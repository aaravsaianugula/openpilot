#!/usr/bin/env python3
"""
Laptop half of the laptop GPU; the comma half and the protocol are openpilot/selfdrive/modeld/laptop.py.

    server.py              serve the comma on :8390 until killed
    server.py --prepare    fetch, compile and check the default model, then exit
    server.py --bench 200  time 200 frames of the default model, then exit

The comma names the chestnut model selected on it or in sunnylink by its upstream ref. The big
ONNX at that ref (comma's newest published one when the ref has none) is fetched from git-LFS,
compiled once for this GPU with tinygrad's compile_onnx.py into ~/.cache/laptop_gpu/<oid>.pkl, and
served only after it matches onnxruntime on the CPU: tinygrad's CUDA renderer has been silently
wrong on openpilot models before (tinygrad #11705).
"""
import argparse
import base64
import math
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tinygrad_repo"), str(REPO / ".elantra")]
import numpy as np  # noqa: E402
import fetch_lfs_object as lfs  # noqa: E402
from openpilot.selfdrive.modeld import laptop as link  # noqa: E402

CACHE = Path(os.getenv("LAPTOP_GPU_CACHE", Path.home() / ".cache/laptop_gpu"))
ONNX_PATH = "openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
MAX_DIFF = 0.2  # fp16 model: known-good backends land at ~0.08, the broken CUDA path at ~150
SESSION_TIMEOUT = 10  # a comma that goes this long without a frame is gone
_preparing: set[str] = set()


def resolve(ref: str | None, name: str | None) -> tuple[str, int, str]:
  """(oid, size, name) of the big ONNX at the comma's selected ref, else the default."""
  if ref:
    try:
      return (*lfs.read_pointer(REPO, ONNX_PATH, ref), name or ref[:9])
    except Exception:
      pass
  return link.DEFAULT_ONNX


def random_feeds(specs: dict, seed: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  return {n: (rng.integers(0, 256, shape, dtype=np.uint8) if np.dtype(dtype) == np.uint8 else rng.standard_normal(shape).astype(dtype))
          for n, (shape, dtype, _) in specs.items()}


class Model:
  def __init__(self, pkl: Path):
    from tinygrad import Tensor
    from tinygrad.dtype import _from_np_dtype
    from tinygrad.engine.realize import lower_and_compile
    from tinygrad.uop.ops import UOp
    from examples.openpilot.helpers import load_pickle

    def view(buffer, shape, dtype, offset):  # modeld's input_view
      v = buffer.view(math.prod(shape) * dtype.itemsize, offset).ensure_allocated()
      return Tensor(UOp.from_buffer(v, dtype)).reshape(shape)

    jits = load_pickle(pkl, out_of_band=True)
    self.run = jits['run']
    self.run.captured._linear = lower_and_compile(self.run.captured._linear)
    self.specs, out_shapes = jits['input_specs'], jits['metadata']['output_shapes']
    self.dev = self.specs['new_img'][2]
    self.state = [n for n in self.specs if f'next_{n}' in out_shapes]
    self.feed = [(n, list(shape), np.dtype(dtype).name) for n, (shape, dtype, _) in self.specs.items() if n not in self.state]
    if sorted(n for n, _, _ in self.feed) != sorted(link.FEED):
      raise RuntimeError(f"model takes {[n for n, _, _ in self.feed]}, the comma's modeld fills {link.FEED}")
    out_shape, out_dtype, _ = jits['output_specs']['outputs']
    assert np.dtype(out_dtype) == np.float32, out_dtype
    self.numel = math.prod(out_shape)
    self.slices = {k: [s.start, s.stop] for k, s in pickle.loads(base64.b64decode(jits['metadata']['metadata']['output_slices'])).items()}

    # the frame message lands in self.host and goes to the GPU in one copy, inputs read in place
    offsets, size = link.layout(self.feed)
    self.host = np.zeros(size, dtype=np.uint8)
    self.host_buf = Tensor(self.host, device='NPY')._buffer()
    self.device_buf = Tensor(self.host, device=self.dev)._buffer()
    self.inputs = {n: view(self.device_buf, shape, _from_np_dtype(np.dtype(dtype)), offsets[n]) for n, shape, dtype in self.feed}
    self.inputs |= {n: Tensor(np.zeros(self.specs[n][0], self.specs[n][1]), device=self.dev).realize() for n in self.state}
    self.outputs = {n: Tensor(np.zeros(shape, dtype), device=d).realize() for n, (shape, dtype, d) in jits['output_specs'].items()}
    for n in self.state:
      self.outputs[f'next_{n}'] = view(self.inputs[n]._buffer(), self.specs[n][0], _from_np_dtype(np.dtype(self.specs[n][1])), 0)
    self.feed_views = {n: np.ndarray(shape, dtype=dtype, buffer=self.host, offset=offsets[n]) for n, shape, dtype in self.feed}
    self.reset()

  def reset(self) -> None:
    for n in self.state:
      self.inputs[n].assign(0).realize()

  def step(self) -> np.ndarray:
    self.device_buf.copy_from(self.host_buf)
    self.run(output_buffers=self.outputs, **self.inputs)
    return self.outputs['outputs'].numpy()


def check(model: Model, onnx: Path, steps: int = 2) -> float:
  """Max abs diff against onnxruntime over `steps` frames, recurrent state fed back on both sides."""
  import onnxruntime as ort
  sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
  names = [o.name for o in sess.get_outputs()]
  state = {n: np.zeros(model.specs[n][0], model.specs[n][1]) for n in model.state}
  model.reset()
  worst = 0.
  for i in range(steps):
    feeds = {n: v for n, v in random_feeds(model.specs, 42 + i).items() if n not in model.state}
    for n, v in feeds.items():
      model.feed_views[n][...] = v
    got = model.step()
    ref = dict(zip(names, sess.run(names, feeds | state), strict=True))
    state = {n: ref[f'next_{n}'] for n in model.state}
    worst = max(worst, float(np.max(np.abs(got.astype(np.float64) - ref['outputs'].reshape(got.shape)))))  # NaN propagates and fails below
  model.reset()
  return worst


def prepare(oid: str, size: int) -> Path:
  """The checked, compiled model for this GPU, building it if needed."""
  pkl, ok = CACHE / f"{oid}.pkl", CACHE / f"{oid}.ok"
  if not ok.is_file():
    CACHE.mkdir(parents=True, exist_ok=True)
    if not pkl.is_file():
      lfs.fetch(REPO, oid, size)
      tmp = CACHE / f"{oid}.tmp"
      subprocess.run([sys.executable, str(REPO / "tinygrad_repo/examples/openpilot/compile_onnx.py"), str(lfs.object_path(REPO, oid)), str(tmp),
                      "--out-of-band", "--benchmark-runs", "3"], check=True, cwd=CACHE, env=dict(os.environ, PYTHONPATH=str(REPO / "tinygrad_repo")))
      tmp.replace(pkl)
    diff = check(Model(pkl), lfs.object_path(REPO, oid))
    print(f"{oid[:12]}: max abs diff vs onnxruntime {diff:.4f} (pass <= {MAX_DIFF})", flush=True)
    if not diff <= MAX_DIFF:
      raise RuntimeError(f"compiled model is wrong on this GPU: max abs diff {diff} > {MAX_DIFF}")
    ok.write_text(f"{diff}\n")
  return pkl


def prepare_in_background(oid: str, size: int) -> None:
  def work():
    try:
      prepare(oid, size)
    except Exception as e:
      print(f"preparing {oid[:12]} failed: {e!r}", flush=True)
    finally:
      _preparing.discard(oid)
  if oid not in _preparing:
    _preparing.add(oid)
    threading.Thread(target=work, daemon=True).start()


def session(conn: socket.socket, models: dict) -> None:
  hello = link.recv_msg(conn)
  oid, size, name = resolve(hello.get("ref"), hello.get("name"))
  if not (CACHE / f"{oid}.ok").is_file():
    prepare_in_background(oid, size)
    return link.send_msg(conn, {"ok": False, "error": f"preparing {name} for this GPU, ready on a later drive"})
  try:
    model = models[oid] if oid in models else models.setdefault(oid, Model(CACHE / f"{oid}.pkl"))
  except Exception as e:
    return link.send_msg(conn, {"ok": False, "error": repr(e)})
  model.reset()
  link.send_msg(conn, {"ok": True, "model": name, "dev": model.dev, "feed": model.feed, "output_slices": model.slices, "output_numel": model.numel})
  print(f"serving {name} on {model.dev}", flush=True)
  frame = memoryview(model.host)
  while True:
    link.recv_into(conn, frame)
    seq, = link.FRAME.unpack_from(model.host)
    conn.sendall(link.REPLY.pack(seq) + model.step().tobytes())


def serve() -> None:
  models: dict[str, Model] = {}
  try:  # load the default up front so the first drive does not wait for it
    models[link.DEFAULT_ONNX[0]] = Model(prepare(*link.DEFAULT_ONNX[:2]))
  except Exception as e:
    print(f"default model not ready: {e!r}", flush=True)
  srv = socket.create_server(("0.0.0.0", link.PORT))
  print(f"listening on :{link.PORT}", flush=True)
  while True:
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(SESSION_TIMEOUT)
    try:
      session(conn, models)
    except (OSError, ConnectionError) as e:
      print(f"{addr[0]} gone: {e!r}", flush=True)
    finally:
      conn.close()


def bench(frames: int) -> None:
  model = Model(prepare(*link.DEFAULT_ONNX[:2]))
  for n, v in random_feeds(model.specs, 0).items():
    if n not in model.state:
      model.feed_views[n][...] = v
  times = []
  for _ in range(frames):
    t = time.perf_counter()
    model.step()
    times.append((time.perf_counter() - t) * 1e3)
  times = np.array(times[10:])
  print(f"{model.dev}: median {np.median(times):.2f} ms, p99 {np.percentile(times, 99):.2f} ms, max {times.max():.2f} ms over {len(times)} frames")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--prepare", action="store_true")
  ap.add_argument("--bench", type=int)
  args = ap.parse_args()
  if args.prepare:
    prepare(*link.DEFAULT_ONNX[:2])
  elif args.bench:
    bench(args.bench)
  else:
    serve()
