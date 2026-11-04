#!/usr/bin/env python3
"""
Laptop half of the laptop GPU; the comma half and the protocol are openpilot/selfdrive/modeld/laptop.py.

    server.py              serve the comma on :8390 until killed
    server.py --prepare    fetch and check the default model, then exit
    server.py --bench 200  time 200 frames of the default model, then exit

The comma names the chestnut model selected on it or in sunnylink by its upstream ref. The big
ONNX at that ref (comma's newest published one when the ref has none) is fetched from git-LFS and
run with onnxruntime on this GPU, recurrent state kept on the GPU. A model is served only after
its GPU outputs match onnxruntime on the CPU.
"""
import argparse
import base64
import pickle
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / ".elantra")]
import numpy as np  # noqa: E402
import fetch_lfs_object as lfs  # noqa: E402
from openpilot.selfdrive.modeld import laptop as link  # noqa: E402

CACHE = REPO  # LFS objects land in REPO/.git/lfs/objects, checks beside them
FORK = "https://raw.githubusercontent.com/aaravsaianugula/openpilot"
ONNX_PATH = "openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
LFS_ENDPOINTS = ["https://huggingface.co/sunnypilot/sunnypilot-lfs.git/info/lfs", *lfs.FALLBACK_ENDPOINTS]
MAX_DIFF = 0.05  # |gpu - cpu| / (1 + |cpu|): fp16 rounding gives ~0.015, a broken backend was off by ~150
SESSION_TIMEOUT = 10  # a comma that goes this long without a frame is gone
TYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(uint8)": "uint8"}
_preparing: set[str] = set()


def resolve(ref: str | None, name: str | None) -> tuple[str, int, str]:
  """(oid, size, name) of the big ONNX at the comma's selected ref, else the default."""
  if ref:
    try:
      with urllib.request.urlopen(f"{FORK}/{ref}/{ONNX_PATH}", timeout=10) as r:
        return (*lfs.parse_pointer(r.read().decode()), name or ref[:9])
    except Exception:
      pass
  return link.DEFAULT_ONNX


def onnx_path(oid: str) -> Path:
  return lfs.object_path(CACHE, oid)


class Model:
  def __init__(self, onnx: Path):
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
      ort.preload_dlls()  # CUDA and cuDNN from the nvidia pip wheels
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.sess = ort.InferenceSession(str(onnx), opts, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1", "cudnn_conv_algo_search": "EXHAUSTIVE"})])  # one graph launch per frame: 4x faster, no jitter
    if self.sess.get_providers()[0] != "CUDAExecutionProvider":
      raise RuntimeError(f"no CUDA on this laptop: onnxruntime offers {self.sess.get_providers()}")
    self.dev = "CUDA"
    ins = {i.name: i for i in self.sess.get_inputs()}
    outs = {o.name: o for o in self.sess.get_outputs()}
    self.state = [n for n in ins if f"next_{n}" in outs]
    self.specs = {n: (list(i.shape), TYPES[i.type]) for n, i in ins.items()}
    self.feed = [(n, *self.specs[n]) for n in ins if n not in self.state]
    if sorted(n for n, _, _ in self.feed) != sorted(link.FEED):
      raise RuntimeError(f"model takes {[n for n, _, _ in self.feed]}, the comma's modeld fills {link.FEED}")
    if outs["outputs"].type != "tensor(float)":
      raise RuntimeError(f"outputs are {outs['outputs'].type}, the comma reads float32")
    meta = self.sess.get_modelmeta().custom_metadata_map
    self.slices = {k: [s.start, s.stop] for k, s in pickle.loads(base64.b64decode(meta["output_slices"])).items()}  # our own LFS-verified file
    self.numel = int(np.prod(outs["outputs"].shape))

    # the frame message lands in self.host; CUDA graphs replay fixed addresses, so every input and output
    # lives in a GPU buffer bound once, and the recurrent state alternates between two of them (A->B, B->A)
    offsets, size = link.layout(self.feed)
    self.host = np.zeros(size, dtype=np.uint8)
    self.feed_views = {n: np.ndarray(shape, dtype=dt, buffer=self.host, offset=offsets[n]) for n, shape, dt in self.feed}
    gpu = lambda shape, dt: ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dt), "cuda", 0)  # noqa: E731
    self.feeds = {n: gpu(shape, dt) for n, shape, dt in self.feed}
    self.out = gpu(outs["outputs"].shape, np.float32)
    self.states = [{n: gpu(*self.specs[n]) for n in self.state} for _ in range(2)]
    self.runs = []
    for graph, (cur, nxt) in enumerate(((0, 1), (1, 0)), start=1):
      io = self.sess.io_binding()
      for n, v in self.feeds.items():
        io.bind_ortvalue_input(n, v)
      for n in self.state:
        io.bind_ortvalue_input(n, self.states[cur][n])
        io.bind_ortvalue_output(f"next_{n}", self.states[nxt][n])
      io.bind_ortvalue_output("outputs", self.out)
      opts = ort.RunOptions()
      opts.add_run_config_entry("gpu_graph_id", str(graph))
      self.runs.append((io, opts))
    self.turn = 0
    for _ in range(2):  # capture both graphs
      self.step()
    self.reset()

  def reset(self) -> None:
    for states in self.states:
      for n, v in states.items():
        v.update_inplace(np.zeros(*self.specs[n]))
    self.turn = 0

  def step(self) -> np.ndarray:
    for n, v in self.feeds.items():
      v.update_inplace(self.feed_views[n])
    self.sess.run_with_iobinding(*self.runs[self.turn])
    self.turn ^= 1
    return self.out.numpy()


def check(model: Model, onnx: Path, steps: int = 2) -> float:
  """Worst |gpu - cpu| / (1 + |cpu|) against onnxruntime on the CPU over `steps` frames, state fed back on both sides."""
  import onnxruntime as ort
  cpu = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
  names = [o.name for o in cpu.get_outputs()]
  state = {n: np.zeros(*model.specs[n]) for n in model.state}
  rng = np.random.default_rng(42)
  model.reset()
  worst = 0.
  for _ in range(steps):
    for n, shape, dt in model.feed:
      model.feed_views[n][...] = rng.integers(0, 256, shape) if dt == "uint8" else rng.random(shape)
    got = model.step().copy()
    ref = dict(zip(names, cpu.run(names, {n: v.copy() for n, v in model.feed_views.items()} | state), strict=True))
    state = {n: ref[f"next_{n}"] for n in model.state}
    worst = max(worst, float(np.max(np.abs(got.astype(np.float64) - ref["outputs"]) / (1 + np.abs(ref["outputs"])))))  # NaN fails the gate
  model.reset()
  return worst


def prepare(oid: str, size: int) -> Path:
  """The checked ONNX for this GPU, fetching and checking it if needed."""
  onnx, ok = onnx_path(oid), onnx_path(oid).with_suffix(".ok")
  if not ok.is_file():
    lfs.fetch(CACHE, oid, size, endpoints=LFS_ENDPOINTS)
    diff = check(Model(onnx), onnx)
    print(f"{oid[:12]}: worst relative diff vs CPU {diff:.4f} (pass <= {MAX_DIFF})", flush=True)
    if not diff <= MAX_DIFF:
      raise RuntimeError(f"model is wrong on this GPU: relative diff {diff} > {MAX_DIFF}")
    ok.write_text(f"{diff}\n")
  return onnx


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


def serve_one(conn: socket.socket, models: dict) -> None:
  hello = link.recv_msg(conn)
  oid, size, name = resolve(hello.get("ref"), hello.get("name"))
  if not onnx_path(oid).with_suffix(".ok").is_file():
    prepare_in_background(oid, size)
    return link.send_msg(conn, {"ok": False, "error": f"preparing {name} for this GPU, ready on a later drive"})
  try:
    model = models[oid] if oid in models else models.setdefault(oid, Model(onnx_path(oid)))
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
      serve_one(conn, models)
    except (OSError, ConnectionError) as e:
      print(f"{addr[0]} gone: {e!r}", flush=True)
    finally:
      conn.close()


def bench(frames: int) -> None:
  model = Model(prepare(*link.DEFAULT_ONNX[:2]))
  rng = np.random.default_rng(0)
  for n, shape, dt in model.feed:
    model.feed_views[n][...] = rng.integers(0, 256, shape) if dt == "uint8" else rng.random(shape)
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
