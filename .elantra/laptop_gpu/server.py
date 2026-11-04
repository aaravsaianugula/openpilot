#!/usr/bin/env python3
"""
Laptop half of the laptop GPU; the comma half and the protocol are openpilot/selfdrive/modeld/laptop.py.

    server.py              serve the comma on :8390 until killed
    server.py --prepare    fetch and check every catalog chestnut model comma published weights for
    server.py --bench 200  time 200 frames of each prepared model, then exit

The comma names the chestnut model selected on it or in sunnylink by its upstream ref. The big
ONNX comma published at that ref is fetched from git-LFS and run with onnxruntime on this GPU,
recurrent state kept on the GPU. Exports from before Sep 14 2026 get their history rebuilt as
state first. A model is served only after its GPU outputs match onnxruntime on the CPU. Models
comma shipped only as AMD binaries for the chestnut are refused, and the comma drives its own.
"""
import argparse
import base64
import json
import math
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

CACHE = REPO  # LFS objects land in REPO/.git/lfs/objects, checks and rebuilt models beside them
COMMA = "https://raw.githubusercontent.com/commaai/openpilot"
ONNX_PATHS = ("openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx", "selfdrive/modeld/models/big_driving_supercombo.onnx")
# keep in step with MODEL_URL_CHESTNUT in openpilot/sunnypilot/models/fetcher.py
CHESTNUT_CATALOG = "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v28.json"
LFS_ENDPOINTS = ["https://huggingface.co/sunnypilot/sunnypilot-lfs.git/info/lfs", *lfs.FALLBACK_ENDPOINTS]
SKIP = 4  # MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ: the history keeps every 4th frame
MAX_DIFF = 0.05  # |gpu - cpu| / (1 + |cpu|): fp16 rounding gives ~0.015, a broken backend was off by ~150
SESSION_TIMEOUT = 10  # a comma that goes this long without a frame is gone
# one graph launch per frame (4x faster, no jitter); NHWC convs with the bias fused in: 11% faster, same gate diff
CUDA_EP = {"enable_cuda_graph": "1", "cudnn_conv_algo_search": "EXHAUSTIVE", "prefer_nhwc": "1", "fuse_conv_bias": "1"}
TYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(uint8)": "uint8"}
_preparing: set[str] = set()


def resolve(ref: str) -> tuple[str, int] | None:
  """(oid, size) of the big ONNX comma published at `ref`; None for the AMD-binary-only models (Sep 16 2026 on)."""
  for path in ONNX_PATHS:
    try:
      with urllib.request.urlopen(f"{COMMA}/{ref}/{path}", timeout=10) as r:
        return lfs.parse_pointer(r.read().decode())
    except Exception:
      pass
  return None


def onnx_path(oid: str) -> Path:
  return lfs.object_path(CACHE, oid)


def prepared() -> dict[str, dict]:
  """oid -> {"onnx", "name", "diff"} for every model checked on this GPU."""
  oks = {ok.stem: json.loads(ok.read_text()) for ok in (CACHE / ".git/lfs/objects").glob("*/*/*.ok")}
  return {oid: p for oid, p in oks.items() if p.get("ep") == CUDA_EP}  # a check under other kernels does not count


def wrap_history(src: Path) -> Path:
  """Rebuild a pre-Sep-14 export, whose frame, desire and feature history modeld kept and fed in, with that
  history as state in and out, the form every export since has. Same queues, sampling and hidden-state
  feedback as that modeld's run_policy (6107da1cd6)."""
  import onnx
  from onnx import TensorProto, helper, numpy_helper
  dst = src.with_name(src.name + ".history.onnx")
  if dst.is_file():
    return dst
  m = onnx.load(str(src))
  g = m.graph
  ins = {i.name: i for i in g.input}
  dims = {n: [d.dim_value for d in i.type.tensor_type.shape.dim] for n, i in ins.items()}
  _, c, H, W = dims["img"]
  _, D, P = dims["desire_pulse"]
  fb = dims["features_buffer"]
  L, T, F = SKIP * (c // 6 - 1) + 1, fb[1], math.prod(fb[2:])
  hidden = pickle.loads(base64.b64decode({p.key: p.value for p in m.metadata_props}["output_slices"]))["hidden_state"]  # our own LFS-verified file
  assert hidden.stop - hidden.start == F, (hidden, F)

  def const(name, value):
    g.initializer.append(numpy_helper.from_array(np.array(value, np.int64), name))
    return name

  pre, post = [], []  # nodes that feed the model, and ones that read its output: graphs are topologically sorted

  def node(op, inputs, output, after=False, **attrs):
    (post if after else pre).append(helper.make_node(op, inputs, [output], **attrs))

  raw = g.output[0].name + "_raw"
  for n in g.node:
    n.output[:] = [raw if o == g.output[0].name else o for o in n.output]
  zero, one, skip = const("lg_0", [0]), const("lg_1", [1]), const("lg_skip", [SKIP])

  # frames: shift the newest warped pair in, keep every SKIP-th, one stack per camera
  node("Slice", ["state_img_q", one, const("lg_L", [L]), one], "lg_img_old")
  node("Unsqueeze", ["new_img", one], "lg_img_new")
  node("Concat", ["lg_img_old", "lg_img_new"], "next_state_img_q", axis=1)
  node("Slice", ["next_state_img_q", zero, const("lg_L2", [L]), one, skip], "lg_img_s")
  for cam, name in enumerate(("img", "big_img")):
    node("Gather", ["lg_img_s", const(f"lg_cam{cam}", cam)], f"lg_{name}_g", axis=0)
    node("Reshape", [f"lg_{name}_g", const(f"lg_{name}_shape", [1, c, H, W])], f"lg_{name}_r")
    node("Cast", [f"lg_{name}_r"], name, to=ins[name].type.tensor_type.elem_type)
  # desire: shift the pulse in, max over each SKIP-frame group
  node("Slice", ["state_desire_q", one, const("lg_SD", [SKIP * D]), zero], "lg_des_old")
  node("Reshape", ["desire", const("lg_des_new_shape", [1, 1, P])], "lg_des_new")
  node("Concat", ["lg_des_old", "lg_des_new"], "next_state_desire_q", axis=0)
  node("Reshape", ["next_state_desire_q", const("lg_des_grp", [D, SKIP, 1, P])], "lg_des_g")
  node("ReduceMax", ["lg_des_g", one], "lg_des_max", keepdims=0)
  node("Reshape", ["lg_des_max", const("lg_des_shape", [1, D, P])], "lg_des_r")
  node("Cast", ["lg_des_r"], "desire_pulse", to=ins["desire_pulse"].type.tensor_type.elem_type)
  # features: the queue holds up to the previous frame's hidden state; this frame's goes in for the next
  node("Slice", ["state_feat_q", zero, const("lg_ST", [SKIP * T]), zero, skip], "lg_feat_s")
  node("Reshape", ["lg_feat_s", const("lg_feat_shape", fb)], "lg_feat_r")
  node("Cast", ["lg_feat_r"], "features_buffer", to=ins["features_buffer"].type.tensor_type.elem_type)
  node("Cast", [raw], "outputs", after=True, to=TensorProto.FLOAT)
  node("Slice", ["outputs", const("lg_h0", [hidden.start]), const("lg_h1", [hidden.stop]), one], "lg_hidden", after=True)
  node("Reshape", ["lg_hidden", const("lg_hidden_shape", [1, 1, F])], "lg_hidden_r", after=True)
  node("Slice", ["state_feat_q", one, const("lg_ST2", [SKIP * T]), zero], "lg_feat_old")
  node("Concat", ["lg_feat_old", "lg_hidden_r"], "next_state_feat_q", after=True, axis=0)
  nodes = pre + list(g.node) + post
  del g.node[:]
  g.node.extend(nodes)

  u8, f32, vi = TensorProto.UINT8, TensorProto.FLOAT, helper.make_tensor_value_info
  states = [("state_img_q", u8, [2, L, 6, H, W]), ("state_desire_q", f32, [SKIP * D, 1, P]), ("state_feat_q", f32, [SKIP * T, 1, F])]
  keep = [i for i in g.input if i.name not in ("img", "big_img", "desire_pulse", "features_buffer")]
  n_out = g.output[0].type.tensor_type.shape.dim[1].dim_value
  del g.input[:], g.output[:]
  g.input.extend(keep + [vi("new_img", u8, [2, 6, H, W]), vi("desire", f32, [P])] + [vi(*s) for s in states])
  g.output.extend([vi("outputs", f32, [1, n_out])] + [vi(f"next_{n}", t, s) for n, t, s in states])
  onnx.checker.check_model(m)
  tmp = dst.with_suffix(".tmp")
  onnx.save(m, str(tmp))
  tmp.replace(dst)
  return dst


class Model:
  def __init__(self, onnx: Path):
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
      ort.preload_dlls()  # CUDA and cuDNN from the nvidia pip wheels
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.sess = ort.InferenceSession(str(onnx), opts, providers=[("CUDAExecutionProvider", CUDA_EP)])
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


def random_feed(model: Model, rng) -> None:
  for n, shape, dt in model.feed:
    model.feed_views[n][...] = rng.integers(0, 256, shape) if dt == "uint8" else rng.random(shape)


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
    random_feed(model, rng)
    got = model.step().copy()
    ref = dict(zip(names, cpu.run(names, {n: v.copy() for n, v in model.feed_views.items()} | state), strict=True))
    state = {n: ref[f"next_{n}"] for n in model.state}
    worst = max(worst, float(np.max(np.abs(got.astype(np.float64) - ref["outputs"]) / (1 + np.abs(ref["outputs"])))))  # NaN fails the gate
  model.reset()
  return worst


def prepare(oid: str, size: int, name: str) -> Path:
  """The checked, servable ONNX for this GPU, fetching, rebuilding and checking it if needed."""
  ok = onnx_path(oid).with_suffix(".ok")
  if oid not in prepared():
    lfs.fetch(CACHE, oid, size, endpoints=LFS_ENDPOINTS)
    import onnxruntime as ort
    onnx = onnx_path(oid)
    if "features_buffer" in {i.name for i in ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]).get_inputs()}:
      onnx = wrap_history(onnx)
    diff = check(Model(onnx), onnx)
    print(f"{name}: worst relative diff vs CPU {diff:.4f} (pass <= {MAX_DIFF})", flush=True)
    if not diff <= MAX_DIFF:
      raise RuntimeError(f"{name} is wrong on this GPU: relative diff {diff} > {MAX_DIFF}")
    ok.write_text(json.dumps({"onnx": onnx.name, "name": name, "diff": diff, "ep": CUDA_EP}))
  return onnx_path(oid).with_name(json.loads(ok.read_text())["onnx"])


def prepare_in_background(oid: str, size: int, name: str) -> None:
  def work():
    try:
      prepare(oid, size, name)
    except Exception as e:
      print(f"preparing {name} failed: {e!r}", flush=True)
    finally:
      _preparing.discard(oid)
  if oid not in _preparing:
    _preparing.add(oid)
    threading.Thread(target=work, daemon=True).start()


def prepare_catalog() -> None:
  """Every chestnut model in sunnypilot's catalog that comma published weights for, ready before its first drive."""
  with urllib.request.urlopen(CHESTNUT_CATALOG, timeout=20) as r:
    bundles = json.load(r)["bundles"]
  for b in bundles:
    if (found := resolve(b["ref"])) is None:
      print(f"{b['display_name']}: AMD binaries only, needs a real chestnut", flush=True)
      continue
    try:
      prepare(*found, b["display_name"])
    except Exception as e:
      print(f"{b['display_name']}: {e!r}", flush=True)


def serve_one(conn: socket.socket, models: dict) -> None:
  hello = link.recv_msg(conn)
  ref, name = hello.get("ref"), hello.get("name") or "comma's default big model"
  found = resolve(ref) if ref else None
  if found is None:
    ready = sorted(p["name"] for p in prepared().values())
    return link.send_msg(conn, {"ok": False, "error": f"{name} ships only as AMD binaries for a real chestnut; this laptop runs: {', '.join(ready)}"})
  oid, size = found
  if oid not in prepared():
    prepare_in_background(oid, size, name)
    return link.send_msg(conn, {"ok": False, "error": f"preparing {name} for this GPU, ready on a later drive"})
  if oid not in models:
    models.clear()  # one model in GPU memory at a time: a 4 GB laptop GPU holds one
    models[oid] = Model(prepare(oid, size, name))
    (CACHE / ".git/lfs/last_model").write_text(json.dumps([oid, size, name]))
  model = models[oid]
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
  try:  # load the last model served so the next drive does not wait for it
    oid, size, name = json.loads((CACHE / ".git/lfs/last_model").read_text())
    models[oid] = Model(prepare(oid, size, name))
  except Exception as e:
    print(f"no model preloaded: {e!r}", flush=True)
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
  for oid, p in prepared().items():
    model = Model(onnx_path(oid).with_name(p["onnx"]))
    random_feed(model, np.random.default_rng(0))
    times = []
    for _ in range(frames):
      t = time.perf_counter()
      model.step()
      times.append((time.perf_counter() - t) * 1e3)
    times = np.array(times[10:])
    print(f"{p['name']}: median {np.median(times):.2f} ms, p99 {np.percentile(times, 99):.2f} ms, max {times.max():.2f} ms (diff {p['diff']:.4f})", flush=True)
    del model


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--prepare", action="store_true")
  ap.add_argument("--bench", type=int)
  args = ap.parse_args()
  if sys.platform == "win32":  # what OMEN's game booster does for a game: high priority, so background apps can't delay a frame
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)  # HIGH_PRIORITY_CLASS
  if args.prepare:
    prepare_catalog()
  elif args.bench:
    bench(args.bench)
  else:
    serve()
