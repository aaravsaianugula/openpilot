#!/usr/bin/env python3
"""
Does the eGPU compute the *right* numbers, or just fast wrong ones?

A model that loads and runs proves nothing. tinygrad issue #11705 is openpilot's supercombo
producing outputs with a max absolute error over 50,000 against onnxruntime on a backend that
raised no error and dropped no frame -- everything looked healthy the whole time. A corrupted
transfer or a miscompiled kernel on gfx1032 fails exactly that way, so the card does not get to
drive until it has answered this tool.

Two stages, in this order, because the second is worthless if the first is broken:

  1. copyout  Write a known byte pattern into device memory and read it back byte-exact, 10,000
              times, through the same Allocator._copyin/_copyout pair every tensor uses. On the
              chestnut that is `AMDAllocator._copyout`'s SCSI-read loop over USB
              (tinygrad/runtime/ops_amd.py) -- bespoke to this setup, and nothing else in
              openpilot exercises it directly. One mismatched byte ends the run and prints the
              iteration and the offset.

  2. model    Run one onnx model through the eGPU (tinygrad) and through onnxruntime's
              CPUExecutionProvider on identical inputs, over >= 1000 consecutive frames of a real
              recorded route, and report the MAX absolute difference per output tensor. The mean
              is useless here: #11705's signature is a handful of catastrophically wrong elements
              in an otherwise sane tensor, and averaging buries exactly that.

Thresholds. The model is fp16 and the two backends order float math differently, so numpy's
default rtol of 1e-5 fails on a known-good backend. What matters is whether the difference is
small next to the quantities these outputs feed -- metres of path, m/s^2 of acceleration:

    max <= 0.2    pass
    0.2 .. 1.0    inconclusive, reported with the number, exit 2 -- never a silent pass
    max >  1.0    fail

Both backends see the *same* inputs on every frame, features buffer included, which is driven
from the CPU reference's own rollout. Letting each backend feed its own features back would
measure how fast a chaotic loop diverges rather than whether the eGPU computes the model
correctly, and would make the answer depend on how long the route is.

The frames and the per-frame model inputs come out of the route with openpilot's own readers,
tools/lib/logreader.py and tools/lib/framereader.py -- no new log parser. Finding the segment
directories is a listdir instead of tools/lib/route.py, for the reason in resolve_segments. The
warp from camera frame to model frame is reproduced in numpy from compile_modeld.py because both
backends must be handed byte-identical images; `--check-warp` holds that reproduction against
tinygrad's own implementation on CPU instead of asking you to trust it.

Nothing above the stage functions imports tinygrad, onnxruntime or openpilot, so this file reads,
reviews and --helps on a machine with no card attached.

What stage 2 needs on the device, and what AGNOS 19.6 does not ship: onnxruntime. Everything else
is there -- ffmpeg and ffprobe in /usr/local/venv/bin, capnp, zstandard, tinygrad. /usr/local/venv
is not writable by `comma`, so installing it takes root. Expect roughly 250 ms/frame of numpy warp
on top of the two forward passes, so budget the better part of an hour for 1000 frames.

Usage, on the comma 4, offroad, with modeld stopped:
    cd /data/openpilot
    export PATH=/usr/comma/shims:/usr/local/venv/bin:$PATH
    export PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo
    python3 .elantra/validate_numerics.py --route 0000008b--f5329831c9
    python3 .elantra/validate_numerics.py --stage copyout      # transfer path only
    python3 .elantra/validate_numerics.py --stage model --check-warp --route 0000008b--f5329831c9
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS_MAX_DIFF = 0.2
FAIL_MAX_DIFF = 1.0

COPYOUT_ITERS = 10_000
# The USB interface hands the allocator a single 0x80000 staging buffer (ops_amd.py, USBIface),
# and _copyout chunks against its size. A payload larger than that is what walks the chunk loop,
# where an off-by-one or a stale-buffer bug would live.
COPYOUT_BYTES = 1 << 20

MIN_FRAMES = 1000
# what PolicyInputs.step builds, and therefore the only model this tool can drive
POLICY_INPUT_NAMES = frozenset({"img", "big_img", "desire_pulse", "features_buffer", "traffic_convention", "action_t"})
DEFAULT_DATA_DIR = Path("/data/media/0/realdata")
DEFAULT_ONNX = REPO / "openpilot" / "selfdrive" / "modeld" / "models" / "big_driving_supercombo.onnx"

EXIT_OK, EXIT_FAIL, EXIT_INCONCLUSIVE = 0, 1, 2

ORT_DTYPES = {
  "tensor(float)": "float32",
  "tensor(float16)": "float16",
  "tensor(uint8)": "uint8",
  "tensor(int8)": "int8",
  "tensor(int32)": "int32",
  "tensor(int64)": "int64",
}


def head(title: str) -> None:
  print("")
  print("=" * 72)
  print(title)
  print("=" * 72)


def ok(label: str, detail: str = "") -> None:
  print("  ok    " + label + ((": " + detail) if detail else ""))


def info(label: str, detail: str = "") -> None:
  print("  --    " + label + ((": " + detail) if detail else ""))


def bad(label: str, detail: str = "") -> None:
  print("  FAIL  " + label + ((": " + detail) if detail else ""))


def report_tinygrad() -> None:
  """Say which tinygrad answered.

  This repo carries tinygrad_repo on the rdna2-am branch and the device also has one installed in
  its venv. A result that does not name the file it tested is not reproducible, and on this
  project a package resolving to the wrong lineage has already cost a day.
  """
  import tinygrad
  info("tinygrad", str(Path(tinygrad.__file__).resolve().parent))


# ---------------------------------------------------------------------------- stage 1: copyout

def stage_copyout(device: str, iters: int, nbytes: int) -> bool:
  head(f"stage 1: {device} _copyout round trip -- {iters} x {nbytes} bytes")

  report_tinygrad()
  from tinygrad.device import Device

  dev = Device[device]
  allocator = dev.allocator
  info("device", f"{dev.device}")

  buf = allocator.alloc(nbytes)
  rng = np.random.default_rng(0xC0FFEE)
  src, dst = np.empty(nbytes, dtype=np.uint8), np.empty(nbytes, dtype=np.uint8)
  src_mv, dst_mv = memoryview(src), memoryview(dst)
  st = time.monotonic()
  try:
    for i in range(iters):
      # a fresh pattern every iteration on purpose: a constant one passes even when the readback
      # returns whatever was in the staging buffer last time, which is a failure we care about
      src[:] = np.frombuffer(rng.bytes(nbytes), dtype=np.uint8)
      allocator._copyin(buf, src_mv)
      dev.synchronize()
      allocator._copyout(dst_mv, buf)

      if not np.array_equal(src, dst):
        diff = np.flatnonzero(src != dst)
        off = int(diff[0])
        detail = f"{diff.size} of {nbytes} bytes differ, first at offset {off}"
        bad(f"mismatch on iteration {i}", f"{detail} (wrote 0x{src[off]:02x}, read 0x{dst[off]:02x})")
        return False

      if (i + 1) % 500 == 0:
        moved = 2 * (i + 1) * nbytes / 1e9
        el = time.monotonic() - st
        info(f"{i + 1}/{iters}", f"{moved:.2f} GB moved, {el:.1f}s, {moved * 1e3 / el:.0f} MB/s")
  finally:
    allocator.free(buf, nbytes)

  ok(f"{iters} round trips byte-exact", f"{2 * iters * nbytes / 1e9:.2f} GB in {time.monotonic() - st:.1f}s")
  return True


# ------------------------------------------------------------------- route -> model inputs

def resolve_segments(data_dir: Path, route: str) -> tuple[str, list[Path]]:
  """Segment directories for a route, in order.

  Not Route(name, data_dir=...): loggerd names on-device routes `<count>--<uid>` with no dongle
  id (system/loggerd/logger.cc), and tools.lib.route.RouteName asserts a 16-char dongle id, so it
  cannot parse what is actually on this device. Everything below the directory listing -- the
  logs and the frames -- is read with openpilot's own readers.
  """
  name, _, seg_slice = route.partition("/")
  segs = []
  for d in sorted(data_dir.iterdir()):
    prefix, _, num = d.name.rpartition("--")
    if d.is_dir() and prefix == name and num.isdigit():
      segs.append((int(num), d))
  if not segs:
    raise FileNotFoundError(f"no segment directories for route {name!r} under {data_dir}")

  segs.sort()
  if seg_slice:
    start, colon, end = seg_slice.partition(":")
    lo = int(start) if start else 0
    hi = int(end) if end else (segs[-1][0] + 1 if colon else lo + 1)
    segs = [(n, d) for n, d in segs if lo <= n < hi]
    if not segs:
      raise FileNotFoundError(f"route {name!r} has no segments in range {seg_slice}")
  return name, [d for _, d in segs]


class RouteScan:
  """Per-frame model inputs and camera indices, read out of a route's rlogs one segment at a time.

  `frames` maps a narrow-road frameId to the state modeld had when it ran that frame; `cameras`
  maps a model input name to {frameId: (segment dir, index into that segment's hevc)}.

  Keyed on modelV2 because modeld stamps it with the frame it just ran
  (fill_model_msg: modelV2.frameId = vipc_frame_id) and publishes it at the end of that
  iteration, so every other message already in flight is what modeld's SubMaster had read.

  One segment at a time because a full route is an hour of rlogs -- 5 minutes of parsing on the
  device to find 50 seconds of frames. The caller stops as soon as it has the run it needs.
  """

  ENCODE_KEY = {"narrowRoadEncodeIdx": "img", "wideRoadEncodeIdx": "big_img"}

  def __init__(self):
    self.frames: dict[int, dict] = {}
    self.cameras: dict[str, dict[int, tuple[Path, int]]] = {"img": {}, "big_img": {}}
    self.long_actuator_delay: float | None = None
    self._cur: dict = {"rpy_calib": None, "is_rhd": False, "lat_delay": None, "sensor": None, "device_type": None}
    # DesireHelper runs after the model, so the state published alongside frame N is what feeds
    # frame N+1; carried across segments here the same way.
    self._pending_desire = (0, 0, 0)

  def add_segment(self, seg: Path) -> None:
    from openpilot.tools.lib.logreader import LogReader

    rlog = next((p for p in (seg / "rlog.zst", seg / "rlog.bz2", seg / "rlog") if p.is_file()), None)
    if rlog is None:
      raise FileNotFoundError(f"no rlog in {seg}")

    cur, seen_model = self._cur, False
    for msg in LogReader(str(rlog), sort_by_time=True):
      which = msg.which()
      if which in self.ENCODE_KEY:
        idx = getattr(msg, which)
        if str(idx.type) == "fullHEVC":
          self.cameras[self.ENCODE_KEY[which]][idx.frameId] = (seg, idx.segmentId)
      elif which == "extrinsicsCalibration":
        cur["rpy_calib"] = np.array(msg.extrinsicsCalibration.rpyCalib, dtype=np.float32)
      elif which == "driverMonitoringState":
        cur["is_rhd"] = bool(msg.driverMonitoringState.isRHD)
      elif which == "lateralDelay":
        cur["lat_delay"] = float(msg.lateralDelay.lateralDelay)
      elif which == "narrowRoadCameraState":
        cur["sensor"] = str(msg.narrowRoadCameraState.sensor)
      elif which == "deviceState":
        cur["device_type"] = str(msg.deviceState.deviceType)
      elif which == "carParams" and self.long_actuator_delay is None:
        self.long_actuator_delay = float(msg.carParams.longitudinalActuatorDelay)
      elif which == "modelV2":
        meta = msg.modelV2.meta
        seen_model = True
        self.frames[int(msg.modelV2.frameId)] = {**cur, "desire_state": self._pending_desire}
        self._pending_desire = (enum_int(meta.laneChangeState), enum_int(meta.laneChangeDirection), self._pending_desire[2])
      elif which == "modelDataV2SP" and seen_model:
        self._pending_desire = (self._pending_desire[0], self._pending_desire[1], enum_int(msg.modelDataV2SP.laneTurnDirection))


def enum_int(v) -> int:
  """pycapnp hands back a plain int for a schema enumerant and a _DynamicEnum for one read off a
  message; the latter compares equal to its int but is not one."""
  return int(v.raw) if hasattr(v, "raw") else int(v)


def desire_index(lane_change_state: int, lane_change_direction: int, lane_turn_direction: int) -> int:
  """The Desire modeld fed the model, from the state it published.

  Mirrors the tail of DesireHelper.update (selfdrive/controls/lib/desire_helper.py). Reading the
  three published enums beats re-running DesireHelper offline, which needs Params and a live
  carState; the mapping between them is the four lines below and nothing else.
  """
  from openpilot.cereal import custom, log

  turn_desires = {
    enum_int(custom.ModelDataV2SP.TurnDirection.turnLeft): enum_int(log.Desire.turnLeft),
    enum_int(custom.ModelDataV2SP.TurnDirection.turnRight): enum_int(log.Desire.turnRight),
  }
  if lane_turn_direction in turn_desires:
    return turn_desires[lane_turn_direction]
  if lane_change_state == enum_int(log.LaneChangeState.laneChangeStarting):
    if lane_change_direction == enum_int(log.LaneChangeDirection.left):
      return enum_int(log.Desire.laneChangeLeft)
    if lane_change_direction == enum_int(log.LaneChangeDirection.right):
      return enum_int(log.Desire.laneChangeRight)
  return enum_int(log.Desire.none)


# ----------------------------------------------------------------- camera frame -> model frame

def warp_nearest(src_flat: np.ndarray, m_inv: np.ndarray, dst_wh: tuple[int, int],
                 src_hw: tuple[int, int], stride_pad: int) -> np.ndarray:
  """numpy transcription of warp_perspective_tinygrad (selfdrive/modeld/compile_modeld.py).

  Both backends have to be handed the same uint8 image, so the warp runs once on the host rather
  than on either device. --check-warp holds this against tinygrad's version.
  """
  w_dst, h_dst = dst_wh
  h_src, w_src = src_hw
  x = np.tile(np.arange(w_dst, dtype=np.float32), h_dst)
  y = np.repeat(np.arange(h_dst, dtype=np.float32), w_dst)

  src_x = m_inv[0, 0] * x + m_inv[0, 1] * y + m_inv[0, 2]
  src_y = m_inv[1, 0] * x + m_inv[1, 1] * y + m_inv[1, 2]
  src_w = m_inv[2, 0] * x + m_inv[2, 1] * y + m_inv[2, 2]
  src_x = src_x / src_w
  src_y = src_y / src_w

  x_nn = np.clip(np.round(src_x), 0, w_src - 1).astype(np.int32)
  y_nn = np.clip(np.round(src_y), 0, h_src - 1).astype(np.int32)
  return src_flat[y_nn * (w_src + stride_pad) + x_nn]


def frames_to_tensor(yuv: np.ndarray) -> np.ndarray:
  """numpy transcription of frames_to_tensor (selfdrive/modeld/compile_modeld.py)."""
  h = (yuv.shape[0] * 2) // 3
  w = yuv.shape[1]
  return np.concatenate([yuv[0:h:2, 0::2], yuv[1:h:2, 0::2], yuv[0:h:2, 1::2], yuv[1:h:2, 1::2],
                         yuv[h:h + h // 4].reshape(h // 2, w // 2),
                         yuv[h + h // 4:h + h // 2].reshape(h // 2, w // 2)], axis=0).reshape(6, h // 2, w // 2)


def frame_prepare(nv12_buf: np.ndarray, m_inv: np.ndarray, cam_wh: tuple[int, int],
                  model_wh: tuple[int, int], stride: int, y_height: int, uv_height: int) -> np.ndarray:
  """numpy transcription of frame_prepare_tinygrad (selfdrive/modeld/compile_modeld.py)."""
  cam_w, cam_h = cam_wh
  model_w, model_h = model_wh
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  # UV_SCALE @ M_inv @ UV_SCALE_INV collapses to this elementwise scaling
  m_inv_uv = m_inv * np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], dtype=np.float32)
  uv = nv12_buf[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)

  y = warp_nearest(nv12_buf[:cam_h * stride], m_inv, (model_w, model_h), (cam_h, cam_w), stride_pad)
  u = warp_nearest(np.ascontiguousarray(uv[:cam_h // 2, :cam_w:2]).ravel(), m_inv_uv,
                   (model_w // 2, model_h // 2), (cam_h // 2, cam_w // 2), 0)
  v = warp_nearest(np.ascontiguousarray(uv[:cam_h // 2, 1:cam_w:2]).ravel(), m_inv_uv,
                   (model_w // 2, model_h // 2), (cam_h // 2, cam_w // 2), 0)
  return frames_to_tensor(np.concatenate([y, u, v]).reshape(model_h * 3 // 2, model_w))


def pack_nv12(decoded: np.ndarray, cam_w: int, cam_h: int, stride: int, y_height: int, uv_height: int) -> np.ndarray:
  """Lay a decoded NV12 frame out the way camerad's buffer is, padding and all.

  ffmpeg hands back tight rows; camerad's buffer is stride-aligned (system/camerad/cameras/
  nv12_info.py) and the warp indexes straight into it, so the padding has to be there for the
  gather indices to land where modeld's would. uv_height, not cam_h // 2: at 1344x760 the UV
  plane is padded from 380 to 384 rows and the warp slices the full padded plane.
  """
  buf = np.zeros(stride * (y_height + uv_height), dtype=np.uint8)
  buf[:cam_h * stride].reshape(cam_h, stride)[:, :cam_w] = decoded[:cam_h * cam_w].reshape(cam_h, cam_w)
  uv_off = stride * y_height
  buf[uv_off:uv_off + (cam_h // 2) * stride].reshape(cam_h // 2, stride)[:, :cam_w] = \
    decoded[cam_h * cam_w:].reshape(cam_h // 2, cam_w)
  return buf


# A transcription bug -- wrong stride, wrong plane, swapped axes -- moves thousands to millions
# of pixels. What is left over is the nearest-neighbour tie: where the source coordinate lands on
# exactly .5, tinygrad's backend division and numpy's disagree in the last float32 bit and the two
# round to neighbouring pixels. Measured at 1 pixel in 196608 on a CUDA backend.
WARP_TIE_TOLERANCE = 1e-4


def check_warp(cam_wh: tuple[int, int], model_wh: tuple[int, int]) -> bool:
  """Hold the numpy warp against tinygrad's own, on CPU, on a random frame."""
  head("warp check: numpy transcription vs compile_modeld.frame_prepare_tinygrad (CPU)")
  # compile_modeld reads WARP_DEV once at import; Context(DEV=...) keeps the rest of the warp's
  # tensors (the aranges) off the card, so this check never touches the eGPU
  os.environ["WARP_DEV"] = "CPU"
  from tinygrad.helpers import Context
  from tinygrad.tensor import Tensor

  from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, make_frame_prepare
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  cam_w, cam_h = cam_wh
  stride, y_height, uv_height, size = get_nv12_info(cam_w, cam_h)
  rng = np.random.default_rng(7)
  buf = np.zeros(size, dtype=np.uint8)
  buf[:stride * (y_height + uv_height)] = rng.integers(0, 256, stride * (y_height + uv_height), dtype=np.uint8)
  m_inv = np.array([[0.62, -0.03, 220.0], [0.01, 0.61, 180.0], [1e-5, -2e-5, 1.0]], dtype=np.float32)

  ref = make_frame_prepare(NV12Frame(cam_w, cam_h, stride, y_height, uv_height, size), *model_wh)
  with Context(DEV="CPU"):
    expected = ref(Tensor(buf, device="CPU"), Tensor(m_inv, device="CPU")).numpy()
  got = frame_prepare(buf, m_inv, cam_wh, model_wh, stride, y_height, uv_height)

  n = int(np.count_nonzero(expected != got))
  budget = max(4, int(expected.size * WARP_TIE_TOLERANCE))
  if n > budget:
    bad("numpy warp does not match tinygrad's", f"{n} of {expected.size} pixels differ, budget {budget}")
    return False
  ok("numpy warp matches tinygrad's", f"{expected.shape} uint8, {n} of {expected.size} pixels differ (budget {budget})")
  return True


# ------------------------------------------------------------------------- policy input queues

class PolicyInputs:
  """The rolling buffers compile_modeld.make_input_queues keeps on the device, on the host.

  They live here rather than on either device so both backends are handed the same arrays: a
  queue kept on the eGPU would carry that backend's own errors into the next frame's input and
  turn a per-frame comparison into a divergence race.
  """

  def __init__(self, input_shapes: dict[str, tuple[int, ...]], frame_skip: int, desire_len: int):
    img = input_shapes["img"]
    n_frames = img[1] // 6
    self.frame_skip = frame_skip
    self.img_q = {k: np.zeros((frame_skip * (n_frames - 1) + 1, 6, img[2], img[3]), dtype=np.uint8)
                  for k in ("img", "big_img")}
    fb = input_shapes["features_buffer"]
    dp = input_shapes["desire_pulse"]
    self.feat_q = np.zeros((frame_skip * fb[1], fb[0], fb[2]), dtype=np.float32)
    self.desire_q = np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32)
    self.prev_desire = np.zeros(desire_len, dtype=np.float32)

  @staticmethod
  def _push(buf: np.ndarray, new_val: np.ndarray) -> None:
    buf[:-1] = buf[1:]
    buf[-1] = new_val

  def step(self, warped: dict[str, np.ndarray], desire_pulse: np.ndarray, traffic_convention: np.ndarray,
           action_t: np.ndarray, prev_feat: np.ndarray) -> dict[str, np.ndarray]:
    # modeld zeroes desire 0 and pulses only on the rising edge (ModelState.run in modeld.py)
    desire_pulse = desire_pulse.copy()
    desire_pulse[0] = 0
    desire = np.where(desire_pulse - self.prev_desire > .99, desire_pulse, 0).astype(np.float32)
    self.prev_desire[:] = desire_pulse

    inputs = {}
    for k, q in self.img_q.items():
      self._push(q, warped[k])
      inputs[k] = q[::self.frame_skip].reshape(1, -1, q.shape[2], q.shape[3])

    self._push(self.desire_q, desire.reshape(1, -1))
    self._push(self.feat_q, prev_feat.reshape(1, -1))
    dq = self.desire_q
    inputs["desire_pulse"] = dq.reshape(-1, self.frame_skip, *dq.shape[1:]).max(1).reshape(1, -1, dq.shape[2])
    fq = self.feat_q
    inputs["features_buffer"] = fq[::self.frame_skip].reshape(1, -1, fq.shape[2])
    inputs["traffic_convention"] = traffic_convention.reshape(1, -1).astype(np.float32)
    inputs["action_t"] = action_t.reshape(1, -1).astype(np.float32)
    return inputs


# ------------------------------------------------------------------------------ stage 2: model

def materialize_onnx(path: Path) -> tuple[Path, Path | None]:
  """onnxruntime wants a real file; the models are chunked in git (common/file_chunker.py)."""
  from openpilot.common.file_chunker import get_manifest_path, open_file_chunked

  if path.is_file() and not Path(get_manifest_path(str(path))).is_file():
    return path, None
  tmpdir = Path(tempfile.mkdtemp(prefix="validate_numerics-"))
  out = tmpdir / path.name
  with open(out, "wb") as f, open_file_chunked(str(path)) as src:
    shutil.copyfileobj(src, f)
  return out, tmpdir


def build_frame_plan(frames: dict, cameras: dict, want: int) -> tuple[list[int], bool]:
  """The longest run of consecutive frameIds we have everything for, capped at `want`."""
  have_wide = bool(cameras["big_img"])
  usable = sorted(fid for fid, st in frames.items()
                  if fid in cameras["img"] and (not have_wide or fid in cameras["big_img"])
                  and st["rpy_calib"] is not None and st["lat_delay"] is not None
                  and st["sensor"] and st["device_type"])
  if not usable:
    return [], have_wide

  best_start, best_len, start = 0, 1, 0
  for i in range(1, len(usable)):
    if usable[i] != usable[i - 1] + 1:
      start = i
    if i - start + 1 > best_len:
      best_start, best_len = start, i - start + 1
    if best_len >= want:
      break
  return usable[best_start:best_start + min(best_len, want)], have_wide


def stage_model(device: str, onnx_path: Path, data_dir: Path, route: str, want_frames: int) -> int:
  head("stage 2: eGPU vs onnxruntime CPUExecutionProvider on a recorded route")

  try:
    import onnxruntime as ort
  except ImportError:
    bad("onnxruntime is not installed", "there is no reference to compare the eGPU against, so this stage cannot run")
    info("fix", "install onnxruntime into the venv that runs modeld, then re-run; AGNOS 19.6 ships without it")
    return EXIT_FAIL

  from openpilot.common.realtime import DT_MDL
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  from openpilot.common.transformations.model import get_warp_matrix
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
  from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.tools.lib.framereader import FrameReader

  route_name, seg_dirs = resolve_segments(data_dir, route)
  info("route", route_name)
  info("segments available", f"{seg_dirs[0].name} .. {seg_dirs[-1].name} ({len(seg_dirs)} segments)")
  info("onnx", str(onnx_path))

  model_file, tmpdir = materialize_onnx(onnx_path)
  try:
    metadata = make_metadata_dict(str(model_file))
    input_shapes = metadata["input_shapes"]
    output_slices = metadata["output_slices"]
    info("model checkpoint", str(metadata["model_checkpoint"]))

    model_h, model_w = input_shapes["img"][2] * 2, input_shapes["img"][3] * 2
    frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ

    scan = RouteScan()
    plan: list[int] = []
    have_wide = False
    for i, seg in enumerate(seg_dirs):
      scan.add_segment(seg)
      plan, have_wide = build_frame_plan(scan.frames, scan.cameras, want_frames)
      info(f"scanned {seg.name}", f"{len(plan)}/{want_frames} consecutive frames")
      if len(plan) >= want_frames:
        seg_dirs = seg_dirs[:i + 1]
        break
    frames, cameras = scan.frames, scan.cameras

    if not cameras["img"]:
      bad("no narrow road camera frames in this route",
          "the main model stream and the DEVICE_CAMERAS lookup both come from the narrow camera; pick a route with fcamera.hevc")
      return EXIT_FAIL
    if len(plan) < want_frames:
      bad("not enough usable consecutive frames",
          f"longest run with calibration, logs and frames is {len(plan)}, need {want_frames}")
      return EXIT_FAIL
    if scan.long_actuator_delay is None:
      bad("no carParams in the rlogs read", "cannot reconstruct the model's action_t input")
      return EXIT_FAIL
    info("frame ids", f"{plan[0]} .. {plan[-1]} ({len(plan)} consecutive)")
    info("cameras", "fcamera.hevc + ecamera.hevc" if have_wide else "fcamera.hevc only (no wide camera in this route)")

    if not have_wide:
      cameras["big_img"] = cameras["img"]

    st0 = frames[plan[0]]
    dc = DEVICE_CAMERAS[(st0["device_type"], st0["sensor"])]
    # modeld's main stream is the narrow road camera when it exists; big_img is always the wide
    # one when there is a second stream (modeld.py main loop)
    main_intrinsics = dc.narrow_road.intrinsics
    extra_intrinsics = dc.wide_road.intrinsics if have_wide else dc.narrow_road.intrinsics
    info("device camera", f"{st0['device_type']} / {st0['sensor']}")

    readers: dict[tuple[str, Path], FrameReader] = {}
    nv12_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    ort_dtypes = {i.name: np.dtype(ORT_DTYPES[i.type]) for i in sess.get_inputs()}
    ort_out_names = [o.name for o in sess.get_outputs()]
    info("onnxruntime", f"{ort.__version__}, inputs {len(ort_dtypes)}, outputs {len(ort_out_names)}")
    missing = set(ort_dtypes) - POLICY_INPUT_NAMES
    if missing:
      raise LookupError(f"the model wants inputs this tool does not build: {sorted(missing)}")

    report_tinygrad()
    from tinygrad.device import Device
    from tinygrad.helpers import Context
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.tensor import Tensor
    runner = OnnxRunner(str(model_file))
    info("tinygrad device", Device[device].device)

    queues = PolicyInputs(input_shapes, frame_skip, ModelConstants.DESIRE_LEN)
    prev_feat = np.zeros(input_shapes["features_buffer"][2], dtype=np.float32)
    max_diff = dict.fromkeys(ort_out_names, 0.0)
    worst_frame = dict.fromkeys(ort_out_names, -1)
    slice_diff = dict.fromkeys(output_slices, 0.0)
    nonfinite: list[tuple[str, int, int, int]] = []

    st = time.monotonic()
    for n, fid in enumerate(plan):
      state = frames[fid]
      warped = {}
      for key, intrinsics, big in (("img", main_intrinsics, False), ("big_img", extra_intrinsics, True)):
        seg, seg_idx = cameras[key][fid]
        hevc = seg / ("ecamera.hevc" if (big and have_wide) else "fcamera.hevc")
        reader = readers.get((key, seg))
        if reader is None:
          readers = {k: v for k, v in readers.items() if k[0] != key}  # one decoder per camera, walking forward
          reader = readers[(key, seg)] = FrameReader(str(hevc), pix_fmt="nv12")
        cam_w, cam_h = reader.w, reader.h
        if (cam_w, cam_h) not in nv12_cache:
          nv12_cache[(cam_w, cam_h)] = get_nv12_info(cam_w, cam_h)
        stride, y_height, uv_height, _ = nv12_cache[(cam_w, cam_h)]
        buf = pack_nv12(reader.get(seg_idx), cam_w, cam_h, stride, y_height, uv_height)
        tfm = get_warp_matrix(state["rpy_calib"], intrinsics, big).astype(np.float32)
        warped[key] = frame_prepare(buf, tfm, (cam_w, cam_h), (model_w, model_h), stride, y_height, uv_height)

      desire_pulse = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
      desire_pulse[desire_index(*state["desire_state"])] = 1
      traffic_convention = np.zeros(2, dtype=np.float32)
      traffic_convention[int(state["is_rhd"])] = 1
      # modeld.py: one DT_MDL for the age of the frame, half of one for the middle of the interval
      lat_action_t = state["lat_delay"] + LAT_SMOOTH_SECONDS + DT_MDL + DT_MDL / 2
      long_action_t = scan.long_actuator_delay + LONG_SMOOTH_SECONDS + DT_MDL + DT_MDL / 2
      action_t = np.array([lat_action_t, long_action_t], dtype=np.float32)

      inputs = queues.step(warped, desire_pulse, traffic_convention, action_t, prev_feat)
      feeds = {name: np.ascontiguousarray(inputs[name], dtype=dt) for name, dt in ort_dtypes.items()}
      cpu_out = sess.run(ort_out_names, feeds)
      with Context(DEV=device):
        gpu_raw = runner({k: Tensor(v, device=device) for k, v in feeds.items()})
        gpu_out = [gpu_raw[name].numpy() for name in ort_out_names]

      for name, a, b in zip(ort_out_names, cpu_out, gpu_out, strict=True):
        # A non-finite output has to be caught here rather than folded into the max. np.max of an
        # array containing NaN is NaN, and every comparison against NaN is False -- so `d >
        # max_diff` never fires, max_diff keeps its 0.0 initial value, and an eGPU emitting
        # nothing but NaN is reported as a flawless pass. That is the exact failure this tool
        # exists to catch, so it is a hard stop, not a large number.
        if not np.all(np.isfinite(b)):
          nonfinite.append((name, fid, int(np.count_nonzero(~np.isfinite(b))), int(b.size)))
          continue
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        if d > max_diff[name]:
          max_diff[name], worst_frame[name] = d, fid
      flat_cpu, flat_gpu = cpu_out[0].reshape(-1), gpu_out[0].reshape(-1)
      for name, sl in output_slices.items():
        # max(0.0, nan) returns 0.0, so the same hole exists here. Guarded by the check above,
        # which has already recorded this frame and skipped its diffs.
        d = float(np.max(np.abs(flat_cpu[sl].astype(np.float64) - flat_gpu[sl].astype(np.float64))))
        if np.isfinite(d):
          slice_diff[name] = max(slice_diff[name], d)

      prev_feat = flat_cpu[output_slices["hidden_state"]].astype(np.float32)
      if (n + 1) % 100 == 0:
        info(f"{n + 1}/{len(plan)}", f"max abs diff so far {max(max_diff.values()):.6g}, {time.monotonic() - st:.0f}s")
  finally:
    if tmpdir is not None:
      shutil.rmtree(tmpdir, ignore_errors=True)

  head("result")
  print(f"  route          {route_name}")
  print(f"  segments read  {seg_dirs[0].name} .. {seg_dirs[-1].name}")
  print(f"  frame ids      {plan[0]} .. {plan[-1]} ({len(plan)} consecutive frames)")
  print(f"  model          {onnx_path.name} ({metadata['model_checkpoint']})")
  print(f"  eGPU           {Device[device].device}   reference: onnxruntime CPUExecutionProvider")
  print("")
  print("  max abs difference per output tensor")
  for name in ort_out_names:
    print(f"    {name:<24} {max_diff[name]:>14.6g}   worst at frameId {worst_frame[name]}")
  print("")
  print("  max abs difference per output slice (worst 8)")
  for name, d in sorted(slice_diff.items(), key=lambda kv: -kv[1])[:8]:
    print(f"    {name:<24} {d:>14.6g}")

  # Ahead of the verdict on purpose: a non-finite output is not a large difference to be compared
  # against a threshold, it is the tool's whole reason for existing. np.max over NaN is NaN and
  # every comparison against NaN is False, so folding it into `overall` would report a flawless
  # pass for an eGPU emitting nothing but NaN.
  if nonfinite:
    nf_name, nf_fid, nf_bad, nf_total = nonfinite[0]
    bad("non-finite model output", f"{len(nonfinite)} frame/output pair(s)")
    info("first occurrence", f"{nf_name} on frameId {nf_fid}: {nf_bad} of {nf_total} values not finite")
    return EXIT_FAIL

  overall = max(max_diff.values())
  print("")
  if overall <= PASS_MAX_DIFF:
    ok("PASS", f"max abs diff {overall:.6g} <= {PASS_MAX_DIFF}")
    return EXIT_OK
  if overall > FAIL_MAX_DIFF:
    bad("FAIL", f"max abs diff {overall:.6g} > {FAIL_MAX_DIFF} -- the eGPU's output is wrong")
    return EXIT_FAIL
  bad("INCONCLUSIVE", f"max abs diff {overall:.6g} is between {PASS_MAX_DIFF} and {FAIL_MAX_DIFF}")
  return EXIT_INCONCLUSIVE


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", choices=("all", "copyout", "model"), default="all")
  p.add_argument("--device", default="AMD", help="tinygrad device under test (default: AMD, the eGPU)")
  p.add_argument("--route", help="route name, optionally with a segment range: NAME or NAME/2:6")
  p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"default: {DEFAULT_DATA_DIR}")
  p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help=f"default: {DEFAULT_ONNX}")
  p.add_argument("--frames", type=int, default=MIN_FRAMES, help=f"consecutive frames to compare (default: {MIN_FRAMES})")
  p.add_argument("--copyout-iters", type=int, default=COPYOUT_ITERS)
  p.add_argument("--copyout-bytes", type=int, default=COPYOUT_BYTES)
  p.add_argument("--check-warp", action="store_true",
                 help="also prove the numpy warp matches tinygrad's, on CPU, before comparing models")
  args = p.parse_args()

  if args.stage in ("all", "model") and not args.route:
    p.error("--route is required for the model stage; a result nobody can reproduce is not a result")
  if args.stage in ("all", "model") and args.frames < MIN_FRAMES:
    p.error(f"--frames must be at least {MIN_FRAMES}: fewer frames cannot see a rare wrong output")

  # modeld sets these for the usbgpu path; the eGPU behaves differently without them
  os.environ.setdefault("GMMU", "0")
  os.environ.setdefault("HCQDEV_WAIT_TIMEOUT_MS", "3000")

  if args.stage in ("all", "copyout"):
    if not stage_copyout(args.device, args.copyout_iters, args.copyout_bytes):
      head("result")
      bad("FAIL", "the transfer path is broken; comparing model outputs over it would be meaningless")
      return EXIT_FAIL
  if args.stage == "copyout":
    return EXIT_OK

  if args.check_warp:
    from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
    model_file, tmpdir = materialize_onnx(args.onnx)
    try:
      shapes = make_metadata_dict(str(model_file))["input_shapes"]["img"]
    finally:
      if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
    _, seg_dirs = resolve_segments(args.data_dir, args.route)
    from openpilot.tools.lib.framereader import FrameReader
    fr = FrameReader(str(seg_dirs[0] / "fcamera.hevc"), pix_fmt="nv12")
    if not check_warp((fr.w, fr.h), (shapes[3] * 2, shapes[2] * 2)):
      return EXIT_FAIL

  return stage_model(args.device, args.onnx, args.data_dir, args.route, args.frames)


if __name__ == "__main__":
  sys.exit(main())
