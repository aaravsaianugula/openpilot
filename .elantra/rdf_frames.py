#!/usr/bin/env python3
"""Build a causal two-camera RDF replay packet from one real comma segment.

Frames are selected by recorded EncodeIdx.segmentId presentation indices, never
an assumed container FPS. Exact camera frame IDs pair within 25 ms. Input time is
the later of the two recorded EOF times; every dynamic input is joined as-of that
decision time. The raw images are decoded originals with only Venus padding added.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller_extract import decompress_log, load_schema, sha256
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

WIDTH, HEIGHT = 1344, 760
WARMUP_FRAMES = 132


class PacketError(RuntimeError):
  """Missing, corrupt or temporally inadmissible packet inputs."""


def pack_nv12(raw, width=WIDTH, height=HEIGHT):
  if width <= 0 or height <= 0 or width % 2 or height % 2 or len(raw) != width * height * 3 // 2:
    raise PacketError("NV12 input size must match positive even frame dimensions")
  stride, y_height, uv_height, size = get_nv12_info(width, height)
  padded = bytearray(size)
  pixels = np.frombuffer(raw, dtype=np.uint8)
  y = np.frombuffer(padded, dtype=np.uint8, count=stride*y_height).reshape(y_height, stride)
  uv = np.frombuffer(padded, dtype=np.uint8, count=stride*uv_height, offset=stride*y_height).reshape(uv_height, stride)
  y[:height, :width] = pixels[:width*height].reshape(height, width)
  uv[:height//2, :width] = pixels[width*height:].reshape(height//2, width)
  return padded


def pair_frames(narrow, wide):
  wide_by_id = {}
  for row in wide:
    if row["frame_id"] in wide_by_id:
      raise PacketError("duplicate wide frame ID")
    wide_by_id[row["frame_id"]] = row
  result = []
  for row in narrow:
    mate = wide_by_id.get(row["frame_id"])
    if mate is None or not row["valid"] or not mate["valid"]:
      raise PacketError(f"missing/invalid exact camera frame {row['frame_id']}")
    if abs(row["t"] - mate["t"]) > .025 + 1e-9:
      raise PacketError("exact camera pair exceeds 25 ms skew")
    result.append({"t": max(row["t"], mate["t"]), "narrow_t": row["t"], "wide_t": mate["t"],
                   "frame_id": row["frame_id"], "wide_frame_id": mate["frame_id"],
                   "road_index": row["segment_id"], "wide_index": mate["segment_id"]})
  if not result:
    raise PacketError("no selected camera pairs")
  for a, b in zip(result[:-1], result[1:], strict=True):
    if (b["frame_id"] != a["frame_id"] + 1 or b["road_index"] != a["road_index"] + 1
        or b["wide_index"] != a["wide_index"] + 1 or not .025 <= b["t"] - a["t"] <= .075):
      raise PacketError("camera pairs must be strictly chronological, contiguous 20 Hz source frames")
  return result


def asof(rows, t, max_age, name):
  if not rows:
    raise PacketError(f"missing recorded {name}")
  index = bisect.bisect_right([row["t"] for row in rows], t) - 1
  if index < 0:
    raise PacketError(f"only future {name} available")
  row = rows[index]
  age = t - row["t"]
  if not row["valid"]:
    raise PacketError(f"latest recorded {name} is invalid")
  if age > max_age + 1e-9:
    raise PacketError(f"recorded {name} is stale by {age:.6f}s")
  return row, age


def desire_index(state, direction, turn):
  # This is the checked DesireHelper final-state mapping, without model probabilities.
  if turn in ("turnLeft", "turnRight"):
    return 1 if turn == "turnLeft" else 2
  if turn != "none" or state not in ("off", "preLaneChange", "laneChangeStarting"):
    raise PacketError("unknown recorded lane-change/turn state")
  if state == "laneChangeStarting":
    if direction not in ("left", "right"):
      raise PacketError("unknown lane-change direction")
    return 3 if direction == "left" else 4
  return 0


def warps(rpy, device, sensor):
  camera = DEVICE_CAMERAS.get((device, sensor))
  if camera is None or not np.isfinite(rpy).all() or np.shape(rpy) != (3,):
    raise PacketError("unknown camera geometry or invalid recorded calibration")
  if camera.narrow_road.size != (WIDTH, HEIGHT) or camera.wide_road.size != (WIDTH, HEIGHT):
    raise PacketError("recorded camera geometry does not match 1344x760 frames")
  return (get_warp_matrix(rpy, camera.narrow_road.intrinsics, False).astype(np.float32),
          get_warp_matrix(rpy, camera.wide_road.intrinsics, True).astype(np.float32))


def read_log(path, schema_dir, segment_index):
  schema = load_schema(schema_dir)
  streams = defaultdict(list)
  static = {}
  for event in schema.Event.read_multiple_bytes(decompress_log(path), traversal_limit_in_words=2**31):
    name = event.which()
    t, valid = int(event.logMonoTime) * 1e-9, bool(event.valid)
    row = {"t": t, "valid": valid}
    if name in ("narrowRoadEncodeIdx", "wideRoadEncodeIdx"):
      data = getattr(event, name)
      if int(data.segmentNum) != segment_index or str(data.type) != "fullHEVC":
        continue
      eof = int(data.timestampEof) * 1e-9
      if eof <= 0:
        raise PacketError("EncodeIdx lacks a real capture EOF timestamp")
      row.update(t=eof, publication_t=t, frame_id=int(data.frameId), segment_id=int(data.segmentId))
    elif name == "carState":
      data = event.carState
      row.update(v=float(data.vEgo), left_blinker=bool(data.leftBlinker), right_blinker=bool(data.rightBlinker))
    elif name == "extrinsicsCalibration":
      data = event.extrinsicsCalibration
      row.update(rpy=np.array(data.rpyCalib), calibrated=str(data.calStatus) == "calibrated")
    elif name in ("driverMonitoringState", "driverMonitoringStateDEPRECATED"):
      row["is_rhd"] = bool(getattr(event, name).isRHD)
      name = "traffic"
    elif name == "modelV2":
      data = event.modelV2.meta
      row.update(state=str(data.laneChangeState), direction=str(data.laneChangeDirection))
      name = "desire_state"
    elif name == "modelDataV2SP":
      row["turn"] = str(event.modelDataV2SP.laneTurnDirection)
      name = "turn_state"
    elif name in ("lateralPlan", "lateralPlanDEPRECATED"):
      row["desire"] = str(getattr(event, name).desire)
      name = "direct_desire"
    elif name == "lateralDelay":
      row.update(delay=float(event.lateralDelay.lateralDelay), status=str(event.lateralDelay.status))
    elif name == "carParams":
      if not valid:
        raise PacketError("recorded carParams invalid")
      data = event.carParams
      static["cp"] = {"steer_delay": float(data.steerActuatorDelay), "long_delay": float(data.longitudinalActuatorDelay),
                      "fingerprint": str(data.carFingerprint), "publication_t": t}
      continue
    elif name == "initData":
      static.update(device=str(event.initData.deviceType), source_commit=str(event.initData.gitCommit),
                    source_dirty=bool(event.initData.dirty))
      continue
    elif name in ("narrowRoadCameraState", "wideRoadCameraState"):
      key = "narrow_sensor" if name.startswith("narrow") else "wide_sensor"
      static[key] = str(getattr(event, name).sensor)
      continue
    else:
      continue
    streams[name].append(row)
  for rows in streams.values():
    rows.sort(key=lambda row: row["t"])
  if "cp" not in static or static.get("device") != "mici" or static.get("narrow_sensor") != static.get("wide_sensor"):
    raise PacketError("target log lacks consistent recorded comma4 hardware configuration")
  return streams, static


def packet_inputs(pairs, streams, static, require_neutral=True):
  arrays = {key: np.asarray([p[key] for p in pairs]) for key in
            ("t", "narrow_t", "wide_t", "frame_id", "wide_frame_id", "road_index", "wide_index")}
  for key in ("tfm", "big_tfm", "desire", "traffic_convention", "action_t", "v_ego", "rpy_calib",
              "calibration_age", "calibration_valid", "car_state_age", "traffic_age", "desire_state_age",
              "lat_delay", "lat_delay_age", "desire_proof", "turn_state_valid"):
    arrays[key] = []
  desire_names = {"none": 0, "turnLeft": 1, "turnRight": 2, "laneChangeLeft": 3, "laneChangeRight": 4,
                  "keepLeft": 5, "keepRight": 6}
  for pair in pairs:
    t = pair["t"]
    state, state_age = asof(streams["carState"], t, .2, "carState")
    calib, calibration_age = asof(streams["extrinsicsCalibration"], t, .5, "calibration")
    traffic, traffic_age = asof(streams["traffic"], t, .2, "driverMonitoringState traffic convention")
    if not calib["calibrated"] or not np.isfinite(state["v"]):
      raise PacketError("uncalibrated camera or non-finite actual ego speed")
    if streams["direct_desire"]:
      direct, desire_age = asof(streams["direct_desire"], t, .2, "lateralPlan desire")
      if direct["desire"] not in desire_names:
        raise PacketError("unknown recorded direct desire")
      desire = desire_names[direct["desire"]]
      desire_proof = 0
    else:
      lane, lane_age = asof(streams["desire_state"], t, .2, "published prior lane-change state")
      if lane["state"] == "off" and lane["direction"] == "none" and not state["left_blinker"] and not state["right_blinker"]:
        # LaneTurnController.update_lane_turn's explicit else branch yields none.
        # DesireHelper then yields none when lane_change_state is off. No invalid
        # modelDataV2SP value is consumed to establish this neutral-only proof.
        desire, desire_age, desire_proof = 0, max(lane_age, state_age), 2
      else:
        turn, turn_age = asof(streams["turn_state"], t, .2, "published prior turn state")
        desire_age = max(lane_age, turn_age)
        desire = desire_index(lane["state"], lane["direction"], turn["turn"])
        desire_proof = 1
    if require_neutral and (desire != 0 or state["left_blinker"] or state["right_blinker"]):
      raise PacketError("requested neutral-desire packet contains a recorded maneuver/blinker")
    if streams["lateralDelay"]:
      delay, delay_age = asof(streams["lateralDelay"], t, .5, "recorded lateralDelay")
      lat_delay = delay["delay"]
      if delay["status"] == "invalid":
        raise PacketError("recorded lateral delay estimate invalid")
    else:
      lat_delay, delay_age = static["cp"]["steer_delay"], 0.
    if not np.isfinite([lat_delay, static["cp"]["long_delay"]]).all() or min(lat_delay, static["cp"]["long_delay"]) < 0:
      raise PacketError("invalid recorded physical controller delays")
    road, wide = warps(calib["rpy"], static["device"], static["narrow_sensor"])
    vec = np.zeros(8, dtype=np.float32)
    vec[desire] = 1
    convention = np.zeros(2, dtype=np.float32)
    convention[int(traffic["is_rhd"])] = 1
    turn_index = bisect.bisect_right([r["t"] for r in streams["turn_state"]], t) - 1
    turn_valid = float(streams["turn_state"][turn_index]["valid"]) if turn_index >= 0 else 0.
    row = {"tfm": road, "big_tfm": wide, "desire": vec, "traffic_convention": convention,
           "action_t": [lat_delay + .075, static["cp"]["long_delay"] + .075], "v_ego": max(0., state["v"]),
           "rpy_calib": calib["rpy"], "calibration_age": calibration_age, "calibration_valid": 1.,
           "car_state_age": state_age, "traffic_age": traffic_age, "desire_state_age": desire_age,
           "lat_delay": lat_delay, "lat_delay_age": delay_age, "desire_proof": desire_proof, "turn_state_valid": turn_valid}
    for key, value in row.items():
      arrays[key].append(value)
  for key in arrays:
    if key not in ("t", "narrow_t", "wide_t", "frame_id", "wide_frame_id", "road_index", "wide_index"):
      arrays[key] = np.asarray(arrays[key], dtype=np.float32)
    if not np.isfinite(arrays[key]).all():
      raise PacketError(f"non-finite required packet input {key}")
  arrays["lateral_control_params"] = np.stack([arrays["v_ego"], arrays["lat_delay"]], axis=1)
  return arrays


def prepare_output(out, segment):
  target, source = out.resolve(), segment.resolve()
  if target.is_relative_to(source) or target == source:
    raise PacketError("packet output cannot be inside the original segment")
  if target.exists() and (not target.is_dir() or any(target.iterdir())):
    raise PacketError("packet output must be a new or empty directory")
  target.mkdir(parents=True, exist_ok=True)
  return target


def decode_camera(source, indices, target):
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise PacketError("existing ffmpeg CLI is unavailable")
  if any(b != a + 1 for a, b in zip(indices[:-1], indices[1:], strict=True)):
    raise PacketError("decoder requires contiguous source presentation indices")
  first, last = int(indices[0]), int(indices[-1])
  # Avoid both Windows raw-frame pipe stalls and unread-stderr pipe deadlocks.
  # Temporary compact pixels are bounded by the requested exact frame count.
  with tempfile.TemporaryDirectory(prefix="decode-", dir=target.parent) as temporary:
    directory = Path(temporary).resolve()
    if not directory.is_relative_to(target.parent.resolve()):
      raise PacketError("temporary decoder directory escaped packet output")
    compact = directory / "compact.nv12"
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "2", "-filter_threads", "1", "-i", str(source),
               "-vf", f"select=between(n\\,{first}\\,{last})", "-fps_mode", "passthrough", "-frames:v", str(len(indices)),
               "-threads:v", "1", "-pix_fmt", "nv12", "-f", "rawvideo", str(compact)]
    with tempfile.TemporaryFile() as errors:
      try:
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors, timeout=120)
      except subprocess.TimeoutExpired as exc:
        raise PacketError("ffmpeg decode exceeded the 120s bound") from exc
      errors.seek(0)
      detail = errors.read(600).decode("utf-8", "replace")
      if process.returncode != 0:
        raise PacketError(f"ffmpeg decode failed exit{process.returncode}: {detail}")
    frame_bytes = WIDTH * HEIGHT * 3 // 2
    if not compact.is_file() or compact.stat().st_size != len(indices) * frame_bytes:
      raise PacketError("ffmpeg compact byte count differs from exact selected frames")
    with compact.open("rb") as pixels, target.open("xb") as output:
      for _ in indices:
        output.write(pack_nv12(pixels.read(frame_bytes)))
  return command


def available_memory_bytes():
  """Query current allocatable physical/commit memory; never assume free RAM."""
  if os.name == "nt":
    class MemoryStatus(ctypes.Structure):
      _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                           "virtual_total", "virtual_available", "extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    query = ctypes.windll.kernel32.GlobalMemoryStatusEx
    query.argtypes, query.restype = [ctypes.POINTER(MemoryStatus)], ctypes.c_int
    if not query(ctypes.byref(status)):
      raise PacketError("available memory query failed")
    return int(min(status.available, status.page_available))
  try:
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
  except (ValueError, OSError, AttributeError) as exc:
    raise PacketError("available memory query is unsupported; no RAM decoder fallback") from exc


def _compact_spec(indices, max_frames, max_bytes):
  values = np.asarray(indices)
  if (values.ndim != 1 or not len(values) or values.dtype.kind not in "iuf" or
      not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 2**31 - 1) or
      np.any(values != np.floor(values)) or np.any(np.diff(values) != 1)):
    raise PacketError("decoder requires nonnegative contiguous integer presentation indices")
  if WIDTH <= 0 or HEIGHT <= 0 or WIDTH % 2 or HEIGHT % 2:
    raise PacketError("decoder requires positive even frame dimensions")
  frame_bytes = WIDTH * HEIGHT * 3 // 2
  expected = len(values) * frame_bytes
  if max_frames < 1 or max_bytes < 1 or len(values) > max_frames or expected > max_bytes:
    raise PacketError("compact decode exceeds declared frame/byte limit")
  return values.astype(np.int64), frame_bytes, expected


def _compact_command(source, indices, target):
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise PacketError("existing ffmpeg CLI is unavailable")
  return [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "2", "-filter_threads", "1", "-i", str(source),
          "-vf", f"select=between(n\\,{int(indices[0])}\\,{int(indices[-1])})", "-fps_mode", "passthrough", "-frames:v", str(len(indices)),
          "-threads:v", "1", "-pix_fmt", "nv12", "-f", "rawvideo", str(target)]


def _decode_compact_ram(source, indices, expected, directory, timeout):
  required = 2 * expected + 512 * 1024**2
  available = available_memory_bytes()
  if available < required:
    raise PacketError(f"insufficient free memory: need{required} available{available}; no disk fallback")
  if not 0 < timeout <= 120:
    raise PacketError("RAM decoder timeout must be within the120s bound")
  try:
    pixels = bytearray(expected)
  except MemoryError as exc:
    raise PacketError("bounded compact RAM allocation failed; no disk fallback") from exc
  command = _compact_command(source, indices, "pipe:1")
  state = {"bytes": 0, "error": None}
  with tempfile.TemporaryFile(dir=directory) as errors:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)

    def drain():
      try:
        view = memoryview(pixels)
        while state["bytes"] < expected:
          start = state["bytes"]
          count = process.stdout.readinto(view[start:min(expected, start + 8 * 1024**2)])
          if not count:
            break
          state["bytes"] += count
        if state["bytes"] == expected and process.stdout.read(1):
          state["error"] = PacketError("ffmpeg compact byte count exceeds exact selected frames")
      except Exception as exc:
        state["error"] = exc

    reader = threading.Thread(target=drain, daemon=True)
    started = time.monotonic()
    reader.start()
    try:
      reader.join(timeout)
      if reader.is_alive():
        raise subprocess.TimeoutExpired(command, timeout)
      if state["error"] is not None and process.poll() is None:
        process.kill()
      process.wait(timeout=max(.001, timeout - (time.monotonic() - started)))
    except subprocess.TimeoutExpired as exc:
      process.kill()
      process.wait(timeout=5)
      reader.join(5)
      if reader.is_alive():
        raise PacketError("ffmpeg RAM decode timed out; drain thread did not terminate after kill, stream not closed") from exc
      raise PacketError(f"ffmpeg RAM decode exceeded the{timeout:g}s bound") from exc
    finally:
      if not reader.is_alive():
        process.stdout.close()
    errors.seek(0)
    detail = errors.read(600).decode("utf-8", "replace")
    if state["error"] is not None:
      raise PacketError(f"ffmpeg compact byte count/read failure: {state['error']}") from state["error"]
    if process.returncode != 0:
      raise PacketError(f"ffmpeg decode failed exit{process.returncode}: {detail}")
    if state["bytes"] != expected:
      raise PacketError("ffmpeg compact byte count differs from exact selected frames")
  return pixels, {"command": command, "decode_exit_code": process.returncode, "available_memory_bytes": available,
                  "required_free_memory_bytes": required, "compact_bytes": expected, "kind": "ram_decode"}


def _compress_frames(frames, target, expected_compact_sha=None):
  import zstandard

  native_hash, compact_hash = hashlib.sha256(), hashlib.sha256()
  native_bytes, created = 0, False
  try:
    with target.open("xb") as output:
      created = True
      with zstandard.ZstdCompressor(level=3).stream_writer(output, closefd=False) as compressed:
        for compact in frames:
          compact_hash.update(compact)
          padded = pack_nv12(compact)
          native_hash.update(padded)
          native_bytes += len(padded)
          compressed.write(padded)
    if expected_compact_sha is not None and compact_hash.hexdigest() != expected_compact_sha:
      raise PacketError("reused compact SHA differs from actual decode receipt")
  except Exception as exc:
    if created:
      failed = target.with_name(target.name + f".failed-{time.time_ns()}")
      target.rename(failed)
      with failed.with_name(failed.name + ".json").open("x", encoding="utf-8") as receipt:
        json.dump({"status": "failed_not_a_packet", "error": f"{type(exc).__name__}: {exc}",
                   "partial_compressed": str(failed.resolve()), "partial_compressed_sha256": sha256(failed),
                   "partial_compressed_bytes": failed.stat().st_size, "native_bytes_processed": native_bytes,
                   "compact_sha256_processed": compact_hash.hexdigest(), "expected_compact_sha256": expected_compact_sha}, receipt, indent=2)
    raise
  return {"native_sha256": native_hash.hexdigest(), "native_bytes": native_bytes,
          "compact_sha256": compact_hash.hexdigest(), "compressed_sha256": sha256(target),
          "compressed_bytes": target.stat().st_size}


def decode_camera_compressed(source, indices, target, *, max_frames=1200, max_bytes=2 * 1024**3, timeout=120):
  """Decode one bounded compact camera in RAM, then pad/compress without raw disk writes."""
  if target.exists():
    raise PacketError("compressed camera output already exists")
  indices, frame_bytes, expected = _compact_spec(indices, max_frames, max_bytes)
  pixels, receipt = _decode_compact_ram(source, indices, expected, target.parent, timeout)
  frames = (memoryview(pixels)[i * frame_bytes:(i + 1) * frame_bytes] for i in range(len(indices)))
  receipt.update(_compress_frames(frames, target))
  receipt.update(frames=len(indices), source_presentation_indices=[int(indices[0]), int(indices[-1])])
  return receipt


def reuse_camera_compact(source, indices, frame_ids, target, receipt_path, context, *, max_frames=1200, max_bytes=2 * 1024**3):
  """Reuse only a successful exact-index decode with current source/selection hashes.

  Hash compact pixels during direct padding/compression. A mismatch preserves a
  clearly failed output/receipt and cannot produce a completed packet. No fresh decode
  is claimed; its original executed command and proof receipt remain explicit.
  """
  indices, frame_bytes, expected = _compact_spec(indices, max_frames, max_bytes)
  receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  compact = Path(receipt["compact"])
  ids = np.asarray(frame_ids)
  digest = receipt.get("compact_sha256", "")
  if (ids.shape != indices.shape or ids.dtype.kind not in "iuf" or not np.isfinite(ids).all() or
      np.any(ids < 0) or np.any(ids != np.floor(ids)) or np.any(np.diff(ids) != 1) or
      not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
    raise PacketError("compact reuse requires exact frame IDs and a recorded SHA256")
  expected_fields = {"frames": len(indices), "width": WIDTH, "height": HEIGHT, "compact_bytes": expected,
                     "source_presentation_indices": [int(indices[0]), int(indices[-1])],
                     "source_frame_ids": [int(ids[0]), int(ids[-1])], **context}
  if (receipt.get("exit_code") != 0 or
      Path(receipt.get("source_camera", "")).resolve() != source.resolve() or
      any(receipt.get(key) != value for key, value in expected_fields.items())):
    raise PacketError("compact reuse receipt differs from successful source/hash/frame selection proof")
  command = receipt.get("command", [])
  if (not command or Path(command[0]).name.lower() not in ("ffmpeg", "ffmpeg.exe") or
      command[1:] != _compact_command(source, indices, compact)[1:]):
    raise PacketError("compact reuse command does not prove the exact NV12 decode")
  if not compact.is_file() or compact.stat().st_size != expected or compact.resolve() == target.resolve():
    raise PacketError("compact reuse file size/path differs from receipt")
  with compact.open("rb") as pixels:
    hashes = _compress_frames((pixels.read(frame_bytes) for _ in indices), target, digest)
  return {**hashes, "kind": "reused_predecoded_compact", "command": command, "decode_exit_code": receipt["exit_code"],
          "compact_bytes": expected, "frames": len(indices), "source_presentation_indices": expected_fields["source_presentation_indices"],
          "proof_receipt": str(receipt_path.resolve()), "proof_receipt_sha256": sha256(receipt_path),
          "compact_source": str(compact.resolve()), "decode_claim": "prior actual successful command reused; no fresh decode claimed"}


def compress(path):
  import zstandard
  target = path.with_suffix(path.suffix + ".zst")
  with path.open("rb") as src, target.open("xb") as dst:
    zstandard.ZstdCompressor(level=3).copy_stream(src, dst)
  return target


def build(segment, schema_dir, out, start, seconds, compressed=False, *, compressed_only=False,
          max_compact_frames=1200, max_compact_bytes=2 * 1024**3, reuse_road_compact_receipt=None,
          selection_receipt=None, camera_receipt=None):
  if not np.isfinite([start, seconds]).all() or start < 0 or seconds < 6.65:
    raise PacketError("nonnegative start and at least 6.65 seconds including real warmup required")
  if reuse_road_compact_receipt is not None and (not compressed_only or selection_receipt is None or camera_receipt is None):
    raise PacketError("compact reuse requires compressed-only plus selection and camera receipts")
  log = segment / "rlog.zst"
  cameras = [segment / "fcamera.hevc", segment / "ecamera.hevc"]
  if not all(p.is_file() for p in (log, *cameras)):
    raise PacketError("target segment must contain full rlog and both real road cameras")
  segment_index = int(segment.name.rsplit("--", 1)[1])
  streams, static = read_log(log, schema_dir, segment_index)
  narrow = sorted(streams["narrowRoadEncodeIdx"], key=lambda row: row["segment_id"])
  if not narrow:
    raise PacketError("missing narrow EncodeIdx")
  origin = narrow[0]["t"]
  chosen = [row for row in narrow if origin + start <= row["t"] < origin + start + seconds]
  pairs = pair_frames(chosen, streams["wideRoadEncodeIdx"])
  if len(pairs) < WARMUP_FRAMES + 1 or pairs[-1]["t"] - pairs[0]["t"] < seconds - .1:
    raise PacketError("requested chronological interval is unavailable or too short")
  arrays = packet_inputs(pairs, streams, static)
  if compressed_only:
    _compact_spec(arrays["road_index"], max_compact_frames, max_compact_bytes)
    _compact_spec(arrays["wide_index"], max_compact_frames, max_compact_bytes)
  source_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in (log, *cameras)}
  source_hashes = {p.name: sha256(p) for p in cameras}
  log_hash = sha256(log)
  output = prepare_output(out, segment)
  commands, decoder_receipts = [], {}
  for source, key, name in zip(cameras, ("road_index", "wide_index"), ("road.nv12", "wide.nv12"), strict=True):
    if compressed_only:
      if name == "road.nv12" and reuse_road_compact_receipt is not None:
        context = {"source_camera_sha256": source_hashes[source.name], "source_log_sha256": log_hash,
                   "selection_sha256": sha256(selection_receipt), "camera_receipt_sha256": sha256(camera_receipt)}
        receipt = reuse_camera_compact(source, arrays[key], arrays["frame_id"], output / (name + ".zst"),
                                       reuse_road_compact_receipt, context, max_frames=max_compact_frames, max_bytes=max_compact_bytes)
      else:
        receipt = decode_camera_compressed(source, arrays[key], output / (name + ".zst"),
                                           max_frames=max_compact_frames, max_bytes=max_compact_bytes)
      decoder_receipts[name] = receipt
      commands.append(receipt["command"])
    else:
      commands.append(decode_camera(source, arrays[key], output / name))
  np.savez_compressed(output / "packet.npz", **arrays)
  if compressed and not compressed_only:
    for name in ("road.nv12", "wide.nv12"):
      compress(output / name)
  if any(source_stats[p] != (p.stat().st_size, p.stat().st_mtime_ns) for p in source_stats):
    raise PacketError("raw camera/log source changed during packet build")
  compressed_hashes = {name + ".zst": receipt["compressed_sha256"] for name, receipt in decoder_receipts.items()}
  artifact_hashes = {p.name: compressed_hashes.get(p.name) or sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
  artifact_bytes = {p.name: p.stat().st_size for p in sorted(output.iterdir()) if p.is_file()}
  if compressed_only:
    for name, receipt in decoder_receipts.items():
      artifact_hashes[name], artifact_bytes[name] = receipt["native_sha256"], receipt["native_bytes"]
  stride, y_height, uv_height, size = get_nv12_info(WIDTH, HEIGHT)
  repo = Path(__file__).resolve().parent.parent
  commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
  gitlink = subprocess.run(["git", "-C", str(repo), "ls-tree", "HEAD", "opendbc_repo"],
                          capture_output=True, text=True, check=True).stdout.split()[2]
  metadata = {"format_version": 1, "frames": len(pairs), "width": WIDTH, "height": HEIGHT, "stride": stride,
    "y_height": y_height, "uv_height": uv_height, "buffer_size": size, "frame_copy_size": stride*(y_height+uv_height),
    "warmup_frames": WARMUP_FRAMES, "warmup_seconds": 6.6, "source_segment": str(segment.resolve()),
    "source_log_sha256": log_hash, "source_cameras_sha256": source_hashes,
    "source_commit": static["source_commit"], "repo_commit": commit, "opendbc_gitlink": gitlink,
    "source_dirty": static["source_dirty"], "source_opendbc_gitlink": None,
    "source_opendbc_identity_status": "unknown; packet builder does not infer archived opendbc identity from superproject commit",
    "opendbc_gitlink_basis": "current replay checkout only; separate from unknown archived-source opendbc gitlink",
    "schema_sha256": {str(p.relative_to(schema_dir)): sha256(p) for p in sorted(schema_dir.rglob("*.capnp"))},
    "calibration_status": "every frame calibrated, valid and causal-as-of; maximum age500ms",
    "camera_sensor": static["narrow_sensor"], "causal_join": "all dynamic inputs as-of max(narrow EOF,wide EOF); no future publication used",
    "camera_pairing": "exact recorded frame IDs; source presentation segmentId; no repeats/interpolation; max25ms capture skew",
    "max_wide_age_s": .025, "max_actual_camera_skew_s": float(np.max(abs(arrays["narrow_t"] - arrays["wide_t"]))),
    "desire_source": "proof0:direct recorded plan;1:valid as-of DH state;2:fresh valid blinkers off+lane state off proves neutral, invalid SP not consumed",
    "desire_proof_counts": {str(int(k)): int(np.sum(arrays["desire_proof"] == k)) for k in np.unique(arrays["desire_proof"])},
    "turn_state_invalid_packet_rows": int(np.sum(arrays["turn_state_valid"] != 1)),
    "desire_proof_source_sha256": {str(p.relative_to(repo)): sha256(p) for p in
      (repo / "openpilot/selfdrive/controls/lib/desire_helper.py", repo / "openpilot/sunnypilot/selfdrive/controls/lib/lane_turn_desire.py")},
    "traffic_source": "as-of recorded driverMonitoringState.isRHD, matching production modeld; not assumed from vehicle market",
    "action_t_basis": "as-of learned lateralDelay else CPsteer; CPlong; each+.075 production frame/action offset; no model smoothing assumed",
    "static_car_params": static["cp"],
    "static_config_timing": "hardware CP file snapshot may publish after camera interval; static hardware config, never future-motion GT",
    "capture_interval": [float(arrays["t"][0]), float(arrays["t"][-1])], "segment_start_eof": origin,
    "requested_start_s": start, "requested_seconds": seconds, "ffmpeg_commands": commands,
    "host_storage": {"mode": "compressed_only" if compressed_only else "raw_and_optional_compressed",
      "stored_files": sorted(p.name for p in output.iterdir() if p.is_file()),
      "raw_artifact_receipt_basis": "computed while padding; device expansion verifies" if compressed_only else "host files hashed",
      "raw_files_present": not compressed_only},
    "compact_decode_limits": {"max_frames": max_compact_frames, "max_bytes": max_compact_bytes,
      "memory_policy": "one compact camera buffer; require2xcompactbytes+512MiB free; no disk fallback"} if compressed_only else None,
    "decoder_receipts": decoder_receipts,
    "padding": "only zero-added Venus stride/plane/kernel storage; original decoded Y and UV pixels preserved",
    "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes,
    "scope": "real chronological offline replay inputs; no GT/future positions supplied; no device or production configuration changed"}
  (output / "packet.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  return metadata


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  command = sub.add_parser("build")
  command.add_argument("--segment", type=Path, required=True)
  command.add_argument("--schema-dir", type=Path, required=True)
  command.add_argument("--out", type=Path, required=True)
  command.add_argument("--start", type=float, required=True)
  command.add_argument("--seconds", type=float, default=12)
  command.add_argument("--compress", action="store_true")
  command.add_argument("--compressed-only", action="store_true", help="Bounded compact RAM decode and direct zstd; no host raw NV12 writes")
  command.add_argument("--max-compact-frames", type=int, default=1200)
  command.add_argument("--max-compact-bytes", type=int, default=2 * 1024**3)
  command.add_argument("--reuse-road-compact-receipt", type=Path)
  command.add_argument("--selection-receipt", type=Path)
  command.add_argument("--camera-receipt", type=Path)
  args = parser.parse_args()
  metadata = build(args.segment, args.schema_dir, args.out, args.start, args.seconds, args.compress,
                   compressed_only=args.compressed_only, max_compact_frames=args.max_compact_frames, max_compact_bytes=args.max_compact_bytes,
                   reuse_road_compact_receipt=args.reuse_road_compact_receipt, selection_receipt=args.selection_receipt,
                   camera_receipt=args.camera_receipt)
  print(json.dumps({"packet": str((args.out / "packet.json").resolve()), "frames": metadata["frames"],
                    "buffer_size": metadata["buffer_size"], "interval": metadata["capture_interval"],
                    "max_camera_skew_s": metadata["max_actual_camera_skew_s"]}))


if __name__ == "__main__":
  main()
