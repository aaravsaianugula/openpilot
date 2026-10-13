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
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

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


def compress(path):
  import zstandard
  target = path.with_suffix(path.suffix + ".zst")
  with path.open("rb") as src, target.open("xb") as dst:
    zstandard.ZstdCompressor(level=3).copy_stream(src, dst)
  return target


def build(segment, schema_dir, out, start, seconds, compressed=False):
  if not np.isfinite([start, seconds]).all() or start < 0 or seconds < 6.65:
    raise PacketError("nonnegative start and at least 6.65 seconds including real warmup required")
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
  output = prepare_output(out, segment)
  commands = []
  for source, key, name in zip(cameras, ("road_index", "wide_index"), ("road.nv12", "wide.nv12"), strict=True):
    commands.append(decode_camera(source, arrays[key], output / name))
  np.savez_compressed(output / "packet.npz", **arrays)
  if compressed:
    for name in ("road.nv12", "wide.nv12"):
      compress(output / name)
  stride, y_height, uv_height, size = get_nv12_info(WIDTH, HEIGHT)
  repo = Path(__file__).resolve().parent.parent
  commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
  gitlink = subprocess.run(["git", "-C", str(repo), "ls-tree", "HEAD", "opendbc_repo"],
                          capture_output=True, text=True, check=True).stdout.split()[2]
  metadata = {"format_version": 1, "frames": len(pairs), "width": WIDTH, "height": HEIGHT, "stride": stride,
    "y_height": y_height, "uv_height": uv_height, "buffer_size": size, "frame_copy_size": stride*(y_height+uv_height),
    "warmup_frames": WARMUP_FRAMES, "warmup_seconds": 6.6, "source_segment": str(segment.resolve()),
    "source_log_sha256": sha256(log), "source_cameras_sha256": {p.name: sha256(p) for p in cameras},
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
    "padding": "only zero-added Venus stride/plane/kernel storage; original decoded Y and UV pixels preserved",
    "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    "artifact_bytes": {p.name: p.stat().st_size for p in sorted(output.iterdir()) if p.is_file()},
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
  args = parser.parse_args()
  metadata = build(args.segment, args.schema_dir, args.out, args.start, args.seconds, args.compress)
  print(json.dumps({"packet": str((args.out / "packet.json").resolve()), "frames": metadata["frames"],
                    "buffer_size": metadata["buffer_size"], "interval": metadata["capture_interval"],
                    "max_camera_skew_s": metadata["max_actual_camera_skew_s"]}))


if __name__ == "__main__":
  main()
