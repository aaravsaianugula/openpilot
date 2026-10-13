"""Render timestamp-matched recorded footage and explicitly offline controller forecasts.

This tool never obtains frames from a guessed camera rate. EncodeIdx.segmentId
is presentation order inside fcamera.hevc; timestampEof is its recorded clock.
The MP4 output rate is a viewing choice, independently recorded in the receipt.
"""

import argparse
import json
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GT = "#77edac"
BASELINE = "#67b7ff"
TUNED = "#ffbd69"
MODEL = "#d3a0ff"
FG = "#eaf0f6"
MUTED = "#99aaba"
BG = "#0e151e"
PANEL = "#18232f"
MAX_CAMERA_GAP = 0.15


class RenderError(ValueError):
  """Missing or ambiguous evidence must stop the render."""


def camera_mapping(data, segment_num):
  """Return source-frame indices and timestamps, refusing inferred frame order."""
  keys = ("raw_encode_segment_id", "raw_encode_t")
  if any(key not in data for key in keys):
    raise RenderError("raw narrowRoadEncodeIdx segmentId/timestampEof mapping is required")
  indices = np.asarray(data[keys[0]], dtype=float)
  times = np.asarray(data[keys[1]], dtype=float)
  if indices.ndim != 1 or indices.shape != times.shape:
    raise RenderError("EncodeIdx arrays must be matching one-dimensional arrays")
  selected = np.ones(len(times), dtype=bool)
  if "raw_encode_segment_num" in data:
    selected &= np.asarray(data["raw_encode_segment_num"]) == segment_num
  indices, times = indices[selected], times[selected]
  if not len(times) or not np.isfinite(indices).all() or not np.isfinite(times).all():
    raise RenderError("EncodeIdx has no finite frames for this segment")
  if np.any(indices < 0) or np.any(indices != np.floor(indices)):
    raise RenderError("EncodeIdx segmentId must be nonnegative integer presentation indices")
  order = np.argsort(indices, kind="stable")
  indices, times = indices[order].astype(int), times[order]
  unique_indices, unique_times = [], []
  for index, timestamp in zip(indices, times, strict=True):
    if unique_indices and index == unique_indices[-1]:
      if timestamp != unique_times[-1]:
        raise RenderError(f"conflicting EncodeIdx timestamps for source frame {index}")
      continue
    unique_indices.append(index)
    unique_times.append(timestamp)
  indices, times = np.asarray(unique_indices), np.asarray(unique_times)
  if not np.array_equal(indices, np.arange(len(indices))):
    raise RenderError("missing EncodeIdx presentation indices; exact source mapping is incomplete")
  if np.any(np.diff(times) <= 0):
    raise RenderError("EncodeIdx timestamps must be strictly increasing in presentation order")
  return indices, times


def frame_schedule(mapping, start, seconds, fps):
  if seconds <= 0 or fps <= 0:
    raise RenderError("seconds and fps must be positive")
  source_indices, source_times = mapping
  count = int(math.ceil(seconds * fps - 1e-9))
  times = start + np.arange(count) / fps
  positions = np.searchsorted(source_times, times, side="right") - 1
  if np.any(positions < 0) or times[-1] > source_times[-1] + MAX_CAMERA_GAP:
    raise RenderError("requested interval exceeds recorded camera timestamps")
  errors = times - source_times[positions]
  if np.any(errors > MAX_CAMERA_GAP):
    raise RenderError("camera timestamp gap exceeds 150 ms; no interpolation across gaps")
  return times, source_indices[positions]


def contiguous_path(xy, valid=None):
  xy = np.asarray(xy, dtype=float)
  if xy.ndim != 2 or xy.shape[1] != 2:
    raise RenderError("trajectory must have shape [horizon,2]")
  good = np.isfinite(xy).all(axis=1)
  if valid is not None:
    good &= np.asarray(valid, dtype=bool)
  bad = np.flatnonzero(~good)
  return xy[:int(bad[0]) if len(bad) else len(xy)]


def reanchor_paths(paths, offset):
  """Express all forecasts in the measured ego frame at the displayed timestamp."""
  gt = paths[0]
  if offset < 0 or offset >= len(gt) - 1:
    return [np.empty((0, 2)) for _ in paths]
  direction = gt[offset + 1] - gt[max(offset - 1, 0)]
  heading = math.atan2(direction[1], direction[0])
  c, s = math.cos(heading), math.sin(heading)
  rotation = np.array([[c, -s], [s, c]])
  return [(path[offset:] - gt[offset]) @ rotation if len(path) > offset else np.empty((0, 2)) for path in paths]


def project_ground(xy, intrinsics, rpy, height):
  """Project x-forward/y-right ground coordinates; height is an explicit assumption."""
  roll, pitch, yaw = rpy
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rotation = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                       [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                       [-sp, cp * sr, cp * cr]])
  device = np.column_stack((xy, np.zeros(len(xy)))) @ rotation.T
  view = device[:, [1, 2, 0]] + [0, height, 0]
  pixels = view @ np.asarray(intrinsics).T
  positive = view[:, 2] > 1.
  uv = np.full((len(xy), 2), np.nan)
  uv[positive] = pixels[positive, :2] / pixels[positive, 2, None]
  return uv


def causal_index(times, timestamp, max_age=0.2):
  times = np.asarray(times)
  i = int(np.searchsorted(times, timestamp, side="right") - 1)
  return i if i >= 0 and timestamp - times[i] <= max_age else None


def forecast_paths(replay, timestamp):
  index = causal_index(replay["forecast_t"], timestamp, 0.55)
  if index is None:
    return [np.empty((0, 2)) for _ in range(3)], None
  valid = replay["forecast_valid"][index] if "forecast_valid" in replay else None
  paths = [contiguous_path(replay[key][index], valid) for key in ("gt_xy", "baseline_xy", "tuned_xy")]
  # A missing candidate must not hide measured GT. Each forecast is limited to
  # the contiguous measured horizon, and its own first missing point.
  paths = [path[:len(paths[0])] for path in paths]
  dt = float(replay["forecast_dt"])
  offset = int(math.floor((timestamp - replay["forecast_t"][index]) / dt + 1e-7))
  return reanchor_paths(paths, offset), float(replay["forecast_t"][index])


@lru_cache(maxsize=20)
def font(size):
  for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
    try:
      return ImageFont.truetype(name, size)
    except OSError:
      continue
  return ImageFont.load_default(size=size)


def text(draw, xy, label, size=18, fill=FG):
  draw.text(xy, label, font=font(size), fill=fill)


def draw_line(draw, points, color, width=3, bounds=None):
  previous = None
  for point in np.asarray(points):
    good = np.isfinite(point).all()
    if bounds is not None and good:
      good = bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]
    if good and previous is not None:
      draw.line([tuple(previous), tuple(point)], fill=color, width=width)
    previous = point if good else None


def plot_trace(draw, box, title, series, current, limits, rail=None):
  left, top, right, bottom = box
  draw.rounded_rectangle(box, radius=12, fill=PANEL)
  text(draw, (left + 12, top + 10), title, 19)
  x0, x1, y0, y1 = left + 38, right - 14, top + 48, bottom - 26
  lower, upper = limits
  begin, end = current - 6., current + 0.3
  def transform(t, v):
    return np.column_stack((x0 + (t - begin) / (end - begin) * (x1 - x0),
                            y1 - (v - lower) / (upper - lower) * (y1 - y0)))
  for value in (lower, 0, upper):
    y = y1 - (value - lower) / (upper - lower) * (y1 - y0)
    if y0 <= y <= y1:
      draw.line((x0, y, x1, y), fill="#344454")
      text(draw, (left + 3, y - 7), f"{value:g}", 12, MUTED)
  if rail is not None:
    for value in (-rail, rail):
      y = y1 - (value - lower) / (upper - lower) * (y1 - y0)
      draw.line((x0, y, x1, y), fill="#ff7272", width=1)
  for timestamps, values, color in series:
    timestamps = np.asarray(timestamps)
    selected = (timestamps >= begin) & (timestamps <= current) & np.isfinite(timestamps)
    # Keeping NaN values in the path breaks lines at gaps rather than connecting them.
    draw_line(draw, transform(timestamps[selected], np.asarray(values)[selected]), color, bounds=(x0, y0, x1, y1))
  draw.line((x1 - 15, y0, x1 - 15, y1), fill=MUTED)
  text(draw, (x0, bottom - 23), "past 6 s", 12, MUTED)


def model_path(data, timestamp):
  if "model_t" not in data or "model_x" not in data:
    return np.empty((0, 2))
  i = causal_index(data["t"], timestamp)
  if i is None or not np.isfinite(data["model_t"][i]) or timestamp - data["model_t"][i] > .2:
    return np.empty((0, 2))
  xy = np.column_stack((data["model_x"][i], data["model_y"][i]))
  horizon = np.asarray(data["model_horizon"][i])
  valid = np.isfinite(horizon) & (horizon >= 0)
  if len(horizon) > 1:
    valid[1:] &= np.diff(horizon) > 0
  return contiguous_path(xy, valid)


def scalar_at(data, key, timestamp):
  i = causal_index(data["t"], timestamp)
  return float(data[key][i]) if i is not None and key in data else float("nan")


def display_value(value, precision=1):
  return f"{value:.{precision}f}" if np.isfinite(value) else "missing"


def projection_config(data, metadata, source_size):
  if "camera_intrinsics" in data:
    intrinsics = np.asarray(data["camera_intrinsics"], dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
      raise RenderError("camera_intrinsics must be a finite 3x3 matrix")
    return intrinsics, "logged sensor intrinsics"
  sensor = metadata.get("camera_sensor")
  if sensor is None:
    sensor = metadata.get("camera", {}).get("sensor")
  # These narrow-camera constants match common/transformations/camera.py. The
  # native dimensions must agree, so an unknown crop is never silently resized.
  configs = {"os04c10": (1344, 760, 1141.5), "ar0231": (1928, 1208, 2648.), "ox03c10": (1928, 1208, 2648.)}
  if sensor in configs and tuple(source_size) == configs[sensor][:2]:
    width, height, focal = configs[sensor]
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]), f"{sensor} source intrinsics"
  width, height = source_size
  return np.array([[width * .85, 0, width / 2], [0, width * .85, height / 2], [0, 0, 1]]), "NOMINAL intrinsics"


def calibration_at(data, timestamp, assumed_height):
  i = causal_index(data["t"], timestamp)
  required = ("rpy_calib", "calibration_t", "calibration_height", "calibrated", "calibration_valid")
  if i is not None and all(key in data for key in required):
    age = timestamp - float(data["calibration_t"][i])
    rpy = np.asarray(data["rpy_calib"][i])
    height = float(np.asarray(data["calibration_height"][i]).reshape(-1)[0])
    if 0 <= age <= .5 and data["calibrated"][i] == 1 and data["calibration_valid"][i] == 1 \
        and np.isfinite(rpy).all() and np.isfinite(height) and height > 0:
      return rpy, height, True
  return np.zeros(3), assumed_height, False


def compose(frame, data, replay, timestamp, origin, segment, intrinsics, projection_label, ground_height):
  canvas = Image.new("RGB", (1600, 1000), BG)
  draw = ImageDraw.Draw(canvas)
  text(draw, (24, 10), "CN7 / comma 4    CONTROLLER EXPERIMENT", 24)
  text(draw, (24, 43), "OFFLINE SIMULATION — recorded targets; no model feedback • tuned controller was NOT driven", 20, TUNED)
  status = " | UNTRUSTED PLANT: diagnostic only" if bool(replay.get("diagnostic_only", False)) else ""
  if not bool(replay.get("lateral_model_available", True)):
    status += " | lateral response unavailable"
  text(draw, (24, 74), f"{segment}     recorded time +{timestamp - origin:05.2f} s" + status, 17, MUTED)
  source_width, source_height = frame.size
  frame = frame.resize((1040, 588))
  canvas.paste(frame, (16, 112))
  draw = ImageDraw.Draw(canvas)
  paths, anchor = forecast_paths(replay, timestamp)
  recorded_model = model_path(data, timestamp)
  all_paths = [*paths, recorded_model]
  colors = (GT, BASELINE, TUNED, MODEL)
  rpy, ground_height, logged_calibration = calibration_at(data, timestamp, ground_height)
  rotation_label = "logged calibration rotation" if logged_calibration else "NOMINAL rotation"
  calibrated = "NOMINAL" not in projection_label and logged_calibration
  for path, color in zip(all_paths, colors, strict=True):
    uv = project_ground(path, intrinsics, rpy, ground_height)
    uv[:, 0] = uv[:, 0] * 1040 / source_width + 16
    uv[:, 1] = uv[:, 1] * 588 / source_height + 112
    draw_line(draw, uv, color, 4, (16, 112, 1056, 700))
  draw.rectangle((16, 656, 1056, 700), fill="#0e151e")
  projection_mode = "Calibrated flat-ground projection" if calibrated else "NOMINAL ground projection"
  height_label = "logged height" if logged_calibration else "assumed height"
  text(draw, (26, 661), f"{projection_mode} ({ground_height:.2f} m {height_label})", 17, TUNED)
  text(draw, (26, 682), f"{projection_label}; {rotation_label}", 13, MUTED)
  if len(recorded_model) == 0:
    draw.rectangle((16, 112, 378, 141), fill=BG)
    text(draw, (26, 116), "Recorded model trajectory unavailable", 15, MODEL)
  draw.rounded_rectangle((1072, 112, 1584, 700), radius=12, fill=PANEL)
  text(draw, (1092, 128), "BEV / measured ego frame", 22)
  text(draw, (1092, 158), "+x forward • +y right • metres", 15, MUTED)
  x0, y0, scale_x, scale_y = 1328, 654, 13., 6.5
  for x in range(0, 71, 10):
    py = y0 - x * scale_y
    draw.line((1095, py, 1562, py), fill="#344454")
    text(draw, (1096, py - 19), str(x), 13, MUTED)
  for y in (-15, -10, -5, 0, 5, 10, 15):
    px = x0 + y * scale_x
    draw.line((px, 192, px, y0), fill="#344454")
    text(draw, (px - 9, 662), str(y), 12, MUTED)
  for path, color in zip(all_paths, colors, strict=True):
    points = np.column_stack((x0 + path[:, 1] * scale_x, y0 - path[:, 0] * scale_y))
    draw_line(draw, points, color, 4, (1095, 192, 1562, 655))
  draw.polygon([(x0, y0 - 14), (x0 - 7, y0 + 5), (x0 + 7, y0 + 5)], fill=FG)
  horizon = max(len(paths[0]) - 1, 0) * float(replay["forecast_dt"])
  caption = f"valid measured horizon {horizon:.2f} s" if anchor is not None else "forecast unavailable at this time"
  if len(paths[2]) == 0:
    caption += " | tuned unavailable"
  text(draw, (1092, 682), caption, 14, MUTED)
  for x, label, color in ((24, "Measured future GT", GT), (350, "Baseline simulation", BASELINE),
                           (690, "Tuned simulation", TUNED), (1020, "Recorded model path", MODEL)):
    draw.line((x, 728, x + 25, 728), fill=color, width=4)
    text(draw, (x + 35, 714), label, 19, color)
  interventions, missing_flags = [], []
  for key, label in (("steering_pressed", "DRIVER STEERING"), ("brake_pressed", "BRAKE"),
                      ("gas_pressed", "GAS"), ("eps_fault", "EPS FAULT")):
    value = scalar_at(data, key, timestamp)
    if not np.isfinite(value):
      missing_flags.append(label)
    elif value > .5:
      interventions.append(label)
  state = ", ".join(interventions) if interventions else "none in available flags"
  if missing_flags:
    state += " | missing: " + ", ".join(missing_flags)
  text(draw, (24, 748), "Interventions: " + state, 15, FG)
  lat = scalar_at(data, "lat_active", timestamp)
  long = scalar_at(data, "long_active", timestamp)
  text(draw, (900, 748), f"lat active {display_value(lat, 0)} | long active {display_value(long, 0)}", 17, MUTED)
  text(draw, (24, 767), "Green = measured. Blue/orange = torque & acceleration COMMANDS; yaw = plant simulation; speed simulated only with valid long fit.",
       11, MUTED)
  measured_torque = data["can_torque"] if "can_torque" in data else data["torque"]
  traces = [
    ("Torque counts / rail ±409", measured_torque, "baseline_torque", "tuned_torque", (-450, 450), 409),
    ("Accel m/s²: GT + commands", data["a"], "baseline_accel", "tuned_accel", (-4, 3), None),
    ("Speed m/s", data["v"], "baseline_v", "tuned_v", (0, max(30, float(np.nanmax(data["v"])) + 2)), None),
    ("Yaw rate rad/s", data["yaw"], "baseline_yaw", "tuned_yaw", (-.8, .8), None),
  ]
  for j, (title, measured, baseline, tuned, limits, rail) in enumerate(traces):
    left = 16 + 396 * j
    # Replay and data can have different clocks/grids: plot each on its own time.
    series = [(data["t"], measured, GT), (replay["t"], replay[baseline], BASELINE), (replay["t"], replay[tuned], TUNED)]
    plot_trace(draw, (left, 782, left + 382, 984), title, series, timestamp, limits, rail)
  return canvas, horizon, calibrated


def probe_video(path):
  result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(path)],
                          capture_output=True, text=True, check=True)
  streams = json.loads(result.stdout)["streams"]
  if not streams or "nb_read_frames" not in streams[0]:
    raise RenderError("ffprobe did not confirm a decoded frame count")
  return streams[0]


def read_exact(stream, count):
  chunks = bytearray()
  while len(chunks) < count:
    part = stream.read(count - len(chunks))
    if not part:
      break
    chunks.extend(part)
  return bytes(chunks)


def validate_inputs(data, replay):
  for source, keys in ((data, ("t", "v", "a", "yaw")),
                       (replay, ("t", "baseline_torque", "tuned_torque", "baseline_accel", "tuned_accel",
                                 "baseline_yaw", "tuned_yaw", "baseline_v", "tuned_v"))):
    missing = [key for key in keys if key not in source]
    if missing:
      raise RenderError("missing telemetry arrays: " + ", ".join(missing))
    timestamps = source["t"]
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
      raise RenderError("telemetry timestamps must be finite and strictly increasing")
    if any(np.asarray(source[key]).shape != timestamps.shape for key in keys):
      raise RenderError("telemetry arrays must match their source timestamp grid")
  if "torque" not in data and "can_torque" not in data:
    raise RenderError("measured torque or can_torque array is required")
  missing = [key for key in ("forecast_t", "forecast_dt", "gt_xy", "baseline_xy", "tuned_xy") if key not in replay]
  if missing:
    raise RenderError("missing forecast arrays: " + ", ".join(missing))
  dt = float(replay["forecast_dt"])
  times = replay["forecast_t"]
  if not np.isfinite(dt) or dt <= 0 or times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
    raise RenderError("forecast timestamps/step must be finite, positive and increasing")
  shape = replay["gt_xy"].shape
  if len(shape) != 3 or shape[0] != len(times) or shape[2] != 2:
    raise RenderError("forecast trajectories must have shape [anchors,horizon,2]")
  if any(replay[key].shape != shape for key in ("baseline_xy", "tuned_xy")):
    raise RenderError("forecast trajectories must have matching shapes")


def validate_output_path(output, segment, data_path, replay_path):
  targets = (output.resolve(), output.with_suffix(".json").resolve())
  raw_directory = segment.resolve()
  inputs = [segment / "fcamera.hevc", data_path, data_path.with_suffix(".json"), replay_path, replay_path.with_suffix(".json")]
  for target in targets:
    if target.is_relative_to(raw_directory):
      raise RenderError("output and receipt must be outside the raw source segment directory")
    for source in inputs:
      source = source.resolve()
      if target == source or (target.exists() and source.exists() and target.samefile(source)):
        raise RenderError(f"output/receipt collision with input: {source}")
  if output.suffix.lower() != ".mp4":
    raise RenderError("output must use the .mp4 extension")


def render(args):
  from controller_extract import hold_calibration

  for executable in ("ffmpeg", "ffprobe"):
    if shutil.which(executable) is None:
      raise RenderError(f"{executable} is not installed")
  validate_output_path(args.out, args.segment, args.data, args.replay)
  data = hold_calibration(dict(np.load(args.data, allow_pickle=False)))
  replay = dict(np.load(args.replay, allow_pickle=False))
  validate_inputs(data, replay)
  metadata_file = args.data.with_suffix(".json")
  metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
  source = args.segment / "fcamera.hevc"
  if not source.is_file():
    raise RenderError(f"missing narrow camera footage: {source}")
  segment_num = int(args.segment.name.rsplit("--", 1)[1])
  mapping = camera_mapping(data, segment_num)
  origin = float(mapping[1][0])
  times, frame_ids = frame_schedule(mapping, origin + args.start, args.seconds, args.fps)
  probe = probe_video(source)
  width, height = int(probe["width"]), int(probe["height"])
  count = int(probe["nb_read_frames"])
  if count != len(mapping[0]):
    raise RenderError(f"decoded source has {count} frames but EncodeIdx maps {len(mapping[0])}; refusing ambiguous mapping")
  intrinsics, projection_label = projection_config(data, metadata, (width, height))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  decode_cmd = ["ffmpeg", "-v", "error", "-i", str(source), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
  encode_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1600x1000",
                "-r", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
  receipt_frames, horizons, calibrated_frames = [], [], 0
  with subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as decoder, \
       subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
    try:
      source_index = -1
      frame = None
      for output_index, (timestamp, wanted) in enumerate(zip(times, frame_ids, strict=True)):
        while source_index < wanted:
          raw = read_exact(decoder.stdout, width * height * 3)
          source_index += 1
          if len(raw) != width * height * 3:
            raise RenderError(f"decoder ended before mapped source frame {wanted}")
          frame = Image.frombytes("RGB", (width, height), raw)
        source_t = float(mapping[1][wanted])
        canvas, horizon, calibrated = compose(frame, data, replay, source_t, origin, args.segment.name,
                                              intrinsics, projection_label, args.ground_height)
        encoder.stdin.write(canvas.tobytes())
        horizons.append(horizon)
        calibrated_frames += calibrated
        receipt_frames.append({"output_frame": output_index, "source_frame": int(wanted), "requested_t": float(timestamp),
                               "source_t": source_t, "overlay_t": source_t, "age_ms": float((timestamp - source_t) * 1000),
                               "calibrated_flat_ground_projection": calibrated})
      encoder.stdin.close()
      if encoder.wait() != 0:
        raise RenderError("ffmpeg encoder failed: " + encoder.stderr.read().decode(errors="replace"))
    finally:
      if decoder.poll() is None:
        decoder.terminate()
      decoder.wait()
  output_probe = probe_video(args.out)
  if int(output_probe["nb_read_frames"]) != len(times):
    raise RenderError("encoded video frame count does not match requested render schedule")
  receipt = {"source": str(source.resolve()), "data": str(args.data.resolve()), "replay": str(args.replay.resolve()),
             "output": str(args.out.resolve()), "source_probe": probe, "output_probe": output_probe,
             "output_fps": args.fps, "requested_seconds": args.seconds, "start_relative_camera_s": args.start,
             "alignment": "narrowRoadEncodeIdx.segmentId => timestampEof; causal source frame selection",
             "container_frame_rate_used_for_alignment": False, "source_frames_mapped": count,
             "output_frames": len(times), "distinct_source_frames": len(np.unique(frame_ids)),
             "max_camera_age_ms": max(row["age_ms"] for row in receipt_frames),
             "projection_intrinsics": projection_label, "calibrated_flat_ground_frames": calibrated_frames,
             "nominal_projection_frames": len(times) - calibrated_frames, "assumed_height_when_missing_m": args.ground_height,
             "analysis_calibration_max_age_s": .5, "held_calibration_rows": int(np.count_nonzero(data["calibration_held"])),
             "forecast_semantics": "offline simulation; frozen recorded targets; tuned controller not driven; no model feedback",
             "trace_semantics": {"green": "recorded measured torque, acceleration, speed, yaw",
                                 "baseline_torque_tuned_torque": "controller commands",
                                 "baseline_accel_tuned_accel": "acceleration commands, not plant/measured acceleration",
                                 "baseline_yaw_tuned_yaw": "simulated plant yaw",
                                 "baseline_v_tuned_v": "simulated where longitudinal fit valid; recorded speed otherwise"},
             "valid_horizon_min_s": min(horizons), "valid_horizon_max_s": max(horizons),
             "frames": receipt_frames, "decode_command": decode_cmd, "encode_command": encode_cmd}
  receipt_path = args.out.with_suffix(".json")
  receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
  print(json.dumps({"video": str(args.out), "receipt": str(receipt_path), "frames": len(times),
                    "max_camera_age_ms": receipt["max_camera_age_ms"]}, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--segment", type=Path, required=True)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--replay", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--start", type=float, default=0.)
  parser.add_argument("--seconds", type=float, default=30.)
  parser.add_argument("--fps", type=float, default=10.)
  parser.add_argument("--ground-height", type=float, default=1.22)
  args = parser.parse_args()
  try:
    render(args)
  except (RenderError, subprocess.CalledProcessError, OSError, KeyError) as exc:
    parser.exit(1, f"controller_video: {exc}\n")


if __name__ == "__main__":
  main()
