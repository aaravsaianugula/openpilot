"""Exact frozen historical longitudinal comparison; never fit or select parameters.

The one validation window is fixed. Recorded future targets are replayed per step;
recorded future yaw conditions the illustration, not the longitudinal controller.
This is a partial longitudinal simulation, with no current RDF or road outcome claim.
"""
from __future__ import annotations

import argparse
import base64
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from controller_extract import hold_calibration
from controller_model_overlay import capture_path
from controller_search import gain_series
import controller_tune as tune
import controller_video as drawing

ROOT = Path(r"D:\comma_four\tuning\controller-tuning-20261012")
SEGMENT = "00000036--244660f5ab--17"
WINDOW = SEGMENT + ":240"
START, COUNT = 240, 61
PARAMETERS = {"acceleration_gain": .4, "braking_gain": .4}
FROZEN = {"search-control/candidate.json": "b747c2af2f8ad55557572612826ee5998f59802b1329081d8596eebe7a6c4570",
          "search-control/search-trace.json": "09328dee1822a0b07ec27b90392994be525af631c2461141b76877812190fbc8",
          "experiment-control/tuning-report.json": "1ecd782f6f621acd7a79da28b6a0721c01766219e88c8551d93edda9f9b1737c"}
SCOPE = "Historical partial longitudinal simulation; frozen recorded targets; no current RDF, full tune or driving-improvement claim."
YAW_SCOPE = "Both simulated paths use recorded future yaw. Conditional illustration, not a lateral response prediction."


class ComparisonError(ValueError):
  pass


def digest(path):
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_json(path, expected):
  if digest(path) != expected:
    raise ComparisonError(f"frozen source hash mismatch: {path}")
  return json.loads(Path(path).read_text(encoding="utf-8"))


def selection_contract(candidate, search, manifest, plant):
  route = SEGMENT.rsplit("--", 1)[0]
  if (candidate.get("production_enabled") is not False or candidate.get("steer_max") != 409
      or candidate.get("complete_controller_candidate") is not None
      or candidate.get("partial_axis_candidates") != PARAMETERS):
    raise ComparisonError("exact frozen partial .4/.4 candidate required")
  if not candidate.get("cohort") or any(x.get("cohort") != candidate["cohort"] for x in (manifest, plant)):
    raise ComparisonError("cohort binding mismatch")
  split = manifest.get("split", {})
  if (route not in split.get("validation", []) or route in split.get("train", [])
      or route in split.get("test", []) or plant.get("split") != split):
    raise ComparisonError("fixed window must belong exclusively to the frozen validation split")
  axis = search.get("axes", {}).get("long", {})
  winner = axis.get("winner") or {}
  if (search.get("partial_candidate") != PARAMETERS or winner.get("parameters") != PARAMETERS
      or winner.get("rejections") != [] or WINDOW not in axis.get("common_window_ids", [])
      or len(axis.get("common_window_ids", [])) < 20 or axis.get("common_coverage", 0) < .6):
    raise ComparisonError("selected window lacks common baseline/candidate support")
  model = plant.get("axes", {}).get("long", {}).get("model")
  if not model or model.get("identified") is not True or model.get("recorded_build_gitlink_known") is not True:
    raise ComparisonError("frozen identified historical longitudinal plant required")
  return model


def make_trace(data, model):
  if START not in tune.window_indices(data, "long", stride=40):
    raise ComparisonError("fixed window fails clean prefix, timing or active-control support")
  z = slice(START, START + COUNT)
  for key in ("car_state_valid", "location_valid", "location_measurement_valid", "planner_valid", "car_output_valid"):
    if key not in data or not np.all(data[key][START-12:START+COUNT] == 1):
      raise ComparisonError("missing valid measured/planner/command source: " + key)
  t = data["t"][z]
  if len(t) != COUNT or np.any(np.abs(np.diff(t) - tune.DT) > .002):
    raise ComparisonError("fixed 61-point20Hz window required")
  target = data["accel_target"][z]
  baseline_a, baseline_command = tune.rollout(data, START, model, "long")
  candidate_a, candidate_command = tune.rollout(data, START, model, "long", gain_series(target, **PARAMETERS))
  if not np.isfinite([baseline_a, baseline_command, candidate_a, candidate_command]).all():
    raise ComparisonError("frozen plant cannot support the full paired window")
  if (np.any(candidate_command < -3.5) or np.any(candidate_command > 2)
      or np.max(np.abs(np.diff(candidate_command))) > tune.DT * 2.5 + 1e-8):
    raise ComparisonError("candidate exceeds existing longitudinal command/slew envelope")
  trace = {"t": t, "target_accel": target, "actual_accel": data["a"][z],
           "recorded_command": data["accel_command"][z], "actual_speed": data["v"][z],
           "recorded_yaw": data["yaw"][z], "baseline_accel": baseline_a, "candidate_accel": candidate_a,
           "baseline_command": baseline_command, "candidate_command": candidate_command}
  for name, acceleration in (("baseline", baseline_a), ("candidate", candidate_a)):
    speed = np.empty(COUNT)
    speed[0] = data["v"][START]
    for i in range(COUNT-1):
      speed[i+1] = max(0, speed[i] + acceleration[i] * tune.DT)
    trace[name + "_speed"] = speed
    trace[name + "_xy"] = tune.integrate(speed, trace["recorded_yaw"], tune.DT)
  trace["actual_xy"] = tune.integrate(trace["actual_speed"], trace["recorded_yaw"], tune.DT)
  return trace


def moving_paths(trace, data, timestamp):
  """Reanchor the SAME frozen window using only measured motion up to display time."""
  anchor = float(trace["t"][0])
  if not anchor <= timestamp < trace["t"][-1]:
    raise ComparisonError("display time must stay inside the single frozen window")
  paths = [capture_path(data, trace[name + "_xy"], trace["t"] - anchor, anchor, timestamp, max_age=3.)
           for name in ("actual", "baseline", "candidate")]
  if any(not len(path) for path in paths):
    raise ComparisonError("measured past-motion reanchor unavailable")
  return paths


def source_inputs(root):
  values = {name: bound_json(root / name, expected) for name, expected in FROZEN.items()}
  manifest_path = root / "search-control/manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  candidate, search, plant = (values[name] for name in FROZEN)
  if manifest.get("plant_report_sha256") != FROZEN["experiment-control/tuning-report.json"]:
    raise ComparisonError("manifest plant source binding mismatch")
  model = selection_contract(candidate, search, manifest, plant)
  matches = [row for row in manifest["sources"] if row.get("segment") == SEGMENT]
  if len(matches) != 1:
    raise ComparisonError("one exact source receipt required")
  source = matches[0]
  npz, metadata = root / "dataset" / (SEGMENT + ".npz"), root / "dataset" / (SEGMENT + ".json")
  if Path(source["npz"]).resolve() != npz.resolve() or Path(source["metadata"]).resolve() != metadata.resolve():
    raise ComparisonError("fixed validation dataset path binding mismatch")
  raw_log = Path(source["source_log"])
  if raw_log.resolve() != (Path(r"F:\routes") / SEGMENT / "rlog.zst").resolve():
    raise ComparisonError("raw segment/log identity mismatch")
  inputs = {"candidate": root / "search-control/candidate.json", "search_trace": root / "search-control/search-trace.json",
            "plant_report": root / "experiment-control/tuning-report.json", "search_manifest": manifest_path,
            "npz": npz, "metadata": metadata, "raw_log": raw_log, "camera": raw_log.parent / "fcamera.hevc",
            "comparison_source": Path(__file__), "rollout_source": Path(tune.__file__),
            "search_source": Path(gain_series.__code__.co_filename), "drawing_source": Path(drawing.__file__),
            "calibration_source": Path(hold_calibration.__code__.co_filename), "reanchor_source": Path(capture_path.__code__.co_filename)}
  hashes = {key: digest(path) for key, path in inputs.items()}
  for key, recorded in (("npz", "npz_sha256"), ("metadata", "metadata_sha256"), ("raw_log", "source_log_sha256")):
    if hashes[key] != source[recorded]:
      raise ComparisonError("dataset/raw source hash mismatch: " + key)
  with np.load(npz, allow_pickle=False) as arrays:
    data = hold_calibration({key: arrays[key] for key in arrays.files})
  return data, model, inputs, hashes, candidate


def unchanged_sources(inputs, hashes):
  if any(digest(path) != hashes[key] for key, path in inputs.items()):
    raise ComparisonError("input/source changed during comparison")


def fresh_output(path, parent):
  path, parent = path.resolve(), parent.resolve()
  if not path.is_relative_to(parent) or path == parent or path.exists():
    raise ComparisonError("output must be new and contained in the declared artifact directory")
  return path


def run_phase(command, phase, attempt, timeout=120):
  """Keep tool failure evidence outside temporary frame buffers, before raising."""
  result, failure = None, None
  proof = {"phase": phase, "command": command, "started_utc": datetime.now(UTC).isoformat(),
           "timeout_s": timeout, "returncode": None, "timed_out": False}
  stdout, stderr = b"", b""
  try:
    result = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    stdout, stderr = result.stdout, result.stderr
    proof["returncode"] = result.returncode
    if result.returncode:
      failure = f"{phase} exited{result.returncode}"
  except subprocess.TimeoutExpired as exc:
    stdout, stderr = exc.stdout or b"", exc.stderr or b""
    proof["timed_out"] = True
    failure = f"{phase} exceeded{timeout}s timeout"
  except OSError as exc:
    failure = f"{phase} could not start: {exc}"
  proof.update(status="failed" if failure else "passed", error=failure,
               stdout=stdout.decode("utf-8", errors="replace"), stderr=stderr.decode("utf-8", errors="replace"),
               stdout_base64=base64.b64encode(stdout).decode("ascii"), stderr_base64=base64.b64encode(stderr).decode("ascii"))
  receipt = attempt / (phase + ".json")
  with receipt.open("x", encoding="utf-8") as stream:
    json.dump(proof, stream, indent=2)
  if failure:
    raise ComparisonError(f"{failure}; exact tool evidence: {receipt}")
  return result


def probe(path, phase, attempt):
  command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(path)]
  streams = json.loads(run_phase(command, phase, attempt).stdout)["streams"]
  if not streams or "nb_read_frames" not in streams[0]:
    raise ComparisonError("ffprobe did not confirm a decoded frame count")
  return streams[0]


def compose(frame, data, trace, timestamp, metadata):
  paths = moving_paths(trace, data, timestamp)
  canvas = Image.new("RGB", (1600, 1000), drawing.BG)
  draw = ImageDraw.Draw(canvas)
  drawing.text(draw, (24, 12), "HISTORICAL SELECTED LONGITUDINAL COMPARISON  |  acceleration .4 / braking .4", 23)
  drawing.text(draw, (24, 47), "Partial simulation / recorded targets / controller never driven / no current RDF or complete tune", 19, drawing.TUNED)
  elapsed = timestamp - float(trace["t"][0])
  drawing.text(draw, (24, 80), f"{WINDOW}  VALIDATION  |  one frozen 3s window, elapsed {elapsed:.2f}s", 17, drawing.MUTED)
  camera_box = (16, 122, 976, 665)
  if frame is None:
    draw.rectangle(camera_box, fill=drawing.PANEL)
    drawing.text(draw, (45, 355), "Layout preview: actual camera has not been decoded", 24, drawing.MUTED)
  else:
    intrinsics, label = drawing.projection_config(data, metadata, frame.size)
    rpy, height, calibrated = drawing.calibration_at(data, timestamp, 1.22)
    if not calibrated or "NOMINAL" in label:
      raise ComparisonError("recorded camera calibration/intrinsics unavailable; no nominal overlay fallback")
    camera = frame.copy()
    camera_draw = ImageDraw.Draw(camera)
    for path, color in zip(paths, (drawing.GT, drawing.BASELINE, drawing.TUNED), strict=True):
      drawing.draw_line(camera_draw, drawing.project_ground(path, intrinsics, rpy, height), color, 5,
                        (0, 0, frame.width, frame.height))
    canvas.paste(camera.resize((960, 543)), camera_box[:2])
  box = (992, 122, 1584, 665)
  draw.rounded_rectangle(box, radius=12, fill=drawing.PANEL)
  forward = max(10., min(90., max(float(np.max(p[:, 0])) for p in paths) + 3))
  lateral = max(3., max(float(np.max(np.abs(p[:, 1]))) for p in paths) + 1)
  for distance in np.arange(0, forward, 10):
    y = 635 - distance / forward * 440
    draw.line((1040, y, 1550, y), fill="#344454")
    drawing.text(draw, (1002, y-8), f"{distance:g}m", 12, drawing.MUTED)
  for path, color in zip(paths, (drawing.GT, drawing.BASELINE, drawing.TUNED), strict=True):
    pixels = np.column_stack((1295 + path[:, 1] / lateral * 250, 635 - path[:, 0] / forward * 440))
    drawing.draw_line(draw, pixels, color, 4, box)
  draw.ellipse((1290, 630, 1300, 640), fill=drawing.FG)
  drawing.text(draw, (1012, 136), "BEV: tail of the same frozen forecast", 18)
  drawing.text(draw, (1012, 165), f"Right positive; width +/-{lateral:.1f}m", 14, drawing.MUTED)
  drawing.text(draw, (24, 682), "Green measured | Blue logged-command plant | Orange selected controller | Purple recorded target", 18)
  drawing.text(draw, (24, 711), YAW_SCOPE, 17, drawing.TUNED)
  drawing.text(draw, (24, 740), "Measured past motion reanchors every path. Simulated lateral response and vision feedback are unavailable.", 16, drawing.MUTED)
  panels = [
    ("Acceleration: measured / plant / selected", "actual_accel", "baseline_accel", "candidate_accel", (-.5, .5)),
    ("Command: target / recorded / selected", "target_accel", "recorded_command", "candidate_command", (-.6, .6)),
    ("Speed m/s: measured / plant / selected", "actual_speed", "baseline_speed", "candidate_speed", (27., 29.)),
  ]
  for i, (title, actual, baseline, candidate, limits) in enumerate(panels):
    drawing.plot_trace(draw, (16 + 528*i, 776, 528 + 528*i, 984), title,
                       [(trace["t"], trace[key], color) for key, color in
                        ((actual, drawing.MODEL if i == 1 else drawing.GT),
                         (baseline, drawing.BASELINE), (candidate, drawing.TUNED))], timestamp, limits)
  return canvas


def prepare(root, output):
  data, model, inputs, hashes, candidate = source_inputs(root)
  trace = make_trace(data, model)
  output = fresh_output(output, root / "historical-selected-comparison")
  output.mkdir(parents=True)
  trace_path = output / "comparison.npz"
  np.savez_compressed(trace_path, **trace)
  metadata = json.loads(inputs["metadata"].read_text())
  preview = output / "preview.png"
  compose(None, data, trace, float(trace["t"][0]) + 1.5, metadata).save(preview)
  unchanged_sources(inputs, hashes)
  receipt = {"window_id": WINDOW, "cohort": candidate["cohort"], "parameters": PARAMETERS, "scope": SCOPE,
             "yaw_assumption": YAW_SCOPE, "partition": "validation", "test_data_opened": False,
             "selection_rerun": False, "source_files": {k: str(v.resolve()) for k, v in inputs.items()},
             "source_sha256": hashes, "trace_sha256": digest(trace_path), "preview_sha256": digest(preview),
             "anchor_t": float(trace["t"][0]), "end_t": float(trace["t"][-1]), "samples": COUNT,
             "dt": tune.DT, "model": model, "state_initialization": "measured acceleration/speed at anchor; recorded command delay prefix",
             "controller_uses": "simulated acceleration feedback plus recorded targets per step; no future measured acceleration or yaw feedback",
             "preselected_window_target_accel_mae": {name: float(np.mean(np.abs(trace[key] - trace["target_accel"])))
                 for name, key in (("baseline", "baseline_accel"), ("selected", "candidate_accel"))},
             "metric_scope": "this preselected validation window only; descriptive, never a new selection/acceptance gate",
             "render_state": "layout preview only; camera not decoded", "production_enabled": False}
  (output / "comparison.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
  return {"comparison": str(output), "preview": str(preview), "trace": str(trace_path), "samples": COUNT}


def render_schedule(data, trace):
  mapping = drawing.camera_mapping(data, int(SEGMENT.rsplit("--", 1)[1]))
  anchor, end = trace["t"][[0, -1]]
  next_frame = int(np.searchsorted(mapping[1], anchor, side="left"))
  if next_frame == len(mapping[1]) or mapping[1][next_frame] - anchor > .075:
    raise ComparisonError("camera does not cover frozen window anchor")
  first = float(mapping[1][next_frame])
  times, indices = drawing.frame_schedule(mapping, first, float(end - first), 10.)
  if np.any(mapping[1][indices] < anchor) or np.any(mapping[1][indices] >= end):
    raise ComparisonError("camera schedule extends beyond frozen supported window")
  return mapping, times, indices


def render(root, comparison, output):
  comparison = comparison.resolve()
  if not comparison.is_relative_to((root / "historical-selected-comparison").resolve()):
    raise ComparisonError("comparison must be inside its artifact directory")
  receipt = json.loads((comparison / "comparison.json").read_text())
  data, model, inputs, hashes, candidate = source_inputs(root)
  if (receipt["source_sha256"] != hashes or receipt["parameters"] != PARAMETERS or receipt["window_id"] != WINDOW
      or digest(comparison / "comparison.npz") != receipt["trace_sha256"]):
    raise ComparisonError("prepared trace/source binding changed")
  with np.load(comparison / "comparison.npz", allow_pickle=False) as arrays:
    trace = {key: arrays[key] for key in arrays.files}
  expected = make_trace(data, model)
  if set(trace) != set(expected) or any(not np.array_equal(trace[k], expected[k]) for k in expected):
    raise ComparisonError("prepared trace differs from deterministic frozen replay")
  output = fresh_output(output, root / "videos")
  if output.suffix != ".mp4" or output.with_suffix(".json").exists():
    raise ComparisonError("new mp4 and receipt required")
  mapping, times, indices = render_schedule(data, trace)
  attempt = Path(tempfile.mkdtemp(prefix="render-attempt-", dir=comparison))
  (attempt / "attempt.json").write_text(json.dumps({"comparison_receipt_sha256": digest(comparison / "comparison.json"),
    "source_sha256": hashes, "trace_sha256": receipt["trace_sha256"], "output": str(output)}, indent=2), encoding="utf-8")
  source_probe = probe(inputs["camera"], "probe-source", attempt)
  width, height = int(source_probe["width"]), int(source_probe["height"])
  if int(source_probe["nb_read_frames"]) != len(mapping[0]) or (width, height) != (1344, 760):
    raise ComparisonError("source frame-count/dimensions disagree with recorded camera")
  executable = shutil.which("ffmpeg")
  if executable is None:
    raise ComparisonError("ffmpeg unavailable")
  output.parent.mkdir(parents=True, exist_ok=True)
  metadata = json.loads(inputs["metadata"].read_text())
  commands = []
  with tempfile.TemporaryDirectory(prefix="buffers-", dir=attempt) as directory:
    temporary = Path(directory)
    camera_rgb, output_rgb = temporary / "camera.rgb", temporary / "output.rgb"
    first, last = int(indices.min()), int(indices.max())
    command = [executable, "-nostdin", "-v", "error", "-xerror", "-threads", "2", "-filter_threads", "1", "-i", str(inputs["camera"]),
               "-vf", f"select=between(n\\,{first}\\,{last})", "-fps_mode", "passthrough", "-frames:v", str(last-first+1),
               "-threads:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", str(camera_rgb)]
    commands.append(command)
    run_phase(command, "decode", attempt)
    frame_bytes = width * height * 3
    if camera_rgb.stat().st_size != (last-first+1) * frame_bytes:
      raise ComparisonError("compact camera decode byte/frame count mismatch")
    rows = []
    with camera_rgb.open("rb") as pixels, output_rgb.open("xb") as encoded_frames:
      for index, requested in zip(indices, times, strict=True):
        pixels.seek((int(index)-first) * frame_bytes)
        frame = Image.frombytes("RGB", (width, height), pixels.read(frame_bytes))
        timestamp = float(mapping[1][index])
        encoded_frames.write(compose(frame, data, trace, timestamp, metadata).tobytes())
        rows.append({"source_frame": int(index), "source_t": timestamp, "requested_t": float(requested),
                     "camera_age_ms": float((requested-timestamp)*1000), "anchor_t": float(trace["t"][0])})
    command = [executable, "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "1", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "1600x1000", "-r", "10", "-i", str(output_rgb), "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-threads:v", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    commands.append(command)
    run_phase(command, "encode", attempt)
  output_probe = probe(output, "probe-output", attempt)
  if int(output_probe["nb_read_frames"]) != len(rows):
    raise ComparisonError("encoded comparison frame count mismatch")
  unchanged_sources(inputs, hashes)
  final = dict(receipt, render_state="actual recorded camera rendered", comparison_receipt_sha256=digest(comparison / "comparison.json"),
               video_sha256=digest(output), output=str(output), frames=rows, output_fps=10,
               source_probe=source_probe, output_probe=output_probe, commands=commands,
               render_attempt=str(attempt), phase_receipts={p.name: digest(p) for p in attempt.glob("*.json")},
               projection="recorded calibrated flat ground; sensor intrinsics; no nominal fallback")
  output.with_suffix(".json").write_text(json.dumps(final, indent=2), encoding="utf-8")
  return {"video": str(output), "frames": len(rows), "receipt": str(output.with_suffix('.json'))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("prepare", "render"))
  parser.add_argument("--root", type=Path, default=ROOT)
  parser.add_argument("--comparison", type=Path, default=ROOT / "historical-selected-comparison/exact-window-v1")
  parser.add_argument("--out", type=Path, default=ROOT / "videos/historical-selected-longitudinal.mp4")
  args = parser.parse_args()
  try:
    result = prepare(args.root, args.comparison) if args.mode == "prepare" else render(args.root, args.comparison, args.out)
  except (ComparisonError, OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
    parser.exit(1, f"historical comparison: {exc}\n")
  print(json.dumps(result))


if __name__ == "__main__":
  main()
