#!/usr/bin/env python3
"""Observation-only archive metrics: recorded model/controller output vs measured motion.

Trajectory GT integrates carState.vEgo and measured LLK yaw, never model velocity.
Model publication is causal as-of, no older than 50 ms. Exact frame-ID matching
anchors each model path and measured future trajectory to camera capture, with
capture age at most 200 ms. Interpolation of measured signals is for offline
labels only. No replay, gain identification or road-performance claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from controller_extract import hold_calibration, model_capture_times, sha256
from controller_tune import DT, clean_mask, cohort_key, enrich_route_metadata, resolve_gitlinks

HORIZONS = (1, 2, 3)
MAX_MODEL_AGE = .05
MAX_CAPTURE_AGE = .2
MODES = ("manual", "lateral_only", "longitudinal_only", "both", "unknown")
ERROR_FIELDS = ("lat_target_minus_measured_mps2", "delivered_minus_requested_fraction",
                "accel_command_minus_measured_mps2", "accel_target_minus_measured_mps2")
TRAJECTORY_FIELDS = ("endpoint_l2_m", "lateral_signed_m", "lateral_abs_m", "model_age_s",
                     "capture_age_s", "capture_to_publication_s")


def mode_indices(data):
  lat, long = data["lat_active"], data["long_active"]
  result = np.full(len(lat), 4, dtype=int)
  known = np.isin(lat, [0, 1]) & np.isin(long, [0, 1]) & (data["car_control_valid"] == 1)
  result[known] = lat[known].astype(int) + 2 * long[known].astype(int)
  return result


def anchor_indices(t):
  if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
    raise ValueError("finite strictly increasing dataset timestamps required")
  result, next_t = [], -np.inf
  for i, stamp in enumerate(t):
    if stamp >= next_t - 1e-8:
      result.append(i)
      next_t = stamp + 1
  return np.asarray(result, dtype=int)


def empty_mode():
  return {"rows": 0, "calibration_held_rows": 0,
          "sampled_torque_counts": Counter(),
          "lat_tracking": {"candidates": 0, "used": 0}, "long_tracking": {"candidates": 0, "used": 0},
          "instantaneous": {key: [] for key in ERROR_FIELDS},
          "model_vs_measured": {str(h): dict(candidates=0, used=0, excluded=Counter(),
                                            **{key: [] for key in TRAJECTORY_FIELDS}) for h in HORIZONS}}


def _window_brackets(data, modes, i, h, capture_t):
  """Admit every measured row used to bracket/interpolate a capture-based label."""
  timestamps = data["t"]
  if capture_t < timestamps[0]:
    return None, "missing_capture_prefix"
  end_t = capture_t + h
  if end_t > timestamps[-1]:
    return None, "future_unavailable"
  lo = int(np.searchsorted(timestamps, capture_t, side="right") - 1)
  hi = int(np.searchsorted(timestamps, end_t, side="left"))
  z = slice(lo, hi + 1)
  t = data["t"][z]
  if np.any(np.abs(np.diff(t) - DT) > .002):
    return None, "time_gap"
  if any(not np.isfinite(data[key][z]).all() for key in ("v", "yaw")) or any(
      not np.all(data[key][z] == 1) for key in ("car_state_valid", "location_valid", "location_measurement_valid")):
    return None, "invalid_measured_telemetry"
  if any(not np.all(data[key][z] == 1) for key in ("calibration_valid", "calibrated")):
    return None, "invalid_calibration"
  pressed = np.stack([data[key][z] for key in ("steering_pressed", "brake_pressed", "gas_pressed")])
  if not np.isfinite(pressed).all():
    return None, "missing_driver_state"
  if np.any(pressed != 0):
    return None, "driver_intervention"
  if modes[i] == 4 or np.any(modes[z] != modes[i]):
    return None, "mode_change"
  return (lo, hi, end_t), None


def window_reason(data, modes, i, h, capture_t=None):
  return _window_brackets(data, modes, i, h, data["t"][i] if capture_t is None else capture_t)[1]


def measured_endpoint(data, modes, i, h, capture_t):
  """Capture-origin measured future position; label-only interpolation, never model input.

  Interpolate speed/yaw at the two requested boundary times only after all
  bracketing rows pass admission. Integrate each actual interval using midpoint
  speed, midpoint yaw and midpoint heading in x-forward/y-right coordinates.
  """
  brackets, reason = _window_brackets(data, modes, i, h, capture_t)
  if reason:
    return None, reason
  lo, hi, end_t = brackets
  t = data["t"][lo:hi+1]
  nodes = np.r_[capture_t, t[(t > capture_t) & (t < end_t)], end_t]
  speed = np.interp(nodes, t, data["v"][lo:hi+1])
  yaw = np.interp(nodes, t, data["yaw"][lo:hi+1])
  dt = np.diff(nodes)
  turns = .5 * (yaw[:-1] + yaw[1:]) * dt
  midpoint_heading = np.cumsum(turns) - .5 * turns
  distance = .5 * (speed[:-1] + speed[1:]) * dt
  return np.array([np.sum(distance * np.cos(midpoint_heading)), np.sum(distance * np.sin(midpoint_heading))]), None


def model_endpoint(data, i, h):
  horizon, x, y = (data[key][i] for key in ("model_horizon", "model_x", "model_y"))
  if (not np.isfinite(horizon).all() or np.any(np.diff(horizon) <= 0)
      or horizon[0] > 0 or horizon[-1] < h):
    return None
  # Only the prefix through the requested horizon is needed; reject holes within it.
  stop = np.searchsorted(horizon, h, side="left") + 1
  if not np.isfinite(x[:stop]).all() or not np.isfinite(y[:stop]).all():
    return None
  return np.array([np.interp(h, horizon, x) - np.interp(0, horizon, x),
                   np.interp(h, horizon, y) - np.interp(0, horizon, y)])


def segment_metrics(data):
  modes = mode_indices(data)
  captures = model_capture_times(data)
  result = {name: empty_mode() for name in MODES if np.any(modes == MODES.index(name))}
  clean_lat, clean_long = clean_mask(data, "lat"), clean_mask(data, "long")
  for mode, value in result.items():
    selected = modes == MODES.index(mode)
    value["rows"] = int(selected.sum())
    value["calibration_held_rows"] = int(np.sum(selected & data.get("calibration_held", np.zeros(len(modes), dtype=bool))))
    torque = data["torque"][selected]
    value["sampled_torque_counts"].update(finite=int(np.isfinite(torque).sum()),
      at_409=int(np.sum(np.abs(torque) == 409)), over_409=int(np.sum(np.abs(torque) > 409)))
    lat, long = selected & clean_lat, selected & clean_long
    value["lat_tracking"].update(candidates=int(np.sum(selected & (data["lat_active"] == 1))), used=int(lat.sum()))
    value["long_tracking"].update(candidates=int(np.sum(selected & (data["long_active"] == 1))), used=int(long.sum()))
    errors = value["instantaneous"]
    errors["lat_target_minus_measured_mps2"].extend((data["desired_curvature"][lat] * data["v"][lat]**2
                                                   - data["v"][lat] * data["yaw"][lat]).tolist())
    delivery = lat & np.isfinite(data["torque_fraction"])
    errors["delivered_minus_requested_fraction"].extend((data["torque"][delivery] / 409 - data["torque_fraction"][delivery]).tolist())
    errors["accel_command_minus_measured_mps2"].extend((data["accel_command"][long] - data["a"][long]).tolist())
    errors["accel_target_minus_measured_mps2"].extend((data["accel_target"][long] - data["a"][long]).tolist())
  for i in anchor_indices(data["t"]):
    mode = result[MODES[modes[i]]]
    age = data["t"][i] - data["model_t"][i]
    capture_t = captures[i]
    capture_age = data["t"][i] - capture_t
    for h in HORIZONS:
      value = mode["model_vs_measured"][str(h)]
      value["candidates"] += 1
      if data["model_valid"][i] != 1 or not np.isfinite(age):
        reason = "invalid_model"
      elif age < -1e-8:
        reason = "future_model"
      elif age > MAX_MODEL_AGE + 1e-8:
        reason = "stale_model"
      elif not np.isfinite(capture_t):
        reason = "unknown_model_capture"
      elif capture_age > MAX_CAPTURE_AGE + 1e-8:
        reason = "stale_model_capture"
      else:
        reason = None
      measured = None
      if reason is None:
        measured, reason = measured_endpoint(data, modes, i, h, capture_t)
      predicted = None if reason else model_endpoint(data, i, h)
      if reason is None and predicted is None:
        reason = "invalid_model_horizon"
      if reason:
        value["excluded"][reason] += 1
        continue
      error = predicted - measured
      value["used"] += 1
      value["endpoint_l2_m"].append(float(np.linalg.norm(error)))
      value["lateral_signed_m"].append(float(error[1]))
      value["lateral_abs_m"].append(float(abs(error[1])))
      value["model_age_s"].append(float(max(0, age)))
      value["capture_age_s"].append(float(capture_age))
      value["capture_to_publication_s"].append(float(data["model_t"][i] - capture_t))
  return result


def summarize(values):
  if not len(values):
    return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
  a = np.asarray(values)
  if not np.isfinite(a).all():
    raise ValueError("non-finite values reached a metrics accumulator")
  return {"n": len(values), "mean": float(np.mean(a)), "median": float(np.median(a)),
          "p95": float(np.percentile(a, 95)), "min": float(np.min(a)), "max": float(np.max(a))}


def finalize_mode(value):
  output = {key: value[key] for key in ("rows", "calibration_held_rows", "sampled_torque_counts")}
  for key in ("lat_tracking", "long_tracking"):
    counts = value[key]
    output[key] = dict(counts, excluded=counts["candidates"] - counts["used"],
                       coverage=counts["used"] / counts["candidates"] if counts["candidates"] else None)
  output["instantaneous"] = {}
  for key, values in value["instantaneous"].items():
    output["instantaneous"][key] = {"signed": summarize(values), "absolute": summarize(np.abs(values))}
  output["model_vs_measured"] = {}
  for h, metrics in value["model_vs_measured"].items():
    output["model_vs_measured"][h] = {key: metrics[key] for key in ("candidates", "used", "excluded")}
    output["model_vs_measured"][h].update({key: summarize(metrics[key]) for key in TRAJECTORY_FIELDS})
    output["model_vs_measured"][h]["coverage"] = metrics["used"] / metrics["candidates"] if metrics["candidates"] else None
  return output


def merge_mode(target, source):
  for key in ("rows", "calibration_held_rows"):
    target[key] += source[key]
  target["sampled_torque_counts"].update(source["sampled_torque_counts"])
  for key in ("lat_tracking", "long_tracking"):
    for count in ("candidates", "used"):
      target[key][count] += source[key][count]
  for key in ERROR_FIELDS:
    target["instantaneous"][key].extend(source["instantaneous"][key])
  for h in map(str, HORIZONS):
    a, b = target["model_vs_measured"][h], source["model_vs_measured"][h]
    for key in ("candidates", "used"):
      a[key] += b[key]
    a["excluded"].update(b["excluded"])
    for key in TRAJECTORY_FIELDS:
      a[key].extend(b[key])


def cohort_identity(meta):
  model = {key: value for key, value in (meta.get("model_identity") or {}).items()
           if not key.startswith(("ModelManager_LastSyncTime", "ModelManager_ModelsCache", "ModelManager_PrevBundle"))}
  identity = {key: meta.get(key) for key in ("git_commit", "dirty", "fingerprint", "op_long", "opendbc_gitlink",
                                            "torque_factor", "torque_friction", "params")}
  identity["model_identity_sha256"] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
  identity["model_identity_keys"] = sorted(model)
  identity["model_identity_status"] = "recorded parameter identity; model weight contents not established by this analysis"
  identity["opendbc_identity_status"] = "resolved clean historical commit" if meta.get("opendbc_gitlink") else "unknown"
  return identity


def source_reference(path, meta):
  return {"segment": meta["segment"], "route": meta["route"], "npz": str(path.resolve()), "npz_sha256": sha256(path),
          "metadata": str(path.with_suffix(".json").resolve()), "metadata_sha256": sha256(path.with_suffix(".json")),
          "source_log": meta.get("source_log"), "source_log_sha256": meta.get("source_log_sha256"),
          "schema_sha256": meta.get("schema_sha256"), "cohort": cohort_key(meta),
          "route_provenance_source": meta.get("route_provenance_source"),
          "route_context_inherited_fields": meta.get("route_context_inherited_fields", [])}


def validate_output(dataset, output):
  root, target = dataset.resolve(), output.resolve()
  if target.suffix.lower() != ".json" or target.is_relative_to(root):
    raise ValueError("report must be a .json file outside the extracted dataset directory")
  if target.exists() and any(item.is_file() and target.samefile(item) for item in root.iterdir()):
    raise ValueError("report aliases an existing dataset file")


def extraction_context(dataset):
  path = dataset / "manifest.json"
  if not path.is_file():
    return {"extraction_manifest": None, "extraction_selected": None, "extraction_errors": None}
  manifest = json.loads(path.read_text(encoding="utf-8"))
  selected, extracted, failures = (manifest.get(key, []) for key in ("selected", "extracted", "extraction_errors"))
  return {"extraction_manifest": str(path.resolve()), "extraction_manifest_sha256": sha256(path),
          "extraction_status": manifest.get("status"), "extraction_selected": len(selected), "extraction_succeeded": len(extracted),
          "extraction_errors": len(failures), "extraction_failures": failures,
          "source_routes_selected": len({row["segment"].rsplit("--", 1)[0] for row in selected})}


def analyze(paths):
  cohorts, sources = {}, []
  for path, meta in paths:
    key = cohort_key(meta)
    if key not in cohorts:
      cohorts[key] = {"identity": cohort_identity(meta), "segments": 0, "routes": set(),
                      "raw_can_counts": Counter(), "modes": {}}
    cohort = cohorts[key]
    with np.load(path, allow_pickle=False) as archive:
      data = hold_calibration({k: archive[k] for k in archive.files})
    for mode, values in segment_metrics(data).items():
      merge_mode(cohort["modes"].setdefault(mode, empty_mode()), values)
    cohort["segments"] += 1
    cohort["routes"].add(meta["route"])
    cohort["raw_can_counts"].update({name: int(meta.get("can_counts", {}).get(name, 0)) for name in
      ("tx_lkas_frames", "tx_at_409_frames", "tx_over_409_frames", "eps_frames", "eps_fault_frames")})
    sources.append(source_reference(path, meta))
  for cohort in cohorts.values():
    cohort["route_ids"] = sorted(cohort["routes"])
    cohort["routes"] = len(cohort["routes"])
    cohort["modes"] = {mode: finalize_mode(values) for mode, values in cohort["modes"].items()}
  return {"segments": len(paths), "routes": len({meta["route"] for _, meta in paths}),
          "cohorts": cohorts, "sources": sources}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--limit", type=int, help="Bounded smoke only; omitted means every NPZ presently in the dataset")
  args = parser.parse_args()
  try:
    validate_output(args.dataset, args.out)
  except ValueError as exc:
    parser.error(str(exc))
  if args.limit is not None and args.limit < 1:
    parser.error("--limit must be positive")
  available = sorted(args.dataset.glob("*.npz"))
  if not available:
    parser.error("dataset contains no NPZ records")
  paths = [(p, json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))) for p in available[:args.limit]]
  enrich_route_metadata(paths)
  resolve_gitlinks(paths, args.source_repo)
  report = analyze(paths)
  report.update(extraction_context(args.dataset))
  report.update({"dataset": str(args.dataset.resolve()), "npz_records_present_at_start": len(available), "smoke_limit": args.limit,
    "trajectory_timing_status": "exact_frame_capture_anchored_measured_gt",
    "analysis_version": 2,
    "contract": {"scope": "all selected extracted records; source archive/extraction completeness must be checked separately",
      "sample_rate_hz": 1 / DT, "anchor_spacing_s": 1, "trajectory_horizons_s": HORIZONS,
      "gt": "capture-origin midpoint integration of measured speed/yaw on actual intervals; x forward,y right; boundary interpolation is labels only",
      "model": "recorded nonuniform modelV2.position.t; linear endpoint interpolation; subtract position at horizon0",
      "alignment": "exact model_frame_id to narrowRoadEncodeIdx.frameId exposure timestamp; publication causal and age<=50ms; capture age<=200ms",
      "model_age_s": "modelV2 publication age only; capture_age_s and capture_to_publication_s reported separately",
      "capture_age_s": "analysis-row time minus exact matched EncodeIdx.timestampEof; never publication fallback",
      "future_window": "capture through capture+horizon including both bracketing rows contiguous, measured-valid, calibrated, driver-free and constant mode",
      "missing_prefix": "capture before this segment's first measured row is excluded; no backward fill or implicit segment stitching",
      "label_causality": "future measured motion/interpolation used only as offline evaluation labels; never model/controller inputs or tuning features",
      "calibration": "analysis-only hold_calibration<=500ms for identical valid source event; dynamic telemetry keeps extraction freshness",
      "lat_tracking": "desired_curvature*vEgo^2 minus measured vEgo*yaw; >=3m/s clean active rows",
      "torque_normalization": "delivered CAN counts/409 minus requested carControl fraction;409 is a comparison reference, historical ceiling unverified",
      "long_tracking": "instantaneous delivered acceleration command/planner target minus measured carState.aEgo; no delay compensation or gain inference",
      "raw_can_counts": "metadata raw LKAS11 src128 and MDPS12 0x251 src0 counts; not attributable to20Hz modes; faults count messages not onsets",
      "interpretation": "descriptive per-provenance/per-mode metrics; no pooled model ranking, causal tune improvement or road acceptance"}})
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  print(json.dumps({"out": str(args.out.resolve()), "segments": report["segments"], "routes": report["routes"],
                    "cohorts": len(report["cohorts"]), "smoke_limit": args.limit}))


if __name__ == "__main__":
  main()
