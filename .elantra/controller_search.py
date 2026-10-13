#!/usr/bin/env python3
"""Finite gain search against a previously validated observational plant.

This reuses controller_tune.rollout, its fixed steering limits and experimental
longitudinal slew limit. It simulates
frozen recorded targets; it is neither fresh RDF inference nor a closed-loop
model/car replay. Validation chooses gains; held-out test is read only afterward.
An untrusted axis remains unavailable, even if a different axis supports research.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

import controller_tune as tune
from controller_extract import sha256
from controller_metrics import source_reference

GRID = (0., .025, .05, .1, .2, .4)
MIN_COVERAGE = .6
MIN_WINDOWS = 20
MIN_DIRECTION_POINTS = 50
LIMITS = {"lat_jerk_p95": 5., "long_jerk_p95": 3., "saturation_fraction": .05,
          "saturation_increase": .01, "direction_mae_ratio": 1.02, "direction_mae_slack": .005,
          "direction_p95_ratio": 1.05, "direction_p95_slack": .01, "minimum_tracking_improvement": .01,
          "speed_tail_slack_mps": .05, "path_endpoint_tail_slack_m": .1}


def gain_series(target, acceleration_gain, braking_gain):
  """Only a frozen negative target selects braking feedback; zero uses acceleration gain."""
  return np.where(np.asarray(target) < 0, braking_gain, acceleration_gain)


def trusted_models(report):
  models, rejected = {}, {}
  for axis in ("lat", "long"):
    entry = report.get("axes", {}).get(axis, {})
    model, score = entry.get("model"), entry.get("validation_baseline", {})
    reasons = []
    if report.get("source_gitlink_known") is not True:
      reasons.append("unknown_recorded_build")
    if entry.get("identified") is not True:
      reasons.append("untrusted_plant")
    if not model or not (model.get("table") or model.get("cells")):
      reasons.append("missing_plant_support")
    if model and model.get("boundary_optimum") is not False:
      reasons.append("boundary_or_unknown_identification")
    if (score.get("used_windows", 0) < 20 or score.get("coverage", 0) < .5 or score.get("mae") is None
        or score.get("mae", float("inf")) > (.25 if axis == "lat" else .3)):
      reasons.append("insufficient_plant_validation")
    if reasons:
      rejected[axis] = reasons
    else:
      models[axis] = model
  return models, rejected


def validate_split(paths, split):
  if set(split) != {"train", "validation", "test"} or any(not split[key] for key in split):
    raise ValueError("nonempty immutable train/validation/test route splits required")
  flat = [route for name in ("train", "validation", "test") for route in split[name]]
  if len(flat) != len(set(flat)):
    raise ValueError("route partitions must be disjoint")
  walls = {}
  for _, meta in paths:
    wall = meta.get("init_data", {}).get("wallTimeNanos")
    if not wall:
      raise ValueError("recorded wall time required for chronological whole-route split")
    walls[meta["route"]] = min(walls.get(meta["route"], wall), wall)
  if set(flat) != set(walls):
    raise ValueError("split routes must exactly match the selected provenance cohort")
  if (max(walls[r] for r in split["train"]) >= min(walls[r] for r in split["validation"]) or
      max(walls[r] for r in split["validation"]) >= min(walls[r] for r in split["test"])):
    raise ValueError("immutable route split must be chronological")
  return {key: list(split[key]) for key in ("train", "validation", "test")}


def _population(records, axis, maximum):
  windows = []
  for meta, data in sorted(records, key=lambda row: (row[0]["route"], row[0].get("segment_index", 0), row[0]["segment"])):
    for start in tune.window_indices(data, axis, stride=40):
      windows.append((f"{meta['segment']}:{start}", data, int(start)))
  if len(windows) > maximum:
    windows = [windows[i] for i in np.linspace(0, len(windows) - 1, maximum, dtype=int)]
  return windows


def evaluate_axis(records, model, axis, parameters, max_windows=120):
  windows = _population(records, axis, max_windows)
  evaluated = {}
  for name, data, start in windows:
    z = slice(start, start + 61)
    target = data["desired_curvature"][z] * data["v"][z]**2 if axis == "lat" else data["accel_target"][z]
    if parameters is None:
      gain = None
    elif axis == "lat":
      gain = parameters["lateral_gain"]
    else:
      gain = gain_series(target, parameters["acceleration_gain"], parameters["braking_gain"])
    state, command = tune.rollout(data, start, model, axis, gain)
    if not np.isfinite(state).all() or not np.isfinite(command).all():
      continue
    actual = data["v"][z] * data["yaw"][z] if axis == "lat" else data["a"][z]
    evaluated[name] = window_metrics(state, command, target, actual, axis)
  return {"parameters": parameters, "windows": evaluated, "attempted": len(windows)}


def window_metrics(state, command, target, actual, axis):
  masks = {"lateral": np.ones(len(state), dtype=bool)} if axis == "lat" else {
    "acceleration": target > .05, "braking": target < -.05}
  tracking = {direction: {"errors": np.abs(state[mask] - target[mask]).tolist(), "points": int(mask.sum())}
              for direction, mask in masks.items()}
  saturated = np.abs(command) >= 409 - 1e-8 if axis == "lat" else (command <= -3.5 + 1e-8) | (command >= 2 - 1e-8)
  return {"tracking": tracking, "gt_errors": np.abs(state - actual).tolist(),
          "jerk": (np.abs(np.diff(state)) / tune.DT).tolist(), "saturated": saturated.astype(float).tolist(),
          "command_rate": (np.abs(np.diff(command)) / tune.DT).tolist()}


def _summary(raw, common, axis):
  values = [raw["windows"][key] for key in common]
  result = {"windows": len(common), "tracking": {}}
  for direction in (("lateral",) if axis == "lat" else ("acceleration", "braking")):
    errors = [error for value in values for error in value["tracking"].get(direction, {}).get("errors", [])]
    result["tracking"][direction] = {"points": len(errors),
      "windows": sum(bool(value["tracking"].get(direction, {}).get("points")) for value in values),
      "mae": float(np.mean(errors)) if errors else None, "p95": float(np.percentile(errors, 95)) if errors else None}
  for key in ("jerk", "command_rate", "gt_errors", "saturated"):
    array = [item for value in values for item in value[key]]
    result[key] = {"mean": float(np.mean(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))} if array else None
  scores = [entry["mae"] for entry in result["tracking"].values()]
  result["target_objective"] = float(np.mean(scores)) if all(value is not None for value in scores) else None
  return result


def _constraints(metrics, baseline, axis, min_direction_points, require_improvement=True):
  reasons = []
  for direction, score in metrics["tracking"].items():
    before = baseline["tracking"][direction]
    if min(score["points"], before["points"]) < min_direction_points or min(score["windows"], before["windows"]) < 2:
      reasons.append(f"missing_{direction}_support")
      continue
    if score["mae"] > before["mae"] * LIMITS["direction_mae_ratio"] + LIMITS["direction_mae_slack"]:
      reasons.append(f"{direction}_tracking_regression")
    if score["p95"] > before["p95"] * LIMITS["direction_p95_ratio"] + LIMITS["direction_p95_slack"]:
      reasons.append(f"{direction}_tail_regression")
  if metrics["jerk"]["p95"] > LIMITS[f"{axis}_jerk_p95"]:
    reasons.append("jerk_limit")
  if metrics["saturated"]["mean"] > min(LIMITS["saturation_fraction"], baseline["saturated"]["mean"] + LIMITS["saturation_increase"]):
    reasons.append("saturation_limit")
  if axis == "long" and metrics["command_rate"]["max"] > 2.5 + 1e-8:
    reasons.append("longitudinal_slew_limit")
  objective, before = metrics["target_objective"], baseline["target_objective"]
  if require_improvement and (objective is None or before is None or objective >= before * (1 - LIMITS["minimum_tracking_improvement"])):
    reasons.append("no_meaningful_target_improvement")
  return reasons


def select_axis(evaluations, baseline, axis, attempted, min_coverage=MIN_COVERAGE,
                min_windows=MIN_WINDOWS, min_direction_points=MIN_DIRECTION_POINTS):
  """Select on target tracking over one common population, never GT similarity."""
  trace, admitted = [], []
  baseline_ok = len(baseline["windows"]) >= min_windows and len(baseline["windows"]) / max(1, attempted) >= min_coverage
  for raw in evaluations:
    reasons = []
    if not baseline_ok:
      reasons.append("insufficient_baseline_support")
    elif len(raw["windows"]) < min_windows or len(raw["windows"]) / max(1, attempted) < min_coverage:
      reasons.append("insufficient_individual_support")
    trace.append({"parameters": raw["parameters"], "individual_windows": len(raw["windows"]), "rejections": reasons})
    if not reasons:
      admitted.append((raw, trace[-1]))
  common = set(baseline["windows"])
  for raw, _ in admitted:
    common &= set(raw["windows"])
  common = sorted(common)
  if not admitted or len(common) < min_windows or len(common) / max(1, attempted) < min_coverage:
    for _, row in admitted:
      row["rejections"].append("insufficient_common_population")
    return {"winner": None, "trace": trace, "common_window_ids": common, "attempted_windows": attempted}
  before = _summary(baseline, common, axis)
  eligible = []
  for raw, row in admitted:
    metrics = _summary(raw, common, axis)
    row["metrics"] = metrics
    row["rejections"].extend(_constraints(metrics, before, axis, min_direction_points))
    if not row["rejections"]:
      eligible.append(row)
  winner = min(eligible, key=lambda row: (row["metrics"]["target_objective"], row["metrics"]["jerk"]["p95"],
                                         json.dumps(row["parameters"], sort_keys=True))) if eligible else None
  return {"winner": winner, "trace": trace, "baseline": before, "common_window_ids": common,
          "attempted_windows": attempted, "common_coverage": len(common) / max(1, attempted)}


def speed_from_acceleration(initial_speed, acceleration):
  speed = np.full(len(acceleration), np.nan)
  speed[0] = initial_speed
  for j in range(1, len(speed)):
    speed[j] = max(0., speed[j-1] + acceleration[j-1] * tune.DT) if np.isfinite(acceleration[j-1]) else np.nan
  return speed


def evaluate_joint(records, models, parameters, max_windows=120, long_cache=None):
  """Simulated longitudinal speed drives lateral demand and physical response.

  Targets are frozen recorded publications, never future GT; motion arrays used
  as inputs are initial state and already-recorded roll. This is moving-regime
  inverse-response research, not native stopping or changed-camera perception.
  """
  cache = {} if long_cache is None else long_cache
  windows = []
  for meta, data in sorted(records, key=lambda row: (row[0]["route"], row[0].get("segment_index", 0), row[0]["segment"])):
    starts = np.intersect1d(tune.window_indices(data, "lat", stride=40), tune.window_indices(data, "long", stride=40))
    windows.extend((f"{meta['segment']}:{start}", data, int(start)) for start in starts)
  if len(windows) > max_windows:
    windows = [windows[i] for i in np.linspace(0, len(windows) - 1, max_windows, dtype=int)]
  evaluated = {}
  for name, data, start in windows:
    z = slice(start, start + 61)
    target_a, curvature = data["accel_target"][z], data["desired_curvature"][z]
    pair = None if parameters is None else (parameters["acceleration_gain"], parameters["braking_gain"])
    key = name, pair
    if key not in cache:
      gains = None if pair is None else gain_series(target_a, *pair)
      acceleration, accel_command = tune.rollout(data, start, models["long"], "long", gains)
      speed = speed_from_acceleration(data["v"][start], acceleration)
      cache[key] = acceleration, accel_command, speed
    acceleration, accel_command, speed = cache[key]
    gain = None if parameters is None else parameters["lateral_gain"]
    lateral, torque = tune.rollout(data, start, models["lat"], "lat", gain, speed_override=speed)
    if not all(np.isfinite(value).all() for value in (acceleration, accel_command, speed, lateral, torque)):
      continue
    reference_v = speed_from_acceleration(data["v"][start], target_a)
    actual_path = tune.integrate(speed, lateral / np.maximum(speed, 1.), tune.DT)
    target_path = tune.integrate(reference_v, curvature * reference_v, tune.DT)
    evaluated[name] = {
      "lat": window_metrics(lateral, torque, curvature * speed**2, data["v"][z] * data["yaw"][z], "lat"),
      "long": window_metrics(acceleration, accel_command, target_a, data["a"][z], "long"),
      "speed_errors": np.abs(speed - reference_v).tolist(),
      "path_endpoint_error": float(np.linalg.norm(actual_path[-1] - target_path[-1]))}
  return {"parameters": parameters, "windows": evaluated, "attempted": len(windows)}


def joint_summary(raw, common):
  result = {axis: _summary({"windows": {key: raw["windows"][key][axis] for key in common}}, common, axis)
            for axis in ("lat", "long")}
  speed = [error for key in common for error in raw["windows"][key]["speed_errors"]]
  path = [raw["windows"][key]["path_endpoint_error"] for key in common]
  result.update(speed_error_p95=float(np.percentile(speed, 95)), path_endpoint_p95=float(np.percentile(path, 95)))
  return result


def joint_constraints(metrics, baseline, min_direction_points=MIN_DIRECTION_POINTS):
  reasons, ratios = [], []
  for axis in ("lat", "long"):
    reasons.extend(f"{axis}:{reason}" for reason in _constraints(
      metrics[axis], baseline[axis], axis, min_direction_points, require_improvement=False))
    for direction, score in metrics[axis]["tracking"].items():
      before, after = baseline[axis]["tracking"][direction]["mae"], score["mae"]
      if before is None or after is None or (before <= 1e-8 and after > 1e-8):
        ratios.append(None)
      else:
        ratios.append(after / before if before > 1e-8 else 1.)
  for field, slack in (("speed_error_p95", LIMITS["speed_tail_slack_mps"]),
                       ("path_endpoint_p95", LIMITS["path_endpoint_tail_slack_m"])):
    if metrics[field] > baseline[field] * 1.05 + slack:
      reasons.append(f"{field}_regression")
  objective = None if any(value is None for value in ratios) else float(np.mean(ratios))
  if objective is None or objective >= 1 - LIMITS["minimum_tracking_improvement"]:
    reasons.append("no_meaningful_joint_target_improvement")
  return reasons, objective


def select_joint(evaluations, baseline, min_windows=MIN_WINDOWS, min_coverage=MIN_COVERAGE):
  attempted, admitted, trace = baseline["attempted"], [], []
  baseline_ok = len(baseline["windows"]) >= min_windows and len(baseline["windows"]) / max(1, attempted) >= min_coverage
  for raw in evaluations:
    reasons = []
    if not baseline_ok:
      reasons.append("insufficient_baseline_joint_support")
    elif len(raw["windows"]) < min_windows or len(raw["windows"]) / max(1, attempted) < min_coverage:
      reasons.append("insufficient_individual_joint_support")
    row = {"parameters": raw["parameters"], "individual_windows": len(raw["windows"]), "rejections": reasons}
    trace.append(row)
    if not reasons:
      admitted.append((raw, row))
  common = set(baseline["windows"])
  for raw, _ in admitted:
    common &= set(raw["windows"])
  common = sorted(common)
  result = {"winner": None, "trace": trace, "common_window_ids": common, "attempted_windows": attempted,
            "common_coverage": len(common) / max(1, attempted), "scope": "coupled moving regime only; native stopping unsupported"}
  if not admitted or len(common) < min_windows or result["common_coverage"] < min_coverage:
    for _, row in admitted:
      row["rejections"].append("insufficient_common_joint_support")
    return result
  before, eligible = joint_summary(baseline, common), []
  result["baseline"] = before
  for raw, row in admitted:
    row["metrics"] = joint_summary(raw, common)
    reasons, objective = joint_constraints(row["metrics"], before)
    row.update(rejections=reasons, objective=objective)
    if not reasons:
      eligible.append(row)
  if eligible:
    result["winner"] = min(eligible, key=lambda row: (row["objective"], json.dumps(row["parameters"], sort_keys=True)))
  return result


def validation_search(records, models, partition="validation", max_windows=120):
  if partition != "validation":
    raise ValueError("gain selection is restricted to validation records")
  axes = {}
  for axis, model in models.items():
    baseline = evaluate_axis(records, model, axis, None, max_windows)
    grid = ([{"lateral_gain": gain} for gain in GRID] if axis == "lat" else
            [{"acceleration_gain": accel, "braking_gain": brake} for accel, brake in product(GRID, repeat=2)])
    raw = [evaluate_axis(records, model, axis, params, max_windows) for params in grid]
    axes[axis] = select_axis(raw, baseline, axis, baseline["attempted"])
  candidate = {}
  for result in axes.values():
    if result["winner"]:
      candidate.update(result["winner"]["parameters"])
  joint = None
  if all(axis in models for axis in ("lat", "long")):
    cache = {}
    baseline = evaluate_joint(records, models, None, max_windows, cache)
    raw_joint = [evaluate_joint(records, models, {"lateral_gain": lat, "acceleration_gain": accel, "braking_gain": brake},
                                max_windows, cache) for lat, accel, brake in product(GRID, repeat=3)]
    joint = select_joint(raw_joint, baseline)
  global_trace = []
  for lat, accel, brake in product(GRID, repeat=3):
    reasons = []
    for axis, params in (("lat", {"lateral_gain": lat}), ("long", {"acceleration_gain": accel, "braking_gain": brake})):
      if axis not in axes:
        reasons.append(f"unavailable_{axis}_plant")
        continue
      row = next(entry for entry in axes[axis]["trace"] if entry["parameters"] == params)
      reasons.extend(f"{axis}:{reason}" for reason in row["rejections"])
    if joint is not None:
      joint_row = next(row for row in joint["trace"] if row["parameters"] == {
        "lateral_gain": lat, "acceleration_gain": accel, "braking_gain": brake})
      reasons = list(joint_row["rejections"])
    if joint is not None and joint["winner"] is None:
      reasons.append("global_joint_unverified")
    if joint is not None and not reasons:
      reasons.append("native_low_speed_stopping_unverified")
    global_trace.append({"parameters": {"lateral_gain": lat, "acceleration_gain": accel, "braking_gain": brake},
                         "rejections": reasons})
  return {"status": "insufficient_evidence", "candidate": None,
          "partial_candidate": candidate or None, "axes": axes, "global_trace": global_trace,
          "moving_candidate": joint["winner"]["parameters"] if joint is not None and joint["winner"] else None,
          "joint_evaluation": joint if joint is not None else {"status": "unavailable_joint_plant"},
          "selection_uses": "validation target tracking only; comfort breaks tracking ties after hard constraints"}


def heldout_test(records, models, partial_candidate, max_windows=120):
  result = {}
  for axis, model in models.items():
    keys = ("lateral_gain",) if axis == "lat" else ("acceleration_gain", "braking_gain")
    if not partial_candidate or any(key not in partial_candidate for key in keys):
      continue
    parameters = {key: partial_candidate[key] for key in keys}
    baseline = evaluate_axis(records, model, axis, None, max_windows)
    candidate = evaluate_axis(records, model, axis, parameters, max_windows)
    common = sorted(set(baseline["windows"]) & set(candidate["windows"]))
    result[axis] = {"attempted_windows": baseline["attempted"], "common_windows": len(common),
                   "coverage": len(common) / max(1, baseline["attempted"]), "window_ids": common,
                   "baseline": _summary(baseline, common, axis), "candidate": _summary(candidate, common, axis)} if common else {
                     "attempted_windows": baseline["attempted"], "common_windows": 0, "status": "insufficient_test_support"}
    if common:
      result[axis]["constraint_rejections"] = _constraints(result[axis]["candidate"], result[axis]["baseline"], axis, MIN_DIRECTION_POINTS)
      result[axis]["support_sufficient"] = len(common) >= MIN_WINDOWS and result[axis]["coverage"] >= MIN_COVERAGE
  return result


def heldout_joint_test(records, models, selected, max_windows=120):
  """Evaluate a frozen moving-policy selection once; never select on test."""
  if selected is None:
    return {"status": "no_frozen_moving_candidate"}
  baseline = evaluate_joint(records, models, None, max_windows)
  candidate = evaluate_joint(records, models, selected, max_windows)
  common = sorted(set(baseline["windows"]) & set(candidate["windows"]))
  result = {"attempted_windows": baseline["attempted"], "common_windows": len(common), "window_ids": common,
            "coverage": len(common) / max(1, baseline["attempted"])}
  if not common:
    return {**result, "status": "insufficient_joint_test_support"}
  before, after = joint_summary(baseline, common), joint_summary(candidate, common)
  reasons, objective = joint_constraints(after, before)
  result.update(baseline=before, candidate=after, constraint_rejections=reasons, objective=objective,
                support_sufficient=len(common) >= MIN_WINDOWS and result["coverage"] >= MIN_COVERAGE)
  return result


def validate_output(dataset, plant_report, output):
  if output.resolve().is_relative_to(dataset.resolve()):
    raise ValueError("search outputs must stay outside the extracted dataset")
  inputs = [plant_report, *dataset.glob("*")]
  for name in ("manifest.json", "search-trace.json", "candidate.json"):
    target = output / name
    if target.resolve() == plant_report.resolve() or (target.exists() and any(path.is_file() and target.samefile(path) for path in inputs)):
      raise ValueError("search output aliases an input evidence file")


def source_runtime_evidence():
  directory = Path(__file__).resolve().parent
  return {"canonical_source_sha256": {name: sha256(directory / name) for name in
                                      ("controller_tune.py", "controller_extract.py", "controller_metrics.py",
                                       "controller_dynamics.py", "lateral_report.py")},
          "python_version": sys.version, "python_executable": sys.executable, "numpy_version": np.__version__}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--plant-report", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--max-windows", type=int, default=120)
  args = parser.parse_args()
  if args.max_windows < MIN_WINDOWS:
    parser.error(f"--max-windows must be at least {MIN_WINDOWS}")
  validate_output(args.dataset, args.plant_report, args.out)
  plant_report = json.loads(args.plant_report.read_text(encoding="utf-8"))
  paths = [(path, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))) for path in sorted(args.dataset.glob("*.npz"))]
  tune.enrich_route_metadata(paths)
  tune.resolve_gitlinks(paths, args.source_repo)
  paths = [(path, meta) for path, meta in paths if tune.cohort_key(meta) == plant_report["cohort"]]
  split = validate_split(paths, plant_report["split"])
  models, rejected = trusted_models(plant_report)
  validation = tune.load_records([(p, m) for p, m in paths if m["route"] in split["validation"]]) if models else []
  result = validation_search(validation, models, max_windows=args.max_windows)
  # Selection is frozen before any test arrays are loaded or evaluated.
  selected = json.loads(json.dumps(result["partial_candidate"]))
  moving_selected = json.loads(json.dumps(result["moving_candidate"]))
  test_records = tune.load_records([(p, m) for p, m in paths if m["route"] in split["test"]]) if selected or moving_selected else []
  test = heldout_test(test_records, models, selected, args.max_windows)
  joint_test = heldout_joint_test(test_records, models, moving_selected, args.max_windows)
  manifest = {"cohort": plant_report["cohort"], "split": split, "status": result["status"],
              "plant_report": str(args.plant_report.resolve()), "plant_report_sha256": sha256(args.plant_report),
              "search_source_sha256": sha256(Path(__file__)), "grid": GRID, "max_windows_per_axis_partition": args.max_windows,
              "minimum_common_coverage": MIN_COVERAGE, "minimum_windows": MIN_WINDOWS,
              "minimum_direction_points": MIN_DIRECTION_POINTS, "constraints": LIMITS,
              "rejected_plants": rejected, "heldout_test": test, "sources": [source_reference(p, m) for p, m in paths],
              "heldout_joint_test": joint_test,
              "scope": "historical same-cohort frozen-target plant simulation; not fresh RDF/current-model or closed-loop replay",
              "joint_evaluation": result["joint_evaluation"],
              "stops": "below3m/s and override/stale/gap windows excluded by canonical clean_mask/window_indices; stopping behavior unsupported",
              "production_enabled": False, "steer_max": 409}
  manifest.update(source_runtime_evidence())
  candidate = {"production_enabled": False, "steer_max": 409, "cohort": plant_report["cohort"],
               "complete_controller_candidate": result["candidate"], "partial_axis_candidates": selected,
               "moving_controller_candidate": moving_selected,
               "status": result["status"], "plant_report_sha256": manifest["plant_report_sha256"],
               "warning": manifest["scope"]}
  args.out.mkdir(parents=True, exist_ok=True)
  for name, value in (("manifest.json", manifest), ("search-trace.json", result), ("candidate.json", candidate)):
    (args.out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  print(json.dumps({"out": str(args.out.resolve()), "cohort": manifest["cohort"], "status": result["status"],
                    "partial_candidate": selected, "rejected_plants": rejected, "evaluated_axes": list(result["axes"])}))


if __name__ == "__main__":
  main()
