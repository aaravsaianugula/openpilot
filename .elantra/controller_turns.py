"""Research turn scoring on frozen FRD curvature; no controller or file I/O.

Positive reference curvature means RIGHT, negative means LEFT. Direction and
phase never depend on candidate speed, acceleration, yaw or command. Tracking
uses caller-supplied physical lateral-acceleration targets (m/s²); jerk is m/s³.
Near-straight and slope thresholds segment geometry, not controller gains.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

DIRECTIONS = ("left", "right")
PHASES = ("entry", "steady", "exit", "unknown")
STEER_MAX = 409
NEAR_STRAIGHT = .001  # 1/m: radius >= 1 km is excluded from turn objectives.
PHASE_SLOPE = .002  # 1/(m*s), causal change in absolute reference curvature.
CONTRACT = {"kind": "turn_metrics_v1", "frame": "x forward,y right,z down; positive yaw/curvature RIGHT",
            "tracking_units": "m/s^2", "jerk_units": "m/s^3", "near_straight_per_m": NEAR_STRAIGHT,
            "phase_slope_per_m_s": PHASE_SLOPE, "steer_max": STEER_MAX,
            "phase": "backward absolute-curvature difference only; absent/nonpast/stale prefix is unknown",
            "policy": "identical common windows/reference/points; both sides and all entry/steady/exit strata required",
            "limits": {"direction_mae_ratio": 1.02, "direction_mae_slack": .005, "direction_p95_ratio": 1.05,
                       "direction_p95_slack": .01, "jerk_p95": 5., "saturation_fraction": .05,
                       "saturation_increase": .01, "minimum_balanced_improvement": .01},
            "default_support": {"direction_points": 50, "phase_points": 10, "windows_per_stratum": 2}}


def window_metrics(reference_curvature, simulated_ay, target_ay, torque, *, dt_s=.05,
                   reference_prefix=None, measured_yaw=None):
  """Score one uniform-grid window; prefix=(strictly past age seconds, curvature).

  Prefix age must match one sample period within 2 ms. First-row jerk is unknown;
  no preceding candidate state is invented. Missing values and commands beyond
  409 are retained as unsupported rows, never silently accepted after pooling.
  Measured yaw is an optional sign diagnostic, never a segmentation input.
  """
  k, state, target, command = (np.asarray(a, dtype=float) for a in
                               (reference_curvature, simulated_ay, target_ay, torque))
  if (not np.isfinite(dt_s) or dt_s <= 0 or k.ndim != 1 or not len(k)
      or any(a.shape != k.shape for a in (state, target, command))):
    raise ValueError("matching nonempty 1D arrays and finite positive dt_s required")
  yaw = np.full(len(k), np.nan) if measured_yaw is None else np.asarray(measured_yaw, dtype=float)
  if yaw.shape != k.shape:
    raise ValueError("measured_yaw must match the reference grid")
  prefix, prefix_rejection = None, None
  if reference_prefix is not None:
    if len(reference_prefix) != 2:
      raise ValueError("reference_prefix must be (age_s, curvature)")
    age, curvature = reference_prefix
    if not np.isfinite([age, curvature]).all() or age <= 0 or abs(age-dt_s) > .002 + 1e-9:
      prefix_rejection = "nonpast_stale_or_missing_reference_prefix"
    else:
      prefix = [float(age), float(curvature)]
  direction = np.full(len(k), "unknown", dtype=object)
  direction[np.isfinite(k) & (np.abs(k) <= NEAR_STRAIGHT)] = "straight"
  direction[np.isfinite(k) & (k < -NEAR_STRAIGHT)] = "left"
  direction[np.isfinite(k) & (k > NEAR_STRAIGHT)] = "right"
  phase = np.full(len(k), "unknown", dtype=object)
  phase[direction == "straight"] = "straight"
  for i in np.flatnonzero(np.isin(direction, DIRECTIONS)):
    previous = k[i-1] if i else prefix[1] if prefix is not None else np.nan
    if not np.isfinite(previous) or (previous*k[i] < 0 and abs(previous) > NEAR_STRAIGHT):
      continue  # A reversal with no observed straight crossing has unknown turn phase.
    slope = (abs(k[i])-abs(previous)) / (dt_s if i else prefix[0])
    phase[i] = "entry" if slope > PHASE_SLOPE else "exit" if slope < -PHASE_SLOPE else "steady"
  valid = np.isfinite(k) & np.isfinite(state) & np.isfinite(target) & np.isfinite(command)
  invalid_torque = np.isfinite(command) & (np.abs(command) > STEER_MAX + 1e-8)
  valid &= ~invalid_torque
  jerk = np.r_[np.nan, np.abs(np.diff(state))/dt_s]
  jerk_valid = valid & np.r_[False, valid[:-1]] & np.isfinite(jerk)

  def points(mask):
    good, differentiated = mask & valid, mask & jerk_valid
    return {"attempted_rows": np.flatnonzero(mask).tolist(), "rows": np.flatnonzero(good).tolist(),
            "errors": np.abs(state[good]-target[good]).tolist(),
            "jerk_rows": np.flatnonzero(differentiated).tolist(), "jerk": jerk[differentiated].tolist(),
            "saturated": (np.abs(command[good]) >= STEER_MAX-1e-8).astype(float).tolist()}

  identity = json.dumps({"contract": CONTRACT, "dt_s": float(dt_s), "prefix": prefix}, sort_keys=True).encode()
  canonical = np.where(np.isfinite(k), k, np.nan).astype("<f8")
  turn = np.isin(direction, DIRECTIONS)
  opposite = np.isfinite(yaw) & (((direction == "left") & (yaw > 0)) | ((direction == "right") & (yaw < 0)))
  return {"contract": dict(CONTRACT), "reference_sha256": hashlib.sha256(identity+canonical.tobytes()).hexdigest(),
          "reference_direction": direction.tolist(), "reference_phase": phase.tolist(), "prefix_rejection": prefix_rejection,
          "missing_reference_rows": np.flatnonzero(~np.isfinite(k)).tolist(),
          "unsupported_rows": np.flatnonzero(~valid).tolist(), "invalid_torque_rows": np.flatnonzero(invalid_torque).tolist(),
          "tracking": {side: points(direction == side) for side in DIRECTIONS},
          "phases": {side: {name: points((direction == side) & (phase == name)) for name in PHASES} for side in DIRECTIONS},
          "measured_yaw": {"opposite_reference_points": int(opposite.sum()),
                           "unknown_turn_points": int(np.sum(turn & ~np.isfinite(yaw)))}, "production_enabled": False}


def _pool(rows):
  pooled = {key: [] for key in ("errors", "jerk", "saturated", "point_ids", "attempted_point_ids", "jerk_point_ids")}
  windows = 0
  for name, row in rows:
    windows += bool(row["rows"])
    for key in ("errors", "jerk", "saturated"):
      pooled[key].extend(row[key])
    for key, source in (("point_ids", "rows"), ("attempted_point_ids", "attempted_rows"), ("jerk_point_ids", "jerk_rows")):
      pooled[key].extend([name, i] for i in row[source])
  errors, jerk, saturated = pooled["errors"], pooled["jerk"], pooled["saturated"]
  pooled.update(points=len(errors), attempted_points=len(pooled["attempted_point_ids"]), windows=windows,
                unsupported_points=len(pooled["attempted_point_ids"])-len(errors),
                mae=float(np.mean(errors)) if errors else None, p95=float(np.percentile(errors, 95)) if errors else None,
                jerk_p95=float(np.percentile(jerk, 95)) if jerk else None,
                saturation_fraction=float(np.mean(saturated)) if saturated else None)
  return pooled


def _missing_window_fields(row):
  if not isinstance(row, dict):
    return ["turn_metrics"]
  required = {"reference_sha256": str, "tracking": dict, "phases": dict, "measured_yaw": dict,
              "unsupported_rows": list, "missing_reference_rows": list, "invalid_torque_rows": list}
  missing = [key for key, kind in required.items() if not isinstance(row.get(key), kind)]
  if row.get("contract") != CONTRACT:
    missing.append("contract")
  if missing:
    return missing
  for key in ("opposite_reference_points", "unknown_turn_points"):
    if not isinstance(row["measured_yaw"].get(key), int):
      missing.append(f"measured_yaw.{key}")
  for side in DIRECTIONS:
    nodes = [(f"tracking.{side}", row["tracking"].get(side))]
    if not isinstance(row["phases"].get(side), dict):
      missing.append(f"phases.{side}")
    else:
      nodes.extend((f"phases.{side}.{phase}", row["phases"][side].get(phase)) for phase in PHASES)
    for name, node in nodes:
      if not isinstance(node, dict):
        missing.append(name)
        continue
      missing.extend(f"{name}.{field}" for field in ("attempted_rows", "rows", "errors", "jerk_rows", "jerk", "saturated")
                     if not isinstance(node.get(field), list))
  return missing


def summarize(windows, common):
  """Pool exactly the declared common windows; old pooled-only rows fail closed."""
  if len(set(common)) != len(common):
    raise ValueError("common window IDs must be unique")
  common = sorted(common)
  missing = [name for name in common if name not in windows]
  invalid_fields = {name: fields for name in common if name in windows and (fields := _missing_window_fields(windows[name]))}
  invalid = list(invalid_fields)
  admitted = [name for name in common if name not in missing and name not in invalid]
  tracking = {side: _pool([(name, windows[name]["tracking"][side]) for name in admitted]) for side in DIRECTIONS}
  scores = [tracking[side]["mae"] for side in DIRECTIONS]
  return {"kind": "turn_summary_v1", "contract": dict(CONTRACT), "common_window_ids": common,
          "missing_windows": missing, "invalid_windows": invalid, "invalid_window_fields": invalid_fields,
          "reference_by_window": {name: windows[name]["reference_sha256"] for name in admitted},
          "unsupported_points": sum(len(windows[name]["unsupported_rows"]) for name in admitted),
          "missing_reference_points": sum(len(windows[name]["missing_reference_rows"]) for name in admitted),
          "invalid_torque_points": sum(len(windows[name]["invalid_torque_rows"]) for name in admitted),
          "measured_yaw": {key: sum(windows[name]["measured_yaw"][key] for name in admitted)
                           for key in ("opposite_reference_points", "unknown_turn_points")},
          "tracking": tracking,
          "phases": {side: {phase: _pool([(name, windows[name]["phases"][side][phase]) for name in admitted])
                            for phase in PHASES} for side in DIRECTIONS},
          "target_objective": float(np.mean(scores)) if all(score is not None for score in scores) else None,
          "production_enabled": False}


def constraints(metrics, baseline, *, min_direction_points=50, min_phase_points=10, min_windows=2, require_improvement=True):
  """Hard side/phase gates precede a balanced objective; these are research limits."""
  if min(min_direction_points, min_phase_points, min_windows) < 1:
    raise ValueError("positive direction/phase/window support minima required")
  if any(not isinstance(row, dict) or row.get("kind") != "turn_summary_v1" or row.get("contract") != CONTRACT or row.get("invalid_windows")
         for row in (metrics, baseline)):
    return ["missing_turn_metrics_contract"]
  reasons = []
  limits = CONTRACT["limits"]
  if metrics["common_window_ids"] != baseline["common_window_ids"] or any(row["missing_windows"] for row in (metrics, baseline)):
    reasons.append("missing_turn_windows")
  if metrics["reference_by_window"] != baseline["reference_by_window"]:
    reasons.append("turn_reference_mismatch")
  if any(row["unsupported_points"] for row in (metrics, baseline)):
    reasons.append("unsupported_turn_points")
  if any(row["missing_reference_points"] for row in (metrics, baseline)):
    reasons.append("missing_turn_reference")
  for side in DIRECTIONS:
    after, before = metrics["tracking"][side], baseline["tracking"][side]
    if after["point_ids"] != before["point_ids"] or after["attempted_point_ids"] != before["attempted_point_ids"]:
      reasons.append("turn_point_population_mismatch")
    for name, a, b, minimum in [(side, after, before, min_direction_points), *[
      (f"{side}_{phase}", metrics["phases"][side][phase], baseline["phases"][side][phase], min_phase_points)
      for phase in PHASES[:3]]]:
      if min(a["points"], b["points"]) < minimum or min(a["windows"], b["windows"]) < min_windows:
        reasons.append(f"missing_{name}_support")
        continue
      if a["mae"] > b["mae"]*limits["direction_mae_ratio"] + limits["direction_mae_slack"]:
        reasons.append(f"{name}_tracking_regression")
      if a["p95"] > b["p95"]*limits["direction_p95_ratio"] + limits["direction_p95_slack"]:
        reasons.append(f"{name}_tail_regression")
    if after["jerk_p95"] is None or before["jerk_p95"] is None or after["jerk_p95"] > limits["jerk_p95"]:
      reasons.append(f"{side}_jerk_limit")
    if (after["saturation_fraction"] is None or before["saturation_fraction"] is None
        or after["saturation_fraction"] > min(limits["saturation_fraction"],
                                             before["saturation_fraction"]+limits["saturation_increase"])):
      reasons.append(f"{side}_saturation_limit")
  objective, before = metrics["target_objective"], baseline["target_objective"]
  if require_improvement and (objective is None or before is None or objective >= before*(1-limits["minimum_balanced_improvement"])):
    reasons.append("no_meaningful_balanced_turn_improvement")
  return sorted(set(reasons))
