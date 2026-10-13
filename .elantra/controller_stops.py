#!/usr/bin/env python3
"""Observation-only stopping and acceleration response evidence from recorded drives.

The model's predicted stationary position is a prediction, never stop-line GT.
Actual travel comes only from measured speed/yaw. Candidate braking parameters
use training-route command/response latency; margins are unoptimized sensitivity
settings, and no candidate has driven or changed these recordings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from controller_extract import hold_calibration, model_capture_times
from controller_metrics import cohort_identity, extraction_context, source_reference, summarize, validate_output
from controller_tune import DT, cohort_key, enrich_route_metadata, resolve_gitlinks
from lateral_report import band_of

MARGINS = (0, .5, 1, 1.5, 2)
RESPONSE_PREFIX = int(round(.5 / DT))
COMMAND_PREFIX = int(round(.3 / DT))
FIELDS = ("t", "v", "a", "yaw", "accel_command", "accel_target", "long_active", "model_t", "model_capture_t", "model_horizon",
          "model_x", "model_y", "model_valid", "car_state_valid", "car_control_valid", "car_output_valid",
          "location_valid", "location_measurement_valid", "calibration_valid", "calibrated",
          "planner_valid", "age_longitudinalPlan", "age_carOutput", "age_carState",
          "steering_pressed", "brake_pressed", "gas_pressed")


def stopping_request(v, dist, latency, margin):
  """Experimental law using current speed and recorded model prediction only.

  Unreachable means latency/margin exhaust distance, or the required deceleration
  exceeds the existing 3.5 m/s² limit. No larger command is invented.
  """
  if not np.isfinite([v, dist, latency, margin]).all() or min(v, dist, latency, margin) < 0:
    raise ValueError("finite nonnegative speed, distance, latency and margin required")
  if v == 0:
    return 0., False
  remaining = dist - v * latency - margin
  if remaining <= 0:
    return -3.5, True
  required = -v * v / (2 * remaining)
  return float(max(-3.5, required)), bool(required < -3.5)


def prediction_at(d, i):
  age = d["t"][i] - d["model_t"][i]
  if d["model_valid"][i] != 1 or not np.isfinite(age) or not -1e-8 <= age <= .05 + 1e-8:
    return None
  capture = d["model_capture_t"][i]
  capture_age = d["t"][i] - capture
  if (not np.isfinite(capture) or capture > d["model_t"][i] + 1e-8
      or capture < d["t"][0] or not 0 <= capture_age <= .2 + 1e-8):
    return None
  h, x, y = (d[key][i] for key in ("model_horizon", "model_x", "model_y"))
  if not np.isfinite([h, x, y]).all() or h[0] != 0 or np.any(np.diff(h) <= 0):
    return None
  arc = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
  stationary = np.diff(arc) / np.diff(h) < .3
  start = None
  for j, slow in enumerate(stationary):
    if slow:
      start = j if start is None else start
      duration = h[j+1] - h[start]
      if duration >= 1:
        return {"distance_m": float(arc[start]), "stop_horizon_s": float(h[start]),
                "stationary_s": float(duration), "model_age_s": float(max(0, age)),
                "capture_t": float(capture), "capture_age_s": float(capture_age),
                "model_stop_xy": [float(x[start] - x[0]), float(y[start] - y[0])]}
    else:
      start = None
  return None


def motion_valid(d):
  finite = np.logical_and.reduce([np.isfinite(d[k]) for k in ("t", "v", "a", "yaw")])
  return finite & np.logical_and.reduce([d[k] == 1 for k in
    ("car_state_valid", "location_valid", "location_measurement_valid", "calibration_valid", "calibrated")])


def stitch_route(paths, calibration_hold=True):
  """Stitch only chronological contiguous segment boundaries, then break invalid motion."""
  pieces, coverage = [], Counter()
  for path, _meta in paths:
    with np.load(path, allow_pickle=False) as archive:
      raw = {key: archive[key] for key in archive.files}
    data = hold_calibration(raw) if calibration_hold else raw
    data["model_capture_t"] = model_capture_times(raw)
    t = data["t"]
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
      raise ValueError(f"nonempty strictly monotonic extracted timestamps required: {path}")
    piece = {key: data[key] for key in FIELDS}
    piece["source_segment"] = np.full(len(t), path.stem)
    piece["source_segment_start_t"] = np.full(len(t), t[0])
    pieces.append(piece)
    coverage["input_rows"] += len(t)
    coverage["unknown_capture_rows"] += int(np.sum(~np.isfinite(data["model_capture_t"])))
  pieces.sort(key=lambda d: d["t"][0])
  runs, current = [], []
  for piece in pieces:
    if current:
      delta = piece["t"][0] - current[-1]["t"][-1]
      if 0 < delta <= .075 + 1e-8:
        coverage["contiguous_segment_joins"] += 1
      else:
        coverage["segment_gap_breaks"] += 1
        if delta <= 0:
          coverage["overlapping_segment_boundaries"] += 1
        runs.append(current)
        current = []
    current.append(piece)
  if current:
    runs.append(current)
  blocks = []
  for pieces in runs:
    d = {key: np.concatenate([p[key] for p in pieces]) for key in (*FIELDS, "source_segment", "source_segment_start_t")}
    valid = motion_valid(d)
    coverage["invalid_motion_rows"] += int((~valid).sum())
    start = None
    for i in range(len(valid) + 1):
      gap = i < len(valid) and i > 0 and not 0 < d["t"][i] - d["t"][i-1] <= .075 + 1e-8
      if (i == len(valid) or not valid[i] or gap) and start is not None:
        blocks.append({key: a[start:i] for key, a in d.items()})
        start = None
      if i < len(valid) and valid[i] and start is None:
        start = i
  coverage["valid_motion_rows"] = sum(len(b["t"]) for b in blocks)
  coverage["motion_blocks"] = len(blocks)
  return blocks, dict(coverage)


def first_sustained_time(values, t, duration):
  start = None
  for i, value in enumerate(values):
    start = (i if start is None else start) if value else None
    if start is not None and t[i] - t[start] >= duration - 1e-8:
      return start
  return None


def signal_onset(d, start, end, key, valid_key, age_key, threshold, duration=0.):
  values, ages = d[key][start:end], d[age_key][start:end]
  available = (d[valid_key][start:end] == 1) & np.isfinite(values) & np.isfinite(ages) & (ages >= 0) & (ages <= .2 + 1e-8)
  onset = first_sustained_time(available & (values < threshold), d["t"][start:end], duration)
  coverage = {"rows": len(values), "valid_fresh_rows": int(available.sum()), "unavailable_rows": int((~available).sum()),
              "status": "observed_valid_negative" if onset is not None else "no_valid_negative_observed",
              "unavailable_rows_before_onset": int((~available[:onset+1]).sum()) if onset is not None else None,
              "maximum_service_age_s": .2}
  return None if onset is None else float(d["t"][start+onset] - d["t"][start]), coverage


def travel(d, start, end):
  t = d["t"][start:end]
  if not len(t):
    return 0., [0., 0.]
  return travel_between(d, float(t[0]), float(t[-1]))


def travel_between(d, start_t, end_t):
  """Measured capture-origin label; interpolate only admitted boundary samples.

  For capture-to-decision distance every sample is already in the decision's
  past. Future samples used to label the actual stop are never controller inputs.
  """
  t = d["t"]
  if not np.isfinite([start_t, end_t]).all() or start_t < t[0] or end_t > t[-1] or start_t > end_t:
    raise ValueError("measured travel requires a complete in-range time interval")
  if start_t == end_t:
    return 0., [0., 0.]
  lo = int(np.searchsorted(t, start_t, side="right") - 1)
  hi = int(np.searchsorted(t, end_t, side="left"))
  if np.any(np.diff(t[lo:hi+1]) > .075 + 1e-8) or not np.all(motion_valid(d)[lo:hi+1]):
    raise ValueError("invalid measured motion or time gap in travel interval")
  nodes = np.r_[start_t, t[(t > start_t) & (t < end_t)], end_t]
  speed = np.interp(nodes, t, d["v"])
  yaw = np.interp(nodes, t, d["yaw"])
  dt = np.diff(nodes)
  turns = .5 * (yaw[:-1] + yaw[1:]) * dt
  heading = np.cumsum(turns) - .5 * turns
  step = .5 * (speed[:-1] + speed[1:]) * dt
  xy = [float(np.sum(step * np.cos(heading))), float(np.sum(step * np.sin(heading)))]
  return float(np.sum(np.abs(step))), xy


def stop_event(d, i, prediction, route, op_long):
  capture_t = prediction["capture_t"]
  capture_lo = int(np.searchsorted(d["t"], capture_t, side="right") - 1)
  past_travel, _ = travel_between(d, capture_t, float(d["t"][i]))
  end = int(np.searchsorted(d["t"], d["t"][i] + 15 + 1e-8, side="right"))
  local_stop = first_sustained_time(d["v"][i:end] <= .3, d["t"][i:end], .5)
  actual_stop = None if local_stop is None else i + local_stop
  confirmed = None if actual_stop is None else int(np.searchsorted(d["t"], d["t"][actual_stop] + .5 - 1e-8))
  last = end - 1 if confirmed is None else confirmed
  if op_long is not True:
    regime = "cp_long_disabled" if op_long is False else "unknown_cp_long"
  elif d["car_control_valid"][i] != 1 or d["long_active"][i] not in (0, 1):
    regime = "unknown_control_state"
  else:
    regime = "comma_longitudinal_active" if d["long_active"][i] == 1 else "longitudinal_inactive"
  source_origin = d.get("source_segment_start_t", np.full(len(d["t"]), d["t"][0]))[i]
  event = {"route": route, "source_segment": str(d["source_segment"][i]), "t": float(d["t"][i]),
           "prediction_t": float(d["t"][i]), "actual_stop_t": None if actual_stop is None else float(d["t"][actual_stop]),
           "capture_t": capture_t, "capture_age_s": prediction["capture_age_s"],
           "travel_since_capture_m": past_travel,
           "remaining_predicted_stop_distance_m": max(0., prediction["distance_m"] - past_travel),
           "prediction_segment_offset_s": float(d["t"][i] - source_origin),
           "covering_segments": list(dict.fromkeys(str(s) for s in d["source_segment"][i:last+1])),
           "model_stop_xy": prediction["model_stop_xy"], "measured_stop_xy": None,
           "model_age_basis": "exact EncodeIdx frame capture",
           "control_regime": regime, "op_long_at_prediction": op_long,
           "long_active_at_prediction": float(d["long_active"][i]) if np.isfinite(d["long_active"][i]) else None,
           "speed_mps": float(d["v"][i]), "predicted_stop_distance_m": prediction["distance_m"],
           "predicted_stop_horizon_s": prediction["stop_horizon_s"], "model_age_s": prediction["model_age_s"],
           "outcome": "stopped" if confirmed is not None else ("timeout_15s" if d["t"][last] - d["t"][i] >= 14.95 else "motion_block_ended"),
           "actual_stop_distance_m": None, "prediction_error_m": None, "actual_stop_endpoint_xy_m": None}
  if actual_stop is not None:
    distance, endpoint = travel_between(d, capture_t, float(d["t"][actual_stop]))
    event.update(actual_stop_distance_m=distance, prediction_error_m=distance - prediction["distance_m"],
                 actual_stop_endpoint_xy_m=endpoint, measured_stop_xy=endpoint,
                 actual_stop_s=float(d["t"][actual_stop] - d["t"][i]),
                 stop_confirmed_s=float(d["t"][confirmed] - d["t"][i]))
  event["distance_until_observation_end_m"] = travel_between(d, capture_t, float(d["t"][last]))[0]
  censor = None
  conditions = {"manual_brake": d["brake_pressed"] == 1, "manual_gas": d["gas_pressed"] == 1,
                "manual_steer": d["steering_pressed"] == 1, "disengage": d["long_active"] == 0,
                "unknown_driver_state": np.logical_or.reduce([~np.isfinite(d[k]) for k in
                  ("brake_pressed", "gas_pressed", "steering_pressed")]),
                "unknown_control_state": ~np.isin(d["long_active"], [0, 1]) | (d["car_control_valid"] != 1)}
  for name, values in conditions.items():
    hits = np.flatnonzero(values[i:last+1])
    index = i + int(hits[0]) if len(hits) else None
    event[f"first_{name}_s"] = None if index is None else float(d["t"][index] - d["t"][i])
    if index is not None:
      censor = index if censor is None else min(censor, index)
  event["onset_signal_coverage"] = {}
  for name, key, valid_key, age_key, threshold, duration in (
      ("planner_negative", "accel_target", "planner_valid", "age_longitudinalPlan", -.05, 0.),
      ("command_negative", "accel_command", "car_output_valid", "age_carOutput", -.05, 0.),
      ("measured_deceleration", "a", "car_state_valid", "age_carState", -.1, .15)):
    event[f"first_{name}_s"], event["onset_signal_coverage"][name] = signal_onset(
      d, i, last+1, key, valid_key, age_key, threshold, duration)
  if regime != "comma_longitudinal_active":
    censor = i
  prefix = slice(capture_lo, i+1)
  prefix_valid = (np.all(d["car_control_valid"][prefix] == 1) and np.all(d["long_active"][prefix] == 1)
                  and all(np.all(d[k][prefix] == 0) for k in ("brake_pressed", "gas_pressed", "steering_pressed")))
  event["capture_prefix_driver_control_valid"] = bool(prefix_valid)
  event["comma_only_eligible"] = confirmed is not None and censor is None and prefix_valid and regime == "comma_longitudinal_active"
  event["comma_only_censor_s"] = None if censor is None else float(d["t"][censor] - d["t"][i])
  event["comma_only_prefix_distance_m"] = travel(d, i, (last if censor is None else censor) + 1)[0]
  event["censor_t"] = float(d["t"][censor if censor is not None else last])
  event["censor_cause"] = (["not_comma_longitudinal_at_prediction"] if regime != "comma_longitudinal_active" else
    [name for name, values in conditions.items() if censor is not None and values[censor]])
  if censor is None:
    event["censor_cause"] = ["actual_stop_confirmed" if confirmed is not None else event["outcome"]]
  return event


def stop_events(d, route, op_long):
  events, i = [], 0
  while i < len(d["t"]):
    prediction = prediction_at(d, i)
    if prediction is None or d["v"][i] < 2:
      i += 1
      continue
    events.append(stop_event(d, i, prediction, route, op_long))
    # A new episode requires a confirmed actual stop or two seconds without a model stop.
    stationary, absent = None, None
    i += 1
    while i < len(d["t"]):
      stationary = (d["t"][i] if stationary is None else stationary) if d["v"][i] <= .3 else None
      absent = (d["t"][i] if absent is None else absent) if prediction_at(d, i) is None else None
      reset = (stationary is not None and d["t"][i] - stationary >= .5 - 1e-8) or (
        absent is not None and d["t"][i] - absent >= 2 - 1e-8)
      i += 1
      if reset:
        break
  return events


def response_clean(d):
  clean = motion_valid(d)
  clean &= np.logical_and.reduce([np.isfinite(d[k]) for k in ("accel_command", "long_active")])
  clean &= (d["car_control_valid"] == 1) & (d["car_output_valid"] == 1) & (d["long_active"] == 1)
  for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
    clean &= d[key] == 0
  return clean & (d["accel_command"] >= -3.5) & (d["accel_command"] <= 2)


def response_events(d, route, op_long):
  if op_long is not True:
    return []
  clean, events, next_t = response_clean(d), [], -np.inf
  for i in range(RESPONSE_PREFIX, len(d["t"])):
    if d["t"][i] < next_t or not np.isfinite(d["accel_command"][i-COMMAND_PREFIX:i+1]).all():
      continue
    previous = float(np.median(d["accel_command"][i-COMMAND_PREFIX:i]))
    change, command = d["accel_command"][i] - previous, d["accel_command"][i]
    direction = -1 if change <= -.25 and command <= -.2 else (1 if change >= .25 and command >= .2 else 0)
    if not direction:
      continue
    next_t = d["t"][i] + 3
    end = min(len(d["t"]), int(np.searchsorted(d["t"], d["t"][i] + 3, side="right")))
    baseline = float(np.median(d["a"][i-RESPONSE_PREFIX:i]))
    onset = first_sustained_time(direction * (d["a"][i:end] - baseline) >= .15, d["t"][i:end], .15)
    response_end = None if onset is None else int(np.searchsorted(d["t"], d["t"][i+onset] + .15 - 1e-8)) + 1
    effect_end = end if onset is None else response_end
    event = {"route": route, "source_segment": str(d["source_segment"][i]), "t": float(d["t"][i]),
             "direction": "braking" if direction < 0 else "acceleration", "speed_mps": float(d["v"][i]),
             "command_before_mps2": previous, "command_after_mps2": float(command), "measured_before_mps2": baseline,
             "response_latency_s": None, "measured_response_mps2": None}
    if not np.all(clean[i-RESPONSE_PREFIX:effect_end]) or np.any(np.diff(d["t"][i-RESPONSE_PREFIX:effect_end]) > .075):
      event["status"] = "intervention_or_invalid_window"
    elif direction * (float(np.median(d["a"][i-3:i])) - baseline) >= .15:
      event["status"] = "already_changing_before_command"
    elif onset is None:
      event["status"] = "no_observed_response" if d["t"][end-1] - d["t"][i] >= 2.95 else "motion_block_ended"
    else:
      event.update(status="observed_response", response_latency_s=float(d["t"][i+onset] - d["t"][i]),
                   measured_response_mps2=float(np.median(d["a"][i+onset:response_end])))
    events.append(event)
  return events


def chronological_split(route_times):
  known = sorted((r for r, wall in route_times.items() if wall is not None), key=lambda r: (route_times[r], r))
  a = max(1, int(len(known) * .6))
  b = max(a, int(len(known) * .8))
  return {r: ("train" if i < a else "validation" if i < b else "test") for i, r in enumerate(known)}


def latency_parameters(events, split, known_build):
  output = {}
  for direction in ("braking", "acceleration"):
    train = [e for e in events if e["direction"] == direction and e["status"] == "observed_response" and split.get(e["route"]) == "train"]
    routes = len({e["route"] for e in train})
    values = [e["response_latency_s"] for e in train]
    output[direction] = {"latency_s": float(np.median(values)) if values else None,
      "training_events": len(values), "training_routes": routes,
      "validity": "training_median_only" if len(values) >= 10 and routes >= 2 and known_build else "underidentified",
      "training_latency_s": summarize(values)}
  return output


def response_summary(events, split):
  output = {}
  for partition in ("train", "validation", "test", "unknown"):
    output[partition] = {}
    for direction in ("braking", "acceleration"):
      chosen = [e for e in events if e["direction"] == direction and split.get(e["route"], "unknown") == partition]
      used = [e for e in chosen if e["status"] == "observed_response"]
      output[partition][direction] = {"candidates": len(chosen), "statuses": dict(Counter(e["status"] for e in chosen)),
        "routes_with_response": len({e["route"] for e in used}), "observed_latency_s": summarize([e["response_latency_s"] for e in used])}
  return output


def speed_response_bins(events, split):
  bins = defaultdict(list)
  for event in events:
    if event["status"] == "observed_response" and split.get(event["route"]) == "train":
      key = event["direction"], band_of(event["speed_mps"])
      bins[key].append(event)
  return [{"direction": direction, "speed_band": band, "events": len(values), "routes": len({e["route"] for e in values}),
           "command_median_mps2": float(np.median([e["command_after_mps2"] for e in values])),
           "response_median_mps2": float(np.median([e["measured_response_mps2"] for e in values])),
           "use": "descriptive onset-response median; not a settled plant map"} for (direction, band), values in sorted(bins.items())]


def stop_regime_summary(stops):
  result = {}
  for regime in sorted({e["control_regime"] for e in stops}):
    selected = [e for e in stops if e["control_regime"] == regime]
    result[regime] = {"episodes": len(selected), "outcomes": dict(Counter(e["outcome"] for e in selected)),
      "comma_only_completed": sum(e["comma_only_eligible"] for e in selected),
      "predicted_stop_distance_m": summarize([e["predicted_stop_distance_m"] for e in selected]),
      "prediction_error_m_all_observed": summarize([e["prediction_error_m"] for e in selected if e["prediction_error_m"] is not None])}
  return result


def analyze(paths):
  grouped, sources = defaultdict(list), []
  for path, meta in paths:
    grouped[cohort_key(meta)].append((path, meta))
    sources.append(source_reference(path, meta))
  cohorts = {}
  for key, members in grouped.items():
    routes, walls = defaultdict(list), {}
    for path, meta in members:
      routes[meta["route"]].append((path, meta))
      wall = meta.get("init_data", {}).get("wallTimeNanos")
      walls[meta["route"]] = min(walls.get(meta["route"]) or wall, wall) if wall else walls.get(meta["route"])
    split = chronological_split(walls)
    stops, responses, coverage = [], [], Counter()
    for route, route_members in routes.items():
      blocks, route_coverage = stitch_route(route_members)
      coverage.update(route_coverage)
      for block in blocks:
        stops.extend(stop_events(block, route, members[0][1].get("op_long")))
        responses.extend(response_events(block, route, members[0][1].get("op_long")))
    identity = cohort_identity(members[0][1])
    params = latency_parameters(responses, split, bool(identity["opendbc_gitlink"]) and identity["dirty"] is False)
    for stop in stops:
      stop["partition"] = split.get(stop["route"], "unknown")
      latency = params["braking"]["latency_s"]
      stop["candidate_margin_sensitivity"] = [] if latency is None else [
        {"margin_m": margin, "request_mps2": stopping_request(stop["speed_mps"], stop["remaining_predicted_stop_distance_m"], latency, margin)[0],
         "unreachable": stopping_request(stop["speed_mps"], stop["remaining_predicted_stop_distance_m"], latency, margin)[1]} for margin in MARGINS]
    raw_can = Counter()
    for _, meta in members:
      raw_can.update({name: int(meta.get("can_counts", {}).get(name, 0)) for name in
                     ("tx_at_409_frames", "tx_over_409_frames", "eps_frames", "eps_fault_frames")})
    cohorts[key] = {"identity": identity, "segments": len(members), "routes": len(routes), "route_partitions": split,
      "route_wall_time_nanos": walls, "coverage": dict(coverage), "raw_can_counts": dict(raw_can),
      "response": response_summary(responses, split), "latency_parameters": params,
      "training_response_by_speed": speed_response_bins(responses, split),
      "stops_by_control_regime": stop_regime_summary(stops),
      "stop_summary": {"episodes": len(stops), "outcomes": dict(Counter(e["outcome"] for e in stops)),
        "control_regimes": dict(Counter(e["control_regime"] for e in stops)),
        "comma_only_completed": sum(e["comma_only_eligible"] for e in stops),
        "predicted_stop_distance_m": summarize([e["predicted_stop_distance_m"] for e in stops]),
        "actual_stop_distance_m_all_observed": summarize([e["actual_stop_distance_m"] for e in stops if e["actual_stop_distance_m"] is not None]),
        "prediction_error_m_comma_only": summarize([e["prediction_error_m"] for e in stops if e["comma_only_eligible"]])},
      "stops": stops, "response_events": responses}
  return {"segments": len(paths), "routes": len({m["route"] for _, m in paths}), "cohorts": cohorts, "sources": sources}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--limit", type=int, help="Bounded smoke; whole-route reporting requires omission")
  args = parser.parse_args()
  try:
    validate_output(args.dataset, args.out)
  except ValueError as exc:
    parser.error(str(exc))
  if args.limit is not None and args.limit < 1:
    parser.error("--limit must be positive")
  present = sorted(args.dataset.glob("*.npz"))
  if not present:
    parser.error("dataset contains no extracted NPZ records")
  paths = [(p, json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))) for p in present[:args.limit]]
  enrich_route_metadata(paths)
  resolve_gitlinks(paths, args.source_repo)
  report = analyze(paths)
  report.update(extraction_context(args.dataset))
  report.update({"dataset": str(args.dataset.resolve()), "npz_present": len(present), "smoke_limit": args.limit,
    "trajectory_timing_status": "exact_frame_capture_anchored_measured_gt", "version": 2,
    "contract": {"gt": "measured carState speed and LLK yaw integrated using actual monotonic dt; never model velocity",
      "stop_prediction": ("modelV2 nonuniform path arc speed<0.3m/s sustained>=1s; publication age<=50ms; "
                          + "exact EncodeIdx capture age<=200ms; complete capture prefix"),
      "stop_episode": ("first model stationary prediction while speed>=2m/s; retain inactive/unknown/CP-disabled regimes separately; "
                       + "reset after actual stop or2s no prediction"),
      "stop_observation": "speed<=0.3m/s sustained>=0.5s within15s; gaps>75ms or invalid motion/calibration end observations",
      "stop_error": ("capture-origin measured travel minus initial predicted stationary distance; "
                     + "prediction discrepancy, NOT stop-line ground truth or causal overshoot"),
      "interventions": "retain all episodes; comma-only completion excluded at first brake/gas/steering/disengage or unknown driver/control state",
      "signal_onsets": "first valid negative planner/output/aEgo; finite,valid,age<=200ms throughout sustained interval; unavailable rows counted",
      "response": ("command change>=0.25m/s² vs previous0.3s median; negative command<=-0.2 or positive>=0.2; "
                   + "measured aEgo change>=0.15 sustained0.15s within3s"),
      "response_admission": "clean0.5s prefix and effect through response; no v>=3 gate; rejects already-changing acceleration and driver overrides",
      "latency": "training-route median observed event response only; descriptive, not identified hardware delay; separate braking/acceleration",
      "law": ("decel=-v²/[2*(remaining_predicted_stop_distance-v*training_brake_latency-margin)] clipped[-3.5,0]; "
              + "subtract already-travelled capture-to-decision distance causally; exhausted distance/excess demand flaggedunreachable"),
      "margins_m": MARGINS, "margin_selection": "unoptimized sensitivity settings; no optimized margin or counterfactual plant claim",
      "cohorts": "separate model/hardware/control provenance; whole routes partitioned by recorded wallTime; unknown walls excluded from parameter training",
      "calibration": "same recorded calibration event held<=500ms in analysis; no source changes",
      "scope": "observation-only experiment; no device, panda, production control or model changes"}})
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  print(json.dumps({"out": str(args.out.resolve()), "segments": report["segments"], "routes": report["routes"],
                    "cohorts": len(report["cohorts"]), "stops": sum(c["stop_summary"]["episodes"] for c in report["cohorts"].values())}))


if __name__ == "__main__":
  main()
