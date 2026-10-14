"""Pure TRAIN five-state low-speed/hold medians with independent standstill.

Caller-authenticated queue witnesses are required, not inferred from grids or
returned CAN. This observational component establishes no contact or road tune.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import re

import numpy as np

import controller_multirate_joint as moving
import controller_multirate_stop_hold as lateral_hold

base = moving.base
FitError, Unsupported, identity_hash = moving.FitError, moving.Unsupported, moving.identity_hash
NATIVE_POLICY_SHA = lateral_hold.NATIVE_POLICY_SHA
CARD_SHA = "6e54bc5145634e451d6b48de9dd80691bf891ba81fab77f7d4de59eebe6c191b"
QUEUE_KIND = "source_bound_queued_lateral_apply_v1"
KIND = "joint_v_a_angle_yaw_unsigned_rate_stop_hold_categorical_50ms_v1"
APPROACH_KIND = "joint_v_a_angle_yaw_unsigned_rate_approach_stop_hold_categorical_50ms_v1"
STATE_KEYS = moving.STATE_KEYS
FEATURES = (*moving.FEATURES, *(f"effective_active_{i}" for i in range(5)), *(f"request_{i}" for i in range(5)))
LOW_SPEED_MAX_MPS = 3.  # Declared identification domain, NOT native activation.
QUEUE_CLOCKS = ("echo_t_ns", "apply_t_ns", "consume_t_ns", "consumed_cc_t_ns", "compute_t_ns", "feedback_cs_t_ns")


def _metadata(records, split, cp, sp):
  if (not isinstance(split, dict) or set(split) != {"train", "validation", "test"}
      or any(not isinstance(v, list) or any(not isinstance(r, str) or not r for r in v) or len(v) != len(set(v)) for v in split.values())):
    raise FitError("frozen disjoint string routes without duplicates required")
  contract, verified, anchors = lateral_hold._metadata(records, split, cp, sp)
  for record in records:  # No array access until EVERY source/metadata gate passes.
    moving._binding_metadata(record, split)
    m, q = record["metadata"], record.get("queued_apply_metadata", {})
    if not isinstance(q, dict):
      raise FitError("queued apply source/policy/lineage metadata unavailable or differs")
    body = {k: v for k, v in q.items() if k != "sha256"}
    if (q.get("kind") != QUEUE_KIND or q.get("sha256") != identity_hash(body)
        or any(q.get(k) != m[k] for k in ("route", "source_log_sha256", "static_identity_sha256", "policy_identity_sha256"))
        or q.get("card_source_sha256") != CARD_SHA or q.get("native_policy_source_sha256") != NATIVE_POLICY_SHA
        or any(not isinstance(q.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", q[k])
               for k in ("arrays_sha256", "motion_arrays_sha256", "view_arrays_sha256"))
        or q["motion_arrays_sha256"] != record["view_receipt"]["input_arrays_sha256"]
        or q["view_arrays_sha256"] != record["view_receipt"]["output_view_arrays_sha256"]):
      raise FitError("queued apply source/policy/lineage metadata unavailable or differs")
  return contract, verified, anchors


def _streams(record):
  streams = moving._streams(record)
  cs = streams["carState"]
  value, present = moving._extra(record["motion_arrays"], "carState", "standstill", len(cs["t"]))
  cs["v"].update(standstill=value, standstill_present=present)
  q = record.get("queued_apply_arrays")
  if not isinstance(q, dict):
    raise FitError("queued_apply_witness_missing")
  if identity_hash(q) != record["queued_apply_metadata"]["arrays_sha256"]:
    raise FitError("queued_apply_content_binding")
  if set(q) != {*QUEUE_CLOCKS, "valid", "effective_active"}:
    raise FitError("queued_apply_fields")
  count = len(streams["LKAS11Tx"]["t"])
  if (any(not isinstance(q[k], np.ndarray) or q[k].shape != (count,)
          or q[k].dtype != (np.dtype(np.uint64) if k in QUEUE_CLOCKS else np.dtype(bool)) for k in q)
      or not np.array_equal(q["echo_t_ns"], streams["LKAS11Tx"]["t"])):
    raise FitError("queued_apply_shape_clock_or_echo_binding")
  return streams, q


def _quality(stream, name, indices):
  # Replace only moving activation/domain checks. All original validity,
  # overrides, LLK uncertainty, command bounds and interpositions survive.
  reasons = [r for r in moving._quality(stream, name, indices)
             if r not in ("lateral_inactive", "request_false", "outside_moving_regime")]
  if name == "carState":
    v = stream["v"]
    if not v["standstill_present"][indices].all() or not np.isin(v["standstill"][indices], [0, 1]).all():
      reasons.append("hardware_standstill_invalid")
    if ((v["standstill"][indices] == 1) & (v["v"][indices] > .3)).any():
      reasons.append("hardware_standstill_above_native_low_speed")
  if name == "carControl" and not np.isin(stream["v"]["lat_active"][indices], [0, 1]).all():
    reasons.append("lateral_active_invalid")
  if name == "LKAS11Tx" and not np.isin(stream["v"]["request"][indices], [0, 1]).all():
    reasons.append("request_invalid")
  return reasons


def _mode(active, request, command):
  if np.any(active & ~request):
    return "active_request_false"
  if np.any(~active & (request | (command != 0))):
    return "inactive_command_or_request"
  return None


def _queued(streams, q, indices, anchor):
  ti = indices["LKAS11Tx"]
  if not q["valid"][ti]:
    raise FitError("queued_apply_invalid")
  echo, apply, consume, cc_time, compute, cs_time = (int(q[k][ti]) for k in QUEUE_CLOCKS)
  if not cs_time < compute < cc_time < consume < apply <= echo or echo-apply > 20_000_000:
    raise FitError("queued_apply_order")
  cs, cc = streams["carState"], streams["carControl"]
  ci, ui = bisect_left(cs["t"], compute)-1, bisect_left(cc["t"], consume)-1
  if (ci < 0 or ui < 0 or cs["t"][ci] != cs_time or cc["t"][ui] != cc_time
      or compute-cs_time > 15_000_000 or consume-cc_time > 20_000_000
      or compute in cs["t"] or consume in cc["t"]):
    raise FitError("queued_latest_causal_snapshot_missing")
  if anchor is None or min(cs_time, cc_time) < anchor:
    raise FitError("before_static_or_policy_prefix")
  reasons = []
  for name, first in (("carState", ci), ("carControl", ui)):
    reasons += _quality(streams[name], name, np.arange(first, indices[name]+1))
  if reasons:
    return bool(q["effective_active"][ti]), reasons
  c, u = cs["v"], cc["v"]
  # Exact bf07 policy under metadata-proven minSteerSpeed0/steerAtStandstillFalse;
  # CC.latActive retains upstream engagement/blinker/MADS decisions, not defaults.
  expected = bool(u["lat_active"][ui] and not c["fault_temporary"][ci] and not c["fault_permanent"][ci]
                  and not (c["v"][ci] <= .3 or c["standstill"][ci]))
  if bool(q["effective_active"][ti]) != expected:
    raise FitError("queued_effective_active_mismatch")
  return expected, []


def _windows(record, anchor, *, maximum_initial_mps=LOW_SPEED_MAX_MPS):
  m = record["metadata"]
  provenance = {k: m[k] for k in ("route", "source_log_sha256")}
  try:
    streams, q = _streams(record)
  except (FitError, KeyError, TypeError, ValueError) as error:
    return [], [{**provenance, "start_ns": None, "end_ns": None, "reasons": [str(error)]}], 0, 0
  llk, cs, tx = (streams[k] for k in ("liveLocationKalman", "carState", "LKAS11Tx"))
  accepted, excluded = [], []
  for i, (start, end) in enumerate(zip(llk["t"], llk["t"][1:], strict=False)):
    reasons = [] if abs(end-start-base.MACRO_NS) <= base.DT_TOLERANCE_NS else ["macro_dt"]
    reasons += ["static_policy_timing_unknown"] if anchor is None else []
    reasons += _quality(llk, "liveLocationKalman", np.array([i, i+1]))
    for name in moving.SERVICES:
      ids, gap = base._span(streams[name], start, end, 15_000_000 if name == "carState" else 20_000_000)
      if gap:
        reasons.append(name+"_gap_or_prefix")
      if anchor is not None and any(streams[name]["t"][j] < anchor for j in ids):
        reasons.append("before_static_or_policy_prefix")
      reasons += _quality(streams[name], name, ids)
    nodes = [start+j*base.NATIVE_NS for j in range(5)]
    joined, active = [], []
    if not reasons:
      points = sorted({start-20_000_000, *nodes, end, *(t for s in streams.values() for t in s["t"]
                       [bisect_left(s["t"], start-20_000_000):bisect_right(s["t"], end)])})
      try:
        for point in points:
          indices = moving._join(streams, point)
          for name, index in indices.items():
            reasons += _quality(streams[name], name, np.array([index]))
            if anchor is not None and streams[name]["t"][index] < anchor:
              reasons.append("before_static_or_policy_prefix")
          flag, more = _queued(streams, q, indices, anchor)
          reasons += more
          ti = indices["LKAS11Tx"]
          mode_reason = _mode(np.array([flag]), np.array([bool(tx["v"]["request"][ti])]), tx["v"]["torque"][[ti]])
          if mode_reason:
            reasons.append(mode_reason)
        if not reasons:
          joined = [moving._join(streams, node) for node in nodes]
          active = [_queued(streams, q, row, anchor)[0] for row in joined]
      except FitError as error:
        reasons.append(str(error))
    source = {**provenance, "start_ns": start, "end_ns": end}
    if reasons:
      excluded.append({**source, "reasons": sorted(set(reasons))})
      continue
    ci, ui, oi, ti = ([row[name] for row in joined] for name in moving.SERVICES)
    endpoint = moving._join(streams, end)["carState"]
    def state(index, location):
      return np.array([cs["v"][k][index] for k in ("v", "a", "steering_angle")]
                      +[llk["v"]["yaw"][location], cs["v"]["vendor_rate_unsigned"][index]])
    initial, final = state(ci[0], i), state(endpoint, i+1)
    if maximum_initial_mps is not None and initial[0] > maximum_initial_mps:
      excluded.append({**source, "reasons": ["outside_declared_low_speed_identification_domain"]})
      continue
    features = np.r_[initial, llk["v"]["roll"][i], streams["carControl"]["v"]["accel_request"][ui],
                     streams["carOutput"]["v"]["accel_reported"][oi], tx["v"]["torque"][ti], active, tx["v"]["request"][ti]]
    accepted.append({**source, "features": features, "delta": final-initial, "dt_s": (end-start)*1e-9,
      "hardware_standstill": bool(cs["v"]["standstill"][ci[0]]), "next_hardware_standstill": bool(cs["v"]["standstill"][endpoint]),
      "node_t_ns": nodes, "effective_active": active,
      **{k: [int(q[k][j]) for j in ti] for k in QUEUE_CLOCKS},
      "requested_t_ns": [streams["carControl"]["t"][j] for j in ui], "reported_t_ns": [streams["carOutput"]["t"][j] for j in oi]})
  return accepted, excluded, sum(s["repairs"] for s in streams.values()), sum(s["ignored"] for s in streams.values())


def _pattern(standstill, features):
  return str(int(standstill))+"|"+"".join(str(int(v)) for v in features[21:31])


def fit(records, split, cp, sp, *, quantile_bins=2, min_samples=100, min_routes=2):
  """Legacy <=3m/s identification domain; never automatic native activation."""
  return _fit(records, split, cp, sp, quantile_bins=quantile_bins, min_samples=min_samples,
              min_routes=min_routes, kind=KIND, maximum_initial_mps=LOW_SPEED_MAX_MPS)


def fit_approach(records, split, cp, sp, *, quantile_bins=2, min_samples=100, min_routes=2):
  """One categorical model across observed TRAIN speed cells; no regime stitching."""
  return _fit(records, split, cp, sp, quantile_bins=quantile_bins, min_samples=min_samples,
              min_routes=min_routes, kind=APPROACH_KIND, maximum_initial_mps=None)


def _fit(records, split, cp, sp, *, quantile_bins, min_samples, min_routes, kind, maximum_initial_mps):
  if (not records or type(quantile_bins) is not int or not 1 <= quantile_bins <= 8
      or type(min_samples) is not int or min_samples < 100 or type(min_routes) is not int or min_routes < 2):
    raise FitError("records,1..8quantile bins and >=100samples/two physical TRAIN routes required")
  contract, verified, anchors = _metadata(records, split, cp, sp)
  for record in records:
    moving._bind(record, split)
  windows, exclusions, repairs, ignored = [], [], 0, 0
  for record, anchor in zip(records, anchors, strict=True):
    rows, failures, order, discarded = _windows(record, anchor, maximum_initial_mps=maximum_initial_mps)
    windows += rows
    exclusions += failures
    repairs += order
    ignored += discarded
  model = {"kind": kind, **dict(zip(("cohort", "static_identity_sha256", "policy_identity_sha256", "opendbc_gitlink"), contract, strict=True)),
    "features": list(FEATURES), "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5, "macro_dt_tolerance_s": .002,
    **({"low_speed_max_initial_mps": LOW_SPEED_MAX_MPS} if kind == KIND else
       {"initial_speed_domain": "retained_train_global_and_cell_bounds"}),
    "native_policy_source_sha256": NATIVE_POLICY_SHA, "card_source_sha256": CARD_SHA,
    "source_eligible": verified and all(a is not None for a in anchors)
                       and all(r["motion_metadata"]["source_eligible"] and not r["motion_metadata"]["eligibility_reasons"] for r in records),
    "source_pin_verified": verified, "static_policy_timing_verified": all(a is not None for a in anchors),
    "min_samples": min_samples, "min_routes": min_routes, "split": {k: list(v) for k, v in split.items()},
    "record_hashes": [identity_hash(r) for r in records], "cells": {}, "patterns": [],
    "input_eligibility": [{"source_log_sha256": r["metadata"]["source_log_sha256"],
      "source_eligible": r["motion_metadata"]["source_eligible"], "reasons": list(r["motion_metadata"]["eligibility_reasons"])} for r in records],
    "accepted_windows": len(windows), "attempted_windows": len(windows)+len(exclusions), "exclusions": exclusions,
    "excluded_reasons": dict(Counter(r for e in exclusions for r in e["reasons"])), "clock_order_repairs": repairs,
    "ignored_non_echo_rows": ignored, "window_evidence": [{k: v for k, v in w.items() if k not in ("features", "delta")} for w in windows],
    "contact_hold_validated": False, "hundred_hz_feedback_validated": False, "velocity_acceleration_consistency_validated": False,
    "joint_median_coherence_validated": False, "selection_eligible": False, "selection": None, "candidate": None,
    "complete_controller_candidate": None, "production_enabled": False,
    "limitations": ["31-feature componentwise median and next categorical standstill are observational hypotheses, not contact physics",
      "Independent hardware standstill is never derived fromv; conflicting categories are unsupported, not majority-voted",
      "Explicit queued CC/CS/compute/consume/apply/echo witnesses are caller-authenticated, not proven acquisition clocks",
      "Requested acceleration, software report and returned src128/0x340 CAN are distinct unshifted command proxies, not delivery or identified lag",
      "Requested5 features are latest published request-asof nodes, distinct from consumedCC; held reported5 is independently published",
      "Queue witness binds consumed effective-active snapshots only; it does not equate published requests with applied force",
      "No intermediate measured v/a features, contact clamp, rack freeze, invented restart or old model fallback",
      "50ms endpoints/100Hz ZOH, Markov history sufficiency, joint coherence and v/a kinematics remain unvalidated",
      ("Initialv<=3m/s is a declared overlapping identification domain, never native activation or automatic model routing" if kind == KIND else
       "Approach speed domain is retained TRAIN global/cell bounds, never native activation or an unbounded physical model"),
      "Source/static/policy, stopping uncertainty and physical contact evidence remain separate selection gates"]}
  if windows:
    x = np.stack([w["features"] for w in windows])
    lo, hi = x.min(axis=0), x.max(axis=0)
    edges = [np.unique(np.quantile(x[:, j], np.arange(1, quantile_bins)/quantile_bins)) for j in range(31)]
    edges = [e[(e >= a) & (e < b)] if a < b else np.array([]) for e, a, b in zip(edges, lo, hi, strict=True)]
    groups = defaultdict(list)
    for row in windows:
      groups[_pattern(row["hardware_standstill"], row["features"])+"|"+base._cell_key(edges, row["features"])].append(row)
    for key, rows in groups.items():
      categories, routes = sorted({r["next_hardware_standstill"] for r in rows}), sorted({r["route"] for r in rows})
      reasons = ([] if len(rows) >= min_samples else ["sample_support"])+([] if len(routes) >= min_routes else ["route_support"])
      reasons += [] if len(categories) == 1 else ["ambiguous_next_hardware_standstill"]
      values = np.stack([r["features"] for r in rows])
      model["cells"][key] = {"samples": len(rows), "routes": routes, "supported": not reasons, "rejections": reasons,
        "feature_min": values.min(axis=0).tolist(), "feature_max": values.max(axis=0).tolist(),
        "joint_delta_median": np.median(np.stack([r["delta"] for r in rows]), axis=0).tolist(),
        "next_hardware_standstill_values": categories}
    model.update(edges=[e.tolist() for e in edges], feature_min=lo.tolist(), feature_max=hi.tolist(),
      actual_dt_s={"min": min(w["dt_s"] for w in windows), "max": max(w["dt_s"] for w in windows)},
      patterns=sorted({_pattern(w["hardware_standstill"], w["features"]) for w in windows}))
  else:
    model.update(edges=[], feature_min=[], feature_max=[], actual_dt_s={"min": None, "max": None})
  model["supported_cells"] = sum(c["supported"] for c in model["cells"].values())
  model["model_sha256"] = identity_hash(model)
  return model


def predict_macro(model, state, accel_request, accel_reported, can_command, effective_active, request, roll, hardware_standstill):
  """Only five completed candidate-own command nodes; no controller/report generator."""
  return _predict(model, state, accel_request, accel_reported, can_command, effective_active, request, roll,
                  hardware_standstill, kind=KIND, maximum_initial_mps=LOW_SPEED_MAX_MPS)


def predict_approach_macro(model, state, accel_request, accel_reported, can_command, effective_active, request, roll, hardware_standstill):
  """Observed categorical approach/hold cells only; no v-derived category or fallback."""
  return _predict(model, state, accel_request, accel_reported, can_command, effective_active, request, roll,
                  hardware_standstill, kind=APPROACH_KIND, maximum_initial_mps=None)


def _predict(model, state, accel_request, accel_reported, can_command, effective_active, request, roll,
             hardware_standstill, *, kind, maximum_initial_mps):
  if type(hardware_standstill) is not bool:
    return Unsupported("hardware_standstill_required")
  if accel_reported is None:
    return Unsupported("reported_channel_required")
  try:
    active, request = np.asarray(effective_active), np.asarray(request)
    if (set(state) != set(STATE_KEYS) or any(np.asarray(v).shape != (5,) for v in
        (accel_request, accel_reported, can_command, active, request)) or active.dtype.kind != "b" or request.dtype.kind != "b"):
      return Unsupported("invalid_macro_inputs")
    initial, command = np.array([state[k] for k in STATE_KEYS], float), np.asarray(can_command, float)
    features = np.r_[initial, roll, np.asarray(accel_request, float), np.asarray(accel_reported, float), command, active, request]
    if (features.shape != (31,) or not np.isfinite(features).all() or initial[0] < 0 or initial[4] < 0
        or (maximum_initial_mps is not None and initial[0] > maximum_initial_mps) or (hardware_standstill and initial[0] > .3)
        or (np.abs(command) > 409).any() or (command != np.rint(command)).any()):
      return Unsupported("invalid_macro_inputs")
    reason = _mode(active, request, command)
    if reason:
      return Unsupported(reason)
    domain_differs = (model.get("low_speed_max_initial_mps") != LOW_SPEED_MAX_MPS if kind == KIND else
                      model.get("initial_speed_domain") != "retained_train_global_and_cell_bounds")
    if (model.get("kind") != kind or model.get("features") != list(FEATURES)
        or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
        or model.get("native_policy_source_sha256") != NATIVE_POLICY_SHA or model.get("card_source_sha256") != CARD_SHA
        or domain_differs
        or type(model["min_samples"]) is not int or model["min_samples"] < 100 or type(model["min_routes"]) is not int or model["min_routes"] < 2):
      return Unsupported("model_contract")
    if not model["cells"]:
      return Unsupported("no_supported_population")
    pattern = _pattern(hardware_standstill, features)
    if pattern not in model["patterns"]:
      return Unsupported("unsupported_pattern")
    lower, upper = (np.asarray(model[k], float) for k in ("feature_min", "feature_max"))
    edges = [np.asarray(e, float) for e in model["edges"]]
    if (lower.shape != (31,) or upper.shape != (31,) or not np.isfinite([lower, upper]).all() or (lower > upper).any()
        or len(edges) != 31 or any(e.ndim != 1 or not np.isfinite(e).all() or (np.diff(e) <= 0).any()
          or (e < a).any() or (e >= b).any() for e, a, b in zip(edges, lower, upper, strict=True))):
      return Unsupported("model_contract")
    if (features < lower).any() or (features > upper).any():
      return Unsupported("outside_training_bounds")
    cell = model["cells"].get(pattern+"|"+base._cell_key(edges, features))
    if cell is None:
      return Unsupported("unsupported_cell")
    lo, hi = (np.asarray(cell[k], float) for k in ("feature_min", "feature_max"))
    delta, categories, routes = np.asarray(cell["joint_delta_median"], float), cell["next_hardware_standstill_values"], cell["routes"]
    if (lo.shape != (31,) or hi.shape != (31,) or delta.shape != (5,) or not np.isfinite([lo, hi]).all() or not np.isfinite(delta).all()
        or (lo > hi).any() or (lo < lower).any() or (hi > upper).any() or type(cell["supported"]) is not bool
        or type(cell["samples"]) is not int or cell["samples"] < 1 or not isinstance(routes, list) or len(routes) != len(set(routes))
        or any(not isinstance(r, str) or r not in model["split"]["train"] for r in routes)
        or not isinstance(categories, list) or not categories or any(type(v) is not bool for v in categories)):
      return Unsupported("model_contract")
    if len(categories) != 1:
      return Unsupported("ambiguous_next_hardware_standstill")
    if not cell["supported"]:
      return Unsupported("cell_support")
    if cell["samples"] < model["min_samples"] or len(routes) < model["min_routes"]:
      return Unsupported("model_contract")
    if (features < lo).any() or (features > hi).any():
      return Unsupported("outside_cell_bounds")
    prediction = initial+delta
    if not np.isfinite(prediction).all() or prediction[0] < 0 or prediction[4] < 0 or (categories[0] and prediction[0] > .3):
      return Unsupported("invalid_joint_prediction")
    return {"state": dict(zip(STATE_KEYS, prediction.tolist(), strict=True)), "hardware_standstill": categories[0]}
  except (TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError):
    return Unsupported("model_contract")
