"""Separate TRAIN-only 50ms transition/hold hypothesis; no contact or100Hz proof.

The five command/speed/active/request nodes are consumed only at the macro END.
Inactive angle/yaw/rate changes are empirical medians, never forced to zero/hold.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

import numpy as np

import controller_multirate_lateral as base
from controller_native_lateral_runtime import ImmutableParams, TYPES

FitError, Unsupported, identity_hash = base.FitError, base.Unsupported, base.identity_hash
KIND = "joint_stop_hold_angle_yaw_unsigned_rate_50ms_zoh_v1"
NATIVE_POLICY_SHA = "bf07d35158cfa8a7f895fdc75c231882bf88b457e8fa411a3387d2afbe31a95f"
STATE_KEYS = base.STATE_KEYS
FEATURES = (*base.FEATURES, *(f"effective_active_{i}" for i in range(5)), *(f"request_{i}" for i in range(5)))
FIELDS = {**base.FIELDS, "carState": (*base.FIELDS["carState"], "standstill"),
          "carControl": ("lat_active", "long_active", "torque_request"), "carOutput": ("torque_reported", "torque_counts")}


def _metadata(records, split, cp, sp, *, partition="train"):
  contract, verified, anchors = base._metadata(records, split, partition=partition)
  if (not isinstance(cp, dict) or not isinstance(sp, dict) or cp.get("carFingerprint") != "HYUNDAI_ELANTRA_2024"
      or cp.get("brand") != "hyundai" or cp.get("steerControlType") != "torque" or cp.get("minSteerSpeed") != 0.
      or cp.get("steerAtStandstill") is not False or cp.get("openpilotLongitudinalControl") is not True
      or contract[1] != identity_hash({"cp": cp, "sp": sp})):
    raise FitError("full static identity and native Elantra min0/no-standstill torque policy required")
  for record in records:  # ALL metadata checks precede any raw array access.
    m, raw = record["metadata"], record["metadata"].get("policy_values")
    if (not isinstance(raw, dict) or set(raw) != set(TYPES) or identity_hash(raw) != m["policy_identity_sha256"]
        or m.get("native_policy_source_sha256") != NATIVE_POLICY_SHA):
      raise FitError("full typed policy identity/native policy source required")
    provider = ImmutableParams(raw)
    try:
      values = {key: provider.get(key) for key in TYPES}
    except ValueError as error:
      raise FitError("invalid recorded typed policy: "+str(error)) from error
    if (not values["EnforceTorqueControl"] or values["TorqueControlTune"] != 0.
        or not values["LateralJerkTorqueController"] or values["NeuralNetworkLateralControl"]):
      raise FitError("recorded native v0/jerk/NN-disabled policy required")
  return contract, verified, anchors


def _stream(arrays, name):
  required = ["t_ns", "valid", *(k+s for k in FIELDS[name] for s in ("", "_present"))]
  if name == "LKAS11Tx":
    required += ["origin", "src", "address", "decoded_ok"]
  if any(name+"_"+key not in arrays for key in required):
    raise FitError("missing_fields")
  values = {key: np.asarray(arrays[name+"_"+key]) for key in required}
  t = values.pop("t_ns")
  if t.ndim != 1 or t.dtype != np.uint64 or not len(t):
    raise FitError("raw_clock_dtype_or_empty")
  if any(v.shape != t.shape for v in values.values()):
    raise FitError("raw_shape")
  if (values["valid"].dtype.kind != "b" or any(values[k+"_present"].dtype.kind != "b" for k in FIELDS[name])
      or any(values[k].dtype.kind not in "fiu" for k in FIELDS[name])):
    raise FitError("raw_validity_or_numeric_dtype")
  ignored = 0
  if name == "LKAS11Tx":
    if values["decoded_ok"].dtype.kind != "b":
      raise FitError("raw_validity_or_numeric_dtype")
    keep = (values["origin"] == 0) & (values["src"] == 128) & (values["address"] == 0x340)
    ignored = int((~keep).sum())
    t, values = t[keep], {k: v[keep] for k, v in values.items()}
    if not len(t):
      raise FitError("no_returned_bus0_echo")
  repairs = sum(int(a) > int(b) for a, b in zip(t, t[1:], strict=False))
  order = np.argsort(t, kind="stable")
  t, values = t[order], {k: v[order] for k, v in values.items()}
  duplicate = np.r_[False, t[1:] == t[:-1]] | np.r_[t[1:] == t[:-1], False]
  return {"t": [int(v) for v in t], "v": values, "duplicate": duplicate, "repairs": repairs, "ignored": ignored}


def _quality(stream, name, indices):
  v, reasons = stream["v"], []
  if not len(indices) or not v["valid"][indices].all():
    reasons.append(name+"_invalid")
  if stream["duplicate"][indices].any():
    reasons.append(name+"_duplicate")
  if any(not v[k+"_present"][indices].all() or not np.isfinite(v[k][indices]).all() for k in FIELDS[name]):
    reasons.append(name+"_missing_or_nonfinite")
  if name in ("carState", "liveLocationKalman"):
    reasons += base._quality(stream, name, indices)
  if name == "carState":
    if (v["driver_torque"][indices] != 0).any():
      reasons.append("nonzero_driver_torque")
    if not np.isin(v["standstill"][indices], [0, 1]).all():
      reasons.append("standstill_invalid")
    if ((v["standstill"][indices] != 0) & (v["v"][indices] > .3)).any():
      reasons.append("hardware_standstill_unmodeled")
  elif name == "carControl":
    if not np.isin(v["lat_active"][indices], [0, 1]).all():
      reasons.append("lateral_active_invalid")
    if (v["long_active"][indices] != 1).any():
      reasons.append("comma_long_inactive")
  elif name == "carOutput":
    count = v["torque_counts"][indices]
    if (np.abs(count) > 409).any() or (count != np.rint(count)).any() or (np.abs(v["torque_reported"][indices]) > 1).any():
      reasons.append("output_envelope")
  elif name == "LKAS11Tx":
    if not v["decoded_ok"][indices].all():
      reasons.append("echo_decode_invalid")
    if not np.isin(v["request"][indices], [0, 1]).all() or (v["torque_fault"][indices] != 0).any():
      reasons.append("request_fault_or_invalid")
    count = v["torque"][indices]
    if (np.abs(count) > 409).any() or (count != np.rint(count)).any():
      reasons.append("torque_envelope")
  return reasons


def _mode_reasons(speed, command, active, request, *, scope=True):
  reasons = []
  if np.any(active & (speed <= .3)):
    reasons.append("effective_active_speed_mismatch")
  if np.any(active & ~request):
    reasons.append("active_request_false")
  if np.any(~active & (request | (command != 0))):
    reasons.append("inactive_command_or_request")
  if np.any(~active & (speed > .3)):
    reasons.append("inactive_above_low_speed")
  if scope and active.all() and request.all():
    reasons.append("moving_macro_uses_moving_model")
  return reasons


def _join(streams, node):
  indices = {}
  for name in ("carState", "carControl", "LKAS11Tx", "carOutput"):
    stream = streams[name]
    index = bisect_right(stream["t"], node)-1
    age = 15_000_000 if name == "carState" else 20_000_000
    if index < 0 or node-stream["t"][index] > age:
      raise FitError("policy_asof_missing")
    indices[name] = index
  cs, cc, tx = (streams[k]["v"] for k in ("carState", "carControl", "LKAS11Tx"))
  ci, ui, ti = (indices[k] for k in ("carState", "carControl", "LKAS11Tx"))
  speed = cs["v"][ci]
  active = bool(cc["lat_active"][ui] and not cs["fault_temporary"][ci] and not cs["fault_permanent"][ci]
                and not (speed <= .3 or cs["standstill"][ci]))
  return indices, speed, tx["torque"][ti], active, bool(tx["request"][ti])


def _windows(record, anchor):
  m = record["metadata"]
  try:
    streams = {name: _stream(record["arrays"], name) for name in FIELDS}
  except FitError as error:
    return [], [{"source_log_sha256": m["source_log_sha256"], "route": m["route"], "start_ns": None, "end_ns": None,
                 "reasons": [str(error)]}], 0, 0
  llk, cs, co, cc, tx = (streams[k] for k in ("liveLocationKalman", "carState", "carOutput", "carControl", "LKAS11Tx"))
  accepted, excluded = [], []
  for i, (start, end) in enumerate(zip(llk["t"], llk["t"][1:], strict=False)):
    reasons = [] if abs(end-start-base.MACRO_NS) <= base.DT_TOLERANCE_NS else ["macro_dt"]
    reasons += (["static_policy_timing_unknown"] if anchor is None else
                ["before_static_or_policy_prefix"] if start-20_000_000 < anchor else [])
    reasons += _quality(llk, "liveLocationKalman", np.array([i, i+1]))
    for name in FIELDS.keys()-{"liveLocationKalman"}:
      indices, gap = base._span(streams[name], start, end, 15_000_000 if name == "carState" else 20_000_000)
      if gap:
        reasons.append(name+"_gap_or_prefix")
      reasons += _quality(streams[name], name, indices)
    provenance = {"source_log_sha256": m["source_log_sha256"], "route": m["route"], "start_ns": start, "end_ns": end}
    nodes = [start+j*base.NATIVE_NS for j in range(5)]
    joined = []
    if not reasons:
      # Every interposed publication and the20ms prefix are checked, not just five nodes.
      points = sorted({start-20_000_000, *nodes, end,
                       *(t for s in streams.values() for t in s["t"][bisect_left(s["t"], start-20_000_000):bisect_right(s["t"], end)])})
      try:
        for point in points:
          indices, v, command, active, request = _join(streams, point)
          consumed_reasons = [reason for name, index in indices.items()
                              for reason in _quality(streams[name], name, np.array([index]))]
          reasons += consumed_reasons
          if not consumed_reasons:
            reasons += _mode_reasons(np.array([v]), np.array([command]), np.array([active]), np.array([request]), scope=False)
        if not reasons:
          joined = [_join(streams, node) for node in nodes]
      except FitError as error:
        reasons.append(str(error))
    if joined:
      speed, command = (np.array([row[j] for row in joined]) for j in (1, 2))
      active, request = (np.array([row[j] for row in joined], bool) for j in (3, 4))
      reasons += _mode_reasons(speed, command, active, request)
    if reasons:
      excluded.append({**provenance, "reasons": sorted(set(reasons))})
      continue
    ci, oi, ui, ti = ([row[0][name] for row in joined] for name in ("carState", "carOutput", "carControl", "LKAS11Tx"))
    endpoint = bisect_right(cs["t"], end)-1
    initial = np.array([cs["v"]["steering_angle"][ci[0]], llk["v"]["yaw"][i], cs["v"]["vendor_rate_unsigned"][ci[0]]])
    final = np.array([cs["v"]["steering_angle"][endpoint], llk["v"]["yaw"][i+1], cs["v"]["vendor_rate_unsigned"][endpoint]])
    features = np.r_[initial, llk["v"]["roll"][i], speed, command, active, request]
    accepted.append({**provenance, "features": features, "delta": final-initial, "dt_s": (end-start)*1e-9,
                     "evidence": {"standstill": cs["v"]["standstill"][ci].tolist(), "long_active": cc["v"]["long_active"][ui].tolist(),
                                  "output_t_ns": [co["t"][j] for j in oi], "echo_t_ns": [tx["t"][j] for j in ti],
                                  "car_state_t_ns": [cs["t"][j] for j in ci], "car_control_t_ns": [cc["t"][j] for j in ui],
                                  "echo_counts": command.tolist(),
                                  "output_counts": co["v"]["torque_counts"][oi].tolist(),
                                  "output_reported": co["v"]["torque_reported"][oi].tolist()}})
  return accepted, excluded, sum(s["repairs"] for s in streams.values()), sum(s["ignored"] for s in streams.values())


def _pattern(features):
  return "".join(str(int(v)) for v in features[14:24])


def fit(records, split, cp, sp, *, quantile_bins=2, min_samples=100, min_routes=2):
  if type(min_samples) is not int or min_samples < 100 or type(min_routes) is not int or min_routes < 2:
    raise FitError("support requires at least100 samples and two physical routes")
  if not records or type(quantile_bins) is not int or not 1 <= quantile_bins <= 8:
    raise FitError("nonempty records and finite1..8quantile bins required")
  contract, verified, anchors = _metadata(records, split, cp, sp)
  windows, excluded, repairs, ignored = [], [], 0, 0
  for record, anchor in zip(records, anchors, strict=True):
    rows, failures, count, discarded = _windows(record, anchor)
    windows += rows
    excluded += failures
    repairs += count
    ignored += discarded
  model = {"kind": KIND, **dict(zip(("cohort", "static_identity_sha256", "policy_identity_sha256", "opendbc_gitlink"), contract, strict=True)),
           "native_policy_source_sha256": NATIVE_POLICY_SHA, "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5,
           "macro_dt_tolerance_s": .002, "features": list(FEATURES), "cells": {}, "patterns": [],
           "source_eligible": verified and all(a is not None for a in anchors), "source_pin_verified": verified,
           "static_policy_timing_verified": all(a is not None for a in anchors), "hundred_hz_feedback_validated": False,
           "contact_hold_validated": False, "nonreversing_contact_validated": False, "selection_eligible": False,
           "production_enabled": False, "candidate": None, "min_samples": min_samples, "min_routes": min_routes,
           "split": {k: list(v) for k, v in split.items()}, "record_hashes": [identity_hash(r) for r in records],
           "exclusions": excluded, "excluded_reasons": dict(Counter(reason for e in excluded for reason in e["reasons"])),
           "accepted_windows": len(windows), "attempted_windows": len(windows)+len(excluded), "clock_order_repairs": repairs,
           "ignored_non_echo_rows": ignored, "window_evidence": [{k: v for k, v in w.items() if k not in ("features", "delta")} for w in windows],
           "limitations": ["50ms joint empirical delta; native100Hz state is unvalidated ZOH between macro endpoints",
                           "Unshifted returned CAN src128/0x340 is a command proxy, not delivery or identified actuator lag",
                           "No longitudinal contact, nonreversing or standstill-hold model is established by this lateral fit",
                           "Publication clocks are not proven acquisition clocks; CO/echo remain independently clocked observations",
                           "48..52ms actual deltas are retained without rescaling; physical LLKroll is not LP.roll",
                           "Source pin and sample counts cannot establish contact/hold/ZOH validation"]}
  if windows:
    x = np.stack([w["features"] for w in windows])
    lower, upper = x.min(axis=0), x.max(axis=0)
    edges = [np.unique(np.quantile(x[:, j], np.arange(1, quantile_bins)/quantile_bins)) for j in range(24)]
    edges = [e[(e >= lo) & (e < hi)] if lo < hi else np.array([]) for e, lo, hi in zip(edges, lower, upper, strict=True)]
    grouped = defaultdict(list)
    for row in windows:
      grouped[_pattern(row["features"])+"|"+base._cell_key(edges, row["features"])].append(row)
    for key, rows in grouped.items():
      values, routes = np.stack([r["features"] for r in rows]), sorted({r["route"] for r in rows})
      reasons = ([] if len(rows) >= min_samples else ["sample_support"])+([] if len(routes) >= min_routes else ["route_support"])
      model["cells"][key] = {"samples": len(rows), "routes": routes, "supported": not reasons, "rejections": reasons,
        "feature_min": values.min(axis=0).tolist(), "feature_max": values.max(axis=0).tolist(),
        "joint_delta_median": np.median(np.stack([r["delta"] for r in rows]), axis=0).tolist()}
    model.update(edges=[e.tolist() for e in edges], feature_min=lower.tolist(), feature_max=upper.tolist(),
                 patterns=sorted({_pattern(w["features"]) for w in windows}),
                 actual_dt_s={"min": min(w["dt_s"] for w in windows), "max": max(w["dt_s"] for w in windows)})
  else:
    model.update(edges=[], feature_min=[], feature_max=[], actual_dt_s={"min": None, "max": None})
  model["supported_cells"] = sum(c["supported"] for c in model["cells"].values())
  model["model_sha256"] = identity_hash(model)
  return model


def predict_macro(model, state, speed, command, active, request, roll):
  """Only completed five-node inputs; no measured future feedback or contact rule."""
  try:
    if set(state) != set(STATE_KEYS) or any(np.asarray(a).shape != (5,) for a in (speed, command, active, request)):
      return Unsupported("invalid_macro_inputs")
    active, request = np.asarray(active), np.asarray(request)
    if active.dtype.kind != "b" or request.dtype.kind != "b":
      return Unsupported("boolean_pattern_required")
    initial, speed, command = np.array([state[k] for k in STATE_KEYS], float), np.asarray(speed, float), np.asarray(command, float)
    features = np.r_[initial, roll, speed, command, active, request]
    if (features.shape != (24,) or not np.isfinite(features).all() or initial[2] < 0 or (speed < 0).any()
        or (np.abs(command) > 409).any() or (command != np.rint(command)).any()):
      return Unsupported("invalid_macro_inputs")
  except (TypeError, ValueError, KeyError):
    return Unsupported("invalid_macro_inputs")
  reasons = _mode_reasons(speed, command, active, request)
  if reasons:
    return Unsupported(reasons[0])
  if (model.get("kind") != KIND or model.get("macro_dt_s") != .05 or model.get("native_dt_s") != .01
      or model.get("micro_nodes") != 5 or model.get("native_policy_source_sha256") != NATIVE_POLICY_SHA):
    return Unsupported("model_contract")
  if not model["cells"]:
    return Unsupported("no_supported_population")
  pattern = _pattern(features)
  if pattern not in model["patterns"]:
    return Unsupported("unsupported_pattern")
  if (features < model["feature_min"]).any() or (features > model["feature_max"]).any():
    return Unsupported("outside_training_bounds")
  cell = model["cells"].get(pattern+"|"+base._cell_key(model["edges"], features))
  if cell is None:
    return Unsupported("unsupported_cell")
  if not cell["supported"]:
    return Unsupported("cell_support")
  if (features < cell["feature_min"]).any() or (features > cell["feature_max"]).any():
    return Unsupported("outside_cell_bounds")
  prediction = initial+cell["joint_delta_median"]
  if not np.isfinite(prediction).all() or prediction[2] < 0:
    return Unsupported("invalid_joint_prediction")
  return dict(zip(STATE_KEYS, prediction.tolist(), strict=True))
