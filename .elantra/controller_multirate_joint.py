"""Pure TRAIN five-state/report-conditioned nominal50ms observational hypothesis.

No native report generation, file/schema authentication, or contact physics.
Five commands from each independent publication stream precede the macro update.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

import numpy as np

import controller_motion_longitudinal as view
import controller_multirate_lateral as base

FitError, Unsupported, identity_hash = base.FitError, base.Unsupported, base.identity_hash
KIND = "joint_v_a_angle_yaw_unsigned_rate_report_conditioned_50ms_v1"
STATE_KEYS = ("v", "a", "angle", "yaw", "vendor_rate")
FEATURES = (*STATE_KEYS, "physical_llk_roll", *(f"accel_request_{i}" for i in range(5)),
            *(f"accel_reported_{i}" for i in range(5)), *(f"can_command_{i}" for i in range(5)))
SERVICES = ("carState", "carControl", "carOutput", "LKAS11Tx")


def _binding_metadata(record, split):
  """All original/enriched source gates precede any record's array access."""
  try:
    m, original, receipt = record["metadata"], record["motion_metadata"], record["view_receipt"]
    segment = original.get("segment", "") if isinstance(original, dict) else ""
    match = view.SEGMENT_RE.fullmatch(segment) if isinstance(segment, str) else None
    if (any(not isinstance(v, dict) for v in (original, receipt)) or receipt.get("kind") != view.KIND
        or any(original.get(k) != m.get(k) or receipt.get(k) != m.get(k) for k in ("cohort", "segment", "source_log_sha256"))
        or not match or match[1] != m["route"] or match[1] not in split["train"]
        or type(original.get("source_eligible")) is not bool or not isinstance(original.get("eligibility_reasons"), list)
        or any(not isinstance(r, str) for r in original["eligibility_reasons"])
        or receipt.get("route") != m["route"] or receipt.get("input_source_eligible") != original.get("source_eligible")
        or receipt.get("eligibility_reasons") != original.get("eligibility_reasons") or receipt.get("reason_bits") != view.REASONS
        or receipt.get("source_eligible") is not False or receipt.get("selection_eligible") is not False
        or receipt.get("production_enabled") is not False or original.get("opendbc_gitlink") != m.get("opendbc_gitlink")):
      raise FitError("original/view/enriched source metadata differs")
  except (KeyError, TypeError, ValueError, AttributeError) as error:
    raise FitError("Motion/view source metadata: "+str(error)) from error


def _bind(record, split):
  """Content claims only: a reviewed caller must authenticate files and schema."""
  try:
    m, original, receipt = record["metadata"], record["motion_metadata"], record["view_receipt"]
    a, derived = record["motion_arrays"], record["view_arrays"]
    route, binding = view._admit(a, original, split, receipt["input_proof"])
    if (route != m["route"] or receipt.get("input_arrays_sha256") != binding["arrays_sha256"]
        or not isinstance(derived, dict) or any(not isinstance(v, np.ndarray) or v.dtype.hasobject for v in derived.values())
        or receipt.get("output_view_arrays_sha256") != identity_hash(derived)):
      raise FitError("original/view array content binding differs")
    for name in ("carState", "carControl", "carOutput"):
      view._stream(a, name)  # Original full payload boundaries/clocks/markers.
      for key in view.PRESERVED:
        field = name+"_"+key
        if field not in derived or not np.array_equal(derived[field], a[field]) or derived[field].dtype != a[field].dtype:
          raise FitError("derived stream markers differ from original")
  except (KeyError, TypeError, ValueError, AttributeError) as error:
    raise FitError("Motion/view binding: "+str(error)) from error


def _extra(arrays, name, key, count):
  if name+"_"+key not in arrays or name+"_"+key+"_present" not in arrays:
    raise FitError("missing_original_fields")
  value, present = (np.asarray(arrays[name+"_"+key+s]) for s in ("", "_present"))
  if value.shape != (count,) or value.dtype.kind not in "fiu" or present.shape != (count,) or present.dtype.kind != "b":
    raise FitError("original_field_shape_or_dtype")
  return value, present


def _streams(record):
  a, derived = record["motion_arrays"], record["view_arrays"]
  streams = {name: base._stream(a, name) for name in base.FIELDS}
  for name, key in (("carState", "a"), ("carControl", "long_active")):
    value, present = _extra(a, name, key, len(streams[name]["t"]))
    streams[name]["v"].update({key: value, key+"_present": present})
  streams["carOutput"] = {"t": [int(t) for t in a["carOutput_t_ns"]], "v": {},
                          "duplicate": a["carOutput_clock_duplicate"], "repairs": 0, "ignored": 0}
  for name, column in (("carState", "can_valid"), ("carControl", "accel_request"), ("carOutput", "accel_reported")):
    count, values = len(streams[name]["t"]), streams[name]["v"]
    if name+"_"+column not in derived:
      raise FitError("missing_reported_channel" if name == "carOutput" else "missing_longitudinal_channel")
    for key in (column, "longitudinal_usable", "longitudinal_reason_bits", "row_usable", "valid", "clock_conflict"):
      value = np.asarray(derived[name+"_"+key])
      kinds = "b" if key in ("can_valid", "longitudinal_usable", "row_usable", "valid", "clock_conflict") else "fiu"
      if (value.shape != (count,) or value.dtype.kind not in kinds
          or (key == "longitudinal_reason_bits" and value.dtype != np.uint16)):
        raise FitError("view_field_shape_or_dtype")
      values[key] = value
    if not np.array_equal(values["longitudinal_usable"], values["longitudinal_reason_bits"] == 0):
      raise FitError("view_reason_usability_mismatch")
  return streams


def _quality(stream, name, indices):
  values, reasons = stream["v"], []
  if name in base.FIELDS:
    reasons += base._quality(stream, name, indices)
  else:
    if not len(indices) or not values["valid"][indices].all():
      reasons.append(name+"_invalid")
    if stream["duplicate"][indices].any():
      reasons.append(name+"_duplicate")
  if name in ("carState", "carControl", "carOutput"):
    if (not values["longitudinal_usable"][indices].all() or not values["row_usable"][indices].all()
        or values["clock_conflict"][indices].any()):
      reasons.append(name+"_longitudinal_invalid")
    column = {"carState": "can_valid", "carControl": "accel_request", "carOutput": "accel_reported"}[name]
    if not np.isfinite(values[column][indices]).all() or (name == "carState" and not values[column][indices].all()):
      reasons.append(name+"_longitudinal_invalid")
  if name in ("carState", "carControl"):
    key = "a" if name == "carState" else "long_active"
    if not values[key+"_present"][indices].all() or not np.isfinite(values[key][indices]).all():
      reasons.append(name+"_missing_or_nonfinite")
    if name == "carControl" and (values[key][indices] != 1).any():
      reasons.append("comma_long_inactive")
    if name == "carState" and (values["driver_torque"][indices] != 0).any():
      reasons.append("nonzero_driver_torque")
    if name == "carState" and (values["v"][indices] <= .3).any():
      reasons.append("outside_moving_regime")
  return reasons


def _join(streams, node):
  indices = {}
  for name in SERVICES:
    stream = streams[name]
    index = bisect_right(stream["t"], node)-1
    if index < 0 or node-stream["t"][index] > (15_000_000 if name == "carState" else 20_000_000):
      raise FitError("micro_asof_missing")
    indices[name] = index
  return indices


def _windows(record, anchor):
  m = record["metadata"]
  try:
    streams = _streams(record)
  except (FitError, KeyError) as error:
    return [], [{"source_log_sha256": m["source_log_sha256"], "route": m["route"], "start_ns": None, "end_ns": None,
                 "reasons": [str(error)]}], 0, 0
  llk, cs, cc, co, tx = (streams[n] for n in ("liveLocationKalman", *SERVICES))
  accepted, excluded = [], []
  for i, (start, end) in enumerate(zip(llk["t"], llk["t"][1:], strict=False)):
    reasons = [] if abs(end-start-base.MACRO_NS) <= base.DT_TOLERANCE_NS else ["macro_dt"]
    reasons += (["static_policy_timing_unknown"] if anchor is None else ["before_static_or_policy_prefix"] if start-20_000_000 < anchor else [])
    reasons += _quality(llk, "liveLocationKalman", np.array([i, i+1]))
    checked = {}
    for name in SERVICES:
      indices, gap = base._span(streams[name], start, end, 15_000_000 if name == "carState" else 20_000_000)
      checked[name] = set(indices.tolist())
      if anchor is not None and any(streams[name]["t"][j] < anchor for j in indices):
        reasons.append("before_static_or_policy_prefix")
      if gap:
        reasons.append(name+"_gap_or_prefix")
      reasons += _quality(streams[name], name, indices)
    nodes, joined = [start+j*base.NATIVE_NS for j in range(5)], []
    if not reasons:
      points = sorted({start-20_000_000, *nodes, end,
                       *(t for s in streams.values() for t in s["t"][bisect_left(s["t"], start-20_000_000):bisect_right(s["t"], end)])})
      consumed = {name: set() for name in SERVICES}
      try:
        for point in points:
          for name, index in _join(streams, point).items():
            consumed[name].add(index)
        for name in SERVICES:
          if anchor is not None and any(streams[name]["t"][j] < anchor for j in consumed[name]):
            reasons.append("before_static_or_policy_prefix")
          extra = np.array(sorted(consumed[name]-checked[name]), dtype=int)
          if len(extra):
            reasons += _quality(streams[name], name, extra)
        joined = [_join(streams, node) for node in nodes] if not reasons else []
      except FitError as error:
        reasons.append(str(error))
    provenance = {"source_log_sha256": m["source_log_sha256"], "route": m["route"], "start_ns": start, "end_ns": end}
    if reasons:
      excluded.append({**provenance, "reasons": sorted(set(reasons))})
      continue
    ci, ui, oi, ti = ([row[name] for row in joined] for name in SERVICES)
    endpoint = _join(streams, end)["carState"]
    def state(index, llk_index):
      return np.array([cs["v"]["v"][index], cs["v"]["a"][index], cs["v"]["steering_angle"][index],
                       llk["v"]["yaw"][llk_index], cs["v"]["vendor_rate_unsigned"][index]])
    initial, final = state(ci[0], i), state(endpoint, i+1)
    features = np.r_[initial, llk["v"]["roll"][i], cc["v"]["accel_request"][ui], co["v"]["accel_reported"][oi], tx["v"]["torque"][ti]]
    accepted.append({**provenance, "features": features, "delta": final-initial, "dt_s": (end-start)*1e-9,
                     "node_t_ns": nodes, "requested_t_ns": [cc["t"][j] for j in ui], "reported_t_ns": [co["t"][j] for j in oi],
                     "echo_t_ns": [tx["t"][j] for j in ti]})
  return accepted, excluded, sum(s["repairs"] for s in streams.values()), sum(s["ignored"] for s in streams.values())


def fit(records, split, *, quantile_bins=2, min_samples=100, min_routes=2):
  if type(min_samples) is not int or min_samples < 100 or type(min_routes) is not int or min_routes < 2:
    raise FitError("support requires at least100 samples and two physical TRAIN routes")
  if not records or type(quantile_bins) is not int or not 1 <= quantile_bins <= 8:
    raise FitError("records and finite1..8quantile bins required")
  if (not isinstance(split, dict) or set(split) != {"train", "validation", "test"}
      or any(not isinstance(v, list) or any(not isinstance(r, str) for r in v) for v in split.values())
      or any(len(v) != len(set(v)) for v in split.values())
      or any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))):
    raise FitError("frozen disjoint TRAIN/validation/test route lists require strings without duplicates")
  contract, verified, anchors = base._metadata(records, split)
  for record in records:
    _binding_metadata(record, split)
  for record in records:  # Metadata/physical-route gates passed for ALL records first.
    _bind(record, split)
  windows, excluded, repairs, ignored = [], [], 0, 0
  for record, anchor in zip(records, anchors, strict=True):
    rows, failures, count, discarded = _windows(record, anchor)
    windows += rows
    excluded += failures
    repairs += count
    ignored += discarded
  model = {"kind": KIND, **dict(zip(("cohort", "static_identity_sha256", "policy_identity_sha256", "opendbc_gitlink"), contract, strict=True)),
           "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5, "macro_dt_tolerance_s": .002,
           "features": list(FEATURES), "cells": {},
           "source_eligible": verified and all(a is not None for a in anchors)
                              and all(r["motion_metadata"]["source_eligible"] and not r["motion_metadata"]["eligibility_reasons"] for r in records),
           "source_pin_verified": verified, "static_policy_timing_verified": all(a is not None for a in anchors),
           "hundred_hz_feedback_validated": False, "report_conditioned_observational_only": True,
           "velocity_acceleration_consistency_validated": False, "joint_median_coherence_validated": False,
           "selection_eligible": False, "production_enabled": False, "candidate": None,
           "min_samples": min_samples, "min_routes": min_routes, "split": {k: list(v) for k, v in split.items()},
           "record_hashes": [identity_hash(r) for r in records],
           "input_eligibility": [{"source_log_sha256": r["metadata"]["source_log_sha256"],
                                  "source_eligible": r["motion_metadata"]["source_eligible"],
                                  "reasons": list(r["motion_metadata"]["eligibility_reasons"])} for r in records],
           "exclusions": excluded, "excluded_reasons": dict(Counter(reason for e in excluded for reason in e["reasons"])),
           "accepted_windows": len(windows), "attempted_windows": len(windows)+len(excluded), "clock_order_repairs": repairs,
           "ignored_non_echo_rows": ignored, "window_evidence": [{k: v for k, v in w.items() if k not in ("features", "delta")} for w in windows],
           "limitations": ["Independent requested/reported acceleration and returnedCAN publication clocks are not acquisition clocks",
                           "Provided reportedchannel is required; this model neither generates nor authenticates a candidate native report",
                           "Original files/schema and enriched source/static/policy authenticity remain the reviewed caller's responsibility",
                           "Five initial-state deltas occur only at nominal50ms macro end; native100Hz ZOH feedback remains unvalidated",
                           "No intermediate measuredv/a features or future measured candidate feedback; no clipping/contact/reverse law",
                           "Velocity/acceleration kinematic consistency and componentwise-median joint-state coherence remain unvalidated",
                           "Commands are unshifted observations, not delivered force/torque or identified actuator lag",
                           "Actual48..52ms deltas are not rescaled; physicalLLKroll is distinct fromLP.roll; moving/zero-driver only"]}
  if windows:
    x = np.stack([w["features"] for w in windows])
    lower, upper = x.min(axis=0), x.max(axis=0)
    edges = [np.unique(np.quantile(x[:, j], np.arange(1, quantile_bins)/quantile_bins)) for j in range(21)]
    edges = [e[(e >= lo) & (e < hi)] if lo < hi else np.array([]) for e, lo, hi in zip(edges, lower, upper, strict=True)]
    groups = defaultdict(list)
    for row in windows:
      groups[base._cell_key(edges, row["features"])].append(row)
    for key, rows in groups.items():
      features, routes = np.stack([r["features"] for r in rows]), sorted({r["route"] for r in rows})
      reasons = ([] if len(rows) >= min_samples else ["sample_support"])+([] if len(routes) >= min_routes else ["route_support"])
      model["cells"][key] = {"samples": len(rows), "routes": routes, "supported": not reasons, "rejections": reasons,
        "feature_min": features.min(axis=0).tolist(), "feature_max": features.max(axis=0).tolist(),
        "joint_delta_median": np.median(np.stack([r["delta"] for r in rows]), axis=0).tolist()}
    model.update(edges=[e.tolist() for e in edges], feature_min=lower.tolist(), feature_max=upper.tolist(),
                 actual_dt_s={"min": min(w["dt_s"] for w in windows), "max": max(w["dt_s"] for w in windows)})
  else:
    model.update(edges=[], feature_min=[], feature_max=[], actual_dt_s={"min": None, "max": None})
  model["supported_cells"] = sum(c["supported"] for c in model["cells"].values())
  model["model_sha256"] = identity_hash(model)
  return model


def predict_macro(model, state, accel_request, accel_reported, can_command, roll):
  """Explicit candidate-own reported channel; completion only, no generation/auth."""
  if accel_reported is None:
    return Unsupported("reported_channel_required")
  try:
    if set(state) != set(STATE_KEYS) or any(np.asarray(v).shape != (5,) for v in (accel_request, accel_reported, can_command)):
      return Unsupported("invalid_macro_inputs")
    initial = np.array([state[k] for k in STATE_KEYS], float)
    command = np.asarray(can_command, float)
    features = np.r_[initial, roll, np.asarray(accel_request, float), np.asarray(accel_reported, float), command]
    if (features.shape != (21,) or not np.isfinite(features).all() or initial[0] < 0 or initial[4] < 0
        or (np.abs(command) > 409).any() or (command != np.rint(command)).any()):
      return Unsupported("invalid_macro_inputs")
  except (TypeError, ValueError, KeyError):
    return Unsupported("invalid_macro_inputs")
  # Content authentication belongs to the caller once per trial. These small
  # shape/support checks prevent broadcasting or malformed per-step state.
  try:
    if (model.get("kind") != KIND or model.get("features") != list(FEATURES)
        or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
        or not isinstance(model["cells"], dict) or type(model["min_samples"]) is not int or model["min_samples"] < 100
        or type(model["min_routes"]) is not int or model["min_routes"] < 2):
      return Unsupported("model_contract")
    if not model["cells"]:
      return Unsupported("no_supported_population")
    lower, upper = (np.asarray(model[k], float) for k in ("feature_min", "feature_max"))
    edges = [np.asarray(edge, float) for edge in model["edges"]]
    if (lower.shape != (21,) or upper.shape != (21,) or not np.isfinite([lower, upper]).all() or (lower > upper).any()
        or len(edges) != 21 or any(e.ndim != 1 or not np.isfinite(e).all() or (np.diff(e) <= 0).any()
                                  or (e < lo).any() or (e >= hi).any() for e, lo, hi in zip(edges, lower, upper, strict=True))):
      return Unsupported("model_contract")
    if (features < lower).any() or (features > upper).any():
      return Unsupported("outside_training_bounds")
    cell = model["cells"].get(base._cell_key(edges, features))
    if cell is None:
      return Unsupported("unsupported_cell")
    cmin, cmax = (np.asarray(cell[k], float) for k in ("feature_min", "feature_max"))
    delta = np.asarray(cell["joint_delta_median"], float)
    routes, samples = cell["routes"], cell["samples"]
    if (cmin.shape != (21,) or cmax.shape != (21,) or delta.shape != (5,) or not np.isfinite(delta).all()
        or not np.isfinite([cmin, cmax]).all() or (cmin > cmax).any() or (cmin < lower).any() or (cmax > upper).any()
        or type(cell["supported"]) is not bool or type(samples) is not int or samples < 1
        or not isinstance(routes, list) or any(not isinstance(r, str) or r not in model["split"]["train"] for r in routes)
        or len(routes) != len(set(routes)) or (cell["supported"] and (samples < model["min_samples"] or len(routes) < model["min_routes"]))):
      return Unsupported("model_contract")
    if not cell["supported"]:
      return Unsupported("cell_support")
    if (features < cmin).any() or (features > cmax).any():
      return Unsupported("outside_cell_bounds")
    prediction = initial+delta
  except (TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError):
    return Unsupported("model_contract")
  if not np.isfinite(prediction).all() or prediction[0] < 0 or prediction[4] < 0:
    return Unsupported("invalid_joint_prediction")
  if prediction[0] <= .3:
    return Unsupported("outside_moving_regime")
  return dict(zip(STATE_KEYS, prediction.tolist(), strict=True))
