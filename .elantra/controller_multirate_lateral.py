"""TRAIN-only joint nominal50ms hypothesis, held constant across five native10ms ticks.

This is a binned-median observational model, not validated100Hz feedback physics.
predict_macro is called only AFTER its five actual past command/speed nodes exist.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

KIND = "joint_angle_yaw_unsigned_rate_50ms_zoh_v1"
MACRO_NS, NATIVE_NS, DT_TOLERANCE_NS = 50_000_000, 10_000_000, 2_000_000
STATE_KEYS = ("angle", "yaw", "vendor_rate")
FEATURES = (*STATE_KEYS, "physical_llk_roll", *(f"speed_{i}" for i in range(5)), *(f"command_{i}" for i in range(5)))
FIELDS = {
  "carState": ("v", "steering_angle", "vendor_rate_unsigned", "driver_torque", "steering_pressed", "brake_pressed",
               "gas_pressed", "fault_temporary", "fault_permanent"),
  "liveLocationKalman": ("yaw", "roll", "yaw_std", "roll_std", "yaw_inner_valid", "roll_inner_valid", "status",
                         "inputs_ok", "sensors_ok", "device_stable"),
  "LKAS11Tx": ("torque", "request", "torque_fault"), "carControl": ("lat_active",),
}


class FitError(ValueError):
  pass


@dataclass(frozen=True)
class Unsupported:
  reason: str


def identity_hash(value):
  """Shared static/policy identity: compact sorted JSON, Data hex, typed array hashes."""
  def material(item):
    if isinstance(item, (bytes, bytearray)):
      return {"capnp_data_hex": bytes(item).hex()}
    if isinstance(item, np.ndarray):
      if item.dtype.kind == "O":
        raise FitError("object arrays are forbidden")
      return {"dtype": item.dtype.str, "shape": list(item.shape), "sha256": hashlib.sha256(item.tobytes()).hexdigest()}
    if isinstance(item, dict):
      return {key: material(v) for key, v in sorted(item.items())}
    if isinstance(item, (list, tuple)):
      return [material(v) for v in item]
    return item.item() if isinstance(item, np.generic) else item
  return hashlib.sha256(json.dumps(material(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


_hash = identity_hash


def _metadata(records, split, *, partition="train"):
  if partition not in ("train", "validation"):
    raise FitError("only TRAIN or validation metadata admission is allowed")
  if (set(split) != {"train", "validation", "test"} or any(not isinstance(v, list) for v in split.values())
      or any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))):
    raise FitError("requires disjoint frozen TRAIN/validation/test routes")
  sources, contracts = {}, set()
  for record in records:  # No arrays are accessed until ALL metadata gates pass.
    m = record["metadata"]
    if m.get("route") not in split[partition]:
      raise FitError(f"only reserved {partition.upper()} routes may be opened")
    for key in ("source_log_sha256", "static_identity_sha256", "policy_identity_sha256"):
      if not re.fullmatch(r"[0-9a-f]{64}", m.get(key, "")):
        raise FitError("missing hash-bound source/static/policy contract")
    source, route = m["source_log_sha256"], m["route"]
    if source in sources:
      if sources[source] != route:
        raise FitError("one physical source SHA cannot establish two routes")
      raise FitError("duplicate physical source record")
    sources[source] = route
    if not isinstance(m.get("cohort"), str) or not m["cohort"]:
      raise FitError("missing cohort contract")
    contracts.add((m["cohort"], m["static_identity_sha256"], m["policy_identity_sha256"], m.get("opendbc_gitlink")))
  if len(contracts) != 1:
    raise FitError("records must share one cohort/static/policy/source contract")
  contract = next(iter(contracts))
  verified = all(r["metadata"].get("source_identity_verified") is True for r in records)
  pin = contract[3]
  anchors = []
  for record in records:
    m, times = record["metadata"], []
    for kind in ("static", "policy"):
      proof = m.get(kind+"_anchor", {})
      if not isinstance(proof, dict):
        times = []
        break
      body = {key: value for key, value in proof.items() if key != "sha256"}
      if (type(body.get("t_ns")) is not int or not 0 <= body["t_ns"] <= np.iinfo(np.uint64).max
          or body.get("valid") is not True or body.get("identity_sha256") != m[kind+"_identity_sha256"]
          or body.get("route") != m["route"] or sources.get(body.get("source_log_sha256")) != m["route"]
          or proof.get("sha256") != identity_hash(body)):
        times = []
        break
      times.append(body["t_ns"])
    anchors.append(max(times) if len(times) == 2 else None)
  return contract, verified and isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{40}", pin) is not None, anchors


def _stream(arrays, name):
  required = [name+"_t_ns", name+"_valid"]
  required += [name+"_"+key+suffix for key in FIELDS[name] for suffix in ("", "_present")]
  if name == "LKAS11Tx":
    required += [name+"_"+key for key in ("origin", "src", "address", "decoded_ok")]
  if any(key not in arrays for key in required):
    raise FitError("missing_fields")
  values = {key.removeprefix(name+"_"): np.asarray(arrays[key]) for key in required}
  t = values.pop("t_ns")
  if t.ndim != 1 or t.dtype != np.uint64 or not len(t):
    raise FitError("raw_clock_dtype_or_empty")
  if any(v.shape != t.shape for v in values.values()):
    raise FitError("raw_shape")
  if (values["valid"].dtype.kind != "b" or any(values[key+"_present"].dtype.kind != "b" for key in FIELDS[name])):
    raise FitError("raw_validity_dtype")
  if any(values[key].dtype.kind not in "fiu" for key in FIELDS[name]):
    raise FitError("raw_numeric_dtype")
  ignored = 0
  if name == "LKAS11Tx":
    if values["decoded_ok"].dtype.kind != "b":
      raise FitError("raw_validity_dtype")
    keep = (values["origin"] == 0) & (values["src"] == 128) & (values["address"] == 0x340)
    ignored = int((~keep).sum())
    t, values = t[keep], {k: v[keep] for k, v in values.items()}
    if not len(t):
      raise FitError("no_returned_bus0_echo")
  repairs = sum(int(a) > int(b) for a, b in zip(t, t[1:], strict=False))
  order = np.argsort(t, kind="stable")
  t, values = t[order], {k: v[order] for k, v in values.items()}
  duplicate = np.zeros(len(t), bool)
  duplicate[:-1] |= t[:-1] == t[1:]
  duplicate[1:] |= t[:-1] == t[1:]
  return {"t": [int(v) for v in t], "v": values, "duplicate": duplicate, "repairs": repairs, "ignored": ignored}


def _span(stream, start, end, age_limit):
  times = stream["t"]
  anchor = bisect_right(times, start)-1
  first = max(0, bisect_right(times, start-age_limit)-1)
  last = bisect_right(times, end)
  indices = np.arange(first, last)
  gaps = [b-a for a, b in zip(times[first:last], times[first+1:last], strict=False)]
  missing = (anchor < 0 or times[0] > start-age_limit or start-times[max(anchor, 0)] > age_limit
             or not last or end-times[last-1] > age_limit)
  return indices, missing or any(gap > age_limit for gap in gaps)


def _quality(stream, name, indices):
  values, reasons = stream["v"], []
  if not len(indices) or not values["valid"][indices].all():
    reasons.append(name+"_invalid")
  if stream["duplicate"][indices].any():
    reasons.append(name+"_duplicate")
  for key in FIELDS[name]:
    if not values[key+"_present"][indices].all() or not np.isfinite(values[key][indices]).all():
      reasons.append(name+"_missing_or_nonfinite")
      break
  if name == "carState":
    if (any((values[k][indices] != 0).any() for k in ("steering_pressed", "brake_pressed", "gas_pressed"))
        or (np.abs(values["driver_torque"][indices]) > 50).any()):
      reasons.append("driver_override")
    if (values["fault_temporary"][indices] != 0).any() or (values["fault_permanent"][indices] != 0).any():
      reasons.append("steering_fault")
    if (values["vendor_rate_unsigned"][indices] < 0).any() or (values["v"][indices] < 0).any():
      reasons.append("invalid_unsigned_rate_or_speed")
  elif name == "liveLocationKalman":
    if (any((values[k][indices] != 1).any() for k in ("yaw_inner_valid", "roll_inner_valid", "inputs_ok", "sensors_ok", "device_stable"))
        or (values["status"][indices] != 2).any() or (values["yaw_std"][indices] < 0).any() or (values["roll_std"][indices] < 0).any()):
      reasons.append("location_quality")
  elif name == "carControl":
    if (values["lat_active"][indices] != 1).any():
      reasons.append("lateral_inactive")
  else:
    if not values["decoded_ok"][indices].all():
      reasons.append("echo_decode_invalid")
    if (values["request"][indices] != 1).any():
      reasons.append("request_false")
    if (values["torque_fault"][indices] != 0).any():
      reasons.append("request_fault")
    torque = values["torque"][indices]
    if (np.abs(torque) > 409).any() or (torque != np.rint(torque)).any():
      reasons.append("torque_envelope")
  return reasons


def _windows(record, anchor_t):
  arrays, m = record["arrays"], record["metadata"]
  try:
    streams = {name: _stream(arrays, name) for name in FIELDS}
  except FitError as error:
    return [], [{"source_log_sha256": m["source_log_sha256"], "start_ns": None, "end_ns": None, "reasons": [str(error)]}], 0, 0
  llk, cs, tx = (streams[name] for name in ("liveLocationKalman", "carState", "LKAS11Tx"))
  accepted, excluded = [], []
  for i, (start, end) in enumerate(zip(llk["t"], llk["t"][1:], strict=False)):
    reasons = [] if abs(end-start-MACRO_NS) <= DT_TOLERANCE_NS else ["macro_dt"]
    if anchor_t is None:
      reasons.append("static_policy_timing_unknown")
    elif start < anchor_t:
      reasons.append("before_static_or_policy_anchor")
    reasons += _quality(llk, "liveLocationKalman", np.array([i, i+1]))
    for name in ("carState", "LKAS11Tx", "carControl"):
      age = 15_000_000 if name == "carState" else 20_000_000
      indices, gap = _span(streams[name], start, end, age)
      if gap:
        reasons.append(name+"_gap_or_prefix")
      reasons += _quality(streams[name], name, indices)
    provenance = {"source_log_sha256": m["source_log_sha256"], "route": m["route"], "start_ns": start, "end_ns": end}
    if reasons:
      excluded.append({**provenance, "reasons": sorted(set(reasons))})
      continue
    nodes = [start+j*NATIVE_NS for j in range(5)]
    ci = [bisect_right(cs["t"], node)-1 for node in nodes]
    ti = [bisect_right(tx["t"], node)-1 for node in nodes]
    endpoint = bisect_right(cs["t"], end)-1
    if (min(ci+ti+[endpoint]) < 0 or any(node-cs["t"][j] > 15_000_000 for node, j in zip(nodes, ci, strict=True))
        or any(node-tx["t"][j] > 20_000_000 for node, j in zip(nodes, ti, strict=True))):
      excluded.append({**provenance, "reasons": ["micro_asof_missing"]})
      continue
    initial = np.array([cs["v"]["steering_angle"][ci[0]], llk["v"]["yaw"][i], cs["v"]["vendor_rate_unsigned"][ci[0]]])
    final = np.array([cs["v"]["steering_angle"][endpoint], llk["v"]["yaw"][i+1], cs["v"]["vendor_rate_unsigned"][endpoint]])
    features = np.r_[initial, llk["v"]["roll"][i], cs["v"]["v"][ci], tx["v"]["torque"][ti]]
    accepted.append({**provenance, "features": features, "delta": final-initial, "dt_s": (end-start)*1e-9})
  return accepted, excluded, sum(s["repairs"] for s in streams.values()), sum(s["ignored"] for s in streams.values())


def _cell_key(edges, features):
  return ":".join(str(int(np.searchsorted(edge, value, side="left"))) for edge, value in zip(edges, features, strict=True))


def fit(records, split, *, quantile_bins=2, min_samples=100, min_routes=2):
  if type(min_samples) is not int or min_samples < 100 or type(min_routes) is not int or min_routes < 2:
    raise FitError("support requires at least100 samples and two physical routes")
  if type(quantile_bins) is not int or not 1 <= quantile_bins <= 8 or not records:
    raise FitError("requires records and a finite1..8quantile grid")
  contract, pin_verified, anchors = _metadata(records, split)
  windows, exclusions, repairs, ignored = [], [], 0, 0
  for record, anchor_t in zip(records, anchors, strict=True):
    accepted, excluded, count, discarded = _windows(record, anchor_t)
    windows += accepted
    exclusions += excluded
    repairs += count
    ignored += discarded
  identity = dict(zip(("cohort", "static_identity_sha256", "policy_identity_sha256", "opendbc_gitlink"), contract, strict=True))
  model = {"kind": KIND, **identity, "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5,
           "macro_dt_tolerance_s": .002, "features": list(FEATURES), "cells": {},
           "source_eligible": pin_verified and all(t is not None for t in anchors), "source_pin_verified": pin_verified,
           "static_policy_timing_verified": all(t is not None for t in anchors),
           "hundred_hz_feedback_validated": False, "selection_eligible": False, "production_enabled": False, "candidate": None,
           "min_samples": min_samples, "min_routes": min_routes, "split": {k: list(v) for k, v in split.items()},
           "record_hashes": [_hash({"metadata": r["metadata"], "arrays": r["arrays"]}) for r in records],
           "exclusions": exclusions, "excluded_reasons": dict(Counter(reason for e in exclusions for reason in e["reasons"])),
           "accepted_windows": len(windows), "attempted_windows": len(windows)+len(exclusions), "clock_order_repairs": repairs,
           "ignored_non_echo_rows": ignored,
           "semantics": "Nominal50ms joint delta at end of five past10ms commands/speeds; native feedback state is ZOH between macro ends.",
           "request_contract": "Returned can src128 LKAS11 request=true/fault=false, including20ms prefix; command proxy, not delivery proof.",
           "limitations": ["100Hz angle feedback/ZOH not validated", "Publication timing is not sensor acquisition timing",
                           "Initial joint state and five echo commands form an unvalidated Markov/command-proxy hypothesis, not identified actuator lag",
                           "Commands are unshifted returned CAN; CP/model delay must not be applied again to this fitted contract",
                           "Actual48..52ms deltas retained without rescaling", "Physical LLK roll is distinct from native LP.roll",
                           "No low-speed/stop or current-source acceptance is established", "Driver torque within50 counts is not proven zero"]}
  if not windows:
    model.update(edges=[], feature_min=[], feature_max=[], actual_dt_s={"min": None, "max": None}, supported_cells=0)
    model["model_sha256"] = _hash(model)
    return model
  x = np.stack([w["features"] for w in windows])
  lower, upper = x.min(axis=0), x.max(axis=0)
  edges = []
  for j in range(x.shape[1]):
    values = np.unique(np.quantile(x[:, j], np.arange(1, quantile_bins)/quantile_bins))
    edges.append(values[(values >= lower[j]) & (values < upper[j])] if lower[j] < upper[j] else np.array([]))
  grouped = defaultdict(list)
  for window in windows:
    grouped[_cell_key(edges, window["features"])].append(window)
  for key, rows in grouped.items():
    features = np.stack([row["features"] for row in rows])
    routes = sorted({row["route"] for row in rows})
    supported = len(rows) >= min_samples and len(routes) >= min_routes
    model["cells"][key] = {"samples": len(rows), "routes": routes, "supported": supported,
      "rejections": ([] if len(rows) >= min_samples else ["sample_support"])+([] if len(routes) >= min_routes else ["route_support"]),
      "feature_min": features.min(axis=0).tolist(), "feature_max": features.max(axis=0).tolist(),
      "joint_delta_median": np.median(np.stack([row["delta"] for row in rows]), axis=0).tolist()}
  model.update(edges=[edge.tolist() for edge in edges], feature_min=lower.tolist(), feature_max=upper.tolist(),
               actual_dt_s={"min": min(w["dt_s"] for w in windows), "max": max(w["dt_s"] for w in windows)},
               supported_cells=sum(cell["supported"] for cell in model["cells"].values()))
  model["model_sha256"] = _hash(model)
  return model


def predict_macro(model, state, speed, command, roll):
  """Consume only the initial joint state and completed macro's five causal nodes."""
  try:
    if set(state) != set(STATE_KEYS) or len(speed) != 5 or len(command) != 5:
      return Unsupported("invalid_macro_inputs")
    initial, speed, command = np.array([state[k] for k in STATE_KEYS], float), np.asarray(speed, float), np.asarray(command, float)
    features = np.r_[initial, roll, speed, command]
    if (features.shape != (14,) or not np.isfinite(features).all() or initial[2] < 0 or (speed < 0).any()
        or (np.abs(command) > 409).any() or (command != np.rint(command)).any()):
      return Unsupported("invalid_macro_inputs")
  except (TypeError, ValueError, KeyError):
    return Unsupported("invalid_macro_inputs")
  if model.get("kind") != KIND or model.get("macro_dt_s") != .05 or model.get("native_dt_s") != .01:
    return Unsupported("model_contract")
  if not model["cells"]:
    return Unsupported("no_supported_population")
  if (features < model["feature_min"]).any() or (features > model["feature_max"]).any():
    return Unsupported("outside_training_bounds")
  cell = model["cells"].get(_cell_key(model["edges"], features))
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
