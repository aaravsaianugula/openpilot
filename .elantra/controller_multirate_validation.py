"""Pure frozen-model VALIDATION diagnostics, never fitting or tune admission.

Completed50ms transitions are conditioned on observed commands/reports during
that interval. They are not forecasts available at its start. File/schema/helper
authenticity is a reviewed caller's responsibility, separate from content claims.
"""
from collections import Counter
from copy import deepcopy
import re

import numpy as np

import controller_multirate_joint as moving
import controller_multirate_joint_stop_hold as categorical

base, policy = moving.base, categorical.lateral_hold
PROVENANCE_KIND = "frozen_multirate_validation_provenance_v1"
_SHA = re.compile(r"[0-9a-f]{64}")
_PIN = re.compile(r"[0-9a-f]{40}")
_CONTEXT = {"kind", "model_sha256", "split_sha256", "static_identity_sha256", "training_receipt_sha256",
            "training_helpers_sha256", "validation_helpers_sha256"}
_REFERENCES = {"source_log_sha256", "motion_npz_sha256", "motion_arrays_sha256", "motion_metadata_sha256",
               "view_arrays_sha256", "view_receipt_sha256"}


class ValidationError(ValueError):
  pass


def _hash(value):
  try:
    return base.identity_hash(value)
  except (TypeError, ValueError) as error:
    raise ValidationError("noncanonical validation content") from error


def _array(value):
  try:
    return np.asarray(value, float)
  except (TypeError, ValueError) as error:
    raise ValidationError("nonnumeric frozen model claim") from error


def required_helpers(kind):
  if not isinstance(kind, str):
    raise ValidationError("unsupported frozen model kind")
  common = ("controller_multirate_lateral.py", "controller_multirate_joint.py", "controller_multirate_stop_hold.py",
            "controller_motion_longitudinal.py", "controller_native_lateral_runtime.py")
  if kind == moving.KIND:
    return common
  if kind in (categorical.KIND, categorical.APPROACH_KIND):
    return (*common, "controller_multirate_joint_stop_hold.py")
  raise ValidationError("unsupported frozen model kind")


def _split(split):
  if (not isinstance(split, dict) or set(split) != {"train", "validation", "test"}
      or any(not isinstance(v, list) or any(not isinstance(r, str) or not r for r in v)
             or len(v) != len(set(v)) for v in split.values())
      or any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))):
    raise ValidationError("frozen disjoint route partitions without duplicates required")


def _model(model, split, cp, sp):
  if not isinstance(model, dict):
    raise ValidationError("frozen model mapping required")
  required_helpers(model.get("kind"))
  _split(model.get("split"))
  category = model["kind"] != moving.KIND
  dimensions = 31 if category else 21
  if (model.get("model_sha256") != _hash({k: v for k, v in model.items() if k != "model_sha256"})
      or model.get("split") != split or model.get("static_identity_sha256") != _hash({"cp": cp, "sp": sp})
      or not isinstance(model.get("cohort"), str) or not model["cohort"]
      or any(model.get(k) is not True for k in ("source_eligible", "source_pin_verified", "static_policy_timing_verified"))
      or not isinstance(model.get("opendbc_gitlink"), str) or not _PIN.fullmatch(model["opendbc_gitlink"])
      or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
      or not isinstance(model.get("features"), list) or any(not isinstance(v, str) for v in model["features"])
      or model["features"] != list(categorical.FEATURES if category else moving.FEATURES)
      or type(model.get("min_samples")) is not int or model["min_samples"] < 100
      or type(model.get("min_routes")) is not int or model["min_routes"] < 2):
    raise ValidationError("frozen model source/split/static/policy/clock identity unavailable or differs")
  if category and (model.get("native_policy_source_sha256") != categorical.NATIVE_POLICY_SHA
      or model.get("card_source_sha256") != categorical.CARD_SHA
      or (model["kind"] == categorical.KIND and model.get("low_speed_max_initial_mps") != 3.)
      or (model["kind"] == categorical.APPROACH_KIND and
          (model.get("initial_speed_domain") != "retained_train_global_and_cell_bounds" or "low_speed_max_initial_mps" in model))):
    raise ValidationError("categorical native policy or identification domain differs")
  lo, hi = (_array(model.get(k)) for k in ("feature_min", "feature_max"))
  edges, cells = model.get("edges"), model.get("cells")
  if (lo.shape != (dimensions,) or hi.shape != (dimensions,) or not np.isfinite([lo, hi]).all() or (lo > hi).any()
      or not isinstance(edges, list) or len(edges) != dimensions or not isinstance(cells, dict)):
    raise ValidationError("malformed frozen feature population")
  for e, a, b in zip(edges, lo, hi, strict=True):
    values = _array(e)
    if values.ndim != 1 or not np.isfinite(values).all() or (np.diff(values) <= 0).any() or (values < a).any() or (values >= b).any():
      raise ValidationError("malformed frozen feature edges")
  patterns = model.get("patterns")
  if category and (not isinstance(patterns, list) or any(not isinstance(p, str) or not re.fullmatch(r"[01]\|[01]{10}", p)
                                                      for p in patterns) or len(patterns) != len(set(patterns))):
    raise ValidationError("malformed categorical patterns")
  for key, cell in cells.items():
    parts = key.split("|") if isinstance(key, str) else []
    if not isinstance(cell, dict) or len(parts) != (3 if category else 1) or (category and "|".join(parts[:2]) not in patterns):
      raise ValidationError("malformed frozen cell key")
    tokens = parts[-1].split(":")
    if len(tokens) != dimensions or any(not re.fullmatch(r"0|[1-9][0-9]*", v) for v in tokens) or any(
        int(v) > len(e) for v, e in zip(tokens, edges, strict=True)):
      raise ValidationError("malformed frozen bin key")
    bounds = [_array(cell.get(k)) for k in ("feature_min", "feature_max")]
    delta, routes, samples = _array(cell.get("joint_delta_median")), cell.get("routes"), cell.get("samples")
    if (any(v.shape != (dimensions,) or not np.isfinite(v).all() for v in bounds)
        or (bounds[0] > bounds[1]).any() or (bounds[0] < lo).any() or (bounds[1] > hi).any()
        or delta.shape != (5,) or not np.isfinite(delta).all() or type(cell.get("supported")) is not bool
        or type(samples) is not int or samples < 1 or not isinstance(routes, list)
        or any(not isinstance(r, str) or r not in split["train"] for r in routes) or len(routes) != len(set(routes))):
      raise ValidationError("malformed frozen cell bounds/delta/routes")
    if category:
      values = cell.get("next_hardware_standstill_values")
      if (not isinstance(values, list) or not values or any(type(v) is not bool for v in values)
          or len(values) != len(set(values)) or not np.array_equal(bounds[0][-10:], list(map(int, parts[1])))
          or not np.array_equal(bounds[1][-10:], list(map(int, parts[1])))):
        raise ValidationError("malformed frozen categorical cell")
    if cell["supported"] and (samples < model["min_samples"] or len(routes) < model["min_routes"]
                              or (category and len(values) != 1)):
      raise ValidationError("frozen cell lacks declared independent TRAIN support")
  if type(model.get("supported_cells")) is not int or model["supported_cells"] != sum(c["supported"] for c in cells.values()):
    raise ValidationError("frozen cell support denominator differs")
  eligibility, records = model.get("input_eligibility"), model.get("record_hashes")
  if (not isinstance(eligibility, list) or not eligibility or not isinstance(records, list) or len(records) != len(eligibility)
      or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in records) or len(records) != len(set(records))):
    raise ValidationError("frozen training record/source catalog missing")
  sources = set()
  for row in eligibility:
    if (not isinstance(row, dict) or not isinstance(row.get("source_log_sha256"), str) or not _SHA.fullmatch(row["source_log_sha256"])
        or row.get("source_eligible") is not True or row.get("reasons") != [] or row["source_log_sha256"] in sources):
      raise ValidationError("untrusted or duplicate TRAIN source")
    sources.add(row["source_log_sha256"])
  routes = {}
  evidence, exclusions = model.get("window_evidence"), model.get("exclusions")
  if not isinstance(evidence, list) or not isinstance(exclusions, list) or any(not isinstance(r, dict) for r in (*evidence, *exclusions)):
    raise ValidationError("malformed TRAIN source route evidence")
  for row in (*evidence, *exclusions):
    source, route = row.get("source_log_sha256"), row.get("route")
    if (not isinstance(source, str) or not isinstance(route, str) or source not in sources
        or route not in split["train"] or source in routes and routes[source] != route):
      raise ValidationError("frozen TRAIN physical-source route binding differs")
    routes[source] = route
  if set(routes) != sources:
    raise ValidationError("TRAIN source route evidence incomplete")
  return category, sources


def _provenance(record, model, split, cp, sp, category):
  p = record.get("validation_provenance")
  refs = _REFERENCES | ({"queue_arrays_sha256", "queue_metadata_sha256"} if category else set())
  keys = _CONTEXT | refs | {"sha256"}
  if not isinstance(p, dict) or set(p)-{"evaluator_sha256"} != keys:
    raise ValidationError("required frozen training receipt/helper provenance missing")
  body = {k: v for k, v in p.items() if k != "sha256"}
  if (p["kind"] != PROVENANCE_KIND or p["sha256"] != _hash(body)
      or any(not isinstance(p.get(k), str) or not _SHA.fullmatch(p[k]) for k in
             (refs | {"model_sha256", "split_sha256", "static_identity_sha256", "training_receipt_sha256"}))
      or ("evaluator_sha256" in p and (not isinstance(p["evaluator_sha256"], str) or not _SHA.fullmatch(p["evaluator_sha256"])))
      or p["model_sha256"] != _hash(model) or p["split_sha256"] != _hash(split)
      or p["static_identity_sha256"] != _hash({"cp": cp, "sp": sp})):
    raise ValidationError("frozen validation provenance identity differs")
  expected_helpers = set(required_helpers(model["kind"]))
  for name in ("training_helpers_sha256", "validation_helpers_sha256"):
    h = p[name]
    if not isinstance(h, dict) or set(h) != expected_helpers or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in h.values()):
      raise ValidationError("exact reused collector/predictor helper claim map required")
  if p["training_helpers_sha256"] != p["validation_helpers_sha256"]:
    raise ValidationError("training and validation helper claims differ")
  original, receipt = record["motion_metadata"], record["view_receipt"]
  expected = {"source_log_sha256": original.get("source_log_sha256"), "motion_npz_sha256": original.get("output_npz_sha256"),
    "motion_arrays_sha256": receipt.get("input_arrays_sha256"), "motion_metadata_sha256": _hash(original),
    "view_arrays_sha256": receipt.get("output_view_arrays_sha256"), "view_receipt_sha256": _hash(receipt)}
  if category:
    q = record.get("queued_apply_metadata")
    if not isinstance(q, dict):
      raise ValidationError("queued metadata provenance missing")
    expected.update(queue_arrays_sha256=q.get("arrays_sha256"), queue_metadata_sha256=_hash(q))
  if any(p[k] != v for k, v in expected.items()):
    raise ValidationError("original/view/queue provenance refs differ")
  return {k: deepcopy(v) for k, v in p.items() if k in _CONTEXT or k == "evaluator_sha256"}


def _preflight(model, records, split, cp, sp):
  _split(split)
  category, train_sources = _model(model, split, cp, sp)
  if not isinstance(records, (list, tuple)) or not records:
    raise ValidationError("explicit VALIDATION records required")
  contexts, sources = [], set()
  for row in records:  # ALL metadata, including the final record, before ANY arrays.
    if not isinstance(row, dict) or any(not isinstance(row.get(k), dict) for k in ("metadata", "motion_metadata", "view_receipt")):
      raise ValidationError("record source metadata missing")
    m, original = row["metadata"], row["motion_metadata"]
    source = m.get("source_log_sha256")
    if (m.get("route") not in split["validation"] or not isinstance(source, str) or not _SHA.fullmatch(source)
        or source in sources or source in train_sources
        or any(m.get(k) != model.get(k) for k in ("cohort", "static_identity_sha256", "policy_identity_sha256", "opendbc_gitlink"))
        or m.get("source_identity_verified") is not True or original.get("source_eligible") is not True
        or original.get("eligibility_reasons") != [] or original.get("opendbc_gitlink") != model["opendbc_gitlink"]):
      raise ValidationError("VALIDATION-only independent trusted original source/model identity required")
    sources.add(source)
    contexts.append(_provenance(row, model, split, cp, sp, category))
  if any(v != contexts[0] for v in contexts[1:]):
    raise ValidationError("validation records require one frozen training/provenance context")
  if category:
    _, verified, anchors = categorical._metadata(records, split, cp, sp, partition="validation")
  else:
    _, verified, anchors = policy._metadata(records, split, cp, sp, partition="validation")
    for row in records:
      moving._binding_metadata(row, split, partition="validation")
  if not verified or any(v is None for v in anchors):
    raise ValidationError("source/static/policy anchor proof unavailable")
  return category, anchors, contexts[0]


def _prediction(model, window, category):
  x = window["features"]
  state = dict(zip(moving.STATE_KEYS, x[:5].tolist(), strict=True))
  inputs = {"initial_state": state, "physical_llk_roll": float(x[5]), "accel_request5": x[6:11].tolist(),
            "accel_reported5": x[11:16].tolist(), "returned_can5": x[16:21].tolist()}
  args = (model, state, x[6:11], x[11:16], x[16:21])
  if category:
    inputs.update(effective_active5=x[21:26].astype(bool).tolist(), request5=x[26:31].astype(bool).tolist(),
                  initial_hardware_standstill=window["hardware_standstill"])
    fn = categorical.predict_approach_macro if model["kind"] == categorical.APPROACH_KIND else categorical.predict_macro
    result = fn(*args, x[21:26].astype(bool), x[26:31].astype(bool), float(x[5]), window["hardware_standstill"])
  else:
    result = moving.predict_macro(*args, float(x[5]))
  actual = dict(zip(moving.STATE_KEYS, (x[:5]+window["delta"]).tolist(), strict=True))
  item = {"prediction_inputs": inputs, "actual_endpoint": actual,
          "measured_motion_direction": "LEFT" if x[3] < 0 else "RIGHT" if x[3] > 0 else "zero",
          "category": {"status": "not_predicted", "predicted_next": None}, "reasons": []}
  if category:
    item["category"] = {"status": "unsupported", "initial_observed": window["hardware_standstill"],
      "actual_next": window["next_hardware_standstill"], "predicted_next": None, "next_error": None, "transition_error": None}
  if isinstance(result, base.Unsupported):
    item.update(status="unsupported", predicted_endpoint=None, endpoint_residual=None, reasons=[result.reason])
  else:
    prediction = result["state"] if category else result
    item.update(status="predicted", predicted_endpoint=prediction,
                endpoint_residual={k: actual[k]-prediction[k] for k in moving.STATE_KEYS})
    if category:
      initial, predicted, observed = window["hardware_standstill"], result["hardware_standstill"], window["next_hardware_standstill"]
      item["category"] = {"status": "predicted", "initial_observed": initial, "predicted_next": predicted, "actual_next": observed,
                          "next_error": predicted != observed, "transition_error": (predicted != initial) != (observed != initial)}
  return item


def _record_ref(record, index):
  """Metadata-only claims remain reportable when raw content must stay unopened."""
  def mapping(key):
    value = record.get(key) if isinstance(record, dict) else None
    return value if isinstance(value, dict) else {}
  m, original, view, queue = (mapping(k) for k in ("metadata", "motion_metadata", "view_receipt", "queued_apply_metadata"))
  identity = {k: m.get(k) if isinstance(m.get(k), str) else None for k in ("route", "segment", "source_log_sha256")}
  refs = {"motion_npz_sha256": original.get("output_npz_sha256"), "motion_arrays_sha256": view.get("input_arrays_sha256"),
          "view_arrays_sha256": view.get("output_view_arrays_sha256"), "queue_arrays_sha256": queue.get("arrays_sha256")}
  reasons = original.get("eligibility_reasons")
  return {**identity, "id": _hash(identity), "record_index": index,
    "claimed_content_refs": {k: v if isinstance(v, str) and _SHA.fullmatch(v) else None for k, v in refs.items()},
    "original_eligibility_reasons": list(reasons) if isinstance(reasons, list) and all(isinstance(r, str) for r in reasons) else None}


def evaluate_validation_transitions(model, records, split, cp, sp):
  """No fitting, sourcefile reads, quantiles, mutation or tune acceptance."""
  report = {"status": "diagnostic", "windows": [], "record_rejections": [], "counts": {}, "routes": {}, "regimes": {}, "directions": {},
    "category_counts": {"predicted": 0, "unsupported": 0, "not_predicted": 0, "quality_unknown": 0,
                        "known_label_denominator": 0, "error_denominator": 0, "next_errors": 0, "transition_errors": 0, "confusion": {}},
    "aggregate_score": None, "predictive_validated": False, "selection": None, "production_enabled": False,
    "state_units": {"v": "m/s", "a": "m/s^2", "angle": "degrees", "yaw": "rad/s", "vendor_rate": "unsigned degrees/s"},
    "direction_semantics": "Measured initial calibrated yaw LEFT<0/RIGHT>0/zero; never requested turn, countersteering or target phase",
    "caller_authentication_verified": False, "window_population_known": True,
    "semantics": "Completed50ms conditional transition given observed commands/reports; not a start-time forecast",
    "limitations": ["Caller training/helper/file/schema provenance is hash-bound claimed evidence, not independently authenticated here",
                    "Unsupported and quality-excluded populations remain explicit; no successful-only overall PASS or ranking",
                    "No physical acceptance thresholds,100Hz/ZOH/contact/coherence or road-tune validation inferred"]}
  claimed = model.get("model_sha256") if isinstance(model, dict) else None
  report["claimed_model_body_sha256"] = claimed if isinstance(claimed, str) and _SHA.fullmatch(claimed) else None
  report["unhashable_input_metadata"] = []
  for key, value in (("model_sha256", model), ("split_sha256", split), ("static_identity_sha256", {"cp": cp, "sp": sp})):
    try:
      report[key] = _hash(value)
    except ValidationError:
      report[key] = None
      report["unhashable_input_metadata"].append(key)
  try:
    category, anchors, context = _preflight(model, records, split, cp, sp)
  except (ValidationError, base.FitError) as error:
    report.update(status="metadata_rejected", window_population_known=False, metadata_rejection=str(error))
    for index, row in enumerate(records if isinstance(records, (list, tuple)) else []):
      rejection = {**_record_ref(row, index), "stage": "metadata_batch", "start_ns": None, "end_ns": None, "reasons": [str(error)]}
      report["record_rejections"].append(rejection)
      route = rejection["route"]
      if route is not None:
        report["routes"].setdefault(route, {"metadata_rejected_records": 0})["metadata_rejected_records"] += 1
    report["counts"] = {"attempted": 0, "predicted": 0, "unsupported": 0, "quality_excluded": 0,
                        "metadata_rejected_records": len(report["record_rejections"])}
    report["coverage"] = None
    return report
  report.update(model_sha256=_hash(model), split_sha256=_hash(split), static_identity_sha256=_hash({"cp": cp, "sp": sp}),
                provenance_context=context, record_bindings=[deepcopy(r["validation_provenance"]) for r in records])
  for index, (record, anchor) in enumerate(zip(records, anchors, strict=True)):
    try:
      moving._bind(record, split, partition="validation")
    except base.FitError as error:
      report["window_population_known"] = False
      report["record_rejections"].append({**_record_ref(record, index), "stage": "content", "start_ns": None, "end_ns": None, "reasons": [str(error)]})
      continue
    if category:
      maximum = 3. if model["kind"] == categorical.KIND else None
      accepted, excluded, _, _ = categorical._windows(record, anchor, maximum_initial_mps=maximum)
    else:
      accepted, excluded, _, _ = moving._windows(record, anchor)
    for ordinal, (window, valid) in enumerate(sorted([(w, True) for w in accepted]+[(w, False) for w in excluded],
                                                   key=lambda p: (p[0]["start_ns"] is not None, p[0]["start_ns"] or 0, p[0]["end_ns"] or 0))):
      identity = {k: window[k] for k in ("route", "source_log_sha256", "start_ns", "end_ns")}
      item = {**identity, "id": _hash({**identity, "ordinal": ordinal}), "regime": model["kind"]}
      if valid:
        item.update(_prediction(model, window, category), actual_dt_s=window["dt_s"])
      else:
        item.update(status="quality_excluded", measured_motion_direction="unknown", reasons=list(window["reasons"]))
        if window["start_ns"] is None:
          report["window_population_known"] = False
          report["record_rejections"].append({**_record_ref(record, index), "stage": "collector", "start_ns": None, "end_ns": None,
                                              "reasons": list(window["reasons"])})
          continue  # No raw macro clock means no invented one-window denominator.
      report["windows"].append(item)
  statuses = Counter(w["status"] for w in report["windows"])
  report["counts"] = {"attempted": len(report["windows"]), **{k: statuses[k] for k in ("predicted", "unsupported", "quality_excluded")},
                      "metadata_rejected_records": 0, "content_rejected_records": sum(r["stage"] == "content" for r in report["record_rejections"]),
                      "collector_rejected_records": sum(r["stage"] == "collector" for r in report["record_rejections"])}
  report["coverage"] = statuses["predicted"]/len(report["windows"]) if report["window_population_known"] and report["windows"] else None
  for key, values, field in (("routes", {r["metadata"]["route"] for r in records}, "route"),
                            ("regimes", {model["kind"]}, "regime"),
                            ("directions", {"LEFT", "RIGHT", "zero", "unknown"}, "measured_motion_direction")):
    for value in sorted(values):
      rows = [w for w in report["windows"] if w[field] == value]
      counts = Counter(w["status"] for w in rows)
      residuals = [w["endpoint_residual"] for w in rows if w["status"] == "predicted"]
      report[key][value] = {"attempted": len(rows), **{k: counts[k] for k in ("predicted", "unsupported", "quality_excluded")},
        "conditional_endpoint_mae": {k: float(np.mean([abs(r[k]) for r in residuals])) for k in moving.STATE_KEYS} if residuals else None,
        "conditional_endpoint_p95_abs": {k: float(np.percentile([abs(r[k]) for r in residuals], 95)) for k in moving.STATE_KEYS} if residuals else None,
        "residual_sample_count": len(residuals), "residual_scope": "Supported predictions only; full exclusions/unsupported denominator above"}
  report["conditional_endpoint_errors"] = deepcopy(report["regimes"][model["kind"]])
  confusion = Counter()
  for window in report["windows"]:
    c = window.get("category", {})
    if category and window["status"] == "quality_excluded":
      report["category_counts"]["quality_unknown"] += 1
    if c.get("status") in ("predicted", "unsupported"):
      report["category_counts"]["known_label_denominator"] += 1
    if c.get("status") in ("unsupported", "not_predicted"):
      report["category_counts"][c["status"]] += 1
    if c.get("status") == "predicted":
      report["category_counts"]["predicted"] += 1
      report["category_counts"]["error_denominator"] += 1
      report["category_counts"]["next_errors"] += int(c["next_error"])
      report["category_counts"]["transition_errors"] += int(c["transition_error"])
      confusion[str(c["actual_next"]).lower()+"->"+str(c["predicted_next"]).lower()] += 1
  report["category_counts"]["confusion"] = dict(confusion)
  report["report_sha256"] = _hash(report)
  return report
