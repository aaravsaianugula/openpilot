"""Pure CONDITIONAL diagnostic comparison, never source authentication or a tune.

The caller must independently authenticate predeclaration, files, imports, source,
receipts and expected bindings. Matching hashes here checks conditional integrity;
it cannot establish that authentication or physical/causal plant validity. All
original full-tune rejections survive. No file reads, fitting or native execution.
"""
from collections import Counter
from copy import deepcopy
import json
import re

import numpy as np

import controller_multirate_validation as validation
import controller_native_two_way as runner
from controller_multirate_lateral import identity_hash

joint, moving, category = runner.shared, runner.plant, runner.low_plant
UNITS = {"v": "m/s", "a": "m/s^2", "angle": "degrees", "yaw": "rad/s", "vendor_rate": "unsigned degrees/s"}
SEMANTICS = "Completed50ms conditional transition given observed commands/reports; not a start-time forecast"
SPEC_KIND = "predeclared_five_state_diagnostic_limits_v1"
EXPECTED_KIND = "trusted_caller_predictive_bindings_v1"
SHA = re.compile(r"[0-9a-f]{64}")
PIN = re.compile(r"[0-9a-f]{40}")
# Closed source-literal classification, not a prefix-based 'unvalidated' bypass.
# These remain limitations, not resolved claims, even when diagnostics are ranked.
RETAINED_ASSUMPTIONS = frozenset((
  "unvalidated_100hz_zoh", "unvalidated_joint_median_coherence", "unvalidated_v_a_kinematics",
  "unvalidated_markov_command_history_sufficiency_for_actuator_latency", "native_initialization_and_apply_phase_uncertain",
  "frozen_learner_and_perception", "unvalidated_held_initial_state_during_cold_prefix",
  "unvalidated_sensor_and_state_publication_sampling", "publication_schedule_caller_authenticated_not_receive_clock",
  "CAN_apply_is_command_proxy_not_delivered_torque", "no_validated_two_way_stop_hold_or_coherence", "diagnostic_only",
  "joint_plant_predictive_validation_unavailable"))
STOP_ASSUMPTIONS = frozenset(("contact_and_hold_unvalidated", "independent_categorical_state_dynamics_unvalidated",
                             "frozen_OEM_cruise_standstill_unverified"))
DOMAIN_LABELS = frozenset(("low_speed_stop_hold_plant_missing", "high_speed_to_low_speed_categorical_transition_unimplemented",
                           "fullrange_categorical_contact_response_unvalidated"))


class SelectionError(ValueError):
  pass


def _need(condition, reason):
  if not condition:
    raise SelectionError(reason)


def _hash(value):
  try:
    return identity_hash(value)
  except (TypeError, ValueError, validation.base.FitError) as error:
    raise SelectionError("noncanonical input content") from error


def _sha(value):
  return isinstance(value, str) and SHA.fullmatch(value) is not None


def _number(value):
  return type(value) in (int, float) and np.isfinite(value)


def _state(value):
  _need(isinstance(value, dict) and set(value) == set(moving.STATE_KEYS)
        and all(_number(v) for v in value.values()) and value["v"] >= 0 and value["vendor_rate"] >= 0,
        "unknown/nonfinite five-state endpoint")
  return value


def _specification(spec, expected, aliases):
  _need(isinstance(spec, dict) and spec.get("kind") == SPEC_KIND and isinstance(expected, dict)
        and expected.get("kind") == EXPECTED_KIND, "predeclared specification and trusted-caller expected bindings required")
  _need(spec.get("sha256") == _hash({k: v for k, v in spec.items() if k != "sha256"})
        and expected.get("specification_sha256") == _hash(spec)
        and _sha(spec.get("predeclaration_receipt_sha256"))
        and expected.get("predeclaration_receipt_sha256") == spec["predeclaration_receipt_sha256"], "predeclaration hash/receipt differs")
  _need(spec.get("state_units") == UNITS and spec.get("semantics") == SEMANTICS, "exact state units/conditional semantics required")
  _need(isinstance(spec.get("models"), dict) and set(spec["models"]) == aliases, "explicit limits for every used model required")
  for limits in spec["models"].values():
    _need(isinstance(limits, dict), "malformed quantitative limits")
    for key in ("minimum_attempted", "minimum_predicted", "minimum_routes", "minimum_predicted_per_route", "minimum_predicted_per_direction"):
      _need(type(limits.get(key)) is int and limits[key] > 0, "positive explicit quantitative denominators required")
    for key in ("minimum_prediction_coverage", "maximum_unsupported_fraction", "maximum_quality_excluded_fraction"):
      _need(_number(limits.get(key)) and 0 <= limits[key] <= 1, "explicit finite coverage/failure limits required")
    directions = limits.get("required_measured_directions")
    _need(isinstance(directions, list) and directions and all(v in ("LEFT", "RIGHT", "zero") for v in directions)
          and len(directions) == len(set(directions)), "explicit measured-motion strata required")
    errors = limits.get("state_error_limits")
    _need(isinstance(errors, dict) and set(errors) == set(UNITS), "all five state error limits required")
    _need(all(isinstance(v, dict) and set(v) == {"mae", "p95_abs"} and all(_number(x) and x >= 0 for x in v.values())
              for v in errors.values()), "finite nonnegative state error limits required")
    categorical = limits.get("categorical")
    if categorical is not None:
      _need(isinstance(categorical, dict) and type(categorical.get("minimum_predicted")) is int
            and categorical["minimum_predicted"] > 0, "explicit categorical denominator required")
      _need(all(_number(categorical.get(k)) and 0 <= categorical[k] <= 1 for k in
                ("maximum_next_error_fraction", "maximum_transition_error_fraction")), "explicit finite categorical error limits required")


def _sources(alias, model, report, expected, split, all_sources):
  binding = expected["models"][alias]
  _need(isinstance(binding, dict) and binding.get("model_sha256") == _hash(model)
        and binding.get("validation_report_sha256") == report.get("report_sha256")
        and _sha(binding.get("validation_receipt_sha256")), "frozen model/validation receipt binding differs")
  _need(report.get("model_sha256") == _hash(model) and report.get("claimed_model_body_sha256") == model["model_sha256"]
        and report.get("split_sha256") == expected["split_sha256"]
        and report.get("static_identity_sha256") == expected["static_identity_sha256"], "validation report headers differ")
  training = {r["source_log_sha256"]: r["route"] for r in (*model["window_evidence"], *model["exclusions"])}
  _need(binding.get("training_sources") == training, "TRAIN physical-source catalog differs")
  original = binding.get("validation_sources")
  _need(isinstance(original, dict) and original, "original VALIDATION source catalog missing")
  for source, metadata in original.items():
    _need(_sha(source) and isinstance(metadata, dict) and metadata.get("route") in split["validation"]
          and metadata.get("partition") == "validation" and source not in training
          and metadata.get("source_eligible") is True and metadata.get("eligibility_reasons") == []
          and all(metadata.get(k) == model[k] for k in ("opendbc_gitlink", "static_identity_sha256", "policy_identity_sha256")),
          "unknown/ineligible/nonvalidation original source")
  for source, route in {**training, **{s: m["route"] for s, m in original.items()}}.items():
    _need(source not in all_sources or all_sources[source] == route, "physical source assigned to different routes")
    all_sources[source] = route
  context = report.get("provenance_context")
  _need(isinstance(context, dict) and context == binding.get("provenance_context")
        and context.get("kind") == validation.PROVENANCE_KIND and context.get("model_sha256") == _hash(model)
        and context.get("split_sha256") == expected["split_sha256"]
        and context.get("static_identity_sha256") == expected["static_identity_sha256"]
        and _sha(context.get("training_receipt_sha256")) and _sha(context.get("evaluator_sha256")),
        "training/evaluator receipt or context differs")
  helpers = context.get("training_helpers_sha256")
  _need(isinstance(helpers, dict) and set(helpers) == set(validation.required_helpers(model["kind"]))
        and all(_sha(v) for v in helpers.values()) and context.get("validation_helpers_sha256") == helpers,
        "exact reused training/validation helper bindings required")
  refs = report.get("record_bindings")
  _need(isinstance(refs, list) and len(refs) == len(original), "source/content binding denominator differs")
  seen = set()
  for ref in refs:
    _need(isinstance(ref, dict) and ref.get("source_log_sha256") in original and ref["source_log_sha256"] not in seen
          and ref.get("sha256") == _hash({k: v for k, v in ref.items() if k != "sha256"})
          and all(ref.get(k) == v for k, v in context.items()) and all(_sha(ref.get(k)) for k in validation._REFERENCES),
          "original/view/helper source binding differs")
    if model["kind"] != moving.KIND:
      _need(_sha(ref.get("queue_arrays_sha256")) and _sha(ref.get("queue_metadata_sha256")), "categorical queue witness missing")
    seen.add(ref["source_log_sha256"])
  return binding, original


def _quantitative(model, report, limits, binding, originals):
  categorical = model["kind"] != moving.KIND
  _need(report.get("status") == "diagnostic" and report.get("window_population_known") is True
        and report.get("record_rejections") == [] and report.get("state_units") == UNITS
        and report.get("semantics") == SEMANTICS and limits.get("kind") == model["kind"]
        and (limits.get("categorical") is not None) == categorical, "report source/population/units/category unknown")
  _need(report.get("report_sha256") == _hash({k: v for k, v in report.items() if k != "report_sha256"}), "validation report body hash differs")
  rows = report.get("windows")
  _need(isinstance(rows, list) and rows, "retained validation transition population missing")
  identities, ids, clocks, statuses, labels = [], set(), set(), Counter(), Counter()
  confusion = Counter()
  for row in rows:
    _need(isinstance(row, dict) and _sha(row.get("id")) and row["id"] not in ids
          and row.get("source_log_sha256") in originals and row.get("route") == originals[row["source_log_sha256"]]["route"]
          and row.get("regime") == model["kind"], "validation transition/source identity differs")
    a, b = row.get("start_ns"), row.get("end_ns")
    _need(type(a) is int and type(b) is int and 0 <= a < b < 2**64, "unknown transition source clocks")
    clock = (row["source_log_sha256"], a, b)
    _need(clock not in clocks, "duplicate physical validation transition")
    clocks.add(clock)
    ids.add(row["id"])
    identities.append({k: row[k] for k in ("id", "route", "source_log_sha256", "start_ns", "end_ns")})
    status = row.get("status")
    _need(status in ("predicted", "unsupported", "quality_excluded"), "unknown validation outcome")
    statuses[status] += 1
    if status == "quality_excluded":
      if categorical:
        labels["quality_unknown"] += 1
      continue
    dt = row.get("actual_dt_s")
    _need(_number(dt) and .048 <= dt <= .052 and round(dt*1e9) == b-a, "completed48..52ms transition clock differs")
    initial = _state(row.get("prediction_inputs", {}).get("initial_state"))
    direction = "LEFT" if initial["yaw"] < 0 else "RIGHT" if initial["yaw"] > 0 else "zero"
    _need(row.get("measured_motion_direction") == direction, "measured-motion stratum differs")
    actual = _state(row.get("actual_endpoint"))
    if status == "predicted":
      predicted, residual = _state(row.get("predicted_endpoint")), row.get("endpoint_residual")
      _need(row.get("reasons") == [] and isinstance(residual, dict) and set(residual) == set(UNITS)
            and all(_number(residual[k]) and residual[k] == actual[k]-predicted[k] for k in UNITS), "endpoint residual/label differs")
    else:
      _need(row.get("predicted_endpoint") is None and row.get("endpoint_residual") is None, "unsupported prediction must remain unknown")
    c = row.get("category", {})
    if categorical:
      _need(isinstance(c, dict) and all(type(c.get(k)) is bool for k in ("initial_observed", "actual_next"))
            and row["prediction_inputs"].get("initial_hardware_standstill") == c["initial_observed"]
            and not (c["initial_observed"] and initial["v"] > .3) and not (c["actual_next"] and actual["v"] > .3),
            "independent categorical label unknown/incompatible")
      labels["known_label_denominator"] += 1
      if status == "predicted":
        _need(c.get("status") == "predicted" and type(c.get("predicted_next")) is bool
              and not (c["predicted_next"] and predicted["v"] > .3)
              and type(c.get("next_error")) is bool and c["next_error"] == (c["predicted_next"] != c["actual_next"])
              and type(c.get("transition_error")) is bool
              and c["transition_error"] == ((c["predicted_next"] != c["initial_observed"]) != (c["actual_next"] != c["initial_observed"])),
              "categorical prediction/error differs")
        labels.update(predicted=1, error_denominator=1, next_errors=int(c["next_error"]), transition_errors=int(c["transition_error"]))
        confusion[f'{str(c["actual_next"]).lower()}->{str(c["predicted_next"]).lower()}'] += 1
      else:
        _need(c.get("status") == "unsupported" and c.get("predicted_next") is None and c.get("next_error") is None
              and c.get("transition_error") is None, "unsupported category cannot become false")
        labels["unsupported"] += 1
    else:
      _need(c.get("status") == "not_predicted" and c.get("predicted_next") is None, "21-state category must stay not predicted")
      labels["not_predicted"] += 1
  _need(_hash(identities) == binding.get("validation_population_sha256"), "frozen validation population differs")
  counts = report.get("counts", {})
  _need(type(counts.get("attempted")) is int and counts["attempted"] == len(rows)
        and all(type(counts.get(k)) is int and counts[k] == statuses[k]
        for k in ("predicted", "unsupported", "quality_excluded")) and all(type(counts.get(k)) is int and counts[k] == 0 for k in
        ("metadata_rejected_records", "content_rejected_records", "collector_rejected_records")), "quantitative denominators differ")
  expected_labels = {k: labels[k] for k in ("predicted", "unsupported", "not_predicted", "quality_unknown",
                     "known_label_denominator", "error_denominator", "next_errors", "transition_errors")}
  category_counts = report.get("category_counts")
  _need(isinstance(category_counts, dict) and all(type(category_counts.get(k)) is int for k in expected_labels)
        and isinstance(category_counts.get("confusion"), dict)
        and all(type(v) is int for v in category_counts["confusion"].values())
        and category_counts == {**expected_labels, "confusion": dict(confusion)}, "categorical denominators differ")
  coverage = statuses["predicted"]/len(rows)
  _need(_number(report.get("coverage")) and report["coverage"] == coverage and len(rows) >= limits["minimum_attempted"]
        and statuses["predicted"] >= limits["minimum_predicted"] and len({r["route"] for r in rows}) >= limits["minimum_routes"],
        "insufficient declared validation denominator")
  groups = {"all": rows, **{"route:"+r: [w for w in rows if w["route"] == r] for r in originals_route_set(originals)},
            **{"direction:"+d: [w for w in rows if w.get("measured_motion_direction") == d]
               for d in limits["required_measured_directions"]}}
  summary = {}
  for name, group in groups.items():
    predicted_rows = [r for r in group if r["status"] == "predicted"]
    minimum = limits["minimum_predicted_per_route"] if name.startswith("route:") else limits["minimum_predicted_per_direction"]
    _need(group and predicted_rows and (name == "all" or len(predicted_rows) >= minimum), "missing declared route/direction prediction support")
    _need(len(predicted_rows)/len(group) >= limits["minimum_prediction_coverage"]
          and sum(r["status"] == "unsupported" for r in group)/len(group) <= limits["maximum_unsupported_fraction"]
          and sum(r["status"] == "quality_excluded" for r in group)/len(group) <= limits["maximum_quality_excluded_fraction"],
          "declared coverage/unsupported/quality limits failed")
    errors = {k: {"mae": float(np.mean([abs(r["endpoint_residual"][k]) for r in predicted_rows])),
                  "p95_abs": float(np.percentile([abs(r["endpoint_residual"][k]) for r in predicted_rows], 95))} for k in UNITS}
    _need(all(errors[k][stat] <= limits["state_error_limits"][k][stat] for k in UNITS for stat in ("mae", "p95_abs")),
          "declared conditional endpoint error limits failed")
    if categorical:
      c = limits["categorical"]
      _need((name != "all" or len(predicted_rows) >= c["minimum_predicted"])
            and np.mean([r["category"]["next_error"] for r in predicted_rows]) <= c["maximum_next_error_fraction"]
            and np.mean([r["category"]["transition_error"] for r in predicted_rows]) <= c["maximum_transition_error_fraction"],
            "declared categorical limits failed")
    summary[name] = {"attempted": len(group), "predicted": len(predicted_rows), "coverage": len(predicted_rows)/len(group),
                     "conditional_errors": errors}
  return summary


def originals_route_set(originals):
  return sorted({m["route"] for m in originals.values()})


def _admit(search, models, reports, split, cp, sp, spec, expected):
  validation._split(split)
  _need(isinstance(models, dict) and models and set(models) <= {"moving", "stopping"}
        and isinstance(reports, dict) and set(reports) == set(models), "exact used five-state models/reports required")
  _specification(spec, expected, set(models))
  _need(expected.get("split_sha256") == _hash(split) and expected.get("static_identity_sha256") == _hash({"cp": cp, "sp": sp})
        and isinstance(expected.get("models"), dict) and set(expected["models"]) == set(models)
        and _sha(expected.get("caller_verification_receipt_sha256")) and _sha(expected.get("selector_helper_sha256")),
        "trusted-caller receipt/static/split bindings missing")
  summaries, sources = {}, {}
  for alias, model in models.items():
    validation._model(model, split, cp, sp)
    _need(model["supported_cells"] > 0 and (model["kind"] == moving.KIND) == (alias == "moving")
          and all(model.get(k) == expected.get(k) for k in ("cohort", "policy_identity_sha256", "opendbc_gitlink"))
          and isinstance(model.get("cohort"), str) and model["cohort"] and _sha(model.get("policy_identity_sha256"))
          and isinstance(model.get("opendbc_gitlink"), str) and PIN.fullmatch(model["opendbc_gitlink"]), "unknown model source/support identity")
    binding, original = _sources(alias, model, reports[alias], expected, split, sources)
    summaries[alias] = _quantitative(model, reports[alias], spec["models"][alias], binding, original)
  _need(isinstance(search, dict) and isinstance(search.get("baseline"), dict) and isinstance(search.get("rows"), list)
        and 0 < len(search["rows"]) <= 64 and isinstance(expected.get("declared_grid"), list)
        and len(search["rows"]) == len(expected["declared_grid"]) and expected.get("grid_sha256") == _hash(expected["declared_grid"])
        and search["baseline"].get("parameters") is None and all(isinstance(r, dict) and r.get("parameters") == p
        for r, p in zip(search["rows"], expected["declared_grid"], strict=True)), "finite baseline/grid binding differs")
  _need(expected.get("search_sha256") == _hash(search), "source-bound search report differs")
  common, declared = search.get("common_window_ids"), expected.get("common_windows")
  _need(isinstance(common, list) and all(isinstance(v, str) and v for v in common) and len(common) == len(set(common))
        and common == sorted(common) and len(common) >= joint.long.research.MIN_WINDOWS and isinstance(declared, dict)
        and set(declared) == set(common) and search.get("attempted_windows") == len(common), "fixed common VALIDATION population differs")
  for value in declared.values():
    _need(isinstance(value, dict) and value.get("route") in split["validation"] and value.get("model_key") in models
          and value.get("regime") == ("moving" if value["model_key"] == "moving" else "stopping")
          and _sha(value.get("reference_sha256")) and _sha(value.get("source_binding_sha256"))
          and value.get("source_log_sha256") in expected["models"][value["model_key"]]["validation_sources"]
          and value["route"] == expected["models"][value["model_key"]]["validation_sources"][value["source_log_sha256"]]["route"],
          "unknown/nonvalidation common window source")
  _need({v["model_key"] for v in declared.values()} == set(models), "unused/missing model population")
  return common, declared, summaries


def _metric_shape(value):
  """Finite metric scalars only; unscored trace/GT arrays are not promoted to metrics."""
  if isinstance(value, dict):
    for v in value.values():
      _metric_shape(v)
  elif isinstance(value, (list, tuple)):
    for v in value:
      _metric_shape(v)
  elif isinstance(value, (float, np.floating)):
    _need(np.isfinite(value), "nonfinite metric")
  elif isinstance(value, np.ndarray):
    _need(value.dtype.kind != "O" and (value.dtype.kind not in "fci" or np.isfinite(value).all()), "nonfinite metric array")


def _row(row, baseline, common, declared, models):
  _need(isinstance(row, dict) and row.get("common_window_ids") == common and row.get("coverage") == 1.
        and row.get("failures") == [] and isinstance(row.get("results"), dict) and set(row["results"]) == set(common),
        "incomplete/mismatched common row population")
  reasons = row.get("selection_rejections")
  _need(isinstance(reasons, list) and all(isinstance(v, str) for v in reasons), "unknown row rejection schema")
  reasons = list(reasons)  # Preserve the immutable source report; never append native reasons into it.
  stops = [key for key in common if declared[key]["regime"] == "stopping"]
  allowed = set(RETAINED_ASSUMPTIONS)
  allowed.add("low_speed_stop_hold_plant_missing")  # moving: stopping unassessed; stopping: admitted31 below
  for key in common:
    result, source = row["results"][key], declared[key]
    _need(isinstance(result, dict) and result.get("status") == "complete" and result.get("censor_t") is None
          and result.get("reason") is None and type(result.get("attempted_steps")) is int and result["attempted_steps"] > 0
          and result.get("supported_steps") == result["attempted_steps"] and result["attempted_steps"] % 5 == 0
          and result.get("macro_indices") == list(range(0, result["attempted_steps"], 5))
          and result.get("fixed_reference_sha256") == source["reference_sha256"]
          and _hash(result.get("source_binding")) == source["source_binding_sha256"], "incomplete/unsupported/source-mismatched native row")
    proof = result["source_binding"]
    _need(isinstance(proof, dict), "upstream native source proof mapping required")
    selected_model = models[source["model_key"]]
    model_key = "model" if source["regime"] == "moving" else (
      "low_speed_model" if selected_model["kind"] == category.KIND else "approach_stop_model")
    _need(proof.get(model_key) == _hash(selected_model), "quantitative evidence differs from upstream consumed model")
    rr = result.get("selection_rejections")
    _need(isinstance(rr, list) and all(isinstance(v, str) for v in rr), "unknown native rejection schema")
    result_allowed = set(RETAINED_ASSUMPTIONS) | {"low_speed_stop_hold_plant_missing"}
    if source["regime"] == "stopping":
      result_allowed |= STOP_ASSUMPTIONS
      model = models["stopping"]
      _need(result.get("categorical_plant_kind") == model["kind"]
            and result.get("categorical_plant_model_sha256") == model["model_sha256"], "bound31 stopping model missing")
      if model["kind"] == category.KIND:
        speed = np.asarray(result.get("trace", {}).get("v"), float)
        _need(speed.ndim == 1 and len(speed) == result["attempted_steps"] and np.isfinite(speed).all()
              and (speed >= 0).all() and (speed <= category.LOW_SPEED_MAX_MPS).all(), "low-speed domain limitation not established")
        result_allowed.add("high_speed_to_low_speed_categorical_transition_unimplemented")
      else:
        result_allowed.add("fullrange_categorical_contact_response_unvalidated")
      stop_result = result.get("stopping_result")
      target = stop_result.get("model_target") if isinstance(stop_result, dict) else None
      _need(isinstance(target, dict) and all(_number(target.get(k)) for k in ("lower_m", "upper_m"))
            and _number(stop_result.get("distance_from_request_m")) and stop_result.get("target_feasible") is True
            and target["lower_m"] <= stop_result["distance_from_request_m"] <= target["upper_m"], "model stopping target feasibility unknown")
    _need(not set(rr)-result_allowed, "unclassified/hard native rejection: "+",".join(sorted(set(rr)-result_allowed)))
    allowed |= result_allowed
    reasons += rr
  _need(not set(reasons)-allowed, "unclassified/hard rejection: "+",".join(sorted(set(reasons)-allowed)))
  metrics, before = row.get("metrics"), baseline.get("metrics")
  _need(isinstance(metrics, dict) and isinstance(before, dict), "joint metrics missing")
  _metric_shape(metrics)
  rejections = joint.constraints(metrics, before)
  if stops:
    rejections += runner._stop_constraints(row["results"], baseline["results"], stops)
  _need(not rejections, ",".join(sorted(set(rejections))))
  objective = (runner.stop._objective if stops else joint._objective)(metrics, before)
  _need(_number(objective) and row.get("joint_target_objective") == objective, "joint objective missing/differs")
  return objective


def _leaves(parameters):
  result = {}
  if isinstance(parameters, dict):
    for group, values in parameters.items():
      if not isinstance(values, dict):
        continue
      for name, value in values.items():
        for i, v in enumerate(value if isinstance(value, list) else [value]):
          if _number(v):
            result[f"{group}.{name}"+(f"[{i}]" if isinstance(value, list) else "")] = float(v)
  return result


def select_diagnostic(search_report, *, models, validation_reports, split, cp, sp, specification, expected_bindings):
  """Quantitatively conditional diagnostics only; never accept a caller success boolean."""
  source_rows = search_report.get("rows", []) if isinstance(search_report, dict) else []
  result = {"status": "unavailable", "ranking_status": "unavailable", "diagnostic_winner": None,
    "rows": [{"source_row": r, "diagnostic_rejections": []} for r in source_rows] if isinstance(source_rows, list) else [],
    "baseline": search_report.get("baseline") if isinstance(search_report, dict) else None,
    "selection": None, "full_controller_candidate": None, "production_enabled": False,
    "predictive_validated": False, "caller_authentication_verified": False,
    "authentication_boundary":
      "Conditional on independently authenticated trusted-caller receipts/expected bindings; this pure core performs no authentication",
    "limitations": ["Completed50ms conditional residuals are not start-time forecasts or closed-loop physical validation",
                    "All original full-tune rejections/assumptions remain visible and unresolved"], "evidence_rejections": []}
  try:
    common, declared, quantitative = _admit(search_report, models, validation_reports, split, cp, sp, specification, expected_bindings)
  except (SelectionError, validation.ValidationError, validation.base.FitError, KeyError, TypeError, ValueError, AttributeError) as error:
    result["evidence_rejections"] = [str(error)]
    for row in result["rows"]:
      row["diagnostic_rejections"] = ["quantitative_evidence_unavailable"]
    return result
  result.update(evidence_integrity="conditionally_admitted", quantitative=quantitative,
    expected_bindings_sha256=_hash(expected_bindings), specification_sha256=_hash(specification), common_window_ids=list(common),
    stopping_assessment="conditional_supported" if "stopping" in models else "unassessed")
  baseline = search_report["baseline"]
  try:
    before = _row(baseline, baseline, common, declared, models)
  except (SelectionError, KeyError, TypeError, ValueError, AttributeError) as error:
    result["evidence_rejections"] = ["baseline_inadmissible:"+str(error)]
    return result
  bounds = {}
  for parameter in expected_bindings["declared_grid"]:
    for key, value in _leaves(parameter).items():
      bounds.setdefault(key, []).append(value)
  ranked = []
  for index, entry in enumerate(result["rows"]):
    row = entry["source_row"]
    entry.update(row_index=index, parameters=deepcopy(row.get("parameters")), row_sha256=_hash(row),
      finite_grid_boundary={k: {"minimum": v == min(bounds[k]), "maximum": v == max(bounds[k])} for k, v in _leaves(row.get("parameters")).items()},
      boundary_basis="Declared finite grid, including retained rejected rows; no convergence claim")
    try:
      objective = _row(row, baseline, common, declared, models)
      _need(objective < before, "no_strict_joint_objective_improvement")
      entry["joint_target_objective"] = objective
      ranked.append(entry)
    except (SelectionError, KeyError, TypeError, ValueError, AttributeError) as error:
      entry["diagnostic_rejections"] = [str(error)]
  ranked.sort(key=lambda r: (r["joint_target_objective"], json.dumps(r["parameters"], sort_keys=True, separators=(",", ":")), r["row_index"]))
  for rank, row in enumerate(ranked, 1):
    row["diagnostic_rank"] = rank
  result.update(status="conditional_diagnostic" if ranked else "no_supported_improvement",
                ranking_status="conditional_diagnostic", diagnostic_winner=ranked[0] if ranked else None)
  return result
