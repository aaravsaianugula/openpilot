"""Synthetic admission/selection contracts only; no fitted or road-performance evidence."""
from copy import deepcopy

import numpy as np
import pytest

import controller_multirate_joint as moving
import controller_multirate_joint_stop_hold as category
import controller_multirate_validation as validation
import controller_native_joint as joint
import controller_native_two_way as runner
import controller_predictive_selection as select
from controller_multirate_lateral import identity_hash as H

UNITS = {"v": "m/s", "a": "m/s^2", "angle": "degrees", "yaw": "rad/s", "vendor_rate": "unsigned degrees/s"}
ASSUMPTIONS = ["unvalidated_100hz_zoh", "unvalidated_joint_median_coherence", "unvalidated_v_a_kinematics",
  "unvalidated_markov_command_history_sufficiency_for_actuator_latency", "native_initialization_and_apply_phase_uncertain",
  "frozen_learner_and_perception", "low_speed_stop_hold_plant_missing", "unvalidated_held_initial_state_during_cold_prefix",
  "unvalidated_sensor_and_state_publication_sampling", "publication_schedule_caller_authenticated_not_receive_clock",
  "CAN_apply_is_command_proxy_not_delivered_torque"]
ROW_LABELS = ["no_validated_two_way_stop_hold_or_coherence", "diagnostic_only", "joint_plant_predictive_validation_unavailable"]


def seal(value, key="sha256"):
  value[key] = H({k: v for k, v in value.items() if k != key})
  return value


def model(alias, split, cp, sp, *, approach=False):
  categorical = alias == "stopping"
  n = 31 if categorical else 21
  lo, hi = np.full(n, -100.), np.full(n, 100.)
  if categorical:
    lo[-10:] = hi[-10:] = 1.
  cell = {"feature_min": lo.tolist(), "feature_max": hi.tolist(), "joint_delta_median": [0.] * 5,
          "samples": 200, "routes": split["train"], "supported": True}
  value = {"kind": (category.APPROACH_KIND if approach else category.KIND) if categorical else moving.KIND, "split": deepcopy(split),
    "cohort": "synthetic-source-bound", "opendbc_gitlink": "a" * 40, "static_identity_sha256": H({"cp": cp, "sp": sp}),
    "policy_identity_sha256": "b" * 64, "source_eligible": True, "source_pin_verified": True,
    "static_policy_timing_verified": True, "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5,
    "features": list(category.FEATURES if categorical else moving.FEATURES), "min_samples": 100, "min_routes": 2,
    "feature_min": lo.tolist(), "feature_max": hi.tolist(), "edges": [[] for _ in range(n)], "supported_cells": 1,
    "record_hashes": ["1" * 64, "2" * 64],
    "input_eligibility": [{"source_log_sha256": c * 64, "source_eligible": True, "reasons": []} for c in ("c", "d")],
    "window_evidence": [{"source_log_sha256": c * 64, "route": r} for c, r in zip(("c", "d"), split["train"], strict=True)],
    "exclusions": [], "fixture_only": True, "selection_eligible": False, "production_enabled": False}
  key = ":".join(["0"] * n)
  if categorical:
    value.update(patterns=["0|1111111111"], native_policy_source_sha256=category.NATIVE_POLICY_SHA,
                 card_source_sha256=category.CARD_SHA)
    value.update(initial_speed_domain="retained_train_global_and_cell_bounds") if approach else value.update(low_speed_max_initial_mps=3.)
    cell["next_hardware_standstill_values"] = [False]
    key = "0|1111111111|" + key
  value["cells"] = {key: cell}
  return seal(value, "model_sha256")


def report(alias, model, split, cp, sp):
  cat = alias == "stopping"
  helpers = dict.fromkeys(validation.required_helpers(model["kind"]), "4" * 64)
  context = {"kind": validation.PROVENANCE_KIND, "model_sha256": H(model), "split_sha256": H(split),
    "static_identity_sha256": H({"cp": cp, "sp": sp}), "training_receipt_sha256": "5" * 64,
    "training_helpers_sha256": helpers, "validation_helpers_sha256": helpers, "evaluator_sha256": "6" * 64}
  bindings = []
  for c in ("e", "f"):
    body = {**context, **dict.fromkeys(validation._REFERENCES, "7" * 64), "source_log_sha256": c * 64}
    if cat:
      body.update(queue_arrays_sha256="8" * 64, queue_metadata_sha256="9" * 64)
    bindings.append(seal(body))
  rows = []
  for i in range(4):
    predicted = dict(zip(moving.STATE_KEYS, [2.5 if cat else 5., 0., .2, -.1 if i % 2 else .1, 1.], strict=True))
    actual = {k: v + .05 for k, v in predicted.items()}
    rows.append({"id": H([alias, i]), "route": split["validation"][i % 2], "source_log_sha256": ("e" if i % 2 == 0 else "f") * 64,
      "start_ns": 100_000_000 + i * 50_000_000, "end_ns": 150_000_000 + i * 50_000_000, "regime": model["kind"],
      "status": "predicted", "actual_dt_s": .05, "measured_motion_direction": "LEFT" if i % 2 else "RIGHT",
      "prediction_inputs": {"initial_state": predicted, **({"initial_hardware_standstill": False} if cat else {})},
      "predicted_endpoint": predicted, "actual_endpoint": actual,
      "endpoint_residual": {k: actual[k] - predicted[k] for k in predicted}, "reasons": [],
      "category": {"status": "predicted", "initial_observed": False, "actual_next": False, "predicted_next": False,
                   "next_error": False, "transition_error": False} if cat else {"status": "not_predicted", "predicted_next": None}})
  value = {"status": "diagnostic", "model_sha256": H(model), "claimed_model_body_sha256": model["model_sha256"],
    "split_sha256": H(split), "static_identity_sha256": H({"cp": cp, "sp": sp}), "state_units": dict(UNITS),
    "semantics": "Completed50ms conditional transition given observed commands/reports; not a start-time forecast",
    "provenance_context": context, "record_bindings": bindings, "window_population_known": True, "record_rejections": [],
    "windows": rows, "counts": {"attempted": 4, "predicted": 4, "unsupported": 0, "quality_excluded": 0,
      "metadata_rejected_records": 0, "content_rejected_records": 0, "collector_rejected_records": 0}, "coverage": 1.,
    "category_counts": {"predicted": 4 if cat else 0, "unsupported": 0, "not_predicted": 0 if cat else 4, "quality_unknown": 0,
      "known_label_denominator": 4 if cat else 0, "error_denominator": 4 if cat else 0, "next_errors": 0,
      "transition_errors": 0, "confusion": {"false->false": 4} if cat else {}},
    "predictive_validated": False, "caller_authentication_verified": False, "selection": None, "production_enabled": False}
  return seal(value, "report_sha256")


def fixture(*, stopping=False, approach=False, errors=(.05, .05), factors=(2., 3.)):
  stopping = stopping or approach
  split = {"train": ["train-a", "train-b"], "validation": ["val-a", "val-b"], "test": ["test-reserved"]}
  cp, sp = {"fixture_only": True}, {"fixture_only": True}
  models = {key: model(key, split, cp, sp, approach=approach and key == "stopping")
            for key in (("moving", "stopping") if stopping else ("moving",))}
  reports = {key: report(key, m, split, cp, sp) for key, m in models.items()}
  common = [f"window-{i:02}" for i in range(20)]
  def proof(key, is_stop):
    return {"fixture_only": True, "id": key, "model": H(models["moving"]),
            **({"approach_stop_model" if approach else "low_speed_model": H(models["stopping"])} if is_stop else {})}
  declarations = {key: {"route": split["validation"][i % 2], "source_log_sha256": ("e" if i % 2 == 0 else "f") * 64,
    "model_key": "stopping" if stopping and i < 2 else "moving", "regime": "stopping" if stopping and i < 2 else "moving",
    "reference_sha256": H([key, "fixed-reference"]), "source_binding_sha256": H(proof(key, stopping and i < 2))}
    for i, key in enumerate(common)}
  reference = np.array([-.002, -.004, -.004, -.002, 0., .002, .004, .004, .002])
  target = np.array([-.06] * 4 + [.06] * 5)
  nodes = np.arange(0, 45, 5)
  def row(error, parameters):
    results = {}
    for key in common:
      decl = declarations[key]
      trace = {"a": np.repeat(target + error, 5), "physical_reference": np.repeat(target, 5),
        "accel_request": np.repeat(target, 5), "v": np.full(45, 2.5 if decl["regime"] == "stopping" else 10.),
        "fixed_speed": np.full(45, 2.5 if decl["regime"] == "stopping" else 10.),
        "fixed_path": np.zeros((45, 2)), "path": np.zeros((45, 2))}
      item = {"window_id": key, "status": "complete", "supported_steps": 45, "attempted_steps": 45,
        "macro_indices": nodes.tolist(), "trace": trace, "source_binding": proof(key, decl["regime"] == "stopping"),
        "fixed_reference_sha256": decl["reference_sha256"], "selection_rejections": list(ASSUMPTIONS),
        "turns": runner.turns.window_metrics(reference, reference * 100 + error, reference * 100, np.zeros(9), reference_prefix=(.05, 0.))}
      if decl["regime"] == "stopping":
        item.update(categorical_plant_kind=models["stopping"]["kind"], categorical_plant_model_sha256=models["stopping"]["model_sha256"],
          stopping_result={"distance_from_request_m": 9.+10*error, "time_from_request_s": 1.+error,
            "target_feasible": True, "model_target": {"distance_from_request_m": 9., "lower_m": 8., "upper_m": 11.,
              "request_t": .6, "capture_t": .5, "path_sha256": "1" * 64, "uncertainty_source_sha256": "2" * 64}},
          hold_motion={"reason": None, "supported_zero_speed_s": .5, "angle_excursion_deg": error, "yaw_abs_max_rad_s": error*.1})
        item["selection_rejections"] += ["contact_and_hold_unvalidated", "independent_categorical_state_dynamics_unvalidated",
          "frozen_OEM_cruise_standstill_unverified", "fullrange_categorical_contact_response_unvalidated" if approach else
          "high_speed_to_low_speed_categorical_transition_unimplemented"]
      results[key] = item
    metrics = runner.summarize(results, common)
    if stopping:
      metrics.update(runner._stop_summary(results, common[:2]))
    return {"parameters": parameters, "results": results, "failures": [], "metrics": metrics,
      "common_window_ids": common, "coverage": 1., "selection_rejections": list(ASSUMPTIONS) + ROW_LABELS}
  baseline = row(.1, None)
  rows = [row(error, {"long": None, "lateral": {"factor": factor, "friction": .1}})
          for error, factor in zip(errors, factors, strict=True)]
  for value in [baseline, *rows]:
    value["joint_target_objective"] = (runner.stop._objective if stopping else joint._objective)(value["metrics"], baseline["metrics"])
  search = {"baseline": baseline, "rows": rows, "common_window_ids": common, "attempted_windows": 20,
    "selection": None, "complete_controller_candidate": None, "production_enabled": False, "ranking_status": "unavailable"}
  specification = {"kind": "predeclared_five_state_diagnostic_limits_v1", "predeclaration_receipt_sha256": "3" * 64,
    "state_units": dict(UNITS), "semantics": reports["moving"]["semantics"], "models": {key: {"kind": m["kind"],
      "minimum_attempted": 4, "minimum_predicted": 4, "minimum_routes": 2, "minimum_prediction_coverage": 1.,
      "maximum_unsupported_fraction": 0., "maximum_quality_excluded_fraction": 0., "minimum_predicted_per_route": 1,
      "required_measured_directions": ["LEFT", "RIGHT"], "minimum_predicted_per_direction": 1,
      "state_error_limits": {k: {"mae": .2, "p95_abs": .2} for k in moving.STATE_KEYS},
      "categorical": {"minimum_predicted": 4, "maximum_next_error_fraction": 0., "maximum_transition_error_fraction": 0.} if key == "stopping" else None}
      for key, m in models.items()}}
  seal(specification)
  expected = {"kind": "trusted_caller_predictive_bindings_v1", "specification_sha256": H(specification),
    "predeclaration_receipt_sha256": specification["predeclaration_receipt_sha256"], "split_sha256": H(split),
    "static_identity_sha256": H({"cp": cp, "sp": sp}), "cohort": "synthetic-source-bound", "policy_identity_sha256": "b" * 64,
    "opendbc_gitlink": "a" * 40, "caller_verification_receipt_sha256": "4" * 64, "selector_helper_sha256": "5" * 64,
    "models": {}, "common_windows": declarations, "declared_grid": [r["parameters"] for r in rows]}
  for key, m in models.items():
    expected["models"][key] = {"model_sha256": H(m), "validation_report_sha256": reports[key]["report_sha256"],
      "validation_receipt_sha256": "6" * 64, "provenance_context": deepcopy(reports[key]["provenance_context"]),
      "validation_population_sha256": H([{k: w[k] for k in ("id", "route", "source_log_sha256", "start_ns", "end_ns")} for w in reports[key]["windows"]]),
      "training_sources": {c * 64: r for c, r in zip(("c", "d"), split["train"], strict=True)},
      "validation_sources": {c * 64: {"route": r, "partition": "validation", "source_eligible": True, "eligibility_reasons": [],
        "opendbc_gitlink": "a" * 40, "static_identity_sha256": m["static_identity_sha256"], "policy_identity_sha256": m["policy_identity_sha256"]}
        for c, r in zip(("e", "f"), split["validation"], strict=True)}}
  args = {"models": models, "validation_reports": reports, "split": split, "cp": cp, "sp": sp,
          "specification": specification, "expected_bindings": expected}
  refresh(search, args)
  return search, args


def refresh(search, args):
  e = args["expected_bindings"]
  e["search_sha256"], e["grid_sha256"] = H(search), H(e["declared_grid"])


def test_conditional_winner_strict_improvement_tie_boundary_and_immutable_inputs(monkeypatch):
  search, args = fixture()
  before = H([search, args])
  monkeypatch.setattr(moving, "fit", lambda *a, **k: pytest.fail("fit called"))
  monkeypatch.setattr(category, "fit", lambda *a, **k: pytest.fail("fit called"))
  result = select.select_diagnostic(search, **args)
  assert H([search, args]) == before
  assert result["diagnostic_winner"]["parameters"]["lateral"]["factor"] == 2.
  assert [r["diagnostic_rank"] for r in result["rows"]] == [1, 2]
  assert result["rows"][0]["finite_grid_boundary"]["lateral.factor"] == {"minimum": True, "maximum": False}
  assert result["selection"] is None and result["full_controller_candidate"] is None and not result["production_enabled"]
  assert not result["predictive_validated"] and not result["caller_authentication_verified"]
  assert result["stopping_assessment"] == "unassessed"
  assert result["rows"][0]["source_row"]["selection_rejections"] == search["rows"][0]["selection_rejections"]


def test_success_labels_without_quantitative_spec_or_expected_context_never_rank():
  search, args = fixture()
  args["validation_reports"]["moving"]["predictive_validated"] = True
  args["expected_bindings"] = None
  result = select.select_diagnostic(search, **args)
  assert result["status"] == "unavailable" and result["diagnostic_winner"] is None
  assert len(result["rows"]) == len(search["rows"])


@pytest.mark.parametrize("fault", ["spec_hash", "model", "helper", "split", "source", "original_unknown", "pin_unknown",
                                  "test_partition", "receipt", "population"])
def test_binding_and_source_mismatch_refuses_without_flag_promotion(fault):
  search, args = fixture()
  e, m, r = args["expected_bindings"], args["models"]["moving"], args["validation_reports"]["moving"]
  if fault == "spec_hash":
    e["specification_sha256"] = "0" * 64
  elif fault == "model":
    e["models"]["moving"]["model_sha256"] = "0" * 64
  elif fault == "helper":
    e["models"]["moving"]["provenance_context"]["validation_helpers_sha256"]["controller_multirate_joint.py"] = "0" * 64
  elif fault == "split":
    args["split"]["validation"].append("other")
  elif fault == "source":
    e["models"]["moving"]["training_sources"]["c" * 64] = "train-b"
  elif fault == "original_unknown":
    e["models"]["moving"]["validation_sources"]["e" * 64]["source_eligible"] = False
  elif fault == "pin_unknown":
    m["opendbc_gitlink"] = None
    seal(m, "model_sha256")
  elif fault == "test_partition":
    e["common_windows"]["window-00"]["route"] = "test-reserved"
  elif fault == "receipt":
    e["models"]["moving"]["validation_receipt_sha256"] = None
  else:
    e["models"]["moving"]["validation_population_sha256"] = "0" * 64
  r["predictive_validated"] = True
  seal(r, "report_sha256")
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


@pytest.mark.parametrize("fault", ["units", "missing_limit", "nonfinite", "denominator", "summary_only_lie", "low_coverage"])
def test_quantitative_spec_and_actual_retained_denominators_are_required(fault):
  search, args = fixture()
  spec, r = args["specification"], args["validation_reports"]["moving"]
  if fault == "units":
    spec["state_units"]["yaw"] = "degrees/s"
  elif fault == "missing_limit":
    del spec["models"]["moving"]["state_error_limits"]["a"]
  elif fault == "nonfinite":
    spec["models"]["moving"]["state_error_limits"]["a"]["mae"] = float("nan")
  elif fault == "denominator":
    r["counts"]["predicted"] = 400
  elif fault == "summary_only_lie":
    r["windows"][0]["actual_endpoint"]["a"] += 1.
  else:
    r["windows"][0].update(status="unsupported", predicted_endpoint=None, endpoint_residual=None)
  if fault != "nonfinite":
    seal(spec)
    args["expected_bindings"]["specification_sha256"] = H(spec)
  seal(r, "report_sha256")
  args["expected_bindings"]["models"]["moving"]["validation_report_sha256"] = r["report_sha256"]
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


@pytest.mark.parametrize("fault", ["unknown_reason", "incomplete", "common", "reference", "right_phase", "left_phase",
                                  "braking", "acceleration", "jerk", "saturation", "path", "no_improvement"])
def test_failed_rows_retained_with_unchanged_joint_constraints(fault):
  search, args = fixture(errors=(.05,), factors=(2.,))
  row = search["rows"][0]
  if fault == "unknown_reason":
    row["selection_rejections"].append("unvalidated_new_unknown_contract")
  elif fault == "incomplete":
    row["results"]["window-00"]["status"] = "unsupported"
    row["failures"] = [{"window_id": "window-00", "reason": "missing_cell"}]
  elif fault == "common":
    row["common_window_ids"] = row["common_window_ids"][:-1]
  elif fault == "reference":
    row["metrics"]["fixed_reference_by_window"]["window-00"] = "0" * 64
  elif fault == "right_phase":
    row["metrics"]["turns"]["phases"]["right"]["entry"]["mae"] = .9
  elif fault == "left_phase":
    row["metrics"]["turns"]["phases"]["left"]["exit"]["mae"] = .9
  elif fault == "braking":
    row["metrics"]["long"]["tracking"]["braking"]["mae"] = .9
  elif fault == "acceleration":
    row["metrics"]["long"]["tracking"]["acceleration"]["mae"] = .9
  elif fault == "jerk":
    row["metrics"]["long"]["jerk"]["p95"] = 4.
  elif fault == "saturation":
    row["metrics"]["turns"]["tracking"]["left"]["saturation_fraction"] = .2
  elif fault == "path":
    row["metrics"]["path_p95"] = 2.
  else:
    row["metrics"] = deepcopy(search["baseline"]["metrics"])
  row["joint_target_objective"] = joint._objective(row["metrics"], search["baseline"]["metrics"])
  refresh(search, args)
  result = select.select_diagnostic(search, **args)
  assert len(result["rows"]) == 1 and result["rows"][0]["diagnostic_rejections"]
  assert result["diagnostic_winner"] is None
  assert result["rows"][0]["source_row"]["failures"] == row["failures"]


def test_bound_low_model_and_unchanged_stop_hold_target_gates():
  search, args = fixture(stopping=True, errors=(.05,), factors=(2.,))
  good = select.select_diagnostic(search, **args)
  assert good["diagnostic_winner"] is not None and good["stopping_assessment"] == "conditional_supported"
  search["rows"][0]["results"]["window-00"]["stopping_result"]["target_feasible"] = False
  refresh(search, args)
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


@pytest.mark.parametrize("fault", ["unknown_category", "source_false", "outside_low_domain"])
def test_categorical_unknown_source_and_domain_cannot_be_reclassified_as_assumptions(fault):
  search, args = fixture(stopping=True, errors=(.05,), factors=(2.,))
  r = args["validation_reports"]["stopping"]
  if fault == "unknown_category":
    r["windows"][0]["category"]["actual_next"] = None
    seal(r, "report_sha256")
    args["expected_bindings"]["models"]["stopping"]["validation_report_sha256"] = r["report_sha256"]
  elif fault == "source_false":
    args["models"]["stopping"]["source_eligible"] = False
    seal(args["models"]["stopping"], "model_sha256")
  else:
    search["rows"][0]["results"]["window-00"]["trace"]["v"][0] = 3.1
    refresh(search, args)
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


@pytest.mark.parametrize("fault", ["model_header", "split_header", "static_header", "source_route", "category_count_bool", "rejection_count_bool"])
def test_independent_report_headers_and_strict_denominator_types(fault):
  search, args = fixture(errors=(.05,), factors=(2.,))
  r = args["validation_reports"]["moving"]
  if fault in ("model_header", "split_header", "static_header"):
    key = {"model_header": "model_sha256", "split_header": "split_sha256", "static_header": "static_identity_sha256"}[fault]
    r[key] = "0" * 64
  elif fault == "source_route":
    args["expected_bindings"]["common_windows"]["window-00"]["route"] = "val-b"
  elif fault == "category_count_bool":
    r["category_counts"]["next_errors"] = False
  else:
    r["counts"]["metadata_rejected_records"] = False
  seal(r, "report_sha256")
  args["expected_bindings"]["models"]["moving"]["validation_report_sha256"] = r["report_sha256"]
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


@pytest.mark.parametrize("literal", ["contact_and_hold_unvalidated", "high_speed_to_low_speed_categorical_transition_unimplemented",
                                     "fullrange_categorical_contact_response_unvalidated"])
def test_stop_only_literal_on_moving_result_remains_hard_even_in_mixed_search(literal):
  search, args = fixture(stopping=True, errors=(.05,), factors=(2.,))
  search["rows"][0]["results"]["window-19"]["selection_rejections"].append(literal)
  refresh(search, args)
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


def test_bound_approach_model_uses_exact_selected_kind_without_low_domain_fallback():
  search, args = fixture(approach=True, errors=(.05,), factors=(2.,))
  for row in [search["baseline"], *search["rows"]]:
    for key in ("window-00", "window-01"):
      row["results"][key]["trace"]["v"][:] = 4.
  refresh(search, args)
  result = select.select_diagnostic(search, **args)
  assert result["diagnostic_winner"] is not None and result["stopping_assessment"] == "conditional_supported"
  search["rows"][0]["results"]["window-00"]["selection_rejections"].append("high_speed_to_low_speed_categorical_transition_unimplemented")
  refresh(search, args)
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None


def test_explicit_conditional_error_limit_and_category_error_fail_despite_success_labels():
  search, args = fixture(stopping=True, errors=(.05,), factors=(2.,))
  report = args["validation_reports"]["stopping"]
  window = report["windows"][0]
  window["actual_endpoint"]["a"] += 1.
  window["endpoint_residual"]["a"] = window["actual_endpoint"]["a"] - window["predicted_endpoint"]["a"]
  report["predictive_validated"] = True
  seal(report, "report_sha256")
  args["expected_bindings"]["models"]["stopping"]["validation_report_sha256"] = report["report_sha256"]
  result = select.select_diagnostic(search, **args)
  assert "endpoint error limits failed" in result["evidence_rejections"][0]
  assert result["diagnostic_winner"] is None and not result["predictive_validated"]
  window["actual_endpoint"]["a"] -= 1.
  window["endpoint_residual"]["a"] = window["actual_endpoint"]["a"] - window["predicted_endpoint"]["a"]
  # Independent observed hardware category disagrees; it is not inferred from zero speed.
  window["actual_endpoint"]["v"] = window["predicted_endpoint"]["v"] = .25
  window["endpoint_residual"]["v"] = 0.
  window["category"].update(actual_next=True, next_error=True, transition_error=True)
  report["category_counts"].update(next_errors=1, transition_errors=1, confusion={"false->false": 3, "true->false": 1})
  seal(report, "report_sha256")
  args["expected_bindings"]["models"]["stopping"]["validation_report_sha256"] = report["report_sha256"]
  result = select.select_diagnostic(search, **args)
  assert "categorical limits failed" in result["evidence_rejections"][0]
  assert result["diagnostic_winner"] is None


def test_core_performs_no_file_io_and_repeated_calls_preserve_whole_inputs(monkeypatch):
  search, args = fixture()
  before = H([search, args])
  monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("pure selector opened a file"))
  first = select.select_diagnostic(search, **args)
  second = select.select_diagnostic(search, **args)
  assert H([search, args]) == before
  assert H(first) == H(second) and first["diagnostic_winner"] is not None


def test_bad_baseline_never_allows_an_otherwise_improving_candidate():
  search, args = fixture()
  search["baseline"]["results"]["window-00"]["selection_rejections"].append("actual_source_mismatch")
  refresh(search, args)
  result = select.select_diagnostic(search, **args)
  assert result["diagnostic_winner"] is None and len(result["rows"]) == 2
  assert "baseline_inadmissible" in result["evidence_rejections"][0]
  assert search["baseline"]["results"]["window-00"]["selection_rejections"][-1] == "actual_source_mismatch"


@pytest.mark.parametrize("stopping", [False, True])
def test_quantitatively_admitted_model_must_match_actual_upstream_consumed_model(stopping):
  search, args = fixture(stopping=stopping, errors=(.05,), factors=(2.,))
  for row in [search["baseline"], *search["rows"]]:
    proof = row["results"]["window-00"]["source_binding"]
    proof["low_speed_model" if stopping else "model"] = "0" * 64
  # Even self-consistent, separately supplied proof digests cannot excuse a different model.
  args["expected_bindings"]["common_windows"]["window-00"]["source_binding_sha256"] = H(proof)
  refresh(search, args)
  assert select.select_diagnostic(search, **args)["diagnostic_winner"] is None
