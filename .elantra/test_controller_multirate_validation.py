"""Synthetic validation contracts only; no actual held-out or physical evidence."""
from copy import deepcopy

import numpy as np
import pytest

import controller_multirate_joint as moving
import controller_multirate_joint_stop_hold as category
import controller_multirate_validation as check
from test_controller_multirate_joint import SPLIT as ORIGINAL_SPLIT, ROUTES, Unreadable
from test_controller_multirate_joint_stop_hold import record, approach_record, bind, CP, SP


SPLIT = {**deepcopy(ORIGINAL_SPLIT), "validation": [*ORIGINAL_SPLIT["validation"], "000001c0--abcdef0123"]}


def rebind(row):
  bind(row)
  row["view_receipt"]["input_source_eligible"] = row["motion_metadata"]["source_eligible"]
  row["view_receipt"]["input_proof"]["split_sha256"] = moving.identity_hash(SPLIT)


def admitted_record(route, kind, macros=3):
  row = approach_record(route) if kind == "approach" else record(route, mode="hold" if kind == "low" else "active", macros=macros)
  if kind == "approach" and macros == 3:
    # Keep the original shape/clock contracts, only retain a short prefix.
    end = 1_350_000_000
    for name in ("carState", "carControl", "carOutput", "LKAS11Tx", "liveLocationKalman"):
      count = int(np.searchsorted(row["motion_arrays"][name+"_t_ns"], end))
      original_count = len(row["motion_arrays"][name+"_t_ns"])
      for arrays in (row["motion_arrays"], row["view_arrays"]):
        for key, value in list(arrays.items()):
          if key.startswith(name+"_") and value.shape == (original_count,):
            arrays[key] = value[:count].copy()
        if name+"_payload" in arrays:
          arrays[name+"_payload"] = arrays[name+"_payload"][:count*8].copy()
          arrays[name+"_offsets"] = arrays[name+"_offsets"][:count+1].copy()
      if name == "LKAS11Tx":
        row["queued_apply_arrays"] = {k: v[:count].copy() for k, v in row["queued_apply_arrays"].items()}
  row["metadata"].update(opendbc_gitlink="a"*40, source_identity_verified=True)
  row["motion_metadata"].update(opendbc_gitlink="a"*40, source_eligible=True, eligibility_reasons=[])
  rebind(row)
  return row


def provenance(row, model):
  helpers = {name: moving.identity_hash({"synthetic_helper": name}) for name in check.required_helpers(model["kind"])}
  original, receipt = row["motion_metadata"], row["view_receipt"]
  body = {"kind": check.PROVENANCE_KIND, "model_sha256": moving.identity_hash(model),
    "split_sha256": moving.identity_hash(SPLIT), "static_identity_sha256": moving.identity_hash({"cp": CP, "sp": SP}),
    "training_receipt_sha256": "f"*64, "training_helpers_sha256": helpers, "validation_helpers_sha256": helpers.copy(),
    "source_log_sha256": original["source_log_sha256"], "motion_npz_sha256": original["output_npz_sha256"],
    "motion_arrays_sha256": receipt["input_arrays_sha256"], "motion_metadata_sha256": moving.identity_hash(original),
    "view_arrays_sha256": receipt["output_view_arrays_sha256"], "view_receipt_sha256": moving.identity_hash(receipt)}
  if model["kind"] != moving.KIND:
    body.update(queue_arrays_sha256=row["queued_apply_metadata"]["arrays_sha256"],
                queue_metadata_sha256=moving.identity_hash(row["queued_apply_metadata"]))
  row["validation_provenance"] = {**body, "sha256": moving.identity_hash(body)}


@pytest.fixture(scope="module")
def models():
  result = {}
  for kind in ("moving", "low", "approach"):
    rows = [admitted_record(route, kind, macros=120) for route in ROUTES]
    model = (moving.fit(rows, SPLIT) if kind == "moving" else
             category.fit(rows, SPLIT, CP, SP) if kind == "low" else category.fit_approach(rows, SPLIT, CP, SP))
    assert model["supported_cells"] and model["source_eligible"]
    result[kind] = model
  return result


def fixture(models, kind):
  model = deepcopy(models[kind])
  rows = [admitted_record(route, kind) for route in SPLIT["validation"]]
  for row in rows:
    provenance(row, model)
  return model, rows


@pytest.mark.parametrize("kind", ["moving", "low", "approach"])
def test_known_endpoint_residuals_and_all_route_denominators(models, kind):
  model, rows = fixture(models, kind)
  label = int(np.searchsorted(rows[0]["motion_arrays"]["carState_t_ns"], model["window_evidence"][0]["end_ns"]))
  rows[0]["motion_arrays"]["carState_steering_angle"][label] += 2.
  rebind(rows[0])
  provenance(rows[0], model)
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert report["status"] == "diagnostic" and report["counts"]["predicted"] >= 2
  scored = [w for w in report["windows"] if w["status"] == "predicted"]
  assert any(w["endpoint_residual"]["angle"] == pytest.approx(2.) for w in scored)
  assert set(report["routes"]) == set(SPLIT["validation"])
  assert report["counts"]["attempted"] == len(report["windows"])
  assert report["counts"]["quality_excluded"] > 0
  assert report["coverage"] == report["counts"]["predicted"]/report["counts"]["attempted"]
  assert all(len(w["endpoint_residual"]) == 5 for w in scored)
  if kind == "moving":
    assert all(w["category"]["status"] == "not_predicted" and w["category"]["predicted_next"] is None for w in scored)
  else:
    assert report["category_counts"]["predicted"] == len(scored)
  assert not report["predictive_validated"] and report["selection"] is None and not report["production_enabled"]


def test_conflicting_category_and_unsupported_predictions_are_retained(models):
  model, rows = fixture(models, "low")
  label = int(np.searchsorted(rows[0]["motion_arrays"]["carState_t_ns"], model["window_evidence"][0]["end_ns"]))
  rows[0]["motion_arrays"]["carState_standstill"][label] = 1.
  rebind(rows[0])
  provenance(rows[0], model)
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert report["category_counts"]["next_errors"] >= 1 and report["category_counts"]["transition_errors"] >= 1
  for cell in model["cells"].values():
    cell["next_hardware_standstill_values"] = [False, True]
    cell["supported"] = False
  model["supported_cells"] = 0
  model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  for row in rows:
    provenance(row, model)
  rejected = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert rejected["counts"]["unsupported"] > 0 and rejected["counts"]["predicted"] == 0
  assert any(w["reasons"] == ["ambiguous_next_hardware_standstill"] for w in rejected["windows"] if w["status"] == "unsupported")
  assert rejected["coverage"] == 0. and rejected["aggregate_score"] is None


@pytest.mark.parametrize("fault", ["train", "test", "mixed", "duplicate", "train_alias", "model_split", "source", "model_source",
                                  "unknown_pin", "policy", "provenance", "helpers"])
def test_every_metadata_and_source_gate_precedes_array_access(models, fault):
  model, rows = fixture(models, "low")
  if fault in ("train", "test", "mixed"):
    route = SPLIT["train"][0] if fault == "train" else SPLIT["test"][0]
    rows[0]["metadata"]["route"] = route
    if fault != "mixed":
      rows[1]["metadata"]["route"] = route
  elif fault == "duplicate":
    rows[1]["metadata"]["source_log_sha256"] = rows[0]["metadata"]["source_log_sha256"]
  elif fault == "train_alias":
    rows[0]["metadata"]["source_log_sha256"] = model["input_eligibility"][0]["source_log_sha256"]
  elif fault == "model_split":
    model["split"]["test"] = []
    model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  elif fault == "source":
    rows[0]["motion_metadata"].update(source_eligible=False, eligibility_reasons=["dirty_source_unknown"])
  elif fault in ("model_source", "unknown_pin"):
    model["source_eligible" if fault == "model_source" else "opendbc_gitlink"] = False if fault == "model_source" else None
    model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  elif fault == "policy":
    rows[0]["metadata"]["policy_values"].pop("EnforceTorqueControl")
  elif fault == "provenance":
    rows[0].pop("validation_provenance")
  else:
    proof = rows[0]["validation_provenance"]
    proof["validation_helpers_sha256"] = dict.fromkeys(proof["validation_helpers_sha256"], "9"*64)
    proof["sha256"] = moving.identity_hash({k: v for k, v in proof.items() if k != "sha256"})
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert report["status"] == "metadata_rejected" and report["counts"]["metadata_rejected_records"] == 2
  assert report["counts"]["attempted"] == 0 and not report["window_population_known"]
  assert report["model_sha256"] == moving.identity_hash(model)
  assert all(r["id"] and r["start_ns"] is None for r in report["record_rejections"])


@pytest.mark.parametrize("fault", ["queue_missing", "tie", "invalid"])
def test_queue_and_invalid_interpositions_keep_named_exclusions(models, fault):
  model, rows = fixture(models, "low")
  if fault == "queue_missing":
    rows[0].pop("queued_apply_arrays")
  elif fault == "tie":
    rows[0]["queued_apply_arrays"]["consume_t_ns"][16] = rows[0]["queued_apply_arrays"]["apply_t_ns"][16]
    rebind(rows[0])
    provenance(rows[0], model)
  else:
    rows[0]["motion_arrays"]["carState_valid"][16] = rows[0]["view_arrays"]["carState_valid"][16] = False
    rebind(rows[0])
    provenance(rows[0], model)
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert report["counts"]["quality_excluded"] > 0 and report["counts"]["predicted"] > 0
  reason = {"queue_missing": "queued_apply_witness_missing", "tie": "queued_apply_order", "invalid": "carState_invalid"}[fault]
  exclusions = [w for w in report["windows"] if w["status"] == "quality_excluded"]+report["record_rejections"]
  assert any(reason in w["reasons"] for w in exclusions)
  if fault == "queue_missing":
    assert report["counts"]["collector_rejected_records"] == 1 and report["coverage"] is None


def test_final_label_perturbation_changes_errors_only_and_inputs_are_immutable(models, monkeypatch):
  model, rows = fixture(models, "moving")
  original = moving.identity_hash({"model": model, "records": rows, "split": SPLIT, "cp": CP, "sp": SP})
  for name in ("fit",):
    monkeypatch.setattr(moving, name, lambda *a, **k: pytest.fail("fit invoked during validation"))
  monkeypatch.setattr(category, "fit", lambda *a, **k: pytest.fail("fit invoked during validation"))
  monkeypatch.setattr(category, "fit_approach", lambda *a, **k: pytest.fail("fit invoked during validation"))
  monkeypatch.setattr(np, "quantile", lambda *a, **k: pytest.fail("quantiles relearned during validation"))
  before = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert moving.identity_hash({"model": model, "records": rows, "split": SPLIT, "cp": CP, "sp": SP}) == original
  label_ns = int(rows[0]["motion_arrays"]["carState_t_ns"][8])
  source = rows[0]["metadata"]["source_log_sha256"]
  own = [w for w in before["windows"] if w["status"] == "predicted" and w["source_log_sha256"] == source]
  assert sum(w["end_ns"] == label_ns for w in own) == 1
  assert all(w["start_ns"] != label_ns for w in own)  # Its next LLK70ms gap is excluded, not frozen into an initial state.
  rows[0]["motion_arrays"]["carState_steering_angle"][8] += 2.
  rebind(rows[0])
  provenance(rows[0], model)
  after = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  left = {w["id"]: w for w in before["windows"] if w["status"] == "predicted"}
  right = {w["id"]: w for w in after["windows"] if w["status"] == "predicted"}
  assert left.keys() == right.keys()
  for key in left:
    assert left[key]["prediction_inputs"] == right[key]["prediction_inputs"] and left[key]["predicted_endpoint"] == right[key]["predicted_endpoint"]
  assert any(left[key]["endpoint_residual"] != right[key]["endpoint_residual"] for key in left)


@pytest.mark.parametrize("fit", [moving.fit, moving.base.fit, category.lateral_hold.fit, category.fit, category.fit_approach])
def test_validation_does_not_weaken_default_training_fit_admission(fit):
  rows = [admitted_record(route, "low") for route in SPLIT["validation"]]
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  with pytest.raises(moving.FitError, match="TRAIN"):
    fit(rows, SPLIT, **({} if fit in (moving.fit, moving.base.fit) else {"cp": CP, "sp": SP}))


@pytest.mark.parametrize("fault", ["kind", "pattern", "evidence", "numeric", "routes", "delta"])
def test_malformed_frozen_model_is_named_rejection_before_arrays(models, fault):
  model, rows = fixture(models, "low")
  if fault == "kind":
    model["kind"] = np.array(["bad", "kind"])
  elif fault == "pattern":
    model["patterns"] = [{}]
  elif fault == "evidence":
    model["window_evidence"] = [["bad"]]
  elif fault == "numeric":
    model["feature_min"][0] = "not a number"
  elif fault == "routes":
    next(iter(model["cells"].values()))["routes"] = [{}]
  else:
    next(iter(model["cells"].values()))["joint_delta_median"] = 1.
  model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  result = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert result["status"] == "metadata_rejected" and result["counts"]["attempted"] == 0


@pytest.mark.parametrize("sign,direction", [(-1, "LEFT"), (1, "RIGHT")])
def test_measured_yaw_direction_is_explicit_even_when_population_unsupported(models, sign, direction):
  model, rows = fixture(models, "moving")
  for row in rows:
    row["motion_arrays"]["liveLocationKalman_yaw"][:] = sign*.01
    rebind(row)
    provenance(row, model)
  result = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert result["directions"][direction]["unsupported"] > 0
  assert result["directions"][direction]["residual_sample_count"] == 0
  assert result["directions"][direction]["conditional_endpoint_mae"] is None
  assert "phase" not in result["directions"][direction]


@pytest.mark.parametrize("name", ["base", "policy", "category", "binding_metadata", "binding_content", "view"])
def test_private_partition_plumbing_never_admits_test(name):
  row = admitted_record(SPLIT["validation"][0], "low")
  row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  with pytest.raises(ValueError, match="TRAIN or validation"):
    if name == "base":
      moving.base._metadata([row], SPLIT, partition="test")
    elif name == "policy":
      category.lateral_hold._metadata([row], SPLIT, CP, SP, partition="test")
    elif name == "category":
      category._metadata([row], SPLIT, CP, SP, partition="test")
    elif name == "binding_metadata":
      moving._binding_metadata(row, SPLIT, partition="test")
    elif name == "binding_content":
      moving._bind(row, SPLIT, partition="test")
    else:
      moving.view._admit(row["motion_arrays"], row["motion_metadata"], SPLIT, row["view_receipt"]["input_proof"], partition="test")


@pytest.mark.parametrize("fault", ["metadata", "reference"])
def test_rejected_malformed_record_claims_stay_serializable_without_arrays(models, fault):
  import json
  model, rows = fixture(models, "moving")
  if fault == "metadata":
    rows[0]["metadata"] = None
  else:
    rows[0]["motion_metadata"]["output_npz_sha256"] = object()
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  result = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert result["status"] == "metadata_rejected" and result["counts"]["attempted"] == 0
  assert isinstance(json.dumps(result), str)


@pytest.mark.parametrize("fault", ["conflicting", "missing_cell"])
def test_unsupported_categorical_labels_stay_observed_not_false_or_lost(models, fault):
  model, rows = fixture(models, "low")
  label = int(np.searchsorted(rows[0]["motion_arrays"]["carState_t_ns"], model["window_evidence"][0]["end_ns"]))
  label_ns = int(rows[0]["motion_arrays"]["carState_t_ns"][label])
  rows[0]["motion_arrays"]["carState_standstill"][label] = 1.
  rebind(rows[0])
  if fault == "conflicting":
    for cell in model["cells"].values():
      cell["supported"], cell["next_hardware_standstill_values"] = False, [False, True]
  else:
    model["cells"] = {}
  model["supported_cells"] = 0
  model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  for row in rows:
    provenance(row, model)
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  windows = [w for w in report["windows"] if w["status"] == "unsupported"]
  target = next(w for w in windows if w["source_log_sha256"] == rows[0]["metadata"]["source_log_sha256"] and w["end_ns"] == label_ns)
  assert target["category"] == {"status": "unsupported", "initial_observed": False, "actual_next": True,
                                "predicted_next": None, "next_error": None, "transition_error": None}
  assert target["actual_endpoint"]["v"] == 0. and target["predicted_endpoint"] is None
  assert report["category_counts"]["unsupported"] == len(windows)
  assert report["category_counts"]["known_label_denominator"] == len(windows)
  assert report["category_counts"]["error_denominator"] == 0 and not report["category_counts"]["confusion"]
  assert report["category_counts"]["quality_unknown"] == report["counts"]["quality_excluded"]
  assert all("category" not in w for w in report["windows"] if w["status"] == "quality_excluded")


@pytest.mark.parametrize("field", ["features", "split"])
def test_frozen_feature_names_and_split_arrays_are_metadata_rejected_before_records(models, field):
  model, rows = fixture(models, "low")
  model[field] = np.asarray(model[field] if field == "features" else list(model[field]))
  model["model_sha256"] = moving.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  report = check.evaluate_validation_transitions(model, rows, SPLIT, CP, SP)
  assert report["status"] == "metadata_rejected" and report["counts"]["attempted"] == 0
