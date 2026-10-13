"""Synthetic contract arithmetic only; no contact, clock acquisition or car proof."""
from copy import deepcopy

import numpy as np
import pytest

import controller_multirate_joint_stop_hold as hold
import controller_multirate_lateral as base
from test_controller_multirate_joint import ROUTES, SPLIT, Unreadable, record as moving_record, rebind
from test_controller_multirate_stop_hold import CP, SP, policy


STATE = {"v": 0., "a": 0., "angle": 0., "yaw": 0., "vendor_rate": 12.}


def bind(row):
  rebind(row)
  q = row["queued_apply_metadata"]
  q.update(arrays_sha256=base.identity_hash(row["queued_apply_arrays"]),
           motion_arrays_sha256=row["view_receipt"]["input_arrays_sha256"],
           view_arrays_sha256=row["view_receipt"]["output_view_arrays_sha256"])
  q["sha256"] = base.identity_hash({k: v for k, v in q.items() if k != "sha256"})


def record(route, *, sign=1, mode="hold", next_standstill=False, initial_standstill=False, macros=120):
  row = moving_record(route, macros=macros)
  a, m = row["motion_arrays"], row["metadata"]
  n, times = len(a["carState_t_ns"]), a["carState_t_ns"]
  phase = (np.arange(n)-3) % 12
  end = (phase >= 5) & (np.arange(n) >= 3)
  a["carState_v"][:] = 0. if mode == "hold" else 1.
  if mode == "mixed":
    a["carState_v"][:] = np.where(np.isin(phase, [1, 2, 3, 4]), .2, 1.)
  a["carState_a"][:] = end*.1
  a["carState_steering_angle"][:] = end*sign
  a["carState_vendor_rate_unsigned"][:] = 12.+end*4.
  a["carState_standstill"] = np.where(end, next_standstill, initial_standstill).astype(float)
  a["carState_standstill_present"] = np.ones(n, bool)
  li = np.searchsorted(times, a["liveLocationKalman_t_ns"])
  a["liveLocationKalman_yaw"][:] = end[li]*sign*.01
  feedback = np.maximum(np.arange(n)-2, 0)
  active = (a["carState_v"][feedback] > .3) & (a["carState_standstill"][feedback] == 0)
  a["LKAS11Tx_request"][:] = active
  a["LKAS11Tx_torque"][:] = active*20
  m.update(policy_values=policy(), native_policy_source_sha256=hold.NATIVE_POLICY_SHA,
           static_identity_sha256=base.identity_hash({"cp": CP, "sp": SP}),
           policy_identity_sha256=base.identity_hash(policy()))
  for kind in ("static", "policy"):
    anchor = m[kind+"_anchor"]
    anchor["identity_sha256"] = m[kind+"_identity_sha256"]
    anchor["sha256"] = base.identity_hash({k: v for k, v in anchor.items() if k != "sha256"})
  q = {"echo_t_ns": times.copy(), "apply_t_ns": times-np.uint64(1_000_000),
       "consume_t_ns": times-np.uint64(2_000_000), "consumed_cc_t_ns": times-np.uint64(10_000_000),
       "compute_t_ns": times-np.uint64(11_000_000),
       "feedback_cs_t_ns": times-np.uint64(20_000_000), "valid": np.ones(n, bool), "effective_active": active.astype(bool)}
  q["valid"][:2] = False  # No invented startup predecessor.
  row["queued_apply_arrays"] = q
  row["queued_apply_metadata"] = {"kind": hold.QUEUE_KIND, "route": route, "source_log_sha256": m["source_log_sha256"],
    "card_source_sha256": hold.CARD_SHA, "native_policy_source_sha256": hold.NATIVE_POLICY_SHA,
    "static_identity_sha256": m["static_identity_sha256"], "policy_identity_sha256": m["policy_identity_sha256"]}
  bind(row)
  return row


def fitted(**kwargs):
  return hold.fit([record(route, **kwargs) for route in ROUTES], SPLIT, CP, SP)


def predict(model, *, state=None, active=None, request=None, command=None, standstill=False):
  return hold.predict_macro(model, STATE if state is None else state, [1.]*5, [1.]*5,
                            [0]*5 if command is None else command,
                            np.zeros(5, bool) if active is None else active,
                            np.zeros(5, bool) if request is None else request, .01, standstill)


@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_speed_rack_and_yaw_evolve_with_independent_false_standstill(sign):
  model = fitted(sign=sign)
  result = predict(model)
  assert result["state"] == pytest.approx({"v": 0., "a": .1, "angle": sign, "yaw": sign*.01, "vendor_rate": 16.})
  assert result["hardware_standstill"] is False
  assert len(model["features"]) == 31 and model["actual_dt_s"] == {"min": .05, "max": .05}
  assert not model["source_eligible"] and model["selection"] is None and not model["production_enabled"]
  assert not any(model[k] for k in ("contact_hold_validated", "hundred_hz_feedback_validated", "joint_median_coherence_validated"))


def test_next_hardware_standstill_is_source_category_not_derived_from_speed():
  result = predict(fitted(next_standstill=True))
  assert result["state"]["v"] == 0 and result["hardware_standstill"] is True
  assert predict(fitted(), standstill=True).reason == "unsupported_pattern"
  assert predict(fitted(), standstill=None).reason == "hardware_standstill_required"


def test_ambiguous_next_category_is_retained_and_unsupported_never_majority_voted():
  rows = [record(ROUTES[0]), record(ROUTES[1], next_standstill=True)]
  model = hold.fit(rows, SPLIT, CP, SP)
  assert not model["supported_cells"]
  assert any(c["next_hardware_standstill_values"] == [False, True] for c in model["cells"].values())
  assert predict(model).reason == "ambiguous_next_hardware_standstill"


def test_actual_consumed_policy_uses_earlier_feedback_cs_not_current_apply_speed():
  model = fitted(mode="mixed")
  assert model["accepted_windows"] > 0
  evidence = model["window_evidence"][0]
  assert evidence["effective_active"] != [True]*5
  assert all(c < u < p <= e for c, u, p, e in zip(evidence["consumed_cc_t_ns"], evidence["consume_t_ns"],
                                                evidence["apply_t_ns"], evidence["echo_t_ns"], strict=True))
  assert any(c < f for c, f in zip(evidence["feedback_cs_t_ns"], evidence["node_t_ns"], strict=True))


def test_all_active_low_speed_has_distinct_mode_and_missing_pattern_never_falls_back():
  model = fitted(mode="active")
  result = predict(model, state={**STATE, "v": 1.}, active=np.ones(5, bool), request=np.ones(5, bool), command=[20]*5)
  assert result["state"]["v"] == 1.
  assert predict(model, state={**STATE, "v": 1.}).reason == "unsupported_pattern"
  assert predict(model, active=np.ones(5, bool)).reason == "active_request_false"
  assert predict(model, command=[1]*5).reason == "inactive_command_or_request"


@pytest.mark.parametrize("fault,reason", [
  ("queue_invalid", "queued_apply_invalid"), ("queue_future", "queued_apply_order"),
  ("queue_policy", "queued_effective_active_mismatch"), ("missing_standstill", "missing_original_fields"),
  ("unknown_standstill", "hardware_standstill_invalid"), ("long_inactive", "comma_long_inactive"),
  ("driver", "nonzero_driver_torque"), ("roll_std", "liveLocationKalman_missing_or_nonfinite"),
  ("invalid_cs", "carState_invalid"), ("request", "inactive_command_or_request")])
def test_invalid_prefix_and_interpositions_are_excluded_without_older_fallback(fault, reason):
  rows = [record(route) for route in ROUTES]
  row, a, q = rows[0], rows[0]["motion_arrays"], rows[0]["queued_apply_arrays"]
  index = 16
  if fault == "queue_invalid":
    q["valid"][index] = False
  elif fault == "queue_future":
    q["consumed_cc_t_ns"][index] = q["apply_t_ns"][index]
  elif fault == "queue_policy":
    q["effective_active"][index] = True
  elif fault == "missing_standstill":
    del a["carState_standstill"]
  elif fault == "unknown_standstill":
    a["carState_standstill"][index] = np.nan
  elif fault == "long_inactive":
    a["carControl_long_active"][index] = 0
  elif fault == "driver":
    a["carState_driver_torque"][index] = .01
  elif fault == "roll_std":
    a["liveLocationKalman_roll_std"][2] = np.nan
  elif fault == "invalid_cs":
    a["carState_valid"][index] = row["view_arrays"]["carState_valid"][index] = False
  else:
    a["LKAS11Tx_request"][index] = 1
  bind(row)
  model = hold.fit(rows, SPLIT, CP, SP)
  assert any(reason in e["reasons"] for e in model["exclusions"])


@pytest.mark.parametrize("fault", ["heldout", "duplicate_train", "alias", "policy", "source_pin", "queue_source"])
def test_all_source_static_policy_split_lineage_gates_precede_array_access(fault):
  rows = [record(route) for route in ROUTES]
  split = deepcopy(SPLIT)
  if fault == "heldout":
    rows[1]["metadata"]["route"] = SPLIT["test"][0]
  elif fault == "duplicate_train":
    split["train"].append(ROUTES[0])
  elif fault == "alias":
    rows[1]["metadata"]["source_log_sha256"] = rows[0]["metadata"]["source_log_sha256"]
  elif fault == "policy":
    rows[1]["metadata"]["policy_values"].pop("EnforceTorqueControl")
  elif fault == "source_pin":
    for row in rows:
      row["metadata"].update(opendbc_gitlink="a"*40, source_identity_verified=True)
  else:
    rows[1]["queued_apply_metadata"]["source_log_sha256"] = "9"*64
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  with pytest.raises(hold.FitError):
    hold.fit(rows, split, CP, SP)


def test_componentwise_median_remains_explicit_unvalidated_joint_hypothesis():
  rows = [record(route) for route in ROUTES]
  row = rows[1]
  row["motion_arrays"]["carState_steering_angle"] *= 3
  row["motion_arrays"]["liveLocationKalman_yaw"] *= 3
  row["motion_arrays"]["carState_a"] *= 3
  bind(row)
  model = hold.fit(rows, SPLIT, CP, SP)
  assert predict(model)["state"] == pytest.approx({"v": 0., "a": .2, "angle": 2., "yaw": .02, "vendor_rate": 16.})
  assert not model["joint_median_coherence_validated"] and not model["velocity_acceleration_consistency_validated"]


def test_bounds_negative_state_and_malformed_delta_are_not_clipped_or_broadcast():
  model = fitted()
  assert isinstance(predict(model, state={**STATE, "angle": 99.}), hold.Unsupported)
  for delta, reason in (([-1., 0., 0., 0., 0.], "invalid_joint_prediction"),
                        ([0., 0., 0., 0., -99.], "invalid_joint_prediction"), (1., "model_contract")):
    changed = deepcopy(model)
    for cell in changed["cells"].values():
      cell["joint_delta_median"] = delta
    assert predict(changed).reason == reason


def test_missing_report_and_queue_lineage_are_not_invented():
  rows = [record(route) for route in ROUTES]
  del rows[0]["queued_apply_arrays"]
  model = hold.fit(rows, SPLIT, CP, SP)
  assert any("queued_apply_witness_missing" in e["reasons"] for e in model["exclusions"])
  assert hold.predict_macro(model, STATE, [1.]*5, None, [0]*5, np.zeros(5, bool), np.zeros(5, bool), .01, False).reason == "reported_channel_required"


def test_support_thresholds_and_unknown_dirty_source_never_promote_selection():
  model = hold.fit([record(ROUTES[0])], SPLIT, CP, SP)
  assert not model["supported_cells"] and all("route_support" in c["rejections"] for c in model["cells"].values())
  small = hold.fit([record(route, macros=10) for route in ROUTES], SPLIT, CP, SP)
  assert not small["supported_cells"]
  for kwargs in ({"min_samples": 99}, {"min_routes": 1}):
    with pytest.raises(hold.FitError):
      hold.fit([record(route) for route in ROUTES], SPLIT, CP, SP, **kwargs)
  assert fitted()["selection"] is None and not fitted()["source_eligible"]


def test_unused_future_gt_cannot_enter_pure_prediction():
  model = fitted()
  before = predict(model)
  model["unused_future_measured_state"] = {key: [999., -999.] for key in STATE}
  assert predict(model) == before
  assert not any("future" in feature or "speed_" in feature for feature in hold.FEATURES)


def test_exact_mode_patterns_stay_separate_with_one_bin_and_never_use_old_model():
  rows = [record(route, mode=mode) for route in ROUTES for mode in ("active", "mixed")]
  # Independent physical files, not one source assigned to different patterns.
  for index, row in enumerate(rows):
    row["metadata"]["segment"] = row["motion_metadata"]["segment"] = row["metadata"]["route"]+"--"+str(index % 2)
    source = str(index+1)*64
    row["metadata"]["source_log_sha256"] = row["motion_metadata"]["source_log_sha256"] = source
    row["queued_apply_metadata"]["source_log_sha256"] = source
    for kind in ("static", "policy"):
      anchor = row["metadata"][kind+"_anchor"]
      anchor["source_log_sha256"] = source
      anchor["sha256"] = base.identity_hash({k: v for k, v in anchor.items() if k != "sha256"})
    bind(row)
  model = hold.fit(rows, SPLIT, CP, SP, quantile_bins=1)
  assert len(model["patterns"]) >= 2
  unseen = np.array([True, False, True, False, True])
  assert predict(model, state={**STATE, "v": 1.}, active=unseen, request=unseen,
                 command=np.where(unseen, 20, 0)).reason == "unsupported_pattern"


def test_known_pin_and_verified_enrichment_cannot_promote_original_dirty_source():
  rows = [record(route) for route in ROUTES]
  for row in rows:
    row["metadata"].update(opendbc_gitlink="a"*40, source_identity_verified=True)
    row["motion_metadata"].update(opendbc_gitlink="a"*40, eligibility_reasons=["dirty_source_unknown"])
    bind(row)
  model = hold.fit(rows, SPLIT, CP, SP)
  assert model["supported_cells"] and model["source_pin_verified"]
  assert not model["source_eligible"] and model["complete_controller_candidate"] is None


@pytest.mark.parametrize("fault", ["original", "view", "queue"])
def test_actual_content_mutations_fail_hash_binding(fault):
  rows = [record(route) for route in ROUTES]
  if fault == "queue":
    rows[0]["queued_apply_arrays"]["valid"][16] = False
    model = hold.fit(rows, SPLIT, CP, SP)
    assert any("queued_apply_content_binding" in e["reasons"] for e in model["exclusions"])
  else:
    arrays, key = ((rows[0]["motion_arrays"], "carState_a") if fault == "original" else
                   (rows[0]["view_arrays"], "carOutput_accel_reported"))
    arrays[key][16] += .1
    with pytest.raises(hold.FitError):
      hold.fit(rows, SPLIT, CP, SP)


def test_actual_consumed_prefix_predecessor_cannot_precede_static_or_policy_anchor():
  rows = [record(route) for route in ROUTES]
  for row in rows:
    for kind in ("static", "policy"):
      anchor = row["metadata"][kind+"_anchor"]
      anchor["t_ns"] = 1_095_000_000  #1120ms macro's1100ms echo consumes1080ms CS.
      anchor["sha256"] = base.identity_hash({k: v for k, v in anchor.items() if k != "sha256"})
  model = hold.fit(rows, SPLIT, CP, SP)
  affected = [e for e in model["exclusions"] if e["start_ns"] == 1_120_000_000]
  assert len(affected) == 2 and all("before_static_or_policy_prefix" in e["reasons"] for e in affected)


@pytest.mark.parametrize("bad_metadata", [None, []])
def test_malformed_queue_metadata_is_declared_fit_error_before_arrays(bad_metadata):
  rows = [record(route) for route in ROUTES]
  rows[-1]["queued_apply_metadata"] = bad_metadata
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = row["queued_apply_arrays"] = Unreadable()
  with pytest.raises(hold.FitError, match="queued apply"):
    hold.fit(rows, SPLIT, CP, SP)
