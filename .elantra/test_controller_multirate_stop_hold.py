"""Synthetic transition/hold arithmetic; not vehicle or contact-dynamics evidence."""
from copy import deepcopy

import numpy as np
import pytest

import controller_multirate_stop_hold as hold
from test_controller_multirate_lateral import SPLIT, STATE, Unreadable, record as moving_record
from test_controller_native_lateral_runtime import params


CP = {"carFingerprint": "HYUNDAI_ELANTRA_2024", "brand": "hyundai", "minSteerSpeed": 0.,
      "steerAtStandstill": False, "openpilotLongitudinalControl": True, "steerControlType": "torque",
      "carFw": [{"fwVersion": b"\x00\xff"}]}
SP = {"enableGasInterceptor": False}


def policy():
  return {**params(), "BlinkerLateralReengageDelay": "0", "BlinkerMinLateralControlSpeed": "0",
          "BlinkerPauseLateralControl": "0", "IsMetric": "1"}


def record(route, mode="hold", variant="late", macros=200, monotonic=False):
  row = moving_record(route, variant, macros)
  a, m = row["arrays"], row["metadata"]
  n = len(a["carState_t_ns"])
  pattern = ([.4, .4, .4, .2, .2] if variant == "late" else [.4, .2, .2, .4, .4])
  speed = np.zeros(n) if mode == "hold" else np.resize(pattern, n) if mode == "mixed" else np.ones(n)
  active = speed > .3
  if monotonic:
    change = np.array([1., .01, 4.]) if variant == "late" else np.array([3., .03, 12.])
    states = np.arange(macros+1)[:, None]*change
    states[:, 2] += 12.
    held = states[np.minimum(np.arange(n)//5, macros)]
    a["carState_steering_angle"], a["carState_vendor_rate_unsigned"] = held[:, 0], held[:, 2]
    a["liveLocationKalman_yaw"] = states[:, 1]
  a["carState_v"] = speed
  a["carState_standstill"], a["carState_standstill_present"] = np.zeros(n), np.ones(n, bool)
  a["LKAS11Tx_torque"], a["LKAS11Tx_request"] = active.astype(float)*20., active.astype(float)
  a["carControl_long_active"], a["carControl_long_active_present"] = np.ones(n), np.ones(n, bool)
  a["carControl_torque_request"], a["carControl_torque_request_present"] = active.astype(float)*.05, np.ones(n, bool)
  a["carOutput_t_ns"], a["carOutput_valid"] = a["carState_t_ns"].copy(), np.ones(n, bool)
  for key, values in (("torque_counts", active.astype(float)*20.), ("torque_reported", active.astype(float)*.04)):
    a["carOutput_"+key], a["carOutput_"+key+"_present"] = values, np.ones(n, bool)
  # Valid causal prefixes; no interpolation or missing-prefix defaults.
  for name in hold.FIELDS:
    for key in [k for k in a if k.startswith(name+"_")]:
      if name == "liveLocationKalman":
        continue
      a[key] = np.r_[np.repeat(a[key][:1], 2), a[key]]
    if name != "liveLocationKalman":
      a[name+"_t_ns"][:2] = a[name+"_t_ns"][2]-np.array([20_000_000, 10_000_000], np.uint64)
  m.update(policy_values=policy(), native_policy_source_sha256=hold.NATIVE_POLICY_SHA,
           static_identity_sha256=hold.identity_hash({"cp": CP, "sp": SP}), policy_identity_sha256=hold.identity_hash(policy()))
  for kind in ("static", "policy"):
    proof = m[kind+"_anchor"]
    proof["identity_sha256"] = m[kind+"_identity_sha256"]
    proof["sha256"] = hold.identity_hash({k: v for k, v in proof.items() if k != "sha256"})
  return row


def fit(mode="hold", **kwargs):
  return hold.fit([record(route, mode) for route in SPLIT["train"]], SPLIT, CP, SP, **kwargs)


def test_inactive_request_false_and_zero_speed_do_not_force_frozen_or_zero_state():
  model = fit()
  result = hold.predict_macro(model, STATE, [0.]*5, [0]*5, [False]*5, [False]*5, .01)
  assert result == pytest.approx({"angle": 1., "yaw": .01, "vendor_rate": 16.})
  assert model["accepted_windows"] == 400 and model["supported_cells"] > 0
  assert model["actual_dt_s"] == {"min": .05, "max": .05}
  assert len(model["features"]) == 24 and model["micro_nodes"] == 5
  assert not model["source_eligible"] and not model["selection_eligible"]
  assert not any(model[k] for k in ("contact_hold_validated", "nonreversing_contact_validated", "hundred_hz_feedback_validated"))


def test_exact_mixed_bit_patterns_never_pool_even_with_one_quantile_bin():
  rows = [record(route, "mixed", variant, monotonic=True) for route in SPLIT["train"] for variant in ("early", "late")]
  model = hold.fit(rows, SPLIT, CP, SP, quantile_bins=1)
  for pattern, angle in (([True, True, True, False, False], 1.), ([True, False, False, True, True], 3.)):
    speed, command = [(.4 if a else .2) for a in pattern], [(20 if a else 0) for a in pattern]
    assert hold.predict_macro(model, STATE, speed, command, pattern, pattern, .01)["angle"] == angle
  unseen = [True, False, True, False, True]
  result = hold.predict_macro(model, STATE, [.4 if a else .2 for a in unseen], [20 if a else 0 for a in unseen], unseen, unseen, .01)
  assert result.reason == "unsupported_pattern"


@pytest.mark.parametrize("fault,reason", [("active_false", "active_request_false"), ("inactive_nonzero", "inactive_command_or_request"),
                                         ("invalid", "carState_invalid"), ("driver", "nonzero_driver_torque"),
                                         ("long_inactive", "comma_long_inactive"), ("hardware_standstill", "hardware_standstill_unmodeled"),
                                         ("output_invalid", "carOutput_invalid"), ("missing_standstill", "missing_fields"),
                                         ("llk_std", "liveLocationKalman_missing_or_nonfinite")])
def test_invalid_interpositions_are_retained_without_old_valid_fallback(fault, reason):
  row = record("route-a", "mixed")
  a = row["arrays"]
  if fault == "active_false":
    a["LKAS11Tx_request"][3] = 0
  elif fault == "inactive_nonzero":
    a["LKAS11Tx_torque"][5] = 1
  elif fault == "invalid":
    a["carState_valid"][3] = False
  elif fault == "driver":
    a["carState_driver_torque"][3] = .01
  elif fault == "long_inactive":
    a["carControl_long_active"][3] = 0
  elif fault == "hardware_standstill":
    a["carState_standstill"][3] = 1
  elif fault == "output_invalid":
    a["carOutput_valid"][3] = False
  elif fault == "missing_standstill":
    del a["carState_standstill"]
  else:
    a["liveLocationKalman_roll_std"][1] = np.nan
  model = hold.fit([row, record("route-b", "mixed")], SPLIT, CP, SP)
  assert any(reason in entry["reasons"] for entry in model["exclusions"])


@pytest.mark.parametrize("fault", ["heldout", "source_alias", "static", "policy", "native_policy"])
def test_metadata_refuses_before_any_array_access(fault):
  a, b = record("route-a"), record("route-b")
  a["arrays"] = b["arrays"] = Unreadable()
  if fault == "heldout":
    b["metadata"]["route"] = "route-t"
  elif fault == "source_alias":
    b["metadata"]["source_log_sha256"] = a["metadata"]["source_log_sha256"]
  elif fault == "static":
    for row in (a, b):
      row["metadata"]["static_identity_sha256"] = "c"*64
  elif fault == "policy":
    del a["metadata"]["policy_values"]["CustomTorqueParams"]
  else:
    a["metadata"]["native_policy_source_sha256"] = "f"*64
  with pytest.raises(hold.FitError):
    hold.fit([a, b], SPLIT, CP, SP)


def test_source_pin_does_not_promote_contact_or_zoh_validation():
  rows = [record(route) for route in SPLIT["train"]]
  for row in rows:
    row["metadata"].update(opendbc_gitlink="b"*40, source_identity_verified=True)
  model = hold.fit(rows, SPLIT, CP, SP)
  assert model["source_eligible"] and not model["selection_eligible"] and model["candidate"] is None
  assert not model["contact_hold_validated"] and not model["production_enabled"]


def test_frozen_prediction_never_reads_future_motion_and_does_not_rescale_delta():
  rows = [record(route) for route in SPLIT["train"]]
  model = hold.fit(rows, SPLIT, CP, SP)
  before = hold.predict_macro(model, STATE, [0.]*5, [0]*5, [False]*5, [False]*5, .01)
  for row in rows:
    row["arrays"]["carState_steering_angle"][:] = 999
    row["arrays"]["liveLocationKalman_yaw"][:] = -999
  assert hold.predict_macro(model, STATE, [0.]*5, [0]*5, [False]*5, [False]*5, .01) == before
  assert before["angle"] == 1.


def test_missing_anchor_and_insufficient_cells_remain_explicit():
  rows = [record(route, macros=10) for route in SPLIT["train"]]
  model = hold.fit(rows, SPLIT, CP, SP)
  assert not model["supported_cells"] and all("sample_support" in c["rejections"] for c in model["cells"].values())
  assert hold.predict_macro(model, STATE, [0.]*5, [0]*5, [False]*5, [False]*5, .01).reason == "cell_support"
  rows[0]["metadata"]["policy_anchor"] = None
  model = hold.fit(rows, SPLIT, CP, SP)
  assert not model["static_policy_timing_verified"] and "static_policy_timing_unknown" in model["excluded_reasons"]


def test_prediction_contract_has_no_inactive_torque_or_active_false_request_substitution():
  model = fit()
  assert hold.predict_macro(model, STATE, [0.]*5, [1]*5, [False]*5, [False]*5, .01).reason == "inactive_command_or_request"
  assert hold.predict_macro(model, STATE, [.4]*5, [0]*5, [True]*5, [False]*5, .01).reason == "active_request_false"
  assert hold.predict_macro(model, STATE, [.4]*5, [0]*5, [True]*5, [True]*5, .01).reason == "moving_macro_uses_moving_model"
  assert hold.predict_macro(model, STATE, [0.]*5, [0]*5, [0]*5, [0]*5, .01).reason == "boolean_pattern_required"
  assert hold.predict_macro(model, STATE, [.2]*5, [0]*5, [False]*5, [False]*5, .01).reason == "outside_training_bounds"


def test_native_scope_and_support_thresholds_cannot_be_weakened():
  rows = [record(route) for route in SPLIT["train"]]
  for kwargs in ({"min_samples": 99}, {"min_routes": 1}):
    with pytest.raises(hold.FitError):
      hold.fit(rows, SPLIT, CP, SP, **kwargs)
  cp = deepcopy(CP)
  cp["steerAtStandstill"] = True
  with pytest.raises(hold.FitError):
    hold.fit(rows, SPLIT, cp, SP)
  model = hold.fit([record(route, "moving") for route in SPLIT["train"]], SPLIT, CP, SP)
  assert not model["cells"] and "moving_macro_uses_moving_model" in model["excluded_reasons"]


def test_tied_invalid_clock_and_wrong_can_bus_cannot_supply_support():
  row = record("route-a")
  a = row["arrays"]
  for key in [k for k in a if k.startswith("carOutput_")]:
    a[key] = np.insert(a[key], 4, a[key][3])
  a["carOutput_valid"][4] = False
  other = record("route-b")
  other["arrays"]["LKAS11Tx_src"][:] = 0
  model = hold.fit([row, other], SPLIT, CP, SP)
  assert "carOutput_duplicate" in model["excluded_reasons"] and "carOutput_invalid" in model["excluded_reasons"]
  assert "no_returned_bus0_echo" in model["excluded_reasons"]


def test_policy_anchor_must_precede_the_entire_checked_prefix():
  rows = [record(route) for route in SPLIT["train"]]
  for row in rows:
    proof = row["metadata"]["policy_anchor"]
    proof["t_ns"] = 1_000_000_000
    proof["sha256"] = hold.identity_hash({k: v for k, v in proof.items() if k != "sha256"})
  model = hold.fit(rows, SPLIT, CP, SP)
  assert model["accepted_windows"] == 398
  assert model["excluded_reasons"]["before_static_or_policy_prefix"] == 2


def test_actual_48ms_delta_is_retained_without_rescaling_to_nominal50ms():
  rows = [record(route) for route in SPLIT["train"]]
  for row in rows:
    for name in hold.FIELDS.keys()-{"liveLocationKalman"}:
      for key in [k for k in row["arrays"] if k.startswith(name+"_")]:
        row["arrays"][key] = np.insert(row["arrays"][key], 0, row["arrays"][key][0])
      row["arrays"][name+"_t_ns"][0] -= np.uint64(10_000_000)
    for key in [k for k in row["arrays"] if k.endswith("_t_ns")]:
      row["arrays"][key] = np.array([1_000_000_000+(int(t)-1_000_000_000)*24//25 for t in row["arrays"][key]], np.uint64)
  model = hold.fit(rows, SPLIT, CP, SP)
  assert model["actual_dt_s"] == {"min": .048, "max": .048}
  result = hold.predict_macro(model, STATE, [0.]*5, [0]*5, [False]*5, [False]*5, .01)
  assert result["angle"] == 1. and result["yaw"] == .01


def test_output_and_echo_keep_independent_clocks_without_invented_delivery_equality():
  rows = [record(route) for route in SPLIT["train"]]
  for row in rows:
    row["arrays"]["carOutput_t_ns"] -= np.uint64(2_000_000)
    row["arrays"]["carOutput_torque_counts"][:] = 3
    row["arrays"]["carOutput_torque_reported"][:] = 3/409
  model = hold.fit(rows, SPLIT, CP, SP)
  first = model["window_evidence"][0]
  assert first["evidence"]["output_t_ns"][0] == first["start_ns"]-2_000_000
  assert first["evidence"]["echo_t_ns"][0] == first["start_ns"]
  assert first["evidence"]["output_counts"] == [3.]*5 and first["evidence"]["echo_counts"] == [0.]*5
  assert first["evidence"]["long_active"] == [1.]*5 and not model["selection_eligible"]


def test_declared_effective_active_cannot_override_native_low_speed_policy():
  model = fit()
  active = [True, False, False, False, False]
  result = hold.predict_macro(model, STATE, [0.]*5, [0]*5, active, active, .01)
  assert result.reason == "effective_active_speed_mismatch"


@pytest.mark.parametrize("service,field", [("carState", "driver_torque"), ("carControl", "lat_active"),
                                           ("LKAS11Tx", "request"), ("carOutput", "torque_counts")])
@pytest.mark.parametrize("fault", ["invalid", "missing"])
def test_every_consumed_phased_prefix_predecessor_is_quality_checked(service, field, fault):
  row = record("route-a")
  a = row["arrays"]
  a[service+"_t_ns"] -= np.uint64(5_000_000)
  a[service+("_valid" if fault == "invalid" else "_"+field+"_present")][0] = False
  model = hold.fit([row, record("route-b")], SPLIT, CP, SP)
  reason = service+("_invalid" if fault == "invalid" else "_missing_or_nonfinite")
  assert any(e["start_ns"] == 1_000_000_000 and reason in e["reasons"] for e in model["exclusions"])


def test_nonzero_driver_on_phased_cs_prefix_predecessor_cannot_enter_support():
  row = record("route-a")
  row["arrays"]["carState_t_ns"] -= np.uint64(5_000_000)
  row["arrays"]["carState_driver_torque"][0] = .01
  model = hold.fit([row, record("route-b")], SPLIT, CP, SP)
  assert any(e["start_ns"] == 1_000_000_000 and "nonzero_driver_torque" in e["reasons"] for e in model["exclusions"])
