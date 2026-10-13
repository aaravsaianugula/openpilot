"""Synthetic hash-bound record contracts; no file/schema/native/vehicle proof."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

import controller_multirate_joint as joint
import controller_motion_longitudinal as view
import controller_multirate_lateral as base


ROUTES = ["000001b7--1972f5d799", "000001b8--a9d69ce1d6"]
SPLIT = {"train": ROUTES, "validation": ["000001bf--acf1234567"], "test": ["000001cc--123456789a"]}
STATE = {"v": 10., "a": 0., "angle": 0., "yaw": 0., "vendor_rate": 12.}


def rebind(row):
  a, m, r = row["motion_arrays"], row["motion_metadata"], row["view_receipt"]
  expected = {"cohort": m["cohort"], "segment": m["segment"], "source_log_sha256": m["source_log_sha256"],
              "motion_npz_sha256": m["output_npz_sha256"], "arrays_sha256": base.identity_hash(a),
              "metadata_sha256": base.identity_hash(m), "schema_sha256": deepcopy(m["schema_sha256"]),
              "split_sha256": base.identity_hash(SPLIT), "motion_splits_sha256": m["splits_sha256"]}
  r.update(kind=view.KIND, route=row["metadata"]["route"], cohort=m["cohort"], segment=m["segment"],
           source_log_sha256=m["source_log_sha256"], input_proof=expected, input_arrays_sha256=expected["arrays_sha256"],
           output_view_arrays_sha256=base.identity_hash(row["view_arrays"]), input_source_eligible=False,
           eligibility_reasons=m["eligibility_reasons"].copy(), reason_bits=dict(view.REASONS),
           source_eligible=False, selection_eligible=False, production_enabled=False)


def record(route, variant="late", macros=120):
  n = macros*12+4
  t = np.uint64(970_000_000)+np.arange(n, dtype=np.uint64)*np.uint64(10_000_000)
  phase = (np.arange(n)-3) % 12
  delta = np.array([1., .1, 1., .01, 4.]) if variant == "late" else np.array([3., .3, 3., .03, 12.])
  values = np.array(list(STATE.values()))+((phase >= 5) & (np.arange(n) >= 3))[:, None]*delta
  a = {}
  def stream(name, clocks, columns):
    count = len(clocks)
    a[name+"_t_ns"], a[name+"_valid"] = clocks.copy(), np.ones(count, bool)
    for key, value in columns.items():
      a[name+"_"+key], a[name+"_"+key+"_present"] = np.broadcast_to(value, (count,)).astype(float).copy(), np.ones(count, bool)
  stream("carState", t, {"v": values[:, 0], "a": values[:, 1], "steering_angle": values[:, 2],
                        "vendor_rate_unsigned": values[:, 4], "driver_torque": 0., "steering_pressed": 0., "brake_pressed": 0.,
                        "gas_pressed": 0., "fault_temporary": 0., "fault_permanent": 0.})
  li = np.ravel(np.column_stack((3+np.arange(macros)*12, 8+np.arange(macros)*12)))
  stream("liveLocationKalman", t[li], {"yaw": values[li, 3], "roll": .01, "yaw_std": .003, "roll_std": .002,
                                      "yaw_inner_valid": 1., "roll_inner_valid": 1., "status": 2., "inputs_ok": 1.,
                                      "sensors_ok": 1., "device_stable": 1.})
  stream("LKAS11Tx", t, {"torque": 100., "request": 1., "torque_fault": 0.})
  a.update(LKAS11Tx_origin=np.zeros(n, np.int64), LKAS11Tx_src=np.full(n, 128, np.int64),
           LKAS11Tx_address=np.full(n, 0x340, np.int64), LKAS11Tx_decoded_ok=np.ones(n, bool))
  stream("carControl", t, {"lat_active": 1., "long_active": 1.})
  stream("carOutput", t+np.uint64(2_000_000), {"torque_reported": 100/409, "torque_counts": 100.})
  v = {}
  for name in ("carState", "carControl", "carOutput"):
    for key, value in (("event_seq", np.arange(n)), ("frame_seq", np.zeros(n, np.int64)), ("clock_duplicate", np.zeros(n, bool)),
                       ("clock_conflict", np.zeros(n, bool)), ("row_usable", np.ones(n, bool))):
      a[name+"_"+key] = value
    a[name+"_payload"], a[name+"_offsets"] = np.zeros(n*8, np.uint8), np.arange(n+1, dtype=np.int64)*8
    v.update({name+"_"+key: a[name+"_"+key].copy() for key in view.PRESERVED})
    column = {"carState": "can_valid", "carControl": "accel_request", "carOutput": "accel_reported"}[name]
    v[name+"_"+column] = np.ones(n, bool) if name == "carState" else np.full(n, 1. if name == "carControl" or variant == "late" else .5)
    v[name+"_longitudinal_usable"], v[name+"_longitudinal_reason_bits"] = np.ones(n, bool), np.zeros(n, np.uint16)
  source = sha256((route+variant).encode()).hexdigest()
  m = {"route": route, "segment": route+("--0" if variant == "late" else "--1"), "cohort": "synthetic", "source_log_sha256": source,
       "static_identity_sha256": "c"*64, "policy_identity_sha256": "d"*64, "opendbc_gitlink": None, "source_identity_verified": False}
  for kind in ("static", "policy"):
    anchor = {"t_ns": 900_000_000, "valid": True, "identity_sha256": m[kind+"_identity_sha256"], "source_log_sha256": source, "route": route}
    m[kind+"_anchor"] = {**anchor, "sha256": base.identity_hash(anchor)}
  original = {"segment": m["segment"], "cohort": m["cohort"], "source_log_sha256": source, "output_npz_sha256": "e"*64,
              "schema_sha256": {"car.capnp": "a"*64, "log.capnp": "b"*64}, "splits_sha256": "f"*64,
              "source_eligible": False, "eligibility_reasons": ["unknown_opendbc_gitlink"], "opendbc_gitlink": None}
  row = {"metadata": m, "motion_arrays": a, "motion_metadata": original, "view_arrays": v, "view_receipt": {}}
  rebind(row)
  return row


def records():
  return [record(route, variant) for route in ROUTES for variant in ("late", "early")]


def test_five_state_response_is_conditioned_on_required_report_and_updates_all_components():
  model = joint.fit(records(), SPLIT)
  late = joint.predict_macro(model, STATE, [1.]*5, [1.]*5, [100]*5, .01)
  early = joint.predict_macro(model, STATE, [1.]*5, [.5]*5, [100]*5, .01)
  assert late == pytest.approx(dict(zip(STATE, [11., .1, 1., .01, 16.], strict=True)))
  assert early == pytest.approx(dict(zip(STATE, [13., .3, 3., .03, 24.], strict=True)))
  assert len(model["features"]) == 21 and all(len(c["joint_delta_median"]) == 5 for c in model["cells"].values())
  assert model["actual_dt_s"] == {"min": .05, "max": .05}
  assert model["report_conditioned_observational_only"] and not model["selection_eligible"] and not model["production_enabled"]
  assert not model["source_eligible"] and model["candidate"] is None
  assert not model["velocity_acceleration_consistency_validated"] and not model["joint_median_coherence_validated"]


def test_supported_can_changes_v_and_a_with_longitudinal_channels_fixed():
  rows = records()
  for row in rows:
    early = row["metadata"]["segment"].endswith("--1")
    row["view_arrays"]["carOutput_accel_reported"][:] = 1.
    row["motion_arrays"]["LKAS11Tx_torque"][:] = 50. if early else 100.
    rebind(row)
  model = joint.fit(rows, SPLIT)
  more = joint.predict_macro(model, STATE, [1.]*5, [1.]*5, [100]*5, .01)
  less = joint.predict_macro(model, STATE, [1.]*5, [1.]*5, [50]*5, .01)
  assert (more["v"], more["a"]) == pytest.approx((11., .1))
  assert (less["v"], less["a"]) == pytest.approx((13., .3))


def test_independent_command_clocks_are_causal_and_retained():
  model = joint.fit(records(), SPLIT)
  evidence = model["window_evidence"][0]
  assert evidence["requested_t_ns"] != evidence["reported_t_ns"]
  assert all(t <= node for t, node in zip(evidence["reported_t_ns"], evidence["node_t_ns"], strict=True))
  assert evidence["reported_t_ns"][0] == evidence["start_ns"]-8_000_000


def test_intermediate_measured_speed_acceleration_are_not_features_or_candidate_feedback():
  rows = records()
  first = joint.fit(rows, SPLIT)
  for row in rows:
    index = np.arange(len(row["motion_arrays"]["carState_t_ns"]))
    middle = np.isin((index-3) % 12, [1, 2, 3, 4])
    row["motion_arrays"]["carState_v"][middle] = 999.
    row["motion_arrays"]["carState_a"][middle] = -999.
    rebind(row)
  changed = joint.fit(rows, SPLIT)
  assert first["cells"] == changed["cells"] and first["feature_min"] == changed["feature_min"]
  assert joint.predict_macro(changed, STATE, [1.]*5, [1.]*5, [100]*5, .01)["v"] == 11.


@pytest.mark.parametrize("fault", ["original", "view", "metadata", "markers", "source", "receipt"])
def test_original_view_metadata_and_source_bindings_refuse_tampering(fault):
  rows = records()
  row = rows[0]
  if fault == "original":
    row["motion_arrays"]["carState_a"][4] = 99
  elif fault == "view":
    row["view_arrays"]["carOutput_accel_reported"][4] = 99
  elif fault == "metadata":
    row["motion_metadata"]["schema_sha256"]["car.capnp"] = "9"*64
  elif fault == "markers":
    row["view_arrays"]["carControl_event_seq"][4] += 1
    rebind(row)
  elif fault == "source":
    row["metadata"]["source_log_sha256"] = "9"*64
  else:
    row["view_receipt"]["input_arrays_sha256"] = "9"*64
  with pytest.raises(joint.FitError):
    joint.fit(rows, SPLIT)


class Unreadable(dict):
  def _refuse(self, *args, **kwargs):
    pytest.fail("unadmitted arrays accessed")

  __getitem__ = __iter__ = __contains__ = get = keys = values = items = _refuse


@pytest.mark.parametrize("fault", ["heldout", "source_alias", "cohort", "split", "duplicate_train", "nonstring_route"])
def test_global_metadata_and_train_gates_precede_any_array_access(fault):
  rows = records()
  for row in rows:
    row["motion_arrays"] = row["view_arrays"] = Unreadable()
  split = deepcopy(SPLIT)
  if fault == "heldout":
    rows[-1]["metadata"]["route"] = split["test"][0]
  elif fault == "source_alias":
    rows[-1]["metadata"]["source_log_sha256"] = rows[0]["metadata"]["source_log_sha256"]
  elif fault == "cohort":
    rows[-1]["metadata"]["cohort"] = "other"
  elif fault == "duplicate_train":
    split["train"].append(split["train"][0])
  elif fault == "nonstring_route":
    split["train"].append(42)
  else:
    split["validation"] += split["train"]
  with pytest.raises(joint.FitError):
    joint.fit(rows, split)


@pytest.mark.parametrize("fault,reason", [("can_invalid", "carState_longitudinal_invalid"), ("report_invalid", "carOutput_longitudinal_invalid"),
                                         ("long_inactive", "comma_long_inactive"), ("missing_report", "missing_reported_channel"),
                                         ("invalid_cs", "carState_invalid"), ("driver", "nonzero_driver_torque"),
                                         ("roll_std", "liveLocationKalman_missing_or_nonfinite"), ("request", "request_false")])
def test_invalid_interpositions_and_missing_channels_are_not_filled(fault, reason):
  rows = records()
  row = rows[0]
  if fault == "can_invalid":
    row["view_arrays"]["carState_can_valid"][4] = False
    row["view_arrays"]["carState_longitudinal_usable"][4] = False
    row["view_arrays"]["carState_longitudinal_reason_bits"][4] = view.REASONS["can_invalid"]
  elif fault == "report_invalid":
    row["view_arrays"]["carOutput_longitudinal_usable"][4] = False
    row["view_arrays"]["carOutput_longitudinal_reason_bits"][4] = view.REASONS["nonfinite_value"]
  elif fault == "long_inactive":
    row["motion_arrays"]["carControl_long_active"][4] = 0
  elif fault == "missing_report":
    del row["view_arrays"]["carOutput_accel_reported"]
  elif fault == "invalid_cs":
    row["motion_arrays"]["carState_valid"][4] = row["view_arrays"]["carState_valid"][4] = False
  elif fault == "driver":
    row["motion_arrays"]["carState_driver_torque"][4] = .01
  elif fault == "roll_std":
    row["motion_arrays"]["liveLocationKalman_roll_std"][0] = np.nan
  else:
    row["motion_arrays"]["LKAS11Tx_request"][4] = 0
  rebind(row)
  model = joint.fit(rows, SPLIT)
  assert any(reason in e["reasons"] for e in model["exclusions"])


def test_support_bounds_and_negative_state_have_no_extrapolation_or_contact_clipping():
  model = joint.fit(records(), SPLIT)
  assert joint.predict_macro(model, STATE, [1.]*5, None, [100]*5, .01).reason == "reported_channel_required"
  assert isinstance(joint.predict_macro(model, STATE, [9.]*5, [1.]*5, [100]*5, .01), joint.Unsupported)
  small = joint.fit([record(route, macros=10) for route in ROUTES], SPLIT)
  assert not small["supported_cells"] and all("sample_support" in c["rejections"] for c in small["cells"].values())
  for c in model["cells"].values():
    c["joint_delta_median"][0] = -20.
  assert joint.predict_macro(model, STATE, [1.]*5, [1.]*5, [100]*5, .01).reason == "invalid_joint_prediction"
  for kwargs in ({"min_samples": 99}, {"min_routes": 1}):
    with pytest.raises(joint.FitError):
      joint.fit(records(), SPLIT, **kwargs)


def test_unknown_original_pin_cannot_be_enriched_to_a_known_pin_before_arrays():
  rows = records()
  for row in rows:
    row["metadata"].update(opendbc_gitlink="a"*40, source_identity_verified=True)
    row["motion_arrays"] = row["view_arrays"] = Unreadable()
  with pytest.raises(joint.FitError, match="source metadata"):
    joint.fit(rows, SPLIT)


@pytest.mark.parametrize("reason", ["dirty_source", "missing_typed_policy"])
def test_verified_enrichment_never_promotes_original_ineligibility(reason):
  rows = records()
  for row in rows:
    row["metadata"].update(opendbc_gitlink="a"*40, source_identity_verified=True)
    row["motion_metadata"].update(opendbc_gitlink="a"*40, eligibility_reasons=[reason])
    rebind(row)
  model = joint.fit(rows, SPLIT)
  assert model["supported_cells"] and model["source_pin_verified"]
  assert not model["source_eligible"]
  assert all(item["reasons"] == [reason] for item in model["input_eligibility"])


def test_zero_speed_interposition_is_excluded_from_moving_cells():
  rows = records()
  rows[0]["motion_arrays"]["carState_v"][4] = 0.
  rebind(rows[0])
  model = joint.fit(rows, SPLIT)
  assert any("outside_moving_regime" in item["reasons"] for item in model["exclusions"])
  assert not any(item["start_ns"] == 1_000_000_000 and item["source_log_sha256"] == rows[0]["metadata"]["source_log_sha256"]
                 for item in model["window_evidence"])


@pytest.mark.parametrize("phased", ["carState", "carOutput"])
def test_actual_consumed_prefix_predecessor_must_follow_static_policy_anchor(phased):
  rows = records()
  for row in rows:
    for kind in ("static", "policy"):
      body = {k: v for k, v in row["metadata"][kind+"_anchor"].items() if k != "sha256"}
      body["t_ns"] = 980_000_000
      row["metadata"][kind+"_anchor"] = {**body, "sha256": base.identity_hash(body)}
    if phased == "carState":
      row["motion_arrays"]["carState_t_ns"] += np.uint64(5_000_000)
      row["view_arrays"]["carState_t_ns"] = row["motion_arrays"]["carState_t_ns"].copy()
      # Align CO so this case specifically exercises the CS join predecessor.
      row["motion_arrays"]["carOutput_t_ns"] -= np.uint64(2_000_000)
      row["view_arrays"]["carOutput_t_ns"] = row["motion_arrays"]["carOutput_t_ns"].copy()
    rebind(row)
  model = joint.fit(rows, SPLIT)
  assert all(item["start_ns"] != 1_000_000_000 for item in model["window_evidence"])
  assert any(item["start_ns"] == 1_000_000_000 and "before_static_or_policy_prefix" in item["reasons"]
             for item in model["exclusions"])


@pytest.mark.parametrize("fault", ["scalar_delta", "short_delta", "bounds", "cell_bounds", "edges", "support"])
def test_malformed_model_arrays_and_support_fail_closed_without_broadcasting(fault):
  model = joint.fit(records(), SPLIT)
  cell = next(c for c in model["cells"].values() if c["feature_min"][11] == 1.)
  if fault == "scalar_delta":
    cell["joint_delta_median"] = 1.
  elif fault == "short_delta":
    cell["joint_delta_median"] = [1., 2.]
  elif fault == "bounds":
    model["feature_min"] = [0.]
  elif fault == "cell_bounds":
    cell["feature_max"] = [np.nan]*21
  elif fault == "edges":
    model["edges"] = [0.]*21
  else:
    cell["supported"] = "yes"
  assert joint.predict_macro(model, STATE, [1.]*5, [1.]*5, [100]*5, .01).reason == "model_contract"
