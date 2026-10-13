"""Synthetic motion verifies the hypothesis implementation, not the real plant."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

import controller_multirate_lateral as multi


SPLIT = {"train": ["route-a", "route-b"], "validation": ["route-v"], "test": ["route-t"]}


def record(route, variant="late", macros=200):
  pattern, change = ([0, 0, 0, 0, 100], (1., .01, 4.)) if variant == "late" else ([100, 0, 0, 0, 0], (3., .03, 12.))
  n = macros*5+1
  t = np.uint64(1_000_000_000)+np.arange(n, dtype=np.uint64)*np.uint64(10_000_000)
  anchors = np.arange(macros+1)
  states = np.column_stack(((anchors % 2)*change[0], (anchors % 2)*change[1], 12.+(anchors % 2)*change[2]))
  held = states[np.minimum(np.arange(n)//5, macros)]
  arrays = {}
  def stream(name, clocks, values):
    count = len(clocks)
    arrays[name+"_t_ns"] = clocks.copy()
    arrays[name+"_valid"] = np.ones(count, bool)
    for key, value in values.items():
      arrays[name+"_"+key] = np.broadcast_to(value, (count,)).astype(float).copy()
      arrays[name+"_"+key+"_present"] = np.ones(count, bool)
  stream("carState", t, {"v": 15., "steering_angle": held[:, 0], "vendor_rate_unsigned": held[:, 2], "driver_torque": 0.,
                        "steering_pressed": 0., "brake_pressed": 0., "gas_pressed": 0., "fault_temporary": 0., "fault_permanent": 0.})
  stream("liveLocationKalman", t[::5], {"yaw": states[:, 1], "roll": .01, "yaw_std": .003, "roll_std": .002,
                                      "yaw_inner_valid": 1., "roll_inner_valid": 1., "status": 2., "inputs_ok": 1.,
                                      "sensors_ok": 1., "device_stable": 1.})
  commands = np.array([pattern[i % 5] if (i//5) % 2 == 0 else 0 for i in range(n)])
  stream("LKAS11Tx", t, {"torque": commands, "request": 1., "torque_fault": 0.})
  arrays.update(LKAS11Tx_origin=np.zeros(n, np.int64), LKAS11Tx_src=np.full(n, 128, np.int64),
                LKAS11Tx_address=np.full(n, 0x340, np.int64), LKAS11Tx_decoded_ok=np.ones(n, bool))
  stream("carControl", t, {"lat_active": 1.})
  metadata = {"route": route, "cohort": "cohort", "static_identity_sha256": "c"*64, "policy_identity_sha256": "d"*64,
              "source_log_sha256": sha256((route+variant).encode()).hexdigest(), "opendbc_gitlink": None,
              "source_identity_verified": False}
  for kind in ("static", "policy"):
    anchor = {"t_ns": 900_000_000, "valid": True, "identity_sha256": metadata[kind+"_identity_sha256"],
              "source_log_sha256": metadata["source_log_sha256"], "route": route}
    metadata[kind+"_anchor"] = {**anchor, "sha256": multi.identity_hash(anchor)}
  return {"metadata": metadata, "arrays": arrays}


def records():
  return [record(route, variant) for route in SPLIT["train"] for variant in ("late", "early")]


STATE = {"angle": 0., "yaw": 0., "vendor_rate": 12.}


def test_real_50ms_delta_is_not_divided_into_fictitious_10ms_response():
  model = multi.fit(records(), SPLIT)
  prediction = multi.predict_macro(model, STATE, [15.]*5, [0, 0, 0, 0, 100], .01)
  assert prediction == pytest.approx({"angle": 1., "yaw": .01, "vendor_rate": 16.})
  assert model["macro_dt_s"] == .05 and model["native_dt_s"] == .01
  assert model["actual_dt_s"]["min"] == model["actual_dt_s"]["max"] == .05
  assert not model["selection_eligible"] and model["candidate"] is None


def test_five_commands_retain_sequence_effect_despite_identical_mean():
  model = multi.fit(records(), SPLIT)
  early = multi.predict_macro(model, STATE, [15.]*5, [100, 0, 0, 0, 0], .01)
  late = multi.predict_macro(model, STATE, [15.]*5, [0, 0, 0, 0, 100], .01)
  assert early["angle"] == 3. and late["angle"] == 1.
  assert early["yaw"] == .03 and late["yaw"] == .01


def test_prediction_is_frozen_and_never_reads_future_logged_feedback_or_cs_yawrate():
  data = records()
  model = multi.fit(data, SPLIT)
  before = multi.predict_macro(model, STATE, [15.]*5, [0, 0, 0, 0, 100], .01)
  for row in data:
    row["arrays"]["carState_steering_angle"][:] = 999
    row["arrays"]["carState_yawRate"] = np.full(1001, -999.)
    row["arrays"]["liveLocationKalman_yaw"][:] = -999
  assert multi.predict_macro(model, STATE, [15.]*5, [0, 0, 0, 0, 100], .01) == before
  assert before["yaw"] > 0  # Actual LLK sign, not carState.yawRate.


class Unreadable:
  def __getitem__(self, key):
    pytest.fail("heldout or aliased source arrays were accessed")


@pytest.mark.parametrize("fault", ["heldout", "same_source_two_routes"])
def test_all_train_and_physical_route_gates_precede_array_access(fault):
  a, b = record("route-a"), record("route-b")
  a["arrays"] = b["arrays"] = Unreadable()
  if fault == "heldout":
    b["metadata"]["route"] = "route-t"
  else:
    b["metadata"]["source_log_sha256"] = a["metadata"]["source_log_sha256"]
  with pytest.raises(multi.FitError, match="TRAIN|source.*route"):
    multi.fit([a, b], SPLIT)


@pytest.mark.parametrize("fault,reason", [("invalid_cs", "carState_invalid"), ("request", "request_false"),
                                         ("driver", "driver_override"), ("inactive", "lateral_inactive"), ("missing", "missing_fields")])
def test_invalid_interpositions_are_not_skipped(fault, reason):
  item = record("route-a")
  if fault == "invalid_cs":
    item["arrays"]["carState_valid"][1] = False
  elif fault == "request":
    item["arrays"]["LKAS11Tx_request"][2] = 0
  elif fault == "driver":
    item["arrays"]["carState_steering_pressed"][3] = 1
  elif fault == "inactive":
    item["arrays"]["carControl_lat_active"][4] = 0
  else:
    del item["arrays"]["carState_driver_torque"]
  model = multi.fit([item, record("route-b")], SPLIT)
  assert any(reason in entry["reasons"] for entry in model["exclusions"])


def test_duplicate_llk_and_invalid_middle_publication_do_not_bridge_valid_endpoints():
  item = record("route-a")
  arrays = item["arrays"]
  for key in list(arrays):
    if key.startswith("liveLocationKalman_"):
      arrays[key] = np.insert(arrays[key], 1, arrays[key][0])
  arrays["liveLocationKalman_t_ns"][1] += np.uint64(25_000_000)
  arrays["liveLocationKalman_valid"][1] = False
  model = multi.fit([item, record("route-b")], SPLIT)
  exclusions = [e for e in model["exclusions"] if e["source_log_sha256"] == item["metadata"]["source_log_sha256"]]
  assert "macro_dt" in exclusions[0]["reasons"] and "liveLocationKalman_invalid" in exclusions[0]["reasons"]
  assert exclusions[0]["end_ns"] == 1_025_000_000


def test_uint64_unsorted_ties_and_latest_invalid_are_not_older_valid_fallback():
  item = record("route-a")
  arrays = item["arrays"]
  for key in list(arrays):
    if key.startswith("carState_"):
      arrays[key] = np.insert(arrays[key], 0, arrays[key][0])
  arrays["carState_t_ns"][0] -= np.uint64(10_000_000)
  arrays["carState_valid"][1] = False
  # Preserve actual clocks while changing file order; sorting cannot erase the invalid latest sample.
  for key in list(arrays):
    if key.startswith("carState_"):
      arrays[key][[0, 1]] = arrays[key][[1, 0]]
  model = multi.fit([item, record("route-b")], SPLIT)
  assert "carState_invalid" in model["exclusions"][0]["reasons"]
  assert model["clock_order_repairs"] > 0


def test_request_prefix_absence_or_false_cannot_be_replaced_by_new_valid_request():
  item = record("route-a")
  model = multi.fit([item, record("route-b")], SPLIT)
  assert "LKAS11Tx_gap_or_prefix" in model["exclusions"][0]["reasons"]
  arrays = item["arrays"]
  for key in list(arrays):
    if key.startswith("LKAS11Tx_"):
      arrays[key] = np.insert(arrays[key], 0, arrays[key][0])
  arrays["LKAS11Tx_t_ns"][0] -= np.uint64(20_000_000)
  arrays["LKAS11Tx_request"][0] = 0
  model = multi.fit([item, record("route-b")], SPLIT)
  assert "request_false" in model["exclusions"][0]["reasons"]


def test_bounds_missing_cells_route_support_and_unsigned_rate_fail_explicitly():
  model = multi.fit(records(), SPLIT)
  assert multi.predict_macro(model, STATE, [15.]*5, [200, 0, 0, 0, 0], .01).reason == "outside_training_bounds"
  assert multi.predict_macro(model, STATE, [15.]*5, [100, 0, 0, 0, 100], .01).reason == "unsupported_cell"
  assert multi.predict_macro(model, {**STATE, "vendor_rate": -1.}, [15.]*5, [0]*5, .01).reason == "invalid_macro_inputs"
  one_route = multi.fit([record("route-a", v) for v in ("late", "early")], SPLIT)
  assert multi.predict_macro(one_route, STATE, [15.]*5, [0, 0, 0, 0, 100], .01).reason == "cell_support"
  too_few = multi.fit([record(r, macros=40) for r in SPLIT["train"]], SPLIT)
  assert multi.predict_macro(too_few, STATE, [15.]*5, [0, 0, 0, 0, 100], .01).reason == "cell_support"


def test_unknown_source_remains_diagnostic_and_thresholds_cannot_be_relaxed():
  model = multi.fit(records(), SPLIT)
  assert not model["source_eligible"] and not model["hundred_hz_feedback_validated"]
  assert model["production_enabled"] is False
  with pytest.raises(multi.FitError, match="100|two"):
    multi.fit(records(), SPLIT, min_samples=10)
  with pytest.raises(multi.FitError, match="100|two"):
    multi.fit(records(), SPLIT, min_routes=1)
  mixed = records()
  mixed[0]["metadata"]["static_identity_sha256"] = "e"*64
  with pytest.raises(multi.FitError, match="contract"):
    multi.fit(mixed, SPLIT)


def test_physical_roll_is_an_independent_frozen_feature_not_lp_roll():
  model = multi.fit(records(), SPLIT)
  assert multi.predict_macro(model, STATE, [15.]*5, [0, 0, 0, 0, 100], .02).reason == "outside_training_bounds"
  mirrored = deepcopy(records())
  for item in mirrored:
    item["arrays"]["liveLocationKalman_yaw"] *= -1
  right = multi.fit(mirrored, SPLIT)
  assert multi.predict_macro(right, STATE, [15.]*5, [0, 0, 0, 0, 100], .01)["yaw"] == -.01


def test_semantic_cp_data_bytes_and_arrays_have_one_explicit_identity_algorithm():
  cp = {"carFw": [{"fwVersion": b"\x00\xfe"}]}
  assert multi.identity_hash({"cp": cp, "sp": {}}) == multi.identity_hash({"cp": {"carFw": [{"fwVersion": bytearray(b"\x00\xfe")}]}, "sp": {}})
  changed = {"carFw": [{"fwVersion": b"\x00\xff"}]}
  assert multi.identity_hash({"cp": changed, "sp": {}}) != multi.identity_hash({"cp": cp, "sp": {}})
  assert multi.identity_hash(np.zeros((2, 3))) != multi.identity_hash(np.zeros((3, 2)))


def test_static_policy_anchors_are_hash_bound_causal_and_never_invented():
  items = [record(r) for r in SPLIT["train"]]
  for item in items:
    anchor = item["metadata"]["policy_anchor"]
    anchor["t_ns"] = 1_100_000_000
    anchor["sha256"] = multi.identity_hash({k: v for k, v in anchor.items() if k != "sha256"})
  model = multi.fit(items, SPLIT)
  assert model["static_policy_timing_verified"]
  assert "before_static_or_policy_anchor" in model["exclusions"][0]["reasons"]
  del items[0]["metadata"]["static_anchor"]
  model = multi.fit(items, SPLIT)
  assert not model["static_policy_timing_verified"] and not model["source_eligible"]
  assert "static_policy_timing_unknown" in model["exclusions"][0]["reasons"]


def test_explicit_unknown_anchor_retains_named_exclusion_instead_of_crashing():
  items = [record(r) for r in SPLIT["train"]]
  items[0]["metadata"]["static_anchor"] = None
  model = multi.fit(items, SPLIT)
  assert not model["static_policy_timing_verified"]
  assert "static_policy_timing_unknown" in model["exclusions"][0]["reasons"]
