"""Regressions for explicit, source-proven missing-prefix repair only."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import controller_extract as extract
import controller_route_seams as seams


def tape():
  t = np.array([10., 10.05, 10.1, 11.05])
  data = {"t": t}
  for service, fields in extract.SERVICE_FIELDS.items():
    data[extract.VALID_FIELDS[service]] = np.ones(len(t))
    data["age_" + service] = np.full(len(t), .05)
    for field in fields:
      shape = (len(t), extract.VECTOR_WIDTHS[field]) if field in extract.VECTOR_WIDTHS else len(t)
      data[field] = np.zeros(shape)
  for field in ("v", "lat_active", "long_active", "location_measurement_valid", "calibrated"):
    data[field][:] = 1
  data["calibration_t"] = t - .05
  data["calibration_height"][:] = 1.2
  data.update(valid_lat=np.zeros(len(t), bool), valid_long=np.zeros(len(t), bool), valid=np.zeros(len(t), bool),
              raw_encode_frame_id=np.array([43], dtype=np.uint64), raw_encode_t=np.array([9.95]))
  return data


def missing(data, service, indices=(0,)):
  for field in (*extract.SERVICE_FIELDS[service], extract.VALID_FIELDS[service], "age_" + service):
    data[field][list(indices)] = np.nan


def control(t, valid=True, command=.2):
  return extract.Sample("carControl", t, {"torque_request": 12., "torque_fraction": .03, "accel_request": command,
                        "lat_active": 1., "long_active": 1.}, valid)


def test_positive_adjacent_prefix_keeps_grid_and_raw_capture_mapping():
  data = tape()
  for service in ("carControl", "liveLocationKalman", "extrinsicsCalibration", "modelV2"):
    missing(data, service)
  previous = [control(9.99), extract.Sample("liveLocationKalman", 9.99, {"yaw": .01, "roll": .02, "location_measurement_valid": 1.}, True),
    extract.Sample("extrinsicsCalibration", 9.7, {"calibrated": 1., "rpy_calib": [.01,.02,.03], "calibration_t": 9.7,
                                               "calibration_height": [1.2]}, True),
    extract.Sample("modelV2", 9.99, {"model_t": 9.99, "model_frame_id": 42, "model_x": [0.,1.], "model_y": [0.,0.],
                                    "model_z": [0.,0.], "model_horizon": [0.,1.]}, True)]
  original = deepcopy(data)
  derived, receipt = seams.repair_prefix(data, previous, [control(10.01)])
  assert np.array_equal(derived["t"], original["t"])
  assert derived["long_active"][0] == 1 and derived["calibration_valid"][0] == 1
  assert derived["valid_long"][0] and derived["valid_lat"][0]
  assert derived["age_extrinsicsCalibration"][0] == pytest.approx(.3)
  assert np.array_equal(derived["raw_encode_frame_id"], original["raw_encode_frame_id"])
  assert derived["model_frame_id"][0] == 42 and 42 not in derived["raw_encode_frame_id"]
  assert "model_capture_t" not in derived
  assert receipt["affected_rows"] == [0]
  assert all(r["source_t"] <= r["t"] and r["event_valid"] for r in receipt["repairs"])
  for key in original:
    np.testing.assert_equal(data[key], original[key])
    if len(np.shape(original[key])) and np.shape(original[key])[0] == len(data["t"]):
      np.testing.assert_equal(derived[key][1:], original[key][1:])


@pytest.mark.parametrize("fault", ["invalid_latest", "stale_latest", "future_only", "actual_invalid_current", "measurement_invalid"])
def test_never_replaces_latest_bad_source_with_older_good_value(fault):
  data = tape()
  service = "liveLocationKalman" if fault == "measurement_invalid" else "carControl"
  missing(data, service)
  previous, current = [control(9.85)], []
  if fault == "invalid_latest":
    previous.append(control(9.99, valid=False))
  elif fault == "stale_latest":
    previous = [control(9.79)]
  elif fault == "future_only":
    previous = [control(10.01)]
  elif fault == "actual_invalid_current":
    current = [control(9.99, valid=False)]
    data["age_carControl"][0], data["car_control_valid"][0] = .01, 0
  else:
    previous = [extract.Sample(service, 9.99, {"yaw": .1, "roll": .2, "location_measurement_valid": 0.}, True)]
  original = deepcopy(data)
  derived, receipt = seams.repair_prefix(data, previous, current)
  for key in original:
    np.testing.assert_equal(derived[key], original[key])
  assert not receipt["affected_rows"]
  if fault != "actual_invalid_current":
    assert receipt["refusals"]


def test_missing_current_prefix_disagreement_fails_closed():
  data = tape()
  missing(data, "carControl")
  derived, receipt = seams.repair_prefix(data, [control(9.99)], [control(9.999, valid=False)])
  assert np.isnan(derived["long_active"][0])
  assert any(r["reason"] == "current_log_already_has_source_event" for r in receipt["refusals"])


def test_new_invalid_event_after_first_repair_is_never_held():
  data = tape()
  missing(data, "carControl", (0,))
  data["car_control_valid"][1], data["long_active"][1], data["age_carControl"][1] = 0, np.nan, .03
  derived, receipt = seams.repair_prefix(data, [control(9.99)], [control(10.02, valid=False)])
  assert derived["long_active"][0] == 1
  assert derived["car_control_valid"][1] == 0 and np.isnan(derived["long_active"][1])
  assert receipt["affected_rows"] == [0]


def test_conflicting_raw_events_at_same_timestamp_are_ambiguous():
  data = tape()
  missing(data, "carControl")
  derived, receipt = seams.repair_prefix(data, [control(9.99),control(9.99,command=-1.)], [])
  assert np.isnan(derived["long_active"][0])
  assert any(r["reason"] == "conflicting_latest_source_events" for r in receipt["refusals"])


def test_repair_is_bounded_and_does_not_add_absent_contract_fields():
  data = tape()
  missing(data, "carControl", (0,3))
  del data["torque_fraction"]
  derived, receipt = seams.repair_prefix(data, [control(9.99),control(11.04)], [])
  assert derived["long_active"][0] == 1 and np.isnan(derived["long_active"][3])
  assert "torque_fraction" not in derived and receipt["affected_rows"] == [0]


def test_validity_recompute_cannot_hold_an_unrepaired_stale_calibration():
  data = tape()
  missing(data, "carControl")
  data["age_extrinsicsCalibration"][0] = .3
  for field in ("calibration_valid", "calibrated", "calibration_t"):
    data[field][0] = np.nan
  derived, _ = seams.repair_prefix(data, [control(9.99)], [])
  assert np.isnan(derived["calibration_valid"][0]) and not derived["valid_long"][0]


@pytest.mark.parametrize("age", [0., 1.01, float("nan")])
def test_invalid_prefix_budget_rejected(age):
  with pytest.raises(seams.SeamError):
    seams.repair_prefix(tape(), [], [], max_prefix_s=age)


def metadata():
  return [{"route":"000001b1--18b1c22fda", "segment":"000001b1--18b1c22fda--"+str(n),
           "git_commit":"a"*40, "dirty":True,"schema_sha256":{"log.capnp":"b"*64},
           "car_params":{"mass":1404.}, "car_params_sp":{}, "params":{}, "fingerprint":"HYUNDAI_ELANTRA_2024", "op_long":True}
          for n in (24,25)]


@pytest.mark.parametrize("fault", ["source", "full_cp", "full_sp", "heldout", "wrong_pair", "missing_proof", "future_cp", "future_sp"])
def test_source_and_full_contract_change_or_heldout_refused(fault):
  metas = metadata()
  contract = {"car_params_snapshots":[{"t":0.,"semantic_sha256":"c"*64}],
              "car_params_sp_snapshots":[{"t":0.,"semantic_sha256":"d"*64}]}
  contracts, train = [deepcopy(contract),deepcopy(contract)], [metas[0]["route"]]
  if fault == "source":
    metas[1]["git_commit"] = "e"*40
  elif fault == "full_cp":
    contracts[1]["car_params_snapshots"][0]["semantic_sha256"] = "e"*64
  elif fault == "full_sp":
    contracts[1]["car_params_sp_snapshots"] = []
  elif fault == "heldout":
    train = []
  elif fault == "missing_proof":
    contracts.pop()
  elif fault in ("future_cp", "future_sp"):
    field = "car_params_snapshots" if fault == "future_cp" else "car_params_sp_snapshots"
    for proof in contracts:
      proof[field][0]["t"] = 11.
  else:
    metas[1]["segment"] = metas[1]["route"]+"--26"
  with pytest.raises(seams.SeamError):
    seams.verify_pair(metas,contracts,train,first_t=10.)


def test_output_cannot_collide_with_originals_or_nonempty_directory(tmp_path):
  raw, dataset = tmp_path/"routes", tmp_path/"dataset"
  raw.mkdir()
  dataset.mkdir()
  for out in (raw/"new",dataset/"new"):
    with pytest.raises(seams.SeamError):
      seams.output_directory(out,raw,dataset)
  used=tmp_path/"used"
  used.mkdir()
  (used/"old.json").write_text("original")
  with pytest.raises(seams.SeamError):
    seams.output_directory(used,raw,dataset)
  out=seams.output_directory(tmp_path/"derived",raw,dataset)
  assert out == Path(tmp_path/"derived").resolve()


@pytest.mark.parametrize("changed_source", [False,True])
def test_evaluation_matches_original_census_static_context_admission(monkeypatch,changed_source):
  seed=metadata()[0]
  seed["segment_index"]=24
  orphan={**deepcopy(seed),"segment":seed["route"]+"--32","segment_index":32,"car_params":{},"fingerprint":None,"op_long":None,
          "summary":{"rows":100},"unrelated_signal_contract":"unchanged"}
  if changed_source:
    orphan["git_commit"]="f"*40
  monkeypatch.setattr(seams,"COHORT",seams.cohort_key(seed))
  members,excluded=seams.evaluation_members([(Path("seed.npz"),seed),(Path("orphan.npz"),orphan)])
  assert len(members)==(1 if changed_source else 2)
  assert bool(excluded)==changed_source
  if not changed_source:
    assert orphan["route_provenance_source"]==seed["segment"] and orphan["car_params"]==seed["car_params"]
  assert orphan["summary"]=={"rows":100} and orphan["unrelated_signal_contract"]=="unchanged"
