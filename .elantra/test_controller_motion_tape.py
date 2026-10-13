from hashlib import sha256
import json
import os
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

import controller_motion_tape as motion


ROUTE = "000001b8--a9d69ce1d6"
SEGMENT = ROUTE+"--8"


class Struct(NS):
  def as_builder(self):
    return self
  def to_bytes(self):
    return b"full-struct-payload"


def event(name, t, valid=True, **values):
  return NS(which=lambda: name, logMonoTime=t, valid=valid, **{name: Struct(**values)})


def cs(t, angle=1., rate=12., valid=True):
  return event("carState", t, valid, vEgo=15., aEgo=.1, steeringAngleDeg=angle, steeringRateDeg=rate,
               steeringTorque=0., steeringPressed=False, brakePressed=False, gasPressed=False, standstill=False,
               steerFaultTemporary=False, steerFaultPermanent=False)


def llk(t, valid=True):
  return event("liveLocationKalman", t, valid, angularVelocityCalibrated=NS(value=[0., 0., .02], std=[.01, .01, .003], valid=True),
               calibratedOrientationNED=NS(value=[.01, 0., 0.], std=[.002, .003, .004], valid=True), status="valid",
               inputsOK=True, sensorsOK=True, deviceStable=True, timeSinceReset=5.)


def frame(address, src, data):
  return NS(address=address, src=src, dat=data)


def can_event(name, t, frames):
  return NS(which=lambda: name, logMonoTime=t, valid=True, **{name: frames})


def lkas(torque, request):
  return (((torque+1024) << 16) | (request << 27)).to_bytes(8, "little")


def files(tmp_path, partition="train"):
  raw = tmp_path/"routes"
  log = raw/SEGMENT/"rlog.zst"
  log.parent.mkdir(parents=True)
  log.write_bytes(b"not-decoded-by-fixtures")
  schema = tmp_path/"schema"
  schema.mkdir()
  (schema/"log.capnp").write_text("schema")
  selection = tmp_path/"selection.json"
  selection.write_text(json.dumps({"cohort": "test", "segments": [{"segment": SEGMENT, "source_log_sha256": sha256(log.read_bytes()).hexdigest()}]}))
  splits = tmp_path/"splits.json"
  splits.write_text(json.dumps({"cohorts": {"test": {"split": {"train": [ROUTE] if partition == "train" else [],
                                                              "validation": [], "test": [] if partition == "train" else [ROUTE]}}}}))
  return selection, splits, schema, raw


def test_heldout_refused_before_decode_or_schema_load(tmp_path, monkeypatch):
  selection, splits, schema, raw = files(tmp_path, "test")
  monkeypatch.setattr(motion, "iter_events", lambda *a: pytest.fail("heldout was decoded"))
  monkeypatch.setattr(motion, "load_schema", lambda *a: pytest.fail("schema load happened before TRAIN admission"))
  with pytest.raises(motion.TapeError, match="TRAIN"):
    motion.build(selection, splits, schema, raw, tmp_path/"out")
  assert not (tmp_path/"out").exists()


def test_raw_clocks_invalid_rows_and_conflicting_ties_are_preserved():
  arrays, report = motion.collect([cs(20_000_000), cs(10_000_000, valid=False), cs(20_000_000, angle=2.), llk(15_000_000)])
  assert arrays["carState_t_ns"].dtype == np.uint64
  assert arrays["carState_t_ns"].tolist() == [10_000_000, 20_000_000, 20_000_000]
  assert arrays["carState_valid"].tolist() == [False, True, True]
  assert arrays["carState_event_seq"].tolist() == [1, 0, 2]
  assert report["cadence"]["carState"]["duplicate_timestamps"] == 1
  assert report["cadence"]["carState"]["conflicting_timestamps"] == 1
  assert not arrays["carState_row_usable"].any()
  assert len(arrays["liveLocationKalman_t_ns"]) == 1  # No artificial 100Hz rows.


def test_unsigned_vendor_rate_is_retained_without_signed_derivative_substitution():
  arrays, report = motion.collect([cs(0, angle=4., rate=12.), cs(50_000_000, angle=3., rate=12.)])
  assert arrays["carState_steering_angle"].tolist() == [4., 3.]
  assert arrays["carState_vendor_rate_unsigned"].tolist() == [12., 12.]
  assert "signed_steering_rate" not in arrays
  assert report["semantics"]["vendor_rate_unsigned"].startswith("Recorded carState.steeringRateDeg")


def test_llk_uncertainty_and_missing_schema_fields_are_explicit():
  item = llk(50_000_000)
  del item.liveLocationKalman.sensorsOK
  arrays, report = motion.collect([item])
  assert arrays["liveLocationKalman_yaw_std"].tolist() == [.003]
  assert arrays["liveLocationKalman_roll_std"].tolist() == [.002]
  assert np.isnan(arrays["liveLocationKalman_sensors_ok"][0])
  assert not arrays["liveLocationKalman_sensors_ok_present"][0]
  assert report["missing_fields"]["liveLocationKalman"]["sensors_ok"] == 1
  assert not arrays["liveLocationKalman_row_usable"][0]


def test_can_bus_request_and_event_origin_are_not_aggregated_or_conflated():
  arrays, report = motion.collect([
    can_event("can", 10, [frame(0x340, 128, lkas(409, 1)), frame(0x340, 128, lkas(-4, 0)),
                          frame(0x251, 0, (1 << 14).to_bytes(8, "little")), frame(0x251, 1, bytes(8)), frame(0x340, 0, lkas(9, 1))]),
    can_event("sendcan", 20, [frame(0x340, 0, lkas(5, 1)), frame(0x340, 128, lkas(6, 1)), frame(0x340, 1, lkas(7, 1)),
                              frame(0x251, 0, bytes(8)), frame(0x2B0, 0, bytes(5))])])
  assert arrays["LKAS11Tx_torque"].tolist() == [409., -4., 5.]
  assert arrays["LKAS11Tx_request"].tolist() == [1., 0., 1.]
  assert arrays["LKAS11Tx_origin"].tolist() == [0, 0, 1]
  assert arrays["MDPS12Rx_fault"].tolist() == [1.]
  assert report["ignored_can_bus_frames"] == 6


def test_invalid_sendcan_bus0_request_remains_invalid_and_is_not_delivery_proof():
  item = can_event("sendcan", 10, [frame(0x340, 0, lkas(100, 1))])
  item.valid = False
  arrays, report = motion.collect([item])
  assert arrays["LKAS11Tx_torque"].tolist() == [100.]
  assert arrays["LKAS11Tx_src"].tolist() == [0]
  assert arrays["LKAS11Tx_valid"].tolist() == [False]
  assert not arrays["LKAS11Tx_row_usable"].any()
  assert "neither" in report["semantics"]["can_origin"].lower()


def test_short_can_frame_is_preserved_and_marked_invalid():
  arrays, report = motion.collect([can_event("can", 10, [frame(0x340, 128, b"\x01")])])
  assert not arrays["LKAS11Tx_decoded_ok"][0]
  assert np.isnan(arrays["LKAS11Tx_torque"][0])
  assert arrays["LKAS11Tx_payload"].dtype == np.uint8
  assert report["short_can_frames"] == 1


def test_unknown_pin_and_declared_pin_cannot_become_verified_source_identity():
  init = event("initData", 0, gitCommit="a"*40, dirty=True, deviceType="mici", params=NS(entries=[]))
  arrays, report = motion.collect([init, cs(10), llk(20)], declared={"opendbc_gitlink": "b"*40})
  assert not report["source_eligible"]
  assert report["opendbc_gitlink"] is None and report["declared_opendbc_gitlink"] == "b"*40
  assert "unknown_opendbc_gitlink" in report["eligibility_reasons"]
  assert "dirty_recorded_source" in report["eligibility_reasons"]


def test_full_static_payloads_and_three_typed_params_are_retained_without_secrets():
  entries = [NS(key=k, value=v) for k, v in [("CustomTorqueParams", b"0"), ("LiveTorqueParamsToggle", b"1"),
                                            ("LiveTorqueParamsRelaxedToggle", b"0"), ("AccessToken", b"secret")]]
  events = [event("carParams", 0, carFingerprint="HYUNDAI_ELANTRA_2024"), event("carParamsSP", 0),
            event("initData", 0, gitCommit="a"*40, dirty=False, deviceType="mici", params=NS(entries=entries))]
  arrays, report = motion.collect(events)
  assert bytes(arrays["carParams_payload"]) == b"full-struct-payload"
  assert report["params_snapshots"][0]["values"]["LiveTorqueParamsToggle"] == "1"
  assert "AccessToken" not in json.dumps(report) and "secret" not in json.dumps(report)
  assert all(value.dtype.kind != "O" for value in arrays.values())


def test_conflicting_typed_entries_are_retained_and_never_silently_last_wins():
  entries = [NS(key="LiveTorqueParamsToggle", value=b"0"), NS(key="LiveTorqueParamsToggle", value=b"1")]
  item = event("initData", 1, gitCommit="a"*40, dirty=False, deviceType="mici", params=NS(entries=entries))
  _, report = motion.collect([item])
  snapshot = report["params_snapshots"][0]
  assert [row["value"] for row in snapshot["entries"]] == ["0", "1"]
  assert "LiveTorqueParamsToggle" not in snapshot["values"]
  assert "conflicting_typed_params" in report["eligibility_reasons"]


@pytest.mark.parametrize("location", ["existing", "raw", "dataset"])
def test_output_guards_refuse_before_decode(tmp_path, monkeypatch, location):
  selection, splits, schema, raw = files(tmp_path)
  dataset = tmp_path/"dataset-recent"
  dataset.mkdir()
  out = {"existing": tmp_path/"existing", "raw": raw/"new-output", "dataset": dataset/"new-output"}[location]
  if location == "existing":
    out.mkdir()
    (out/"preserve").write_text("unchanged")
  monkeypatch.setattr(motion, "iter_events", lambda *a: pytest.fail("invalid output decoded input"))
  with pytest.raises(motion.TapeError, match="output"):
    motion.build(selection, splits, schema, raw, out, dataset_roots=(dataset,))


def test_exclusive_numeric_artifact_and_failed_second_build_do_not_overwrite(tmp_path, monkeypatch):
  selection, splits, schema, raw = files(tmp_path)
  monkeypatch.setattr(motion, "load_schema", lambda path: NS())
  monkeypatch.setattr(motion, "iter_events", lambda *a: iter([cs(10), llk(20)]))
  out = tmp_path/"out"
  result = motion.build(selection, splits, schema, raw, out)
  assert result["completed"] == 1 and result["failed"] == 0
  path = out/(SEGMENT+".npz")
  before = sha256(path.read_bytes()).hexdigest()
  with np.load(path, allow_pickle=False) as archive:
    assert archive["carState_t_ns"].tolist() == [10]
  with pytest.raises(motion.TapeError, match="output"):
    motion.build(selection, splits, schema, raw, out)
  assert sha256(path.read_bytes()).hexdigest() == before


def test_input_change_during_decode_is_retained_failure_with_no_tape(tmp_path, monkeypatch):
  selection, splits, schema, raw = files(tmp_path)
  monkeypatch.setattr(motion, "load_schema", lambda path: NS())
  def changing(*args):
    yield cs(10)
    selection.write_text(selection.read_text()+" ")
  monkeypatch.setattr(motion, "iter_events", changing)
  out = tmp_path/"out"
  result = motion.build(selection, splits, schema, raw, out)
  assert result["failed"] == 1 and "changed" in result["results"][0]["error"]
  assert not (out/(SEGMENT+".npz")).exists()


def directory_alias(link, target):
  try:
    link.symlink_to(target, target_is_directory=True)
  except OSError:
    if os.name != "nt":
      raise
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("alias", ["directory", "file", "hardlink", "duplicate_physical_file"])
def test_train_alias_to_heldout_or_duplicate_file_refused_before_any_decode(tmp_path, monkeypatch, alias):
  selection, splits, schema, raw = files(tmp_path)
  train_log = raw/SEGMENT/"rlog.zst"
  heldout_route = "000001d8--1122334455"
  target = raw/(heldout_route+"--1")/"rlog.zst"
  target.parent.mkdir()
  target.write_bytes(train_log.read_bytes())
  split = json.loads(splits.read_text())
  split["cohorts"]["test"]["split"]["test"] = [heldout_route]
  splits.write_text(json.dumps(split))
  train_log.unlink()
  if alias == "directory":
    train_log.parent.rmdir()
    directory_alias(train_log.parent, target.parent)
  elif alias == "file":
    try:
      train_log.symlink_to(target)
    except OSError:
      if os.name != "nt":
        raise
      # Junctions exercise the same component guard when file symlinks need privilege.
      train_log.parent.rmdir()
      directory_alias(train_log.parent, target.parent)
  else:
    train_log.hardlink_to(target)
    if alias == "duplicate_physical_file":
      second = raw/(ROUTE+"--9")/"rlog.zst"
      second.parent.mkdir()
      second.hardlink_to(target)
      selected = json.loads(selection.read_text())
      selected["segments"].append({"segment": second.parent.name, "source_log_sha256": sha256(target.read_bytes()).hexdigest()})
      selection.write_text(json.dumps(selected))
  monkeypatch.setattr(motion, "iter_events", lambda *a: pytest.fail("aliased physical source was decoded"))
  monkeypatch.setattr(motion, "load_schema", lambda *a: pytest.fail("aliased source reached schema parsing"))
  with pytest.raises(motion.TapeError, match="alias|hardlink|physical"):
    motion.build(selection, splits, schema, raw, tmp_path/"out")
  assert not (tmp_path/"out").exists()
