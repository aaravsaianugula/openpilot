"""Real fork-capnp fixtures, never a recorded-data/physical admission claim."""
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import capnp
import numpy as np
import pytest

import controller_motion_longitudinal as view
import controller_motion_tape as motion
from controller_extract import load_schema
from controller_multirate_lateral import identity_hash


@pytest.fixture(scope="module")
def schemas():
  directory = Path(r"D:\comma_four\tuning\lat-tracking\schemas")
  log = load_schema(directory)
  car = capnp.load(str(directory/"car.capnp"), imports=[str(directory), str(directory/"include")])
  hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*.capnp")}
  return log, car, hashes


def event(log, service, timestamp, *, valid=True, accel=None, can_valid=True, absent=False):
  result = log.Event.new_message(logMonoTime=timestamp, valid=valid)
  message = result.init(service)
  if service == "carState":
    message.canValid, message.vEgo, message.aEgo = can_valid, 12., .2
  elif not absent:
    output = message.init("actuators" if service == "carControl" else "actuatorsOutput")
    output.accel = accel
    if service == "carControl":
      message.latActive, message.longActive = True, True
  return result.as_reader()


def proof(arrays, metadata, split):
  return {"cohort": metadata["cohort"], "segment": metadata["segment"], "source_log_sha256": metadata["source_log_sha256"],
          "motion_npz_sha256": metadata["output_npz_sha256"], "arrays_sha256": identity_hash(arrays),
          "metadata_sha256": identity_hash(metadata), "schema_sha256": metadata["schema_sha256"],
          "split_sha256": identity_hash(split), "motion_splits_sha256": metadata["splits_sha256"]}


def fixture(schemas, extra=()):
  log, car, hashes = schemas
  origin = 2**53+1
  events = [event(log, "carState", origin), event(log, "carControl", origin+1_000_000, accel=1.25),
            event(log, "carOutput", origin+3_000_000, accel=.75), event(log, "carState", origin+10_000_000),
            event(log, "carControl", origin+11_000_000, accel=-2.), event(log, "carOutput", origin+13_000_000, accel=-1.5), *extra]
  arrays, metadata = motion.collect(events)
  payload = io.BytesIO()
  np.savez_compressed(payload, **arrays)
  metadata.update(segment="000001b8--a9d69ce1d6--8", cohort="synthetic", source_log_sha256="b"*64,
                  output_npz_sha256=hashlib.sha256(payload.getvalue()).hexdigest(), schema_sha256=hashes, splits_sha256="c"*64)
  split = {"train": ["000001b8--a9d69ce1d6"], "validation": ["000001bf--acf1234567"], "test": ["000001cc--123456789a"]}
  return arrays, metadata, split, car


def derive(inputs, **kwargs):
  arrays, metadata, split, car = inputs
  return view.derive_arrays(arrays, metadata, split, car_schema=car, proof=proof(arrays, metadata, split), **kwargs)


def test_requested_and_post_controller_commands_keep_independent_raw_clocks(schemas):
  inputs = fixture(schemas)
  before = identity_hash(inputs[0])
  result, receipt = derive(inputs)
  np.testing.assert_array_equal(result["carControl_accel_request"], [1.25, -2.])
  np.testing.assert_array_equal(result["carOutput_accel_reported"], [.75, -1.5])
  for service in ("carState", "carControl", "carOutput"):
    for key in view.PRESERVED:
      name = service+"_"+key
      np.testing.assert_array_equal(result[name], inputs[0][name])
      assert not np.shares_memory(result[name], inputs[0][name])
  assert result["carState_t_ns"][0] == 2**53+1 and result["carState_t_ns"].dtype == np.uint64
  assert not np.array_equal(result["carControl_t_ns"], result["carOutput_t_ns"])
  assert identity_hash(inputs[0]) == before
  assert receipt["source_eligible"] is False and receipt["selection_eligible"] is False and not receipt["production_enabled"]
  assert "unknown_opendbc_gitlink" in receipt["eligibility_reasons"]
  assert receipt["input_arrays_sha256"] == before


def test_actual_can_invalid_and_invalid_interposed_events_are_retained(schemas):
  log = schemas[0]
  inputs = fixture(schemas, [event(log, "carState", 2**53+1+5_000_000, can_valid=False),
                             event(log, "carControl", 2**53+1+6_000_000, valid=False, accel=.5)])
  result, receipt = derive(inputs)
  assert result["carState_can_valid"].tolist() == [True, False, True]
  assert result["carState_longitudinal_usable"].tolist() == [True, False, True]
  assert result["carControl_longitudinal_usable"].tolist() == [True, False, True]
  assert result["carState_longitudinal_reason_bits"][1] & view.REASONS["can_invalid"]
  assert result["carControl_longitudinal_reason_bits"][1] & view.REASONS["event_invalid"]
  assert receipt["streams"]["carState"]["rows"] == 3


@pytest.mark.parametrize("fault", ["absent_actuators", "nonfinite_accel", "zero_accel"])
@pytest.mark.parametrize("service,column", [("carControl", "accel_request"), ("carOutput", "accel_reported")])
def test_missing_actuator_pointer_is_distinct_from_published_scalar_zero(schemas, fault, service, column):
  log = schemas[0]
  extra = event(log, service, 2**53+1+6_000_000, accel=np.nan if fault == "nonfinite_accel" else 0.,
                absent=fault == "absent_actuators")
  result, receipt = derive(fixture(schemas, [extra]))
  value = result[service+"_"+column][1]
  if fault == "zero_accel":
    assert value == 0 and result[service+"_longitudinal_usable"][1]
  else:
    assert np.isnan(value) and not result[service+"_longitudinal_usable"][1]
    reason = "actuator_struct_absent" if fault == "absent_actuators" else "nonfinite_value"
    assert result[service+"_longitudinal_reason_bits"][1] & view.REASONS[reason]
    assert receipt["streams"][service]["reasons"][reason] == 1


def test_conflicting_duplicate_publications_preserve_order_and_are_unusable(schemas):
  inputs = fixture(schemas, [event(schemas[0], "carControl", 2**53+1+1_000_000, accel=-3.)])
  result, _ = derive(inputs)
  assert result["carControl_accel_request"].tolist() == [1.25, -3., -2.]
  assert result["carControl_event_seq"].tolist() == [1, 6, 4]
  assert result["carControl_clock_conflict"].tolist() == [True, True, False]
  assert result["carControl_longitudinal_usable"].tolist() == [False, False, True]


@pytest.mark.parametrize("fault", ["offsets", "payload", "missing_stream", "bad_clock", "bad_valid", "bad_duplicates", "object_array"])
def test_malformed_raw_contract_or_payload_refuses_without_partial_view(schemas, fault):
  inputs = fixture(schemas)
  arrays = inputs[0]
  bound = proof(*inputs[:3])
  if fault == "offsets":
    arrays["carOutput_offsets"][-1] += 1
  elif fault == "payload":
    arrays["carControl_payload"] = np.asarray([1, 2, 3], np.uint8)
    arrays["carControl_offsets"] = np.asarray([0, 1, 3], np.int64)
  elif fault == "missing_stream":
    del arrays["carState_payload"]
  elif fault == "bad_clock":
    arrays["carState_t_ns"] = arrays["carState_t_ns"].astype(float)
  elif fault == "bad_valid":
    arrays["carState_valid"] = arrays["carState_valid"].astype(np.uint8)
  elif fault == "bad_duplicates":
    arrays["carState_clock_duplicate"][:] = True
  else:
    arrays["unexpected"] = np.asarray([object()], object)
  with pytest.raises(view.ViewError):
    if fault == "object_array":
      view.derive_arrays(*inputs[:3], car_schema=inputs[3], proof=bound)
    else:
      derive(inputs)


@pytest.mark.parametrize("fault", ["heldout", "route_alias", "schema_hash", "input_hash", "source_hash", "schema_type", "metadata_contract"])
def test_train_source_schema_and_content_binding_refuse_before_decode(schemas, monkeypatch, fault):
  inputs = fixture(schemas)
  arrays, metadata, split, car = inputs
  bound = proof(arrays, metadata, split)
  if fault == "heldout":
    split["test"] += split["train"]
    split["train"] = []
    bound["split_sha256"] = identity_hash(split)
  elif fault == "route_alias":
    metadata["route"] = "other-route"
    bound = proof(arrays, metadata, split)
  elif fault == "schema_hash":
    bound["schema_sha256"] = {**bound["schema_sha256"], "car.capnp": "f"*64}
  elif fault == "input_hash":
    arrays["carControl_t_ns"][0] += np.uint64(1)
  elif fault == "source_hash":
    bound["source_log_sha256"] = "f"*64
  elif fault == "metadata_contract":
    metadata["eligibility_reasons"] = None
    bound = proof(arrays, metadata, split)
  else:
    car = SimpleNamespace(CarControl=car.CarControl, CarState=car.CarState)
  monkeypatch.setattr(view, "decode_payload", lambda *a: pytest.fail("payload decode before metadata/schema admission"))
  with pytest.raises(view.ViewError):
    view.derive_arrays(arrays, metadata, split, car_schema=car, proof=bound)


def test_original_input_quality_is_not_promoted_by_new_scalar(schemas):
  inputs = fixture(schemas)
  inputs[0]["carOutput_row_usable"][0] = False
  result, _ = derive(inputs)
  assert result["carOutput_accel_reported"][0] == .75 and not result["carOutput_longitudinal_usable"][0]
  assert result["carOutput_longitudinal_reason_bits"][0] & view.REASONS["original_row_unusable"]


def test_schema_hash_receipt_does_not_alias_caller_metadata(schemas):
  inputs = fixture(schemas)
  inputs[1]["schema_sha256"] = dict(inputs[1]["schema_sha256"])
  _, receipt = derive(inputs)
  expected = receipt["input_proof"]["schema_sha256"]["car.capnp"]
  inputs[1]["schema_sha256"]["car.capnp"] = "f"*64
  assert receipt["input_proof"]["schema_sha256"]["car.capnp"] == expected
