"""Regression tests for the standalone, causal controller log extractor."""


import numpy as np
import pytest

from controller_extract import ExtractionError, Sample, align_samples, decompress_log, inventory, select_segments


def state(t, v=8.0, valid=True):
  return Sample("carState", t, {"v": v, "a": 0.0, "steering_pressed": 0.0,
                              "brake_pressed": 0.0, "gas_pressed": 0.0}, valid)


def test_join_is_causal_sorted_and_does_not_backfill():
  arrays, _ = align_samples([state(1.1, 11.0), state(1.0, 10.0),
                             Sample("carControl", 1.075, {"torque_request": 204.5}, True)])
  np.testing.assert_allclose(arrays["t"], [1.0, 1.05, 1.1])
  np.testing.assert_allclose(arrays["v"], [10, 10, 11])
  assert np.isnan(arrays["torque_request"][:2]).all()
  assert arrays["torque_request"][2] == 204.5


def test_join_rejects_stale_data_and_keeps_invalid_event_explicit():
  arrays, _ = align_samples([state(1.0, valid=False), state(1.35, 9.0)])
  assert arrays["v"][4] == 8.0  # Exactly 200 ms is allowed.
  assert np.isnan(arrays["v"][5:7]).all()
  assert arrays["car_state_valid"][0] == 0.0
  assert not arrays["valid"].any()


def test_no_car_state_is_an_error_instead_of_an_empty_success():
  with pytest.raises(ExtractionError, match="carState"):
    align_samples([Sample("carControl", 1.0, {"torque_request": 0.0}, True)])


def test_model_horizon_is_not_replaced_with_invented_times():
  arrays, _ = align_samples([state(1), state(1.1), Sample("modelV2", 1,
                             {"model_x": [0, 1], "model_y": [0, -1], "model_z": [0, 0],
                              "model_horizon": [0, 0.05], "model_frame_id": 21.0, "model_t": 1.0}, True)])
  assert arrays["model_x"].shape == (3, 33)
  np.testing.assert_array_equal(arrays["model_y"][0, :2], [0, -1])
  assert np.isnan(arrays["model_x"][:, 2:]).all()
  assert arrays["model_horizon"][0, 1] == 0.05


def test_telemetry_and_transmitted_can_torque_remain_independent():
  arrays, _ = align_samples([state(1), state(1.1),
                             Sample("carOutput", 1, {"torque": 350.0}, True),
                             Sample("canTorque", 1, {"can_torque": 409.0}, True)])
  assert arrays["torque"][0] == 350
  assert arrays["can_torque"][0] == 409


def test_corrupt_or_truncated_compressed_log_fails(tmp_path):
  import zstandard

  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(b"valid bytes")[:-2])
  with pytest.raises(ExtractionError, match="truncated|corrupt"):
    decompress_log(path)
  path.write_bytes(b"garbage")
  with pytest.raises(ExtractionError, match="truncated|corrupt"):
    decompress_log(path)


def test_concatenated_zstd_frames_are_all_decoded(tmp_path):
  import zstandard

  path = tmp_path / "rlog.zst"
  encoder = zstandard.ZstdCompressor()
  path.write_bytes(encoder.compress(b"one") + encoder.compress(b"two"))
  assert decompress_log(path) == b"onetwo"


@pytest.fixture(scope="module")
def tiny_schema(tmp_path_factory):
  import capnp

  path = tmp_path_factory.mktemp("schema") / "extract_test.capnp"
  path.write_text("""@0xe4c948b754f4c1d8;
struct Event {
  logMonoTime @0 :UInt64;
  valid @1 :Bool = true;
  union { carState @2 :CarState; reserved @3 :Void; }
}
struct CarState {
  vEgo @0 :Float32; aEgo @1 :Float32;
  steeringPressed @2 :Bool; brakePressed @3 :Bool; gasPressed @4 :Bool;
  steeringTorque @5 :Float32; steeringAngleDeg @6 :Float32;
}
""", encoding="utf-8")
  return capnp.load(str(path))


def test_real_schema_decode_preserves_invalid_flags_and_counts_missing_services(tmp_path, tiny_schema):
  import zstandard
  from controller_extract import decode_log

  event = tiny_schema.Event.new_message(logMonoTime=1_000_000_000, valid=False)
  event.init("carState").vEgo = 8.0
  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(event.to_bytes()))
  samples, metadata = decode_log(path, tiny_schema)
  arrays, summary = align_samples(samples)
  assert metadata["invalid_message_counts"]["carState"] == 1
  assert arrays["car_state_valid"][0] == 0
  assert "carControl" in summary["missing_services"]
  assert metadata["opendbc_gitlink"] is None


def test_truncated_capnp_is_not_a_partial_success(tmp_path, tiny_schema):
  import zstandard
  from controller_extract import decode_log

  event = tiny_schema.Event.new_message(logMonoTime=1_000_000_000)
  event.init("carState").vEgo = 8.0
  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(event.to_bytes()[:-1]))
  with pytest.raises(ExtractionError, match="corrupt/incompatible"):
    decode_log(path, tiny_schema)


def test_unknown_event_requires_matching_schema_instead_of_skipping(tmp_path, tiny_schema):
  import capnp
  import zstandard
  from controller_extract import decode_log

  source = tmp_path / "newer.capnp"
  source.write_text("""@0x9cad7421e032847b;
struct Event {
  logMonoTime @0 :UInt64; valid @1 :Bool = true;
  union { carState @2 :CarState; reserved @3 :Void; newAncillary @4 :Void; }
}
struct CarState {
  vEgo @0 :Float32; aEgo @1 :Float32;
  steeringPressed @2 :Bool; brakePressed @3 :Bool; gasPressed @4 :Bool;
  steeringTorque @5 :Float32; steeringAngleDeg @6 :Float32;
}
""", encoding="utf-8")
  newer = capnp.load(str(source))
  state_event = newer.Event.new_message(logMonoTime=1_000_000_000)
  state_event.init("carState").vEgo = 8.0
  ancillary = newer.Event.new_message(logMonoTime=1_010_000_000)
  ancillary.newAncillary = None
  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(state_event.to_bytes() + ancillary.to_bytes()))
  with pytest.raises(ExtractionError, match="Event union/schema mismatch at message 1"):
    decode_log(path, tiny_schema)
  samples, metadata = decode_log(path, newer)
  assert len(samples) == 1
  assert metadata["message_counts"] == {"carState": 1, "newAncillary": 1}


def test_iq_camera_aliases_require_original_ordinals_and_preserve_raw_counts(tmp_path):
  import capnp
  import zstandard
  from controller_extract import decode_log

  def make_schema(path, schema_id, road_ordinal):
    named = {road_ordinal: "roadCameraState :FrameData", 15: "roadEncodeIdx :EncodeIndex", 22: "carState :CarState"}
    fields = [f"{named[i].split(' :')[0]} @{i} :{named[i].split(' :')[1]};" if i in named
              else f"reserved{i} @{i} :Void;" for i in range(1, 67)]
    path.write_text(f"@{schema_id};\nstruct Event {{ logMonoTime @0 :UInt64; valid @67 :Bool = true; union {{ " +
                   " ".join(fields) + " } }\n" + """
struct CarState {
  vEgo @0 :Float32; aEgo @1 :Float32; steeringPressed @2 :Bool;
  brakePressed @3 :Bool; gasPressed @4 :Bool; steeringTorque @5 :Float32; steeringAngleDeg @6 :Float32;
}
struct FrameData { frameId @0 :UInt32; timestampEof @1 :UInt64; sensor @2 :Text; }
struct EncodeIndex {
  frameId @0 :UInt32; timestampEof @1 :UInt64; segmentNum @2 :Int32;
  segmentId @3 :UInt32; segmentIdEncode @4 :UInt32; encodeId @5 :UInt32;
}
""", encoding="utf-8")
    return capnp.load(str(path))

  schema = make_schema(tmp_path / "iq.capnp", "0xc9ad7421e032847b", 2)
  events = []
  for name in ("carState", "roadCameraState", "roadEncodeIdx"):
    event = schema.Event.new_message(logMonoTime=1_000_000_000)
    d = event.init(name)
    if name == "carState":
      d.vEgo = 8.0
    else:
      d.frameId, d.timestampEof = 123, 990_000_000
    events.append(event.to_bytes())
  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(b"".join(events)))
  samples, metadata = decode_log(path, schema)
  arrays, summary = align_samples(samples)
  assert metadata["service_aliases"] == {"roadCameraState": "narrowRoadCameraState", "roadEncodeIdx": "narrowRoadEncodeIdx"}
  assert metadata["message_counts"]["roadCameraState"] == 1
  assert "narrowRoadCameraState" not in metadata["message_counts"]
  assert arrays["camera_frame_id"][0] == 123
  assert arrays["raw_encode_frame_id"][0] == 123
  assert "liveLocationKalman" in summary["missing_services"]
  assert not arrays["valid_lat"].any()
  wrong = make_schema(tmp_path / "wrong.capnp", "0xb9ad7421e032847b", 3)
  with pytest.raises(ExtractionError, match="roadCameraState.*ordinal"):
    decode_log(path, wrong)


def test_missing_log_is_an_error(tmp_path):
  with pytest.raises(ExtractionError, match="missing"):
    decompress_log(tmp_path / "missing.zst")


def test_inventory_counts_missing_sources_and_route_pairs(tmp_path):
  for route in ("00000001--abc", "00000002--def", "00000003--ghi"):
    for index in (0, 1, 2):
      folder = tmp_path / f"{route}--{index}"
      folder.mkdir()
      (folder / "rlog.zst").write_bytes(b"x")
      (folder / "fcamera.hevc").write_bytes(b"x")
  (tmp_path / "00000004--jkl--0").mkdir()
  result = inventory([tmp_path, tmp_path / "missing-root"])
  assert result["coverage"]["segments"] == 10
  assert result["coverage"]["with_rlog"] == 9
  assert result["coverage"]["with_fcamera"] == 9
  assert len(result["errors"]) == 1
  selected = select_segments(result["segments"], 4)
  assert len(selected) == 4
  assert len({row["route"] for row in selected}) == 2
  for route in {row["route"] for row in selected}:
    assert {row["index"] for row in selected if row["route"] == route} == {0, 1}


def test_duplicate_copy_is_not_an_extra_independent_route(tmp_path):
  roots = [tmp_path / "one", tmp_path / "two"]
  for root in roots:
    folder = root / "00000001--abc--0"
    folder.mkdir(parents=True)
    (folder / "rlog.zst").write_bytes(b"x")
  assert len(select_segments(inventory(roots)["segments"], 20)) == 1


def test_wrong_bus_does_not_create_eps_faults():
  from controller_extract import can_values

  frames = [{"address": 0x251, "src": 1, "dat": b"\x00\x40"},
            {"address": 0x251, "src": 130, "dat": b"\x00\x40"},
            {"address": 0x251, "src": 0, "dat": b"\x00\x00"}]
  values, counters = can_values(frames)
  assert values["eps_fault"] == 0
  assert counters["eps_frames"] == 1
  assert counters["eps_fault_frames"] == 0


def test_only_bus_128_transmitted_lkas_is_counted():
  from controller_extract import can_values

  dat = ((409 + 1024) << 16).to_bytes(4, "little")
  values, counters = can_values([{"address": 0x340, "src": 2, "dat": dat},
                                 {"address": 0x340, "src": 130, "dat": dat},
                                 {"address": 0x340, "src": 128, "dat": dat}])
  assert values["can_torque"] == 409
  assert counters["tx_lkas_frames"] == 1


def test_unknown_opendbc_identity_is_never_inferred_from_safety_parameter():
  from controller_extract import provenance_tag

  metadata = {"car_params": {"safetyConfigs": [{"safetyParam": 409}]}, "init_data": {"gitCommit": "abc"}}
  tagged = provenance_tag(metadata)
  assert tagged["opendbc_gitlink"] is None
  assert tagged["opendbc_identity_status"] == "unknown"


def test_long_mask_does_not_require_lateral_yaw_or_steering_output():
  samples = [state(1), state(1.1),
             Sample("carControl", 1, {"long_active": 1.0}, True),
             Sample("carOutput", 1, {"accel_command": 0.2}, True),
             Sample("longitudinalPlan", 1, {"accel_target": 0.3}, True),
             Sample("extrinsicsCalibration", 1, {"calibrated": 1.0}, True),
             Sample("modelV2", 1, {}, True)]
  arrays, _ = align_samples(samples)
  assert arrays["valid_long"].all()
  assert not arrays["valid_lat"].any()


def calibration_gap_data():
  samples = [state(1), state(1.55),
             Sample("carControl", 1, {"lat_active": 1., "long_active": 1.}, True),
             Sample("carOutput", 1, {"torque": 0., "accel_command": .2}, True),
             Sample("controlsState", 1, {"desired_curvature": 0.}, True),
             Sample("liveLocationKalman", 1, {"yaw": 0., "roll": 0., "location_measurement_valid": 1.}, True),
             Sample("longitudinalPlan", 1, {"accel_target": .3}, True),
             Sample("extrinsicsCalibration", 1, {"calibrated": 1., "calibration_t": 1.,
                                                   "rpy_calib": [0., .01, .02], "calibration_height": [1.2]}, True),
             Sample("modelV2", 1, {}, True)]
  arrays, _ = align_samples(samples)
  # Dynamic channels are independently fresh throughout this artificial fixture.
  for key in ("v", "a", "torque", "accel_command", "desired_curvature", "yaw", "roll", "accel_target",
              "lat_active", "long_active", "car_state_valid", "car_control_valid", "car_output_valid", "controls_valid",
              "location_valid", "location_measurement_valid", "planner_valid", "model_valid"):
    arrays[key][:] = arrays[key][0]
  return arrays


def test_calibration_hold_is_explicit_bounded_and_does_not_mutate_source():
  from controller_extract import hold_calibration

  raw = calibration_gap_data()
  held = hold_calibration(raw)
  assert np.isnan(raw["calibrated"][5])
  np.testing.assert_array_equal(held["calibrated"][:11], np.ones(11))
  assert held["valid_lat"][:11].all() and held["valid_long"][:11].all()
  assert np.isnan(held["calibrated"][11])
  assert not held["valid_lat"][11] and not held["valid_long"][11]
  np.testing.assert_array_equal(held["rpy_calib"][8], [0., .01, .02])
  assert held["calibration_height"][8, 0] == 1.2
  np.testing.assert_array_equal(raw["age_extrinsicsCalibration"], held["age_extrinsicsCalibration"])


def test_calibration_hold_never_crosses_a_new_or_invalid_source_event():
  from controller_extract import hold_calibration

  raw = calibration_gap_data()
  raw["age_extrinsicsCalibration"][5:] -= .1  # A new source event was recorded, with no finite state to recover.
  held = hold_calibration(raw)
  assert np.isnan(held["calibrated"][7])
  raw = calibration_gap_data()
  raw["calibration_valid"][4] = 0
  held = hold_calibration(raw)
  assert np.isnan(held["calibrated"][5])


def test_calibration_hold_keeps_all_dynamic_freshness_gates():
  from controller_extract import hold_calibration

  raw = calibration_gap_data()
  raw["yaw"][7] = np.nan
  raw["car_output_valid"][8] = np.nan
  raw["planner_valid"][9] = 0
  held = hold_calibration(raw)
  assert not held["valid_lat"][7]
  assert held["valid_long"][7]
  assert not held["valid_lat"][8] and not held["valid_long"][8]
  assert held["valid_lat"][9] and not held["valid_long"][9]
  np.testing.assert_array_equal(raw["car_output_valid"], held["car_output_valid"])


def test_calibration_hold_rejects_missing_age_contract():
  from controller_extract import ExtractionError, hold_calibration

  raw = calibration_gap_data()
  del raw["age_extrinsicsCalibration"]
  with pytest.raises(ExtractionError, match="age_extrinsicsCalibration"):
    hold_calibration(raw)


CAPTURE_PROVENANCE = {"model_capture_identity_status": "recorded_narrow_frame_id"}


def test_model_capture_requires_identity_proof_and_rejects_iq_rewritten_ids():
  from controller_extract import model_capture_times

  data = {"t": np.array([2.]), "model_t": np.array([1.9]), "model_frame_id": np.array([123.]),
          "raw_encode_frame_id": np.array([123.]), "raw_encode_t": np.array([1.8])}
  assert np.isnan(model_capture_times(data)).all()
  assert model_capture_times(data, CAPTURE_PROVENANCE)[0] == 1.8
  # IQ source can relabel an older BIG path as frame123, so exact123 is not exposure proof.
  data.update(model_frame_id_trusted=np.ones(1), model_frame_age=np.zeros(1), model_big=np.zeros(1))
  assert np.isnan(model_capture_times(data, {"recorded_software_family": "IQ.Pilot"})).all()
  assert np.isnan(model_capture_times(data, {"init_data": {"version": "IQ.Pilot 1.0c"}})).all()
  assert np.isnan(model_capture_times(data, {"message_counts": {"iqDriveModelData": 1}})).all()
  data["model_frame_id_trusted"] = np.ones(2)
  assert np.isnan(model_capture_times(data, CAPTURE_PROVENANCE)).all()


def test_model_capture_legacy_sunnypilot_provenance_remains_available():
  from controller_extract import model_capture_times

  data = {"t": np.array([2.]), "model_t": np.array([1.9]), "model_frame_id": np.array([1.]),
          "raw_encode_frame_id": np.array([1.]), "raw_encode_t": np.array([1.8])}
  metadata = {"message_counts": {"modelV2": 1, "narrowRoadCameraState": 1, "narrowRoadEncodeIdx": 1},
              "source_log_sha256": "a" * 64, "schema_sha256": {"log.capnp": "b" * 64}}
  assert model_capture_times(data, metadata)[0] == 1.8
  del metadata["schema_sha256"]
  assert np.isnan(model_capture_times(data, metadata)).all()


def test_new_iq_extraction_records_age_and_big_but_never_invents_capture_trust(tmp_path):
  import capnp
  import zstandard
  from controller_extract import decode_log, model_capture_times

  source = tmp_path / "iq_capture.capnp"
  source.write_text("""@0xdba7b32f08ea509c;
struct Event {
  logMonoTime @0 :UInt64; valid @1 :Bool = true;
  union { modelV2 @2 :Model; iqDriveModelData @3 :Void; }
}
struct Model { frameId @0 :UInt32; position @1 :Position; frameAge @2 :UInt32; big @3 :Bool; }
struct Position { x @0 :List(Float32); y @1 :List(Float32); z @2 :List(Float32); t @3 :List(Float32); }
""", encoding="utf-8")
  schema = capnp.load(str(source))
  event = schema.Event.new_message(logMonoTime=1_000_000_000)
  d = event.init("modelV2")
  d.frameId, d.frameAge, d.big = 123, 0, False
  path = tmp_path / "rlog.zst"
  path.write_bytes(zstandard.ZstdCompressor().compress(event.to_bytes()))
  samples, metadata = decode_log(path, schema)
  arrays, _ = align_samples([state(1), *samples])
  assert metadata["model_capture_identity_status"] == "untrusted_selector_frame_id"
  assert arrays["model_frame_age"][0] == 0
  assert arrays["model_big"][0] == 0
  assert arrays["model_frame_id_trusted"][0] == 0
  arrays.update(raw_encode_frame_id=np.array([123]), raw_encode_t=np.array([.9]))
  assert np.isnan(model_capture_times(arrays, metadata)).all()


def test_model_capture_time_matches_exact_frame_id_not_nearest_or_publication():
  from controller_extract import model_capture_times

  data = {"t": np.array([2., 3., 4., 5.]), "model_t": np.array([1.9, 2.9, 3.9, 4.9]),
          "model_frame_id": np.array([9., 8., 10., 11.]),
          "raw_encode_frame_id": np.array([8., 9., 10.]), "raw_encode_t": np.array([2.5, 1.5, 3.5])}
  result = model_capture_times(data, CAPTURE_PROVENANCE)
  np.testing.assert_array_equal(result[:3], [1.5, 2.5, 3.5])
  assert np.isnan(result[3])
  np.testing.assert_array_equal(data["model_t"], [1.9, 2.9, 3.9, 4.9])


def test_model_capture_rejects_future_exposure_or_unpublished_model():
  from controller_extract import model_capture_times

  data = {"t": np.array([2., 2., 2.]), "model_t": np.array([1.9, 2., 2.1]),
          "model_frame_id": np.array([1., 2., 3.]), "raw_encode_frame_id": np.array([1., 2., 3.]),
          "raw_encode_t": np.array([1.95, 2.01, 1.9])}
  assert np.isnan(model_capture_times(data, CAPTURE_PROVENANCE)).all()


def test_model_capture_conflicting_duplicates_are_ambiguous_but_identical_duplicates_are_safe():
  from controller_extract import model_capture_times

  data = {"t": np.full(3, 2.), "model_t": np.full(3, 1.9), "model_frame_id": np.array([7., 8., 9.]),
          "raw_encode_frame_id": np.array([7., 7., 8., 8., 9., 9.]),
          "raw_encode_t": np.array([1., 1., 1.2, 1.3, 1.4, np.nan])}
  result = model_capture_times(data, CAPTURE_PROVENANCE)
  assert result[0] == 1.
  assert np.isnan(result[1:]).all()


def test_model_capture_missing_encode_metadata_never_uses_publication_as_exposure():
  from controller_extract import model_capture_times

  data = {"t": np.array([2., 3.]), "model_t": np.array([1.9, 2.9]), "model_frame_id": np.array([1., 2.])}
  assert np.isnan(model_capture_times(data, CAPTURE_PROVENANCE)).all()
  data.update(raw_encode_frame_id=np.array([1., 2.]), raw_encode_t=np.array([1., 2.]))
  del data["model_t"]
  assert np.isnan(model_capture_times(data, CAPTURE_PROVENANCE)).all()


def test_model_capture_noninteger_ids_are_rejected_and_integer_64bit_ids_are_exact():
  from controller_extract import model_capture_times

  data = {"t": np.full(3, 2.), "model_t": np.full(3, 1.9),
          "model_frame_id": np.array([1.5, np.nan, float(2**53)]),
          "raw_encode_frame_id": np.array([1.5, 2., float(2**53)]), "raw_encode_t": np.ones(3)}
  assert np.isnan(model_capture_times(data, CAPTURE_PROVENANCE)).all()
  ids = np.array([2**63 + 7, 2**63 + 9], dtype=np.uint64)
  data = {"t": np.full(2, 2.), "model_t": np.full(2, 1.9), "model_frame_id": ids,
          "raw_encode_frame_id": ids[::-1], "raw_encode_t": np.array([1.2, 1.1])}
  np.testing.assert_array_equal(model_capture_times(data, CAPTURE_PROVENANCE), [1.1, 1.2])


def test_model_capture_stale_exact_match_reports_actual_age_without_inventing_freshness():
  from controller_extract import model_capture_times

  data = {"t": np.array([10.]), "model_t": np.array([9.99]), "model_frame_id": np.array([1.]),
          "raw_encode_frame_id": np.array([1.]), "raw_encode_t": np.array([1.])}
  assert model_capture_times(data, CAPTURE_PROVENANCE)[0] == 1.
  assert data["t"][0] - model_capture_times(data, CAPTURE_PROVENANCE)[0] == 9.
