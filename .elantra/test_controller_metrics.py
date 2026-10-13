"""Measured-trajectory and coverage regressions for archive controller metrics."""
import json
import os

import numpy as np
import pytest

import controller_metrics as cm


def record(seconds=4, speed=10., yaw=0.):
  t = np.arange(int(seconds / cm.DT) + 1) * cm.DT
  n = len(t)
  horizon = (np.arange(33) / 32) ** 2 * 10
  data = {"t": t, "v": np.full(n, speed), "a": np.zeros(n), "yaw": np.full(n, yaw),
          "roll": np.zeros(n), "torque": np.full(n, 204.5), "torque_fraction": np.full(n, .5),
          "desired_curvature": np.full(n, yaw / speed), "accel_command": np.zeros(n),
          "accel_target": np.zeros(n), "model_t": t.copy(), "model_horizon": np.tile(horizon, (n, 1)),
          "model_x": np.tile(speed * horizon, (n, 1)), "model_y": np.zeros((n, 33)),
          "model_frame_id": np.arange(n, dtype=np.int64), "raw_encode_frame_id": np.arange(n, dtype=np.int64),
          "raw_encode_t": t.copy(),
          "lat_active": np.ones(n), "long_active": np.ones(n)}
  for key in ("calibrated", "calibration_valid", "model_valid", "car_state_valid", "car_control_valid",
              "car_output_valid", "controls_valid", "planner_valid", "location_valid", "location_measurement_valid",
              "valid_lat", "valid_long"):
    data[key] = np.ones(n)
  for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
    data[key] = np.zeros(n)
  return data


def test_nonuniform_horizon_uses_timestamps_and_measured_speed():
  data = record()
  # Decoy model-derived velocity must never change measured GT.
  data["model_velocity"] = np.full(len(data["t"]), 1000.)
  result = cm.segment_metrics(data)
  metric = result["both"]["model_vs_measured"]["1"]
  assert metric["used"] == 4
  assert metric["candidates"] == 5
  assert metric["excluded"] == {"future_unavailable": 1}
  assert cm.summarize(metric["endpoint_l2_m"])["max"] < 1e-10
  assert cm.summarize(result["both"]["model_vs_measured"]["3"]["endpoint_l2_m"])["max"] < 1e-10
  # An index-based 33-point horizon would put this endpoint elsewhere.
  assert data["model_horizon"][0, 3] != pytest.approx(1)


def test_right_positive_yaw_and_model_origin_are_preserved():
  data = record(yaw=.2)
  h = data["model_horizon"][0]
  data["model_x"][:] = 50 * np.sin(.2 * h) + 7
  data["model_y"][:] = 50 * (1 - np.cos(.2 * h)) + 4
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert cm.summarize(result["endpoint_l2_m"])["max"] < .04
  assert cm.summarize(result["lateral_signed_m"])["max"] > 0
  data["model_y"][:] = 4 - 50 * (1 - np.cos(.2 * h))
  wrong = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert cm.summarize(wrong["lateral_abs_m"])["min"] > 1.9


@pytest.mark.parametrize(("field", "value", "reason"), [
  ("steering_pressed", 1, "driver_intervention"),
  ("gas_pressed", 1, "driver_intervention"),
  ("brake_pressed", 1, "driver_intervention"),
  ("location_measurement_valid", 0, "invalid_measured_telemetry"),
  ("yaw", np.nan, "invalid_measured_telemetry"),
  ("calibrated", 0, "invalid_calibration"),
  ("long_active", 0, "mode_change"),
])
def test_entire_gt_horizon_rejects_invalid_data_and_override(field, value, reason):
  data = record(seconds=1)
  data[field][10] = value
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["used"] == 0
  assert result["excluded"][reason] == 1
  assert sum(result["excluded"].values()) == result["candidates"]


def test_future_gap_is_not_interpolated_away():
  data = record(seconds=1)
  data["t"][10:] += .1
  data["model_t"] = data["t"].copy()
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["used"] == 0
  assert result["excluded"]["time_gap"] == 1


def test_model_age_is_causal_and_strictly_bounded():
  data = record(seconds=2)
  data["model_t"] -= .051
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["excluded"]["stale_model"] == 3
  data["model_t"] = data["t"] + .001
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["excluded"]["future_model"] == 3
  data["model_t"] = data["t"] - .05
  data["raw_encode_t"] = data["model_t"].copy()
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["used"] == 1
  assert result["model_age_s"] == pytest.approx([.05])


def delayed_record(seconds=4):
  data = record(seconds=seconds)
  data["t"] += 10.
  data["model_t"] = data["t"] - .01
  data["raw_encode_t"] = data["t"] - .08
  data["v"] = 4. + .7 * (data["t"] - 10.)
  data["a"][:] = .7
  horizon = np.r_[[0., .03, .11, .26, .5, .8, 1., 1.3, 1.8, 2., 2.5, 3.], np.linspace(3.2, 10., 21)]
  data["model_horizon"][:] = horizon
  capture_v = 4. + .7 * (data["raw_encode_t"] - 10.)
  data["model_x"][:] = capture_v[:, None] * horizon + .35 * horizon**2 + 7.
  data["model_y"][:] = 4.
  return data


def test_exact_80ms_capture_delay_anchors_gt_to_exposure_with_nonuniform_model_horizon():
  data = delayed_record()
  data["model_velocity"] = np.full(len(data["t"]), 999.)
  metrics = cm.segment_metrics(data)["both"]["model_vs_measured"]
  one = metrics["1"]
  assert one["used"] == 3 and one["candidates"] == 5
  assert one["excluded"] == {"missing_capture_prefix": 1, "future_unavailable": 1}
  assert cm.summarize(one["endpoint_l2_m"])["max"] < 1e-10
  assert one["capture_age_s"] == pytest.approx([.08] * 3)
  assert one["capture_to_publication_s"] == pytest.approx([.07] * 3)
  assert metrics["3"]["used"] == 1
  assert cm.summarize(metrics["3"]["endpoint_l2_m"])["max"] < 1e-10


@pytest.mark.parametrize(("field", "value", "reason"), [
  ("brake_pressed", 1., "driver_intervention"), ("location_measurement_valid", 0., "invalid_measured_telemetry"),
  ("car_state_valid", 0., "invalid_measured_telemetry"), ("calibration_valid", 0., "invalid_calibration"),
])
def test_capture_prefix_interpolation_requires_both_bracketing_rows_valid(field, value, reason):
  data = delayed_record(seconds=2)
  data[field][18] = value  # 10.90 is before capture10.92 and publication10.99.
  one = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert one["used"] == 0
  assert one["excluded"][reason] == 1


def test_capture_gt_rejects_missing_mapping_and_stale_capture_without_publication_fallback():
  data = delayed_record(seconds=2)
  del data["raw_encode_frame_id"]
  one = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert one["used"] == 0 and one["excluded"]["unknown_model_capture"] == 3
  data = delayed_record(seconds=2)
  data["raw_encode_t"] = data["t"] - .201
  one = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert one["used"] == 0 and one["excluded"]["stale_model_capture"] == 3


def test_capture_gt_never_interpolates_across_prefix_gap_or_uses_future_gt_as_a_model_input():
  data = delayed_record(seconds=3)
  data["t"][19:] += .1
  data["model_t"] = data["t"] - .01
  data["raw_encode_t"] = data["t"] - .08
  one = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert one["excluded"]["time_gap"] >= 1
  data = delayed_record(seconds=2)
  before = data["model_x"].copy()
  data["v"][30:] += 2.
  one = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert one["used"] == 1
  assert one["endpoint_l2_m"][0] > .7
  np.testing.assert_array_equal(data["model_x"], before)


def test_insufficient_or_nonmonotonic_model_horizon_is_explicit():
  data = record(seconds=1)
  data["model_horizon"][:, -1] = .1
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["used"] == 0
  assert result["excluded"]["invalid_model_horizon"] == 1
  data = record(seconds=1)
  data["model_horizon"] *= .05
  result = cm.segment_metrics(data)["both"]["model_vs_measured"]["1"]
  assert result["excluded"]["invalid_model_horizon"] == 1


def test_409_is_an_explicit_normalization_reference_not_historical_limit():
  data = record(seconds=1)
  data["torque_fraction"][:] = 1
  data["torque"][:] = 384
  result = cm.segment_metrics(data)["both"]
  assert result["rows"] == 21
  assert cm.summarize(result["instantaneous"]["delivered_minus_requested_fraction"])["mean"] == pytest.approx(384 / 409 - 1)
  data["torque"][:2] = [409, 450]
  result = cm.segment_metrics(data)["both"]
  assert result["sampled_torque_counts"] == {"finite": 21, "at_409": 1, "over_409": 1}
  assert result["lat_tracking"]["used"] == 20


def test_modes_never_pool_instantaneous_longitudinal_metrics():
  data = record(seconds=1)
  data["a"][:] = -.25
  data["accel_command"][:] = -.5
  data["accel_target"][:] = -.75
  data["long_active"][10:] = 0
  result = cm.segment_metrics(data)
  assert result["both"]["long_tracking"]["used"] == 10
  assert result["lateral_only"]["long_tracking"]["used"] == 0
  assert cm.summarize(result["both"]["instantaneous"]["accel_command_minus_measured_mps2"])["mean"] == -.25
  assert cm.summarize(result["both"]["instantaneous"]["accel_target_minus_measured_mps2"])["mean"] == -.5


def test_source_report_is_whitelisted_and_does_not_expose_vin_or_model_credentials(tmp_path):
  p = tmp_path / "segment.npz"
  np.savez_compressed(p, t=np.array([0.]))
  metadata = {"segment": "segment", "route": "route", "source_log": "rlog.zst", "source_log_sha256": "abc",
              "git_commit": "def", "fingerprint": "HYUNDAI_ELANTRA_2024", "carVin": "private-vin",
              "model_identity": {"ModelApiToken": "private-token"}, "car_params_sp": {"carVin": "private-vin"}}
  p.with_suffix(".json").write_text(json.dumps(metadata))
  source = cm.source_reference(p, metadata)
  identity = cm.cohort_identity(metadata)
  serialized = json.dumps([source, identity])
  assert "private-vin" not in serialized
  assert "private-token" not in serialized
  assert source["source_log_sha256"] == "abc"
  assert len(source["npz_sha256"]) == 64


def test_anchor_sampling_never_exceeds_one_per_second():
  indices = cm.anchor_indices(np.arange(100) * .05)
  assert list(indices) == [0, 20, 40, 60, 80]
  with pytest.raises(ValueError, match="increasing"):
    cm.anchor_indices(np.array([0., 0.]))


def test_report_output_cannot_overwrite_dataset_or_hardlinked_metadata(tmp_path):
  dataset = tmp_path / "dataset"
  dataset.mkdir()
  metadata = dataset / "segment.json"
  metadata.write_text("original")
  with pytest.raises(ValueError, match="outside"):
    cm.validate_output(dataset, metadata)
  with pytest.raises(ValueError, match="outside"):
    cm.validate_output(dataset, tmp_path / "report.npz")
  alias = tmp_path / "alias.json"
  os.link(metadata, alias)
  with pytest.raises(ValueError, match="aliases"):
    cm.validate_output(dataset, alias)
  cm.validate_output(dataset, tmp_path / "report.json")
  assert metadata.read_text() == "original"


def test_extraction_denominator_counts_selected_and_named_failures(tmp_path):
  manifest = {"status": "failed", "selected": [{"segment": "r--0"}, {"segment": "r--1"}, {"segment": "s--0"}],
              "extracted": [{"segment": "r--0"}, {"segment": "s--0"}],
              "extraction_errors": [{"segment": "r--1", "error": "truncated"}], "errors": []}
  (tmp_path / "manifest.json").write_text(json.dumps(manifest))
  context = cm.extraction_context(tmp_path)
  assert context["extraction_selected"] == 3
  assert context["extraction_succeeded"] == 2
  assert context["extraction_errors"] == 1
  assert context["source_routes_selected"] == 2
  assert context["extraction_failures"][0]["segment"] == "r--1"
  assert context["extraction_status"] == "failed"
