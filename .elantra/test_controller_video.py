"""Timestamp, coordinate and missing-data regression tests for evidence videos."""

import numpy as np
import pytest

from controller_video import (
  RenderError, calibration_at, camera_mapping, contiguous_path, forecast_paths,
  frame_schedule, project_ground, projection_config, reanchor_paths, validate_output_path,
)
from controller_model_overlay import (
  capture_path, measured_forecasts, model_path, native_index, rdf_path, validate_native_output, validate_rdf_output,
)


def test_nonuniform_camera_timestamps_choose_logged_frames_not_container_rate():
  mapping = camera_mapping({"raw_encode_t": np.array([100., 100.04, 100.13, 100.17]),
                            "raw_encode_segment_id": np.arange(4),
                            "raw_encode_segment_num": np.ones(4)}, 1)
  times, indices = frame_schedule(mapping, 100.04, 0.15, 20)
  np.testing.assert_allclose(times, [100.04, 100.09, 100.14])
  np.testing.assert_array_equal(indices, [1, 1, 2])


def test_mapping_uses_presentation_segment_id_and_filters_other_segments():
  mapping = camera_mapping({"raw_encode_t": np.array([2., 3., 3.05]),
                            "raw_encode_segment_id": np.array([99, 0, 1]),
                            "raw_encode_segment_num": np.array([0, 1, 1])}, 1)
  np.testing.assert_array_equal(mapping[0], [0, 1])
  np.testing.assert_allclose(mapping[1], [3, 3.05])


@pytest.mark.parametrize("data,reason", [
  ({"raw_encode_t": [1, 2], "raw_encode_segment_id": [0, 0]}, "conflicting"),
  ({"raw_encode_t": [2, 1], "raw_encode_segment_id": [0, 1]}, "increasing"),
  ({"raw_encode_t": [1, 2], "raw_encode_segment_id": [0, 2]}, "missing"),
  ({"camera_t": [1, 2], "camera_frame_id": [100, 101]}, "EncodeIdx"),
])
def test_camera_mapping_errors_are_explicit(data, reason):
  with pytest.raises(RenderError, match=reason):
    camera_mapping({k: np.array(v) for k, v in data.items()}, 1)


def test_source_timestamp_gap_is_not_interpolated_or_silently_dropped():
  with pytest.raises(RenderError, match="gap"):
    frame_schedule((np.array([0, 1, 2]), np.array([1., 1.05, 1.4])), 1., 0.4, 10)


def test_incomplete_forecast_stops_at_first_invalid_point():
  xy = np.array([[0., 0.], [1., 0.], [np.nan, 0.], [3., 0.]])
  np.testing.assert_array_equal(contiguous_path(xy), [[0, 0], [1, 0]])
  np.testing.assert_array_equal(contiguous_path(np.nan_to_num(xy), [True, True, False, True]), [[0, 0], [1, 0]])


def test_candidates_are_trimmed_to_common_valid_measured_horizon():
  gt = np.array([[[0., 0.], [1, 0], [2, 0], [np.nan, np.nan]]])
  simulated = np.nan_to_num(gt)
  paths, anchor = forecast_paths({"forecast_t": np.array([1.]), "forecast_dt": np.array(.05),
                                  "gt_xy": gt, "baseline_xy": simulated, "tuned_xy": simulated}, 1.05)
  assert anchor == 1
  assert [len(path) for path in paths] == [2, 2, 2]


def test_unavailable_candidate_does_not_hide_recorded_ground_truth():
  gt = np.array([[[0., 0.], [1, 0], [2, 0]]])
  paths, _ = forecast_paths({"forecast_t": np.array([1.]), "forecast_dt": np.array(.05),
                            "gt_xy": gt, "baseline_xy": gt, "tuned_xy": np.full_like(gt, np.nan)}, 1.)
  assert [len(path) for path in paths] == [3, 3, 0]


def test_reanchor_keeps_right_positive_and_uses_measured_pose_for_all_paths():
  gt = np.array([[0., 0.], [1., 0.], [2., 0.]])
  candidate = gt + [0, 0.5]
  measured, simulated = reanchor_paths([gt, candidate], 1)
  np.testing.assert_allclose(measured, [[0, 0], [1, 0]])
  np.testing.assert_allclose(simulated, [[0, 0.5], [1, 0.5]])


def test_ground_projection_right_is_screen_right_and_behind_camera_is_invalid():
  k = np.array([[100., 0, 100], [0, 100, 50], [0, 0, 1]])
  uv = project_ground(np.array([[10., 0], [10, 1], [-1, 0]]), k, np.zeros(3), 1.2)
  assert uv[1, 0] > uv[0, 0]
  assert uv[0, 1] > 50
  assert np.isnan(uv[2]).all()


def test_sensor_intrinsics_require_matching_native_dimensions():
  intrinsics, label = projection_config({}, {"camera_sensor": "os04c10"}, (1344, 760))
  assert intrinsics[0, 0] == 1141.5
  assert "NOMINAL" not in label
  _, label = projection_config({}, {"camera_sensor": "os04c10"}, (1928, 1208))
  assert "NOMINAL" in label


def test_missing_forecast_is_displayed_as_missing_not_backfilled():
  paths, anchor = forecast_paths({"forecast_t": np.array([10.])}, 9.)
  assert anchor is None
  assert all(len(path) == 0 for path in paths)


def test_calibration_must_be_calibrated_and_no_more_than_half_second_old():
  data = {"t": np.array([10.49, 10.5, 10.51]), "calibration_t": np.array([10., 10., 10.]),
          "rpy_calib": np.tile([.01, .02, .03], (3, 1)), "calibration_height": np.ones((3, 1)) * 1.3,
          "calibrated": np.ones(3), "calibration_valid": np.ones(3)}
  assert calibration_at(data, 10.5, 1.22)[2]
  assert not calibration_at(data, 10.51, 1.22)[2]
  data["calibrated"][0] = 0
  assert not calibration_at(data, 10.49, 1.22)[2]
  data["calibration_valid"][1] = 0
  assert not calibration_at(data, 10.5, 1.22)[2]


def test_render_cannot_overwrite_inputs_or_their_metadata(tmp_path):
  segment = tmp_path / "raw--0"
  data, replay = tmp_path / "data.npz", tmp_path / "replay.npz"
  with pytest.raises(RenderError, match="collision"):
    validate_output_path(data, segment, data, replay)
  with pytest.raises(RenderError, match="collision"):
    validate_output_path(tmp_path / "data.mp4", segment, data, replay)
  with pytest.raises(RenderError, match="raw source"):
    validate_output_path(segment / "evidence.mp4", segment, data, replay)
  validate_output_path(tmp_path / "videos" / "evidence.mp4", segment, data, replay)


def test_existing_hardlink_to_input_is_rejected(tmp_path):
  data = tmp_path / "data.npz"
  data.write_bytes(b"source")
  output = tmp_path / "video.mp4"
  output.hardlink_to(data)
  with pytest.raises(RenderError, match="collision"):
    validate_output_path(output, tmp_path / "raw--0", data, tmp_path / "replay.npz")


def motion_data(n=150):
  d = {"t": np.arange(n) * .05, "v": np.full(n, 10.), "yaw": np.zeros(n)}
  for key in ("car_state_valid", "location_valid", "location_measurement_valid"):
    d[key] = np.ones(n)
  return d


def test_render_forecast_uses_exact_camera_phase_and_past_measured_pose():
  d = motion_data()
  xy = np.column_stack((np.arange(61) * .5, np.zeros(61)))[None]
  replay = {"forecast_t": np.array([.1]), "forecast_dt": np.array(.05),
            "gt_xy": xy, "baseline_xy": xy, "tuned_xy": np.full_like(xy, np.nan)}
  paths, anchor = forecast_paths(replay, .125, d)
  assert anchor == .1
  np.testing.assert_allclose(paths[0][0], [0, 0], atol=1e-10)
  assert paths[0][-1, 0] == pytest.approx(29.75)
  assert not len(paths[2])


def test_fresh_evidence_gt_retains_measured_driver_motion_but_never_fills_sensor_holes():
  d = motion_data()
  d.update(calibration_valid=np.ones(150), calibrated=np.ones(150),
           steering_pressed=np.ones(150), long_active=np.zeros(150))
  replay = measured_forecasts(d)
  assert replay["gt_xy"][0, -1, 0] == pytest.approx(30.)
  assert np.isnan(replay["tuned_xy"]).all()
  d["location_valid"][20] = 0
  replay = measured_forecasts(d)
  assert np.isfinite(replay["gt_xy"][0, 19]).all()
  assert np.isnan(replay["gt_xy"][0, 20:]).all()
  d["t"][20:] += 1
  replay = measured_forecasts(d)
  assert np.isnan(replay["gt_xy"][0, 20:]).all()


def test_capture_path_compensates_only_measured_past_and_rejects_gaps():
  d = motion_data()
  xy = np.column_stack((np.arange(33), np.zeros(33)))
  h = np.arange(33) * .1
  path = capture_path(d, xy, h, .1, .2)
  np.testing.assert_allclose(path[0], [0., 0.], atol=1e-10)
  assert path[-1, 0] == pytest.approx(31.)
  d["v"][5:] = 1000  # future motion cannot change the displayed pose
  np.testing.assert_allclose(capture_path(d, xy, h, .1, .2), path)
  assert not len(capture_path(d, xy, h, -.1, .1))
  assert not len(capture_path(d, xy, h, .1, .301))
  d["location_valid"][2] = 0
  assert not len(capture_path(d, xy, h, .1, .2))
  d["location_valid"][2] = 1
  d["location_valid"][4] = 0  # displayed endpoint admission is required too
  assert not len(capture_path(d, xy, h, .1, .2))
  short = {key: value[:2].copy() for key, value in motion_data().items()}
  short["t"] = np.array([0., .15])
  assert not len(capture_path(short, xy, h, .1, .15))


def test_recorded_model_overlay_requires_exact_capture_and_causal_publication():
  d = motion_data()
  n = len(d["t"])
  d.update(model_t=d["t"].copy(), model_capture_t=d["t"] - .05, model_valid=np.ones(n),
           model_x=np.tile(np.arange(33), (n, 1)), model_y=np.zeros((n, 33)),
           model_horizon=np.tile(np.arange(33) * .1, (n, 1)))
  assert len(model_path(d, .2))
  d["model_t"][4] = .201
  assert not len(model_path(d, .2))
  d["model_t"][4] = .2
  d["model_capture_t"][4] = np.nan
  assert not len(model_path(d, .2))


def test_fresh_rdf_identity_warmup_availability_and_source_match():
  from rdf_bench import WHOLE_SHA

  d = motion_data()
  n = len(d["t"])
  d.update(raw_encode_frame_id=np.arange(n), raw_encode_t=d["t"].copy())
  position = np.zeros((n, 33, 3))
  position[:, :, 0] = np.arange(33)
  rdf = {"input_t": d["t"].copy(), "narrow_t": d["t"].copy(), "output_ready_t": d["t"] + .035,
         "frame_id": np.arange(n), "position": position, "horizon": np.arange(33) * .1,
         "scored_mask": np.arange(n) >= 132, "desired_curvature": np.zeros(n), "desired_acceleration": np.zeros(n),
         "model_sha256": np.asarray(WHOLE_SHA)}
  validate_rdf_output(rdf, d)
  rdf["input_t"] -= .001
  with pytest.raises(RenderError, match="availability"):
    validate_rdf_output(rdf, d)
  rdf["input_t"] += .001
  assert rdf_path(rdf, d, 6.63)[1] is None  # frame132 not ready; earlier frames are history
  assert rdf_path(rdf, d, 6.65)[1] == 132
  rdf["output_ready_t"] += 1.
  assert rdf_path(rdf, d, 7.4)[1] is None  # delayed output is never displayed at capture time
  rdf["scored_mask"][0] = True
  with pytest.raises(RenderError, match="warmup"):
    validate_rdf_output(rdf, d)
  rdf["scored_mask"][0] = False
  rdf["frame_id"][0] = 999
  with pytest.raises(RenderError, match="exactly match"):
    validate_rdf_output(rdf, d)
  rdf["model_sha256"] = np.asarray("wrong")
  with pytest.raises(RenderError, match="identity"):
    validate_rdf_output(rdf, d)


def native_evidence():
  rdf_t = np.arange(150) * .05
  rdf = {"input_t": rdf_t, "narrow_t": rdf_t, "output_ready_t": rdf_t + .04,
         "scored_mask": np.arange(150) >= 132, "desired_acceleration": np.ones(150) * .1,
         "should_stop": np.ones(150, dtype=bool)}
  t = 6.6 + np.arange(100) * .01
  indices = np.searchsorted(rdf["output_ready_t"], t, side="right") - 1
  admitted = rdf["scored_mask"][indices]
  native = {"t": t, "admitted": admitted, "initialization_uncertain": np.ones(100, dtype=bool),
            "native_accel": np.where(admitted, -.52, np.nan), "native_state": np.where(admitted, 2., np.nan),
            "should_stop": admitted.copy(), "rdf_index": np.where(admitted, indices, -1),
            "rdf_accel_target": np.where(admitted, .1, np.nan),
            "rdf_capture_age_s": np.where(admitted, t-rdf_t[indices], np.nan)}
  receipt = {"mode": "native_replay", "rows": 100, "dt": .01, "no_actuation": True, "no_params_writes": True,
             "initialization": "cold/reset; uncertain until recorded disengagement",
             "rdf_output_sha256": "a"*64, "source_log_sha256": "b"*64, "output_npz_sha256": "c"*64}
  return native, receipt, rdf


def test_native_binding_rejects_wrong_rdf_log_or_npz():
  native, receipt, rdf = native_evidence()
  validate_native_output(native, receipt, rdf, "a"*64, "b"*64, "c"*64)
  for key in ("rdf_output_sha256", "source_log_sha256", "output_npz_sha256"):
    changed = {**receipt, key: "0"*64}
    with pytest.raises(RenderError, match="hash"):
      validate_native_output(native, changed, rdf, "a"*64, "b"*64, "c"*64)


@pytest.mark.parametrize("change", ["clock", "shape", "flags", "state", "future", "warmup", "stop"])
def test_native_malformed_or_lookahead_evidence_rejected(change):
  native, receipt, rdf = native_evidence()
  if change == "clock":
    native["t"][10] += .001
  elif change == "shape":
    native["native_accel"] = native["native_accel"][:-1]
  elif change == "flags":
    native["admitted"] = native["admitted"].astype(float)
  elif change == "state":
    native["native_state"][20] = 3
  elif change == "future":
    native["rdf_index"][20] += 1
  elif change == "warmup":
    native["admitted"][0] = True
  elif change == "stop":
    native["should_stop"][20] = False
  with pytest.raises(RenderError):
    validate_native_output(native, receipt, rdf, "a"*64, "b"*64, "c"*64)


def test_native_selection_never_backfills_unavailable_or_future_rows():
  native, _, _ = native_evidence()
  assert native_index(native, 6.59) is None
  assert native_index(native, native["t"][0]) is None
  assert native_index(native, native["t"][20] + .001) == 20
  native["admitted"][21] = False
  assert native_index(native, native["t"][21] + .001) is None
  assert native_index(native, native["t"][-1] + .02) is None


def test_render_cannot_overwrite_native_evidence_or_receipt(tmp_path):
  native = tmp_path / "native.npz"
  with pytest.raises(RenderError, match="collision"):
    validate_output_path(tmp_path / "native.mp4", tmp_path / "raw--0", tmp_path / "data.npz", tmp_path / "replay.npz",
                         (native, native.with_suffix(".json")))
