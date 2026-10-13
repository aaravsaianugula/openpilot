"""Timestamp, coordinate and missing-data regression tests for evidence videos."""

import numpy as np
import pytest

from controller_video import (
  RenderError, calibration_at, camera_mapping, contiguous_path, forecast_paths,
  frame_schedule, project_ground, projection_config, reanchor_paths, validate_output_path,
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
