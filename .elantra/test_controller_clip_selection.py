import json

import numpy as np
import pytest

import controller_clip_selection as clips


def data():
  n = 1200
  d = {key: np.ones(n) for key in ("calibrated", "calibration_valid", "valid_long", "valid_lat", "long_active", "lat_active")}
  d.update({key: np.zeros(n) for key in ("steering_pressed", "brake_pressed", "gas_pressed", "yaw", "torque")})
  d.update(t=np.arange(n)*.05, raw_encode_t=np.arange(n)*.05,
           raw_encode_frame_id=np.arange(n)+100, raw_encode_segment_id=np.arange(n),
           rpy_calib=np.zeros((n, 3)), v=np.maximum(0., 8.-np.arange(n)*.02), a=np.full(n, -.4))
  return d


META = {"segment": "route--0", "route": "route", "source_log_sha256": "a"*64}


def test_observed_stop_is_selected_with_genuine_history_and_capture_identity():
  rows, _ = clips.segment_candidates(META, data())
  stops = [row for row in rows if row["kind"] == "stopping"]
  assert stops and stops[0]["history_s"] == 6.6
  assert stops[0]["last_frame_id"] - stops[0]["first_frame_id"] == 639
  assert stops[0]["evidence"]["final_speed_mps"] == 0


@pytest.mark.parametrize("field,value", [("brake_pressed", 1), ("long_active", 0), ("calibration_valid", 0)])
def test_override_disengagement_and_missing_calibration_cannot_be_a_controlled_stop(field, value):
  d = data()
  d[field][:] = value
  assert not clips.segment_candidates(META, d)[0]


def test_camera_gap_cannot_be_hidden_by_regular_motion_timestamps():
  d = data()
  d["raw_encode_frame_id"][1:] += 1
  d["raw_encode_frame_id"][601:] += 1
  rows, rejected = clips.segment_candidates(META, d)
  assert not rows and rejected["camera_gap_or_identity"] > 0


def test_even_one_invalid_axis_row_cannot_claim_uninterrupted_control():
  d = data()
  d["valid_long"][599] = 0
  assert not clips.segment_candidates(META, d)[0]


def test_behavior_selection_uses_distinct_routes_and_deterministic_ties():
  rows = [{"kind": kind, "route": route, "segment": route+"--0", "score": score, "camera_start_s": 0}
          for kind, route, score in (("stopping", "a", 3), ("acceleration", "a", 10),
                                     ("acceleration", "b", 2), ("turning", "c", 1))]
  selected = clips.select_diverse(rows)
  assert [row["route"] for row in selected] == ["a", "b", "c"]
  assert clips.select_diverse(list(reversed(rows))) == selected


def test_training_path_filter_never_opens_validation_or_test_arrays(tmp_path, monkeypatch):
  for route in ("train", "validation", "test"):
    (tmp_path/(route+".npz")).write_bytes(b"must not open arrays")
    (tmp_path/(route+".json")).write_text(json.dumps({"route": route}))
  monkeypatch.setattr(clips, "enrich_route_metadata", lambda paths: None)
  monkeypatch.setattr(clips, "cohort_key", lambda meta: "cohort")
  reservation = {"cohorts": {"cohort": {"split": {key: [key] for key in ("train", "validation", "test")}}}}
  paths = clips.training_paths(tmp_path, reservation, "cohort")
  assert [meta["route"] for _, meta in paths] == ["train"]


def turning_data(frames):
  d = {key: value[:frames].copy() for key, value in data().items()}
  d["v"][:] = 10.
  d["yaw"][:] = .1
  d["torque"][:] = 30.
  return d


@pytest.mark.parametrize("seconds,frames,scored", [(16., 320, 188), (24., 480, 348)])
def test_explicit_short_duration_preserves_exact_frames_and132_history(seconds, frames, scored):
  rows, rejected = clips.segment_candidates(META, turning_data(frames), seconds=seconds)
  assert not rejected
  turns = [row for row in rows if row["kind"] == "turning"]
  assert len(turns) == 1
  assert turns[0]["seconds"] == seconds
  assert turns[0]["history_s"] * 20 == 132
  assert turns[0]["scored_rows"] == scored
  assert turns[0]["last_frame_id"]-turns[0]["first_frame_id"]+1 == frames


@pytest.mark.parametrize("seconds", [16., 24.])
@pytest.mark.parametrize("field,value", [("valid_lat", 0), ("lat_active", 0), ("torque", 410),
                                        ("yaw", np.nan), ("brake_pressed", 1), ("calibration_valid", 0)])
def test_short_clip_keeps_every_scored_row_gate(seconds, field, value):
  d = turning_data(round(seconds*20))
  d[field][-1] = value
  rows, _ = clips.segment_candidates(META, d, seconds=seconds)
  assert not [row for row in rows if row["kind"] == "turning"]


@pytest.mark.parametrize("seconds", [True, None, "16", 0., -16., 8.55, 8.61, 16.025, np.nan, np.inf])
def test_invalid_duration_is_rejected_before_source_inspection(seconds):
  with pytest.raises(ValueError, match="duration"):
    clips.segment_candidates(META, {}, seconds=seconds)


def test_short_sources_never_fabricate_longer_footage_or_motion():
  d = turning_data(320)
  assert clips.segment_candidates(META, d, seconds=24.)[0] == []
  assert clips.segment_candidates(META, d)[0] == []
  d = turning_data(480)
  for key in ("t", "v", "a", "yaw", "torque"):
    d[key] = d[key][:320]
  rows, rejected = clips.segment_candidates(META, d, seconds=24.)
  assert not rows and rejected["motion_coverage"] == 1


def test_default32_second_contract_is_unchanged_and_minimum_duration_is_explicit():
  d = turning_data(1200)
  assert clips.segment_candidates(META, d) == clips.segment_candidates(META, d, seconds=32.)
  rows, _ = clips.segment_candidates(META, turning_data(172), seconds=8.6)
  assert rows[0]["scored_rows"] == 40
