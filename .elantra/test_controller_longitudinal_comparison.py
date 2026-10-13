import copy
import base64
import json
import subprocess

import numpy as np
import pytest

import controller_longitudinal_comparison as comparison


def contracts():
  parameters = comparison.PARAMETERS.copy()
  route = comparison.SEGMENT.rsplit("--", 1)[0]
  split = {"train": ["older"], "validation": [route], "test": ["newer"]}
  candidate = {"cohort": "frozen", "production_enabled": False, "steer_max": 409,
               "complete_controller_candidate": None, "partial_axis_candidates": parameters}
  search = {"partial_candidate": parameters, "axes": {"long": {"winner": {"parameters": parameters, "rejections": []},
            "common_window_ids": [comparison.WINDOW] + [str(i) for i in range(20)], "common_coverage": .8}}}
  manifest = {"cohort": "frozen", "split": split}
  model = {"table": {"18+": {"command": [-1, 0, 1], "response": [-1, 0, 1]}}, "delay_s": 0, "tau_s": .5,
           "identified": True, "recorded_build_gitlink_known": True}
  plant = {"cohort": "frozen", "split": copy.deepcopy(split), "axes": {"long": {"model": model}}}
  return candidate, search, manifest, plant


def data():
  n = 400
  d = {"t": 100 + np.arange(n)*.05, "v": np.full(n, 28.), "a": np.zeros(n), "yaw": np.full(n, .02),
       "roll": np.zeros(n), "accel_target": np.full(n, .2), "accel_command": np.zeros(n)}
  for key in ("calibrated", "long_active", "valid_long", "car_state_valid", "location_valid", "location_measurement_valid",
              "planner_valid", "car_output_valid"):
    d[key] = np.ones(n)
  for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
    d[key] = np.zeros(n)
  return d


def test_exact_candidate_and_common_validation_contract():
  values = contracts()
  assert comparison.selection_contract(*values)["tau_s"] == .5


@pytest.mark.parametrize("change", ["candidate", "cohort", "unsupported", "test", "rejected", "unknown_plant"])
def test_mismatched_candidate_support_partition_or_provenance_is_rejected(change):
  candidate, search, manifest, plant = contracts()
  if change == "candidate":
    candidate["partial_axis_candidates"] = {"acceleration_gain": .2, "braking_gain": .2}
  elif change == "cohort":
    plant["cohort"] = "other"
  elif change == "unsupported":
    search["axes"]["long"]["common_window_ids"].remove(comparison.WINDOW)
  elif change == "test":
    manifest["split"]["test"].append(comparison.SEGMENT.rsplit("--", 1)[0])
  elif change == "rejected":
    search["axes"]["long"]["winner"]["rejections"] = ["unsupported"]
  else:
    plant["axes"]["long"]["model"]["recorded_build_gitlink_known"] = False
  with pytest.raises(comparison.ComparisonError):
    comparison.selection_contract(candidate, search, manifest, plant)


def test_frozen_file_hash_and_output_collision_fail_closed(tmp_path):
  path = tmp_path / "receipt.json"
  path.write_text(json.dumps({"a": 1}))
  expected = comparison.digest(path)
  assert comparison.bound_json(path, expected) == {"a": 1}
  path.write_text(json.dumps({"a": 2}))
  with pytest.raises(comparison.ComparisonError, match="hash"):
    comparison.bound_json(path, expected)
  with pytest.raises(comparison.ComparisonError, match="new and contained"):
    comparison.fresh_output(path, tmp_path)
  with pytest.raises(comparison.ComparisonError, match="contained"):
    comparison.fresh_output(tmp_path.parent / "escaped.npz", tmp_path)
  with pytest.raises(comparison.ComparisonError, match="changed"):
    comparison.unchanged_sources({"input": path}, {"input": expected})


def test_trace_acceleration_slew_speed_and_right_positive_motion_math():
  d = data()
  trace = comparison.make_trace(d, contracts()[3]["axes"]["long"]["model"])
  assert trace["candidate_command"][0] == .125
  assert trace["candidate_accel"][1] == pytest.approx(.125 * .05 / .55)
  assert trace["candidate_speed"][1] == 28
  assert trace["candidate_speed"][2] == pytest.approx(28 + trace["candidate_accel"][1] * .05)
  assert np.all(trace["baseline_accel"] == 0)
  assert trace["actual_xy"][1, 0] == pytest.approx(28*.05*np.cos(.02*.05/2))
  assert trace["actual_xy"][1, 1] > 0
  assert np.max(np.abs(np.diff(trace["candidate_command"]))) <= .125 + 1e-8


def test_future_measured_gt_never_enters_longitudinal_command_or_state():
  d = data()
  model = contracts()[3]["axes"]["long"]["model"]
  first = comparison.make_trace(d, model)
  d["a"][comparison.START+1:] = -3
  d["v"][comparison.START+1:] = 99
  d["yaw"][comparison.START+1:] = -.1
  changed = comparison.make_trace(d, model)
  for key in ("candidate_command", "candidate_accel", "candidate_speed", "baseline_accel", "baseline_speed"):
    np.testing.assert_array_equal(first[key], changed[key])
  assert not np.array_equal(first["actual_xy"], changed["actual_xy"])


@pytest.mark.parametrize("issue", ["gap", "override", "invalid", "stale_planner"])
def test_window_rejects_gap_override_invalid_and_stale_inputs(issue):
  d = data()
  if issue == "gap":
    d["t"][comparison.START+20:] += .1
  elif issue == "override":
    d["gas_pressed"][comparison.START-1] = 1
  elif issue == "invalid":
    d["valid_long"][comparison.START+1] = 0
  else:
    d["planner_valid"][comparison.START-1] = 0
  with pytest.raises(comparison.ComparisonError):
    comparison.make_trace(d, contracts()[3]["axes"]["long"]["model"])


def test_reanchor_uses_only_measured_past_and_expires_at_frozen_window_end():
  d = data()
  trace = comparison.make_trace(d, contracts()[3]["axes"]["long"]["model"])
  timestamp = float(trace["t"][0]) + 1.25
  first = comparison.moving_paths(trace, d, timestamp)
  d["v"][d["t"] > timestamp] = 99
  d["yaw"][d["t"] > timestamp] = -1
  for a, b in zip(first, comparison.moving_paths(trace, d, timestamp), strict=True):
    np.testing.assert_array_equal(a, b)
  np.testing.assert_allclose(first[0][0], [0, 0], atol=1e-9)
  with pytest.raises(comparison.ComparisonError, match="inside"):
    comparison.moving_paths(trace, d, float(trace["t"][-1]))


def test_camera_schedule_stays_causal_and_inside_fixed_scored_window():
  d = data()
  trace = comparison.make_trace(d, contracts()[3]["axes"]["long"]["model"])
  d["raw_encode_t"] = d["t"] - .013
  d["raw_encode_segment_id"] = np.arange(len(d["t"]))
  d["raw_encode_segment_num"] = np.full(len(d["t"]), 17)
  mapping, times, indices = comparison.render_schedule(d, trace)
  assert len(times) == 30
  assert np.all(mapping[1][indices] <= times)
  assert np.all(mapping[1][indices] >= trace["t"][0])
  assert np.all(mapping[1][indices] < trace["t"][-1])


@pytest.mark.parametrize("failure", ["exit", "timeout", "launch"])
def test_failed_external_phase_keeps_exact_command_and_output_before_raising(tmp_path, monkeypatch, failure):
  command, stdout, stderr = ["ffmpeg", "exact input.hevc"], b"partial stdout", b"decode failed\xff"
  def run(received, **options):
    assert received == command and options == {"capture_output": True, "check": False, "timeout": 120}
    if failure == "timeout":
      raise subprocess.TimeoutExpired(command, 120, output=stdout, stderr=stderr)
    if failure == "launch":
      raise OSError("executable absent")
    return subprocess.CompletedProcess(command, 7, stdout, stderr)
  monkeypatch.setattr(comparison.subprocess, "run", run)
  with pytest.raises(comparison.ComparisonError, match="exact tool evidence"):
    comparison.run_phase(command, "decode", tmp_path)
  receipt = json.loads((tmp_path / "decode.json").read_text())
  assert receipt["command"] == command and receipt["status"] == "failed"
  assert receipt["timed_out"] == (failure == "timeout")
  assert receipt["returncode"] == (7 if failure == "exit" else None)
  if failure != "launch":
    assert base64.b64decode(receipt["stdout_base64"]) == stdout
    assert base64.b64decode(receipt["stderr_base64"]) == stderr
  else:
    assert "executable absent" in receipt["error"]
