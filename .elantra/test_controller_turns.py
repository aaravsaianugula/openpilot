"""Fixed-reference direction/phase and common-population turn acceptance."""
from copy import deepcopy

import numpy as np
import pytest

import controller_turns as turns


REFERENCE = np.array([-.002, -.004, -.004, -.002, 0., .002, .004, .004, .002])


def window(error=.1, reference=REFERENCE, command=None, yaw=None):
  target = reference * 100
  errors = np.broadcast_to(error, target.shape)
  return turns.window_metrics(reference, target + errors, target,
                              np.zeros(len(target)) if command is None else command,
                              reference_prefix=(.05, 0.), measured_yaw=yaw)


def summary(row):
  return turns.summarize({"a": row, "b": deepcopy(row)}, ["b", "a"])


def constraints(after, before, **kwargs):
  return turns.constraints(after, before, min_direction_points=2, min_phase_points=2,
                           min_windows=2, **kwargs)


def test_left_improvement_cannot_mask_right_regression():
  baseline = summary(window())
  error = np.where(REFERENCE < 0, 0., .15)
  candidate = summary(window(error))
  assert candidate["target_objective"] < baseline["target_objective"]
  assert "right_tracking_regression" in constraints(candidate, baseline)


def test_left_only_coverage_never_establishes_balanced_turn_candidate():
  baseline = summary(window(reference=-np.abs(REFERENCE)))
  candidate = summary(window(.05, reference=-np.abs(REFERENCE)))
  assert candidate["target_objective"] is None
  assert "missing_right_support" in constraints(candidate, baseline)


def test_candidate_speed_motion_and_measured_yaw_cannot_change_reference_labels():
  first = window(yaw=-np.sign(REFERENCE))
  # Same frozen curvature, different physical demand/speed and measured motion.
  second = turns.window_metrics(REFERENCE, REFERENCE * 400, REFERENCE * 400,
                                np.zeros(9), reference_prefix=(.05, 0.), measured_yaw=np.sign(REFERENCE))
  assert first["reference_direction"] == second["reference_direction"]
  assert first["reference_phase"] == second["reference_phase"]
  assert first["reference_sha256"] == second["reference_sha256"]
  assert first["measured_yaw"]["opposite_reference_points"] == 8
  assert second["measured_yaw"]["opposite_reference_points"] == 0


def test_phase_uses_only_present_and_past_reference_and_requires_past_prefix():
  reference = np.array([.002, .004, .004, .002])
  row = turns.window_metrics(reference, reference, reference, np.zeros(4))
  assert row["reference_phase"] == ["unknown", "entry", "steady", "exit"]
  future_changed = turns.window_metrics([.002, .004, .5, -.5], reference, reference, np.zeros(4))
  assert future_changed["reference_phase"][:2] == row["reference_phase"][:2]
  for prefix in ((0., 0.), (-.05, 0.), (.2, 0.), (.05, np.nan)):
    result = turns.window_metrics(reference, reference, reference, np.zeros(4), reference_prefix=prefix)
    assert result["reference_phase"][0] == "unknown"
    assert result["prefix_rejection"] is not None


def test_phase_coverage_and_phase_regression_are_separate_hard_gates():
  reference = np.array([-.004] * 5 + [.004] * 5)
  baseline = summary(window(reference=reference))
  candidate = summary(window(.05, reference=reference))
  assert "missing_left_exit_support" in constraints(candidate, baseline)
  baseline = summary(window())
  error = np.where(np.arange(9) == 1, .5, 0.)
  candidate = summary(window(error))
  assert "left_entry_tracking_regression" in constraints(candidate, baseline)


def test_409_is_saturation_and_over_ceiling_is_unsupported():
  command = np.array([-409., 409., 410., -410., 0., 0., 0., 0., 0.])
  row = window(command=command)
  assert row["invalid_torque_rows"] == [2, 3]
  assert row["tracking"]["left"]["saturated"] == [1., 1.]
  assert "unsupported_turn_points" in constraints(summary(row), summary(window()))
  saturated = summary(window(.05, command=np.full(9, 409.)))
  assert "left_saturation_limit" in constraints(saturated, summary(window()))


def test_missing_high_error_rows_cannot_win_by_smaller_scored_population():
  baseline = summary(window())
  row = window(.01)
  candidate_ay = REFERENCE * 100 + .01
  candidate_ay[1] = np.nan
  row = turns.window_metrics(REFERENCE, candidate_ay, REFERENCE * 100, np.zeros(9), reference_prefix=(.05, 0.))
  candidate = summary(row)
  assert candidate["tracking"]["left"]["unsupported_points"] == 2
  reasons = constraints(candidate, baseline)
  assert "turn_point_population_mismatch" in reasons
  assert "unsupported_turn_points" in reasons


def test_common_windows_and_reference_bindings_must_match_exactly():
  baseline = summary(window())
  candidate = turns.summarize({"a": window(.05)}, ["a", "b"])
  assert candidate["missing_windows"] == ["b"]
  assert "missing_turn_windows" in constraints(candidate, baseline)
  changed = summary(window(.05, reference=REFERENCE * 2))
  assert "turn_reference_mismatch" in constraints(changed, baseline)
  with pytest.raises(ValueError, match="unique"):
    turns.summarize({"a": window()}, ["a", "a"])


def test_old_pooled_rows_fail_closed_and_summary_is_deterministic():
  old = turns.summarize({"a": {"tracking": {"lateral": {"errors": [0.]}}}}, ["a"])
  assert old["invalid_windows"] == ["a"]
  assert "missing_turn_metrics_contract" in constraints(old, old)
  missing = turns.summarize({"a": None}, ["a"])
  assert missing["invalid_windows"] == ["a"]
  assert "missing_turn_metrics_contract" in constraints(missing, missing)
  row = window()
  assert turns.summarize({"a": row, "b": row}, ["a", "b"]) == turns.summarize({"b": row, "a": row}, ["b", "a"])


def test_absent_summary_and_contract_only_rows_fail_with_named_fields():
  valid = summary(window())
  assert turns.constraints(None, valid) == ["missing_turn_metrics_contract"]
  assert turns.constraints(valid, None) == ["missing_turn_metrics_contract"]
  row = {"contract": dict(turns.CONTRACT)}
  result = turns.summarize({"a": row}, ["a"])
  assert result["invalid_windows"] == ["a"]
  assert "tracking" in result["invalid_window_fields"]["a"]
  row = window()
  del row["phases"]["left"]["entry"]["errors"]
  result = turns.summarize({"a": row}, ["a"])
  assert "phases.left.entry.errors" in result["invalid_window_fields"]["a"]
  row = window()
  row["tracking"]["left"]["errors"] = None
  assert turns.summarize({"a": row}, ["a"])["invalid_windows"] == ["a"]


def test_missing_reference_and_jerk_are_named_without_future_fill():
  row = turns.window_metrics([-.002, np.nan, .004], [0., np.nan, 1.], [0., 0., 0.], [0., 0., 0.])
  assert row["missing_reference_rows"] == [1]
  assert row["reference_direction"] == ["left", "unknown", "right"]
  assert row["reference_phase"] == ["unknown", "unknown", "unknown"]
  assert row["tracking"]["right"]["jerk"] == []
  infinite = turns.window_metrics([np.inf, -np.inf], [0., 0.], [0., 0.], [0., 0.])
  assert infinite["reference_direction"] == ["unknown", "unknown"]
  infinite_yaw = window(yaw=np.where(REFERENCE < 0, np.inf, -np.inf))
  assert infinite_yaw["measured_yaw"]["opposite_reference_points"] == 0
  assert infinite_yaw["measured_yaw"]["unknown_turn_points"] == 8


def test_valid_balanced_candidate_and_jerk_constraints():
  before, after = summary(window()), summary(window(.05))
  assert constraints(after, before) == []
  rough = turns.window_metrics(REFERENCE, np.arange(9) * 10., np.arange(9) * 10., np.zeros(9), reference_prefix=(.05, 0.))
  assert "left_jerk_limit" in constraints(summary(rough), before)


@pytest.mark.parametrize("dt", [0., -.05, np.nan])
def test_invalid_dt_is_rejected(dt):
  with pytest.raises(ValueError):
    turns.window_metrics([.002], [0.], [0.], [0.], dt_s=dt)


def test_malformed_shapes_are_rejected():
  with pytest.raises(ValueError):
    turns.window_metrics([.002], [0., 1.], [0.], [0.])
