"""Finite-search regressions: independent holdouts, fixed support and separate gains."""

from copy import deepcopy

import numpy as np
import pytest

import controller_search as search


def raw_result(params, ids=("a", "b"), error=.1, jerk=0., saturation=0.):
  return {"parameters": params, "windows": {name: {
    "tracking": {"acceleration": {"errors": [.1] * 60, "points": 60},
                 "braking": {"errors": [error] * 60, "points": 60}},
    "gt_errors": [999.] * 60, "jerk": [jerk] * 60, "saturated": [saturation] * 60,
    "command_rate": [0.] * 60} for name in ids}}


def selection(raw, baseline=None):
  return search.select_axis(raw, baseline or raw_result(None), "long", attempted=2,
                            min_coverage=.6, min_windows=2, min_direction_points=50)


def test_acceleration_gain_never_applies_to_negative_braking_target():
  gains = search.gain_series(np.array([.2, -.2, -1., 0.]), .4, .025)
  np.testing.assert_array_equal(gains, [.4, .025, .025, .4])


def test_winner_tracks_frozen_target_not_similarity_to_recorded_gt():
  better = raw_result({"acceleration_gain": .1, "braking_gain": .2}, error=.02)
  worse = raw_result({"acceleration_gain": .2, "braking_gain": .1}, error=.08)
  for window in worse["windows"].values():
    window["gt_errors"] = [0.] * 60
  result = selection([worse, better])
  assert result["winner"]["parameters"] == better["parameters"]


def test_support_coverage_cheating_is_rejected_before_candidate_ranking():
  cheating = raw_result({"acceleration_gain": .4, "braking_gain": .4}, ids=("a",), error=0.)
  supported = raw_result({"acceleration_gain": .1, "braking_gain": .1}, error=.03)
  result = selection([cheating, supported])
  assert result["winner"]["parameters"] == supported["parameters"]
  assert result["trace"][0]["rejections"] == ["insufficient_individual_support"]
  assert result["common_window_ids"] == ["a", "b"]


def test_every_eligible_candidate_is_scored_on_identical_common_windows():
  a = raw_result({"acceleration_gain": .1, "braking_gain": .1}, ids=("a", "b", "c"), error=.03)
  b = raw_result({"acceleration_gain": .2, "braking_gain": .2}, ids=("a", "b", "d"), error=.02)
  baseline = raw_result(None, ids=("a", "b", "c", "d"))
  result = search.select_axis([a, b], baseline, "long", attempted=4,
                              min_coverage=.5, min_windows=2, min_direction_points=50)
  assert result["common_window_ids"] == ["a", "b"]
  assert all(row["metrics"]["windows"] == 2 for row in result["trace"])


@pytest.mark.parametrize(("jerk", "saturation", "reason"), [(4., 0., "jerk_limit"), (0., .1, "saturation_limit")])
def test_comfort_and_saturation_constraints_precede_tracking_ranking(jerk, saturation, reason):
  bad = raw_result({"acceleration_gain": .4, "braking_gain": .4}, error=0., jerk=jerk, saturation=saturation)
  good = raw_result({"acceleration_gain": .1, "braking_gain": .1}, error=.03)
  result = selection([bad, good])
  assert reason in result["trace"][0]["rejections"]
  assert result["winner"]["parameters"] == good["parameters"]


def test_winner_is_deterministic_and_parameters_are_not_mutated():
  a = raw_result({"acceleration_gain": .1, "braking_gain": .2}, error=.02)
  b = raw_result({"acceleration_gain": .2, "braking_gain": .1}, error=.02)
  original = deepcopy([a, b])
  first = selection([a, b])["winner"]["parameters"]
  second = selection([b, a])["winner"]["parameters"]
  assert first == second == a["parameters"]
  assert [a, b] == original


def test_untrusted_boundary_missing_support_plants_are_never_approved():
  axis = {"identified": True, "model": {"table": {"3-7": {}}, "boundary_optimum": False},
          "validation_baseline": {"used_windows": 30, "coverage": .8, "mae": .1}}
  report = {"axes": {"long": axis}, "source_gitlink_known": True}
  assert "long" in search.trusted_models(report)[0]
  for field, value in (("identified", False), ("model", None)):
    broken = deepcopy(report)
    broken["axes"]["long"][field] = value
    assert not search.trusted_models(broken)[0]
  broken = deepcopy(report)
  broken["axes"]["long"]["model"]["boundary_optimum"] = True
  assert not search.trusted_models(broken)[0]


def test_route_split_cannot_leak_or_reorder_holdout_labels():
  paths = [(None, {"route": route, "init_data": {"wallTimeNanos": t}}) for route, t in (("a", 1), ("b", 2), ("c", 3))]
  split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
  assert search.validate_split(paths, split) == split
  with pytest.raises(ValueError, match="disjoint"):
    search.validate_split(paths, dict(split, test=["b", "c"]))
  with pytest.raises(ValueError, match="chronological"):
    search.validate_split(paths, {"train": ["c"], "validation": ["b"], "test": ["a"]})


def test_validation_search_rejects_test_partition_and_does_not_read_holdout(monkeypatch):
  def forbidden(*_args, **_kwargs):
    raise AssertionError("held-out rollout must not run during validation selection")
  monkeypatch.setattr(search, "evaluate_axis", forbidden)
  with pytest.raises(ValueError, match="validation"):
    search.validation_search([], {}, partition="test")
  result = search.validation_search([], {}, partition="validation")
  assert result["status"] == "insufficient_evidence" and result["candidate"] is None


def test_heldout_failure_is_reported_without_retuning_selected_gains(monkeypatch):
  selected = {"acceleration_gain": .1, "braking_gain": .2}
  original = selected.copy()

  def evaluate(_records, _model, _axis, params, _maximum):
    raw = raw_result(params, error=.1 if params is None else .5)
    raw["attempted"] = 2
    return raw

  def forbidden(*_args, **_kwargs):
    raise AssertionError("held-out evaluation must never select or retune")

  monkeypatch.setattr(search, "evaluate_axis", evaluate)
  monkeypatch.setattr(search, "select_axis", forbidden)
  result = search.heldout_test([], {"long": {}}, selected)
  assert "braking_tracking_regression" in result["long"]["constraint_rejections"]
  assert selected == original


def test_both_trusted_axis_winners_cannot_claim_a_complete_controller_without_joint_evaluation(monkeypatch):
  def evaluate(_records, _model, axis, params, _maximum):
    raw = raw_result(params, ids=tuple(f"window-{i}" for i in range(24)), error=.1 if params is None else .03)
    raw["attempted"] = 24
    if axis == "lat":
      for window in raw["windows"].values():
        window["tracking"] = {"lateral": window["tracking"]["braking"]}
    return raw

  monkeypatch.setattr(search, "evaluate_axis", evaluate)
  result = search.validation_search([], {"lat": {}, "long": {}})
  assert set(result["partial_candidate"]) == {"lateral_gain", "acceleration_gain", "braking_gain"}
  assert result["candidate"] is None and result["status"] == "insufficient_evidence"
  assert len(result["global_trace"]) == 216
  assert all("global_joint_unverified" in row["rejections"] for row in result["global_trace"])


def test_canonical_source_hashes_and_runtime_versions_are_recorded():
  evidence = search.source_runtime_evidence()
  assert set(evidence["canonical_source_sha256"]) == {"controller_tune.py", "controller_extract.py", "controller_metrics.py"}
  assert all(len(value) == 64 for value in evidence["canonical_source_sha256"].values())
  assert evidence["numpy_version"] == np.__version__
  assert evidence["python_version"] and evidence["python_executable"]


def test_missing_direction_or_regressed_braking_rejects_candidate():
  candidate = raw_result({"acceleration_gain": .1, "braking_gain": .2}, error=.2)
  assert selection([candidate])["winner"] is None
  assert "braking_tracking_regression" in selection([candidate])["trace"][0]["rejections"]
  candidate = raw_result({"acceleration_gain": .1, "braking_gain": .2}, error=.02)
  for window in candidate["windows"].values():
    del window["tracking"]["acceleration"]
  assert selection([candidate])["winner"] is None


def test_output_cannot_alias_dataset_or_plant_report(tmp_path):
  dataset = tmp_path / "dataset"
  dataset.mkdir()
  report = tmp_path / "report.json"
  report.write_text("{}")
  with pytest.raises(ValueError):
    search.validate_output(dataset, report, dataset / "search")
  out = tmp_path / "search"
  out.mkdir()
  import os
  os.link(report, out / "manifest.json")
  with pytest.raises(ValueError, match="input"):
    search.validate_output(dataset, report, out)
