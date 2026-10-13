"""Regression checks for the experimental, offline hardware-calibration workflow."""
import numpy as np
import pytest

import controller_tune as ct


def test_split_keeps_whole_routes_chronological():
    split = ct.route_split(["r5", "r1", "r2", "r1", "r4", "r3", "r6"])
    assert split == {"train": ["r1", "r2", "r3"], "validation": ["r4"], "test": ["r5", "r6"]}
    assert not (set(split["train"]) & set(split["test"]))
    with pytest.raises(ValueError, match="five"):
        ct.route_split(["a", "b"])


def test_lookup_refuses_extrapolation_and_missing_speed_band():
    table = {"3-7": {"command": [-200, 0, 200], "response": [1, 0, -1]}}
    assert ct.lookup(table, 5, 100) == pytest.approx(-0.5)
    assert np.isnan(ct.lookup(table, 5, 300))
    assert np.isnan(ct.lookup(table, 20, 0))
    assert ct.inverse(table, 5, 0.5) == pytest.approx(-100)
    assert np.isnan(ct.inverse(table, 5, 2))


def test_hardware_limit_preserves_409_and_asymmetric_rate():
    assert ct.limit_torque(900, 400, 0.05) == 409
    assert ct.limit_torque(-900, -400, 0.05) == -409
    assert ct.limit_torque(409, 0, 0.05) == 15
    assert ct.limit_torque(0, 100, 0.05) == 65
    assert ct.limit_torque(-409, 10, 0.05) == -12
    with pytest.raises(ValueError):
        ct.limit_torque(409, 0, -0.1)


def test_ground_truth_uses_measured_speed_and_right_positive_yaw():
    xy = ct.integrate(np.full(61, 10.), np.full(61, 0.1), 0.05)
    assert xy.shape == (61, 2)
    assert xy[-1, 0] == pytest.approx(29.55, abs=0.1)
    assert xy[-1, 1] > 4.0
    assert np.array_equal(xy[0], [0, 0])
    assert np.isnan(ct.integrate([10, np.nan, 10], [0, 0, 0], 0.05)[-1]).all()


def test_contiguous_windows_never_span_overrides_stale_data_or_gaps():
    d = {"t": np.arange(100) * .05, "v": np.ones(100) * 8,
         "a": np.zeros(100), "yaw": np.zeros(100), "roll": np.zeros(100),
         "torque": np.zeros(100), "desired_curvature": np.zeros(100),
         "lat_active": np.ones(100), "long_active": np.ones(100),
         "accel_command": np.zeros(100), "accel_target": np.zeros(100),
         "calibrated": np.ones(100), "valid": np.ones(100),
         "steering_pressed": np.zeros(100), "brake_pressed": np.zeros(100),
         "gas_pressed": np.zeros(100)}
    assert ct.window_indices(d, "lat", seconds=1).size > 0
    d["steering_pressed"][20:80] = 1
    starts = ct.window_indices(d, "lat", seconds=1)
    assert all(not (s <= 79 and s + 20 >= 20) for s in starts)
    d["t"][15:] += 1
    assert all(not (s < 15 <= s + 20) for s in ct.window_indices(d, "lat", seconds=1))


def test_nonmonotonic_and_sparse_tables_are_rejected():
    assert not ct.usable_cell_table([-100, 0, 100], [0, 1, 0], "lat")
    assert ct.usable_cell_table([-100, 0, 100], [1, 0, -1], "lat")
    assert not ct.usable_cell_table([-3, 0, 2], [-1, 0, .01], "long")


def test_validation_never_promotes_unidentified_or_unknown_build():
    assert not ct.acceptance({"mae": .1, "p95": .2}, {"mae": .08, "p95": .19}, False)
    assert ct.acceptance({"mae": .1, "p95": .2}, {"mae": .08, "p95": .19}, True)
    assert not ct.acceptance({"mae": .1, "p95": .2}, {"mae": .11, "p95": .18}, True)


def test_cohorts_separate_model_and_hardware_contracts():
    base = {"git_commit": "abc", "model_identity": {"Model": "a"},
            "car_params": {"longitudinalActuatorDelay": .5}}
    other = dict(base, model_identity={"Model": "b"})
    assert ct.cohort_key(base) != ct.cohort_key(other)
    other = dict(base, car_params={"longitudinalActuatorDelay": .3})
    assert ct.cohort_key(base) != ct.cohort_key(other)


def test_common_population_does_not_reward_dropping_hard_windows():
    a = {"window_scores": {"easy": {"absolute": [0.1], "target": [0.1], "endpoint": .1},
                            "hard": {"absolute": [2.], "target": [2.], "endpoint": 2.}},
         "attempted_windows": 2}
    b = {"window_scores": {"easy": {"absolute": [0.05], "target": [.05], "endpoint": .05}},
         "attempted_windows": 2}
    paired = ct.paired_scores([a, b])
    assert paired[0]["mae"] == pytest.approx(.1)
    assert paired[0]["used_windows"] == paired[1]["used_windows"] == 1
    assert paired[1]["coverage"] == .5


def test_provenance_ignores_catalog_refresh_but_keeps_active_model():
    a = {"model_identity": {"ModelManager_ActiveBundle": "model-a",
                            "ModelManager_LastSyncTime": "1", "ModelManager_ModelsCache": "cache-a"}}
    b = {"model_identity": {"ModelManager_ActiveBundle": "model-a",
                            "ModelManager_LastSyncTime": "2", "ModelManager_ModelsCache": "cache-b"}}
    assert ct.cohort_key(a) == ct.cohort_key(b)
    b["model_identity"]["ModelManager_ActiveBundle"] = "model-b"
    assert ct.cohort_key(a) != ct.cohort_key(b)


def test_static_car_params_inherit_only_same_route_and_source_commit():
    seed = {"route": "r1", "segment": "r1--0", "segment_index": 0, "git_commit": "a", "dirty": False,
            "car_params": {"carFingerprint": "HYUNDAI_ELANTRA_2024"}, "fingerprint": "HYUNDAI_ELANTRA_2024"}
    child = {"route": "r1", "segment": "r1--1", "git_commit": "a", "dirty": False, "car_params": {}}
    wrong = {"route": "r1", "segment": "r1--2", "git_commit": "b", "dirty": False, "car_params": {}}
    other = {"route": "r2", "segment": "r2--1", "git_commit": "a", "dirty": False, "car_params": {}}
    ct.enrich_route_metadata([(None, m) for m in (seed, child, wrong, other)])
    assert child["fingerprint"] == "HYUNDAI_ELANTRA_2024"
    assert child["route_provenance_source"] == "r1--0"
    assert wrong["car_params"] == other["car_params"] == {}


def test_transient_rollout_uses_delayed_commands_and_native_limiter():
    d = {k: np.zeros(40) for k in ("a", "yaw", "roll", "torque")}
    d.update(t=np.arange(40) * .05, v=np.full(40, 5.), desired_curvature=np.full(40, .01))
    cell = {"state_min": -.25, "state_max": .25, "command": [-100., 0., 100.],
            "delta": [.03, 0., -.03]}
    model = {"kind": "median_transition", "delay_s": .1, "state_edges": [-.25, .25],
             "cells": {"3-7": {"0": cell}}}
    baseline, _ = ct.rollout(d, 12, model, "lat", seconds=.5)
    tuned, commands = ct.rollout(d, 12, model, "lat", .1, seconds=.5)
    assert np.all(baseline == 0)
    assert np.isfinite(tuned).all() and tuned[-1] > 0
    assert np.all(tuned[:3] == 0)  # two delayed command samples
    assert commands[0] == -15 and np.all(np.abs(commands) <= 409)


def test_rollout_gain_series_matches_scalar_and_applies_current_sample_gain():
    d = {k: np.zeros(40) for k in ("a", "yaw", "roll", "torque")}
    d.update(t=np.arange(40) * .05, v=np.full(40, 5.), desired_curvature=np.full(40, .01))
    cell = {"state_min": -.25, "state_max": .25, "command": [-100., 0., 100.], "delta": [.03, 0., -.03]}
    model = {"kind": "median_transition", "delay_s": 0., "state_edges": [-.25, .25],
             "cells": {"3-7": {"0": cell}}}
    scalar = ct.rollout(d, 12, model, "lat", .1, seconds=.5)
    series = ct.rollout(d, 12, model, "lat", np.full(11, .1), seconds=.5)
    np.testing.assert_array_equal(scalar, series)
    gains = np.r_[np.zeros(5), np.full(6, .1)]
    state, command = ct.rollout(d, 12, model, "lat", gains, seconds=.5)
    assert np.all(state[:6] == 0) and state[6] > 0
    assert np.all(command[:5] == 0) and command[5] < 0
    for invalid in ([.1], np.full((11, 1), .1), np.full(11, np.nan), -.1):
        with pytest.raises(ValueError, match="gain"):
            ct.rollout(d, 12, model, "lat", invalid, seconds=.5)
