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


def dynamic_score(ids, error=1., attempted=100):
    return {"attempted_windows": attempted, "window_scores": {
        str(i): {"absolute": [error], "target": [error], "endpoint": error} for i in ids}}


def test_training_dynamics_rejects_unsupported_before_fixing_common_population():
    raw = [dynamic_score(range(20)), dynamic_score(range(5, 20), .5),
           dynamic_score([], 0.), dynamic_score(range(9), 0.)]
    scores, support = ct.training_dynamic_scores(raw)
    assert support["data_partition"] == "training_only"
    assert support["common_population_windows"] == 15
    assert [score["eligible"] for score in scores] == [True, True, False, False]
    assert scores[0]["individual_used_windows"] == 20
    assert scores[0]["used_windows"] == scores[1]["used_windows"] == 15
    assert scores[0]["window_ids"] == scores[1]["window_ids"]
    assert scores[1]["coverage"] == pytest.approx(.15)
    assert scores[2]["individual_coverage"] == scores[2]["common_coverage"] == 0
    assert "insufficient_individual_windows" in scores[2]["rejection_reasons"]
    assert scores[3]["individual_used_windows"] == 9
    assert scores[3]["common_supported_windows"] == 4
    assert scores[3]["mae"] is None


def test_training_dynamics_cannot_shrink_denominator_or_cherry_pick_common_support():
    scores, support = ct.training_dynamic_scores([
        dynamic_score(range(100), attempted=1000), dynamic_score(range(10), 0., attempted=10)])
    assert support["attempted_windows"] == 1000
    assert scores[1]["reported_attempted_windows"] == 10
    assert scores[1]["individual_coverage"] == pytest.approx(.01)
    assert not scores[1]["eligible"]
    assert "insufficient_individual_coverage" in scores[1]["rejection_reasons"]
    assert scores[0]["eligible"] and scores[0]["used_windows"] == 100
    scores, support = ct.training_dynamic_scores([
        dynamic_score(range(20)), dynamic_score(range(15, 35), 0.)])
    assert support["individually_eligible_candidates"] == 2
    assert support["common_population_windows"] == 5
    assert all(not score["eligible"] for score in scores)
    assert all("insufficient_common_windows" in score["rejection_reasons"] for score in scores)
    assert all(score["coverage"] == pytest.approx(.05) for score in scores)


def test_identify_recovers_interior_response_slower_than_old_tau_grid(monkeypatch):
    table = {"3-7": {"command": [-250., 0., 250.], "response": [1.25, 0., -1.25]}}
    monkeypatch.setattr(ct, "fit_table", lambda records, axis: (table, {}))
    records = []
    for route in ("training-a", "training-b"):
        n = 900
        d = {key: np.zeros(n) for key in ("a", "yaw", "roll", "desired_curvature",
             "steering_pressed", "brake_pressed", "gas_pressed")}
        d.update(t=np.arange(n) * ct.DT, v=np.full(n, 5.), lat_active=np.ones(n),
                 calibrated=np.ones(n), valid_lat=np.ones(n))
        d["torque"] = 150 * np.sin(d["t"] * 1.1) + 40 * np.sin(d["t"] * 3.2)
        state = np.zeros(n)
        for i in range(n - 1):
            command = d["torque"][i - 6] if i >= 6 else 0.
            state[i + 1] = state[i] + ct.DT / (1.2 + ct.DT) * (-command * .005 - state[i])
        d["yaw"] = state / d["v"]
        records.append(({"route": route, "segment": route + "--0"}, d))
    model, support = ct.identify(records, "lat")
    assert model["tau_s"] == 1.2 and model["delay_s"] == .3
    assert model["training_prediction"]["mae"] < 1e-12
    assert not model["boundary_optimum"] and model["boundary_parameters"] == {}
    assert model["search_status"] == "finite_training_grid_optimum_not_convergence"
    assert len(support["dynamic_grid"]) == 63
    assert {row["tau_s"] for row in support["dynamic_grid"]} == {.05, .1, .2, .35, .5, .8, 1.2, 2., 3.}


def test_identify_retains_failed_candidate_without_erasing_supported_training(monkeypatch):
    monkeypatch.setattr(ct, "fit_table", lambda records, axis: ({"known": {}}, {}))
    def score(records, model, axis, stride):
        if model["tau_s"] == .05 and model["delay_s"] == .6:
            return dynamic_score([])
        return dynamic_score(range(20), abs(model["tau_s"] - 1.2) + abs(model["delay_s"] - .3))
    monkeypatch.setattr(ct, "score_model", score)
    model, support = ct.identify(["training-only"], "lat")
    assert (model["delay_s"], model["tau_s"]) == (.3, 1.2)
    assert model["training_prediction"]["used_windows"] == 20
    rejected = [row for row in support["dynamic_grid"] if not row["eligible"]]
    assert len(support["dynamic_grid"]) == 63 and len(rejected) == 1
    assert rejected[0]["individual_used_windows"] == 0
    assert "insufficient_individual_windows" in rejected[0]["rejection_reasons"]


@pytest.mark.parametrize(("delay", "tau", "boundaries"), [
    (0., 1.2, {"delay_s": "minimum"}), (.6, 1.2, {"delay_s": "maximum"}),
    (.3, .05, {"tau_s": "minimum"}), (.3, 3., {"tau_s": "maximum"}), (.3, .1, {})])
def test_identify_marks_actual_grid_boundaries(monkeypatch, delay, tau, boundaries):
    monkeypatch.setattr(ct, "fit_table", lambda records, axis: ({"known": {}}, {}))
    monkeypatch.setattr(ct, "score_model", lambda records, model, axis, stride: dynamic_score(
        range(20), abs(model["delay_s"] - delay) + abs(model["tau_s"] - tau)))
    model, _ = ct.identify(["training-only"], "long")
    assert model["boundary_parameters"] == boundaries
    assert model["boundary_optimum"] == bool(boundaries)


def test_transient_keeps_empty_cells_and_unsupported_candidates(monkeypatch):
    import controller_dynamics

    monkeypatch.setattr(controller_dynamics, "fit", lambda records, delay: {
        "delay_s": delay, "cells": {} if delay == .6 else {"known": {}}, "support": {"delay_s": delay}})
    monkeypatch.setattr(ct, "score_model", lambda records, model, axis, stride: dynamic_score(
        [] if model["delay_s"] == .05 else range(20), abs(model["delay_s"] - .3)))
    model, support = ct.identify_transient(["training-only"])
    assert model["delay_s"] == .3 and not model["boundary_optimum"]
    assert len(support["dynamic_grid"]) == len(support["cell_support"]) == 8
    empty = next(row for row in support["dynamic_grid"] if row["delay_s"] == .6)
    assert not empty["eligible"] and "no_supported_transient_cells" in empty["rejection_reasons"]
    unsupported = next(row for row in support["dynamic_grid"] if row["delay_s"] == .05)
    assert not unsupported["eligible"] and unsupported["individual_used_windows"] == 0
    assert model["training_prediction"]["used_windows"] == 20


def test_no_supported_table_retains_declared_grid_and_reasons(monkeypatch):
    monkeypatch.setattr(ct, "fit_table", lambda records, axis: ({}, {"cells": []}))
    model, support = ct.identify([], "lat")
    assert model is None and len(support["dynamic_grid"]) == 63
    assert all("no_supported_steady_table" in row["rejection_reasons"] for row in support["dynamic_grid"])


@pytest.mark.parametrize("delay", [0., .6])
def test_transient_marks_both_delay_grid_boundaries(monkeypatch, delay):
    import controller_dynamics

    monkeypatch.setattr(controller_dynamics, "fit", lambda records, value: {
        "delay_s": value, "cells": {"known": {}}, "support": {}})
    monkeypatch.setattr(ct, "score_model", lambda records, model, axis, stride: dynamic_score(
        range(20), abs(model["delay_s"] - delay)))
    model, _ = ct.identify_transient(["training-only"])
    assert model["boundary_optimum"]
    assert model["boundary_parameters"] == {"delay_s": "minimum" if delay == 0 else "maximum"}
