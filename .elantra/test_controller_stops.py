"""Regressions for measured stop episodes and separate brake/acceleration response."""
import numpy as np
import pytest

import controller_stops as cs


def data(seconds=8):
  t = np.arange(int(seconds / cs.DT) + 1) * cs.DT
  n = len(t)
  h = (np.arange(33) / 32) ** 2 * 10
  d = {"t": t, "v": np.full(n, 5.), "a": np.zeros(n), "yaw": np.zeros(n),
       "accel_command": np.zeros(n), "accel_target": np.zeros(n),
       "model_t": t.copy(), "model_horizon": np.tile(h, (n, 1)),
       "model_x": np.tile(5 * np.minimum(h, 2), (n, 1)), "model_y": np.zeros((n, 33)),
       "source_segment": np.full(n, "route--0")}
  for key in ("long_active", "model_valid", "car_state_valid", "car_control_valid", "car_output_valid",
              "location_valid", "location_measurement_valid", "calibration_valid", "calibrated", "planner_valid"):
    d[key] = np.ones(n)
  for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
    d[key] = np.zeros(n)
  for key in ("age_longitudinalPlan", "age_carOutput", "age_carState"):
    d[key] = np.zeros(n)
  return d


def test_stopping_law_uses_latency_margin_and_existing_accel_limit():
  request, unreachable = cs.stopping_request(10, 30, .5, 0)
  assert request == pytest.approx(-2)
  assert not unreachable
  assert cs.stopping_request(10, 30, .5, 1)[0] < request
  assert cs.stopping_request(10, 5, .5, 0) == (-3.5, True)
  assert cs.stopping_request(10, 10, 0, 0) == (-3.5, True)
  assert cs.stopping_request(0, 0, 1, 2) == (0., False)
  assert cs.stopping_request(2, 100, 0, 0)[0] <= 0


@pytest.mark.parametrize("values", [(-1, 5, 0, 0), (1, -1, 0, 0), (1, 5, -1, 0),
                                     (1, 5, 0, -1), (np.nan, 5, 0, 0), (1, np.inf, 0, 0)])
def test_stopping_law_rejects_invalid_inputs(values):
  with pytest.raises(ValueError):
    cs.stopping_request(*values)


def test_nonuniform_model_path_detects_sustained_stationary_prediction():
  d = data()
  p = cs.prediction_at(d, 0)
  assert p["distance_m"] == pytest.approx(10)
  assert p["stationary_s"] >= 1
  # One slow point or a short pause cannot establish a predicted stop.
  d["model_x"][:] = 5 * d["model_horizon"]
  assert cs.prediction_at(d, 0) is None
  d["model_x"][0, 10] = d["model_x"][0, 9]
  assert cs.prediction_at(d, 0) is None


def test_model_prediction_rejects_future_stale_or_invalid_horizons():
  d = data()
  d["model_t"][0] = -.051
  assert cs.prediction_at(d, 0) is None
  d["model_t"][0] = .001
  assert cs.prediction_at(d, 0) is None
  d["model_t"][0] = 0
  d["model_horizon"][0, 10] = -.1
  assert cs.prediction_at(d, 0) is None


def test_measured_stop_distance_and_yaw_are_not_model_ground_truth():
  d = data()
  d["v"][40:] = 0
  d["yaw"][:] = .1
  events = cs.stop_events(d, "route", op_long=True)
  assert len(events) == 1
  e = events[0]
  assert e["outcome"] == "stopped"
  assert e["actual_stop_distance_m"] == pytest.approx(10, abs=.001)
  assert e["actual_stop_endpoint_xy_m"][1] > .9
  assert e["predicted_stop_distance_m"] == pytest.approx(10)
  assert e["comma_only_eligible"]
  assert e["prediction_error_m"] == pytest.approx(0, abs=.001)
  disabled = cs.stop_events(d, "route", op_long=False)[0]
  assert disabled["control_regime"] == "cp_long_disabled"
  assert not disabled["comma_only_eligible"]
  assert e["prediction_t"] == 0
  assert e["actual_stop_t"] == 2
  assert e["model_stop_xy"] == [10., 0.]
  assert e["measured_stop_xy"][1] > .9


def test_manual_intervention_censors_acceptance_but_keeps_actual_event():
  d = data()
  d["v"][40:] = 0
  d["brake_pressed"][20:] = 1
  d["long_active"][25:] = 0
  e = cs.stop_events(d, "route", op_long=True)[0]
  assert e["actual_stop_distance_m"] == pytest.approx(10)
  assert not e["comma_only_eligible"]
  assert e["comma_only_censor_s"] == 1
  assert e["first_manual_brake_s"] == 1
  assert e["first_disengage_s"] == 1.25
  assert e["comma_only_prefix_distance_m"] == pytest.approx(5)
  assert e["censor_t"] == 1
  assert e["censor_cause"] == ["manual_brake"]


def test_inactive_predicted_stop_is_retained_separately_from_comma_acceptance():
  d = data()
  d["long_active"][:] = 0
  d["v"][40:] = 0
  e = cs.stop_events(d, "route", op_long=True)[0]
  assert e["control_regime"] == "longitudinal_inactive"
  assert not e["comma_only_eligible"]
  assert e["actual_stop_distance_m"] == pytest.approx(10)
  assert e["censor_cause"] == ["not_comma_longitudinal_at_prediction"]


def test_braking_and_acceleration_delay_both_admit_below_three_mps():
  for direction in (-1, 1):
    d = data(seconds=5)
    d["v"][:] = 2
    d["accel_command"][20:] = direction * .6
    d["a"][26:] = direction * .4
    events = cs.response_events(d, "route", op_long=True)
    valid = [e for e in events if e["status"] == "observed_response"]
    assert len(valid) == 1
    assert valid[0]["direction"] == ("braking" if direction < 0 else "acceleration")
    assert valid[0]["response_latency_s"] == pytest.approx(.3)


@pytest.mark.parametrize("where", [18, 23])
def test_response_rejects_driver_override_in_prefix_or_effect(where):
  d = data(seconds=5)
  d["accel_command"][20:] = -.6
  d["a"][26:] = -.4
  d["gas_pressed"][where] = 1
  events = cs.response_events(d, "route", op_long=True)
  assert not any(e["status"] == "observed_response" for e in events)
  assert events[0]["status"] == "intervention_or_invalid_window"


def test_response_reports_no_response_without_inventing_latency():
  d = data(seconds=5)
  d["accel_command"][20:] = -.6
  e = cs.response_events(d, "route", op_long=True)[0]
  assert e["status"] == "no_observed_response"
  assert e["response_latency_s"] is None


def test_segment_stitch_and_invalid_telemetry_break_actual_motion(tmp_path):
  a, b = data(seconds=1), data(seconds=1)
  b["t"] += 1.05
  p0, p1 = tmp_path / "r--0.npz", tmp_path / "r--1.npz"
  np.savez_compressed(p0, **a)
  np.savez_compressed(p1, **b)
  blocks, coverage = cs.stitch_route([(p1, {}), (p0, {})], calibration_hold=False)
  assert len(blocks) == 1
  assert len(blocks[0]["t"]) == 42
  assert coverage["contiguous_segment_joins"] == 1
  b["t"] += .1
  b["location_measurement_valid"][10] = 0
  np.savez_compressed(p1, **b)
  blocks, coverage = cs.stitch_route([(p0, {}), (p1, {})], calibration_hold=False)
  assert len(blocks) == 3
  assert coverage["invalid_motion_rows"] == 1
  assert coverage["segment_gap_breaks"] == 1


def test_training_latency_uses_training_events_only_and_separates_direction():
  events = [{"route": "train", "direction": "braking", "status": "observed_response", "response_latency_s": .3},
            {"route": "test", "direction": "braking", "status": "observed_response", "response_latency_s": 2.},
            {"route": "train", "direction": "acceleration", "status": "observed_response", "response_latency_s": .6}]
  result = cs.latency_parameters(events, {"train": "train", "test": "test"}, known_build=True)
  assert result["braking"]["latency_s"] == .3
  assert result["acceleration"]["latency_s"] == .6
  assert result["braking"]["validity"] == "underidentified"
  assert result["braking"]["training_events"] == 1


def test_stop_onsets_ignore_invalid_stale_and_nonfinite_negative_signals():
  d = data()
  d["v"][40:] = 0
  d["accel_target"][:] = -.5
  d["accel_command"][:] = -.6
  d["planner_valid"][:10] = 0
  d["age_carOutput"][:15] = .25
  d["accel_command"][15] = np.nan
  e = cs.stop_events(d, "route", op_long=True)[0]
  assert e["first_planner_negative_s"] == pytest.approx(.5)
  assert e["first_command_negative_s"] == pytest.approx(.8)
  assert e["onset_signal_coverage"]["planner_negative"]["unavailable_rows"] == 10
  assert e["onset_signal_coverage"]["command_negative"]["unavailable_rows"] == 16
  d["planner_valid"][:] = 0
  e = cs.stop_events(d, "route", op_long=True)[0]
  assert e["first_planner_negative_s"] is None
  assert e["onset_signal_coverage"]["planner_negative"]["valid_fresh_rows"] == 0


def test_measured_deceleration_requires_fresh_valid_whole_sustained_slice():
  d = data(seconds=1)
  d["a"][:] = -.5
  d["age_carState"][2] = .3
  onset, coverage = cs.signal_onset(d, 0, len(d["t"]), "a", "car_state_valid", "age_carState", -.1, .15)
  assert onset == pytest.approx(.15)
  assert coverage["unavailable_rows"] == 1
