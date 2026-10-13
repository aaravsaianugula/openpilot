"""Behavioral checks for empirical transient lateral dynamics."""

import numpy as np
import pytest

import controller_dynamics as dynamics


def record(route, deltas=(.18, .03, -.03, -.18), repetitions=100):
  commands = (-300., -120., 120., 300.)
  n = len(commands) * repetitions * 3
  data = {"t": np.arange(n) * .05, "v": np.full(n, 5.), "yaw": np.zeros(n), "roll": np.zeros(n),
          "torque": np.zeros(n), "lat_active": np.ones(n), "steering_pressed": np.zeros(n),
          "brake_pressed": np.zeros(n), "gas_pressed": np.zeros(n), "calibrated": np.ones(n),
          "calibration_valid": np.ones(n), "model_valid": np.ones(n), "valid_lat": np.ones(n)}
  for group, (command, delta) in enumerate(zip(commands, deltas, strict=True)):
    for j in range(repetitions):
      i = (group * repetitions + j) * 3
      data["torque"][i:i+3] = command
      data["yaw"][i+1] = delta / 5.
      data["valid_lat"][i+2] = 0  # Only the intended transition is usable.
  return {"route": route}, data


def fitted():
  return dynamics.fit([record("train-a"), record("train-b")], 0.)


def test_nonlinear_transition_uses_medians_without_a_fitted_slope():
  model = fitted()
  assert model["kind"] == "median_transition" and model["dt"] == .05
  assert dynamics.forward(model, 5., 0., -300.) == pytest.approx(.18)
  assert dynamics.forward(model, 5., 0., -120.) == pytest.approx(.03)
  assert dynamics.forward(model, 5., 0., 120.) == pytest.approx(-.03)
  assert dynamics.inverse(model, 5., 0., .03) == pytest.approx(-120.)
  assert dynamics.reachable(model, 5., 0.) == pytest.approx((-.18, .18))
  assert dynamics.forward(model, 5., 0., 0.) == pytest.approx(0.)


def test_no_command_state_speed_or_response_extrapolation():
  model = fitted()
  for speed, state, command in ((5., 0., 409.), (5., .3, 0.), (20., 0., 0.), (np.nan, 0., 0.)):
    assert np.isnan(dynamics.forward(model, speed, state, command))
  assert np.isnan(dynamics.inverse(model, 5., 0., .3))
  assert dynamics.reachable(model, 5., .3) is None
  assert dynamics.reachable(model, 20., 0.) is None


def test_reachable_endpoint_remains_invertible_after_floating_point_subtraction():
  model = fitted()
  cell = next(iter(next(iter(model["cells"].values())).values()))
  cell["state_min"], cell["state_max"] = -.2, .2
  low, high = dynamics.reachable(model, 5., .1)
  assert dynamics.inverse(model, 5., .1, low) == pytest.approx(300.)
  assert dynamics.inverse(model, 5., .1, high) == pytest.approx(-300.)
  assert np.isnan(dynamics.inverse(model, 5., .1, high + 1e-5))


def test_sparse_single_route_nonmonotone_and_weak_response_have_no_default():
  cases = [[record("one-route")], [record("a", repetitions=20), record("b", repetitions=20)],
           [record("a", (.18, -.03, .03, -.18)), record("b", (.18, -.03, .03, -.18))],
           [record("a", (.003, .001, -.001, -.003)), record("b", (.003, .001, -.001, -.003))]]
  for records in cases:
    model = dynamics.fit(records, 0.)
    assert not model["cells"]
    assert np.isnan(dynamics.forward(model, 5., 0., 0.))
    assert np.isnan(dynamics.inverse(model, 5., 0., 0.))


def test_delay_requires_clean_whole_command_to_effect_prefix():
  # Every fixture has an invalid prefix row within each four-row .1 s delay interval.
  model = dynamics.fit([record("a"), record("b")], .1)
  assert not model["cells"]
  assert model["support"]["excluded"]["unclean_interval"] > 0
  with pytest.raises(ValueError, match="20 Hz"):
    dynamics.fit([record("a"), record("b")], .025)


def test_delayed_transition_uses_past_command_instead_of_current_command():
  records = []
  for route in ("a", "b"):
    meta, original = record(route)
    chunks = len(original["t"]) // 3
    data = {key: np.repeat(value[::3], 5) for key, value in original.items()}
    data["t"] = np.arange(chunks * 5) * .05
    data["yaw"][:] = 0
    data["valid_lat"][:] = 1
    data["torque"][:] = 0
    for j in range(chunks):
      source, start = j * 3, j * 5
      data["torque"][start] = original["torque"][source]
      data["yaw"][start + 3] = original["yaw"][source + 1]
      data["valid_lat"][start + 4] = 0
    records.append((meta, data))
  model = dynamics.fit(records, .1)
  assert dynamics.forward(model, 5., 0., -120.) == pytest.approx(.03)
  cell = next(iter(next(iter(model["cells"].values())).values()))
  assert 0. not in cell["command"]


@pytest.mark.parametrize("key,value", [("steering_pressed", 1.), ("brake_pressed", 1.), ("gas_pressed", 1.),
                                       ("valid_lat", 0.), ("model_valid", 0.), ("calibrated", np.nan),
                                       ("calibration_valid", 0.), ("torque", 410.), ("yaw", np.nan)])
def test_overrides_stale_missing_and_over_ceiling_data_cannot_train(key, value):
  records = [record("a"), record("b")]
  for _, data in records:
    data[key][:] = value
  model = dynamics.fit(records, 0.)
  assert not model["cells"]


def test_missing_contract_and_timestamp_gaps_are_named():
  records = [record("a"), record("b")]
  del records[0][1]["model_valid"]
  records[1][1]["t"][1::3] += .1
  model = dynamics.fit(records, 0.)
  assert not model["cells"]
  assert model["support"]["missing_fields"] == [{"route": "a", "fields": ["model_valid"]}]
  assert model["support"]["excluded"]["time_gap"] > 0


def test_training_population_is_explicit_and_inputs_are_not_mutated():
  train = [record("train-a"), record("train-b")]
  holdout = record("holdout-never-read", (10., 5., -5., -10.))
  original = train[0][1]["yaw"].copy()
  model = dynamics.fit(train, 0.)
  np.testing.assert_array_equal(train[0][1]["yaw"], original)
  assert model["support"]["training_routes"] == ["train-a", "train-b"]
  assert holdout[0]["route"] not in str(model)
  assert model["support"]["accepted_cells"] > 0
  cell = next(iter(next(iter(model["cells"].values())).values()))
  assert all(n == 200 for n in cell["n"])
  assert all(n == 2 for n in cell["routes"])
  assert "delta_p25" in cell and "delta_p75" in cell


def test_roll_is_removed_before_transition_binning():
  records = [record("a"), record("b")]
  for _, data in records:
    data["roll"][:] = .02
    data["yaw"] += 9.81 * .02 / data["v"]
  model = dynamics.fit(records, 0.)
  assert dynamics.forward(model, 5., 0., -120.) == pytest.approx(.03)
