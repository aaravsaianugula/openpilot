"""The action head -> steering/accel conversion, checked against OpenPilot's.

This is the arithmetic that actually drives the car when an RDF model is active,
and it is the one part of the chain that can be exercised without a GPU. Values
are compared against commaai/openpilot modeld.get_action_from_model and
drive_helpers.should_stop.
"""

import numpy as np
import pytest

from cereal import log
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
from openpilot.sunnypilot.modeld_v2.modeld import ModelState


LAT_SMOOTH = 0.1   # rdf-driving's LAT_SMOOTH_SECONDS, carried as the bundle's lat override
LONG_SMOOTH = 0.3  # LONG_SMOOTH_SECONDS


MIN_LAT_CONTROL_SPEED = 0.3  # same value as OpenPilot's


def make_model(lat=LAT_SMOOTH, long=LONG_SMOOTH):
  """A ModelState with only what get_action_from_model touches."""
  model = ModelState.__new__(ModelState)
  model.LAT_SMOOTH_SECONDS = lat
  model.LONG_SMOOTH_SECONDS = long
  model.MIN_LAT_CONTROL_SPEED = MIN_LAT_CONTROL_SPEED
  model.PLANPLUS_CONTROL = 1.0
  model.constants = ModelConstants()
  model.generation = 12  # RDF bundles
  return model


def action_output(curv_term: float, accel: float):
  # slice 2062:2066 is 4 wide; only [0] and [1] are read.
  return {'action': np.array([[curv_term, accel, 0.0, 0.0]], dtype=np.float32)}


def zero_action():
  return log.ModelDataV2.Action(desiredCurvature=0.0, desiredAcceleration=0.0, shouldStop=False)


def test_accel_comes_from_action_index_1():
  model = make_model(long=0.0)  # no smoothing, read the raw value through
  out = model.get_action_from_model(action_output(0.0, -1.25), zero_action(), 0.2, 0.2, v_ego=20.0)

  assert out.desiredAcceleration == pytest.approx(-1.25, abs=1e-5)


@pytest.mark.parametrize("v_ego,curv_term,expected", [
  (20.0, 4.0, 4.0 / 400.0),    # divided by v_ego squared
  (10.0, 4.0, 4.0 / 100.0),
  (0.5, 4.0, 4.0 / 1.0),       # max(1.0, v_ego) floors the divisor at 1
])
def test_curvature_is_action_index_0_over_v_ego_squared(v_ego, curv_term, expected):
  model = make_model(lat=0.0)
  out = model.get_action_from_model(action_output(curv_term, 0.0), zero_action(), 0.2, 0.2, v_ego=v_ego)

  assert out.desiredCurvature == pytest.approx(expected, rel=1e-5)


def test_curvature_is_held_at_the_previous_value_below_min_lat_control_speed():
  # OpenPilot holds the last curvature below MIN_LAT_CONTROL_SPEED rather than
  # acting on the model at a standstill, where curvature is ill-conditioned.
  model = make_model(lat=0.0)
  prev = log.ModelDataV2.Action(desiredCurvature=0.05, desiredAcceleration=0.0, shouldStop=False)

  held = model.get_action_from_model(action_output(4.0, 0.0), prev, 0.2, 0.2, v_ego=0.1)
  assert held.desiredCurvature == pytest.approx(0.05)

  # Just above the threshold the model's own curvature takes over again.
  live = model.get_action_from_model(action_output(4.0, 0.0), prev, 0.2, 0.2, v_ego=0.4)
  assert live.desiredCurvature == pytest.approx(4.0 / 1.0, rel=1e-5)


def test_curvature_sign_is_preserved():
  # A sign flip here steers the wrong way; pin both directions.
  model = make_model(lat=0.0)
  left = model.get_action_from_model(action_output(2.0, 0.0), zero_action(), 0.2, 0.2, v_ego=10.0)
  right = model.get_action_from_model(action_output(-2.0, 0.0), zero_action(), 0.2, 0.2, v_ego=10.0)

  assert left.desiredCurvature > 0
  assert right.desiredCurvature < 0
  assert left.desiredCurvature == pytest.approx(-right.desiredCurvature)


@pytest.mark.parametrize("v_ego,accel,expected", [
  (0.2, 0.0, True),     # stopped and not accelerating -> stop
  (0.2, 0.5, False),    # accelerating away
  (5.0, -2.0, False),   # braking hard but still moving
  (0.3, 0.0, False),    # boundary: v_ego < 0.3 is strict
])
def test_should_stop_matches_openpilot(v_ego, accel, expected):
  # openpilot drive_helpers: should_stop(v_ego, a) -> v_ego < 0.3 and a < 0.1
  model = make_model(long=0.0)
  out = model.get_action_from_model(action_output(0.0, accel), zero_action(), 0.2, 0.2, v_ego=v_ego)

  assert out.shouldStop is expected


def test_smoothing_pulls_toward_the_previous_action():
  # smooth_value is a first-order lag, so one step must land strictly between
  # the previous value and the new one - never overshoot either.
  model = make_model()
  prev = log.ModelDataV2.Action(desiredCurvature=0.0, desiredAcceleration=0.0, shouldStop=False)
  out = model.get_action_from_model(action_output(0.0, -2.0), prev, 0.2, 0.2, v_ego=20.0)

  assert -2.0 < out.desiredAcceleration < 0.0


def test_unsmoothed_output_is_the_raw_action():
  model = make_model(lat=0.0, long=0.0)
  out = model.get_action_from_model(action_output(8.0, -1.0), zero_action(), 0.2, 0.2, v_ego=20.0)

  assert out.desiredAcceleration == pytest.approx(-1.0, abs=1e-5)
  assert out.desiredCurvature == pytest.approx(8.0 / 400.0, rel=1e-5)


def test_action_path_ignores_plan_entirely():
  # With an action head present the plan must not influence control, matching
  # openpilot. A plan full of garbage must not change the commanded values.
  model = make_model(lat=0.0, long=0.0)
  out = action_output(4.0, -0.5)
  out['plan'] = np.full((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), 1e6, dtype=np.float32)

  result = model.get_action_from_model(out, zero_action(), 0.2, 0.2, v_ego=20.0)

  assert result.desiredAcceleration == pytest.approx(-0.5, abs=1e-5)
  assert result.desiredCurvature == pytest.approx(4.0 / 400.0, rel=1e-5)
