"""
Tests for the CN7 low-speed feedforward gain schedule.

The load-bearing property is that the schedule touches the FEEDFORWARD and nothing else: for a
given error the proportional contribution must be bit-identical at every speed, and at and above
LEARNER_MIN_VEL the whole controller must be bit-identical to today. Both are asserted with `==`,
not approximately, because "close enough on the highway" is not the claim being made.

`openpilot.common.params` is ALWAYS replaced here with an in-memory stand-in, deliberately.

These tests select controller variants by writing LateralJerkTorqueController and
NeuralNetworkLateralControl. Against the real param store -- which is what you get when this runs
ON THE DEVICE -- those writes change the car's live configuration, and a run that aborts partway
leaves it changed. A suite that can switch the feedforward correction off on a real vehicle as a
side effect of testing it is not a trade worth making for any coverage it buys. It also races:
Params.put_bool is non-blocking by default, so an immediately following read can return the stale
value, which is exactly what it did the first time this ran on the car.

Nothing here needs a real param store. The check that the CAR is configured correctly lives in
.elantra/verify_lat_fix.py, which reads the real params and writes nothing.

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import ast
import math
import os
import sys
import types

import numpy as np
import pytest

_stub = types.ModuleType("openpilot.common.params")

if True:  # keeps the stand-in definitions indented together, and out of the way

  class _StubParams:
    _store: dict = {}

    def __init__(self, d=None):
      pass

    def get_bool(self, key, *args, **kwargs):
      return bool(_StubParams._store.get(key, False))

    def put_bool(self, key, value, **kwargs):
      _StubParams._store[key] = bool(value)

    def get(self, key, *args, **kwargs):
      return _StubParams._store.get(key)

    def put(self, key, value, **kwargs):
      _StubParams._store[key] = value

    def remove(self, key):
      _StubParams._store.pop(key, None)

  class _UnknownKeyName(Exception):
    pass

  _stub.Params = _StubParams
  _stub.UnknownKeyName = _UnknownKeyName
  # Unconditional, and before anything that imports it transitively. On the device the real store
  # IS the car's configuration -- see the module docstring.
  sys.modules["openpilot.common.params"] = _stub
  from openpilot.common.params import Params

from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI, CarControllerParams, HyundaiFlags
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log
from openpilot.common.basedir import BASEDIR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib import lat_accel_factor_schedule as sched
from openpilot.sunnypilot.selfdrive.controls.lib.lat_accel_factor_schedule import (
  FF_LAT_ACCEL_GAIN_BP,
  FF_LAT_ACCEL_GAIN_V,
  KP_SCALE_BP,
  KP_SCALE_V,
  LEARNER_MIN_VEL,
  ff_gain_applies,
  kp_low_speed_scale,
  lat_accel_factor_gain,
  scaled_kp_interp,
)
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import (
  INTERP_SPEEDS,
  KP_INTERP,
  LatControlTorque as LatControlTorqueV0,
)

CN7 = HYUNDAI.HYUNDAI_ELANTRA_2024
OTHER_HYUNDAI = HYUNDAI.HYUNDAI_SONATA
LOW_SPEEDS = (3.0, 5.0, 6.7, 8.0, 10.0, 12.0, 14.0, 14.99)
HIGH_SPEEDS = (LEARNER_MIN_VEL, 16.0, 22.0, 30.0, 40.0)
GAIN_FLOOR = 0.38
KP_SCALE_FLOOR = 0.26
EPS_CEILING = 409

# The measured p90 tracking error per band, from the 409-count archive, hands-off turn frames.
# These are what the KP cap was derived from; if the derivation is ever redone they move together.
MEASURED_ERR_P90 = ((2.5, 0.316), (3.5, 0.331), (4.5, 0.537), (5.5, 0.582),
                    (7.0, 0.634), (9.0, 0.548), (11.5, 0.346), (14.5, 0.379))


def _make_controller(car_name=CN7, jerk_aware=True):
  params = Params()
  params.put_bool("EnforceTorqueControl", True)
  params.put_bool("LateralJerkTorqueController", jerk_aware)
  params.put_bool("NeuralNetworkLateralControl", False)

  car_interface = interfaces[car_name]
  CP = car_interface.get_non_essential_params(car_name)
  CP_SP = car_interface.get_non_essential_params_sp(CP, car_name)
  CI = car_interface(CP, CP_SP)
  VM = VehicleModel(CP)
  controller = LatControlTorqueV0(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
  return controller, VM, CP


def _make_model_v2(v_ego):
  model = log.ModelDataV2.new_message()
  n = len(ModelConstants.T_IDXS)
  for field, values in (("position", [float(x) for x in v_ego * np.array(ModelConstants.T_IDXS)]),
                        ("orientation", [0.0] * n),
                        ("velocity", [float(v_ego)] * n),
                        ("acceleration", [0.0] * n)):
    data = log.XYZTData.new_message()
    data.x = values
    data.y = [0.0] * n
    data.z = [0.0] * n
    setattr(model, field, data)
  return model


def _drive(controller, VM, v_ego, desired_curvature, frames=1, lat_delay=0.2, steering_angle_deg=0.0):
  """Run `frames` identical control cycles; return the last (torque, pid_log)."""
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.aEgo = 0.0
  CS.steeringAngleDeg = steering_angle_deg
  CS.steeringRateDeg = 0.0
  CS.steeringPressed = False
  vehicle_params = log.VehicleParameters.new_message()  # roll and angleOffsetDeg are 0.0
  model_v2 = _make_model_v2(v_ego)

  out = None
  for _ in range(frames):
    # controlsd's ordering: refresh the live torque params and the limits, then step the controller.
    # Note LatControlTorqueExtBase binds self._pid to the base controller's PID inside update(), so
    # on the very first cycle update_limits() acts on the extension's own throwaway PIDController and
    # the real one keeps its lat-accel-space limits until cycle two. Mirrored here on purpose.
    controller.extension.update_limits()
    controller.extension.update_model_v2(model_v2)
    controller.extension.update_lateral_lag(lat_delay)
    # calibrated_pose is only read by the neural feedforward, which is off here.
    torque, _, pid_log = controller.update(True, CS, VM, vehicle_params, False, desired_curvature, None, False, lat_delay)
    out = (torque, pid_log)
  return out


def _arm(blend, v_ego, curvature, frames=1, car_name=CN7, jerk_aware=True, tracking=False):
  """One fresh controller at one blend setting. Fresh so the PID state matches between arms."""
  previous = sched.LOW_SPEED_FF_BLEND
  sched.LOW_SPEED_FF_BLEND = blend
  try:
    controller, VM, CP = _make_controller(car_name, jerk_aware)
    # `tracking` puts the wheel where the vehicle model says this curvature lives, so the error is
    # ~0 and the output is the feedforward rather than a railed proportional term.
    angle = math.degrees(VM.get_steer_from_curvature(-curvature, v_ego, 0.0)) if tracking else 0.0
    torque, pid_log = _drive(controller, VM, v_ego, curvature, frames, steering_angle_deg=angle)
    return torque, pid_log, CP
  finally:
    sched.LOW_SPEED_FF_BLEND = previous


# ---- the schedule itself ---------------------------------------------------


def test_schedule_shape():
  assert len(FF_LAT_ACCEL_GAIN_BP) == len(FF_LAT_ACCEL_GAIN_V) >= 2
  assert list(FF_LAT_ACCEL_GAIN_BP) == sorted(FF_LAT_ACCEL_GAIN_BP)
  assert len(set(FF_LAT_ACCEL_GAIN_BP)) == len(FF_LAT_ACCEL_GAIN_BP), "np.interp needs increasing breakpoints"
  assert list(FF_LAT_ACCEL_GAIN_V) == sorted(FF_LAT_ACCEL_GAIN_V), "the gain must not fall as speed rises"
  assert FF_LAT_ACCEL_GAIN_V[-1] == 1.0
  assert max(FF_LAT_ACCEL_GAIN_V) == 1.0, "the schedule may never command more than the learner above its floor"
  assert min(FF_LAT_ACCEL_GAIN_V) >= GAIN_FLOOR, "1/0.38 = 2.63x is the most feedforward this may add"
  assert FF_LAT_ACCEL_GAIN_BP[-1] == LEARNER_MIN_VEL
  assert LEARNER_MIN_VEL == 15.0


def test_learner_floor_matches_torqued():
  """The schedule ends at MIN_VEL because that is where the learner's own evidence begins.

  Read out of the source rather than imported: torqued pulls in msgq, which is not present on an
  unbuilt checkout, and this equality has to be checkable everywhere.
  """
  source = open(os.path.join(BASEDIR, "openpilot", "selfdrive", "locationd", "torqued.py"), encoding="utf-8").read()
  found = [ast.literal_eval(node.value) for node in ast.parse(source).body
           if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MIN_VEL"]
  assert len(found) == 1, "could not read MIN_VEL out of torqued.py"
  assert float(found[0]) == LEARNER_MIN_VEL


def test_gain_is_exactly_one_at_and_above_the_learner_floor():
  for v in HIGH_SPEEDS + (1e6,):
    assert lat_accel_factor_gain(v) == 1.0, f"gain must be exactly 1.0 at {v} m/s"


def test_gain_is_bounded_and_monotone():
  previous = 0.0
  for v in np.arange(0.0, 40.0, 0.05):
    gain = lat_accel_factor_gain(float(v))
    assert 0.0 < gain <= 1.0
    assert 1.0 / gain <= 1.0 / GAIN_FLOOR + 1e-9
    assert gain >= previous - 1e-12, f"the gain fell going up in speed at {v} m/s"
    previous = gain


def test_nan_v_ego_falls_back_to_today():
  assert lat_accel_factor_gain(float("nan")) == 1.0


def test_blend_zero_is_the_identity():
  previous = sched.LOW_SPEED_FF_BLEND
  sched.LOW_SPEED_FF_BLEND = 0.0
  try:
    for v in (0.0, 3.0, 6.0, 10.0, 14.0, 20.0):
      assert lat_accel_factor_gain(v) == 1.0
  finally:
    sched.LOW_SPEED_FF_BLEND = previous


# ---- the proportional-gain cap ---------------------------------------------


def test_kp_scale_shape():
  assert len(KP_SCALE_BP) == len(KP_SCALE_V) >= 2
  assert list(KP_SCALE_BP) == sorted(KP_SCALE_BP)
  assert len(set(KP_SCALE_BP)) == len(KP_SCALE_BP), "np.interp needs increasing breakpoints"
  assert list(KP_SCALE_V) == sorted(KP_SCALE_V), "the gain must not be cut harder as speed rises"
  assert KP_SCALE_V[-1] == 1.0
  assert max(KP_SCALE_V) == 1.0, "this may only ever reduce the stock gain, never raise it"
  assert min(KP_SCALE_V) >= KP_SCALE_FLOOR, "a 3.8x cut is the most this may apply"


def test_kp_scale_is_exactly_one_above_the_cap_range():
  for v in (KP_SCALE_BP[-1], 10.0, 15.0, 22.0, 30.0, 40.0, 1e6):
    assert kp_low_speed_scale(v) == 1.0, f"KP scale must be exactly 1.0 at {v} m/s"


def test_kp_scale_bounded_monotone_and_nan_safe():
  previous = 0.0
  for v in np.arange(0.0, 40.0, 0.05):
    scale = kp_low_speed_scale(float(v))
    assert 0.0 < scale <= 1.0
    assert scale >= previous - 1e-12, f"the KP cut deepened going up in speed at {v} m/s"
    previous = scale
  assert kp_low_speed_scale(float("nan")) == 1.0


def test_kp_table_is_untouched_from_10_ms_up():
  """The stock knots at 10, 15 and 30 m/s must survive exactly, so np.interp above 10 is identical."""
  scaled = scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, CN7)
  for speed, stock, new in zip(INTERP_SPEEDS, KP_INTERP, scaled, strict=True):
    if speed >= 10.0:
      assert new == stock, f"KP moved at {speed} m/s, where the measurement shows no over-gain"
    else:
      assert new < stock, f"KP did not move at {speed} m/s"


def test_kp_table_is_untouched_for_other_cars():
  assert scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, OTHER_HYUNDAI) == list(KP_INTERP)
  assert scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, HONDA.HONDA_CIVIC) == list(KP_INTERP)


def test_kp_blend_zero_is_the_identity():
  previous = sched.LOW_SPEED_KP_BLEND
  sched.LOW_SPEED_KP_BLEND = 0.0
  try:
    assert scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, CN7) == list(KP_INTERP)
    for v in (0.0, 2.0, 3.0, 5.0, 8.0, 20.0):
      assert kp_low_speed_scale(v) == 1.0
  finally:
    sched.LOW_SPEED_KP_BLEND = previous


def test_p_term_cannot_rail_on_a_routine_error():
  """The whole point of the cap: P alone may reach full scale at the worst error observed, not at
  an ordinary one. Stock fails this by up to 4.8x; the capped schedule must not."""
  scaled = scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, CN7)
  # A representative latAccelFactor. It is NOT a constant on this car -- it is learned at runtime
  # and moved 2.72-3.56 across the archive -- but the cap was derived against a value in this
  # range, and this test checks the cap's shape, not the exact factor.
  lat_accel_factor = 3.169
  for speed, err_p90 in MEASURED_ERR_P90:
    stock_kp = float(np.interp(speed, INTERP_SPEEDS, KP_INTERP))
    new_kp = float(np.interp(speed, INTERP_SPEEDS, scaled))
    stock_counts = stock_kp * err_p90 / lat_accel_factor * EPS_CEILING
    new_counts = new_kp * err_p90 / lat_accel_factor * EPS_CEILING
    assert new_counts <= 1.4 * EPS_CEILING, \
      f"at {speed} m/s the P term alone still asks {new_counts:.0f} counts at the p90 error"
    if speed <= 8.0:
      assert new_counts < stock_counts, f"the cap did not bind at {speed} m/s, where it must"


def test_controller_pid_actually_uses_the_capped_gain():
  """Behavioural, not textual: build the real controller and read the gain back off its PID.

  The guard checks the construction line by text; this checks the number the loop will use, so a
  refactor that keeps the symbol and loses the effect is caught here rather than on the road.
  """
  controller, _, _ = _make_controller()
  for speed in (3.0, 5.0, 7.5):
    controller.pid.speed = speed
    stock = float(np.interp(speed, INTERP_SPEEDS, KP_INTERP))
    assert controller.pid.k_p < stock, f"the PID still uses the stock gain {stock} at {speed} m/s"
  for speed in (10.0, 15.0, 30.0):
    controller.pid.speed = speed
    stock = float(np.interp(speed, INTERP_SPEEDS, KP_INTERP))
    assert controller.pid.k_p == stock, f"the gain moved at {speed} m/s, where the cap must not bind"

  other, _, _ = _make_controller(OTHER_HYUNDAI)
  for speed in (3.0, 5.0, 10.0):
    other.pid.speed = speed
    assert other.pid.k_p == float(np.interp(speed, INTERP_SPEEDS, KP_INTERP)),       f"a non-CN7 car had its proportional gain changed at {speed} m/s"


def test_gain_applies_to_the_measured_platform_only():
  assert ff_gain_applies(CN7)
  assert not ff_gain_applies(OTHER_HYUNDAI)
  assert not ff_gain_applies(HONDA.HONDA_CIVIC)
  # Same body and the same port, but it substitutes HYUNDAI_SONATA's torque tune, so the ratios
  # here are relative to the wrong latAccelFactor and nobody has measured its plant gain.
  assert not ff_gain_applies(HYUNDAI.HYUNDAI_ELANTRA_HEV_2024)


# ---- the controller --------------------------------------------------------


def test_only_the_feedforward_is_scheduled():
  """P must be untouched at every speed, and f must move by exactly the predicted amount."""
  curvature = 0.02
  for v in LOW_SPEEDS + HIGH_SPEEDS:
    torque_0, log_0, CP = _arm(0.0, v, curvature)
    torque_1, log_1, _ = _arm(1.0, v, curvature)

    assert log_0.error == log_1.error, f"error changed at {v} m/s -- the schedule leaked into the error path"
    assert log_0.p == log_1.p, f"P changed at {v} m/s -- the closed-loop gain must not move"
    assert log_0.i == log_1.i, f"I changed on the first frame at {v} m/s"

    gain = lat_accel_factor_gain(v)
    lat_accel_factor = CP.lateralTuning.torque.latAccelFactor
    expected = (curvature * v ** 2 / lat_accel_factor) * (1.0 / gain - 1.0)
    assert log_1.f - log_0.f == pytest.approx(expected, abs=1e-6), f"feedforward delta wrong at {v} m/s"
    if gain == 1.0:
      assert log_1.f == log_0.f, f"feedforward moved at {v} m/s where the gain is exactly 1.0"
      assert torque_1 == torque_0, f"output torque moved at {v} m/s where the gain is exactly 1.0"


def test_highway_is_bit_identical():
  """Every logged term and the output, over a settling run, at and above the learner floor."""
  for v in HIGH_SPEEDS:
    torque_0, log_0, _ = _arm(0.0, v, 0.01, frames=50)
    torque_1, log_1, _ = _arm(1.0, v, 0.01, frames=50)
    assert (torque_1, log_1.f, log_1.p, log_1.i, log_1.d, log_1.error, log_1.output) == \
           (torque_0, log_0.f, log_0.p, log_0.i, log_0.d, log_0.error, log_0.output), \
           f"the highway is not bit-identical at {v} m/s"


def test_low_speed_feedforward_actually_moved():
  """The mirror of the identity tests: a change that quietly did nothing must fail here."""
  for v in (3.0, 5.0, 8.5, 12.0):
    _, log_0, _ = _arm(0.0, v, 0.05)
    _, log_1, _ = _arm(1.0, v, 0.05)
    assert abs(log_1.f) > abs(log_0.f) * 1.2, f"the feedforward did not increase at {v} m/s"


def test_output_follows_the_feedforward_when_not_saturated():
  """With the wheel already tracking, the error is ~0 and the whole delta reaches the output.

  Worth separating from the test above: at low speed with a standing error the speed-scheduled KP
  rails the output on its own, and the feedforward change is then invisible at the output even
  though it is real. That is a property of the plant, not a failure of the change.
  """
  for v in (3.0, 5.0, 8.5, 12.0):
    # A fixed 1.0 m/s^2 demand at every speed, which the scheduled feedforward can carry without
    # clipping even at the 0.38 floor: 1.0 / (3.169 * 0.38) = 0.83 of full scale.
    curvature = 1.0 / v ** 2
    torque_0, log_0, _ = _arm(0.0, v, curvature, frames=2, tracking=True)
    torque_1, log_1, _ = _arm(1.0, v, curvature, frames=2, tracking=True)
    assert abs(torque_0) < 1.0 and abs(torque_1) < 1.0, f"expected an unsaturated command at {v} m/s"
    assert abs(torque_1) > abs(torque_0) + 1e-6, f"the extra feedforward did not reach the output at {v} m/s"
    moved = abs(torque_1) - abs(torque_0)
    expected = abs(log_1.f - log_0.f)
    assert moved == pytest.approx(expected, abs=1e-6),       f"the output moved by something other than the feedforward delta at {v} m/s"


def test_other_cars_are_untouched():
  """Same brand, same controller, same code path -- only the fingerprint gate excludes it."""
  for v in (3.0, 6.0, 10.0):
    torque_0, log_0, _ = _arm(0.0, v, 0.05, car_name=OTHER_HYUNDAI)
    torque_1, log_1, _ = _arm(1.0, v, 0.05, car_name=OTHER_HYUNDAI)
    assert torque_1 == torque_0 and log_1.f == log_0.f, f"a non-CN7 car was retuned at {v} m/s"


def test_v0_feedforward_is_untouched_when_jerk_aware_is_off():
  """The FEEDFORWARD correction is deliberately not duplicated into the v0 path. The KP cap is a
  separate change and does apply there, so this compares the feedforward term only."""
  previous = sched.LOW_SPEED_KP_BLEND
  sched.LOW_SPEED_KP_BLEND = 0.0
  try:
    for v in (3.0, 6.0, 10.0):
      torque_0, log_0, _ = _arm(0.0, v, 0.05, jerk_aware=False)
      torque_1, log_1, _ = _arm(1.0, v, 0.05, jerk_aware=False)
      assert torque_1 == torque_0 and log_1.f == log_0.f, f"the v0 feedforward moved at {v} m/s"
  finally:
    sched.LOW_SPEED_KP_BLEND = previous


def test_counts_cannot_exceed_the_eps_ceiling():
  """Asserted at the opendbc limiter, which is what actually reaches the wire."""
  class _Probe:
    flags = HyundaiFlags.RAISED_LIMITS
    carFingerprint = CN7

  limits = CarControllerParams(_Probe())
  assert limits.STEER_MAX == EPS_CEILING, "the MDPS accepts 409 counts and faults at 410"

  previous = sched.LOW_SPEED_FF_BLEND
  sched.LOW_SPEED_FF_BLEND = 1.0
  try:
    for v in (3.0, 6.7, 10.0, 14.0):
      for driver_torque in (-300.0, 0.0, 300.0):
        controller, VM, _ = _make_controller()
        last = 0
        for _ in range(200):
          torque, _ = _drive(controller, VM, v, 0.2)  # an absurd demand, deliberately
          applied = apply_driver_steer_torque_limits(int(round(torque * limits.STEER_MAX)), last,
                                                     driver_torque, limits)
          assert abs(applied) <= EPS_CEILING, f"commanded {applied} counts at {v} m/s"
          last = applied
  finally:
    sched.LOW_SPEED_FF_BLEND = previous


def test_pid_limits_stay_in_torque_space():
  controller, VM, _ = _make_controller()
  _drive(controller, VM, 6.0, 0.02, frames=3)
  assert abs(controller.pid.pos_limit) <= 1.0
  assert abs(controller.pid.neg_limit) <= 1.0


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
