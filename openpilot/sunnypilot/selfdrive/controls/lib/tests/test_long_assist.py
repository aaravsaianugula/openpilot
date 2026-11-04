from collections import deque
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist import (
  LongAssist, Mode, State, plan_stop_distance, A_MIN, A_MAP_CURVE_MAX, BRAKE_DELIVERY, CORROBORATE_FRAMES,
  DILEMMA_FRAMES, STOP_CAP_START,
)
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY, Kind, MapAhead

T = np.array(ModelConstants.T_IDXS)


def make_model(v0, stop_at=None, accel=0., kappa=None, lead=None, desired_accel=0., lane_change=False):
  if stop_at is not None:
    a = v0 ** 2 / (2 * max(stop_at, 0.1))
    ts = v0 / a if a > 0 else 0.
    tc = np.minimum(T, ts)
    v = np.maximum(v0 - a * T, 0.)
    x = v0 * tc - a * tc ** 2 / 2
  else:
    v = np.maximum(v0 + accel * T, 0.)
    x = v0 * T + accel * T ** 2 / 2
  m = log.ModelDataV2.new_message()
  m.position.x = x.tolist()
  m.position.t = T.tolist()
  m.velocity.x = v.tolist()
  k = np.zeros_like(T) if kappa is None else kappa(x)
  m.orientationRate.z = (k * v).tolist()
  if lead is not None:
    ld = m.init('leadsV3', 1)[0]
    ld.prob = 0.9
    ld.x, ld.v, ld.a = [lead[0]] * 6, [lead[1]] * 6, [lead[2]] * 6
  m.action.desiredAcceleration = desired_accel
  m.meta.laneChangeState = 'laneChangeStarting' if lane_change else 'off'
  return m


def make_lead(present=False, d=0., v=0., a=0., radar=True):
  return log.RadarState.LeadData.new_message(present=present, dRel=d, vLead=v, aLeadK=a, radar=radar)


def make_cs(v, gas=False, blinker=False):
  return SimpleNamespace(vEgo=v, gasPressed=gas, brakePressed=False, leftBlinker=blinker, rightBlinker=False)


def run(la, frames, model, lead=None, cs=None, v_cruise=30., v_limit=0., map_ahead=EMPTY):
  for _ in range(frames):
    la.update(model, lead or make_lead(), cs, True, v_cruise, v_limit, True, map_ahead)


def hazard(dist, kind):
  return MapAhead(np.zeros(0), np.zeros(0), dist, kind)


class TestPlanStopDistance:
  def test_stop_found(self):
    m = make_model(15., stop_at=60.)
    d = plan_stop_distance(T, np.array(m.position.x), np.array(m.velocity.x))
    assert d is not None and abs(d - 60.) < 2.

  def test_no_stop_when_cruising(self):
    m = make_model(15.)
    assert plan_stop_distance(T, np.array(m.position.x), np.array(m.velocity.x)) is None


class TestLongAssist:
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.full

  def test_off_and_shadow_never_change_output(self):
    for mode in (Mode.off, Mode.shadow):
      la = LongAssist()
      la.mode = mode
      run(la, 20, make_model(15., stop_at=30.), cs=make_cs(15.))
      assert la.apply(0.5, True) == (0.5, True)
    assert la.a_brake is not None  # shadow still computes

  def test_dilemma_then_fast_ramp(self):
    # stop needing > 3 m/s^2 that just appeared: not amplified while fresh, then opens fast
    m = make_model(20., stop_at=40.)
    run(self.la, 5, m, cs=make_cs(20.))
    assert self.la.a_brake is None
    run(self.la, DILEMMA_FRAMES + 8, m, cs=make_cs(20.))
    assert self.la.a_brake is not None and self.la.a_brake <= -3.0

  def test_radar_confirmed_lead_stop_needs_no_wait(self):
    m = make_model(15., stop_at=50.)
    run(self.la, 5, m, lead=make_lead(True, 55., 0., 0.), cs=make_cs(15.))
    assert self.la.a_brake is not None and self.la.a_brake <= -STOP_CAP_START

  def test_stop_waits_for_corroboration_then_ramps(self):
    m = make_model(15., stop_at=80.)
    run(self.la, 4 + CORROBORATE_FRAMES - 2, m, cs=make_cs(15.))
    assert self.la.a_brake is None  # a flicker of a few frames never brakes
    run(self.la, 2, m, cs=make_cs(15.))
    assert abs(self.la.a_brake + STOP_CAP_START / BRAKE_DELIVERY) < 0.15
    run(self.la, 20, m, cs=make_cs(15.))
    assert A_MIN < self.la.a_brake < -STOP_CAP_START / BRAKE_DELIVERY - 0.5
    assert self.la.state == State.stop

  def test_flicker_never_brakes(self):
    for _ in range(5):
      run(self.la, 6, make_model(15., stop_at=60.), cs=make_cs(15.))
      run(self.la, 6, make_model(15.), cs=make_cs(15.))
      assert self.la.a_brake is None

  def test_lead_braking_without_plan_stop(self):
    run(self.la, 3, make_model(20.), lead=make_lead(True, 40., 10., -2.), cs=make_cs(20.))
    assert self.la.state == State.lead
    assert A_MIN <= self.la.a_brake < -3.0

  def test_curve_ahead(self):
    m = make_model(20., kappa=lambda x: np.where(x > 100., 0.015, 0.))
    run(self.la, 3, m, cs=make_cs(20.))
    assert self.la.state == State.curve
    assert -1.5 / BRAKE_DELIVERY < self.la.a_brake < -0.5
    assert self.la.v_curve < 20.
    run(self.la, 3, make_model(20., kappa=lambda x: np.where(x > 100., 0.01, 0.), lane_change=True), cs=make_cs(20.))
    assert self.la.state != State.curve

  def test_turn_hands_off(self):
    m = make_model(8., stop_at=25., kappa=lambda x: np.full_like(x, 0.05))
    run(self.la, 15, m, cs=make_cs(8., blinker=True))
    assert self.la.turning and self.la.state == State.turn
    assert self.la.a_brake is None and self.la.a_lift == 0.

  def test_lift_to_speed_limit(self):
    m = make_model(10., accel=0.3)
    run(self.la, 40, m, cs=make_cs(10.), v_cruise=20., v_limit=12.)
    assert self.la.state == State.cruise
    assert abs(self.la.a_lift - 0.8) < 0.05
    a, _ = self.la.apply(0.1, False)
    assert a == self.la.a_lift
    # lift never lowers a stronger model accel
    assert self.la.apply(1.5, False)[0] == 1.5

  def test_lift_blocked(self):
    m = make_model(10., accel=0.3)
    for kwargs in ({'cs': make_cs(10., gas=True), 'model': m},
                   {'cs': make_cs(10.), 'model': make_model(10., accel=0.3, desired_accel=-0.5)},
                   {'cs': make_cs(10.), 'model': make_model(10., accel=0.3, lane_change=True)}):
      la = LongAssist()
      la.mode = Mode.full
      run(la, 40, kwargs['model'], cs=kwargs['cs'], v_cruise=20.)
      assert la.a_lift == 0.

  def test_launch_and_stopped_lead(self):
    m = make_model(0., accel=1.0)
    run(self.la, 40, m, cs=make_cs(0.), v_cruise=20.)
    assert self.la.state == State.launch and self.la.clear_stop
    assert self.la.apply(0.0, True) == (self.la.a_lift, False)

    la = LongAssist()
    la.mode = Mode.full
    run(la, 40, m, lead=make_lead(True, 6., 0., 0.), cs=make_cs(0.), v_cruise=20.)
    assert not la.clear_stop and la.a_lift < 0.2

  def test_closed_loop_stop_with_stuck_model(self):
    # The model plans a stop at X_STOP but its own action only asks for -0.3 m/s^2, and the car delivers
    # BRAKE_DELIVERY of whatever it is asked, 0.5 s late -- both measured on this car.
    X_STOP, v, pos = 90., 15., 0.
    delay = deque([0.] * int(0.5 / DT_MDL), maxlen=int(0.5 / DT_MDL))
    a_min = 0.
    for _ in range(int(30 / DT_MDL)):
      m = make_model(v, stop_at=X_STOP - pos, desired_accel=-0.3)
      self.la.update(m, make_lead(), make_cs(v), True, 30., 0., True)
      a_cmd, _ = self.la.apply(-0.3, False)
      a_act = delay[0] * (BRAKE_DELIVERY if delay[0] < 0. else 1.)
      delay.append(a_cmd)
      a_min = min(a_min, a_act)
      v = max(v + a_act * DT_MDL, 0.)
      pos += v * DT_MDL
      if v == 0.:
        break
    assert v == 0.
    assert X_STOP - 4. < pos <= X_STOP
    assert a_min >= A_MIN

  def test_zero_lift_never_cancels_model_brake(self):
    assert self.la.a_lift == 0.
    assert self.la.apply(-0.3, True) == (-0.3, True)

  def test_map_curve_brakes_gently(self):
    curve = MapAhead(np.array([150., 200.]), np.array([8., 8.]), -1., Kind.none)
    run(self.la, 3, make_model(25.), cs=make_cs(25.), map_ahead=curve)
    assert self.la.state == State.curve
    assert self.la.a_brake == -A_MAP_CURVE_MAX / BRAKE_DELIVERY
    assert self.la.v_curve < 25.

  def test_stop_sign_trusted_at_once_even_when_late(self):
    m = make_model(20., stop_at=40.)
    run(self.la, 4, m, cs=make_cs(20.), map_ahead=hazard(42., Kind.stop))
    assert self.la.a_brake is not None  # no corroboration wait, no yellow-light skip
    run(self.la, 6, m, cs=make_cs(20.), map_ahead=hazard(42., Kind.stop))
    assert self.la.a_brake < -3.0

  def test_signal_keeps_dilemma_rule(self):
    run(self.la, 5, make_model(20., stop_at=40.), cs=make_cs(20.), map_ahead=hazard(42., Kind.signal))
    assert self.la.a_brake is None

  def test_primed_blocks_lift_at_speed_caps_launch(self):
    run(self.la, 40, make_model(10., accel=0.3), cs=make_cs(10.), v_cruise=20., map_ahead=hazard(100., Kind.signal))
    assert self.la.a_lift == 0. and self.la.state == State.primed

    la = LongAssist()
    la.mode = Mode.full
    run(la, 60, make_model(0., accel=1.0), cs=make_cs(0.), v_cruise=20., map_ahead=hazard(15., Kind.signal))
    assert 0.9 < la.a_lift <= 1.0
