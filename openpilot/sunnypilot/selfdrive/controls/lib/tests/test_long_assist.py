from collections import deque
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist import (
  LongAssist, Mode, State, lead_decel, plan_stop_distance, stop_decel, A_MIN, A_MAP_CURVE_MAX, A_STOP_EMERG,
  A_LAUNCH_BP, A_LAUNCH_V, CORROBORATE_FRAMES, DILEMMA_FRAMES, J_STOP_ON, STOP_GAP, STOP_ONSET, TAU_ONSET,
  BRAKE_ENGAGE_BP, BRAKE_ENGAGE_V, J_ENGAGE, J_LIFT_DOWN, J_STOP_ON_MAX, LIFT_GAP_S0, T_FOLLOW_BP, T_FOLLOW_V, WET_ACCEL,
  MAP_ON_FRAMES, SIGNAL_GO_FRAMES,
)
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY, Cert, Kind, MapAhead

T = np.array(ModelConstants.T_IDXS)


def make_model(v0, stop_at=None, accel=0., kappa=None, lead=None, desired_accel=0., lane_change=False, path_y=None):
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
  if path_y is not None:
    m.position.y = path_y(x).tolist()
  if lead is not None:
    ld = m.init('leadsV3', 1)[0]
    ld.prob = 0.9
    ld.x, ld.v, ld.a = [lead[0]] * 6, [lead[1]] * 6, [lead[2]] * 6
  m.action.desiredAcceleration = desired_accel
  m.meta.laneChangeState = 'laneChangeStarting' if lane_change else 'off'
  return m


def make_lead(present=False, d=0., v=0., a=0., radar=True):
  return log.RadarState.LeadData.new_message(present=present, dRel=d, vLead=v, aLeadK=a, radar=radar)


def make_cs(v, gas=False, blinker=False, a=0., buttons=()):
  return SimpleNamespace(vEgo=v, aEgo=a, gasPressed=gas, brakePressed=False, leftBlinker=blinker, rightBlinker=False,
                         buttonEvents=[SimpleNamespace(type=b, pressed=True) for b in buttons])


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
    assert self.la.a_brake is not None and self.la.a_brake <= -A_STOP_EMERG

  def test_stop_waits_for_corroboration_then_ramps(self):
    m = make_model(15., stop_at=80.)
    run(self.la, 4 + CORROBORATE_FRAMES - 2, m, cs=make_cs(15.))
    assert self.la.a_brake is None  # a flicker of a few frames never brakes
    run(self.la, 2, m, cs=make_cs(15.))
    assert -0.3 < self.la.a_brake < 0.  # the stop profile ramps in
    run(self.la, 10, m, cs=make_cs(15.))
    assert -J_STOP_ON_MAX * 12 * DT_MDL - 1e-6 <= self.la.a_brake <= -J_STOP_ON * 12 * DT_MDL + 1e-6
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

  def test_stopped_lead_found_far_ramps_to_its_stop(self):
    # 2026-11-04 commute: stopped car first seen at 79 m from 14.6 m/s, closing rate 0.12/s
    assert 1.3 < stop_decel(79. - STOP_GAP, 14.6, 0., TAU_ONSET) < 1.9
    d = 79.
    for n in range(1, 21):
      d -= 14.6 * DT_MDL
      self.la.update(make_model(14.6), make_lead(True, d, 0., 0.), make_cs(14.6), True, 30., 0., True)
      if n == 3:
        assert self.la.state == State.lead and self.la.a_brake < 0.
    need = stop_decel(d - STOP_GAP, 14.6, 0., TAU_ONSET)  # the car never slows here: the need keeps growing
    assert -need - 1e-6 <= self.la.a_brake <= -J_STOP_ON * 18 * DT_MDL + 1e-6  # ramping in since frame 3

  def test_curve_ahead(self):
    m = make_model(20., kappa=lambda x: np.where(x > 100., 0.015, 0.))
    run(self.la, 20, m, cs=make_cs(20.))
    assert self.la.state == State.curve
    assert -1.5 <= self.la.a_brake < -0.4
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
    assert abs(self.la.a_lift - 1.25) < 0.05  # (12 - 10) / LIFT_TAU, capped at 1.25 at 10 m/s
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
    # The model plans a stop at X_STOP but its own action only asks for -0.3 m/s^2; the car is Car below
    X_STOP = 90.
    car = Car(15.)
    a_min = 0.
    for _ in range(int(30 / DT_MDL)):
      m = make_model(car.v, stop_at=X_STOP - car.x, desired_accel=-0.3)
      self.la.update(m, make_lead(), make_cs(car.v, a=car.a), True, 30., 0., True)
      a_t, ss = self.la.apply(-0.3, False)
      car.step(self.la.actuate(a_t, car.v, car.a), ss)
      a_min = min(a_min, car.a)
      if car.v == 0.:
        break
    assert car.v == 0.
    assert X_STOP - 4. < car.x <= X_STOP
    assert a_min >= -2.0

  def test_zero_lift_never_cancels_model_brake(self):
    assert self.la.a_lift == 0.
    assert self.la.apply(-0.3, True) == (-0.3, True)

  def test_map_curve_brakes_gently(self):
    curve = MapAhead(np.array([150., 200.]), np.array([8., 8.]), -1., Kind.none)
    run(self.la, 20, make_model(25.), cs=make_cs(25.), map_ahead=curve)
    assert self.la.state == State.curve
    assert self.la.a_brake == -A_MAP_CURVE_MAX
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

  def test_primed_blocks_lift_at_speed_not_launch(self):
    run(self.la, 40, make_model(10., accel=0.3), cs=make_cs(10.), v_cruise=20., map_ahead=hazard(100., Kind.signal))
    assert self.la.a_lift == 0. and self.la.state == State.primed

    # a launch is the model's own go at that signal: not capped (2026-11-07/08 road test)
    la = LongAssist()
    la.mode = Mode.full
    run(la, 60, make_model(0., accel=1.0), cs=make_cs(0.), v_cruise=20., map_ahead=hazard(15., Kind.signal))
    assert la.a_lift == A_LAUNCH_V[0]


class TestLeadDecel:
  def test_stopped_lead_is_its_stop_point(self):
    # 13 m/s, stopped car at 52 m: needed ~2.2 m/s^2; the old relative formula asked 3.5 with a noisy a_lead
    assert abs(lead_decel(52.4, 13., 0., -1.5) - 13. ** 2 / (2 * (48.4 - 13. * 0.7))) < 1e-6

  def test_slower_lead_at_constant_speed(self):
    # 20 m/s on a 10 m/s lead 60 m ahead: close 7 m during the latency, then dv^2 / 2 gap
    assert abs(lead_decel(60., 20., 10., 0.) - 100. / (2 * (56. - 7.))) < 0.1

  def test_brake_tap_is_not_a_full_stop(self):
    full_stop = 20. ** 2 / (2 * (36. + 20. ** 2 / 4. - 20. * 0.7))
    assert 0.3 < lead_decel(40., 20., 20., -2.) < 0.8 * full_stop

  def test_following_with_alead_noise_asks_nothing(self):
    assert lead_decel(30., 15., 15., -0.5) < STOP_ONSET  # below the brake onset
    assert lead_decel(30., 15., 16., 0.3) == 0.

  def test_standstill_and_creep(self):
    assert lead_decel(3., 0., 0.5, 0.) == 0.          # launch behind a close lead is never braked
    assert lead_decel(3.5, 1., 0., 0.) <= 1.5          # creeping up behind a stopped car ends gently


class Car:
  """This Elantra in closed loop: 0.5 s actuator delay, then a 0.3 s lag. Brakes as logged 2026-11-07..09 (19 routes):
  the SCC slows us with the powertrain alone (<= 0.35 m/s^2) until asked past ENGAGE_V for 0.3 s, then the friction
  brakes track 1:1 until the request is above -0.25; slower than 3 m/s they answer at once, ~0.6x below 2 m/s, where
  idle creep beats small decels; longcontrol's stopping state (v < 0.3 with a target < 0.1, or any shouldStop) ramps to
  stopAccel -2 at 1 m/s^3."""
  N_DELAY = int(0.5 / DT_MDL)
  ENGAGE_BP, ENGAGE_V = [3., 8., 12.], [0.75, 1.15, 1.45]  # a little under what LA asks: -1.55 engaged, -1.1 mostly not
  POWERTRAIN = 0.35

  def __init__(self, v):
    self.v, self.a, self.x = v, 0., 0.
    self.delay = deque([0.] * self.N_DELAY, maxlen=self.N_DELAY)
    self.stopping = False
    self.out = 0.
    self.brakes = False
    self.ask = 0.

  def step(self, a_target, should_stop):
    should_stop = should_stop or (self.v < 0.3 and a_target < 0.1)
    self.stopping = should_stop
    self.out = max(min(self.out, 0.) - 1.0 * DT_MDL, -2.0) if self.stopping else a_target
    cmd = self.delay[0]
    self.delay.append(self.out)
    self.ask = self.ask + DT_MDL if -cmd >= np.interp(self.v, self.ENGAGE_BP, self.ENGAGE_V) else 0.
    if cmd > -0.25:
      self.brakes = False
    elif self.v < 3. or self.ask >= 0.3:
      self.brakes = True
    if cmd < 0.:
      a_cmd = np.interp(self.v, [1., 3.], [0.6, 1.0]) * cmd if self.brakes else max(cmd, -self.POWERTRAIN)
    else:
      a_cmd = cmd
    creep = 0. if self.stopping and self.v == 0. else 0.4 * max(0., 1. - self.v / 2.)  # 238 t=569: crept at -2 + StopReq
    # a positive request is tracked on the total: idle creep only shows below it (238 t=752: 0.2-0.3 asked, 0.5 got)
    a_new = a_cmd + creep if cmd < 0. else max(cmd, creep)
    self.a += (a_new - self.a) * DT_MDL / 0.3
    self.v = max(self.v + self.a * DT_MDL, 0.)
    if self.v == 0.:
      self.a = max(self.a, 0.) if a_new >= 0. else 0.
    self.x += self.v * DT_MDL


def drive_to_lead(la, v0, lead_x, lead_v, lead_a, base, model_stop_short=8., seconds=40.):
  """Close the loop on a lead that brakes to a stop. base(v, gap) is the stock planner (model/MPC) without LA;
  the model plans its own stop model_stop_short behind the lead once the lead has stopped, as it did on 2026-11-05."""
  car = Car(v0)
  log = []
  for _ in range(int(seconds / DT_MDL)):
    lead_v = max(lead_v + lead_a * DT_MDL, 0.)
    lead_a = lead_a if lead_v > 0. else 0.
    lead_x += lead_v * DT_MDL
    gap = lead_x - car.x
    a_base = base(car.v, gap)
    stop_at = gap - model_stop_short if lead_v == 0. and car.v > 1. else None
    m = make_model(car.v, stop_at=max(stop_at, 0.5) if stop_at is not None else None, accel=a_base, desired_accel=a_base)
    la.update(m, make_lead(True, float(gap), float(lead_v), float(lead_a)), make_cs(car.v, a=car.a), True, 30., 0., True)
    a_t, ss = la.apply(a_base, False)
    car.step(la.actuate(min(a_t, a_base), car.v, car.a), ss)
    log.append((car.v, gap, car.a, a_t, getattr(la, 'hold', False)))
  return np.array(log)


def creeping_base(v, gap):
  # stock e2e approaching a stopped car on 2026-11-05: -0.3..-0.5 down to ~1.5 m/s, then ~-0.1 at a crawl
  return -0.08 if v < 1.5 else (-0.4 if gap < 40. else 0.)


class TestClosedLoopStops:
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.brake

  def check_stop(self, log, gap_lo=4.0, gap_hi=6.5, a_min=-2.6):
    v, gap, a = log[:, 0], log[:, 1], log[:, 2]
    stop = np.flatnonzero(v < 0.05)
    assert stop.size, "never stopped"
    k = stop[0]
    assert gap_lo <= gap[k] <= gap_hi, f"stopped {gap[k]:.1f} m behind"
    assert v[k:].max() < 0.05 and gap[k:].min() > gap[k] - 0.2, "rolled after the stop"
    assert a.min() >= a_min
    crawl = np.sum((v > 0.05) & (v < 1.5)) * DT_MDL
    assert crawl < 2.5, f"crawled {crawl:.1f} s"
    return k

  def test_creep_to_stopped_lead(self):
    # 0000023b t=86-104: 10.6 m/s, lead 33.5 m ahead at 7.4 m/s braking to a stop. Stock + old LA stopped 5.8 m
    # behind, crept at 0.3-0.6 m/s while longcontrol bounced at 0.3 m/s, and the driver braked at 3.8 m.
    log = drive_to_lead(self.la, 10.6, 33.5, 7.4, -0.9, creeping_base)
    self.check_stop(log)
    assert self.la.hold and self.la.apply(0., False)[1]

  def test_creep_after_short_stop(self):
    # 00000238 t=558-581: 5.3 m/s, lead 18 m ahead at 5.6 m/s braking to a stop: old LA let the car crawl
    # 7.5 -> 3.2 m over 14 s and only braked at 4.1 m
    self.check_stop(drive_to_lead(self.la, 5.3, 18., 5.6, -1.4, creeping_base))

  def test_queue_stop_not_short(self):
    # 00000238 t=1148-1170: 18.8 m/s, a queue at a light; the model planned its stop ~10 m behind the minivan and LA
    # followed it to 11 m, the driver gassed up. Now the stop target is STOP_GAP behind the lead.
    # a lead braking at -4: deployed LA peaked -3.3 on this plant (the brakes came in late)
    log = drive_to_lead(self.la, 18.8, 68., 16.8, -4.0, lambda v, gap: -0.3, model_stop_short=10.)
    self.check_stop(log, a_min=-2.7)

  def test_far_stopped_lead_coasts_then_one_decel(self):
    # 238 t=128: stopped queue ~110 m ahead from 15.4 m/s, stock planner not braking yet. Road test 2026-11-09 asked
    # for the decel to start sooner and smaller: it lifts off at once, brings the brakes in once, one smooth decel.
    # Deployed LA on this plant: braked from 97 m, peak -1.7, crawl 2.2 s
    log = drive_to_lead(self.la, 15.4, 130., 0., 0., lambda v, gap: 0.)
    gap, a, a_t = log[:, 1], log[:, 2], log[:, 3]
    k = self.check_stop(log)
    assert gap[np.flatnonzero(a < -0.2)[0]] > 105.  # off the gas as soon as the queue is seen
    assert np.all(np.abs(np.diff(a_t[:k])) / DT_MDL < 3.0)  # requests jerk-limited (the brakes' own bite is the car's)
    assert -1.6 <= a[:k].min() <= -0.9
    assert np.sum((log[:k, 0] > 0.05) & (log[:k, 0] < 1.5)) * DT_MDL < 2.0

  def test_late_stopped_lead_brakes_at_once(self):
    # a stopped car found at 35 m from 15 m/s needs ~5 m/s^2: never ramped, same as before
    run(self.la, 3, make_model(15.), lead=make_lead(True, 35., 0., 0.), cs=make_cs(15.))
    assert self.la.a_brake == A_MIN

  def test_hold_releases_when_lead_leaves(self):
    log = drive_to_lead(self.la, 5.3, 18., 5.6, -1.4, creeping_base, seconds=20.)
    assert self.la.hold
    gap = log[-1, 1]
    for _ in range(10):  # the lead pulls away and the plan launches
      gap += 0.1
      self.la.update(make_model(0., accel=1.0), make_lead(True, float(gap), 2.0, 1.0), make_cs(0.), True, 30., 0., True)
    assert not self.la.hold and self.la.a_brake is None
    assert self.la.apply(1.0, False) == (1.0, False)


def mpc_like(v, gap, v_l, a_l):
  """Stand-in for the lead MPC (aggressive): pulls toward its 6 m + 1.25 s comfort distance and the lead's speed. Fit to
  its logged launches behind a car pulling away (238 t=752, 22a t=770): 0.15-0.2 at first, 0.5-0.75 after 2 s."""
  desired = 6. + 1.25 * v + (v ** 2 - v_l ** 2) / (2 * 2.5)
  return float(np.clip(0.1 * (gap - desired) + 0.2 * (v_l - v) + 0.25 * max(a_l, 0.), -3.5, 2.0))


def model_launch_da(t_go):
  # the model's own launch on 2026-11 drives: ~0.3-0.9 m/s^2 over the first 2 s
  return float(np.clip(0.4 * t_go, 0., 0.8))


def launch(la, lead=None, seconds=12., v_cruise=15., green=1.0, map_ahead=EMPTY, lag_v0=0.15, a_noise=0.):
  """Closed loop from rest. lead=(gap, accel, t_move): a stopped lead that pulls away at t_move. The planner is
  min(MPC stand-in, cruise, LA-applied e2e) with shouldStop = any; the model's plan lags our speed by lag_v0.
  a_noise: the radar's aLeadK reads this for the lead's first 1 s of motion (logged: -0.1..-0.4 at walking pace)."""
  car = Car(0.)
  car.stopping = True
  a_cruise, lead_x, lead_v, lead_a = 0., 1e3, 30., 0.
  ml_v = 0.
  if lead is not None:
    lead_x, lead_a_go, t_move = lead
    lead_v = 0.
  log = []
  for k in range(int(seconds / DT_MDL)):
    t = k * DT_MDL
    if lead is not None:
      lead_a = lead_a_go if t >= t_move else 0.
      lead_v += lead_a * DT_MDL
      lead_x += lead_v * DT_MDL
      ml_v += (lead_v - ml_v) * DT_MDL / 0.2  # logged launches: the model's lead speed trails the radar's 0.1-0.2 s
    gap = lead_x - car.x
    t_go = t - green if lead is None else t - t_move - 0.3
    go = t_go > 0.
    da = model_launch_da(t_go) if go else -0.05
    v0 = max(car.v - lag_v0, 0.)
    m = make_model(v0, accel=1.0 if go else 0., desired_accel=da,
                   lead=(float(gap), float(ml_v), float(lead_a)) if lead is not None else None)
    a_read = a_noise if a_noise and lead is not None and 0. < t - t_move < 1.0 else lead_a
    radar = make_lead(lead is not None, float(gap), float(lead_v), float(a_read))
    la.update(m, radar, make_cs(car.v, a=car.a), True, v_cruise, 0., True, map_ahead)
    a_e2e, ss_e2e = la.apply(da, car.v < 0.3 and da < 0.1)
    a_mpc = mpc_like(car.v, gap, lead_v, lead_a) if lead is not None else 2.0
    a_cruise = float(np.clip(np.clip(v_cruise - car.v, -0.5, 2.0), a_cruise - 1.6 * DT_MDL, a_cruise + 1.6 * DT_MDL))
    a_t = min(a_mpc, a_cruise, a_e2e)
    ss = ss_e2e or any(car.v < 0.3 and a < 0.1 for a in (a_mpc, a_cruise))
    car.step(la.actuate(a_t, car.v, car.a), ss)
    log.append((t, car.v, car.a, a_t, a_mpc, a_e2e, gap, la.hold))
  return np.array(log)


def launch_stats(log, t0):
  t, v, a = log[:, 0], log[:, 1], log[:, 2]
  move = t[np.flatnonzero((t >= t0) & (v >= 0.3))[0]]
  t8 = t[np.flatnonzero(v >= 8.)[0]] - move
  a2 = a[(t >= move) & (t <= move + 2.)].max()
  return move - t0, t8, a2


class TestClosedLoopLaunch:
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.full

  def check_smooth(self, log):
    moving = (log[:-1, 1] > 0.05) & (log[1:, 1] > 0.05)  # at rest the plant's accel is not physical
    jerk = np.abs(np.diff(log[:, 2]))[moving] / DT_MDL
    assert jerk.max() <= 3.0, f"jerk {jerk.max():.1f}"

  def test_no_lead_green_light(self):
    # driver: 0-8 m/s in 6.2 s, first-2 s peak 1.88; before this fix lift built over ~1 s and the model paced the rest
    log = self.run_check(launch(self.la, green=1.0), 1.0)
    assert log[:, 1].max() <= 15. + 1e-6

  def test_no_lead_at_mapped_signal(self):
    # most launches are at mapped signals: lift was capped at 1.0 there and cut to 0 above 5 m/s
    self.run_check(launch(self.la, green=1.0, map_ahead=hazard(8., Kind.signal)), 1.0)

  def run_check(self, log, t0):
    delay, t8, a2 = launch_stats(log, t0)
    assert delay < 1.25, f"moved {delay:.2f} s after the plan went"  # 0.5 s delay + 0.3 s lag + jerk ramp to 0.3 m/s
    assert t8 < 6.5, f"0-8 m/s in {t8:.1f} s"
    assert 1.3 <= a2 <= 1.8, f"first-2 s peak {a2:.2f}"
    self.check_smooth(log)
    return log

  def test_lead_pulls_away_noisy_alead(self):
    # a car pulling away reads a noisy negative aLeadK at walking pace: that was a stop point a metre on, and LA held
    # us at the line until it read >= 0
    self.test_lead_pulls_away(a_noise=-0.4)

  def test_lead_pulls_away(self, a_noise=0.):
    # LA's own stop behind a lead, then the car pulls away at 1.5 m/s^2: the hold lets go as soon as it moves, LA never
    # paces the launch (the MPC does) and never asks more than it
    stop = drive_to_lead(self.la, 5.3, 18., 5.6, -1.4, creeping_base, seconds=20.)
    assert self.la.hold
    log = launch(self.la, lead=(float(stop[-1, 1]), 1.5, 1.0), a_noise=a_noise)
    t, v, a_t, a_mpc, a_e2e, gap, hold = log[:, 0], log[:, 1], log[:, 3], log[:, 4], log[:, 5], log[:, 6], log[:, 7]
    assert np.all(a_t <= a_mpc + 1e-9)
    assert v[t < 1.0].max() < 0.05 and hold[t < 1.0].all()  # held until the lead moves
    released = t[np.flatnonzero(hold == 0)[0]]
    assert released < 1.0 + 0.6  # the lead reads > 0.5 m/s after 0.35 s, then the 3-frame miss
    # from release on, LA (and the slower model under it) never paces below min(MPC, launch profile)
    paced = (t > released + 0.6) & (v < 8.)
    assert np.all(a_e2e[paced] >= np.minimum(a_mpc[paced], np.interp(v[paced], A_LAUNCH_BP, A_LAUNCH_V)) - 1e-6)
    assert gap.min() >= 4.0
    self.check_smooth(log)

  def test_lead_launch_then_stop_again(self):
    # stop-and-go: after the launch the lead brakes to a stop again; LA still stops behind it as before
    la = self.la
    log = launch(la, lead=(5., 1.5, 1.0), seconds=9.)
    car_v, gap = log[-1, 1], log[-1, 6]
    lead_v = 1.5 * 8.
    assert car_v > 5.
    TestClosedLoopStops.check_stop(None, drive_to_lead(la, car_v, gap, lead_v, -2.0, creeping_base, seconds=25.))

  def test_spurious_stop_at_launch_ignored(self):
    # the model's own speed reads < V_STOP for ~0.5 s after we pass it while its plan accelerates: not a stop
    for v in np.arange(0., 1.5, 0.05):
      self.la.update(make_model(max(v - 0.2, 0.), accel=1.0, desired_accel=0.5), make_lead(), make_cs(v), True, 15., 0., True)
      assert self.la.stop_frames == 0 and self.la.a_brake is None


class TestActuate:
  """The planner's final target -> what this car's brakes need (see long_assist A_COAST)."""
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.full
    run(self.la, 3, make_model(15.), cs=make_cs(15.))

  def test_passes_through_off_shadow_and_powertrain_range(self):
    for a in (1.0, 0., -0.2, -0.5):
      assert self.la.actuate(a, 15., 0.) == a
    la = LongAssist()
    la.mode = Mode.shadow
    run(la, 3, make_model(15.), cs=make_cs(15.))
    assert la.actuate(-1.2, 15., 0.) == -1.2

  def test_own_brake_brings_the_brakes_in_then_follows_the_need(self):
    self.la.a_brake = -0.8
    level = float(np.interp(15., BRAKE_ENGAGE_BP, BRAKE_ENGAGE_V))
    cmds = [self.la.actuate(-0.8, 15., 0.) for _ in range(20)]
    assert np.all(np.diff(cmds) >= -J_ENGAGE * DT_MDL - 1e-9)  # the extra request ramps in
    assert abs(cmds[-1] + level) < 1e-6
    for _ in range(20):  # the brakes are on
      cmd = self.la.actuate(-0.8, 15., -0.9)
    assert abs(cmd + 0.8) < 1e-6

  def test_mpc_following_stays_on_the_powertrain(self):
    # a mild MPC brake is not ours: no engaging request (closed loop that made steady following cycle)
    assert self.la.a_brake is None
    assert all(self.la.actuate(-0.8, 15., 0.) == -0.8 for _ in range(20))
    assert min(self.la.actuate(-1.2, 15., 0.) for _ in range(20)) < -1.2  # a real need is

  def test_never_less_braking_than_planned(self):
    rng = np.random.default_rng(0)
    self.la.a_brake = -1.
    for a, v, ae in zip(rng.uniform(-3.5, 1., 500), rng.uniform(0., 30., 500), rng.uniform(-3., 1., 500), strict=True):
      assert self.la.actuate(float(a), float(v), float(ae)) <= a + 1e-9

  def test_crawl_compensated(self):
    cmd = [self.la.actuate(-0.6, 1., 0.) for _ in range(10)][-1]
    assert abs(cmd + 1.) < 1e-6  # ~0.6x below 2 m/s


class TestLiftGates:
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.full

  def test_soft_gate_ramps_down_hard_gate_cuts(self):
    run(self.la, 40, make_model(10., accel=0.3), cs=make_cs(10.), v_cruise=20.)
    lift = self.la.a_lift
    assert lift > 0.8
    run(self.la, 1, make_model(10., accel=0.3, desired_accel=-0.4), cs=make_cs(10.), v_cruise=20.)  # mild intent
    assert abs(self.la.a_lift - (lift - J_LIFT_DOWN * DT_MDL)) < 1e-6
    run(self.la, 3, make_model(10., accel=0.3), cs=make_cs(10.), v_cruise=20.)
    assert self.la.a_lift > lift - J_LIFT_DOWN * DT_MDL  # back up without a 0.5 s fade
    run(self.la, 1, make_model(10., accel=0.3, desired_accel=-1.0), cs=make_cs(10.), v_cruise=20.)  # hard braking
    assert self.la.a_lift == 0.

  def test_following_mild_model_brake_does_not_hold_us_back(self):
    # a lead pulling away 30 m ahead: the model's own -0.4 keeps its longer gap; the lift catches up (MPC still caps)
    m = make_model(10., accel=0.3, desired_accel=-0.4)
    run(self.la, 40, m, lead=make_lead(True, 30., 12., 0.5), cs=make_cs(10.), v_cruise=20.)
    assert self.la.a_lift > 0.8
    la = LongAssist()
    la.mode = Mode.full
    run(la, 40, m, cs=make_cs(10.), v_cruise=20.)  # no lead: mild intent still holds it
    assert la.a_lift == 0.

  def test_lift_gap_outside_the_mpc(self):
    # at the MPC's own gap (6 m + t_follow * v) the lift asks nothing: it never pushes in further than the MPC
    v = 13.4
    gap = 6. + float(np.interp(v, T_FOLLOW_BP, T_FOLLOW_V)) * v
    run(self.la, 40, make_model(v), lead=make_lead(True, gap, v, 0.), cs=make_cs(v), v_cruise=30.)
    assert self.la.a_lift == 0. and self.la.v_target < v
    run(self.la, 40, make_model(v), lead=make_lead(True, gap + (LIFT_GAP_S0 - 6.) + 6., v, 0.), cs=make_cs(v), v_cruise=30.)
    assert self.la.a_lift > 0.3


class TestWet:
  def test_mpc_gap(self):
    la = LongAssist()
    P = log.LongitudinalPersonality
    assert la.mpc_t_follow(10., P.aggressive) is None  # mode off: stock
    la.mode = Mode.full
    t = la.mpc_t_follow(10., P.aggressive)
    assert t == la.mpc_t_follow(10., P.standard) and 0.85 <= t < 1.0  # wet keeps aggressive's distance
    assert la.mpc_t_follow(25., P.aggressive) == 1.25
    assert la.mpc_t_follow(10., P.relaxed) is None

  def test_wet_launch_gentler(self):
    dry, wet = LongAssist(), LongAssist()
    for la in (dry, wet):
      la.mode = Mode.full
    wet.wet = True
    for la in (dry, wet):
      run(la, 60, make_model(0., accel=1.0), cs=make_cs(0.), v_cruise=20.)
    assert dry.a_lift == A_LAUNCH_V[0] and abs(wet.a_lift - A_LAUNCH_V[0] * WET_ACCEL) < 1e-6

  def test_wet_brakes_earlier(self):
    # a stopped car found far enough that dry still only coasts: wet brakes from farther and no harder
    out = {}
    for w in (False, True):
      la = LongAssist()
      la.mode = Mode.brake
      la.wet = w
      log_ = drive_to_lead(la, 13., 150., 0., 0., lambda v, gap: 0.)
      a = log_[:, 2]
      out[w] = (log_[np.flatnonzero(a < -0.4)[0], 1], a.min())
      TestClosedLoopStops.check_stop(None, log_)
    assert out[True][0] >= out[False][0] and out[True][1] >= out[False][1] - 0.05


class TestClosedLoopFollow:
  def test_lead_braking_to_a_stop_from_speed(self):
    # 248 t=1096: lead 78.7 m ahead at 19 m/s braking at ~-0.8 to a stop, from 20.7. Deployed LA asked -0.4 building to
    # -1.7 over 5 s, got only the powertrain, then the brakes came in and peaked at -3.9
    la = LongAssist()
    la.mode = Mode.brake
    log_ = drive_to_lead(la, 20.7, 78.7, 19.0, -0.8, lambda v, gap: -0.3, seconds=45.)
    TestClosedLoopStops.check_stop(None, log_, a_min=-2.4)


def mapped(d, kind=Kind.stop, cert=Cert.sure, hid=7, y=0.):
  return MapAhead(np.zeros(0), np.zeros(0), float(d), kind, cert, hid, y)


def drive_to_line(la, v0, line, kind, cert, model, seconds=45., gas_at=None, resume_at=None, blinker=False):
  """Close the loop toward a mapped stop line `line` m ahead on our approach. model(v, d_line) -> (plan, shouldStop) is
  the stock model; the planner is min(model, LA) with shouldStop = any. gas_at=(t0, t1): the driver's gas (override
  resets LA); resume_at: a resume press."""
  car = Car(v0)
  log = []
  for k in range(int(seconds / DT_MDL)):
    tt = k * DT_MDL
    d_line = line - car.x
    m, ss = model(car.v, d_line)
    gas = gas_at is not None and gas_at[0] <= tt < gas_at[1]
    buttons = ('accelCruise',) if resume_at is not None and abs(tt - resume_at) < DT_MDL / 2 else ()
    la.update(m, make_lead(), make_cs(car.v, gas=gas, a=car.a, blinker=blinker, buttons=buttons), not gas, 30., 0., True,
              mapped(d_line, kind, cert) if d_line > -4. else EMPTY)
    a_t, ss = la.apply(m.action.desiredAcceleration, ss)
    if gas:
      car.step(1.0, False)
    else:
      car.step(la.actuate(a_t, car.v, car.a), ss)
    log.append((car.v, d_line, car.a, a_t, la.hold, la.a_brake if la.a_brake is not None else 0.))
  return np.array(log)


def ignores_sign(v, d):
  # the model drives through the stop sign; at rest it wants to go
  return (make_model(v, accel=1.0, desired_accel=0.5), False) if v < 0.5 else (make_model(v), False)


def red_light(v, d):
  # 249 t=469 / 24a t=582: the model brakes gently from ~190 m (its plan slowing ~1.5 m/s^2, its action -0.6), shows the
  # stop itself only from ~110 m, and its own decel then comes late (0.7x what the stop needs)
  if d > 190.:
    return make_model(v), False
  if d > 110.:
    return make_model(v, accel=-1.5, desired_accel=-0.6), False
  need = v ** 2 / (2 * max(d - 1., 1.))
  return make_model(v, stop_at=max(d - 1., 0.5), desired_accel=-float(min(0.7 * need, 3.5))), v < 0.5


class TestIntersections:
  def setup_method(self):
    self.la = LongAssist()
    self.la.mode = Mode.full

  def check_line_stop(self, log, a_min=-2.2):
    v, d, a = log[:, 0], log[:, 1], log[:, 2]
    stop = np.flatnonzero(v < 0.05)
    assert stop.size, "ran it"
    k = stop[0]
    assert 0. <= d[k] <= 3.5, f"stopped {d[k]:.1f} m before the line"
    assert a[:k].min() >= a_min, f"peak {a[:k].min():.2f}"
    assert np.sum((v[:k] > 0.05) & (v[:k] < 1.5)) * DT_MDL < 2.5
    return k

  def test_stop_sign_full_stop_hold_until_resume(self):
    # a mapped stop sign the model never sees, from 35 mph: one decel started early enough for the brakes, a full stop
    # at the line, held while the model wants to go, away on resume and not stopped again
    log = drive_to_line(self.la, 15.6, 220., Kind.stop, Cert.sure, ignores_sign, seconds=45., resume_at=38.)
    k = self.check_line_stop(log)
    v, d, a = log[:, 0], log[:, 1], log[:, 2]
    assert d[np.flatnonzero(a < -0.5)[0]] > 80.
    r = int(38. / DT_MDL)
    assert v[k:r].max() < 0.05 and log[k + 5:r, 4].all()
    assert v[-1] > 1.

  def test_stop_sign_gas_goes_without_a_second_stop(self):
    log = drive_to_line(self.la, 15.6, 220., Kind.stop, Cert.sure, ignores_sign, seconds=40., gas_at=(30., 30.6))
    self.check_line_stop(log)
    v = log[:, 0]
    assert v[int(31. / DT_MDL):].min() > 0.3 and v[-1] > 2.

  def test_stop_sign_wet_starts_earlier(self):
    onset = {}
    for wet in (False, True):
      la = LongAssist()
      la.mode = Mode.full
      la.wet = wet
      log = drive_to_line(la, 15.6, 220., Kind.stop, Cert.sure, ignores_sign, seconds=40.)
      self.check_line_stop(log, a_min=-2.0 if wet else -2.2)
      onset[wet] = log[np.flatnonzero(log[:, 2] < -0.4)[0], 1]
    assert onset[True] >= onset[False]

  def test_red_light_committed_early_one_smooth_decel(self):
    log = drive_to_line(self.la, 19., 260., Kind.signal, Cert.sure, red_light, seconds=40.)
    k = self.check_line_stop(log, a_min=-2.0)
    v, d, a = log[:, 0], log[:, 1], log[:, 2]
    assert d[np.flatnonzero(log[:, 5] < -0.3)[0]] > 170.  # committed when the model first slows for it
    assert d[np.flatnonzero(a < -0.6)[0]] > 120.          # the brakes are on long before the model's own stop
    assert v[k:].max() < 0.05                              # held at the red (the plan does not go)
    # the model alone (LA shadow): powertrain only until its own late braking, then much harder or through the line
    base = LongAssist()
    base.mode = Mode.shadow
    b = drive_to_line(base, 19., 260., Kind.signal, Cert.sure, red_light, seconds=40.)
    kb = np.flatnonzero(b[:, 0] < 0.05)
    assert not kb.size or b[kb[0], 1] < 0. or b[:, 2].min() < a.min() - 0.8

  def test_late_yellow_left_to_the_model(self):
    # the model first shows a stop at the signal 44 m out at 18 m/s (~6.7 m/s^2 needed): LA neither slams nor adds later
    m = make_model(18., stop_at=44., desired_accel=-2.)
    for _ in range(40):
      self.la.update(m, make_lead(), make_cs(18.), True, 30., 0., True, mapped(45., Kind.signal))
      assert self.la.a_brake is None
    assert self.la.map_go and not self.la.map_commit
    assert self.la.apply(-2., False)[0] == -2.  # the model's own braking still goes through

  def test_green_light_never_brakes(self):
    log = drive_to_line(self.la, 17., 200., Kind.signal, Cert.sure, lambda v, d: (make_model(v), False), seconds=14.)
    assert (log[:, 5] == 0.).all() and log[:, 0].min() > 16.

  def test_light_turns_green_releases(self):
    slowing = make_model(18., accel=-1.5, desired_accel=-0.6)
    run(self.la, 30, slowing, cs=make_cs(18.), map_ahead=mapped(150., Kind.signal))
    assert self.la.map_commit and self.la.a_brake is not None and self.la.a_brake < 0.
    run(self.la, SIGNAL_GO_FRAMES + 25, make_model(18., accel=0.3), cs=make_cs(18.), map_ahead=mapped(140., Kind.signal))
    assert not self.la.map_commit and self.la.a_brake is None

  def test_moving_car_ahead_slowing_is_not_a_red_light(self):
    # the model slows for a car ahead before the signal (22a/23a/249: lights that changed as the queue moved): the
    # MPC follows the car, LA does not stop at the line
    slowing = make_model(18., accel=-1.5, desired_accel=-0.6)
    run(self.la, 40, slowing, lead=make_lead(True, 50., 12., -0.5), cs=make_cs(18.), map_ahead=mapped(150., Kind.signal))
    assert not self.la.map_commit

  def test_side_street_or_unsure_sign_needs_the_model(self):
    run(self.la, 40, make_model(15.), cs=make_cs(15.), map_ahead=mapped(60., cert=Cert.ambiguous))
    assert self.la.a_brake is None and not self.la.map_commit
    run(self.la, 5, make_model(15., stop_at=59.), cs=make_cs(15.), map_ahead=mapped(60., cert=Cert.ambiguous))
    assert self.la.a_brake is not None  # the model stopping at that line is trusted at once

  def test_path_turns_off_before_the_line(self):
    turn = make_model(15., path_y=lambda x: np.where(x > 50., -(x - 50.) ** 2 / 15., 0.))
    run(self.la, 40, turn, cs=make_cs(15., blinker=True), map_ahead=mapped(120.))
    assert not self.la.map_commit and self.la.a_brake is None

  def test_committed_stop_sign_braked_through_the_turn(self):
    # stopping at the sign before turning right there: the path bends in the first second, the blinker is on
    curving = make_model(6., kappa=lambda x: np.full_like(x, 0.03))
    run(self.la, MAP_ON_FRAMES + 10, curving, cs=make_cs(6., blinker=True), map_ahead=mapped(25.))
    assert self.la.turning and self.la.map_commit and self.la.a_brake is not None and self.la.a_brake < -0.3

  def test_model_stop_held_at_rest(self):
    # a stop only the model shows (no lead, no map): at rest its shouldStop drops and it asks a creep while its plan
    # still stops; LA holds until the plan goes (it used to let go 0.2 s after stopping, 249 t=587)
    la = LongAssist()
    la.mode = Mode.brake
    car = Car(10.)
    for k in range(int(30. / DT_MDL)):
      d = 40. - car.x
      going = k * DT_MDL > 25.
      if going:
        m, da = make_model(car.v, accel=1.0), 1.0
      elif car.v > 0.5:
        m, da = make_model(car.v, stop_at=max(d, 0.5)), -0.3
      else:
        m, da = make_model(0., accel=0.05), 0.15
      la.update(m, make_lead(), make_cs(car.v, a=car.a), True, 30., 0., True)
      a_t, ss = la.apply(da, False)
      car.step(la.actuate(a_t, car.v, car.a), ss)
      if 15. < k * DT_MDL < 25.:
        assert car.v < 0.05 and la.hold
    assert car.v > 0.5 and not la.hold
