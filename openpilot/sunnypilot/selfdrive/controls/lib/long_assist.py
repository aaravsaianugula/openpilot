"""
Long Assist (Elantra overlay): helps the car execute the longitudinal plan the model already shows.

Brake requests can only lower the model (e2e) candidate, and a stop held below V_HOLD only adds shouldStop;
lift can only raise it. The planner still
takes min() with the lead MPC and cruise candidates, so a lift can never out-accelerate them and any
brake from them, the model or this module wins. Mode is the LongAssistMode param (0 off, 1 shadow,
2 brake, 3 full); shadow computes and logs longitudinalPlanSP.longAssist without touching the output.
Constants are first-principles starting points; tune them on shadow logs with .elantra/long_assist_scan.py.
"""
from collections import deque
from enum import IntEnum

import numpy as np

from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY as MAP_EMPTY, Kind, MapAhead

State = custom.LongitudinalPlanSP.LongAssist.State
LaneChangeState = log.LaneChangeState


class Mode(IntEnum):
  off = 0
  shadow = 1  # compute and log only
  brake = 2   # brake requests applied
  full = 3    # brake + lift applied


A_MIN = -3.5             # ACCEL_MIN, panda clamp
TAU = 0.7                # s, actuator 0.5 + model/CAN latency 0.2; the brake ramps in, so no jerk term
# The car delivers ~0.8x the braking it is commanded: 38 min of engaged logs, aEgo 0.75 s after the command,
# 0.69-0.86x across every bin from -0.5 to -3.5 (acceleration 0.85-1.0x). Plan, actuators and CAN output agree,
# so the shortfall is in the car, and Hyundai longcontrol has kiV = 0, so nothing closes it. Requests below are
# physical decelerations; they are divided by this before reaching the planner.
# shortcut: one constant, upgrade to an online estimate if it varies with load, grade or weather in later logs.
BRAKE_DELIVERY = 0.8

# model stop point
V_STOP = 0.5             # m/s, plan speed treated as stopped
STOP_BUF = 1.0           # m short of the model's own stop point
STOP_ONSET = 0.3         # m/s^2, below this coasting/drag covers it (lead following)
STOP_HIST = 10           # frames of world-frame stop observations
STOP_MIN_OBS = 4         # observations in STOP_HIST needed to latch
STOP_MEDIAN_N = 5
STOP_MISS_MAX = 2        # frames the plan may drop the stop before the latch (and its hold time) resets
# Replay (2026-11-03, 8 drives): plans that flicker to a stop for a few frames, and far stops the model
# drops a second later, were most brake episodes. So nothing until the stop has held CORROBORATE_FRAMES.
CORROBORATE_FRAMES = 10
DILEMMA_ACCEL = 3.0      # a fresh stop needing more than this is likely a yellow light: leave it to the model
DILEMMA_FRAMES = 10

# stop profile (model stop point or a stopped lead), shaped on 44 manual stops behind a stopped lead (2026-11-06):
# the driver holds speed, then brakes at a constant-equivalent 1.1 m/s^2 (-1.0 building to -1.25, peak -2) to a
# firm stop a median 5.0 m behind (3.4-6.3). Brake mode (2026-11-05) instead braked as soon as anything was
# needed, often ending 7-11 m short, then let the car creep: below ~2 m/s this car delivers ~0.6x a small decel
# and its idle creep beats -0.1..-0.6, so it rolled 7.5 -> 3.2 m and 5.8 -> 3.8 m (driver braked) behind
# stopped cars while longcontrol bounced between stopping and pid at 0.3 m/s.
STOP_GAP = 5.0           # m behind a stopped lead
A_STOP_ON_BP = [2., 8.]
A_STOP_ON_V = [0.6, 1.2]  # required decel that starts the profile: hold speed until then
A_STOP_EMERG = 2.5       # a need this big is braked at once, no onset ramp
J_STOP_ON = 1.5          # m/s^3 onset ramp
V_STOP_FIN = 2.0         # m/s, below this, near the target, the profile never asks less than A_STOP_FIN
A_STOP_FIN = 0.8
S_STOP_FIN = 3.0         # m short of the target where the firm finish and hold start; farther back the stock
                         # planner's creep closes in first (238 t=1165: a stop 11 m back made the driver gas up)
V_HOLD = 0.5             # m/s, a latched stop asserts shouldStop below this and holds it at standstill
V_LEAD_STOPPED = 0.5     # m/s, a lead predicted to end below this is a stop target
STOP_LOW_V = 3.0         # m/s, below this a stop is ended at most this hard (creeping up to a car)
A_STOP_LOW_V = 1.5

# leads
LEAD_PROB = 0.5
LEAD_S0 = 4.0
LEAD_BRAKE_T = 5.0       # s a lead keeps braking; replay of 74 takeovers: 3 s braked before the driver 15x, 5 s 22x, 8 s 27x, phantoms 2/6/14
LEAD_T = np.arange(0.25, 15.01, 0.25)
LEAD_LOW_V = 3.0         # m/s, creeping up behind a car is ended gently, not at A_MIN
LEAD_LOW_V_MAX = 1.5
TTC_SAFETY = 2.0         # s, the only brake kept during a turn
CUTOUT_BLOCK_FRAMES = int(1.5 / DT_MDL)
LEAD_MIN_FRAMES = 3      # replay: 1-2 frame lead-accel spikes became 0.2-0.5 s brake taps

# curves
A_LAT_BP = [0., 8., 15., 25., 35.]
A_LAT_V = [3.0, 2.8, 2.5, 2.4, 2.4]  # replay: 1.5-1.8 at 25-32 m/s braked for highway curves driven fine
CURVE_ONSET = 0.5
A_CURVE_MAX_BP = [20., 30.]
A_CURVE_MAX_V = [2.0, 1.0]           # highway curvature from the plan is noisy: never brake hard on it
A_CURVE_DECEL = 1.2
CURVE_V_MIN = 3.0        # plan points slower than this give unreliable curvature
CURVE_RANGE = 120.

# map: mapd curve targets cap gently (the vision path takes over within its range); hazards prime
A_MAP_CURVE_MAX = 1.0
HAZARD_MATCH = 25.       # m, a model stop this close to a mapped stop/signal is trusted at once
PRIME_V = 5.             # m/s, above this no lift toward a mapped stop/signal
A_LIFT_PRIMED = 1.0      # launch cap while still inside a mapped intersection

# turns
TURN_KAPPA = 0.02        # 1/m, R < 50 m within TURN_LOOKAHEAD_T with a blinker on
TURN_KAPPA_EXIT = 0.015
TURN_LOOKAHEAD_T = 1.0
TURN_EXIT_FRAMES = int(1.0 / DT_MDL)

# lift. 230 engaged gas presses (2026-10..11): 181 while following at a 1.63 s gap (driver alone: 1.6 s city,
# 1.1-1.4 s highway), 23 at launch (driver: 0-8 m/s at 1.3 m/s^2, up to ~1.9 in the first 2 s). Faster lift response
# and the aggressive gap covered 39 of them in shadow vs 26; following tighter is the MPC personality's job.
LIFT_TAU = 2.5
A_LIFT_BP = [0., 5., 8., 20.]
A_LIFT_V = [1.5, 1.5, 1.2, 0.8]
J_LIFT = 1.5
LIFT_FADE_FRAMES = int(0.5 / DT_MDL)
GAP_GAIN = 0.15
T_FOLLOW_LIFT = 1.25    # aggressive personality
INTENT_T = 2.0
INTENT_DECEL_DV = -0.5
E2E_BRAKING = -0.3
LAUNCH_PLAN_V = 1.0
CLEAR_STOP_ACCEL = 0.3
J_BRAKE_RELEASE = 2.0

# confidence
C_ON = 0.7
C_OFF = 0.5
C_PLAN_ALPHA = DT_MDL / 0.5
T_CMP = np.arange(0., 4.01, 0.5)


def plan_stop_distance(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> float | None:
  below = np.flatnonzero(v < V_STOP)
  if below.size:
    k = below[0]
    if k == 0:
      return 0.
    return float(np.interp(V_STOP, [v[k], v[k - 1]], [x[k], x[k - 1]]))
  # plan still tapering at the horizon: extrapolate at its end decel
  if v[-1] < 2.0:
    a_end = (v[-1] - v[-4]) / (t[-1] - t[-4])
    if a_end < -0.1:
      return float(x[-1] + v[-1] ** 2 / (2 * -a_end))
  return None


def lead_decel(d: float, v_ego: float, v_l: float, a_l: float) -> float:
  """Least constant deceleration, starting after TAU, that keeps LEAD_S0 behind the lead's predicted path.

  The lead keeps its braking for up to LEAD_BRAKE_T (stopping if it reaches zero), then holds its speed.
  Replay of 10 drives (2026-11-04): the bare relative formula a_l - dv^2/2d extrapolates every brake through
  zero speed (-3.5 asked for a stop needing -2.2), and gating it on closing rate missed a stopped car found
  at 79 m until 0.1 s before the driver braked.
  """
  if v_ego < 0.1:
    return 0.
  v_l, a_l = max(v_l, 0.), min(a_l, 0.)
  tb = min(LEAD_BRAKE_T, v_l / -a_l) if a_l < 0. else 0.
  tt = np.minimum(LEAD_T, tb)
  v_f = v_l + a_l * tb
  x = d - LEAD_S0 + v_l * tt + a_l * tt ** 2 / 2 + v_f * (LEAD_T - tt)
  overrun = v_ego * LEAD_T - x            # how far past the buffer holding our speed would take us
  u = LEAD_T - TAU
  room = x - v_ego * TAU
  with np.errstate(divide='ignore', invalid='ignore'):
    a_moving = 2 * overrun / u ** 2         # still rolling at that time
    a_stopped = np.where(room > 0., v_ego ** 2 / (2 * room), np.inf)  # stopped before it
  a = np.where(u > 0., np.where(a_moving * u <= v_ego, a_moving, a_stopped), np.inf)
  req = float(np.max(np.where(overrun > 0., a, 0.)))
  if v_f < 0.1:  # a stopped lead binds where it stopped, even past the time grid
    req = max(req, v_ego ** 2 / (2 * room[-1]) if room[-1] > 0. else np.inf)
  return min(req, LEAD_LOW_V_MAX) if v_ego < LEAD_LOW_V else min(req, 50.)


def lead_stop_point(d: float, v_l: float, a_l: float) -> float | None:
  """Where the lead comes to rest if it keeps braking for up to LEAD_BRAKE_T, None if it keeps moving."""
  v_l, a_l = max(v_l, 0.), min(a_l, 0.)
  if v_l < V_LEAD_STOPPED:
    return d
  if a_l < 0. and v_l / -a_l <= LEAD_BRAKE_T:
    return d + v_l ** 2 / (2 * -a_l)
  return None


def stop_decel(s: float, v_ego: float, a_ego: float) -> float:
  """Constant decel, from the end of the latency, that stops in s. The car keeps its current decel through TAU."""
  a0 = max(-a_ego, 0.)
  t0 = min(TAU, v_ego / a0) if a0 > 0. else TAU
  v_tau = max(v_ego - a0 * t0, 0.)
  room = s - (v_ego + v_tau) / 2 * t0
  if v_tau < 0.1:
    return max(v_ego ** 2 / (2 * s), 0.) if s > 0.1 else np.inf
  return v_tau ** 2 / (2 * room) if room > 0.1 else np.inf


class LongAssist:
  def __init__(self):
    self.mode = Mode.off
    self.reset()

  def reset(self) -> None:
    self.odom = 0.
    self.stop_hist: deque = deque(maxlen=STOP_HIST)
    self.stop_frames = 0
    self.stop_miss = 0
    self.prev_plan: tuple[np.ndarray, np.ndarray] | None = None
    self.c_plan = 1.
    self.conf_ok = False
    self.lift_weight = 0.
    self.turning = False
    self.turn_exit = 0
    self.had_lead = False
    self.lead_frames = 0
    self.lead_stop_frames = 0
    self.cutout_block = 0
    self.stop_latched = False
    self.stop_target_miss = 0
    self.hold = False

    self.state = State.off
    self.confidence = 0.
    self.c_lead = 1.
    self.stop_dist = -1.
    self.a_brake: float | None = None
    self.a_lift = 0.
    self.clear_stop = False
    self.v_curve = 0.
    self.v_target = 0.
    self.hazard_dist = -1.
    self.hazard_kind = Kind.none

  def apply(self, a_e2e: float, should_stop_e2e: bool) -> tuple[float, bool]:
    if self.mode >= Mode.full and self.a_lift > max(a_e2e, 0.):
      a_e2e = self.a_lift
      should_stop_e2e = should_stop_e2e and not self.clear_stop
    if self.mode >= Mode.brake and self.a_brake is not None:
      a_e2e = min(a_e2e, self.a_brake)
    if self.mode >= Mode.brake and self.hold:
      should_stop_e2e = True  # stopping state at crawl: this car creeps through small decels in pid
    return a_e2e, should_stop_e2e

  def update(self, model, lead_one, CS, enabled: bool, v_cruise: float, v_limit: float, healthy: bool,
             map_ahead: MapAhead = MAP_EMPTY) -> None:
    if self.mode == Mode.off or not enabled:
      self.reset()
      return

    v_ego = CS.vEgo
    self.odom += v_ego * DT_MDL
    self.hazard_dist, self.hazard_kind = map_ahead.hazard_dist, map_ahead.hazard_kind
    primed = self.hazard_dist >= 0.

    x = np.array(model.position.x)
    v = np.maximum(np.array(model.velocity.x), 0.)
    t = np.array(model.position.t) if len(model.position.t) == len(x) else np.array(ModelConstants.T_IDXS[:len(x)])
    yaw_rate = np.array(model.orientationRate.z)
    if len(x) < 4 or len(v) != len(x) or len(yaw_rate) != len(x):
      self.reset()
      self.state = State.standby
      return

    # --- leads: radard lead (camera SCC11 point or vision) and the model's own lead
    leads = []
    if lead_one.present:
      leads.append((lead_one.dRel, lead_one.vLead, lead_one.aLeadK))
    model_lead = len(model.leadsV3) > 0 and model.leadsV3[0].prob > LEAD_PROB
    if model_lead:
      ml = model.leadsV3[0]
      leads.append((ml.x[0], ml.v[0], ml.a[0]))
    lead = min(leads, key=lambda ld: ld[0]) if leads else None

    if self.had_lead and lead is None:
      self.cutout_block = CUTOUT_BLOCK_FRAMES
    self.had_lead = lead is not None
    self.cutout_block = max(self.cutout_block - 1, 0)

    ttc = np.inf
    if lead is not None:
      d, v_l, a_l = lead
      if v_ego > v_l:
        ttc = d / (v_ego - v_l)
    # every lead hypothesis (camera radar point, model lead) is a possible obstacle: the most demanding one
    # brakes. Closest-wins picked a vision lead at 58 m doing 13 m/s over a stopped radar car at 79 m.
    # A lead coming to rest is a stop target (the stop profile below); a moving one is followed with lead_decel,
    # which aims LEAD_S0 behind its predicted path (normal following needs far less, the MPC keeps a time gap).
    a_lead = None
    a_req = 0.
    lead_stops = []
    for ld in leads:
      x_l = lead_stop_point(*ld)
      if x_l is None:
        a_req = max(a_req, lead_decel(ld[0], v_ego, ld[1], ld[2]))
      else:
        lead_stops.append(x_l)
    if a_req > STOP_ONSET:
      a_lead = max(-a_req, A_MIN)
    self.lead_frames = self.lead_frames + 1 if a_lead is not None else 0
    if self.lead_frames < LEAD_MIN_FRAMES:
      a_lead = None
    self.lead_stop_frames = self.lead_stop_frames + 1 if lead_stops else 0
    if self.lead_stop_frames < LEAD_MIN_FRAMES:
      lead_stops = []

    # --- confidence
    cur = np.interp(T_CMP, t, v)
    if self.prev_plan is not None:
      prev = np.interp(T_CMP + DT_MDL, *self.prev_plan)
      rms = float(np.sqrt(np.mean((cur - prev) ** 2)))
      self.c_plan += (1. - min(rms, 1.) - self.c_plan) * C_PLAN_ALPHA
    self.prev_plan = (t, v)

    radar_lead = lead_one.present and lead_one.radar
    if radar_lead and model_lead:
      ml = model.leadsV3[0]
      dd = (lead_one.dRel - ml.x[0]) / max(5., 0.1 * lead_one.dRel)
      dvl = (lead_one.vLead - ml.v[0]) / 1.5
      self.c_lead = float(np.exp(-dd ** 2 - dvl ** 2))
    elif radar_lead or model_lead:
      self.c_lead = 0.5  # single source
    else:
      self.c_lead = 1.

    # --- model stop point, latched in world frame so a late brake cannot make it chase away
    d_obs = plan_stop_distance(t, x, v) if v_ego > V_STOP else None
    self.stop_hist.append(None if d_obs is None else self.odom + d_obs)
    obs = [s for s in self.stop_hist if s is not None]
    s_model = None
    stop_spread = 0.
    self.stop_miss = 0 if d_obs is not None else self.stop_miss + 1
    if len(obs) >= STOP_MIN_OBS and self.stop_miss <= STOP_MISS_MAX:
      recent = obs[-STOP_MEDIAN_N:]
      self.stop_frames += 1
      stop_spread = float(np.std(recent))
      d = float(np.median(recent)) - self.odom
      behind_lead = lead is not None and d > lead[0] - 12.
      at_hazard = primed and abs(d - self.hazard_dist) < HAZARD_MATCH
      a_req = stop_decel(d - STOP_BUF, v_ego, CS.aEgo)
      # a stop sign, give-way or crossing has no yellow phase, so a late stop there is still braked for
      dilemma_possible = not (behind_lead or at_hazard) or self.hazard_kind == Kind.signal
      # a mapped stop, or a stopped lead the camera radar also sees, needs no waiting
      corroborated = at_hazard or (behind_lead and lead_one.present and lead_one.radar)
      held = self.stop_frames - (0 if corroborated else CORROBORATE_FRAMES)
      if not ((dilemma_possible and a_req > DILEMMA_ACCEL and self.stop_frames <= DILEMMA_FRAMES) or held < 0):
        # behind a lead the model plans its stop 8-10 m back at a queue (2026-11-05: stopped 11 m short, the
        # driver gassed up); the lead only moves on, so STOP_GAP behind where it is now is always safe
        s_model = max(d - STOP_BUF, lead[0] - STOP_GAP) if behind_lead else d - STOP_BUF
    else:
      self.stop_frames = 0

    c_plan = self.c_plan * float(np.exp(-(stop_spread / 1.5) ** 2))
    self.confidence = c_plan ** 0.4 * self.c_lead ** 0.3 * (1. if healthy else 0.)
    self.conf_ok = self.confidence >= (C_OFF if self.conf_ok else C_ON)

    # --- curves from the plan path: v_c = sqrt(a_lat / |kappa|)
    kappa = np.abs(yaw_rate) / np.maximum(v, CURVE_V_MIN)
    valid = (v > CURVE_V_MIN) & (x < CURVE_RANGE)
    a_curve = None
    self.v_curve = float(v_cruise)
    if valid.any():
      a_lat = np.interp(v[valid], A_LAT_BP, A_LAT_V)
      v_c = np.sqrt(a_lat / np.maximum(kappa[valid], 1e-4))
      reach = np.maximum(x[valid] - v_ego * TAU, 0.)
      self.v_curve = float(min(self.v_curve, np.min(np.sqrt(v_c ** 2 + 2 * A_CURVE_DECEL * reach))))
      # at least 1 s to shed speed, so being slightly fast mid-curve is not a hard brake
      a_req = float(np.max((v_ego ** 2 - np.minimum(v_c, v_ego) ** 2) / (2 * (reach + v_ego))))
      if a_req > CURVE_ONSET:
        a_curve = -min(a_req, float(np.interp(v_ego, A_CURVE_MAX_BP, A_CURVE_MAX_V)))
    if map_ahead.curve_dist.size:
      reach = np.maximum(map_ahead.curve_dist - v_ego * TAU, 0.)
      self.v_curve = float(min(self.v_curve, np.min(np.sqrt(map_ahead.curve_v ** 2 + 2 * A_CURVE_DECEL * reach))))
      a_req = float(np.max((v_ego ** 2 - np.minimum(map_ahead.curve_v, v_ego) ** 2) / (2 * (reach + v_ego))))
      if a_req > CURVE_ONSET:
        a_curve = min(a_curve if a_curve is not None else 0., -min(a_req, A_MAP_CURVE_MAX))
    lane_change = model.meta.laneChangeState != LaneChangeState.off
    if lane_change:
      a_curve = None  # the plan's yaw rate during a lane change is the lane change, not the road

    # --- turn hand-off: once the path bends into the turn, the model owns it
    k_near = float(np.max(kappa[t <= TURN_LOOKAHEAD_T]))
    if (CS.leftBlinker or CS.rightBlinker) and k_near > TURN_KAPPA:
      self.turning = True
      self.turn_exit = TURN_EXIT_FRAMES
    elif self.turning:
      self.turn_exit = self.turn_exit - 1 if k_near < TURN_KAPPA_EXIT else TURN_EXIT_FRAMES
      self.turning = self.turn_exit > 0

    # --- stop profile toward the nearest stop target: hold speed until the required decel reaches the onset,
    # ramp in, track the constant decel that stops there, never fade into a creep, hold at standstill
    targets = [x_l - STOP_GAP for x_l in lead_stops] + ([s_model] if s_model is not None else [])
    s_stop = min(targets) if targets else None
    self.stop_dist = -1. if s_stop is None else s_stop
    self.stop_target_miss = 0 if s_stop is not None else self.stop_target_miss + 1
    a_stop = None
    if s_stop is not None:
      a_req = stop_decel(s_stop, v_ego, CS.aEgo)
      if v_ego < STOP_LOW_V:
        a_req = min(a_req, A_STOP_LOW_V)
      near = v_ego < V_STOP_FIN and s_stop < v_ego ** 2 / (2 * A_STOP_FIN) + S_STOP_FIN
      onset = a_req >= np.interp(v_ego, A_STOP_ON_BP, A_STOP_ON_V) or near
      self.stop_latched = self.stop_latched or (onset and v_ego > V_STOP)
      if self.stop_latched:
        a_stop = -max(a_req, A_STOP_FIN) if near else -a_req
    elif self.stop_target_miss > STOP_MISS_MAX:
      self.stop_latched = False
    hold = self.stop_latched and (self.hold or (v_ego < V_HOLD and s_stop is not None and s_stop < S_STOP_FIN))
    released_at_rest = self.hold and not hold
    self.hold = hold
    stop_state = State.lead if lead_stops and s_stop == min(lead_stops) - STOP_GAP else State.stop

    # --- brake: lowest request wins; onset ramps at J_STOP_ON from the current decel (2026-11-05: 1.2-1.6 m/s^2
    # steps when a lead was found), except a need of A_STOP_EMERG or more; release is rate limited
    requests = {State.lead: a_lead, State.curve: a_curve} if not self.turning else {State.lead: a_lead} if ttc < TTC_SAFETY else {}
    if a_stop is not None and (not self.turning or ttc < TTC_SAFETY):
      requests[stop_state] = min(a_stop, requests.get(stop_state) or 0.)
    requests = {k: a for k, a in requests.items() if a is not None}
    brake_state = min(requests, key=requests.get) if requests else None
    a_brake = max(requests[brake_state] / BRAKE_DELIVERY, A_MIN) if requests else None
    if a_brake is not None and -requests[brake_state] < A_STOP_EMERG:
      prev = self.a_brake if self.a_brake is not None else min(CS.aEgo, 0.) / BRAKE_DELIVERY
      a_brake = max(a_brake, prev - J_STOP_ON * DT_MDL / BRAKE_DELIVERY)
    if self.a_brake is not None and not released_at_rest:
      released = self.a_brake + J_BRAKE_RELEASE * DT_MDL
      if a_brake is None:
        a_brake = released if released < 0. else None
      else:
        a_brake = min(a_brake, released)
    self.a_brake = a_brake

    # --- lift toward min(cruise, speed limit + offset, curve speed, lead gap)
    v_t = min(v_cruise, self.v_curve)
    if v_limit > 0.:
      v_t = min(v_t, v_limit)
    if lead is not None:
      d, v_l, a_l = lead
      v_t = min(v_t, v_l + GAP_GAIN * (d - LEAD_S0 - T_FOLLOW_LIFT * v_ego)) if a_l > -0.2 else min(v_t, v_ego)
    self.v_target = float(v_t)

    v_plan = float(np.interp(INTENT_T, t, v))
    intent_decel = v_plan - v_ego < INTENT_DECEL_DV or model.action.desiredAcceleration < E2E_BRAKING
    lift_ok = (self.conf_ok and not (CS.gasPressed or CS.brakePressed) and not self.turning and
               not lane_change and not intent_decel and
               self.stop_frames == 0 and self.cutout_block == 0 and self.a_brake is None and
               (v_ego > V_STOP or v_plan > LAUNCH_PLAN_V) and not (primed and v_ego > PRIME_V))
    self.lift_weight = min(self.lift_weight + 1. / LIFT_FADE_FRAMES, 1.) if lift_ok else 0.
    a_lift_max = min(np.interp(v_ego, A_LIFT_BP, A_LIFT_V), A_LIFT_PRIMED if primed else np.inf)
    a_lift = np.clip((v_t - v_ego) / LIFT_TAU, 0., a_lift_max) * self.lift_weight
    self.a_lift = float(min(a_lift, self.a_lift + J_LIFT * DT_MDL))
    self.clear_stop = self.a_lift > CLEAR_STOP_ACCEL

    if CS.gasPressed:
      self.state = State.override
    elif brake_state is not None:
      self.state = brake_state
    elif self.turning:
      self.state = State.turn
    elif self.a_lift > 0.:
      self.state = State.launch if v_ego < 5. else State.cruise
    else:
      self.state = State.primed if primed else State.standby
