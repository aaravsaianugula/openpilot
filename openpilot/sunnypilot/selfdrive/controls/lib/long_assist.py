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
TAU = 0.7                # s, actuator 0.5 + model/CAN latency 0.2, brakes already on
TAU_ONSET = 1.1          # s from coasting: brakes in ~1.1 s after a -0.3..-1 request, 0.4-0.6 s after -2.4..-3.5 (18 onsets
                         # 2026-11-08/09); the engaging request (BRAKE_ENGAGE_V) sits between

# Brake actuation (19 routes 2026-11-07..09). The SCC only uses the friction brakes once asked for more than
# BRAKE_ENGAGE_V; below that the powertrain alone slows us, however long it is asked. Steady requests at 9-20 m/s:
# -0.3 -> -0.24..-0.33, -0.5 -> -0.30..-0.41, -0.8 -> -0.42..-0.57, -1.1 -> -0.49..-0.81, -1.55 -> -1.55. Once on, the
# brakes track 1:1 (-0.54 -> -0.58, -0.84 -> -0.90, -1.25 -> -1.35) until released. 248 t=1101: a lead braking to a
# stop, LA asked -0.4 building to -1.7 over 5 s and got -0.35..-0.5, then the brakes came in and peaked at -3.9; that
# is the late-hard-then-crawl stop. Requests below (and the planner's target, see LongAssist.actuate) are physical
# decelerations: a need past the powertrain asks firmly enough to bring the brakes in, then follows the need.
A_COAST = 0.3            # m/s^2 the powertrain delivers without the brakes
BRAKE_ENGAGE_BP = [3., 8., 12.]
BRAKE_ENGAGE_V = [0.8, 1.2, 1.5]
A_ENGAGE = 0.6           # need of our own brake (stop, lead, curve) that brings the brakes in
A_ENGAGE_ANY = 1.0       # any planner need: below it the MPC's following stays on the powertrain. Closed loop on the
                         # measured plant, bringing them in at 0.6 for the MPC turned steady following into a +-1.4 cycle
ENGAGE_SEEN = 0.6        # m/s^2 of decel past the powertrain: the brakes are on
ENGAGE_WAIT = 1.0        # s at the engage request without them coming in before asking a little more
J_ENGAGE = 3.0           # m/s^3 ramp of the extra request; the car's own response lags ~1 s
J_ENGAGE_OFF = 2.0
CRAWL_DELIVERY_BP = [1., 3.]
CRAWL_DELIVERY_V = [0.6, 1.0]  # below 2 m/s ~0.6x and idle creep (2026-11-06)

# wet roads: the "standard" distance setting. Same following distance as aggressive; launches and lift gentler,
# brakes brought in earlier and smaller, stops planned with more latency margin
WET_ACCEL = 0.8
WET_JERK = 0.7
A_ENGAGE_WET = 0.45
WET_TAU_MARGIN = 0.3

# following distance (MPC t_follow, aggressive and wet). Road test 2026-11-09: too far back below 40 mph; engaged
# gaps matched the MPC's 1.25 s + 6 m (20-30 mph 20 m, 30-40 mph 25.6 m). Stock 1.25 s from 45 mph up.
T_FOLLOW_BP = [0., 9., 13.4, 17.9, 20.]
T_FOLLOW_V = [0.85, 0.9, 1.0, 1.15, 1.25]

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
# required decel that starts the profile. Road test 2026-11-09: holding speed until 1.2 then braking (made harder by
# the brakes coming in late, see A_COAST) felt late and hard; now it coasts from A_STOP_COAST and brakes from here
A_STOP_ON_V = [0.5, 0.7]
A_STOP_ON_V_WET = [0.4, 0.5]
A_STOP_COAST = 0.2       # need that lifts off: the powertrain's A_COAST, no brakes, no latch
A_STOP_EMERG = 2.5       # a need this big is braked at once, no onset ramp
J_STOP_ON = 1.5          # m/s^3 onset ramp
STOP_RAMP_T = 0.8        # s, a bigger need found late ramps in this fast, up to J_STOP_ON_MAX: at J_STOP_ON a stopped car
J_STOP_ON_MAX = 2.5      # found at 78 m from 14.6 m/s (1.7 needed) escalated to -3.1 on the measured plant
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
LEAD_JUMP = 15.          # m, a lead stop this much nearer than the recent median (a cut-in) is taken at once

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
# Road test 2026-11-09: in motion it sped up in bursts and slowly. Replay of 5 drives: the lift dropped to 0 at once
# 21x/min when a gate flickered (plan confidence 19% of frames, model intent 12%, mapped signal 6%, cut-out 5%) and
# came back over 0.5 s; a 2.5 s approach was slow near the target. Now soft gates ramp it down, hard ones cut it.
LIFT_TAU = 1.6
A_LIFT_BP = [0., 5., 8., 20., 30.]
A_LIFT_V = [1.5, 1.5, 1.3, 1.0, 0.7]
J_LIFT = 2.0
J_LIFT_DOWN = 2.0
GAP_GAIN = 0.15
# the lift only catches up a gap the MPC would also close: aimed 1 m outside the MPC's 6 m + t_follow * v. Aimed 2 m
# inside it (LEAD_S0), it spent every bit of accel the MPC allowed and steady following cycled +-0.9 on the plant
LIFT_GAP_S0 = 7.
# Following a lead the MPC paces, the model's mild braking is mostly it keeping its own ~1.6-1.9 s gap (engaged gaps
# matched it), which held us back whenever the lead pulled away: there only its hard braking cuts the lift
FOLLOW_RANGE = 60.
FOLLOW_CLOSING = 1.0     # m/s, closing faster than this is not steady following
INTENT_T = 2.0
INTENT_DECEL_DV = -0.5
E2E_BRAKING = -0.3
INTENT_HARD_DV = -1.5    # the plan slows this much by INTENT_T, or the model brakes this hard: lift cut at once
E2E_BRAKING_HARD = -0.8
LAUNCH_PLAN_V = 1.0
CLEAR_STOP_ACCEL = 0.3
J_BRAKE_RELEASE = 2.0

# launch: from rest until V_LAUNCH_END. Road test 2026-11-07/08: launches too slow with and without a lead. Replay of
# 114 recorded launches: the lift took ~1 s to build and was then capped at 1.0 at mapped signals, cut above 5 m/s with
# a signal within 150 m, blocked for 1.5 s by every flicker of a weak model lead, and behind a lead the gap law asked
# ~0.3; the model's own speed reads < V_STOP for ~0.5 s after we pass it, which looked like a stop at 0 m (blocked lift,
# sometimes latched a -1.9 brake). Driver: 0-8 m/s in 6.2 s, mean 1.28, first-2 s peak 1.88.
A_LAUNCH_BP = [0., 5., 8.]
A_LAUNCH_V = [1.6, 1.6, 1.2]  # meets A_LIFT_V at V_LAUNCH_END
J_LAUNCH = 2.5                # m/s^3, also the ramp-down of a launch's lift for soft reasons; the car's SCC allows 3 in pid
V_LAUNCH_END = 8.
LAUNCH_GO_T = 5.              # s, a plan back above LAUNCH_PLAN_V by then is a launch, not a stop
LAUNCH_STOP_NEAR = 1.         # m, only a stop this close can be the model's lagging speed estimate

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


def lead_decel(d: float, v_ego: float, v_l: float, a_l: float, tau: float = TAU) -> float:
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
  u = LEAD_T - tau
  room = x - v_ego * tau
  with np.errstate(divide='ignore', invalid='ignore'):
    a_moving = 2 * overrun / u ** 2         # still rolling at that time
    a_stopped = np.where(room > 0., v_ego ** 2 / (2 * room), np.inf)  # stopped before it
  a = np.where(u > 0., np.where(a_moving * u <= v_ego, a_moving, a_stopped), np.inf)
  req = float(np.max(np.where(overrun > 0., a, 0.)))
  if v_f < 0.1:  # a stopped lead binds where it stopped, even past the time grid
    req = max(req, v_ego ** 2 / (2 * room[-1]) if room[-1] > 0. else np.inf)
  return min(req, LEAD_LOW_V_MAX) if v_ego < LEAD_LOW_V else min(req, 50.)


def lead_stop_point(d: float, v_l: float, a_l: float, at_rest: bool = False) -> float | None:
  """Where the lead comes to rest if it keeps braking for up to LEAD_BRAKE_T, None if it keeps moving.

  At rest only a lead that is stopped now counts: a car pulling away reads a noisy -0.1..-0.4 a_l at walking pace,
  which extrapolated to a stop a metre or two on and held us at the line.
  """
  v_l, a_l = max(v_l, 0.), min(a_l, 0.)
  if v_l < V_LEAD_STOPPED:
    return d
  if a_l < 0. and v_l / -a_l <= LEAD_BRAKE_T and not at_rest:
    return d + v_l ** 2 / (2 * -a_l)
  return None


def stop_decel(s: float, v_ego: float, a_ego: float, tau: float = TAU) -> float:
  """Constant decel, from the end of the latency, that stops in s. The car keeps its current decel through tau."""
  a0 = max(-a_ego, 0.)
  t0 = min(tau, v_ego / a0) if a0 > 0. else tau
  v_tau = max(v_ego - a0 * t0, 0.)
  room = s - (v_ego + v_tau) / 2 * t0
  if v_tau < 0.1:
    return max(v_ego ** 2 / (2 * s), 0.) if s > 0.1 else np.inf
  return v_tau ** 2 / (2 * room) if room > 0.1 else np.inf


class LongAssist:
  def __init__(self):
    self.mode = Mode.off
    self.wet = False  # the planner sets it from the distance setting each frame
    self.reset()

  def reset(self) -> None:
    self.odom = 0.
    self.stop_hist: deque = deque(maxlen=STOP_HIST)
    self.stop_frames = 0
    self.stop_miss = 0
    self.prev_plan: tuple[np.ndarray, np.ndarray] | None = None
    self.c_plan = 1.
    self.conf_ok = False
    self.turning = False
    self.turn_exit = 0
    self.had_lead = False
    self.lead_frames = 0
    self.lead_stop_frames = 0
    self.cutout_block = 0
    self.stop_latched = False
    self.stop_target_miss = 0
    self.hold = False
    self.launching = False
    self.lead_stop_hist: deque = deque(maxlen=STOP_MEDIAN_N)
    self.brakes_on = False
    self.engage_t = 0.
    self.a_extra = 0.

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

  def mpc_t_follow(self, v_ego: float, personality) -> float | None:
    """The lead MPC's time gap for aggressive and wet (standard); None keeps the stock personality gap."""
    if self.mode < Mode.full or personality == log.LongitudinalPersonality.relaxed:
      return None
    return float(np.interp(v_ego, T_FOLLOW_BP, T_FOLLOW_V))

  def actuate(self, a_target: float, v_ego: float, a_ego: float) -> float:
    """The command that makes this car deliver the planner's final target (see A_COAST). Never less braking than
    planned: a need past the powertrain adds an engaging request until the brakes are on, then follows the need."""
    if self.mode < Mode.brake or self.state == State.off:
      return a_target
    need = -a_target
    extra = 0.
    if need <= A_COAST - 0.1:
      self.brakes_on = False
      self.engage_t = 0.
    elif v_ego < BRAKE_ENGAGE_BP[0]:
      self.brakes_on = True  # slow, the brakes answer at once (2-5 m/s: -1.12 -> -1.06); below 2 m/s ~0.6x
      extra = need - need / float(np.interp(v_ego, CRAWL_DELIVERY_BP, CRAWL_DELIVERY_V))
    elif not self.brakes_on and (need > A_ENGAGE_ANY or (self.a_brake is not None and a_target >= self.a_brake - 1e-3 and
                                                          need > (A_ENGAGE_WET if self.wet else A_ENGAGE))):
      self.engage_t += DT_MDL
      level = float(np.interp(v_ego, BRAKE_ENGAGE_BP, BRAKE_ENGAGE_V)) + min(0.4 * max(self.engage_t - ENGAGE_WAIT, 0.), 0.5)
      extra = min(need - level, 0.)
    self.brakes_on = self.brakes_on or (need > A_COAST and a_ego < -ENGAGE_SEEN)
    if extra < self.a_extra:
      self.a_extra = max(extra, self.a_extra - J_ENGAGE * DT_MDL)
    else:
      self.a_extra = min(extra, self.a_extra + J_ENGAGE_OFF * DT_MDL) if need > A_COAST - 0.1 else 0.
    return float(max(a_target + self.a_extra, A_MIN))

  def update(self, model, lead_one, CS, enabled: bool, v_cruise: float, v_limit: float, healthy: bool,
             map_ahead: MapAhead = MAP_EMPTY) -> None:
    if self.mode == Mode.off or not enabled:
      self.reset()
      return

    v_ego = CS.vEgo
    self.odom += v_ego * DT_MDL
    # from rest until V_LAUNCH_END; a brake while rolling makes it ordinary driving again
    self.launching = ((self.launching or v_ego < V_STOP) and v_ego < V_LAUNCH_END and
                      not (self.a_brake is not None and v_ego > V_STOP))
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
    tau = (TAU if self.brakes_on or CS.aEgo < -ENGAGE_SEEN else TAU_ONSET) + (WET_TAU_MARGIN if self.wet else 0.)
    # every lead hypothesis (camera radar point, model lead) is a possible obstacle: the most demanding one
    # brakes. Closest-wins picked a vision lead at 58 m doing 13 m/s over a stopped radar car at 79 m.
    # A lead coming to rest is a stop target (the stop profile below); a moving one is followed with lead_decel,
    # which aims LEAD_S0 behind its predicted path (normal following needs far less, the MPC keeps a time gap).
    a_lead = None
    a_req = 0.
    lead_stops = []
    for ld in leads:
      x_l = lead_stop_point(*ld, at_rest=v_ego < 0.1)
      if x_l is None:
        a_req = max(a_req, lead_decel(ld[0], v_ego, ld[1], ld[2], tau))
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
    # world-frame median: a vision lead's range jumps 5-10 m frame to frame (24d t=655: 59 -> 49 m asked -3.4)
    self.lead_stop_hist.append(self.odom + min(lead_stops) if lead_stops else None)
    recent = [s for s in self.lead_stop_hist if s is not None]
    if lead_stops and len(recent) >= 3:
      x_med = float(np.median(recent)) - self.odom
      lead_stops = [min(lead_stops) if min(lead_stops) < x_med - LEAD_JUMP else x_med]

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
    if (self.launching and d_obs is not None and d_obs < LAUNCH_STOP_NEAR and
        np.interp(LAUNCH_GO_T, t, v) > LAUNCH_PLAN_V):
      d_obs = None  # the plan's own speed still reads < V_STOP just after we pass it, while the plan accelerates
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

    # --- stop profile toward the nearest stop target: lift off once a little is needed, brake from the onset (ramp
    # in, track the constant decel that stops there), never fade into a creep, hold at standstill
    targets = [x_l - STOP_GAP for x_l in lead_stops] + ([s_model] if s_model is not None else [])
    s_stop = min(targets) if targets else None
    self.stop_dist = -1. if s_stop is None else s_stop
    self.stop_target_miss = 0 if s_stop is not None else self.stop_target_miss + 1
    a_stop = None
    if s_stop is not None:
      a_req = stop_decel(s_stop, v_ego, CS.aEgo, tau)
      if v_ego < STOP_LOW_V:
        a_req = min(a_req, A_STOP_LOW_V)
      near = v_ego < V_STOP_FIN and s_stop < v_ego ** 2 / (2 * A_STOP_FIN) + S_STOP_FIN
      onset = a_req >= np.interp(v_ego, A_STOP_ON_BP, A_STOP_ON_V_WET if self.wet else A_STOP_ON_V) or near
      self.stop_latched = self.stop_latched or (onset and v_ego > V_STOP)
      if self.stop_latched:
        a_stop = -max(a_req, A_STOP_FIN) if near else -a_req
      elif a_req >= A_STOP_COAST and v_ego > V_STOP:
        a_stop = -min(a_req, A_COAST)
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
    a_brake = max(requests[brake_state], A_MIN) if requests else None
    if a_brake is not None and -requests[brake_state] < A_STOP_EMERG:
      prev = self.a_brake if self.a_brake is not None else min(CS.aEgo, 0.)
      j_on = float(np.clip(-a_brake / STOP_RAMP_T, J_STOP_ON, J_STOP_ON_MAX)) * (WET_JERK if self.wet else 1.)
      a_brake = max(a_brake, prev - j_on * DT_MDL)
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
    # launching behind leads that all pull away, the lead MPC (min() in the planner) paces us: the gap law follows the
    # lead's current speed and asked ~0.3 m/s^2 for the first 2 s of 80 recorded launches behind a lead
    pulling_away = self.launching and all(v_l > max(v_ego, V_LEAD_STOPPED) for _, v_l, _ in leads)
    if lead is not None and not pulling_away:
      d, v_l, a_l = lead
      gap = LIFT_GAP_S0 + float(np.interp(v_ego, T_FOLLOW_BP, T_FOLLOW_V)) * v_ego
      v_t = min(v_t, v_l + GAP_GAIN * (d - gap)) if a_l > -0.2 else min(v_t, v_ego)
    self.v_target = float(v_t)

    v_plan = float(np.interp(INTENT_T, t, v))
    da = model.action.desiredAcceleration
    intent_decel = v_plan - v_ego < INTENT_DECEL_DV or da < E2E_BRAKING
    intent_hard = v_plan - v_ego < INTENT_HARD_DV or da < E2E_BRAKING_HARD
    # hard gates cut the lift at once: the driver, a brake or stop of ours, a turn, the model braking hard, rest
    hard_ok = (not (CS.gasPressed or CS.brakePressed) and not self.turning and not intent_hard and
               self.stop_frames == 0 and self.a_brake is None and (v_ego > V_STOP or v_plan > LAUNCH_PLAN_V))
    # soft gates ramp it down: plan confidence, a lane change, mild model intent, a cut-out, a mapped stop/signal at
    # speed (a launch is the model's own go at that signal: no mapped-signal cap or block until V_LAUNCH_END)
    following = lead is not None and lead[0] < FOLLOW_RANGE and lead[1] > v_ego - FOLLOW_CLOSING
    soft_ok = (self.conf_ok and not lane_change and not (intent_decel and not following) and
               (self.cutout_block == 0 or self.launching) and not (primed and v_ego > PRIME_V and not self.launching))
    if self.launching:
      a_lift_max, j_up, j_down = float(np.interp(v_ego, A_LAUNCH_BP, A_LAUNCH_V)), J_LAUNCH, J_LAUNCH
    else:
      a_lift_max, j_up, j_down = min(np.interp(v_ego, A_LIFT_BP, A_LIFT_V), A_LIFT_PRIMED if primed else np.inf), J_LIFT, J_LIFT_DOWN
    if self.wet:
      a_lift_max, j_up = a_lift_max * WET_ACCEL, j_up * WET_JERK
    target = float(np.clip((v_t - v_ego) / LIFT_TAU, 0., a_lift_max)) if soft_ok else 0.
    a_lift = float(np.clip(target, self.a_lift - j_down * DT_MDL, self.a_lift + j_up * DT_MDL)) if hard_ok else 0.
    self.a_lift = a_lift
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
