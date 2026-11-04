"""
Long Assist (Elantra overlay): helps the car execute the longitudinal plan the model already shows.

Brake requests can only lower the model (e2e) candidate; lift can only raise it. The planner still
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

# model stop point
V_STOP = 0.5             # m/s, plan speed treated as stopped
STOP_BUF = 1.0           # m short of the model's own stop point
STOP_ONSET = 0.3         # m/s^2, below this coasting/drag covers it
STOP_OVERBRAKE = 1.1     # >1 so the required decel shrinks as we approach
STOP_HIST = 10           # frames of world-frame stop observations
STOP_MIN_OBS = 4         # observations in STOP_HIST needed to latch
STOP_MEDIAN_N = 5
STOP_MISS_MAX = 2        # frames the plan may drop the stop before the latch (and its hold time) resets
# Replay (2026-11-03, 8 drives): plans that flicker to a stop for a few frames, and far stops the model
# drops a second later, were most brake episodes. So nothing until the stop has held CORROBORATE_FRAMES,
# then the cap opens from STOP_CAP_START at STOP_CAP_RATE per second while it keeps holding.
CORROBORATE_FRAMES = 10
STOP_CAP_START = 1.0
STOP_CAP_RATE = 1.0      # m/s^2 per s
DILEMMA_ACCEL = 3.0      # a fresh stop needing more than this is likely a yellow light: leave it to the model
DILEMMA_FRAMES = 10

# leads
LEAD_PROB = 0.5
LEAD_S0 = 4.0
CLOSING_RATE = 0.2       # 1/s, (v_ego - v_lead) / d, i.e. TTC < 5 s
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
STOP_ONSET_PRIMED = 0.15
PRIME_V = 5.             # m/s, above this no lift toward a mapped stop/signal
A_LIFT_PRIMED = 1.0      # launch cap while still inside a mapped intersection

# turns
TURN_KAPPA = 0.02        # 1/m, R < 50 m within TURN_LOOKAHEAD_T with a blinker on
TURN_KAPPA_EXIT = 0.015
TURN_LOOKAHEAD_T = 1.0
TURN_EXIT_FRAMES = int(1.0 / DT_MDL)

# lift
LIFT_TAU = 2.5
A_LIFT_BP = [0., 5., 8., 20.]
A_LIFT_V = [1.5, 1.5, 1.2, 0.8]
J_LIFT = 1.0
LIFT_FADE_FRAMES = int(1.0 / DT_MDL)
GAP_GAIN = 0.15
T_FOLLOW_LIFT = 1.45
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
    self.cutout_block = 0

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

    a_lead = None
    ttc = np.inf
    if lead is not None:
      d, v_l, a_l = lead
      dv = v_ego - v_l
      if dv > 0.:
        ttc = d / dv
        if dv / max(d, 1.) > CLOSING_RATE or not lead_one.present:
          a_req = min(a_l, 0.) - dv ** 2 / (2 * max(d - LEAD_S0, 0.5))
          if a_req < -STOP_ONSET:
            a_lead = max(a_req, A_MIN)
    self.lead_frames = self.lead_frames + 1 if a_lead is not None else 0
    if self.lead_frames < LEAD_MIN_FRAMES:
      a_lead = None

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
    a_stop = None
    stop_spread = 0.
    self.stop_miss = 0 if d_obs is not None else self.stop_miss + 1
    if len(obs) >= STOP_MIN_OBS and self.stop_miss <= STOP_MISS_MAX:
      recent = obs[-STOP_MEDIAN_N:]
      self.stop_frames += 1
      stop_spread = float(np.std(recent))
      d = float(np.median(recent)) - self.odom
      self.stop_dist = d
      behind_lead = lead is not None and d > lead[0] - 12.
      at_hazard = primed and abs(d - self.hazard_dist) < HAZARD_MATCH
      d_eff = d - STOP_BUF - v_ego * TAU
      if v_ego < 2. and d - STOP_BUF < 2.:
        a_stop = -float(np.clip(0.8 * v_ego, 0.3, 1.0))
        a_req = -a_stop
      else:
        a_req = v_ego ** 2 / (2 * d_eff) if d_eff > 0.5 else np.inf
        if a_req > (STOP_ONSET_PRIMED if at_hazard else STOP_ONSET):
          a_stop = max(-STOP_OVERBRAKE * a_req, A_MIN)
      if a_stop is not None:
        # a stop sign, give-way or crossing has no yellow phase, so a late stop there is still braked for
        dilemma_possible = not (behind_lead or at_hazard) or self.hazard_kind == Kind.signal
        # a mapped stop, or a stopped lead the camera radar also sees, needs no waiting
        corroborated = at_hazard or (behind_lead and lead_one.present and lead_one.radar)
        held = self.stop_frames - (0 if corroborated else CORROBORATE_FRAMES)
        if (dilemma_possible and a_req > DILEMMA_ACCEL and self.stop_frames <= DILEMMA_FRAMES) or held < 0:
          a_stop = None
        else:
          a_stop = max(a_stop, -(STOP_CAP_START + max(STOP_CAP_RATE, a_req) * held * DT_MDL))
    else:
      self.stop_frames = 0
      self.stop_dist = -1.

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

    # --- brake: lowest request wins; release is rate limited, onset is not
    if self.turning:
      requests = {State.lead: a_lead} if ttc < TTC_SAFETY else {}
    else:
      requests = {State.stop: a_stop, State.lead: a_lead, State.curve: a_curve}
    requests = {k: a for k, a in requests.items() if a is not None}
    brake_state = min(requests, key=requests.get) if requests else None
    a_brake = requests[brake_state] if requests else None
    if self.a_brake is not None:
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
