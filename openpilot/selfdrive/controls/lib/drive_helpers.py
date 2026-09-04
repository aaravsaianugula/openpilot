import numpy as np
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MIN_STABLE_DELAY = 0.3

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2

# CN7: the ISO comfort limit is a lateral ACCELERATION, so the tightest radius it will command is
# v^2 / limit -- 15 m at 15 mph, 33 m at 22 mph, 81 m at 35 mph, against real turn radii of 7.5-20 m.
# Measured over 3.41M engaged frames it is active on 8% of turn frames at 11-13 mph rising to 80% at
# 31-36 mph, removes a median 11-22% of the model's demanded curvature, and accounts for 92.3% of the
# frames that raise "Turn Exceeds Steering Limit". The archive holds ZERO turn frames tight enough to
# reach it above 18 m/s, so the schedule is back on the stock value before highway speed and the
# highway is bit-identical. 4.0 is where the gain stops being realisable: the MDPS accepts 409 counts,
# which buys about 2.9 m/s^2 at 14-18 m/s -- measured 2.67 at 13-16 and 3.26 at 16-22 by
# .elantra/plant_gain.py, NOT the 3.65 an earlier note claimed. On that reproduction the EPS
# saturates before the stock 3.0 clamp binds at EVERY speed below 16 m/s, which means this
# raise mostly moves the failure from "clamped" to "saturated" down there rather than buying
# curvature. It is kept because that reproduction pools frames where the command was still
# moving, and the one band with enough genuinely settled frames measured 28% higher; the road
# test is what decides it. See FINDINGS-2026-09-03.md.
# .elantra/guards.py pins both ends of this schedule.
LAT_ACCEL_LIMIT_BP = [16.0, 22.0]                     # m/s
LAT_ACCEL_LIMIT_V = [4.0, MAX_LATERAL_ACCEL_NO_ROLL]  # m/s^2


def should_stop(v_ego: float, a_target: float) -> bool:
  return bool(v_ego < 0.3 and a_target < 0.1)

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll) -> tuple[float, bool]:
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)

  max_lat_accel_no_roll = float(np.interp(v_ego, LAT_ACCEL_LIMIT_BP, LAT_ACCEL_LIMIT_V))
  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = max_lat_accel_no_roll + roll_compensation
  min_lat_accel = -max_lat_accel_no_roll + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  return a_target

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
