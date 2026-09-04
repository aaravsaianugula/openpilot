"""
CN7 low-speed feedforward gain schedule.

`latAccelFactor` is the plant gain the torque controller inverts to turn a desired lateral
acceleration into a torque request. It is a single scalar, and `torqued.py` only ever fits it on
samples with `vEgo > MIN_VEL` (15 m/s) AND `|lateral_acc| <= LAT_ACC_THRESHOLD` (1 m/s^2). It has
therefore never seen a frame below 34 mph, and never a frame harder than a gentle highway curve.

Measured on this car, over the 20 archived routes that ran a 409-count build (344 segments,
1,992,979 frames), as the regression of roll-compensated yaw-rate lateral acceleration on delivered
counts, lag-aligned by 0.2 s. F is the lateral acceleration produced at a full 409-count command:

    v (m/s)   3-4   4-5   5-6   6-8   8-10  10-13  13-16  16-22  22-32
    F        1.08  1.57  1.90  1.99   2.20   2.40   2.81   3.08   3.04
    F/3.157  0.34  0.50  0.60  0.63   0.70   0.76   0.89   0.98   0.96

At 16-22 m/s the measurement returns 3.08 against the 3.157 the learner fits from exactly that
range -- agreement to 3%, from an independent estimator. Below 8 m/s the true gain is roughly half
what the controller assumes, so the feedforward asks for roughly half the torque the turn needs and
the shortfall has to be made up by the proportional term, which cannot act until the car has
already run wide.

This schedule corrects the FEEDFORWARD ONLY. The error path keeps the unscheduled latAccelFactor,
so the closed-loop P gain is unchanged at every speed and this introduces no new stability margin
question. `KP_INTERP` in latcontrol_torque_v0.py is already speed-scheduled 12.7x over this range;
the error path over-compensates and the feedforward under-compensates, and only the second is wrong.

Every value is rounded so it sits AT OR ABOVE the measured ratio -- the schedule never asks for more
torque than the measurement says the turn needs. The last breakpoint is LEARNER_MIN_VEL because that
is where the learner's own evidence begins: at and above it the gain is exactly 1.0, and since
x / 1.0 is bit-identical in IEEE-754, every speed at or above 15 m/s behaves exactly as it does
today -- same feedforward, same anti-windup bound, same integrator, same commanded counts.

The gain is capped at 1.0 and never rises above it. The measured ratio at 16-32 m/s is 0.96-0.98,
so there is nothing to gain there, and the highway is the one band of this car that already works.

.elantra/guards.py::guard_ff_lat_accel_schedule pins all of the above, including the equality with
torqued.py's MIN_VEL.
"""
import numpy as np

# This correction was measured on HYUNDAI_ELANTRA_2024 and on nothing else.
# latcontrol_torque_jerk_aware is fork-wide code that runs on every car with
# LateralJerkTorqueController on, so the schedule is gated on the fingerprint rather than applied to
# the fleet. Adding a platform here is a claim that somebody measured ITS plant gain the same way.
#
# HYUNDAI_ELANTRA_HEV_2024 is deliberately absent even though it shares this port: the ratios below
# are relative to a latAccelFactor of 3.157, which the ICE car inherits from HYUNDAI_ELANTRA_2021,
# and the hybrid substitutes HYUNDAI_SONATA instead. Same body, different tune, no measurement.
MEASURED_PLATFORMS = ("HYUNDAI_ELANTRA_2024",)

# openpilot/selfdrive/locationd/torqued.py MIN_VEL. Named here because it is the REASON the schedule
# ends where it does, not a coincidence: above this speed the learner has evidence and we defer to
# it. guards.py reads both and fails if they diverge.
LEARNER_MIN_VEL = 15.0  # m/s

FF_LAT_ACCEL_GAIN_BP = [3.0, 4.5, 6.0, 8.0, 10.0, 13.0, 15.0]  # m/s
FF_LAT_ACCEL_GAIN_V = [0.38, 0.53, 0.64, 0.69, 0.75, 0.86, 1.00]  # multiplier on latAccelFactor

# 0.0 reproduces today's behaviour arithmetically; 1.0 is the full measured schedule. This exists so
# the empty-lot stage of the road test can be driven part-way, and so the rollback is one number.
LOW_SPEED_FF_BLEND = 1.0


def lat_accel_factor_gain(v_ego: float) -> float:
  """Multiplier on latAccelFactor for the FEEDFORWARD term only. Always in (0, 1]."""
  # Negated `<` on purpose: a NaN vEgo takes this branch and returns 1.0, i.e. falls back to today's
  # behaviour, rather than falling through to np.interp and propagating NaN into the torque command.
  # It also makes "exactly 1.0 above the learner floor" a property of this function rather than of
  # np.interp's clamping.
  if not (v_ego < FF_LAT_ACCEL_GAIN_BP[-1]):
    return 1.0

  gain = float(np.interp(v_ego, FF_LAT_ACCEL_GAIN_BP, FF_LAT_ACCEL_GAIN_V))
  return 1.0 + LOW_SPEED_FF_BLEND * (gain - 1.0)


def ff_gain_applies(car_fingerprint) -> bool:
  """True only for the platform this schedule was measured on."""
  return str(car_fingerprint) in MEASURED_PLATFORMS


# ---------------------------------------------------------------------------------------------
# The proportional gain, and why it also has to move.
#
# Fixing the feedforward is necessary but not sufficient. Measured on the same 409-count archive,
# on hands-off turn frames below 13 mph the controller REQUESTS full torque on 71% of frames but
# the EPS receives >=405 counts on only 29%, and the request holds at max for a median of just
# 0.34 s -- 53% of bursts are under half a second. The 3 counts/frame rate limiter needs 1.36 s to
# ramp 0 -> 409, so only 13% of bursts last long enough to arrive. The delivered torque is a
# saw-tooth averaging 311 counts against a ceiling it demonstrably reaches (p90 = 409).
#
# The command chatters because the loop gain is far too high down there, not because the error is
# noisy. The measured tracking error is small and smooth: median 0.17 m/s^2 at 1-4 m/s, moving
# 0.023 m/s^2 per frame. But KP_INTERP rises 25x from highway to 3.5 m/s while the plant only
# weakens 2.8x, so the low-speed loop gain is ~9x the highway's and a perfectly ordinary error
# saturates the command:
#
#     v (m/s)      3.5    4.5    5.5    7.0    9.0   11.5   14.5    19+
#     KP vs hwy   25.4x  16.1x  10.3x   6.7x   4.3x   3.0x   2.2x   1.0x
#     plant gain   2.8x   1.9x   1.6x   1.5x   1.4x   1.3x   1.1x   1.0x
#
# The cap below is the largest KP for which the P term ALONE cannot exceed full scale at that
# band's measured p90 tracking error -- i.e. P may still ask for everything the actuator has, but
# only at the worst error actually observed, not at a routine one. Derived per band:
#
#     v (m/s)     2-3    3-4    4-5    5-6    6-8   8-10   10-13   13-16
#     err p90    0.316  0.331  0.537  0.582  0.634  0.548  0.346   0.379
#     KP stock    38.9   24.1   14.6   10.4    7.2    4.3    3.1     2.2
#     KP cap      10.3    9.6    5.9    4.9    5.0    5.8    9.1     7.5
#     binds?      3.8x   2.5x   2.5x   2.1x   1.4x   no     no      no
#
# It binds only below about 8 m/s. At and above KP_SCALE_BP[-1] the multiplier is exactly 1.0, so
# every speed at or above 20 mph -- the whole highway, and most of the 15-35 mph band -- keeps the
# stock gain untouched.
#
# BELOW 2.5 m/s THERE IS NO MEASUREMENT: the archive holds no hands-off turn frames under 2 m/s.
# np.interp clamps to the first value, so the correction there is a bounded 3.8x extrapolation
# rather than the 25x a constant-loop-gain target would have implied. That regime is what the
# empty-lot stage of the road test exists to cover.
#
# This is a CLOSED-LOOP change, unlike the feedforward schedule above, and an offline replay cannot
# validate it -- a gain change moves the trajectory, so replaying it against a recorded trajectory
# answers nothing. Only the road test can. It is also only safe BECAUSE the feedforward correction
# above now carries the demand: cutting the gain without it would simply remove authority, which is
# why guards.py refuses a non-zero KP blend when the feedforward blend is zero.
KP_SCALE_BP = [2.5, 3.5, 4.5, 5.5, 7.0, 9.0]  # m/s
KP_SCALE_V = [0.26, 0.40, 0.40, 0.47, 0.70, 1.00]  # multiplier on the stock KP_INTERP

# 0.0 keeps the stock gain exactly; 1.0 applies the full measured cap. Same role as
# LOW_SPEED_FF_BLEND, and the same one-number rollback.
LOW_SPEED_KP_BLEND = 1.0


def kp_low_speed_scale(v_ego: float) -> float:
  """Multiplier on the stock speed-scheduled KP. Always in (0, 1]."""
  # Same negated comparison as above: a NaN vEgo returns 1.0 and keeps the stock gain rather than
  # propagating NaN into the proportional term.
  if not (v_ego < KP_SCALE_BP[-1]):
    return 1.0

  scale = float(np.interp(v_ego, KP_SCALE_BP, KP_SCALE_V))
  return 1.0 + LOW_SPEED_KP_BLEND * (scale - 1.0)


def scaled_kp_interp(speeds, kp_interp, car_fingerprint) -> list:
  """The stock KP schedule with its low-speed rise brought down to the measured cap.

  Returned as a plain table so the gain stays a pure function of speed, computed once, with no
  per-frame work and nothing new in the 100 Hz path.
  """
  if not ff_gain_applies(car_fingerprint):
    return list(kp_interp)
  return [kp * kp_low_speed_scale(float(v)) for v, kp in zip(speeds, kp_interp, strict=True)]
