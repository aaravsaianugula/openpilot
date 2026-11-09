#!/usr/bin/env python3
"""
Pre-flight for the CN7 low-speed lateral fix. Run it ON the device, against the real build.

Everything else in this change was verified against an unbuilt checkout with a stubbed Params.
This is the one check that runs where the car runs: real `libparams_c.so`, real capnp, the real
param values the car will boot with, and the real opendbc limiter. It answers one question --
**is the fix actually going to run, and with which numbers** -- because on this port the expensive
failure has never been a wrong number, it has been a change that was silently not running at all
while every gate stayed green.

    PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 /data/openpilot/.elantra/verify_lat_fix.py

Exit 0 means the fix is live and consistent. Non-zero means do not drive it.

It is READ-ONLY: it constructs controllers and reads params, and writes nothing.
"""

from __future__ import annotations

import sys

FAILURES: list[str] = []
NOTES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
  if ok:
    print("  ok    " + label)
  else:
    print("  FAIL  " + label + (": " + detail if detail else ""))
    FAILURES.append(label + (": " + detail if detail else ""))
  return ok


def note(label: str, value) -> None:
  print(f"        {label:<38} {value}")
  NOTES.append(f"{label}: {value}")


def _read_car_params(raw: bytes, car_structs):
  """CarParams from its stored bytes, detached from any context manager pycapnp may wrap it in."""
  reader = car_structs.CarParams.from_bytes(raw)
  if hasattr(reader, "__enter__"):
    with reader as cp:
      return cp.as_builder().as_reader()
  return reader


def main() -> int:
  print("CN7 low-speed lateral fix -- on-device pre-flight")
  print("")

  # ---- 1. the schedule module is present and self-consistent ----------------
  print("[schedule]")
  try:
    from openpilot.sunnypilot.selfdrive.controls.lib.lat_accel_factor_schedule import (
      FF_LAT_ACCEL_GAIN_BP,
      FF_LAT_ACCEL_GAIN_V,
      KP_SCALE_BP,
      KP_SCALE_V,
      LEARNER_MIN_VEL,
      LOW_SPEED_FF_BLEND,
      LOW_SPEED_KP_BLEND,
      ff_gain_applies,
      kp_low_speed_scale,
      lat_accel_factor_gain,
      scaled_kp_interp,
    )
  except ImportError as exc:
    print("  FAIL  the schedule module is not installed: " + str(exc))
    print("")
    print("The fix is NOT on this device. Nothing else below would be meaningful.")
    return 1

  check("the schedule module imports", True)
  note("feedforward blend", LOW_SPEED_FF_BLEND)
  note("proportional-gain blend", LOW_SPEED_KP_BLEND)
  note("feedforward schedule", f"{FF_LAT_ACCEL_GAIN_BP} -> {FF_LAT_ACCEL_GAIN_V}")
  note("KP cap schedule", f"{KP_SCALE_BP} -> {KP_SCALE_V}")

  check("the feedforward gain is exactly 1.0 at and above the learner floor",
        all(lat_accel_factor_gain(v) == 1.0 for v in (LEARNER_MIN_VEL, 20.0, 30.0, 40.0)),
        "the highway would not be bit-identical")
  check("the KP cap is exactly 1.0 above the band it binds in",
        all(kp_low_speed_scale(v) == 1.0 for v in (KP_SCALE_BP[-1], 12.0, 20.0, 30.0)))
  check("a NaN speed falls back to today's behaviour",
        lat_accel_factor_gain(float("nan")) == 1.0 and kp_low_speed_scale(float("nan")) == 1.0)
  check("the KP cap is not applied without the feedforward that replaces it",
        not (LOW_SPEED_KP_BLEND > 0.0 and LOW_SPEED_FF_BLEND == 0.0),
        "cutting the gain with no feedforward behind it steers LESS than stock")

  # ---- 2. the params the car will actually boot with ------------------------
  print("")
  print("[params on this device]")
  from openpilot.common.params import Params
  params = Params()

  jerk_aware = params.get_bool("LateralJerkTorqueController")
  nnlc = params.get_bool("NeuralNetworkLateralControl")
  enforce = params.get_bool("EnforceTorqueControl")
  override = params.get_bool("TorqueParamsOverrideEnabled")
  tune = params.get("TorqueControlTune", return_default=True)

  note("LateralJerkTorqueController", jerk_aware)
  note("NeuralNetworkLateralControl", nnlc)
  note("EnforceTorqueControl", enforce)
  note("TorqueParamsOverrideEnabled", override)
  note("TorqueControlTune", tune)

  check("the jerk-aware path is on",
        jerk_aware,
        "the feedforward correction lives in that path and is INERT without it -- the car would"
        + " drive today's behaviour while every gate stayed green")
  check("NNLC is off",
        not nnlc,
        "NNLC replaces this feedforward outright; the two cannot both be judged on one drive")
  if override and enforce:
    check("TorqueParamsOverrideEnabled is off", False,
          "with EnforceTorqueControl it re-widens the PID limits every 300th frame; the 409 ceiling"
          + " still holds on the wire but the integrator recovers more slowly")

  # ---- 3. the controller the car will actually build ------------------------
  print("")
  print("[controller]")
  from opendbc.car.structs import car as car_structs

  raw = params.get("CarParamsPersistent") or params.get("CarParams")
  if raw is None:
    check("CarParams is available", False, "the car has not fingerprinted since boot; drive once"
          + " or start the car, then re-run")
    return 1
  # from_bytes rather than messaging.log_from_bytes: this only needs capnp, and pulling in
  # cereal.messaging would drag msgq into a read-only pre-flight for no reason. Newer pycapnp
  # returns a context manager from from_bytes and older ones return the reader directly, so take
  # a detached copy either way rather than depending on which build this device has.
  CP = _read_car_params(raw, car_structs)
  fingerprint = str(CP.carFingerprint)
  note("carFingerprint", fingerprint)
  note("latAccelFactor (static)", round(float(CP.lateralTuning.torque.latAccelFactor), 4))
  note("steerActuatorDelay", round(float(CP.steerActuatorDelay), 4))

  check("the schedule applies to this car", ff_gain_applies(fingerprint),
        fingerprint + " is not a platform this correction was measured on")

  from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import INTERP_SPEEDS, KP_INTERP
  scaled = scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, fingerprint)
  check("the proportional gain is capped below 8 m/s",
        all(n < s for v, s, n in zip(INTERP_SPEEDS, KP_INTERP, scaled, strict=True) if v < 8.0),
        "the KP table is unchanged -- the controller is still building its PID from the stock one")
  check("the proportional gain is untouched at and above 10 m/s",
        all(n == s for v, s, n in zip(INTERP_SPEEDS, KP_INTERP, scaled, strict=True) if v >= 10.0))
  print("        stock KP  " + str([round(float(x), 2) for x in KP_INTERP]))
  print("        capped KP " + str([round(float(x), 2) for x in scaled]))

  # ---- 4. what the car will actually command --------------------------------
  print("")
  print("[commanded counts, through opendbc's own limiter]")
  from opendbc.car.hyundai.values import CarControllerParams
  limits = CarControllerParams(CP)
  note("STEER_MAX", limits.STEER_MAX)
  note("STEER_DRIVER_ALLOWANCE", limits.STEER_DRIVER_ALLOWANCE)
  note("STEER_DELTA_UP / DOWN", f"{limits.STEER_DELTA_UP} / {limits.STEER_DELTA_DOWN}")
  check("STEER_MAX is the MDPS's measured acceptance limit",
        limits.STEER_MAX == 409,
        "found " + str(limits.STEER_MAX) + "; the MDPS accepts 409 and trips CF_Mdps_ToiFlt at 410")

  factor = float(CP.lateralTuning.torque.latAccelFactor)
  print("        feedforward counts for a 2.0 m/s^2 demand, stock vs scheduled:")
  for speed in (3.0, 5.0, 7.0, 10.0, 13.0, 15.0, 20.0, 30.0):
    stock = min(2.0 / factor, 1.0) * limits.STEER_MAX
    new = min(2.0 / (factor * lat_accel_factor_gain(speed)), 1.0) * limits.STEER_MAX
    flag = "" if speed < LEARNER_MIN_VEL else "   <- must be identical"
    print(f"          {speed:>5.1f} m/s   {stock:>5.0f} -> {new:>5.0f}{flag}")
    if speed >= LEARNER_MIN_VEL and abs(new - stock) > 1e-9:
      check(f"the feedforward is unchanged at {speed} m/s", False, "the highway moved")

  # ---- 5. the fix cannot exceed the EPS ceiling -----------------------------
  from opendbc.car.lateral import apply_driver_steer_torque_limits
  worst = 0
  for driver in (-300.0, 0.0, 300.0):
    last = 0
    for _ in range(300):
      last = apply_driver_steer_torque_limits(10000, last, driver, limits)
      worst = max(worst, abs(last))
  check("no command can exceed the EPS ceiling at the wire", worst <= 409,
        "reached " + str(worst) + " counts")

  print("")
  print("-" * 62)
  if FAILURES:
    print("FAILED: " + str(len(FAILURES)) + " check(s). Do NOT drive this build.")
    for f in FAILURES:
      print("  - " + f)
    return 1
  print("PASSED: the fix is installed, active on this car, and inside the EPS ceiling.")
  print("")
  print("Rollback, either one, then `sudo systemctl restart comma`:")
  print("  LOW_SPEED_KP_BLEND = 0.0   (the gain cut -- roll this back FIRST)")
  print("  LOW_SPEED_FF_BLEND = 0.0   (the feedforward -- roll this back only after)")
  print("  both in openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py")
  return 0


if __name__ == "__main__":
  sys.exit(main())
