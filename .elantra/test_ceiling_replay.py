#!/usr/bin/env python3
"""Tests for ceiling_replay.py. Plain main()-style harness, matching the .elantra convention:
run it directly and read the exit code. pytest collects nothing from this.

    PYTHONPATH=<opendbc> python .elantra/test_ceiling_replay.py

The load-bearing claim in this tool is MAX_ALLOWANCE_NO_REFLASH: that opendbc's driver-torque
window can be widened from 50 to 101 counts and still sit strictly inside the window panda
already enforces, so the change needs no reflash. That is an assertion about two pieces of code
in different languages, and the arithmetic is done here against BOTH of them -- opendbc's real
apply_driver_steer_torque_limits, and the constants read out of panda's own C header -- rather
than against a remembered formula. If either side moves, this fails.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ceiling_replay as cr

FAIL = []
OPENDBC = os.environ.get("OPENDBC_PATH", r"D:/Coding/opendbc-elantra-wt-cn7")


def check(label, cond, detail=""):
  if cond:
    print("  ok    " + label)
  else:
    print("  FAIL  " + label + ((" -- " + detail) if detail else ""))
    FAIL.append(label)


def opendbc_ceiling(steer_max, allowance, driver, sign=1):
  """The steady-state |applied| opendbc permits at this driver torque, via its OWN limiter.

  Run to convergence because apply_driver_steer_torque_limits applies the driver clamp and THEN
  a rate limit against the last applied value; a single call measures the rate limit.
  """
  from opendbc.car.lateral import apply_driver_steer_torque_limits

  limits = cr.Limits(steer_max, 3, 7, allowance)
  last = 0
  want = sign * 100000
  for _ in range(4000):
    last = apply_driver_steer_torque_limits(want, last, driver, limits)
  return last


def panda_constants():
  """(max_torque_raised, driver_torque_allowance, driver_torque_multiplier) from hyundai.h.

  Returns None when the header cannot be read, rather than raising. The caller already has a
  check whose failure message says to set OPENDBC_PATH, and that message was unreachable: open()
  raised FileNotFoundError first, so a mis-set path took the whole run down with a traceback
  instead of failing one named check. Same rule as guards.py -- a check may report that it could
  not verify something; it may not abort the run it is part of.
  """
  path = os.path.join(OPENDBC, "opendbc/safety/modes/hyundai.h")
  try:
    with open(path, encoding="utf-8") as fh:
      src = fh.read()
  except OSError:
    return None
  allow = re.search(r"\.driver_torque_allowance\s*=\s*(\d+)", src)
  mult = re.search(r"\.driver_torque_multiplier\s*=\s*(\d+)", src)
  raised = re.search(r"HYUNDAI_STEERING_LIMITS_RAISED\s*=\s*HYUNDAI_LIMITS\((\d+)", src)
  if not (allow and mult and raised):
    return None
  return int(raised.group(1)), int(allow.group(1)), int(mult.group(1))


def panda_ceiling(driver, sign=1):
  """panda's own driver-clamped window, from its C constants. Mirrors opendbc/safety/lateral.h."""
  consts = panda_constants()
  if consts is None:
    return None
  max_torque, allowance, mult = consts
  if sign >= 0:
    return max(min(max_torque, max_torque + (allowance + driver) * mult), 0)
  return -max(min(max_torque, max_torque + (allowance - driver) * mult), 0)


def test_panda_constants_are_readable():
  print("panda constants")
  consts = panda_constants()
  check("hyundai.h yields (max_torque, allowance, multiplier)", consts is not None,
        "cannot read opendbc/safety/modes/hyundai.h -- set OPENDBC_PATH")
  if consts:
    check("panda's raised ceiling is 512, above opendbc's 409", consts[0] == cr.PANDA_CEILING,
          f"header says {consts[0]}, tool says {cr.PANDA_CEILING}")
    check("panda's driver allowance is what the tool assumes",
          consts[1] == cr.PANDA_DRIVER_ALLOWANCE, f"header {consts[1]}")
    check("panda's driver multiplier is what the tool assumes",
          consts[2] == cr.PANDA_DRIVER_MULTIPLIER, f"header {consts[2]}")


def test_max_allowance_stays_inside_pandas_window():
  """The claim the whole no-reflash argument rests on, checked at every driver torque."""
  print("driver-window headroom")
  if panda_constants() is None:
    check("panda constants available", False)
    return

  drivers = list(range(-1024, 1025, 8))
  worst = None
  for d in drivers:
    for sign in (1, -1):
      mine = opendbc_ceiling(409, cr.MAX_ALLOWANCE_NO_REFLASH, d, sign)
      theirs = panda_ceiling(d, sign)
      if abs(mine) > abs(theirs):
        worst = (d, sign, mine, theirs)
        break
  check(f"allowance {cr.MAX_ALLOWANCE_NO_REFLASH} never commands more than panda permits",
        worst is None, f"driver={worst[0]} sign={worst[1]}: opendbc {worst[2]} vs panda {worst[3]}"
        if worst else "")

  # ...and one count higher must actually break it, or the bound is not a bound.
  broke = False
  for d in drivers:
    for sign in (1, -1):
      if abs(opendbc_ceiling(409, cr.MAX_ALLOWANCE_NO_REFLASH + 1, d, sign)) > abs(panda_ceiling(d, sign)):
        broke = True
        break
  check(f"allowance {cr.MAX_ALLOWANCE_NO_REFLASH + 1} does exceed it, so the bound is tight",
        broke, "no driver torque distinguished them -- the bound may be arbitrary")


def test_todays_allowance_is_the_cars_allowance():
  print("grid baseline")
  from opendbc.car.hyundai.values import CarControllerParams, HyundaiFlags

  class _Probe:
    carFingerprint = "HYUNDAI_ELANTRA_2024"
    flags = int(HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.RAISED_LIMITS)

  real = CarControllerParams(_Probe())
  check("grid row 0 is today's ceiling", cr.GRID[0][0] == real.STEER_MAX,
        f"grid {cr.GRID[0][0]} vs car {real.STEER_MAX}")
  check("grid row 0 is today's rate", cr.GRID[0][1:3] == (real.STEER_DELTA_UP, real.STEER_DELTA_DOWN))
  check("grid row 0 is today's driver allowance",
        cr.GRID[0][3] == real.STEER_DRIVER_ALLOWANCE,
        f"grid {cr.GRID[0][3]} vs car {real.STEER_DRIVER_ALLOWANCE}")
  check("Limits defaults to today's allowance", cr.Limits(409, 3, 7).STEER_DRIVER_ALLOWANCE == 50)


def test_driver_clamp_actually_binds_and_a_wider_window_relieves_it():
  print("clamp behaviour")
  # 100 counts of opposing column torque against a 50-count allowance costs 2*(100-50) = 100.
  at_50 = opendbc_ceiling(409, 50, -100)
  at_100 = opendbc_ceiling(409, 100, -100)
  check("50 counts of allowance clamps a 100-count opposing torque to 309",
        at_50 == 309, str(at_50))
  check("100 counts of allowance leaves the full 409", at_100 == 409, str(at_100))
  check("with the wheel quiet the clamp does nothing either way",
        opendbc_ceiling(409, 50, 0) == 409 == opendbc_ceiling(409, 100, 0))
  # And a torque that AIDS the command is never clamped, at any allowance.
  check("torque in the commanded direction is never clamped",
        opendbc_ceiling(409, 50, 200) == 409)


def test_replay_is_monotonic_in_allowance():
  print("replay monotonicity")
  # Opposing driver torque throughout, so the clamp is what limits delivery.
  trace = [(6.0, 1.0, -150.0)] * 2000
  means = [cr.replay(trace, 409, 3, 7, a)["mean_applied"] for a in (50, 75, 100)]
  check("a wider driver window delivers more, never less",
        means[0] < means[1] < means[2], str(means))
  # With no driver torque at all the allowance must change nothing.
  quiet = [(6.0, 1.0, 0.0)] * 2000
  flat = [cr.replay(quiet, 409, 3, 7, a)["mean_applied"] for a in (50, 75, 100)]
  check("with a quiet wheel the allowance is inert", len(set(flat)) == 1, str(flat))


def test_per_band_accounting_is_exhaustive_and_exclusive():
  print("per-band accounting")
  trace = [(v, 1.0, 0.0) for v in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)] * 100
  r = cr.replay(trace, 409, 3, 7, 50)
  counts = {b: n for b, (n, _) in r["per_band"].items()}
  check("every band that should have frames has them",
        all(n == 100 for n in counts.values()), str(counts))
  check("frames are not double-counted across bands",
        sum(counts.values()) == len(trace), f"{sum(counts.values())} vs {len(trace)}")
  # A speed outside every band must be dropped, not folded into the nearest one.
  r = cr.replay([(30.0, 1.0, 0.0)] * 100 + trace, 409, 3, 7, 50)
  check("a speed above the last band is not folded into it",
        sum(n for n, _ in r["per_band"].values()) == len(trace), "highway frames leaked into a band")


def test_trace_round_trip():
  print("trace file")
  import tempfile

  trace = [(6.25, 0.5, -128.0), (12.5, -0.25, 64.0), (0.0, 0.0, 0.0)]
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "t.bin")
    cr.dump_trace(trace, path)
    back = cr.load_trace(path)
    check("round-trips the frame count", len(back) == len(trace), f"{len(back)} vs {len(trace)}")
    # Every value here is exactly representable in float32, so this must be exact.
    check("round-trips the values exactly", all(
      all(abs(a - b) < 1e-6 for a, b in zip(x, y, strict=True))
      for x, y in zip(trace, back, strict=True)), str(back))

    with open(path, "r+b") as fh:
      fh.write(b"XXXXXXXX")
    try:
      cr.load_trace(path)
      check("a file with the wrong magic is rejected", False, "it was accepted")
    except SystemExit:
      check("a file with the wrong magic is rejected", True)


def main():
  for fn in (test_panda_constants_are_readable,
             test_max_allowance_stays_inside_pandas_window,
             test_todays_allowance_is_the_cars_allowance,
             test_driver_clamp_actually_binds_and_a_wider_window_relieves_it,
             test_replay_is_monotonic_in_allowance,
             test_per_band_accounting_is_exhaustive_and_exclusive,
             test_trace_round_trip):
    fn()
  print()
  if FAIL:
    print(f"FAILED {len(FAIL)}: {FAIL}")
    return 1
  print("all checks passed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
