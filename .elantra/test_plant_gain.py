#!/usr/bin/env python3
"""Tests for plant_gain.py -- the measurement the feedforward schedule is built on.

Aimed at the ways this specific measurement can be wrong while looking right, because the table
it replaces was produced by an unversioned script and could not be checked at all:

  * a synthetic plant with a KNOWN gain must be recovered exactly, or the estimator is not
    measuring what its docstring says;
  * an injected actuation lag must be FOUND by the sweep, not assumed;
  * a moving command must not be counted as settled -- that distinction is the entire reason
    this tool exists, since a transient gain understates the plant and makes the schedule
    over-boost;
  * a thin band must report its real frame count, not zero.

No logs and no network. Pure functions over synthetic frames.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plant_gain as pg

FAILS: list[str] = []


def check(label, cond, detail=""):
    if cond:
        print("  ok    " + label)
    else:
        print("  FAIL  " + label + (": " + detail if detail else ""))
        FAILS.append(label)


def frame(v, norm, yaw, roll=0.0, driver=0.0, pressed=False):
    return (v, norm, yaw, roll, driver, pressed)


# ---------------------------------------------------------------- band_of
def test_band_of():
    print("band_of")
    check("lower edge is inclusive", pg.band_of(3.0) == "3-4")
    check("upper edge belongs to the next band", pg.band_of(4.0) == "4-5")
    check("below the first band is None", pg.band_of(1.9) is None)
    check("above the last band is None", pg.band_of(200.0) is None)
    check("bands are contiguous and non-overlapping",
          all(pg.BANDS[i][1] == pg.BANDS[i + 1][0] for i in range(len(pg.BANDS) - 1)))


# ---------------------------------------------------------------- settled_flags
def test_settled_flags():
    print("settled flags")
    n = pg.SETTLE_FRAMES
    steady = [frame(5.0, 0.5, 0.1)] * (n + 5)
    flags = pg.settled_flags(steady, 409.0)
    check("a stable command is not settled before the window fills", not any(flags[: n - 1]))
    check("a stable command is settled once the window fills", flags[n - 1] and flags[-1])

    # a command ramping faster than SETTLE_SPAN counts across the window is never settled
    ramp = [frame(5.0, (i * 2.0) / 409.0, 0.1) for i in range(n + 10)]
    check("a ramping command is never settled", not any(pg.settled_flags(ramp, 409.0)),
          "2 counts/frame over a 20-frame window is 40 counts of travel, well over SETTLE_SPAN")

    # The bug this pins: frames carry a NORMALISED command but SETTLE_SPAN is in counts, so
    # comparing them directly makes every frame settled and the whole split vacuous.
    check("the window is compared in counts, not normalised units",
          not any(pg.settled_flags(ramp, 409.0)) and any(pg.settled_flags(ramp, 1.0)),
          "a steer_max of 1.0 leaves the ramp inside SETTLE_SPAN -- which is the bug, shown")

    # a None (disengaged / excluded) must RESET the window, not be bridged across
    gap = [frame(5.0, 0.5, 0.1)] * n + [None] + [frame(5.0, 0.5, 0.1)] * (n - 2)
    flags = pg.settled_flags(gap, 409.0)
    check("a gap resets the settle window rather than being bridged",
          flags[n - 1] and not any(f for f in flags[n + 1:]),
          "frames after a gap must re-earn settled status")


# ---------------------------------------------------------------- pearson / pct
def test_stats():
    print("statistics")
    xs = [float(i) for i in range(100)]
    check("perfect positive correlation is 1", abs(pg.pearson(xs, xs) - 1.0) < 1e-9)
    check("perfect negative correlation is -1",
          abs(pg.pearson(xs, [-x for x in xs]) + 1.0) < 1e-9)
    check("a constant series correlates 0 rather than dividing by zero",
          pg.pearson(xs, [7.0] * 100) == 0.0)
    check("too few samples correlate 0", pg.pearson([1.0, 2.0], [1.0, 2.0]) == 0.0)
    vals = [float(i) for i in range(101)]
    check("p10/p50/p90 land where they should",
          pg.pct(vals, 0.10) == 10 and pg.pct(vals, 0.50) == 50 and pg.pct(vals, 0.90) == 90)
    check("an empty sample is nan, not 0", math.isnan(pg.pct([], 0.5)))


# ---------------------------------------------------------------- the estimator itself
def synthetic(gain, lag_frames, n=4000, v=5.0, norm=0.5, roll=0.0, sm=409.0):
    """A plant with a KNOWN gain and a KNOWN lag, in the REAL sign convention.

    controlsState carries -output_torque, so a command that produces positive yaw is stored
    negative. Building the fixture the other way round would let a sign error in the tool pass.
    """
    norms = [norm if (i // 200) % 2 == 0 else -norm for i in range(n)]   # alternate direction
    frames = []
    for i in range(n):
        src = i - lag_frames
        a_tyre = gain * norms[src] if 0 <= src < n else 0.0
        yaw = (a_tyre + roll * pg.G) / v
        frames.append(frame(v, pg.COMMAND_SIGN * norms[i], yaw, roll))
    return {"frames": frames, "steer_max": sm, "laf": 3.169}


def test_recovers_a_known_gain():
    print("recovers a known plant gain")
    acc = pg.new_acc()
    pg.accumulate(synthetic(gain=2.0, lag_frames=0), acc)
    res = pg.summarise(acc, {3.169}, {409.0}, 1, 1)
    cell = res["bands"]["5-6"]
    check("the gain is recovered exactly at zero lag",
          abs(cell["F_all"] - 2.0) < 1e-6, f"got {cell['F_all']}")
    check("the chosen lag is zero when there is none", cell["lag_ms"] == 0,
          f"got {cell['lag_ms']} ms")

    # roll must be removed, not left in: the same plant on a banked road reports the same gain
    acc = pg.new_acc()
    pg.accumulate(synthetic(gain=2.0, lag_frames=0, roll=0.05), acc)
    cell = pg.summarise(acc, {3.169}, {409.0}, 1, 1)["bands"]["5-6"]
    check("roll is removed from the measured lateral accel",
          abs(cell["F_all"] - 2.0) < 1e-6, f"got {cell['F_all']} on a 0.05 rad bank")


def test_sign_convention():
    print("sign convention")
    check("the command sign is folded in explicitly", pg.COMMAND_SIGN == -1.0,
          "controlsState carries -output_torque; a positive F must mean the car turned "
          + "the way it was asked")
    acc = pg.new_acc()
    pg.accumulate(synthetic(gain=2.0, lag_frames=0), acc)
    cell = pg.summarise(acc, {3.169}, {409.0}, 1, 1)["bands"]["5-6"]
    check("a car that steers the way it is asked reports a POSITIVE gain", cell["F_all"] > 0,
          f"got {cell['F_all']}")


def test_finds_an_injected_lag():
    print("finds an injected actuation lag")
    lag_frames = 20                      # 200 ms at 100 Hz
    acc = pg.new_acc()
    pg.accumulate(synthetic(gain=2.0, lag_frames=lag_frames), acc)
    cell = pg.summarise(acc, {3.169}, {409.0}, 1, 1)["bands"]["5-6"]
    check("the sweep finds the injected 200 ms lag", cell["lag_ms"] == 200,
          f"got {cell['lag_ms']} ms")
    check("and recovers the gain at that lag", abs(cell["F_all"] - 2.0) < 1e-6,
          f"got {cell['F_all']}")
    check("the lag is not pinned to the edge of the sweep",
          pg.LAG_SWEEP_MS[0] < cell["lag_ms"] < pg.LAG_SWEEP_MS[-1])


def test_transient_understates_the_gain():
    """The finding this tool exists to make measurable, as an executable claim."""
    print("a transient command understates the gain")
    v, sm, gain, lag = 5.0, 409.0, 2.0, 20
    frames = []
    # a command that ramps and is never held: the yaw always trails, so a/norm reads low
    for i in range(4000):
        norm = 0.9 * math.sin(i / 40.0)
        src = i - lag
        a = gain * (0.9 * math.sin(src / 40.0)) if src >= 0 else 0.0
        frames.append(frame(v, pg.COMMAND_SIGN * norm, a / v))
    acc = pg.new_acc()
    pg.accumulate({"frames": frames, "steer_max": sm, "laf": 3.169}, acc)
    cell = pg.summarise(acc, {3.169}, {409.0}, 1, 1)["bands"]["5-6"]
    at_zero_lag = pg.med(acc[("5-6", 0)]["ratio"])
    check("measuring a lagged plant at zero lag understates the gain",
          at_zero_lag < gain - 0.05, f"got {at_zero_lag:.3f} against a true {gain}")
    check("scanning the lag recovers it", abs(cell["F_all"] - gain) < 0.05,
          f"got {cell['F_all']} at {cell['lag_ms']} ms")


def test_filters():
    print("frame filters")
    v, sm = 5.0, 409.0
    # a command below MIN_COUNTS must be excluded: a/norm is noise over noise down there
    weak = pg.MIN_COUNTS / sm * 0.5
    acc = pg.new_acc()
    pg.accumulate({"frames": [frame(v, weak, 0.4)] * 3000, "steer_max": sm, "laf": 3.169}, acc)
    check("a command under MIN_COUNTS is excluded", len(acc[("5-6", 0)]["ratio"]) == 0)

    # the driver must be excluded on torque, not only on steeringPressed
    acc = pg.new_acc()
    pg.accumulate({"frames": [frame(v, 0.5, 0.4, driver=pg.MAX_DRIVER + 1.0, pressed=False)] * 3000,
                   "steer_max": sm, "laf": 3.169}, acc)
    check("driver torque above MAX_DRIVER is excluded even when not pressed",
          len(acc[("5-6", 0)]["ratio"]) == 0,
          "steeringPressed latches at 150 counts; the clamp starts cutting at 50")

    acc = pg.new_acc()
    pg.accumulate({"frames": [frame(v, 0.5, 0.4, pressed=True)] * 3000,
                   "steer_max": sm, "laf": 3.169}, acc)
    check("steeringPressed is excluded", len(acc[("5-6", 0)]["ratio"]) == 0)


def test_thin_band_reports_its_real_count():
    """Regression: the too-few branch used to print n=0 for a band that had frames."""
    print("a thin band reports its real n")
    n = pg.MIN_BAND_N // 2
    acc = pg.new_acc()
    pg.accumulate({"frames": [frame(5.0, -0.5, 0.4)] * n, "steer_max": 409.0, "laf": 3.169}, acc)
    cell = pg.summarise(acc, {3.169}, {409.0}, 1, 1)["bands"]["5-6"]
    check("a band under MIN_BAND_N reports no median", cell.get("F_all") is None)
    check("but it reports how many frames it did have, not zero",
          cell["n"] == n, f"reported n={cell['n']}, had {n}")


def test_multiple_lat_accel_factors_are_flagged():
    print("more than one latAccelFactor is not a single schedule")
    acc = pg.new_acc()
    pg.accumulate(synthetic(gain=2.0, lag_frames=0), acc)
    res = pg.summarise(acc, {3.169, 2.89}, {409.0}, 1, 1)
    check("both factors are carried into the result", len(res["lat_accel_factors"]) == 2,
          "g = F/latAccelFactor is meaningless if the denominator moved mid-set")


def main():
    for fn in (test_band_of, test_settled_flags, test_stats, test_recovers_a_known_gain,
               test_sign_convention, test_finds_an_injected_lag, test_transient_understates_the_gain, test_filters,
               test_thin_band_reports_its_real_count, test_multiple_lat_accel_factors_are_flagged):
        fn()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
