#!/usr/bin/env python3
"""Would a higher ceiling actually reach the car, or does the slew rate eat it first?

Replays the recorded normalised torque trace -- carControl.actuators.torque, exactly what the
controller asked for on the drive -- back through opendbc's OWN rate limiter under a grid of
(STEER_MAX, STEER_DELTA_UP) and reports what would have reached the EPS.

apply_driver_steer_torque_limits is imported, not reimplemented, so this cannot drift from the
code the car runs. .elantra/torque_projection.py already prices a ceiling change and is left
alone; what it cannot do is vary the RATE, which is the half of the limiter this is about.

WHAT THIS IS AND IS NOT. It is open loop: the recorded actuators.torque is held fixed while the
ceiling moves underneath it. In the real closed loop a car that finally got the torque it asked
for would back off, so a replay OVERSTATES the benefit of a raise. That asymmetry is the point.
If even the generous open-loop replay says a raise buys little, the closed-loop truth is smaller
still -- a negative result here is sound, and a positive one is only an upper bound.

panda enforces HYUNDAI_LIMITS(512, 3, 7): it caps the ceiling at 512 AND the rate at 3/7. Any
rate above 3 in this grid is therefore a PANDA REFLASH, not just an opendbc edit, and the table
labels it so.
"""

from __future__ import annotations

import glob
import json
import sys

PANDA_CEILING = 512
PANDA_RATE_UP = 3
PANDA_RATE_DOWN = 7

# opendbc/safety/modes/hyundai.h: HYUNDAI_LIMITS sets .max_rt_delta = 112, described in
# declarations.h as "max change in torque per 250ms interval". At 100 Hz that is 25 frames, so
# panda's REAL sustained slew ceiling is 112/25 = 4.48 counts/frame no matter what max_rate_up
# says. A build that raises rate_up to 10 without also raising max_rt_delta does not get rate 10;
# it gets a safety fault. Any row whose 250 ms travel exceeds this needs BOTH numbers moved.
PANDA_MAX_RT_DELTA = 112
RT_INTERVAL_FRAMES = 25

# The band the shortfall actually lives in, per the flat-409 road data.
LOW_SPEED = (3.0, 14.0)

# (steer_max, rate_up, rate_down). Row 0 is what the car runs today.
GRID = [
    (409, 3, 7),     # today
    (409, 4, 7),     # the most rate today's max_rt_delta=112 can actually sustain
    (409, 10, 10),   # carrotpilot's rate; needs max_rt_delta raised too
    (450, 3, 7),     # ceiling only -- and the EPS has already refused this, see below
    (450, 10, 10),
    (500, 3, 7),
    (500, 10, 10),
]


class Limits:
    def __init__(self, steer_max: int, rate_up: int, rate_down: int):
        self.STEER_MAX = steer_max
        self.STEER_DELTA_UP = rate_up
        self.STEER_DELTA_DOWN = rate_down
        self.STEER_DRIVER_ALLOWANCE = 50
        self.STEER_DRIVER_MULTIPLIER = 2
        self.STEER_DRIVER_FACTOR = 1


def assert_limiter_matches() -> None:
    """The grid's non-rate constants are a copy of CarControllerParams, and copies drift."""
    from opendbc.car.hyundai.values import CarControllerParams, HyundaiFlags

    class _Probe:
        # RAISED_LIMITS must be here or CarControllerParams falls through to the 384 HKG default
        # and the STEER_MAX check below compares the grid against the wrong number. This is the
        # CarParams.flags bit (2**27), NOT the safetyParam bit (1024) -- they are different flags
        # with the same name and reading the wrong one fails silently in both directions.
        carFingerprint = "HYUNDAI_ELANTRA_2024"
        flags = int(HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.RAISED_LIMITS)

    real = CarControllerParams(_Probe())
    mine = Limits(real.STEER_MAX, PANDA_RATE_UP, PANDA_RATE_DOWN)
    bad = {n: (getattr(mine, n), getattr(real, n))
           for n in ("STEER_DRIVER_ALLOWANCE", "STEER_DRIVER_MULTIPLIER", "STEER_DRIVER_FACTOR",
                     "STEER_DELTA_UP", "STEER_DELTA_DOWN")
           if getattr(mine, n) != getattr(real, n)}
    if bad:
        raise SystemExit(f"CarControllerParams drifted from this tool's copy: {bad!r}")
    if GRID[0][0] != real.STEER_MAX:
        raise SystemExit(f"grid row 0 is {GRID[0]} but the car runs STEER_MAX={real.STEER_MAX}")


def collect(segs: list) -> list:
    """(vEgo, normalised torque, driver column torque) for every engaged frame."""
    from openpilot.tools.lib.logreader import LogReader

    trace = []
    for seg in segs:
        v = 0.0
        driver = 0.0
        try:
            for ev in LogReader(seg, sort_by_time=True):
                w = ev.which()
                if w == "carState":
                    v = float(ev.carState.vEgo)
                    driver = float(ev.carState.steeringTorque)
                elif w == "carControl":
                    cc = ev.carControl
                    if cc.latActive:
                        trace.append((v, float(cc.actuators.torque), driver))
        except Exception as exc:
            print(f"# skipped {seg}: {type(exc).__name__}", file=sys.stderr)
    return trace


def replay(trace: list, steer_max: int, rate_up: int, rate_down: int) -> dict:
    from opendbc.car.lateral import apply_driver_steer_torque_limits

    limits = Limits(steer_max, rate_up, rate_down)
    last = 0
    lo, hi = LOW_SPEED
    applied_low = []
    asked_low = []
    at_ceiling = 0

    for v, frac, driver in trace:
        want = int(round(frac * steer_max))
        last = apply_driver_steer_torque_limits(want, last, driver, limits)
        if lo <= v < hi:
            applied_low.append(abs(last))
            asked_low.append(abs(want))
            if abs(last) >= steer_max - 1:
                at_ceiling += 1

    n = len(applied_low)
    if not n:
        return {"frames": 0}
    return {
        "frames": n,
        # Mean applied counts is the honest summary: it is proportional to the mean torque the
        # EPS actually receives over the band, which is what "authority" means here.
        "mean_applied": round(sum(applied_low) / n, 1),
        "mean_asked": round(sum(asked_low) / n, 1),
        "pct_of_ask_delivered": round(100.0 * sum(applied_low) / max(sum(asked_low), 1), 1),
        "pct_frames_at_ceiling": round(100.0 * at_ceiling / n, 2),
        "p95_applied": round(sorted(applied_low)[int(n * 0.95)], 1),
    }


def main() -> int:
    routes = sys.argv[1:]
    if not routes:
        print("usage: ceiling_replay.py <route> [route ...]", file=sys.stderr)
        return 2
    assert_limiter_matches()

    segs = []
    for r in routes:
        segs += sorted(glob.glob(f"/data/media/0/realdata/{r}--*/rlog.zst"))
    print(f"# {len(segs)} segments; collecting engaged frames", flush=True)
    trace = collect(segs)
    print(f"# {len(trace)} engaged frames, " +
          f"{sum(1 for v, _, _ in trace if LOW_SPEED[0] <= v < LOW_SPEED[1])} in " +
          f"{LOW_SPEED[0]}-{LOW_SPEED[1]} m/s\n", flush=True)

    base = None
    out = []
    print(f"{'ceiling':>7} {'rate':>6} {'what it needs':>34} {'mean applied':>13} " +
          f"{'% of ask':>9} {'% at ceil':>10} {'vs today':>9}")
    for steer_max, ru, rd in GRID:
        r = replay(trace, steer_max, ru, rd)
        if not r["frames"]:
            continue
        if base is None:
            base = r["mean_applied"]
        needs = []
        if steer_max > 409:
            needs.append("opendbc")
            # Not an option, and the table has to say so. The MDPS accepts 409 and trips
            # CF_Mdps_ToiFlt at 410; 19 onsets were recorded at commanded counts 410-433, all
            # of them between 2.1 and 8.6 m/s where the schedules that produced them were
            # already flat. A gain printed for these rows is a gain the car cannot collect.
            needs.append("EPS FAULTS >409 (measured)")
        if ru != PANDA_RATE_UP or rd != PANDA_RATE_DOWN:
            needs.append("panda rate")
        # The rate limit and the real-time delta are two separate panda checks and raising one
        # without the other just moves which check rejects you.
        if ru * RT_INTERVAL_FRAMES > PANDA_MAX_RT_DELTA:
            needs.append(f"panda max_rt_delta {ru * RT_INTERVAL_FRAMES}")
        if steer_max > PANDA_CEILING:
            needs.append("OVER PANDA CEILING")
        r.update({"steer_max": steer_max, "rate_up": ru, "rate_down": rd,
                  "needs": needs or ["nothing - this is today"],
                  "gain_vs_today_pct": round((r["mean_applied"] / base - 1) * 100, 1)})
        out.append(r)
        rate = f"{ru}/{rd}"
        why = " + ".join(r["needs"])
        print(f"{steer_max:>7} {rate:>6} {why:>34} " +
              f"{r['mean_applied']:>13} {r['pct_of_ask_delivered']:>9} " +
              f"{r['pct_frames_at_ceiling']:>10} {r['gain_vs_today_pct']:>8}%")

    with open("/data/ceiling_replay.json", "w") as f:
        json.dump({"routes": routes, "low_speed_band": LOW_SPEED, "results": out}, f, indent=1)
    print("\n# wrote /data/ceiling_replay.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
