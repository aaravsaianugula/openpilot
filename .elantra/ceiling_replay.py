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

The 10/10 row is NOT "what carrotpilot runs", whatever this file used to say. Read from
ajouatom/openpilot @ carrot-wip on 2026-09-03: their panda declares HYUNDAI_LIMITS(512, 10, 10),
but their CarControllerParams commands STEER_DELTA_UP = 3 / STEER_DELTA_DOWN = 7 -- the same
rate this car already uses -- and they keep max_rt_delta = 112, which caps any sustained ramp at
112/25 = 4.48 counts/frame regardless. Their 10/10 is a permission their car controller never
exercises and their own panda could not sustain. There is no ramp-rate setting to copy from
them.
"""

from __future__ import annotations

import argparse
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

# panda runs the SAME driver clamp shape as opendbc -- TorqueDriverLimited, allowance 50,
# multiplier 2 (opendbc/safety/modes/hyundai.h HYUNDAI_LIMITS) -- but against max_torque 512
# rather than opendbc's 409. Its window is therefore 103 counts wider at every driver torque,
# and opendbc may spend that headroom without a reflash. Solving
#   409 + (A + d)*2  <=  512 + (50 + d)*2      for all d
# gives A <= (512 + 100 - 409)/2 = 101.5, so 101 is the largest allowance that stays strictly
# inside what panda already enforces. Above it, opendbc would command torque panda rejects.
PANDA_DRIVER_ALLOWANCE = 50
PANDA_DRIVER_MULTIPLIER = 2
MAX_ALLOWANCE_NO_REFLASH = (PANDA_CEILING + PANDA_DRIVER_ALLOWANCE * PANDA_DRIVER_MULTIPLIER
                            - 409) // PANDA_DRIVER_MULTIPLIER

# Reported per band as well as pooled: the driver clamp bites hardest below 10 m/s, where the
# column-torque reaction is largest, and a pooled 3-14 m/s figure averages that away.
BANDS = ((1.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, 14.0), (14.0, 18.0))

# The band the shortfall actually lives in, per the flat-409 road data.
LOW_SPEED = (3.0, 14.0)

# (steer_max, rate_up, rate_down, driver_allowance). Row 0 is what the car runs today.
GRID = [
    (409, 3, 7, 100),   # today
    (409, 3, 7, 50),    # what the archive was recorded under, before the allowance raise
    (409, 3, 7, 75),    # the halfway point, kept so the curve is visible
    (409, 3, 7, 125),   # OVER panda's window: shown to prove the guard, not as an option
    (409, 4, 7, 50),    # the most rate today's max_rt_delta=112 can actually sustain
    (409, 10, 10, 50),  # NOT carrotpilot's rate -- see below; needs max_rt_delta raised too
    (450, 3, 7, 50),    # ceiling only -- and the EPS has already refused this, see below
    (500, 3, 7, 50),
]


class Limits:
    def __init__(self, steer_max: int, rate_up: int, rate_down: int, allowance: int = 50):
        self.STEER_MAX = steer_max
        self.STEER_DELTA_UP = rate_up
        self.STEER_DELTA_DOWN = rate_down
        self.STEER_DRIVER_ALLOWANCE = allowance
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
    if GRID[0][3] != real.STEER_DRIVER_ALLOWANCE:
        raise SystemExit(f"grid row 0 allowance {GRID[0][3]} but the car runs {real.STEER_DRIVER_ALLOWANCE}")


def _reader():
    """openpilot's LogReader on the device, the local schema loader off it.

    Same two-backend arrangement as turn_tracking.py, so this sweep can be priced against the
    archive on a laptop instead of only against whatever the device still holds -- the device
    rotates oldest-first and the archive goes back much further.
    """
    try:
        from openpilot.tools.lib.logreader import LogReader
        return lambda p: LogReader(p, sort_by_time=True)
    except ImportError:
        import oplog
        return oplog.events


TRACE_MAGIC = b"CRTR0001"


def dump_trace(trace: list, path: str) -> None:
    """Write (vEgo, torque, driver) as three flat float32 blocks behind a small header.

    Flat binary rather than pickle: this file exists only to carry numbers between two
    processes, and a format that cannot execute anything is the right one for a file a tool
    reads back without checking where it came from.

    float32 is deliberate and sufficient: torque is a normalised float that opendbc rounds to
    an integer count anyway, driver torque is already an integer, and vEgo only picks a speed
    band. It keeps a 3.4M-frame archive trace around 40 MB instead of a few hundred.
    """
    import array
    import struct

    with open(path, "wb") as fh:
        fh.write(TRACE_MAGIC + struct.pack("<Q", len(trace)))
        for i in range(3):
            col = array.array("f", [t[i] for t in trace])
            if sys.byteorder == "big":
                col.byteswap()
            col.tofile(fh)


def load_trace(path: str) -> list:
    import array
    import struct

    with open(path, "rb") as fh:
        head = fh.read(len(TRACE_MAGIC) + 8)
        if head[:len(TRACE_MAGIC)] != TRACE_MAGIC:
            raise SystemExit(f"{path} is not a ceiling_replay trace")
        n = struct.unpack("<Q", head[len(TRACE_MAGIC):])[0]
        cols = []
        for _ in range(3):
            col = array.array("f")
            col.fromfile(fh, n)
            if sys.byteorder == "big":
                col.byteswap()
            cols.append(col)
    return list(zip(*cols, strict=True))


def collect(segs: list) -> list:
    """(vEgo, normalised torque, driver column torque) for every engaged frame."""
    events = _reader()

    trace = []
    for seg in segs:
        v = 0.0
        driver = 0.0
        try:
            for ev in events(seg):
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


def replay(trace: list, steer_max: int, rate_up: int, rate_down: int,
           allowance: int = 50) -> dict:
    from opendbc.car.lateral import apply_driver_steer_torque_limits

    limits = Limits(steer_max, rate_up, rate_down, allowance)
    last = 0
    lo, hi = LOW_SPEED
    applied_low = []
    asked_low = []
    at_ceiling = 0
    per_band = {b: [0, 0.0, 0.0] for b in BANDS}  # frames, sum applied, sum asked

    for v, frac, driver in trace:
        want = int(round(frac * steer_max))
        last = apply_driver_steer_torque_limits(want, last, driver, limits)
        for b in BANDS:
            if b[0] <= v < b[1]:
                per_band[b][0] += 1
                per_band[b][1] += abs(last)
                per_band[b][2] += abs(want)
                break
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
        "per_band": {f"{b[0]:g}-{b[1]:g}": (c[0], round(c[1] / c[0], 1) if c[0] else None)
                     for b, c in per_band.items()},
        # Mean applied counts is the honest summary: it is proportional to the mean torque the
        # EPS actually receives over the band, which is what "authority" means here.
        "mean_applied": round(sum(applied_low) / n, 1),
        "mean_asked": round(sum(asked_low) / n, 1),
        "pct_of_ask_delivered": round(100.0 * sum(applied_low) / max(sum(asked_low), 1), 1),
        "pct_frames_at_ceiling": round(100.0 * at_ceiling / n, 2),
        "p95_applied": round(sorted(applied_low)[int(n * 0.95)], 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("routes", nargs="*",
                    help="route ids to replay; omit to use every route under --routes-dir")
    ap.add_argument("--routes-dir", default="/data/media/0/realdata",
                    help="where the segment directories live")
    ap.add_argument("--out", default="/data/lat-tracking/ceiling_replay.json")
    ap.add_argument("--dump-trace", metavar="FILE",
                    help="decode only: write the trace here and exit, importing no opendbc")
    ap.add_argument("--trace", metavar="FILE",
                    help="replay a trace written by --dump-trace, decoding nothing")
    a = ap.parse_args()

    if a.trace:
        assert_limiter_matches()
        trace = load_trace(a.trace)
        print(f"# {len(trace)} engaged frames from {a.trace}", flush=True)
    else:
        if a.routes:
            segs = []
            for r in a.routes:
                segs += sorted(glob.glob(f"{a.routes_dir}/{r}--*/rlog.zst"))
        else:
            segs = sorted(glob.glob(f"{a.routes_dir}/*--*--*/rlog.zst"))
        print(f"# {len(segs)} segments; collecting engaged frames", flush=True)
        trace = collect(segs)
        if a.dump_trace:
            dump_trace(trace, a.dump_trace)
            print(f"# wrote {len(trace)} frames to {a.dump_trace}", flush=True)
            return 0
        # Deferred until after decoding: on a machine using the local backend this import is
        # what collides with the decoder, so it must not happen while oplog is still needed.
        assert_limiter_matches()
    low = sum(1 for v, _, _ in trace if LOW_SPEED[0] <= v < LOW_SPEED[1])
    print(f"# {len(trace)} engaged frames, {low} in {LOW_SPEED[0]}-{LOW_SPEED[1]} m/s\n", flush=True)

    base = None
    out = []
    print(f"{'ceiling':>7} {'rate':>6} {'allow':>6} {'what it needs':>36} "
          + f"{'mean applied':>13} {'% of ask':>9} {'% at ceil':>10} {'vs today':>9}")
    for steer_max, ru, rd, allow in GRID:
        r = replay(trace, steer_max, ru, rd, allow)
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
        if allow != PANDA_DRIVER_ALLOWANCE:
            # An allowance at or under MAX_ALLOWANCE_NO_REFLASH keeps opendbc's driver window
            # strictly inside panda's, which is already 103 counts wider at every driver torque.
            # Above it, opendbc would command torque panda rejects, and the row is not an option.
            needs.append("opendbc driver clamp" if allow <= MAX_ALLOWANCE_NO_REFLASH
                         else f"OVER PANDA WINDOW (max {MAX_ALLOWANCE_NO_REFLASH})")
        r.update({"steer_max": steer_max, "rate_up": ru, "rate_down": rd, "allowance": allow,
                  "needs": needs or ["nothing - this is today"],
                  "gain_vs_today_pct": round((r["mean_applied"] / base - 1) * 100, 1)})
        out.append(r)
        why = " + ".join(r["needs"])
        print(f"{steer_max:>7} {f'{ru}/{rd}':>6} {allow:>6} {why:>36} "
              + f"{r['mean_applied']:>13} {r['pct_of_ask_delivered']:>9} "
              + f"{r['pct_frames_at_ceiling']:>10} {r['gain_vs_today_pct']:>8}%")

    # Per band, because the column-torque reaction that drives the driver clamp scales inversely
    # with speed: a pooled 3-14 m/s mean averages the worst of it away.
    print("\n# mean applied counts by speed band (frames in brackets)")
    band_names = [f"{b[0]:g}-{b[1]:g}" for b in BANDS]
    print(f"{'ceiling/rate/allow':>20} " + " ".join(f"{b:>13}" for b in band_names))
    for r in out:
        label = f"{r['steer_max']}/{r['rate_up']}-{r['rate_down']}/{r['allowance']}"
        cells = []
        for b in band_names:
            n, mean = r["per_band"].get(b, (0, None))
            cells.append(f"{mean:>6} ({n:>5})" if mean is not None else f"{'-':>13}")
        print(f"{label:>20} " + " ".join(cells))

    with open(a.out, "w") as f:
        json.dump({"routes": a.routes or "all", "routes_dir": a.routes_dir,
                   "low_speed_band": LOW_SPEED, "results": out}, f, indent=1)
    print(f"\n# wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
