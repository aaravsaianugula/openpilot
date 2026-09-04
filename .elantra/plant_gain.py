#!/usr/bin/env python3
"""Measure the CN7 plant gain -- the lateral acceleration a full steering command actually buys.

This is the measurement the low-speed feedforward schedule is built on
(openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py). It existed as a
table in a docstring with no script behind it, so it could not be reproduced, re-run on new
drives, or checked. This is that script.

WHAT IT MEASURES

    F(v) = tyre lateral acceleration / (delivered counts / STEER_MAX)

i.e. the m/s^2 the car produces at a FULL command, per speed band. The feedforward divides a
desired lateral acceleration by latAccelFactor to get a torque request, so the request is only
right when latAccelFactor equals F(v). One scalar cannot equal a function of speed, and
torqued.py fits that scalar only above 15 m/s -- hence a schedule.

The gain the controller should use is therefore  g(v) = F(v) / latAccelFactor_in_force,
and BOTH terms are read from the log rather than assumed. latAccelFactor is NOT a constant of
this port: it is whatever lateralTorqueParameters carried on that drive (this fork renamed
liveTorqueParameters; the upstream name returns nothing, silently). STEER_MAX is not constant
across the archive either -- most of it ran 384 -- so it is recovered per segment.

THREE THINGS THAT MAKE THIS DIFFERENT FROM THE TABLE IT REPLACES

1. SETTLED VS TRANSIENT, reported separately, and this is the whole point.
   The same archive shows the command chattering: full-torque asks hold for a median 0.34 s
   against a rate limiter needing 1.36 s to ramp. A gain measured across frames where the
   command is still moving measures the CAR RESPONSE LAG, not its steady-state gain, and
   understates it. Understating F makes g too small, and the feedforward divides by g -- so an
   understated F causes the schedule to OVER-boost. A settled-only estimate is the honest one
   for a steady-state gain; the difference between the two columns is the size of that error.

2. THE ACTUATION LAG IS SCANNED, NOT ASSUMED.
   Yaw follows the command by steerActuatorDelay plus the vehicle own response. The lag is
   found per band by maximising |correlation| over a sweep, and printed, so a wrong assumption
   shows up as a lag at the edge of the sweep instead of quietly biasing every gain.

3. THE DRIVER IS EXCLUDED, not just steeringPressed.
   steeringPressed does not latch until STEER_THRESHOLD = 150 counts while the driver clamp
   starts cutting at 50, so "not pressed" still contains frames the driver was influencing.

ROLL. The controller feedforwards (desired_lat_accel - roll*g), so the tyres supply
(measured - roll*g) and that is what is regressed. Yaw is
liveLocationKalman.angularVelocityCalibrated -- carState.yawRate is always 0 on this fork.

Binned medians, never fits: the response is convex below ~10 m/s and a fit reports a slope the
car does not have. p10/p90 and n are printed beside every median so a thin band is visible.

    python .elantra/plant_gain.py --routes-dir D:/comma_four/routes --out plant_gain.json
    python .elantra/plant_gain.py --report plant_gain.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turn_tracking import (  # reuse: one decode path, one ceiling recovery
    BACKEND,
    _events,
    med,
    rlog_of,
    routes_under,
    steer_max_of,
)

G = 9.81
DT = 0.01                       # controlsState is 100 Hz

# Narrower at the bottom than turn_tracking bands: the schedule breakpoints are 3/4.5/6/8/10/13/15
# and a band that straddles two breakpoints cannot check either.
BANDS = ((2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 10.0),
         (10.0, 13.0), (13.0, 16.0), (16.0, 22.0), (22.0, 99.0))

LAG_SWEEP_MS = list(range(0, 501, 20))   # command -> yaw actuation lag, scanned not assumed
LAG_REPORT_MS = (0, 200, 400)   # settled ratio is reported at each: it MUST be flat across them
MIN_COUNTS = 80.0               # ratio denominator floor; below this a/(c/SM) is noise over noise
MAX_DRIVER = 30.0               # counts; the clamp starts cutting at 50, pressed latches at 150

# The settle window must be LONGER than the largest lag in the sweep. That is the whole point:
# in genuine steady state a = F*c regardless of lag, so a settled estimate is lag-invariant and
# the spread across LAG_REPORT_MS is a self-check on the criterion rather than a free parameter.
# At 20 frames the criterion admitted 99.7% of frames and the ratio still moved 13% with lag,
# which proved those frames were not settled at all.
SETTLE_FRAMES = 60              # 0.6 s of stable command, longer than the 0.5 s sweep
SETTLE_SPAN = 8.0               # counts of peak-to-peak movement allowed inside that window

# Sign convention, measured rather than assumed. carOutput.actuatorsOutput.torque is the
# delivered normalised command; openpilot's lateral convention has it opposing the lateral
# acceleration it produces (the controller returns -output_torque, "left is positive in this
# convention"). Folding the sign in here keeps F positive, so a NEGATIVE F means "this car
# steered the wrong way" -- a finding, rather than a convention nobody wrote down.
# Verified against the data: the command/accel correlation is -0.88 to -0.94 in every band.
COMMAND_SIGN = -1.0
MIN_BAND_N = 200                # below this a band reports its n and no median

# The band torqued.py itself fits (MIN_VEL = 15 m/s upward). g is expressed relative to THIS
# band rather than to a constant, so the schedule stays correct as the learner drifts.
LEARNER_BAND = "16-22"


def band_of(v):
    for lo, hi in BANDS:
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}"
    return None


def pearson(xs, ys):
    n = len(xs)
    if n < 30:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sxx = syy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx, dy = x - mx, y - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    d = math.sqrt(sxx * syy)
    return (sxy / d) if d > 0 else 0.0


def settled_flags(frames, steer_max):
    """True where the command has been stable for SETTLE_FRAMES, so the car has caught up.

    steer_max is REQUIRED: frames carry the normalised command in [-1, 1] while SETTLE_SPAN is
    in counts. Comparing the two directly makes every frame settled (|norm| <= 1 < 8) and the
    settled-vs-transient split -- the entire point of this tool -- silently vacuous.
    """
    out = [False] * len(frames)
    win: list[float] = []
    for i, f in enumerate(frames):
        if f is None:
            win = []
            continue
        win.append(f[1] * steer_max)
        if len(win) > SETTLE_FRAMES:
            win.pop(0)
        if len(win) == SETTLE_FRAMES and (max(win) - min(win)) <= SETTLE_SPAN:
            out[i] = True
    return out


def collect_segment(segdir):
    """Per-frame (v, out, yaw, roll, driver, pressed) plus this segment constants.

    The command is what the EPS was actually SENT, not what the controller asked for. Those
    differ by the driver clamp and the rate limiter, and the plant responds to the first. Taking
    controlsState.torqueState.output instead inflates the command on exactly the frames where
    the limiters bind, and so understates the gain -- measured at 7-26% in the low bands.

    That means the frame stream is driven by carOutput (from card) rather than controlsState
    (from controlsd), carrying the most recent carState / liveLocationKalman values. Both run at
    100 Hz, and keying off the message that carries the delivered value removes the cross-process
    join entirely rather than learning it: the k-th message of one process does not pair with the
    k-th of the other, and that mis-pairing has produced a wrong headline number here before.
    """
    path = rlog_of(segdir)
    if not path:
        return None
    v = roll = driver = 0.0
    lat = pressed = False
    yaw = None
    ok = True
    laf = None
    pairs, frames = [], []
    try:
        for e in _events(path):
            w = e.which()
            if w == "carState":
                c = e.carState
                v, driver, pressed = float(c.vEgo), float(c.steeringTorque), bool(c.steeringPressed)
            elif w == "carControl":
                lat = bool(e.carControl.latActive)
            elif w == "vehicleParameters":
                try:
                    roll = float(e.vehicleParameters.roll)
                except (AttributeError, ValueError, TypeError):
                    pass
            elif w == "lateralTorqueParameters":
                # The fork name. Upstream liveTorqueParameters returns nothing, silently.
                try:
                    f = float(e.lateralTorqueParameters.latAccelFactorFiltered)
                    if f > 0.1:
                        laf = f
                except (AttributeError, ValueError, TypeError):
                    pass
            elif w == "liveLocationKalman":
                m = e.liveLocationKalman.angularVelocityCalibrated
                yaw = float(m.value[2]) if bool(m.valid) else None
            elif w == "selfdriveState":
                a1 = e.selfdriveState.alertText1 or ""
                ok = not ("Calibrat" in a1 or "Big Model Failed" in a1)
            elif w == "carOutput":
                a = e.carOutput.actuatorsOutput
                pairs.append((float(a.torqueOutputCan), float(a.torque)))
                frames.append(None if not (lat and ok and yaw is not None)
                              else (v, float(a.torque), yaw, roll, driver, pressed))
    except Exception as ex:
        print(f"    ! {os.path.basename(segdir)}: {type(ex).__name__}: {str(ex)[:60]}",
              file=sys.stderr)
    sm = steer_max_of(pairs)
    if sm is None:
        return None                       # never steered hard enough to say; not a default
    return {"frames": frames, "steer_max": sm, "laf": laf}


def new_acc():
    return defaultdict(lambda: {"ratio": [], "a": [], "c": [], "ratio_settled": []})


def accumulate(seg, acc):
    """Add one segment usable frames into the per-band, per-lag accumulators."""
    frames, sm = seg["frames"], seg["steer_max"]
    settled = settled_flags(frames, sm)
    n = len(frames)
    for lag_ms in LAG_SWEEP_MS:
        k = int(round(lag_ms / 1000.0 / DT))
        for i in range(n - k):
            f, fy = frames[i], frames[i + k]
            if f is None or fy is None:
                continue
            v, out, _, roll, driver, pressed = f
            norm = COMMAND_SIGN * out       # normalised command in [-1, 1], yaw-positive
            counts = norm * sm
            if v < BANDS[0][0] or abs(counts) < MIN_COUNTS:
                continue
            if pressed or abs(driver) > MAX_DRIVER:
                continue
            b = band_of(v)
            if b is None:
                continue
            a_tyre = fy[2] * fy[0] - roll * G      # yaw*v, less the bank gravity term
            ratio = a_tyre / norm                   # lateral accel at a full command
            cell = acc[(b, lag_ms)]
            cell["ratio"].append(ratio)
            cell["a"].append(a_tyre)
            cell["c"].append(norm)
            if settled[i] and settled[i + k]:
                cell["ratio_settled"].append(ratio)


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def summarise(acc, laf_seen, steer_max_seen, segs, routes):
    """Pick each band lag by |correlation|, then report medians at that lag."""
    out = {}
    for b in [f"{lo:g}-{hi:g}" for lo, hi in BANDS]:
        best, best_r = None, -1.0
        for lag_ms in LAG_SWEEP_MS:
            cell = acc.get((b, lag_ms))
            if not cell or len(cell["ratio"]) < MIN_BAND_N:
                continue
            r = abs(pearson(cell["c"], cell["a"]))
            if r > best_r:
                best, best_r = lag_ms, r
        if best is None:
            # Report the real count at lag 0. A band with 180 frames and a band with none are
            # different facts, and printing 0 for both is the kind of thing this tool exists to
            # stop doing.
            seen = len(acc.get((b, LAG_SWEEP_MS[0]), {"ratio": []})["ratio"])
            out[b] = {"n": seen, "note": f"fewer than {MIN_BAND_N} frames"}
            continue
        cell = acc[(b, best)]
        enough = len(cell["ratio_settled"]) >= MIN_BAND_N
        by_lag = {}
        for lag in LAG_REPORT_MS:
            c2 = acc.get((b, lag))
            rs = c2["ratio_settled"] if c2 else []
            by_lag[lag] = round(med(rs), 3) if len(rs) >= MIN_BAND_N else None
        vals = [x for x in by_lag.values() if x is not None]
        spread = (max(vals) - min(vals)) / abs(med(vals)) if len(vals) > 1 and med(vals) else None
        out[b] = {
            "lag_ms": best,
            "corr": round(best_r, 3),
            "n": len(cell["ratio"]),
            "F_all": round(med(cell["ratio"]), 3),
            "F_all_p10": round(pct(cell["ratio"], 0.10), 3),
            "F_all_p90": round(pct(cell["ratio"], 0.90), 3),
            "n_settled": len(cell["ratio_settled"]),
            "F_settled": round(med(cell["ratio_settled"]), 3) if enough else None,
            "F_settled_p10": round(pct(cell["ratio_settled"], 0.10), 3) if enough else None,
            "F_settled_p90": round(pct(cell["ratio_settled"], 0.90), 3) if enough else None,
            "F_settled_by_lag": by_lag,
            "lag_spread": round(spread, 4) if spread is not None else None,
        }
    return {"bands": out, "lat_accel_factors": sorted(laf_seen),
            "steer_max_values": sorted(steer_max_seen), "segments": segs, "routes": routes,
            "backend": BACKEND, "min_counts": MIN_COUNTS, "max_driver": MAX_DRIVER,
            "settle_frames": SETTLE_FRAMES, "settle_span": SETTLE_SPAN,
            "lag_sweep_ms": [LAG_SWEEP_MS[0], LAG_SWEEP_MS[-1]]}


def report(res):
    laf = res["lat_accel_factors"]
    print(f"routes {res['routes']}  segments {res['segments']}  backend {res['backend']}")
    print(f"STEER_MAX seen: {res['steer_max_values']}")
    print(f"latAccelFactor in force: {laf}")
    if len(laf) != 1:
        print("  !! more than one latAccelFactor across these drives -- "
              + "g = F/latAccelFactor is not a single schedule for this set")
    print()
    print("F = lateral accel (m/s^2) at a FULL command.  g = F / latAccelFactor is what the")
    print("feedforward should divide by.  settled = command stable "
          + f"{res['settle_frames']} frames within {res['settle_span']:g} counts.")
    print()
    print(f"{'band':>9} {'lag':>5} {'corr':>5} {'n':>8} {'F_all':>7} {'n_set':>7} "
          + f"{'F_set':>7} {'p10':>6} {'p90':>6} {'F@0':>6} {'F@200':>6} {'F@400':>6} "
          + f"{'spread':>7} {'g':>6}")
    ref = (res["bands"].get(LEARNER_BAND) or {}).get("F_settled")
    if ref:
        print(f"g is F(v) / F({LEARNER_BAND}) = F(v) / {ref:.2f}, the band torqued itself fits.")
    for b, c in res["bands"].items():
        if "lag_ms" not in c:
            print(f"{b:>9} {'-':>5} {'-':>5} {c.get('n', 0):>8}  (too few frames)")
            continue
        fs = c["F_settled"]
        g = (fs / ref) if (fs is not None and ref) else None
        tail = (f"{fs:>7.2f} {c['F_settled_p10']:>6.2f} {c['F_settled_p90']:>6.2f} "
                if fs is not None else f"{'-':>7} {'-':>6} {'-':>6} ")
        bl = c.get("F_settled_by_lag") or {}
        cols = "".join(f"{bl.get(lag):>6.2f} " if bl.get(lag) is not None else f"{'-':>6} "
                       for lag in LAG_REPORT_MS)
        sp = c.get("lag_spread")
        print(f"{b:>9} {c['lag_ms']:>5} {c['corr']:>5.2f} {c['n']:>8} {c['F_all']:>7.2f} "
              + f"{c['n_settled']:>7} " + tail + cols
              + (f"{100 * sp:>6.1f}%" if sp is not None else f"{'-':>7}")
              + (f" {g:>5.3f}" if g is not None else f" {'-':>5}"))
    print()
    print("F@0 / F@200 / F@400 are the SETTLED gain measured at three different assumed")
    print("actuation lags. In genuine steady state a = F*c regardless of lag, so those three")
    print("must agree: `spread` is their range over their median and is the self-check on the")
    print("settle criterion. A spread over ~5% means the frames are not settled and the")
    print("headline number is a function of an assumed lag rather than a measurement.")
    print()
    print("g = F(v) / F(learner band) is the multiplier the feedforward should carry -- a RATIO")
    print("of gains, not F over a fixed constant, because latAccelFactor is learned at runtime")
    print("and moved 2.72-3.56 across this archive. A schedule divided by a fixed number is only")
    print("correct at the one runtime value it was fitted against.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes-dir")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--limit", type=int, default=0, help="stop after N segments (smoke test)")
    ap.add_argument("--steer-max", type=float, default=0.0,
                    help="only segments whose recovered ceiling is this (the archive mixes "
                         + "384 and 409 builds, and pooling them mixes two different cars)")
    ap.add_argument("--routes-file",
                    help="newline-delimited route names to include; the archive mixes builds "
                         + "and the set under measurement has to be nameable and re-runnable")
    a = ap.parse_args()

    if a.report:
        with open(a.report) as fh:
            report(json.load(fh))
        return 0
    if not (a.routes_dir and a.out):
        ap.error("pass --routes-dir and --out to measure, or --report to print a result")

    acc = new_acc()
    laf_seen, sm_seen = set(), set()
    segs = 0
    routes = routes_under(a.routes_dir)
    names = sorted(routes)
    if a.routes_file:
        with open(a.routes_file) as fh:
            want = {ln.strip() for ln in fh if ln.strip()}
        missing = want - set(names)
        if missing:
            print(f"  !! {len(missing)} named routes are not in {a.routes_dir}: "
                  + ", ".join(sorted(missing)[:5]), file=sys.stderr)
        names = [n for n in names if n in want]
    for ri, rname in enumerate(names, 1):
        for segdir in routes[rname]:
            seg = collect_segment(segdir)
            if seg is None:
                continue
            if a.steer_max and seg["steer_max"] != a.steer_max:
                continue
            accumulate(seg, acc)
            sm_seen.add(seg["steer_max"])
            if seg["laf"]:
                laf_seen.add(round(seg["laf"], 4))
            segs += 1
            if a.limit and segs >= a.limit:
                break
        print(f"  [{ri}/{len(names)}] {rname}  segments so far {segs}", file=sys.stderr)
        if a.limit and segs >= a.limit:
            break

    res = summarise(acc, laf_seen, sm_seen, segs, len(names))
    res["routes_file"] = a.routes_file
    res["steer_max_filter"] = a.steer_max or None
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
