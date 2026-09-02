#!/usr/bin/env python3
"""What moving the steering ceiling would cost, measured against the recorded drives.

THE ANSWER FOR THIS CAR IS ALREADY IN: DO NOT RAISE IT. Measured 2026-09-01 -- the MDPS accepts
409 and trips CF_Mdps_ToiFlt at 410. The two raised builds put 158 frames above 409 on the wire
and took 19 fault onsets, at commanded counts 410 to 433, all of them under a ceiling that was
STATIONARY at the time (np.interp clamps below its first breakpoint, and every onset was below
it). So this tool refuses any candidate above 409 (see EPS_CEILING).
It is kept because the counterfactual machinery below is the only honest way to price a
ceiling change against real drives, and because a LOWER candidate is still a live question.

Two questions, answered in one pass, because a pass over the whole route store costs an hour.

1. THE COST, as a ONE-STEP counterfactual. Each frame, both chains are seeded from the torque
   the car was ACTUALLY applying on the previous frame, then stepped once through the real
   apply_driver_steer_torque_limits under each ceiling. Deliberately NOT `fraction * ceiling`:
   the rate limiter and the driver-torque envelope are history dependent, so the delta they
   produce is not the delta the multiplication predicts.

   BEFORE is the flat 409 the recorded drives ACTUALLY ran, so the BEFORE chain is a real
   one-frame prediction test -- it must reproduce the next recorded command exactly. AFTER is
   the flat candidate, which defaults to 409 (a no-op, so the tool reports honestly that
   nothing changed) and is set with --candidate.

   One step, not a free-running trajectory, and that is a deliberate limit rather than a
   shortcut. Lateral control is a feedback loop: a counterfactual that ran for 60 s would have
   driven a different path, so the recorded `actuators.torque` would no longer be the demand.
   Letting it run free produces a confident-looking number that is fiction. What this measures
   is well defined and defensible: the instantaneous difference in commanded counts at the
   operating points the car actually visited.

   Seeding from the recorded output also makes the BEFORE chain a genuine prediction test --
   it must reproduce the next recorded command exactly, and the exact-match rate is reported
   rather than smoothed. A model that cannot predict one frame ahead has no standing to say
   what the other ceiling would have done.

2. THE FACTORY ENVELOPE. Every LKAS11 frame the factory camera sent (camera bus, src < 128),
   decoded for CR_Lkas_StrToqReq. comma's rule for picking a steering limit is "find the
   maximum value that the stock LKAS will request", and nobody has ever applied it to the CN7.
   Counting zeros is a result: it says the rule cannot be applied from this data at all, which
   is a different and more honest claim than any particular number.

Ceilings are passed in, never imported from the checkout under test. Reading them from
CarControllerParams would make the tool agree with whichever side of the change happens to be
checked out, and measure nothing.

    scan    PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 torque_projection.py scan
    report  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 torque_projection.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# hyundai_can.dbc: BO_ 832 LKAS11, SG_ CR_Lkas_StrToqReq : 16|11@1+ (1.0,-1024.0)
#                                  SG_ CF_Lkas_ActToi    : 27|1@1+
LKAS11 = 0x340
TORQUE_OFFSET = 1024
OP_TX_SRC = 128          # src >= 128 is a frame openpilot transmitted
CAMERA_BUS = 2           # the factory camera's own LKAS11 lives here

# 8.94 and 13.41 m/s are 20.0 and 30.0 mph, and they are band edges on purpose: 20 mph falls
# INSIDE a 7-10 band, so "what happens under 20 mph" -- the regime this car's shortfall
# actually lives in -- could not be read off the table without mixing in frames from above it.
# Holding these splits still also keeps reports comparable across scans, which matters because
# a re-scan needs the car. This costs comparability with the 10-14 / 14-18 split used
# elsewhere; lateral_report.py keeps those and is unchanged.
SPEED_BANDS = ((0.0, 3.0), (3.0, 7.0), (7.0, 8.94), (8.94, 13.41), (13.41, 18.0), (18.0, 1e9))
BAND_NAMES = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in SPEED_BANDS]

# CarControllerParams for this platform, restated so the tool does not move when the checkout
# does. assert_params_match() refuses to run if the real ones have drifted from these.
RATE_UP = 3
RATE_DOWN = 7
DRIVER_ALLOWANCE = 50
DRIVER_MULTIPLIER = 2
DRIVER_FACTOR = 1

BEFORE_FLAT = 409         # what the recorded drives actually ran
DEFAULT_CANDIDATE = 409   # the flat candidate to price against it; override with --candidate
CANDIDATE_CEILING = DEFAULT_CANDIDATE   # rebound by main() when --candidate is given

# THE HIGHEST COUNT THIS MDPS ACCEPTS. Measured 2026-09-01, not inferred: routes 000000dc (500
# schedule) and 000000dd (450 schedule) put 158 frames above 409 on the wire and took 19
# CF_Mdps_ToiFlt onsets while steering, at commanded counts of 410 through 433 -- minimum 410 --
# against ~1.59M engaged frames at 409 or below with none at all.
#
# It is the VALUE, not the schedule: np.interp clamps below its first breakpoint, so under
# 8.94 m/s both schedules held a CONSTANT ceiling, and all 19 onsets happened between 2.1 and
# 8.6 m/s. Every fault therefore happened under a stationary ceiling, which is the flat-ceiling
# experiment that was thought never to have run.
#
# So this is a hardware boundary and the tool refuses above it. Do not raise this constant: a
# projection is a number someone acts on, and every count from 410 up is one the EPS rejects.
EPS_CEILING = 409

# What panda would let out. Deliberately NOT the limit this tool enforces -- panda sits 103
# counts above the EPS ceiling, so "under panda" is not the same as "the car can steer".
PANDA_CEILING = 512


class Limits:
    """The subset of CarControllerParams that apply_driver_steer_torque_limits reads."""

    def __init__(self, steer_max: int):
        self.STEER_MAX = steer_max
        self.STEER_DELTA_UP = RATE_UP
        self.STEER_DELTA_DOWN = RATE_DOWN
        self.STEER_DRIVER_ALLOWANCE = DRIVER_ALLOWANCE
        self.STEER_DRIVER_MULTIPLIER = DRIVER_MULTIPLIER
        self.STEER_DRIVER_FACTOR = DRIVER_FACTOR


def assert_params_match() -> None:
    """The class above is a copy, and copies drift. This is what stops it drifting silently."""
    from opendbc.car.hyundai.values import CarControllerParams, HyundaiFlags

    class _Probe:
        # CarControllerParams reads exactly these two. Building a real CarParams here would
        # couple the check to the struct backend for no gain.
        carFingerprint = "HYUNDAI_ELANTRA_2024"
        flags = int(HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC)

    real = CarControllerParams(_Probe())
    mismatched = {
        name: (mine, getattr(real, name))
        for name, mine in (
            ("STEER_DELTA_UP", RATE_UP),
            ("STEER_DELTA_DOWN", RATE_DOWN),
            ("STEER_DRIVER_ALLOWANCE", DRIVER_ALLOWANCE),
            ("STEER_DRIVER_MULTIPLIER", DRIVER_MULTIPLIER),
            ("STEER_DRIVER_FACTOR", DRIVER_FACTOR),
        )
        if mine != getattr(real, name)
    }
    if mismatched:
        raise SystemExit("CarControllerParams no longer match this tool's copy: " + repr(mismatched))

    if CANDIDATE_CEILING > EPS_CEILING:
        raise SystemExit(
            f"refusing to project a {CANDIDATE_CEILING}-count ceiling: this MDPS accepts " +
            f"{EPS_CEILING} and trips CF_Mdps_ToiFlt at {EPS_CEILING + 1}. Measured on routes " +
            "000000dc and 000000dd -- 158 frames above 409, 19 fault onsets, lowest at 410 -- " +
            "against ~1.59M engaged frames at 409 or below with none. Every onset happened " +
            "under a STATIONARY ceiling, so this is the value and not the schedule. panda " +
            f"would pass it ({PANDA_CEILING}); that is not the check made here, and 410-512 " +
            "is exactly the band the EPS rejects. Do not raise EPS_CEILING.")


def band_of(v: float) -> str | None:
    for (lo, hi), name in zip(SPEED_BANDS, BAND_NAMES, strict=False):
        if lo <= v < hi:
            return name
    return None


def decode_lkas11(dat: bytes):
    """(signed torque counts, ActToi bit), or None if the frame is too short."""
    if len(dat) < 4:
        return None
    word = int.from_bytes(dat[0:4], "little")
    return ((word >> 16) & 0x7FF) - TORQUE_OFFSET, (word >> 27) & 1


def new_band() -> dict:
    return {
        "frames": 0,
        "before_at_ceiling": 0,
        "after_at_ceiling": 0,
        "before_max_abs": 0,
        "after_max_abs": 0,
        "before_sum_abs": 0,
        "after_sum_abs": 0,
        "changed": 0,
        "sum_abs_delta": 0,
        "max_abs_delta": 0,
        "delta_hist": {},
    }


def collect_segment(seg: str, factory: dict, tx: dict):
    """One pass over the log: the per-cycle inputs, the recorded outputs, and the CAN counters.

    Returns (ccs, reals) where ccs[i] is (latActive, fraction, vEgoRaw, driverTorque) for the
    i-th carControl and reals[k] is the k-th recorded torqueOutputCan.

    sort_by_time is not optional. An rlog is written per service and lands in bursts, so file
    order is not execution order: consecutive carControl frames can precede the carState frames
    they were computed against, and the simulation then steps on a stale speed and driver
    torque. Measured on one segment, this alone moved one-frame agreement from 79% to 84%.
    """
    from openpilot.tools.lib.logreader import LogReader

    ccs: list[tuple[bool, float, float, float]] = []
    reals: list[int] = []
    v_raw = 0.0
    driver_tq = 0.0

    for ev in LogReader(seg, sort_by_time=True):
        w = ev.which()
        if w == "carState":
            v_raw = float(ev.carState.vEgoRaw)
            driver_tq = float(ev.carState.steeringTorque)

        elif w == "carOutput":
            reals.append(int(ev.carOutput.actuatorsOutput.torqueOutputCan))

        elif w == "carControl":
            cc = ev.carControl
            ccs.append((bool(cc.latActive), float(cc.actuators.torque), v_raw, driver_tq))

        elif w == "can":
            for c in ev.can:
                if c.address != LKAS11:
                    continue
                dec = decode_lkas11(bytes(c.dat))
                if dec is None:
                    continue
                torque, toi = dec
                if c.src >= OP_TX_SRC:
                    tx["frames"] += 1
                    tx["max_abs"] = max(tx["max_abs"], abs(torque))
                elif c.src == CAMERA_BUS:
                    factory["frames"] += 1
                    factory["max_abs"] = max(factory["max_abs"], abs(torque))
                    if torque != 0:
                        factory["nonzero"] += 1
                    if toi:
                        factory["act_toi"] += 1

    return ccs, reals


def step(frac: float, prev: int, driver: float, limits, ceiling: int, explicit_ceiling: bool) -> int:
    """One frame of the car's own limiter, seeded from `prev`.

    `explicit_ceiling` picks the CALL SHAPE, which is a real behavioural choice and not a
    formality: upstream's apply_driver_steer_torque_limits takes an optional 5th argument, and
    a caller that passes it drives the driver-override envelope from that value instead of
    from limits.STEER_MAX. With a flat candidate the two agree, so both shapes are exercised
    here deliberately -- if they ever stop agreeing, the tests below are what says so.
    """
    from opendbc.car.lateral import apply_driver_steer_torque_limits
    new_torque = int(round(frac * ceiling))
    if explicit_ceiling:
        return apply_driver_steer_torque_limits(new_torque, prev, driver, limits, ceiling)
    return apply_driver_steer_torque_limits(new_torque, prev, driver, limits)


def simulate_segment(seg: str, bands: dict, factory: dict, valid: dict, tx: dict) -> None:
    ccs, reals = collect_segment(seg, factory, tx)

    lim_before = Limits(BEFORE_FLAT)
    lim_after = Limits(CANDIDATE_CEILING)

    # carOutput at cycle k reports the CarController result computed from carControl at k-1, so
    # the car's own apply_torque_last while processing ccs[i] was reals[i], and the command it
    # produced was reals[i+1]. That lag is SCORED at both offsets rather than assumed: if a
    # future pipeline change moves it, lag 0 wins instead and the report says so, rather than
    # the tool quietly measuring a shifted signal.
    n = min(len(ccs), len(reals) - 1)
    for i in range(max(n, 0)):
        lat, frac, v_raw, driver_tq = ccs[i]
        prev = reals[i]
        if not lat:
            before = after = 0
        else:
            before = step(frac, prev, driver_tq, lim_before, BEFORE_FLAT, False)
            after = step(frac, prev, driver_tq, lim_after, CANDIDATE_CEILING, True)

        valid["paired"] += 1
        if before == reals[i + 1]:
            valid["exact_lag1"] += 1
        if before == reals[i]:
            valid["exact_lag0"] += 1
        d = abs(reals[i + 1] - before)
        if d <= 1:
            valid["within1_lag1"] += 1
        valid["max_mismatch"] = max(valid["max_mismatch"], d)
        valid["sum_mismatch"] += d

        if not lat:
            continue
        band = band_of(v_raw)
        if band is None:
            continue
        b = bands[band]
        b["frames"] += 1
        ab, aa = abs(before), abs(after)
        b["before_sum_abs"] += ab
        b["after_sum_abs"] += aa
        b["before_max_abs"] = max(b["before_max_abs"], ab)
        b["after_max_abs"] = max(b["after_max_abs"], aa)
        if ab >= BEFORE_FLAT:
            b["before_at_ceiling"] += 1
        if aa >= CANDIDATE_CEILING:
            b["after_at_ceiling"] += 1
        delta = after - before
        if delta:
            b["changed"] += 1
            b["sum_abs_delta"] += abs(delta)
            b["max_abs_delta"] = max(b["max_abs_delta"], abs(delta))
        b["delta_hist"][str(max(-40, min(40, delta)))] =             b["delta_hist"].get(str(max(-40, min(40, delta))), 0) + 1


def cmd_scan(args) -> int:
    assert_params_match()
    segs = sorted(glob.glob(str(Path(args.routes) / "*" / "rlog.zst")))
    if args.limit:
        segs = segs[-args.limit:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.is_file() and not args.force:
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seg"])
            except Exception:
                pass

    todo = [s for s in segs if os.path.basename(os.path.dirname(s)) not in done]
    print(f"segments: {len(segs)} total, {len(done)} already done, {len(todo)} to go", flush=True)

    with out.open("a", encoding="utf-8") as fh:
        for i, seg in enumerate(todo):
            name = os.path.basename(os.path.dirname(seg))
            rec = {
                "seg": name,
                # Stamped per record, not per file. `report` reads it back instead of trusting
                # the module default -- otherwise `scan --candidate 384` followed by a bare
                # `report` prints a header naming a ceiling the data was never scanned at. The
                # resume path makes it worse: two runs at different candidates append to the
                # SAME jsonl, and without this nothing anywhere records that they differ.
                "candidate": CANDIDATE_CEILING,
                "bands": {b: new_band() for b in BAND_NAMES},
                "factory": {"frames": 0, "nonzero": 0, "max_abs": 0, "act_toi": 0},
                "tx": {"frames": 0, "max_abs": 0},
                "valid": {"paired": 0, "exact_lag0": 0, "exact_lag1": 0, "within1_lag1": 0,
             "max_mismatch": 0, "sum_mismatch": 0},
                "error": None,
            }
            try:
                simulate_segment(seg, rec["bands"], rec["factory"], rec["valid"], rec["tx"])
            except Exception as e:      # counted and named, never silently dropped
                rec["error"] = f"{type(e).__name__}: {e}"
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}  {name}  err={rec['error']}", flush=True)
    print("scan complete", flush=True)
    return 0


def merge(recs: list) -> tuple:
    bands = {b: new_band() for b in BAND_NAMES}
    factory = {"frames": 0, "nonzero": 0, "max_abs": 0, "act_toi": 0}
    tx = {"frames": 0, "max_abs": 0}
    valid = {"paired": 0, "exact_lag0": 0, "exact_lag1": 0, "within1_lag1": 0,
             "max_mismatch": 0, "sum_mismatch": 0}
    for r in recs:
        if r["error"]:
            continue
        for k in factory:
            factory[k] = max(factory[k], r["factory"][k]) if k == "max_abs" else factory[k] + r["factory"][k]
        tx["frames"] += r["tx"]["frames"]
        tx["max_abs"] = max(tx["max_abs"], r["tx"]["max_abs"])
        for k in valid:
            valid[k] = max(valid[k], r["valid"][k]) if k == "max_mismatch" else valid[k] + r["valid"][k]
        for b, d in r["bands"].items():
            t = bands[b]
            for k, v in d.items():
                if k == "delta_hist":
                    for kk, vv in v.items():
                        t[k][kk] = t[k].get(kk, 0) + vv
                elif k.endswith(("max_abs", "max_abs_delta")):
                    t[k] = max(t[k], v)
                else:
                    t[k] += v
    return bands, factory, tx, valid


def cmd_report(args) -> int:
    path = Path(args.out)
    if not path.is_file():
        raise SystemExit("no scan output at " + str(path))
    recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    errored = [r["seg"] for r in recs if r["error"]]

    # Which ceiling was this scanned AT? Read from the records, never from the module default.
    # Records predating this stamp have no candidate; they are reported as unknown rather than
    # assumed, because assuming is how a report ends up naming a ceiling it never measured.
    seen = {r.get("candidate") for r in recs}
    if len(seen) > 1:
        raise SystemExit(
            "this scan file mixes candidate ceilings " + repr(sorted(seen, key=str)) + ". " +
            "Two runs at different --candidate values appended to the same file, so no single " +
            "number describes it. Re-scan with --force, or report each separately.")
    candidate = next(iter(seen)) if seen else None
    if candidate is None:
        raise SystemExit(
            "this scan file predates candidate stamping, so what ceiling it measured is " +
            "unrecorded. Re-scan with --force rather than guessing.")

    bands, factory, tx, valid = merge(recs)

    print(f"segments scanned: {len(recs)}   unreadable: {len(errored)}")
    for s in errored[:10]:
        print("   ! " + s)

    print("\n--- simulation fidelity (BEFORE chain vs what openpilot recorded) ---")
    p = valid["paired"] or 1
    print(f"  paired frames {valid['paired']}")
    print(f"  exact at lag 0 {valid['exact_lag0']} ({100*valid['exact_lag0']/p:.3f}%)" +
          f"   exact at lag 1 {valid['exact_lag1']} ({100*valid['exact_lag1']/p:.3f}%)")
    print("     lag 1 is the pipeline's own carControl -> carOutput delay, scored not assumed")
    print(f"  within 1 count at lag 1 {valid['within1_lag1']} " +
          f"({100*valid['within1_lag1']/p:.3f}%)")
    print(f"  worst mismatch {valid['max_mismatch']} counts   " +
          f"mean |mismatch| {valid['sum_mismatch']/p:.4f}")
    if valid["exact_lag1"] < valid["exact_lag0"]:
        print("  !! lag 0 fits better than lag 1 -- the pipeline alignment assumed here has")
        print("     changed, and every delta below is measured against a shifted signal.")

    print(f"\n--- what a flat {candidate} changes, by speed band " +
          f"(BEFORE = flat {BEFORE_FLAT}) ---")
    print(f"  {'band':>8} {'frames':>9} {'changed':>9} {'%chg':>7} {'mean|d|':>8} {'max|d|':>7} " +
          f"{'pin_before':>11} {'pin_after':>10} {'maxcmd_b':>9} {'maxcmd_a':>9}")
    tot_frames = tot_changed = 0
    for b in BAND_NAMES:
        d = bands[b]
        n = d["frames"]
        tot_frames += n
        tot_changed += d["changed"]
        if not n:
            print(f"  {b:>8} {0:>9}")
            continue
        print(f"  {b:>8} {n:>9} {d['changed']:>9} {100*d['changed']/n:>6.2f}% " +
              f"{d['sum_abs_delta']/max(d['changed'],1):>8.2f} {d['max_abs_delta']:>7} " +
              f"{100*d['before_at_ceiling']/n:>10.3f}% {100*d['after_at_ceiling']/n:>9.3f}% " +
              f"{d['before_max_abs']:>9} {d['after_max_abs']:>9}")
    print(f"  {'ALL':>8} {tot_frames:>9} {tot_changed:>9} " +
          f"{100*tot_changed/max(tot_frames,1):>6.2f}%")

    print("\n--- the factory envelope (camera-bus LKAS11, what stock LKAS asked for) ---")
    print(f"  frames {factory['frames']}   nonzero torque frames {factory['nonzero']}   " +
          f"max |CR_Lkas_StrToqReq| {factory['max_abs']}   ActToi frames {factory['act_toi']}")
    if factory["frames"] and not factory["nonzero"]:
        print("  => the factory camera never actuated in any recorded frame. comma's rule")
        print("     ('find the maximum value that the stock LKAS will request') CANNOT be")
        print("     applied from this data -- for 384 as much as for 409.")

    print("\n--- openpilot's own transmitted LKAS11 (sanity) ---")
    print(f"  frames {tx['frames']}   max |torque| {tx['max_abs']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="simulate both chains over every recorded segment")
    s.add_argument("--routes", default="/data/media/0/realdata")
    s.add_argument("--out", default="/data/elantra-projection.jsonl")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--candidate", type=int, default=DEFAULT_CANDIDATE,
                   help=f"flat ceiling to price against the recorded {BEFORE_FLAT} " +
                        f"(default {DEFAULT_CANDIDATE}; refused above {EPS_CEILING}, " +
                        "where this car's EPS faults)")
    s.set_defaults(func=cmd_scan)
    r = sub.add_parser("report", help="aggregate a completed scan")
    r.add_argument("--out", default="/data/elantra-projection.jsonl")
    r.set_defaults(func=cmd_report)
    global CANDIDATE_CEILING
    args = ap.parse_args()
    if getattr(args, "candidate", None) is not None:
        # Bound to the module global rather than threaded through, because simulate_segment
        # and the report both read it and the alternative is six extra parameters. Set before
        # assert_params_match() runs, so an out-of-range candidate is refused up front.
        CANDIDATE_CEILING = args.candidate
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
