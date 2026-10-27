#!/usr/bin/env python3
"""Where does the low-speed steering demand actually go at a flat 409 ceiling?

The planner asks for more than the car delivers between 3 and 14 m/s. That shortfall has six
candidate owners and this walks the chain frame by frame to name the one that owns it:

    planner demand      carControl.actuators.curvature * vEgo^2
      -> controller     controlsState...torqueState  (f, p, i, output, saturated)
      -> normalised     carControl.actuators.torque          pinned at +-1.0?
      -> counts         carOutput.actuatorsOutput.torqueOutputCan  == round(torque * 409)?
      -> delivered      MDPS12 CR_Mdps_OutTq                 does the EPS follow?

THE MODEL-DOMAIN CHECK IS THE POINT. torque_from_lateral_accel divides by a single, speed
independent latAccelFactor, and selfdrive/locationd/torqued.py only ever LEARNS that number from

    vego > MIN_VEL (15 m/s)   and   abs(lateral_acc) <= LAT_ACC_THRESHOLD (1 m/s^2)

so every sample behind it comes from gentle highway curves. It is then applied at 4 m/s and at
3 m/s^2. If the car's true lateral-accel-per-unit-torque is lower down there -- tyre scrub,
steering friction, and the EPS's own speed-dependent boost curve all move -- then the controller
systematically under-commands, the P term has to make up the whole difference, and it saturates.

So we measure the gain empirically per speed band and compare it to the constant the controller
actually used. `empirical` is the median of actualLateralAccel / normalised torque over frames
where the command is big enough to be meaningful. If empirical << model at low speed, the loss is
the tune's domain, not the ceiling, and raising the ceiling by 10% recovers 10% of a bigger gap.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

MDPS12 = 0x251
LKAS11 = 0x340
POWERTRAIN_BUS = 0
OP_TX_SRC = 128
TORQUE_OFFSET = 1024

STEER_MAX = 409          # the flat ceiling these routes ran
# opendbc/car/hyundai/values.py, and panda enforces the same 3/7 via HYUNDAI_LIMITS(512, 3, 7),
# so neither can be moved without reflashing the other.
STEER_DELTA_UP = 3
# The driver window AS THE ARCHIVE RAN IT. This tool measures recorded drives, all of which
# predate the CN7 raise to 100, so 50 is what reproduces them; today's value would
# mis-model every one. Raise it only for drives recorded after that change.
STEER_DRIVER_ALLOWANCE = 50
PINNED_EPS = 0.995       # |actuators.torque| at or above this is "asking for everything"
GAIN_MIN_TORQUE = 0.15   # below this the ratio is dominated by noise and roll
INTEGRATOR_FREEZE_SPEED = 5.0   # latcontrol_torque*.py: freeze_integrator = ... or CS.vEgo < 5

SPEED_BANDS = ((0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 999.0))
BAND_NAMES = [f"{lo:g}-{hi:g}" if hi < 999 else f"{lo:g}+" for lo, hi in SPEED_BANDS]


def band_of(v: float):
    for (lo, hi), name in zip(SPEED_BANDS, BAND_NAMES, strict=False):
        if lo <= v < hi:
            return name
    return None


def decode_lkas11(dat: bytes):
    if len(dat) < 4:
        return None
    return ((int.from_bytes(dat[0:4], "little") >> 16) & 0x7FF) - TORQUE_OFFSET


def decode_mdps12(dat: bytes):
    if len(dat) < 8:
        return None
    w = int.from_bytes(dat[0:8], "little")
    return {
        "active": (w >> 13) & 1,
        "str_tq": ((w >> 40) & 0xFFF) * 0.01 - 20.48,
        "out_tq": ((w >> 52) & 0xFFF) * 0.1 - 204.8,
    }


def new_band():
    return {
        "frames": 0,
        "pinned_norm": 0,          # |actuators.torque| >= PINNED_EPS
        "sat_flag": 0,             # torqueState.saturated
        "int_frozen": 0,           # vEgo < 5
        "rate_limited": 0,         # counts applied < counts asked
        "counts_lost": [],         # how many counts the limiter/clip took
        "limiter_binding": 0,      # |d(out_can)| >= STEER_DELTA_UP: the slew rate itself, and
                                   # unlike counts_lost this needs no cross-message alignment
        "applied_when_pinned": [], # what actually reaches the car while asking for EVERYTHING
        "driver_opposing": 0,      # column torque past the allowance, where the clip can bind
        "demand": [],              # |desiredLateralAccel|
        "actual": [],              # |actualLateralAccel|
        "demand_when_pinned": [],
        "actual_when_pinned": [],
        # actualLateralAccel per unit of torque the car ACTUALLY RECEIVED (applied counts /
        # STEER_MAX), not per unit of what the controller asked for. Dividing by the ask is wrong
        # here and understates the car: at 3-7 m/s the rate limiter means the median applied is
        # 236 of a commanded 409, so the ask is not the input the plant saw.
        "gain_samples": [],
        # ...and the same ratio against the ASK, kept only to show the size of that error.
        "gain_vs_ask": [],
        "f_share": [], "p_share": [], "i_share": [],
        "driver_tq_when_pinned": [],
        "out_tq_by_cmd": defaultdict(list),
    }


def summarise(d: dict) -> dict:
    n = d["frames"]
    if not n:
        return {"frames": 0}

    def med(xs):
        return round(statistics.median(xs), 3) if xs else None

    def pct(k):
        return round(100.0 * d[k] / n, 3)

    dem, act = d["demand_when_pinned"], d["actual_when_pinned"]
    shortfall = None
    if dem and act:
        md, ma = statistics.median(dem), statistics.median(act)
        shortfall = round((md - ma) / md * 100.0, 1) if md > 1e-6 else None

    return {
        "frames": n,
        "pct_pinned_norm": pct("pinned_norm"),
        "pct_saturated_flag": pct("sat_flag"),
        "pct_integrator_frozen": pct("int_frozen"),
        "pct_rate_limited": pct("rate_limited"),
        "pct_limiter_binding": pct("limiter_binding"),
        "pct_driver_opposing_past_allowance": pct("driver_opposing"),
        "median_counts_lost_when_limited": med(d["counts_lost"]),
        # The headline for "is the car using the ceiling it already has". Asking for 1.0 means
        # asking for STEER_MAX; this is what the rate limiter and the driver clip actually let
        # through. If it sits well under STEER_MAX, raising STEER_MAX buys nothing.
        "applied_when_pinned": {
            "n": len(d["applied_when_pinned"]),
            "median": med(d["applied_when_pinned"]),
            "p90": (round(sorted(d["applied_when_pinned"])[int(len(d["applied_when_pinned"]) * 0.9)], 1)
                    if d["applied_when_pinned"] else None),
            "pct_reaching_ceiling": (
                round(100.0 * sum(1 for x in d["applied_when_pinned"] if x >= STEER_MAX - 1)
                      / len(d["applied_when_pinned"]), 1) if d["applied_when_pinned"] else None),
        },
        "median_demand_lataccel": med(d["demand"]),
        "median_actual_lataccel": med(d["actual"]),
        "when_pinned": {
            "n": len(dem),
            "median_demand": med(dem),
            "median_actual": med(act),
            "shortfall_pct": shortfall,
            "median_driver_torque_nm": med(d["driver_tq_when_pinned"]),
        },
        "empirical_gain_lataccel_per_unit_torque": {
            "n": len(d["gain_samples"]),
            "median": med(d["gain_samples"]),
            "median_vs_ask_DO_NOT_USE": med(d["gain_vs_ask"]),
        },
        "output_shares_when_pinned": {
            "f": med(d["f_share"]), "p": med(d["p_share"]), "i": med(d["i_share"]),
        },
        "delivered_nm_by_commanded_counts": {
            str(k): {"n": len(v), "median_nm": round(statistics.median(v), 2)}
            for k, v in sorted(d["out_tq_by_cmd"].items()) if len(v) >= 30
        },
    }


def scan(segs: list) -> dict:
    from openpilot.tools.lib.logreader import LogReader

    bands = {b: new_band() for b in BAND_NAMES}
    model_gain = None
    learner = {}
    versions = defaultdict(int)
    skipped = []

    for seg in segs:
        lat = False
        v = 0.0
        norm_tq = 0.0
        out_can = 0.0
        prev_out_can = 0.0
        driver_col_tq = 0.0
        ts = None
        last_cmd = 0
        try:
            for ev in LogReader(seg, sort_by_time=True):
                w = ev.which()
                if w == "carParams":
                    model_gain = round(float(ev.carParams.lateralTuning.torque.latAccelFactor), 4)
                # This build names it lateralTorqueParameters, not upstream's
                # liveTorqueParameters, and it is carried in the qlog rather than the rlog -- so
                # reading the rlog alone silently reports "no learner" on a car whose learner is
                # converged and in use (calPerc 100, useParams True).
                elif w == "lateralTorqueParameters":
                    p = ev.lateralTorqueParameters
                    learner = {
                        "latAccelFactorFiltered": round(float(p.latAccelFactorFiltered), 4),
                        "frictionCoefficientFiltered": round(float(p.frictionCoefficientFiltered), 4),
                        "latAccelOffsetFiltered": round(float(p.latAccelOffsetFiltered), 4),
                        "calPerc": int(p.calPerc),
                        "useParams": bool(p.useParams),
                    }
                elif w == "carState":
                    v = float(ev.carState.vEgo)
                    driver_col_tq = float(ev.carState.steeringTorque)
                elif w == "carOutput":
                    prev_out_can = out_can
                    out_can = float(ev.carOutput.actuatorsOutput.torqueOutputCan)
                elif w == "controlsState":
                    lcs = ev.controlsState.lateralControlState
                    ts = lcs.torqueState if lcs.which() == "torqueState" else None
                    if ts is not None:
                        versions[int(ts.version)] += 1
                elif w == "carControl":
                    cc = ev.carControl
                    lat = bool(cc.latActive)
                    norm_tq = float(cc.actuators.torque)
                    if not lat:
                        continue
                    b = band_of(v)
                    if b is None:
                        continue
                    d = bands[b]
                    d["frames"] += 1

                    pinned = abs(norm_tq) >= PINNED_EPS
                    if pinned:
                        d["pinned_norm"] += 1
                        d["applied_when_pinned"].append(abs(out_can))
                    if v < INTEGRATOR_FREEZE_SPEED:
                        d["int_frozen"] += 1
                    # The clip can only bind when the driver is pushing back past the allowance:
                    # max_steer_allowed = min(steer_max, steer_max + (50 + col_tq) * 2).
                    if driver_col_tq * (1 if norm_tq >= 0 else -1) < -STEER_DRIVER_ALLOWANCE:
                        d["driver_opposing"] += 1
                    # The slew rate binding on its own terms -- no cross-message alignment needed,
                    # so this is the number to trust if counts_lost and it ever disagree.
                    if abs(out_can - prev_out_can) >= STEER_DELTA_UP:
                        d["limiter_binding"] += 1

                    # What the carcontroller asked for before the rate limiter and the driver
                    # allowance clip, versus what actually went out.
                    asked = round(norm_tq * STEER_MAX)
                    if abs(asked) - abs(out_can) > 1:
                        d["rate_limited"] += 1
                        d["counts_lost"].append(abs(asked) - abs(out_can))

                    if ts is not None:
                        dem, act = abs(float(ts.desiredLateralAccel)), abs(float(ts.actualLateralAccel))
                        d["demand"].append(dem)
                        d["actual"].append(act)
                        if ts.saturated:
                            d["sat_flag"] += 1
                        if pinned:
                            d["demand_when_pinned"].append(dem)
                            d["actual_when_pinned"].append(act)
                            tot = abs(float(ts.f)) + abs(float(ts.p)) + abs(float(ts.i))
                            if tot > 1e-6:
                                d["f_share"].append(abs(float(ts.f)) / tot)
                                d["p_share"].append(abs(float(ts.p)) / tot)
                                d["i_share"].append(abs(float(ts.i)) / tot)
                        # Empirical lateral-accel-per-unit-torque, against what the EPS actually
                        # received. Compared against the single speed-independent latAccelFactor
                        # the controller divides by.
                        applied_norm = abs(out_can) / STEER_MAX
                        if applied_norm >= GAIN_MIN_TORQUE:
                            d["gain_samples"].append(act / applied_norm)
                        if abs(norm_tq) >= GAIN_MIN_TORQUE:
                            d["gain_vs_ask"].append(act / abs(norm_tq))
                elif w == "can":
                    for c in ev.can:
                        if c.address == LKAS11 and c.src >= OP_TX_SRC:
                            t = decode_lkas11(bytes(c.dat))
                            if t is not None:
                                last_cmd = t
                        elif c.address == MDPS12 and c.src == POWERTRAIN_BUS and lat:
                            m = decode_mdps12(bytes(c.dat))
                            if m is None:
                                continue
                            b = band_of(v)
                            if b is None:
                                continue
                            bands[b]["out_tq_by_cmd"][abs(last_cmd) // 25 * 25].append(abs(m["out_tq"]))
                            if abs(norm_tq) >= PINNED_EPS:
                                bands[b]["driver_tq_when_pinned"].append(abs(m["str_tq"]))
        except Exception as exc:
            skipped.append({"seg": os.path.basename(os.path.dirname(seg)),
                            "reason": type(exc).__name__ + ": " + str(exc)[:100]})

    return {
        "model_lat_accel_factor": model_gain,
        "live_torque_params": learner,
        "torque_controller_versions": dict(versions),
        "segments_skipped": skipped,
        "bands": {b: summarise(d) for b, d in bands.items()},
    }


def main() -> int:
    routes = sys.argv[1:]
    if not routes:
        print("usage: demand_decomp.py <route> [route ...]", file=sys.stderr)
        return 2
    segs = []
    for r in routes:
        segs += sorted(glob.glob(f"/data/media/0/realdata/{r}--*/rlog.zst"))
    print(f"# {len(segs)} segments across {len(routes)} routes", flush=True)

    rep = scan(segs)
    rep["routes"] = routes
    with open("/data/demand_decomp.json", "w") as f:
        json.dump(rep, f, indent=1)

    print(f"model latAccelFactor: {rep['model_lat_accel_factor']}   learner: {rep['live_torque_params']}")
    print(f"controller versions: {rep['torque_controller_versions']}\n")
    print(f"{'band':>8} {'frames':>8} {'pin%':>6} {'slew%':>6} {'lost':>5} " +
          f"{'applied|pinned':>14} {'@ceil%':>7} {'gain':>6} {'model':>6} {'froz%':>6} {'drv%':>5}")
    for b in BAND_NAMES:
        s = rep["bands"][b]
        if not s["frames"]:
            continue
        g, aw = s["empirical_gain_lataccel_per_unit_torque"], s["applied_when_pinned"]
        print(f"{b:>8} {s['frames']:>8} {s['pct_pinned_norm']:>6} {s['pct_limiter_binding']:>6} " +
              f"{str(s['median_counts_lost_when_limited']):>5} {str(aw['median']):>14} " +
              f"{str(aw['pct_reaching_ceiling']):>7} {str(g['median']):>6} " +
              f"{str(rep['model_lat_accel_factor']):>6} {s['pct_integrator_frozen']:>6} " +
              f"{s['pct_driver_opposing_past_allowance']:>5}")
    print("\n# wrote /data/demand_decomp.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
