#!/usr/bin/env python3
"""What ceiling did each recorded route actually run, and did the EPS fault?

Two things here are new, and both exist because the previous pass got a misleading answer.

CEILING RECOVERY. A route's build label proves nothing and neither does its peak command: route
000000bd was the 500-count schedule build and topped out at 337, because the drive never asked
for enough to reach the ceiling. But STEER_MAX is not only a clip, it is a GAIN --
hyundai/carcontroller.py does

    new_torque = int(round(actuators.torque * self.params.STEER_MAX))

so on any frame where the rate limiter and the driver-allowance clip are both idle,

    torqueOutputCan / actuators.torque  ==  STEER_MAX

exactly. That recovers the ceiling a build was compiled with from a gentle drive that never came
near it. Limited frames are excluded rather than trusted: STEER_DELTA_DOWN holds the output up
while the command falls away under it, so a limited frame can read far ABOVE the real ceiling as
well as below. We report the histogram and the high percentile per speed band, so a SCHEDULED
ceiling shows up as a ramp across the bands and a flat one shows up as a flat line.

The companion number, ceiling_lower_bound, is a BOUND and not the value: asking for 1.0 asks for
STEER_MAX, but the rate limiter and the driver clip sit downstream and can hold the applied count
under it for a whole drive. Route 000000d8 asked for 1.0 and never got past 394 on a 409 build.

FAULT CHANNELS. The previous counter read CF_Mdps_ToiFlt (bit 14) alone and found 3 frames in
1.59M. But openpilot itself faults on `ToiUnavail != 0 or ToiFlt != 0` (hyundai/carstate.py), and
"the MDPS dropped out rather than delivering" is more naturally ToiActive going to 0. So we carry
every MDPS12 status bit, plus carState's own fault flags and the onroadEvents the driver actually
saw. A fault the driver reports and the log does not is a measurement bug, not an absence.

MDPS12 (0x251), from _hyundai_can_common.dbc, and ONLY on src 0 -- the same message appears on
src 1 carrying something else entirely, and accepting it invented 599 phantom faults per segment
against a true count of zero.

    CR_Mdps_StrColTq  0|11@1+ (1.0,-1024.0)     driver column torque, counts
    CF_Mdps_Def      11|1@1+
    CF_Mdps_ToiUnavail 12|1@1+
    CF_Mdps_ToiActive  13|1@1+                  1 = the MDPS is actuating
    CF_Mdps_ToiFlt     14|1@1+
    CF_Mdps_FailStat   15|1@1+
    CR_Mdps_StrTq    40|12@1+ (0.01,-20.48)     driver torque, Nm
    CR_Mdps_OutTq    52|12@1+ (0.1,-204.8)      what the EPS ACTUALLY DELIVERED

LKAS11 (0x340) CR_Lkas_StrToqReq 16|11@1+ (1.0,-1024.0), src >= 128 for openpilot's own TX.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

LKAS11 = 0x340
MDPS12 = 0x251
OP_TX_SRC = 128
POWERTRAIN_BUS = 0
TORQUE_OFFSET = 1024

# Ratio frames are only trustworthy when the normalised command is big enough that integer
# rounding of the count does not swamp it. At |torque| >= 0.5 the rounding error on the recovered
# ceiling is at most 1 count; below 0.2 it is worthless.
RATIO_MIN_TORQUE = 0.5

# ...and only when the rate limiter is idle. The first cut of this assumed a limited frame can
# only drag the ratio DOWN, which is false: STEER_DELTA_DOWN holds the OUTPUT UP while the
# normalised command falls away beneath it, and the ratio then reads far ABOVE the real ceiling.
# That is what made flat-409 route 000000c9 report a p99 of 502.6 and look like the 500 build.
# A frame that moved by less than the up-rate is one the limiter was not acting on.
STEER_DELTA_UP = 3

# ToiActive flickers for a frame or two either side of an engage/disengage boundary, which is not
# a dropout. Only count one after lateral has been continuously active for half a second.
DROPOUT_MIN_ENGAGED_FRAMES = 50

SPEED_BANDS = ((0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 999.0))
BAND_NAMES = [f"{lo:g}-{hi:g}" if hi < 999 else f"{lo:g}+" for lo, hi in SPEED_BANDS]

STEER_EVENTS = ("steerUnavailable", "steerTempUnavailable", "steerTempUnavailableSilent",
                "steerSaturated", "ldw")


def band_of(v: float):
    for (lo, hi), name in zip(SPEED_BANDS, BAND_NAMES, strict=False):
        if lo <= v < hi:
            return name
    return None


def decode_lkas11(dat: bytes):
    """(signed torque counts, CF_Lkas_ActToi).

    ActToi matters as much as the torque. openpilot DELIBERATELY drops it -- carcontroller.py's
    common_fault_avoidance cuts the bit for two frames out of every 89 when the wheel is past 85
    degrees, precisely to stop the EPS faulting. The MDPS answers a commanded cut by clearing
    CF_Mdps_ToiActive, which is indistinguishable from a real dropout unless you check who asked.
    """
    if len(dat) < 4:
        return None
    word = int.from_bytes(dat[0:4], "little")
    return ((word >> 16) & 0x7FF) - TORQUE_OFFSET, (word >> 27) & 1


def decode_mdps12(dat: bytes):
    """Every status bit and both torque signals, or None if the frame is short."""
    if len(dat) < 8:
        return None
    w = int.from_bytes(dat[0:8], "little")
    return {
        "col_tq": (w & 0x7FF) - TORQUE_OFFSET,
        "def": (w >> 11) & 1,
        "unavail": (w >> 12) & 1,
        "active": (w >> 13) & 1,
        "flt": (w >> 14) & 1,
        "failstat": (w >> 15) & 1,
        "str_tq": ((w >> 40) & 0xFFF) * 0.01 - 20.48,
        "out_tq": ((w >> 52) & 0xFFF) * 0.1 - 204.8,
    }


def provenance(seg0: str) -> dict:
    from openpilot.tools.lib.logreader import LogReader

    out = {"git_commit": None, "git_branch": None, "dirty": None, "fingerprint": None,
           "safety_param": None, "lat_accel_factor": None, "friction": None, "params": {}}
    want = {"initData", "carParams"}
    keys = ("NeuralNetworkLateralControl", "LateralJerkTorqueController", "EnforceTorqueControl",
            "TorqueControlTune", "LiveTorqueParamsToggle", "CustomTorqueParams")
    try:
        for ev in LogReader(seg0):
            w = ev.which()
            if w not in want:
                continue
            want.discard(w)
            if w == "initData":
                d = ev.initData
                out["git_commit"] = str(d.gitCommit)[:9]
                out["git_branch"] = str(d.gitBranch)
                out["dirty"] = bool(d.dirty)
                entries = {e.key: e.value for e in d.params.entries}
                for k in keys:
                    v = entries.get(k)
                    out["params"][k] = None if v is None else v.decode("utf-8", "replace")
            else:
                cp = ev.carParams
                out["fingerprint"] = str(cp.carFingerprint)
                if len(cp.safetyConfigs):
                    out["safety_param"] = int(cp.safetyConfigs[-1].safetyParam)
                out["lat_accel_factor"] = round(float(cp.lateralTuning.torque.latAccelFactor), 4)
                out["friction"] = round(float(cp.lateralTuning.torque.friction), 4)
            if not want:
                break
    except Exception as exc:
        out["error"] = type(exc).__name__ + ": " + str(exc)[:100]
    return out


def scan_route(route: str, segs: list) -> dict:
    from openpilot.tools.lib.logreader import LogReader

    engaged = 0
    max_cmd_can = 0          # from openpilot's own LKAS11 TX
    max_cmd_out = 0          # from carOutput, the same number one layer up
    max_norm_torque = 0.0    # peak |actuators.torque|; 1.0 makes max_cmd_out a LOWER BOUND on
                             # STEER_MAX, not the value -- see ceiling_lower_bound below
    above_409 = 0
    ratios = []                                  # (band, implied_steer_max)
    ratio_by_band = defaultdict(list)
    mdps_engaged = Counter()                     # bit -> frames set, while engaged
    mdps_any = Counter()                         # bit -> frames set, engaged or not
    onsets = []                                  # every 0->1 transition of a fault bit
    dropouts = []                                # ToiActive 1->0 while openpilot was steering
    uncommanded_dropouts = []                    # ...of those, the ones openpilot did NOT ask for
    cs_fault_engaged = Counter()
    events = Counter()
    versions = Counter()
    saturated = 0
    torque_state_frames = 0
    # delivered-vs-commanded, bucketed by commanded magnitude: the EPS boost curve
    deliver = defaultdict(list)
    skipped = []

    prev = {"unavail": 0, "flt": 0, "failstat": 0, "def": 0, "active": 1}

    for seg in segs:
        lat = False
        v = 0.0
        angle = 0.0
        out_can = 0.0
        prev_out_can = 0.0
        norm_tq = 0.0
        last_cmd = 0
        last_act_toi = 1
        engaged_run = 0
        try:
            for ev in LogReader(seg, sort_by_time=True):
                w = ev.which()
                if w == "carState":
                    cs = ev.carState
                    v = float(cs.vEgo)
                    angle = float(cs.steeringAngleDeg)
                    if lat:
                        if cs.steerFaultTemporary:
                            cs_fault_engaged["temporary"] += 1
                        if cs.steerFaultPermanent:
                            cs_fault_engaged["permanent"] += 1
                elif w == "carOutput":
                    prev_out_can = out_can
                    out_can = float(ev.carOutput.actuatorsOutput.torqueOutputCan)
                    if lat:
                        max_cmd_out = max(max_cmd_out, abs(out_can))
                elif w == "carControl":
                    cc = ev.carControl
                    lat = bool(cc.latActive)
                    norm_tq = float(cc.actuators.torque)
                    engaged_run = engaged_run + 1 if lat else 0
                    if lat:
                        engaged += 1
                        max_norm_torque = max(max_norm_torque, abs(norm_tq))
                        # Ceiling recovery. Needs a command big enough for integer rounding not to
                        # swamp it AND a frame the rate limiter was not acting on -- a limited
                        # frame reads high or low depending on which way the command was moving.
                        limiter_idle = abs(out_can - prev_out_can) < STEER_DELTA_UP
                        if abs(norm_tq) >= RATIO_MIN_TORQUE and out_can != 0.0 and limiter_idle:
                            r = abs(out_can) / abs(norm_tq)
                            b = band_of(v)
                            ratios.append(r)
                            if b:
                                ratio_by_band[b].append(r)
                elif w == "controlsState":
                    lcs = ev.controlsState.lateralControlState
                    if lcs.which() == "torqueState":
                        ts = lcs.torqueState
                        torque_state_frames += 1
                        versions[int(ts.version)] += 1
                        if ts.saturated:
                            saturated += 1
                elif w == "onroadEvents":
                    for e in ev.onroadEvents:
                        n = str(e.name)
                        if n in STEER_EVENTS:
                            events[n] += 1
                elif w == "can":
                    for c in ev.can:
                        if c.address == LKAS11 and c.src >= OP_TX_SRC:
                            dec = decode_lkas11(bytes(c.dat))
                            if dec is None:
                                continue
                            t, last_act_toi = dec
                            last_cmd = t
                            if lat:
                                a = abs(t)
                                max_cmd_can = max(max_cmd_can, a)
                                if a > 409:
                                    above_409 += 1
                        elif c.address == MDPS12 and c.src == POWERTRAIN_BUS:
                            m = decode_mdps12(bytes(c.dat))
                            if m is None:
                                continue
                            for bit in ("def", "unavail", "flt", "failstat"):
                                if m[bit]:
                                    mdps_any[bit] += 1
                                    if lat:
                                        mdps_engaged[bit] += 1
                                if m[bit] and not prev[bit]:
                                    onsets.append({
                                        "seg": os.path.basename(os.path.dirname(seg)),
                                        "bit": bit, "engaged": lat, "v_ego": round(v, 2),
                                        "angle_deg": round(angle, 1), "cmd": last_cmd,
                                        "out_tq": round(m["out_tq"], 1),
                                        "driver_tq": round(m["str_tq"], 2),
                                        "col_tq": m["col_tq"], "toi_active": m["active"],
                                    })
                            if (lat and prev["active"] and not m["active"]
                                    and engaged_run >= DROPOUT_MIN_ENGAGED_FRAMES):
                                # act_toi 0 means OPENPILOT asked the MDPS to stop actuating, so
                                # ToiActive falling is the commanded answer and not a fault at
                                # all. Only act_toi 1 here is the MDPS letting go on its own.
                                rec = {
                                    "seg": os.path.basename(os.path.dirname(seg)),
                                    "v_ego": round(v, 2), "angle_deg": round(angle, 1),
                                    "cmd": last_cmd, "out_tq": round(m["out_tq"], 1),
                                    "driver_tq": round(m["str_tq"], 2),
                                    "unavail": m["unavail"], "flt": m["flt"],
                                    "act_toi": last_act_toi,
                                }
                                dropouts.append(rec)
                                if last_act_toi:
                                    uncommanded_dropouts.append(rec)
                            if lat and m["active"] and abs(last_cmd) > 0:
                                deliver[abs(last_cmd) // 25 * 25].append(abs(m["out_tq"]))
                            prev = {k: m[k] for k in prev}
        except Exception as exc:
            skipped.append({"seg": os.path.basename(os.path.dirname(seg)),
                            "reason": type(exc).__name__ + ": " + str(exc)[:100]})

    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(len(s) * p))], 1)

    return {
        "route": route,
        "segments": len(segs),
        "segments_skipped": skipped,
        "engaged_frames": engaged,
        "max_cmd_can": max_cmd_can,
        "max_cmd_out": round(max_cmd_out, 1),
        "max_norm_torque": round(max_norm_torque, 4),
        # A LOWER BOUND on STEER_MAX, not the value. Asking for 1.0 asks for STEER_MAX, but the
        # rate limiter and the driver-allowance clip both sit downstream and can hold the applied
        # count under it for the whole drive -- route 000000d8 asked for 1.0 and never got past
        # 394 on a 409 build. So this proves "the ceiling was AT LEAST this", which is exactly
        # what is needed to prove a route ran raised, and never proves a ceiling was 409.
        "ceiling_lower_bound": round(max_cmd_out) if max_norm_torque >= 0.995 else None,
        "frames_above_409": above_409,
        "implied_steer_max": {
            "n": len(ratios),
            "p50": pct(ratios, 0.50), "p95": pct(ratios, 0.95), "p99": pct(ratios, 0.99),
            "max": round(max(ratios), 1) if ratios else None,
            "mode": Counter(round(r) for r in ratios).most_common(3),
        },
        "implied_by_band": {b: {"n": len(xs), "p95": pct(xs, 0.95), "max": round(max(xs), 1)}
                            for b, xs in sorted(ratio_by_band.items()) if xs},
        "mdps_bits_engaged": dict(mdps_engaged),
        "mdps_bits_any": dict(mdps_any),
        "fault_onsets": onsets[:80],
        "fault_onsets_total": len(onsets),
        "toi_dropouts": dropouts[:80],
        "toi_dropouts_total": len(dropouts),
        "uncommanded_dropouts": uncommanded_dropouts[:80],
        "uncommanded_dropouts_total": len(uncommanded_dropouts),
        "carstate_faults_engaged": dict(cs_fault_engaged),
        "onroad_events": dict(events),
        "torque_versions": dict(versions),
        "torque_state_frames": torque_state_frames,
        "saturated_frames": saturated,
        "delivered_curve": {str(k): {"n": len(xs), "median_nm": round(statistics.median(xs), 1)}
                            for k, xs in sorted(deliver.items()) if len(xs) >= 20},
    }


def main() -> int:
    only = sys.argv[1:] or None
    segs = sorted(glob.glob("/data/media/0/realdata/*/rlog.zst"))
    by_route = defaultdict(list)
    for s in segs:
        by_route[os.path.basename(os.path.dirname(s)).split("--")[0]].append(s)

    routes = sorted(r for r in by_route if not only or r in only)
    print(f"# {len(segs)} segments, {len(by_route)} routes; scanning {len(routes)}", flush=True)

    out = []
    for route in routes:
        prov = provenance(by_route[route][0])
        rep = scan_route(route, by_route[route])
        rep["provenance"] = prov
        out.append(rep)
        ism = rep["implied_steer_max"]
        print(f"{route} segs={rep['segments']:>3} eng={rep['engaged_frames']:>7} " +
              f"maxcmd={rep['max_cmd_can']:>4} >409={rep['frames_above_409']:>6} " +
              f"CEIL={rep['ceiling_lower_bound']} (ntq={rep['max_norm_torque']}) " +
              f"ratio_p95={ism['p95']} mode={ism['mode'][:1]} " +
              f"mdps={dict(rep['mdps_bits_engaged'])} " +
              f"drop={rep['uncommanded_dropouts_total']}/{rep['toi_dropouts_total']} " +
              f"ev={dict(rep['onroad_events'])} commit={prov['git_commit']} " +
              f"sp={prov['safety_param']}", flush=True)

    dest = os.environ.get("CENSUS_OUT", "/data/eps_census.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n# wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
