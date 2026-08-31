#!/usr/bin/env python3
"""
Lateral authority report for the CN7 Elantra -- where the car runs out of steering torque.

Runs on the comma device against recorded routes:

    PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 .elantra/lateral_report.py scan
    PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 .elantra/lateral_report.py compare <before> <after>

Three things here are deliberate, because each one has already produced a wrong answer:

1. TWO INDEPENDENT ESTIMATORS. Pinning is computed twice from different sources -- once from
   what openpilot recorded it commanded (carOutput/carControl), once by decoding the LKAS11
   frames it actually transmitted (src >= 128, addr 0x340). They are never averaged. If they
   disagree beyond a frame-timing tolerance, that is a finding.

2. PROVENANCE, AND A REFUSAL TO MIX IT. Every route is tagged with the git commit, the
   safetyParam, the offline torque seed and the lateral params in force when it was recorded
   -- all of which segment 0 already carries. Routes with different tags can never land on
   the same side of a before/after. A previous campaign reported a "baseline" of 151 segments
   against an "after" of 150 and called it a comparison; the populations were simply different.

3. NO SILENT DROPS. Every segment that cannot be read is counted and named, and `compare`
   refuses outright if either side lost one. The same previous campaign lost segments to two
   uncounted early returns, which moved the denominator without moving the headline.

Medians are binned by speed, never fitted. The CN7 torque -> lateral-accel response is convex
below ~10 m/s, and a straight line through it reports a slope the car does not have.

It also watches, on every route and for free, for the one number nobody has ever measured: a
non-zero steering torque request from the FACTORY camera on bus 2. While openpilot is engaged
the camera never actuates, so this normally reads zero -- but if a genuinely passive window
ever occurs, this catches it without anyone having to switch something on.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

# --- CAN, straight off the wire -------------------------------------------------------
# hyundai_can.dbc: BO_ 832 LKAS11, SG_ CR_Lkas_StrToqReq : 16|11@1+ (1.0,-1024.0)
#                                  SG_ CF_Lkas_ActToi    : 27|1@1+
#                  BO_ 593 MDPS12, SG_ CF_Mdps_ToiFlt    : 14|1@1+
LKAS11 = 0x340
MDPS12 = 0x251
TORQUE_OFFSET = 1024

# src >= 128 is a frame openpilot transmitted; src 0/1/2 is one it received on that bus.
OP_TX_SRC = 128
POWERTRAIN_BUS = 0
CAMERA_BUS = 2

# Address filters must pin the bus, not just "anything openpilot did not send". Measured on
# this car: 0x251 arrives on src 0 (the real MDPS, 6000 frames in a segment), src 130
# (openpilot forwarding it to the camera) and src 1 (1200 frames of something else entirely
# -- 599 of them have bit 14 set). Accepting every src < 128 turned that bus-1 traffic into
# 599 phantom EPS faults per segment, against a true count of zero confirmed by CANParser.

SPEED_BANDS = ((0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 999.0))
BAND_NAMES = [f"{lo:g}-{hi:g}" if hi < 999 else f"{lo:g}+" for lo, hi in SPEED_BANDS]

# Reported at each threshold rather than against one hardcoded ceiling, so a before/after
# across a schedule change stays interpretable on both sides.
THRESHOLDS = (384, 409)

# The two estimators watch the same drive through different messages, so they differ by a few
# frames of timing. Past this it is a defect in one of them, not noise.
ESTIMATOR_TOLERANCE = 0.02
# The absolute tolerance above is 2 percentage points, several times the headline pinned
# rate this tool reports -- on its own it can never fail. The relative bound is what makes
# the cross-check an actual check.
ESTIMATOR_REL_TOLERANCE = 0.25
UNREADABLE_FILE = "_unreadable.json"

# Params that change lateral behaviour. Part of the provenance tag: a route recorded with NNLC
# on is not comparable to one recorded with it off, and must never share a bin.
LATERAL_PARAMS = (
    "NeuralNetworkLateralControl",
    "LateralJerkTorqueController",
    "EnforceTorqueControl",
    "TorqueControlTune",
    "LiveTorqueParamsToggle",
    "LiveTorqueParamsRelaxedToggle",
    "CustomTorqueParams",
    "TorqueParamsOverrideEnabled",
    "TorqueParamsOverrideLatAccelFactor",
    "TorqueParamsOverrideFriction",
)


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


def decode_toiflt(dat: bytes):
    if len(dat) < 2:
        return None
    return (int.from_bytes(dat[0:2], "little") >> 14) & 1


def is_eps_fault_frame(address: int, src: int, dat: bytes) -> bool:
    """Is this frame the real MDPS reporting a steering fault?

    The bus matters as much as the address. 0x251 shows up on src 0 (the MDPS itself), src 130
    (openpilot forwarding it to the camera) and src 1, which on this car carries something else
    entirely -- 599 of its 1200 frames per segment have bit 14 set. Accepting any src < 128
    reported those as EPS faults against a true count of zero, confirmed against CANParser.
    """
    if address != MDPS12 or src != POWERTRAIN_BUS:
        return False
    return decode_toiflt(dat) == 1


def new_band() -> dict:
    return {
        "frames": 0,
        "pinned": {str(t): 0 for t in THRESHOLDS},
        "max_abs": 0,
        "demand": [],
        "delivered": [],
        # toi_flt counts EVERY MDPS fault frame in the band, engaged or not. That makes it
        # useless for comparing two builds: a route where openpilot never engaged still
        # accumulates faults, and dividing by engaged frames then invents a rate. Keep it for
        # continuity, but toi_flt_engaged is the one that can be compared.
        "toi_flt": 0,
        "toi_flt_engaged": 0,
    }


def finalize(store: dict) -> dict:
    for d in store.values():
        d["demand_median"] = statistics.median(d["demand"]) if d["demand"] else None
        d["delivered_median"] = statistics.median(d["delivered"]) if d["delivered"] else None
        del d["demand"], d["delivered"]
    return store


def provenance(seg0: str) -> dict:
    """Read the route's identity out of its own first segment.

    Never from the live device: today's params say nothing about what was in force when this
    drive was recorded, and that confusion is exactly what a provenance tag exists to prevent.
    """
    from openpilot.tools.lib.logreader import LogReader

    out: dict = {
        "git_commit": None, "git_branch": None, "dirty": None,
        "fingerprint": None, "safety_param": None,
        "lat_accel_factor": None, "friction": None,
        "nnlc_model": None, "nnlc_fuzzy": None, "params": {},
    }
    want = {"initData", "carParams", "carParamsSP"}
    for ev in LogReader(seg0):
        w = ev.which()
        if w not in want:
            continue
        want.discard(w)
        if w == "initData":
            d = ev.initData
            out["git_commit"] = str(d.gitCommit)
            out["git_branch"] = str(d.gitBranch)
            out["dirty"] = bool(d.dirty)
            entries = {e.key: e.value for e in d.params.entries}
            for k in LATERAL_PARAMS:
                v = entries.get(k)
                out["params"][k] = None if v is None else v.decode("utf-8", "replace")
        elif w == "carParams":
            cp = ev.carParams
            out["fingerprint"] = str(cp.carFingerprint)
            if len(cp.safetyConfigs):
                out["safety_param"] = int(cp.safetyConfigs[-1].safetyParam)
            out["lat_accel_factor"] = round(float(cp.lateralTuning.torque.latAccelFactor), 6)
            out["friction"] = round(float(cp.lateralTuning.torque.friction), 6)
        else:
            n = ev.carParamsSP.neuralNetworkLateralControl
            out["nnlc_model"] = str(n.model.name)
            out["nnlc_fuzzy"] = bool(n.fuzzyFingerprint)
        if not want:
            break
    return out


def tag_of(prov: dict) -> str:
    """A short stable key for "the car was configured like this"."""
    material = json.dumps(
        {
            "git_commit": prov["git_commit"],
            "safety_param": prov["safety_param"],
            "fingerprint": prov["fingerprint"],
            "lat_accel_factor": prov["lat_accel_factor"],
            "friction": prov["friction"],
            "nnlc_model": prov["nnlc_model"],
            "params": prov["params"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def scan_route(route: str, segs: list) -> dict:
    from openpilot.tools.lib.logreader import LogReader

    A = {b: new_band() for b in BAND_NAMES}   # estimator A: carOutput + carControl
    B = {b: new_band() for b in BAND_NAMES}   # estimator B: transmitted LKAS11 + carState
    factory = {"frames": 0, "max_abs_torque": 0, "act_toi_frames": 0}
    op_tx_frames = 0
    skipped: list = []
    engaged_a = engaged_b = 0

    for seg in segs:
        lat = False
        v = 0.0
        out_can = 0
        try:
            for ev in LogReader(seg):
                w = ev.which()
                if w == "carState":
                    v = float(ev.carState.vEgo)
                elif w == "carOutput":
                    out_can = int(ev.carOutput.actuatorsOutput.torqueOutputCan)
                elif w == "carControl":
                    cc = ev.carControl
                    lat = bool(cc.latActive)
                    if not lat:
                        continue
                    band = band_of(v)
                    if band is None:
                        continue
                    engaged_a += 1
                    d = A[band]
                    d["frames"] += 1
                    a = abs(out_can)
                    d["max_abs"] = max(d["max_abs"], a)
                    for t in THRESHOLDS:
                        if a >= t:
                            d["pinned"][str(t)] += 1
                    if a >= THRESHOLDS[0]:
                        d["demand"].append(abs(float(cc.actuators.curvature)) * v * v)
                        d["delivered"].append(abs(float(cc.currentCurvature)) * v * v)
                elif w == "can":
                    for c in ev.can:
                        if c.address == LKAS11:
                            dec = decode_lkas11(bytes(c.dat))
                            if dec is None:
                                continue
                            tq, toi = dec
                            if c.src >= OP_TX_SRC:
                                op_tx_frames += 1
                                if not lat:
                                    continue
                                band = band_of(v)
                                if band is None:
                                    continue
                                engaged_b += 1
                                d = B[band]
                                d["frames"] += 1
                                a = abs(tq)
                                d["max_abs"] = max(d["max_abs"], a)
                                for t in THRESHOLDS:
                                    if a >= t:
                                        d["pinned"][str(t)] += 1
                            elif c.src == CAMERA_BUS:
                                factory["frames"] += 1
                                factory["max_abs_torque"] = max(factory["max_abs_torque"], abs(tq))
                                factory["act_toi_frames"] += toi
                        elif is_eps_fault_frame(c.address, c.src, bytes(c.dat)):
                            band = band_of(v)
                            if band is not None:
                                A[band]["toi_flt"] += 1
                                if lat:
                                    A[band]["toi_flt_engaged"] += 1
        except Exception as exc:
            skipped.append({
                "segment": os.path.basename(os.path.dirname(seg)),
                "reason": type(exc).__name__ + ": " + str(exc)[:120],
            })

    return {
        "route": route,
        "segments_total": len(segs),
        "segments_read": len(segs) - len(skipped),
        "segments_skipped": skipped,
        "engaged_frames": {"estimator_a": engaged_a, "estimator_b": engaged_b},
        "bands": {"estimator_a": finalize(A), "estimator_b": finalize(B)},
        # A drive with zero openpilot LKAS11 transmissions is a genuinely passive drive, and
        # the only kind in which the factory numbers below mean anything.
        "op_lkas11_tx_frames": op_tx_frames,
        "passive_drive": op_tx_frames == 0,
        "factory_lkas": factory,
    }


def routes_under(root: Path) -> dict:
    out: dict = {}
    for path in sorted(glob.glob(str(root / "*" / "rlog.zst"))):
        seg_dir = os.path.basename(os.path.dirname(path))
        if "--" not in seg_dir:
            continue
        route, _, idx = seg_dir.rpartition("--")
        if not idx.isdigit():
            continue
        # Reports are written as <route>.json alongside the leading-underscore marker files,
        # and cmd_compare filters those out by prefix. A route starting with "_" would have its
        # report silently excluded from every comparison. Real dongle IDs are lowercase hex and
        # cannot start with one, so refusing here costs nothing and closes the ambiguity.
        if route.startswith("_"):
            continue
        out.setdefault(route, []).append(path)
    for segs in out.values():
        segs.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).rpartition("--")[2]))
    return out


def cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.routes)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    routes = routes_under(root)
    if not routes:
        print("no routes under " + str(root))
        return 1

    unreadable: dict = {}
    names = sorted(routes)
    if args.limit:
        names = names[-args.limit:]
    print(str(len(routes)) + " route(s) found, analysing " + str(len(names)))

    for i, route in enumerate(names, 1):
        dest = outdir / (route + ".json")
        head = "  [" + str(i) + "/" + str(len(names)) + "] " + route
        if dest.exists() and not args.force:
            print(head + "  (cached)")
            continue
        segs = routes[route]
        try:
            prov = provenance(segs[0])
        except Exception as exc:
            # Counted, not skipped. A route that vanished here used to leave every denominator
            # silently -- the exact failure this file's docstring claims to prevent. It cannot be
            # attributed to a tag either (the tag comes from the provenance we just failed to
            # read), so `compare` refuses outright rather than guessing which side it belonged to.
            print(head + "  UNREADABLE segment 0 (" + type(exc).__name__ + ") -- recorded")
            unreadable[route] = type(exc).__name__
            continue
        report = scan_route(route, segs)
        report["provenance"] = prov
        report["tag"] = tag_of(prov)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        note = ""
        if report["passive_drive"]:
            note += "  PASSIVE-DRIVE"
        if report["factory_lkas"]["max_abs_torque"]:
            note += "  FACTORY-TORQUE=" + str(report["factory_lkas"]["max_abs_torque"])
        print(head + "  tag=" + report["tag"]
              + "  engaged=" + str(report["engaged_frames"]["estimator_a"])
              + "  lost=" + str(len(report["segments_skipped"])) + note)

    marker = outdir / UNREADABLE_FILE
    if unreadable:
        marker.write_text(json.dumps(unreadable, indent=2), encoding="utf-8")
        print(str(len(unreadable)) + " route(s) could not be read; recorded in " + marker.name)
    elif marker.exists():
        marker.unlink()
    return 0


def merge(reports: list, estimator: str) -> dict:
    merged = {b: new_band() for b in BAND_NAMES}
    for r in reports:
        for b, src in r["bands"][estimator].items():
            d = merged[b]
            d["frames"] += src["frames"]
            for t in THRESHOLDS:
                d["pinned"][str(t)] += src["pinned"][str(t)]
            d["max_abs"] = max(d["max_abs"], src["max_abs"])
            d["toi_flt"] += src["toi_flt"]
            d["toi_flt_engaged"] += src.get("toi_flt_engaged", 0)
            for key in ("demand", "delivered"):
                m = src.get(key + "_median")
                if m is not None:
                    d[key].append(m)
    return finalize(merged)


def print_table(title: str, merged: dict) -> None:
    print("\n" + title)
    print(f"  {'speed':<8s} {'frames':>8s} {'>=384':>8s} {'%':>7s} {'>=409':>8s} " +
          f"{'max':>6s} {'demand':>8s} {'deliv':>8s} {'ratio':>7s} {'Flt/eng':>9s} {'Flt/all':>8s}")
    for b in BAND_NAMES:
        d = merged[b]
        if not d["frames"]:
            continue
        p384 = d["pinned"]["384"]
        dem, dlv = d["demand_median"], d["delivered_median"]
        print("  {:<8s} {:8d} {:8d} {:6.1f}% {:8d} {:6d} {!s:>8} {!s:>8} {!s:>7} {:9d} {:8d}".format(
            b, d["frames"], p384, 100.0 * p384 / d["frames"], d["pinned"]["409"], d["max_abs"],
            (f"{dem:8.3f}") if dem else "-",
            (f"{dlv:8.3f}") if dlv else "-",
            ("%7.3f" % (dem / dlv)) if dem and dlv else "-",
            d.get("toi_flt_engaged", 0), d["toi_flt"]))


def summarise(label: str, reports: list, failures: list) -> dict:
    lost = sum(len(r["segments_skipped"]) for r in reports)
    total = sum(r["segments_total"] for r in reports)
    read = sum(r["segments_read"] for r in reports)
    print("\n[" + label + "] routes=" + str(len(reports))
          + "  segments " + str(read) + "/" + str(total) + " read")
    if lost:
        for r in reports:
            for s in r["segments_skipped"]:
                print("    lost " + s["segment"] + ": " + s["reason"])
        failures.append(label + ": " + str(lost) + " segment(s) could not be read")

    a, b = merge(reports, "estimator_a"), merge(reports, "estimator_b")
    fa = sum(d["frames"] for d in a.values())
    fb = sum(d["frames"] for d in b.values())
    pa = sum(d["pinned"]["384"] for d in a.values())
    pb = sum(d["pinned"]["384"] for d in b.values())
    ra = pa / fa if fa else 0.0
    rb = pb / fb if fb else 0.0
    print(f"  estimator A (carOutput): {fa:8d} frames, pinned>=384 {pa} ({100 * ra:.2f}%)")
    print(f"  estimator B (LKAS11 TX): {fb:8d} frames, pinned>=384 {pb} ({100 * rb:.2f}%)")
    if fa == 0 or fb == 0:
        # Zero frames makes both rates 0.0, which compared equal and printed AGREE over an empty
        # table. Nothing was measured; that is a failure, not a clean result.
        print("  independent estimators NOT COMPARED -- no engaged frames " +
              "(A=" + str(fa) + ", B=" + str(fb) + ")")
        failures.append(label + ": no engaged frames on one or both estimators -- " +
                        "nothing was measured, this is not a clean run")
    else:
        rel = abs(ra - rb) / max(ra, rb) if max(ra, rb) > 0 else 0.0
        agree = abs(ra - rb) <= ESTIMATOR_TOLERANCE and rel <= ESTIMATOR_REL_TOLERANCE
        verdict = "AGREE" if agree else "DISAGREE"
        print(f"  independent estimators {verdict} (delta {100 * abs(ra - rb):.2f} pp / " +
              f"{100 * rel:.0f}% rel; tol {100 * ESTIMATOR_TOLERANCE:.0f} pp and " +
              f"{100 * ESTIMATOR_REL_TOLERANCE:.0f}%)")
        if not agree:
            failures.append(label + ": estimators disagree -- one of them is wrong, " +
                                    "do not average them")
    print_table("[" + label + "] estimator A, binned medians", a)
    return a


def cmd_compare(args: argparse.Namespace) -> int:
    outdir = Path(args.out)
    reports = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(outdir.glob("*.json")) if not p.name.startswith("_")]
    if not reports:
        print("no reports in " + str(outdir) + " -- run `scan` first")
        return 1

    by_tag: dict = {}
    for r in reports:
        by_tag.setdefault(r["tag"], []).append(r)

    if args.before is None or args.after is None:
        print("available tags:\n")
        for tag, rs in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
            p = rs[0]["provenance"]
            print("  {}  routes={:<4d} branch={!s:<10} param={!s:<5} laf={!s:<7} fric={!s:<8} NNLC={}".format(
                     tag, len(rs), p["git_branch"], p["safety_param"],
                     p["lat_accel_factor"], p["friction"],
                     p["params"].get("NeuralNetworkLateralControl")))
        print("\nPass two tags to compare them.")
        return 0

    for tag in (args.before, args.after):
        if tag not in by_tag:
            print("unknown tag: " + tag)
            return 1
    if args.before == args.after:
        print("before and after are the same tag -- nothing to compare")
        return 1

    # A mixed bin is impossible by construction: the tag IS the provenance. What is still
    # possible, and was the actual failure last time, is silently losing segments.
    failures: list = []
    summarise("before " + args.before, by_tag[args.before], failures)
    summarise("after  " + args.after, by_tag[args.after], failures)

    print("\n" + "-" * 60)
    marker = outdir / UNREADABLE_FILE
    if marker.exists():
        for route, why in json.loads(marker.read_text(encoding="utf-8")).items():
            failures.append("route " + route + " was unreadable (" + why + ") and cannot be " +
                            "attributed to either side")

    if failures:
        print("COMPARISON REFUSED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("Both sides clean: no lost segments, independent estimators agree.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="analyse routes and cache one JSON report each")
    s.add_argument("--routes", default="/data/media/0/realdata")
    s.add_argument("--out", default="/data/elantra-lateral-reports")
    s.add_argument("--limit", type=int, default=0, help="only the N most recent routes")
    s.add_argument("--force", action="store_true", help="re-analyse cached routes")
    s.set_defaults(func=cmd_scan)

    c = sub.add_parser("compare", help="before/after across two provenance tags")
    c.add_argument("before", nargs="?")
    c.add_argument("after", nargs="?")
    c.add_argument("--out", default="/data/elantra-lateral-reports")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
