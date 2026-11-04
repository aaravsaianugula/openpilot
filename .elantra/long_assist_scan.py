#!/usr/bin/env python3
"""What would Long Assist have done, and where did it disagree with the planner?

Runs on the device over /data/media/0/realdata (or the routes named on the command line) and reads
longitudinalPlanSP.longAssist, logged in every mode from shadow up. Per route it reports:

  * the state histogram and the share of frames with confidence >= C_ON;
  * BRAKE episodes: Long Assist asked for at least MARGIN m/s^2 more braking than the planner
    output. In shadow mode these are the late stops it exists to fix; in brake/full mode they
    should be gone, because its request is part of the planner's min();
  * LIFT episodes: it would have accelerated at least MARGIN harder than the planner (shadow),
    or did (full).

Episodes shorter than MIN_FRAMES are dropped so one noisy frame is not a finding. Output is
JSON at $LA_SCAN_OUT (default /data/long_assist_scan.json) plus one line per route.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

MARGIN = 0.3       # m/s^2
MIN_FRAMES = 4     # 0.2 s at 20 Hz
C_ON = 0.7


def episodes(frames: list[dict], key: str) -> list[dict]:
    """Group consecutive frames whose `key` is true into episodes of at least MIN_FRAMES."""
    out, run = [], []
    for f in frames + [{key: False}]:
        if f.get(key):
            run.append(f)
            continue
        if len(run) >= MIN_FRAMES:
            out.append({
                "t": round(run[0]["t"], 2),
                "dur": round(run[-1]["t"] - run[0]["t"], 2),
                "v": round(run[0]["v"], 1),
                "state": Counter(r["state"] for r in run).most_common(1)[0][0],
                "worst_gap": round(max(abs(r["la"] - r["plan"]) for r in run), 2),
                "stop_dist": round(run[0]["stop_dist"], 1),
            })
        run = []
    return out


def summarize(frames: list[dict]) -> dict:
    for f in frames:
        f["brake"] = f["brake_active"] and f["a_brake"] < f["plan"] - MARGIN
        # a zero lift under a braking plan is not a lift: apply() only raises e2e when a_lift > 0
        f["lift"] = not f["brake_active"] and f["lift_v"] > max(f["plan"], 0.) + MARGIN
    brake = episodes([dict(f, la=f["a_brake"]) for f in frames], "brake")
    lift = episodes([dict(f, la=f["lift_v"]) for f in frames], "lift")
    n = len(frames)
    return {
        "frames": n,
        "modes": dict(Counter(f["mode"] for f in frames)),
        "states": dict(Counter(f["state"] for f in frames)),
        "conf_on_share": round(sum(f["conf"] >= C_ON for f in frames) / n, 3) if n else 0.,
        "brake_episodes": brake,
        "lift_episodes": lift,
    }


def scan_route(segs: list[str]) -> dict:
    from openpilot.tools.lib.logreader import LogReader

    frames = []
    t0 = None
    v = a_plan = 0.
    for seg in segs:
        try:
            for ev in LogReader(seg, sort_by_time=True):
                w = ev.which()
                if w == "carState":
                    v = float(ev.carState.vEgo)
                elif w == "longitudinalPlan":
                    a_plan = float(ev.longitudinalPlan.aTarget)
                elif w == "longitudinalPlanSP":
                    la = ev.longitudinalPlanSP.longAssist
                    if la.mode == 0:
                        continue
                    t0 = ev.logMonoTime if t0 is None else t0
                    frames.append({
                        "t": (ev.logMonoTime - t0) * 1e-9, "v": v, "plan": a_plan, "mode": int(la.mode),
                        "state": str(la.state), "conf": float(la.confidence), "brake_active": bool(la.brakeActive),
                        "a_brake": float(la.aBrake), "lift_v": float(la.aLift), "stop_dist": float(la.stopDistance),
                    })
        except Exception as e:  # a corrupt or partial segment must not lose the rest of the route
            print(f"# skipped {seg}: {e}", file=sys.stderr)
    return summarize(frames)


def main() -> int:
    only = sys.argv[1:] or None
    by_route = defaultdict(list)
    for s in sorted(glob.glob("/data/media/0/realdata/*/rlog.zst")):
        by_route[os.path.basename(os.path.dirname(s)).split("--")[0]].append(s)
    routes = sorted(r for r in by_route if not only or r in only)

    out = {}
    for route in routes:
        rep = scan_route(by_route[route])
        if not rep["frames"]:
            continue
        out[route] = rep
        print(f"{route} frames={rep['frames']} modes={rep['modes']} conf_on={rep['conf_on_share']} " +
              f"brake_ep={len(rep['brake_episodes'])} lift_ep={len(rep['lift_episodes'])} states={rep['states']}", flush=True)

    dest = os.environ.get("LA_SCAN_OUT", "/data/long_assist_scan.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"# {len(out)} routes with Long Assist data; wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
