#!/usr/bin/env python3
"""Replay Long Assist over recorded drives: what would it have done, without driving?

Feeds each route's own modelV2 / radarState / carState / carControl / longitudinalPlanSP back
through LongAssist in shadow mode, frame by frame in the order plannerd saw them, and compares its
brake/lift requests with the planner output the car actually ran. Reuses long_assist_scan's
episode logic, then labels every brake episode with what happened next:

  stopped   the car got below 1 m/s within STOP_WINDOW s -- a real stop it may have been late for;
  slowed    it did not stop but lost >= 30% of its speed -- a slow lead or a curve;
  phantom   neither -- Long Assist would have braked for nothing. These are the ones to tune out.

Also reports how often the radard lead was radar-matched (camera SCC11 point) vs vision-only; which
planner candidate was winning during each lift episode (a lift only reaches the car when the model's
e2e candidate is the one holding the output down); and exec_gap, the median of a_ego 0.5 s later minus
the planner's braking target, i.e. how much less the car actually brakes than it is told to.

    python .elantra/long_assist_replay.py [route ...]       # on the device, newest 10 by default
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import long_assist_scan as las

STOP_WINDOW = 15.
EXEC_DELAY_FRAMES = 10   # 0.5 s, the actuator delay
DEFAULT_ROUTES = 10


def label(ep: dict, frames: list[dict]) -> str:
  after = [f["v"] for f in frames if ep["t"] <= f["t"] <= ep["t"] + STOP_WINDOW]
  if not after:
    return "phantom"
  if min(after) < 1.:
    return "stopped"
  return "slowed" if min(after) <= 0.7 * after[0] else "phantom"


def replay_route(segs: list[str]) -> dict:
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.common.constants import CV
  from openpilot.sunnypilot.selfdrive.controls.lib.long_assist import LongAssist, Mode
  from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY, MapContext

  la = LongAssist()
  la.mode = Mode.shadow
  # hazards replay from the logged GPS; mapd's curve targets are a mem param and are not logged
  map_ctx = MapContext()
  gps = None
  primed = 0
  latest: dict = {}
  frames = []
  t0 = None
  leads = radar_leads = 0
  for seg in segs:
    try:
      for ev in LogReader(seg, sort_by_time=True):
        w = ev.which()
        if w in ("gpsLocation", "gpsLocationExternal"):
          gps = getattr(ev, w)
          continue
        if w in ("modelV2", "radarState", "carState", "carControl", "longitudinalPlanSP"):
          latest[w] = getattr(ev, w)
          continue
        # plannerd publishes longitudinalPlan right after it consumes a modelV2
        if w != "longitudinalPlan" or len(latest) < 5:
          continue
        cs, cc, sp, lead = latest["carState"], latest["carControl"], latest["longitudinalPlanSP"], latest["radarState"].leadOne
        enabled = cc.enabled and not cc.cruiseControl.override
        v_cruise = sp.vTarget if sp.vTarget > 0 else cs.vCruise * CV.KPH_TO_MS
        res = sp.speedLimit.resolver
        v_limit = res.speedLimitFinal if res.speedLimitValid else 0.
        map_ahead = map_ctx.ahead(gps.latitude, gps.longitude, gps.bearingDeg, gps.speed, gps.hasFix) if gps else EMPTY
        la.update(latest["modelV2"], lead, cs, enabled, v_cruise, v_limit, True, map_ahead)
        if not enabled:
          continue
        primed += map_ahead.hazard_dist >= 0.
        leads += lead.present
        radar_leads += lead.present and lead.radar
        t0 = ev.logMonoTime if t0 is None else t0
        frames.append({
          "t": (ev.logMonoTime - t0) * 1e-9, "v": float(cs.vEgo), "plan": float(ev.longitudinalPlan.aTarget),
          "mode": 1, "state": str(la.state), "conf": float(la.confidence), "brake_active": la.a_brake is not None,
          "a_brake": float(la.a_brake or 0.), "lift_v": float(la.a_lift), "stop_dist": float(la.stop_dist),
          "a_ego": float(cs.aEgo), "src": str(ev.longitudinalPlan.longitudinalPlanSource),
        })
    except Exception as e:  # a corrupt or partial segment must not lose the rest of the route
      print(f"# skipped {seg}: {e}", file=sys.stderr)

  rep = las.summarize(frames)
  for ep in rep["brake_episodes"]:
    ep["outcome"] = label(ep, frames)
  for ep in rep["lift_episodes"]:
    srcs = [f["src"] for f in frames if ep["t"] <= f["t"] <= ep["t"] + ep["dur"]]
    ep["src"] = max(set(srcs), key=srcs.count) if srcs else ""
  gaps = [frames[i + EXEC_DELAY_FRAMES]["a_ego"] - f["plan"] for i, f in enumerate(frames[:-EXEC_DELAY_FRAMES])
          if f["plan"] < -1.]
  rep["exec_gap"] = round(float(sorted(gaps)[len(gaps) // 2]), 2) if gaps else 0.
  rep["lead_frames"] = leads
  rep["radar_lead_share"] = round(radar_leads / leads, 3) if leads else 0.
  rep["hazards_loaded"] = map_ctx.hazards is not None
  rep["primed_share"] = round(primed / len(frames), 3) if frames else 0.
  return rep


def main() -> int:
  by_route = defaultdict(list)
  for s in sorted(glob.glob("/data/media/0/realdata/*/rlog.zst")):
    by_route[os.path.basename(os.path.dirname(s)).split("--")[0]].append(s)
  routes = sys.argv[1:] or sorted(by_route)[-DEFAULT_ROUTES:]

  out = {}
  for route in routes:
    rep = replay_route(sorted(by_route[route], key=lambda s: int(os.path.basename(os.path.dirname(s)).split("--")[-1])))
    if not rep["frames"]:
      continue
    out[route] = rep
    outcomes = [ep["outcome"] for ep in rep["brake_episodes"]]
    print(f"{route} engaged_frames={rep['frames']} radar_lead={rep['radar_lead_share']} conf_on={rep['conf_on_share']} " +
          f"brake_ep={len(outcomes)} stopped={outcomes.count('stopped')} slowed={outcomes.count('slowed')} " +
          f"phantom={outcomes.count('phantom')} primed={rep['primed_share']} lift_ep={len(rep['lift_episodes'])} " +
          f"lift_ep_e2e={sum(ep['src'] == 'e2e' for ep in rep['lift_episodes'])} exec_gap={rep['exec_gap']} states={rep['states']}", flush=True)

  dest = os.environ.get("LA_REPLAY_OUT", "/data/long_assist_replay.json")
  with open(dest, "w") as f:
    json.dump(out, f, indent=1)
  print(f"# {len(out)} routes; wrote {dest}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
