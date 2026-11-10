#!/usr/bin/env python3
"""Build Long Assist's offline map layer of stop signs, traffic signals, give-way signs and level crossings.

mapd gives speed limits and curve speeds but no intersection controls, so Long Assist reads this file
(long_assist_map.HAZARDS_PATH) to know a stop or signal is coming before the model sees it.

Besides each node's position and kind, the file holds the approaches the node controls: the road centreline leading
into it (up to 250 m back), the stop line's offset before the node and how sure the map is that this approach stops.
Long Assist matches our position and heading to an approach, so a side-street sign, the other direction's sign or a
signal on a parallel road never applies. How an approach is decided:
  * direction=forward/backward (traffic_signals:direction for signals): the approach travelling that way along the way;
  * stop=all, a signal or a level crossing at a junction: every approach;
  * a stop/give-way node at a junction centre: the minor road(s) (lower class), else the road that ends there (T);
    a crossroads of equal roads with no stop=all is ambiguous (Long Assist then needs the model to agree);
  * a node on a road, no direction: traffic heading toward the nearest junction within 60 m.

Downloads go through the project's security gate: fetch the Overpass response into a quarantine folder inside a
container, scan it, then build from the saved file:

    python -I .elantra/osm_hazards.py --query --bbox 47.21,-122.34,47.49,-121.88 > q.txt        # the Overpass query
    python .elantra/osm_hazards.py --json overpass.json --bbox 47.21,-122.34,47.49,-121.88 --merge old.npz --out hazards.npz
    scp hazards.npz comma@<device>:/data/media/0/long_assist/hazards.npz   # read when plannerd starts

Stdlib + numpy only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

DEVICE_PATH = "/data/media/0/long_assist/hazards.npz"
# tag -> long_assist_map.Kind
KINDS = {("highway", "stop"): 1, ("highway", "traffic_signals"): 2, ("highway", "give_way"): 3, ("railway", "level_crossing"): 4}
DRIVABLE = "|".join(("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
                     "service", "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"))
RANK = {"motorway": 7, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2, "residential": 2, "road": 2,
        "living_street": 1, "service": 0}
MINOR_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}  # never make a junction
HALF_WIDTH = [4., 4.5, 5.5, 7., 8., 9., 11., 12.]  # m by rank: road half-width (a crossing road's centre to our stop line)
AP_S = np.array([0., 15., 30., 60., 100., 150., 200., 250.])  # m back along an approach
JUNCTION_NEAR = 60.     # m: a stop/give-way on a road controls traffic heading to its nearest junction within this
RAIL_STOP_OFF = 5.      # m before a crossing node (on the tracks) where traffic stops
CROSSWALK = 2.          # m from the crossing road's edge back to the stop line
# signals at a junction: drivers first in line stopped a median 13.4 m (4-21) short of half-width + crosswalk (17 red
# lights, 2026-10..11 drives): turn lanes and wide crossings set arterial stop lines farther back
SIGNAL_SETBACK = 12.
R_EARTH = 6371000.
DEG = math.pi / 180.


def query(box: tuple[float, float, float, float]) -> str:
  b = ",".join(f"{v:.5f}" for v in box)
  ways = f'way["highway"~"^({DRIVABLE})$"]({b})->.w;.w out body qt;node(w.w)->.n;.n out skel qt;'
  nodes = f'(node["highway"~"^(stop|traffic_signals|give_way)$"]({b});node["railway"="level_crossing"]({b}););out body qt;'
  return "[out:json][timeout:900][maxsize:1073741824];" + ways + nodes


def kind_of(tags: dict) -> int:
  if tags.get("highway") == "traffic_signals":
    sig = tags.get("traffic_signals")
    return 0 if sig == "emergency" else 4 if sig == "level_crossing" else 2
  return next((code for (key, val), code in KINDS.items() if tags.get(key) == val), 0)


def rank(tags: dict) -> int:
  return RANK.get(tags.get("highway", "").replace("_link", ""), 0)


def half_width(tags: dict) -> float:
  try:
    return max(1.7 * float(tags["lanes"]) + 1., 4.)
  except (KeyError, ValueError):
    return HALF_WIDTH[rank(tags)]


class Graph:
  def __init__(self, elements: list[dict]):
    self.ll: dict[int, tuple[float, float]] = {}
    self.ways: list[dict] = []
    self.hazards: list[tuple[int, dict]] = []
    for el in elements:
      if el.get("type") == "node":
        self.ll[el["id"]] = (el["lat"], el["lon"])
        if el.get("tags") and kind_of(el["tags"]):
          self.hazards.append((el["id"], el["tags"]))
      elif el.get("type") == "way" and el.get("tags", {}).get("highway"):
        self.ways.append(el)
    # adj[n]: (neighbour, way index, travel neighbour -> n allowed)
    self.adj: dict[int, list[tuple[int, int, bool]]] = {}
    nbrs: dict[int, set] = {}
    for wi, w in enumerate(self.ways):
      t, nodes = w["tags"], w["nodes"]
      ow = t.get("oneway", "")
      fwd_only = ow in ("yes", "true", "1") or t.get("junction") in ("roundabout", "circular") or \
        (t["highway"] in ("motorway", "motorway_link") and ow != "no")
      rev_only = ow == "-1"
      junction_way = t.get("service") not in MINOR_SERVICE
      for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        if a == b or a not in self.ll or b not in self.ll:
          continue
        self.adj.setdefault(b, []).append((a, wi, not rev_only))
        self.adj.setdefault(a, []).append((b, wi, not fwd_only))
        if junction_way:
          nbrs.setdefault(a, set()).add(b)
          nbrs.setdefault(b, set()).add(a)
    self.deg = {n: len(s) for n, s in nbrs.items()}

  def local(self, h: int, n: int) -> tuple[float, float]:
    lat0, lon0 = self.ll[h]
    lat, lon = self.ll[n]
    return (lon - lon0) * DEG * R_EARTH * math.cos(lat0 * DEG), (lat - lat0) * DEG * R_EARTH

  def junction(self, n: int) -> bool:
    return self.deg.get(n, 0) >= 3

  def way_dir(self, h: int, nb: int, wi: int) -> int:
    """+1 if travel nb -> h goes along way wi's node order, -1 against it."""
    nodes = self.ways[wi]["nodes"]
    for i, n in enumerate(nodes):
      if n == h:
        if i > 0 and nodes[i - 1] == nb:
          return 1
        if i + 1 < len(nodes) and nodes[i + 1] == nb:
          return -1
    return 0

  def walk(self, h: int, nb: int, max_len: float, junction_stop: bool = False) -> tuple[list[tuple[float, float]], float | None]:
    """Walk from h through nb away from h, straightest continuation. Returns the local points (from h) and, with
    junction_stop, the distance to the first junction met (None if none within max_len)."""
    pts = [(0., 0.), self.local(h, nb)]
    total = math.hypot(*pts[1])
    prev, cur = h, nb
    while total < max_len:
      if junction_stop and self.junction(cur):
        return pts, total
      dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
      best, best_cos = None, math.cos(50 * DEG)
      for n2, _, _ in self.adj.get(cur, []):
        if n2 in (prev, cur):
          continue
        p = self.local(h, n2)
        ex, ey = p[0] - pts[-1][0], p[1] - pts[-1][1]
        L = math.hypot(ex, ey) * math.hypot(dx, dy)
        if L > 0 and (ex * dx + ey * dy) / L > best_cos:
          best, best_cos = (n2, p), (ex * dx + ey * dy) / L
      if best is None:
        break
      prev, cur = cur, best[0]
      total += math.hypot(best[1][0] - pts[-1][0], best[1][1] - pts[-1][1])
      pts.append(best[1])
    if junction_stop:
      return pts, (total if self.junction(cur) else None)
    return pts, None


def resample(pts: list[tuple[float, float]]) -> tuple[np.ndarray, int]:
  p = np.array(pts)
  s = np.concatenate(([0.], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
  n = int(np.sum(AP_S <= s[-1]))
  if n < len(AP_S) and s[-1] > AP_S[n - 1] + 5.:  # keep the end of a shorter road as the last point
    n += 1
  ss = np.minimum(AP_S, s[-1])
  return np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], axis=1), n


def approaches(g: Graph, h: int, tags: dict, kind: int) -> list[tuple[int, int, float, int]]:
  """(neighbour, way index, stop line offset, certainty 2 sure / 1 likely / 0 ambiguous) for each approach it controls."""
  inc = [(nb, wi) for nb, wi, ok in g.adj.get(h, []) if ok]
  if not inc:
    return []
  junc = g.junction(h)
  all_ways = {wi for _, wi, _ in g.adj[h]}
  dir_tag = tags.get("traffic_signals:direction" if kind == 2 else "direction", tags.get("direction"))

  def off(wi: int) -> float:
    if kind == 4:
      return RAIL_STOP_OFF
    if not junc:
      return 0.
    # the crossing road's half-width (not our own road's continuation) plus a crosswalk
    name = g.ways[wi]["tags"].get("name")
    others = [w for w in all_ways if w != wi]
    cross = [w for w in others if name is None or g.ways[w]["tags"].get("name") != name] or others
    return max([half_width(g.ways[w]["tags"]) for w in cross] + [4.]) + CROSSWALK + (SIGNAL_SETBACK if kind == 2 else 0.)

  def directed(sign: int, cert: int):
    return [(nb, wi, off(wi), cert) for nb, wi in inc if g.way_dir(h, nb, wi) == sign]

  if dir_tag in ("forward", "backward"):
    return directed(1 if dir_tag == "forward" else -1, 2)
  if kind == 4 or (kind == 2 and (junc or dir_tag == "both")) or (kind in (1, 3) and tags.get("stop") == "all"):
    return [(nb, wi, off(wi), 2) for nb, wi in inc]
  if junc:
    if kind == 2:
      return [(nb, wi, off(wi), 2) for nb, wi in inc]
    ranks = {wi: rank(g.ways[wi]["tags"]) for wi in all_ways}
    top = max(ranks.values())
    minor = [(nb, wi) for nb, wi in inc if ranks[wi] < top]
    if minor:
      return [(nb, wi, off(wi), 2 if tags.get("stop") == "minor" else 1) for nb, wi in minor]
    # equal roads: the one that ends here (no straight continuation with the same name) stops
    out = []
    for nb, wi in inc:
      ax, ay = g.local(h, nb)
      name = g.ways[wi]["tags"].get("name")
      through = False
      for nb2, wi2, _ in g.adj[h]:
        if nb2 == nb:
          continue
        bx, by = g.local(h, nb2)
        c = (ax * bx + ay * by) / max(math.hypot(ax, ay) * math.hypot(bx, by), 1e-6)
        if c < -math.cos(30 * DEG) and (name is None or g.ways[wi2]["tags"].get("name") == name):
          through = True
      if not through:
        out.append((nb, wi, off(wi), 1))
    return out if out and len(out) < len(inc) else [(nb, wi, off(wi), 0) for nb, wi in inc]
  # on a road: toward the nearest junction
  dists = []
  for nb, _ in inc:
    # junction beyond h, seen by traffic coming from nb: walk the other way from h
    ahead = [n2 for n2, w2, _ in g.adj[h] if n2 != nb]
    d = min((dj for n2 in ahead if (dj := g.walk(h, n2, 150., junction_stop=True)[1]) is not None), default=None)
    dists.append(d)
  if kind == 2:
    best = [i for i, d in enumerate(dists) if d is not None and d <= JUNCTION_NEAR]
    return [(nb, wi, off(wi), 2) for i, (nb, wi) in enumerate(inc) if i in best or not best]
  known = [d for d in dists if d is not None]
  if not known:
    return [(nb, wi, off(wi), 0) for nb, wi in inc]
  dmin = min(known)
  others = [d for d in dists if d is not None and d != dmin]
  sure = dmin <= JUNCTION_NEAR and all(d >= 1.5 * dmin for d in others)
  return [(nb, wi, off(wi), 2 if sure else 0) for (nb, wi), d in zip(inc, dists, strict=True) if d == dmin or not sure]


def build(elements: list[dict]) -> dict[str, np.ndarray]:
  g = Graph(elements)
  rows = []
  for h, tags in g.hazards:
    kind = kind_of(tags)
    aps = []
    for nb, wi, off_m, cert in approaches(g, h, tags, kind):
      pts, _ = g.walk(h, nb, AP_S[-1] + 10.)
      p, n = resample(pts)
      aps.append((p, n, off_m, cert, half_width(g.ways[wi]["tags"])))
    rows.append((g.ll[h][0], g.ll[h][1], kind, aps))
  rows.sort(key=lambda r: r[0])
  return pack(rows)


def pack(rows) -> dict[str, np.ndarray]:
  i0 = np.cumsum([0] + [len(r[3]) for r in rows]).astype(np.int32)
  aps = [a for r in rows for a in r[3]]
  return {"lat": np.array([r[0] for r in rows], np.float64), "lon": np.array([r[1] for r in rows], np.float64),
          "kind": np.array([r[2] for r in rows], np.uint8), "ap_i0": i0,
          "ap_pts": np.array([a[0] for a in aps], np.float32).reshape(-1, len(AP_S), 2),
          "ap_n": np.array([a[1] for a in aps], np.uint8), "ap_off": np.array([a[2] for a in aps], np.float32),
          "ap_cert": np.array([a[3] for a in aps], np.uint8), "ap_w": np.array([a[4] for a in aps], np.float32)}


def merge(new: dict[str, np.ndarray], old_path: str, box) -> dict[str, np.ndarray]:
  """Keep the old file's nodes outside box (position and kind only: Long Assist then needs the model to agree)."""
  with np.load(old_path) as o:
    keep = ~((o["lat"] > box[0]) & (o["lat"] < box[2]) & (o["lon"] > box[1]) & (o["lon"] < box[3]))
    rows = [(float(a), float(b), int(k), []) for a, b, k in zip(o["lat"][keep], o["lon"][keep], o["kind"][keep], strict=True)]
  n = len(new["lat"])
  for i in range(n):
    a, b = new["ap_i0"][i], new["ap_i0"][i + 1]
    rows.append((float(new["lat"][i]), float(new["lon"][i]), int(new["kind"][i]),
                 [(new["ap_pts"][j], int(new["ap_n"][j]), float(new["ap_off"][j]), int(new["ap_cert"][j]), float(new["ap_w"][j]))
                  for j in range(a, b)]))
  rows.sort(key=lambda r: r[0])
  return pack(rows)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--bbox", required=True, help="south,west,north,east")
  ap.add_argument("--query", action="store_true", help="print the Overpass query and exit")
  ap.add_argument("--json", help="a saved Overpass response for that query")
  ap.add_argument("--merge", help="an older hazards.npz whose nodes outside bbox are kept")
  ap.add_argument("--out", default=DEVICE_PATH if os.path.isdir("/data/media/0") else "hazards.npz")
  args = ap.parse_args()
  box = tuple(float(v) for v in args.bbox.split(","))
  if args.query:
    print(query(box))
    return 0
  if not args.json:
    ap.error("--json is required (fetch it through the security gate first)")
  with open(args.json) as f:
    arrays = build(json.load(f)["elements"])
  if args.merge:
    arrays = merge(arrays, args.merge, box)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  np.savez_compressed(args.out, **arrays)
  counts = {name: int((arrays["kind"] == code).sum()) for (_, name), code in KINDS.items()}
  cert = np.bincount(arrays["ap_cert"], minlength=3).tolist()
  print(f"wrote {args.out}: {len(arrays['lat'])} nodes {counts}, {len(arrays['ap_n'])} approaches (ambiguous/likely/sure {cert})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
