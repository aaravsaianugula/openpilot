"""
Long Assist map context (Elantra overlay).

Two map sources, both projected onto where we are going:
  * mapd's MapTargetVelocities mem param: curve target speeds at points along the road ahead;
  * an offline OSM layer of stop signs, traffic signals, give-way signs and level crossings, built by
    .elantra/osm_hazards.py into HAZARDS_PATH (mapd has none of these).

Each hazard carries the approaches it controls (the road centreline leading into it, the stop line's offset and how
sure the map is). We report the nearest hazard whose approach we are on: within the road's half-width plus GPS error of
its centreline and heading along it. A side-street sign, the other direction's sign or a signal on a parallel road does
not match. Nodes without approach data (an older file, or outside its area) fall back to a corridor along our heading
and are reported as ambiguous. Long Assist decides what to do with it (long_assist.py).
"""
import json
import os
from enum import IntEnum
from typing import NamedTuple

import numpy as np

HAZARDS_PATH = "/data/media/0/long_assist/hazards.npz"
R_EARTH = 6371000.
DEG = np.pi / 180.
CURVE_RANGE = 400.     # m of road ahead used for map curve speeds
NEAREST_MAX = 50.      # m, farther than this from every target point means the list is not for our road
HAZARD_RANGE = 300.    # m ahead: one smooth stop from 45 mph starts ~250 m out
# corridor fallback, m either side of our heading line. Stop and give-way nodes sit on the road they control; signal
# and crossing nodes are often mapped at the junction centre, so they keep a wide one.
HAZARD_LATERAL = np.array([0., 6., 20., 6., 20.])  # indexed by Kind
CORRIDOR_RANGE = 150.
WINDOW_DEG = 0.004     # ~450 m latitude pre-filter before projecting
BEARING_MIN_SPEED = 2.  # m/s, GPS bearing is noise below this: carried on by the yaw rate
HEADING_TOL = 35.      # deg between our heading and the approach's direction of travel
LAT_GPS = 3.           # m added to the road's half-width for lane position and GPS error (plus the fix's own accuracy)
GPS_ACC_MAX = 10.      # m, a fix worse than this matches nothing
S_PAST = 4.            # m past the stop line still reported (holding at the line)


class Kind(IntEnum):
  none = 0
  stop = 1
  signal = 2
  give_way = 3
  rail = 4


class Cert(IntEnum):
  ambiguous = 0  # the map cannot say this approach stops (or no approach data): only with the model's agreement
  likely = 1     # junction-centre sign on the minor road / the road that ends there
  sure = 2       # tagged direction, stop=all, a signal at a junction, or toward the nearest junction


class MapAhead(NamedTuple):
  curve_dist: np.ndarray   # m along the road to each mapd target point
  curve_v: np.ndarray      # m/s target speed at that point
  hazard_dist: float       # m to the nearest mapped hazard's stop line ahead on our approach, -1 when none
  hazard_kind: int
  hazard_cert: int = Cert.ambiguous
  hazard_id: int = -1      # index in the hazard file, stable for one file
  hazard_y: float = 0.     # m, the stop line's lateral offset from our heading line (left +)


EMPTY = MapAhead(np.zeros(0), np.zeros(0), -1., Kind.none)


def to_local(lat0: float, lon0: float, lat, lon) -> tuple[np.ndarray, np.ndarray]:
  """Equirectangular east/north metres from (lat0, lon0); exact enough over a few hundred metres."""
  return (np.asarray(lon) - lon0) * DEG * R_EARTH * np.cos(lat0 * DEG), (np.asarray(lat) - lat0) * DEG * R_EARTH


def parse_target_velocities(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  try:
    pts = json.loads(raw) if raw else None
    if not pts:
      return None
    return (np.array([p["latitude"] for p in pts], dtype=float), np.array([p["longitude"] for p in pts], dtype=float),
            np.array([p["velocity"] for p in pts], dtype=float))
  except (ValueError, KeyError, TypeError):
    return None


def load_hazards(path: str) -> dict | None:
  if not os.path.isfile(path):
    return None
  try:
    with np.load(path) as d:
      hz = {k: d[k] for k in d.files}
    if "ap_i0" not in hz:  # older file: positions only
      order = np.argsort(hz["lat"])
      hz = {k: hz[k][order] for k in ("lat", "lon", "kind")}
      hz["ap_i0"] = np.zeros(len(hz["lat"]) + 1, np.int32)
    return hz
  except (OSError, ValueError, KeyError):
    return None


def match_approaches(ce: np.ndarray, cn: np.ndarray, pts: np.ndarray, n: np.ndarray, off: np.ndarray, w: np.ndarray,
                     bearing: float, acc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Our position (ce, cn, relative to each approach's hazard node) against each approach polyline (pts[:, 0] is the
  node, then back along the road). Returns per approach: matched, distance to its stop line along the road, and the
  stop line's position relative to us (east, north)."""
  A, B = pts[:, :-1], pts[:, 1:]                       # segment k runs from A (nearer the node) back to B
  u = B - A
  L = np.maximum(np.hypot(u[..., 0], u[..., 1]), 1e-3)
  S = np.concatenate((np.zeros((len(pts), 1)), np.cumsum(L, axis=1)), axis=1)
  c = np.stack([ce, cn], axis=1)[:, None, :] - A
  t = (c[..., 0] * u[..., 0] + c[..., 1] * u[..., 1]) / L ** 2
  t = np.clip(t, np.where(np.arange(t.shape[1]) == 0, -S_PAST / L, 0.), 1.)
  d = c - t[..., None] * u
  lat = np.hypot(d[..., 0], d[..., 1])
  # travelling toward the node is B -> A
  seg_brg = np.degrees(np.arctan2(-u[..., 0], -u[..., 1]))
  dpsi = np.abs((bearing - seg_brg + 180.) % 360. - 180.)
  valid = np.arange(t.shape[1])[None, :] < (n[:, None] - 1)
  tol = (w + LAT_GPS + min(acc, GPS_ACC_MAX))[:, None]
  ok = valid & (lat < tol) & (dpsi < HEADING_TOL)
  k = np.argmin(np.where(ok, lat, np.inf), axis=1)
  r = np.arange(len(pts))
  matched = ok[r, k]
  s = S[r, k] + t[r, k] * L[r, k]
  # stop line: off back from the node along the first segment
  u0 = u[:, 0] / L[:, :1]
  line = pts[:, 0] + u0 * off[:, None] - np.stack([ce, cn], axis=1)
  return matched, s - off, line


class MapContext:
  def __init__(self, hazards_path: str = HAZARDS_PATH):
    self.hazards = load_hazards(hazards_path)
    self.targets = None
    self._targets_raw = None
    self.bearing = 0.

  def set_targets(self, raw) -> None:
    if raw != self._targets_raw:
      self._targets_raw = raw
      self.targets = parse_target_velocities(raw)

  def ahead(self, lat: float, lon: float, bearing_deg: float, speed: float, valid: bool, acc: float = 0.,
            yaw_rate: float = 0., dt: float = 0.) -> MapAhead:
    """yaw_rate (rad/s, clockwise +, the model's orientationRate.z) carries the heading on below BEARING_MIN_SPEED."""
    if speed > BEARING_MIN_SPEED:
      self.bearing = bearing_deg
    else:
      self.bearing = (self.bearing + np.degrees(yaw_rate) * dt) % 360.
    if not valid or acc > GPS_ACC_MAX:
      return EMPTY
    psi = self.bearing * DEG
    s, c = np.sin(psi), np.cos(psi)

    curve_dist, curve_v = EMPTY.curve_dist, EMPTY.curve_v
    if self.targets is not None:
      t_lat, t_lon, t_v = self.targets
      e, n = to_local(lat, lon, t_lat, t_lon)
      i = int(np.argmin(np.hypot(e, n)))
      if np.hypot(e[i], n[i]) < NEAREST_MAX:
        # distance along the polyline from the nearest point, offset by where that point sits along our heading
        seg = np.hypot(np.diff(e[i:]), np.diff(n[i:]))
        dist = e[i] * s + n[i] * c + np.concatenate(([0.], np.cumsum(seg)))
        keep = (dist > 0.) & (dist < CURVE_RANGE)
        curve_dist, curve_v = dist[keep], t_v[i:][keep]

    best = (np.inf, Kind.none, Cert.ambiguous, -1, 0.)
    hz = self.hazards
    if hz is not None:
      lo, hi = np.searchsorted(hz["lat"], [lat - WINDOW_DEG, lat + WINDOW_DEG])
      if hi > lo:
        e, n = to_local(lat, lon, hz["lat"][lo:hi], hz["lon"][lo:hi])
        near = np.hypot(e, n) < HAZARD_RANGE + 60.
        i0 = hz["ap_i0"]
        has_ap = i0[lo + 1:hi + 1] > i0[lo:hi]
        # hazards with approach data
        hs = lo + np.flatnonzero(near & has_ap)
        if hs.size:
          js = np.concatenate([np.arange(i0[h], i0[h + 1]) for h in hs])
          owner = np.repeat(hs, i0[hs + 1] - i0[hs])
          ce, cn = -e[owner - lo], -n[owner - lo]  # us relative to each approach's node
          m, dist, line = match_approaches(ce, cn, hz["ap_pts"][js].astype(float), hz["ap_n"][js], hz["ap_off"][js].astype(float),
                                           hz["ap_w"][js].astype(float), self.bearing, acc)
          m &= (dist > -S_PAST) & (dist < HAZARD_RANGE)
          if m.any():
            q = int(np.argmin(np.where(m, dist, np.inf)))
            y = float(-line[q, 0] * c + line[q, 1] * s)
            best = (float(dist[q]), int(hz["kind"][owner[q]]), int(hz["ap_cert"][js[q]]), int(owner[q]), y)
        # nodes without approach data: a corridor along our heading
        along, cross = e * s + n * c, e * c - n * s
        kinds = hz["kind"][lo:hi]
        mc = near & ~has_ap & (along > 0.) & (along < CORRIDOR_RANGE) & (np.abs(cross) < HAZARD_LATERAL[kinds])
        if mc.any():
          j = int(np.argmin(np.where(mc, along, np.inf)))
          if along[j] < best[0]:
            best = (float(along[j]), int(kinds[j]), Cert.ambiguous, lo + j, float(-cross[j]))

    if best[3] < 0:
      return MapAhead(curve_dist, curve_v, -1., Kind.none)
    return MapAhead(curve_dist, curve_v, best[0], best[1], best[2], best[3], best[4])
