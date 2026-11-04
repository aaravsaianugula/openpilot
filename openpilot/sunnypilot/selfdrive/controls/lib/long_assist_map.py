"""
Long Assist map context (Elantra overlay).

Two map sources, both projected onto where we are going:
  * mapd's MapTargetVelocities mem param: curve target speeds at points along the road ahead;
  * an offline OSM layer of stop signs, traffic signals, give-way signs and level crossings, built by
    .elantra/osm_hazards.py into HAZARDS_PATH (mapd has none of these).

Map data only primes Long Assist (no lift into a controlled intersection, a matching model stop is
trusted at once). It never brakes on its own: the model plan decides whether to stop.
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
HAZARD_RANGE = 150.    # m ahead
# m either side of our heading line. Stop and give-way nodes sit on the road they control, so a tight
# corridor skips the side-street signs along an arterial (replay: primed on up to 44% of frames at 25 m);
# signal and crossing nodes are often mapped at the junction centre, so they keep a wide one.
HAZARD_LATERAL = np.array([0., 6., 20., 6., 20.])  # indexed by Kind
WINDOW_DEG = 0.004     # ~450 m latitude pre-filter before projecting
BEARING_MIN_SPEED = 2.  # m/s, GPS bearing is noise below this: keep the last good one


class Kind(IntEnum):
  none = 0
  stop = 1
  signal = 2
  give_way = 3
  rail = 4


class MapAhead(NamedTuple):
  curve_dist: np.ndarray   # m along the road to each mapd target point
  curve_v: np.ndarray      # m/s target speed at that point
  hazard_dist: float       # m to the nearest mapped hazard ahead, -1 when none
  hazard_kind: int


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


def load_hazards(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  if not os.path.isfile(path):
    return None
  try:
    with np.load(path) as d:
      order = np.argsort(d["lat"])
      return d["lat"][order], d["lon"][order], d["kind"][order]
  except (OSError, ValueError, KeyError):
    return None


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

  def ahead(self, lat: float, lon: float, bearing_deg: float, speed: float, valid: bool) -> MapAhead:
    if not valid:
      return EMPTY
    if speed > BEARING_MIN_SPEED:
      self.bearing = bearing_deg * DEG
    s, c = np.sin(self.bearing), np.cos(self.bearing)

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

    hazard_dist, hazard_kind = -1., Kind.none
    if self.hazards is not None:
      h_lat, h_lon, h_kind = self.hazards
      lo, hi = np.searchsorted(h_lat, [lat - WINDOW_DEG, lat + WINDOW_DEG])
      if hi > lo:
        e, n = to_local(lat, lon, h_lat[lo:hi], h_lon[lo:hi])
        along, cross = e * s + n * c, e * c - n * s
        m = (along > 0.) & (along < HAZARD_RANGE) & (np.abs(cross) < HAZARD_LATERAL[h_kind[lo:hi]])
        if m.any():
          j = int(np.argmin(np.where(m, along, np.inf)))
          hazard_dist, hazard_kind = float(along[j]), int(h_kind[lo:hi][j])

    return MapAhead(curve_dist, curve_v, hazard_dist, hazard_kind)
