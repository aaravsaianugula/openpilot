#!/usr/bin/env python3
"""Tests for osm_hazards.py. No network: the Overpass response is a fixture."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import osm_hazards as oh

R = 6371000.
LAT0, LON0 = 47.36, -122.02


def node(i, n, e, **tags):
  el = {"type": "node", "id": i, "lat": LAT0 + n / R * 180 / math.pi,
        "lon": LON0 + e / (R * math.cos(math.radians(LAT0))) * 180 / math.pi}
  if tags:
    el["tags"] = tags
  return el


def way(i, nodes, **tags):
  return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


def fixture(stop_tags=None, j_tags=None):
  """Main (secondary) runs north through junction J (id 3) to junction X (id 5, a signal) with Cross (tertiary); Side
  (residential) ends at J from the east, with node 20 10 m along it."""
  els = [node(1, -300, 0), node(2, -150, 0), node(3, 0, 0, **(j_tags or {})), node(4, 150, 0), node(5, 300, 0, highway="traffic_signals"),
         node(6, 300, -150), node(7, 300, 150), node(20, 0, 10, **(stop_tags or {})), node(21, 0, 60), node(22, 0, 200)]
  els += [way(100, [1, 2, 3, 4, 5], highway="secondary", name="Main"), way(101, [6, 5, 7], highway="tertiary", name="Cross"),
          way(102, [3, 20, 21, 22], highway="residential", name="Side")]
  return els


def approaches(arrays, n, e):
  """(direction (north, east) back from the node along each approach, stop line offset, cert) of the hazard at (n, e) m"""
  dn = (arrays["lat"] - LAT0) * math.pi / 180 * R - n
  de = (arrays["lon"] - LON0) * math.pi / 180 * R * math.cos(math.radians(LAT0)) - e
  i = int(np.argmin(np.hypot(dn, de)))
  out = []
  for j in range(arrays["ap_i0"][i], arrays["ap_i0"][i + 1]):
    d = arrays["ap_pts"][j][1] - arrays["ap_pts"][j][0]
    L = float(np.hypot(*d))
    out.append((round(float(d[1]) / L), round(float(d[0]) / L), round(float(arrays["ap_off"][j]), 3), int(arrays["ap_cert"][j])))
  return sorted(out)


def test_side_street_stop_controls_only_traffic_heading_to_its_junction():
  arrays = oh.build(fixture(stop_tags={"highway": "stop"}))
  assert approaches(arrays, 0, 10) == [(0, 1, 0., 2)]  # entered from the east (westbound, toward J), line at the node


def test_tagged_direction_wins():
  arrays = oh.build(fixture(stop_tags={"highway": "stop", "direction": "forward"}))  # Side's node order: eastbound
  assert approaches(arrays, 0, 10) == [(0, -1, 0., 2)]


def test_junction_centre_stop_on_the_minor_road():
  for tags, cert in (({"highway": "stop", "stop": "minor"}, 2), ({"highway": "stop"}, 1)):
    aps = approaches(oh.build(fixture(j_tags=tags)), 0, 0)
    assert len(aps) == 1 and aps[0][:2] == (0, 1) and aps[0][3] == cert  # only from Side
    assert aps[0][2] == oh.HALF_WIDTH[oh.RANK["secondary"]] + oh.CROSSWALK  # Main's half-width + crosswalk


def test_all_way_stop_and_junction_signal():
  arrays = oh.build(fixture(j_tags={"highway": "stop", "stop": "all"}))
  assert len(approaches(arrays, 0, 0)) == 3
  sig = approaches(arrays, 300, 0)  # from Main (south) and both sides of Cross
  assert [a[:2] for a in sig] == [(-1, 0), (0, -1), (0, 1)] and all(a[3] == 2 for a in sig)
  main = [a for a in sig if a[:2] == (-1, 0)][0]
  assert main[2] == oh.HALF_WIDTH[oh.RANK["tertiary"]] + oh.CROSSWALK + oh.SIGNAL_SETBACK


def test_merge_keeps_old_nodes_outside_the_box(tmp_path):
  old = tmp_path / "old.npz"
  np.savez(old, lat=np.array([47.0, 47.36]), lon=np.array([-122.0, -122.02]), kind=np.array([2, 1], np.uint8))
  arrays = oh.merge(oh.build(fixture(stop_tags={"highway": "stop"})), str(old), (47.3, -122.1, 47.4, -121.9))
  assert arrays["lat"][0] == 47.0 and arrays["ap_i0"][1] == 0  # outside: position only
  assert len(arrays["lat"]) == 3  # the old node inside the box is replaced by the new data


def test_query():
  q = oh.query((47.3, -122.1, 47.4, -121.9))
  assert 'way["highway"~' in q and "node(w.w)" in q and '"railway"="level_crossing"' in q


if __name__ == "__main__":
  import pytest
  sys.exit(pytest.main([__file__, "-q"]))
