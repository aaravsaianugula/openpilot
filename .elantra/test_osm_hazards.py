#!/usr/bin/env python3
"""Tests for osm_hazards.py. No network: the Overpass response is a fixture."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import osm_hazards as oh


def test_parse_keeps_known_nodes_sorted_by_lat():
  arrays = oh.parse([
    {"type": "node", "lat": 47.5, "lon": -122.0, "tags": {"highway": "traffic_signals"}},
    {"type": "node", "lat": 47.1, "lon": -122.1, "tags": {"highway": "stop", "stop": "all"}},
    {"type": "node", "lat": 47.2, "lon": -122.2, "tags": {"highway": "crossing"}},
    {"type": "way", "tags": {"highway": "stop"}},
    {"type": "node", "lat": 47.3, "lon": -122.3, "tags": {"railway": "level_crossing"}},
  ])
  assert arrays["lat"].tolist() == [47.1, 47.3, 47.5]
  assert arrays["kind"].tolist() == [1, 4, 2]


def test_bbox_and_query():
  s, w, n, e = oh.bbox(47.36, -122.02, 10.)
  assert abs((n - s) * 111.32 - 20.) < 0.01 and e - w > n - s  # longitude degrees are shorter here
  q = oh.query((s, w, n, e))
  assert q.count("node[") == len(oh.KINDS) and '"highway"="traffic_signals"' in q


if __name__ == "__main__":
  import pytest
  sys.exit(pytest.main([__file__, "-q"]))
