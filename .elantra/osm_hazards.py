#!/usr/bin/env python3
"""Build Long Assist's offline map layer of stop signs, traffic signals, give-way signs and level crossings.

mapd gives speed limits and curve speeds but no intersection controls, so Long Assist reads this file
(long_assist_map.HAZARDS_PATH) to know a stop or signal is coming before the model sees it. It only
primes: no lift toward it, and a model stop that lines up with it is trusted at once.

One Overpass API query for a box around a centre point; the result is a few hundred KB of npz.

    python .elantra/osm_hazards.py --center 47.36,-122.02 --radius-km 60 --out hazards.npz
    scp hazards.npz comma@<device>:/data/media/0/long_assist/hazards.npz   # read at the next drive

Stdlib + numpy only, so it also runs on the device itself (default --out is HAZARDS_PATH there).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEVICE_PATH = "/data/media/0/long_assist/hazards.npz"
# tag -> long_assist_map.Kind
KINDS = {("highway", "stop"): 1, ("highway", "traffic_signals"): 2, ("highway", "give_way"): 3, ("railway", "level_crossing"): 4}


def bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
  dlat = radius_km / 111.32
  dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
  return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def query(box: tuple[float, float, float, float]) -> str:
  b = ",".join(f"{v:.5f}" for v in box)
  nodes = "".join(f'node["{k}"="{v}"]({b});' for k, v in KINDS)
  return f"[out:json][timeout:300];({nodes});out body qt;"


def parse(elements: list[dict]) -> dict[str, np.ndarray]:
  lat, lon, kind = [], [], []
  for el in elements:
    tags = el.get("tags", {})
    k = next((code for (key, val), code in KINDS.items() if tags.get(key) == val), 0)
    if el.get("type") == "node" and k:
      lat.append(el["lat"])
      lon.append(el["lon"])
      kind.append(k)
  order = np.argsort(lat)
  return {"lat": np.array(lat, dtype=np.float64)[order], "lon": np.array(lon, dtype=np.float64)[order],
          "kind": np.array(kind, dtype=np.uint8)[order]}


def fetch(box) -> list[dict]:
  data = urllib.parse.urlencode({"data": query(box)}).encode()
  req = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": "sunnypilot-elantra-long-assist"})
  with urllib.request.urlopen(req, timeout=330) as r:
    return json.load(r)["elements"]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--center", required=True, help="lat,lon")
  ap.add_argument("--radius-km", type=float, default=60.)
  ap.add_argument("--out", default=DEVICE_PATH if os.path.isdir("/data/media/0") else "hazards.npz")
  args = ap.parse_args()

  lat, lon = (float(v) for v in args.center.split(","))
  arrays = parse(fetch(bbox(lat, lon, args.radius_km)))
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  np.savez_compressed(args.out, **arrays)
  counts = {name: int((arrays["kind"] == code).sum()) for (_, name), code in
            zip(KINDS, KINDS.values(), strict=True)}
  print(f"wrote {args.out}: {len(arrays['lat'])} nodes {counts}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
