#!/usr/bin/env python3
"""Select diverse real replay clips from reserved training routes only.

Measured motion ranks footage acquisition, never candidate performance or model
ground truth. Cameras remain required; this does not establish RDF input readiness.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from controller_extract import hold_calibration, sha256
from controller_tune import cohort_key, enrich_route_metadata

KINDS = ("stopping", "acceleration", "turning")
SECONDS, HISTORY_S = 32., 6.6


def segment_candidates(meta, data, seconds=SECONDS):
  """Rank continuous footage after the unchanged genuine 132-frame history.

  Duration is an acquisition choice, never a relaxation of control/validity
  gates. It must cover at least two scored seconds on the exact 20 Hz grid.
  """
  if isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, (int, float, np.integer, np.floating)):
    raise ValueError("duration must be a finite number of seconds")
  seconds = float(seconds)
  frames = seconds * 20
  if (not np.isfinite(frames) or seconds < HISTORY_S + 2 or abs(frames-round(frames)) > 1e-8):
    raise ValueError("duration must cover history+2 seconds and match the 20 Hz frame grid")
  frames = int(round(frames))
  result, rejected = [], Counter()
  capture = data["raw_encode_t"]
  frame_id, segment_id = data["raw_encode_frame_id"], data["raw_encode_segment_id"]
  if not len(capture) or any(len(value) != len(capture) for value in (frame_id, segment_id)):
    return result, {"missing_camera_mapping": 1}
  order = np.argsort(capture)
  capture, frame_id, segment_id = capture[order], frame_id[order], segment_id[order]
  for camera_start in range(0, len(capture) - frames + 1, 100):
    z = slice(camera_start, camera_start + frames)
    camera_t = capture[z]
    if (not np.isfinite(camera_t).all() or np.any(np.abs(np.diff(camera_t) - .05) > .005)
        or np.any(np.diff(frame_id[z]) != 1) or np.any(np.diff(segment_id[z]) != 1)):
      rejected["camera_gap_or_identity"] += 1
      continue
    start = float(camera_t[0])
    indices = np.flatnonzero((data["t"] >= start) & (data["t"] < start + seconds))
    if len(indices) < frames - 1 or np.any(np.abs(np.diff(data["t"][indices]) - .05) > .002):
      rejected["motion_coverage"] += 1
      continue
    scored = indices[data["t"][indices] >= start + HISTORY_S]
    fields = ("v", "a", "steering_pressed", "brake_pressed", "gas_pressed", "calibrated")
    if (not all(np.isfinite(data[key][indices]).all() for key in fields)
        or not np.isfinite(data["rpy_calib"][indices]).all()
        or not np.all(data["calibrated"][indices] == 1)
        or not np.all(data["calibration_valid"][indices] == 1)):
      rejected["calibration_or_vehicle_missing"] += 1
      continue
    if any(np.any(data[key][scored] != 0) for key in ("steering_pressed", "brake_pressed", "gas_pressed")):
      rejected["driver_intervention"] += 1
      continue
    common = {"segment": meta["segment"], "route": meta["route"], "camera_start_s": start - float(capture[0]),
              "seconds": seconds, "history_s": HISTORY_S, "capture_start_t": start,
              "first_frame_id": int(frame_id[z][0]), "last_frame_id": int(frame_id[z][-1]),
              "scored_rows": len(scored), "source_log_sha256": meta["source_log_sha256"]}
    v, a = data["v"][scored], data["a"][scored]
    first_v, last_v = float(np.median(v[:20])), float(np.median(v[-20:]))
    long_ok = np.all(data["long_active"][scored] == 1) and np.all(data["valid_long"][scored] == 1)
    lat_ok = (np.all(data["lat_active"][scored] == 1) and np.all(data["valid_lat"][scored] == 1)
              and np.isfinite(data["yaw"][scored]).all() and np.isfinite(data["torque"][scored]).all()
              and np.all(np.abs(data["torque"][scored]) <= 409))
    if long_ok and first_v >= 3 and last_v < .3 and np.sum(v < .3) >= 20:
      result.append(dict(common, kind="stopping", score=first_v-last_v,
                         evidence={"initial_speed_mps": first_v, "final_speed_mps": last_v,
                                   "observed_travel_m": float(np.sum(v) * .05)}))
    if long_ok and last_v-first_v >= 2 and np.sum(a > .3) >= 40:
      result.append(dict(common, kind="acceleration", score=last_v-first_v,
                         evidence={"initial_speed_mps": first_v, "final_speed_mps": last_v,
                                   "acceleration_p95_mps2": float(np.percentile(a, 95))}))
    if lat_ok and np.min(v) >= 3:
      lateral = np.abs(v * data["yaw"][scored])
      if np.sum(lateral >= .5) >= 40:
        result.append(dict(common, kind="turning", score=float(np.percentile(lateral, 95)),
                           evidence={"lateral_accel_p95_mps2": float(np.percentile(lateral, 95)),
                                     "torque_abs_p95": float(np.percentile(np.abs(data["torque"][scored]), 95))}))
  return result, dict(rejected)


def select_diverse(candidates):
  """Deterministic ranking, one distinct route per behavior; keep every row."""
  selected, routes = [], set()
  for kind in KINDS:
    ranked = sorted((row for row in candidates if row["kind"] == kind),
                    key=lambda row: (-row["score"], row["segment"], row["camera_start_s"]))
    winner = next((row for row in ranked if row["route"] not in routes), None)
    if winner is not None:
      selected.append(winner)
      routes.add(winner["route"])
  return selected


def training_paths(dataset, reservation, cohort):
  split = reservation["cohorts"][cohort]["split"]
  if not split or set(split) != {"train", "validation", "test"}:
    raise ValueError("frozen whole-route split required")
  flat = [route for key in split for route in split[key]]
  if len(flat) != len(set(flat)):
    raise ValueError("route split is not disjoint")
  paths = [(path, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))) for path in sorted(dataset.glob("*.npz"))]
  enrich_route_metadata(paths)
  return [(path, meta) for path, meta in paths if meta["route"] in split["train"] and cohort_key(meta) == cohort]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--splits", type=Path, required=True)
  parser.add_argument("--cohort", required=True)
  parser.add_argument("--inventory", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  if args.out.exists() or args.out.resolve().is_relative_to(args.dataset.resolve()):
    parser.error("output must be new and outside the dataset")
  reservation = json.loads(args.splits.read_text(encoding="utf-8"))
  inventory = {row["name"]: row for row in json.loads(args.inventory.read_text(encoding="utf-8"))["files"]}
  candidates, coverage = [], Counter()
  for path, meta in training_paths(args.dataset, reservation, args.cohort):
    coverage["training_segments"] += 1
    cameras = inventory.get(meta["segment"], {}).get("cameras", {})
    if not all(cameras.get(name, {}).get("available_device_bytes", 0) > 0 for name in ("fcamera.hevc", "ecamera.hevc")):
      coverage["missing_device_camera_pair"] += 1
      continue
    with np.load(path, allow_pickle=False) as archive:
      data = hold_calibration({key: archive[key] for key in archive.files})
    rows, rejected = segment_candidates(meta, data)
    candidates.extend(rows)
    coverage.update(rejected)
  selected = select_diverse(candidates)
  result = {"cohort": args.cohort, "partition": "train", "selected": selected, "candidates": candidates,
            "coverage": dict(coverage), "missing_behaviors": sorted(set(KINDS) - {row["kind"] for row in selected}),
            "splits_sha256": sha256(args.splits), "inventory_sha256": sha256(args.inventory),
            "selector_sha256": sha256(Path(__file__)),
            "criteria": {"seconds": SECONDS, "genuine_history_s": HISTORY_S, "axis_valid_fraction_min": 1.,
                         "continuous_post_history_control": True, "driver_override_rows_allowed": 0,
                         "steer_max": 409},
            "scope": "Observed-motion training footage acquisition only; not stop-line GT, model readiness or tune improvement"}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
  print(json.dumps({"selected": selected, "coverage": dict(coverage), "missing_behaviors": result["missing_behaviors"]}))


if __name__ == "__main__":
  main()
