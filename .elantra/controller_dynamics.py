"""Empirical transient lateral response for offline prediction experiments.

The model stores binned median 50 ms state increments, not fitted response slopes.
It is a predictive hypothesis learned from the supplied training records. Logged
feedback, road geometry and unobserved disturbances confound these observations;
neither fitting nor inverting this table establishes causal system identification.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from lateral_report import band_of

DT = .05
STATE_EDGES = np.array([-4., -2.5, -1.5, -.75, -.25, .25, .75, 1.5, 2.5, 4.])
COMMAND_EDGES = np.array([-409.5, -330., -250., -170., -90., -30., 30., 90., 170., 250., 330., 409.5])
MIN_SAMPLES = 100
MIN_ROUTES = 2
MIN_COMMANDS = 3
# At least 0.4 m/s^3 of distinguishable command-dependent response across the
# supported command range; tiny floating-point or measurement differences fail.
MIN_DELTA_RANGE = .02
REQUIRED = ("t", "v", "yaw", "roll", "torque", "lat_active", "steering_pressed", "brake_pressed",
            "gas_pressed", "calibrated", "calibration_valid", "model_valid", "valid_lat")


def fit(records, delay_s):
  """Learn from supplied training tuples ``(metadata, held-calibration arrays)``.

  The transition is s[i+1]-s[i], s=v*yaw-9.81*roll; its command is the actual
  recorded torque at i-delay. Every row from that command through the measured
  effect must be fresh, calibrated, engaged, free of driver overrides and within
  409 counts. Unsupported input records/cells are named in the returned support.
  """
  if not np.isfinite(delay_s) or delay_s < 0 or abs(delay_s / DT - round(delay_s / DT)) > 1e-8:
    raise ValueError("delay_s must be a finite nonnegative multiple of the 20 Hz sample period")
  delay_n = int(round(delay_s / DT))
  groups = defaultdict(list)
  route_support = defaultdict(set)
  excluded = Counter()
  support = {"training_routes": [], "training_records": 0, "candidate_transitions": 0, "usable_transitions": 0,
             "missing_fields": [], "malformed_records": [], "cells": [], "accepted_cells": 0,
             "rejected_cells": [], "boundary_cells": 0,
             "min_samples": MIN_SAMPLES, "min_routes": MIN_ROUTES, "min_commands": MIN_COMMANDS,
             "min_delta_range": MIN_DELTA_RANGE,
             "shadow_diagnostics": "Only training operating points are counted; held-out prediction is required."}
  training_routes = set()
  for meta, data in records:
    route = meta.get("route")
    if not isinstance(route, str) or not route:
      raise ValueError("each training record requires a nonempty metadata route identity")
    training_routes.add(route)
    support["training_records"] += 1
    missing = sorted(set(REQUIRED) - set(data))
    if missing:
      support["missing_fields"].append({"route": route, "fields": missing})
      continue
    n = len(data["t"])
    if any(np.asarray(data[key]).shape != (n,) for key in REQUIRED):
      support["malformed_records"].append({"route": route, "error": "required arrays must share the 1D timestamp shape"})
      continue
    if n < 2:
      excluded["too_short"] += n
      continue
    support["candidate_transitions"] += n - 1
    clean = np.logical_and.reduce([np.isfinite(data[key]) for key in REQUIRED])
    for key in ("lat_active", "calibrated", "calibration_valid", "model_valid", "valid_lat"):
      clean &= data[key] == 1
    for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
      clean &= data[key] == 0
    clean &= (np.abs(data["torque"]) <= 409) & (data["v"] >= 3)
    if "eps_fault" in data:
      clean &= ~(np.isfinite(data["eps_fault"]) & (data["eps_fault"] != 0))
    state = data["v"] * data["yaw"] - 9.81 * data["roll"]
    clean &= np.isfinite(state)
    indices = np.arange(n - 1)
    excluded["history_unavailable"] += int(np.sum(indices < delay_n))
    indices = indices[indices >= delay_n]
    if not len(indices):
      continue
    lo = indices - delay_n
    dirty_prefix = np.r_[0, np.cumsum(~clean)]
    bad = dirty_prefix[indices + 2] - dirty_prefix[lo] != 0
    excluded["unclean_interval"] += int(bad.sum())
    indices, lo = indices[~bad], lo[~bad]
    gaps = ~np.isfinite(np.diff(data["t"])) | (np.abs(np.diff(data["t"]) - DT) > .002)
    gap_prefix = np.r_[0, np.cumsum(gaps)]
    bad = gap_prefix[indices + 1] - gap_prefix[lo] != 0
    excluded["time_gap"] += int(bad.sum())
    indices = indices[~bad]
    state_bins = np.searchsorted(STATE_EDGES, state[indices], side="right") - 1
    command = data["torque"][indices - delay_n]
    command_bins = np.searchsorted(COMMAND_EDGES, command, side="right") - 1
    valid_bins = (state_bins >= 0) & (state_bins < len(STATE_EDGES) - 1)
    excluded["outside_state_bins"] += int((~valid_bins).sum())
    indices, state_bins, command, command_bins = [a[valid_bins] for a in (indices, state_bins, command, command_bins)]
    valid_bins = (command_bins >= 0) & (command_bins < len(COMMAND_EDGES) - 1)
    excluded["outside_command_bins"] += int((~valid_bins).sum())
    indices, state_bins, command, command_bins = [a[valid_bins] for a in (indices, state_bins, command, command_bins)]
    bands = np.array([band_of(float(v)) for v in data["v"][indices]], dtype=object)
    known_band = bands != None  # noqa: E711 - object array comparison is elementwise.
    excluded["unknown_speed_band"] += int((~known_band).sum())
    indices, state_bins, command, command_bins, bands = [a[known_band] for a in (indices, state_bins, command, command_bins, bands)]
    support["usable_transitions"] += len(indices)
    values = np.column_stack((command, state[indices + 1] - state[indices], state[indices], data["v"][indices]))
    keys = set(zip(bands, state_bins, command_bins, strict=True))
    for band, state_bin, command_bin in keys:
      mask = (bands == band) & (state_bins == state_bin) & (command_bins == command_bin)
      key = band, int(state_bin), int(command_bin)
      groups[key].append(values[mask])
      route_support[key].add(route)
  by_state = defaultdict(list)
  for (band, state_bin, command_bin), chunks in sorted(groups.items()):
    values = np.concatenate(chunks)
    row = {"speed_band": band, "state_bin": state_bin, "command_bin": command_bin,
           "n": len(values), "routes": len(route_support[band, state_bin, command_bin]),
           "command": float(np.median(values[:, 0])), "delta": float(np.median(values[:, 1])),
           "delta_p25": float(np.percentile(values[:, 1], 25)), "delta_p75": float(np.percentile(values[:, 1], 75)),
           "delta_min": float(values[:, 1].min()), "delta_max": float(values[:, 1].max()),
           "state_min": float(values[:, 2].min()), "state_max": float(values[:, 2].max()),
           "state_median": float(np.median(values[:, 2])),
           "speed_min": float(values[:, 3].min()), "speed_max": float(values[:, 3].max())}
    support["cells"].append(row)
    if row["n"] >= MIN_SAMPLES and row["routes"] >= MIN_ROUTES:
      by_state[band, state_bin].append(row)
  cells = {}
  for (band, state_bin), rows in sorted(by_state.items()):
    command = np.array([row["command"] for row in rows])
    delta = np.array([row["delta"] for row in rows])
    reason = None
    if len(rows) < MIN_COMMANDS:
      reason = "fewer_than_three_supported_commands"
    elif not np.all(np.diff(command) > 0) or not np.all(np.diff(delta) < 0):
      reason = "nonmonotonic_command_response"
    elif delta[0] - delta[-1] < MIN_DELTA_RANGE:
      reason = "insufficient_delta_range"
    if reason:
      support["rejected_cells"].append({"speed_band": band, "state_bin": state_bin, "reason": reason})
      continue
    cell = {key: [row[key] for row in rows] for key in
            ("command", "delta", "delta_p25", "delta_p75", "delta_min", "delta_max", "n", "routes")}
    cell.update(state_min=min(row["state_min"] for row in rows), state_max=max(row["state_max"] for row in rows),
                state_median=float(np.median([row["state_median"] for row in rows])),
                state_bin_low=float(STATE_EDGES[state_bin]), state_bin_high=float(STATE_EDGES[state_bin + 1]),
                speed_min=min(row["speed_min"] for row in rows), speed_max=max(row["speed_max"] for row in rows))
    cells.setdefault(band, {})[str(state_bin)] = cell
    support["accepted_cells"] += 1
    support["boundary_cells"] += int(state_bin in (0, len(STATE_EDGES) - 2) or
                                     rows[0]["command_bin"] == 0 or rows[-1]["command_bin"] == len(COMMAND_EDGES) - 2)
  support["training_routes"] = sorted(training_routes)
  support["excluded"] = dict(excluded)
  return {"kind": "median_transition", "dt": DT, "delay_s": float(delay_s),
          "state_edges": STATE_EDGES.tolist(), "command_edges": COMMAND_EDGES.tolist(), "cells": cells,
          "support": support,
          "limitations": "Predictive transient medians only; observational feedback is confounded; no causal identification or road validation."}


def _cell(model, speed, state):
  if not np.isfinite([speed, state]).all():
    return None
  edges = model["state_edges"]
  index = int(np.searchsorted(edges, state, side="right") - 1)
  cell = model["cells"].get(band_of(float(speed)), {}).get(str(index))
  if cell is None or state < cell["state_min"] - 1e-9 or state > cell["state_max"] + 1e-9:
    return None
  return cell


def forward(model, speed, state, command):
  """Predict the next corrected state inside supported state/command bounds."""
  cell = _cell(model, speed, state)
  if cell is None or not np.isfinite(command) or command < cell["command"][0] or command > cell["command"][-1]:
    return float("nan")
  return float(state + np.interp(command, cell["command"], cell["delta"]))


def inverse(model, speed, state, target_next):
  """Invert a supported next-state request, without extrapolation or saturation."""
  cell = _cell(model, speed, state)
  if cell is None or not np.isfinite(target_next):
    return float("nan")
  delta = target_next - state
  if delta < cell["delta"][-1] - 1e-12 or delta > cell["delta"][0] + 1e-12:
    return float("nan")
  # Reachable endpoints can differ by a rounding bit after subtracting state.
  delta = np.clip(delta, cell["delta"][-1], cell["delta"][0])
  return float(np.interp(delta, cell["delta"][::-1], cell["command"][::-1]))


def reachable(model, speed, state):
  """Return empirical minimum/maximum next state, or None without support."""
  cell = _cell(model, speed, state)
  if cell is None:
    return None
  return float(state + cell["delta"][-1]), float(state + cell["delta"][0])
