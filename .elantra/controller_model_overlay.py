"""Capture-aligned model paths and fresh RDF output admission for evidence renders."""

import numpy as np


def measured_forecasts(data, seconds=3, stride=10):
  """Display-only measured future labels, including visibly flagged driver actions.

  Control engagement is not a motion sensor validity condition. Never pass these
  future labels to model inference or controller search.
  """
  from controller_tune import DT, integrate

  count = int(round(seconds / DT)) + 1
  valid = np.isfinite(data["v"]) & np.isfinite(data["yaw"])
  for key in ("car_state_valid", "location_valid", "location_measurement_valid", "calibration_valid", "calibrated"):
    valid &= data[key] == 1
  paths, timestamps = [], []
  for start in range(0, len(data["t"]), stride):
    end = min(len(data["t"]), start + count)
    speed, yaw = np.full(count, np.nan), np.full(count, np.nan)
    good = valid[start:end].copy()
    good[1:] &= np.abs(np.diff(data["t"][start:end]) - DT) <= .002
    speed[:end-start] = np.where(good, data["v"][start:end], np.nan)
    yaw[:end-start] = np.where(good, data["yaw"][start:end], np.nan)
    path = integrate(speed, yaw, DT)
    missing = np.flatnonzero(~good)
    if len(missing):
      path[missing[0]:] = np.nan
    timestamps.append(data["t"][start])
    paths.append(path)
  gt = np.asarray(paths).reshape(-1, count, 2)
  return {"forecast_t": np.asarray(timestamps), "forecast_dt": np.asarray(DT), "gt_xy": gt,
          "baseline_xy": np.full_like(gt, np.nan), "tuned_xy": np.full_like(gt, np.nan)}


class RenderError(ValueError):
  """Missing or ambiguous evidence must stop the render."""


def causal_index(times, timestamp, max_age=0.2):
  times = np.asarray(times)
  i = int(np.searchsorted(times, timestamp, side="right") - 1)
  return i if i >= 0 and timestamp - times[i] <= max_age else None


def contiguous_path(xy, valid=None):
  xy = np.asarray(xy, dtype=float)
  if xy.ndim != 2 or xy.shape[1] != 2:
    raise RenderError("trajectory must have shape [horizon,2]")
  good = np.isfinite(xy).all(axis=1)
  if valid is not None:
    good &= np.asarray(valid, dtype=bool)
  bad = np.flatnonzero(~good)
  return xy[:int(bad[0]) if len(bad) else len(xy)]


def capture_path(data, xy, horizon, capture_t, timestamp, max_age=.2):
  """Move a capture-origin path into the displayed measured ego frame using past motion only."""
  empty = np.empty((0, 2))
  age = timestamp - capture_t
  if not np.isfinite(age) or not 0 <= age <= max_age + 1e-8 or len(xy) < 2:
    return empty
  t = data["t"]
  if capture_t < t[0] or timestamp > t[-1]:
    return empty
  nodes = np.r_[capture_t, t[(t > capture_t) & (t < timestamp)], timestamp]
  indices = np.searchsorted(t, nodes[:-1], side="right") - 1
  last = int(np.searchsorted(t, timestamp, side="right") - 1)
  observed = slice(indices[0], last+1)
  if (np.any(np.diff(t[observed]) > .075 + 1e-8) or capture_t - t[indices[0]] > .075 + 1e-8
      or timestamp - t[last] > .075 + 1e-8):
    return empty
  for key in ("car_state_valid", "location_valid", "location_measurement_valid"):
    if key not in data or not np.all(data[key][observed] == 1):
      return empty
  v, yaw = data["v"][indices], data["yaw"][indices]
  if not np.isfinite([data["v"][observed], data["yaw"][observed]]).all():
    return empty
  dt = np.diff(nodes)
  turns = yaw * dt
  heading = np.cumsum(turns) - turns / 2
  displacement = np.array([np.sum(v * dt * np.cos(heading)), np.sum(v * dt * np.sin(heading))])
  angle = float(np.sum(turns))
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  h = np.asarray(horizon)[:len(xy)]
  if not np.isfinite(h).all() or np.any(np.diff(h) <= 0) or h[0] > age or h[-1] <= age:
    return empty
  present = np.array([np.interp(age, h, xy[:, axis]) for axis in (0, 1)])
  future = np.vstack((present, xy[h > age])) - xy[0] - displacement
  return future @ rotation


def model_path(data, timestamp):
  if "model_t" not in data or "model_x" not in data:
    return np.empty((0, 2))
  i = causal_index(data["t"], timestamp)
  if (i is None or data.get("model_valid", np.zeros(len(data["t"])))[i] != 1
      or not np.isfinite(data["model_t"][i]) or not 0 <= timestamp - data["model_t"][i] <= .2):
    return np.empty((0, 2))
  xy = np.column_stack((data["model_x"][i], data["model_y"][i]))
  horizon = np.asarray(data["model_horizon"][i])
  valid = np.isfinite(horizon) & (horizon >= 0)
  if len(horizon) > 1:
    valid[1:] &= np.diff(horizon) > 0
  capture = data.get("model_capture_t", np.full(len(data["t"]), np.nan))[i]
  return capture_path(data, contiguous_path(xy, valid), horizon, capture, timestamp)


def validate_rdf_output(rdf, data):
  from rdf_bench import HISTORY_WARMUP_FRAMES, WHOLE_SHA

  required = ("input_t", "output_ready_t", "narrow_t", "frame_id", "position", "horizon", "scored_mask",
              "desired_curvature", "desired_acceleration", "model_sha256")
  missing = [key for key in required if key not in rdf]
  if missing:
    raise RenderError("missing fresh RDF arrays: " + ", ".join(missing))
  n = len(rdf["input_t"])
  if str(rdf["model_sha256"]) != WHOLE_SHA or rdf["position"].shape != (n, 33, 3) or rdf["horizon"].shape != (33,):
    raise RenderError("RDF identity or native plan shape mismatch")
  for key in ("input_t", "output_ready_t", "narrow_t"):
    if rdf[key].shape != (n,) or not np.isfinite(rdf[key]).all() or np.any(np.diff(rdf[key]) <= 0):
      raise RenderError("fresh RDF clocks must be finite and strictly increasing")
  for key in ("frame_id", "scored_mask", "desired_curvature", "desired_acceleration"):
    if rdf[key].shape != (n,):
      raise RenderError("fresh RDF arrays must match the frame count")
  if (np.any(rdf["narrow_t"] > rdf["input_t"]) or np.any(rdf["output_ready_t"] < rdf["input_t"])
      or np.any(rdf["scored_mask"][:HISTORY_WARMUP_FRAMES])):
    raise RenderError("fresh RDF output availability or genuine history warmup violated")
  frames = data.get("raw_encode_frame_id", np.array([]))
  times = data.get("raw_encode_t", np.array([]))
  for frame_id, capture in zip(rdf["frame_id"], rdf["narrow_t"], strict=True):
    matched = times[frames == frame_id]
    if len(matched) != 1 or abs(float(matched[0]) - capture) > 1e-6:
      raise RenderError("fresh RDF frame identity does not exactly match this footage")


def rdf_path(rdf, data, timestamp):
  i = causal_index(rdf["output_ready_t"], timestamp, .2)
  if i is None or not rdf["scored_mask"][i]:
    return np.empty((0, 2)), None
  path = capture_path(data, contiguous_path(rdf["position"][i, :, :2]), rdf["horizon"],
                      float(rdf["narrow_t"][i]), timestamp)
  return path, i if len(path) else None


def validate_native_output(native, receipt, rdf, rdf_sha, log_sha, output_sha):
  """Bind native commands to this exact model output/log and reject inadmissible rows."""
  for key, actual in (("rdf_output_sha256", rdf_sha), ("source_log_sha256", log_sha), ("output_npz_sha256", output_sha)):
    if receipt.get(key) != actual:
      raise RenderError(f"native evidence hash mismatch: {key}")
  if (receipt.get("mode") != "native_replay" or receipt.get("dt") != .01
      or receipt.get("no_actuation") is not True or receipt.get("no_params_writes") is not True
      or not isinstance(receipt.get("initialization"), str) or not receipt["initialization"]):
    raise RenderError("native evidence requires the diagnostic 100 Hz replay contract")
  required = ("t", "admitted", "initialization_uncertain", "native_accel", "native_state", "should_stop",
              "rdf_index", "rdf_accel_target", "rdf_capture_age_s")
  if any(key not in native for key in required):
    raise RenderError("missing native evidence arrays")
  n = len(native["t"])
  if n < 2 or receipt.get("rows") != n or any(np.asarray(native[key]).shape != (n,) for key in required):
    raise RenderError("native evidence shapes/count must match the receipt")
  numeric = ("t", "native_accel", "native_state", "rdf_accel_target", "rdf_capture_age_s")
  if any(np.asarray(native[key]).dtype.kind not in "fi" for key in numeric):
    raise RenderError("native evidence clocks/commands must be real numeric arrays")
  if not np.isfinite(native["t"]).all() or not np.allclose(np.diff(native["t"]), .01, atol=1e-8, rtol=0):
    raise RenderError("native evidence clocks must be continuous and increasing at 100 Hz")
  if any(np.asarray(native[key]).dtype.kind != "b" for key in ("admitted", "initialization_uncertain", "should_stop")):
    raise RenderError("native evidence flags must be explicit booleans")
  admitted = native["admitted"]
  for key in ("native_accel", "native_state", "rdf_accel_target", "rdf_capture_age_s"):
    if not np.isfinite(native[key][admitted]).all() or not np.isnan(native[key][~admitted]).all():
      raise RenderError("native commands/states must be finite only on admitted rows")
  if not np.isin(native["native_state"][admitted], (0, 1, 2)).all() or np.asarray(native["rdf_index"]).dtype.kind != "i":
    raise RenderError("invalid native state or RDF index")
  if "should_stop" not in rdf or np.asarray(rdf["should_stop"]).dtype.kind != "b":
    raise RenderError("native evidence requires exact boolean RDF shouldStop outputs")
  indices = native["rdf_index"][admitted]
  expected = np.searchsorted(rdf["output_ready_t"], native["t"][admitted], side="right") - 1
  if (np.any(indices < 0) or np.any(indices >= len(rdf["input_t"])) or not np.array_equal(indices, expected)):
    raise RenderError("native evidence used future or mismatched RDF outputs")
  age = native["t"][admitted] - rdf["narrow_t"][indices]
  if (not rdf["scored_mask"][indices].all() or np.any(age < 0) or np.any(age > .2 + 1e-8)
      or not np.allclose(age, native["rdf_capture_age_s"][admitted], atol=1e-8, rtol=0)
      or not np.array_equal(native["rdf_accel_target"][admitted], rdf["desired_acceleration"][indices])
      or not np.array_equal(native["should_stop"][admitted], rdf["should_stop"][indices])):
    raise RenderError("native evidence violated RDF warmup, freshness, target or shouldStop identity")


def native_index(native, timestamp):
  """Select only the latest native tick; never substitute a prior admitted row."""
  index = causal_index(native["t"], timestamp, .01000001)
  return index if index is not None and native["admitted"][index] else None
