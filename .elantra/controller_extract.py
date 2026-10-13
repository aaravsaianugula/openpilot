#!/usr/bin/env python3
"""Extract recorded controller/model/vehicle signals without an openpilot runtime.

This is observation-only. Each output row is a 20 Hz causal as-of join of messages
sorted by logMonoTime, with a maximum service age of 200 ms. Missing or stale values
remain NaN. Recorded model paths are predictions, and measured yaw is angular
velocity, not carState.yawRate. No counterfactual or hardware claim is made here.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from lateral_report import band_of, decode_lkas11, decode_toiflt, routes_under


class ExtractionError(RuntimeError):
  """A source cannot satisfy the extraction contract."""


@dataclass
class Sample:
  service: str
  t: float
  values: dict
  valid: bool


SERVICE_FIELDS = {
  "carState": ("v", "a", "steering_pressed", "brake_pressed", "gas_pressed", "driver_torque", "steering_angle"),
  "carControl": ("torque_request", "torque_fraction", "accel_request", "lat_active", "long_active"),
  "carOutput": ("torque", "accel_command"),
  "controlsState": ("desired_curvature", "actual_curvature"),
  "liveLocationKalman": ("yaw", "roll", "location_measurement_valid"),
  "extrinsicsCalibration": ("calibrated", "rpy_calib", "calibration_t", "calibration_height"),
  "longitudinalPlan": ("accel_target", "planner_model_t"),
  "modelV2": ("model_t", "model_frame_id", "model_x", "model_y", "model_z", "model_horizon",
              "model_frame_age", "model_big", "model_frame_id_trusted"),
  "narrowRoadCameraState": ("camera_t", "camera_frame_id"),
  "narrowRoadEncodeIdx": ("encode_t", "encode_event_t", "encode_frame_id", "encode_segment_num",
                          "encode_segment_id", "encode_segment_id_encode", "encode_id"),
  "canTorque": ("can_torque",),
  "epsStatus": ("eps_fault",),
  "lateralTorqueParameters": ("learned_torque_factor", "learned_torque_friction", "learned_torque_offset",
                              "learner_valid", "learner_use_params"),
  "lateralDelay": ("learned_lateral_delay", "delay_estimated"),
}
VALID_FIELDS = {
  "carState": "car_state_valid", "carControl": "car_control_valid", "carOutput": "car_output_valid",
  "controlsState": "controls_valid", "liveLocationKalman": "location_valid",
  "extrinsicsCalibration": "calibration_valid", "longitudinalPlan": "planner_valid",
  "modelV2": "model_valid", "narrowRoadCameraState": "camera_valid",
  "narrowRoadEncodeIdx": "encode_valid", "canTorque": "can_torque_valid", "epsStatus": "eps_valid",
  "lateralTorqueParameters": "learner_event_valid", "lateralDelay": "delay_event_valid",
}
VECTOR_WIDTHS = {"model_x": 33, "model_y": 33, "model_z": 33, "model_horizon": 33, "rpy_calib": 3,
                 "calibration_height": 1}
CP_FIELDS = (
  "brand", "carFingerprint", "flags", "openpilotLongitudinalControl", "pcmCruise", "safetyConfigs",
  "longitudinalTuning", "lateralTuning", "lateralParams", "mass", "wheelbase", "centerToFront",
  "steerRatio", "steerRatioRear", "steerActuatorDelay", "longitudinalActuatorDelay", "minSteerSpeed",
  "tireStiffnessFactor", "tireStiffnessFront", "tireStiffnessRear", "rotationalInertia", "wheelSpeedFactor",
  "steerControlType", "stopAccel", "dashcamOnly", "passive", "radarUnavailable", "networkLocation",
)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def decompress_log(path: Path) -> bytes:
  if not path.is_file():
    raise ExtractionError(f"missing source log: {path}")
  payload = path.read_bytes()
  if not payload:
    raise ExtractionError(f"empty source log: {path}")
  try:
    if path.suffix == ".zst":
      import zstandard

      chunks = []
      remaining = payload
      while remaining:
        decoder = zstandard.ZstdDecompressor().decompressobj()
        chunks.append(decoder.decompress(remaining))
        if not decoder.eof:
          raise ExtractionError(f"truncated compressed log: {path}")
        remaining = decoder.unused_data
      raw = b"".join(chunks)
    elif path.suffix == ".bz2":
      raw = bz2.decompress(payload)
    else:
      raw = payload
  except ExtractionError:
    raise
  except Exception as exc:
    raise ExtractionError(f"corrupt compressed log {path}: {type(exc).__name__}: {exc}") from exc
  if not raw:
    raise ExtractionError(f"empty decompressed log: {path}")
  return raw


def inventory(roots: list[Path]) -> dict:
  rows = []
  errors = []
  root_info = []
  for root in roots:
    root = root.resolve()
    if not root.is_dir():
      errors.append({"root": str(root), "error": "missing source root"})
      root_info.append({"path": str(root), "exists": False})
      continue
    known_logs = {str(Path(p).parent.resolve()) for paths in routes_under(root).values() for p in paths}
    count = 0
    for folder in sorted(root.iterdir()):
      if not folder.is_dir():
        continue
      route, sep, index = folder.name.rpartition("--")
      if not sep or not index.isdigit() or route.startswith("_"):
        continue
      files = {p.name: {"path": str(p.resolve()), "bytes": p.stat().st_size}
               for p in folder.iterdir() if p.is_file() and p.name in
               ("rlog.zst", "rlog.bz2", "rlog", "qlog.zst", "fcamera.hevc", "ecamera.hevc", "dcamera.hevc", "qcamera.ts")}
      log = next((files[name]["path"] for name in ("rlog.zst", "rlog.bz2", "rlog") if name in files), None)
      rows.append({"root": str(root), "path": str(folder.resolve()), "segment": folder.name,
                   "route": route, "index": int(index), "log": log, "files": files,
                   "known_to_routes_under": str(folder.resolve()) in known_logs})
      count += 1
    root_info.append({"path": str(root), "exists": True, "segments": count})
  coverage = {"segments": len(rows), "logical_segments": len({r["segment"] for r in rows}),
              "routes": len({r["route"] for r in rows}),
              "with_rlog": sum(r["log"] is not None for r in rows)}
  for name in ("fcamera", "ecamera", "dcamera"):
    coverage[f"with_{name}"] = sum(f"{name}.hevc" in r["files"] for r in rows)
  coverage["with_rlog_and_both_road_cameras"] = sum(
    r["log"] is not None and "fcamera.hevc" in r["files"] and "ecamera.hevc" in r["files"] for r in rows)
  missing_sources = [{"segment": row["segment"], "path": row["path"],
                      "missing": (["rlog"] if row["log"] is None else []) +
                                 [name for name in ("fcamera.hevc", "ecamera.hevc") if name not in row["files"]]}
                     for row in rows if row["log"] is None or "fcamera.hevc" not in row["files"] or "ecamera.hevc" not in row["files"]]
  return {"roots": root_info, "coverage": coverage, "segments": rows, "missing_sources": missing_sources, "errors": errors}


def select_segments(rows: list[dict], limit: int | None) -> list[dict]:
  """Choose whole route pairs, spread across the route range; never split a route population."""
  unique = {}
  for row in rows:
    if row["log"]:
      unique.setdefault(row["segment"], row)
  grouped: dict[str, list[dict]] = {}
  for row in unique.values():
    grouped.setdefault(row["route"], []).append(row)
  for route in grouped:
    grouped[route].sort(key=lambda row: row["index"])
  if limit is None:
    return [row for route in sorted(grouped) for row in grouped[route]]
  if limit < 1:
    raise ExtractionError("--limit must be positive")
  routes = sorted(grouped)
  if not routes:
    return []
  route_count = min(len(routes), max(1, limit // 2))
  chosen = [routes[i] for i in np.linspace(0, len(routes) - 1, route_count, dtype=int)]
  selected = []
  for route in chosen:
    candidates = grouped[route]
    zero = next((row for row in candidates if row["index"] == 0), None)
    nonzero = next((row for row in candidates if row["index"] > 0), None)
    selected.extend(row for row in (zero, nonzero) if row is not None)
  # Fill spare capacity, retaining selected routes before bringing in another route.
  for route in chosen + [r for r in routes if r not in chosen]:
    for row in grouped[route]:
      if len(selected) >= limit:
        return selected[:limit]
      if row not in selected:
        selected.append(row)
  return selected[:limit]


def can_values(frames) -> tuple[dict, Counter]:
  values = {}
  counts = Counter()
  for frame in frames:
    address, src, dat = int(frame["address"]), int(frame["src"]), bytes(frame["dat"])
    if address == 0x340 and src == 128:
      decoded = decode_lkas11(dat)
      if decoded is None:
        counts["short_lkas_frames"] += 1
        continue
      values["can_torque"] = float(decoded[0])
      counts["tx_lkas_frames"] += 1
      counts["tx_over_409_frames"] += abs(decoded[0]) > 409
      counts["tx_at_409_frames"] += abs(decoded[0]) == 409
    elif address == 0x251 and src == 0:
      fault = decode_toiflt(dat)
      if fault is None:
        counts["short_eps_frames"] += 1
        continue
      values["eps_fault"] = float(fault)
      counts["eps_frames"] += 1
      counts["eps_fault_frames"] += fault
  return values, counts


def align_samples(samples: list[Sample], rate: float = 20.0, max_age: float = 0.2) -> tuple[dict, dict]:
  samples.sort(key=lambda sample: sample.t)
  states = [sample for sample in samples if sample.service == "carState"]
  if not states:
    raise ExtractionError("no carState messages in source log")
  t = states[0].t + np.arange(int(np.floor((states[-1].t - states[0].t) * rate + 1e-7)) + 1) / rate
  arrays = {"t": t}
  grouped: dict[str, list[Sample]] = {}
  for sample in samples:
    grouped.setdefault(sample.service, []).append(sample)
  for service, fields in SERVICE_FIELDS.items():
    rows = grouped.get(service, [])
    for field in fields:
      arrays[field] = np.full((len(t), VECTOR_WIDTHS[field]) if field in VECTOR_WIDTHS else len(t), np.nan)
    arrays[VALID_FIELDS[service]] = np.full(len(t), np.nan)
    arrays[f"age_{service}"] = np.full(len(t), np.nan)
    if not rows:
      continue
    source_t = np.array([row.t for row in rows])
    indices = np.searchsorted(source_t, t, side="right") - 1
    for i in np.flatnonzero(indices >= 0):
      sample = rows[indices[i]]
      age = t[i] - sample.t
      arrays[f"age_{service}"][i] = age
      if age > max_age + 1e-9:
        continue
      arrays[VALID_FIELDS[service]][i] = float(sample.valid)
      for field in fields:
        if field not in sample.values:
          continue
        value = sample.values[field]
        if field in VECTOR_WIDTHS:
          count = min(len(value), VECTOR_WIDTHS[field])
          arrays[field][i, :count] = value[:count]
        else:
          arrays[field][i] = value
  required = ("v", "a", "yaw", "roll", "torque", "desired_curvature", "lat_active")
  arrays["valid_lat"] = np.logical_and.reduce([np.isfinite(arrays[name]) for name in required])
  for name in ("car_state_valid", "car_control_valid", "car_output_valid", "controls_valid", "location_valid",
               "location_measurement_valid", "calibration_valid", "calibrated", "model_valid"):
    arrays["valid_lat"] &= arrays[name] == 1
  arrays["valid"] = arrays["valid_lat"].copy()
  arrays["valid_long"] = np.logical_and.reduce([np.isfinite(arrays[name]) for name in
                                                ("v", "a", "accel_command", "accel_target", "long_active")])
  for name in ("car_state_valid", "car_control_valid", "car_output_valid", "planner_valid", "model_valid",
               "calibration_valid", "calibrated"):
    arrays["valid_long"] &= arrays[name] == 1
  for service, prefix, fields in (
    ("narrowRoadEncodeIdx", "raw_encode", ("encode_t", "encode_event_t", "encode_frame_id", "encode_segment_num", "encode_segment_id")),
    ("narrowRoadCameraState", "raw_camera", ("camera_t", "camera_frame_id")),
  ):
    for field in fields:
      arrays[f"{prefix}_{field.split('_', 1)[1]}"] = np.array([row.values[field] for row in grouped.get(service, [])])
  for service, prefix in (("lateralTorqueParameters", "learner"), ("lateralDelay", "delay")):
    arrays[f"raw_{prefix}_t"] = np.array([row.t for row in grouped.get(service, [])])
    for field in SERVICE_FIELDS[service]:
      arrays[f"raw_{field}"] = np.array([row.values[field] for row in grouped.get(service, [])])
  summary = {"rows": len(t), "valid_rows": int(arrays["valid"].sum()),
             "lat_active_rows": int(np.nansum(arrays["lat_active"] == 1)),
             "long_active_rows": int(np.nansum(arrays["long_active"] == 1)),
             "valid_lat_rows": int(arrays["valid_lat"].sum()), "valid_long_rows": int(arrays["valid_long"].sum()),
             "service_samples": {key: len(value) for key, value in sorted(grouped.items())},
             "missing_services": sorted(set(SERVICE_FIELDS) - set(grouped)),
             "speed_band_rows": dict(Counter(band_of(float(v)) or "missing" for v in arrays["v"]))}
  return arrays, summary


def hold_calibration(data: dict, max_age: float = 0.5) -> dict:
  """Explicit analysis-only persistence for the 4 Hz calibration state.

  Extraction retains its universal 200 ms freshness contract. This separate helper
  returns a copy with calibration fields held for at most 500 ms by default,
  only when the recorded source-event age identifies the same previously observed
  valid event. New or invalid events terminate a hold. Dynamic telemetry keeps its
  original 200 ms gates; lateral/longitudinal validity is recomputed accordingly.
  ``calibration_held`` identifies the affected analysis rows. This helper neither
  writes an extracted dataset nor changes production controller behavior.
  """
  if not np.isfinite(max_age) or not 0.2 <= max_age <= 0.5:
    raise ExtractionError("analysis calibration max_age must be between 0.2 and 0.5 seconds")
  fields = ("calibrated", "calibration_valid", "rpy_calib", "calibration_t", "calibration_height")
  for key in ("t", "age_extrinsicsCalibration", *fields):
    if key not in data:
      raise ExtractionError(f"calibration hold requires recorded {key}")
  out = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()}
  out["calibration_held"] = np.zeros(len(out["t"]), dtype=bool)
  cached_t, cached = None, None
  for i, t in enumerate(out["t"]):
    age = data["age_extrinsicsCalibration"][i]
    if not np.isfinite([t, age]).all() or age < 0:
      cached_t, cached = None, None
      continue
    source_t = t - age
    known_t = data["calibration_t"][i]
    if np.isfinite(known_t):
      if data["calibration_valid"][i] == 1 and abs(source_t - known_t) <= 1e-6:
        cached_t = known_t
        cached = {key: np.array(data[key][i], copy=True) for key in fields}
      else:
        cached_t, cached = None, None
    elif cached_t is not None and abs(source_t - cached_t) > 1e-6:
      cached_t, cached = None, None
    if cached is not None and 0.2 + 1e-9 < age <= max_age + 1e-9 and abs(source_t - cached_t) <= 1e-6:
      for key in fields:
        out[key][i] = cached[key]
      out["calibration_held"][i] = True
  required = ("v", "a", "yaw", "roll", "torque", "desired_curvature", "lat_active")
  out["valid_lat"] = np.logical_and.reduce([np.isfinite(out[name]) for name in required])
  for name in ("car_state_valid", "car_control_valid", "car_output_valid", "controls_valid", "location_valid",
               "location_measurement_valid", "calibration_valid", "calibrated", "model_valid"):
    out["valid_lat"] &= out[name] == 1
  out["valid"] = out["valid_lat"].copy()
  out["valid_long"] = np.logical_and.reduce([np.isfinite(out[name]) for name in
                                             ("v", "a", "accel_command", "accel_target", "long_active")])
  for name in ("car_state_valid", "car_control_valid", "car_output_valid", "planner_valid", "model_valid",
               "calibration_valid", "calibrated"):
    out["valid_long"] &= out[name] == 1
  return out


def _exact_frame_id(value):
  if isinstance(value, (bool, np.bool_)):
    return None
  if isinstance(value, (int, np.integer)):
    return int(value) if 0 <= value <= 2**64 - 1 else None
  if isinstance(value, (float, np.floating)) and np.isfinite(value) and 0 <= value <= 2**53 - 1 and value == int(value):
    return int(value)
  return None


def _model_frame_trust(data: dict, metadata: dict | None, shape: tuple) -> np.ndarray:
  """IQ selectors rewrite frame IDs; no current IQ path has proven exposure identity."""
  unknown = np.zeros(shape, dtype=bool)
  meta = metadata or {}
  counts = meta.get("message_counts", {})
  status = meta.get("model_capture_identity_status")
  iq = (meta.get("recorded_software_family") == "IQ.Pilot" or
        str(meta.get("init_data", {}).get("version", "")).startswith("IQ.Pilot") or
        any(counts.get(name, 0) for name in ("iqDriveModelData", "iqLiveLocation", "iqCarParams")))
  if iq or (status is not None and status != "recorded_narrow_frame_id"):
    return unknown
  if "model_frame_id_trusted" in data:
    trusted = np.asarray(data["model_frame_id_trusted"])
    if trusted.shape != shape or trusted.dtype.kind not in "biuf":
      return unknown
    return np.isfinite(trusted) & (trusted == 1)
  # Legacy archived Sunnypilot NPZs predate the trust array. Require their source
  # and schema receipt plus original raw narrow-service names, never an IQ alias.
  legacy = (all(counts.get(name, 0) for name in ("modelV2", "narrowRoadCameraState", "narrowRoadEncodeIdx")) and
            len(str(meta.get("source_log_sha256", ""))) == 64 and
            len(str(meta.get("schema_sha256", {}).get("log.capnp", ""))) == 64)
  return np.ones(shape, dtype=bool) if status == "recorded_narrow_frame_id" or legacy else unknown


def model_capture_times(data: dict, metadata: dict | None = None) -> np.ndarray:
  """Match each model's exact frame ID to its narrow-camera exposure timestamp.

  ``model_t`` is Event.logMonoTime publication time, never exposure time. This
  analysis-only helper matches model_frame_id to raw_encode_frame_id/raw_encode_t
  (EncodeIdx.timestampEof seconds). It performs no nearest/as-of ID match, alias,
  interpolation or future fill. Missing/malformed identities, conflicting duplicate
  ID mappings, future publication, and capture after publication or the analysis row
  return NaN. Identical duplicate mappings are allowed. Integer 64-bit IDs retain
  their exact values; float IDs outside the exact-integer range are rejected.

  Exact IDs alone are insufficient: production consumers must pass their segment
  metadata. IQ.Pilot selectors can relabel held BIG outputs, so all IQ paths remain
  untrusted even with frameAge0/nonBIG until producer exposure identity is proven.
  Unknown provenance without an explicit trusted-row array returns NaN. Legacy
  Sunnypilot outputs require original raw narrow-service names and schema/source
  receipts. Neither frameAge0 nor a caller-provided flag overrides IQ metadata.

  An old but exact exposure is returned unchanged: consumers must measure and gate
  capture age themselves. A publication-age limit cannot bound capture-to-GT age.
  Existing archived NPZs are not rewritten; new extraction records trust/age/BIG.
  """
  t = np.asarray(data.get("t", []))
  if t.ndim != 1:
    raise ExtractionError("model capture alignment requires a 1D timestamp grid")
  result = np.full(len(t), np.nan)
  trusted = _model_frame_trust(data, metadata, t.shape)
  required = ("model_t", "model_frame_id", "raw_encode_frame_id", "raw_encode_t")
  if any(key not in data for key in required):
    return result
  publication, model_ids, encode_ids, captures = (np.asarray(data[key]) for key in required)
  if (any(a.ndim != 1 for a in (publication, model_ids, encode_ids, captures)) or
      publication.shape != t.shape or model_ids.shape != t.shape or encode_ids.shape != captures.shape or
      any(a.dtype.kind not in "fiu" for a in (t, publication, captures))):
    return result
  mapping, ambiguous = {}, set()
  for raw_id, capture in zip(encode_ids, captures, strict=True):
    frame_id = _exact_frame_id(raw_id)
    if frame_id is None:
      continue
    if not np.isfinite(capture) or capture < 0:
      ambiguous.add(frame_id)
    elif frame_id in mapping and mapping[frame_id] != capture:
      ambiguous.add(frame_id)
    else:
      mapping[frame_id] = float(capture)
  for i, raw_id in enumerate(model_ids):
    if not trusted[i]:
      continue
    frame_id = _exact_frame_id(raw_id)
    if frame_id is None or frame_id in ambiguous or frame_id not in mapping:
      continue
    capture = mapping[frame_id]
    if np.isfinite([t[i], publication[i]]).all() and 0 <= capture <= publication[i] <= t[i]:
      result[i] = capture
  return result


def provenance_tag(metadata: dict) -> dict:
  cp = metadata.get("car_params", {})
  init = metadata.get("init_data", {})
  material = {"car_params": cp, "git_commit": init.get("gitCommit"), "dirty": init.get("dirty"),
              "model_identity": metadata.get("model_identity", {}), "opendbc_gitlink": None}
  return {"tag": hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16],
          "opendbc_gitlink": None, "opendbc_identity_status": "unknown",
          "warning": "Superproject gitCommit and safetyParam do not establish the recorded opendbc gitlink."}


def load_schema(directory: Path):
  import capnp

  if not (directory / "log.capnp").is_file():
    raise ExtractionError(f"missing log.capnp in explicit --schema-dir: {directory}")
  return capnp.load(str((directory / "log.capnp").resolve()), imports=[str(directory.resolve()), str((directory / "include").resolve())])


def _measurement(message, index: int) -> float:
  return float(message.value[index]) if len(message.value) > index else np.nan


def _schema_service_aliases(schema) -> dict:
  """Normalize only verified original narrow-camera Event ordinals.

  IQ.Pilot uses the original names at @2/@15; sunnypilot calls those same
  services narrowRoad*. Raw message counts retain the source names. This does
  not map IQ location/model services onto unrelated vehicle-pose contracts.
  """
  aliases = {}
  for source, target, ordinal, fields in (
      ("roadCameraState", "narrowRoadCameraState", 2, ("frameId", "timestampEof", "sensor")),
      ("roadEncodeIdx", "narrowRoadEncodeIdx", 15,
       ("frameId", "timestampEof", "segmentNum", "segmentId", "segmentIdEncode", "encodeId"))):
    field = schema.Event.schema.fields.get(source)
    if field is None:
      continue
    if field.proto.ordinal.which() != "explicit" or field.proto.ordinal.explicit != ordinal:
      raise ExtractionError(f"unsupported {source} Event ordinal; expected @{ordinal}")
    if target in schema.Event.schema.fields or any(name not in field.schema.fields for name in fields):
      raise ExtractionError(f"ambiguous or incomplete {source} camera service contract")
    aliases[source] = target
  return aliases


def decode_log(path: Path, schema) -> tuple[list[Sample], dict]:
  raw = decompress_log(path)
  aliases = _schema_service_aliases(schema)
  samples = []
  schema_fields = schema.Event.schema.fields
  frame_contract = ("untrusted_selector_frame_id" if "iqDriveModelData" in schema_fields else
                    "recorded_narrow_frame_id" if all(k in schema_fields for k in
                      ("narrowRoadCameraState", "narrowRoadEncodeIdx")) else "unknown")
  metadata = {"car_params": {}, "init_data": {}, "model_identity": {}, "can_counts": Counter(),
              "message_counts": Counter(), "invalid_message_counts": Counter(), "source_log": str(path.resolve()),
              "source_log_sha256": sha256(path), "camera_sensor": None, "params": {}, "service_aliases": aliases,
              "model_capture_identity_status": frame_contract}
  try:
    for index, event in enumerate(schema.Event.read_multiple_bytes(raw, traversal_limit_in_words=2**31)):
      try:
        source_name = event.which()
      except Exception as exc:
        raise ExtractionError(f"Event union/schema mismatch at message {index} in {path}; " +
                              "decode with the recorded build's explicit schema snapshot") from exc
      name = aliases.get(source_name, source_name)
      metadata["message_counts"][source_name] += 1
      if not event.valid:
        metadata["invalid_message_counts"][source_name] += 1
      t, valid = int(event.logMonoTime) * 1e-9, bool(event.valid)
      values = None
      if name == "carState":
        d = event.carState
        values = {"v": float(d.vEgo), "a": float(d.aEgo), "steering_pressed": float(d.steeringPressed),
                  "brake_pressed": float(d.brakePressed), "gas_pressed": float(d.gasPressed),
                  "driver_torque": float(d.steeringTorque), "steering_angle": float(d.steeringAngleDeg)}
      elif name == "carControl":
        d = event.carControl
        values = {"lat_active": float(d.latActive), "long_active": float(d.longActive),
                  "torque_request": float(d.actuators.torque) * 409, "torque_fraction": float(d.actuators.torque),
                  "accel_request": float(d.actuators.accel)}
      elif name == "carOutput":
        d = event.carOutput.actuatorsOutput
        values = {"torque": float(d.torqueOutputCan), "accel_command": float(d.accel)}
      elif name == "controlsState":
        values = {"desired_curvature": float(event.controlsState.desiredCurvature),
                  "actual_curvature": float(event.controlsState.curvature)}
      elif name == "liveLocationKalman":
        d = event.liveLocationKalman
        values = {"yaw": _measurement(d.angularVelocityCalibrated, 2),
                  "roll": _measurement(d.calibratedOrientationNED, 0),
                  "location_measurement_valid": float(d.angularVelocityCalibrated.valid and d.calibratedOrientationNED.valid
                                                       and str(d.status) == "valid" and d.inputsOK and d.sensorsOK)}
      elif name == "extrinsicsCalibration":
        d = event.extrinsicsCalibration
        values = {"calibrated": float(str(d.calStatus) == "calibrated"),
                  "rpy_calib": list(d.rpyCalib), "calibration_t": t, "calibration_height": list(d.height)}
        metadata["calibration_height"] = list(d.height)
      elif name == "lateralTorqueParameters":
        d = event.lateralTorqueParameters
        values = {"learned_torque_factor": float(d.latAccelFactorFiltered),
                  "learned_torque_friction": float(d.frictionCoefficientFiltered),
                  "learned_torque_offset": float(d.latAccelOffsetFiltered),
                  "learner_valid": float(d.valid), "learner_use_params": float(d.useParams)}
      elif name == "lateralDelay":
        d = event.lateralDelay
        values = {"learned_lateral_delay": float(d.lateralDelay), "delay_estimated": float(str(d.status) == "estimated")}
      elif name == "longitudinalPlan":
        d = event.longitudinalPlan
        values = {"accel_target": float(d.aTarget), "planner_model_t": int(d.modelMonoTime) * 1e-9}
      elif name == "modelV2":
        d = event.modelV2
        fields = d.schema.fields
        values = {"model_t": t, "model_frame_id": float(d.frameId), "model_x": list(d.position.x),
                  "model_y": list(d.position.y), "model_z": list(d.position.z), "model_horizon": list(d.position.t),
                  "model_frame_age": float(d.frameAge) if "frameAge" in fields else np.nan,
                  "model_big": float(d.big) if "big" in fields else np.nan,
                  "model_frame_id_trusted": float(metadata["model_capture_identity_status"] == "recorded_narrow_frame_id")}
      elif name == "narrowRoadCameraState":
        d = getattr(event, source_name)
        values = {"camera_t": int(d.timestampEof) * 1e-9, "camera_frame_id": float(d.frameId)}
        metadata["camera_sensor"] = str(d.sensor)
      elif name == "narrowRoadEncodeIdx":
        d = getattr(event, source_name)
        values = {"encode_t": int(d.timestampEof) * 1e-9, "encode_event_t": t, "encode_frame_id": float(d.frameId),
                  "encode_segment_num": float(d.segmentNum), "encode_segment_id": float(d.segmentId),
                  "encode_segment_id_encode": float(d.segmentIdEncode), "encode_id": float(d.encodeId)}
      elif name in ("can", "sendcan"):
        frames = ({"address": frame.address, "src": frame.src, "dat": frame.dat} for frame in getattr(event, name)
                  if (frame.address == 0x340 and frame.src == 128) or (frame.address == 0x251 and frame.src == 0))
        can_data, counters = can_values(frames)
        metadata["can_counts"].update(counters)
        if "can_torque" in can_data:
          samples.append(Sample("canTorque", t, {"can_torque": can_data["can_torque"]}, valid))
        if "eps_fault" in can_data:
          samples.append(Sample("epsStatus", t, {"eps_fault": can_data["eps_fault"]}, valid))
      elif name == "carParams":
        cp = event.carParams.to_dict()
        metadata["car_params"] = {key: cp[key] for key in CP_FIELDS if key in cp}
      elif name == "initData":
        d = event.initData
        metadata["init_data"] = {"gitCommit": str(d.gitCommit), "gitSrcCommit": str(d.gitSrcCommit),
                                 "gitBranch": str(d.gitBranch), "dirty": bool(d.dirty), "version": str(d.version),
                                 "deviceType": str(d.deviceType), "wallTimeNanos": int(d.wallTimeNanos)}
        if str(d.version).startswith("IQ.Pilot"):
          metadata["model_capture_identity_status"] = "untrusted_selector_frame_id"
        safe_params = ("NeuralNetworkLateralControl", "CustomTorqueParams", "LiveTorqueParamsToggle", "TorqueControlTune",
                       "LateralJerkTorqueController", "EnforceTorqueControl", "LiveTorqueParamsRelaxedToggle",
                       "TorqueParamsOverrideEnabled", "TorqueParamsOverrideLatAccelFactor", "TorqueParamsOverrideFriction")
        for entry in d.params.entries:
          if "model" in str(entry.key).lower() or str(entry.key) in safe_params:
            data = bytes(entry.value)
            try:
              value = data.decode("utf-8") if len(data) <= 1024 and b"\x00" not in data else None
            except UnicodeDecodeError:
              value = None
            value = value if value is not None else {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            destination = "params" if str(entry.key) in safe_params else "model_identity"
            metadata[destination][str(entry.key)] = value
      elif name == "carParamsSP":
        metadata["car_params_sp"] = event.carParamsSP.to_dict()
      if values is not None:
        samples.append(Sample(name, t, values, valid))
  except ExtractionError:
    raise
  except Exception as exc:
    raise ExtractionError(f"corrupt/incompatible rlog {path}: {type(exc).__name__}: {exc}") from exc
  if not samples:
    raise ExtractionError(f"no recognized telemetry in source log: {path}")
  metadata["provenance"] = provenance_tag(metadata)
  cp, init = metadata["car_params"], metadata["init_data"]
  torque_tune = cp.get("lateralTuning", {}).get("torque", {})
  metadata.update({"source": metadata["source_log"], "source_sha256": metadata["source_log_sha256"],
                   "git_commit": init.get("gitCommit"), "dirty": init.get("dirty"), "fingerprint": cp.get("carFingerprint"),
                   "op_long": cp.get("openpilotLongitudinalControl"), "torque_factor": torque_tune.get("latAccelFactor"),
                   "torque_friction": torque_tune.get("friction"), "opendbc_gitlink": None})
  return samples, metadata


def write_json(path: Path, value: dict) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


_WORKER_SCHEMAS = {}


def extract_one(row: dict, schema_dir_text: str, schema_hashes: dict, out_text: str) -> dict:
  """Worker entry point: schemas stay process-local; all manifest writes stay in the parent."""
  started = time.perf_counter()
  schema_dir, out = Path(schema_dir_text), Path(out_text)
  if schema_dir_text not in _WORKER_SCHEMAS:
    _WORKER_SCHEMAS[schema_dir_text] = load_schema(schema_dir)
  samples, metadata = decode_log(Path(row["log"]), _WORKER_SCHEMAS[schema_dir_text])
  arrays, summary = align_samples(samples)
  metadata.update({"segment": row["segment"], "route": row["route"], "segment_index": row["index"],
                   "source_segment": row["path"], "source_files": row["files"], "schema_dir": str(schema_dir),
                   "schema_sha256": schema_hashes,
                   "schema_hash": hashlib.sha256(json.dumps(schema_hashes, sort_keys=True).encode()).hexdigest(),
                   "extractor_format_version": 2, "summary": summary,
                   "contract": {"rate_hz": 20, "max_age_s": 0.2, "join": "causal as-of logMonoTime; sorted before alignment",
                                "torque": "carOutput.actuatorsOutput.torqueOutputCan; signed CAN counts; never substituted",
                                "torque_request": "carControl torque times normalization reference409; actual historical limit unverified",
                                "can_torque": "independent LKAS11 address0x340 src128 signed torque counts",
                                "eps_fault": "MDPS12 address0x251 src0 bit14; excludes all other buses",
                                "yaw": "liveLocationKalman.angularVelocityCalibrated.value[2], rad/s in calibrated vehicle frame",
                                "roll": "liveLocationKalman.calibratedOrientationNED.value[0], rad",
                                "accel_target": "longitudinalPlan.aTarget; m/s^2; controlsState deprecated aTarget not used",
                                "model_coordinates": "recorded modelV2.position device frame; x forward,y right,z down; meters",
                                "model_t": "modelV2 Event.logMonoTime publication time; not camera exposure time",
                                "model_capture_alignment": "analysis model_capture_times matches exact frame ID to EncodeIdx.timestampEof; unknown stays NaN",
                                "model_frame_identity": "metadata trust gate required; IQ selector rewrites frameId, so IQ capture remains blocked",
                                "model_frame_age_big": "recorded modelV2.frameAge/big when fields exist; zero/nonBIG alone do not prove IQ exposure identity",
                                "model_horizon": "recorded modelV2.position.t; seconds; no synthetic horizon padding",
                                "valid_lat": "fresh valid lat telemetry, location measurements, calibration and model; overrides not excluded",
                                "valid_long": "fresh valid carState/control/output, planner, calibration and model; overrides not excluded",
                                "encode_segment_id": "narrowRoadEncodeIdx.segmentId, presentation index into segment camera file",
                                "missing": "NaN; no backward fill or interpolation; flags are float0/1/NaN"}})
  target = out / f"{row['segment']}.npz"
  np.savez_compressed(target, **arrays)
  write_json(target.with_suffix(".json"), metadata)
  return {"segment": row["segment"], "npz": str(target), "metadata": str(target.with_suffix('.json')),
          "seconds": round(time.perf_counter() - started, 3), **summary}


def extract(args: argparse.Namespace) -> int:
  start = time.perf_counter()
  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  manifest = inventory([Path(root) for root in args.root])
  manifest.update({"format_version": 2, "rate_hz": 20, "max_age_s": 0.2,
                   "selection_policy": "route-disjoint representative route pairs: segment0 plus first nonzero; evenly spread route names",
                   "selected": [], "extracted": [], "extraction_errors": [], "status": "inventory"})
  write_json(out / "manifest.json", manifest)
  selected = select_segments(manifest["segments"], args.limit)
  manifest["selected"] = [{"segment": row["segment"], "path": row["path"]} for row in selected]
  schema_dir = Path(args.schema_dir).resolve()
  schema_hashes = {str(path.relative_to(schema_dir)): sha256(path) for path in sorted(schema_dir.rglob("*.capnp"))}
  try:
    load_schema(schema_dir)  # Fail before dispatch if the explicit schema bundle is unavailable.
  except Exception as exc:
    manifest["status"] = "failed"
    manifest["extraction_errors"].append({"source": str(schema_dir), "error": f"schema load failed: {exc}"})
    write_json(out / "manifest.json", manifest)
    raise ExtractionError(f"schema load failed: {exc}") from exc
  if args.workers < 1:
    raise ExtractionError("--workers must be positive")
  manifest["workers"] = args.workers
  manifest["schema_sha256"] = schema_hashes
  manifest["status"] = "extracting"
  write_json(out / "manifest.json", manifest)

  def record_result(row: dict, future=None):
    try:
      result = future.result() if future is not None else extract_one(row, str(schema_dir), schema_hashes, str(out))
      manifest["extracted"].append(result)
      print(json.dumps(result), flush=True)
    except Exception as exc:
      error = {"segment": row["segment"], "source": row["log"], "error": f"{type(exc).__name__}: {exc}"}
      manifest["extraction_errors"].append(error)
      print(json.dumps(error), file=sys.stderr, flush=True)
    manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    write_json(out / "manifest.json", manifest)
  if args.workers == 1:
    for row in selected:
      record_result(row)
  else:
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
      futures = {executor.submit(extract_one, row, str(schema_dir), schema_hashes, str(out)): row for row in selected}
      for future in as_completed(futures):
        record_result(futures[future], future)
  manifest["status"] = "failed" if manifest["errors"] or manifest["extraction_errors"] or not manifest["extracted"] else "complete"
  manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
  write_json(out / "manifest.json", manifest)
  print(json.dumps({"status": manifest["status"], "inventory": manifest["coverage"], "selected": len(selected),
                    "extracted": len(manifest["extracted"]), "errors": len(manifest["errors"]) + len(manifest["extraction_errors"]),
                    "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
  return 0 if manifest["status"] == "complete" else 1


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  command = commands.add_parser("extract")
  command.add_argument("--root", action="append", required=True, help="Existing flat route archive root; repeat for multiple roots")
  command.add_argument("--schema-dir", required=True, help="Exact recorded-build-compatible cereal schema directory")
  command.add_argument("--out", required=True)
  command.add_argument("--limit", type=int, help="Maximum representative segments; omission extracts every available logical segment")
  command.add_argument("--workers", type=int, default=1, help="Independent process workers; manifest writes remain serialized")
  args = parser.parse_args()
  try:
    return extract(args)
  except ExtractionError as exc:
    print(str(exc), file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
