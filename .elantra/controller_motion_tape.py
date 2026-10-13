"""Selected TRAIN-only raw multirate motion tapes; no resampling or plant fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import stat

import numpy as np

import controller_extract as extract
import controller_native_lateral as serialization
import controller_native_lateral_runtime as param_contracts
import lateral_report as can_helpers
from controller_extract import Sample, decompress_log, load_schema, sha256
from controller_native_lateral import encode_payloads
from controller_native_lateral_runtime import TYPES
from lateral_report import decode_lkas11, decode_mdps_status

FIELDS = {
  "carState": {"v": ("vEgo",), "a": ("aEgo",), "steering_angle": ("steeringAngleDeg",),
    "vendor_rate_unsigned": ("steeringRateDeg",), "driver_torque": ("steeringTorque",), "steering_pressed": ("steeringPressed",),
    "brake_pressed": ("brakePressed",), "gas_pressed": ("gasPressed",), "standstill": ("standstill",),
    "fault_temporary": ("steerFaultTemporary",), "fault_permanent": ("steerFaultPermanent",)},
  "liveLocationKalman": {"yaw": ("angularVelocityCalibrated", "value", 2), "yaw_std": ("angularVelocityCalibrated", "std", 2),
    "yaw_inner_valid": ("angularVelocityCalibrated", "valid"), "roll": ("calibratedOrientationNED", "value", 0),
    "roll_std": ("calibratedOrientationNED", "std", 0), "roll_inner_valid": ("calibratedOrientationNED", "valid"),
    "status": ("status",), "inputs_ok": ("inputsOK",), "sensors_ok": ("sensorsOK",), "device_stable": ("deviceStable",),
    "time_since_reset": ("timeSinceReset",)},
  "carControl": {"lat_active": ("latActive",), "long_active": ("longActive",), "torque_request": ("actuators", "torque"),
    "desired_curvature": ("actuators", "curvature")},
  "carOutput": {"torque_reported": ("actuatorsOutput", "torque"), "torque_counts": ("actuatorsOutput", "torqueOutputCan")},
}
CAN_FIELDS = {"LKAS11Tx": ("torque", "request", "torque_fault"),
              "MDPS12Rx": ("fault", "unavailable", "active", "failure", "defect"),
              "SAS11Rx": ("steering_angle", "vendor_rate_unsigned")}
SEMANTICS = {
  "clock": "Raw uint64 event.logMonoTime nanoseconds; publication/batch clock, not proven sensor acquisition clock.",
  "vendor_rate_unsigned": "Recorded carState.steeringRateDeg; Hyundai classic SAS11.SAS_Speed unsigned magnitude,4deg/s quantization.",
  "steering_angle": "Recorded signed degrees; no VM-inverse angle, derivative or artificial time grid.",
  "yaw": "LLK.angularVelocityCalibrated.value[2] rad/s; std[2] and inner/outer validity retained.",
  "roll": "LLK.calibratedOrientationNED.value[0] rad; std[0] retained.",
  "a": "Native filtered carState.aEgo m/s^2, not independent accelerometer truth.",
  "torque": "LKAS11 signed CAN counts and request/fault bits; request=false does not prove delivered actuation.",
  "can_origin": "0=returned can echo LKAS11 src128,1=outgoing sendcan LKAS11 bus0; neither proves delivered actuation. Raw source retained.",
}
SEGMENT_RE = re.compile(r"([0-9a-f]{8}--[0-9a-f]{10})--([0-9]+)")


class TapeError(ValueError):
  pass


def iter_events(path, schema):
  yield from schema.Event.read_multiple_bytes(decompress_log(path), traversal_limit_in_words=2**31)


def _value(message, path):
  try:
    for key in path:
      message = message[key] if isinstance(key, int) else getattr(message, key)
    if path == ("status",):
      message = message.raw if hasattr(message, "raw") else {"uninitialized": 0, "uncalibrated": 1, "valid": 2}[str(message)]
    return float(message), True
  except (AttributeError, IndexError, KeyError, TypeError, ValueError):
    return np.nan, False


def _row(service, t_ns, sequence, valid, values, present, payload, *, frame=-1, origin=-1, address=-1, src=-1, decoded=True):
  return {"sample": Sample(service, t_ns*1e-9, values, valid), "t_ns": t_ns, "sequence": sequence,
          "frame": frame, "present": present, "payload": payload, "origin": origin, "address": address, "src": src, "decoded": decoded}


def _json_hash(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _can_rows(event, name, t_ns, sequence, streams, counters):
  for index, frame in enumerate(getattr(event, name)):
    address, src, data = int(frame.address), int(frame.src), bytes(frame.dat)
    service, values = None, None
    if address == 0x340:
      if (name == "can" and src == 128) or (name == "sendcan" and src == 0):
        service, decoded = "LKAS11Tx", decode_lkas11(data)
        if decoded is not None:
          values = {"torque": decoded[0], "request": decoded[1], "torque_fault": (int.from_bytes(data[:4], "little") >> 28) & 1}
    elif address == 0x251:
      if name == "can" and src == 0:
        service, decoded = "MDPS12Rx", decode_mdps_status(data)
        if decoded is not None:
          values = dict(zip(CAN_FIELDS[service], (decoded[k] for k in ("flt", "unavail", "active", "failstat", "def")), strict=True))
    elif address == 0x2B0 and name == "can" and src == 0:
      service = "SAS11Rx"
      if len(data) >= 3:
        values = {"steering_angle": int.from_bytes(data[:2], "little", signed=True)*.1, "vendor_rate_unsigned": data[2]*4}
    if service is None:
      counters["ignored_can_bus_frames"] += address in (0x340, 0x251, 0x2B0)
      continue
    decoded = values is not None
    counters["short_can_frames"] += not decoded
    present = dict.fromkeys(CAN_FIELDS[service], decoded)
    values = values if decoded else dict.fromkeys(CAN_FIELDS[service], np.nan)
    streams[service].append(_row(service, t_ns, sequence, bool(event.valid), values, present, data,
                                 frame=index, origin=int(name == "sendcan"), address=address, src=src, decoded=decoded))


def _stream_arrays(service, rows):
  original = [row["t_ns"] for row in rows]
  rows = sorted(rows, key=lambda row: (row["t_ns"], row["sequence"], row["frame"]))
  fields = tuple(FIELDS.get(service, CAN_FIELDS.get(service, ())))
  data = {"t_ns": np.array([r["t_ns"] for r in rows], np.uint64), "event_seq": np.array([r["sequence"] for r in rows], np.int64),
          "frame_seq": np.array([r["frame"] for r in rows], np.int64), "valid": np.array([r["sample"].valid for r in rows], bool)}
  missing = {}
  for key in fields:
    data[key] = np.array([r["sample"].values[key] for r in rows], float)
    data[key+"_present"] = np.array([r["present"][key] for r in rows], bool)
    missing[key] = int(np.count_nonzero(~data[key+"_present"]))
  if service in CAN_FIELDS:
    for key in ("origin", "address", "src"):
      data[key] = np.array([r[key] for r in rows], np.int64)
    data["decoded_ok"] = np.array([r["decoded"] for r in rows], bool)
  data["payload"], data["offsets"] = encode_payloads([r["payload"] for r in rows])
  dt = np.diff(data["t_ns"]).astype(float)*1e-9
  positive = dt[dt > 0]
  conflicts = np.zeros(len(rows), bool)
  duplicates = np.zeros(len(rows), bool)
  groups = defaultdict(list)
  for i, row in enumerate(rows):
    groups[row["t_ns"]].append(i)
  for indices in groups.values():
    if len(indices) > 1:
      duplicates[indices] = True
    reference = rows[indices[0]]
    for i in indices[1:]:
      first, other = reference["sample"], rows[i]["sample"]
      if (not np.array_equal(list(first.values.values()), list(other.values.values()), equal_nan=True)
          or reference["payload"] != rows[i]["payload"] or first.valid != other.valid):
        conflicts[indices] = True
  data["clock_conflict"] = conflicts
  data["clock_duplicate"] = duplicates
  usable = data["valid"] & ~duplicates
  for key in fields:
    usable &= data[key+"_present"] & np.isfinite(data[key])
  if service in ("carState", "SAS11Rx"):
    usable &= data["vendor_rate_unsigned"] >= 0
  if service == "liveLocationKalman":
    for key in ("yaw_inner_valid", "roll_inner_valid", "inputs_ok", "sensors_ok", "device_stable"):
      usable &= data[key] == 1
    usable &= (data["status"] == 2) & (data["yaw_std"] >= 0) & (data["roll_std"] >= 0)
  if service in CAN_FIELDS:
    usable &= data["decoded_ok"]
  data["row_usable"] = usable  # Quality only; cadence/support/delay remain separate fit gates.
  summary = {"messages": len(rows), "invalid_messages": int((~data["valid"]).sum()),
             "out_of_order_in_file": sum(a > b for a, b in zip(original, original[1:], strict=False)),
             "duplicate_timestamps": int((dt == 0).sum()), "conflicting_timestamps": len(set(data["t_ns"][conflicts].tolist())),
             "quality_usable_rows": int(usable.sum()),
             "positive_dt_median_s": float(np.median(positive)) if len(positive) else None,
             "positive_dt_p95_s": float(np.quantile(positive, .95)) if len(positive) else None,
             "max_gap_s": float(max(positive)) if len(positive) else None,
             "first_t_ns": int(data["t_ns"][0]) if rows else None, "last_t_ns": int(data["t_ns"][-1]) if rows else None}
  return {service+"_"+key: value for key, value in data.items()}, summary, missing


def collect(events, *, declared=None, aliases=None):
  streams, counters, static, params, identities = defaultdict(list), Counter(), defaultdict(list), [], []
  aliases, declared = aliases or {}, declared or {}
  for sequence, event in enumerate(events):
    source_name = event.which()
    raw_t = event.logMonoTime
    if not isinstance(raw_t, (int, np.integer)) or isinstance(raw_t, bool) or not 0 <= raw_t <= np.iinfo(np.uint64).max:
      raise TapeError("invalid raw uint64 publication clock")
    name, t_ns = aliases.get(source_name, source_name), int(raw_t)
    valid = bool(event.valid)
    if name in FIELDS:
      message = getattr(event, source_name)
      extracted = {key: _value(message, path) for key, path in FIELDS[name].items()}
      values, present = ({key: result[index] for key, result in extracted.items()} for index in (0, 1))
      streams[name].append(_row(name, t_ns, sequence, valid, values, present, message.as_builder().to_bytes()))
    elif name in ("can", "sendcan"):
      _can_rows(event, source_name, t_ns, sequence, streams, counters)
    elif name in ("carParams", "carParamsSP"):
      message = getattr(event, source_name)
      payload = message.as_builder().to_bytes()
      streams[name].append(_row(name, t_ns, sequence, valid, {}, {}, payload))
      static[name].append({"t_ns": t_ns, "valid": valid, "payload_sha256": hashlib.sha256(payload).hexdigest()})
    elif name == "initData":
      d, values, entries, conflicted = getattr(event, source_name), {}, [], set()
      for entry in d.params.entries:
        key = str(entry.key)
        if key in TYPES:
          raw = bytes(entry.value)
          try:
            value = raw.decode("utf-8")
            if len(raw) > 1024:
              raise ValueError("large typed value")
          except (UnicodeDecodeError, ValueError):
            value = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
          entries.append({"key": key, "value": value, "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)})
          if key in values and value != values[key]:
            counters["conflicting_param_entries"] += 1
            conflicted.add(key)
          if key not in conflicted:
            values[key] = value
      values = {key: value for key, value in values.items() if key not in conflicted}
      params.append({"t_ns": t_ns, "valid": valid, "values": values, "entries": entries, "values_sha256": _json_hash(values)})
      identities.append({"t_ns": t_ns, "valid": valid, "git_commit": str(d.gitCommit), "dirty": bool(d.dirty), "device_type": str(d.deviceType)})
  arrays, cadence, missing = {}, {}, {}
  for service in (*FIELDS, *CAN_FIELDS, "carParams", "carParamsSP"):
    data, cadence[service], missing[service] = _stream_arrays(service, streams[service])
    arrays.update(data)
  reasons = ["unknown_opendbc_gitlink"]  # InitData does not prove its nested source pin.
  if any(row["dirty"] for row in identities):
    reasons.append("dirty_recorded_source")
  if not identities or any(not row["valid"] for row in identities):
    reasons.append("missing_or_invalid_source_identity")
  if len({(row["git_commit"], row["dirty"]) for row in identities}) > 1:
    reasons.append("source_identity_changed")
  for name in ("carParams", "carParamsSP"):
    if not static[name] or any(not r["valid"] for r in static[name]):
      reasons.append("missing_or_invalid_"+name)
    if len({r["payload_sha256"] for r in static[name]}) > 1:
      reasons.append("changed_"+name)
  for name in FIELDS:
    if not streams[name]:
      reasons.append("missing_"+name)
  required = {"CustomTorqueParams", "LiveTorqueParamsToggle", "LiveTorqueParamsRelaxedToggle"}
  if not params or any(required-set(row["values"]) for row in params):
    reasons.append("missing_runtime_policy_params")
  if counters["conflicting_param_entries"]:
    reasons.append("conflicting_typed_params")
  if len({row["values_sha256"] for row in params}) > 1:
    reasons.append("typed_params_changed")
  report = {"semantics": SEMANTICS, "cadence": cadence, "missing_fields": missing, "static_snapshots": dict(static),
            "params_snapshots": params, "recorded_source_identity": identities, "source_eligible": False,
            "opendbc_gitlink": None, "declared_opendbc_gitlink": declared.get("opendbc_gitlink"), "eligibility_reasons": reasons,
            "no_resampling": True, "no_capture_inference": True, "production_enabled": False,
            "ignored_can_bus_frames": counters["ignored_can_bus_frames"], "short_can_frames": counters["short_can_frames"],
            "conflicting_param_entries": counters["conflicting_param_entries"],
            "static_timing": "All snapshots retained at actual publication clocks; no inherited or future static snapshot applied."}
  return arrays, report


def _output_guard(out, raw_root, schema_dir, sources, dataset_roots):
  out = out.resolve()
  if out.exists() or any(out == p or out.is_relative_to(p) for p in (raw_root, schema_dir, *dataset_roots)):
    raise TapeError("output must be a new exclusive directory outside raw/schema/datasets")
  if any(p == out or p.is_relative_to(out) for p in sources):
    raise TapeError("output collides with source inputs")
  if any(part == "dataset" or part.startswith("dataset-") for part in out.parts):
    raise TapeError("output inside a dataset is forbidden")
  return out


def _selection(selection, splits, raw_root):
  cohort, selected = selection.get("cohort"), selection.get("segments")
  if not isinstance(selected, list) or not 1 <= len(selected) <= 32:
    raise TapeError("requires a bounded explicit1..32segment allowlist")
  split = splits.get("cohorts", {}).get(cohort, {}).get("split", {})
  if set(split) != {"train", "validation", "test"} or any(not isinstance(v, list) for v in split.values()):
    raise TapeError("requires frozen complete TRAIN/validation/test route split")
  partitions = [set(split[k]) for k in ("train", "validation", "test")]
  if any(a & b for i, a in enumerate(partitions) for b in partitions[i+1:]):
    raise TapeError("overlapping frozen route partitions")
  rows, seen = [], set()
  for row in selected:
    name = row.get("segment", "")
    match = SEGMENT_RE.fullmatch(name)
    if not match or name in seen or match[1] not in partitions[0]:
      raise TapeError("every unique explicit segment must belong to frozen TRAIN routes")
    seen.add(name)
    expected = row.get("source_log_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
      raise TapeError("every source requires an immutable SHA256")
    log = raw_root/name/"rlog.zst"
    if not log.is_relative_to(raw_root):
      raise TapeError("source path escapes authoritative raw root")
    rows.append((name, log, expected, row.get("metadata", {})))
  return rows, _physical_sources(rows)


def _physical_identity(path):
  """Reject reparse/symlink components and multiply linked or unidentifiable files."""
  try:
    for component in (path, *path.parents):
      info = component.lstat()
      if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise TapeError("source component alias/symlink/junction is forbidden")
    if path.resolve(strict=True) != path:
      raise TapeError("source physical path differs from exact flat logical path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
      raise TapeError("source hardlinks/nonregular files are forbidden")
    if not info.st_ino:
      raise TapeError("source physical file identity unavailable")
    return int(info.st_dev), int(info.st_ino)
  except OSError as error:
    raise TapeError("source physical file identity unavailable") from error


def _physical_sources(rows):
  identities = {}
  for _, path, _, _ in rows:
    identity = _physical_identity(path)
    if identity in identities.values():
      raise TapeError("duplicate physical source file alias is forbidden")
    identities[path] = identity
  return identities


def _write_json(path, value):
  with path.open("x", encoding="utf-8") as output:
    json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
    output.write("\n")


def build(selection_path, splits_path, schema_dir, raw_root, out, *, dataset_roots=()):
  selection_path, splits_path, schema_dir = (Path(p).resolve() for p in (selection_path, splits_path, schema_dir))
  raw_root = Path(raw_root).absolute()  # Preserve logical components for alias detection.
  raw_inputs = {p: p.read_bytes() for p in (selection_path, splits_path)}
  selection, splits = (json.loads(raw_inputs[p]) for p in (selection_path, splits_path))
  input_hashes = {p: hashlib.sha256(data).hexdigest() for p, data in raw_inputs.items()}
  rows, physical = _selection(selection, splits, raw_root)  # ALL TRAIN/physical admission precedes log/schema decode.
  sources = [selection_path, splits_path, *(r[1] for r in rows)]
  out = _output_guard(Path(out), raw_root, schema_dir, sources, tuple(Path(p).resolve() for p in dataset_roots))
  for _, log, expected, _ in rows:
    if not log.is_file() or sha256(log) != expected:
      raise TapeError("selected raw log missing or changed before decode")
  schema_paths = sorted(schema_dir.rglob("*.capnp"))
  schema_hashes = {str(p.relative_to(schema_dir)): sha256(p) for p in schema_paths}
  if not schema_hashes:
    raise TapeError("explicit recorded schema snapshot absent")
  schema = load_schema(schema_dir)
  helpers = {"producer": Path(__file__), "extractor": Path(extract.__file__), "serialization": Path(serialization.__file__),
             "typed_params": Path(param_contracts.__file__), "can_decoder": Path(can_helpers.__file__)}
  helper_hashes = {key: sha256(path) for key, path in helpers.items()}
  input_hashes.update({path: schema_hashes[str(path.relative_to(schema_dir))] for path in schema_paths})
  input_hashes.update({path: helper_hashes[key] for key, path in helpers.items()})
  aliases = extract._schema_service_aliases(schema) if hasattr(schema, "Event") else {}
  out.mkdir(parents=True, exist_ok=False)
  results = []
  for name, log, expected, declared in rows:
    try:
      if _physical_sources(rows) != physical:
        raise TapeError("source physical file changed before decode")
      arrays, report = collect(iter_events(log, schema), declared=declared, aliases=aliases)
      if (_physical_sources(rows) != physical or sha256(log) != expected
          or any(sha256(path) != digest for path, digest in input_hashes.items())):
        raise TapeError("source/selection/split/schema/helper input changed during decode")
      npz, sidecar = out/(name+".npz"), out/(name+".json")
      with npz.open("xb") as output:
        np.savez_compressed(output, **arrays)
      report.update(segment=name, source_log=str(log), source_log_sha256=expected, schema_sha256=schema_hashes,
                    source_file_identity={"device": physical[log][0], "inode": physical[log][1], "links": 1},
                    selection_sha256=input_hashes[selection_path], splits_sha256=input_hashes[splits_path], helper_sha256=helper_hashes,
                    output_npz_sha256=sha256(npz), cohort=selection["cohort"])
      _write_json(sidecar, report)
      results.append({"segment": name, "status": "completed", "npz": str(npz), "npz_sha256": report["output_npz_sha256"],
                      "receipt_sha256": sha256(sidecar), "source_eligible": report["source_eligible"]})
    except Exception as error:
      results.append({"segment": name, "status": "failed", "error_type": type(error).__name__, "error": str(error), "source_log_sha256": expected})
  manifest = {"cohort": selection["cohort"], "selected": len(rows), "completed": sum(r["status"] == "completed" for r in results),
              "failed": sum(r["status"] == "failed" for r in results), "results": results, "production_enabled": False,
              "selection_sha256": input_hashes[selection_path], "splits_sha256": input_hashes[splits_path],
              "schema_sha256": schema_hashes, "helper_sha256": helper_hashes}
  _write_json(out/"manifest.json", manifest)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("selection", "splits", "schema-dir", "raw-root", "out"):
    parser.add_argument("--"+name, type=Path, required=True)
  parser.add_argument("--dataset", type=Path, action="append", default=[])
  args = parser.parse_args()
  try:
    result = build(args.selection, args.splits, args.schema_dir, args.raw_root, args.out, dataset_roots=args.dataset)
    print(json.dumps({key: result[key] for key in ("selected", "completed", "failed")}))
    return int(result["failed"] > 0)
  except (TapeError, OSError, ValueError) as error:
    print(f"motion tape failed: {type(error).__name__}: {error}")
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
