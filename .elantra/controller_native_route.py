#!/usr/bin/env python3
"""Hash-bound two-source native LongControl command diagnostic, with takeover censor.

Prepare from both original logs on the host; run only in the matching POSIX source
snapshot. Recorded motion remains fixed. No gain changes, publishers or actuation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

import controller_native_replay as native
from controller_native_replay import ReplayError


FIELDS = {
  "carState": ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise"),
  "carControl": ("long_active", "recorded_command"),
  "carOutput": ("recorded_output",), "controlsState": ("recorded_state",),
}
FLAGS = ("brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "long_active")


def normalized_digest(path):
  return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def adapter_sources():
  return {Path(module).name: {"raw_sha256": native.digest(module), "normalized_sha256": normalized_digest(module)}
          for module in (Path(__file__), Path(native.__file__))}


def validate_route_raw(spec, records, statics, captures, packet, rdf):
  """Bind every native raw source to the independently reconstructed route packet."""
  chunks = spec["source_chunks"]
  if len(chunks) != 2 or any(len(items) != 2 for items in (records, statics, captures)):
    raise ReplayError("native route requires both ordered raw/static/camera sources")
  native.validate_rdf(rdf)
  first = float(rdf["input_t"][0])
  for key in ("car_params_snapshots", "car_params_sp_snapshots"):
    snapshots = [record.get("full_static_snapshots", {}).get(key, []) for record in records]
    if (not all(snapshots) or len({row["semantic_sha256"] for rows in snapshots for row in rows}) != 1
        or not all(np.isfinite(row["t"]) for rows in snapshots for row in rows)
        or not any(row["t"] <= first for rows in snapshots for row in rows)):
      raise ReplayError("native route requires unchanged full CP/SP and actual past configuration")
  for static in statics:
    native.validate_parameters(static.get("carParams", {}), static.get("carParamsSP", {}))
    for key in ("carParams", "carParamsSP"):
      if static[key] != statics[0][key]:
        raise ReplayError("native consumed CP/SP differs across raw sources")
  n, offset = len(rdf["input_t"]), 0
  for key in ("source_chunk_index", "source_num", "road_index", "frame_id", "narrow_t"):
    if key not in packet or np.asarray(packet[key]).shape != (n,):
      raise ReplayError("native route packet mapping missing")
  combined = {}
  for index, (chunk, record, capture) in enumerate(zip(chunks, records, captures, strict=True)):
    if record.get("segment") != chunk["segment"] or record.get("source_log_sha256") != chunk["source_log_sha256"]:
      raise ReplayError("native source order/log hash differs from plural selection")
    count = chunk["frames"]
    part = slice(offset, offset + count)
    expected_ids = np.arange(chunk["frame_ids"][0], chunk["frame_ids"][1] + 1)
    local = np.arange(chunk["road_presentation_indices"][0], chunk["road_presentation_indices"][1] + 1)
    if (len(expected_ids) != count or len(local) != count or not np.array_equal(packet["frame_id"][part], expected_ids)
        or not np.array_equal(rdf["frame_id"][part], expected_ids) or not np.array_equal(packet["road_index"][part], local)
        or not np.all(packet["source_chunk_index"][part] == index)
        or not np.all(packet["source_num"][part] == int(chunk["segment"].rsplit("--", 1)[1]))):
      raise ReplayError("native route global/local/source frame mapping differs")
    for frame, stamp in capture.items():
      if frame in combined and combined[frame] != stamp:
        raise ReplayError("conflicting native route camera identity")
      combined[frame] = stamp
    for frame, stamp in zip(expected_ids, rdf["narrow_t"][part], strict=True):
      if int(frame) not in capture or abs(capture[int(frame)] - stamp) > 1e-6:
        raise ReplayError("native RDF exposure does not match its exact raw source")
    offset += count
  if offset != n or not np.array_equal(packet["narrow_t"], rdf["narrow_t"]):
    raise ReplayError("native plural packet exposure count/clock differs")
  return {key: statics[0][key] for key in ("carParams", "carParamsSP")}


def validate_tape(tape):
  rdf = {key[4:]: value for key, value in tape.items() if key.startswith("rdf_")}
  native.validate_rdf(rdf)
  t = np.asarray(tape.get("t"))
  if (t.ndim != 1 or t.dtype.kind not in "fi" or len(t) < 2 or not np.isfinite(t).all()
      or not np.allclose(np.diff(t), .01, rtol=0, atol=1e-9)):
    raise ReplayError("native route requires explicit continuous synthetic100Hz phase")
  censor = np.asarray(tape.get("raw_censor_t"))
  if censor.shape != () or censor.dtype.kind not in "fi" or not np.isfinite(censor) or not t[0] < float(censor) < t[-1]:
    raise ReplayError("missing or invalid route raw brake censor")
  for service, fields in FIELDS.items():
    stamps, valid = np.asarray(tape.get(service + "_t")), np.asarray(tape.get(service + "_valid"))
    if stamps.shape != t.shape or stamps.dtype.kind not in "fi" or valid.shape != t.shape or valid.dtype.kind != "b":
      raise ReplayError("native route raw service clocks/flags shape differs")
    finite = np.isfinite(stamps)
    if np.any(stamps[finite] > t[finite] + 1e-9) or np.any(valid & (~finite | (t - stamps > .2 + 1e-9))):
      raise ReplayError("native route raw service is future or stale while admitted")
    for key in fields:
      values = np.asarray(tape.get(key))
      if values.shape != t.shape or values.dtype.kind not in "fi" or not np.isfinite(values[valid]).all():
        raise ReplayError("native route admitted raw scalar missing/nonfinite")
      if key in FLAGS and not np.isin(values[valid], (0., 1.)).all():
        raise ReplayError("native route admitted raw boolean flag differs")
  for key in ("controller_admissible", "deadline_admissible"):
    values = np.asarray(rdf.get(key))
    if values.dtype.kind != "b" or values.shape != rdf["input_t"].shape:
      raise ReplayError("native route requires exact RDF deadline/controller flags")
  return rdf, float(censor)


def gate_native_tape(tape):
  """Invalidate only the latest rejected RDF tick; never seek an older target.

  The original adapter requires the exact genuine-history scored_mask. Preserve it
  and temporarily close its carControl admission gate for a failed RDF deadline.
  Original raw validity is restored in the output and a separate reason is saved.
  """
  rdf, _ = validate_tape(tape)
  indices = np.searchsorted(rdf["output_ready_t"], tape["t"], side="right") - 1
  display = (rdf["scored_mask"] & rdf["controller_admissible"] & rdf["deadline_admissible"]
             & (rdf["output_ready_t"] - rdf["narrow_t"] <= .2 + 1e-8))
  rejected = (indices >= 132) & ~display[np.maximum(indices, 0)]
  gated = {**tape, "carControl_valid": np.asarray(tape["carControl_valid"]).copy() & ~rejected}
  return gated, rejected, indices


def replay_route(tape, static, controller_type, accel_limits):
  rdf, censor = validate_tape(tape)
  gated, rejected, latest = gate_native_tape(tape)
  result = native.replay_native(gated, static["carParams"], static["carParamsSP"], controller_type, accel_limits)
  result["carControl_valid"] = np.asarray(tape["carControl_valid"]).copy()
  result["reason"][rejected] = 6
  result["model_display_rejected"], result["latest_ready_rdf_index"] = rejected, latest
  # This route never claims that an off tick reconstructs the historical integral.
  result["initialization_uncertain"][:] = True
  result["raw_censor_t"] = np.asarray(censor)
  selected = result["admitted"]
  safe = np.maximum(latest, 0)
  controlled = (selected & (tape["t"] < censor) & (rdf["narrow_t"][safe] < censor)
                & (rdf["output_ready_t"][safe] < censor))
  result["pre_censor_admitted"] = controlled
  result["controlled_command_available"] = controlled & (tape["long_active"] == 1)
  for key in ("brake_pressed", "gas_pressed", "steering_pressed"):
    result["controlled_command_available"] &= tape[key] == 0
  result["post_censor_diagnostic"] = selected & (tape["t"] >= censor)
  result["accel_min"], result["accel_max"] = np.full(len(selected), np.nan), np.full(len(selected), np.nan)
  cp, sp = native.validate_parameters(static["carParams"], static["carParamsSP"])
  for j in np.flatnonzero(selected):
    lower, upper = accel_limits(cp, sp, float(tape["v"][j]), float(tape["v_cruise"][j]) / 3.6)
    if not lower - 1e-9 <= result["native_accel"][j] <= upper + 1e-9:
      raise ReplayError("native acceleration exceeds actual interface limits")
    result["accel_min"][j], result["accel_max"][j] = lower, upper
  return result


def prepare(args):
  from controller_route_overlay import load_route_evidence
  import rdf_route_frames as route

  spec, packet, _, _, rdf, _, proof = load_route_evidence(
    args.selection, args.selection_sha256, args.packet, args.raw_root, args.dataset_dir, args.schema_dir, args.rdf_output, args.transfer_receipt)
  native.validate_output(args.out, [Path(path) for path in proof["input_sha256"]], args.raw_root)
  stream_sets, statics, captures = [], [], []
  for chunk in spec["source_chunks"]:
    log = args.raw_root / chunk["segment"] / "rlog.zst"
    if native.digest(log) != chunk["source_log_sha256"]:
      raise ReplayError("native raw log differs before preparation")
    streams, static, capture = native.read_recorded(log, args.schema_dir, int(chunk["segment"].rsplit("--", 1)[1]))
    if native.digest(log) != chunk["source_log_sha256"]:
      raise ReplayError("native raw log changed while preparing")
    stream_sets.append(streams)
    statics.append(static)
    captures.append(capture)
  static = validate_route_raw(spec, proof["source_segments"], statics, captures, packet, rdf)
  streams, duplicates = route.merge_streams(stream_sets)
  tape = native.prepare_arrays(streams, rdf)
  tape["raw_censor_t"] = np.asarray(proof["raw_censor_t"])
  validate_tape(tape)
  if any(native.digest(path) != digest for path, digest in proof["input_sha256"].items()):
    raise ReplayError("route source changed during native preparation")
  receipt = {"mode": "prepared_native_route", "multi_source": True, "route_evidence": proof, "static": static,
             "raw_native_duplicate_counts": duplicates, "prepare_adapter_sources": adapter_sources(),
             "expected_native_sources_normalized_sha256": native.source_hashes(args.repo),
             "causality": "exact plural raw logs sorted/deduplicated; as-of <=200ms; latest invalid message never skipped; synthetic100Hz phase",
             "historical_opendbc_identity": "unknown dirty e748 source; conditional current-master command diagnostic only",
             "no_actuation": True, "no_params_writes": True}
  return tape, receipt


def run(args):
  native.validate_output(args.out, [args.input, args.input.with_suffix(".json")])
  prepared = json.loads(args.input.with_suffix(".json").read_text())
  if (prepared.get("mode") != "prepared_native_route" or prepared.get("multi_source") is not True
      or native.digest(args.input) != prepared.get("output_npz_sha256")):
    raise ReplayError("requires hash-bound plural prepared native route")
  actual = adapter_sources()
  if {key: value["normalized_sha256"] for key, value in actual.items()} != {
      key: value["normalized_sha256"] for key, value in prepared.get("prepare_adapter_sources", {}).items()}:
    raise ReplayError("native route prepare/run adapter source changed")
  hashes = native.source_hashes(args.repo)
  if hashes != prepared["expected_native_sources_normalized_sha256"]:
    raise ReplayError("native source/dependency snapshot differs from prepared current-master")
  sys.path[:0] = [str(args.repo), str(args.repo / "opendbc_repo")]
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl
  from openpilot.common.realtime import DT_CTRL
  from opendbc.car.hyundai.interface import CarInterface

  origin = Path(inspect.getfile(LongControl)).resolve()
  limits_origin = Path(inspect.getfile(CarInterface.get_pid_accel_limits)).resolve()
  if (DT_CTRL != .01 or origin != (args.repo / native.SOURCES[0]).resolve()
      or limits_origin != (args.repo / "opendbc_repo/opendbc/car/interfaces.py").resolve()):
    raise ReplayError("actual native LongControl/interface import or time step differs")
  with np.load(args.input, allow_pickle=False) as source:
    tape = dict(source)
  arrays = replay_route(tape, prepared["static"], LongControl, CarInterface.get_pid_accel_limits)
  admitted = arrays["admitted"]
  receipt = {"mode": "native_route_replay", "multi_source": True, "input": str(args.input.resolve()),
             "input_npz_sha256": native.digest(args.input), "input_receipt_sha256": native.digest(args.input.with_suffix(".json")),
             "route_evidence": prepared["route_evidence"], "static": prepared["static"],
             "prepare_adapter_sources": prepared["prepare_adapter_sources"], "run_adapter_sources": actual,
             "native_sources_normalized_sha256": hashes, "native_import": str(origin), "limits_import": str(limits_origin),
             "initialization": ("native constructor cold/off/last_output0/integral0; no historical-state equivalence; "
                                +
                                "EVERY admitted tick uncertain, including after disengagement"),
             "admitted_rows": int(admitted.sum()), "controlled_command_available_rows": int(arrays["controlled_command_available"].sum()),
             "post_censor_diagnostic_rows": int(arrays["post_censor_diagnostic"].sum()),
             "positive_accel_should_stop_rows": int(np.count_nonzero(admitted & arrays["should_stop"] & (arrays["rdf_accel_target"] > 0))),
             "reason_counts": {int(k): int(v) for k, v in zip(*np.unique(arrays["reason"], return_counts=True), strict=True)},
             "reason_codes": {0: "admitted", 1: "model unavailable/history warmup", 2: "model capture stale",
                              3: "carState stale/invalid", 4: "carControl stale/invalid", 5: "nonfinite/nonboolean vehicle input",
                              6: "latest ready RDF rejected by controller/deadline/display mask; no fallback"},
             "controlled_metrics": "none; exact brake censor preserved; no controlled3s outcome score or completed-stop claim",
             "semantics": ("actual current-master LongControl command on fixed recorded motion; unchanged shouldStop; "
                           +
                           "no gains/plant optimization or physical improvement"),
             "no_actuation": True, "no_params_writes": True}
  return arrays, receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("prepare", "run"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--input", type=Path)
  parser.add_argument("--selection", type=Path)
  parser.add_argument("--selection-sha256")
  parser.add_argument("--packet", type=Path)
  parser.add_argument("--raw-root", type=Path)
  parser.add_argument("--dataset-dir", type=Path)
  parser.add_argument("--schema-dir", type=Path)
  parser.add_argument("--rdf-output", type=Path)
  parser.add_argument("--transfer-receipt", type=Path)
  args = parser.parse_args()
  if args.out.suffix != ".npz":
    raise ReplayError("native route output must be NPZ with a separate JSON receipt")
  if args.mode == "prepare":
    required = ("selection", "selection_sha256", "packet", "raw_root", "dataset_dir", "schema_dir", "rdf_output", "transfer_receipt")
    if any(getattr(args, name) is None for name in required):
      raise ReplayError("prepare requires all plural route inputs")
    arrays, receipt = prepare(args)
  else:
    if args.input is None:
      raise ReplayError("run requires --input")
    arrays, receipt = run(args)
  with args.out.open("xb") as output:
    np.savez_compressed(output, **arrays)
  receipt.update(output=str(args.out.resolve()), output_npz_sha256=native.digest(args.out), rows=len(arrays["t"]), dt=.01, argv=sys.argv)
  with args.out.with_suffix(".json").open("x", encoding="utf-8") as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
  print(json.dumps({key: receipt[key] for key in ("mode", "output", "rows", "output_npz_sha256")}))


if __name__ == "__main__":
  main()
