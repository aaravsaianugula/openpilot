#!/usr/bin/env python3
"""Diagnostic native LongControl replay on recorded motion, without vehicle actuation.

Prepare on the archive host; run on a POSIX openpilot runtime. No alternate plant
response is inferred. The 100 Hz grid is an explicit replay phase, not a recovered
controlsd execution schedule. Unknown controller initialization stays labelled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

MODEL_SHA = "52fcf48bfb991f327a8982037eb0855d9a63437d78e9f4828d2be54df0f32567"
SOURCES = ("openpilot/selfdrive/controls/lib/longcontrol.py", "openpilot/common/pid.py",
           "openpilot/common/realtime.py", "openpilot/selfdrive/controls/lib/drive_helpers.py",
           "openpilot/selfdrive/modeld/constants.py", "opendbc_repo/opendbc/car/interfaces.py",
           "opendbc_repo/opendbc/car/hyundai/interface.py", "opendbc_repo/opendbc/car/car.capnp")


class ReplayError(RuntimeError):
  pass


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(repo):
  return {name: hashlib.sha256((repo / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in SOURCES}


def validate_parameters(cp, sp):
  required = ("carFingerprint", "openpilotLongitudinalControl", "stopAccel", "longitudinalTuning")
  if any(key not in cp for key in required) or "enableGasInterceptor" not in sp:
    raise ReplayError("missing recorded CP/CP_SP controller contract")
  if cp["carFingerprint"] != "HYUNDAI_ELANTRA_2024" or cp["openpilotLongitudinalControl"] is not True:
    raise ReplayError("requires recorded HYUNDAI_ELANTRA_2024 openpilot longitudinal configuration")
  tuning = cp["longitudinalTuning"]
  bp, gains = np.asarray(tuning.get("kiBP", [])), np.asarray(tuning.get("kiV", []))
  if (bp.ndim != 1 or not len(bp) or bp.shape != gains.shape or not np.isfinite(bp).all()
      or not np.isfinite(gains).all() or np.any(np.diff(bp) <= 0) or np.any(gains < 0)):
    raise ReplayError("missing/nonfinite/inconsistent recorded integral gain arrays")
  if not np.isfinite(cp["stopAccel"]) or cp["stopAccel"] >= 0 or type(sp["enableGasInterceptor"]) is not bool:
    raise ReplayError("invalid recorded stopAccel or enableGasInterceptor")
  return (SimpleNamespace(stopAccel=float(cp["stopAccel"]), longitudinalTuning=SimpleNamespace(kiBP=bp, kiV=gains)),
          SimpleNamespace(enableGasInterceptor=sp["enableGasInterceptor"]))


def asof_indices(source_t, source_valid, target_t, max_age=.2):
  source_t, source_valid, target_t = map(np.asarray, (source_t, source_valid, target_t))
  if not len(source_t) or source_t.shape != source_valid.shape or np.any(np.diff(source_t) < 0):
    raise ReplayError("missing or unsorted source service")
  indices = np.searchsorted(source_t, target_t, side="right") - 1
  safe = np.maximum(indices, 0)
  age = target_t - source_t[safe]
  return indices, (indices >= 0) & source_valid[safe] & (age >= 0) & (age <= max_age + 1e-9)


def validate_rdf(rdf):
  keys = ("input_t", "narrow_t", "output_ready_t", "frame_id", "desired_acceleration", "should_stop", "scored_mask")
  if any(key not in rdf for key in (*keys, "model_sha256")) or str(rdf["model_sha256"]) != MODEL_SHA:
    raise ReplayError("missing exact RDF v2 identity/output contract")
  n = len(rdf["input_t"])
  if n <= 132 or any(np.asarray(rdf[key]).shape != (n,) for key in keys):
    raise ReplayError("RDF outputs must include genuine 132-frame history and matching arrays")
  for key in ("input_t", "narrow_t", "output_ready_t"):
    if not np.isfinite(rdf[key]).all() or np.any(np.diff(rdf[key]) <= 0):
      raise ReplayError("RDF clocks must be finite and increasing")
  if (np.any(rdf["output_ready_t"] < rdf["input_t"]) or np.any(rdf["narrow_t"] > rdf["input_t"])
      or np.any(rdf["input_t"] - rdf["narrow_t"] > .025) or np.any(np.diff(rdf["frame_id"]) != 1)
      or not np.array_equal(rdf["scored_mask"], np.arange(n) >= 132)
      or not np.isfinite(rdf["desired_acceleration"]).all() or np.asarray(rdf["should_stop"]).dtype.kind != "b"):
    raise ReplayError("RDF warmup, stop flags, acceleration or causal availability violated")


def read_recorded(log_path, schema_dir, segment_num):
  from controller_extract import decompress_log, load_schema

  schema = load_schema(schema_dir)
  streams, static, capture = defaultdict(list), {}, {}
  for event in schema.Event.read_multiple_bytes(decompress_log(log_path), traversal_limit_in_words=2**31):
    name, t, valid = event.which(), int(event.logMonoTime) * 1e-9, bool(event.valid)
    row = {"t": t, "valid": valid}
    if name == "carState":
      d = event.carState
      row.update(v=float(d.vEgo), a=float(d.aEgo), brake_pressed=float(d.brakePressed),
                 gas_pressed=float(d.gasPressed), steering_pressed=float(d.steeringPressed),
                 cruise_standstill=float(d.cruiseState.standstill), v_cruise=float(d.vCruise))
    elif name == "carControl":
      row.update(long_active=float(event.carControl.longActive), recorded_command=float(event.carControl.actuators.accel))
    elif name == "carOutput":
      row["recorded_output"] = float(event.carOutput.actuatorsOutput.accel)
    elif name == "controlsState":
      row["recorded_state"] = float(event.controlsState.longControlState.raw)
    elif name in ("carParams", "carParamsSP"):
      if not valid:
        raise ReplayError(f"invalid recorded {name}")
      d = getattr(event, name)
      fields = ("carFingerprint", "openpilotLongitudinalControl", "stopAccel", "longitudinalTuning") if name == "carParams" else ("enableGasInterceptor",)
      value = d.to_dict()
      if any(key not in value for key in fields):
        raise ReplayError(f"missing required {name} schema/recorded fields")
      selected = {key: value[key] for key in fields}
      if name in static and static[name] != selected:
        raise ReplayError("CP/CP_SP changed within the source segment")
      static[name], static[name + "_t"] = selected, t
      continue
    elif name == "narrowRoadEncodeIdx":
      d = event.narrowRoadEncodeIdx
      if valid and int(d.segmentNum) == segment_num and str(d.type) == "fullHEVC":
        key = int(d.frameId)
        value = int(d.timestampEof) * 1e-9
        if key in capture and capture[key] != value:
          raise ReplayError("conflicting camera frame identity")
        capture[key] = value
      continue
    else:
      continue
    streams[name].append(row)
  for rows in streams.values():
    rows.sort(key=lambda row: row["t"])
  if "carParams" not in static or "carParamsSP" not in static:
    raise ReplayError("recorded CP/CP_SP absent")
  validate_parameters(static["carParams"], static["carParamsSP"])
  return streams, static, capture


def prepare_arrays(streams, rdf):
  validate_rdf(rdf)
  first, last = float(rdf["input_t"][0]), float(rdf["input_t"][-1] + .05)
  count = int(np.ceil((last - first) / .01 - 1e-6))
  arrays = {"t": first + np.arange(count) * .01}
  fields = {"carState": ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise"),
            "carControl": ("long_active", "recorded_command"), "carOutput": ("recorded_output",),
            "controlsState": ("recorded_state",)}
  for service, names in fields.items():
    rows = streams.get(service, [])
    source_t, valid = np.array([r["t"] for r in rows]), np.array([r["valid"] for r in rows], dtype=bool)
    indices, admitted = asof_indices(source_t, valid, arrays["t"])
    safe = np.maximum(indices, 0)
    arrays[service + "_t"] = np.where(indices >= 0, source_t[safe], np.nan)
    arrays[service + "_valid"] = admitted
    for name in names:
      values = np.asarray([r[name] for r in rows])
      arrays[name] = np.where(admitted, values[safe], np.nan)
  arrays.update({"rdf_" + key: np.asarray(value) for key, value in rdf.items()})
  return arrays


def replay_native(tape, cp_dict, sp_dict, controller_type, accel_limits):
  cp, sp = validate_parameters(cp_dict, sp_dict)
  rdf = {key[4:]: value for key, value in tape.items() if key.startswith("rdf_")}
  validate_rdf(rdf)
  t = tape["t"]
  if not np.isfinite(t).all() or len(t) < 2 or not np.allclose(np.diff(t), .01, atol=1e-9, rtol=0):
    raise ReplayError("native controller requires a continuous 100 Hz replay grid")
  n = len(t)
  output = {key: np.full(n, np.nan) for key in ("native_accel", "native_state", "pid_integral", "rdf_accel_target", "rdf_capture_age_s")}
  output.update(t=t.copy(), admitted=np.zeros(n, dtype=bool), controller_reset=np.zeros(n, dtype=bool),
                initialization_uncertain=np.ones(n, dtype=bool), rdf_index=np.full(n, -1, dtype=np.int64),
                should_stop=np.zeros(n, dtype=bool), reason=np.ones(n, dtype=np.uint8))
  model_indices = np.searchsorted(rdf["output_ready_t"], t, side="right") - 1
  controller, uncertain = None, True
  for j, timestamp in enumerate(t):
    i = model_indices[j]
    reason = 0
    if i < 0 or not rdf["scored_mask"][max(i, 0)]:
      reason = 1
    elif not 0 <= timestamp - rdf["narrow_t"][i] <= .2 + 1e-9:
      reason = 2
    for code, service in ((3, "carState"), (4, "carControl")):
      age = timestamp - tape[service + "_t"][j]
      if not tape[service + "_valid"][j] or not np.isfinite(age) or not 0 <= age <= .2 + 1e-9:
        reason = code
    fields = ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise", "long_active")
    if not np.isfinite([tape[key][j] for key in fields]).all():
      reason = 5
    if any(tape[key][j] not in (0., 1.) for key in
           ("brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "long_active")):
      reason = 5
    if reason:
      output["reason"][j] = reason
      controller, uncertain = None, True
      continue
    if controller is None:
      controller = controller_type(cp, sp)
      output["controller_reset"][j] = True
    active = bool(tape["long_active"][j])
    cs = SimpleNamespace(vEgo=float(tape["v"][j]), aEgo=float(tape["a"][j]), brakePressed=bool(tape["brake_pressed"][j]),
                         cruiseState=SimpleNamespace(standstill=bool(tape["cruise_standstill"][j])))
    limits = accel_limits(cp, sp, cs.vEgo, float(tape["v_cruise"][j]) / 3.6)
    if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
      raise ReplayError("native interface produced invalid acceleration limits")
    target, stop = float(rdf["desired_acceleration"][i]), bool(rdf["should_stop"][i])
    value = float(controller.update(active, cs, target, stop, limits))
    if not np.isfinite(value):
      raise ReplayError("native controller produced nonfinite acceleration")
    if not active:
      uncertain = False  # native off branch resets PID and last_output_accel to zero
    output["native_accel"][j], output["native_state"][j] = value, int(controller.long_control_state)
    output["pid_integral"][j], output["rdf_accel_target"][j] = float(controller.pid.i), target
    output["should_stop"][j], output["rdf_capture_age_s"][j], output["rdf_index"][j] = stop, timestamp - rdf["narrow_t"][i], i
    output["initialization_uncertain"][j], output["admitted"][j], output["reason"][j] = uncertain, True, 0
  for key in ("v", "a", "long_active", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill",
              "recorded_command", "recorded_output", "recorded_state"):
    output[key] = np.asarray(tape[key]).copy()
  for service in ("carState", "carControl", "carOutput", "controlsState"):
    for suffix in ("_t", "_valid"):
      output[service + suffix] = np.asarray(tape[service + suffix]).copy()
  return output


def validate_output(out, inputs, source_directory=None):
  targets = (out.resolve(), out.with_suffix(".json").resolve())
  sources = [p.resolve() for p in inputs]
  for target in targets:
    if target.exists() or target in sources:
      raise ReplayError("output/receipt already exists or collides with an input")
    if source_directory is not None and target.is_relative_to(source_directory.resolve()):
      raise ReplayError("outputs must be outside the raw source segment")
  out.parent.mkdir(parents=True, exist_ok=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("prepare", "run"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--segment", type=Path)
  parser.add_argument("--schema-dir", type=Path)
  parser.add_argument("--rdf-output", type=Path)
  parser.add_argument("--input", type=Path)
  args = parser.parse_args()
  if args.out.suffix != ".npz":
    raise ReplayError("--out must be an NPZ with a separate JSON receipt")
  if args.mode == "prepare":
    if None in (args.segment, args.schema_dir, args.rdf_output):
      raise ReplayError("prepare requires --segment --schema-dir --rdf-output")
    log_path = next((args.segment / name for name in ("rlog.zst", "rlog.bz2", "rlog") if (args.segment / name).is_file()), None)
    if log_path is None:
      raise ReplayError("no source rlog")
    validate_output(args.out, [log_path, args.rdf_output, args.rdf_output.with_suffix(".json")], args.segment)
    rdf = dict(np.load(args.rdf_output, allow_pickle=False))
    streams, static, captures = read_recorded(log_path, args.schema_dir, int(args.segment.name.rsplit("--", 1)[1]))
    validate_rdf(rdf)
    for frame, timestamp in zip(rdf["frame_id"], rdf["narrow_t"], strict=True):
      if int(frame) not in captures or abs(captures[int(frame)] - timestamp) > 1e-6:
        raise ReplayError("RDF camera frame identity does not match this source rlog")
    arrays = prepare_arrays(streams, rdf)
    receipt = {"mode": "prepared", "source_segment": str(args.segment.resolve()), "source_log": str(log_path.resolve()),
               "source_log_sha256": digest(log_path), "rdf_output_sha256": digest(args.rdf_output), "static": static,
               "expected_native_sources_normalized_sha256": source_hashes(args.repo),
               "schema_sha256": {str(p.relative_to(args.schema_dir)): digest(p) for p in sorted(args.schema_dir.rglob("*.capnp"))},
               "static_config_timing": "CP/CP_SP are recorded static configuration; their publication times are preserved separately",
               "causality": "raw per-service timestamps sorted; latest valid-as-of <=200ms; invalid latest messages never skipped",
               "historical_opendbc_gitlink": None, "historical_opendbc_identity": "unknown; no controller equivalence claim"}
  else:
    if args.input is None:
      raise ReplayError("run requires --input")
    validate_output(args.out, [args.input, args.input.with_suffix(".json")])
    prepared = json.loads(args.input.with_suffix(".json").read_text())
    if digest(args.input) != prepared["output_npz_sha256"]:
      raise ReplayError("prepared input hash mismatch")
    hashes = source_hashes(args.repo)
    if hashes != prepared["expected_native_sources_normalized_sha256"]:
      raise ReplayError("native controller/dependency source differs from prepared current-master contract")
    sys.path[:0] = [str(args.repo), str(args.repo / "opendbc_repo")]
    from openpilot.selfdrive.controls.lib.longcontrol import LongControl
    from openpilot.common.realtime import DT_CTRL
    from opendbc.car.hyundai.interface import CarInterface

    if DT_CTRL != .01 or Path(inspect.getfile(LongControl)).resolve() != (args.repo / SOURCES[0]).resolve():
      raise ReplayError("actual native controller import or time step mismatch")
    tape = dict(np.load(args.input, allow_pickle=False))
    arrays = replay_native(tape, prepared["static"]["carParams"], prepared["static"]["carParamsSP"],
                           LongControl, CarInterface.get_pid_accel_limits)
    receipt = {"mode": "native_replay", "input": str(args.input.resolve()), "input_npz_sha256": digest(args.input),
               "source_segment": prepared["source_segment"], "source_log_sha256": prepared["source_log_sha256"],
               "rdf_output_sha256": prepared["rdf_output_sha256"], "native_sources_normalized_sha256": hashes,
               "native_import": inspect.getfile(LongControl), "limits_import": inspect.getfile(CarInterface.get_pid_accel_limits),
               "static": prepared["static"],
               "initialization": ("native constructor cold/reset: off, last_output=0, PID integral=0; unknown match to recorded controller; "
                                  +
                                  "labelled uncertain until an admitted disengagement resets state; gaps force a new cold reset"),
               "reason_codes": {0: "admitted", 1: "model unavailable/history warmup", 2: "model capture stale",
                                3: "carState stale/invalid", 4: "carControl stale/invalid", 5: "nonfinite/nonboolean vehicle input"},
               "admitted_rows": int(arrays["admitted"].sum()),
               "initialization_uncertain_admitted_rows": int(np.count_nonzero(arrays["initialization_uncertain"] & arrays["admitted"])),
               "max_admitted_model_capture_age_ms": float(np.nanmax(arrays["rdf_capture_age_s"]) * 1000),
               "reason_counts": {int(k): int(v) for k, v in zip(*np.unique(arrays["reason"], return_counts=True), strict=True)},
               "semantics": ("actual current-master LongControl at synthetic100Hz phase on causal recorded motion/control-active inputs; "
                             +
                             "no gain changes, plant response, optimization, or road improvement claim"),
               "no_actuation": True, "no_params_writes": True}
  np.savez_compressed(args.out, **arrays)
  receipt.update(output=str(args.out.resolve()), output_npz_sha256=digest(args.out), rows=len(arrays["t"]),
                 dt=.01, argv=sys.argv, adapter_sha256=digest(Path(__file__)))
  args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
  print(json.dumps({key: receipt[key] for key in ("mode", "output", "rows", "output_npz_sha256")}))


if __name__ == "__main__":
  main()
