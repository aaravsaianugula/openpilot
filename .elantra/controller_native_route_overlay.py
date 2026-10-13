"""Explicit plural-source native command overlay; recorded feedback stays fixed."""
import json
from pathlib import Path

import numpy as np

import controller_native_replay as native
import controller_native_route as route
from controller_model_overlay import RenderError


def validate_plural_arrays(output, tape, rdf, censor):
  route.validate_tape(tape)
  if not np.array_equal(output.get("t"), tape["t"]) or float(output.get("raw_censor_t", np.nan)) != censor:
    raise RenderError("plural native clock/censor differs from prepared raw tape")
  n = len(tape["t"])
  numeric = ("native_accel", "native_state", "pid_integral", "rdf_accel_target", "rdf_capture_age_s", "accel_min", "accel_max")
  flags = ("admitted", "initialization_uncertain", "should_stop", "model_display_rejected", "pre_censor_admitted",
           "controlled_command_available", "post_censor_diagnostic")
  for key in (*numeric, *flags, "rdf_index", "latest_ready_rdf_index"):
    array = np.asarray(output.get(key))
    if array.shape != (n,) or (key in flags and array.dtype.kind != "b"):
      raise RenderError("plural native output fields/flags differ")
  if any(np.asarray(output[key]).dtype.kind not in "fi" for key in numeric):
    raise RenderError("plural native output must have numeric scalar traces")
  for key in ("rdf_index", "latest_ready_rdf_index"):
    if np.asarray(output[key]).dtype.kind != "i":
      raise RenderError("plural native indices must be integers")
  raw = ("v", "a", "long_active", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill",
         "recorded_command", "recorded_output", "recorded_state")
  for key in (*raw, *(service+suffix for service in route.FIELDS for suffix in ("_t", "_valid"))):
    if key not in output or not np.array_equal(output[key], tape[key], equal_nan=True):
      raise RenderError("plural native recorded input changed: " + key)
  _, rejected, latest = route.gate_native_tape(tape)
  safe = np.maximum(latest, 0)
  age = tape["t"] - rdf["narrow_t"][safe]
  expected = ((latest >= 132) & ~rejected & (age >= 0) & (age <= .2+1e-9)
              & tape["carState_valid"] & tape["carControl_valid"])
  inputs = ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise", "long_active")
  expected &= np.logical_and.reduce([np.isfinite(tape[key]) for key in inputs])
  expected &= np.logical_and.reduce([np.isin(tape[key], (0., 1.)) for key in route.FLAGS])
  for service in ("carState", "carControl"):
    service_age = tape["t"] - tape[service+"_t"]
    expected &= np.isfinite(service_age) & (service_age >= 0) & (service_age <= .2+1e-9)
  if (not np.array_equal(output["admitted"], expected) or not np.array_equal(output["model_display_rejected"], rejected)
      or not np.array_equal(output["latest_ready_rdf_index"], latest) or not output["initialization_uncertain"].all()):
    raise RenderError("plural native admission/latest-ready/cold-state proof differs")
  for key in numeric:
    if not np.isfinite(output[key][expected]).all() or not np.isnan(output[key][~expected]).all():
      raise RenderError("plural native unavailable samples must stay missing")
  indices = latest[expected]
  if (not np.array_equal(output["rdf_index"][expected], indices)
      or not np.array_equal(output["rdf_accel_target"][expected], rdf["desired_acceleration"][indices])
      or not np.array_equal(output["should_stop"][expected], rdf["should_stop"][indices])
      or not np.allclose(output["rdf_capture_age_s"][expected], age[expected], atol=1e-8, rtol=0)
      or not np.isin(output["native_state"][expected], (0., 1., 2.)).all()):
    raise RenderError("plural native RDF target/state/capture proof differs")
  if (not np.all(output["accel_min"][expected] == -3.5) or not np.all(output["accel_max"][expected] == 2.)
      or np.any(output["native_accel"][expected] < -3.5) or np.any(output["native_accel"][expected] > 2.)):
    raise RenderError("plural native actual Hyundai acceleration limits differ")
  before = expected & (tape["t"] < censor) & (rdf["narrow_t"][safe] < censor) & (rdf["output_ready_t"][safe] < censor)
  controlled = before & (tape["long_active"] == 1)
  for key in ("brake_pressed", "gas_pressed", "steering_pressed"):
    controlled &= tape[key] == 0
  if (not np.array_equal(output["pre_censor_admitted"], before)
      or not np.array_equal(output["controlled_command_available"], controlled)
      or not np.array_equal(output["post_censor_diagnostic"], expected & (tape["t"] >= censor))):
    raise RenderError("plural native takeover masks differ")
  return {"native_admitted": int(expected.sum()), "prebrake_controlled_command_available": int(controlled.sum()),
          "post_censor_diagnostic": int(output["post_censor_diagnostic"].sum()),
          "admitted_should_stop": int(np.count_nonzero(expected & output["should_stop"]))}


def load_plural_native(output_path, transfer_path, input_path, evidence, rdf, repo):
  receipt_path, prepared_path = output_path.with_suffix(".json"), input_path.with_suffix(".json")
  receipt, prepared, transfer = [json.loads(path.read_text()) for path in (receipt_path, prepared_path, transfer_path)]
  if (transfer_path.parent.resolve() != output_path.parent.resolve() or transfer.get("status") != "verified"
      or transfer.get("transfer_exit_code") != 0 or receipt.get("mode") != "native_route_replay"
      or prepared.get("mode") != "prepared_native_route" or receipt.get("multi_source") is not True
      or prepared.get("multi_source") is not True or receipt.get("no_actuation") is not True or receipt.get("no_params_writes") is not True
      or receipt.get("route_evidence") != evidence or prepared.get("route_evidence") != evidence):
    raise RenderError("requires explicitly bound plural native route and verified transfer")
  for name, row in transfer.get("files", {}).items():
    path = transfer_path.parent / name
    if Path(name).name != name or path.is_symlink() or path.stat().st_size != row["bytes"] or native.digest(path) != row["sha256"]:
      raise RenderError("plural native transferred file changed")
  if (output_path.name not in transfer["files"] or receipt_path.name not in transfer["files"]
      or receipt.get("output_npz_sha256") != native.digest(output_path)
      or receipt.get("input_npz_sha256") != native.digest(input_path)
      or prepared.get("output_npz_sha256") != native.digest(input_path)
      or receipt.get("input_receipt_sha256") != native.digest(prepared_path)
      or receipt.get("static") != prepared.get("static")):
    raise RenderError("plural native input/output/CP receipt hashes differ")
  current = route.adapter_sources()
  normalized = {key: row["normalized_sha256"] for key, row in current.items()}
  for key in ("prepare_adapter_sources", "run_adapter_sources"):
    if {name: row["normalized_sha256"] for name, row in receipt.get(key, {}).items()} != normalized:
      raise RenderError("plural native executed adapter differs from reviewed source")
  if (receipt.get("native_sources_normalized_sha256") != native.source_hashes(repo)
      or receipt["native_sources_normalized_sha256"] != prepared.get("expected_native_sources_normalized_sha256")):
    raise RenderError("plural native source snapshot changed")
  with np.load(output_path, allow_pickle=False) as loaded:
    output = dict(loaded)
  with np.load(input_path, allow_pickle=False) as loaded:
    tape = dict(loaded)
  counts = validate_plural_arrays(output, tape, rdf, evidence["raw_censor_t"])
  paths = [output_path, receipt_path, input_path, prepared_path, transfer_path, Path(__file__), Path(route.__file__), Path(native.__file__)]
  return output, {"input_sha256": {str(path.resolve()): native.digest(path) for path in paths}, "counts": counts,
                  "scope": "actual native commands on fixed recorded feedback; all cold uncertain; takeover diagnostic only"}
