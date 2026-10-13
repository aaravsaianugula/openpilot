"""Native longitudinal/identified-plant research core; no file writes or deployment.

LongControl receives simulated acceleration/speed. Recorded future GT is never
feedback or a scoring reference. Targets/images and cruise-standstill inputs stay
frozen; perception/OEM standstill feedback and a complete lateral tune are absent.
Unit factories are code fixtures, not native-runtime or vehicle evidence.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import inspect
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np

import controller_native_replay as native
import controller_search as research
import controller_stops as stops
import controller_tune as tune

NATIVE_DT = .01
MODEL_DT = .05
ACTION_OFFSET_S = .075
LONG_SOURCE = native.SOURCES[0]
LONG_SHA = "fed627d936b8f3166ad6b6af9e63561b8747bab46bdb0fb493164f9ca1ea73b4"
KI_GRID = (0., .05, .1, .2)
STOP_GRID = (-1.5, -2., -2.5)
ACCEL_LIMITS = (-3.5, 2.)


class SearchError(ValueError):
  pass


@dataclass(frozen=True)
class NativeRuntime:
  controller_type: Callable
  accel_limits: Callable
  source_hashes: dict
  opendbc_gitlink: str
  actual_native: bool
  static_builder: Callable = native.validate_parameters


def _material(value):
  if isinstance(value, (bytes, bytearray)):
    return {"capnp_data_hex": bytes(value).hex()}
  if isinstance(value, np.ndarray):
    if value.dtype.hasobject:
      raise SearchError("object arrays are not a native input contract")
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
  if isinstance(value, dict):
    if any(not isinstance(key, str) for key in value):
      raise SearchError("contract dictionary keys must be strings")
    return {key: _material(item) for key, item in sorted(value.items())}
  if isinstance(value, (list, tuple)):
    return [_material(item) for item in value]
  return value.item() if isinstance(value, np.generic) else value


def value_hash(value):
  return hashlib.sha256(json.dumps(_material(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def rdf_from_tape(tape):
  return {key[4:]: value for key, value in tape.items() if key.startswith("rdf_")}


def source_hashes(repo):
  hashes = native.source_hashes(repo)
  name = "openpilot/cereal/custom.capnp"
  hashes[name] = hashlib.sha256((Path(repo) / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
  return hashes


def native_messages(cp_dict, sp_dict):
  native.validate_parameters(cp_dict, sp_dict)
  from opendbc.car.structs import car
  from openpilot.cereal import custom
  return car.CarParams.new_message(**cp_dict), custom.CarParamsSP.new_message(**sp_dict)


def load_native_runtime(repo, expected_sources, expected_gitlink):
  """Owner-serialized POSIX API validation; never invoke this in host fixtures."""
  if os.name != "posix":
    raise SearchError("actual native runtime loading requires POSIX")
  repo = Path(repo).resolve()
  hashes = source_hashes(repo)
  if hashes != expected_sources or hashes.get(LONG_SOURCE) != LONG_SHA:
    raise SearchError("native source changed; Kp0/stopping-ramp1 contract is fixed")
  if not (repo / "opendbc_repo/.git").exists():
    raise SearchError("partial source snapshot lacks independently verified host-pin manifest; no clone or identity inference performed")
  pin = subprocess.run(["git", "-C", str(repo / "opendbc_repo"), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
  if not re.fullmatch(r"[0-9a-f]{40}", expected_gitlink) or pin != expected_gitlink:
    raise SearchError("actual native opendbc pin differs")
  sys.path[:0] = [str(repo), str(repo / "opendbc_repo")]
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl
  from openpilot.common.realtime import DT_CTRL
  from opendbc.car.hyundai.interface import CarInterface
  if (DT_CTRL != NATIVE_DT or Path(inspect.getfile(LongControl)).resolve() != repo / LONG_SOURCE
      or Path(inspect.getfile(CarInterface.get_pid_accel_limits)).resolve() != repo / "opendbc_repo/opendbc/car/interfaces.py"):
    raise SearchError("native import/acceleration-limit source mismatch")
  return NativeRuntime(LongControl, CarInterface.get_pid_accel_limits, hashes, pin, True, native_messages)


def native_parameters(cp, parameters=None):
  baseline = {"ki_v": list(cp["longitudinalTuning"]["kiV"]), "stop_accel": cp["stopAccel"]}
  if parameters is None:
    return baseline
  if set(parameters) != {"ki_v", "stop_accel"}:
    raise SearchError("only declared KiV/stopAccel parameters may change; model delays are fixed")
  values = np.asarray(parameters["ki_v"], dtype=float)
  if (values.shape != np.asarray(baseline["ki_v"]).shape or not np.isfinite(values).all()
      or any(value not in KI_GRID for value in values) or parameters["stop_accel"] not in STOP_GRID):
    raise SearchError("candidate outside finite native KiV/stopAccel allowlist")
  return {"ki_v": values.tolist(), "stop_accel": float(parameters["stop_accel"])}


def resampled_alpha(tau):
  if not np.isfinite(tau) or tau <= 0 or tune.DT != MODEL_DT:
    raise SearchError("positive20Hz identified time constant required")
  alpha_20 = MODEL_DT / (tau + MODEL_DT)
  return 1 - (1-alpha_20)**(NATIVE_DT/MODEL_DT)


def admit(window, report, proof, runtime, parameters):
  hashes = {"window_sha256": value_hash(window), "plant_report_sha256": value_hash(report),
            "rdf_sha256": value_hash(rdf_from_tape(window["tape"])), "model_inputs_sha256": value_hash(window["model_inputs"])}
  if any(proof.get(key) != value for key, value in hashes.items()):
    raise SearchError("window/plant/model-input hash binding mismatch")
  if (proof.get("native_sources") != runtime.source_hashes or runtime.source_hashes.get(LONG_SOURCE) != LONG_SHA
      or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in runtime.source_hashes.values())):
    raise SearchError("native source hash contract differs")
  if runtime.actual_native:
    actual = Path(inspect.getfile(runtime.controller_type)).read_bytes().replace(b"\r\n", b"\n")
    limits = Path(inspect.getfile(runtime.accel_limits)).read_bytes().replace(b"\r\n", b"\n")
    if (hashlib.sha256(actual).hexdigest() != LONG_SHA or runtime.static_builder is not native_messages
        or hashlib.sha256(limits).hexdigest() != runtime.source_hashes.get("opendbc_repo/opendbc/car/interfaces.py")):
      raise SearchError("actual native controller/message/limit factory differs from source proof")
  pin = window.get("opendbc_gitlink")
  if (not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{40}", pin)
      or pin != runtime.opendbc_gitlink or pin != proof.get("plant_opendbc_gitlink")):
    raise SearchError("same known plant/window/native opendbc pin required")
  if (not window.get("cohort") or window["cohort"] != report.get("cohort")
      or not re.fullmatch(r"[0-9a-f]{64}", window.get("source_log_sha256", ""))
      or proof.get("source_log_sha256") != window["source_log_sha256"]):
    raise SearchError("same-cohort source-log provenance required")
  walls = report.get("route_start_wall_time_nanos", {})
  catalog = [(None, {"route": route, "init_data": {"wallTimeNanos": wall}}) for route, wall in walls.items()]
  research.validate_split(catalog, report["split"])
  if window.get("partition") != "validation" or window["route"] not in report["split"]["validation"]:
    raise SearchError("native search refuses heldout/train windows; immutable validation only")
  models, rejected = research.trusted_models(report)
  model = models.get("long")
  if (model is None or model.get("identified") is not True or model.get("recorded_build_gitlink_known") is not True
      or not model.get("table") or model.get("kind") == "median_transition"):
    raise SearchError("untrusted longitudinal plant: " + repr(rejected.get("long", [])))
  native.validate_parameters(window["cp"], window["sp"])
  candidate = native_parameters(window["cp"], parameters)
  inputs, rdf = window["model_inputs"], rdf_from_tape(window["tape"])
  native.validate_rdf(rdf)
  for field in ("controller_admissible", "deadline_admissible"):
    flag = np.asarray(rdf.get(field))
    if flag.shape != (len(rdf["input_t"]),) or flag.dtype.kind != "b":
      raise SearchError("explicit boolean RDF controller/deadline admission required")
  delay = window["cp"].get("longitudinalActuatorDelay")
  action_t = np.asarray(inputs.get("action_t"))
  packet_t = np.asarray(inputs.get("packet_action_t"))
  consumed_t, recorded_packet_t = np.asarray(rdf.get("action_t")), np.asarray(rdf.get("packet_action_t"))
  includes_smoothing = inputs.get("packet_includes_model_smoothing")
  smoothing = .3 if includes_smoothing is True else 0.
  offset = np.array([0., 0.]) if includes_smoothing is True else np.array([.1, .3])
  if (delay is None or not np.isfinite(delay) or delay < 0 or inputs.get("fresh_inference") is not True
      or inputs.get("model_sha256") != native.MODEL_SHA or action_t.shape != (len(rdf["input_t"]), 2)
      or type(includes_smoothing) is not bool or packet_t.shape != action_t.shape
      or not np.array_equal(action_t, consumed_t) or not np.array_equal(packet_t, recorded_packet_t)
      or not np.isfinite(action_t).all() or not np.isfinite(packet_t).all()
      or not np.allclose(action_t, packet_t+offset, atol=1e-7, rtol=0)
      or not np.allclose(packet_t[:, 1], delay + ACTION_OFFSET_S+smoothing, atol=1e-7, rtol=0)):
    raise SearchError("fixed CP delay requires matching fresh RDF action_t inputs; no posthoc delay relabelling")
  if window.get("reference_mode") not in ("capture_horizon", "command_target_only"):
    raise SearchError("explicit physical-time or command-target-only reference required")
  if window.get("regime", "moving") not in ("moving", "stopping"):
    raise SearchError("explicit moving or stopping evaluation regime required")
  return model, candidate


def command_prefix(window, delay):
  if not np.isfinite(delay) or delay < 0:
    raise SearchError("finite nonnegative physical plant delay required")
  count = int(round(delay/NATIVE_DT))
  if abs(count*NATIVE_DT-delay) > 1e-9:
    raise SearchError("physical plant delay must be a nonnegative native-grid multiple")
  if not count:
    return []
  source_t, command, valid = (np.asarray(window[key]) for key in ("history_t", "history_command", "history_valid"))
  anchor = float(window["tape"]["t"][0])
  if (source_t.ndim != 1 or source_t.shape != command.shape or source_t.shape != valid.shape
      or not len(source_t) or not np.isfinite(source_t).all() or np.any(np.diff(source_t) <= 0)
      or np.any(source_t >= anchor) or valid.dtype.kind != "b"):
    raise SearchError("strictly past chronological transmitted-command delay prefix required")
  nodes = anchor - np.arange(count, 0, -1)*NATIVE_DT
  indices, admitted = native.asof_indices(source_t, valid, nodes, max_age=.075)
  if not admitted.all():
    raise SearchError("physical command delay prefix missing/stale/invalid")
  values = command[indices]
  if not np.isfinite(values).all() or np.any(values < ACCEL_LIMITS[0]) or np.any(values > ACCEL_LIMITS[1]):
    raise SearchError("invalid delay-prefix transmitted commands")
  return values.tolist()


def physical_reference(rdf, index, timestamp, mode):
  if mode == "command_target_only":
    return float("nan")
  acceleration, horizon = np.asarray(rdf.get("acceleration")), np.asarray(rdf.get("horizon"))
  n = len(rdf["input_t"])
  if (acceleration.shape != (n, 33, 3) or horizon.shape != (33,) or not np.isfinite(horizon).all()
      or np.any(np.diff(horizon) <= 0) or not np.isfinite(acceleration[index, :, 0]).all()):
    raise SearchError("capture+horizon-aligned raw model acceleration reference unavailable")
  age = timestamp - float(rdf["narrow_t"][index])
  if not horizon[0] <= age <= horizon[-1]:
    raise SearchError("physical tick outside raw prediction horizon; no extrapolation")
  return float(np.interp(age, horizon, acceleration[index, :, 0]))


def stopping_support(model):
  support = model.get("low_speed_validation", {})
  return (model.get("stopping_identified") is True and support.get("used_windows", 0) >= 20
          and support.get("coverage", 0) >= .5 and support.get("standstill_hold_identified") is True)


def reset_observed(window):
  evidence = window.get("reset_evidence", {})
  timestamp = evidence.get("t")
  return (evidence.get("valid") is True and timestamp is not None and np.isfinite(timestamp)
          and 0 < window["tape"]["t"][0]-timestamp <= .02+1e-9
          and evidence.get("long_active") is False and evidence.get("native_state") == 0
          and evidence.get("transmitted_accel") == 0.)


def stop_prediction_identity(window):
  inputs = window["model_inputs"]
  return {"cohort": window["cohort"], "opendbc_gitlink": window["opendbc_gitlink"], "model_sha256": native.MODEL_SHA,
          "native_source_sha256": LONG_SHA, "model_source_sha256": inputs.get("model_source_sha256"),
          "cp_sha256": value_hash(window["cp"]), "sp_sha256": value_hash(window["sp"]),
          "packet_includes_model_smoothing": inputs["packet_includes_model_smoothing"],
          "long_action_t": float(inputs["action_t"][0, 1]), "packet_long_action_t": float(inputs["packet_action_t"][0, 1])}


def verified_stop_uncertainty(window, train_routes):
  """Check frozen retained TRAIN residuals; never fit or fill an absent margin."""
  uncertainty = window.get("stop_target_uncertainty", {})
  body, bound = uncertainty.get("evidence", {}), uncertainty.get("bound_m")
  identity = stop_prediction_identity(window)
  if (not body or uncertainty.get("source_sha256") != value_hash(body) or body.get("identity") != identity
      or not isinstance(identity["model_source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", identity["model_source_sha256"])
      or body.get("method") != "empirical_higher_absolute_model_vs_observed_stop_residual" or body.get("quantile") != .95
      or bound is None or not np.isfinite(bound) or bound < 0):
    return None
  rows = body.get("outcomes", [])
  if not isinstance(rows, list) or len(rows) < 20 or body.get("sample_count") != len(rows):
    return None
  residuals, by_route, log_routes, ids = [], {}, {}, set()
  for row in rows:
    route, event_id = row.get("route"), row.get("event_id")
    values = [row.get(key) for key in ("capture_t", "observed_stop_t", "model_distance_m", "observed_distance_m", "residual_m")]
    if (any(value is None for value in values) or not np.isfinite(values).all() or min(values[2:4]) < 0
        or not 0 < values[1]-values[0] <= 15 or route not in train_routes or not event_id or event_id in ids
        or row.get("partition") != "train" or row.get("identity_sha256") != value_hash(identity)
        or row.get("uninterrupted") is not True or row.get("long_active_entire_episode") is not True
        or row.get("valid_motion_entire_episode") is not True or row.get("interventions") != []
        or any(not re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")) for key in ("source_log_sha256", "raw_model_plan_sha256"))
        or abs(values[4]-(values[3]-values[2])) > 1e-9):
      return None
    log_sha = row["source_log_sha256"]
    if log_sha in log_routes and log_routes[log_sha] != route:
      return None  # One retained physical log cannot establish independence across route labels.
    log_routes[log_sha] = route
    ids.add(event_id)
    by_route.setdefault(route, []).append((values[0], values[1]))
    residuals.append(abs(values[4]))
  if len(by_route) < 2 or body.get("route_count") != len(by_route):
    return None
  for episodes in by_route.values():
    ordered = sorted(episodes)
    if any(before[1] > after[0] for before, after in zip(ordered, ordered[1:], strict=False)):
      return None
  ordered = sorted(residuals)
  empirical = ordered[int(np.ceil(.95*(len(ordered)-1)))]
  return float(bound) if abs(bound-empirical) <= 1e-9 else None


def model_stopping_target(window, trace, origin, train_routes):
  """Causal model prediction minus simulated past travel; never measured stop GT."""
  rdf = rdf_from_tape(window["tape"])
  uncertainty = window.get("stop_target_uncertainty", {})
  bound = verified_stop_uncertainty(window, train_routes)
  if bound is None:
    return None
  index = int(trace["rdf_index"][origin])
  capture, request = float(rdf["narrow_t"][index]), float(trace["t"][origin])
  position, horizon = np.asarray(rdf.get("position")), np.asarray(rdf.get("horizon"))
  if position.shape != (len(rdf["input_t"]), 33, 3) or horizon.shape != (33,) or not trace["t"][0] <= capture <= request:
    return None  # No recorded-future or invented pre-anchor displacement fallback.
  # prediction_at consumes only row1 plus t[0] as the covered interval origin.
  sample = {"t": np.array([trace["t"][0], request]), "model_t": np.array([np.nan, rdf["output_ready_t"][index]]),
            "model_capture_t": np.array([np.nan, capture]), "model_valid": np.array([np.nan, 1.]),
            "model_horizon": np.array([np.full(33, np.nan), horizon]),
            "model_x": np.array([np.full(33, np.nan), position[index, :, 0]]),
            "model_y": np.array([np.full(33, np.nan), position[index, :, 1]])}
  prediction = stops.prediction_at(sample, 1)
  if prediction is None:
    return None
  past = float(trace["distance"][origin] - np.interp(capture, trace["t"][:origin+1], trace["distance"][:origin+1]))
  remaining = prediction["distance_m"] - past
  if not np.isfinite(remaining) or remaining < 0:
    return None
  return {"distance_from_request_m": remaining, "lower_m": max(0., remaining-bound), "upper_m": remaining+bound,
          "rdf_index": index, "capture_t": capture, "available_t": float(rdf["output_ready_t"][index]), "request_t": request,
          "path_sha256": value_hash({"position": position[index], "horizon": horizon}),
          "uncertainty_source_sha256": uncertainty["source_sha256"], "past_travel_source": "simulated past only",
          "semantics": "causal model stationary-position prediction and frozen TRAIN uncertainty, not stopline ground truth"}


def evaluate_trial(window, report, proof, runtime, parameters=None, *, stopping=False):
  model, parameters = admit(window, report, proof, runtime, parameters)
  tape, rdf = window["tape"], rdf_from_tape(window["tape"])
  t = np.asarray(tape["t"])
  fields = ("long_active", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "valid", "v_cruise")
  timing = tuple(service+suffix for service in ("carState", "carControl") for suffix in ("_t", "_valid"))
  if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.allclose(np.diff(t), NATIVE_DT, rtol=0, atol=1e-9)
      or any(np.asarray(tape[key]).shape != t.shape for key in (*fields, *timing))):
    raise SearchError("continuous100Hz inputs and matching causal exogenous fields required")
  cp_dict = deepcopy(window["cp"])
  cp_dict["longitudinalTuning"]["kiV"], cp_dict["stopAccel"] = parameters["ki_v"], parameters["stop_accel"]
  cp, sp = runtime.static_builder(cp_dict, window["sp"])
  history = command_prefix(window, model["delay_s"])
  alpha = resampled_alpha(model["tau_s"])
  speed, acceleration = float(window["initial_v"]), float(window["initial_a"])
  if not np.isfinite([speed, acceleration]).all() or speed < 0:
    raise SearchError("finite measured anchor acceleration/nonnegative speed required")
  controller = runtime.controller_type(cp, sp)
  trace = {key: np.full(len(t), np.nan) for key in ("v", "a", "command", "native_state", "pid_integral",
            "feedforward_target", "physical_reference", "rdf_index", "delayed_command", "distance")}
  trace["t"] = t.copy()
  completed, reason, censor_t, terminal_state = 0, None, None, None
  low_speed_supported = stopping_support(model)
  if stopping and not low_speed_supported:
    reason = "stopping_plant_unidentified"
  distance = 0.
  indices = np.searchsorted(rdf["output_ready_t"], t, side="right") - 1
  for j, timestamp in enumerate(t):
    if reason:
      break
    if not np.isfinite([speed, acceleration]).all():
      reason = "nonfinite_simulated_plant_state"
      break
    values = [tape[key][j] for key in fields]
    if (not np.isfinite(values).all() or any(tape[key][j] not in (0, 1) for key in fields[:-1])
        or tape["v_cruise"][j] < 0 or not tape["valid"][j]
        or any(not tape[service+"_valid"][j] or not np.isfinite(tape[service+"_t"][j])
               or not 0 <= timestamp-tape[service+"_t"][j] <= .2+1e-9 for service in ("carState", "carControl"))):
      reason, censor_t = "invalid_causal_vehicle_input", float(timestamp)
      break
    cs = native.SimpleNamespace(vEgo=speed, aEgo=acceleration, brakePressed=bool(tape["brake_pressed"][j]),
                                cruiseState=native.SimpleNamespace(standstill=bool(tape["cruise_standstill"][j])))
    limits = runtime.accel_limits(cp, sp, speed, float(tape["v_cruise"][j])/3.6)
    if not np.array_equal(np.asarray(limits), ACCEL_LIMITS):
      raise SearchError("fixed native Hyundai acceleration/safety limits changed")
    if not tape["long_active"][j]:
      controller.reset()
      controller.update(False, cs, 0., False, limits)
      terminal_state = int(controller.long_control_state)
      reason, censor_t = "longitudinal_disengagement", float(timestamp)
      break
    if any(tape[key][j] for key in ("brake_pressed", "gas_pressed", "steering_pressed")):
      reason, censor_t = "driver_takeover", float(timestamp)
      break
    index = int(indices[j])
    if (index < 0 or not rdf["scored_mask"][index] or not 0 <= timestamp-rdf["narrow_t"][index] <= .2+1e-9):
      reason, censor_t = "model_unavailable_warmup_or_stale", float(timestamp)
      break
    if not rdf["controller_admissible"][index] or not rdf["deadline_admissible"][index]:
      reason, censor_t = "latest_model_controller_or_deadline_inadmissible", float(timestamp)
      break
    target, stop = float(rdf["desired_acceleration"][index]), bool(rdf["should_stop"][index])
    reference = physical_reference(rdf, index, timestamp, window["reference_mode"])
    command = float(controller.update(True, cs, target, stop, limits))
    if not np.isfinite(command) or not limits[0] <= command <= limits[1]:
      reason = "native_command_nonfinite_or_out_of_bounds"
      break
    history.append(command)
    delayed = history.pop(0)
    steady = tune.lookup(model["table"], speed, delayed)
    if not np.isfinite(steady):
      reason = "unsupported_speed_or_command_cell"
      break
    next_speed = max(0., speed + acceleration*NATIVE_DT)
    next_acceleration = acceleration + alpha*(steady-acceleration)
    if next_speed == 0 and next_acceleration < 0:
      if not low_speed_supported:
        reason = "unidentified_standstill_transition"
        break
      next_acceleration = 0.  # Non-reversing constraint, requires separately validated standstill support.
    for key, value in (("v", speed), ("a", acceleration), ("command", command), ("native_state", int(controller.long_control_state)),
                       ("pid_integral", float(controller.pid.i)), ("feedforward_target", target), ("physical_reference", reference),
                       ("rdf_index", index), ("delayed_command", delayed), ("distance", distance)):
      trace[key][j] = value
    completed += 1
    distance += speed*NATIVE_DT
    speed, acceleration = next_speed, next_acceleration
  physical = window["reference_mode"] == "capture_horizon"
  rejections = []
  if not runtime.actual_native:
    rejections.append("code_fixture_not_native_runtime_evidence")
  if not physical:
    rejections.append("physical_time_reference_unavailable")
  if not low_speed_supported:
    rejections.append("native_stopping_plant_unidentified")
  if not reset_observed(window):
    rejections.append("native_initialization_uncertain")
  if completed != len(t):
    rejections.append("incomplete_supported_horizon")
  stop_result = None
  if stopping:
    requested = np.flatnonzero(np.isfinite(trace["native_state"]) & (trace["native_state"] == 2))
    sustained = int(round(.5/NATIVE_DT)) + 1
    stationary = np.isfinite(trace["v"]) & (trace["v"] <= .3) & (trace["native_state"] == 2)
    found = [j for j in range(len(t)-sustained+1) if stationary[j:j+sustained].all()]
    if requested.size and found:
      origin, end = int(requested[0]), found[0]
      target = model_stopping_target(window, trace, origin, report["split"]["train"])
      stop_result = {"distance_from_request_m": float(trace["distance"][end]-trace["distance"][origin]),
                     "time_from_request_s": float(t[end]-t[origin]), "stationary_confirmation_s": .5, "model_target": target,
                     "speed_at_native_stop_request_mps": float(trace["v"][origin]),
                     "max_simulated_approach_speed_mps": float(np.max(trace["v"][:origin+1]))}
      if stop_result["max_simulated_approach_speed_mps"] < 2:
        rejections.append("moving_approach_history_unavailable")
      if not np.all(trace["native_state"][origin:end+1] == 2):
        rejections.append("noncontinuous_native_stopping_episode")
      stop_result["target_feasible"] = target is not None and target["lower_m"] <= stop_result["distance_from_request_m"] <= target["upper_m"]
      if target is None:
        rejections.append("model_stopping_target_feasibility_unverified")
      elif not stop_result["target_feasible"]:
        rejections.append("stop_outside_model_prediction_uncertainty_interval")
    else:
      rejections.append("native_stop_not_completed_within_horizon")
  return {"window_id": window["id"], "parameters": parameters, "trace": trace,
          "status": "complete" if completed == len(t) else "censored" if censor_t is not None else "unsupported",
          "reason": reason, "censor_t": censor_t, "terminal_native_state": terminal_state,
          "used_steps": completed, "attempted_steps": len(t), "coverage": completed/len(t),
          "reference_mode": window["reference_mode"], "stopping_supported": low_speed_supported,
          "stopping_result": stop_result,
          "selection_eligible": not rejections, "selection_rejections": rejections,
          "evidence": "actual native class" if runtime.actual_native else "unit code fixture only",
          "assumptions": ["20Hz backward-Euler pole resampled at100Hz; high-frequency plant response unproven",
                          "frozen recorded targets/images and OEM cruise standstill; no model/OEM feedback simulation"],
          "source_binding": {key: proof[key] for key in ("window_sha256", "plant_report_sha256", "rdf_sha256", "model_inputs_sha256")},
          "production_enabled": False}


def _window_score(result):
  trace = result["trace"]
  if result["status"] != "complete" or result["reference_mode"] != "capture_horizon":
    return None
  error, target = np.abs(trace["a"]-trace["physical_reference"]), trace["physical_reference"]
  return {"errors": error.tolist(), "acceleration_errors": error[target > .05].tolist(),
          "braking_errors": error[target < -.05].tolist(),
          "jerk": (np.abs(np.diff(trace["a"]))/NATIVE_DT).tolist(),
          "command_rate": (np.abs(np.diff(trace["command"]))/NATIVE_DT).tolist(),
          "saturated": ((trace["command"] <= -3.5+1e-8) | (trace["command"] >= 2-1e-8)).astype(float).tolist(),
          "stopping_result": result["stopping_result"]}


def summarize(windows, common):
  metrics = {"tracking": {}, "windows": len(common)}
  for direction in ("acceleration", "braking"):
    values = [error for key in common for error in windows[key][direction+"_errors"]]
    metrics["tracking"][direction] = {"points": len(values), "windows": sum(bool(windows[key][direction+"_errors"]) for key in common),
      "mae": float(np.mean(values)) if values else None, "p95": float(np.percentile(values, 95)) if values else None}
  for name in ("jerk", "command_rate", "saturated"):
    values = [value for key in common for value in windows[key][name]]
    metrics[name] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
  scores = [score["mae"] for score in metrics["tracking"].values()]
  metrics["target_objective"] = float(np.mean(scores)) if all(value is not None for value in scores) else None
  return metrics


def select_trials(rows, attempted):
  """One fixed common population; conservative policy constraints are not native slew limits."""
  before, candidates = rows[0], rows[1:]
  eligible = [row for row in candidates if row["individually_supported"]]
  common = set(before["windows"])
  for row in eligible:
    common &= set(row["windows"])
  common = sorted(common) if eligible and before["individually_supported"] else []
  enough = len(common) >= research.MIN_WINDOWS and len(common)/attempted >= research.MIN_COVERAGE
  baseline = summarize(before["windows"], common) if enough else None
  for row in rows:
    supported = row["individually_supported"] and enough
    row["common_window_ids"], row["common_coverage"] = common if supported else [], len(common)/attempted if supported else 0.
    row["metrics"] = summarize(row["windows"], common) if supported else None
    if not supported:
      row["selection_rejections"].append("insufficient_fixed_common_population")
      continue
    if row is before:
      continue
    row["selection_rejections"].extend(research._constraints(row["metrics"], baseline, "long", research.MIN_DIRECTION_POINTS))
    stops = [key for key in common if before["windows"][key]["stopping_result"] is not None]
    if len(stops) < 2:
      row["selection_rejections"].append("insufficient_completed_native_stopping_windows")
    for key in stops:
      old, new = before["windows"][key]["stopping_result"], row["windows"][key]["stopping_result"]
      if new is None:
        row["selection_rejections"].append("native_stop_not_completed")
      elif (old.get("model_target") is None or new.get("model_target") is None or new.get("target_feasible") is not True
            or not new["model_target"]["lower_m"] <= new["distance_from_request_m"] <= new["model_target"]["upper_m"]):
        row["selection_rejections"].append("model_stopping_target_feasibility_unverified")
      elif any(old["model_target"].get(field) != new["model_target"].get(field) for field in ("request_t", "capture_t", "path_sha256")):
        row["selection_rejections"].append("stopping_reference_anchor_mismatch")
      elif new["distance_from_request_m"] > old["distance_from_request_m"]+.1 or new["time_from_request_s"] > old["time_from_request_s"]+.1:
        row["selection_rejections"].append("native_stopping_regression")
  ranked = sorted([row for row in candidates if row["metrics"] and row["metrics"]["target_objective"] is not None],
                  key=lambda row: (row["metrics"]["target_objective"], row["metrics"]["jerk"]["p95"], json.dumps(row["parameters"], sort_keys=True)))
  for rank, row in enumerate(ranked, 1):
    row["rank"] = rank
  return next((row for row in ranked if not row["selection_rejections"]), None), common


def validation_search(windows, report, proofs, runtime, *, grid=None):
  """Finite native-parameter diagnostics only; never open, select or score test data."""
  if not windows or len({window["id"] for window in windows}) != len(windows):
    raise SearchError("nonempty unique validation windows required")
  baseline = native_parameters(windows[0]["cp"])
  if any(value_hash(w["cp"]) != value_hash(windows[0]["cp"]) or value_hash(w["sp"]) != value_hash(windows[0]["sp"]) for w in windows):
    raise SearchError("one same-cohort CP/SP contract per search required")
  grid = ([{"ki_v": [value]*len(baseline["ki_v"]), "stop_accel": stop} for value, stop in product(KI_GRID, STOP_GRID)]
          if grid is None else list(grid))
  if not grid or len(grid) > 64:
    raise SearchError("finite declared native grid must contain1..64 trials")
  rows = []
  for trial_index, parameters in enumerate([None, *grid]):
    row = {"parameters": baseline if parameters is None else parameters, "windows": {}, "failures": [], "selection_rejections": []}
    for window in windows:
      try:
        if trial_index and not isinstance(parameters, dict):
          raise SearchError("native grid candidate must explicitly declare KiV and stopAccel")
        result = evaluate_trial(window, report, proofs[window["id"]], runtime, parameters, stopping=window.get("regime") == "stopping")
        score = _window_score(result)
        if score is not None:
          row["windows"][window["id"]] = score
        else:
          row["failures"].append({"window_id": window["id"], "status": result["status"], "reason": result["reason"],
                                  "censor_t": result["censor_t"], "used_steps": result["used_steps"],
                                  "reference_rejection": "physical_time_reference_unavailable" if result["reference_mode"] != "capture_horizon" else None})
        row["selection_rejections"].extend(result["selection_rejections"])
      except Exception as exc:
        # Retain the complete failure; no failed candidate/window silently disappears.
        row["failures"].append({"window_id": window["id"], "status": "rejected", "error_type": type(exc).__name__,
                                "reason": str(exc), "traceback": traceback.format_exc()})
    row["individual_windows"] = len(row["windows"])
    row["individual_coverage"] = len(row["windows"])/len(windows)
    row["individually_supported"] = len(row["windows"]) >= research.MIN_WINDOWS and row["individual_coverage"] >= research.MIN_COVERAGE
    row["selection_rejections"] = sorted(set(row["selection_rejections"]))
    if not row["individually_supported"]:
      row["selection_rejections"].append("insufficient_individual_support")
    rows.append(row)
  winner, common = select_trials(rows, len(windows))
  before, candidates = rows[0], rows[1:]
  for row in candidates:
    p = row["parameters"]
    row["finite_grid_boundaries"] = {}
    if isinstance(p, dict) and isinstance(p.get("ki_v"), (list, tuple)) and "stop_accel" in p:
      row["finite_grid_boundaries"] = {"ki_v": ["minimum" if value == min(KI_GRID) else "maximum" if value == max(KI_GRID) else None
                                                  for value in p["ki_v"]],
                                       "stop_accel": ("minimum" if p["stop_accel"] == min(STOP_GRID)
                                                      else "maximum" if p["stop_accel"] == max(STOP_GRID) else None)}
  return {"status": "incomplete_native_controller_research", "baseline_parameters": baseline, "baseline": before, "rows": candidates,
          "attempted_windows": len(windows), "common_window_ids": common, "minimum_common_coverage": research.MIN_COVERAGE,
          "minimum_common_windows": research.MIN_WINDOWS, "complete_controller_candidate": None, "production_enabled": False,
          "native_longitudinal_candidate": winner["parameters"] if winner else None,
          "selection": winner, "constraints": research.LIMITS,
          "constraint_scope": "Conservative offline research comfort/direction/saturation gates;2.5m/s3 command slew is not a native LongControl limit.",
          "stopping_constraints": ("at least2 completed native stops inside causal model-target/TRAIN-uncertainty interval; "
                                   + "no distance/time regression beyond.1m/.1s"),
          "scope": "Finite native longitudinal selection only; frozen test evaluation, lateral and closed-loop acceptance remain separate."}
