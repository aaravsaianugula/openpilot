"""Conditional native joint diagnostics: frozen vision, one-way speed -> lateral.

Native LongControl is evaluated once per content-bound long parameter pair.
LatHandle consumes that trace's PRE-transition v/a and a 50ms joint state held
unchanged between macro endpoints. This is an UNVALIDATED 100Hz ZOH-feedback
hypothesis, not invented 100Hz physics or a production controller. Raw future
measured steering/yaw/speed never supplies candidate feedback or targets.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re
import traceback

import numpy as np

import controller_multirate_lateral as plant
import controller_native_lateral_runtime as lateral
import controller_native_search as long
import controller_turns as turns

DT, MACRO_DT, MICRO_NODES = .01, .05, 5
MIN_PLAN_SPEED = .3
CONTRACT = {"kind": "native_joint_diagnostic_v1", "native_dt_s": DT, "macro_dt_s": MACRO_DT,
            "micro_nodes": MICRO_NODES, "steer_max": 409, "minimum_plan_speed_mps": MIN_PLAN_SPEED,
            "reference": "capture-age raw orientationRate.z / hypot(velocity.x,velocity.y); positive RIGHT",
            "reference_assumption": "planar/no-sideslip and heading consistency UNVALIDATED",
            "feedback": "joint angle/yaw/unsigned vendor rate held until fifth completed native command",
            "learner": "explicit causal anchor snapshot only; future adaptive-estimator feedback absent",
            "inference_delay": "exact frame/capture witness -> packet lat_delay/control_params/action_t; distinct from later native core and extension lag",
            "scope": "moving-joint diagnostics; one-way simulated speed -> lateral, frozen perception/exogenous flags; standstill regime absent",
            "delivery": "CAN command/echo and request are observations, not proof of delivered actuator torque"}
_SHA = re.compile(r"[0-9a-f]{64}")
_PIN = re.compile(r"[0-9a-f]{40}")
_STATE = {"angle", "yaw", "vendor_rate"}
_INIT = {"previous_curvature", "last_transmitted_counts", "angle_counter", "steer_limited"}


class JointError(ValueError):
  pass


def _finite(value):
  return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)) and np.isfinite(value)


def static_identity(cp, sp):
  return plant.identity_hash({"cp": cp, "sp": sp})


def policy_identity(params):
  return plant.identity_hash(params)


def helper_hashes():
  return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in
          (("joint", __import__(__name__)), ("long", long), ("lateral", lateral), ("plant", plant), ("turns", turns))}


def bind_proof(window, report, long_proof, envelope, model, long_runtime, lat_runtime):
  """Content binding, not an independent assertion that a recorded source is clean."""
  return {"window_sha256": long.value_hash(window), "report_sha256": long.value_hash(report),
          "long_proof_sha256": long.value_hash(long_proof), "lateral_window_sha256": long.value_hash(envelope),
          "lateral_model_sha256": long.value_hash(model), "long_sources": dict(long_runtime.source_hashes),
          "lateral_sources": dict(lat_runtime.sources), "helpers": helper_hashes()}


class LongCache:
  """In-memory content cache; copies prevent a caller mutating a later trial."""
  def __init__(self):
    self._entries = {}

  def evaluate(self, window, report, proof, runtime, parameters, stopping):
    _, normalized = long.admit(window, report, proof, runtime, parameters)
    key = long.value_hash({"window": window, "report": report, "proof": proof, "native_sources": runtime.source_hashes,
                           "actual_native": runtime.actual_native, "parameters": normalized, "stopping": stopping})
    if key not in self._entries:
      try:
        result = long.evaluate_trial(window, report, proof, runtime, parameters, stopping=stopping)
      except Exception as error:
        result = {"status": "rejected", "reason": str(error), "error_type": type(error).__name__,
                  "traceback": traceback.format_exc(), "selection_rejections": ["long_trial_failed"], "production_enabled": False}
      self._entries[key] = deepcopy(result)
    result = deepcopy(self._entries[key])
    return result, {"key": key, "result_sha256": long.value_hash(result)}


def _parameters(cp, parameters, grid):
  try:
    if not isinstance(grid, dict) or set(grid) != {"factor", "friction"} or any(not isinstance(v, (list, tuple)) or not v for v in grid.values()):
      raise JointError("declared finite factor/friction grid required even for baseline")
    lateral._candidate_params({}, {key: values[0] for key, values in grid.items()}, grid)
  except (ValueError, TypeError, KeyError) as error:
    raise JointError("malformed lateral grid: " + str(error)) from error
  if parameters is None:
    return {"long": None, "lateral": None}
  if not isinstance(parameters, dict) or set(parameters) != {"long", "lateral"}:
    raise JointError("joint candidate requires exact long/lateral parameter declarations")
  try:
    long.native_parameters(cp, parameters["long"])
    lateral._candidate_params({}, parameters["lateral"], grid)
  except (ValueError, TypeError, KeyError) as error:
    raise JointError("malformed or out-of-grid joint candidate: " + str(error)) from error
  return deepcopy(parameters)


def _event(row, anchor, max_age=None):
  if (not isinstance(row, dict) or row.get("valid") is not True or not _finite(row.get("t"))
      or not 0 <= anchor-row["t"] or (max_age is not None and anchor-row["t"] > max_age+1e-9)):
    raise JointError("missing, invalid, future or stale causal anchor event")


def _state(state):
  if not isinstance(state, dict) or set(state) != _STATE or any(not _finite(v) for v in state.values()) or state["vendor_rate"] < 0:
    raise JointError("finite joint angle/yaw/unsigned vendor-rate state required")
  return {key: float(value) for key, value in state.items()}


def _model_population(model, report):
  if (model.get("split") != report["split"] or model.get("min_samples", 0) < 100 or model.get("min_routes", 0) < 2
      or not model.get("cells") or model.get("supported_cells", 0) < 1 or model.get("features") != list(plant.FEATURES)
      or not model.get("record_hashes") or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in model["record_hashes"])
      or model.get("model_sha256") != plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})):
    raise JointError("hash-bound supported TRAIN joint plant population required")
  bounds = [np.asarray(model.get(key), dtype=float) for key in ("feature_min", "feature_max")]
  edges = model.get("edges")
  if (any(a.shape != (14,) or not np.isfinite(a).all() for a in bounds) or np.any(bounds[0] > bounds[1])
      or not isinstance(edges, list) or len(edges) != 14
      or any(np.asarray(e).ndim != 1 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in edges)):
    raise JointError("malformed joint feature bounds/edges")
  supported = 0
  for cell in model["cells"].values():
    lower, upper, delta = (np.asarray(cell.get(key), dtype=float) for key in ("feature_min", "feature_max", "joint_delta_median"))
    if (lower.shape != (14,) or upper.shape != (14,) or delta.shape != (3,)
        or not all(np.isfinite(a).all() for a in (lower, upper, delta)) or np.any(lower > upper)):
      raise JointError("malformed joint empirical cell")
    if cell.get("supported") is True:
      routes = cell.get("routes", [])
      if (cell.get("samples", 0) < model["min_samples"] or len(set(routes)) < model["min_routes"]
          or any(route not in report["split"]["train"] for route in routes)):
        raise JointError("joint cell lacks independent TRAIN-route support")
      supported += 1
  if supported != model["supported_cells"]:
    raise JointError("joint supported-cell denominator mismatch")


def _delay_witness(window, env):
  """Source-bound caller contract, not independent proof a caller's metadata is genuine."""
  rdf = long.rdf_from_tape(window["tape"])
  n, witness = len(rdf["input_t"]), env.get("lateral_delay_witness")
  if not isinstance(witness, dict):
    raise JointError("independent inference-time lateral delay witness missing")
  arrays = {key: np.asarray(witness.get(key)) for key in
            ("frame_id", "capture_t", "delay", "source_publication_t", "source_valid", "source_kind", "source_log_sha256", "packet_lat_delay")}
  if (any(value.shape != (n,) for value in arrays.values()) or arrays["frame_id"].dtype.kind not in "iu"
      or arrays["source_valid"].dtype.kind != "b" or not arrays["source_valid"].all()
      or arrays["source_kind"].dtype.kind != "U" or arrays["source_log_sha256"].dtype.kind != "U"
      or not np.array_equal(arrays["frame_id"], rdf["frame_id"]) or not np.array_equal(arrays["capture_t"], rdf["narrow_t"])
      or not (arrays["source_log_sha256"] == env["source_log_sha256"]).all()
      or witness.get("packet_sha256") != env["source_evidence"]["packet_sha256"]):
    raise JointError("lateral delay witness frame/capture/source/packet binding differs")
  for key in ("capture_t", "delay", "source_publication_t", "packet_lat_delay"):
    if arrays[key].dtype.kind not in "fiu" or not np.isfinite(arrays[key]).all():
      raise JointError("non-finite inference-time lateral delay witness")
  recorded, fallback = arrays["source_kind"] == "lateralDelay", arrays["source_kind"] == "carParams"
  age = arrays["capture_t"]-arrays["source_publication_t"]
  if (not (recorded | fallback).all() or (age <= 0).any() or (arrays["delay"] < 0).any()
      or (age[recorded] > .5+1e-9).any()):
    raise JointError("lateral delay witness source invalid, future, or stale")
  cp_delay = window["cp"].get("steerActuatorDelay")
  if fallback.any() and (not _finite(cp_delay) or cp_delay < 0
      or not np.allclose(arrays["delay"][fallback], cp_delay, atol=1e-7, rtol=0)
      or not np.allclose(arrays["source_publication_t"][fallback], env["cp_t"], atol=1e-9, rtol=0)):
    raise JointError("CP lateral delay fallback lacks exact prior CP/value witness")
  controls = np.asarray(witness.get("packet_lateral_control_params"))
  smoothing = .1 if window["model_inputs"]["packet_includes_model_smoothing"] else 0.
  if (controls.shape != (n, 2) or controls.dtype.kind not in "fiu" or not np.isfinite(controls).all() or (controls[:, 0] < 0).any()
      or not np.allclose(controls[:, 1], arrays["delay"], atol=1e-7, rtol=0)
      or not np.allclose(arrays["packet_lat_delay"], arrays["delay"], atol=1e-7, rtol=0)
      or not np.allclose(rdf["packet_action_t"][:, 0], arrays["delay"]+.075+smoothing, atol=1e-7, rtol=0)):
    raise JointError("witness/packet lateral delay/control parameters/action_t mismatch")


def _admit(window, report, long_proof, env, model, lr, lat, proof, parameters, grid):
  if not isinstance(lat, lateral.Runtime) or not isinstance(lr, long.NativeRuntime):
    raise JointError("existing admitted native Runtime adapters required")
  if proof != bind_proof(window, report, long_proof, env, model, lr, lat):
    raise JointError("joint source/input/helper hash binding differs")
  parameters = _parameters(window["cp"], parameters, grid)
  long.admit(window, report, long_proof, lr, parameters["long"])
  pin = window.get("opendbc_gitlink")
  if (not isinstance(pin, str) or not _PIN.fullmatch(pin) or pin != lat.pin
      or any(env.get(k) != window.get(k) for k in ("route", "cohort", "source_log_sha256", "opendbc_gitlink"))):
    raise JointError("same known source/cohort/route/pin required for joint native feedback")
  if not lat.sources or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in lat.sources.values()):
    raise JointError("lateral native source hashes absent")
  if lat.bindings.actual_native and (lat.pin != lateral.OPENDBC_PIN or lat.sources.get(lateral.V0_PATH) != lateral.V0_SHA
      or any(Path(inspect.getfile(fn)).resolve() != Path(lateral.__file__).resolve()
             for fn in (lat.bindings.decode_static, lat.bindings.controller_factory, lat.bindings.build_model))):
    raise JointError("actual lateral adapter factory/source proof differs")
  if (model.get("kind") != plant.KIND or model.get("source_eligible") is not True
      or model.get("static_policy_timing_verified") is not True or model.get("source_pin_verified") is not True
      or model.get("macro_dt_s") != MACRO_DT or model.get("native_dt_s") != DT or model.get("micro_nodes") != MICRO_NODES
      or any(model.get(key) != env.get(key) for key in ("cohort", "opendbc_gitlink", "static_identity_sha256", "policy_identity_sha256"))):
    raise JointError("untrusted or incompatible joint plant source/clock/identity")
  _model_population(model, report)
  evidence = env.get("source_evidence", {})
  if (evidence.get("verified") is not True or evidence.get("source_log_sha256") != window["source_log_sha256"]
      or any(not isinstance(evidence.get(k), str) or not _SHA.fullmatch(evidence[k])
             for k in ("schema_sha256", "extractor_sha256", "packet_sha256"))):
    raise JointError("verified log/schema/extractor/packet evidence required")
  t = np.asarray(window["tape"]["t"])
  if (t.ndim != 1 or len(t) < MICRO_NODES or len(t) % MICRO_NODES or not np.isfinite(t).all()
      or not np.allclose(np.diff(t), DT, rtol=0, atol=1e-9)):
    raise JointError("continuous100Hz joint clock and complete five-tick macros required")
  horizon = np.asarray(long.rdf_from_tape(window["tape"]).get("horizon"))
  if horizon.shape != (33,) or not np.allclose(horizon, lat.bindings.horizon, rtol=0, atol=1e-7):
    raise JointError("raw RDF horizon differs from actual native model contract")
  anchor = float(t[0])
  if (any(not _finite(env.get(key)) or env[key] > anchor for key in ("static_t", "cp_t", "sp_t"))
      or abs(env["static_t"]-max(env["cp_t"], env["sp_t"])) > 1e-9):
    raise JointError("full CP/SP require their distinct prior publication times")
  _delay_witness(window, env)
  raw = []
  for key in ("cp_raw", "sp_raw"):
    value = np.asarray(env.get(key))
    if value.dtype != np.uint8 or value.ndim != 1 or not len(value):
      raise JointError("full CP/SP nonempty serialized uint8 payloads required")
    raw.append(value.tobytes())
  cp, sp = lat.bindings.decode_static(*raw)
  if long.value_hash(cp.to_dict()) != long.value_hash(window["cp"]) or long.value_hash(sp.to_dict()) != long.value_hash(window["sp"]):
    raise JointError("full CP/SP semantic equality to longitudinal window required")
  if env.get("static_identity_sha256") != static_identity(window["cp"], window["sp"]):
    raise JointError("static identity does not bind full semantic CP/SP")
  for key, age in (("lp", .2), ("params", None), ("initial", .052), ("roll", .2)):
    _event(env.get(key), anchor, age)
  lp, params = env["lp"]["values"], env["params"]["values"]
  if (set(lp) != set(lateral.LP_KEYS) or any(not _finite(v) for v in lp.values()) or lp["stiffnessFactor"] <= 0 or lp["steerRatio"] <= 0
      or set(params) != set(lateral.TYPES) or env.get("policy_identity_sha256") != policy_identity(params)):
    raise JointError("complete fixed causal LP/typed Params identity required")
  provider = lateral.ImmutableParams(params)
  for key in lateral.TYPES:
    provider.get(key)
  state = _state(env["initial"]["state"])
  if not _finite(env["roll"].get("value")):
    raise JointError("finite physical LLK roll anchor required independently of LP.roll")
  quality = env.get("location_quality", {})
  _event(quality, anchor, .075)
  if (any(not _finite(quality.get(k)) or quality[k] < 0 for k in ("yaw_std", "roll_std"))
      or any(quality.get(k) != 1 for k in ("yaw_inner_valid", "roll_inner_valid", "inputs_ok", "sensors_ok", "device_stable"))
      or quality.get("status") != 2 or abs(quality["t"]-env["roll"]["t"]) > 1e-9
      or not _finite(env.get("car_state_t")) or not 0 <= anchor-env["car_state_t"] <= .02+1e-9):
    raise JointError("recorded finite LLK uncertainty/status and causal initial CS clock required")
  learner = env.get("learner")
  if (not isinstance(learner, dict) or any(type(learner.get(k)) is not bool for k in ("valid", "admitted", "useParams", "frequency_valid"))
      or (learner.get("t") is not None and (not _finite(learner["t"]) or learner["t"] > anchor))
      or (not learner["admitted"] and not learner.get("reason"))):
    raise JointError("explicit causal learner snapshot or source-bound unavailability required")
  if learner["admitted"] and (not learner["valid"] or not learner["frequency_valid"] or learner["t"] is None
      or anchor-learner["t"] > 2.5 or (learner["useParams"] and
         (any(not _finite(learner.get(k)) for k in ("factor", "offset", "friction")) or learner["factor"] <= 0 or learner["friction"] < 0))):
    raise JointError("native-admitted learner source/values missing or stale")
  initial, history = env.get("initialization", {}), env.get("history", {})
  if (set(initial) != _INIT or not _finite(initial["previous_curvature"])
      or type(initial["last_transmitted_counts"]) is not int or abs(initial["last_transmitted_counts"]) > 409
      or type(initial["angle_counter"]) is not int or initial["angle_counter"] < 0 or type(initial["steer_limited"]) is not bool):
    raise JointError("explicit native limiter initialization required")
  h = {key: np.asarray(history.get(key)) for key in ("t", "command", "request", "valid")}
  if (any(value.shape != (2,) for value in h.values()) or not np.isfinite(h["t"]).all()
      or not np.allclose(h["t"], anchor+np.array([-.02, -.01]), rtol=0, atol=1e-9)
      or h["command"].dtype.kind not in "iu" or np.any(np.abs(h["command"]) > 409)
      or any(h[key].dtype.kind != "b" or not h[key].all() for key in ("request", "valid"))
      or int(h["command"][-1]) != initial["last_transmitted_counts"]):
    raise JointError("valid strictly past20ms request-true native CAN prefix required")
  for key in ("lat_active", "driver_torque", "fault_temporary", "fault_permanent", "published_delay", "lat_input_t", "lat_input_valid"):
    if np.asarray(window["tape"].get(key)).shape != t.shape:
      raise JointError("missing joint causal input: " + key)
  censor = env.get("censor_t")
  if censor is not None and (not _finite(censor) or censor <= t[-1]):
    raise JointError("native window must end before its explicit external censor")
  return parameters, raw, state


def fixed_reference(window):
  """Frozen planar model geometry; no candidate speed, future measured GT or extrapolation."""
  rdf, t = long.rdf_from_tape(window["tape"]), np.asarray(window["tape"]["t"])
  plan, horizon = np.asarray(rdf.get("plan")), np.asarray(rdf.get("horizon"))
  n = len(rdf["input_t"])
  future = np.asarray(rdf.get("desired_curvature"))
  if (plan.shape != (n, 33, 15) or horizon.shape != (33,) or future.shape != (n,)
      or not np.isfinite(plan).all() or not np.isfinite(horizon).all() or not np.isfinite(future).all()
      or np.any(np.diff(horizon) <= 0) or window.get("reference_mode") != "capture_horizon"):
    raise JointError("full finite raw RDF33x15 plan/nonuniform horizon/future action required")
  indices = np.searchsorted(rdf["output_ready_t"], t, side="right")-1
  curvature, acceleration = np.full(len(t), np.nan), np.full(len(t), np.nan)
  valid = np.zeros(len(t), bool)
  for j, (timestamp, index) in enumerate(zip(t, indices, strict=True)):
    if (index < 0 or not rdf["scored_mask"][index] or not rdf["controller_admissible"][index]
        or not rdf["deadline_admissible"][index] or not 0 <= timestamp-rdf["narrow_t"][index] <= .2+1e-9):
      continue  # latest unavailable output is never replaced by an older one.
    age = timestamp-rdf["narrow_t"][index]
    if not horizon[0] <= age <= horizon[-1]:
      continue
    vx, vy, yaw = (float(np.interp(age, horizon, plan[index, :, column])) for column in (3, 4, 14))
    speed = np.hypot(vx, vy)
    if speed <= MIN_PLAN_SPEED:
      continue
    curvature[j] = yaw/speed
    acceleration[j] = long.physical_reference(rdf, index, timestamp, window["reference_mode"])
    valid[j] = True
  speed = np.full(len(t), np.nan)
  path = np.full((len(t), 2), np.nan)
  heading = 0.
  speed[0], path[0] = float(window["initial_v"]), 0.
  for j in range(1, len(t)):
    if not valid[j-1] or not np.isfinite(speed[j-1]):
      break
    next_speed = speed[j-1]+acceleration[j-1]*DT
    if next_speed < 0:
      break  # A model-reference standstill law is not invented.
    next_heading = heading+curvature[j-1]*speed[j-1]*DT
    path[j] = path[j-1]+.5*DT*(speed[j-1]*np.array([np.cos(heading), np.sin(heading)])
                              + next_speed*np.array([np.cos(next_heading), np.sin(next_heading)]))
    speed[j], heading = next_speed, next_heading
  return {"curvature": curvature, "acceleration": acceleration, "speed": speed, "path": path,
          "valid": valid & np.isfinite(speed) & np.isfinite(path).all(axis=1), "rdf_index": indices,
          "sha256": long.value_hash({"curvature": curvature, "acceleration": acceleration, "speed": speed, "path": path})}


def evaluate_joint(window, report, long_proof, envelope, model, long_runtime, lat_runtime, proof, *,
                   parameters=None, lateral_grid, cache=None):
  parameters, raw, state = _admit(window, report, long_proof, envelope, model, long_runtime, lat_runtime, proof, parameters, lateral_grid)
  reference = fixed_reference(window)
  if not reference["valid"][0]:
    raise JointError("physical curvature/reference unavailable at causal anchor")
  if cache is None:
    cache = LongCache()
  if not isinstance(cache, LongCache):
    raise JointError("content-bound LongCache required")
  result, receipt = cache.evaluate(window, report, long_proof, long_runtime, parameters["long"], window.get("regime") == "stopping")
  if result["status"] == "rejected":
    return {"window_id": window["id"], "status": "rejected", "reason": "long_trial_failed", "long_result": result,
            "long_cache": receipt, "selection_rejections": result["selection_rejections"], "complete_controller_candidate": None,
            "production_enabled": False, "parameters": parameters}
  tape, rdf, t = window["tape"], long.rdf_from_tape(window["tape"]), np.asarray(window["tape"]["t"])
  handle = lat_runtime.create(*raw, envelope["lp"]["values"], envelope["params"]["values"],
                              candidate=parameters["lateral"], grid=lateral_grid, initialization=envelope["initialization"])
  trace = {key: np.full(len(t), np.nan) for key in ("angle", "yaw", "vendor_rate", "can_torque", "native_torque", "future_curvature")}
  trace.update(t=t.copy(), v=result["trace"]["v"].copy(), a=result["trace"]["a"].copy(),
               physical_curvature=reference["curvature"].copy(), fixed_speed=reference["speed"].copy(), fixed_path=reference["path"].copy())
  trace["path"] = np.full((len(t), 2), np.nan)
  trace["future_action_t"] = np.full((len(t), 2), np.nan)
  trace["steer_request"] = np.zeros(len(t), bool)
  completed, supported, reason, censor = 0, 0, None, None
  heading, position = 0., np.zeros(2)
  commands, speeds = [], []
  prior = envelope["initialization"]["last_transmitted_counts"]/409.
  for j, timestamp in enumerate(t):
    if j >= result["used_steps"]:
      reason, censor = result["reason"] or "long_horizon_incomplete", result["censor_t"]
      break
    if envelope["censor_t"] is not None and timestamp >= envelope["censor_t"]:
      reason, censor = "frozen_censor", float(timestamp)
      break
    if not reference["valid"][j]:
      reason, censor = "latest_physical_model_unavailable", float(timestamp)
      break
    flags = ("lat_active", "fault_temporary", "fault_permanent", "lat_input_valid")
    if (any(tape[k][j] not in (0, 1) for k in flags) or not tape["lat_input_valid"][j]
        or not _finite(tape["lat_input_t"][j]) or not 0 <= timestamp-tape["lat_input_t"][j] <= .2+1e-9
        or not _finite(tape["driver_torque"][j]) or not _finite(tape["published_delay"][j]) or tape["published_delay"][j] < 0):
      reason, censor = "invalid_causal_lateral_input", float(timestamp)
      break
    if not tape["lat_active"][j] or tape["fault_temporary"][j] or tape["fault_permanent"][j]:
      reason, censor = "lateral_inactive_or_fault", float(timestamp)
      break
    if tape["driver_torque"][j] != 0:
      reason, censor = "driver_torque_intervention", float(timestamp)
      break
    index = int(reference["rdf_index"][j])
    if result["trace"]["rdf_index"][j] != index:
      raise JointError("cached long and lateral latest-ready model identities differ")
    speed, accel = trace["v"][j], trace["a"][j]
    output = handle.step(t=float(timestamp), active=True, v=float(speed), a=float(accel), steering_angle=state["angle"],
                         steering_rate=state["vendor_rate"], driver_torque=float(tape["driver_torque"][j]), steering_pressed=False,
                         standstill=bool(speed <= .3), fault_temporary=False, fault_permanent=False, prior_delivered_torque=prior,
                         model=lat_runtime.model_from_rdf(rdf["plan"][index], rdf["horizon"]),
                         desired_curvature=float(rdf["desired_curvature"][index]), published_delay=float(tape["published_delay"][j]),
                         learner=envelope["learner"] if j == 0 else None)
    if not output["active"] or not output["steer_request"]:
      reason, censor = "active_steer_request_false", float(timestamp)
      break
    command = output["can_torque"]
    if not _finite(command) or command != int(command) or abs(command) > 409 or not _finite(output["torque"]):
      raise JointError("native CAN output outside fixed integer409 contract")
    for key in _STATE:
      trace[key][j] = state[key]
    trace["can_torque"][j], trace["native_torque"][j] = command, output["torque"]
    trace["steer_request"][j], trace["future_curvature"][j], trace["path"][j] = True, rdf["desired_curvature"][index], position
    trace["future_action_t"][j] = rdf["action_t"][index]
    position = position+speed*DT*np.array([np.cos(heading), np.sin(heading)])
    heading += state["yaw"]*DT
    prior = command/409.
    commands.append(int(command))
    speeds.append(float(speed))
    completed += 1
    if len(commands) == MICRO_NODES:
      next_state = plant.predict_macro(model, state, speeds, commands, envelope["roll"]["value"])
      if isinstance(next_state, plant.Unsupported):
        reason = "macro_unsupported:"+next_state.reason
        break
      state = _state(next_state)
      commands, speeds = [], []
      supported = completed
  nodes = np.arange(0, supported, MICRO_NODES)
  metrics = turns.window_metrics(reference["curvature"][nodes], (trace["v"]*trace["yaw"])[nodes],
                                (reference["curvature"]*trace["v"]**2)[nodes], trace["can_torque"][nodes],
                                dt_s=MACRO_DT, measured_yaw=trace["yaw"][nodes]) if len(nodes) else None
  rejections = list(result["selection_rejections"])
  rejections.extend(["unvalidated_100hz_zoh_feedback", "unvalidated_planar_heading_consistency", "native_lateral_initialization_uncertain",
                     "frozen_learner_adaptation_unverified", "coupled_standstill_regime_unavailable"])
  if not lat_runtime.bindings.actual_native:
    rejections.append("code_fixture_not_native_lateral_evidence")
  if model.get("selection_eligible") is not True:
    rejections.append("lateral_plant_not_selection_eligible")
  if completed != len(t) or reason:
    rejections.append("incomplete_joint_supported_horizon")
  return {"window_id": window["id"], "parameters": parameters, "trace": trace, "long_result": result, "long_cache": receipt,
          "status": "complete" if completed == len(t) and reason is None else "censored" if censor is not None else "unsupported",
          "reason": reason, "censor_t": censor, "used_steps": completed, "supported_steps": supported,
          "attempted_steps": len(t), "macro_indices": nodes.tolist(),
          "turns": metrics, "fixed_reference_sha256": reference["sha256"], "source_binding": deepcopy(proof),
          "selection_rejections": sorted(set(rejections)), "selection_eligible": False, "complete_controller_candidate": None,
          "contract": dict(CONTRACT), "production_enabled": False}


def summarize(results, common):
  """Fixed declared population; a failed/missing window never improves coverage."""
  missing = [key for key in common if key not in results or results[key].get("status") != "complete"]
  turn_summary = turns.summarize({key: results[key].get("turns") for key in common if key not in missing}, common)
  long_rows = {key: long._window_score(results[key]["long_result"]) for key in common if key not in missing and "long_result" in results[key]}
  metrics = long.summarize(long_rows, common) if not missing and all(long_rows.get(k) is not None for k in common) and common else None
  speed, path = [], []
  for key in common:
    if key not in missing:
      trace = results[key]["trace"]
      speed.extend(np.abs(trace["v"]-trace["fixed_speed"]).tolist())
      path.extend(np.linalg.norm(trace["path"]-trace["fixed_path"], axis=1)[results[key]["macro_indices"]].tolist())
  return {"missing_windows": missing, "common_window_ids": sorted(common), "turns": turn_summary, "long": metrics,
          "fixed_reference_by_window": {key: results[key].get("fixed_reference_sha256") for key in common if key in results},
          "speed_p95": float(np.percentile(speed, 95)) if speed else None, "path_p95": float(np.percentile(path, 95)) if path else None}


def constraints(metrics, baseline):
  reasons = turns.constraints(metrics["turns"], baseline["turns"], require_improvement=False)
  if metrics.get("missing_windows") or baseline.get("missing_windows"):
    reasons.append("missing_common_windows")
  if metrics.get("fixed_reference_by_window") != baseline.get("fixed_reference_by_window"):
    reasons.append("fixed_physical_reference_mismatch")
  if metrics.get("long") is None or baseline.get("long") is None:
    reasons.append("missing_independent_acceleration_braking_metrics")
  else:
    reasons.extend(long.research._constraints(metrics["long"], baseline["long"], "long", long.research.MIN_DIRECTION_POINTS,
                                             require_improvement=False))
  for key, slack in (("speed_p95", .05), ("path_p95", .1)):
    a, b = metrics.get(key), baseline.get(key)
    if a is None or b is None or not np.isfinite([a, b]).all() or a > b+slack:
      reasons.append(key+"_regression_or_missing")
  return sorted(set(reasons))


def _objective(metrics, baseline):
  ratios = []
  for node in ("turns", "long"):
    a, b = metrics.get(node), baseline.get(node)
    if a is None or b is None or a.get("target_objective") is None or b.get("target_objective") is None:
      return None
    current, previous = a["target_objective"], b["target_objective"]
    if not np.isfinite([current, previous]).all() or previous < 0 or current < 0:
      return None
    if previous == 0:
      if current:
        return None
      ratios.append(1.)
    else:
      ratios.append(current/previous)
  return float(np.mean(ratios))


def _stop_constraints(results, baseline, common):
  stops = [key for key in common if baseline.get(key, {}).get("long_result", {}).get("stopping_result") is not None]
  reasons = [] if len(stops) >= 2 else ["insufficient_completed_native_stopping_windows"]
  for key in stops:
    old = baseline[key]["long_result"]["stopping_result"]
    new = results.get(key, {}).get("long_result", {}).get("stopping_result")
    if new is None:
      reasons.append("native_stop_not_completed")
    elif (not isinstance(new.get("model_target"), dict) or not isinstance(old.get("model_target"), dict)
          or new.get("target_feasible") is not True
          or not new["model_target"]["lower_m"] <= new["distance_from_request_m"] <= new["model_target"]["upper_m"]):
      reasons.append("model_stopping_target_feasibility_unverified")
    elif any(old["model_target"].get(field) != new["model_target"].get(field) for field in ("request_t", "capture_t", "path_sha256")):
      reasons.append("stopping_reference_anchor_mismatch")
    elif new["distance_from_request_m"] > old["distance_from_request_m"]+.1 or new["time_from_request_s"] > old["time_from_request_s"]+.1:
      reasons.append("native_stopping_regression")
  return reasons


def validation_search(windows, report, long_proofs, envelopes, model, lr, lat, proofs, *, grid, lateral_grid):
  """Finite joint diagnostics; baseline retained, no production or complete winner."""
  if not windows or len({w["id"] for w in windows}) != len(windows) or not isinstance(grid, (list, tuple)) or not 0 < len(grid) <= 64:
    raise JointError("unique nonempty validation windows and finite1..64 joint grid required")
  if any(static_identity(w["cp"], w["sp"]) != static_identity(windows[0]["cp"], windows[0]["sp"]) for w in windows):
    raise JointError("one full CP/SP contract required for joint search")
  normalized = [_parameters(windows[0]["cp"], p, lateral_grid) for p in grid]
  signatures = [long.value_hash(p) for p in normalized]
  if (len(signatures) != len(set(signatures)) or any(p == {"long": None, "lateral": None} for p in normalized)):
    raise JointError("grid must contain distinct explicit candidates, with baseline preserved separately")
  common, cache, rows = sorted(w["id"] for w in windows), LongCache(), []
  for parameters in [None, *normalized]:
    row = {"parameters": parameters, "results": {}, "failures": [], "selection_rejections": []}
    for window in windows:
      name = window["id"]
      try:
        result = evaluate_joint(window, report, long_proofs[name], envelopes[name], model, lr, lat, proofs[name],
                                parameters=parameters, lateral_grid=lateral_grid, cache=cache)
        row["results"][name] = result
        row["selection_rejections"].extend(result["selection_rejections"])
        if result["status"] != "complete":
          row["failures"].append({"window_id": name, "status": result["status"], "reason": result["reason"]})
      except Exception as error:
        failure = {"window_id": name, "status": "rejected", "reason": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()}
        row["results"][name] = failure
        row["failures"].append(failure)
    row["metrics"] = summarize(row["results"], common)
    before = rows[0] if rows else row
    row["selection_rejections"].extend(constraints(row["metrics"], before["metrics"]))
    row["selection_rejections"].extend(_stop_constraints(row["results"], before["results"], common))
    row["joint_target_objective"] = _objective(row["metrics"], before["metrics"])
    if rows and (row["joint_target_objective"] is None or row["joint_target_objective"] >= .99):
      row["selection_rejections"].append("no_meaningful_joint_target_improvement")
    row["coverage"] = (len(common)-len(row["metrics"]["missing_windows"]))/len(common)
    if len(common) < long.research.MIN_WINDOWS:
      row["selection_rejections"].append("insufficient_fixed_common_population")
    row["selection_rejections"] = sorted(set(row["selection_rejections"]))
    row["common_window_ids"] = common
    rows.append(row)
  ranked = sorted([row for row in rows[1:] if row["joint_target_objective"] is not None],
                  key=lambda row: (row["joint_target_objective"], json.dumps(row["parameters"], sort_keys=True)))
  for rank, row in enumerate(ranked, 1):
    row["diagnostic_rank"] = rank
  return {"status": "insufficient_evidence", "baseline": rows[0], "rows": rows[1:], "common_window_ids": common,
          "attempted_windows": len(common), "complete_controller_candidate": None, "selection": None,
          "minimum_common_windows": long.research.MIN_WINDOWS, "minimum_common_coverage": long.research.MIN_COVERAGE,
          "production_enabled": False, "contract": dict(CONTRACT)}
