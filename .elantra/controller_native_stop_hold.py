"""Conditional approach -> stop -> hold native diagnostics, with frozen vision.

One continuous native lateral Handle consumes the cached native LongControl
PRE-transition speed/acceleration trace. Inactive zero CAN is not rate-limited
as active steering. Both empirical models advance only after five completed
ticks; no rack/yaw freeze, invented standstill curvature, or future measured
feedback. Physical contact, initialization and 100Hz ZOH remain unvalidated.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

import controller_multirate_stop_hold as hold
import controller_native_joint as joint

long = joint.long
DT, MACRO_DT = joint.DT, joint.MACRO_DT
CONTRACT = {"kind": "conditional_native_stop_hold_v1", "dt_s": DT, "macro_dt_s": MACRO_DT,
            "moving_approach_prefix_s": .5, "moving_macro_min_speed_mps": 2., "zero_speed_hold_min_s": .5,
            "geometry": "candidate-independent finite prefix only; unknown geometry never re-enters turn support",
            "raw_car_state": "hash-bound complete publication stream; latest asof matches tape clock, 15ms freshness/gaps; eligibility only",
            "simulated_standstill": "simulated v == 0 is an unvalidated conditional counterfactual, not recorded hardware feedback",
            "hold": "same continuous native Handle; exact effective-active/request pattern; unshifted CAN proxy",
            "contact": "conditional existing Long plant nonreversing speed, not validated vehicle contact/hold",
            "objective_weights": {"balanced_turn_tracking": .25, "balanced_acceleration_braking_tracking": .25,
                                  "absolute_model_stop_distance_error": .5},
            "scope": "one-way speed->lateral; frozen perception, learner and OEM flags; no release/restart feedback"}


class StopHoldError(joint.JointError):
  pass


def policy_source_hash():
  return hashlib.sha256(Path(joint.lateral.__file__).read_bytes()).hexdigest()


def bind_proof(joint_proof, hold_model, assumptions):
  return {"joint_proof_sha256": long.value_hash(joint_proof), "hold_model_sha256": long.value_hash(hold_model),
          "contact_assumptions_sha256": long.value_hash(assumptions),
          "helpers": {"stop_hold": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "hold_plant": hashlib.sha256(Path(hold.__file__).read_bytes()).hexdigest()}}


def _hold_admit(model, moving, window, report):
  if (model.get("kind") != hold.KIND or model.get("features") != list(hold.FEATURES) or model.get("split") != report["split"]
      or any(model.get(k) != moving.get(k) for k in ("cohort", "opendbc_gitlink", "static_identity_sha256", "policy_identity_sha256"))
      or any(model.get(k) is not True for k in ("source_eligible", "source_pin_verified", "static_policy_timing_verified"))
      or model.get("native_policy_source_sha256") != policy_source_hash()
      or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
      or model.get("model_sha256") != hold.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
      or model.get("min_samples", 0) < 100 or model.get("min_routes", 0) < 2 or not model.get("cells")
      or not model.get("record_hashes") or any(not isinstance(v, str) or not joint._SHA.fullmatch(v) for v in model["record_hashes"])
      or window["cp"].get("minSteerSpeed") != 0 or window["cp"].get("steerAtStandstill") is not False):
    raise StopHoldError("unsupported hold plant/source/static/policy/TRAIN contract")
  bounds = [np.asarray(model.get(k)) for k in ("feature_min", "feature_max")]
  if (any(v.shape != (24,) or not np.isfinite(v).all() for v in bounds) or np.any(bounds[0] > bounds[1])
      or not isinstance(model.get("edges"), list) or len(model["edges"]) != 24
      or any(np.asarray(v).ndim != 1 or not np.isfinite(v).all() or np.any(np.diff(v) <= 0) for v in model["edges"])):
    raise StopHoldError("malformed hold empirical bounds")
  supported = 0
  for key, cell in model["cells"].items():
    pattern, sep, indices = key.partition("|")
    lo, hi, delta = (np.asarray(cell.get(k)) for k in ("feature_min", "feature_max", "joint_delta_median"))
    if (len(pattern) != 10 or any(v not in "01" for v in pattern) or pattern not in model.get("patterns", [])
        or not sep or len(indices.split(":")) != 24 or lo.shape != (24,) or hi.shape != (24,) or delta.shape != (3,)
        or any(not np.isfinite(v).all() for v in (lo, hi, delta)) or np.any(lo > hi)):
      raise StopHoldError("malformed hold pattern/cell")
    if cell.get("supported") is True:
      routes = cell.get("routes", [])
      if cell.get("samples", 0) < model["min_samples"] or len(set(routes)) < model["min_routes"] or any(r not in report["split"]["train"] for r in routes):
        raise StopHoldError("hold cell lacks independent TRAIN support")
      supported += 1
  if not supported or supported != model.get("supported_cells"):
    raise StopHoldError("hold supported population missing or inconsistent")


def _references(window):
  fixed = joint.fixed_reference(window)
  rdf, t = long.rdf_from_tape(window["tape"]), np.asarray(window["tape"]["t"])
  position = np.asarray(rdf.get("position"))
  if position.shape != (len(rdf["input_t"]), 33, 3) or not np.array_equal(position, rdf["plan"][:, :, :3]):
    raise StopHoldError("stop target position must equal the full raw RDF plan")
  available = np.zeros(len(t), bool)
  requests = np.zeros(len(t), bool)
  for j, (timestamp, index) in enumerate(zip(t, fixed["rdf_index"], strict=True)):
    if (index >= 0 and rdf["scored_mask"][index] and rdf["controller_admissible"][index] and rdf["deadline_admissible"][index]
        and 0 <= timestamp-rdf["narrow_t"][index] <= .2+1e-9
        and rdf["horizon"][0] <= timestamp-rdf["narrow_t"][index] <= rdf["horizon"][-1]):
      available[j], requests[j] = True, rdf["should_stop"][index]
  changes = np.flatnonzero(~requests[:-1] & requests[1:])+1
  if not available.all() or requests[0] or len(changes) != 1 or not requests[changes[0]:].all():
    raise StopHoldError("one fully available causal False->True request required; release/restart unsupported")
  origin = int(changes[0])
  if t[origin]-t[0] < CONTRACT["moving_approach_prefix_s"]-1e-9 or rdf["narrow_t"][fixed["rdf_index"][origin]] < t[0]:
    raise StopHoldError("genuine moving prefix and covered request capture required")
  if np.asarray(window["tape"]["cruise_standstill"][:origin]).any():
    raise StopHoldError("OEM standstill-driven early stop is unsupported")
  geometry = np.logical_and.accumulate(fixed["valid"])
  if not geometry[0]:
    raise StopHoldError("initial moving geometry unavailable")
  for key in ("curvature", "speed", "path"):
    fixed[key] = fixed[key].copy()
    fixed[key][~geometry] = np.nan
  fixed.update(available=available, requests=requests, geometry=geometry, request_index=origin)
  fixed["sha256"] = long.value_hash({key: fixed[key] for key in ("curvature", "speed", "path", "available", "requests", "geometry")})
  return fixed


def _raw_car_state_eligibility(window, env):
  """Source-bound caller witness, not independent proof its raw data is genuine.

  Complete raw CS publications retain invalid interpositions. Only the initial
  asof predecessor through the last consumed publication is quality-checked;
  future rows do not supply eligibility or candidate feedback. t_ns is the
  original uint64 Event.logMonoTime; booleans are Event.valid, CS.canValid and
  CS.standstill, and v_ego is recorded CS.vEgo, never the simulated speed.
  Joint proof binds this envelope plus the window and source log identities.
  """
  witness = env.get("raw_car_state_witness")
  if (not isinstance(witness, dict) or witness.get("kind") != "raw_car_state_eligibility_v1"
      or witness.get("route") != window.get("route") or witness.get("source_log_sha256") != window.get("source_log_sha256")
      or witness.get("sha256") != long.value_hash({k: v for k, v in witness.items() if k != "sha256"})):
    raise StopHoldError("raw carState witness missing or source/content binding differs")
  fields = {key: np.asarray(witness.get(key)) for key in ("t_ns", "event_valid", "can_valid", "standstill", "v_ego")}
  ns = fields["t_ns"]
  t, recorded_t = (np.asarray(window["tape"].get(key)) for key in ("t", "carState_t"))
  if (ns.ndim != 1 or not len(ns) or ns.dtype != np.dtype(np.uint64) or any(v.shape != ns.shape for v in fields.values())
      or any(fields[k].dtype.kind != "b" for k in ("event_valid", "can_valid", "standstill")) or fields["v_ego"].dtype.kind not in "fiu"
      or t.ndim != 1 or not len(t) or t.dtype.kind not in "fiu" or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
      or recorded_t.shape != t.shape or recorded_t.dtype.kind not in "fiu" or not np.isfinite(recorded_t).all()
      or np.any(ns[1:] <= ns[:-1])):
    raise StopHoldError("raw carState witness malformed flags, speed, or original uint64 clocks")
  # Match frozen raw-log producers exactly; division can differ by one ULP.
  publication_t = np.asarray([int(value)*1e-9 for value in ns], dtype=np.float64)
  if np.any(np.diff(publication_t) <= 0):
    raise StopHoldError("raw carState witness publication clocks are ambiguous")
  indices = np.searchsorted(publication_t, t, side="right")-1
  if np.any(indices < 0):
    raise StopHoldError("raw carState witness lacks causal predecessor")
  age = t-publication_t[indices]
  if not np.array_equal(publication_t[indices], recorded_t) or np.any((age < 0) | (age > .015+1e-9)):
    raise StopHoldError("raw carState witness latest publication differs from tape clock or is stale/future")
  span = slice(int(indices[0]), int(indices[-1])+1)
  speed = fields["v_ego"][span]
  if (not fields["event_valid"][span].all() or not fields["can_valid"][span].all()
      or not np.isfinite(speed).all() or np.any(speed < 0) or np.any(np.diff(ns[span]) > 15_000_000)
      or np.any(fields["standstill"][span] & (speed > .3))):
    raise StopHoldError("raw carState witness contains invalid/interposed/gapped state or hardware standstill above .3m/s")


def _admit(window, report, lp, env, moving, model, lr, lat, bound, assumptions, proof, parameters, grid):
  if proof != bind_proof(bound, model, assumptions):
    raise StopHoldError("stop/hold source/helper/contact hash binding differs")
  _raw_car_state_eligibility(window, env)
  parameters, raw, state = joint._admit(window, report, lp, env, moving, lr, lat, bound, parameters, grid)
  _hold_admit(model, moving, window, report)
  long_model = report["axes"]["long"]["model"]
  if window.get("regime") != "stopping" or not long.stopping_support(long_model):
    raise StopHoldError("identified longitudinal low-speed and standstill support required")
  if long.verified_stop_uncertainty(window, report["split"]["train"]) is None:
    raise StopHoldError("frozen20-outcome/two-TRAIN-route stop margin unavailable")
  expected = {"mode": "conditional_existing_long_nonreversing_contact", "validated": False,
              "source_log_sha256": window["source_log_sha256"], "long_model_sha256": long.value_hash(long_model),
              "hold_model_sha256": long.value_hash(model), "cp_sha256": long.value_hash(window["cp"]), "sp_sha256": long.value_hash(window["sp"])}
  if assumptions != expected:
    raise StopHoldError("explicit source-bound conditional contact/hold assumptions required; no validation inferred")
  return parameters, raw, state, _references(window)


def _hold_motion(trace, used, long_result):
  zero = np.flatnonzero(trace["v"][:used] == 0)
  if not len(zero):
    return {"supported_zero_speed_s": 0., "reason": "zero_speed_not_reached"}
  first = int(zero[0])
  if np.any(trace["v"][first:used] != 0) or np.any(long_result["trace"]["native_state"][first:used] != 2):
    return {"supported_zero_speed_s": 0., "reason": "release_or_restart_unsupported"}
  duration = float(trace["t"][used-1]-trace["t"][first])
  return {"supported_zero_speed_s": duration, "start_index": first, "end_index": used-1,
          "angle_excursion_deg": float(np.max(np.abs(trace["angle"][first:used]-trace["angle"][first]))),
          "yaw_abs_max_rad_s": float(np.max(np.abs(trace["yaw"][first:used]))),
          "vendor_rate_max_deg_s": float(np.max(trace["vendor_rate"][first:used])),
          "reason": None if duration >= .5-1e-9 else "insufficient_supported_zero_speed_hold"}


def evaluate_stop_hold(window, report, long_proof, envelope, moving_model, hold_model, long_runtime, lat_runtime,
                       joint_proof, assumptions, proof, *, parameters=None, lateral_grid, cache=None):
  parameters, raw, state, fixed = _admit(window, report, long_proof, envelope, moving_model, hold_model, long_runtime, lat_runtime,
                                       joint_proof, assumptions, proof, parameters, lateral_grid)
  cache = joint.LongCache() if cache is None else cache
  if not isinstance(cache, joint.LongCache):
    raise StopHoldError("shared content-bound LongCache required")
  result, receipt = cache.evaluate(window, report, long_proof, long_runtime, parameters["long"], True)
  if result["status"] == "rejected":
    return {"window_id": window["id"], "status": "rejected", "reason": "long_trial_failed", "long_result": result,
            "selection_rejections": ["long_trial_failed"], "complete_controller_candidate": None, "production_enabled": False}
  t, tape, rdf = np.asarray(window["tape"]["t"]), window["tape"], long.rdf_from_tape(window["tape"])
  origin = fixed["request_index"]
  moving = (result["trace"]["v"][:origin] >= 2.) & (result["trace"]["native_state"][:origin] == 1)
  genuine = any(moving[i:i+5].all() for i in range(0, max(0, len(moving)-4), 5))
  if result["status"] == "complete" and (not genuine or not np.all(result["trace"]["native_state"][origin:] == 2)):
    raise StopHoldError("native moving approach/continuous stopping clock not established")
  handle = lat_runtime.create(*raw, envelope["lp"]["values"], envelope["params"]["values"], candidate=parameters["lateral"],
                              grid=lateral_grid, initialization=envelope["initialization"])
  trace = {key: np.full(len(t), np.nan) for key in ("angle", "yaw", "vendor_rate", "can_torque", "native_torque")}
  trace.update(t=t.copy(), v=result["trace"]["v"].copy(), a=result["trace"]["a"].copy(), physical_curvature=fixed["curvature"],
               fixed_speed=fixed["speed"], fixed_path=fixed["path"], effective_active=np.zeros(len(t), bool), request=np.zeros(len(t), bool),
               path=np.full((len(t), 2), np.nan))
  used, supported, reason, censor, failure = 0, 0, None, None, None
  heading, position, macro, reached_zero = 0., np.zeros(2), [], False
  prior = envelope["initialization"]["last_transmitted_counts"]/409.
  for j, timestamp in enumerate(t):
    if j >= result["used_steps"]:
      reason, censor = result["reason"] or "long_horizon_incomplete", result["censor_t"]
      break
    fields = ("lat_active", "fault_temporary", "fault_permanent", "lat_input_valid")
    if (any(tape[k][j] not in (0, 1) for k in fields) or not tape["lat_input_valid"][j]
        or not joint._finite(tape["lat_input_t"][j]) or not 0 <= timestamp-tape["lat_input_t"][j] <= .2+1e-9
        or not joint._finite(tape["published_delay"][j]) or tape["published_delay"][j] < 0
        or not joint._finite(tape["driver_torque"][j]) or tape["driver_torque"][j] != 0
        or tape["fault_temporary"][j] or tape["fault_permanent"][j]):
      reason, censor = "invalid_or_intervened_lateral_input", float(timestamp)
      break
    index, speed = int(fixed["rdf_index"][j]), float(trace["v"][j])
    if reached_zero and speed != 0:
      reason, censor = "release_or_restart_unsupported", float(timestamp)
      break
    reached_zero |= speed == 0
    if result["trace"]["rdf_index"][j] != index:
      raise StopHoldError("long/lateral latest-ready frame mismatch")
    output = handle.step(t=float(timestamp), active=bool(tape["lat_active"][j]), v=speed, a=float(trace["a"][j]),
                         steering_angle=state["angle"], steering_rate=state["vendor_rate"], driver_torque=0., steering_pressed=False,
                         standstill=bool(speed == 0), fault_temporary=False, fault_permanent=False, prior_delivered_torque=prior,
                         model=lat_runtime.model_from_rdf(rdf["plan"][index], rdf["horizon"]),
                         desired_curvature=float(rdf["desired_curvature"][index]), published_delay=float(tape["published_delay"][j]),
                         learner=envelope["learner"] if j == 0 else None)
    active, request, command = output["active"], output["steer_request"], output["can_torque"]
    if active and not request:
      reason, failure = "active_steer_request_false", deepcopy(output)
      break
    if (not joint._finite(command) or command != int(command) or abs(command) > 409 or not joint._finite(output["torque"])
        or (not active and (request or command != 0))):
      raise StopHoldError("native effective-active/request/CAN contract violated")
    for key in joint._STATE:
      trace[key][j] = state[key]
    trace["can_torque"][j], trace["native_torque"][j], trace["path"][j] = command, output["torque"], position
    trace["effective_active"][j], trace["request"][j] = active, request
    position = position+speed*DT*np.array([np.cos(heading), np.sin(heading)])
    heading += state["yaw"]*DT
    prior = command/409.
    macro.append((speed, int(command), bool(active), bool(request)))
    used += 1
    if len(macro) == 5:
      speeds, commands, active_flags, requests = map(list, zip(*macro, strict=True))
      moving_macro = all(active_flags) and all(requests)
      next_state = (joint.plant.predict_macro(moving_model, state, speeds, commands, envelope["roll"]["value"]) if moving_macro else
                    hold.predict_macro(hold_model, state, speeds, commands, active_flags, requests, envelope["roll"]["value"]))
      if isinstance(next_state, hold.Unsupported):
        reason = ("moving" if moving_macro else "hold")+"_unsupported:"+next_state.reason
        break
      state, macro = joint._state(next_state), []
      supported = used
  nodes = np.arange(0, supported, 5)
  geometric = nodes[fixed["geometry"][nodes]]
  turn_metrics = joint.turns.window_metrics(fixed["curvature"][geometric], (trace["v"]*trace["yaw"])[geometric],
                                          (fixed["curvature"]*trace["v"]**2)[geometric], trace["can_torque"][geometric],
                                          dt_s=.05, measured_yaw=trace["yaw"][geometric]) if len(geometric) else None
  motion = _hold_motion(trace, supported, result)
  rejections = [*result["selection_rejections"], "contact_and_hold_unvalidated", "nonreversing_contact_unvalidated",
                "unvalidated_100hz_zoh_feedback", "native_lateral_initialization_uncertain", "frozen_learner_adaptation_unverified",
                "unvalidated_planar_heading_consistency", "frozen_OEM_standstill_feedback_unverified",
                "simulated_carState_standstill_equals_zero_unvalidated"]
  if (~fixed["geometry"]).any():
    rejections.append("full_horizon_speed_path_reference_unknown")
  if not lat_runtime.bindings.actual_native:
    rejections.append("code_fixture_not_native_lateral_evidence")
  if motion["reason"]:
    rejections.append(motion["reason"])
  if result.get("stopping_result") is None or result["stopping_result"].get("target_feasible") is not True:
    rejections.append("model_stop_target_feasibility_unverified")
  if used != len(t) or reason:
    rejections.append("incomplete_supported_stop_hold_horizon")
  return {"window_id": window["id"], "parameters": parameters, "trace": trace, "long_result": result, "long_cache": receipt,
          "status": "complete" if used == len(t) and reason is None else "censored" if censor is not None else "unsupported",
          "used_steps": used, "supported_steps": supported, "attempted_steps": len(t), "reason": reason, "censor_t": censor, "failure_event": failure,
          "raw_plan_available": fixed["available"], "geometry_prefix_mask": fixed["geometry"], "macro_indices": nodes.tolist(),
          "unknown_geometry_macro_indices": np.arange(0, len(t), 5)[~fixed["geometry"][::5]].tolist(),
          "turns": turn_metrics, "hold_motion": motion, "fixed_reference_sha256": fixed["sha256"], "stop_request_index": origin,
          "raw_car_state_witness_sha256": envelope["raw_car_state_witness"]["sha256"],
          "contact_assumptions": deepcopy(assumptions), "source_binding": deepcopy(proof), "contract": dict(CONTRACT),
          "selection_rejections": sorted(set(rejections)), "selection_eligible": False, "complete_controller_candidate": None,
          "production_enabled": False}


def _summary(results, common):
  # A cached native failure has no trace; preserve it in results, but never
  # pass it to the existing complete-trace scorer or drop its declared ID.
  scored = {name: ({k: v for k, v in row.items() if k != "long_result"} if "long_result" in row
                   and "trace" not in row["long_result"] else row) for name, row in results.items()}
  metrics = joint.summarize(scored, common)
  for key in ("speed_p95", "path_p95"):
    if metrics[key] is not None and not np.isfinite(metrics[key]):
      metrics[key] = None
  metrics["unknown_geometry_by_window"] = {name: result.get("unknown_geometry_macro_indices", []) for name, result in results.items()}
  errors = []
  for name in common:
    row = results.get(name, {})
    if "hold_motion" not in row:
      continue
    target = row.get("long_result", {}).get("stopping_result")
    if row.get("status") != "complete" or not target or not target.get("model_target"):
      errors = None
      break
    errors.append(abs(target["distance_from_request_m"]-target["model_target"]["distance_from_request_m"]))
  metrics["model_stop_distance_mae_m"] = float(np.mean(errors)) if errors else None
  return metrics


def _objective(metrics, baseline):
  joint_score = joint._objective(metrics, baseline)
  after, before = metrics["model_stop_distance_mae_m"], baseline["model_stop_distance_mae_m"]
  if joint_score is None or after is None or before is None or (before == 0 and after > 0):
    return None
  return .5*joint_score+.5*(after/before if before else 1.)


def _constraints(results, baseline, common):
  reasons = joint._stop_constraints(results, baseline, common)
  for name in common:
    if "hold_motion" not in baseline.get(name, {}):
      continue
    old, new = baseline[name]["hold_motion"], results.get(name, {}).get("hold_motion")
    if new is None or new.get("reason") or new.get("supported_zero_speed_s", 0) < .5:
      reasons.append("missing_supported_zero_speed_hold")
    elif not old.get("reason"):
      if new["angle_excursion_deg"] > old["angle_excursion_deg"]*1.02+.01 or new["yaw_abs_max_rad_s"] > old["yaw_abs_max_rad_s"]*1.02+.001:
        reasons.append("hold_motion_regression")
  return reasons


def validation_search(windows, report, long_proofs, envelopes, moving_model, hold_model, lr, lat, joint_proofs, assumptions,
                      stop_proofs, *, grid, lateral_grid):
  """One fixed mixed validation population/grid/cache; all failures retained."""
  if not windows or len({w["id"] for w in windows}) != len(windows) or not isinstance(grid, (list, tuple)) or not 0 < len(grid) <= 64:
    raise StopHoldError("finite1..64 grid and unique nonempty validation population required")
  candidates = [joint._parameters(windows[0]["cp"], p, lateral_grid) for p in grid]
  signatures = [long.value_hash(p) for p in candidates]
  if len(set(signatures)) != len(signatures) or any(p == {"long": None, "lateral": None} for p in candidates):
    raise StopHoldError("distinct explicit candidates and separate unchanged baseline required")
  if any(joint.static_identity(w["cp"], w["sp"]) != joint.static_identity(windows[0]["cp"], windows[0]["sp"]) for w in windows):
    raise StopHoldError("same complete CP/SP for all mixed windows required")
  common, cache, rows = sorted(w["id"] for w in windows), joint.LongCache(), []
  for parameters in [None, *candidates]:
    row = {"parameters": parameters, "results": {}, "failures": [], "selection_rejections": []}
    for window in windows:
      name = window["id"]
      try:
        args = (window, report, long_proofs[name], envelopes[name], moving_model)
        kwargs = {"parameters": parameters, "lateral_grid": lateral_grid, "cache": cache}
        if window.get("regime", "moving") == "moving":
          result = joint.evaluate_joint(*args, lr, lat, joint_proofs[name], **kwargs)
        elif window["regime"] == "stopping":
          result = evaluate_stop_hold(*args, hold_model, lr, lat, joint_proofs[name], assumptions[name], stop_proofs[name], **kwargs)
        else:
          raise StopHoldError("unknown evaluation regime")
        row["results"][name] = result
        row["selection_rejections"].extend(result["selection_rejections"])
        if result["status"] != "complete":
          row["failures"].append({"window_id": name, "status": result["status"], "reason": result["reason"]})
      except Exception as error:
        failure = {"window_id": name, "status": "rejected", "reason": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()}
        row["results"][name] = failure
        row["failures"].append(failure)
    row["metrics"] = _summary(row["results"], common)
    before = rows[0] if rows else row
    row["selection_rejections"].extend(joint.constraints(row["metrics"], before["metrics"]))
    row["selection_rejections"].extend(_constraints(row["results"], before["results"], common))
    row["joint_target_objective"] = _objective(row["metrics"], before["metrics"])
    if len(common) < long.research.MIN_WINDOWS:
      row["selection_rejections"].append("insufficient_fixed_common_population")
    if rows and (row["joint_target_objective"] is None or row["joint_target_objective"] >= .99):
      row["selection_rejections"].append("no_meaningful_joint_target_improvement")
    row["selection_rejections"] = sorted(set(row["selection_rejections"]))
    row["common_window_ids"] = common
    row["coverage"] = (len(common)-len(row["metrics"]["missing_windows"]))/len(common)
    rows.append(row)
  ranked = sorted([row for row in rows[1:] if row["joint_target_objective"] is not None],
                  key=lambda row: (row["joint_target_objective"], json.dumps(row["parameters"], sort_keys=True)))
  for rank, row in enumerate(ranked, 1):
    row["diagnostic_rank"] = rank
  return {"status": "insufficient_evidence", "baseline": rows[0], "rows": rows[1:], "common_window_ids": common,
          "attempted_windows": len(common), "selection": None, "complete_controller_candidate": None,
          "production_enabled": False, "contract": dict(CONTRACT)}
