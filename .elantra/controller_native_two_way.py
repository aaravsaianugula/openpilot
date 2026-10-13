"""Conditional two-way native diagnostics with one shared five-state macro plant.

Caller-authenticated publication schedules are research evidence, not receive
clocks or road acceptance. No scalar longitudinal plant, LongCache or future
measured feedback. Native initialization, ZOH and state coherence are unproven.
"""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import traceback

import numpy as np

import controller_multirate_joint as plant
import controller_native_joint as shared
import controller_native_longitudinal_output as output
import controller_native_stop_hold as stop

long, lateral, turns = shared.long, shared.lateral, shared.turns
DT, MICRO = .01, 5
CARD_PATH = "openpilot/selfdrive/car/card.py"
CARD_SHA = output.PINNED[CARD_PATH]
EVENTS = ("cs_capture", "consume", "co_publish", "cs_publish", "compute", "cc_publish", "apply")
INPUT_FIELDS = ("long_active", "lat_active", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "valid",
                "v_cruise", "driver_torque", "fault_temporary", "fault_permanent", "published_delay", "lat_input_t", "lat_input_valid",
                "carState_t", "carState_valid", "carControl_t", "carControl_valid")


class TwoWayError(shared.JointError):
  pass


def helper_hashes():
  return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in
          (("runner", __import__(__name__)), ("plant", plant), ("shared", shared), ("long", long),
           ("lateral", lateral), ("output", output), ("raw_cs", stop), ("turns", turns))}


def bind_proof(window, report, env, model, lr, lat, out):
  return {"window": long.value_hash(window), "report": long.value_hash(report), "envelope": long.value_hash(env),
          "model": long.value_hash(model), "long_sources": dict(lr.source_hashes), "lateral_sources": dict(lat.sources),
          "output_sources": dict(out.sources), "output_bootstrap": dict(out.bootstrap), "helpers": helper_hashes()}


def _body_hash(value):
  return long.value_hash({k: v for k, v in value.items() if k != "sha256"})


def _latest(rdf, timestamp):
  index = int(np.searchsorted(rdf["output_ready_t"], timestamp, side="right")-1)
  if (index < 0 or not rdf["scored_mask"][index] or not rdf["controller_admissible"][index]
      or not rdf["deadline_admissible"][index] or not 0 <= timestamp-rdf["narrow_t"][index] <= .2+1e-9):
    raise TwoWayError("latest RDF unavailable/warmup/stale/controller-or-deadline inadmissible")
  return index


def _native_inputs(w, report, lr, lat, out, parameters, grid):
  """Deliberately repeat input-only long.admit checks; never its scalar plant gate."""
  if not isinstance(lr, long.NativeRuntime) or not isinstance(lat, lateral.Runtime) or not isinstance(out, output.Runtime):
    raise TwoWayError("three existing native Runtime adapters required")
  p = shared._parameters(w["cp"], parameters, grid)
  pin = w.get("opendbc_gitlink")
  if (not isinstance(pin, str) or not shared._PIN.fullmatch(pin) or pin != lr.opendbc_gitlink or pin != lat.pin or pin != out.pin
      or not isinstance(w.get("cohort"), str) or not w["cohort"] or w["cohort"] != report.get("cohort")
      or not shared._SHA.fullmatch(w.get("source_log_sha256", ""))):
    raise TwoWayError("same known source/cohort/native pin required")
  for sources in (lr.source_hashes, lat.sources, out.sources):
    if not sources or any(not isinstance(v, str) or not shared._SHA.fullmatch(v) for v in sources.values()):
      raise TwoWayError("native source manifest missing/malformed")
  if lr.source_hashes.get(long.LONG_SOURCE) != long.LONG_SHA:
    raise TwoWayError("native LongControl source differs")
  if lr.actual_native:
    paths = ((lr.controller_type, long.LONG_SOURCE), (lr.accel_limits, "opendbc_repo/opendbc/car/interfaces.py"))
    if lr.static_builder is not long.native_messages or any(
        hashlib.sha256(Path(inspect.getfile(fn)).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != lr.source_hashes.get(name)
        for fn, name in paths):
      raise TwoWayError("actual LongControl/limits/message origins differ")
  if lat.bindings.actual_native and (lat.pin != lateral.OPENDBC_PIN or lat.sources.get(lateral.V0_PATH) != lateral.V0_SHA
      or any(Path(inspect.getfile(fn)).resolve() != Path(lateral.__file__).resolve()
             for fn in (lat.bindings.decode_static, lat.bindings.controller_factory, lat.bindings.build_model))):
    raise TwoWayError("actual lateral binding differs")
  out._actual_binding_check()
  walls = report.get("route_start_wall_time_nanos", {})
  long.research.validate_split([(None, {"route": r, "init_data": {"wallTimeNanos": wall}}) for r, wall in walls.items()], report["split"])
  if w.get("partition") != "validation" or w["route"] not in report["split"]["validation"] or w.get("regime", "moving") != "moving":
    raise TwoWayError("reserved validation moving windows only; heldout/stopping refused")
  long.native.validate_parameters(w["cp"], w["sp"])
  if (w["cp"].get("brand") != "hyundai" or type(w["cp"].get("radarUnavailable")) is not bool
      or type(w["sp"].get("flags")) is not int or w["sp"]["flags"] < 0):
    raise TwoWayError("complete native Hyundai output static contract required")
  selected = long.native_parameters(w["cp"], p["long"])
  rdf, inputs = long.rdf_from_tape(w["tape"]), w["model_inputs"]
  long.native.validate_rdf(rdf)
  n = len(rdf["input_t"])
  for name in ("controller_admissible", "deadline_admissible"):
    flag = np.asarray(rdf.get(name))
    if flag.shape != (n,) or flag.dtype.kind != "b":
      raise TwoWayError("explicit RDF admission flags required")
  included = inputs.get("packet_includes_model_smoothing")
  action, packet = np.asarray(inputs.get("action_t")), np.asarray(inputs.get("packet_action_t"))
  delay = w["cp"].get("longitudinalActuatorDelay")
  if (type(included) is not bool or not shared._finite(delay) or delay < 0 or inputs.get("fresh_inference") is not True
      or inputs.get("model_sha256") != long.native.MODEL_SHA or action.shape != (n, 2) or packet.shape != (n, 2)
      or not np.array_equal(action, rdf.get("action_t")) or not np.array_equal(packet, rdf.get("packet_action_t"))
      or not np.isfinite(action).all() or not np.isfinite(packet).all()
      or not np.allclose(action, packet+([0., 0.] if included else [.1, .3]), atol=1e-7, rtol=0)
      or not np.allclose(packet[:, 1], delay+.075+(.3 if included else 0.), atol=1e-7, rtol=0)):
    raise TwoWayError("fixed native delays require exact consumed fresh RDF action_t inputs")
  return p, selected, rdf


def _population(model, report, env):
  if (model.get("kind") != plant.KIND or model.get("features") != list(plant.FEATURES) or model.get("split") != report["split"]
      or any(model.get(k) is not True for k in ("source_eligible", "source_pin_verified", "static_policy_timing_verified"))
      or any(model.get(k) != env.get(k) for k in ("cohort", "opendbc_gitlink", "static_identity_sha256", "policy_identity_sha256"))
      or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
      or model.get("model_sha256") != plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
      or type(model.get("min_samples")) is not int or model["min_samples"] < 100
      or type(model.get("min_routes")) is not int or model["min_routes"] < 2
      or not model.get("record_hashes") or any(not isinstance(v, str) or not shared._SHA.fullmatch(v) for v in model["record_hashes"])):
    raise TwoWayError("hash-bound source-admitted five-state TRAIN plant required")
  lo, hi = (np.asarray(model.get(k), float) for k in ("feature_min", "feature_max"))
  edges = model.get("edges")
  if (lo.shape != (21,) or hi.shape != (21,) or not np.isfinite([lo, hi]).all() or (lo > hi).any()
      or not isinstance(edges, list) or len(edges) != 21
      or any(np.asarray(e).ndim != 1 or not np.isfinite(e).all() or (np.diff(e) <= 0).any() for e in edges)):
    raise TwoWayError("malformed five-state feature population")
  supported = 0
  for key, cell in model.get("cells", {}).items():
    bounds = [np.asarray(cell.get(k), float) for k in ("feature_min", "feature_max")]
    delta = np.asarray(cell.get("joint_delta_median"), float)
    if (len(key.split(":")) != 21 or any(v.shape != (21,) or not np.isfinite(v).all() for v in bounds)
        or (bounds[0] > bounds[1]).any() or delta.shape != (5,) or not np.isfinite(delta).all()
        or type(cell.get("supported")) is not bool):
      raise TwoWayError("malformed five-state empirical cell")
    if cell["supported"]:
      routes = cell.get("routes", [])
      if (type(cell.get("samples")) is not int or cell["samples"] < model["min_samples"] or len(routes) != len(set(routes))
          or len(routes) < model["min_routes"] or any(r not in report["split"]["train"] for r in routes)):
        raise TwoWayError("cell lacks independent TRAIN support")
      supported += 1
  if not supported or supported != model.get("supported_cells"):
    raise TwoWayError("supported five-state cell denominator differs")


def _schedule(w, env, t):
  s = env.get("publication_schedule", {})
  if (s.get("sha256") != _body_hash(s) or s.get("source_log_sha256") != w["source_log_sha256"]
      or s.get("card_source_sha256") != CARD_SHA or s.get("cold_start") is not True or s.get("phase_known") is not False):
    raise TwoWayError("source-bound cold/phase-uncertain card schedule required")
  prefix = np.asarray(s.get("prefix_node_t"))
  if (prefix.ndim != 1 or not len(prefix) or len(prefix) > 10 or prefix.dtype.kind not in "fiu" or not np.isfinite(prefix).all()
      or not np.allclose(np.diff(np.r_[prefix, t[0]]), DT, atol=1e-9, rtol=0)
      or t[0]-prefix[0] > .1+1e-9 or s.get("prefix_state_assumption") != {"mode": "fixed_initial_five_state", "validated": False}):
    raise TwoWayError("explicit bounded causal cold pre-roll required")
  nodes = np.r_[prefix, t]
  booleans = ("apply_call", "co_valid", "startup_empty", "cc_valid", "co_seen", "co_alive", "co_frequency_valid")
  for key in (*[e+"_t" for e in EVENTS], *[e+"_seq" for e in EVENTS], *booleans):
    a = np.asarray(s.get(key))
    kind = "b" if key in booleans else "iu" if key.endswith("_seq") else "fiu"
    if a.shape != nodes.shape or a.dtype.kind not in kind or not np.isfinite(a).all():
      raise TwoWayError("malformed explicit publication schedule")
  inputs = s.get("inputs", {})
  if (set(inputs) != set(INPUT_FIELDS) or any(np.asarray(v).shape != nodes.shape or not np.isfinite(v).all() for v in inputs.values())
      or any(not np.array_equal(inputs[k][len(prefix):], w["tape"].get(k)) for k in INPUT_FIELDS)
      or not np.allclose(np.diff(s["compute_t"]), DT, rtol=0, atol=1e-9)):
    raise TwoWayError("native100Hz and explicit prefix/main exogenous inputs required")
  numeric = {"v_cruise", "driver_torque", "published_delay", "lat_input_t", "carState_t", "carControl_t"}
  if (any(not np.isin(inputs[k], [0, 1]).all() for k in set(INPUT_FIELDS)-numeric)
      or (inputs["published_delay"] < 0).any() or (inputs["v_cruise"] < 0).any()
      or not np.array_equal(s["cs_publish_t"], inputs["carState_t"])
      or not np.array_equal(s["cc_valid"], (inputs["valid"] == 1) & (inputs["carState_valid"] == 1)
                             & (inputs["carControl_valid"] == 1) & (inputs["lat_input_valid"] == 1))):
    raise TwoWayError("invalid exogenous bit/value or CS/CC validity publication witness")
  def pairs(event, i):
    return float(s[event+"_t"][i]), int(s[event+"_seq"][i])
  seen = set()
  for i, node in enumerate(nodes):
    row = {e: pairs(e, i) for e in EVENTS}
    if (any(pair in seen for pair in row.values()) or len(set(row.values())) != len(EVENTS)
        or any(pair[0] > node or (i and pair[0] <= nodes[i-1]) for pair in row.values())
        or not row["cs_capture"] < row["consume"] < row["co_publish"] < row["cs_publish"] < row["apply"]
        or not row["compute"] < row["cc_publish"]):
      raise TwoWayError("future/ambiguous card event order or macro-node crossing")
    seen.update(row.values())
    consumed = [j for j in range(i+1) if pairs("cc_publish", j) < row["consume"]]
    if not consumed and (i != 0 or not s["startup_empty"][i] or s["apply_call"][i]):
      raise TwoWayError("missing candidate CC prefix; empty startup cannot apply")
    if consumed and s["startup_empty"][i]:
      raise TwoWayError("startup-empty witness conflicts with prior candidate CC")
    expected = bool(consumed and s["cc_valid"][consumed[-1]] and s["co_seen"][i] and s["co_alive"][i] and s["co_frequency_valid"][i]
                    and 0 <= row["consume"][0]-pairs("cc_publish", consumed[-1])[0] <= .2+1e-9)
    if bool(s["co_valid"][i]) != expected or (not consumed and s["co_seen"][i]) or (i >= len(prefix) and not expected):
      raise TwoWayError("CO.valid lacks consumed candidate CC valid/seen/alive/frequency/freshness witness")
  return s


def _calibration(rdf, env, inputs):
  c, n = env.get("calibration_witness", {}), len(rdf["input_t"])
  consumed = inputs.get("packet_calibration", {})
  if (not isinstance(consumed, dict) or consumed.get("sha256") != _body_hash(consumed) or _body_hash(consumed) != _body_hash(c)
      or c.get("decision_time_origin") != "max_camera_eof" or c.get("calibration_cutoff") != "decision"
      or c.get("sha256") != _body_hash(c) or c.get("packet_sha256") != env["source_evidence"]["packet_sha256"]
      or not np.array_equal(c.get("frame_id"), rdf["frame_id"]) or not np.array_equal(c.get("decision_t"), rdf["input_t"])):
    raise TwoWayError("calibration/packet/frame/clock binding differs")
  valid, clock = np.asarray(c.get("valid")), np.asarray(c.get("calibration_t"))
  if (valid.shape != (n,) or valid.dtype.kind != "b" or not valid.all() or clock.shape != (n,)
      or not np.isfinite(clock).all() or (clock > c["decision_t"]).any() or (c["decision_t"]-clock > .5).any()
      or not np.array_equal(c.get("narrow_t"), rdf["narrow_t"]) or np.asarray(c.get("wide_t")).shape != (n,)
      or not np.isfinite(c["wide_t"]).all() or not np.array_equal(c["decision_t"], np.maximum(c["narrow_t"], c["wide_t"]))
      or (np.abs(c["wide_t"]-c["narrow_t"]) > .025).any()
      or any(np.asarray(c.get(k)).shape != (n, 3, 3) or not np.isfinite(c[k]).all() for k in ("tfm", "big_tfm"))):
    raise TwoWayError("missing/future/invalid calibrated camera transforms")


def _macro_prefix(cc, co, can, start, end):
  """Same valid/request-true20ms prefix regime as the empirical collector."""
  for stream, name in ((cc, "CC"), (co, "CO"), (can, "CAN")):
    times = [row["key"][0] if "key" in row else row["t"] for row in stream]
    first = int(np.searchsorted(times, start, side="right")-1)
    if first < 0 or start-times[first] > .02+1e-9:
      raise TwoWayError(name+" valid macro prefix unavailable/stale")
    rows = stream[first:]
    span = [time for time in times[first:] if time <= end]
    if any(b-a > .02+1e-9 for a, b in zip(span, span[1:], strict=False)):
      raise TwoWayError(name+" macro prefix gap")
    for entry in rows[:len(span)]:
      valid = entry["request"] if name == "CAN" else entry["valid"]
      if not valid or (name == "CC" and (not entry["active"] or not entry["long_active"])):
        raise TwoWayError(name+" invalid/inactive/request-false macro prefix")


def _envelope(w, env, lat, compute_start):
  """Shared envelope semantics without invoking old joint/long plant admission."""
  if any(env.get(k) != w.get(k) for k in ("route", "cohort", "source_log_sha256", "opendbc_gitlink")):
    raise TwoWayError("envelope source identity differs")
  evidence = env.get("source_evidence", {})
  if (evidence.get("verified") is not True or evidence.get("source_log_sha256") != w["source_log_sha256"]
      or any(not isinstance(evidence.get(k), str) or not shared._SHA.fullmatch(evidence[k])
             for k in ("schema_sha256", "extractor_sha256", "packet_sha256"))):
    raise TwoWayError("verified source/schema/extractor/packet evidence required")
  if (any(not shared._finite(env.get(k)) or env[k] > compute_start for k in ("static_t", "cp_t", "sp_t"))
      or abs(env["static_t"]-max(env["cp_t"], env["sp_t"])) > 1e-9):
    raise TwoWayError("distinct prior CP/SP publication witnesses required")
  raw = [np.asarray(env.get(k)) for k in ("cp_raw", "sp_raw")]
  if any(v.dtype != np.uint8 or v.ndim != 1 or not len(v) for v in raw):
    raise TwoWayError("serialized full CP/SP payloads required")
  cp, sp = lat.bindings.decode_static(*(v.tobytes() for v in raw))
  if (long.value_hash(cp.to_dict()) != long.value_hash(w["cp"]) or long.value_hash(sp.to_dict()) != long.value_hash(w["sp"])
      or env.get("static_identity_sha256") != shared.static_identity(w["cp"], w["sp"])):
    raise TwoWayError("full static semantic identity differs")
  torque = w["cp"].get("lateralTuning", {}).get("torque", {})
  limits = lat.bindings.limits_factory(cp)
  if (w["cp"].get("steerControlType") != "torque" or not {"latAccelFactor", "latAccelOffset", "friction"} <= set(torque)
      or any(not shared._finite(torque.get(k)) for k in ("latAccelFactor", "latAccelOffset", "friction"))
      or torque["latAccelFactor"] <= 0 or torque["friction"] < 0
      or tuple(getattr(limits, k) for k in lateral.LIMIT_KEYS) != (409, 3, 7, 50, 2, 1)):
    raise TwoWayError("baseline native torque parameters or literal409/3/7 envelope invalid")
  for key, age in (("lp", .2), ("params", None), ("initial", .052), ("roll", .2), ("location_quality", .075)):
    shared._event(env.get(key), compute_start, age)
  lp, params = env["lp"]["values"], env["params"]["values"]
  if (set(lp) != set(lateral.LP_KEYS) or any(not shared._finite(v) for v in lp.values())
      or lp["stiffnessFactor"] <= 0 or lp["steerRatio"] <= 0 or set(params) != set(lateral.TYPES)
      or env.get("policy_identity_sha256") != shared.policy_identity(params)):
    raise TwoWayError("complete causal LP and typed policy identity required")
  provider = lateral.ImmutableParams(params)
  for key in lateral.TYPES:
    provider.get(key)
  if (not provider.get_bool("EnforceTorqueControl") or provider.get("TorqueControlTune") != 0.
      or not provider.get_bool("LateralJerkTorqueController") or provider.get_bool("NeuralNetworkLateralControl")):
    raise TwoWayError("native torqueV0/jerk/NN-off policy required")
  state = env["initial"].get("state", {})
  if (set(state) != set(plant.STATE_KEYS) or any(not shared._finite(v) for v in state.values())
      or state["v"] <= .3 or state["vendor_rate"] < 0 or state["v"] != w["initial_v"] or state["a"] != w["initial_a"]):
    raise TwoWayError("complete causal moving five-state anchor required")
  q = env["location_quality"]
  if (not shared._finite(env["roll"].get("value")) or abs(q["t"]-env["roll"]["t"]) > 1e-9
      or any(not shared._finite(q.get(k)) or q[k] < 0 for k in ("yaw_std", "roll_std"))
      or any(q.get(k) != 1 for k in ("yaw_inner_valid", "roll_inner_valid", "inputs_ok", "sensors_ok", "device_stable")) or q.get("status") != 2):
    raise TwoWayError("physical roll/LLK uncertainty and quality required")
  learner = env.get("learner", {})
  if (any(type(learner.get(k)) is not bool for k in ("valid", "admitted", "useParams", "frequency_valid"))
      or (learner.get("t") is not None and (not shared._finite(learner["t"]) or learner["t"] > compute_start))
      or (not learner["admitted"] and not learner.get("reason"))):
    raise TwoWayError("explicit causal learner admission/unavailability required")
  if learner["admitted"] and (not learner["valid"] or not learner["frequency_valid"] or learner["t"] is None
      or compute_start-learner["t"] > 2.5 or (learner["useParams"] and
        (any(not shared._finite(learner.get(k)) for k in ("factor", "offset", "friction")) or learner["factor"] <= 0 or learner["friction"] < 0))):
    raise TwoWayError("stale/malformed admitted learner")
  init, h = env.get("initialization", {}), env.get("history", {})
  if (set(init) != shared._INIT or not shared._finite(init["previous_curvature"]) or type(init["last_transmitted_counts"]) is not int
      or abs(init["last_transmitted_counts"]) > 409 or type(init["angle_counter"]) is not int or init["angle_counter"] < 0
      or type(init["steer_limited"]) is not bool):
    raise TwoWayError("explicit native limiter initialization required")
  arrays = {k: np.asarray(h.get(k)) for k in ("t", "command", "request", "valid")}
  if (any(v.shape != (2,) for v in arrays.values()) or not np.isfinite(arrays["t"]).all()
      or not np.allclose(np.diff(arrays["t"]), .01, atol=1e-9, rtol=0) or not (arrays["t"] < compute_start).all()
      or compute_start-arrays["t"][-1] > .02+1e-9
      or arrays["command"].dtype.kind not in "iu" or (np.abs(arrays["command"]) > 409).any()
      or any(arrays[k].dtype.kind != "b" or not arrays[k].all() for k in ("request", "valid"))
      or arrays["command"][-1] != init["last_transmitted_counts"]):
    raise TwoWayError("strictly past valid request-true CAN history required")
  return {k: float(v) for k, v in state.items()}


def _admit(w, report, env, model, lr, lat, out, proof, parameters, grid):
  if proof != bind_proof(w, report, env, model, lr, lat, out):
    raise TwoWayError("source/input/model/helper binding differs")
  p, selected, rdf = _native_inputs(w, report, lr, lat, out, parameters, grid)
  t = np.asarray(w["tape"]["t"])
  if (t.ndim != 1 or len(t) < MICRO or len(t) % MICRO or not np.isfinite(t).all()
      or not np.allclose(np.diff(t), DT, atol=1e-9, rtol=0)):
    raise TwoWayError("continuous five-node native macros required")
  if not shared._finite(w.get("end_t")) or abs(w["end_t"]-(t[-1]+DT)) > 1e-9:
    raise TwoWayError("explicit covered50ms terminal boundary required")
  s = _schedule(w, env, t)
  state = _envelope(w, env, lat, float(min(s[e+"_t"][0] for e in EVENTS)))
  _population(model, report, env)
  initial_features = np.array([state[k] for k in plant.STATE_KEYS]+[env["roll"]["value"]])
  if (initial_features < np.asarray(model["feature_min"])[:6]).any() or (initial_features > np.asarray(model["feature_max"])[:6]).any():
    raise TwoWayError("initial state/physical roll outside five-state plant support")
  # Validate complete prefix/scored/terminal raw eligibility, never feedback.
  raw_clock = np.asarray([int(v)*1e-9 for v in env["raw_car_state_witness"]["t_ns"]])
  query = np.r_[s["prefix_node_t"], t, w["end_t"]]
  ids = np.searchsorted(raw_clock, query, side="right")-1
  if (ids < 0).any():
    raise TwoWayError("raw CS prefix/terminal coverage missing")
  validation_window = {**w, "tape": {"t": query, "carState_t": np.r_[s["inputs"]["carState_t"], raw_clock[ids[-1]]]}}
  stop._raw_car_state_eligibility(validation_window, env)
  _calibration(rdf, env, w["model_inputs"])
  shared._delay_witness(w, env)
  horizon = np.asarray(rdf.get("horizon"))
  if horizon.shape != (33,) or not np.allclose(horizon, lat.bindings.horizon, rtol=0, atol=1e-7):
    raise TwoWayError("actual native RDF horizon differs")
  inputs = s["inputs"]
  for j, timestamp in enumerate(s["compute_t"]):
    if any(not inputs[service+"_valid"][j] or not 0 <= timestamp-inputs[service+"_t"][j] <= .2+1e-9 for service in ("carState", "carControl")):
      raise TwoWayError("future/stale/invalid native vehicle publication")
    if not inputs["lat_input_valid"][j] or not 0 <= timestamp-inputs["lat_input_t"][j] <= .2+1e-9:
      raise TwoWayError("future/stale/invalid lateral input")
    _latest(rdf, timestamp)
  reference = shared.fixed_reference(w)
  if not reference["valid"].all():
    raise TwoWayError("complete fixed physical reference unavailable")
  reference["target_ay"] = reference["curvature"]*reference["speed"]**2
  reference["sha256"] = long.value_hash({k: reference[k] for k in ("curvature", "acceleration", "speed", "path", "target_ay")})
  censor = env.get("censor_t")
  if censor is not None and not shared._finite(censor):
    raise TwoWayError("finite external censor required")
  effective = deepcopy(w["cp"])
  effective["longitudinalTuning"]["kiV"], effective["stopAccel"] = selected["ki_v"], selected["stop_accel"]
  return p, selected, effective, state, s, rdf, reference


def evaluate_two_way(window, report, envelope, model, long_runtime, lat_runtime, output_runtime, proof, *, parameters=None, lateral_grid):
  try:
    p, selected, effective, state, schedule, rdf, fixed = _admit(
      window, report, envelope, model, long_runtime, lat_runtime, output_runtime, proof, parameters, lateral_grid)
  except (ValueError, TypeError, KeyError, AttributeError) as error:
    raise TwoWayError("admission refused: "+str(error)) from error
  cp, sp = long_runtime.static_builder(effective, window["sp"])
  controller = long_runtime.controller_type(cp, sp)
  lat = lat_runtime.create(cp.to_bytes(), sp.to_bytes(), envelope["lp"]["values"], envelope["params"]["values"],
                           candidate=p["lateral"], grid=lateral_grid, initialization=envelope["initialization"])
  out = output_runtime.create(effective, window["sp"], static_identity_sha256=output.static_identity(effective, window["sp"]))
  t = np.asarray(window["tape"]["t"])
  names = (*plant.STATE_KEYS, "native_torque", "can_torque", "accel_request", "published_accel", "held_accel_after_apply",
           "consumed_cc_index", "requested_cc_index", "applied_accel_request", "native_state", "rdf_index", "published_torque", "pid_integral",
           "output_simulated_a", "output_simulated_v", "native_feedback_v", "native_feedback_a")
  trace = {k: np.full(len(t), np.nan) for k in names}
  trace.update(t=t.copy(), apply_call=np.asarray(schedule["apply_call"])[len(schedule["prefix_node_t"]):].copy(), path=np.full((len(t), 2), np.nan),
               target_ay=fixed["target_ay"].copy(), physical_reference=fixed["acceleration"].copy(),
               fixed_speed=fixed["speed"].copy(), fixed_path=fixed["path"].copy())
  cc_queue, co_queue, cs_queue, attempts = [], [], [], []
  used = supported = 0
  pending = None
  reason = censor = None
  held_counts, held_request = envelope["initialization"]["last_transmitted_counts"], bool(envelope["history"]["request"][-1])
  can_history = [{"t": float(timestamp), "command": int(command), "request": bool(request)} for timestamp, command, request in
                 zip(envelope["history"]["t"], envelope["history"]["command"], envelope["history"]["request"], strict=True)]
  heading, position = 0., np.zeros(2)
  prefix_count = len(schedule["prefix_node_t"])
  inputs = schedule["inputs"]
  for row, node in enumerate(np.r_[schedule["prefix_node_t"], t]):
    j = row-prefix_count
    if (any(inputs[k][row] for k in ("brake_pressed", "gas_pressed", "steering_pressed", "fault_temporary", "fault_permanent"))
        or inputs["driver_torque"][row] != 0 or not inputs["valid"][row]
        or (j >= 0 and (not inputs["long_active"][row] or not inputs["lat_active"][row]))):
      reason, censor = "intervention_inactive_or_invalid", float(schedule["cs_capture_t"][row])
      break
    consumed = computed = published = captured = None
    events = sorted((float(schedule[e+"_t"][row]), int(schedule[e+"_seq"][row]), e) for e in EVENTS)
    try:
      for timestamp, seq, event in events:
        cutoff = envelope.get("censor_t")
        if pending is not None and timestamp >= pending[0] and (cutoff is None or pending[0] < cutoff):
          state, supported, pending = pending[1], pending[2], None
        if cutoff is not None and timestamp >= cutoff:
          reason, censor = "frozen_censor", float(cutoff)
          break
        if event == "cs_capture":
          captured = deepcopy(state)
        elif event == "consume":
          eligible = [row for row in cc_queue if row["key"] < (timestamp, seq)]
          consumed = eligible[-1] if eligible else None
        elif event == "co_publish":
          if consumed is None and schedule["apply_call"][row]:
            raise TwoWayError("empty candidate CC cannot apply")
          snapshot = out.receipt()  # Fenced peek only; never execute a hypothetical later apply.
          published = {"published_accel": snapshot["held_output"], "published_event_valid": bool(schedule["co_valid"][row]),
                       "simulated_v": captured["v"], "simulated_a": captured["a"], "co_t": timestamp}
          co_queue.append({"key": (timestamp, seq), "accel": published["published_accel"], "torque": held_counts/409.,
                           "valid": bool(schedule["co_valid"][row])})
        elif event == "cs_publish":
          cs_queue.append({"key": (timestamp, seq), "state": deepcopy(captured)})
        elif event == "compute":
          prior = [row for row in co_queue if row["key"] < (timestamp, seq)]
          feedback = [item for item in cs_queue if item["key"] < (timestamp, seq)]
          if not prior or not feedback or (j >= 0 and not prior[-1]["valid"]):
            raise TwoWayError("candidate CS/valid CO prefix unavailable; no recorded seed")
          observed = feedback[-1]["state"]
          index = _latest(rdf, timestamp)
          cs = long.native.SimpleNamespace(vEgo=observed["v"], aEgo=observed["a"], brakePressed=False,
                                          cruiseState=long.native.SimpleNamespace(standstill=bool(inputs["cruise_standstill"][row])))
          limits = long_runtime.accel_limits(cp, sp, observed["v"], float(inputs["v_cruise"][row])/3.6)
          if not np.array_equal(limits, long.ACCEL_LIMITS):
            raise TwoWayError("native Hyundai acceleration bounds changed")
          long_active, lat_active = bool(inputs["long_active"][row]), bool(inputs["lat_active"][row])
          if not long_active:
            controller.reset()
          request = float(controller.update(long_active, cs, float(rdf["desired_acceleration"][index]), bool(rdf["should_stop"][index]), limits))
          if not shared._finite(request) or not limits[0] <= request <= limits[1]:
            raise TwoWayError("native acceleration request outside bounds")
          lateral_result = lat.compute(t=timestamp, active=lat_active, v=observed["v"], a=observed["a"], steering_angle=observed["angle"],
            steering_rate=observed["vendor_rate"], driver_torque=0., steering_pressed=False, standstill=observed["v"] <= .3,
            fault_temporary=False, fault_permanent=False, prior_published_torque=prior[-1]["torque"],
            model=lat_runtime.model_from_rdf(rdf["plan"][index], rdf["horizon"]), desired_curvature=float(rdf["desired_curvature"][index]),
            published_delay=float(inputs["published_delay"][row]), learner=envelope["learner"] if row == 0 else None)
          computed = {"index": j, "accel": request, "state": int(controller.long_control_state), "long_active": long_active,
                      "torque": lateral_result["torque"], "active": lateral_result["active"], "rdf_index": index,
                      "feedback_state": deepcopy(observed), "valid": bool(schedule["cc_valid"][row])}
        elif event == "cc_publish":
          if computed is None:
            raise TwoWayError("CC publication precedes its native computation")
          cc_queue.append({**computed, "key": (timestamp, seq)})
        elif event == "apply":
          actual_apply = bool(schedule["apply_call"][row])
          if consumed is None and actual_apply:
            raise TwoWayError("no consumed candidate CC")
          native_state = int(controller.long_control_state) if consumed is None else consumed["state"]
          requested = float(controller.last_output_accel) if consumed is None else consumed["accel"]
          applied_output = out.step(t=timestamp, apply_call=actual_apply, requested_accel=requested, long_state=output.STATES[native_state],
                                    long_active=False if consumed is None else consumed["long_active"], simulated_v=captured["v"],
                                    simulated_a=captured["a"], published_event_valid=bool(schedule["co_valid"][row]))
          if applied_output["published_accel"] != published["published_accel"]:
            raise TwoWayError("native held output changed between CO publication and actual apply")
          published["held_accel_after_apply"] = applied_output["held_accel_after_apply"]
          if actual_apply:
            applied = lat.apply(t=timestamp, torque=consumed["torque"], active=consumed["active"], driver_torque=0., steering_angle=captured["angle"])
            held_counts, held_request = applied["can_torque"], applied["steer_request"]
            can_history.append({"t": timestamp, "command": held_counts, "request": bool(held_request)})
        attempts.append({"node_index": j, "event": event, "t": timestamp, "sequence": seq,
                         "computed": deepcopy(computed) if event == "compute" else None,
                         "publication": deepcopy(published) if event == "co_publish" else None,
                         "captured_state": deepcopy(captured) if event in ("cs_capture", "apply") else None})
      if reason:
        break
      cutoff = envelope.get("censor_t")
      if pending is not None and node >= pending[0] and (cutoff is None or pending[0] < cutoff):
        state, supported, pending = pending[1], pending[2], None
      if cutoff is not None and node >= cutoff:
        reason, censor = "frozen_censor", float(cutoff)
        break
      if j < 0:
        continue  # Every prefix event retained; no invalid cold report enters a plant macro.
      if not held_request or not computed["active"]:
        raise TwoWayError("moving active request-false unsupported")
      for stream, name in ((cc_queue, "CC"), (co_queue, "CO")):
        if not stream or not 0 <= node-stream[-1]["key"][0] <= .02+1e-9:
          raise TwoWayError(name+" model-node publication missing/stale")
      if not can_history or not 0 <= node-can_history[-1]["t"] <= .02+1e-9:
        raise TwoWayError("CAN model-node transmission proxy missing/stale")
      if j % MICRO == 0:
        _macro_prefix(cc_queue, co_queue, can_history, float(node-.02), float(node))
      for key in plant.STATE_KEYS:
        trace[key][j] = state[key]
      trace["native_torque"][j], trace["can_torque"][j] = computed["torque"], held_counts
      trace["accel_request"][j], trace["published_accel"][j] = cc_queue[-1]["accel"], co_queue[-1]["accel"]
      trace["held_accel_after_apply"][j], trace["published_torque"][j] = published["held_accel_after_apply"], co_queue[-1]["torque"]
      trace["output_simulated_a"][j], trace["output_simulated_v"][j] = published["simulated_a"], published["simulated_v"]
      trace["consumed_cc_index"][j], trace["requested_cc_index"][j] = -1 if consumed is None else consumed["index"], cc_queue[-1]["index"]
      trace["applied_accel_request"][j] = np.nan if consumed is None or not schedule["apply_call"][row] else consumed["accel"]
      trace["native_state"][j], trace["rdf_index"][j], trace["pid_integral"][j] = computed["state"], computed["rdf_index"], float(controller.pid.i)
      trace["native_feedback_v"][j], trace["native_feedback_a"][j] = computed["feedback_state"]["v"], computed["feedback_state"]["a"]
      trace["path"][j] = position
      position += state["v"]*DT*np.array([np.cos(heading), np.sin(heading)])
      heading += state["yaw"]*DT
      used += 1
      if used % MICRO == 0:
        sl = slice(used-MICRO, used)
        next_state = plant.predict_macro(model, state, trace["accel_request"][sl], trace["published_accel"][sl],
                                         trace["can_torque"][sl], envelope["roll"]["value"])
        if isinstance(next_state, plant.Unsupported):
          reason = "macro_unsupported:"+next_state.reason
          break
        pending = (float(t[used]) if used < len(t) else float(window["end_t"]), next_state, used)
    except Exception as error:
      reason = str(error)
      attempts.append({"node_index": j, "failure_type": type(error).__name__, "traceback": traceback.format_exc()})
      break
  if reason is None and pending is not None:
    if envelope.get("censor_t") is not None and envelope["censor_t"] <= pending[0]:
      reason, censor = "frozen_censor_before_terminal_boundary", float(envelope["censor_t"])
    elif window["end_t"]+1e-9 >= pending[0]:
      state, supported = pending[1], pending[2]
    else:
      reason = "terminal_macro_boundary_uncovered"
  nodes = np.arange(0, supported, MICRO)
  metrics = turns.window_metrics(fixed["curvature"][nodes], (trace["v"]*trace["yaw"])[nodes], fixed["target_ay"][nodes],
                                 trace["can_torque"][nodes], dt_s=.05, measured_yaw=trace["yaw"][nodes]) if len(nodes) else None
  return {"window_id": window["id"], "parameters": p, "native_long_parameters": selected, "trace": trace, "event_attempts": attempts,
          "status": "complete" if supported == len(t) and reason is None else "censored" if censor is not None else "unsupported",
          "reason": reason, "censor_t": censor, "used_steps": used, "supported_steps": supported, "attempted_steps": len(t),
          "macro_indices": nodes.tolist(), "turns": metrics, "fixed_reference_sha256": fixed["sha256"], "source_binding": deepcopy(proof),
          "effective_static_sha256": output.static_identity(effective, window["sp"]), "output_receipt": out.receipt(),
          "limitations": ["Only events wholly before each declared model node are supported; cross-node event interleaving is refused",
                          "Source-bound CS capture/publication times are an unvalidated sampling hypothesis, never sensor-acquisition proof",
                          "Per-node intervention censors the whole tick conservatively; earlier partial-tick commands are not simulated",
                          "Macro samples and reported metrics are50ms conditional observations; replicated ZOH ticks are not independent physics proof"],
          "selection_rejections": ["unvalidated_100hz_zoh", "unvalidated_joint_median_coherence", "unvalidated_v_a_kinematics",
            "unvalidated_markov_command_history_sufficiency_for_actuator_latency",
            "native_initialization_and_apply_phase_uncertain", "frozen_learner_and_perception", "low_speed_stop_hold_plant_missing",
            "unvalidated_held_initial_state_during_cold_prefix", "unvalidated_sensor_and_state_publication_sampling",
            "publication_schedule_caller_authenticated_not_receive_clock", "CAN_apply_is_command_proxy_not_delivered_torque"],
          "selection_eligible": False, "complete_controller_candidate": None, "production_enabled": False}


def summarize(results, common):
  missing = [key for key in common if results.get(key, {}).get("status") != "complete"]
  admitted = [key for key in common if key not in missing]
  turn_metrics = turns.summarize({key: results[key]["turns"] for key in admitted}, common)
  scores, speeds, paths = {}, [], []
  for key in admitted:
    r, trace = results[key], results[key]["trace"]
    nodes = np.asarray(r["macro_indices"], int)
    if len(nodes) < 2:
      continue
    acceleration, target, command = (trace[name][nodes] for name in ("a", "physical_reference", "accel_request"))
    error = np.abs(acceleration-target)
    scores[key] = {"errors": error.tolist(), "acceleration_errors": error[target > .05].tolist(), "braking_errors": error[target < -.05].tolist(),
                   "jerk": (np.abs(np.diff(acceleration))/.05).tolist(), "command_rate": (np.abs(np.diff(command))/.05).tolist(),
                   "saturated": ((command <= -3.5+1e-8) | (command >= 2.-1e-8)).astype(float).tolist()}
    speeds.extend(np.abs(trace["v"][nodes]-trace["fixed_speed"][nodes]).tolist())
    paths.extend(np.linalg.norm(trace["path"][nodes]-trace["fixed_path"][nodes], axis=1).tolist())
  return {"missing_windows": missing, "common_window_ids": list(common), "turns": turn_metrics,
          "long": long.summarize(scores, common) if common and not missing and len(scores) == len(common) else None,
          "fixed_reference_by_window": {key: results[key].get("fixed_reference_sha256") for key in common if key in results},
          "speed_p95": float(np.percentile(speeds, 95)) if speeds else None, "path_p95": float(np.percentile(paths, 95)) if paths else None}


def validation_search(windows, report, envelopes, model, lr, lat, out, proofs, *, grid, lateral_grid):
  if not windows or len({w["id"] for w in windows}) != len(windows) or not isinstance(grid, (list, tuple)) or not 0 < len(grid) <= 64:
    raise TwoWayError("unique validation windows and declared finite1..64 grid required")
  common, rows = sorted(w["id"] for w in windows), []
  for parameters in (None, *grid):
    row = {"parameters": deepcopy(parameters), "results": {}, "failures": [], "selection_rejections": []}
    for window in windows:
      key = window["id"]
      try:
        result = evaluate_two_way(window, report, envelopes[key], model, lr, lat, out, proofs[key],
                                  parameters=parameters, lateral_grid=lateral_grid)
      except Exception as error:
        result = {"window_id": key, "status": "rejected", "reason": str(error), "error_type": type(error).__name__,
                  "traceback": traceback.format_exc()}
      row["results"][key] = result
      row["selection_rejections"].extend(result.get("selection_rejections", []))
      if result["status"] != "complete":
        row["failures"].append({"window_id": key, "status": result["status"], "reason": result["reason"]})
    row["metrics"] = summarize(row["results"], common)
    before = rows[0] if rows else row
    row["selection_rejections"].extend(shared.constraints(row["metrics"], before["metrics"]))
    row["joint_target_objective"] = shared._objective(row["metrics"], before["metrics"])
    row["selection_rejections"].extend(["no_validated_two_way_stop_hold_or_coherence", "diagnostic_only"])
    if len(common) < long.research.MIN_WINDOWS:
      row["selection_rejections"].append("insufficient_fixed_common_population")
    row["selection_rejections"] = sorted(set(row["selection_rejections"]))
    row["common_window_ids"], row["coverage"] = list(common), (len(common)-len(row["metrics"]["missing_windows"]))/len(common)
    rows.append(row)
  ranked = sorted((r for r in rows[1:] if r["joint_target_objective"] is not None), key=lambda r: r["joint_target_objective"])
  for rank, row in enumerate(ranked, 1):
    row["diagnostic_rank"] = rank
  return {"status": "insufficient_evidence", "baseline": rows[0], "rows": rows[1:], "common_window_ids": common,
          "attempted_windows": len(common), "selection": None, "complete_controller_candidate": None, "production_enabled": False}
