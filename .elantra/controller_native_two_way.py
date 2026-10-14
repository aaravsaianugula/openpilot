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
import controller_multirate_joint_stop_hold as low_plant
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


def helper_hashes(*, include_low_speed=False):
  result = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in
          (("runner", __import__(__name__)), ("plant", plant), ("shared", shared), ("long", long),
           ("lateral", lateral), ("output", output), ("raw_cs", stop), ("turns", turns))}
  if include_low_speed:
    result["low_speed_plant"] = hashlib.sha256(Path(low_plant.__file__).read_bytes()).hexdigest()
  return result


def _choose_stop_model(window, low_speed_model, approach_stop_model):
  if low_speed_model is not None and approach_stop_model is not None:
    raise TwoWayError("low_speed_model and approach_stop_model are mutually exclusive")
  if window.get("regime", "moving") != "stopping" and (low_speed_model is not None or approach_stop_model is not None):
    raise TwoWayError("categorical stopping model supplied to moving-only window")
  return approach_stop_model if approach_stop_model is not None else low_speed_model


def bind_proof(window, report, env, model, lr, lat, out, *, low_speed_model=None, approach_stop_model=None):
  selected = _choose_stop_model(window, low_speed_model, approach_stop_model)
  result = {"window": long.value_hash(window), "report": long.value_hash(report), "envelope": long.value_hash(env),
          "model": long.value_hash(model), "long_sources": dict(lr.source_hashes), "lateral_sources": dict(lat.sources),
          "output_sources": dict(out.sources), "output_bootstrap": dict(out.bootstrap),
          "helpers": helper_hashes(include_low_speed=selected is not None)}
  if low_speed_model is not None:
    result["low_speed_model"] = long.value_hash(low_speed_model)
  if approach_stop_model is not None:
    result["approach_stop_model"] = long.value_hash(approach_stop_model)
  return result


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
  if w.get("partition") != "validation" or w["route"] not in report["split"]["validation"] or w.get("regime", "moving") not in ("moving", "stopping"):
    raise TwoWayError("reserved validation moving/stopping windows only; heldout/unknown regime refused")
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
  _population_structure(model, report, 21)


def _population_structure(model, report, dimensions, *, categorical=False):
  """One pre-factory structural gate for both empirical five-state schemas."""
  lo, hi = (np.asarray(model.get(k), float) for k in ("feature_min", "feature_max"))
  raw_edges, patterns, cells = model.get("edges"), model.get("patterns"), model.get("cells")
  if (lo.shape != (dimensions,) or hi.shape != (dimensions,) or not np.isfinite([lo, hi]).all() or (lo > hi).any()
      or not isinstance(raw_edges, list) or len(raw_edges) != dimensions or not isinstance(cells, dict)):
    raise TwoWayError("malformed empirical feature population")
  edges = [np.asarray(e, float) for e in raw_edges]
  if any(e.ndim != 1 or not np.isfinite(e).all() or (np.diff(e) <= 0).any() or (e < a).any() or (e >= b).any()
         for e, a, b in zip(edges, lo, hi, strict=True)):
    raise TwoWayError("malformed empirical feature edges/bounds")
  if categorical and (not isinstance(patterns, list) or any(not isinstance(v, str) or len(v) != 12 or v[1] != "|"
      or any(c not in "01" for c in v.replace("|", "")) for v in patterns) or len(patterns) != len(set(patterns))):
    raise TwoWayError("malformed low-speed pattern population")
  supported = 0
  for key, cell in cells.items():
    if not isinstance(key, str) or not isinstance(cell, dict):
      raise TwoWayError("malformed empirical cell/key")
    parts = key.split("|") if categorical else [key]
    if categorical and (len(parts) != 3 or "|".join(parts[:2]) not in patterns):
      raise TwoWayError("malformed empirical mode key")
    tokens = parts[-1].split(":")
    if (len(tokens) != dimensions or any(not token.isascii() or not token.isdigit() or str(int(token)) != token for token in tokens)
        or any(int(token) > len(edge) for token, edge in zip(tokens, edges, strict=True))):
      raise TwoWayError("malformed empirical bin key")
    bounds = [np.asarray(cell.get(k), float) for k in ("feature_min", "feature_max")]
    delta, routes, samples = np.asarray(cell.get("joint_delta_median"), float), cell.get("routes"), cell.get("samples")
    if (any(v.shape != (dimensions,) or not np.isfinite(v).all() for v in bounds)
        or (bounds[0] > bounds[1]).any() or delta.shape != (5,) or not np.isfinite(delta).all()
        or (bounds[0] < lo).any() or (bounds[1] > hi).any() or type(cell.get("supported")) is not bool
        or type(samples) is not int or samples < 1 or not isinstance(routes, list)
        or any(not isinstance(r, str) or r not in report["split"]["train"] for r in routes) or len(routes) != len(set(routes))):
      raise TwoWayError("malformed empirical cell bounds/routes/delta")
    categories = cell.get("next_hardware_standstill_values") if categorical else [False]
    if categorical:
      bits = np.array(list(map(int, parts[1])))
      if (not isinstance(categories, list) or not categories or any(type(v) is not bool for v in categories)
          or len(categories) != len(set(categories)) or not np.array_equal(bounds[0][-10:], bits)
          or not np.array_equal(bounds[1][-10:], bits)):
        raise TwoWayError("malformed empirical categorical state/pattern bounds")
    if cell["supported"]:
      if samples < model["min_samples"] or len(routes) < model["min_routes"] or len(categories) != 1:
        raise TwoWayError("cell lacks independent TRAIN/category support")
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


def _macro_prefix(cc, co, can, start, end, *, stopping=False):
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
      if stopping and name == "CAN":
        valid = (entry["active"] and entry["request"]) or (not entry["active"] and not entry["request"] and entry["command"] == 0)
      if not valid or (name == "CC" and (not entry["long_active"] or (not stopping and not entry["active"]))):
        raise TwoWayError(name+" invalid/inactive/request-false macro prefix")


def _low_population(model, report, env, *, approach=False):
  """Validate only the used31-state model; no legacy or unused moving cells."""
  kind = low_plant.APPROACH_KIND if approach else low_plant.KIND
  if not isinstance(model, dict):
    raise TwoWayError("source-bound categorical31-state TRAIN population unavailable")
  domain_differs = ((model.get("initial_speed_domain") != "retained_train_global_and_cell_bounds"
                     or "low_speed_max_initial_mps" in model) if approach else
                    model.get("low_speed_max_initial_mps") != low_plant.LOW_SPEED_MAX_MPS)
  if (model.get("kind") != kind or model.get("features") != list(low_plant.FEATURES)
      or model.get("split") != report["split"] or any(model.get(k) is not True for k in
        ("source_eligible", "source_pin_verified", "static_policy_timing_verified"))
      or any(model.get(k) != env.get(k) for k in ("cohort", "opendbc_gitlink", "static_identity_sha256", "policy_identity_sha256"))
      or (model.get("macro_dt_s"), model.get("native_dt_s"), model.get("micro_nodes")) != (.05, .01, 5)
      or domain_differs
      or model.get("native_policy_source_sha256") != low_plant.NATIVE_POLICY_SHA or model.get("card_source_sha256") != CARD_SHA
      or model.get("model_sha256") != low_plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
      or type(model.get("min_samples")) is not int or model["min_samples"] < 100
      or type(model.get("min_routes")) is not int or model["min_routes"] < 2
      or not model.get("record_hashes") or any(not isinstance(v, str) or not shared._SHA.fullmatch(v) for v in model["record_hashes"])):
    raise TwoWayError("source-bound categorical31-state TRAIN population unavailable")
  _population_structure(model, report, 31, categorical=True)


def _initial_standstill(env, compute_start, first_prefix_node):
  initial, raw = env["initial"], env["raw_car_state_witness"]
  flag, ns = initial.get("hardware_standstill"), initial.get("hardware_standstill_t_ns")
  if type(flag) is not bool or type(ns) is not int or not 0 <= ns <= np.iinfo(np.uint64).max:
    raise TwoWayError("independent initial hardware standstill witness required")
  found = np.flatnonzero(raw["t_ns"] == ns)
  timestamp = ns*1e-9
  publication = np.asarray([int(v)*1e-9 for v in raw["t_ns"]])
  latest = int(np.searchsorted(publication, compute_start, side="right")-1)
  if (len(found) != 1 or latest < 0 or int(found[0]) != latest
      or not 0 <= initial["t"]-timestamp <= .015+1e-9 or timestamp > compute_start):
    raise TwoWayError("initial hardware standstill must precede earliest cold-prefix use")
  i = int(found[0])
  if (not raw["event_valid"][i] or not raw["can_valid"][i] or bool(raw["standstill"][i]) != flag
      or not np.isfinite(raw["v_ego"][i]) or raw["v_ego"][i] != initial["state"]["v"] or (flag and raw["v_ego"][i] > .3)):
    raise TwoWayError("initial hardware standstill raw source differs/invalid")
  last = int(np.searchsorted(publication, first_prefix_node, side="right")-1)
  bridge = slice(i, last+1)
  speed = raw["v_ego"][bridge]
  if (last < i or not raw["event_valid"][bridge].all() or not raw["can_valid"][bridge].all()
      or not np.isfinite(speed).all() or (speed < 0).any() or (raw["standstill"][bridge] & (speed > .3)).any()
      or any(int(b)-int(a) > 15_000_000 for a, b in zip(raw["t_ns"][i:last], raw["t_ns"][i+1:last+1], strict=True))):
    raise TwoWayError("initial category to cold-prefix bridge invalid/gapped/incompatible")
  active = np.asarray(env["history"].get("effective_active"))
  if active.shape != (2,) or active.dtype.kind != "b" or not active.all():
    raise TwoWayError("causal active/request-true approach CAN history required")
  return flag


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


def _admit(w, report, env, model, lr, lat, out, proof, parameters, grid, low_speed_model, approach_stop_model):
  categorical_model = _choose_stop_model(w, low_speed_model, approach_stop_model)
  if proof != bind_proof(w, report, env, model, lr, lat, out, low_speed_model=low_speed_model, approach_stop_model=approach_stop_model):
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
  stopping = w.get("regime", "moving") == "stopping"
  if stopping:
    _low_population(categorical_model, report, env, approach=approach_stop_model is not None)
    if approach_stop_model is None and state["v"] > low_plant.LOW_SPEED_MAX_MPS:
      raise TwoWayError("categorical_standstill_transition_unavailable_outside_low_speed_domain")
  else:
    _population(model, report, env)
  used_model = categorical_model if stopping else model
  initial_features = np.array([state[k] for k in plant.STATE_KEYS]+[env["roll"]["value"]])
  if (initial_features < np.asarray(used_model["feature_min"])[:6]).any() or (initial_features > np.asarray(used_model["feature_max"])[:6]).any():
    raise TwoWayError("initial state/physical roll outside five-state plant support")
  # Validate complete prefix/scored/terminal raw eligibility, never feedback.
  raw_clock = np.asarray([int(v)*1e-9 for v in env["raw_car_state_witness"]["t_ns"]])
  query = np.r_[s["prefix_node_t"], t, w["end_t"]]
  ids = np.searchsorted(raw_clock, query, side="right")-1
  if (ids < 0).any():
    raise TwoWayError("raw CS prefix/terminal coverage missing")
  validation_window = {**w, "tape": {"t": query, "carState_t": np.r_[s["inputs"]["carState_t"], raw_clock[ids[-1]]]}}
  stop._raw_car_state_eligibility(validation_window, env)
  if stopping:
    _initial_standstill(env, float(min(s[e+"_t"][0] for e in EVENTS)), float(s["prefix_node_t"][0]))
    if long.verified_stop_uncertainty(w, report["split"]["train"]) is None:
      raise TwoWayError("frozen20-outcome/two-TRAIN-route stopping uncertainty unavailable")
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
  reference = stop._references(w) if stopping else shared.fixed_reference(w)
  if stopping:
    prefix_count = len(s["prefix_node_t"])
    if (s["inputs"]["cruise_standstill"][:prefix_count+reference["request_index"]].any()
        or any(rdf["should_stop"][_latest(rdf, float(v))] for v in s["compute_t"][:prefix_count])):
      raise TwoWayError("pre-request OEM standstill or cold-prefix prior stop/release unsupported")
  if not stopping and not reference["valid"].all():
    raise TwoWayError("complete fixed physical reference unavailable")
  reference["target_ay"] = reference["curvature"]*reference["speed"]**2
  reference["sha256"] = long.value_hash({k: reference[k] for k in
    (("curvature", "acceleration", "speed", "path", "target_ay", "geometry", "requests") if stopping else
     ("curvature", "acceleration", "speed", "path", "target_ay"))})
  censor = env.get("censor_t")
  if censor is not None and not shared._finite(censor):
    raise TwoWayError("finite external censor required")
  effective = deepcopy(w["cp"])
  effective["longitudinalTuning"]["kiV"], effective["stopAccel"] = selected["ki_v"], selected["stop_accel"]
  return p, selected, effective, state, s, rdf, reference


def evaluate_two_way(window, report, envelope, model, long_runtime, lat_runtime, output_runtime, proof, *, parameters=None, lateral_grid,
                     low_speed_model=None, approach_stop_model=None):
  try:
    p, selected, effective, state, schedule, rdf, fixed = _admit(
      window, report, envelope, model, long_runtime, lat_runtime, output_runtime, proof, parameters, lateral_grid,
      low_speed_model, approach_stop_model)
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
  stopping = window.get("regime", "moving") == "stopping"
  categorical_model = approach_stop_model if approach_stop_model is not None else low_speed_model
  categorical_predict = low_plant.predict_approach_macro if approach_stop_model is not None else low_plant.predict_macro
  categorical_speed_max = None if approach_stop_model is not None else low_plant.LOW_SPEED_MAX_MPS
  hardware_standstill = envelope["initial"]["hardware_standstill"] if stopping else None
  if stopping:
    trace.update(hardware_standstill=np.full(len(t), np.nan), effective_active=np.zeros(len(t), bool),
                 request=np.zeros(len(t), bool), distance=np.full(len(t), np.nan), native_compute_t=np.full(len(t), np.nan))
  cc_queue, co_queue, cs_queue, attempts = [], [], [], []
  used = supported = 0
  pending = None
  reason = censor = None
  held_counts, held_request = envelope["initialization"]["last_transmitted_counts"], bool(envelope["history"]["request"][-1])
  held_active = bool(envelope["history"]["effective_active"][-1]) if stopping else True
  can_history = [{"t": float(timestamp), "command": int(command), "request": bool(request), "active": held_active} for timestamp, command, request in
                 zip(envelope["history"]["t"], envelope["history"]["command"], envelope["history"]["request"], strict=True)]
  heading, position, distance, reached_stop = 0., np.zeros(2), 0., False
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
          if stopping:
            hardware_standstill = pending[3]
          state, supported, pending = pending[1], pending[2], None
        if cutoff is not None and timestamp >= cutoff:
          reason, censor = "frozen_censor", float(cutoff)
          break
        if event == "cs_capture":
          captured = deepcopy(state)
          if stopping:
            captured["hardware_standstill"] = hardware_standstill
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
          if stopping and j >= 0 and int(controller.long_control_state) in (0, 3):
            raise TwoWayError("native_stop_release_or_restart_unsupported")
          if not shared._finite(request) or not limits[0] <= request <= limits[1]:
            raise TwoWayError("native acceleration request outside bounds")
          lateral_result = lat.compute(t=timestamp, active=lat_active, v=observed["v"], a=observed["a"], steering_angle=observed["angle"],
            steering_rate=observed["vendor_rate"], driver_torque=0., steering_pressed=False,
            standstill=observed["hardware_standstill"] if stopping else observed["v"] <= .3,
            fault_temporary=False, fault_permanent=False, prior_published_torque=prior[-1]["torque"],
            model=lat_runtime.model_from_rdf(rdf["plan"][index], rdf["horizon"]), desired_curvature=float(rdf["desired_curvature"][index]),
            published_delay=float(inputs["published_delay"][row]), learner=envelope["learner"] if row == 0 else None)
          computed = {"index": j, "accel": request, "state": int(controller.long_control_state), "long_active": long_active,
                      "torque": lateral_result["torque"], "active": lateral_result["active"], "rdf_index": index,
                      "feedback_state": deepcopy(observed), "valid": bool(schedule["cc_valid"][row])}
          if stopping:
            computed["native_compute_t"] = timestamp
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
            held_active = bool(consumed["active"])
            can_history.append({"t": timestamp, "command": held_counts, "request": bool(held_request), "active": held_active})
        attempts.append({"node_index": j, "event": event, "t": timestamp, "sequence": seq,
                         "computed": deepcopy(computed) if event == "compute" else None,
                         "publication": deepcopy(published) if event == "co_publish" else None,
                         "captured_state": deepcopy(captured) if event in ("cs_capture", "apply") else None})
      if reason:
        break
      cutoff = envelope.get("censor_t")
      if pending is not None and node >= pending[0] and (cutoff is None or pending[0] < cutoff):
        if stopping:
          hardware_standstill = pending[3]
        state, supported, pending = pending[1], pending[2], None
      if cutoff is not None and node >= cutoff:
        reason, censor = "frozen_censor", float(cutoff)
        break
      if j < 0:
        continue  # Every prefix event retained; no invalid cold report enters a plant macro.
      if not stopping and (not held_request or not computed["active"]):
        raise TwoWayError("moving active request-false unsupported")
      if stopping and ((held_active and not held_request) or (not held_active and (held_request or held_counts != 0))):
        raise TwoWayError("inactive_command_or_request_or_active_request_false")
      for stream, name in ((cc_queue, "CC"), (co_queue, "CO")):
        if not stream or not 0 <= node-stream[-1]["key"][0] <= .02+1e-9:
          raise TwoWayError(name+" model-node publication missing/stale")
      if not can_history or not 0 <= node-can_history[-1]["t"] <= .02+1e-9:
        raise TwoWayError("CAN model-node transmission proxy missing/stale")
      if j % MICRO == 0:
        _macro_prefix(cc_queue, co_queue, can_history, float(node-.02), float(node), stopping=stopping)
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
      if stopping:
        trace["hardware_standstill"][j], trace["effective_active"][j], trace["request"][j] = hardware_standstill, held_active, held_request
        trace["distance"][j] = distance
        trace["native_compute_t"][j] = computed["native_compute_t"]
      distance += state["v"]*DT
      position += state["v"]*DT*np.array([np.cos(heading), np.sin(heading)])
      heading += state["yaw"]*DT
      used += 1
      if used % MICRO == 0:
        sl = slice(used-MICRO, used)
        if stopping:
          if categorical_speed_max is not None and state["v"] > categorical_speed_max:
            raise TwoWayError("categorical_standstill_transition_unavailable_outside_low_speed_domain")
          predicted = categorical_predict(categorical_model, state, trace["accel_request"][sl], trace["published_accel"][sl],
            trace["can_torque"][sl], trace["effective_active"][sl], trace["request"][sl], envelope["roll"]["value"], hardware_standstill)
          next_state = predicted if isinstance(predicted, plant.Unsupported) else predicted["state"]
          next_hardware = None if isinstance(predicted, plant.Unsupported) else predicted["hardware_standstill"]
          if not isinstance(predicted, plant.Unsupported):
            if reached_stop and (next_state["v"] != 0 or not next_hardware):
              raise TwoWayError("source_predicted_release_or_restart_unsupported")
            reached_stop |= next_state["v"] == 0 and next_hardware
        else:
          next_state = plant.predict_macro(model, state, trace["accel_request"][sl], trace["published_accel"][sl],
                                           trace["can_torque"][sl], envelope["roll"]["value"])
          next_hardware = None
        if isinstance(next_state, plant.Unsupported):
          reason = "macro_unsupported:"+next_state.reason
          break
        pending = (float(t[used]) if used < len(t) else float(window["end_t"]), next_state, used, next_hardware)
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
  geometric = nodes[fixed["geometry"][nodes]] if stopping else nodes
  metrics = turns.window_metrics(fixed["curvature"][geometric], (trace["v"]*trace["yaw"])[geometric], fixed["target_ay"][geometric],
                                 trace["can_torque"][geometric], dt_s=.05, measured_yaw=trace["yaw"][geometric]) if len(geometric) else None
  outcome, motion = _stopping_outcome(window, report, trace, supported) if stopping else (None, None)
  if stopping and reason is None and (outcome is None or motion["reason"]):
    reason = "incomplete_supported_native_approach_stop_hold"
  result = {"window_id": window["id"], "parameters": p, "native_long_parameters": selected, "trace": trace, "event_attempts": attempts,
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
  if stopping:
    result.update(stopping_result=outcome, hold_motion=motion, geometry_prefix_mask=fixed["geometry"].copy(),
      unknown_geometry_macro_indices=np.arange(0, len(t), MICRO)[~fixed["geometry"][::MICRO]].tolist(),
      stop_request_index=fixed["request_index"], categorical_plant_kind=categorical_model["kind"],
      categorical_plant_model_sha256=categorical_model["model_sha256"],
      independent_initial_standstill_witness_ns=envelope["initial"]["hardware_standstill_t_ns"])
    result["selection_rejections"].extend(["contact_and_hold_unvalidated", "independent_categorical_state_dynamics_unvalidated",
      "frozen_OEM_cruise_standstill_unverified"])
    if approach_stop_model is None:
      result["low_speed_domain_contract"] = "initialmacro v<=3m/s only; no above-domain categorical transition"
      result["selection_rejections"].append("high_speed_to_low_speed_categorical_transition_unimplemented")
    else:
      result["categorical_domain_contract"] = "ONE source-bound approach model across retained TRAIN speed/cell domains; no3m/s switch or fallback"
      result["selection_rejections"].append("fullrange_categorical_contact_response_unvalidated")
    result["limitations"].extend(["2m/s aligned approach and0.5s approach/hold requirements are declared research support, not native settings",
      "Initial category is latest valid raw CS before earliest cold-prefix capture; held prefix state remains unvalidated",
      "Published request features are independent of earlier consumed/applied CC and held software reports",
      "Native lateral speed gate and independent hardware category remain separate; inactive native reset preserves its source semantics",
      "Post-request recorded OEM cruise standstill is an exogenous unvalidated flag, not predicted hardware standstill"])
    if outcome is None or outcome.get("target_feasible") is not True:
      result["selection_rejections"].append("model_stopping_target_feasibility_unverified")
    if (~fixed["geometry"]).any():
      result["selection_rejections"].append("full_horizon_path_reference_unknown")
  return result


def _stopping_outcome(window, report, trace, supported):
  """Supported causal states only; model target is never stopline truth."""
  motion = {"supported_zero_speed_s": 0., "reason": "supported_native_stop_hold_missing"}
  requested = np.flatnonzero(trace["native_state"][:supported] == 2)
  if not len(requested):
    return None, motion
  origin = int(requested[0])
  compute = np.asarray(trace.get("native_compute_t"))
  feedback_v, feedback_a = (np.asarray(trace.get(k)) for k in ("native_feedback_v", "native_feedback_a"))
  grid = np.asarray(trace["t"][:supported])
  if (any(a.ndim != 1 or len(a) < supported or not np.isfinite(a[:supported]).all() for a in (compute, feedback_v, feedback_a))
      or (np.diff(compute[:supported]) <= 0).any() or origin < 1
      or not grid[0] <= compute[origin] <= grid[origin] or compute[origin] <= grid[origin-1]):
    motion["reason"] = "native_stop_event_outside_covered_simulated_grid"
    return None, motion
  request_t = float(compute[origin])
  moving = (feedback_v[:origin] >= 2.) & (trace["native_state"][:origin] == 1)
  if (request_t-compute[0] < .5-1e-9
      or not any(moving[i:i+MICRO].all() for i in range(0, max(0, len(moving)-MICRO+1), MICRO))
      or not (trace["native_state"][origin:supported] == 2).all()):
    motion["reason"] = "genuine_aligned_moving_approach_or_continuous_native_stop_missing"
    return None, motion
  stationary = (trace["v"][:supported] == 0) & (trace["hardware_standstill"][:supported] == 1)
  zero = np.flatnonzero(stationary)
  if not len(zero) or not stationary[zero[0]:].all():
    return None, motion
  first = int(zero[0])
  if grid[first] < request_t:
    motion["reason"] = "stationary_precedes_native_stop_request"
    return None, motion
  duration = float(trace["t"][supported-1]-trace["t"][first])
  motion = {"supported_zero_speed_s": duration, "start_index": first, "end_index": supported-1,
    "angle_excursion_deg": float(np.max(np.abs(trace["angle"][first:supported]-trace["angle"][first]))),
    "yaw_abs_max_rad_s": float(np.max(np.abs(trace["yaw"][first:supported]))),
    "vendor_rate_max_deg_s": float(np.max(trace["vendor_rate"][first:supported])),
    "reason": None if duration >= .5-1e-9 else "insufficient_supported_zero_speed_hold"}
  # The covered prior node's state is held through this event. At a50ms
  # boundary, compute precedes the boundary and must not see its new state.
  prior = int(np.searchsorted(grid, request_t, side="right")-1)
  physical_v = float(trace["v"][prior])
  request_distance = float(trace["distance"][prior]+physical_v*(request_t-grid[prior]))
  if not np.isfinite([physical_v, request_distance]).all() or physical_v < 0 or request_distance < 0:
    motion["reason"] = "native_stop_event_simulated_distance_unavailable"
    return None, motion
  target_trace = {**trace, "t": trace["t"].copy(), "distance": trace["distance"].copy()}
  target_trace["t"][origin], target_trace["distance"][origin] = request_t, request_distance
  target = long.model_stopping_target(window, target_trace, origin, report["split"]["train"])
  distance = float(trace["distance"][first]-request_distance)
  return {"distance_from_request_m": distance, "time_from_request_s": float(grid[first]-request_t),
    "native_stop_compute_t": request_t, "stop_request_node_t": float(grid[origin]), "stationary_start_node_t": float(grid[first]),
    "approach_start_compute_t": float(compute[0]), "approach_elapsed_to_request_s": float(request_t-compute[0]),
    "simulated_distance_at_request_m": request_distance,
    "native_feedback_speed_at_stop_request_mps": float(feedback_v[origin]),
    "native_feedback_acceleration_at_stop_request_mps2": float(feedback_a[origin]),
    "simulated_physical_speed_at_request_mps": physical_v,
    "speed_at_native_stop_request_mps": float(feedback_v[origin]), "model_target": target,
    "target_feasible": target is not None and target["lower_m"] <= distance <= target["upper_m"],
    "native_stop_origin_index": origin,
    "clock_semantics": "Actual native compute event; conditional ZOH-integrated simulated distance; conservative supported-node stop/hold interval",
    "semantics": "source-predicted conditional standstill; no contact or stopline ground truth"}, motion


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
    geometric = nodes[np.isfinite(trace["fixed_speed"][nodes]) & np.isfinite(trace["fixed_path"][nodes]).all(axis=1)]
    speeds.extend(np.abs(trace["v"][geometric]-trace["fixed_speed"][geometric]).tolist())
    paths.extend(np.linalg.norm(trace["path"][geometric]-trace["fixed_path"][geometric], axis=1).tolist())
  return {"missing_windows": missing, "common_window_ids": list(common), "turns": turn_metrics,
          "long": long.summarize(scores, common) if common and not missing and len(scores) == len(common) else None,
          "fixed_reference_by_window": {key: results[key].get("fixed_reference_sha256") for key in common if key in results},
          "speed_p95": float(np.percentile(speeds, 95)) if speeds else None, "path_p95": float(np.percentile(paths, 95)) if paths else None}


def _stop_summary(results, declared):
  missing, errors = [], []
  for key in declared:
    row = results.get(key, {})
    target = row.get("stopping_result", {}).get("model_target") if row.get("stopping_result") else None
    if row.get("status") != "complete" or target is None:
      missing.append(key)
    else:
      errors.append(abs(row["stopping_result"]["distance_from_request_m"]-target["distance_from_request_m"]))
  return {"declared_stop_window_ids": list(declared), "missing_stop_windows": missing,
    "model_stop_distance_mae_m": float(np.mean(errors)) if errors and not missing else None,
    "hold_motion_by_window": {key: results.get(key, {}).get("hold_motion") for key in declared},
    "unknown_geometry_by_window": {key: results.get(key, {}).get("unknown_geometry_macro_indices") for key in declared}}


def _stop_constraints(results, baseline, declared):
  reasons = [] if len(declared) >= 2 else ["insufficient_completed_native_stopping_windows"]
  for key in declared:
    before, after = baseline.get(key, {}), results.get(key, {})
    old, new = before.get("stopping_result"), after.get("stopping_result")
    if before.get("status") != "complete" or after.get("status") != "complete" or old is None or new is None:
      reasons.append("missing_supported_stop_common_window")
      continue
    a, b = old.get("model_target"), new.get("model_target")
    if a is None or b is None or not new.get("target_feasible"):
      reasons.append("model_stopping_target_feasibility_unverified")
    elif any(a.get(k) != b.get(k) for k in ("request_t", "capture_t", "path_sha256", "uncertainty_source_sha256")):
      reasons.append("stopping_reference_anchor_mismatch")
    elif new["distance_from_request_m"] > old["distance_from_request_m"]+.1 or new["time_from_request_s"] > old["time_from_request_s"]+.1:
      reasons.append("native_stopping_regression")
    before_hold, after_hold = before.get("hold_motion"), after.get("hold_motion")
    if not before_hold or not after_hold or before_hold["reason"] or after_hold["reason"] or after_hold["supported_zero_speed_s"] < .5:
      reasons.append("missing_supported_zero_speed_hold")
    elif (after_hold["angle_excursion_deg"] > before_hold["angle_excursion_deg"]*1.02+.01
          or after_hold["yaw_abs_max_rad_s"] > before_hold["yaw_abs_max_rad_s"]*1.02+.001):
      reasons.append("hold_motion_regression")
  return reasons


def validation_search(windows, report, envelopes, model, lr, lat, out, proofs, *, grid, lateral_grid,
                      low_speed_model=None, approach_stop_model=None):
  if low_speed_model is not None and approach_stop_model is not None:
    raise TwoWayError("low_speed_model and approach_stop_model are mutually exclusive")
  if not windows or len({w["id"] for w in windows}) != len(windows) or not isinstance(grid, (list, tuple)) or not 0 < len(grid) <= 64:
    raise TwoWayError("unique validation windows and declared finite1..64 grid required")
  common, rows = sorted(w["id"] for w in windows), []
  stop_ids = sorted(w["id"] for w in windows if w.get("regime") == "stopping")
  if not stop_ids and (low_speed_model is not None or approach_stop_model is not None):
    raise TwoWayError("categorical stopping model supplied to moving-only windows")
  if stop_ids:
    try:
      identities = {(shared.static_identity(w["cp"], w["sp"]),
                     shared.policy_identity(envelopes[w["id"]]["params"]["values"])) for w in windows}
      if len(identities) != 1 or any(envelopes[w["id"]].get("policy_identity_sha256") != next(iter(identities))[1] for w in windows):
        raise TwoWayError("mixed windows require identical full static and typed policy identities")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
      raise TwoWayError("mixed static/policy population refused: "+str(error)) from error
  for parameters in (None, *grid):
    row = {"parameters": deepcopy(parameters), "results": {}, "failures": [], "selection_rejections": []}
    for window in windows:
      key = window["id"]
      try:
        result = evaluate_two_way(window, report, envelopes[key], model, lr, lat, out, proofs[key],
                                  parameters=parameters, lateral_grid=lateral_grid,
                                  low_speed_model=low_speed_model if window.get("regime") == "stopping" else None,
                                  approach_stop_model=approach_stop_model if window.get("regime") == "stopping" else None)
      except Exception as error:
        result = {"window_id": key, "status": "rejected", "reason": str(error), "error_type": type(error).__name__,
                  "traceback": traceback.format_exc()}
      row["results"][key] = result
      row["selection_rejections"].extend(result.get("selection_rejections", []))
      if result["status"] != "complete":
        row["failures"].append({"window_id": key, "status": result["status"], "reason": result["reason"]})
    row["metrics"] = summarize(row["results"], common)
    if stop_ids:
      row["metrics"].update(_stop_summary(row["results"], stop_ids))
    before = rows[0] if rows else row
    row["selection_rejections"].extend(shared.constraints(row["metrics"], before["metrics"]))
    if stop_ids:
      row["selection_rejections"].extend(_stop_constraints(row["results"], before["results"], stop_ids))
    row["joint_target_objective"] = (stop._objective(row["metrics"], before["metrics"]) if stop_ids else
                                     shared._objective(row["metrics"], before["metrics"]))
    row["selection_rejections"].extend(["no_validated_two_way_stop_hold_or_coherence", "diagnostic_only",
                                        "joint_plant_predictive_validation_unavailable"])
    if len(common) < long.research.MIN_WINDOWS:
      row["selection_rejections"].append("insufficient_fixed_common_population")
    row["selection_rejections"] = sorted(set(row["selection_rejections"]))
    row["common_window_ids"], row["coverage"] = list(common), (len(common)-len(row["metrics"]["missing_windows"]))/len(common)
    rows.append(row)
  # Tracking against frozen targets cannot rank an unvalidated response plant.
  # Preserve every objective/trace until an authenticated predictive gate exists.
  return {"status": "insufficient_evidence", "baseline": rows[0], "rows": rows[1:], "common_window_ids": common,
          "ranking_status": "unavailable", "ranking_rejections": ["joint_plant_predictive_validation_unavailable"],
          "attempted_windows": len(common), "selection": None, "complete_controller_candidate": None, "production_enabled": False}
