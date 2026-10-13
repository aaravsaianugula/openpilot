"""Conditional stop/hold code fixtures; no physical contact or vehicle proof."""
from copy import deepcopy

import numpy as np
import pytest

import controller_native_stop_hold as stop
from test_controller_native_joint import example as joint_example, GRID
from test_controller_native_search import uncertainty


def hash_witness(witness):
  witness["sha256"] = stop.long.value_hash({k: v for k, v in witness.items() if k != "sha256"})


def example(monkeypatch):
  args = list(joint_example(monkeypatch))
  window, report, env, moving, lr, lat = args[:6]
  window.update(regime="stopping", initial_v=3., initial_a=-3.5)
  t = 6.7+np.arange(200)*.01
  tape = window["tape"]
  for key, values in list(tape.items()):
    if not key.startswith("rdf_"):
      tape[key] = np.resize(values, len(t))
    elif np.asarray(values).ndim and len(values) == 170:
      tape[key] = np.take(values, np.minimum(np.arange(220), 169), axis=0)
  tape["t"] = t
  for name in ("carState", "carControl", "lat_input"):
    tape[name+"_t"] = t-.001
  raw_cs_ns = np.rint(tape["carState_t"]*1e9).astype(np.uint64)
  # Construct the recorded tape with the frozen raw-log producer expression.
  tape["carState_t"] = np.asarray([int(value)*1e-9 for value in raw_cs_ns])
  recorded_speed = np.maximum(0., 3.-3.5*(t-t[0]))
  env["raw_car_state_witness"] = {"kind": "raw_car_state_eligibility_v1", "route": window["route"],
    "source_log_sha256": window["source_log_sha256"], "t_ns": raw_cs_ns, "event_valid": np.ones(len(t), bool),
    "can_valid": np.ones(len(t), bool), "standstill": recorded_speed == 0, "v_ego": recorded_speed}
  hash_witness(env["raw_car_state_witness"])
  capture, h = np.arange(220)*.05, tape["rdf_horizon"]
  tape.update(rdf_frame_id=np.arange(220), rdf_narrow_t=capture, rdf_input_t=capture+.001, rdf_output_ready_t=capture+.02,
              rdf_scored_mask=np.arange(220) >= 132, rdf_should_stop=capture >= 7.2, rdf_desired_acceleration=np.full(220, -3.5))
  plan = np.zeros((220, 33, 15))
  for i, captured in enumerate(capture):
    v = max(0., 3.-3.5*(captured-t[0]))
    phase = np.minimum(h, v/3.5)
    plan[i, :, 0] = v*phase-1.75*phase**2
    plan[i, :, 3] = np.maximum(0., v-3.5*h)
    plan[i, :, 6] = np.where(h < v/3.5, -3.5, 0.)
    plan[i, :, 11] = .002*plan[i, :, 0]
    plan[i, :, 14] = .002*plan[i, :, 3]
  tape.update(rdf_plan=plan, rdf_position=plan[:, :, :3].copy(), rdf_acceleration=plan[:, :, 6:9].copy())
  for key in ("action_t", "packet_action_t"):
    window["model_inputs"][key] = tape["rdf_"+key].copy()
  witness = env["lateral_delay_witness"]
  witness.update(frame_id=np.arange(220), capture_t=capture, delay=np.full(220, .1), source_publication_t=capture-.01,
                 source_valid=np.ones(220, bool), source_kind=np.full(220, "lateralDelay"),
                 source_log_sha256=np.full(220, window["source_log_sha256"]), packet_lat_delay=np.full(220, .1),
                 packet_lateral_control_params=np.tile([3., .1], (220, 1)))
  model = report["axes"]["long"]["model"]
  model.update(table={stop.long.tune.band_of(v): {"command": [-3.5, 0., 2.], "response": [-3.5, 0., 2.]} for v in range(31)},
               delay_s=0., stopping_identified=True,
               low_speed_validation={"used_windows": 20, "coverage": .8, "standstill_hold_identified": True})
  window["reset_evidence"] = {"t": t[0]-.01, "valid": True, "long_active": False, "native_state": 0, "transmitted_accel": 0.}
  window["stop_target_uncertainty"] = uncertainty(window)
  hold = deepcopy(moving)
  hold.update(kind=stop.hold.KIND, features=list(stop.hold.FEATURES), native_policy_source_sha256=stop.policy_source_hash(),
              contact_hold_validated=False, nonreversing_contact_validated=False,
              feature_min=[-500.]*24, feature_max=[500.]*24, edges=[[] for _ in range(24)])
  hold["patterns"] = [("1"*n+"0"*(5-n))*2 for n in range(5)]
  hold["cells"] = {pattern+"|"+":".join(["0"]*24): {"supported": True, "samples": 100, "routes": ["route-t", "route-u"],
                    "feature_min": [-500.]*24, "feature_max": [500.]*24, "joint_delta_median": [.01, .001, 0.]}
                   for pattern in hold["patterns"]}
  hold["supported_cells"] = len(hold["cells"])
  hold["model_sha256"] = stop.hold.identity_hash({k: v for k, v in hold.items() if k != "model_sha256"})
  assumptions = {"mode": "conditional_existing_long_nonreversing_contact", "validated": False,
                 "source_log_sha256": window["source_log_sha256"], "long_model_sha256": stop.long.value_hash(model),
                 "hold_model_sha256": stop.long.value_hash(hold), "cp_sha256": stop.long.value_hash(window["cp"]),
                 "sp_sha256": stop.long.value_hash(window["sp"])}
  calls = []
  def predict(model, state, speed, command, active, request, roll):
    calls.append((list(speed), list(command), list(active), list(request)))
    return {"angle": state["angle"]+.01, "yaw": state["yaw"]+.001, "vendor_rate": 0.}
  monkeypatch.setattr(stop.hold, "predict_macro", predict)
  return args, hold, assumptions, calls


def evaluate(fixture, **kwargs):
  args, hold, assumptions, _ = fixture
  window, report, env, moving, lr, lat = args[:6]
  lp = {"window_sha256": stop.long.value_hash(window), "plant_report_sha256": stop.long.value_hash(report),
        "rdf_sha256": stop.long.value_hash(stop.long.rdf_from_tape(window["tape"])), "model_inputs_sha256": stop.long.value_hash(window["model_inputs"]),
        "source_log_sha256": window["source_log_sha256"], "plant_opendbc_gitlink": window["opendbc_gitlink"],
        "native_sources": lr.source_hashes.copy(), "initialization": "observed_off"}
  bound = stop.joint.bind_proof(window, report, lp, env, moving, lr, lat)
  proof = stop.bind_proof(bound, hold, assumptions)
  return stop.evaluate_stop_hold(window, report, lp, env, moving, hold, lr, lat, bound, assumptions, proof,
                                 lateral_grid=GRID, **kwargs)


def test_one_continuous_handle_transitions_to_inactive_zero_and_supported_hold(monkeypatch):
  fixture = example(monkeypatch)
  original = fixture[0][5].create
  handles = []
  def create(*a, **kw):
    handles.append(original(*a, **kw))
    return handles[-1]
  monkeypatch.setattr(fixture[0][5], "create", create)
  result = evaluate(fixture)
  assert result["status"] == "complete" and result["used_steps"] == 200 and len(handles) == 1
  assert len(fixture[0][-1])+len(fixture[-1]) == 40
  trace = result["trace"]
  inactive = ~trace["effective_active"]
  assert inactive.any() and not trace["request"][inactive].any() and (trace["can_torque"][inactive] == 0).all()
  assert result["hold_motion"]["supported_zero_speed_s"] >= .5
  assert np.ptp(trace["angle"][trace["v"] == 0]) > 0 and np.ptp(trace["yaw"][trace["v"] == 0]) > 0
  assert result["complete_controller_candidate"] is None and not result["production_enabled"]
  assert "contact_and_hold_unvalidated" in result["selection_rejections"]


def test_missing_curvature_does_not_block_raw_plan_or_turn_into_zero_score(monkeypatch):
  fixture = example(monkeypatch)
  result = evaluate(fixture)
  assert result["status"] == "complete" and result["raw_plan_available"].all()
  mask = result["geometry_prefix_mask"]
  assert mask.any() and (~mask).any() and np.isnan(result["trace"]["physical_curvature"][~mask]).all()
  assert result["unknown_geometry_macro_indices"] and len(result["macro_indices"]) == 40
  assert len(result["turns"]["reference_direction"]) < 40
  assert np.isnan(result["trace"]["fixed_path"][~mask]).all()


def test_model_speed_jitter_cannot_reenter_turn_geometry_after_unknown_prefix(monkeypatch):
  fixture = example(monkeypatch)
  tape = fixture[0][0]["tape"]
  tape["rdf_plan"][170:, :, 3], tape["rdf_plan"][170:, :, 14] = 3., -.03
  result = evaluate(fixture)
  first_missing = np.flatnonzero(~result["geometry_prefix_mask"])[0]
  assert not result["geometry_prefix_mask"][first_missing:].any()
  assert "left" not in result["turns"]["reference_direction"]


def test_future_measured_motion_cannot_change_stop_native_feedback(monkeypatch):
  fixture = example(monkeypatch)
  first = evaluate(fixture)
  fixture[0][0]["tape"].update(v=np.full(200, 90.), a=np.full(200, 9.), yaw=np.full(200, -10.), steering_angle=np.full(200, 999.))
  second = evaluate(fixture)
  for key in ("v", "a", "angle", "yaw", "can_torque", "physical_curvature"):
    np.testing.assert_array_equal(first["trace"][key], second["trace"][key])


@pytest.mark.parametrize("fault", ["all_true", "release", "margin", "low_speed", "contact", "hold_source"])
def test_missing_stop_contracts_refuse_before_any_native_factory(monkeypatch, fault):
  fixture = example(monkeypatch)
  window, report = fixture[0][:2]
  if fault == "all_true":
    window["tape"]["rdf_should_stop"][:] = True
  elif fault == "release":
    window["tape"]["rdf_should_stop"][160:] = False
  elif fault == "margin":
    window["stop_target_uncertainty"] = {}
  elif fault == "low_speed":
    report["axes"]["long"]["model"]["low_speed_validation"]["standstill_hold_identified"] = False
  elif fault == "contact":
    fixture[2]["validated"] = True
  else:
    fixture[1]["source_eligible"] = False
  monkeypatch.setattr(stop.long, "evaluate_trial", lambda *a, **k: pytest.fail("Long native before refusal"))
  monkeypatch.setattr(fixture[0][5], "create", lambda *a, **k: pytest.fail("Lat native before refusal"))
  with pytest.raises(stop.StopHoldError):
    evaluate(fixture)


def test_active_request_cut_is_unsupported_but_inactive_zero_edge_is_not_rate_limited(monkeypatch):
  fixture = example(monkeypatch)
  original = fixture[0][5].create
  def create(*a, **kw):
    h = original(*a, **kw)
    step = h.step
    def changed(**values):
      result = step(**values)
      if values["t"] == fixture[0][0]["tape"]["t"][3]:
        result["steer_request"] = False
      return result
    h.step = changed
    return h
  monkeypatch.setattr(fixture[0][5], "create", create)
  result = evaluate(fixture)
  assert result["used_steps"] == 3 and result["reason"] == "active_steer_request_false"
  assert not fixture[-1]


def test_hold_support_failure_and_driver_censor_retain_prefix(monkeypatch):
  fixture = example(monkeypatch)
  monkeypatch.setattr(stop.hold, "predict_macro", lambda *a: stop.hold.Unsupported("pattern_not_supported"))
  result = evaluate(fixture)
  assert result["status"] == "unsupported" and result["reason"] == "hold_unsupported:pattern_not_supported"
  fixture[0][0]["tape"]["brake_pressed"][7] = True
  result = evaluate(fixture)
  assert result["used_steps"] == 7 and result["censor_t"] == fixture[0][0]["tape"]["t"][7]


def test_finite_mixed_search_keeps_failed_rows_and_fixed_common_windows(monkeypatch):
  fixture = example(monkeypatch)
  args, hold, assumptions, _ = fixture
  w, report, env, moving, lr, lat = args[:6]
  from test_controller_native_search import proof
  m_args = joint_example(monkeypatch)
  m_args[0]["id"] = "moving-window"
  windows = [w, m_args[0]]
  envelopes = {w["id"]: env, m_args[0]["id"]: m_args[2]}
  proofs = {item["id"]: proof(item, report, lr) for item in windows}
  bounds = {item["id"]: stop.joint.bind_proof(item, report, proofs[item["id"]], envelopes[item["id"]], moving, lr, lat) for item in windows}
  bound = bounds[w["id"]]
  stop_proof = stop.bind_proof(bound, hold, assumptions)
  calls, run, create = [], stop.long.evaluate_trial, lat.create
  def count(*a, **kw):
    calls.append(1)
    return run(*a, **kw)
  def construct(*a, **kw):
    if kw["candidate"] and kw["candidate"]["factor"] == 3.:
      raise RuntimeError("retained lateral candidate construction failure")
    return create(*a, **kw)
  monkeypatch.setattr(stop.long, "evaluate_trial", count)
  monkeypatch.setattr(lat, "create", construct)
  grid = [{"long": None, "lateral": {"factor": f, "friction": .1}} for f in (2., 3.)]
  result = stop.validation_search(windows, report, proofs, envelopes, moving, hold, lr, lat,
                                  bounds, {w["id"]: assumptions}, {w["id"]: stop_proof}, grid=grid, lateral_grid=GRID)
  common = sorted([w["id"], m_args[0]["id"]])
  assert len(result["rows"]) == 2 and result["common_window_ids"] == common and len(calls) == 2
  assert all(row["common_window_ids"] == common for row in [result["baseline"], *result["rows"]])
  assert result["rows"][1]["coverage"] == 0 and len(result["rows"][1]["failures"]) == 2
  assert result["complete_controller_candidate"] is None and result["selection"] is None


def test_only_aligned_completed_moving_macro_establishes_genuine_approach(monkeypatch):
  fixture = example(monkeypatch)
  result = evaluate(fixture)
  forged = deepcopy(result["long_result"])
  origin = result["stop_request_index"]
  forged["trace"]["v"][:origin] = 1.
  forged["trace"]["v"][3:8] = 2.1  # Five supported ticks, split over macro0 and macro1.
  monkeypatch.setattr(stop.joint.LongCache, "evaluate", lambda *a, **k: (forged, {"key": "fixture", "result_sha256": stop.long.value_hash(forged)}))
  monkeypatch.setattr(fixture[0][5], "create", lambda *a, **k: pytest.fail("Lat factory before aligned-macro refusal"))
  with pytest.raises(stop.StopHoldError, match="moving approach"):
    evaluate(fixture)


def test_inactive_zero_can_edge_does_not_receive_active_slew_constraint(monkeypatch):
  fixture = example(monkeypatch)
  fixture[0][2]["initialization"]["last_transmitted_counts"] = 409
  fixture[0][2]["history"]["command"][:] = 409
  result = evaluate(fixture)
  first = np.flatnonzero(~result["trace"]["effective_active"])[0]
  assert result["status"] == "complete" and result["trace"]["can_torque"][first] == 0
  assert abs(result["trace"]["can_torque"][first-1]) > 7


def test_cached_long_failure_is_retained_without_summary_crash(monkeypatch):
  fixture = example(monkeypatch)
  args, hold, assumptions, _ = fixture
  w, report, env, moving, lr, lat = args[:6]
  from test_controller_native_search import proof
  lp = proof(w, report, lr)
  bound = stop.joint.bind_proof(w, report, lp, env, moving, lr, lat)
  stop_proof = stop.bind_proof(bound, hold, assumptions)
  def fail(*a, **k):
    raise RuntimeError("retained native long constructor failure")
  monkeypatch.setattr(stop.long, "evaluate_trial", fail)
  result = stop.validation_search([w], report, {w["id"]: lp}, {w["id"]: env}, moving, hold, lr, lat,
                                  {w["id"]: bound}, {w["id"]: assumptions}, {w["id"]: stop_proof},
                                  grid=[{"long": None, "lateral": {"factor": 2., "friction": .1}}], lateral_grid=GRID)
  assert result["baseline"]["failures"] and result["rows"][0]["failures"]
  assert result["baseline"]["results"][w["id"]]["long_result"]["error_type"] == "RuntimeError"


def test_simulated_release_after_zero_is_censored_without_forcing_hold(monkeypatch):
  fixture = example(monkeypatch)
  original = evaluate(fixture)
  changed = deepcopy(original["long_result"])
  index = int(np.flatnonzero(changed["trace"]["v"] == 0)[0])+5
  changed["trace"]["v"][index] = .01
  monkeypatch.setattr(stop.joint.LongCache, "evaluate", lambda *a, **k: (changed, {"key": "fixture", "result_sha256": stop.long.value_hash(changed)}))
  result = evaluate(fixture)
  assert result["reason"] == "release_or_restart_unsupported" and result["used_steps"] == index
  assert np.isnan(result["trace"]["angle"][index:]).all()


def test_short_zero_speed_suffix_does_not_establish_half_second_hold(monkeypatch):
  fixture = example(monkeypatch)
  tape = fixture[0][0]["tape"]
  for key, value in list(tape.items()):
    if not key.startswith("rdf_"):
      tape[key] = value[:100]
  witness = fixture[0][2]["raw_car_state_witness"]
  for key in ("t_ns", "event_valid", "can_valid", "standstill", "v_ego"):
    witness[key] = witness[key][:100]
  hash_witness(witness)
  result = evaluate(fixture)
  assert result["hold_motion"]["supported_zero_speed_s"] < .5
  assert "insufficient_supported_zero_speed_hold" in result["selection_rejections"]


def test_unsupported_final_macro_cannot_extend_supported_hold_or_turn_population(monkeypatch):
  fixture = example(monkeypatch)
  complete = evaluate(fixture)
  count, predict = 0, stop.hold.predict_macro
  expected_calls = len(fixture[-1])
  def fail_last(*args):
    nonlocal count
    count += 1
    return stop.hold.Unsupported("last_macro") if count == expected_calls else predict(*args)
  monkeypatch.setattr(stop.hold, "predict_macro", fail_last)
  result = evaluate(fixture)
  assert result["used_steps"] == 200 and result["status"] == "unsupported"
  assert result["supported_steps"] == 195 and result["macro_indices"][-1] == 190
  assert result["hold_motion"]["end_index"] == 194
  assert result["hold_motion"]["supported_zero_speed_s"] == pytest.approx(complete["hold_motion"]["supported_zero_speed_s"]-.05)


@pytest.mark.parametrize("fault", ["missing", "standstill_above_low_speed", "interposed_invalid", "invalid_event", "invalid_can",
                                   "stale", "future", "clock_mismatch", "wrong_source", "wrong_route", "duplicate_clock", "gap",
                                   "negative_speed", "nonfinite_speed", "nonboolean_standstill", "non_uint64_clock", "hash"])
def test_raw_car_state_eligibility_refuses_before_either_factory_or_cache(monkeypatch, fault):
  fixture = example(monkeypatch)
  env, tape = fixture[0][2], fixture[0][0]["tape"]
  witness = env["raw_car_state_witness"]
  if fault == "missing":
    del env["raw_car_state_witness"]
  elif fault == "standstill_above_low_speed":
    witness["standstill"][2] = True
  elif fault == "interposed_invalid":
    index = 7
    for key in ("t_ns", "event_valid", "can_valid", "standstill", "v_ego"):
      value = witness[key][index-1]
      if key == "t_ns":
        value += np.uint64(5_000_000)
      if key == "event_valid":
        value = False
      witness[key] = np.insert(witness[key], index, value)
  elif fault in ("invalid_event", "invalid_can"):
    witness["event_valid" if fault == "invalid_event" else "can_valid"][5] = False
  elif fault in ("stale", "future"):
    shift = -20_000_000 if fault == "stale" else 20_000_000
    witness["t_ns"] = (witness["t_ns"].astype(np.int64)+shift).astype(np.uint64)
  elif fault == "clock_mismatch":
    witness["t_ns"][3] += np.uint64(1000)
  elif fault == "wrong_source":
    witness["source_log_sha256"] = "f"*64
  elif fault == "wrong_route":
    witness["route"] = "other-route"
  elif fault == "duplicate_clock":
    witness["t_ns"][3] = witness["t_ns"][2]
  elif fault == "gap":
    for key in ("t_ns", "event_valid", "can_valid", "standstill", "v_ego"):
      witness[key] = np.delete(witness[key], 3)
    tape["carState_t"][3] = tape["carState_t"][2]
  elif fault == "negative_speed":
    witness["v_ego"][5] = -.01
  elif fault == "nonfinite_speed":
    witness["v_ego"][5] = np.nan
  elif fault == "nonboolean_standstill":
    witness["standstill"] = witness["standstill"].astype(np.uint8)
  elif fault == "non_uint64_clock":
    witness["t_ns"] = witness["t_ns"].astype(np.float64)
  if fault != "missing":
    hash_witness(witness)
  if fault == "hash":
    witness["sha256"] = "0"*64
  monkeypatch.setattr(stop.joint.LongCache, "evaluate", lambda *a, **k: pytest.fail("cache before raw-CS refusal"))
  monkeypatch.setattr(fixture[0][5], "create", lambda *a, **k: pytest.fail("Lat factory before raw-CS refusal"))
  with pytest.raises(stop.StopHoldError, match="raw carState"):
    evaluate(fixture)


def test_raw_car_state_values_are_eligibility_only_and_post_window_invalidity_is_not_consumed(monkeypatch):
  fixture = example(monkeypatch)
  before = evaluate(fixture)
  witness = fixture[0][2]["raw_car_state_witness"]
  witness["v_ego"][:] = .2
  witness["standstill"][:] = False
  for key in ("t_ns", "event_valid", "can_valid", "standstill", "v_ego"):
    value = witness[key][-1]
    if key == "t_ns":
      value += np.uint64(20_000_000)
    if key in ("event_valid", "can_valid"):
      value = False
    if key == "v_ego":
      value = np.nan
    witness[key] = np.append(witness[key], value)
  hash_witness(witness)
  after = evaluate(fixture)
  for key in ("v", "a", "angle", "yaw", "can_torque"):
    np.testing.assert_array_equal(before["trace"][key], after["trace"][key])


def test_valid_raw_producer_timestamp_is_not_recanonicalized_to_division(monkeypatch):
  fixture = example(monkeypatch)
  raw_ns = fixture[0][2]["raw_car_state_witness"]["t_ns"]
  recorded = fixture[0][0]["tape"]["carState_t"].copy()
  assert int(raw_ns[0])*1e-9 == recorded[0]
  assert float(raw_ns[0])/1e9 != recorded[0]  # A deterministic one-ULP counterexample.
  result = evaluate(fixture)
  assert result["status"] == "complete"
  np.testing.assert_array_equal(recorded, fixture[0][0]["tape"]["carState_t"])


def test_original_integer_clock_asof_includes_exact_publication_and_excludes_future():
  ns = np.asarray([6_689_000_000, 6_699_000_000, 6_699_000_001], dtype=np.uint64)
  published = np.asarray([int(value)*1e-9 for value in ns])
  ticks = np.asarray([np.nextafter(published[1], -np.inf), published[1], np.nextafter(published[1], np.inf)])
  window = {"route": "fixture-route", "source_log_sha256": "a"*64,
            "tape": {"t": ticks, "carState_t": np.asarray([published[0], published[1], published[1]])}}
  witness = {"kind": "raw_car_state_eligibility_v1", "route": window["route"], "source_log_sha256": window["source_log_sha256"],
             "t_ns": ns, "event_valid": np.asarray([True, True, False]), "can_valid": np.ones(3, bool),
             "standstill": np.zeros(3, bool), "v_ego": np.full(3, 2.)}
  hash_witness(witness)
  env = {"raw_car_state_witness": witness}
  stop._raw_car_state_eligibility(window, env)
  window["tape"]["t"] = np.asarray([published[2]])
  window["tape"]["carState_t"] = np.asarray([published[2]])
  with pytest.raises(stop.StopHoldError, match="invalid/interposed"):
    stop._raw_car_state_eligibility(window, env)


def test_actual_native_stop_hold_continuous_api_synthetic_inputs(monkeypatch):
  """Actual controller APIs only; synthetic vision/plant/support, not car performance."""
  import hashlib
  import inspect
  import json
  import os
  from pathlib import Path
  import subprocess

  required = ("NATIVE_LATERAL_SOURCE_ROOT", "NATIVE_SEARCH_SOURCE_ROOT", "NATIVE_JOINT_SOURCE_MANIFEST")
  if os.name != "posix" or not all(os.environ.get(key) for key in required):
    pytest.skip("Owner-serialized actual native POSIX/source environment required")
  repo = Path(os.environ[required[0]]).resolve()
  manifest_path = Path(os.environ[required[2]])
  if not repo.is_dir() or not manifest_path.is_file():
    pytest.skip("Actual native POSIX source/manifest unavailable")
  assert Path(os.environ[required[1]]).resolve() == repo
  manifest_bytes = manifest_path.read_bytes()
  assert hashlib.sha256(manifest_bytes).hexdigest() == "2f7c60555e8329ad6cf259ec9b87d6bac93e6ee97d3e2f948afd3f61bd5c02d2"
  manifest = json.loads(manifest_bytes)
  assert len(manifest["expected_native_sources"]) == 44
  assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip() == \
    "13aedc522518fe1b62651d9a8a320e931efc335c"
  loaded = stop.joint.lateral.load_runtime(repo, manifest["expected_native_sources"], manifest["expected_opendbc_gitlink"])
  sources = stop.long.source_hashes(repo)
  assert all(value == manifest["expected_native_sources"][key] for key, value in sources.items() if key in manifest["expected_native_sources"])
  lr = stop.long.load_native_runtime(repo, sources, stop.joint.lateral.OPENDBC_PIN)
  class CountedRuntime(stop.joint.lateral.Runtime):
    def create(self, *args, **kwargs):
      actual_handle = super().create(*args, **kwargs)
      handles.append(actual_handle)
      return actual_handle
  handles = []
  lat = CountedRuntime(loaded.bindings, loaded.sources, loaded.pin)
  # Only reuse construction of synthetic inputs; restore BOTH predictor patches
  # before evaluating. No factory or predictor is replaced in the actual trial.
  with monkeypatch.context() as fixture_patch:
    fixture = example(fixture_patch)
  args, hold, assumptions, _ = fixture
  window, report, env, moving = args[:4]
  from opendbc.car.structs import car
  from opendbc.car.hyundai.values import HyundaiFlags
  from openpilot.cereal import custom
  cp = car.CarParams.new_message(carFingerprint="HYUNDAI_ELANTRA_2024", brand="hyundai", mass=1360., wheelbase=2.72,
                                centerToFront=1.088, rotationalInertia=2500., steerRatio=13.5, steerRatioRear=0.,
                                tireStiffnessFront=80000., tireStiffnessRear=80000., steerActuatorDelay=.1,
                                steerLimitTimer=.4, minSteerSpeed=0., steerAtStandstill=False, steerControlType="torque",
                                flags=int(HyundaiFlags.RAISED_LIMITS), openpilotLongitudinalControl=True,
                                longitudinalActuatorDelay=.5, stopAccel=-2.)
  tuning = cp.lateralTuning.init("torque")
  tuning.latAccelFactor, tuning.latAccelOffset, tuning.friction = 2.5, 0., .1
  cp.longitudinalTuning.kiBP, cp.longitudinalTuning.kiV = [0., 30.], [0., 0.]
  firmware = cp.init("carFw", 1)[0]
  firmware.ecu, firmware.fwVersion = "eps", b"\x00\xff\x80"
  sp = custom.CarParamsSP.new_message(enableGasInterceptor=False)
  sp.neuralNetworkLateralControl.model.path = str(repo/stop.joint.lateral.ASSET)
  window.update(cp=cp.to_dict(), sp=sp.to_dict(), opendbc_gitlink=stop.joint.lateral.OPENDBC_PIN,
                fixture_scope="Synthetic raw plan/CS/plant/uncertainty/support identities; actual native API only")
  report["fixture_scope"] = window["fixture_scope"]
  env.update(cp_raw=np.frombuffer(cp.to_bytes(), np.uint8).copy(), sp_raw=np.frombuffer(sp.to_bytes(), np.uint8).copy(),
             opendbc_gitlink=stop.joint.lateral.OPENDBC_PIN, static_identity_sha256=stop.joint.static_identity(window["cp"], window["sp"]))
  tape, horizon = window["tape"], np.asarray(lat.bindings.horizon)
  tape["rdf_horizon"] = horizon
  plan = np.zeros_like(tape["rdf_plan"])
  for i, capture in enumerate(tape["rdf_narrow_t"]):
    speed = max(0., 3.-3.5*(capture-tape["t"][0]))
    phase = np.minimum(horizon, speed/3.5)
    plan[i, :, 0], plan[i, :, 3] = speed*phase-1.75*phase**2, np.maximum(0., speed-3.5*horizon)
    plan[i, :, 6] = np.where(horizon < speed/3.5, -3.5, 0.)
    plan[i, :, 11], plan[i, :, 14] = .002*plan[i, :, 0], .002*plan[i, :, 3]
  tape.update(rdf_plan=plan, rdf_position=plan[:, :, :3].copy(), rdf_acceleration=plan[:, :, 6:9].copy())
  for model, features, delta in ((moving, 14, [-.001, .0001, 0.]), (hold, 24, [.0002, .00001, 0.])):
    model.update(opendbc_gitlink=stop.joint.lateral.OPENDBC_PIN, static_identity_sha256=env["static_identity_sha256"],
                 edges=[[] for _ in range(features)], production_enabled=False, candidate=None,
                 fixture_scope="Constructed median cells/support labels only; no empirical fit or physical validation")
    keys = [":".join(["0"]*14)] if features == 14 else [pattern+"|"+":".join(["0"]*24) for pattern in model["patterns"]]
    model["cells"] = {key: {"supported": True, "samples": 100, "routes": report["split"]["train"],
                            "feature_min": [-500.]*features, "feature_max": [500.]*features, "joint_delta_median": delta} for key in keys}
    model["supported_cells"] = len(keys)
    model["model_sha256"] = stop.hold.identity_hash({key: value for key, value in model.items() if key != "model_sha256"})
  args[4:6] = [lr, lat]
  assumptions.update(cp_sha256=stop.long.value_hash(window["cp"]), sp_sha256=stop.long.value_hash(window["sp"]),
                     hold_model_sha256=stop.long.value_hash(hold))
  window["stop_target_uncertainty"] = uncertainty(window)  # Synthetic20-outcome fixture bound to the final full CP/SP.
  assert Path(inspect.getfile(stop.joint.plant.predict_macro)).resolve() == Path(stop.joint.plant.__file__).resolve()
  assert Path(inspect.getfile(stop.hold.predict_macro)).resolve() == Path(stop.hold.__file__).resolve()
  result = evaluate(fixture)
  assert lr.actual_native and lat.bindings.actual_native and result["long_result"]["evidence"] == "actual native class"
  assert result["status"] == "complete" and result["used_steps"] == result["supported_steps"] == 200
  assert len(result["macro_indices"]) == 40 and len(handles) == 1
  handle = handles[0]
  assert handle.last_t == tape["t"][-1] and handle.limits.STEER_MAX == 409
  assert Path(inspect.getfile(type(handle.controller))).resolve() == repo/stop.joint.lateral.V0_PATH
  assert Path(inspect.getfile(lr.controller_type)).resolve() == repo/stop.long.LONG_SOURCE
  trace, long_trace = result["trace"], result["long_result"]["trace"]
  np.testing.assert_array_equal(trace["v"], long_trace["v"])
  np.testing.assert_array_equal(trace["a"], long_trace["a"])
  origin = result["stop_request_index"]
  assert np.any(long_trace["native_state"][:origin] == 1) and (long_trace["native_state"][origin:] == 2).all()
  active, counts = trace["effective_active"], trace["can_torque"]
  assert active[0] and (~active).any() and not active[-1]
  assert not trace["request"][~active].any() and (counts[~active] == 0).all()
  assert np.max(np.abs(counts)) <= 409 and np.max(np.abs(np.diff(np.r_[0., counts])[active])) <= 7
  assert result["hold_motion"]["supported_zero_speed_s"] >= .5
  assert np.ptp(trace["angle"][trace["v"] == 0]) > 0 and np.ptp(trace["yaw"][trace["v"] == 0]) > 0
  assert window["cp"]["carFw"][0]["fwVersion"] == b"\x00\xff\x80"
  assert result["complete_controller_candidate"] is None and not result["production_enabled"] and not result["selection_eligible"]
  assert "contact_and_hold_unvalidated" in result["selection_rejections"]
  assert "simulated_carState_standstill_equals_zero_unvalidated" in result["selection_rejections"]
