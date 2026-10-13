"""Synthetic adapter tests; no native-runtime, identification or vehicle proof."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import controller_native_joint as joint
from test_controller_native_search import example as long_example, proof as long_proof
from test_controller_native_lateral_runtime import fixture_runtime, LP, params, GRID


def recorded_fixture_params():
  return {**params(), "BlinkerLateralReengageDelay": "0", "BlinkerMinLateralControlSpeed": "0",
          "BlinkerPauseLateralControl": "0", "IsMetric": "1"}


def example(monkeypatch):
  window, report, long_runtime = long_example()
  runtime, cp, _, _ = fixture_runtime()
  cp_dict = deepcopy(window["cp"])
  cp_dict.update(brand="hyundai", steerControlType="torque", minSteerSpeed=0., steerAtStandstill=False, steerActuatorDelay=.1,
                 lateralTuning={"torque": {"latAccelFactor": 2.5, "latAccelOffset": 0., "friction": .1}})
  window["cp"] = cp_dict
  def decode(c, s):
    selected = deepcopy(cp)
    selected.to_dict = lambda: json.loads(c)
    return selected, SimpleNamespace(to_dict=lambda: json.loads(s))
  horizon = window["tape"]["rdf_horizon"]
  runtime = joint.lateral.Runtime(replace(runtime.bindings, decode_static=decode, horizon=horizon),
                                  {"fixture.py": "f"*64}, long_runtime.opendbc_gitlink)
  n = len(window["tape"]["rdf_input_t"])
  plan = np.zeros((n, 33, 15))
  plan[:, :, 3] = 20.
  plan[:, :, 6] = window["tape"]["rdf_acceleration"][:, :, 0]
  plan[:, :, 14] = .04
  window["tape"]["rdf_plan"] = plan
  window["tape"]["rdf_desired_curvature"] = np.full(n, .002)
  t = window["tape"]["t"]
  window["tape"].update(lat_active=np.ones(len(t), bool), driver_torque=np.zeros(len(t)),
                        fault_temporary=np.zeros(len(t), bool), fault_permanent=np.zeros(len(t), bool),
                        published_delay=np.full(len(t), .1), lat_input_t=t-.001, lat_input_valid=np.ones(len(t), bool))
  anchor = float(t[0])
  envelope = {key: window[key] for key in ("route", "cohort", "source_log_sha256", "opendbc_gitlink")}
  envelope.update(cp_raw=np.frombuffer(json.dumps(cp_dict).encode(), np.uint8).copy(),
                  sp_raw=np.frombuffer(json.dumps(window["sp"]).encode(), np.uint8).copy(), static_t=anchor-1., cp_t=-1., sp_t=anchor-1.,
                  lp={"values": LP.copy(), "t": anchor-.01, "valid": True},
                  params={"values": recorded_fixture_params(), "t": anchor-1., "valid": True},
                  initial={"state": {"angle": 0., "yaw": 0., "vendor_rate": 0.}, "t": anchor, "valid": True},
                  roll={"value": .01, "t": anchor-.01, "valid": True},
                  location_quality={"t": anchor-.01, "valid": True, "yaw_std": .01, "roll_std": .01,
                                    "yaw_inner_valid": 1, "roll_inner_valid": 1, "inputs_ok": 1, "sensors_ok": 1,
                                    "device_stable": 1, "status": 2}, car_state_t=anchor-.001,
                  learner={"t": anchor-.01, "valid": False, "admitted": False, "useParams": False,
                           "frequency_valid": False, "reason": "explicit unit fixture unavailable learner"},
                  initialization={"previous_curvature": 0., "last_transmitted_counts": 0, "angle_counter": 0, "steer_limited": False},
                  history={"t": anchor+np.array([-.02, -.01]), "command": np.zeros(2, int),
                           "request": np.ones(2, bool), "valid": np.ones(2, bool)}, censor_t=None,
                  static_identity_sha256=joint.static_identity(cp_dict, window["sp"]),
                  policy_identity_sha256=joint.policy_identity(recorded_fixture_params()),
                  source_evidence={"verified": True, "source_log_sha256": window["source_log_sha256"],
                                   "schema_sha256": "3"*64, "extractor_sha256": "4"*64, "packet_sha256": "5"*64})
  envelope["lateral_delay_witness"] = {
    "frame_id": window["tape"]["rdf_frame_id"].copy(), "capture_t": window["tape"]["rdf_narrow_t"].copy(),
    "delay": np.full(n, .1), "source_publication_t": window["tape"]["rdf_narrow_t"]-.01, "source_valid": np.ones(n, bool),
    "source_kind": np.full(n, "lateralDelay"), "source_log_sha256": np.full(n, window["source_log_sha256"]),
    "packet_sha256": "5"*64, "packet_lat_delay": np.full(n, .1), "packet_lateral_control_params": np.tile([20., .1], (n, 1))}
  model = {"kind": joint.plant.KIND, "cohort": window["cohort"], "opendbc_gitlink": window["opendbc_gitlink"],
           "static_identity_sha256": envelope["static_identity_sha256"], "policy_identity_sha256": envelope["policy_identity_sha256"],
           "source_eligible": True, "static_policy_timing_verified": True, "source_pin_verified": True,
           "hundred_hz_feedback_validated": False, "selection_eligible": False, "macro_dt_s": .05, "native_dt_s": .01, "micro_nodes": 5,
           "split": deepcopy(report["split"]), "record_hashes": ["7"*64, "8"*64], "min_samples": 100, "min_routes": 2,
           "features": list(joint.plant.FEATURES), "edges": [[] for _ in range(14)], "feature_min": [-500.]*14, "feature_max": [500.]*14,
           "supported_cells": 1, "cells": {"fixture": {"supported": True, "samples": 100, "routes": ["route-t", "route-u"],
                                                       "feature_min": [-500.]*14, "feature_max": [500.]*14, "joint_delta_median": [0., 0., 0.]}}}
  model["model_sha256"] = joint.plant.identity_hash(model)
  calls = []
  def predict(model, state, speed, command, roll):
    calls.append((deepcopy(state), np.array(speed), np.array(command), roll))
    return {"angle": state["angle"]+float(np.sum(command))*.001,
            "yaw": state["yaw"]+float(np.mean(speed))*.00001, "vendor_rate": 4.}
  monkeypatch.setattr(joint.plant, "predict_macro", predict)
  return window, report, envelope, model, long_runtime, runtime, calls


def evaluate(args, **kwargs):
  window, report, envelope, model, long_runtime, runtime, _ = args
  lp = long_proof(window, report, long_runtime)
  proof = joint.bind_proof(window, report, lp, envelope, model, long_runtime, runtime)
  return joint.evaluate_joint(window, report, lp, envelope, model, long_runtime, runtime, proof,
                              lateral_grid=GRID, **kwargs)


def test_future_recorded_feedback_cannot_change_joint_commands(monkeypatch):
  args = example(monkeypatch)
  first = evaluate(args)
  args[0]["tape"].update(v=np.full(100, 99.), a=np.full(100, -9.),
                         steering_angle=np.full(100, 999.), yaw=np.full(100, -99.))
  second = evaluate(args)
  for field in ("v", "a", "angle", "yaw", "can_torque", "physical_curvature", "fixed_path"):
    np.testing.assert_array_equal(first["trace"][field], second["trace"][field])
  assert first["complete_controller_candidate"] is None
  assert "unvalidated_100hz_zoh_feedback" in first["selection_rejections"]
  assert not first["production_enabled"]


def test_simulated_long_speed_couples_into_macro_but_fixed_reference_does_not(monkeypatch):
  args = example(monkeypatch)
  first = evaluate(args)
  second = evaluate(args, parameters={"long": {"ki_v": [.2, .2], "stop_accel": -2.}, "lateral": None})
  assert not np.array_equal(first["trace"]["v"], second["trace"]["v"])
  assert not np.array_equal(first["trace"]["yaw"], second["trace"]["yaw"])
  np.testing.assert_array_equal(first["trace"]["physical_curvature"], second["trace"]["physical_curvature"])
  np.testing.assert_array_equal(first["trace"]["fixed_path"], second["trace"]["fixed_path"])
  assert first["turns"]["reference_sha256"] == second["turns"]["reference_sha256"]


def test_macro_runs_after_fifth_command_and_never_rescales_delta(monkeypatch):
  args = example(monkeypatch)
  result = evaluate(args)
  assert len(args[-1]) == 20
  assert all(len(speed) == len(command) == 5 for _, speed, command, _ in args[-1])
  np.testing.assert_array_equal(result["trace"]["angle"][:5], np.zeros(5))
  assert result["trace"]["angle"][5] == pytest.approx(sum(args[-1][0][2])*.001)
  assert result["turns"]["tracking"]["right"]["rows"] == list(range(20))
  assert result["macro_indices"] == list(range(0, 100, 5))


def test_current_capture_age_reference_is_not_future_feedforward_and_keeps_both_signs(monkeypatch):
  args = example(monkeypatch)
  rdf = args[0]["tape"]
  rdf["rdf_plan"][:, :, 4] = 15.
  rdf["rdf_plan"][:, :, 14] = .04 + rdf["rdf_horizon"]
  rdf["rdf_plan"][:140, :, 14] *= -1
  rdf["rdf_desired_curvature"][:] = .02
  result = evaluate(args)
  assert result["trace"]["physical_curvature"][0] == pytest.approx(-(.04+.05)/25.)
  assert result["trace"]["future_curvature"][0] == .02
  assert set(result["turns"]["reference_direction"]) >= {"left", "right"}


@pytest.mark.parametrize("field", ["request", "valid"])
def test_bad_request_prefix_refused_before_native_creation(monkeypatch, field):
  args = example(monkeypatch)
  args[2]["history"][field][0] = False
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("native construction before refusal"))
  with pytest.raises(joint.JointError, match="prefix"):
    evaluate(args)


def test_request_false_and_unsupported_macro_retain_partial_trace(monkeypatch):
  args = example(monkeypatch)
  original = args[5].create
  def create(*a, **k):
    handle = original(*a, **k)
    step = handle.step
    def changed(**values):
      output = step(**values)
      if values["t"] == args[0]["tape"]["t"][3]:
        output["steer_request"] = False
      return output
    handle.step = changed
    return handle
  monkeypatch.setattr(args[5], "create", create)
  result = evaluate(args)
  assert result["reason"] == "active_steer_request_false" and result["used_steps"] == 3
  assert not args[-1] and np.isnan(result["trace"]["can_torque"][3:]).all()
  monkeypatch.setattr(args[5], "create", original)
  monkeypatch.setattr(joint.plant, "predict_macro", lambda *a: joint.plant.Unsupported("outside_support"))
  result = evaluate(args)
  assert result["reason"] == "macro_unsupported:outside_support" and result["used_steps"] == 5


def test_brake_censor_prevents_future_native_calls_and_macro_completion(monkeypatch):
  args = example(monkeypatch)
  args[0]["tape"]["brake_pressed"][7] = True
  result = evaluate(args)
  assert result["used_steps"] == 7 and result["censor_t"] == args[0]["tape"]["t"][7]
  assert len(args[-1]) == 1 and result["macro_indices"] == [0, 5]
  assert np.isnan(result["trace"]["angle"][7:]).all()


@pytest.mark.parametrize("fault", ["test", "source", "model_source", "cp", "lp_future", "hash", "zero_speed"])
def test_bad_proofs_and_contracts_fail_before_controller_creation(monkeypatch, fault):
  args = example(monkeypatch)
  if fault == "test":
    args[0]["partition"] = "test"
  elif fault == "source":
    args[2]["source_evidence"]["verified"] = False
  elif fault == "model_source":
    args[3]["source_eligible"] = False
  elif fault == "cp":
    args[0]["cp"]["stopAccel"] = -2.1
  elif fault == "lp_future":
    args[2]["lp"]["t"] = args[0]["tape"]["t"][0]+.001
  elif fault == "zero_speed":
    args[0]["tape"]["rdf_plan"][:, :, 3:5] = 0.
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("native construction before refusal"))
  if fault == "hash":
    window, report, env, model, long_runtime, runtime, _ = args
    lp = long_proof(window, report, long_runtime)
    proof = joint.bind_proof(window, report, lp, env, model, long_runtime, runtime)
    proof["lateral_window_sha256"] = "0"*64
    with pytest.raises(joint.JointError):
      joint.evaluate_joint(window, report, lp, env, model, long_runtime, runtime, proof, lateral_grid=GRID)
  else:
    with pytest.raises((joint.JointError, joint.long.SearchError)):
      evaluate(args)


def test_finite_joint_grid_validates_every_entry_before_any_native_call(monkeypatch):
  args = example(monkeypatch)
  window, report, env, model, lr, lat, _ = args
  proof = long_proof(window, report, lr)
  bound = joint.bind_proof(window, report, proof, env, model, lr, lat)
  for grid in ([], [None], [{"long": None, "lateral": {"factor": np.nan, "friction": .1}}],
               [{"long": None, "lateral": None}]):
    monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("native construction before grid refusal"))
    with pytest.raises(joint.JointError):
      joint.validation_search([window], report, {window["id"]: proof}, {window["id"]: env}, model, lr, lat,
                               {window["id"]: bound}, grid=grid, lateral_grid=GRID)


def test_cached_long_result_is_reused_and_mutation_cannot_poison_it(monkeypatch):
  args = example(monkeypatch)
  cache = joint.LongCache()
  calls = []
  original = joint.long.evaluate_trial
  def run(*a, **k):
    calls.append(1)
    return original(*a, **k)
  monkeypatch.setattr(joint.long, "evaluate_trial", run)
  first = evaluate(args, cache=cache)
  first["trace"]["v"][:] = -100
  second = evaluate(args, cache=cache, parameters={"long": None, "lateral": {"factor": 3., "friction": .1}})
  assert len(calls) == 1 and np.all(second["trace"]["v"] > 0)
  assert first["long_cache"]["result_sha256"] == second["long_cache"]["result_sha256"]


def test_joint_summary_rejects_missing_window_instead_of_improving_coverage(monkeypatch):
  args = example(monkeypatch)
  result = evaluate(args)
  metrics = joint.summarize({result["window_id"]: result}, [result["window_id"], "missing"])
  assert metrics["missing_windows"] == ["missing"]
  assert "missing_common_windows" in joint.constraints(metrics, metrics)


def test_joint_constraints_do_not_allow_right_improvement_to_hide_left_regression():
  def row(left, right):
    k = np.r_[-np.linspace(.002, .01, 60), np.linspace(.01, .002, 60), np.linspace(.002, .01, 60), -np.linspace(.01, .002, 60)]
    reference = np.sign(k)
    errors = np.where(k < 0, left, right)
    return joint.turns.window_metrics(k, reference+errors, reference, np.zeros(len(k)))
  before = joint.turns.summarize({"a": row(.1, 1.), "b": row(.1, 1.)}, ["a", "b"])
  after = joint.turns.summarize({"a": row(.3, .1), "b": row(.3, .1)}, ["a", "b"])
  baseline = {"missing_windows": [], "turns": before, "long": None, "speed_p95": 0., "path_p95": 0.}
  candidate = {**baseline, "turns": after}
  assert "left_tracking_regression" in joint.constraints(candidate, baseline)


@pytest.mark.parametrize("fault", ["empty_cells", "roll_std", "split"])
def test_unusable_quality_or_training_population_refused_before_native(monkeypatch, fault):
  args = example(monkeypatch)
  if fault == "empty_cells":
    args[3]["cells"], args[3]["supported_cells"] = {}, 0
  elif fault == "roll_std":
    args[2]["location_quality"]["roll_std"] = None
  else:
    args[3]["split"]["train"] = ["route-v"]
  body = {key: value for key, value in args[3].items() if key != "model_sha256"}
  args[3]["model_sha256"] = joint.plant.identity_hash(body)
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("native construction before refusal"))
  with pytest.raises(joint.JointError):
    evaluate(args)


def test_future_learner_snapshot_is_refused_before_native(monkeypatch):
  args = example(monkeypatch)
  args[2]["learner"]["t"] = args[0]["tape"]["t"][0]+.01
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("native construction before refusal"))
  with pytest.raises(joint.JointError, match="learner"):
    evaluate(args)


def test_learner_is_applied_only_at_causal_anchor_not_from_future_updates(monkeypatch):
  args = example(monkeypatch)
  original = args[5].create
  received = []
  def create(*a, **k):
    handle = original(*a, **k)
    step = handle.step
    def capture(**values):
      received.append(values["learner"])
      return step(**values)
    handle.step = capture
    return handle
  monkeypatch.setattr(args[5], "create", create)
  args[2]["learner"] = {"t": args[0]["tape"]["t"][0]-.05, "valid": True, "admitted": True, "frequency_valid": True,
                        "useParams": True, "factor": 3., "offset": .01, "friction": .2}
  result = evaluate(args)
  assert received[0]["factor"] == 3. and all(value is None for value in received[1:])
  assert "frozen_learner_adaptation_unverified" in result["selection_rejections"]


def test_future_unready_plan_is_not_used_and_latest_invalid_has_no_older_fallback(monkeypatch):
  args = example(monkeypatch)
  before = evaluate(args)
  args[0]["tape"]["rdf_plan"][-1] *= -900
  after = evaluate(args)
  np.testing.assert_array_equal(before["trace"]["physical_curvature"], after["trace"]["physical_curvature"])
  args[0]["tape"]["rdf_controller_admissible"][133] = False
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("native construction before refusal"))
  with pytest.raises(joint.JointError, match="reference unavailable"):
    evaluate(args)


def test_missing_recorded_policy_is_not_defaulted(monkeypatch):
  args = example(monkeypatch)
  del args[2]["params"]["values"]["CustomTorqueParams"]
  args[2]["policy_identity_sha256"] = joint.policy_identity(args[2]["params"]["values"])
  args[3]["policy_identity_sha256"] = args[2]["policy_identity_sha256"]
  args[3]["model_sha256"] = joint.plant.identity_hash({k: v for k, v in args[3].items() if k != "model_sha256"})
  monkeypatch.setattr(joint.long, "evaluate_trial", lambda *a, **k: pytest.fail("long native call before refusal"))
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("lateral native call before refusal"))
  with pytest.raises(joint.JointError, match="typed Params"):
    evaluate(args)


def test_actual_grid_keeps_baseline_failed_rows_and_identical_window_population(monkeypatch):
  args = example(monkeypatch)
  window, report, env, model, lr, lat, _ = args
  lp = long_proof(window, report, lr)
  bound = joint.bind_proof(window, report, lp, env, model, lr, lat)
  original_create, original_long = lat.create, joint.long.evaluate_trial
  long_calls = []
  def create(*a, **kw):
    if kw["candidate"] and kw["candidate"]["factor"] == 3.:
      raise RuntimeError("retained candidate constructor failure")
    return original_create(*a, **kw)
  def long_run(*a, **kw):
    long_calls.append(1)
    return original_long(*a, **kw)
  monkeypatch.setattr(lat, "create", create)
  monkeypatch.setattr(joint.long, "evaluate_trial", long_run)
  grid = [{"long": None, "lateral": {"factor": factor, "friction": .1}} for factor in (2., 3.)]
  result = joint.validation_search([window], report, {window["id"]: lp}, {window["id"]: env}, model, lr, lat,
                                   {window["id"]: bound}, grid=grid, lateral_grid=GRID)
  assert len(long_calls) == 1 and len(result["rows"]) == 2
  assert result["baseline"]["results"][window["id"]]["parameters"] == {"long": None, "lateral": None}
  assert all(row["common_window_ids"] == [window["id"]] for row in [result["baseline"], *result["rows"]])
  failed = result["rows"][1]
  assert failed["coverage"] == 0 and failed["failures"][0]["error_type"] == "RuntimeError"
  assert "retained candidate constructor failure" in failed["failures"][0]["traceback"]
  assert result["selection"] is None and result["complete_controller_candidate"] is None


@pytest.mark.parametrize("bad_grid", [None, {"factor": [np.nan], "friction": [.1]}])
def test_declared_lateral_grid_must_be_valid_even_for_baseline(monkeypatch, bad_grid):
  args = example(monkeypatch)
  window, report, env, model, lr, lat, _ = args
  lp = long_proof(window, report, lr)
  bound = joint.bind_proof(window, report, lp, env, model, lr, lat)
  monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("native construction before grid refusal"))
  with pytest.raises(joint.JointError, match="grid"):
    joint.evaluate_joint(window, report, lp, env, model, lr, lat, bound, lateral_grid=bad_grid)


@pytest.mark.skipif(os.name != "posix", reason="Owner-serialized actual native POSIX API case; synthetic inputs only")
def test_actual_native_joint_moving_api_synthetic_full_static_and_speed_coupling(monkeypatch):
  """Both actual classes + actual median macro API; no real plant/data acceptance."""
  repo = Path(os.environ["NATIVE_LATERAL_SOURCE_ROOT"]).resolve()
  assert Path(os.environ["NATIVE_SEARCH_SOURCE_ROOT"]).resolve() == repo
  manifest_bytes = Path(os.environ["NATIVE_JOINT_SOURCE_MANIFEST"]).read_bytes()
  assert hashlib.sha256(manifest_bytes).hexdigest() == "2f7c60555e8329ad6cf259ec9b87d6bac93e6ee97d3e2f948afd3f61bd5c02d2"
  manifest = json.loads(manifest_bytes)
  assert len(manifest["expected_native_sources"]) == 44
  assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip() == \
    "13aedc522518fe1b62651d9a8a320e931efc335c"
  lat = joint.lateral.load_runtime(repo, manifest["expected_native_sources"], manifest["expected_opendbc_gitlink"])
  long_sources = joint.long.source_hashes(repo)
  assert all(value == manifest["expected_native_sources"][key] for key, value in long_sources.items()
             if key in manifest["expected_native_sources"])
  lr = joint.long.load_native_runtime(repo, long_sources, joint.lateral.OPENDBC_PIN)
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
  sp.neuralNetworkLateralControl.model.path = str(repo/joint.lateral.ASSET)
  args = list(example(monkeypatch))
  monkeypatch.undo()  # Only the code-fixture setup used a patched macro; actual case must not.
  window, report, env, model = args[:4]
  window.update(cp=cp.to_dict(), sp=sp.to_dict(), opendbc_gitlink=joint.lateral.OPENDBC_PIN)
  env.update(cp_raw=np.frombuffer(cp.to_bytes(), np.uint8).copy(), sp_raw=np.frombuffer(sp.to_bytes(), np.uint8).copy(),
             opendbc_gitlink=joint.lateral.OPENDBC_PIN,
             static_identity_sha256=joint.static_identity(window["cp"], window["sp"]))
  model.update(opendbc_gitlink=joint.lateral.OPENDBC_PIN, static_identity_sha256=env["static_identity_sha256"])
  horizon = np.asarray(lat.bindings.horizon)
  tape = window["tape"]
  tape["rdf_horizon"] = horizon
  tape["rdf_acceleration"][:, :, 0] = .1+horizon
  tape["rdf_plan"][:, :, 6] = .1+horizon
  lower = [-10., -1., 0., -1., *([0.]*5), *([-409.]*5)]
  upper = [10., 1., 100., 1., *([40.]*5), *([409.]*5)]
  edges = [[] for _ in range(14)]
  edges[4] = [20.5]
  cells = {}
  for high in (False, True):
    lo, hi, key = lower.copy(), upper.copy(), ["0"]*14
    if high:
      lo[4], key[4] = 20.5, "1"
    else:
      hi[4] = 20.5
    cells[":".join(key)] = {"supported": True, "samples": 100, "routes": report["split"]["train"],
                            "feature_min": lo, "feature_max": hi,
                            "joint_delta_median": [-.001 if high else -.0005, .0004 if high else .0002, 0.]}
  model.update(edges=edges, feature_min=lower, feature_max=upper, cells=cells, supported_cells=2)
  model["model_sha256"] = joint.plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  args[4:6] = [lr, lat]
  first = evaluate(args)
  window["initial_v"] = 21.  # A second explicit synthetic anchor, not a comparable validation candidate.
  second = evaluate(args)
  assert first["status"] == second["status"] == "complete"
  assert first["used_steps"] == second["used_steps"] == 100 and len(first["macro_indices"]) == 20
  assert lr.actual_native and lat.bindings.actual_native
  assert first["long_result"]["evidence"] == "actual native class"
  assert not np.array_equal(first["trace"]["angle"], second["trace"]["angle"])
  assert not np.array_equal(first["trace"]["yaw"], second["trace"]["yaw"])
  for result in (first, second):
    counts = result["trace"]["can_torque"]
    assert np.max(np.abs(counts)) <= 409 and np.max(np.abs(np.diff(np.r_[0., counts]))) <= 7
    assert abs(counts[0]) <= 3 and result["complete_controller_candidate"] is None and not result["production_enabled"]
  assert window["cp"]["carFw"][0]["fwVersion"] == b"\x00\xff\x80"


@pytest.mark.parametrize("fault", ["partial_macro", "native_horizon", "missing_witness", "delay_mismatch", "future_delay_source", "retagged_lateral_action"])
def test_review_admission_refuses_before_either_native_factory(monkeypatch, fault):
  args = example(monkeypatch)
  window, report, env, model, lr, lat, _ = args
  if fault == "partial_macro":
    for key, value in window["tape"].items():
      if not key.startswith("rdf_"):
        window["tape"][key] = value[:7]
  elif fault == "native_horizon":
    lat.bindings = replace(lat.bindings, horizon=np.asarray(lat.bindings.horizon)+.01)
  elif fault == "missing_witness":
    del env["lateral_delay_witness"]
  elif fault == "delay_mismatch":
    env["lateral_delay_witness"]["packet_lat_delay"] += .2
  elif fault == "retagged_lateral_action":
    for key in ("action_t", "packet_action_t"):
      window["model_inputs"][key][:, 0] += .2
      window["tape"]["rdf_"+key][:, 0] += .2
  else:
    env["lateral_delay_witness"]["source_publication_t"][0] = env["lateral_delay_witness"]["capture_t"][0]+.01
  monkeypatch.setattr(joint.long, "evaluate_trial", lambda *a, **k: pytest.fail("LongControl reached before admission refusal"))
  monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("LatHandle reached before admission refusal"))
  with pytest.raises(joint.JointError):
    evaluate(args)


def test_cp_delay_fallback_uses_actual_cp_publication_not_sp(monkeypatch):
  args = example(monkeypatch)
  env, witness = args[2], args[2]["lateral_delay_witness"]
  env["sp_t"], env["static_t"] = -.5, -.5
  witness["source_kind"][:] = "carParams"
  witness["source_publication_t"][:] = env["cp_t"]
  assert evaluate(args)["status"] == "complete"
  witness["source_publication_t"][:] = env["sp_t"]
  monkeypatch.setattr(joint.long, "evaluate_trial", lambda *a, **k: pytest.fail("LongControl before CP witness refusal"))
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("LatHandle before CP witness refusal"))
  with pytest.raises(joint.JointError, match="prior CP"):
    evaluate(args)


def test_later_core_delay_and_extension_cache_do_not_relabel_inference_delay(monkeypatch):
  args = example(monkeypatch)
  args[0]["tape"]["published_delay"][:] = .25
  result = evaluate(args)
  assert result["status"] == "complete"
  np.testing.assert_allclose(result["trace"]["future_action_t"][:, 0], .275)
  np.testing.assert_allclose(args[2]["lateral_delay_witness"]["delay"], .1)
  assert args[2]["params"]["values"]["LagdValueCache"] == "0.2"
