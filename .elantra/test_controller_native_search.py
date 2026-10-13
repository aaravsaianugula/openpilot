"""Code fixtures prove replay invariants, never vehicle performance."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import controller_native_search as search
from controller_native_replay import MODEL_SHA


class FixtureController:
  calls = []

  def __init__(self, cp, sp):
    self.cp, self.sp = cp, sp
    self.long_control_state, self.last_output_accel = 0, 0.
    self.pid = SimpleNamespace(i=0.)

  def reset(self):
    self.pid.i = 0.

  def update(self, active, cs, target, should_stop, limits):
    self.calls.append((cs.vEgo, cs.aEgo, should_stop, active))
    self.long_control_state = 0 if not active else 2 if should_stop else 1
    if not active:
      self.reset()
      value = 0.
    elif should_stop:
      self.reset()
      value = self.last_output_accel - .01 if self.last_output_accel > self.cp.stopAccel else self.last_output_accel
    else:
      self.pid.i += float(self.cp.longitudinalTuning.kiV[0]) * .01 * (target-cs.aEgo)
      value = target + self.pid.i
    self.last_output_accel = float(np.clip(value, *limits))
    return self.last_output_accel


def example():
  n = 170
  capture = np.arange(n)*.05
  horizon = np.linspace(0, np.sqrt(5), 33)**2
  rdf = {"input_t": capture+.001, "narrow_t": capture, "output_ready_t": capture+.02,
         "frame_id": np.arange(n), "desired_acceleration": np.full(n, .2), "should_stop": np.zeros(n, bool),
         "scored_mask": np.arange(n) >= 132, "model_sha256": np.array(MODEL_SHA),
         "controller_admissible": np.ones(n, bool), "deadline_admissible": np.ones(n, bool),
         "acceleration": np.tile(np.column_stack((.1+horizon, np.zeros(33), np.zeros(33))), (n, 1, 1)), "horizon": horizon}
  rdf["packet_action_t"] = np.tile([.175, .575], (n, 1))
  rdf["action_t"] = rdf["packet_action_t"] + [.1, .3]
  t = 6.7 + np.arange(100)*.01
  tape = {"t": t, "long_active": np.ones(100), "brake_pressed": np.zeros(100), "gas_pressed": np.zeros(100),
          "steering_pressed": np.zeros(100), "cruise_standstill": np.zeros(100), "valid": np.ones(100, bool),
          "v_cruise": np.full(100, 100.), "v": np.full(100, 20.), "a": np.zeros(100)}
  for service in ("carState", "carControl"):
    tape[service+"_t"], tape[service+"_valid"] = t-.001, np.ones(100, bool)
  tape.update({"rdf_"+key: value for key, value in rdf.items()})
  cp = {"carFingerprint": "HYUNDAI_ELANTRA_2024", "openpilotLongitudinalControl": True,
        "stopAccel": -2., "longitudinalTuning": {"kiBP": [0., 30.], "kiV": [.07, .12]}, "longitudinalActuatorDelay": .5}
  window = {"id": "validation-window", "route": "route-v", "cohort": "known", "partition": "validation",
            "opendbc_gitlink": "a"*40, "source_log_sha256": "b"*64, "tape": tape, "cp": cp,
            "sp": {"enableGasInterceptor": False}, "initial_v": 20., "initial_a": 0.,
            "history_t": np.array([6.5, 6.55, 6.6, 6.65]), "history_command": np.zeros(4), "history_valid": np.ones(4, bool),
            "model_inputs": {"fresh_inference": True, "model_sha256": MODEL_SHA,
                             "model_source_sha256": "d"*64,
                             "action_t": rdf["action_t"].copy(), "packet_action_t": rdf["packet_action_t"].copy(),
                             "packet_includes_model_smoothing": False}, "reference_mode": "capture_horizon"}
  model = {"table": {"18+": {"command": [-3.5, 0, 2.], "response": [-3.5, 0, 2.]}},
           "delay_s": .1, "tau_s": .5, "identified": True, "boundary_optimum": False,
           "recorded_build_gitlink_known": True}
  report = {"cohort": "known", "source_gitlink_known": True,
            "split": {"train": ["route-t", "route-u"], "validation": ["route-v"], "test": ["route-x"]},
            "route_start_wall_time_nanos": {"route-t": 1, "route-u": 2, "route-v": 3, "route-x": 4},
            "axes": {"long": {"identified": True, "model": model,
                              "validation_baseline": {"used_windows": 30, "coverage": .8, "mae": .1}}}}
  runtime = search.NativeRuntime(FixtureController, lambda cp, sp, v, cruise: (-3.5, 2.),
                                 {search.LONG_SOURCE: search.LONG_SHA}, "a"*40, False)
  return window, report, runtime


def proof(window, report, runtime):
  return {"window_sha256": search.value_hash(window), "plant_report_sha256": search.value_hash(report),
          "rdf_sha256": search.value_hash(search.rdf_from_tape(window["tape"])),
          "model_inputs_sha256": search.value_hash(window["model_inputs"]),
          "source_log_sha256": window["source_log_sha256"], "plant_opendbc_gitlink": window["opendbc_gitlink"],
          "native_sources": runtime.source_hashes.copy(), "initialization": "cold_constructor"}


def evaluate(window, report, runtime, parameters=None, **kwargs):
  return search.evaluate_trial(window, report, proof(window, report, runtime), runtime, parameters, **kwargs)


def test_future_recorded_gt_cannot_change_native_feedback_or_commands():
  window, report, runtime = example()
  first = evaluate(window, report, runtime)
  changed = deepcopy(window)
  changed["tape"]["v"][:] = 99
  changed["tape"]["a"][:] = -3
  second = evaluate(changed, report, runtime)
  for key in ("command", "v", "a", "native_state", "physical_reference"):
    np.testing.assert_array_equal(first["trace"][key], second["trace"][key])
  assert first["trace"]["v"][-1] > first["trace"]["v"][0]
  assert FixtureController.calls[-1][0] != 99


def test_five_native_substeps_preserve_identified20hz_constant_input_response():
  alpha = search.resampled_alpha(.5)
  state = 0.
  for _ in range(5):
    state += alpha * (1-state)
  assert state == pytest.approx(.05 / (.5+.05), abs=1e-14)
  assert alpha != pytest.approx(.01 / (.5+.01), abs=1e-5)


def test_baseline_preserves_actual_speed_dependent_integral_vector():
  window, report, runtime = example()
  result = evaluate(window, report, runtime)
  assert result["parameters"] == {"ki_v": [.07, .12], "stop_accel": -2.}
  assert window["cp"]["longitudinalTuning"]["kiV"] == [.07, .12]


def test_feedforward_future_action_is_not_physical_scoring_reference():
  window, report, runtime = example()
  window["tape"]["rdf_desired_acceleration"][:] = .7
  result = evaluate(window, report, runtime)
  trace = result["trace"]
  assert trace["feedforward_target"][0] == .7
  assert trace["physical_reference"][0] == pytest.approx(.1 + 6.7 - 6.65)
  assert trace["physical_reference"][0] != trace["feedforward_target"][0]


def test_unsupported_speed_band_and_low_speed_stop_remain_named_failures():
  window, report, runtime = example()
  window["initial_v"] = 9.
  window["tape"]["rdf_should_stop"][:] = True
  result = evaluate(window, report, runtime)
  assert result["status"] == "unsupported"
  assert result["reason"] == "unsupported_speed_or_command_cell"
  assert result["used_steps"] == 0 and not result["stopping_supported"]
  window["initial_v"] = 20.
  result = evaluate(window, report, runtime, stopping=True)
  assert result["status"] == "unsupported" and result["reason"] == "stopping_plant_unidentified"


@pytest.mark.parametrize("field", ["gas_pressed", "brake_pressed", "steering_pressed", "long_active", "valid", "carState_t"])
def test_exact_first_takeover_or_invalid_tick_censors_horizon(field):
  window, report, runtime = example()
  window["tape"][field][7] = (window["tape"]["t"][7]+.01 if field == "carState_t"
                              else 0 if field in ("long_active", "valid") else 1)
  result = evaluate(window, report, runtime)
  assert result["status"] == "censored" and result["used_steps"] == 7
  assert result["censor_t"] == window["tape"]["t"][7]
  assert np.isnan(result["trace"]["command"][7:]).all()
  if field == "long_active":
    assert result["terminal_native_state"] == 0


@pytest.mark.parametrize("fault", ["source", "cohort", "pin", "hash", "delay", "retagged_delay", "test", "untrusted"])
def test_source_cohort_pin_hash_delay_and_heldout_contracts_fail_closed(fault):
  window, report, runtime = example()
  bound = proof(window, report, runtime)
  if fault == "source":
    bound["native_sources"][search.LONG_SOURCE] = "0"*64
  elif fault == "cohort":
    window["cohort"] = "other"
  elif fault == "pin":
    window["opendbc_gitlink"] = None
  elif fault == "hash":
    bound["window_sha256"] = "0"*64
  elif fault in ("delay", "retagged_delay"):
    window["cp"]["longitudinalActuatorDelay"] = .6
    if fault == "retagged_delay":
      window["model_inputs"]["action_t"][:, 1] += .1
      window["model_inputs"]["packet_action_t"][:, 1] += .1
  elif fault == "test":
    window["route"], window["partition"] = "route-x", "test"
  else:
    report["source_gitlink_known"] = False
  if fault not in ("source", "hash"):
    bound = proof(window, report, runtime)
  with pytest.raises(search.SearchError):
    search.evaluate_trial(window, report, bound, runtime)


def test_missing_raw_reference_is_labelled_and_cannot_select_tune():
  window, report, runtime = example()
  window["reference_mode"] = "command_target_only"
  window["tape"].pop("rdf_acceleration")
  result = evaluate(window, report, runtime)
  assert result["status"] == "complete" and not result["selection_eligible"]
  assert np.isnan(result["trace"]["physical_reference"]).all()
  assert "physical_time_reference_unavailable" in result["selection_rejections"]


@pytest.mark.parametrize("flag", ["controller_admissible", "deadline_admissible"])
def test_fresh_scored_latest_inadmissible_output_is_censored_without_older_fallback(flag):
  window, report, runtime = example()
  window["tape"]["rdf_"+flag][133] = False
  result = evaluate(window, report, runtime)
  assert result["used_steps"] == 0 and result["status"] == "censored"
  assert result["reason"] == "latest_model_controller_or_deadline_inadmissible"
  assert window["tape"]["rdf_scored_mask"][133] and window["tape"]["rdf_"+flag][132]


def test_unsupported_candidate_retained_without_destroying_supported_common_population():
  window, report, runtime = example()
  class FailingCandidate(FixtureController):
    def update(self, *args):
      if self.cp.longitudinalTuning.kiV[0] == .2:
        raise RuntimeError("fixture deliberately unsupported candidate")
      return super().update(*args)
  runtime = search.NativeRuntime(FailingCandidate, runtime.accel_limits, runtime.source_hashes, runtime.opendbc_gitlink, False)
  windows, proofs = [], {}
  for i in range(20):
    item = deepcopy(window)
    item["id"] = str(i)
    windows.append(item)
    proofs[str(i)] = proof(item, report, runtime)
  grid = [{"ki_v": [.0, .0], "stop_accel": -2.}, {"ki_v": [.2, .2], "stop_accel": -2.}]
  result = search.validation_search(windows, report, proofs, runtime, grid=grid)
  assert len(result["rows"]) == 2
  assert result["complete_controller_candidate"] is None and not result["production_enabled"]
  assert result["baseline_parameters"]["ki_v"] == [.07, .12]
  assert len(result["common_window_ids"]) == 20
  assert result["rows"][0]["individual_windows"] == 20
  assert result["rows"][1]["individual_windows"] == 0
  assert len(result["rows"][1]["failures"]) == 20
  assert result["rows"][1]["failures"][0]["error_type"] == "RuntimeError"
  assert "fixture deliberately unsupported" in result["rows"][1]["failures"][0]["traceback"]
  assert result["rows"][0]["finite_grid_boundaries"]["ki_v"] == ["minimum", "minimum"]
  malformed = search.validation_search(windows, report, proofs, runtime, grid=[{}])
  assert malformed["selection"] is None and len(malformed["rows"]) == 1
  assert len(malformed["rows"][0]["failures"]) == 20


def score(errors=.1, jerk=.1, saturated=0., stop_distance=5.):
  return {"errors": [errors]*100, "acceleration_errors": [errors]*50, "braking_errors": [errors]*50,
          "jerk": [jerk]*99, "command_rate": [.5]*99, "saturated": [saturated]*100,
          "stopping_result": {"distance_from_request_m": stop_distance, "time_from_request_s": 2., "target_feasible": True,
                              "model_target": {"lower_m": 4.9, "upper_m": 5.1, "request_t": 1., "capture_t": .95, "path_sha256": "c"*64}}}


def trial_row(parameters, sample):
  return {"parameters": parameters, "windows": {str(i): deepcopy(sample) for i in range(20)},
          "individually_supported": True, "selection_rejections": []}


def test_fully_admitted_selector_can_choose_real_finite_native_parameter_winner():
  baseline = trial_row({"ki_v": [.07, .12], "stop_accel": -2.}, score(.1))
  small = trial_row({"ki_v": [.05, .05], "stop_accel": -2.}, score(.08))
  winner = trial_row({"ki_v": [.1, .1], "stop_accel": -2.5}, score(.05))
  selected, common = search.select_trials([baseline, small, winner], 20)
  assert selected is winner and len(common) == 20
  assert winner["rank"] == 1 and small["rank"] == 2


@pytest.mark.parametrize("violation", ["jerk", "saturation", "braking", "stop_distance", "initialization", "missing_stop_target", "anchor"])
def test_native_selector_rejects_comfort_direction_stop_and_initialization_regressions(violation):
  baseline = trial_row({"ki_v": [0.], "stop_accel": -2.}, score(.1))
  candidate = trial_row({"ki_v": [.1], "stop_accel": -2.}, score(.05))
  if violation == "initialization":
    candidate["selection_rejections"] = ["native_initialization_uncertain"]
  for value in candidate["windows"].values():
    if violation == "jerk":
      value["jerk"] = [4.]*99
    elif violation == "saturation":
      value["saturated"] = [1.]*100
    elif violation == "braking":
      value["braking_errors"] = [.2]*50
    elif violation == "stop_distance":
      value["stopping_result"]["distance_from_request_m"] = 5.2
    elif violation == "missing_stop_target":
      value["stopping_result"]["model_target"] = None
    elif violation == "anchor":
      value["stopping_result"]["model_target"]["request_t"] = 1.1
  selected, _ = search.select_trials([baseline, candidate], 20)
  assert selected is None and candidate["selection_rejections"]


def test_claimed_reset_string_is_not_causal_valid_raw_off_evidence():
  window, report, runtime = example()
  bound = proof(window, report, runtime)
  bound["initialization"] = "observed_off_reset"
  result = search.evaluate_trial(window, report, bound, runtime)
  assert "native_initialization_uncertain" in result["selection_rejections"]
  window["reset_evidence"] = {"valid": True, "t": window["tape"]["t"][0]-.01,
                              "long_active": False, "native_state": 0, "transmitted_accel": 0.}
  assert search.reset_observed(window)
  window["reset_evidence"]["t"] = window["tape"]["t"][0]+.01
  assert not search.reset_observed(window)


def test_model_stop_target_uses_causal_raw_plan_and_simulated_past_not_future_gt():
  window, report, runtime = example()
  h = window["tape"]["rdf_horizon"]
  position = np.column_stack((np.minimum(2*h, 5), np.zeros(33), np.zeros(33)))
  window["tape"]["rdf_position"] = np.tile(position, (170, 1, 1))
  window["stop_target_uncertainty"] = uncertainty(window)
  t = window["tape"]["t"]
  trace = {"t": t, "distance": (t-t[0])*2, "rdf_index": np.full(len(t), 138.)}
  target = search.model_stopping_target(window, trace, 25, report["split"]["train"])
  assert target["capture_t"] == pytest.approx(6.9) and target["available_t"] <= t[25]
  assert target["distance_from_request_m"] == pytest.approx(4.9)
  window["tape"]["v"][:] = 999
  window["tape"]["a"][:] = -99
  assert search.model_stopping_target(window, trace, 25, report["split"]["train"]) == target
  window["stop_target_uncertainty"]["evidence"]["outcomes"][0]["route"] = "route-x"
  assert search.model_stopping_target(window, trace, 25, report["split"]["train"]) is None


def uncertainty(window):
  identity = search.stop_prediction_identity(window)
  rows = [{"event_id": str(i), "route": ("route-t", "route-u")[i%2], "partition": "train",
           "capture_t": float(i*10), "observed_stop_t": float(i*10+2), "model_distance_m": 5., "observed_distance_m": 5.1,
           "residual_m": 5.1-5., "identity_sha256": search.value_hash(identity), "uninterrupted": True,
           "long_active_entire_episode": True, "valid_motion_entire_episode": True, "interventions": [],
           "source_log_sha256": ("c", "d")[i%2]*64, "raw_model_plan_sha256": "e"*64} for i in range(20)]
  body = {"identity": identity, "method": "empirical_higher_absolute_model_vs_observed_stop_residual", "quantile": .95,
          "sample_count": 20, "route_count": 2, "outcomes": rows}
  return {"bound_m": 5.1-5., "source_sha256": search.value_hash(body), "evidence": body}


@pytest.mark.parametrize("fault", ["arbitrary_bound", "one_route", "too_few", "intervention", "identity", "residual", "overlap", "heldout"])
def test_model_stop_uncertainty_requires_retained_supported_train_residuals(fault):
  window, report, runtime = example()
  value = uncertainty(window)
  window["stop_target_uncertainty"] = value
  assert search.verified_stop_uncertainty(window, report["split"]["train"]) == pytest.approx(.1)
  body = value["evidence"]
  if fault == "arbitrary_bound":
    value["bound_m"] = 9.
  elif fault == "one_route":
    for row in body["outcomes"]:
      row["route"] = "route-t"
    body["route_count"] = 1
  elif fault == "too_few":
    body["outcomes"].pop()
    body["sample_count"] = 19
  elif fault == "intervention":
    body["outcomes"][0]["interventions"] = ["gas"]
  elif fault == "identity":
    body["outcomes"][0]["identity_sha256"] = "0"*64
  elif fault == "residual":
    body["outcomes"][0]["residual_m"] = 8.
  elif fault == "overlap":
    body["outcomes"][2]["capture_t"] = 1.
    body["outcomes"][2]["observed_stop_t"] = 3.
  else:
    body["outcomes"][0]["route"] = "route-x"
  value["source_sha256"] = search.value_hash(body)
  assert search.verified_stop_uncertainty(window, report["split"]["train"]) is None


def test_model_stop_uncertainty_rejects_one_log_assigned_to_multiple_train_routes():
  window, report, runtime = example()
  value = uncertainty(window)
  window["stop_target_uncertainty"] = value
  for row in value["evidence"]["outcomes"]:
    row["source_log_sha256"] = "c"*64
  value["source_sha256"] = search.value_hash(value["evidence"])
  # Distinct event IDs and nonoverlapping episodes cannot make one log two independent routes.
  assert search.verified_stop_uncertainty(window, report["split"]["train"]) is None


def test_near_standstill_native_stop_entry_retains_prior_moving_approach():
  window, report, runtime = example()
  tape = window["tape"]
  t = 6.7 + np.arange(190)*.01
  for field, values in list(tape.items()):
    if not field.startswith("rdf_"):
      tape[field] = np.resize(values, len(t))
  tape["t"] = t
  for service in ("carState", "carControl"):
    tape[service+"_t"] = t-.001
  tape["rdf_should_stop"][151:] = True  # Preserve native late stopping entry; do not force an early state.
  tape["rdf_desired_acceleration"][:] = -3.5
  window["initial_v"], window["initial_a"] = 3., -3.5
  model = report["axes"]["long"]["model"]
  model["table"] = {search.tune.band_of(v): {"command": [-3.5, 0, 2.], "response": [-3.5, 0, 2.]} for v in range(31)}
  model.update(stopping_identified=True, low_speed_validation={"used_windows": 20, "coverage": .8, "standstill_hold_identified": True})
  result = evaluate(window, report, runtime, stopping=True)
  assert result["status"] == "complete"
  assert result["stopping_result"]["speed_at_native_stop_request_mps"] < 2
  assert result["stopping_result"]["max_simulated_approach_speed_mps"] == 3.
  assert "moving_approach_history_unavailable" not in result["selection_rejections"]
  window["initial_v"] = .2
  result = evaluate(window, report, runtime, stopping=True)
  assert "moving_approach_history_unavailable" in result["selection_rejections"]


@pytest.mark.skipif(os.name != "posix", reason="Actual native class requires owner-serialized POSIX openpilot runtime")
def test_actual_longcontrol_should_stop_uses_native_state_and_stopping_ramp():
  window, report, fixture = example()
  window["tape"]["rdf_should_stop"][:] = True
  repo = Path(os.environ.get("NATIVE_SEARCH_SOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
  pin = subprocess.run(["git", "-C", str(repo / "opendbc_repo"), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
  runtime = search.load_native_runtime(repo, search.source_hashes(repo), pin)
  window["opendbc_gitlink"] = pin
  result = evaluate(window, report, runtime)
  assert result["trace"]["native_state"][0] == 2
  assert result["trace"]["command"][0] == pytest.approx(-.01)
  assert result["trace"]["command"][1] == pytest.approx(-.02)


@pytest.mark.skipif(os.name != "posix", reason="Actual native PID requires owner-serialized POSIX openpilot runtime")
def test_actual_native_pid_uses_simulated_feedback_and_candidate_integral_schedule():
  window, report, fixture = example()
  repo = Path(os.environ.get("NATIVE_SEARCH_SOURCE_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
  pin = subprocess.run(["git", "-C", str(repo / "opendbc_repo"), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
  runtime = search.load_native_runtime(repo, search.source_hashes(repo), pin)
  window["opendbc_gitlink"] = pin
  baseline = evaluate(window, report, runtime)
  candidate = {"ki_v": [.2, .2], "stop_accel": -2.}
  changed_gain = evaluate(window, report, runtime, candidate)
  altered_gt = deepcopy(window)
  altered_gt["tape"]["v"][:] = 99.
  altered_gt["tape"]["a"][:] = -3.
  unchanged_feedback = evaluate(altered_gt, report, runtime, candidate)
  assert baseline["status"] == changed_gain["status"] == unchanged_feedback["status"] == "complete"
  assert np.all(changed_gain["trace"]["native_state"] == 1)
  assert np.max(np.abs(baseline["trace"]["command"] - changed_gain["trace"]["command"])) > 1e-4
  for key in ("command", "pid_integral", "a", "v", "native_state"):
    np.testing.assert_array_equal(changed_gain["trace"][key], unchanged_feedback["trace"][key])
  assert not changed_gain["selection_eligible"]  # Synthetic plant is not car stopping evidence.


def test_full_static_capnp_data_can_be_content_bound_without_stripping_firmware():
  original = {"carFw": [{"ecu": "eps", "fwVersion": b"\x00\xff\x80"}], "lateralTuning": {"torque": {"friction": .1}}}
  changed = deepcopy(original)
  changed["carFw"][0]["fwVersion"] = b"\x00\xfe\x80"
  assert search.value_hash(original) != search.value_hash(changed)
  assert original["carFw"][0]["fwVersion"] == b"\x00\xff\x80"


def test_capnp_data_bytes_and_bytearray_share_identity_but_text_does_not():
  assert search.value_hash({"fwVersion": b"\x00\xff"}) == search.value_hash({"fwVersion": bytearray(b"\x00\xff")})
  assert search.value_hash({"fwVersion": b"\x00\xff"}) != search.value_hash({"fwVersion": "00ff"})


def test_capnp_data_extension_preserves_prior_nondata_hash_contract():
  import hashlib
  import json

  old = {"float": .1, "flags": [True, False], "nested": {"numbers": [1, 2]}}
  encoded = json.dumps(old, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
  assert search.value_hash(old) == hashlib.sha256(encoded).hexdigest()
