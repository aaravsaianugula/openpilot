"""Synthetic contract fixtures only: no car, native API or identified plant proof."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

import controller_native_two_way as runner
import controller_native_joint as old
import controller_multirate_joint as plant
from test_controller_native_search import example as long_example, FixtureController
from test_controller_native_lateral_runtime import fixture_runtime as lat_fixture, LP, GRID, params
from test_controller_native_longitudinal_output import fixture_runtime as output_fixture


class Static(NS):
  def __init__(self, values):
    self.values = deepcopy(values)
    self.__dict__.update(deepcopy(values))
    if "longitudinalTuning" in values:
      self.longitudinalTuning = NS(**deepcopy(values["longitudinalTuning"]))

  def to_dict(self):
    return deepcopy(self.values)

  def to_bytes(self):
    return json.dumps(self.values).encode()


def fixture():
  window, report, lr = long_example()
  report.pop("axes")  # This runner must not depend on a separable long plant.
  cp, sp, tape = window["cp"], window["sp"], window["tape"]
  cp.update(
    brand="hyundai",
    radarUnavailable=False,
    steerControlType="torque",
    minSteerSpeed=0.0,
    steerAtStandstill=False,
    steerActuatorDelay=0.1,
    lateralTuning={"torque": {"latAccelFactor": 2.5, "latAccelOffset": 0.0, "friction": 0.1}},
  )
  sp["flags"] = 0
  lr = replace(lr, static_builder=lambda c, s: (Static(c), Static(s)))
  lat, decoded, _, _ = lat_fixture()

  def decode(c, s):
    selected = deepcopy(decoded)
    selected.to_dict = lambda: json.loads(c)
    return selected, Static(json.loads(s))

  lat = old.lateral.Runtime(replace(lat.bindings, decode_static=decode, horizon=tape["rdf_horizon"]), {"fixture": "f" * 64}, lr.opendbc_gitlink)
  out = output_fixture()
  out.pin = lr.opendbc_gitlink
  out.sources = {"fixture": "f" * 64}
  t, n = tape["t"], len(tape["rdf_input_t"])
  clocks = np.array([round(float(v - 0.007) * 1e9) for v in t], np.uint64)
  publication = np.array([int(v) * 1e-9 for v in clocks])
  tape.update(
    carState_t=publication,
    lat_active=np.ones(len(t), bool),
    driver_torque=np.zeros(len(t)),
    fault_temporary=np.zeros(len(t), bool),
    fault_permanent=np.zeros(len(t), bool),
    published_delay=np.full(len(t), 0.1),
    lat_input_t=publication.copy(),
    lat_input_valid=np.ones(len(t), bool),
  )
  tape["carControl_t"] = t - 0.008
  plan = np.zeros((n, 33, 15))
  plan[:, :, 3] = 20.0
  plan[:, :, 6] = tape["rdf_acceleration"][:, :, 0]
  plan[:, :, 14] = 0.04
  tape.update(rdf_plan=plan, rdf_desired_curvature=np.full(n, 0.00003))
  raw_params = {**params(), "BlinkerLateralReengageDelay": "0", "BlinkerMinLateralControlSpeed": "0", "BlinkerPauseLateralControl": "0", "IsMetric": "1"}
  anchor = float(t[0])
  env = {key: window[key] for key in ("route", "cohort", "source_log_sha256", "opendbc_gitlink")}
  env.update(
    cp_raw=np.frombuffer(json.dumps(cp).encode(), np.uint8).copy(),
    sp_raw=np.frombuffer(json.dumps(sp).encode(), np.uint8).copy(),
    cp_t=-1.0,
    sp_t=anchor - 1.0,
    static_t=anchor - 1.0,
    lp={"values": LP.copy(), "t": anchor - 0.06, "valid": True},
    params={"values": raw_params, "t": anchor - 1.0, "valid": True},
    initial={"state": {"v": 20.0, "a": 0.0, "angle": 0.0, "yaw": 0.0, "vendor_rate": 0.0}, "t": anchor - 0.06, "valid": True},
    roll={"value": 0.01, "t": anchor - 0.06, "valid": True},
    car_state_t=float(publication[0]),
    location_quality={
      "t": anchor - 0.06,
      "valid": True,
      "yaw_std": 0.01,
      "roll_std": 0.01,
      "yaw_inner_valid": 1,
      "roll_inner_valid": 1,
      "inputs_ok": 1,
      "sensors_ok": 1,
      "device_stable": 1,
      "status": 2,
    },
    learner={"t": anchor - 0.06, "valid": False, "admitted": False, "useParams": False, "frequency_valid": False, "reason": "fixture"},
    initialization={"previous_curvature": 0.0, "last_transmitted_counts": 0, "angle_counter": 0, "steer_limited": False},
    history={"t": anchor + np.array([-0.06, -0.05]), "command": np.zeros(2, int), "request": np.ones(2, bool), "valid": np.ones(2, bool)},
    censor_t=None,
    static_identity_sha256=old.static_identity(cp, sp),
    policy_identity_sha256=old.policy_identity(raw_params),
    source_evidence={
      "verified": True,
      "source_log_sha256": window["source_log_sha256"],
      "schema_sha256": "3" * 64,
      "extractor_sha256": "4" * 64,
      "packet_sha256": "5" * 64,
    },
  )
  env["lateral_delay_witness"] = {
    "frame_id": tape["rdf_frame_id"].copy(),
    "capture_t": tape["rdf_narrow_t"].copy(),
    "delay": np.full(n, 0.1),
    "source_publication_t": tape["rdf_narrow_t"] - 0.01,
    "source_valid": np.ones(n, bool),
    "source_kind": np.full(n, "lateralDelay"),
    "source_log_sha256": np.full(n, window["source_log_sha256"]),
    "packet_sha256": "5" * 64,
    "packet_lat_delay": np.full(n, 0.1),
    "packet_lateral_control_params": np.tile([20.0, 0.1], (n, 1)),
  }
  prefix = anchor + np.array([-0.04, -0.03, -0.02, -0.01])
  nodes = np.r_[prefix, t]
  raw_clocks = np.array([round(float(v - 0.007) * 1e9) for v in np.r_[nodes, t[-1] + 0.01]], np.uint64)
  raw = {
    "kind": "raw_car_state_eligibility_v1",
    "route": window["route"],
    "source_log_sha256": window["source_log_sha256"],
    "t_ns": raw_clocks,
    "event_valid": np.ones(len(raw_clocks), bool),
    "can_valid": np.ones(len(raw_clocks), bool),
    "standstill": np.zeros(len(raw_clocks), bool),
    "v_ego": np.full(len(raw_clocks), 20.0),
  }
  raw["sha256"] = old.long.value_hash(raw)
  env["raw_car_state_witness"] = raw
  calibration = {
    "packet_sha256": "5" * 64,
    "frame_id": tape["rdf_frame_id"].copy(),
    "decision_t": tape["rdf_input_t"].copy(),
    "narrow_t": tape["rdf_narrow_t"].copy(),
    "wide_t": tape["rdf_input_t"].copy(),
    "decision_time_origin": "max_camera_eof",
    "calibration_cutoff": "decision",
    "calibration_t": tape["rdf_input_t"] - 0.1,
    "valid": np.ones(n, bool),
    "tfm": np.tile(np.eye(3), (n, 1, 1)),
    "big_tfm": np.tile(np.eye(3), (n, 1, 1)),
  }
  calibration["sha256"] = old.long.value_hash(calibration)
  env["calibration_witness"] = calibration
  window["model_inputs"]["packet_calibration"] = deepcopy(calibration)
  window["end_t"] = float(t[-1] + 0.01)
  schedule = {
    "source_log_sha256": window["source_log_sha256"],
    "card_source_sha256": runner.CARD_SHA,
    "cold_start": True,
    "phase_known": False,
    "prefix_node_t": prefix,
    "prefix_state_assumption": {"mode": "fixed_initial_five_state", "validated": False},
    "cs_capture_t": nodes - 0.0097,
    "compute_t": nodes - 0.006,
    "cc_publish_t": nodes - 0.005,
    "consume_t": nodes - 0.009,
    "co_publish_t": nodes - 0.008,
    "cs_publish_t": nodes - 0.007,
    "apply_t": nodes - 0.004,
    "apply_call": np.ones(len(nodes), bool),
    "co_valid": np.ones(len(nodes), bool),
    "startup_empty": np.zeros(len(nodes), bool),
    "cc_valid": np.ones(len(nodes), bool),
    "co_seen": np.ones(len(nodes), bool),
    "co_alive": np.ones(len(nodes), bool),
    "co_frequency_valid": np.ones(len(nodes), bool),
  }
  schedule["apply_call"][0] = False
  schedule["startup_empty"][0] = True
  for key in ("co_valid", "co_seen", "co_alive", "co_frequency_valid"):
    schedule[key][0] = False
  schedule["inputs"] = {k: np.r_[np.repeat(tape[k][0], len(prefix)), tape[k]] for k in runner.INPUT_FIELDS}
  for key, offset in (("carState_t", 0.007), ("carControl_t", 0.008), ("lat_input_t", 0.007)):
    schedule["inputs"][key][:len(prefix)] = prefix - offset
  raw_publication = np.array([int(v) * 1e-9 for v in raw_clocks])
  for key in ("carState_t", "lat_input_t"):
    schedule["inputs"][key][:len(prefix)] = raw_publication[:len(prefix)]
  schedule["cs_publish_t"] = raw_publication[:-1].copy()
  schedule["inputs"]["long_active"][0] = schedule["inputs"]["lat_active"][0] = False
  for name in runner.EVENTS:
    schedule[name + "_seq"] = np.arange(len(nodes), dtype=np.int64) * 10 + runner.EVENTS.index(name)
  schedule["sha256"] = old.long.value_hash(schedule)
  env["publication_schedule"] = schedule
  model = {
    "kind": plant.KIND,
    "features": list(plant.FEATURES),
    "cohort": window["cohort"],
    "opendbc_gitlink": window["opendbc_gitlink"],
    "static_identity_sha256": env["static_identity_sha256"],
    "policy_identity_sha256": env["policy_identity_sha256"],
    "source_eligible": True,
    "source_pin_verified": True,
    "static_policy_timing_verified": True,
    "macro_dt_s": 0.05,
    "native_dt_s": 0.01,
    "micro_nodes": 5,
    "split": deepcopy(report["split"]),
    "record_hashes": ["7" * 64, "8" * 64],
    "min_samples": 100,
    "min_routes": 2,
    "supported_cells": 2,
    "selection_eligible": False,
    "production_enabled": False,
    "feature_min": [-500.0] * 21,
    "feature_max": [500.0] * 21,
    "edges": [[] for _ in range(21)],
    "cells": {},
  }
  model["edges"][16] = [3.0]
  for key, delta, bounds in (("low", [0.0, 0.0, 0.0, 0.0, 0.0], (-500.0, 3.0)), ("high", [-0.5, -0.2, 0.0, 0.001, 0.0], (4.0, 500.0))):
    bins = [0] * 21
    bins[16] = int(key == "high")
    lo, hi = [-500.0] * 21, [500.0] * 21
    lo[16], hi[16] = bounds
    model["cells"][":".join(map(str, bins))] = {
      "supported": True,
      "samples": 100,
      "routes": ["route-t", "route-u"],
      "feature_min": lo,
      "feature_max": hi,
      "joint_delta_median": delta,
    }
  model["model_sha256"] = plant.identity_hash(model)
  return window, report, env, model, lr, lat, out


def evaluate(args, **kwargs):
  return runner.evaluate_two_way(*args, runner.bind_proof(*args), lateral_grid=GRID, **kwargs)


def test_steering_changes_shared_v_a_and_later_native_long_feedback_without_scalar_plant():
  args = fixture()
  FixtureController.calls.clear()
  baseline = evaluate(args)
  candidate = evaluate(args, parameters={"long": None, "lateral": {"factor": 2.0, "friction": 0.1}})
  assert baseline["status"] == candidate["status"] == "complete"
  assert not np.array_equal(baseline["trace"]["v"], candidate["trace"]["v"])
  assert not np.array_equal(baseline["trace"]["a"], candidate["trace"]["a"])
  assert not np.array_equal(baseline["trace"]["accel_request"], candidate["trace"]["accel_request"])
  np.testing.assert_array_equal(candidate["trace"]["v"][:5], np.full(5, 20.0))
  assert candidate["trace"]["v"][5] == 19.5 and candidate["trace"]["a"][5] == -0.2
  assert candidate["complete_controller_candidate"] is None


def test_consume_precedes_publish_and_apply_uses_old_cc_even_when_new_cc_arrives():
  args = fixture()
  result = evaluate(args)
  trace = result["trace"]
  assert result["status"] == "complete"
  assert trace["consumed_cc_index"][1] == 0 and trace["requested_cc_index"][1] == 1
  assert trace["applied_accel_request"][1] == trace["accel_request"][0]
  assert trace["apply_call"][0] and trace["can_torque"][0] >= 0
  first_prefix = -len(args[2]["publication_schedule"]["prefix_node_t"])
  startup = [e["publication"] for e in result["event_attempts"] if e["node_index"] == first_prefix and e["event"] == "co_publish"]
  assert len(startup) == 1 and startup[0]["published_event_valid"] is False and startup[0]["published_accel"] == 0.0
  assert trace["published_accel"][2] == trace["held_accel_after_apply"][1]


def test_future_recorded_cs_co_and_slowing_do_not_change_frozen_targets_or_population():
  args = fixture()
  before = evaluate(args)
  args[0]["tape"].update(v=np.full(100, 99.0), a=np.full(100, 9.0), recorded_output=np.full(100, -3.0), steering_angle=np.full(100, 999.0))
  after = evaluate(args)
  changed = evaluate(args, parameters={"long": None, "lateral": {"factor": 2.0, "friction": 0.1}})
  for key in ("v", "a", "native_torque", "can_torque", "accel_request", "published_accel"):
    np.testing.assert_array_equal(before["trace"][key], after["trace"][key])
  np.testing.assert_array_equal(before["trace"]["target_ay"], changed["trace"]["target_ay"])
  assert before["turns"]["reference_direction"] == changed["turns"]["reference_direction"]
  assert before["turns"]["reference_phase"] == changed["turns"]["reference_phase"]


@pytest.mark.parametrize("fault", ["source", "calibration", "future_schedule", "consume_order", "test", "latest_deadline"])
def test_admission_and_latest_output_fail_before_any_factory(monkeypatch, fault):
  args = fixture()
  w, _, env, model, lr, lat, out = args
  if fault == "source":
    model["source_eligible"] = False
    model["model_sha256"] = plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  elif fault == "calibration":
    env["calibration_witness"]["valid"][0] = False
  elif fault == "future_schedule":
    env["publication_schedule"]["cc_publish_t"][1] = w["tape"]["t"][1] + 0.1
  elif fault == "consume_order":
    env["publication_schedule"]["consume_t"][1] = env["publication_schedule"]["co_publish_t"][1] + 0.0001
  elif fault == "test":
    w["partition"] = "test"
  else:
    w["tape"]["rdf_deadline_admissible"][133] = False
  for key in ("calibration_witness", "publication_schedule"):
    env[key]["sha256"] = old.long.value_hash({k: v for k, v in env[key].items() if k != "sha256"})
  monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("factory before admission"))
  monkeypatch.setattr(out, "create", lambda *a, **k: pytest.fail("factory before admission"))
  with pytest.raises((runner.TwoWayError, old.long.SearchError)):
    evaluate(args)


def test_unsupported_terminal_macro_and_midmacro_censor_preserve_attempts_not_support():
  args = fixture()
  args[2]["censor_t"] = float(args[0]["tape"]["t"][7])
  result = evaluate(args)
  assert result["used_steps"] == 7 and result["supported_steps"] == 5 and result["macro_indices"] == [0]
  assert result["status"] == "censored"
  summary = runner.summarize({result["window_id"]: result}, [result["window_id"]])
  assert summary["missing_windows"] == [result["window_id"]] and summary["turns"]["missing_windows"] == [result["window_id"]]


@pytest.mark.parametrize("fault", ["transform_rehashed", "unknown_origin", "empty_cohort", "stale_calibration"])
def test_independent_consumed_packet_calibration_and_nonempty_source_before_factory(monkeypatch, fault):
  args = fixture()
  w, report, env, model, _, lat, out = args
  if fault == "transform_rehashed":
    env["calibration_witness"]["tfm"][0, 0, 0] += 1
  elif fault in ("unknown_origin", "stale_calibration"):
    for value in (env["calibration_witness"], w["model_inputs"]["packet_calibration"]):
      if fault == "unknown_origin":
        value["decision_time_origin"] = "unknown"
      else:
        value["calibration_t"] = value["decision_t"]-1.
      value["sha256"] = old.long.value_hash({k: v for k, v in value.items() if k != "sha256"})
  else:
    w["cohort"] = report["cohort"] = env["cohort"] = model["cohort"] = None
    model["model_sha256"] = plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  env["calibration_witness"]["sha256"] = old.long.value_hash({k: v for k, v in env["calibration_witness"].items() if k != "sha256"})
  monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("factory before source/calibration admission"))
  monkeypatch.setattr(out, "create", lambda *a, **k: pytest.fail("factory before source/calibration admission"))
  with pytest.raises(runner.TwoWayError):
    evaluate(args)


def test_fifty_ms_delta_never_supplies_earlier_next_macro_compute_or_apply():
  args = fixture()
  FixtureController.calls.clear()
  result = evaluate(args, parameters={"long": None, "lateral": {"factor": 2.0, "friction": 0.1}})
  assert result["status"] == "complete"
  assert result["trace"]["v"][5] == 19.5
  # Native calculation5 is before macro boundary at node5; it must still see old state.
  assert FixtureController.calls[9][:2] == (20.0, 0.0)  # Four prefix calculations precede scored macros.
  assert FixtureController.calls[10][:2] == (19.5, -0.2)
  assert result["trace"]["output_simulated_a"][5] == 0.0


def test_end_censor_does_not_count_forty_ms_terminal_commands_as_fifty_ms_support():
  args = fixture()
  args[2]["censor_t"] = args[0]["end_t"] - 0.005
  result = evaluate(args)
  assert result["used_steps"] == 100 and result["supported_steps"] == 95
  assert result["status"] == "censored" and result["macro_indices"][-1] == 90


def rebind_schedule(args):
  s = args[2]["publication_schedule"]
  s["sha256"] = old.long.value_hash({k: v for k, v in s.items() if k != "sha256"})


def test_skip_apply_preserves_hardware_history_and_output_phase_with_computation_continuing():
  args = fixture()
  s = args[2]["publication_schedule"]
  first = len(s["prefix_node_t"])
  s["apply_call"][first+3] = False
  rebind_schedule(args)
  result = evaluate(args)
  assert result["status"] == "complete"
  assert result["trace"]["can_torque"][3] == result["trace"]["can_torque"][2]
  assert len([e for e in result["event_attempts"] if e["event"] == "compute"]) == 104


@pytest.mark.parametrize("fault", ["startup_valid", "missing_prefix", "initial_after_capture", "future_cs_compute"])
def test_no_invalid_startup_or_future_snapshot_before_factories(monkeypatch, fault):
  args = fixture()
  s = args[2]["publication_schedule"]
  if fault == "startup_valid":
    s["co_valid"][0] = True
  elif fault == "missing_prefix":
    s["prefix_node_t"] = np.array([])
  elif fault == "initial_after_capture":
    args[2]["initial"]["t"] = s["compute_t"][0] - 0.001
  else:
    # Source CS published after native calculation cannot be silently consumed.
    s["inputs"]["carState_t"][6] = s["compute_t"][6] + 0.0001
  rebind_schedule(args)
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("factory before prefix/snapshot refusal"))
  with pytest.raises(runner.TwoWayError):
    evaluate(args)


def test_failed_terminal_macro_keeps_commands_and_full_fixed_common_population():
  args = fixture()
  model = args[3]
  # First macro is low-supported; subsequent higher CAN values enter a retained unsupported cell.
  for key, cell in model["cells"].items():
    if key.split(":")[16] == "1":
      cell["supported"] = False
  model["supported_cells"] = 1
  model["model_sha256"] = plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  result = evaluate(args)
  assert result["status"] == "unsupported" and result["reason"] == "macro_unsupported:cell_support"
  assert result["used_steps"] > result["supported_steps"]
  assert np.isfinite(result["trace"]["can_torque"][: result["used_steps"]]).all()
  assert result["macro_indices"] == list(range(0, result["supported_steps"], 5))


def test_finite_grid_retains_malformed_and_failed_rows_without_long_cache(monkeypatch):
  args = fixture()
  w, report, env, model, lr, lat, out = args
  monkeypatch.setattr(old.LongCache, "evaluate", lambda *a, **k: pytest.fail("separable long cache used"))
  monkeypatch.setattr(old.long, "evaluate_trial", lambda *a, **k: pytest.fail("scalar long plant used"))
  good = {"long": None, "lateral": {"factor": 2.0, "friction": 0.1}}
  bad = {"long": {"ki_v": [99.0, 99.0], "stop_accel": -2.0}, "lateral": None}
  result = runner.validation_search([w], report, {w["id"]: env}, model, lr, lat, out, {w["id"]: runner.bind_proof(*args)}, grid=[good, bad], lateral_grid=GRID)
  assert len(result["rows"]) == 2 and result["baseline"]["results"][w["id"]]["status"] == "complete"
  assert result["rows"][1]["results"][w["id"]]["status"] == "rejected"
  assert all(row["common_window_ids"] == [w["id"]] for row in [result["baseline"], *result["rows"]])
  assert result["rows"][1]["metrics"]["missing_windows"] == [w["id"]]
  assert result["selection"] is None and result["complete_controller_candidate"] is None


def test_both_turn_directions_and_entry_steady_exit_populations_are_frozen_when_speed_changes():
  args = fixture()
  tape = args[0]["tape"]
  waveform = np.array([-.04, -.08, -.08, -.04, 0., .04, .08, .08, .04, 0.])
  tape["rdf_plan"][133:, :, 14] = np.resize(waveform, len(tape["rdf_plan"])-133)[:, None]
  tape["rdf_acceleration"][:143, :, 0] = -.2
  tape["rdf_acceleration"][143:, :, 0] = .2
  before = evaluate(args)
  after = evaluate(args, parameters={"long": None, "lateral": {"factor": 2., "friction": .1}})
  assert before["status"] == after["status"] == "complete"
  assert set(before["turns"]["reference_direction"]) >= {"left", "right"}
  assert set(before["turns"]["reference_phase"]) >= {"entry", "steady", "exit"}
  assert before["turns"]["reference_sha256"] == after["turns"]["reference_sha256"]
  for side in ("left", "right"):
    for phase in ("entry", "steady", "exit"):
      assert before["turns"]["phases"][side][phase]["attempted_rows"] == after["turns"]["phases"][side][phase]["attempted_rows"]
  np.testing.assert_array_equal(before["trace"]["target_ay"], after["trace"]["target_ay"])
  assert not np.array_equal(before["trace"]["v"], after["trace"]["v"])


def test_cross_node_calculation_order_is_explicitly_refused_without_forcing_times(monkeypatch):
  args = fixture()
  schedule = args[2]["publication_schedule"]
  schedule["compute_t"] += .007
  schedule["cc_publish_t"] += .007
  rebind_schedule(args)
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("factory before unsupported event order"))
  with pytest.raises(runner.TwoWayError, match="event order|crossing"):
    evaluate(args)


def test_output_native_phase_does_not_advance_at_co_publication_before_actual_apply(monkeypatch):
  args = fixture()
  holder, observed = {}, []
  original_out, original_lat = args[6].create, args[5].create
  def create_output(*a, **k):
    holder["output"] = original_out(*a, **k)
    return holder["output"]
  def create_lateral(*a, **k):
    handle = original_lat(*a, **k)
    calculate = handle.compute
    def compute(**values):
      observed.append(holder["output"].receipt()["apply_calls"])
      return calculate(**values)
    handle.compute = compute
    return handle
  monkeypatch.setattr(args[6], "create", create_output)
  monkeypatch.setattr(args[5], "create", create_lateral)
  result = evaluate(args)
  assert result["status"] == "complete"
  flags = args[2]["publication_schedule"]["apply_call"]
  assert observed == [int(flags[:i].sum()) for i in range(len(flags))]


def test_censor_between_co_and_apply_retains_earlier_events_without_hypothetical_apply():
  args = fixture()
  schedule = args[2]["publication_schedule"]
  first = len(schedule["prefix_node_t"])
  args[2]["censor_t"] = float(schedule["compute_t"][first]+.0005)
  result = evaluate(args)
  retained = [e for e in result["event_attempts"] if e["node_index"] == 0]
  assert {e.get("event") for e in retained} >= {"consume", "co_publish", "cs_publish", "compute"}
  assert "apply" not in {e.get("event") for e in retained}
  assert result["output_receipt"]["apply_calls"] == int(schedule["apply_call"][:first].sum())
  assert result["status"] == "censored" and result["supported_steps"] == 0


def test_more_than_twenty_ms_without_can_apply_is_unsupported_not_held_dynamics():
  args = fixture()
  schedule = args[2]["publication_schedule"]
  first = len(schedule["prefix_node_t"])
  schedule["apply_call"][first+3:first+8] = False
  rebind_schedule(args)
  result = evaluate(args)
  assert result["status"] == "unsupported" and "CAN" in result["reason"]
  assert result["used_steps"] < 8 and result["supported_steps"] == 0


def test_native_limiter_history_cannot_come_from_after_first_prefix_use(monkeypatch):
  args = fixture()
  args[2]["history"]["t"] = args[0]["tape"]["t"][0]+np.array([-.02, -.01])
  monkeypatch.setattr(args[5], "create", lambda *a, **k: pytest.fail("factory before future limiter-prefix refusal"))
  with pytest.raises(runner.TwoWayError):
    evaluate(args)


def test_missing_native_output_static_field_refused_before_any_controller_factory(monkeypatch):
  args = fixture()
  window, _, env, model, _, lat, _ = args
  del window["cp"]["radarUnavailable"]
  env["cp_raw"] = np.frombuffer(json.dumps(window["cp"]).encode(), np.uint8).copy()
  env["static_identity_sha256"] = model["static_identity_sha256"] = old.static_identity(window["cp"], window["sp"])
  model["model_sha256"] = plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  monkeypatch.setattr(lat, "create", lambda *a, **k: pytest.fail("factory before complete native static admission"))
  with pytest.raises(runner.TwoWayError):
    evaluate(args)


@pytest.mark.skipif(os.name != "posix" or os.environ.get("RUN_NATIVE_TWO_WAY_API") != "1",
                    reason="Opt-in owner-serialized actual three-adapter API case; synthetic inputs only")
def test_actual_three_adapter_two_way_feedback_and_causal_publications():
  """Actual native composition, not empirical vehicle or candidate evidence.

  These fixed arbitrary deltas deliberately exercise all five state channels.
  Their magnitudes/signs and factor2/3 comparison are declared before execution;
  lack of actual CAN/feedback divergence is a failure, never a reason to retune
  this fixture. The support counts/routes below are synthetic API labels only.
  """
  import hashlib
  from pathlib import Path

  repo = Path(os.environ["NATIVE_TWO_WAY_SOURCE_ROOT"]).resolve()
  manifest_bytes = Path(os.environ["NATIVE_TWO_WAY_SOURCE_MANIFEST"]).read_bytes()
  expected_digest = os.environ["NATIVE_TWO_WAY_SOURCE_MANIFEST_SHA256"]
  assert len(expected_digest) == 64 and all(c in "0123456789abcdef" for c in expected_digest)
  assert hashlib.sha256(manifest_bytes).hexdigest() == expected_digest
  manifest = json.loads(manifest_bytes)
  assert manifest["research_helpers"] == runner.helper_hashes()
  assert manifest["test_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  pin, head = manifest["opendbc_gitlink"], manifest["superproject_head"]
  # The independent host manifest is the expected content, never a target-side
  # source_hashes() calculation substituted as its own proof. Output loader
  # checks bootstrap and cached namespace origins before broad native imports.
  out = runner.output.load_runtime(repo, manifest["output_sources"], pin, head,
                                   expected_bootstrap=manifest["output_bootstrap"])
  lat = runner.lateral.load_runtime(repo, manifest["lateral_sources"], pin)
  lr = runner.long.load_native_runtime(repo, manifest["long_sources"], pin)
  assert lr.actual_native and lat.bindings.actual_native and out.bindings.actual_native

  from opendbc.car.structs import car
  from opendbc.car.hyundai.values import HyundaiFlags
  from openpilot.cereal import custom

  cp = car.CarParams.new_message(
    carFingerprint="HYUNDAI_ELANTRA_2024", brand="hyundai", radarUnavailable=False,
    mass=1360., wheelbase=2.72, centerToFront=1.088, rotationalInertia=2500.,
    steerRatio=13.5, steerRatioRear=0., tireStiffnessFront=80000., tireStiffnessRear=80000.,
    steerActuatorDelay=.1, steerLimitTimer=.4, minSteerSpeed=0., steerAtStandstill=False,
    steerControlType="torque", flags=int(HyundaiFlags.RAISED_LIMITS),
    openpilotLongitudinalControl=True, longitudinalActuatorDelay=.5, stopAccel=-2.)
  torque = cp.lateralTuning.init("torque")
  torque.latAccelFactor, torque.latAccelOffset, torque.friction = 2.5, 0., .1
  cp.longitudinalTuning.kiBP, cp.longitudinalTuning.kiV = [0., 30.], [.05, .1]
  firmware = cp.init("carFw", 1)[0]
  firmware.ecu, firmware.fwVersion = "eps", b"\x00\xff\x80"
  sp = custom.CarParamsSP.new_message(enableGasInterceptor=False)
  sp.neuralNetworkLateralControl.model.path = str(repo / runner.lateral.ASSET)

  args = list(fixture())  # Setup only; no fixture controller/predictor is invoked.
  window, report, env, model = args[:4]
  args[4:] = [lr, lat, out]
  window.update(cp=cp.to_dict(), sp=sp.to_dict(), opendbc_gitlink=pin)
  env.update(cp_raw=np.frombuffer(cp.to_bytes(), np.uint8).copy(),
             sp_raw=np.frombuffer(sp.to_bytes(), np.uint8).copy(), opendbc_gitlink=pin,
             static_identity_sha256=old.static_identity(window["cp"], window["sp"]))
  model.update(opendbc_gitlink=pin, static_identity_sha256=env["static_identity_sha256"])
  assert window["cp"]["carFw"][0]["fwVersion"] == b"\x00\xff\x80"

  tape = window["tape"]
  anchor, native_steps, rdf_rows = float(tape["t"][0]), 600, 280
  t = anchor + np.arange(native_steps) * .01
  horizon = np.asarray(lat.bindings.horizon)
  for key, value in list(tape.items()):
    array = np.asarray(value)
    if not key.startswith("rdf_") and array.shape == (100,):
      tape[key] = np.full(native_steps, array[0], dtype=array.dtype)
    elif key.startswith("rdf_") and array.ndim and array.shape[0] == 170:
      tape[key] = np.repeat(array[:1], rdf_rows, axis=0)
  tape["t"] = t
  capture = np.arange(rdf_rows) * .05
  tape.update(rdf_input_t=capture+.001, rdf_narrow_t=capture, rdf_output_ready_t=capture+.02,
              rdf_frame_id=np.arange(rdf_rows), rdf_scored_mask=np.arange(rdf_rows) >= 132,
              rdf_horizon=horizon, rdf_desired_curvature=np.full(rdf_rows, .0003))
  tape["rdf_acceleration"][:, :, 0] = .1 + horizon
  tape["rdf_plan"][:, :, 3] = 20.
  tape["rdf_plan"][:, :, 6] = .1 + horizon
  tape["rdf_plan"][:, :, 14] = .04
  for name in ("action_t", "packet_action_t"):
    window["model_inputs"][name] = tape["rdf_" + name].copy()
  window["end_t"] = float(t[-1] + .01)

  prefix = env["publication_schedule"]["prefix_node_t"].copy()
  nodes = np.r_[prefix, t]
  raw_clock = np.array([round(float(node-.007)*1e9) for node in np.r_[nodes, window["end_t"]]], np.uint64)
  publication = np.array([int(ns)*1e-9 for ns in raw_clock])  # Exact frozen raw producer convention.
  tape["carState_t"], tape["lat_input_t"] = publication[len(prefix):-1].copy(), publication[len(prefix):-1].copy()
  tape["carControl_t"] = t-.008
  env["car_state_t"] = float(tape["carState_t"][0])
  raw = env["raw_car_state_witness"]
  raw.update(t_ns=raw_clock, event_valid=np.ones(len(raw_clock), bool), can_valid=np.ones(len(raw_clock), bool),
             standstill=np.zeros(len(raw_clock), bool), v_ego=np.full(len(raw_clock), 20.))
  raw["sha256"] = runner._body_hash(raw)
  calibration = env["calibration_witness"]
  calibration.update(frame_id=tape["rdf_frame_id"].copy(), decision_t=tape["rdf_input_t"].copy(),
                     narrow_t=capture.copy(), wide_t=tape["rdf_input_t"].copy(), calibration_t=tape["rdf_input_t"]-.1,
                     valid=np.ones(rdf_rows, bool), tfm=np.tile(np.eye(3), (rdf_rows, 1, 1)),
                     big_tfm=np.tile(np.eye(3), (rdf_rows, 1, 1)))
  calibration["sha256"] = runner._body_hash(calibration)
  window["model_inputs"]["packet_calibration"] = deepcopy(calibration)
  witness = env["lateral_delay_witness"]
  witness.update(frame_id=tape["rdf_frame_id"].copy(), capture_t=capture.copy(), delay=np.full(rdf_rows, .1),
                 source_publication_t=capture-.01, source_valid=np.ones(rdf_rows, bool),
                 source_kind=np.full(rdf_rows, "lateralDelay"), source_log_sha256=np.full(rdf_rows, window["source_log_sha256"]),
                 packet_lat_delay=np.full(rdf_rows, .1), packet_lateral_control_params=np.tile([20., .1], (rdf_rows, 1)))

  schedule = env["publication_schedule"]
  offsets = {"cs_capture": .0097, "consume": .009, "co_publish": .008, "cs_publish": .007,
             "compute": .006, "cc_publish": .005, "apply": .004}
  for name in runner.EVENTS:
    schedule[name+"_t"] = nodes-offsets[name]
    schedule[name+"_seq"] = np.arange(len(nodes), dtype=np.int64)*10 + runner.EVENTS.index(name)
  schedule["cs_publish_t"] = publication[:-1].copy()
  for name in ("apply_call", "co_valid", "cc_valid", "co_seen", "co_alive", "co_frequency_valid"):
    schedule[name] = np.ones(len(nodes), bool)
  schedule["startup_empty"] = np.zeros(len(nodes), bool)
  schedule["startup_empty"][0] = True
  for name in ("apply_call", "co_valid", "co_seen", "co_alive", "co_frequency_valid"):
    schedule[name][0] = False
  skipped = 207  # One fixed skipped apply; never a prolonged stale-CAN experiment.
  schedule["apply_call"][len(prefix)+skipped] = False
  schedule["inputs"] = {key: np.r_[np.repeat(tape[key][0], len(prefix)), tape[key]] for key in runner.INPUT_FIELDS}
  for key in ("carState_t", "lat_input_t"):
    schedule["inputs"][key][:len(prefix)] = publication[:len(prefix)]
  schedule["inputs"]["carControl_t"][:len(prefix)] = prefix-.008
  schedule["inputs"]["long_active"][0] = schedule["inputs"]["lat_active"][0] = False
  schedule["sha256"] = runner._body_hash(schedule)

  # Unpatched median predictor: one predeclared cell for each integer CAN proxy.
  # Labels100/two routes are code fixtures, not independently measured support.
  lower, upper = [-500.]*21, [500.]*21
  edges, cells = [[] for _ in range(21)], {}
  edges[16] = np.arange(-408.5, 409., 1.).tolist()
  for count in range(-409, 410):
    lo, hi, bins = lower.copy(), upper.copy(), [0]*21
    lo[16], hi[16], bins[16] = count-.5, count+.5, count+409
    cells[":".join(map(str, bins))] = {
      "supported": True, "samples": 100, "routes": report["split"]["train"].copy(),
      "feature_min": lo, "feature_max": hi,
      "joint_delta_median": [-abs(count)*1e-6, -abs(count)*1e-5, count*1e-5, count*1e-7, abs(count)*1e-6],
    }
  model.update(feature_min=lower, feature_max=upper, edges=edges, cells=cells, supported_cells=len(cells),
               fixture_only=True, fixture_delta_contract="arbitrary predeclared CAN-conditioned five-state API deltas; no physics proof")
  model["model_sha256"] = plant.identity_hash({k: v for k, v in model.items() if k != "model_sha256"})
  assert runner.plant.predict_macro is plant.predict_macro
  candidate_a = {"long": None, "lateral": {"factor": 2., "friction": .1}}
  candidate_b = {"long": None, "lateral": {"factor": 3., "friction": .1}}
  first = evaluate(args, parameters=candidate_a)
  second = evaluate(args, parameters=candidate_b)
  original_window = deepcopy(window)
  # Only unused FUTURE measured columns change. Eligibility/schedule, initial
  # anchor, model plans and raw-CS witness remain byte/content identical.
  window["tape"]["v"][1:] = 77.
  window["tape"]["a"][1:] = -3.
  window["tape"]["steering_angle"] = np.r_[0., np.full(native_steps-1, 77.)]
  window["tape"]["yaw"] = np.r_[0., np.full(native_steps-1, -2.)]
  window["tape"]["recorded_output"] = np.r_[0., np.full(native_steps-1, -3.)]
  last = evaluate(args, parameters=candidate_a)
  for result in (first, second, last):
    assert result["status"] == "complete", (result["reason"], result["used_steps"], result["supported_steps"])
    assert result["used_steps"] == result["supported_steps"] == native_steps
    assert result["macro_indices"] == list(range(0, native_steps, 5))
    receipt, trace = result["output_receipt"], result["trace"]
    assert receipt["actual_native"] and not receipt["failed"]
    assert receipt["source_hashes"] == manifest["output_sources"] and receipt["bootstrap_sources"] == manifest["output_bootstrap"]
    assert receipt["apply_calls"] == int(schedule["apply_call"].sum())
    assert result["native_long_parameters"]["ki_v"] == window["cp"]["longitudinalTuning"]["kiV"]
    np.testing.assert_array_equal(trace["consumed_cc_index"], np.arange(native_steps)-1)
    np.testing.assert_array_equal(trace["published_accel"][1:], trace["held_accel_after_apply"][:-1])
    np.testing.assert_array_equal(trace["published_torque"][1:], trace["can_torque"][:-1]/409.)
    assert trace["held_accel_after_apply"][skipped] == trace["published_accel"][skipped]
    assert trace["can_torque"][skipped] == trace["can_torque"][skipped-1]
    assert np.isnan(trace["applied_accel_request"][skipped])
    assert np.max(np.abs(trace["can_torque"])) <= 409
    assert np.max(np.abs(np.diff(trace["can_torque"]))) <= 7
    assert abs(trace["can_torque"][0]) <= 3*len(prefix)
    for key in plant.STATE_KEYS:
      assert np.ptp(trace[key]) > 0, (key, "actual composition never exercised this synthetic state channel")
      np.testing.assert_array_equal(np.diff(trace[key])[np.arange(1, native_steps) % 5 != 0], 0.)
    computes = [event for event in result["event_attempts"] if event.get("event") == "compute"]
    assert len(computes) == len(nodes) and computes[0]["node_index"] < 0
    assert any(event["publication"]["published_event_valid"] is False for event in result["event_attempts"] if event.get("event") == "co_publish")
    assert not result["selection_eligible"] and result["complete_controller_candidate"] is None and not result["production_enabled"]
    assert "unvalidated_markov_command_history_sufficiency_for_actuator_latency" in result["selection_rejections"]
  for key in first["trace"]:
    np.testing.assert_array_equal(first["trace"][key], last["trace"][key], err_msg=key)
  assert first["event_attempts"] == last["event_attempts"]
  assert first["fixed_reference_sha256"] == second["fixed_reference_sha256"] == last["fixed_reference_sha256"]
  assert runner.long.value_hash(original_window["model_inputs"]) == runner.long.value_hash(window["model_inputs"])
  # Clear the shared initial rate-limit ramp before testing native consequences.
  can_difference = np.flatnonzero(first["trace"]["can_torque"][100:] != second["trace"]["can_torque"][100:])
  assert len(can_difference), "Predeclared factor comparison produced no actual CAN divergence; do not retune fixture"
  after = int(can_difference[0]+100)
  for key in ("v", "a", "native_feedback_v", "native_feedback_a", "accel_request"):
    assert np.any(first["trace"][key][after:] != second["trace"][key][after:]), key
