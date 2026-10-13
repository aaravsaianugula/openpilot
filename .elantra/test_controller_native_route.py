"""Causal plural-source preparation and real-native stop semantics."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller_native_replay import MODEL_SHA, ReplayError, prepare_arrays
from controller_native_route import gate_native_tape, replay_route, validate_route_raw, validate_tape


@pytest.fixture
def episode():
  n, first = 160, 100.
  clock = first + np.arange(n) * .05
  rdf = {"input_t": clock.copy(), "narrow_t": clock.copy(), "output_ready_t": clock + .04,
         "frame_id": np.arange(1000, 1000 + n), "desired_acceleration": np.full(n, .1),
         "should_stop": np.ones(n, bool), "scored_mask": np.arange(n) >= 132,
         "controller_admissible": np.arange(n) >= 132, "deadline_admissible": np.arange(n) >= 132,
         "model_sha256": np.asarray(MODEL_SHA)}
  cp = {"carFingerprint": "HYUNDAI_ELANTRA_2024", "openpilotLongitudinalControl": True,
        "stopAccel": -2., "longitudinalTuning": {"kiBP": [0.], "kiV": [0.]}}
  sp = {"enableGasInterceptor": False}
  chunks = [{"segment": f"000001b7--1972f5d799--{25 + i}", "source_log_sha256": str(i + 1) * 64,
             "frames": 80, "frame_ids": [1000 + i * 80, 1079 + i * 80],
             "road_presentation_indices": [1120 if i == 0 else 0, 1199 if i == 0 else 79]} for i in range(2)]
  records = [{**c, "full_static_snapshots": {
    key: [{"t": 99. if i == 0 else 110., "semantic_sha256": letter * 64}]
    for key, letter in (("car_params_snapshots", "a"), ("car_params_sp_snapshots", "b"))}}
    for i, c in enumerate(chunks)]
  statics = [{"carParams": copy.deepcopy(cp), "carParamsSP": copy.deepcopy(sp)} for _ in chunks]
  captures = [{int(frame): float(t) for frame, t in zip(rdf["frame_id"][i * 80:(i + 1) * 80], clock[i * 80:(i + 1) * 80], strict=True)}
              for i in range(2)]
  packet = {"source_chunk_index": np.repeat([0, 1], 80), "source_num": np.repeat([25, 26], 80),
            "road_index": np.r_[np.arange(1120, 1200), np.arange(80)], "frame_id": rdf["frame_id"].copy(),
            "narrow_t": clock.copy()}
  streams = {
    "carState": [{"t": float(t), "valid": True, "v": .1, "a": 0., "brake_pressed": 0., "gas_pressed": 0.,
                  "steering_pressed": 0., "cruise_standstill": 0., "v_cruise": 50.} for t in clock],
    "carControl": [{"t": float(t), "valid": True, "long_active": 1., "recorded_command": -.1} for t in clock],
    "carOutput": [{"t": float(t), "valid": True, "recorded_output": -.1} for t in clock],
    "controlsState": [{"t": float(t), "valid": True, "recorded_state": 2.} for t in clock],
  }
  tape = prepare_arrays(streams, rdf)
  tape["raw_censor_t"] = np.asarray(107.4)
  return {"source_chunks": chunks}, records, statics, captures, packet, rdf, tape


def test_two_source_static_and_capture_contract(episode):
  spec, records, statics, captures, packet, rdf, _ = episode
  result = validate_route_raw(spec, records, statics, captures, packet, rdf)
  assert result["carParams"]["stopAccel"] == -2.
  assert result["carParamsSP"] == {"enableGasInterceptor": False}


@pytest.mark.parametrize("mutation", ["omitted", "swapped", "different_cp", "future_cp", "future_sp", "capture", "duplicate", "local_index", "id"])
def test_plural_contract_cannot_be_replaced_by_primary_only(episode, mutation):
  spec, records, statics, captures, packet, rdf, _ = episode
  if mutation == "omitted":
    records.pop()
  elif mutation == "swapped":
    records.reverse()
  elif mutation == "different_cp":
    statics[1]["carParams"]["stopAccel"] = -1.
  elif mutation in ("future_cp", "future_sp"):
    key = "car_params_snapshots" if mutation == "future_cp" else "car_params_sp_snapshots"
    records[0]["full_static_snapshots"][key][0]["t"] = 101.
  elif mutation == "capture":
    captures[1][1080] += .001
  elif mutation == "duplicate":
    captures[1][1000] = captures[0][1000] + .001
  elif mutation == "local_index":
    packet["road_index"][80] = 80
  else:
    packet["frame_id"][80] += 1
  with pytest.raises(ReplayError):
    validate_route_raw(spec, records, statics, captures, packet, rdf)


def test_masked_latest_ready_output_never_falls_back_to_older_output(episode):
  tape = episode[-1]
  tape["rdf_deadline_admissible"][134] = False
  gated, rejected, indices = gate_native_tape(tape)
  ticks = indices == 134
  assert ticks.any() and rejected[ticks].all()
  assert not gated["carControl_valid"][ticks].any()
  assert tape["carControl_valid"][ticks].all()  # Original telemetry never modified.
  assert not rejected[indices == 133].any()


def test_no_future_target_or_post_brake_controlled_availability(episode):
  tape = episode[-1]
  _, _, indices = gate_native_tape(tape)
  first_ready = tape["rdf_output_ready_t"][132]
  assert np.all(indices[tape["t"] < first_ready] < 132)
  late = indices >= 0
  assert np.all(tape["rdf_output_ready_t"][indices[late]] <= tape["t"][late])
  # An exposure before brake with computation ready after brake is display only.
  tape["raw_censor_t"] = np.asarray(tape["rdf_narrow_t"][145] + .02)
  assert tape["rdf_narrow_t"][145] < float(tape["raw_censor_t"]) < tape["rdf_output_ready_t"][145]


@pytest.mark.parametrize("mutation", ["future_cs", "malformed_flag", "bad_valid_shape", "nonfinite_valid_value", "missing_censor"])
def test_prepared_native_tape_refuses_broken_causality_and_flags(episode, mutation):
  tape = episode[-1]
  if mutation == "future_cs":
    tape["carState_t"][100] = tape["t"][100] + .01
  elif mutation == "malformed_flag":
    tape["brake_pressed"][100] = .5
  elif mutation == "bad_valid_shape":
    tape["carControl_valid"] = np.ones(1, bool)
  elif mutation == "nonfinite_valid_value":
    tape["a"][100] = np.nan
  else:
    del tape["raw_censor_t"]
  with pytest.raises(ReplayError):
    validate_tape(tape)


@pytest.fixture
def native():
  if sys.platform == "win32":
    pytest.skip("real LongControl imports require the authorized POSIX runtime")
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl
  from opendbc.car.hyundai.interface import CarInterface
  return LongControl, CarInterface.get_pid_accel_limits


def test_actual_native_positive_target_stop_true_brakes_and_censor_stays_diagnostic(episode, native):
  _, _, static, _, _, _, tape = episode
  result = replay_route(tape, static[0], *native)
  admitted = result["admitted"]
  assert admitted.any()
  assert (result["native_state"][admitted] == 2).all()
  assert (result["native_accel"][admitted] < 0).all()
  assert result["initialization_uncertain"].all()
  assert result["post_censor_diagnostic"].any()
  assert not result["controlled_command_available"][result["t"] >= float(tape["raw_censor_t"])].any()
  assert np.isnan(result["native_accel"][~admitted]).all()


def test_actual_native_masks_deadline_and_preserves_disengage_brake_standstill(episode, native):
  _, _, static, _, _, _, tape = episode
  tape["rdf_should_stop"][:] = False
  tape["rdf_deadline_admissible"][134] = False
  tape["long_active"][(tape["t"] >= 106.8) & (tape["t"] < 106.9)] = 0.
  tape["cruise_standstill"][(tape["t"] >= 107.) & (tape["t"] < 107.1)] = 1.
  tape["brake_pressed"][(tape["t"] >= 107.1) & (tape["t"] < 107.2)] = 1.
  result = replay_route(tape, static[0], *native)
  rejected = result["model_display_rejected"]
  assert rejected.any() and not result["admitted"][rejected].any()
  off = result["admitted"] & (result["long_active"] == 0)
  assert off.any() and (result["native_state"][off] == 0).all()
  assert (result["native_accel"][off] == 0).all() and result["initialization_uncertain"][off].all()
  assert not result["controlled_command_available"][result["brake_pressed"] == 1].any()
