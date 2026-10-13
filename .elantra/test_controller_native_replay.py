import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller_native_replay import ReplayError, asof_indices, prepare_arrays, replay_native, validate_output, validate_parameters, validate_rdf


def parameters():
  return ({"carFingerprint": "HYUNDAI_ELANTRA_2024", "openpilotLongitudinalControl": True,
           "stopAccel": -2., "longitudinalTuning": {"kiBP": [0.], "kiV": [0.]}},
          {"enableGasInterceptor": False})


def rdf_inputs():
  t = np.arange(150) * .05
  return {"input_t": t, "narrow_t": t, "output_ready_t": t + .04,
          "frame_id": np.arange(150), "desired_acceleration": np.ones(150) * .1,
          "should_stop": np.ones(150, dtype=bool), "scored_mask": np.arange(150) >= 132,
          "model_sha256": np.asarray("52fcf48bfb991f327a8982037eb0855d9a63437d78e9f4828d2be54df0f32567")}


def native_inputs():
  n = 200
  t = 6.6 + np.arange(n) * .01
  result = {"t": t}
  for key in ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise",
              "long_active", "recorded_command", "recorded_output", "recorded_state"):
    result[key] = np.zeros(n)
  result["long_active"][:] = 1
  result["v"][:] = .1
  for service in ("carState", "carControl", "carOutput", "controlsState"):
    result[service + "_t"] = t.copy()
    result[service + "_valid"] = np.ones(n, dtype=bool)
  result.update({"rdf_" + key: value for key, value in rdf_inputs().items()})
  return result


def test_asof_has_no_future_and_does_not_skip_invalid_latest():
  indices, valid = asof_indices(np.array([1., 1.1, 1.4]), np.array([True, False, True]),
                               np.array([.99, 1., 1.12, 1.35, 1.41]), .2)
  assert indices.tolist() == [-1, 0, 1, 1, 2]
  assert valid.tolist() == [False, True, False, False, True]


@pytest.mark.parametrize("missing", ["stopAccel", "longitudinalTuning", "carFingerprint"])
def test_missing_parameter_contract_rejected(missing):
  cp, sp = parameters()
  del cp[missing]
  with pytest.raises(ReplayError):
    validate_parameters(cp, sp)


def test_missing_interceptor_and_bad_integral_arrays_rejected():
  cp, _ = parameters()
  with pytest.raises(ReplayError):
    validate_parameters(cp, {})
  cp["longitudinalTuning"]["kiV"] = []
  with pytest.raises(ReplayError):
    validate_parameters(cp, {"enableGasInterceptor": False})


def test_rdf_warmup_and_ready_time_are_contracts():
  rdf = rdf_inputs()
  validate_rdf(rdf)
  rdf["scored_mask"][131] = True
  with pytest.raises(ReplayError):
    validate_rdf(rdf)
  rdf = rdf_inputs()
  rdf["output_ready_t"][20] = rdf["input_t"][20] - .01
  with pytest.raises(ReplayError):
    validate_rdf(rdf)


def test_prepare_requires_every_raw_service():
  with pytest.raises(ReplayError, match="missing or unsorted"):
    prepare_arrays({}, rdf_inputs())


def test_prepare_keeps_last_native_tick_with_real_camera_jitter():
  tape, rdf = native_inputs(), rdf_inputs()
  rdf["input_t"][-1] -= .0005
  rdf["narrow_t"][-1] -= .0005
  streams = {}
  groups = {"carState": ("v", "a", "brake_pressed", "gas_pressed", "steering_pressed", "cruise_standstill", "v_cruise"),
            "carControl": ("long_active", "recorded_command"), "carOutput": ("recorded_output",),
            "controlsState": ("recorded_state",)}
  for service, fields in groups.items():
    streams[service] = [{"t": 0., "valid": True, **{key: tape[key][0] for key in fields}}]
  result = prepare_arrays(streams, rdf)
  assert len(result["t"]) == 750
  assert result["t"][-1] == pytest.approx(7.49)


def test_existing_output_hardlink_and_raw_source_directory_rejected(tmp_path):
  source = tmp_path / "source.npz"
  source.write_bytes(b"preserve")
  target = tmp_path / "alias.npz"
  target.hardlink_to(source)
  with pytest.raises(ReplayError):
    validate_output(target, [source])
  raw = tmp_path / "raw"
  raw.mkdir()
  with pytest.raises(ReplayError):
    validate_output(raw / "output.npz", [source], raw)
  assert source.read_bytes() == b"preserve"


@pytest.fixture
def native():
  if sys.platform == "win32":
    pytest.skip("actual native imports require POSIX hardware dependencies; exercised on comma CPU")
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl
  from opendbc.car.hyundai.interface import CarInterface
  return LongControl, CarInterface.get_pid_accel_limits


def test_positive_acceleration_stop_true_stays_stopping(native):
  cp, sp = parameters()
  result = replay_native(native_inputs(), cp, sp, *native)
  selected = result["admitted"]
  assert selected.any()
  assert (result["native_state"][selected] == 2).all()
  assert (result["native_accel"][selected] <= 0).all()
  assert result["initialization_uncertain"][selected].all()


def test_no_future_or_stale_model_outputs(native):
  cp, sp = parameters()
  result = replay_native(native_inputs(), cp, sp, *native)
  assert not result["admitted"][:4].any()  # frame132 ready at6.64, never at capture6.6
  assert not result["admitted"][110:].any()  # final capture7.45 expires at7.65
  assert np.isnan(result["native_accel"][~result["admitted"]]).all()


def test_stale_vehicle_signal_breaks_controller_continuity(native):
  cp, sp = parameters()
  tape = native_inputs()
  tape["carState_t"][30:50] -= 1.
  result = replay_native(tape, cp, sp, *native)
  assert not result["admitted"][30:50].any()
  assert result["controller_reset"][50]


def test_disengage_resets_then_cruise_standstill_and_brake_block_start(native):
  cp, sp = parameters()
  tape = native_inputs()
  tape["rdf_should_stop"][:] = False
  tape["long_active"][10:20] = 0
  tape["cruise_standstill"][20:35] = 1
  tape["brake_pressed"][35:45] = 1
  result = replay_native(tape, cp, sp, *native)
  assert (result["native_state"][10:20] == 0).all()
  assert (result["native_accel"][10:20] == 0).all()
  assert not result["initialization_uncertain"][10:20].any()
  assert (result["native_state"][20:45] == 2).all()
  assert (result["native_state"][45:70] == 1).all()
