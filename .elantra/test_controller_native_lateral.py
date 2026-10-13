import numpy as np
import pytest

from controller_native_lateral import (
  ReplayError, RecordedParams, add_capture, admit_model, check_adapter_binding, decode_payload, encode_payloads, hardware_step,
  normalized_digest, validate_lateral_rdf, validate_nn_binding, validate_selection,
)


def parameters():
  return {"EnforceTorqueControl": "1", "TorqueControlTune": "0.0", "LateralJerkTorqueController": "1",
          "NeuralNetworkLateralControl": "0", "TorqueParamsOverrideEnabled": "0", "LagdToggle": "1",
          "LagdValueCache": "0.186", "TorqueParamsOverrideLatAccelFactor": "2.5", "TorqueParamsOverrideFriction": "0.1"}


def rdf():
  t = 100 + np.arange(140) * .05
  return {"model_sha256": np.array("52fcf48bfb991f327a8982037eb0855d9a63437d78e9f4828d2be54df0f32567"),
          "input_t": t, "narrow_t": t.copy(), "output_ready_t": t + .04, "frame_id": np.arange(140),
          "desired_acceleration": np.zeros(140), "desired_curvature": np.zeros(140), "should_stop": np.zeros(140, bool),
          "scored_mask": np.arange(140) >= 132, "history_scored_mask": np.arange(140) >= 132,
          "deadline_admissible": np.ones(140, bool), "controller_admissible": np.arange(140) >= 132,
          "plan": np.zeros((140, 33, 15)), "horizon": np.linspace(0, 10, 33)}


def test_recorded_float_selection_and_missing_setting():
  p = RecordedParams(parameters())
  assert type(p.get("TorqueControlTune")) is float
  assert p.get("TorqueControlTune") == 0.0
  validate_selection(p)
  with pytest.raises(ReplayError, match="missing recorded"):
    p.get_bool("IsMetric")
  with pytest.raises(ReplayError, match="read-only"):
    p.put_bool("EnforceTorqueControl", False)


@pytest.mark.parametrize("key,value", [("TorqueControlTune", "1.0"), ("NeuralNetworkLateralControl", "1"),
                                     ("TorqueParamsOverrideEnabled", "1"), ("LateralJerkTorqueController", "0")])
def test_reject_different_controller_configuration(key, value):
  raw = parameters()
  raw[key] = value
  with pytest.raises(ReplayError):
    validate_selection(RecordedParams(raw))


@pytest.mark.parametrize("key,value", [("TorqueControlTune", "nan"), ("EnforceTorqueControl", "true")])
def test_reject_malformed_typed_value(key, value):
  raw = parameters()
  raw[key] = value
  with pytest.raises(ReplayError):
    RecordedParams(raw).get(key)


def test_model_not_available_early_during_history_or_when_stale():
  a = rdf()
  validate_lateral_rdf(a)
  assert admit_model(a, a["input_t"][132] + .039)[0] is None
  assert admit_model(a, a["output_ready_t"][132])[0] == 132
  assert admit_model(a, a["output_ready_t"][-1] + .201)[0] is None


def test_full_plan_and_orientation_required():
  a = rdf()
  a["plan"][136, 10, 10] = np.nan
  with pytest.raises(ReplayError, match="full finite"):
    validate_lateral_rdf(a)


def test_explicit_rdf_unavailable_mask_never_backfilled():
  a = rdf()
  a["controller_admissible"][132] = False
  validate_lateral_rdf(a)
  assert admit_model(a, a["output_ready_t"][132])[0] is None
  a["deadline_admissible"] = a["deadline_admissible"].astype(float)
  with pytest.raises(ReplayError, match="boolean admission"):
    validate_lateral_rdf(a)


def test_payload_roundtrip_and_bad_offsets():
  blob, offsets = encode_payloads([b"abc", b"xy"])
  assert decode_payload(blob, offsets, 1) == b"xy"
  offsets[-1] += 1
  with pytest.raises(ReplayError, match="payload"):
    decode_payload(blob, offsets, 1)


def test_conflicting_duplicate_camera_identity_rejected():
  capture = {}
  add_capture(capture, 50, 100.)
  add_capture(capture, 50, 100.)
  assert capture == {50: 100.}
  with pytest.raises(ReplayError, match="conflicting camera"):
    add_capture(capture, 50, 100.01)


def test_actual_sp_asset_path_must_bind_receipt_and_allowed_file(tmp_path):
  import hashlib
  p = tmp_path / "openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/HYUNDAI_ELANTRA_2021.json"
  p.parent.mkdir(parents=True)
  p.write_bytes(b"{}")
  sha = hashlib.sha256(b"{}").hexdigest()
  validate_nn_binding(str(p), str(p), sha, tmp_path)
  with pytest.raises(ReplayError, match="constructor asset"):
    validate_nn_binding(str(p), str(tmp_path / "foreign.json"), sha, tmp_path)
  with pytest.raises(ReplayError, match="constructor asset"):
    validate_nn_binding(str(p), str(p), "0" * 64, tmp_path)


def test_preparation_adapter_binding_allows_eol_only(tmp_path):
  p = tmp_path / "adapter.py"
  p.write_bytes(b"x = 1\r\n")
  receipt = {"prepare_adapter_normalized_sha256": normalized_digest(p)}
  p.write_bytes(b"x = 1\n")
  check_adapter_binding(receipt, p)
  p.write_bytes(b"x = 2\n")
  with pytest.raises(ReplayError, match="preparation adapter"):
    check_adapter_binding(receipt, p)


def test_params_bool_matches_native_raw_read_even_for_integer_key():
  p = RecordedParams({"BlinkerPauseLateralControl": "0"})
  assert p.get("BlinkerPauseLateralControl") == 0
  assert p.get_bool("BlinkerPauseLateralControl") is False


def test_native_driver_limiter_and_angle_fault_boundaries():
  try:
    from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance
  except (ImportError, OSError):
    pytest.skip("native POSIX opendbc runtime unavailable")
  from types import SimpleNamespace
  p = SimpleNamespace(STEER_MAX=409, STEER_DELTA_UP=3, STEER_DELTA_DOWN=7,
                      STEER_DRIVER_ALLOWANCE=50, STEER_DRIVER_MULTIPLIER=2, STEER_DRIVER_FACTOR=1)
  assert apply_driver_steer_torque_limits(409, 0, 0, p) == 3
  assert apply_driver_steer_torque_limits(0, 409, 0, p) == 402
  assert apply_driver_steer_torque_limits(409, 409, -300, p) < 409
  counter, request = 0, True
  for _ in range(90):
    counter, request = common_fault_avoidance(True, True, counter, 89, 2)
  assert not request
  applied, counter, request = hardware_step(.8, 409, 0, 90, True, p, 89,
                                             apply_driver_steer_torque_limits, common_fault_avoidance)
  assert applied == 402 and not request  # angle cut holds bounded torque, does not invent zero torque
  applied, _, request = hardware_step(.8, 409, 0, 0, False, p, counter,
                                      apply_driver_steer_torque_limits, common_fault_avoidance)
  assert applied == 0 and not request
  applied, _, request = hardware_step(3.5, 0, 0, 0, True, p, 0,
                                      apply_driver_steer_torque_limits, common_fault_avoidance)
  assert applied == 3 and request  # legitimate cold native pre-limit request may exceed nominal +/-1


def test_native_v0_reset_does_not_clear_controller_history(monkeypatch):
  try:
    import openpilot.common.params as params_module
  except (ImportError, OSError):
    pytest.skip("native POSIX Params runtime unavailable")
  monkeypatch.setattr(params_module, "Params", lambda *args, **kwargs: RecordedParams(parameters()))
  from openpilot.cereal import custom, log
  from opendbc.car.structs import car
  from opendbc.car.vehicle_model import VehicleModel
  from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance
  from opendbc.car.hyundai.values import CarControllerParams
  from opendbc.car.hyundai.interface import CarInterface
  from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque
  from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as V1
  from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
  from types import SimpleNamespace
  cp = CarInterface.get_non_essential_params("HYUNDAI_ELANTRA_2024")
  CarInterface.configure_torque_tune(cp.carFingerprint, cp.lateralTuning)
  cp = cp.as_reader()
  sp = custom.CarParamsSP.new_message()
  ci = CarInterface(cp, sp)
  controller = ControlsExt.initialize_lateral_control(SimpleNamespace(params=RecordedParams(parameters()), CP=cp, CP_SP=sp),
                                                      V1(cp, sp, ci, .01), ci, .01)
  assert isinstance(controller, LatControlTorque)
  model = log.ModelDataV2.new_message()
  model.orientation.x = [0.] * 33
  model.acceleration.y = [0.] * 33
  controller.extension.update_model_v2(model)
  controller.extension.update_lateral_lag(.2)
  controller.extension.update_limits()  # still the separate extension PID before first native update
  cs = car.CarState.new_message(vEgo=20., steeringAngleDeg=80.)
  lp = log.VehicleParameters.new_message(stiffnessFactor=1., steerRatio=float(cp.steerRatio))
  raw_torque, _, _ = controller.update(True, cs, VehicleModel(cp), lp, False, 0., None, False, .2)
  assert np.isfinite(raw_torque) and abs(raw_torque) > 1
  applied, _, _ = hardware_step(raw_torque, 0, 0, 80., True, CarControllerParams(cp), 0,
                                apply_driver_steer_torque_limits, common_fault_avoidance)
  assert abs(applied) == 3
  controller.pid.i = .123
  controller.lat_accel_request_buffer.append(1.2)
  controller.previous_measurement = .4
  controller.sat_time = .8
  controller.reset()
  assert controller.pid.i == .123
  assert controller.lat_accel_request_buffer[-1] == 1.2
  assert controller.previous_measurement == .4
  assert controller.sat_time == 0
