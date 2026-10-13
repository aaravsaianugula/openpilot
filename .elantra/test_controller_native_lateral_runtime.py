"""Synthetic fixtures verify adapter logic, never vehicle performance."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import controller_native_lateral_runtime as rt


def params():
  return {"EnforceTorqueControl": "1", "TorqueControlTune": "0.0", "LateralJerkTorqueController": "1",
          "NeuralNetworkLateralControl": "0", "TorqueParamsOverrideEnabled": "0", "CustomTorqueParams": "0",
          "LiveTorqueParamsToggle": "1", "LiveTorqueParamsRelaxedToggle": "0", "LagdToggle": "1", "LagdValueCache": "0.2",
          "TorqueParamsOverrideLatAccelFactor": "2.5", "TorqueParamsOverrideFriction": "0.1"}


GRID = {"factor": (2., 2.5, 3.), "friction": (0., .1, .2)}
LP = {"stiffnessFactor": 1., "steerRatio": 13.5, "angleOffsetDeg": 0., "roll": 0.}


def fixture_runtime():
  aliases = [NS(Params=object()) for _ in range(4)]
  class Policy:
    def __init__(self, cp):
      self.p = aliases[3].Params()
    def initialize_custom_params(self, decimated=False):
      self.use_params = not (self.p.get_bool("CustomTorqueParams") and self.p.get_bool("TorqueParamsOverrideEnabled"))
      self.p.get_bool("LiveTorqueParamsRelaxedToggle")
  class VM:
    def __init__(self, cp):
      pass
    def update_params(self, stiffness, ratio):
      self.ratio = ratio
    def calc_curvature(self, angle, speed, roll):
      return -angle / self.ratio
  class Controller:
    def __init__(self, cp, sp, ci, dt):
      self.params = aliases[0].Params()
      assert aliases[1].Params() is self.params and aliases[2].Params() is self.params
      self.torque_params = deepcopy(cp.lateralTuning.torque)
      self.pid = NS(i=0., pos_limit=2.5, neg_limit=-2.5)
      self.events, self.frame = [], -1
      self.extension = NS(torque_params=self.torque_params, update_limits=self.ext_limits,
                          update_model_v2=lambda m: self.events.append("model"), update_lateral_lag=lambda x: self.events.append("lag"))
    def update_torque_parameters(self, factor, offset, friction):
      self.events.append("learner")
      self.torque_params.latAccelFactor, self.torque_params.latAccelOffset, self.torque_params.friction = factor, offset, friction
      self.pid.pos_limit, self.pid.neg_limit = factor, -factor
    def ext_limits(self):
      self.events.append("extension_limits")
      self.pid.pos_limit, self.pid.neg_limit = 1., -1.
    def reset(self):
      self.events.append("reset")  # native reset preserves PID state
    def update(self, active, cs, vm, lp, limited, curvature, pose, clipped, delay):
      self.events.append("update")
      self.frame += 1
      if self.frame % 300 == 0 and self.params.get_bool("TorqueParamsOverrideEnabled"):
        self.update_torque_parameters(self.params.get("TorqueParamsOverrideLatAccelFactor"), self.torque_params.latAccelOffset,
                                      self.params.get("TorqueParamsOverrideFriction"))
      if active:
        self.pid.i += .001
      value = (curvature * cs.vEgo ** 2 - cs.steeringAngleDeg - cs.aEgo * .01) / self.torque_params.latAccelFactor
      return (value + self.pid.i if active else 0.), 0., NS()
  cp = NS(carFingerprint="HYUNDAI_ELANTRA_2024", brand="hyundai", steerControlType="torque", minSteerSpeed=0., steerAtStandstill=False,
          lateralTuning=NS(which=lambda: "torque", torque=NS(latAccelFactor=2.5, latAccelOffset=0., friction=.1)))
  limits = NS(STEER_MAX=409, STEER_DELTA_UP=3, STEER_DELTA_DOWN=7, STEER_DRIVER_ALLOWANCE=50,
              STEER_DRIVER_MULTIPLIER=2, STEER_DRIVER_FACTOR=1)
  def limiter(wanted, previous, driver, limits):
    return int(np.clip(wanted, max(-409, previous-3), min(409, previous+3)))
  def model(plan, horizon):
    fields = {}
    for i, key in enumerate(("position", "velocity", "acceleration", "orientation", "orientationRate")):
      fields[key] = NS(**{axis: plan[:, i*3+j].tolist() for j, axis in enumerate(("x", "y", "z"))}, t=horizon.tolist())
    return NS(**fields)
  b = rt.Bindings(tuple(aliases), lambda c, s: (deepcopy(c), s), lambda **kw: NS(**kw), Controller,
                  lambda c, s: NS(), VM, lambda c: deepcopy(limits), Policy,
                  lambda v, old, new, roll: (new, False), limiter, lambda hit, active, counter, a, b: (counter+int(hit), active),
                  model,
                  lambda p, d: d if p.get_bool("LagdToggle") else p.get("LagdValueCache"), np.linspace(0, 10, 33),
                  actual_native=False)
  return rt.Runtime(b, {}, "a"*40), cp, NS(), aliases


def make(runtime, cp, sp, candidate=None, raw=None):
  return runtime.create(cp, sp, LP, raw or params(), candidate=candidate, grid=GRID)


def tick(handle, t=10., **changes):
  model = handle.runtime.model_from_rdf(np.zeros((33, 15)), handle.runtime.bindings.horizon)
  values = {"t": t, "active": True, "v": 15., "a": 0., "steering_angle": 0., "steering_rate": 0., "driver_torque": 0.,
            "steering_pressed": False, "standstill": False, "fault_temporary": False, "fault_permanent": False,
            "prior_delivered_torque": 0., "model": model, "desired_curvature": .001, "published_delay": .2, "learner": None}
  values.update(changes)
  return handle.step(**values)


def test_immutable_typed_provider_missing_and_write_fail_closed():
  raw = params()
  p = rt.ImmutableParams(raw)
  raw["LagdValueCache"] = "9"
  assert p.get("LagdValueCache") == .2
  with pytest.raises(TypeError):
    p.raw["LagdValueCache"] = "9"
  with pytest.raises(rt.LateralRuntimeError, match="missing"):
    p.get("unrecorded")
  with pytest.raises(rt.LateralRuntimeError, match="read-only"):
    p.put("LagdValueCache", "9")


@pytest.mark.parametrize("candidate", [{"factor": 0., "friction": .1}, {"factor": 2.2, "friction": .1},
                                       {"factor": 2.5, "friction": -.1}, {"factor": 2.5, "offset": 0.}])
def test_declared_finite_grid_is_required(candidate):
  runtime, cp, sp, _ = fixture_runtime()
  with pytest.raises(rt.LateralRuntimeError):
    make(runtime, cp, sp, candidate)


def test_provider_aliases_restore_and_a_b_a_has_independent_state():
  runtime, cp, sp, aliases = fixture_runtime()
  originals = [m.Params for m in aliases]
  a = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  b = make(runtime, cp, sp, {"factor": 3., "friction": .2})
  again = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  assert [m.Params for m in aliases] == originals
  assert tick(a)["torque"] == tick(again)["torque"]
  assert tick(b)["torque"] != tick(make(runtime, cp, sp, {"factor": 2., "friction": .1}))["torque"]
  assert a.provider is a.controller.params and a.provider is not b.provider


def test_aliases_restore_after_constructor_failure():
  runtime, cp, sp, aliases = fixture_runtime()
  originals = [m.Params for m in aliases]
  raw = params()
  del raw["LiveTorqueParamsRelaxedToggle"]
  with pytest.raises(rt.LateralRuntimeError, match="missing"):
    make(runtime, cp, sp, raw=raw)
  assert [m.Params for m in aliases] == originals


def test_native_learner_order_and_fixed_override_policy_are_distinct():
  runtime, cp, sp, _ = fixture_runtime()
  learner = {"admitted": True, "useParams": True, "factor": 3., "offset": .03, "friction": .2}
  baseline = make(runtime, cp, sp)
  result = tick(baseline, learner=learner)
  assert baseline.controller.events[:4] == ["learner", "extension_limits", "model", "lag"]
  assert result["learner_applied"] and result["effective_factor"] == 3. and result["pid_pos_limit"] == 1.
  candidate = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  result = tick(candidate, learner=learner)
  assert not result["learner_applied"] and not result["learner_policy_use_params"]
  assert result["effective_factor"] == 2. and result["pid_pos_limit"] == 2.


def test_simulated_feedback_and_prior_delivery_drive_commands_without_logged_cs():
  runtime, cp, sp, _ = fixture_runtime()
  left = tick(make(runtime, cp, sp), steering_angle=-1.)
  right = tick(make(runtime, cp, sp), steering_angle=1.)
  assert left["torque"] != right["torque"]
  h = make(runtime, cp, sp)
  first = tick(h)
  second = tick(h, 10.01, prior_delivered_torque=first["can_torque"] / 409.)
  assert abs(first["torque"] - first["can_torque"] / 409.) > .01
  assert second["steer_limited_next"]
  third = tick(h, 10.02, prior_delivered_torque=second["can_torque"] / 409.)
  assert third["steer_limited_feedback"]
  assert abs(first["can_torque"]) <= 3 and abs(second["can_torque"]) <= 6


def test_native_reset_retains_pid_and_cold_restart_is_uncertain():
  runtime, cp, sp, _ = fixture_runtime()
  h = make(runtime, cp, sp)
  tick(h)
  tick(h, 10.01, active=False)
  assert h.controller.pid.i > 0
  fresh = h.cold_restart()
  assert fresh.controller.pid.i == 0. and fresh.initialization_uncertain


def test_override_frame300_preserves_candidate_and_learner_is_not_resurrected():
  runtime, cp, sp, _ = fixture_runtime()
  h = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  for i in range(301):
    result = tick(h, 10.+i*.01, learner={"admitted": True, "useParams": True, "factor": 3., "offset": .1, "friction": .2})
  assert result["override_frame"] == 300 and result["effective_factor"] == 2.
  assert result["effective_friction"] == .1 and not result["learner_applied"]


def test_lag_refresh_runs_after_control_and_strictly_after_three_seconds():
  runtime, cp, sp, _ = fixture_runtime()
  h = make(runtime, cp, sp)
  for i in range(302):
    result = tick(h, 10.+i*.01, published_delay=.4)
    if i == 0:
      assert result["extension_lag"] == .2 and result["core_delay"] == .4
    else:
      assert result["extension_lag"] == .4
  raw = params()
  raw["LagdToggle"] = "0"
  h = make(runtime, cp, sp, raw=raw)
  tick(h, published_delay=.4)
  assert tick(h, 10.01, published_delay=.4)["extension_lag"] == .2


def test_invalid_cp_and_hardware_limits_cannot_be_masked_by_candidate():
  runtime, cp, sp, _ = fixture_runtime()
  cp.lateralTuning.torque.latAccelFactor = 0.
  with pytest.raises(rt.LateralRuntimeError, match="CP torque"):
    make(runtime, cp, sp, {"factor": 2., "friction": .1})
  cp.lateralTuning.torque.latAccelFactor = 2.5
  cp.steerControlType = "angle"
  with pytest.raises(rt.LateralRuntimeError, match="Elantra torque"):
    make(runtime, cp, sp)
  cp.steerControlType = "torque"
  normal_limits = runtime.bindings.limits_factory(cp)
  normal_limits.STEER_MAX = 410
  changed = rt.Runtime(replace(runtime.bindings, limits_factory=lambda c: normal_limits), {}, "a"*40)
  with pytest.raises(rt.LateralRuntimeError, match="409/3/7"):
    make(changed, cp, sp)


def test_source_manifest_and_pin_fail_before_any_native_git_or_import(monkeypatch):
  monkeypatch.setattr(rt, "os", NS(name="posix"))
  monkeypatch.setattr(rt, "source_hashes", lambda repo: {rt.V0_PATH: rt.V0_SHA})
  monkeypatch.setattr(rt.subprocess, "run", lambda *a, **kw: pytest.fail("Git must not run after source admission failure"))
  with pytest.raises(rt.LateralRuntimeError, match="manifest/pin"):
    rt.load_runtime(Path.cwd(), {}, rt.OPENDBC_PIN)
  with pytest.raises(rt.LateralRuntimeError, match="manifest/pin"):
    rt.load_runtime(Path.cwd(), {rt.V0_PATH: rt.V0_SHA}, "a"*40)


def test_host_manifest_can_bind_verified_external_asset_without_runtime_fallback(tmp_path, monkeypatch):
  monkeypatch.setattr(rt, "SOURCES", ("native.py", rt.ASSET))
  (tmp_path/"native.py").write_bytes(b"native\r\n")
  artifact = tmp_path/"verified-artifact.json"
  artifact.write_bytes(b"{}\r\n")
  expected = rt.source_hashes(tmp_path, nn_asset_path=artifact)
  assert expected["native.py"] == hashlib.sha256(b"native\n").hexdigest()
  assert expected[rt.ASSET] == hashlib.sha256(b"{}\r\n").hexdigest()
  with pytest.raises(FileNotFoundError):
    rt.source_hashes(tmp_path)  # Actual loader never supplies the host-only override.


@pytest.mark.parametrize("changes", [{"v": -1.}, {"a": float("nan")}, {"t": float("inf")}, {"active": 1}])
def test_invalid_simulated_inputs_fail_closed(changes):
  runtime, cp, sp, _ = fixture_runtime()
  with pytest.raises(rt.LateralRuntimeError):
    tick(make(runtime, cp, sp), **changes)


def test_time_gap_and_model_horizon_are_not_silently_repaired():
  runtime, cp, sp, _ = fixture_runtime()
  h = make(runtime, cp, sp)
  tick(h)
  with pytest.raises(rt.LateralRuntimeError, match="100Hz"):
    tick(h, 10.03)
  with pytest.raises(rt.LateralRuntimeError, match="horizon"):
    runtime.model_from_rdf(np.zeros((33, 15)), np.linspace(0, 9, 33))


def test_direct_model_with_nonfinite_field_cannot_bypass_converter_validation():
  runtime, cp, sp, _ = fixture_runtime()
  model = runtime.model_from_rdf(np.zeros((33, 15)), runtime.bindings.horizon)
  model.velocity.z[12] = float("nan")
  with pytest.raises(rt.LateralRuntimeError, match="full finite native model"):
    tick(make(runtime, cp, sp), model=model)


@pytest.mark.skipif(os.name != "posix", reason="Owner-serialized real native runtime required")
def test_actual_native_provider_turns_frame300_reset_and_simulated_feedback():
  repo = Path(os.environ["NATIVE_LATERAL_SOURCE_ROOT"])
  runtime = rt.load_runtime(repo, rt.source_hashes(repo), rt.OPENDBC_PIN)
  from opendbc.car.structs import car
  from opendbc.car.hyundai.values import HyundaiFlags
  from openpilot.cereal import custom
  cp = car.CarParams.new_message(carFingerprint="HYUNDAI_ELANTRA_2024", brand="hyundai", mass=1360., wheelbase=2.72,
                                centerToFront=1.088, rotationalInertia=2500., steerRatio=13.5, steerRatioRear=0.,
                                tireStiffnessFront=80000., tireStiffnessRear=80000.,
                                steerActuatorDelay=.1, steerLimitTimer=.4, minSteerSpeed=0., steerAtStandstill=False,
                                steerControlType="torque", flags=int(HyundaiFlags.RAISED_LIMITS))
  tuning = cp.lateralTuning.init("torque")
  tuning.latAccelFactor, tuning.latAccelOffset, tuning.friction = 2.5, 0., .1
  sp = custom.CarParamsSP.new_message()
  sp.neuralNetworkLateralControl.model.path = str(repo/rt.ASSET)
  cp, sp = cp.to_bytes(), sp.to_bytes()
  a = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  b = make(runtime, cp, sp, {"factor": 3., "friction": .2})
  again = make(runtime, cp, sp, {"factor": 2., "friction": .1})
  av, bv, repeat = tick(a), tick(b), tick(again)
  assert av["torque"] == repeat["torque"] and av["torque"] != bv["torque"]
  negative = tick(make(runtime, cp, sp, {"factor": 2., "friction": .1}), desired_curvature=-.001)
  assert av["torque"] * negative["torque"] < 0
  changed = tick(make(runtime, cp, sp, {"factor": 2., "friction": .1}), steering_angle=1.)
  assert changed["torque"] != av["torque"]
  for i in range(1, 301):
    av = tick(a, 10.+i*.01)
  assert av["effective_factor"] == 2.
  assert av["effective_friction"] == pytest.approx(.1, rel=0, abs=1e-7)  # Native capnp stores friction as Float32.
  assert av["override_frame"] == 300
  old_i = a.controller.pid.i
  tick(a, 13.01, active=False)
  assert a.controller.pid.i == old_i
  assert a.cold_restart().controller.pid.i == 0.
