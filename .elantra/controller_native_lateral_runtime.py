"""Native v0/jerk runtime adapter, without a vehicle plant or recorded-future feedback.

The caller owns source/cohort, causal RDF, plant support and scoring admission.
Cold state remains uncertain; native reset only clears saturation bookkeeping.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from types import MappingProxyType, SimpleNamespace

import numpy as np

import controller_native_lateral as lateral

OPENDBC_PIN = "b45eb78def29582bcce43c68cfb1d55f19224d1a"
V0_PATH = "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py"
V0_SHA = "8974572fff8ae55462cb6df459513bf3995b05854849e5dae844fd83f9991add"
ASSET = "openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/HYUNDAI_ELANTRA_2021.json"
SOURCES = (*lateral.SOURCES, "openpilot/common/basedir.py", "openpilot/cereal/__init__.py",
           "openpilot/selfdrive/locationd/torqued.py", "openpilot/sunnypilot/selfdrive/locationd/torqued_ext.py", ASSET)
TYPES = {**lateral.PARAM_TYPES, "CustomTorqueParams": "BOOL", "LiveTorqueParamsToggle": "BOOL", "LiveTorqueParamsRelaxedToggle": "BOOL"}
LP_KEYS = ("stiffnessFactor", "steerRatio", "angleOffsetDeg", "roll")
LIMIT_KEYS = ("STEER_MAX", "STEER_DELTA_UP", "STEER_DELTA_DOWN", "STEER_DRIVER_ALLOWANCE", "STEER_DRIVER_MULTIPLIER", "STEER_DRIVER_FACTOR")
_PARAMS_LOCK = RLock()


class LateralRuntimeError(ValueError):
  pass


def _finite(value):
  if isinstance(value, (bool, np.bool_)):
    return False
  try:
    return math.isfinite(value)
  except (TypeError, ValueError):
    return False


class ImmutableParams:
  def __init__(self, raw):
    self.raw, self.reads = MappingProxyType(dict(raw)), []

  def get(self, key, **kwargs):
    if key not in TYPES or key not in self.raw:
      raise LateralRuntimeError(f"missing recorded typed Params value: {key}")
    self.reads.append(key)
    value, kind = self.raw[key], TYPES[key]
    try:
      if kind == "BOOL":
        if value not in ("0", "1"):
          raise ValueError("requires 0/1")
        return value == "1"
      result = float(value) if kind == "FLOAT" else int(value)
      if not math.isfinite(result):
        raise ValueError("nonfinite")
      return result
    except (ValueError, TypeError) as exc:
      raise LateralRuntimeError(f"invalid recorded {kind} Params value: {key}") from exc

  def get_bool(self, key, **kwargs):
    self.get(key)
    return self.raw[key] == "1"

  def put(self, *args, **kwargs):
    raise LateralRuntimeError("Params provider is read-only")

  put_bool = remove = clear_all = put


@contextmanager
def scoped_params(modules, provider):
  """Restore cached from-import aliases even on construction failure; serialize patching."""
  def factory(*args, **kwargs):
    if args or kwargs:
      raise LateralRuntimeError("native Params path overrides are forbidden")
    if provider is None:
      raise LateralRuntimeError("Params construction outside a trial is forbidden")
    return provider
  with _PARAMS_LOCK:
    saved = list({id(module): (module, module.Params) for module in modules}.values())
    try:
      for module, _ in saved:
        module.Params = factory
      yield
    finally:
      for module, original in saved:
        module.Params = original


def source_hashes(repo, *, nn_asset_path=None):
  """Host manifests may hash a verified asset artifact; runtime always uses repo/ASSET."""
  repo = Path(repo)
  result = {}
  for name in SOURCES:
    path = Path(nn_asset_path) if name == ASSET and nn_asset_path is not None else repo/name
    data = path.read_bytes()
    result[name] = hashlib.sha256(data if name == ASSET else data.replace(b"\r\n", b"\n")).hexdigest()
  return result


@dataclass(frozen=True)
class Bindings:
  params_modules: tuple
  decode_static: object
  build_cs: object
  controller_factory: object
  ci_factory: object
  vm_factory: object
  limits_factory: object
  policy_type: object
  clip_curvature: object
  limiter: object
  fault_avoidance: object
  build_model: object
  get_lat_delay: object
  horizon: object
  actual_native: bool = False


def load_runtime(repo, expected_sources, expected_gitlink):
  """Future POSIX execution verifies files, Git pin and imports; no checkout writes."""
  repo = Path(repo).resolve()
  if os.name != "posix":
    raise LateralRuntimeError("actual native lateral runtime requires POSIX")
  actual = source_hashes(repo)
  if actual != expected_sources or actual.get(V0_PATH) != V0_SHA or expected_gitlink != OPENDBC_PIN:
    raise LateralRuntimeError("native source manifest/pin differs")
  pin = subprocess.run(["git", "-C", str(repo/"opendbc_repo"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
  if pin != expected_gitlink:
    raise LateralRuntimeError("actual runtime opendbc pin differs")
  sys.path[:0] = [str(repo), str(repo/"opendbc_repo")]
  params_module = importlib.import_module("openpilot.common.params")
  with scoped_params((params_module,), None):
    from openpilot.cereal import custom, log
    from opendbc.car.structs import car
    from opendbc.car.hyundai.interface import CarInterface
    from opendbc.car.hyundai.values import CarControllerParams
    from opendbc.car.hyundai.carcontroller import MAX_ANGLE, MAX_ANGLE_FRAMES, MAX_ANGLE_CONSECUTIVE_FRAMES
    from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance
    from opendbc.car.vehicle_model import VehicleModel
    from openpilot.common.realtime import DT_CTRL
    from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
    from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as V1
    from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as V0
    from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
    from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import TorqueEstimatorExt
    from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
  if DT_CTRL != .01 or (MAX_ANGLE, MAX_ANGLE_FRAMES, MAX_ANGLE_CONSECUTIVE_FRAMES) != (85, 89, 2):
    raise LateralRuntimeError("native time/angle contract differs")
  for obj, name in ((V0, V0_PATH), (V1, "openpilot/selfdrive/controls/lib/latcontrol_torque.py"),
                    (ControlsExt, "openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py"),
                    (TorqueEstimatorExt, "openpilot/sunnypilot/selfdrive/locationd/torqued_ext.py"),
                    (CarInterface, "opendbc_repo/opendbc/car/hyundai/interface.py"), (VehicleModel, "opendbc_repo/opendbc/car/vehicle_model.py"),
                    (CarControllerParams, "opendbc_repo/opendbc/car/hyundai/values.py"),
                    (apply_driver_steer_torque_limits, "opendbc_repo/opendbc/car/lateral.py"),
                    (common_fault_avoidance, "opendbc_repo/opendbc/car/lateral.py"),
                    (clip_curvature, "openpilot/selfdrive/controls/lib/drive_helpers.py"),
                    (get_lat_delay, "openpilot/sunnypilot/livedelay/helpers.py")):
    if Path(inspect.getfile(obj)).resolve() != (repo/name).resolve():
      raise LateralRuntimeError(f"native import origin differs: {name}")
  for name in SOURCES:
    if name.endswith(".py"):
      relative = name.removeprefix("opendbc_repo/")
      module_name = relative.removesuffix("/__init__.py") if relative.endswith("/__init__.py") else relative[:-3]
      module = sys.modules.get(module_name.replace("/", "."))
      if module is not None and Path(module.__file__).resolve() != (repo/name).resolve():
        raise LateralRuntimeError(f"cached native dependency origin differs: {name}")
  module_names = ("openpilot.common.params", "openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware",
                  "openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc", "openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override",
                  "openpilot.sunnypilot.selfdrive.locationd.torqued_ext", "openpilot.sunnypilot.modeld_v2.modeld_base")
  modules = tuple(importlib.import_module(name) for name in module_names)
  for module in modules:
    if Path(module.__file__).resolve() != (repo/(module.__name__.replace(".", "/")+".py")).resolve():
      raise LateralRuntimeError("Params alias module origin differs")
  def decode(cp_raw, sp_raw):
    if not isinstance(cp_raw, bytes) or not isinstance(sp_raw, bytes):
      raise LateralRuntimeError("actual runtime requires full CP/SP serialized bytes")
    with car.CarParams.from_bytes(cp_raw) as cp:
      c = cp.as_builder().as_reader()
    with custom.CarParamsSP.from_bytes(sp_raw) as sp:
      s = sp.as_builder().as_reader()
    lateral.validate_nn_binding(str(s.neuralNetworkLateralControl.model.path), str(s.neuralNetworkLateralControl.model.path), actual[ASSET], repo)
    return c, s
  def controller(cp, sp, ci, dt):
    selected = ControlsExt.initialize_lateral_control(SimpleNamespace(params=modules[0].Params(), CP=cp, CP_SP=sp), V1(cp, sp, ci, dt), ci, dt)
    if not isinstance(selected, V0):
      raise LateralRuntimeError("native selector did not construct v0")
    return selected
  def model(plan, horizon):
    result = log.ModelDataV2.new_message()
    for key, sl in (("position", Plan.POSITION), ("velocity", Plan.VELOCITY), ("acceleration", Plan.ACCELERATION),
                    ("orientation", Plan.T_FROM_CURRENT_EULER), ("orientationRate", Plan.ORIENTATION_RATE)):
      value = getattr(result, key)
      for axis, column in zip(("x", "y", "z"), plan[:, sl].T, strict=True):
        setattr(value, axis, column.astype(float).tolist())
      value.t = horizon.tolist()
    return result
  bindings = Bindings(modules, decode, car.CarState.new_message, controller, CarInterface, VehicleModel, CarControllerParams,
                      TorqueEstimatorExt, clip_curvature, apply_driver_steer_torque_limits, common_fault_avoidance, model, get_lat_delay,
                      tuple(ModelConstants.T_IDXS), True)
  return Runtime(bindings, actual, pin)


def _candidate_params(raw, candidate, grid):
  raw = dict(raw)
  if candidate is None:
    return raw
  if not isinstance(candidate, dict) or set(candidate) != {"factor", "friction"} or not isinstance(grid, dict) or set(grid) != set(candidate):
    raise LateralRuntimeError("candidate requires exact factor/friction and a declared finite grid")
  for key in candidate:
    values = grid[key]
    if (not isinstance(values, (tuple, list)) or not 0 < len(values) <= 64 or any(not _finite(v) for v in values)
        or any(v <= 0 if key == "factor" else v < 0 for v in values) or not _finite(candidate[key]) or candidate[key] not in values):
      raise LateralRuntimeError("candidate outside declared finite factor/friction grid")
  raw.update(CustomTorqueParams="1", TorqueParamsOverrideEnabled="1",
             TorqueParamsOverrideLatAccelFactor=str(float(candidate["factor"])), TorqueParamsOverrideFriction=str(float(candidate["friction"])))
  return raw


class Runtime:
  def __init__(self, bindings, sources, pin):
    self.bindings, self.sources, self.pin = bindings, MappingProxyType(dict(sources)), pin

  def model_from_rdf(self, plan, horizon):
    plan, horizon = np.asarray(plan), np.asarray(horizon)
    if (plan.shape != (33, 15) or horizon.shape != (33,) or not np.isfinite(plan).all() or not np.isfinite(horizon).all()
        or not np.allclose(horizon, self.bindings.horizon, atol=1e-7, rtol=0)):
      raise LateralRuntimeError("requires full finite RDF plan and native nonuniform horizon")
    return self.bindings.build_model(plan, horizon)

  def validate_model(self, model):
    try:
      for key in ("position", "velocity", "acceleration", "orientation", "orientationRate"):
        vector = getattr(model, key)
        for axis in ("x", "y", "z", "t"):
          values = np.asarray(getattr(vector, axis), dtype=float)
          if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("invalid vector")
          if axis == "t" and not np.allclose(values, self.bindings.horizon, atol=1e-7, rtol=0):
            raise ValueError("invalid native horizon")
    except (AttributeError, TypeError, ValueError) as exc:
      raise LateralRuntimeError("requires full finite native model with fixed native horizon") from exc

  def create(self, cp_raw, sp_raw, lp, params_raw, *, candidate=None, grid=None, initialization=None):
    raw = _candidate_params(params_raw, candidate, grid)
    provider = ImmutableParams(raw)
    if (not provider.get_bool("EnforceTorqueControl") or provider.get("TorqueControlTune") != 0.
        or not provider.get_bool("LateralJerkTorqueController") or provider.get_bool("NeuralNetworkLateralControl")):
      raise LateralRuntimeError("requires native v0 + jerk-aware and NN disabled")
    if set(lp) != set(LP_KEYS) or any(not _finite(lp[k]) for k in LP_KEYS) or lp["stiffnessFactor"] <= 0 or lp["steerRatio"] <= 0:
      raise LateralRuntimeError("requires finite fixed causal vehicle parameters")
    cp, sp = self.bindings.decode_static(cp_raw, sp_raw)
    if (cp.carFingerprint != "HYUNDAI_ELANTRA_2024" or cp.brand != "hyundai" or cp.lateralTuning.which() != "torque"
        or str(cp.steerControlType) != "torque"):
      raise LateralRuntimeError("requires exact Elantra torque CP")
    torque = cp.lateralTuning.torque
    if (any(not _finite(x) for x in (torque.latAccelFactor, torque.latAccelOffset, torque.friction, cp.minSteerSpeed))
        or torque.latAccelFactor <= 0 or torque.friction < 0):
      raise LateralRuntimeError("invalid baseline CP torque parameters")
    with scoped_params(self.bindings.params_modules, provider):
      policy = self.bindings.policy_type(cp)
      policy.offline_latAccelFactor, policy.offline_friction = cp.lateralTuning.torque.latAccelFactor, cp.lateralTuning.torque.friction
      policy.initialize_custom_params(decimated=False)
      ci, vm, limits = self.bindings.ci_factory(cp, sp), self.bindings.vm_factory(cp), self.bindings.limits_factory(cp)
      controller = self.bindings.controller_factory(cp, sp, ci, .01)
    if tuple(getattr(limits, key) for key in LIMIT_KEYS) != (409, 3, 7, 50, 2, 1):
      raise LateralRuntimeError("native steering envelope differs from 409/3/7 contract")
    if controller.extension.torque_params is not controller.torque_params:
      raise LateralRuntimeError("native torque parameter shared identity differs")
    arguments = (cp_raw, sp_raw, dict(lp), dict(params_raw), deepcopy(candidate), deepcopy(grid))
    return Handle(self, cp, vm, limits, controller, policy, provider, dict(lp), arguments, initialization)


class Handle:
  def __init__(self, runtime, cp, vm, limits, controller, policy, provider, lp, arguments, initialization):
    self.runtime, self.cp, self.vm, self.limits, self.controller, self.policy, self.provider = runtime, cp, vm, limits, controller, policy, provider
    self.lp, self.arguments = SimpleNamespace(**lp), arguments
    initial = dict(initialization or {})
    if set(initial)-{"previous_curvature", "last_transmitted_counts", "angle_counter", "steer_limited"}:
      raise LateralRuntimeError("unknown initialization fields cannot establish native state")
    self.previous_curvature = initial.get("previous_curvature", 0.)
    self.last_transmitted_counts, self.angle_counter = initial.get("last_transmitted_counts", 0), initial.get("angle_counter", 0)
    self.limited = initial.get("steer_limited", False)
    if (not _finite(self.previous_curvature) or type(self.last_transmitted_counts) is not int or abs(self.last_transmitted_counts) > 409
        or type(self.angle_counter) is not int or self.angle_counter < 0 or type(self.limited) is not bool):
      raise LateralRuntimeError("invalid explicit initialization")
    self.last_t, self.param_time, self.initialization_uncertain = None, None, True
    self.extension_lag = provider.get("LagdValueCache")

  def reset_native(self):
    self.controller.reset()  # Does not clear PID, filters, demand buffer or limiter history.

  def cold_restart(self):
    cp, sp, lp, raw, candidate, grid = self.arguments
    return self.runtime.create(cp, sp, lp, raw, candidate=candidate, grid=grid)

  def step(self, *, t, active, v, a, steering_angle, steering_rate, driver_torque, steering_pressed, standstill,
           fault_temporary, fault_permanent, prior_delivered_torque, model, desired_curvature, published_delay, learner=None):
    numeric = (t, v, a, steering_angle, steering_rate, driver_torque, prior_delivered_torque, desired_curvature, published_delay)
    if (any(not _finite(x) for x in numeric) or v < 0 or published_delay < 0 or abs(prior_delivered_torque) > 1
        or any(type(x) is not bool for x in (active, steering_pressed, standstill, fault_temporary, fault_permanent))):
      raise LateralRuntimeError("invalid explicit simulated feedback")
    if self.last_t is not None and abs(t-self.last_t-.01) > 1e-9:
      raise LateralRuntimeError("requires continuous strictly chronological 100Hz steps")
    self.runtime.validate_model(model)
    cs = self.runtime.bindings.build_cs(vEgo=v, aEgo=a, steeringAngleDeg=steering_angle, steeringRateDeg=steering_rate,
                                       steeringTorque=driver_torque, steeringPressed=steering_pressed, standstill=standstill,
                                       steerFaultTemporary=fault_temporary, steerFaultPermanent=fault_permanent)
    stopped = v <= max(self.cp.minSteerSpeed, .3) or standstill
    effective_active = active and not fault_temporary and not fault_permanent and (not stopped or self.cp.steerAtStandstill)
    self.vm.update_params(max(self.lp.stiffnessFactor, .1), max(self.lp.steerRatio, .1))
    measured = -self.vm.calc_curvature(math.radians(steering_angle-self.lp.angleOffsetDeg), v, self.lp.roll)
    learner_applied = False
    if learner is not None:
      if not isinstance(learner, dict) or type(learner.get("admitted")) is not bool or type(learner.get("useParams")) is not bool:
        raise LateralRuntimeError("requires explicit causal learner admission")
      if learner["admitted"] and learner["useParams"] and self.policy.use_params:
        values = [learner.get(k) for k in ("factor", "offset", "friction")]
        if any(not _finite(x) for x in values) or values[0] <= 0 or values[2] < 0:
          raise LateralRuntimeError("invalid admitted learner parameters")
        self.controller.update_torque_parameters(*values)
        self.controller.extension.update_limits()
        learner_applied = True
    self.controller.extension.update_model_v2(model)
    self.controller.extension.update_lateral_lag(self.extension_lag)
    if not effective_active:
      self.reset_native()
    self.previous_curvature, clipped = self.runtime.bindings.clip_curvature(v, self.previous_curvature,
                                                                          desired_curvature if effective_active else measured, self.lp.roll)
    old_limited = self.limited
    torque, _, _ = self.controller.update(effective_active, cs, self.vm, self.lp, old_limited, self.previous_curvature, None, clipped, published_delay)
    torque = float(torque)
    if not math.isfinite(torque):
      raise LateralRuntimeError("native controller produced nonfinite torque")
    applied, self.angle_counter, request = lateral.hardware_step(torque, self.last_transmitted_counts, driver_torque, steering_angle,
                                                                effective_active, self.limits, self.angle_counter,
                                                                self.runtime.bindings.limiter, self.runtime.bindings.fault_avoidance)
    if abs(applied) > 409 or (effective_active and abs(applied-self.last_transmitted_counts) > 7):
      raise LateralRuntimeError("native steering envelope violated")
    if effective_active:
      self.limited = abs(torque-prior_delivered_torque) > 1e-2
    self.last_transmitted_counts, self.last_t = applied, t
    p, pid = self.controller.torque_params, self.controller.pid
    frame = self.controller.extension.frame if hasattr(self.controller.extension, "frame") else self.controller.frame
    result = {"t": t, "active": effective_active, "torque": torque, "can_torque": applied, "steer_request": bool(request),
              "torque_fault": effective_active and not request, "steer_limited_feedback": old_limited, "steer_limited_next": self.limited,
              "desired_curvature": self.previous_curvature, "measured_curvature": measured, "learner_applied": learner_applied,
              "learner_policy_use_params": bool(self.policy.use_params), "effective_factor": float(p.latAccelFactor),
              "effective_offset": float(p.latAccelOffset), "effective_friction": float(p.friction), "pid_i": float(pid.i),
              "pid_pos_limit": float(pid.pos_limit), "pid_neg_limit": float(pid.neg_limit), "core_delay": published_delay,
              "extension_lag": self.extension_lag, "override_frame": frame, "pre_limit_saturated": abs(torque) > 1,
              "initialization_uncertain": True, "actual_native": self.runtime.bindings.actual_native}
    if self.param_time is None or t-self.param_time > 3:
      self.extension_lag = self.runtime.bindings.get_lat_delay(self.provider, published_delay)
      self.param_time = t
    return result
