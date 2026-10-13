#!/usr/bin/env python3
"""Current-master native lateral commands on frozen recorded motion; no actuation.

Prepare full recorded structs on the archive host, then run on POSIX. Engagement
is the recorded latActive trace, not a reconstructed engagement state machine.
Unknown initialization remains uncertain even after native reset/disengagement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import controller_native_replay as native_common
from controller_native_replay import ReplayError, asof_indices, digest, validate_output, validate_rdf

PARAM_TYPES = {
  "EnforceTorqueControl": "BOOL", "TorqueControlTune": "FLOAT", "LateralJerkTorqueController": "BOOL",
  "NeuralNetworkLateralControl": "BOOL", "TorqueParamsOverrideEnabled": "BOOL",
  "TorqueParamsOverrideLatAccelFactor": "FLOAT", "TorqueParamsOverrideFriction": "FLOAT",
  "LagdToggle": "BOOL", "LagdValueCache": "FLOAT", "BlinkerPauseLateralControl": "INT", "IsMetric": "BOOL",
  "BlinkerMinLateralControlSpeed": "INT", "BlinkerLateralReengageDelay": "INT",
}
FIELDS = {
  "carState": ("vEgo", "aEgo", "steeringAngleDeg", "steeringRateDeg", "steeringTorque", "steeringPressed",
               "steerFaultTemporary", "steerFaultPermanent", "standstill", "leftBlinker", "rightBlinker"),
  "vehicleParameters": ("stiffnessFactor", "steerRatio", "angleOffsetDeg", "roll"),
  "lateralTorqueParameters": ("latAccelFactorFiltered", "latAccelOffsetFiltered", "frictionCoefficientFiltered", "useParams", "valid"),
  "lateralDelay": ("lateralDelay",), "carControl": ("latActive", "actuators"), "carOutput": ("actuatorsOutput",),
  "selfdriveState": ("active",), "selfdriveStateSP": ("mads",),
}
SOURCES = (
  "openpilot/common/params_keys.h", "openpilot/common/params.py",
  "openpilot/common/pid.py", "openpilot/common/filter_simple.py", "openpilot/common/realtime.py",
  "openpilot/common/constants.py", "openpilot/cereal/log.capnp", "openpilot/cereal/custom.capnp",
  "openpilot/cereal/messaging/__init__.py", "openpilot/cereal/services.py", "openpilot/selfdrive/controls/controlsd.py",
  "openpilot/selfdrive/controls/lib/latcontrol.py", "openpilot/selfdrive/controls/lib/latcontrol_torque.py",
  "openpilot/selfdrive/controls/lib/drive_helpers.py", "openpilot/selfdrive/modeld/constants.py",
  "openpilot/selfdrive/locationd/helpers.py",
  "openpilot/selfdrive/modeld/modeld.py", "openpilot/sunnypilot/__init__.py",
  "openpilot/sunnypilot/modeld_v2/modeld_base.py", "openpilot/sunnypilot/livedelay/helpers.py",
  "openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_base.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_override.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_jerk_aware.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/nnlc/nnlc.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/nnlc/model.py",
  "openpilot/sunnypilot/selfdrive/controls/lib/nnlc/helpers.py",
  "opendbc_repo/opendbc/car/structs.py", "opendbc_repo/opendbc/car/car.capnp",
  "opendbc_repo/opendbc/car/interfaces.py", "opendbc_repo/opendbc/sunnypilot/car/interfaces.py",
  "opendbc_repo/opendbc/car/vehicle_model.py", "opendbc_repo/opendbc/car/lateral.py",
  "opendbc_repo/opendbc/sunnypilot/car/lateral_ext.py", "opendbc_repo/opendbc/car/hyundai/interface.py",
  "opendbc_repo/opendbc/car/hyundai/carcontroller.py", "opendbc_repo/opendbc/car/hyundai/values.py",
)


class RecordedParams:
  def __init__(self, raw):
    self.raw, self.reads = dict(raw), []

  def get(self, key, **kwargs):
    if key not in self.raw or key not in PARAM_TYPES:
      raise ReplayError(f"missing recorded typed parameter: {key}; native defaults are forbidden")
    self.reads.append(key)
    value, kind = self.raw[key], PARAM_TYPES[key]
    try:
      if kind == "BOOL":
        if value not in ("0", "1"):
          raise ValueError("boolean is not 0/1")
        return value == "1"
      result = float(value) if kind == "FLOAT" else int(value)
      if not np.isfinite(result):
        raise ValueError("nonfinite")
      return result
    except (TypeError, ValueError) as exc:
      raise ReplayError(f"invalid recorded {kind} parameter: {key}") from exc

  def get_bool(self, key, **kwargs):
    self.get(key)  # validate even when the registered type is INT
    return self.raw[key] == "1"  # native get_bool compares bytes directly

  def put_bool(self, *args, **kwargs):
    raise ReplayError("recorded Params are read-only")

  put = remove = clear_all = put_bool


def validate_selection(params):
  if (not params.get_bool("EnforceTorqueControl") or params.get("TorqueControlTune") != 0.0
      or not params.get_bool("LateralJerkTorqueController") or params.get_bool("NeuralNetworkLateralControl")
      or params.get_bool("TorqueParamsOverrideEnabled")):
    raise ReplayError("requires exact recorded v0 + jerk-aware, NN/override disabled configuration")
  params.get_bool("LagdToggle")
  params.get("LagdValueCache")


def validate_lateral_rdf(rdf):
  validate_rdf(rdf)
  n = len(rdf["input_t"])
  if (np.asarray(rdf.get("plan")).shape != (n, 33, 15) or not np.isfinite(rdf["plan"]).all()
      or np.asarray(rdf.get("desired_curvature")).shape != (n,) or not np.isfinite(rdf["desired_curvature"]).all()
      or np.asarray(rdf.get("horizon")).shape != (33,) or not np.isfinite(rdf["horizon"]).all()):
    raise ReplayError("requires full finite RDF plan acceleration/orientation and curvature")
  for key in ("history_scored_mask", "deadline_admissible", "controller_admissible"):
    a = np.asarray(rdf.get(key))
    if a.shape != (n,) or a.dtype.kind != "b":
      raise ReplayError("requires matching boolean admission flags")
  if (not np.array_equal(rdf["history_scored_mask"], rdf["scored_mask"])
      or np.any(rdf["controller_admissible"] & ~(rdf["scored_mask"] & rdf["deadline_admissible"]))
      or np.any(rdf["deadline_admissible"] & (rdf["output_ready_t"] - rdf["input_t"] > .2 + 1e-9))):
    raise ReplayError("RDF declared admission conflicts with causal clocks/history")


def admit_model(rdf, timestamp):
  i = int(np.searchsorted(rdf["output_ready_t"], timestamp, side="right") - 1)
  if i < 0 or not rdf["scored_mask"][max(i, 0)]:
    return None, 1
  if not 0 <= timestamp - rdf["narrow_t"][i] <= .2 + 1e-9:
    return None, 2
  if not rdf["controller_admissible"][i] or not rdf["deadline_admissible"][i]:
    return None, 5
  return i, 0


def encode_payloads(payloads):
  offsets = np.r_[0, np.cumsum([len(p) for p in payloads])].astype(np.int64)
  return np.frombuffer(b"".join(payloads), np.uint8).copy(), offsets


def add_capture(capture, frame_id, timestamp):
  if frame_id in capture and capture[frame_id] != timestamp:
    raise ReplayError("conflicting camera frame identity")
  capture[frame_id] = timestamp


def decode_payload(blob, offsets, index):
  if (blob.ndim != 1 or blob.dtype != np.uint8 or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
      or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(blob) or np.any(np.diff(offsets) <= 0)
      or not 0 <= index < len(offsets) - 1):
    raise ReplayError("invalid full-struct payload offsets")
  return blob[offsets[index]:offsets[index + 1]].tobytes()


def source_hashes(repo):
  return {name: hashlib.sha256((repo / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in SOURCES}


def normalized_digest(path):
  return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_adapter_binding(receipt, source):
  if receipt.get("prepare_adapter_normalized_sha256") != normalized_digest(source):
    raise ReplayError("preparation adapter source changed")


def validate_nn_binding(recorded_path, constructor_path, expected_sha, repo):
  actual = Path(constructor_path)
  allowed = repo / "openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/HYUNDAI_ELANTRA_2021.json"
  if (constructor_path != recorded_path or actual.resolve() != allowed.resolve()
      or not actual.is_file() or digest(actual) != expected_sha):
    raise ReplayError("recorded NN constructor asset missing/changed, despite NN feedforward being disabled")


def hardware_step(torque, previous, driver_torque, steering_angle, active, limits, angle_counter, limiter, fault_avoidance):
  applied = limiter(int(round(torque * limits.STEER_MAX)), previous, driver_torque, limits)
  angle_counter, request = fault_avoidance(abs(steering_angle) >= 85, active, angle_counter, 89, 2)
  return applied if active else 0, angle_counter, request


def prepare(segment, schema_dir, rdf, repo):
  from controller_extract import decompress_log, load_schema

  validate_lateral_rdf(rdf)
  path = next((segment / n for n in ("rlog.zst", "rlog.bz2", "rlog") if (segment / n).is_file()), None)
  if path is None:
    raise ReplayError("missing source rlog")
  schema = load_schema(schema_dir)
  streams, static, params, capture = defaultdict(list), {}, {}, {}
  maneuver = {"messages": 0, "valid": 0, "selected_messages": 0, "selected_valid": 0}
  for event in schema.Event.read_multiple_bytes(decompress_log(path), traversal_limit_in_words=2**31):
    name, t, valid = event.which(), int(event.logMonoTime) * 1e-9, bool(event.valid)
    if name in FIELDS:
      d = getattr(event, name)
      if set(FIELDS[name]) - set(d.schema.fields):
        raise ReplayError(f"missing recorded {name} fields")
      streams[name].append((t, valid, d.as_builder().to_bytes()))
    elif name in ("carParams", "carParamsSP"):
      if not valid:
        raise ReplayError(f"invalid recorded {name}")
      d = getattr(event, name)
      payload = d.as_builder().to_bytes()
      if name in static and static[name][1] != payload:
        raise ReplayError("recorded static CP/SP changed")
      static[name] = (t, payload, d.to_dict())
    elif name == "initData":
      for entry in event.initData.params.entries:
        key = str(entry.key)
        if key in PARAM_TYPES:
          value = bytes(entry.value).decode("utf-8")
          if key in params and params[key] != value:
            raise ReplayError("recorded Params snapshot changed")
          params[key] = value
    elif name == "narrowRoadEncodeIdx" and valid:
      d = event.narrowRoadEncodeIdx
      if int(d.segmentNum) == int(segment.name.rsplit("--", 1)[1]) and str(d.type) == "fullHEVC":
        add_capture(capture, int(d.frameId), int(d.timestampEof) * 1e-9)
    elif name == "lateralManeuverPlan":
      selected = rdf["input_t"][0] <= t <= rdf["input_t"][-1] + .05
      maneuver["messages"] += 1
      maneuver["valid"] += valid
      maneuver["selected_messages"] += bool(selected)
      maneuver["selected_valid"] += bool(selected and valid)
  provider = RecordedParams(params)
  validate_selection(provider)
  if set(static) != {"carParams", "carParamsSP"} or set(streams) != set(FIELDS):
    raise ReplayError("missing full recorded CP/SP or dynamic service")
  for frame, timestamp in zip(rdf["frame_id"], rdf["narrow_t"], strict=True):
    if int(frame) not in capture or abs(capture[int(frame)] - timestamp) > 1e-6:
      raise ReplayError("RDF frame/capture does not bind to this raw log")
  cp, sp = static["carParams"][2], static["carParamsSP"][2]
  required = ("carFingerprint", "lateralTuning", "steerLimitTimer", "steerAtStandstill", "minSteerSpeed", "steerControlType",
              "mass", "rotationalInertia", "wheelbase", "centerToFront", "tireStiffnessFront", "tireStiffnessRear", "steerRatio", "flags")
  if any(k not in cp for k in required) or cp["carFingerprint"] != "HYUNDAI_ELANTRA_2024" or "torque" not in cp["lateralTuning"]:
    raise ReplayError("missing full native CP/VM contract")
  if "neuralNetworkLateralControl" not in sp:
    raise ReplayError("missing recorded NN constructor asset identity")
  arrays = {"rdf_" + key: value for key, value in rdf.items()}
  arrays["t"] = rdf["input_t"][0] + np.arange(int(np.ceil((rdf["input_t"][-1] + .05 - rdf["input_t"][0]) / .01 - 1e-6))) * .01
  coverage = {}
  for name, rows in streams.items():
    rows.sort(key=lambda row: row[0])
    arrays[name + "_t"] = np.array([r[0] for r in rows])
    arrays[name + "_valid"] = np.array([r[1] for r in rows], bool)
    arrays[name + "_payload"], arrays[name + "_offsets"] = encode_payloads([r[2] for r in rows])
    coverage[name] = {"messages": len(rows), "valid": sum(r[1] for r in rows), "fields": FIELDS[name]}
  for name, (_, payload, _) in static.items():
    arrays[name] = np.frombuffer(payload, np.uint8).copy()
  receipt = {"source_segment": str(segment.resolve()), "source_log_sha256": digest(path), "params_raw": params,
             "params_types": PARAM_TYPES, "missing_recorded_params": sorted(set(PARAM_TYPES) - set(params)),
             "coverage": coverage, "static_publication_t": {k: v[0] for k, v in static.items()},
             "lateral_maneuver_plan_coverage": maneuver,
             "target_source": "Explicit diagnostic override to fresh RDF desiredCurvature; native lateralManeuverPlan target priority is not replayed.",
             "nn_asset_recorded_path": sp["neuralNetworkLateralControl"]["model"]["path"],
             "native_helper_sha256": digest(Path(native_common.__file__)),
             "expected_native_sources_normalized_sha256": source_hashes(repo),
             "recorded_static_config_timing": "CP/SP/Params are file snapshots, not a recovered runtime change history",
             "source_opendbc_gitlink": None, "source_opendbc_identity": "unknown; historical controller parity unproven"}
  return arrays, receipt


def native_replay(tape, receipt, repo, monitor):
  sys.path[:0] = [str(repo), str(repo / "opendbc_repo")]
  import openpilot.common.params as params_module

  provider = RecordedParams(receipt["params_raw"])
  validate_selection(provider)
  params_module.Params = lambda *args, **kwargs: provider
  from openpilot.cereal import custom, log, messaging
  from opendbc.car.structs import car
  from opendbc.car.hyundai.interface import CarInterface
  from opendbc.car.hyundai.carcontroller import MAX_ANGLE, MAX_ANGLE_FRAMES, MAX_ANGLE_CONSECUTIVE_FRAMES
  from opendbc.car.hyundai.values import CarControllerParams
  from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance
  from opendbc.car.vehicle_model import VehicleModel
  from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
  from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as V1
  from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as V0
  from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
  from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
  from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

  with car.CarParams.from_bytes(tape["carParams"].tobytes()) as d:
    cp = d.as_builder().as_reader()
  with custom.CarParamsSP.from_bytes(tape["carParamsSP"].tobytes()) as d:
    sp = d.as_builder().as_reader()
  validate_nn_binding(receipt["nn_asset_recorded_path"], str(sp.neuralNetworkLateralControl.model.path), receipt["nn_asset_sha256"], repo)
  ci, vm, limits = CarInterface(cp, sp), VehicleModel(cp), CarControllerParams(cp)
  if (MAX_ANGLE, MAX_ANGLE_FRAMES, MAX_ANGLE_CONSECUTIVE_FRAMES) != (85, 89, 2):
    raise ReplayError("native angle-fault contract differs")
  if tuple(getattr(limits, k) for k in ("STEER_MAX", "STEER_DELTA_UP", "STEER_DELTA_DOWN", "STEER_DRIVER_ALLOWANCE",
                                       "STEER_DRIVER_MULTIPLIER", "STEER_DRIVER_FACTOR")) != (409, 3, 7, 50, 2, 1):
    raise ReplayError("native Hyundai torque limit contract differs")
  rdf = {k[4:]: v for k, v in tape.items() if k.startswith("rdf_")}
  validate_lateral_rdf(rdf)
  if not np.allclose(rdf["horizon"], ModelConstants.T_IDXS, atol=1e-7, rtol=0):
    raise ReplayError("RDF horizon differs from native model constants")
  t = tape["t"]
  if len(t) < 2 or not np.isfinite(t).all() or not np.allclose(np.diff(t), .01, atol=1e-9, rtol=0):
    raise ReplayError("requires continuous explicit100Hz replay phase")
  joins = {name: asof_indices(tape[name + "_t"], tape[name + "_valid"], t,
                              max_age=2.5 if name in ("lateralTorqueParameters", "lateralDelay") else .2) for name in FIELDS}
  types = {"carState": car.CarState, "carControl": car.CarControl, "carOutput": car.CarOutput,
           "vehicleParameters": log.VehicleParameters, "lateralTorqueParameters": log.LateralTorqueParameters, "lateralDelay": log.LateralDelay}
  cache = {}

  def message(name, index):
    if name not in cache or cache[name][0] != index:
      raw = decode_payload(tape[name + "_payload"], tape[name + "_offsets"], int(index))
      with types[name].from_bytes(raw) as d:
        cache[name] = (index, d.as_builder())
    return cache[name][1]

  # Native all_checks frequency/alive logic, with publication time explicitly
  # substituted for unrecorded socket receive time. No socket is created.
  freq = messaging.FrequencyTracker(4., 100., False)
  learner_cursor = 0
  n = len(t)
  names = ("native_torque", "native_can_torque", "desired_curvature", "measured_curvature", "pid_i", "recorded_torque",
           "recorded_can_torque", "published_delay", "extension_lag", "driver_torque", "steering_angle", "steering_rate", "v", "a")
  out = {k: np.full(n, np.nan) for k in names}
  out.update(t=t.copy(), admitted=np.zeros(n, bool), initialization_uncertain=np.ones(n, bool),
             active=np.zeros(n, bool), steer_request=np.zeros(n, bool), torque_fault=np.zeros(n, bool),
             learner_applied=np.zeros(n, bool), steer_limited_feedback=np.zeros(n, bool),
             pre_limit_saturated=np.zeros(n, bool),
             controller_constructed=np.zeros(n, bool), reason=np.zeros(n, np.uint8), rdf_index=np.full(n, -1, np.int64))
  out["rdf_capture_age_s"] = np.full(n, np.nan)
  for name in FIELDS:
    index = joins[name][0]
    out[name + "_t"] = np.where(index >= 0, tape[name + "_t"][np.maximum(index, 0)], np.nan)
    out[name + "_valid"] = joins[name][1].copy()
  controller, previous, limited, applied_last, angle_counter, extension_lag, param_time = None, 0., False, 0, 0, 0., 0.
  for j, timestamp in enumerate(t):
    if j % 100 == 0:
      monitor.check_error()
    i, reason = admit_model(rdf, timestamp)
    required = ("carState", "carControl", "carOutput", "vehicleParameters", "lateralDelay")
    if any(not joins[name][1][j] for name in required):
      reason = 3
    if reason:
      out["reason"][j] = reason
      controller = None  # missing history forces explicitly uncertain fresh state
      continue
    cs, cc, co, lp, delay = (message(name, joins[name][0][j]) for name in required)
    numeric = [getattr(cs, k) for k in ("vEgo", "aEgo", "steeringAngleDeg", "steeringRateDeg", "steeringTorque")]
    numeric += [getattr(lp, k) for k in FIELDS["vehicleParameters"]] + [delay.lateralDelay, co.actuatorsOutput.torque]
    if not np.isfinite(numeric).all() or cs.vEgo < 0 or lp.steerRatio <= 0 or lp.stiffnessFactor <= 0 or delay.lateralDelay < 0:
      raise ReplayError("nonfinite/invalid measured native input")
    active = bool(cc.latActive)
    standstill = abs(cs.vEgo) <= max(cp.minSteerSpeed, .3) or cs.standstill
    if active and (cs.steerFaultTemporary or cs.steerFaultPermanent or (standstill and not cp.steerAtStandstill)):
      out["reason"][j], controller = 4, None
      continue
    if controller is None:
      controller = ControlsExt.initialize_lateral_control(SimpleNamespace(params=provider, CP=cp, CP_SP=sp), V1(cp, sp, ci, .01), ci, .01)
      if not isinstance(controller, V0):
        raise ReplayError("native selector did not construct v0")
      previous, limited, applied_last, angle_counter = 0., False, 0, 0
      extension_lag, param_time = provider.get("LagdValueCache"), timestamp - 3.01
      out["controller_constructed"][j] = True
    vm.update_params(max(lp.stiffnessFactor, .1), max(lp.steerRatio, .1))
    curvature = -vm.calc_curvature(math.radians(cs.steeringAngleDeg - lp.angleOffsetDeg), cs.vEgo, lp.roll)
    lt = tape["lateralTorqueParameters_t"]
    while learner_cursor < len(lt) and lt[learner_cursor] <= timestamp:
      freq.record_recv_time(float(lt[learner_cursor]))
      learner_cursor += 1
    li = joins["lateralTorqueParameters"][0][j]
    if joins["lateralTorqueParameters"][1][j] and timestamp - lt[li] < 2.5 and freq.valid:
      learner = message("lateralTorqueParameters", li)
      if learner.useParams:
        values = [learner.latAccelFactorFiltered, learner.latAccelOffsetFiltered, learner.frictionCoefficientFiltered]
        if not np.isfinite(values).all() or values[0] <= 0 or values[2] < 0:
          raise ReplayError("invalid admitted native learner values")
        controller.update_torque_parameters(*values)
        controller.extension.update_limits()
        out["learner_applied"][j] = True
    model = log.ModelDataV2.new_message()
    plan = rdf["plan"][i]
    for key, sl in (("position", Plan.POSITION), ("velocity", Plan.VELOCITY), ("acceleration", Plan.ACCELERATION),
                    ("orientation", Plan.T_FROM_CURRENT_EULER), ("orientationRate", Plan.ORIENTATION_RATE)):
      value = getattr(model, key)
      for axis, column in zip(("x", "y", "z"), plan[:, sl].T, strict=True):
        setattr(value, axis, column.astype(float).tolist())
      value.t = rdf["horizon"].tolist()
    controller.extension.update_model_v2(model)
    controller.extension.update_lateral_lag(extension_lag)
    if not active:
      controller.reset()  # inherited reset does NOT clear PID/filter/demand buffer
    previous, curvature_limited = clip_curvature(cs.vEgo, previous, float(rdf["desired_curvature"][i]) if active else curvature, lp.roll)
    # Current-master controlsd imports modeld.LAT_SMOOTH_SECONDS literal0.0.
    # It does not import this driving bundle's .1 action-time smoothing value.
    core_delay = float(delay.lateralDelay)
    torque, _, _ = controller.update(active, cs, vm, lp, limited, previous, None, curvature_limited, core_delay)
    torque = float(torque)
    if not np.isfinite(torque):
      raise ReplayError("native controller produced nonfinite torque request")
    applied, angle_counter, request = hardware_step(torque, applied_last, cs.steeringTorque, cs.steeringAngleDeg,
                                                    active, limits, angle_counter, apply_driver_steer_torque_limits, common_fault_avoidance)
    if not isinstance(applied, (int, np.integer)) or abs(applied) > 409 or (active and abs(applied - applied_last) > 7):
      raise ReplayError("native applied steering exceeds verified hardware bounds")
    applied_last = applied
    old_limited = limited
    if active:
      limited = abs(torque - co.actuatorsOutput.torque) > 1e-2  # publish feedback applies on NEXT tick
    out["admitted"][j], out["active"][j], out["rdf_index"][j] = True, active, i
    out["rdf_capture_age_s"][j] = timestamp - rdf["narrow_t"][i]
    out["native_torque"][j], out["native_can_torque"][j], out["desired_curvature"][j] = torque, applied, previous
    out["pre_limit_saturated"][j] = abs(torque) > 1
    out["steer_request"][j], out["torque_fault"][j], out["steer_limited_feedback"][j] = request, active and not request, old_limited
    out["measured_curvature"][j], out["pid_i"][j] = curvature, controller.pid.i
    out["recorded_torque"][j], out["recorded_can_torque"][j] = cc.actuators.torque, co.actuatorsOutput.torqueOutputCan
    out["published_delay"][j], out["extension_lag"][j] = core_delay, extension_lag
    for key, value in zip(("driver_torque", "steering_angle", "steering_rate", "v", "a"),
                          (cs.steeringTorque, cs.steeringAngleDeg, cs.steeringRateDeg, cs.vEgo, cs.aEgo), strict=True):
      out[key][j] = value
    if timestamp - param_time > 3:  # controlsd.get_params_sp runs AFTER state_control/publish
      extension_lag, param_time = get_lat_delay(provider, delay.lateralDelay), timestamp
  if not out["admitted"].any():
    raise ReplayError("no admitted native lateral command rows")
  if any(not np.isfinite(out[name][out["admitted"]]).all() for name in (*names, "rdf_capture_age_s")):
    raise ReplayError("nonfinite admitted native lateral diagnostic scalar")
  dependencies = {}
  for name, module in tuple(sys.modules.items()):
    filename = getattr(module, "__file__", None)
    if filename:
      path = Path(filename).resolve()
      if path.is_file() and path.is_relative_to(repo.resolve()) and path.suffix in (".py", ".so"):
        dependencies[name] = {"path": str(path), "sha256": digest(path)}
  return out, {"native_import": inspect.getfile(V0), "ci_import": inspect.getfile(CarInterface), "vm_import": inspect.getfile(VehicleModel),
               "params_read_keys": sorted(set(provider.reads)), "runtime_loaded_dependencies": dependencies,
               "params_provider": "read-only recorded provider injected before native module imports; no Params instance or publisher created"}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("prepare", "run"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
  parser.add_argument("--segment", type=Path)
  parser.add_argument("--schema-dir", type=Path)
  parser.add_argument("--rdf-output", type=Path)
  parser.add_argument("--nn-asset", type=Path)
  parser.add_argument("--input", type=Path)
  args = parser.parse_args()
  if args.out.suffix != ".npz":
    raise ReplayError("output must use .npz")
  if args.mode == "prepare":
    if None in (args.segment, args.schema_dir, args.rdf_output, args.nn_asset):
      raise ReplayError("prepare requires segment/schema/rdf-output/nn-asset")
    validate_output(args.out, [args.rdf_output, args.rdf_output.with_suffix(".json"), args.nn_asset], args.segment)
    rdf = dict(np.load(args.rdf_output, allow_pickle=False))
    arrays, receipt = prepare(args.segment, args.schema_dir, rdf, args.repo)
    receipt.update(mode="prepared", rdf_output_sha256=digest(args.rdf_output), nn_asset_sha256=digest(args.nn_asset),
                   prepare_adapter_sha256=digest(Path(__file__)), prepare_adapter_normalized_sha256=normalized_digest(Path(__file__)))
  else:
    if args.input is None:
      raise ReplayError("run requires --input")
    validate_output(args.out, [args.input, args.input.with_suffix(".json")])
    receipt = json.loads(args.input.with_suffix(".json").read_text())
    check_adapter_binding(receipt, Path(__file__))
    if (digest(args.input) != receipt["output_npz_sha256"]
        or digest(Path(native_common.__file__)) != receipt["native_helper_sha256"]
        or source_hashes(args.repo) != receipt["expected_native_sources_normalized_sha256"]):
      raise ReplayError("prepared input/native source hash mismatch")
    from rdf_bench import Monitor

    monitor = Monitor(80.)
    monitor.check()
    monitor.thread.start()
    try:
      arrays, imports = native_replay(dict(np.load(args.input, allow_pickle=False)), receipt, args.repo, monitor)
      monitor.check()
      monitor.check_error()
    finally:
      monitor.done.set()
      monitor.thread.join(timeout=3)
    receipt.update(mode="native_lateral_replay", monitor=monitor.snapshots, **imports,
                   initialization=("Cold constructor/PID/filter/request-buffer/previous-curvature/limiter state. " +
                                   "Unknown match to recorded state; inherited reset cannot remove uncertainty."),
                   availability="RDF only after conservative serial output_ready_t;132genuine model history excluded; capture age<=200ms",
                   activation=("Frozen recorded latActive, checked against CS faults/standstill/CP. Engagement/MADS/blinker decision " +
                               "state not reconstructed; missing IsMetric not consumed."),
                   learner_availability=("Native FrequencyTracker4Hz/100Hz +valid/useParams/alive2.5s, using publication as receive-time proxy; " +
                                         "exact historical socket receive phase unknown."),
                   feedback="Previous replay command vs causal recorded carOutput; frozen hardware feedback, no simulated vehicle response.",
                   delay=("Core published lateralDelay+current-master global0.0; extension initialized recorded LagdValueCache then " +
                          "get_lat_delay after publish every>3s. RDF catalog action_t smoothing not added here."),
                   wire_packing="Not performed; full CarController seed/HUD/CC_SP state not reconstructed. Native driver limiter and angle-fault request only.",
                   no_actuation=True, no_params_writes=True,
                   reason_codes={0: "admitted", 1: "model unavailable/history", 2: "model capture stale", 3: "required recorded service stale/invalid",
                                 4: "recorded activation conflicts with native fault/standstill gate", 5: "RDF declared unavailable"},
                   admitted_rows=int(arrays["admitted"].sum()),
                   reason_counts={int(k): int(v) for k, v in zip(*np.unique(arrays["reason"], return_counts=True), strict=True)})
  np.savez_compressed(args.out, **arrays)
  receipt.update(output=str(args.out.resolve()), output_npz_sha256=digest(args.out), rows=len(arrays["t"]), argv=sys.argv)
  receipt["adapter_sha256" if args.mode == "prepare" else "run_adapter_sha256"] = digest(Path(__file__))
  args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2)+"\n")
  print(json.dumps({k:receipt[k] for k in ("mode","output","rows","output_npz_sha256")}))


if __name__ == "__main__":
  main()
