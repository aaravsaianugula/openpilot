"""Strict admission of native lateral command diagnostics into evidence videos."""
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from controller_model_overlay import RenderError, native_index

lateral_index = native_index
SCALARS = ("native_torque", "native_can_torque", "desired_curvature", "measured_curvature", "pid_i", "recorded_torque",
           "recorded_can_torque", "published_delay", "extension_lag", "driver_torque", "steering_angle", "steering_rate", "v", "a",
           "rdf_capture_age_s")
FLAGS = ("admitted", "initialization_uncertain", "active", "steer_request", "torque_fault", "pre_limit_saturated",
         "learner_applied", "steer_limited_feedback", "controller_constructed")


def validate_lateral_output(native, receipt, rdf, rdf_sha, log_sha, output_sha, adapter_sha):
  for key, actual in (("rdf_output_sha256", rdf_sha), ("source_log_sha256", log_sha), ("output_npz_sha256", output_sha),
                      ("prepare_adapter_sha256", adapter_sha), ("run_adapter_sha256", adapter_sha), ("adapter_sha256", adapter_sha),
                      ("prepare_adapter_normalized_sha256", adapter_sha)):
    if receipt.get(key) != actual:
      raise RenderError(f"native lateral evidence hash mismatch: {key}")
  if (receipt.get("mode") != "native_lateral_replay" or receipt.get("no_actuation") is not True
      or receipt.get("no_params_writes") is not True or not isinstance(receipt.get("initialization"), str) or not receipt["initialization"]):
    raise RenderError("native lateral diagnostic receipt required")
  required = ("t", "rdf_index", "reason", *SCALARS, *FLAGS)
  if any(key not in native for key in required):
    raise RenderError("missing native lateral arrays")
  n = len(native["t"])
  expected_count = int(np.ceil((rdf["input_t"][-1] + .05 - rdf["input_t"][0]) / .01 - 1e-6))
  if n < 2 or n != expected_count or receipt.get("rows") != n or any(np.asarray(native[k]).shape != (n,) for k in required):
    raise RenderError("native lateral arrays/receipt shapes disagree")
  if (np.asarray(native["t"]).dtype.kind not in "fi" or not np.isfinite(native["t"]).all()
      or not np.allclose(np.diff(native["t"]), .01, atol=1e-8, rtol=0)
      or abs(native["t"][0] - rdf["input_t"][0]) > 1e-8):
    raise RenderError("native lateral clock must preserve its continuous100Hz input phase")
  if any(np.asarray(native[k]).dtype.kind != "b" for k in FLAGS):
    raise RenderError("native lateral flags must be explicit booleans")
  admitted = native["admitted"]
  if (np.asarray(native["rdf_index"]).dtype.kind != "i" or np.asarray(native["reason"]).dtype.kind not in "iu"
      or not np.array_equal(admitted, native["reason"] == 0) or np.any(native["rdf_index"][~admitted] != -1)):
    raise RenderError("native lateral admission/index/reason mismatch")
  if not native["initialization_uncertain"][admitted].all():
    raise RenderError("cold native lateral initialization must remain uncertain")
  for key in SCALARS:
    a = np.asarray(native[key])
    if a.dtype.kind not in "fi" or not np.isfinite(a[admitted]).all() or not np.isnan(a[~admitted]).all():
      raise RenderError("native lateral scalars must be finite only on admitted rows")
  indices = native["rdf_index"][admitted]
  expected = np.searchsorted(rdf["output_ready_t"], native["t"][admitted], side="right") - 1
  if np.any(indices < 0) or np.any(indices >= len(rdf["input_t"])) or not np.array_equal(indices, expected):
    raise RenderError("native lateral used future or mismatched RDF output")
  for key in ("scored_mask", "controller_admissible", "deadline_admissible"):
    if (key not in rdf or np.asarray(rdf[key]).shape != rdf["input_t"].shape
        or np.asarray(rdf[key]).dtype.kind != "b" or not rdf[key][indices].all()):
      raise RenderError("native lateral violated explicit RDF admission/warmup")
  age = native["t"][admitted] - rdf["narrow_t"][indices]
  if (np.any(age < 0) or np.any(age > .2 + 1e-8)
      or not np.allclose(age, native["rdf_capture_age_s"][admitted], atol=1e-8, rtol=0)
      or not np.isfinite(rdf["desired_curvature"][indices]).all()):
    raise RenderError("native lateral RDF capture age/target invalid")
  constructed = admitted & ~np.r_[False, admitted[:-1]]
  if not np.array_equal(native["controller_constructed"], constructed):
    raise RenderError("native lateral cold-state boundary mismatch")
  if (not np.array_equal(native["pre_limit_saturated"][admitted], np.abs(native["native_torque"][admitted]) > 1)
      or np.any(native["native_can_torque"][admitted] != np.round(native["native_can_torque"][admitted]))):
    raise RenderError("native lateral raw saturation or applied integer counts invalid")
  validate_enforcement(native, receipt)


def validate_enforcement(native, receipt):
  """Recompute only actual native driver/rate/angle enforcement, not vehicle response."""
  repo = Path(__file__).resolve().parent.parent
  sys.path.insert(0, str(repo / "opendbc_repo"))
  from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance

  helper = repo / "opendbc_repo/opendbc/car/lateral.py"
  expected_sha = receipt.get("expected_native_sources_normalized_sha256", {}).get("opendbc_repo/opendbc/car/lateral.py")
  actual_sha = hashlib.sha256(helper.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
  if Path(inspect.getfile(apply_driver_steer_torque_limits)).resolve() != helper.resolve() or expected_sha != actual_sha:
    raise RenderError("native lateral enforcement helper source hash mismatch")
  limits = SimpleNamespace(STEER_MAX=409, STEER_DELTA_UP=3, STEER_DELTA_DOWN=7,
                           STEER_DRIVER_ALLOWANCE=50, STEER_DRIVER_MULTIPLIER=2, STEER_DRIVER_FACTOR=1)
  previous, counter = 0, 0
  for j in np.flatnonzero(native["admitted"]):
    if native["controller_constructed"][j]:
      previous, counter = 0, 0
    active = bool(native["active"][j])
    applied = apply_driver_steer_torque_limits(int(round(native["native_torque"][j] * 409)), previous,
                                               native["driver_torque"][j], limits)
    counter, request = common_fault_avoidance(abs(native["steering_angle"][j]) >= 85, active, counter, 89, 2)
    applied = applied if active else 0
    if (native["native_can_torque"][j] != applied or abs(applied) > 409
        or native["steer_request"][j] != request or native["torque_fault"][j] != (active and not request)):
      raise RenderError("native lateral driver/rate/409 or angle-request enforcement mismatch")
    previous = applied
