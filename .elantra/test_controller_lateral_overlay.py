import numpy as np
import pytest
import hashlib
from pathlib import Path

from controller_lateral_overlay import lateral_index, validate_lateral_output
from controller_model_overlay import RenderError


def evidence():
  rt = np.arange(140) * .05
  rdf = {"input_t": rt, "narrow_t": rt, "output_ready_t": rt + .04,
         "scored_mask": np.arange(140) >= 132, "controller_admissible": np.arange(140) >= 132,
         "deadline_admissible": np.ones(140, bool), "desired_curvature": np.zeros(140)}
  t = np.arange(700) * .01
  index = np.searchsorted(rdf["output_ready_t"], t, side="right") - 1
  admitted = (index >= 132) & (index < 140)
  n = {"t": t, "admitted": admitted, "initialization_uncertain": np.ones(700, bool),
       "rdf_index": np.where(admitted, index, -1), "reason": np.where(admitted, 0, 1).astype(np.uint8)}
  for key in ("active", "steer_request", "learner_applied"):
    n[key] = admitted.copy()
  for key in ("torque_fault", "pre_limit_saturated", "steer_limited_feedback", "controller_constructed"):
    n[key] = np.zeros(700, bool)
  first = np.flatnonzero(admitted)[0]
  n["controller_constructed"][first] = True
  names = ("native_torque", "native_can_torque", "desired_curvature", "measured_curvature", "pid_i", "recorded_torque",
           "recorded_can_torque", "published_delay", "extension_lag", "driver_torque", "steering_angle", "steering_rate", "v", "a")
  for key in names:
    n[key] = np.where(admitted, 0., np.nan)
  n["rdf_capture_age_s"] = np.where(admitted, t - rt[np.maximum(index, 0)], np.nan)
  r = {"mode": "native_lateral_replay", "rows": 700, "no_actuation": True, "no_params_writes": True,
       "initialization": "cold; unknown all controller state", "rdf_output_sha256": "a" * 64,
       "source_log_sha256": "b" * 64, "output_npz_sha256": "c" * 64,
       "prepare_adapter_sha256": "d" * 64, "prepare_adapter_normalized_sha256": "d" * 64,
       "run_adapter_sha256": "d" * 64, "adapter_sha256": "d" * 64}
  helper = Path(__file__).resolve().parent.parent / "opendbc_repo/opendbc/car/lateral.py"
  r["expected_native_sources_normalized_sha256"] = {
    "opendbc_repo/opendbc/car/lateral.py": hashlib.sha256(helper.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
  return n, r, rdf


def validate(n, r, rdf):
  validate_lateral_output(n, r, rdf, "a" * 64, "b" * 64, "c" * 64, "d" * 64)


def test_native_lateral_valid_contract_and_no_backfill():
  n, r, rdf = evidence()
  validate(n, r, rdf)
  i = np.flatnonzero(n["admitted"])[0]
  assert lateral_index(n, n["t"][i] - .001) is None
  assert lateral_index(n, n["t"][i] + .001) == i
  n["admitted"][i + 1] = False
  assert lateral_index(n, n["t"][i + 1] + .001) is None
  assert lateral_index(n, n["t"][-1] + .02) is None


@pytest.mark.parametrize("key", ["rdf_output_sha256", "source_log_sha256", "output_npz_sha256", "run_adapter_sha256",
                               "prepare_adapter_normalized_sha256"])
def test_reject_wrong_evidence_binding(key):
  n, r, rdf = evidence()
  r[key] = "0" * 64
  with pytest.raises(RenderError, match="hash"):
    validate(n, r, rdf)


@pytest.mark.parametrize("change", ["clock", "shape", "flags", "future", "stale", "warmup", "finite_gap", "integer", "driver", "request",
                                   "coverage", "helper", "driver_yield"])
def test_reject_bad_lateral_commands_or_admission(change):
  n, r, rdf = evidence()
  i = np.flatnonzero(n["admitted"])[0] + 5
  if change == "clock":
    n["t"][i] += .001
  elif change == "shape":
    n["native_torque"] = n["native_torque"][:-1]
  elif change == "flags":
    n["admitted"] = n["admitted"].astype(float)
  elif change == "future":
    n["rdf_index"][i] += 1
  elif change == "stale":
    n["rdf_capture_age_s"][i] = .3
  elif change == "warmup":
    rdf["controller_admissible"][132] = False
  elif change == "finite_gap":
    n["native_torque"][0] = 0.
  elif change == "integer":
    n["native_can_torque"][i] = .5
  elif change == "driver":
    n["native_can_torque"][i] = 409
  elif change == "request":
    n["steer_request"][i] = False
  elif change == "coverage":
    n = {key: value[:-1] for key, value in n.items()}
    r["rows"] -= 1
  elif change == "helper":
    r["expected_native_sources_normalized_sha256"]["opendbc_repo/opendbc/car/lateral.py"] = "0" * 64
  elif change == "driver_yield":
    first = np.flatnonzero(n["admitted"])[0]
    n["native_torque"][first] = .4
    n["native_can_torque"][first] = 3
    n["driver_torque"][first] = -300  # real opposing-driver bound yields zero, not three counts
  with pytest.raises(RenderError):
    validate(n, r, rdf)


def test_finite_overrange_raw_request_admitted_only_with_real_native_clamp():
  n, r, rdf = evidence()
  i = np.flatnonzero(n["admitted"])[0]
  n["native_torque"][i] = 3.5
  n["pre_limit_saturated"][i] = True
  n["native_can_torque"][i] = 3
  validate(n, r, rdf)


def test_renderer_rejects_two_native_modes_and_input_receipt_collision(tmp_path, monkeypatch):
  from types import SimpleNamespace
  import controller_video
  monkeypatch.setattr(controller_video.shutil, "which", lambda key: key)
  with pytest.raises(RenderError, match="simultaneous"):
    controller_video.render(SimpleNamespace(native_output=tmp_path / "long.npz", native_lateral_output=tmp_path / "lat.npz",
                                            rdf_output=tmp_path / "rdf.npz"))
  with pytest.raises(RenderError, match="collision"):
    controller_video.validate_output_path(tmp_path / "lat.mp4", tmp_path / "raw--0", tmp_path / "data.npz", tmp_path / "replay.npz",
                                          (tmp_path / "lat.npz", tmp_path / "lat.json"))
