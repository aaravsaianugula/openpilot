"""Synthetic adapter fixtures; actual native API is an explicit later opt-in."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

import controller_native_longitudinal_output as output


class Static(NS):
  def to_dict(self):
    return deepcopy(vars(self))


class FixtureController:
  """A spy with arbitrary outputs, expressly not a Hyundai control-law clone."""

  def __init__(self, cp, sp):
    self.CP, self.CP_SP, self.calls = cp, sp, []
    self.car_config = NS(
      lookahead_jerk_bp=[2.0, 5.0, 20.0],
      lookahead_jerk_upper_v=[0.3, 0.45, 0.6],
      lookahead_jerk_lower_v=[0.3, 0.45, 0.6],
      longitudinal_actuator_delay=0.5,
      jerk_limits=4.0,
    )
    for key in output.STATE_FLOATS:
      setattr(self, key, 0.0)
    self.stopping, self.stopping_count, self.long_control_state_last = False, 0, "off"
    self.tuning = NS(**{key: False if key == "stopping" else 0.0 for key in output.TUNING_FIELDS})

  @property
  def enabled(self):
    return bool(self.CP_SP.flags)

  def update(self, cc, cs):
    self.calls.append((cc, cs))
    self.accel_cmd = cc.actuators.accel
    self.desired_accel = cc.actuators.accel
    self.actual_accel = cc.actuators.accel + cs.out.vEgo / 100 + cs.out.aEgo / 10 + self.CP_SP.flags
    self.accel_last = self.actual_accel
    self.long_control_state_last = cc.actuators.longControlState
    self.stopping = cc.actuators.longControlState == "stopping"
    self.stopping_count += int(self.stopping)
    self.tuning = NS(**{key: getattr(self, key) for key in output.TUNING_FIELDS})


def static():
  return (
    {
      "carFingerprint": "HYUNDAI_ELANTRA_2024",
      "brand": "hyundai",
      "radarUnavailable": False,
      "openpilotLongitudinalControl": True,
      "stopAccel": -2.0,
      "longitudinalTuning": {"kiBP": [0.0, 30.0], "kiV": [0.07, 0.12]},
    },
    {"flags": 0, "enableGasInterceptor": False},
  )


def fixture_runtime():
  bindings = output.Bindings(
    FixtureController,
    lambda c, s: (Static(**deepcopy(c)), Static(**deepcopy(s))),
    lambda a, state, active: NS(actuators=NS(accel=a, longControlState=state), longActive=active),
    lambda v, a: NS(out=NS(vEgo=v, aEgo=a)),
    actual_native=False,
  )
  return output.Runtime(bindings, {}, output.OPENDBC_PIN, output.SUPERPROJECT_HEAD)


def make(runtime=None, cp=None, sp=None):
  c, s = static()
  c, s = cp if cp is not None else c, sp if sp is not None else s
  return (runtime or fixture_runtime()).create(c, s, static_identity_sha256=output.static_identity(c, s))


def step(handle, t=0.0, **changes):
  values = {"t": t, "apply_call": True, "requested_accel": 1.0, "long_state": "pid", "long_active": True, "simulated_v": 10.0, "simulated_a": 0.2}
  values.update(changes)
  return handle.step(**values)


def test_native_update_phase_and_held_report_follow_apply_count_not_wall_clock():
  h = make()
  rows = [step(h, i * 0.37, requested_accel=float(i)) for i in range(11)]
  assert [r["apply_frame_before"] for r in rows if r["native_updated"]] == [0, 5, 10]
  assert len(h.controller.calls) == 3
  assert rows[1]["published_accel"] == rows[0]["held_accel_after_apply"]
  assert all(rows[i]["held_accel_after_apply"] == rows[0]["held_accel_after_apply"] for i in range(1, 5))
  assert rows[5]["published_accel"] == rows[0]["held_accel_after_apply"]
  assert rows[5]["held_accel_after_apply"] != rows[5]["published_accel"]


def test_skipped_apply_does_not_advance_phase_or_reset_state():
  h = make()
  skipped = step(h, apply_call=False)
  assert skipped["apply_frame_before"] == skipped["apply_frame_after"] == 0
  assert skipped["published_accel"] == skipped["held_accel_after_apply"] == 0.0
  first = step(h, 9.0)
  held = step(h, 20.0, apply_call=False, long_active=False, long_state="off", requested_accel=-3.0)
  assert held["apply_frame_after"] == first["apply_frame_after"] == 1
  assert held["native_after"] == first["native_after"] and len(h.controller.calls) == 1
  assert step(h, 20.001)["apply_frame_before"] == 1


def test_constructor_cold_state_and_publication_validity_remain_uncertain():
  h = make()
  first = step(h)
  assert first["published_accel"] == 0.0 and first["initialization_uncertain"]
  assert first["published_event_valid"] is None and not first["selection_eligible"]
  assert first["safety_acceptance"] == "not_evaluated" and first["delivered_force"] == "not_identified"
  invalid = step(h, 1.0, published_event_valid=False)
  assert invalid["published_event_valid"] is False and invalid["published_accel"] == first["held_accel_after_apply"]
  assert step(h, 2.0, published_event_valid=True)["published_event_valid"] is True


def test_flags_stopping_and_simulated_motion_are_passed_to_factory_without_clipping():
  cp, sp = static()
  sp["flags"] = 16
  h = make(cp=cp, sp=sp)
  row = step(h, requested_accel=5.0, long_state="stopping", simulated_v=20.0, simulated_a=-1.0)
  cc, cs = h.controller.calls[0]
  assert (cc.actuators.accel, cc.actuators.longControlState, cs.out.vEgo, cs.out.aEgo) == (5.0, "stopping", 20.0, -1.0)
  assert h.controller.CP_SP.flags == 16 and row["native_after"]["tuning"]["stopping"]
  assert row["held_accel_after_apply"] == pytest.approx(21.1)  # arbitrary spy result, not a native law
  assert row["actual_native"] is False


def test_per_trial_state_static_copy_and_receipt_copies_are_independent():
  cp, sp = static()
  a = make(cp=cp, sp=sp)
  b = make(cp=cp, sp=sp)
  cp["radarUnavailable"], sp["flags"] = True, 32
  row = step(a)
  row["native_after"]["tuning"]["actual_accel"] = 999.0
  assert step(b)["published_accel"] == 0.0 and not a.controller.CP.radarUnavailable
  assert step(a, 1.0)["published_accel"] != 999.0
  receipt = a.receipt()
  assert set(receipt["native_state"]["tuning"]) == set(output.TUNING_FIELDS)
  assert set(receipt["native_state"]["car_config"]) == set(output.CONFIG_FIELDS)
  assert receipt["static_identity_sha256"] == output.static_identity(*static())


@pytest.mark.parametrize(
  "changes",
  [
    {"simulated_v": -1.0},
    {"simulated_v": float("nan")},
    {"simulated_a": float("inf")},
    {"simulated_a": True},
    {"requested_accel": float("nan")},
    {"t": True},
    {"apply_call": 1},
    {"long_active": 1},
    {"long_state": "unknown"},
    {"published_event_valid": 1},
  ],
)
def test_invalid_inputs_do_not_mutate_handle(changes):
  h = make()
  with pytest.raises(output.OutputError):
    step(h, **changes)
  assert h.frame == 0 and h.last_t is None and h.controller.calls == []


def test_clock_must_increase_and_future_recorded_feedback_has_no_input_slot():
  h = make()
  step(h, 3.0)
  with pytest.raises(output.OutputError, match="clock"):
    step(h, 3.0)
  with pytest.raises(TypeError):
    step(h, 4.0, recorded_car_output=123.0)
  with pytest.raises(TypeError):
    step(h, 4.0, future_ground_truth={"v": 0.0})


@pytest.mark.parametrize("which,key", [("cp", "radarUnavailable"), ("sp", "flags"), ("cp", "stopAccel")])
def test_missing_consumed_static_contract_is_rejected(which, key):
  cp, sp = static()
  del (cp if which == "cp" else sp)[key]
  with pytest.raises(output.OutputError):
    make(cp=cp, sp=sp)


def test_static_binding_and_live_static_mutation_fail_closed():
  cp, sp = static()
  with pytest.raises(output.OutputError, match="identity"):
    fixture_runtime().create(cp, sp, static_identity_sha256="0" * 64)
  h = make()
  h.controller.CP.radarUnavailable = True
  with pytest.raises(output.OutputError, match="static"):
    step(h)


def test_fixture_cannot_self_declare_actual_native_factory():
  r = fixture_runtime()
  r.bindings = replace(r.bindings, actual_native=True)
  with pytest.raises(output.OutputError, match="native"):
    make(runtime=r)


def test_native_failure_poisons_handle_without_an_invented_reset():
  h = make()

  def broken(cc, cs):
    raise RuntimeError("synthetic factory failure")

  h._update = broken
  with pytest.raises(output.OutputError, match="native update"):
    step(h)
  with pytest.raises(output.OutputError, match="failed"):
    step(h, 1.0)


def test_runtime_binding_mutation_cannot_change_an_existing_trial():
  r = fixture_runtime()
  h = make(runtime=r)
  r.bindings = replace(r.bindings, control_builder=lambda a, s, active: NS(actuators=NS(accel=999.0, longControlState=s), longActive=active))
  assert step(h)["held_accel_after_apply"] == pytest.approx(1.12)


def test_native_state_injection_cannot_supply_unrecorded_feedback():
  h = make()
  step(h)
  h.controller.tuning.actual_accel = 999.0
  with pytest.raises(output.OutputError, match="state"):
    step(h, 1.0)


def test_foreign_cached_parent_is_rejected_before_any_native_import(monkeypatch, tmp_path):
  repo = tmp_path / "source"
  monkeypatch.setattr(output, "os", NS(name="posix", environ={}))
  monkeypatch.setattr(output, "source_hashes", lambda r: dict(output.PINNED))
  monkeypatch.setattr(output, "bootstrap_hashes", lambda r: dict(output.BOOTSTRAP))

  def git_fixture(command, **kwargs):
    if "ls-tree" in command:
      text = f"160000 commit {output.OPENDBC_PIN}\topendbc_repo\n"
    else:
      text = (output.OPENDBC_PIN if Path(command[2]).name == "opendbc_repo" else output.SUPERPROJECT_HEAD) + "\n"
    return NS(returncode=0, stdout=text)

  monkeypatch.setattr(output.subprocess, "run", git_fixture)
  monkeypatch.setitem(sys.modules, "opendbc", NS(__file__=str(tmp_path / "foreign/__init__.py"), __path__=[str(tmp_path / "foreign")]))
  calls = []

  def import_spy(name):
    calls.append(name)
    raise AssertionError("foreign code would execute here")

  monkeypatch.setattr(output.importlib, "import_module", import_spy)
  with pytest.raises(output.OutputError, match="cached"):
    output.load_runtime(repo, dict(output.PINNED), output.OPENDBC_PIN, output.SUPERPROJECT_HEAD, expected_bootstrap=dict(output.BOOTSTRAP))
  assert calls == []


@pytest.mark.parametrize("namespace", [False, True])
def test_cached_bootstrap_search_path_is_canonical_before_import(monkeypatch, tmp_path, namespace):
  repo = tmp_path / "source"
  initializer = repo / "opendbc_repo/opendbc/__init__.py"
  expected = dict(output.BOOTSTRAP)
  if namespace:
    expected["opendbc_repo/opendbc/__init__.py"] = None
  monkeypatch.setattr(output, "BOOTSTRAP", expected)
  monkeypatch.setattr(output, "os", NS(name="posix", environ={}))
  monkeypatch.setattr(output, "source_hashes", lambda r: dict(output.PINNED))
  monkeypatch.setattr(output, "bootstrap_hashes", lambda r: dict(expected))

  def git_fixture(command, **kwargs):
    if "ls-tree" in command:
      text = f"160000 commit {output.OPENDBC_PIN}\topendbc_repo\n"
    else:
      text = (output.OPENDBC_PIN if Path(command[2]).name == "opendbc_repo" else output.SUPERPROJECT_HEAD) + "\n"
    return NS(returncode=0, stdout=text)

  monkeypatch.setattr(output.subprocess, "run", git_fixture)
  monkeypatch.setitem(sys.modules, "opendbc", NS(__file__=None if namespace else str(initializer), __path__=[str(initializer.parent / "alternate")]))
  calls = []

  def import_spy(name):
    calls.append(name)
    raise AssertionError("alternate in-root child would execute here")

  monkeypatch.setattr(output.importlib, "import_module", import_spy)
  with pytest.raises(output.OutputError, match="cached"):
    output.load_runtime(repo, dict(output.PINNED), output.OPENDBC_PIN, output.SUPERPROJECT_HEAD, expected_bootstrap=expected)
  assert calls == []


@pytest.mark.parametrize("mutation", ["modified", "added", "removed"])
def test_package_bootstrap_changes_are_rejected_before_import(monkeypatch, tmp_path, mutation):
  repo = tmp_path / "source"
  existing, absent = "opendbc_repo/opendbc/__init__.py", "opendbc_repo/opendbc/namespace/__init__.py"
  path = repo / existing
  path.parent.mkdir(parents=True)
  path.write_bytes(b"# legitimate fixture initializer\n")
  expected = {existing: hashlib.sha256(path.read_bytes()).hexdigest(), absent: None}
  monkeypatch.setattr(output, "BOOTSTRAP", expected)
  monkeypatch.setattr(output, "os", NS(name="posix", environ={}))
  monkeypatch.setattr(output, "source_hashes", lambda r: dict(output.PINNED))
  if mutation == "modified":
    path.write_bytes(b"raise RuntimeError('foreign initializer')\n")
  elif mutation == "added":
    path = repo / absent
    path.parent.mkdir()
    path.write_bytes(b"raise RuntimeError('unexpected namespace initializer')\n")
  else:
    path.unlink()
  calls = []
  monkeypatch.setattr(output.importlib, "import_module", lambda name: calls.append(name))
  with pytest.raises(output.OutputError, match="bootstrap"):
    output.load_runtime(repo, dict(output.PINNED), output.OPENDBC_PIN, output.SUPERPROJECT_HEAD, expected_bootstrap=expected)
  assert calls == []


@pytest.mark.parametrize("field,value", [("frame", 4), ("held", 999.0), ("last_t", 99.0), ("failed", True)])
@pytest.mark.parametrize("operation", ["step", "receipt"])
def test_owned_phase_output_clock_and_poison_mutation_is_rejected(field, value, operation):
  h = make()
  setattr(h, field, value)
  with pytest.raises(output.OutputError, match="owned"):
    step(h) if operation == "step" else h.receipt()
  assert h.controller.calls == []


def test_external_poison_clear_cannot_resume_failed_handle():
  h = make()
  h._update = lambda cc, cs: (_ for _ in ()).throw(RuntimeError("fixture failure"))
  with pytest.raises(output.OutputError, match="native update"):
    step(h)
  h.failed = False
  with pytest.raises(output.OutputError, match="owned"):
    step(h, 1.0)
  with pytest.raises(output.OutputError, match="owned"):
    h.receipt()


class SyntheticEnumReader:
  """Only the documented pycapnp enum shape; no actual native execution."""

  def __init__(self, raw, name):
    self.raw, self.name = raw, name

  def __str__(self):
    return self.name


class NativeShapedFixtureController(FixtureController):
  def __init__(self, cp, sp):
    super().__init__(cp, sp)
    self.long_control_state_last = 0  # actual pinned constructor uses LongCtrlState.off

  def update(self, cc, cs):
    super().update(cc, cs)
    name = cc.actuators.longControlState
    self.long_control_state_last = SyntheticEnumReader(output.STATES.index(name), name)


def native_shaped_fixture_runtime(monkeypatch):
  monkeypatch.setitem(sys.modules, "capnp.lib.capnp", NS(_DynamicEnum=SyntheticEnumReader))
  r = fixture_runtime()
  r.bindings = replace(r.bindings, controller_type=NativeShapedFixtureController)
  return r


def test_native_shaped_integer_constructor_and_reader_update_states(monkeypatch):
  h = make(runtime=native_shaped_fixture_runtime(monkeypatch))
  initial = h.receipt()["cold_initial_state"]
  assert initial["long_control_state_last"] == "off" and initial["long_control_state_last_ordinal"] == 0
  assert initial["long_control_state_last_representation"] == "integer"
  for i in range(16):
    name = ("pid", "stopping", "starting", "off")[i // 5]
    row = step(h, float(i), long_state=name)
    state = row["native_after"]
    assert state["long_control_state_last"] == name
    assert state["long_control_state_last_ordinal"] == output.STATES.index(name)
    assert state["long_control_state_last_representation"] == "dynamic_reader"


@pytest.mark.parametrize(
  "invalid",
  [True, False, -1, 4, 99, 0.0, "0", "unknown", NS(raw=0), SyntheticEnumReader(True, "pid"), SyntheticEnumReader(99, "unknown"), SyntheticEnumReader(0, "pid")],
)
def test_native_state_enum_rejects_non_native_or_unknown_values(monkeypatch, invalid):
  native_shaped_fixture_runtime(monkeypatch)
  cp, sp = static()
  controller = FixtureController(Static(**cp), Static(**sp))
  controller.long_control_state_last = invalid
  with pytest.raises(output.OutputError):
    output._snapshot(controller)


def test_native_state_representation_change_is_fenced_outside_update(monkeypatch):
  h = make(runtime=native_shaped_fixture_runtime(monkeypatch))
  h.controller.long_control_state_last = "off"
  with pytest.raises(output.OutputError, match="state"):
    step(h)


def test_actual_native_longitudinal_output_apply_hold_flags_and_stopping():
  """Opt-in actual source/API proof only; all CP, requests and motion are synthetic."""
  keys = ("NATIVE_LONG_OUTPUT_SOURCE_ROOT", "NATIVE_LONG_OUTPUT_SOURCE_MANIFEST", "NATIVE_LONG_OUTPUT_MANIFEST_SHA256")
  if os.name != "posix" or not all(os.environ.get(k) for k in keys):
    pytest.skip("Owner-reviewed isolated POSIX API run required")
  repo, manifest_path = Path(os.environ[keys[0]]), Path(os.environ[keys[1]])
  raw = manifest_path.read_bytes()
  assert hashlib.sha256(raw).hexdigest() == os.environ[keys[2]]
  manifest = json.loads(raw)
  r = output.load_runtime(
    repo, manifest["source_hashes"], manifest["opendbc_gitlink"], manifest["superproject_head"], expected_bootstrap=manifest["bootstrap_sources"]
  )
  from opendbc.car.structs import car
  from openpilot.cereal import custom

  cp = car.CarParams.new_message(
    carFingerprint="HYUNDAI_ELANTRA_2024", brand="hyundai", radarUnavailable=False, openpilotLongitudinalControl=True, stopAccel=-2.0
  )
  cp.longitudinalTuning.kiBP, cp.longitudinalTuning.kiV = [0.0, 30.0], [0.07, 0.12]
  sp = custom.CarParamsSP.new_message(flags=0, enableGasInterceptor=False)
  h = make(runtime=r, cp=cp.to_dict(), sp=sp.to_dict())
  first = step(h, requested_accel=1.0)
  assert first["actual_native"] and first["published_accel"] == 0.0
  assert first["held_accel_after_apply"] == pytest.approx(1.0, abs=1e-7)
  for i in range(1, 5):
    assert step(h, i, requested_accel=-2.0)["held_accel_after_apply"] == pytest.approx(1.0, abs=1e-7)
  changed = step(h, 5.0, requested_accel=-2.0)
  assert changed["published_accel"] == pytest.approx(1.0, abs=1e-7)
  assert changed["held_accel_after_apply"] == pytest.approx(-2.0, abs=1e-7)
  sp.flags = 16  # explicit synthetic dynamic policy, actual native flag semantics
  dynamic = make(runtime=r, cp=cp.to_dict(), sp=sp.to_dict())
  row = step(dynamic, requested_accel=1.0, simulated_v=5.0, simulated_a=0.0)
  assert row["held_accel_after_apply"] == pytest.approx(0.15, abs=1e-7)
  stopping = make(runtime=r, cp=cp.to_dict(), sp=sp.to_dict())
  row = step(stopping, requested_accel=1.0, long_state="stopping")
  assert row["native_after"]["tuning"]["stopping"] and row["held_accel_after_apply"] == 0.0
  assert row["initialization_uncertain"] and not row["selection_eligible"]
