"""Synthetic source/observer contracts, never policy authentication or native proof."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_controller_observation import obs, observer, records, sm

pytest_plugins = ("test_controller_observation",)  # shared isolated schema fixture; fresh process per test module

ROOT = Path(__file__).resolve().parents[1]


def policy_records(o):
  return [r["policy"] for r in records(o) if "policy" in r]


def test_one_completion_preserves_value_exception_and_no_begin_queue_record():
  o = observer()
  with o.policy.group("controlsInit") as group:
    consumer = group.consumer("controls")
    before = len(records(o))
    with consumer.read("selectionEnforce") as read:
      assert len(records(o)) == before
      value = False
      read.returned(value)
    failure = RuntimeError("VIN_PRIVATE_DO_NOT_EXPORT")
    with pytest.raises(RuntimeError) as caught:
      with consumer.read("selectionTune"):
        raise failure
    assert caught.value is failure
  rows = policy_records(o)
  assert [r["kind"] for r in rows] == ["consumerCreated", "readReturned", "readRaised", "groupComplete"]
  assert rows[1]["scalar_kind"] == "bool" and rows[1]["bool_value"] is value
  assert rows[2]["scalar_kind"] == "unknown"
  assert "VIN_PRIVATE" not in repr(rows)
  assert rows[-1]["attempts"] == 2 and rows[-1]["returned"] == rows[-1]["raised"] == 1
  assert rows[-1]["complete"] and rows[1]["begin_ns"] < rows[1]["end_ns"]
  assert all(r["capture_bytes"] <= 1024 and not r["refs"] for r in records(o))


@pytest.mark.parametrize("value,kind", [(None, "none"), (True, "bool"), (2**63-1, "int"), (-2**63, "int"), (.25, "float"),
                                        (float("nan"), "unknown"), (float("inf"), "unknown"), (2**63, "unknown"),
                                        (b"VIN_PRIVATE", "unknown"), ("VIN_PRIVATE", "unknown")])
def test_scalar_types_no_coercion_or_raw_payload(value, kind):
  o = observer()
  with o.policy.group("selection") as group:
    consumer = group.consumer("controls")
    with consumer.read("selectionTune") as read:
      read.returned(value)
  row = policy_records(o)[1]
  assert row["scalar_kind"] == kind and row["raw_presence"] == row["default_used"] == "unknown"
  assert "VIN_PRIVATE" not in repr(policy_records(o))
  assert row["valid"] is (kind != "unknown")


def test_group_loss_never_complete_or_reset_shared_budget():
  o = observer(capacity=2)
  with o.policy.group("controlsInit") as group:
    consumer = group.consumer("controls")
    for _ in range(5):
      with consumer.read("selectionEnforce") as read:
        read.returned(True)
  assert o.dropped and o._loop_records == 2 and o.loss
  assert not any(r["kind"] == "groupComplete" and r["complete"] for r in policy_records(o))
  o._pending.clear()
  with o.policy.group("selection") as group:
    consumer.selected(None, "pid")
  assert not policy_records(o)[-1]["complete"]


def test_non_torque_selection_has_no_borrowed_id():
  o = observer()
  with o.policy.group("selection") as group:
    consumer = group.consumer("controls")
    consumer.selected(SimpleNamespace(), "angle")
  selection = policy_records(o)[1]
  assert selection["kind"] == "controllerSelected" and not selection["selected_known"]
  assert selection["selected_id"] == 0 and selection["selection_reason"] == "nonTorqueNotInstrumented"


def test_noop_external_calls_have_no_policy_sink_and_no_new_params_reads():
  p = ROOT / "openpilot/sunnypilot/livedelay/helpers.py"
  tree = ast.parse(p.read_text())
  function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_lat_delay")
  scope = {"Params": object}
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(p), "exec"), scope)
  class Params:
    def __init__(self):
      self.calls = []
    def get_bool(self, key):
      self.calls.append(key)
      return True
    def get(self, *a, **k):
      pytest.fail("conditional omitted read executed")
  params = Params()
  assert scope["get_lat_delay"](params, .31) == .31 and params.calls == ["LagdToggle"]


def test_policy_schema_and_overlay_registered():
  schema = (ROOT / "openpilot/cereal/custom.capnp").read_text()
  assert "policyObservation @31 :PolicyObservation" in schema
  sync = (ROOT / ".elantra/sync.py").read_text()
  assert "openpilot/selfdrive/controls/lib/controller_policy_observation.py" in sync


def test_minimal_source_refs_learner_not_full_sm():
  o = observer()
  source = sm(obs.CONTROLS_INPUTS)
  with o.policy.group("learnerWrite") as group:
    c = group.consumer("stockTorque")
    c.torque_written("learner", SimpleNamespace(latAccelFactor=2.5, latAccelOffset=.02, friction=.1),
                     source_ns=source.logMonoTime["lateralTorqueParameters"], source_frame=1)
  row = policy_records(o)[1]
  assert row["source_ns"] == 900000 and row["factor"] == 2.5 and row["offset"] == .02
  assert not records(o)[1]["refs"]


def source_fixture(o, *, enforce=True, tune=0., override=True, lagd=False, steering="torque"):
  """Actual observation/getter/selection/MRO source; unrelated native math is absent."""
  calls = []
  values = {"EnforceTorqueControl": enforce, "TorqueControlTune": tune, "TorqueParamsOverrideEnabled": override,
            "LateralJerkTorqueController": True, "NeuralNetworkLateralControl": False, "LagdToggle": lagd,
            "LagdValueCache": .2, "BlinkerPauseLateralControl": False, "IsMetric": True,
            "BlinkerMinLateralControlSpeed": 20, "BlinkerLateralReengageDelay": 0,
            "TorqueParamsOverrideLatAccelFactor": 2.7, "TorqueParamsOverrideFriction": .12}
  class Params:
    def get_bool(self, key):
      calls.append(("get_bool", key))
      return values[key]
    def get(self, key, **kwargs):
      calls.append(("get", key, kwargs))
      return b"CP_PRIVATE" if key == "CarParams" else b"SP_PRIVATE" if key == "CarParamsSP" else values[key]
  torque = SimpleNamespace(as_builder=lambda: SimpleNamespace(latAccelFactor=2.5, latAccelOffset=.02, friction=.1))
  cp = SimpleNamespace(lateralTuning=SimpleNamespace(which=lambda: steering, torque=torque),
                       steerControlType={"angle": 1, "curvature": 2}.get(steering, 0))
  car = SimpleNamespace(CarParams=SimpleNamespace(SteerControlType=SimpleNamespace(angle=1, curvature=2)))
  namespace = {"Params": Params, "create_observer": lambda _: o, "car": car, "structs": car, "DT_CTRL": .01,
               "custom": SimpleNamespace(CarParamsSP=object), "PARAMS_UPDATE_PERIOD": 3,
               "cloudlog": SimpleNamespace(info=lambda _: None), "time": SimpleNamespace(monotonic=lambda: 100.),
               "messaging": SimpleNamespace(log_from_bytes=lambda raw, _: cp if raw == b"CP_PRIVATE" else object())}
  class StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node):
      node.annotation = None
      return node
    def visit_FunctionDef(self, node):
      node.returns = None
      return self.generic_visit(node)
  def load(path, name, methods, transform=None, alias=None):
    cls = copy.deepcopy(next(n for n in ast.parse((ROOT / path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == name))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if transform:
      transform(cls)
    tree = StripAnnotations().visit(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), namespace)
    if alias:
      namespace[alias] = namespace[name]
  base = "openpilot/sunnypilot/selfdrive/controls/lib/"
  load("openpilot/sunnypilot/modeld_v2/modeld_base.py", "ModelStateBase", ["__init__"])
  load(base + "blinker_pause_lateral.py", "BlinkerPauseLateral", ["__init__", "get_params"])
  class Base:
    def __init__(self, lac, *args):
      self.lac_torque = lac
    def update_limits(self):
      pass
    def update_model_v2(self, model):
      pass
    def update_lateral_lag(self, delay):
      pass
  namespace["LatControlTorqueExtBase"] = Base
  load(base + "latcontrol_torque_jerk_aware.py", "LatControlTorqueJerkAware", ["__init__"])
  def trim_nn(cls):
    ctor = cls.body[0]
    cut = next(i for i, n in enumerate(ctor.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model_path" for t in n.targets))
    ctor.body = ctor.body[:cut]
  load(base + "nnlc/nnlc.py", "NeuralNetworkLateralControl", ["__init__"], trim_nn)
  load(base + "latcontrol_torque_ext_override.py", "LatControlTorqueExtOverride", ["__init__", "update_override_torque_params"])
  load(base + "latcontrol_torque_ext.py", "LatControlTorqueExt", ["__init__"])
  namespace["LatControl"] = Base
  def trim_torque(cls):
    ctor = next(n for n in cls.body if n.name == "__init__")
    ctor.body = [n for n in ctor.body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in ("_policy_capture", "torque_params")
                                                   for t in n.targets)) or isinstance(n, ast.If)]
    cls.body.append(ast.parse("def update_limits(self):\n  pass").body[0])
  load("openpilot/selfdrive/controls/lib/latcontrol_torque.py", "LatControlTorque", ["__init__", "update_torque_parameters"], trim_torque, "StockTorque")
  load(base + "latcontrol_torque_v0.py", "LatControlTorque", ["__init__", "update_torque_parameters"], trim_torque, "LatControlTorqueV0")
  namespace["LatControlTorque"] = namespace["StockTorque"]
  for kind in ("Angle", "Curvature", "PID"):
    namespace["LatControl" + kind] = lambda *args: SimpleNamespace()
  helper = ast.parse((ROOT / "openpilot/sunnypilot/livedelay/helpers.py").read_text())
  function = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == "get_lat_delay")
  exec(compile(ast.fix_missing_locations(StripAnnotations().visit(ast.Module(body=[function], type_ignores=[]))), "helper", "exec"), namespace)
  load("openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py", "ControlsExt", ["__init__", "initialize_lateral_control", "get_params_sp"])
  def trim_controls(cls):
    ctor = cls.body[0]
    cut = next(i for i, n in enumerate(ctor.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "CI" for t in n.targets))
    prefix = ctor.body[:cut]
    selection = next(n for n in ctor.body if isinstance(n, ast.If) and "self.CP.steerControlType" in ast.unparse(n.test))
    final = next(n for n in ctor.body if isinstance(n, ast.Assign) and "ControlsExt.initialize_lateral_control" in ast.unparse(n))
    ctor.body = prefix + ast.parse("self.CI = object()").body + [selection, final]
  load("openpilot/selfdrive/controls/controlsd.py", "Controls", ["__init__"], trim_controls)
  controls = namespace["Controls"]()
  source = sm(obs.CONTROLS_INPUTS)
  class Source:
    def __init__(self):
      self.__dict__.update(source.__dict__)
    def __getitem__(self, key):
      return SimpleNamespace(lateralDelay=.31) if key == "lateralDelay" else SimpleNamespace(
        useParams=True, latAccelFactorFiltered=2.6, latAccelOffsetFiltered=.03, frictionCoefficientFiltered=.11)
    def all_checks(self, services):
      calls.append(("all_checks", tuple(services)))
      return True
  controls.sm = Source()
  update = next(n for n in ast.parse((ROOT / "openpilot/selfdrive/controls/controlsd.py").read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "Controls")
  update = next(n for n in update.body if isinstance(n, ast.FunctionDef) and n.name == "state_control")
  learner = next(n for n in update.body if isinstance(n, ast.If) and "self.CP.lateralTuning.which()" in ast.unparse(n.test))
  function = ast.parse("def learner(self):\n  pass").body[0]
  function.body = [learner]
  exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "actual-learner-fragment", "exec"), namespace)
  return controls, calls, namespace["learner"]


@pytest.mark.parametrize("enforce,tune,expected", [(True, 0., 31), (False, 0., 31), (True, 1., 23)])
def test_actual_constructor_mro_selection_counts_and_one_spare(enforce, tune, expected):
  o = observer()
  controls, calls, _ = source_fixture(o, enforce=enforce, tune=tune)
  assert len(records(o)) == expected and not o.dropped and o.loop == 0
  rows = policy_records(o)
  reads = [r for r in rows if r["kind"] == "readReturned"]
  assert len(reads) == len([c for c in calls if c[1] not in ("CarParams", "CarParamsSP")])
  assert [r["site"] for r in reads[3:7]] == ["jerkEnable", "nnEnable", "overrideEnforce", "overrideInitial"]
  selected = next(r for r in rows if r["kind"] == "controllerSelected")
  assert selected["selected_id"] == controls.LaC._policy_capture.id
  assert controls.LaC.extension._policy_capture.parent == controls.LaC._policy_capture.id
  assert len({r["consumer_id"] for r in rows if r["kind"] == "consumerCreated"}) == (6 if expected == 31 else 4)
  assert all(r["complete"] for r in rows if r["kind"] == "groupComplete")
  assert o._loop_bytes <= 32768
  if expected == 31:
    o.capture("computeBegin")
    assert len(records(o)) == 32
    o.capture("computeEnd")
    assert len(records(o)) == 32 and o.dropped == 1


@pytest.mark.parametrize("override,lagd,expected", [(True, False, 28), (False, False, 25), (True, True, 27)])
def test_actual_first_normal_tick28_preserves_order_and_does_not_recreate_consumers(override, lagd, expected):
  o = observer()
  c, calls, learner = source_fixture(o, override=override, lagd=lagd)
  prefix = len(records(o))
  o.begin_loop()
  o.capture("updateBegin")
  o.capture("updateEnd", sm=c.sm)
  learner(c)  # existing all_checks/useParams branch, one actual scalar-write group
  o.capture("computeBegin", sm=c.sm)
  assert c.LaC.extension.update_override_torque_params(c.LaC.torque_params) is override
  o.capture("computeEnd")
  for service in ("controlsState", "carControl"):
    event = SimpleNamespace(logMonoTime=900001, valid=True)
    tap = obs.PublisherTap(SimpleNamespace(sock={}, send=lambda *args: None), o)
    tap.send(service, event)
  o.capture("paramsRefreshBegin")
  c.get_params_sp(c.sm)
  o.capture("paramsRefreshEnd")
  tap.send("carControlSP", SimpleNamespace(logMonoTime=900002, valid=True))
  suffix = records(o)[prefix:]
  assert len(suffix) == expected and o._loop_records == expected and o._loop_bytes == 7584 + (expected-13)*1024
  assert not o.dropped and not any(r.get("policy", {}).get("kind") == "consumerCreated" for r in suffix)
  writes = [r["policy"] for r in suffix if r.get("policy", {}).get("kind") == "torqueParametersWritten"]
  assert [r["write_source"] for r in writes] == (["learner", "override"] if override else ["learner"])
  assert writes[0]["factor"] == 2.6
  if override:
    assert writes[1]["factor"] == 2.7
  assert calls.count(("all_checks", ("lateralTorqueParameters",))) == 1
  before = len(records(o))
  assert not c.LaC.extension.update_override_torque_params(c.LaC.torque_params)
  assert len(records(o)) == before  # non-due update has no synthetic/no-op group


@pytest.mark.parametrize("steering", ["angle", "curvature", "pid"])
def test_actual_non_torque_branch_has_unavailable_lineage_and_disabled_same_return(steering):
  o = observer()
  c, enabled_calls, _ = source_fixture(o, enforce=False, steering=steering)
  disabled, disabled_calls, _ = source_fixture(None, enforce=False, steering=steering)
  assert enabled_calls == disabled_calls
  assert len(records(o)) == 15 and not hasattr(c.LaC, "_policy_capture") and not hasattr(disabled.LaC, "_policy_capture")
  row = next(r for r in policy_records(o) if r["kind"] == "controllerSelected")
  assert not row["selected_known"] and row["selected_class"] == steering and row["selected_id"] == 0


def test_actual_policy_schema_roundtrip_ineligibility_and_one_kib(isolated_schema):
  o = observer()
  source_fixture(o)
  for record in records(o):
    if "policy" not in record:
      continue
    raw = obs.encode_record(record, isolated_schema, publisher_ns=9_000_000)
    assert len(raw) <= 1024 and b"CP_PRIVATE" not in raw and b"SP_PRIVATE" not in raw
    with isolated_schema.Event.from_bytes(raw) as event:
      value = event.controlsObservation.observation
      assert not value.policyComplete and not value.sourceAuthenticated and not value.predictiveEligible
      assert value.policyObservation.present and value.policyObservation.rawPresence == "unknown"
      assert not value.references


def test_actual_float32_effect_is_separate_from_typed_return(isolated_schema):
  event = isolated_schema.Event.new_message()
  params = event.init("carParams").lateralTuning.init("torque")
  params.latAccelFactor, params.latAccelOffset, params.friction = 2.7, .02, .1
  o = observer()
  with o.policy.group("stockConstructor") as group:
    c = group.consumer("stockExtension")
    with c.read("overrideFactor") as read:
      read.returned(2.7)
    c.torque_written("override", params)
  returned = next(r for r in policy_records(o) if r["kind"] == "readReturned")
  written = next(r for r in policy_records(o) if r["kind"] == "torqueParametersWritten")
  assert returned["float_value"] == 2.7 and written["factor"] == params.latAccelFactor
  assert returned["float_value"] != written["factor"]  # stored Float32 is not the getter's Float64


@pytest.mark.parametrize("enforce,tune", [(True, 0.), (False, 0.), (True, 1.)])
def test_actual_disabled_and_enabled_getter_order_and_baseline_values(enforce, tune):
  enabled, calls_on, _ = source_fixture(observer(), enforce=enforce, tune=tune)
  disabled, calls_off, _ = source_fixture(None, enforce=enforce, tune=tune)
  assert calls_on == calls_off
  assert vars(enabled.LaC.torque_params) == vars(disabled.LaC.torque_params)
  assert enabled.lat_delay == disabled.lat_delay


def test_actual_conversion_raises_after_successful_get_without_changing_exception_or_coercing_return():
  o = observer()
  c, _, _ = source_fixture(o)
  get = c.LaC.extension.params.get
  failure = ValueError("VIN_PRIVATE_CAST")
  class Returned:
    casts = 0
    def __float__(self):
      self.casts += 1
      raise failure
  value = Returned()
  c.LaC.extension.params.get = lambda key, **kwargs: value if key == "TorqueParamsOverrideLatAccelFactor" else get(key, **kwargs)
  o.begin_loop()  # real run resets after constructor; do not spend its sole spare record
  with pytest.raises(ValueError) as caught:
    c.LaC.extension.update_override_torque_params(c.LaC.torque_params)
  assert caught.value is failure and value.casts == 1
  row = next(r for r in reversed(policy_records(o)) if r["site"] == "overrideFactor")
  assert row["kind"] == "readReturned" and row["scalar_kind"] == "unknown" and not row["valid"]
  assert not policy_records(o)[-1]["complete"] and "VIN_PRIVATE" not in repr(policy_records(o))


def test_registry_int_returned_bool_and_exact_learner_event_receive_ref():
  o = observer()
  c, calls, learner = source_fixture(o)
  row = next(r for r in policy_records(o) if r["site"] == "blinkerInitial")
  assert row["registry_type"] == "int" and row["scalar_kind"] == "bool" and not row["default_requested"]
  o.begin_loop()
  c.sm.logMonoTime["lateralTorqueParameters"], c.sm.recv_frame["lateralTorqueParameters"] = 2**53+1, 2**53+3
  learner(c)
  row = next(r for r in reversed(policy_records(o)) if r["kind"] == "torqueParametersWritten")
  assert row["source_ns"] == 2**53+1 and row["source_frame"] == 2**53+3 and not row["valid"]  # future ref, not reconstructed
  assert calls.count(("all_checks", ("lateralTorqueParameters",))) == 1


@pytest.mark.parametrize("failure", ["tie", "regression", "clock_error"])
def test_read_clock_failure_cannot_promote_later_end_or_escape_getter(failure):
  o = observer()
  with o.policy.group("controlsInit") as group:
    c = group.consumer("controls")
    last = o._last_clock
    ticks = iter([last if failure == "tie" else last-1, last+100, last+200])
    if failure == "clock_error":
      def clock():
        raise RuntimeError("PRIVATE_CLOCK")
      o.clock = clock
    else:
      o.clock = lambda: next(ticks)
    with c.read("selectionEnforce") as read:
      value = True
      read.returned(value)
    assert value is True
  assert not any(r["kind"] == "groupComplete" and r["complete"] for r in policy_records(o))
  assert not any(r["kind"] == "readReturned" and r["valid"] for r in policy_records(o))


def test_duplicate_read_end_and_group_bound_cannot_coalesce_or_claim_complete():
  o = observer()
  with o.policy.group("controlsInit") as group:
    c = group.consumer("controls")
    read = c.read("selectionEnforce")
    with read:
      read.returned(True)
      read.returned(False)
    read.__exit__(None, None, None)
  row = next(r for r in policy_records(o) if r["kind"] == "readReturned")
  assert row["bool_value"] is True and o.errors == 2 and not policy_records(o)[-1]["complete"]
  o.begin_loop()
  before = len(records(o))
  with o.policy.group("selection"):
    for _ in range(17):
      with c.read("selectionEnforce") as read:
        read.returned(True)
  assert len(records(o)) - before == 16 and o.loss


def test_unknown_policy_fields_refuse_before_copy_and_consumer_ids_never_recycle():
  o = observer()
  class Private:
    def __repr__(self):
      pytest.fail("private value formatted")
  o.capture("policy", _policy={"private": Private()})
  assert not records(o) and o.errors and o.loss
  for _ in range(65):
    o.begin_loop()
    o._pending.clear()
    with o.policy.group("controlsInit") as group:
      c = group.consumer("controls")
  assert c.id == 0 and o.policy.next_consumer == 64 and o.loss


def test_disabled_native_operations_ast_matches_original_snapshots():
  import hashlib
  import json
  # Derived once from frozen original bytes; no generated archive or Git dependency.
  expected = {
    "openpilot/selfdrive/controls/controlsd.py": "39a28c285254be9f01f25af9049a0d870ca832502a5c7e5b52dc842d8d18d5a3",
    "openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py": "991e2601dc6f75450ae3acd61df8b2264c1dce3f7eebe0afcc4073c4ef6617e1",
    "openpilot/selfdrive/controls/lib/latcontrol_torque.py": "7749ffd6ff4ff754135b413145dbc95fda6d11a8d4f4aaf8f3b1bb89db46752a",
    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py": "ce10871798882542eb2439ebb31cfcfb599168a417acb202f21a67e1707e869a",
    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py": "5500af41872164eec32c5a7daca7238f6d9a5996049c738dde8c5784fbb7bb7c",
    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_override.py": "78c09041b318e5507c97551caa3befb28d3d82d9633458d176f73e2a9418c3f5",
    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_jerk_aware.py": "4a79228a867395c22de3b69b0a5dea7ae214e9d246b33addbf22cf073ee54735",
    "openpilot/sunnypilot/selfdrive/controls/lib/nnlc/nnlc.py": "a8518d375356726c32fcdaf8ff40a7e9dcbbb91058ff4405bb98073c3f4d610f",
    "openpilot/sunnypilot/selfdrive/controls/lib/blinker_pause_lateral.py": "95b6d0eeeba4ae5eef71d88c7250557c9b0714d8b8a072f8979aa57f86c9e759",
    "openpilot/sunnypilot/modeld_v2/modeld_base.py": "a721d64014a7d4392c0600479cff49e0475f7f982f4fa2a4a32e37f65c99cbbc",
    "openpilot/sunnypilot/livedelay/helpers.py": "af03a3836445ca1df47e283cfc415194113b6c4ee0051e5f5c36f42fa08b3ac5",
  }
  class Disabled(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
      if "policy_capture" in [a.arg for a in node.args.kwonlyargs]:
        index = [a.arg for a in node.args.kwonlyargs].index("policy_capture")
        del node.args.kwonlyargs[index]
        del node.args.kw_defaults[index]
      node = self.generic_visit(node)
      # The default-None helper returns inside its original branch; subsequent
      # enabled-only statements are unreachable in the disabled projection.
      for index, statement in enumerate(node.body):
        if isinstance(statement, (ast.Return, ast.Raise)):
          node.body = node.body[:index+1]
          break
      return node
    def visit_If(self, node):
      condition = ast.unparse(node.test)
      if condition in ("policy_capture is None", "self._policy_capture is None", "self.observer is None"):
        return [self.visit(n) for n in node.body]
      if condition == "policy_capture is not None":
        return [self.visit(n) for n in node.orelse]
      return self.generic_visit(node)
    def visit_Assign(self, node):
      if any((isinstance(t, ast.Name) and t.id == "policy_capture") or (isinstance(t, ast.Attribute) and t.attr == "_policy_capture") for t in node.targets):
        return None
      return self.generic_visit(node)
  def normalized(node):
    if isinstance(node, ast.AST):
      # Empty type_params was added in3.12; it does not change these native operations.
      return [type(node).__name__, [[key, normalized(value)] for key, value in ast.iter_fields(node)
                                   if not (key == "type_params" and value == [])]]
    if isinstance(node, list):
      return [normalized(value) for value in node]
    if isinstance(node, bytes):
      return {"bytes_hex": node.hex()}
    if node is Ellipsis:
      return {"literal": "Ellipsis"}
    return node
  for path, original_ast_sha256 in expected.items():
    current = Disabled().visit(ast.parse((ROOT / path).read_text()))
    raw = json.dumps(normalized(current), separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == original_ast_sha256, path
