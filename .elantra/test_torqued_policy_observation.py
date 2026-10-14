"""Bounded synthetic/source contracts; no native estimator or policy acceptance."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest_plugins = ("test_controller_observation",)  # one current schema, fresh process


def helper():
  path = ROOT / "openpilot/sunnypilot/selfdrive/locationd/torqued_policy_observation.py"
  spec = importlib.util.spec_from_file_location("torqued_policy_observation", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def instance():
  ticks = iter(range(1000000, 10000000, 100))
  capture = helper().TorquedPolicyCapture(clock=lambda: next(ticks), epoch=b"e"*16)
  return capture, capture.instance(False)


def estimator():
  return SimpleNamespace(frame=1, use_live_torque_params=False, custom_torque_params=True, factor_sanity=.3, friction_sanity=.2)


def event(schema, valid=True):
  value = schema.Event.new_message(logMonoTime=900000, valid=valid)
  target = value.init("lateralTorqueParameters")
  target.useParams, target.latAccelFactorFiltered = True, 2.5
  return value


def test_disabled_creates_no_sink_epoch_clock_or_params(monkeypatch):
  module = helper()
  monkeypatch.delenv("ELANTRA_TORQUED_POLICY_OBSERVER", raising=False)
  monkeypatch.setattr(module, "TorquedPolicyCapture", lambda: pytest.fail("disabled allocated sink"))
  assert module.create_torqued_policy_capture() is None


def test_false_known_unknown_raised_no_private_payload_and_one_end_record():
  capture, sink = instance()
  with sink.read("constructorLive", -1) as read:
    assert not capture.pending
    read.returned(False)
  with sink.read("constructorCustom", -1) as read:
    read.returned(b"VIN_PRIVATE")
  failure = RuntimeError("SECRET_PRIVATE")
  with pytest.raises(RuntimeError) as caught:
    with sink.read("initializeRelaxed", 0):
      raise failure
  assert caught.value is failure
  assert len(capture.pending) == 3
  assert capture.pending[0]["returned_known"] and capture.pending[0]["returned_bool"] is False
  assert not capture.pending[1]["returned_known"] and capture.pending[2]["outcome"] == "raised"
  assert "PRIVATE" not in repr(capture.pending) and capture.loss


def test_present_incomplete_overflow_is_distinct_from_absence(isolated_schema):
  capture, sink = instance()
  for _ in range(9):
    with sink.read("constructorLive", 0) as read:
      read.returned(False)
  msg = event(isolated_schema)
  token = sink.stage(msg, estimator())
  header = msg.lateralTorqueParameters.policyObservation
  assert token is not None and header.present and not header.readCaptureComplete
  assert header.dropped > 0 and len(header.reads) == 8 and len(capture.pending) == 8
  assert msg.valid and msg.lateralTorqueParameters.latAccelFactorFiltered == 2.5
  assert not event(isolated_schema).lateralTorqueParameters.policyObservation.present


def test_retire_only_staged_prefix_not_delivery_ack_and_send_exception_preserved(isolated_schema):
  capture, sink = instance()
  with sink.read("constructorLive", 0) as read:
    read.returned(True)
  token = sink.stage(event(isolated_schema), estimator())
  with sink.read("refreshCustom", 0) as read:
    read.returned(False)
  sink.retire(token)
  assert [row["site"] for row in capture.pending] == ["refreshCustom"]
  token = sink.stage(event(isolated_schema), estimator())
  sink.send_failed(token)
  assert len(capture.pending) == 1 and capture.loss


def test_invalid_clocks_do_not_promote_current_or_later_header(isolated_schema):
  capture, sink = instance()
  ticks = iter([100, 100, 200, 300])
  capture.clock = lambda: next(ticks)
  with sink.read("constructorLive", -1) as read:
    read.returned(True)
  msg = event(isolated_schema)
  sink.stage(msg, estimator())
  assert msg.lateralTorqueParameters.policyObservation.present
  assert not msg.lateralTorqueParameters.policyObservation.readCaptureComplete and capture.loss


def test_schema_wire_bounds_and_constant_untrusted_flags(isolated_schema):
  capture, sink = instance()
  for index in range(8):
    with sink.read("constructorLive", index) as read:
      read.returned(index % 2 == 0)
  msg = event(isolated_schema, valid=False)
  sink.stage(msg, estimator())
  header = msg.lateralTorqueParameters.policyObservation
  assert header.present and header.readCaptureComplete and not msg.valid
  assert not header.policyComplete and not header.sourceAuthenticated and not header.predictiveEligible
  assert header.rawPresence == header.defaultUsed == "unknown"
  envelope = isolated_schema.LateralTorqueParameters.TorquedPolicyObservationV1.new_message(**header.to_dict())
  assert len(envelope.to_bytes()) <= helper().MAX_WIRE_BYTES == 1024
  with isolated_schema.Event.from_bytes(msg.to_bytes()) as restored:
    assert len(restored.lateralTorqueParameters.policyObservation.reads) == 8 and not restored.valid


def ext_class(params):
  """Execute only the actual extension class against synthetic Params, no native imports."""
  import numpy as np
  path = ROOT / "openpilot/sunnypilot/selfdrive/locationd/torqued_ext.py"
  tree = ast.parse(path.read_text())
  selected = next(node for node in tree.body if isinstance(node, ast.ClassDef))
  namespace = {"np": np, "Params": lambda: params, "car": SimpleNamespace(CarParams=object), "DT_MDL": .05,
               "PARAMS_UPDATE_PERIOD": 3, "ALLOWED_CARS": ["hyundai"], "RELAXED_MIN_BUCKET_POINTS": np.array([1, 200])}
  exec(compile(ast.Module(body=[selected], type_ignores=[]), str(path), "exec"), namespace)
  return namespace["TorqueEstimatorExt"]


class RecordedParams:
  def __init__(self, values, *, failure=None):
    self.values, self.failure, self.calls = values, failure, []

  def get_bool(self, key):
    self.calls.append(("get_bool", key))
    value = self.values.get(key, False)
    if isinstance(value, BaseException):
      raise value
    return value

  def get(self, key, **kwargs):
    self.calls.append(("get", key, kwargs))
    return self.values[key]


def cp():
  return SimpleNamespace(brand="hyundai", lateralTuning=SimpleNamespace(which=lambda: "torque"))


@pytest.mark.parametrize("enforce,relaxed,custom", [(False, False, False), (True, False, False), (True, True, True)])
def test_actual_extension_fragments_preserve_one_read_order_values_and_cadence(enforce, relaxed, custom):
  values = {"EnforceTorqueControl": enforce, "LiveTorqueParamsToggle": False, "CustomTorqueParams": custom,
            "LiveTorqueParamsRelaxedToggle": relaxed, "TorqueParamsOverrideEnabled": True,
            "TorqueParamsOverrideLatAccelFactor": 2.0, "TorqueParamsOverrideFriction": .1}
  baseline, observed = RecordedParams(values), RecordedParams(values)
  plain = ext_class(baseline)(cp())
  capture, sink = instance()
  tapped = ext_class(observed)(cp(), policy_capture=sink)
  for value in (plain, tapped):
    value.initialize_custom_params()
    for _ in range(61):
      value.update_use_params()
  assert baseline.calls == observed.calls
  assert sum(row == ("get_bool", "LiveTorqueParamsToggle") for row in observed.calls) == 3
  for field in ("frame", "use_params", "use_live_torque_params", "custom_torque_params", "factor_sanity", "friction_sanity",
                "offline_latAccelFactor", "offline_friction"):
    assert getattr(plain, field) == getattr(tapped, field)
  assert plain._torqued_policy_capture is None
  assert all(row["outcome"] == "returned" for row in capture.pending)
  expected = ["constructorLive", "constructorCustom"]
  if enforce:
    expected += ["initializeRelaxed", "initializeCustom"]
  expected += ["refreshLive", "refreshCustom", "refreshLive", "refreshCustom"]
  assert [row["site"] for row in capture.pending] == expected


def test_getter_return_is_not_later_float_conversion_exception():
  params = RecordedParams({"EnforceTorqueControl": True, "CustomTorqueParams": True,
                           "TorqueParamsOverrideLatAccelFactor": "not a float"})
  capture, sink = instance()
  value = ext_class(params)(cp(), policy_capture=sink)
  with pytest.raises(ValueError):
    value.initialize_custom_params()
  assert capture.pending[-1]["site"] == "initializeCustom"
  assert capture.pending[-1]["outcome"] == "returned" and capture.pending[-1]["returned_bool"] is True
  assert sum(call[0:2] == ("get_bool", "CustomTorqueParams") for call in params.calls) == 2


def test_actual_constructor_getter_exception_object_and_original_order():
  failure = RuntimeError("PRIVATE")
  values = {"LiveTorqueParamsToggle": failure}
  captures = []
  for enabled in (False, True):
    params = RecordedParams(values)
    capture, sink = instance()
    with pytest.raises(RuntimeError) as caught:
      ext_class(params)(cp(), policy_capture=sink if enabled else None)
    assert caught.value is failure
    assert params.calls == [("get_bool", "EnforceTorqueControl"), ("get_bool", "LiveTorqueParamsToggle")]
    captures.append(capture)
  assert not captures[0].pending and captures[1].pending[0]["outcome"] == "raised"


def publish_fragment():
  path = ROOT / "openpilot/selfdrive/locationd/torqued.py"
  main = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
  loop = next(node for node in main.body if isinstance(node, ast.While))
  publish = next(node for node in loop.body if isinstance(node, ast.If) and ast.unparse(node.test) == "sm.frame % 5 == 0")
  return compile(ast.Module(body=[publish], type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("fail", [False, True])
def test_actual_main_publication_single_send_exact_exception_and_prefix_retirement(isolated_schema, fail):
  capture, sink = instance()
  with sink.read("constructorLive", -1) as read:
    read.returned(False)
  calls, failure = [], RuntimeError("PRIVATE_SEND")
  value = estimator()
  value._torqued_policy_capture = sink

  def get_msg(**kwargs):
    calls.append(("get_msg", kwargs))
    return event(isolated_schema, valid=False)

  def send(service, msg):
    calls.append(("send", service, msg.valid, msg.lateralTorqueParameters.policyObservation.present))
    if fail:
      raise failure

  value.get_msg = get_msg
  namespace = {"estimator": value, "pm": SimpleNamespace(send=send), "sm": SimpleNamespace(frame=5, all_checks=lambda: False), "DEBUG": True}
  if fail:
    with pytest.raises(RuntimeError) as caught:
      exec(publish_fragment(), namespace)
    assert caught.value is failure and len(capture.pending) == 1 and capture.loss
  else:
    exec(publish_fragment(), namespace)
    assert not capture.pending and not capture.loss
  assert calls == [("get_msg", {"valid": False, "with_points": True}), ("send", "lateralTorqueParameters", False, True)]


def test_partial_encoding_failure_is_absent_then_sticky_loss_without_native_mutation(isolated_schema):
  capture, sink = instance()
  with sink.read("constructorLive", -1) as read:
    read.returned(False)

  class BrokenHeader:
    present = False

    def __setattr__(self, name, value):
      if name != "present":
        raise ValueError("bounded encoding failure")
      object.__setattr__(self, name, value)

  broken = BrokenHeader()
  msg = SimpleNamespace(logMonoTime=100, valid=True, lateralTorqueParameters=SimpleNamespace(useParams=True, policyObservation=broken))
  token = sink.stage(msg, estimator())
  assert token is None and not broken.present and msg.valid and capture.loss
  sink.retire(token)
  assert len(capture.pending) == 1  # failed attachment never retires an unreported prefix
  next_msg = event(isolated_schema)
  assert sink.stage(next_msg, estimator()) is not None
  header = next_msg.lateralTorqueParameters.policyObservation
  assert header.present and header.epochLoss and not header.readCaptureComplete and len(header.reads) == 1


def test_default_off_main_publication_has_no_capture_and_cache_get_msg_is_untapped(isolated_schema):
  calls = []
  value = estimator()
  value._torqued_policy_capture = None
  msg = event(isolated_schema, valid=False)
  value.get_msg = lambda **kwargs: calls.append(("get_msg", kwargs)) or msg
  pm = SimpleNamespace(send=lambda service, sent: calls.append(("send", service, sent is msg)))
  exec(publish_fragment(), {"estimator": value, "pm": pm, "sm": SimpleNamespace(frame=5, all_checks=lambda: False), "DEBUG": False})
  assert calls == [("get_msg", {"valid": False, "with_points": False}), ("send", "lateralTorqueParameters", True)]
  assert not msg.valid and not msg.lateralTorqueParameters.policyObservation.present
  tree = ast.parse((ROOT / "openpilot/selfdrive/locationd/torqued.py").read_text())
  method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "TorqueEstimator"
                for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "get_msg")
  assert "policy" not in ast.unparse(method)
  main = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "main")
  loop = next(item for item in main.body if isinstance(item, ast.While))
  cache = next(item for item in loop.body if isinstance(item, ast.If) and ast.unparse(item.test) == "sm.frame % 240 == 0")
  assert "policy" not in ast.unparse(cache)


def test_same_key_initialization_read_remains_distinct_from_cached_refresh(isolated_schema):
  capture, sink = instance()
  for site, result in (("constructorCustom", False), ("initializeCustom", True), ("refreshCustom", False)):
    with sink.read(site, 0) as read:
      read.returned(result)
  msg = event(isolated_schema)
  sink.stage(msg, estimator())
  header = msg.lateralTorqueParameters.policyObservation
  assert [row.readId for row in header.reads] == [1, 2, 3]
  assert header.initializeCustomReadId == 2 and header.cachedCustomReadId == 3
  assert [row.returnedBool for row in header.reads] == [False, True, False]


def test_unknown_nonfinite_state_is_not_known_default_or_native_event_failure(isolated_schema):
  capture, sink = instance()
  value = estimator()
  value.factor_sanity, value.use_live_torque_params = float("nan"), b"PRIVATE"
  msg = event(isolated_schema)
  assert sink.stage(msg, value) is not None and msg.valid
  header = msg.lateralTorqueParameters.policyObservation
  assert header.present and not header.readCaptureComplete and header.epochLoss
  assert not header.factorSanityKnown and not header.cachedLiveKnown
  assert b"PRIVATE" not in msg.to_bytes()


def test_instance_token_bounds_and_cross_prefix_rejection(isolated_schema):
  capture, sink = instance()
  for _ in range(63):
    assert capture.instance(False) is not None
  assert capture.instance(False) is None and capture.instances == 64
  with sink.read("constructorLive", -1) as read:
    read.returned(True)
  token = sink.stage(event(isolated_schema), estimator())
  sink.retire(token[:-1] + (token[-1],))  # equal content is not staged-token identity
  assert len(capture.pending) == 1 and capture.loss
  sink.retire(token)
  assert not capture.pending


def normalized(value):
  if isinstance(value, ast.AST):
    return [type(value).__name__, [[key, normalized(item)] for key, item in ast.iter_fields(value) if not (key == "type_params" and not item)]]
  if isinstance(value, list):
    return [normalized(item) for item in value]
  if isinstance(value, bytes):
    return {"bytes_hex": value.hex()}
  if value is Ellipsis:
    return {"literal": "Ellipsis"}
  return value


def ast_hash(tree):
  return hashlib.sha256(json.dumps(normalized(tree), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class DisabledProjection(ast.NodeTransformer):
  """Only remove the new optional supplement; retain original native operations."""
  def visit_ImportFrom(self, node):
    return None if node.module and node.module.endswith("torqued_policy_observation") else node

  def visit_arguments(self, node):
    pairs = [(arg, default) for arg, default in zip(node.kwonlyargs, node.kw_defaults, strict=True) if arg.arg != "policy_capture"]
    node.kwonlyargs, node.kw_defaults = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    return node

  def visit_Assign(self, node):
    if any(ast.unparse(target) in ("policy_capture", "self._torqued_policy_capture") for target in node.targets):
      return None
    return self.generic_visit(node)

  def visit_If(self, node):
    condition = ast.unparse(node.test)
    if condition in ("policy_capture is None", "estimator._torqued_policy_capture is None"):
      return [self.visit(item) for item in node.body]
    if condition == "policy_capture is not None":
      return [self.visit(item) for item in node.orelse]
    return self.generic_visit(node)


def test_disabled_original_operations_ast_and_cache_are_portable():
  # Fixed literals derived once from the immutable pre-change bytes; no generated artifact required.
  expected = {
    "openpilot/selfdrive/locationd/torqued.py": "cc7c5930ed2de2f7097d9245e95a6dcbe5339bb7fe8b36c89d4a1ebb028946d7",
    "openpilot/sunnypilot/selfdrive/locationd/torqued_ext.py": "cf86989423ea4e325fea155d7fc776d680180faf60dfbde623fc245850912c58",
  }
  for relative, digest in expected.items():
    projected = DisabledProjection().visit(ast.parse((ROOT / relative).read_text()))
    assert ast_hash(projected) == digest, relative


def test_overlay_registration_covers_both_modified_producers_and_new_helper():
  tree = ast.parse((ROOT / ".elantra/sync.py").read_text())
  values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("OVERLAY_ADDED", "OVERLAY_MODIFIED")}
  assert "openpilot/sunnypilot/selfdrive/locationd/torqued_policy_observation.py" in values["OVERLAY_ADDED"]
  assert {"openpilot/selfdrive/locationd/torqued.py", "openpilot/sunnypilot/selfdrive/locationd/torqued_ext.py"} <= set(values["OVERLAY_MODIFIED"])
