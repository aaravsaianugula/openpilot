"""Synthetic/source-only model-reference contracts; no native models or assets."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest_plugins = ("test_controller_observation",)  # exactly one current schema per process


def helper():
  path = ROOT / "openpilot/selfdrive/modeld/model_asset_observation.py"
  spec = importlib.util.spec_from_file_location("model_asset_observation", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def capture():
  ticks = iter(range(1000000, 10000000, 100))
  return helper().ModelAssetCapture("tinygrad", clock=lambda: next(ticks), epoch=b"e"*16)


def model(c, chestnut=False, *, origin="bundleLookup", generation=7):
  value = SimpleNamespace(chestnut=chestnut, generation=generation, LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.3,
                          DEV="QCOM")
  value._model_asset_reference = c.reserve(chestnut)
  value._model_asset_reference.begin()
  value._model_asset_reference.origin(origin)
  value._model_asset_reference.loaded(value, 1344, 760, "split")
  return value


def events(schema, *, valid=False):
  model_event = schema.Event.new_message(logMonoTime=900000, valid=valid)
  native = model_event.init("modelV2")
  native.frameId, native.frameIdExtra, native.timestampEof = 71, 70, 800000
  driving = schema.Event.new_message(logMonoTime=900100, valid=valid)
  driving.init("drivingModelData").frameId = 71
  sp = schema.Event.new_message(logMonoTime=900200, valid=valid)
  sp.init("modelDataV2SP").laneTurnDirection = "turnLeft"
  return sp, model_event, driving


def envelope_root(schema, header):
  import capnp
  # custom.capnp is the exact same import already loaded by this isolated log schema.
  path = Path(schema.__file__).with_name("custom.capnp")
  custom = capnp.load(str(path), imports=[str(path.parent), str(path.parent / "include")])
  return custom.ModelDataV2SP.ModelAssetReferenceV1.new_message(**header.to_dict())


def test_disabled_allocates_no_epoch_clock_capture_or_asset_operation(monkeypatch):
  module = helper()
  monkeypatch.delenv("ELANTRA_MODEL_ASSET_OBSERVER", raising=False)
  monkeypatch.setattr(module, "ModelAssetCapture", lambda *args: pytest.fail("OFF allocated capture"))
  assert module.create_model_asset_capture("tinygrad") is None


def test_instance_activation_actual_fallback_same_instance_and_failed_attempt_lineage(isolated_schema):
  c = capture()
  big, small = model(c, True), model(c)
  c.activate(big, small)
  c.attempt(big)
  c.fallback(small)
  assert c.pending[-1]["from_id"] == 1 and c.pending[-1]["to_id"] == 2 and c.pending[-1]["failed_attempt"] == 1
  c.attempt(small)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, small)
  header = sp.modelDataV2SP.modelAssetReference
  assert token is not None and header.present and header.activeInstanceId == header.attemptedInstanceId == 2
  assert header.attemptSequence == 2 and len(header.references) == 4
  assert not sp.valid and not native.valid and sp.modelDataV2SP.laneTurnDirection == "turnLeft"
  assert not header.fullAssetKnown and not header.sourceAuthenticated and not header.policyComplete and not header.predictiveEligible
  assert header.streamLayout == header.consumedDigest == header.fullAssetDigest == "unknown"
  c.retire(token)
  c.attempt(small)
  c.fallback(small)  # native assignment can legitimately be to the same object
  assert c.loss and c.reference_attempts == 5 and not c.pending  # no silent lifetime-budget reset


def test_unavailable_big_snapshot_never_promotes_late_completion(isolated_schema):
  c = capture()
  late = c.reserve(True)
  c.unavailable(late)
  small = model(c)
  c.activate(small, small)
  old = tuple(c.pending)
  late.begin()
  late.origin("envOverride")
  late.loaded(SimpleNamespace(chestnut=True), 1344, 760, "stock")
  assert tuple(c.pending) == old and c.active_id == 2  # worker-private completion, no main journal mutation
  c.attempt(small)
  sp, native, driving = events(isolated_schema)
  c.stage(sp, native, driving, small)
  assert sp.modelDataV2SP.modelAssetReference.activeInstanceId == 2


def test_fixed_bounds_clocks_privacy_and_maximum_wire(isolated_schema):
  c = capture()
  big, small = model(c, True), model(c, origin="envOverride")
  assert c.reserve(False) is None and c.instances == 2 and c.loss
  c.activate(big, small)
  c.attempt(big)
  c.fallback(small)
  c.attempt(small)
  sp, native, driving = events(isolated_schema)
  c.stage(sp, native, driving, small)
  header = sp.modelDataV2SP.modelAssetReference
  envelope = envelope_root(isolated_schema, header)
  raw = envelope.to_bytes()
  assert len(raw) <= 1024 and len(header.references) == 4 and header.epochLoss
  assert b"PRIVATE_PATH" not in raw and not header.referenceCaptureComplete


def test_send_failure_preserves_staged_prefix_and_exact_token_retirement(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  original = tuple(c.pending)
  c.send_failed(token)
  assert tuple(c.pending) == original and c.loss
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  c.retire(token[:-1] + (token[-1],))
  assert tuple(c.pending) == original
  c.retire(token)
  assert not c.pending


def test_invalid_load_clock_and_untrusted_unknown_metadata_never_promote(isolated_schema):
  c = capture()
  value = SimpleNamespace(chestnut=False, generation=b"PRIVATE_PATH", LAT_SMOOTH_SECONDS=float("nan"))
  value._model_asset_reference = c.reserve(False)
  ticks = iter([100, 100])
  value._model_asset_reference.clock = lambda: next(ticks)
  value._model_asset_reference.begin()
  value._model_asset_reference.origin("PRIVATE_PATH")
  value._model_asset_reference.loaded(value, 1344, 760, "PRIVATE_PATH")
  c.activate(value)
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  c.stage(sp, native, driving, value)
  header = sp.modelDataV2SP.modelAssetReference
  assert header.present and header.epochLoss and not header.referenceCaptureComplete
  assert b"PRIVATE_PATH" not in sp.to_bytes()


def parsed(relative):
  return ast.parse((ROOT / relative).read_text())


def source_function(relative, name):
  return next(node for node in parsed(relative).body if isinstance(node, ast.FunctionDef) and node.name == name)


V2 = "openpilot/sunnypilot/modeld_v2/modeld.py"
STOCK = "openpilot/selfdrive/modeld/modeld.py"


def test_actual_manifest_override_selection_retains_declared_bundle_without_extra_probes(isolated_schema):
  calls = []
  bundle = SimpleNamespace(generation=7, overrides=[], models=[SimpleNamespace(artifact=SimpleNamespace(fileName="declared.pkl"))])
  environment = {"COMBINED_MODEL_PKL": "PRIVATE_OVERRIDE"}

  class Environment:
    def get(self, key):
      calls.append(("env", key))
      return environment.get(key)

  namespace = {"ModelStateBase": SimpleNamespace(__init__=lambda self: calls.append(("base",))),
               "os": SimpleNamespace(environ=Environment(), path=SimpleNamespace(exists=lambda path: calls.append(("exists", path)) or False)),
               "get_active_bundle": lambda **kwargs: calls.append(("bundle", kwargs)) or bundle,
               "_pkl_exists": lambda path: calls.append(("pkl_exists", path)) or path == "PRIVATE_OVERRIDE"}
  find = source_function(V2, "_find_driving_pkl")
  exec(compile(ast.Module(body=[find], type_ignores=[]), V2, "exec"), namespace)
  constructor = next(node for cls in parsed(V2).body if isinstance(cls, ast.ClassDef) and cls.name == "ModelState"
                     for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
  cls = ast.ClassDef(name="SyntheticModel", bases=[], keywords=[], body=[constructor], decorator_list=[])
  exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), V2, "exec"), namespace)
  native = namespace["SyntheticModel"]

  def combined(self, path, width, height, declared):
    calls.append(("combined", path, width, height, declared is bundle))
    self._combined_model_type, self.DEV = "split", "QCOM"

  native._init_combined = combined
  c = capture()
  reserved = c.reserve(False)
  value = native(1344, 760, asset_reference=reserved)
  assert value.generation == 7 and reserved.completed["origin"] == "envOverride"
  assert calls == [("base",), ("env", "COMBINED_MODEL_PKL"), ("exists", "PRIVATE_OVERRIDE"),
                   ("bundle", {"chestnut": False}), ("env", "COMBINED_MODEL_PKL"), ("pkl_exists", "PRIVATE_OVERRIDE"),
                   ("combined", "PRIVATE_OVERRIDE", 1344, 760, True)]
  assert "PRIVATE" not in repr(dict(reserved.completed))
  c.activate(value)
  c.attempt(value)
  sp, native_event, driving = events(isolated_schema)
  c.stage(sp, native_event, driving, value)
  header = sp.modelDataV2SP.modelAssetReference
  assert header.references[0].declaredGeneration == 7 and header.references[0].origin == "envOverride"
  assert not header.fullAssetKnown and header.fullAssetDigest == "unknown"


def startup_fragment(relative):
  main = source_function(relative, "main")
  begin = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign) and ast.unparse(node) == "model = None")
  end = next(index for index in range(begin, len(main.body)) if isinstance(main.body[index], ast.Assert)) + 1
  if isinstance(main.body[end], ast.If) and ast.unparse(main.body[end].test) == "asset_observation is not None":
    end += 1
  # Retain the real nested nonlocal handoff in a function scope, not module scope.
  wrapper = ast.FunctionDef(name="startup_fixture", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                            body=[*main.body[begin:end], ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))],
                            decorator_list=[])
  return compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), relative, "exec")


@pytest.mark.parametrize("relative", [V2, STOCK])
@pytest.mark.parametrize("late", [False, True])
def test_actual_source_thread_handoff_late_big_never_replaces_main_snapshot(relative, late):
  calls, threads = [], []
  c = capture()

  class NativeModel:
    def __init__(self, *args, asset_reference=None, **kwargs):
      self.chestnut = args[2] if args else kwargs["chestnut"]
      self.generation, self.DEV = 7, "QCOM"
      self._model_asset_reference = asset_reference
      calls.append(("construct", self.chestnut))
      if asset_reference is not None:
        asset_reference.begin()
        asset_reference.origin("stockBuiltin")
        asset_reference.loaded(self, 1344, 760, "stock")

    def warmup(self):
      calls.append(("warmup", self.chestnut))

  class Thread:
    def __init__(self, *, target, daemon):
      assert daemon is True
      self.target = target
      threads.append(self)

    def start(self):
      calls.append(("start",))
      if not late:
        self.target()

    def join(self, timeout):
      calls.append(("join", timeout))

  namespace = {"CHESTNUT": True, "create_model_asset_capture": lambda runner: c, "ModelState": NativeModel,
               "threading": SimpleNamespace(Thread=Thread), "BIG_MODEL_TIMEOUT": 60,
               "vipc_client_main": SimpleNamespace(width=1344, height=760),
               "params": SimpleNamespace(put_bool=lambda *args: calls.append(("put_bool", *args)),
                                         remove=lambda *args: calls.append(("remove", *args))),
               "cloudlog": SimpleNamespace(exception=lambda *args: calls.append(("load_failure",)))}
  exec(startup_fragment(relative), namespace)
  namespace.update(namespace["startup_fixture"]())
  active, pending = namespace["model"], tuple(c.pending)
  assert active.chestnut is not late and c.active_id == (2 if late else 1)
  assert calls.count(("join", 60)) == 1 and sum(call[0] == "construct" for call in calls) == (1 if late else 2)
  if late:
    main_clock = c.last_clock
    threads[0].target()
    assert namespace["model"] is active and tuple(c.pending) == pending and c.active_id == 2 and c.last_clock == main_clock
  else:
    assert calls.index(("warmup", True)) < calls.index(("construct", False))


def run_try_fragment(relative):
  main = source_function(relative, "main")
  loop = next(node for node in main.body if isinstance(node, ast.While) and any(isinstance(item, ast.Try) for item in node.body))
  branch = next(item for item in loop.body if isinstance(item, ast.Try) and "model.run(" in ast.unparse(item))
  return compile(ast.Module(body=[branch], type_ignores=[]), relative, "exec")


@pytest.mark.parametrize("relative", [V2, STOCK])
@pytest.mark.parametrize("fallback_allowed", [False, True])
def test_actual_run_failure_original_exception_or_actual_assignment_not_failed_output_relabel(relative, fallback_allowed):
  c = capture()
  old, small = model(c, True), model(c)
  c.activate(old, small)
  c.attempt(old)
  error = RuntimeError("PRIVATE_NATIVE_ERROR")
  calls = []

  def fail(*args):
    raise error

  old.run = fail
  namespace = {"asset_observation": c, "model": old, "small_model": small, "chestnut_state": None,
               "bufs": {}, "transforms": {}, "inputs": {}, "run_count": 3,
               "params": SimpleNamespace(get_bool=lambda key: calls.append(("get_bool", key)) or fallback_allowed,
                                         put_bool=lambda *args: calls.append(("put_bool", *args))),
               "cloudlog": SimpleNamespace(exception=lambda *args: None)}
  if not fallback_allowed:
    with pytest.raises(RuntimeError) as caught:
      exec(run_try_fragment(relative), namespace)
    assert caught.value is error and namespace["model"] is old and c.active_id == 1
  else:
    exec(run_try_fragment(relative), namespace)
    assert namespace["model"] is small and namespace["model_output"] is None and c.active_id == 2
    assert c.pending[-1]["from_id"] == 1 and c.pending[-1]["failed_attempt"] == 1
  assert calls[0] == ("get_bool", "ChestnutActive")


def send_fragment(relative):
  main = source_function(relative, "main")
  branches = [node for node in ast.walk(main) if isinstance(node, ast.If) and ast.unparse(node.test) == "asset_observation is None"]
  publish = next(node for node in branches if "staged_asset =" in ast.unparse(node))
  return compile(ast.Module(body=[publish], type_ignores=[]), relative, "exec")


@pytest.mark.parametrize("relative", [V2, STOCK])
@pytest.mark.parametrize("failure_at", [None, "modelV2", "modelDataV2SP"])
def test_actual_send_order_native_fields_exception_and_retirement(relative, failure_at, isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  error, calls = RuntimeError("PRIVATE_SEND"), []

  def send(service, message):
    calls.append(service)
    assert message.valid is False
    if service == failure_at:
      raise error

  namespace = {"asset_observation": c, "model": value, "pm": SimpleNamespace(send=send),
               "mdv2sp_send": sp, "modelv2_send": native, "drivingdata_send": driving, "posenet_send": native}
  if failure_at is not None:
    with pytest.raises(RuntimeError) as caught:
      exec(send_fragment(relative), namespace)
    assert caught.value is error and c.pending and c.loss
  else:
    exec(send_fragment(relative), namespace)
    assert not c.pending and not c.loss
  expected = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"]
  assert calls == expected[:expected.index(failure_at)+1] if failure_at else calls == expected
  assert sp.modelDataV2SP.laneTurnDirection == "turnLeft" and not sp.valid


def test_output_clock_tie_is_present_incomplete_and_partial_encoding_does_not_retire(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  driving.logMonoTime = native.logMonoTime
  assert c.stage(sp, native, driving, value) is not None
  assert sp.modelDataV2SP.modelAssetReference.present and not sp.modelDataV2SP.modelAssetReference.referenceCaptureComplete
  c.send_failed(c.staged)
  pending = tuple(c.pending)
  broken = SimpleNamespace(modelDataV2SP=SimpleNamespace(modelAssetReference=object()))
  assert c.stage(broken, native, driving, value) is None and tuple(c.pending) == pending


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
  def visit_ImportFrom(self, node):
    return None if node.module and node.module.endswith("model_asset_observation") else node

  def visit_arguments(self, node):
    pairs = [(arg, default) for arg, default in zip(node.kwonlyargs, node.kw_defaults, strict=True) if arg.arg != "asset_reference"]
    node.kwonlyargs, node.kw_defaults = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    return node

  def visit_Assign(self, node):
    if any(ast.unparse(target) in ("asset_observation", "self._model_asset_reference", "big_asset_reference") for target in node.targets):
      return None
    return self.generic_visit(node)

  def visit_If(self, node):
    condition = ast.unparse(node.test)
    if condition in ("asset_observation is None", "asset_reference is None"):
      return [self.visit(item) for item in node.body]
    if condition in ("asset_observation is not None", "asset_reference is not None"):
      return [self.visit(item) for item in node.orelse]
    return self.generic_visit(node)


def test_portable_disabled_native_operations_ast_matches_original():
  expected = {V2: "5daad707c3f2affd0e7579a7258e3d227410c7a4ae0cb33e4967b75259056b5c",
              STOCK: "5ab9c5a921a7932472f0cee0a6a0f32112f09bfde619fce67a61cf4bc1346cba"}
  for relative, original in expected.items():
    assert ast_hash(DisabledProjection().visit(parsed(relative))) == original


def test_overlay_registration_covers_exact_changed_sources():
  tree = parsed(".elantra/sync.py")
  entries = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("OVERLAY_ADDED", "OVERLAY_MODIFIED")}
  assert "openpilot/selfdrive/modeld/model_asset_observation.py" in entries["OVERLAY_ADDED"]
  assert {V2, STOCK, "openpilot/cereal/custom.capnp"} <= set(entries["OVERLAY_MODIFIED"])


@pytest.mark.parametrize("relative", [V2, STOCK])
def test_attempt_capture_precedes_existing_native_execution_timer(relative):
  main = source_function(relative, "main")
  loop = next(node for node in main.body if isinstance(node, ast.While) and any(isinstance(item, ast.Try) for item in node.body))
  attempt = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.If) and "asset_observation.attempt(model)" in ast.unparse(node))
  timer = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign) and ast.unparse(node) == "mt1 = time.perf_counter()")
  assert attempt < timer


def test_driving_frame_identity_is_observed_independently_without_borrowing(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  driving.drivingModelData.frameId, driving.drivingModelData.frameIdExtra = 16, 17
  c.stage(sp, native, driving, value)
  header = sp.modelDataV2SP.modelAssetReference
  assert (header.modelFrameId, header.modelFrameIdExtra, header.modelTimestampEof) == (71, 70, 800000)
  assert (header.drivingFrameId, header.drivingFrameIdExtra) == (16, 17)
  assert driving.drivingModelData.frameId == 16 and native.modelV2.frameId == 71 and not driving.valid


def test_first_actual_fallback_to_same_instance_is_not_fabricated_switch(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  c.fallback(value)
  row = c.pending[-1]
  assert row["kind"] == "runtimeFallback" and row["from_id"] == row["to_id"] == 1 and row["failed_attempt"] == 1
  assert row["valid"] and not c.loss
  c.attempt(value)
  sp, native, driving = events(isolated_schema)
  c.stage(sp, native, driving, value)
  assert sp.modelDataV2SP.modelAssetReference.referenceCaptureComplete
