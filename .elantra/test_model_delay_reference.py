"""Source fragments/public synthetic schema only; no native model/Params imports."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_model_asset_observation import DisabledProjection, ast_hash, envelope_root, events, helper, send_fragment

ROOT = Path(__file__).resolve().parents[1]
STOCK = "openpilot/selfdrive/modeld/modeld.py"
V2 = "openpilot/sunnypilot/modeld_v2/modeld.py"
pytest_plugins = ("test_controller_observation",)


def delay_module():
  path = ROOT / "openpilot/selfdrive/modeld/model_delay_observation.py"
  spec = importlib.util.spec_from_file_location("model_delay_fixture", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def capture(runner="tinygrad"):
  ticks = iter(range(1000000, 10000000, 100))
  return helper().ModelAssetCapture(runner, clock=lambda: next(ticks), epoch=b"e"*16, delay_enabled=True)


def source_class(path, name, namespace):
  tree = ast.parse((ROOT / path).read_text())
  node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
  exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
  return namespace[name]


def base_fragment(params):
  return source_class("openpilot/sunnypilot/modeld_v2/modeld_base.py", "ModelStateBase", {"Params": lambda: params})


def delay_function():
  tree = ast.parse((ROOT / "openpilot/sunnypilot/livedelay/helpers.py").read_text())
  node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
  ns = {"Params": object}
  exec(compile(ast.Module(body=[node], type_ignores=[]), "get_lat_delay_fragment", "exec"), ns)
  return ns["get_lat_delay"]


class ParamsFixture:
  def __init__(self, value=.11, toggle=False):
    self.value, self.toggle, self.calls = value, toggle, []

  def get(self, key, *, return_default):
    self.calls.append(("get", key, return_default))
    if isinstance(self.value, BaseException):
      raise self.value
    return self.value

  def get_bool(self, key):
    self.calls.append(("get_bool", key))
    return self.toggle


def constructor_fragment(relative, params):
  tree = ast.parse((ROOT / relative).read_text())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelState")
  fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
  body = []
  for node in fn.body:
    body.append(node)
    if isinstance(node, ast.If) and ast.unparse(node.test) == "asset_reference is not None":
      break
  fn.body = body
  ns = {"ModelStateBase": base_fragment(params)}
  exec(compile(ast.Module(body=[fn], type_ignores=[]), relative, "exec"), ns)
  return ns["__init__"]


def model(c, params=None, chestnut=False, relative=V2):
  params = params or ParamsFixture()
  ref = c.reserve(chestnut)
  value = SimpleNamespace(chestnut=chestnut, generation=7, DEV="QCOM", LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.3)
  constructor_fragment(relative, params)(value, 1344, 760, chestnut, asset_reference=ref)
  if relative == V2:
    ref.origin("envOverride")
  ref.loaded(value, 1344, 760, "split")
  return value


def refresh(c, value, params, published=.5):
  consumer = c.consumer_for(value)
  try:
    value.lat_delay = delay_function()(params, published, policy_capture=consumer)
  except BaseException:
    if consumer is not None:
      consumer.cache_failed()
    raise
  if consumer is not None:
    consumer.delay_written(value.lat_delay, consumer.delay_origin)


def prepare(c, value, *, effective_lat=.32, seen=True, event_ns=810000, published_lat=.6):
  sm = {"lateralDelay": SimpleNamespace(lateralDelay=.5)}
  sm = type("SM", (dict,), {})(sm)
  sm.frame = 120
  for name, v in (("logMonoTime", event_ns), ("recv_frame", 119), ("seen", seen), ("updated", True),
                  ("valid", False), ("alive", True)):
    setattr(sm, name, {"lateralDelay": v})
  main = SimpleNamespace(frame_id=71, timestamp_sof=790000, timestamp_eof=800000)
  extra = SimpleNamespace(frame_id=70, timestamp_sof=789999, timestamp_eof=799999)
  c.delay.prepare(c, value, sm, main, extra,
                  {"action_t": np.array([.395, .675], dtype=np.float32),
                   "lateral_control_params": np.array([12.5, effective_lat], dtype=np.float32)},
                  v_ego=12.5, effective_lat=effective_lat, published_lat=published_lat,
                  lat_action=.395, long_action=.675, frame_delay=.05, action_delay=.025,
                  lat_smoothing=.1, long_smoothing=.4, smoothing_origin="declaredProcessing")


@pytest.mark.parametrize("relative", [STOCK, V2])
def test_private_already_reserved_identity_precedes_getter_and_original_load_begin(relative):
  c = capture()
  params = ParamsFixture()
  value = model(c, params, relative=relative)
  consumer = c.consumer_for(value)
  assert consumer.identity == value._model_asset_reference.id == 1 and consumer.epoch is c.epoch
  assert params.calls == [("get", "LagdValueCache", True)]
  row = consumer.pending[0]
  assert row["beginNs"] < row["endNs"] < consumer.constructor["write_ns"] < value._model_asset_reference.begin_ns
  assert row["instanceId"] == 1 and row["site"] == "delayInitial"


def test_read_returned_none_and_later_cast_failure_preserve_original_exception_and_no_write():
  c = capture()
  value = model(c)
  consumer = c.consumer_for(value)
  before = dict(consumer.cached)
  params = ParamsFixture(value=None)
  with pytest.raises(TypeError):
    refresh(c, value, params)
  assert params.calls == [("get_bool", "LagdToggle"), ("get", "LagdValueCache", True)]
  row = consumer.pending[-1]
  assert row["outcome"] == "returned" and row["valueKind"] == "knownNone" and row["castOutcome"] == "raised"
  assert consumer.cached == before
  failure = KeyboardInterrupt("native getter failure")
  with pytest.raises(KeyboardInterrupt) as actual:
    model(c, ParamsFixture(value=failure))
  assert actual.value is failure


def test_stock_cache_published_offset_and_initial_long_owner_stay_distinct(isolated_schema):
  c = capture("stock")
  big, small = model(c, chestnut=True), model(c)
  c.activate(big, small)
  c.delay.initial_long(c, big, .3, .6, .3, "parsed")
  c.attempt(big)
  prepare(c, big)
  c.delay.output_none(True)
  c.fallback(small)
  refresh(c, small, ParamsFixture(value=.22))
  c.attempt(small)
  prepare(c, small, effective_lat=.5)
  c.delay.postprocess(.395, .675, "unknown")
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, small)
  delay = sp.modelDataV2SP.modelAssetReference.modelDelayReference
  assert token is not None and delay.cachedValue == .22 and delay.publishedDelay == .5
  assert delay.effectiveLatOffset == .5 and delay.publishedLatOffset == .6
  assert delay.initialLongOwnerId == 1 and delay.preparedInstanceId == 2 and delay.longDelay == .6
  assert delay.currentLongSmoothing == .4 and delay.lastDiscardedInstanceId == 1
  assert delay.outputNoneObservedCount == delay.discardedBeforeFinalSpReturnCount == 1
  assert not delay.publishedValid and not delay.graphConsumptionKnown and not delay.policyComplete


def test_failed_send_and_replacement_never_retire_read_prefix_or_relabel_old_attempt(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  consumer = c.consumer_for(value)
  original_ids = [r["readId"] for r in consumer.pending]
  c.delay.final_send_started(token)
  c.send_failed(token)
  assert [r["readId"] for r in consumer.pending] == original_ids
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  delay = sp.modelDataV2SP.modelAssetReference.modelDelayReference
  assert delay.failedFinalSpSendCount == delay.discardedBeforeFinalSpReturnCount == 1
  assert delay.lastDiscardedAttemptId == 1 and delay.preparedAttemptId == 2
  c.delay.final_send_started(token)
  c.retire(token)
  assert not consumer.pending and c.delay.current is None


def test_default_off_and_asset_only_use_no_delay_consumer_clock_or_base_keyword(monkeypatch):
  module = helper()
  monkeypatch.delenv("ELANTRA_MODEL_ASSET_OBSERVER", raising=False)
  monkeypatch.delenv("ELANTRA_MODEL_DELAY_REFERENCE_OBSERVER", raising=False)
  monkeypatch.setattr(module, "ModelAssetCapture", lambda *a, **kw: pytest.fail("OFF capture"))
  assert module.create_model_asset_capture("stock") is None
  c = helper().ModelAssetCapture("stock", clock=lambda: pytest.fail("asset-only private clock"), epoch=b"e"*16)
  ref = c.reserve(False)
  assert ref.delay_consumer() is None and c.delay is None


def test_source_hooks_outside_timer_and_refresh_cadence_and_overlay():
  for path in (STOCK, V2):
    text = (ROOT / path).read_text()
    assert text.index("asset_observation.delay.prepare(") < text.index("mt1 = time.perf_counter()", text.index("asset_observation.attempt(model)"))
    assert text.index("asset_observation.delay.output_none(") > text.index("model_execution_time = mt2 - mt1")
    tree = ast.parse(text)
    if path == V2:
      refresh_block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "sm.frame % 60 == 0")
      assert "policy_capture=delay_consumer" in ast.unparse(refresh_block)
  assert '"openpilot/selfdrive/modeld/model_delay_observation.py"' in (ROOT / ".elantra/sync.py").read_text()


def test_portable_disabled_original_ast_unchanged():
  expected = {V2: "5daad707c3f2affd0e7579a7258e3d227410c7a4ae0cb33e4967b75259056b5c",
              STOCK: "5ab9c5a921a7932472f0cee0a6a0f32112f09bfde619fce67a61cf4bc1346cba"}
  for path, original in expected.items():
    assert ast_hash(DisabledProjection().visit(ast.parse((ROOT / path).read_text()))) == original


def test_full_actual_new_header_fixed100_fields_four_assets_four_reads_budget(isolated_schema):
  c = capture()
  big, small = model(c, chestnut=True), model(c)
  c.activate(big, small)
  c.delay.initial_long(c, big, .3, .6, .3, "parsed")
  c.attempt(big)
  prepare(c, big)
  c.delay.output_none(True)
  c.fallback(small)
  refresh(c, small, ParamsFixture(value=.22))
  refresh(c, small, ParamsFixture(toggle=True))
  c.attempt(small)
  prepare(c, small)
  sp, native, driving = events(isolated_schema)
  assert c.stage(sp, native, driving, small) is not None
  header = sp.modelDataV2SP.modelAssetReference
  assert len(header.references) == len(header.modelDelayReference.reads) == 4
  assert len(header.modelDelayReference.to_dict()) == 100
  assert len(envelope_root(isolated_schema, header).to_bytes()) == 1024
  assert not any((header.sourceAuthenticated, header.policyComplete, header.predictiveEligible))


def test_exhausted_read_id_cannot_alias_old_staged_prefix_or_new_write(isolated_schema):
  c = capture()
  value = model(c)
  consumer = c.consumer_for(value)
  consumer.sequence = 2**64-2
  refresh(c, value, ParamsFixture(value=.22, toggle=True))
  c.activate(value)
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  before = [r["readId"] for r in consumer.pending]
  refresh(c, value, ParamsFixture(value=.33, toggle=True))
  assert [r["readId"] for r in consumer.pending] == before
  assert not consumer.cached["read_known"] and consumer.cached["read_id"] == 0
  assert consumer.loss
  c.delay.final_send_started(token)
  c.retire(token)
  assert not consumer.pending  # exhausted new read was never silently part of old prefix
  assert not consumer.cached["read_known"]


def test_unknown_first_constructor_snapshot_is_immutable_even_on_duplicate_write():
  c = capture()
  ref = c.reserve(False)
  consumer = ref.delay_consumer()
  with consumer.read("delayInitial") as read:
    read.returned(object())
  consumer.delay_written(object(), "constructor")
  first = consumer.constructor
  assert first["kind"] == "unknown"
  consumer.delay_written(.22, "constructor")
  assert consumer.constructor is first and consumer.cached is first
  assert consumer.loss


@pytest.mark.parametrize("seen,event_ns", [(False, 810000), (True, 0), (True, -1)])
def test_unseen_or_missing_publication_is_not_known_event_zero(seen, event_ns):
  c = capture()
  value = model(c)
  c.activate(value)
  c.delay.initial_long(c, value, .3, .6, .3, "parsed")
  c.attempt(value)
  prepare(c, value, seen=seen, event_ns=event_ns)
  data = c.delay.current["payload"]
  assert data["publishedDelayKnown"] and data["publishedDelay"] == .5
  assert not data["publishedReferenceKnown"] and not data["preparationComplete"]
  assert data["publishedSeen"] is seen


def test_consumer_identity_must_match_current_attempt_before_read_or_cache_binding(isolated_schema):
  c = capture()
  first, second = model(c), model(c)
  c.activate(first, second)
  c.delay.initial_long(c, first, .3, .6, .3, "parsed")
  c.attempt(first)
  prepare(c, second)
  assert not c.delay.current["payload"]["preparationComplete"]
  sp, native, driving = events(isolated_schema)
  assert c.stage(sp, native, driving, first) is None


def test_tinygrad_published_scalar_is_known_but_counterfactual_offset_is_unknown():
  c = capture()
  value = model(c)
  c.activate(value)
  c.delay.initial_long(c, value, .3, .6, .3, "parsed")
  c.attempt(value)
  prepare(c, value, published_lat=None)
  data = c.delay.current["payload"]
  assert data["publishedDelayKnown"] and data["publishedDelay"] == .5
  assert not data["publishedLatOffsetKnown"] and data["publishedLatOffset"] == 0.
  assert data["effectiveLatOffsetKnown"] and data["effectiveLatOffset"] == .32
  assert data["preparationComplete"]


@pytest.mark.parametrize("relative", [STOCK, V2])
@pytest.mark.parametrize("failure_at", ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP", None])
def test_extracted_single_send_phase_distinguishes_earlier_failure_and_preserves_exception(relative, failure_at, isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.delay.initial_long(c, value, .3, .6, .3, "parsed")
  c.attempt(value)
  prepare(c, value)
  c.delay.postprocess(.395, .675, "unknown")
  consumer = c.consumer_for(value)
  before = [r["readId"] for r in consumer.pending]
  sp, native, driving = events(isolated_schema)
  error, calls = RuntimeError("native send failure"), []

  def send(service, message):
    calls.append(service)
    assert message.valid is False
    if service == failure_at:
      raise error

  namespace = {"asset_observation": c, "model": value, "pm": SimpleNamespace(send=send),
               "mdv2sp_send": sp, "modelv2_send": native, "drivingdata_send": driving, "posenet_send": native}
  if failure_at is None:
    exec(send_fragment(relative), namespace)
    assert not consumer.pending and c.delay.current is None
  else:
    with pytest.raises(RuntimeError) as actual:
      exec(send_fragment(relative), namespace)
    assert actual.value is error and [r["readId"] for r in consumer.pending] == before and c.pending
    assert c.delay.counts["failedFinalSpSendCount"] == (1 if failure_at == "modelDataV2SP" else 0)
  expected = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"]
  assert calls == (expected if failure_at is None else expected[:expected.index(failure_at)+1])


def test_old_or_invalid_token_cannot_enable_final_failure_counter_or_retirement(isolated_schema):
  c = capture()
  value = model(c)
  c.activate(value)
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  token = c.stage(sp, native, driving, value)
  copied_token = token[:-1] + token[-1:]
  assert copied_token == token and copied_token is not token
  c.delay.final_send_started(copied_token)
  c.send_failed(token)
  assert c.delay.counts["failedFinalSpSendCount"] == 0
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  new_token = c.stage(sp, native, driving, value)
  before = list(c.consumer_for(value).pending)
  c.delay.final_send_started(token)
  c.retire(new_token)
  assert c.consumer_for(value).pending == before and c.pending
  c.delay.final_send_started(new_token)
  c.retire(new_token)
  assert not c.consumer_for(value).pending


def test_clock_tie_duplicate_return_overflow_and_private_values_never_promote_or_leak(isolated_schema):
  c = capture()
  value = model(c)
  consumer = c.consumer_for(value)
  ticks = iter([2000000, 2000000, 2000100])
  consumer.clock = lambda: next(ticks)
  with consumer.read("delayCache") as read:
    read.returned(.22)
    read.returned(object())
  consumer.delay_written(.22, "cachedDelay")
  assert consumer.pending[-1]["endStatus"] == "tie" and not consumer.pending[-1]["valid"] and consumer.loss
  consumer.clock = c.clock
  refresh(c, value, ParamsFixture(toggle=True))
  before = [r["readId"] for r in consumer.pending]
  refresh(c, value, ParamsFixture(value=.33))
  assert len(consumer.pending) == 4 and [r["readId"] for r in consumer.pending][:len(before)] == before
  assert not consumer.cached["read_known"]
  c.activate(value)
  c.attempt(value)
  prepare(c, value)
  sp, native, driving = events(isolated_schema)
  assert c.stage(sp, native, driving, value) is not None
  delay = sp.modelDataV2SP.modelAssetReference.modelDelayReference
  assert delay.epochLoss and not delay.readCaptureComplete and not delay.preparationComplete
  assert b"PRIVATE_VALUE" not in envelope_root(isolated_schema, sp.modelDataV2SP.modelAssetReference).to_bytes()


def test_private_constructor_and_unknown_site_are_not_coerced_or_relabelled():
  class CoercionFault(BaseException):
    pass

  class Private:
    def __float__(self):
      pytest.fail("observer converted private value")

    def __str__(self):
      raise CoercionFault("observer stringified private value")

    def __repr__(self):
      pytest.fail("observer rendered private value")

  c = capture()
  raw = Private()
  value = model(c, ParamsFixture(value=raw))
  consumer = c.consumer_for(value)
  assert value.lat_delay is raw and consumer.constructor["kind"] == "unknown"
  with consumer.read(Private()) as read:
    read.returned(raw)
  assert consumer.pending[-1]["site"] == "unknown" and not consumer.pending[-1]["valid"]
  assert all(type(v) in (int, float, bool, str) for r in consumer.pending for v in r.values())


def test_bool_getter_none_is_unknown_not_known_float_none_or_default_false():
  c = capture()
  value = model(c)
  consumer = c.consumer_for(value)
  with consumer.read("lagdToggle") as read:
    read.returned(None)
  row = consumer.pending[-1]
  assert row["outcome"] == "returned" and row["valueKind"] == "unknown" and not row["valid"]
  assert consumer.loss


def test_fixed_vector_elements_unknown_types_and_absence_do_not_change_inputs():
  module = delay_module()
  array = np.array([12.5, .22], dtype=np.float32)
  assert module._vector({"action_t": array}, "action_t") == ("prepared", 12.5, np.float32(.22).item())
  assert module._vector({}, "action_t") == ("absentOriginalBranch", 0., 0.)
  assert module._vector({"action_t": array.astype(np.float64)}, "action_t")[0] == "unavailable"
  assert module._vector(None, "action_t")[0] == "unavailable"
  assert array.tolist() == [12.5, np.float32(.22).item()]


@pytest.mark.parametrize("relative", [STOCK, V2])
def test_prepared_attempt_is_not_run_enter_when_actual_precall_cadence_fails(relative):
  c = capture()
  value = model(c)
  value.constants = SimpleNamespace(MODEL_FREQ=20)
  value.run = lambda *a: pytest.fail("run entered after prior cadence failure")
  c.activate(value)
  c.attempt(value)
  prepare(c, value)
  tree = ast.parse((ROOT / relative).read_text())
  run_try = next(n for n in ast.walk(tree) if isinstance(n, ast.Try) and any("model_output = model.run(" in ast.unparse(x) for x in n.body))
  namespace = {"asset_observation": c, "model": value, "chestnut_state": object(), "run_count": 1,
               "ModelConstants": SimpleNamespace(MODEL_RUN_FREQ=20),
               "SERVICE_LIST": {"chestnutState": SimpleNamespace(frequency=0)}, "params": ParamsFixture(toggle=False)}
  with pytest.raises(ZeroDivisionError):
    exec(compile(ast.Module(body=[run_try], type_ignores=[]), relative, "exec"), namespace)
  assert c.delay.current["payload"]["preparedStatus"] == "preparedForRun"
  assert c.delay.counts["outputNoneObservedCount"] == 0 and not c.delay.current["payload"]["runCallKnown"]
  old_id = c.delay.current["attempt"]
  c.attempt(value)
  prepare(c, value)
  assert c.delay.last_discard == (old_id, 1, "replacedUnreturned")
