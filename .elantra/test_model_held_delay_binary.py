"""Pure synthetic contracts: no Cereal, models, Params or native imports."""
import ast
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper():
  path = ROOT / "openpilot/selfdrive/modeld/model_observation_binary.py"
  spec = importlib.util.spec_from_file_location("held_binary_fixture", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class Owner:
  def __init__(self):
    self.identity, self.epoch, self.last_read = 1, b"e"*16, None
    self.cached, self.loss, self.errors, self.tick = {}, False, 0, 100

  def now(self):
    self.tick += 10
    return self.tick, "known"

  def error(self):
    self.loss = True
    self.errors += 1

  def write(self, read_id=1, origin="publishedHelper"):
    self.last_read = SimpleNamespace(owner=self, completed=True, retained=True, write_seen=True,
                                    row={"readId": read_id, "instanceId": self.identity, "site": "lagdToggle",
                                         "outcome": "returned", "valid": True, "boolValue": True,
                                         "valueKind": "knownBool", "castOutcome": "notApplied"})
    if origin == "cachedHelper":
      self.last_read.row.update(site="delayCache", boolValue=False, valueKind="knownFloat", castOutcome="succeeded")
    self.cached = {
      'read_id': read_id,
      'read_known': True,
      'write_ns': self.tick+5,
      'clock_status': "known",
      'origin': origin,
    }


class SM:
  def __init__(self):
    self.accesses = []
    self.values = {
      'frame': 9,
      'logMonoTime': 80,
      'recv_frame': 8,
      'seen': True,
      'updated': False,
      'valid': True,
      'alive': True,
    }

  def __getattr__(self, name):
    if name == "frame":
      self.accesses.append(name)
      return self.values[name]
    if name in self.values:
      self.accesses.append(name)
      owner = self

      class Entry:
        def __getitem__(self, service):
          assert service == "lateralDelay"
          owner.accesses.append(name+"[]")
          return owner.values[name]

      return Entry()
    raise AttributeError(name)


def refresh(tracker, owner, sm, value=.22, origin="publishedHelper"):
  token = tracker.begin(sm)
  assert tracker.offer(token, value) is value
  owner.write(origin=origin)
  tracker.finish(token, sm)
  return token


def test_metadata_budget_actual_flags_and_unknown_ownership_never_promotes():
  m, owner, sm = helper(), Owner(), SM()
  tracker = m.HeldCacheTracker(owner)
  refresh(tracker, owner, sm)
  assert len(sm.accesses) == 28
  assert sm.accesses.count("frame") == 4
  for name in ("logMonoTime", "recv_frame", "seen", "updated", "valid", "alive"):
    assert sm.accesses.count(name) == sm.accesses.count(name+"[]") == 2
  row = tracker.snapshot
  assert row["sourceEventNs"] == 80 and row["sourceRecvFrame"] == 8
  assert row["sourceSeen"] and not row["sourceUpdated"] and row["sourceValid"] and row["sourceAlive"]
  assert row["associatedReadKnown"] and row["cacheReadId"] == 1
  assert row["sourceReferenceKnown"] is False and row["valid"] is False
  assert row["associationStatus"] == "unsupported"


@pytest.mark.parametrize("field,value", [("frame", 10), ("logMonoTime", 81), ("valid", False), ("seen", 1)])
def test_changed_or_malformed_fence_retains_failure_without_false_lineage(field, value):
  m, owner, sm = helper(), Owner(), SM()
  tracker = m.HeldCacheTracker(owner)
  token = tracker.begin(sm)
  tracker.offer(token, .22)
  owner.write()
  sm.values[field] = value
  tracker.finish(token, sm)
  assert tracker.snapshot["associationStatus"] == "invalid"
  assert not tracker.snapshot["sourceReferenceKnown"] and not tracker.snapshot["valid"]
  assert owner.loss


def test_offer_returns_private_argument_unchanged_without_conversion():
  class Private:
    def __float__(self):
      pytest.fail("private conversion")

    def __repr__(self):
      pytest.fail("private formatting")

  m, owner, sm, value = helper(), Owner(), SM(), Private()
  tracker = m.HeldCacheTracker(owner)
  token = tracker.begin(sm)
  assert tracker.offer(token, value) is value
  owner.write()
  tracker.finish(token, sm)
  assert tracker.snapshot["argumentKind"] == "unknown" and not tracker.snapshot["argumentCaptured"]


def test_cached_branch_is_offered_not_used_and_duplicate_token_cannot_rebind():
  m, owner, sm = helper(), Owner(), SM()
  tracker = m.HeldCacheTracker(owner)
  token = refresh(tracker, owner, sm, origin="cachedHelper")
  frozen = dict(tracker.snapshot)
  assert frozen["branch"] == "cachedHelper" and frozen["associationStatus"] == "unsupported"
  tracker.finish(token, sm)
  assert dict(tracker.snapshot) == frozen and owner.loss


def test_fixed_payload_decoder_rejects_version_shape_and_private_scalar():
  m = helper()
  anchor = m.empty_anchor()
  assert len(anchor) == 30
  packet = m.empty_packet()
  wire = m.encode(packet)
  assert wire["codecVersion"] == 1 and len(wire["payload"]) == 976 and wire["payload"][:4] == b"ELMW"
  assert m.decode(wire) == packet
  with pytest.raises(m.LayoutError, match="version"):
    m.decode(dict(wire, codecVersion=2))
  with pytest.raises(m.LayoutError, match="reserved_shape"):
    m.decode(dict(wire, payload=b"ELMW"+b"\xff"*4+wire["payload"][8:]))
  with pytest.raises(m.LayoutError):
    m.encode({"shape": packet["shape"], "values": dict(packet["values"], private=object())})


def pure_modules():
  """Actual helper source with only observation-helper imports injected, no NumPy.

  This proves source fragments, not native models, actual SDK or IPC execution.
  """
  binary = helper()

  class ObservationImports(ast.NodeTransformer):
    def visit_Import(self, node):
      return None if any(a.name == "numpy" for a in node.names) else node

    def visit_ImportFrom(self, node):
      return None if node.module and node.module.startswith("openpilot.selfdrive.modeld.model_") else node

  def source(name, namespace):
    path = ROOT / ("openpilot/selfdrive/modeld/"+name+".py")
    tree = ObservationImports().visit(ast.parse(path.read_text(encoding="utf-8")))
    namespace["__name__"] = name+"_source_fixture"
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)

  delay = source("model_delay_observation", {"HeldCacheTracker": binary.HeldCacheTracker})
  asset = source("model_asset_observation", {"DelayConsumer": delay.DelayConsumer, "DelayJournal": delay.DelayJournal,
    "AssetPacket": binary.AssetPacket, "encode_semantic": binary.encode_semantic, "unpack_read": delay.unpack_read})
  return binary, delay, asset


class ParamsFixture:
  def __init__(self, toggle=True):
    self.toggle, self.calls = toggle, []

  def get_bool(self, key):
    self.calls.append(("bool", key))
    return self.toggle

  def get(self, key, *, return_default):
    self.calls.append(("get", key, return_default))
    return .11


def helper_fragment():
  path = ROOT / "openpilot/sunnypilot/livedelay/helpers.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
  ns = {"Params": object}
  exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
  return ns[fn.name]


def refresh_fragment(relative, ns):
  tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
  test = "sm.frame % 60 == 0" if "modeld_v2" in relative else "asset_observation is None"
  matches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == test
             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "get_lat_delay"
                     for call in ast.walk(n))
             and any(isinstance(assign, ast.Assign) and any(ast.unparse(t) == "model.lat_delay" for t in assign.targets)
                     for assign in ast.walk(n))]
  assert len(matches) == 1, "refresh fragment must uniquely contain the actual helper and cache assignment"
  node = matches[0]
  exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), ns)


class PublishedSM(SM):
  def __init__(self):
    super().__init__()
    self.argument, self.data_reads = .22, 0

  def __getitem__(self, service):
    assert service == "lateralDelay"
    self.data_reads += 1
    return SimpleNamespace(lateralDelay=self.argument)


def composition(runner="stock", *, binary=True):
  codec, delay, asset = pure_modules()
  ticks = iter(range(1000, 100000, 10))
  capture = asset.ModelAssetCapture(runner, clock=lambda: next(ticks), epoch=b"e"*16, delay_enabled=True, binary=binary)
  ref = capture.reserve(False)
  consumer = ref.delay_consumer()
  with consumer.read("delayInitial") as read:
    read.returned(.11)
  consumer.delay_written(.11, "constructor")
  model = SimpleNamespace(chestnut=False, generation=7, DEV="QCOM", LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.3,
                          _model_asset_reference=ref)
  ref.begin()
  ref.origin("stockBuiltin" if runner == "stock" else "envOverride")
  ref.loaded(model, 1344, 760, "stock" if runner == "stock" else "split")
  capture.activate(model)
  capture.delay.initial_long(capture, model, .2, .5, .3, "parsed")
  return codec, delay, asset, capture, model, consumer


@pytest.mark.parametrize("runner", ["stock", "tinygrad"])
def test_actual_refresh_fragment_one_argument_one_getter_and_current_B_independent(runner):
  binary, _, _, capture, model, consumer = composition(runner)
  sm, params, offset = PublishedSM(), ParamsFixture(), []
  sm.values["frame"] = 60
  rel = "openpilot/selfdrive/modeld/modeld.py" if runner == "stock" else "openpilot/sunnypilot/modeld_v2/modeld.py"
  ns = {
    'asset_observation': capture,
    'model': model,
    'sm': sm,
    'params': params,
    'get_lat_delay': helper_fragment(),
    'camera_offset_helper': SimpleNamespace(set_offset=offset.append),
  }
  refresh_fragment(rel, ns)
  assert model.lat_delay == .22 and sm.data_reads == 1
  assert params.calls[0] == ("bool", "LagdToggle")
  assert sum(name == "frame" for name in sm.accesses) == (4 if runner == "stock" else 5)
  assert consumer.held.snapshot["sourceEventNs"] == 80
  assert consumer.held.snapshot["associationStatus"] == "unsupported"
  # prepare performs its own unchanged CURRENT publication read B, not held A.
  sm.values.update(frame=61, logMonoTime=90, recv_frame=61, updated=True)
  sm.argument = .33
  capture.attempt(model)
  frame = SimpleNamespace(frame_id=7, timestamp_sof=500, timestamp_eof=600)
  capture.delay.prepare(capture, model, sm, frame, frame, {}, v_ego=5., effective_lat=.32,
    published_lat=.43 if runner == "stock" else None, lat_action=.32, long_action=.5, frame_delay=.01,
    action_delay=.01, lat_smoothing=.1, long_smoothing=.3, smoothing_origin="stockLiteral")
  assert capture.delay.current["payload"]["publishedEventNs"] == 90
  assert capture.delay.current["payload"]["publishedDelay"] == .33
  assert consumer.held.snapshot["sourceEventNs"] == 80
  assert sm.data_reads == 2
  if runner == "tinygrad":
    before = (len(sm.accesses), sm.data_reads, list(params.calls))
    refresh_fragment(rel, ns)
    assert len(sm.accesses) == before[0]+1 and sm.data_reads == before[1] and params.calls == before[2]
  assert binary.PAYLOAD_BYTES == 976


def prepared_composition():
  binary, delay, asset, capture, model, consumer = composition()
  sm = PublishedSM()
  refresh_fragment("openpilot/selfdrive/modeld/modeld.py", {
    'asset_observation': capture,
    'model': model,
    'sm': sm,
    'params': ParamsFixture(),
    'get_lat_delay': helper_fragment(),
  })
  capture.attempt(model)
  frame = SimpleNamespace(frame_id=7, timestamp_sof=500, timestamp_eof=600)
  capture.delay.prepare(capture, model, sm, frame, frame, {}, v_ego=5., effective_lat=.32, published_lat=.32,
    lat_action=.32, long_action=.5, frame_delay=.01, action_delay=.01,
    lat_smoothing=.1, long_smoothing=.3, smoothing_origin="stockLiteral")
  native = SimpleNamespace(logMonoTime=900, valid=False, modelV2=SimpleNamespace(frameId=7, frameIdExtra=7, timestampEof=600))
  driving = SimpleNamespace(logMonoTime=901, valid=False, drivingModelData=SimpleNamespace(frameId=8, frameIdExtra=9))
  sp = SimpleNamespace(logMonoTime=902, valid=False,
                       modelDataV2SP=SimpleNamespace(modelHeldDelayBinaryReference=SimpleNamespace()))
  return binary, delay, asset, capture, model, consumer, sp, native, driving


def test_binary_stage_legacy_absent_prefix_preserved_on_failure_exact_final_return_retirement():
  binary, _, _, capture, model, consumer, sp, native, driving = prepared_composition()
  ids = tuple(row["readId"] for row in consumer.pending)
  token = capture.stage(sp, native, driving, model)
  assert token is not None and not hasattr(sp.modelDataV2SP, "modelAssetReference")
  wire = vars(sp.modelDataV2SP.modelHeldDelayBinaryReference)
  packet = binary.decode(wire)
  assert len(wire["payload"]) == 976 and packet["shape"]["nativeReadCount"] == len(ids)
  assert packet["values"]["parent.drivingFrameId"] == 8
  assert packet["values"]["heldCacheSourceAnchor.sourceReferenceKnown"] is False
  assert packet["values"]["heldCacheSourceAnchor.associationStatus"] == "unsupported"
  frozen = wire["payload"]
  capture.delay.final_send_started(token)
  capture.send_failed(token)
  assert tuple(row["readId"] for row in consumer.pending) == ids and consumer.held.pending
  assert wire["payload"] == frozen and sp.valid is False and native.valid is False
  # A new native attempt uses the same continuous consumers, with original prefix.
  capture.attempt(model)
  frame = SimpleNamespace(frame_id=7, timestamp_sof=500, timestamp_eof=600)
  capture.delay.prepare(capture, model, PublishedSM(), frame, frame, {}, v_ego=5., effective_lat=.32, published_lat=.32,
    lat_action=.32, long_action=.5, frame_delay=.01, action_delay=.01,
    lat_smoothing=.1, long_smoothing=.3, smoothing_origin="stockLiteral")
  token = capture.stage(sp, native, driving, model)
  assert token is not None
  capture.delay.final_send_started(token)
  capture.retire(token)
  assert not consumer.pending and not consumer.held.pending


def test_binary_encoding_failure_does_not_stage_or_retire_or_touch_native_fields():
  _, _, asset, capture, model, consumer, sp, native, driving = prepared_composition()
  ids = tuple(row["readId"] for row in consumer.pending)
  def fail(*args):
    raise ValueError("synthetic encoder failure")
  # The actual stage source resolves this explicitly injected pure encoder.
  asset.ModelAssetCapture.stage.__globals__["encode_semantic"] = fail
  assert capture.stage(sp, native, driving, model) is None
  assert capture.staged is capture.delay.staged is capture.delay.draft is None
  assert tuple(row["readId"] for row in consumer.pending) == ids and consumer.held.pending
  assert vars(sp.modelDataV2SP.modelHeldDelayBinaryReference) == {} and sp.valid is False


def test_factory_exact_flag_precedence_and_old_modes_unchanged(monkeypatch):
  _, _, asset = pure_modules()
  calls = []
  asset.create_model_asset_capture.__globals__["ModelAssetCapture"] = lambda *a, **kw: calls.append((a, kw)) or kw
  for flag in ("ELANTRA_MODEL_ASSET_OBSERVER", "ELANTRA_MODEL_DELAY_REFERENCE_OBSERVER", "ELANTRA_MODEL_HELD_DELAY_BINARY_OBSERVER"):
    monkeypatch.delenv(flag, raising=False)
  assert asset.create_model_asset_capture("stock") is None and not calls
  monkeypatch.setenv("ELANTRA_MODEL_ASSET_OBSERVER", "1")
  assert asset.create_model_asset_capture("stock") == {
    'delay_enabled': False,
    'binary': False,
  }
  monkeypatch.setenv("ELANTRA_MODEL_HELD_DELAY_BINARY_OBSERVER", "true")
  assert asset.create_model_asset_capture("stock")["binary"] is False
  monkeypatch.setenv("ELANTRA_MODEL_HELD_DELAY_BINARY_OBSERVER", "1")
  assert asset.create_model_asset_capture("stock") == {
    'delay_enabled': True,
    'binary': True,
  }


def test_supplied_bit_images_uint64_extrema_and_absence_are_exact():
  m = helper()
  packet = m.empty_packet()
  for path in ("parent.attemptSequence", "heldCacheSourceAnchor.cacheReadId"):
    packet["values"][path] = 2**64-1
  packet["values"]["heldCacheSourceAnchor.argumentValue"] = {"bits": 0x8000000000000000}
  assert m.decode(m.encode(packet)) == packet
  nested = m._nested(packet)
  nested["heldCacheSourceAnchor"]["argumentValue"] = -0.
  # Semantic conversion captures this Python scalar, not hypothetical earlier bits.
  for row in m.PARENT:
    if row["source_type"].startswith("Float"):
      nested["parent"][row["source_path"].rsplit(".", 1)[1]] = 0.
  recovered = m.decode(m.encode_semantic(nested["parent"], nested["heldCacheSourceAnchor"]))
  assert recovered["values"]["heldCacheSourceAnchor.argumentValue"]["bits"] == int.from_bytes(struct.pack("<d", -0.), "little")


def test_outer_decoder_rejects_legacy_binary_conflict_before_payload_access():
  m = helper()
  class Body:
    def _has(self, name):
      return True

    def __getattr__(self, name):
      pytest.fail("conflicting outer payload accessed")
  with pytest.raises(m.LayoutError, match="conflict"):
    m.decode_body(Body())


@pytest.mark.parametrize("failure", ["payload_setter", "commit"])
def test_attachment_write_then_raise_invalidates_actual_binary_target(failure):
  _, _, _, capture, model, consumer, sp, native, driving = prepared_composition()
  class WriteThenRaise:
    def __setattr__(self, name, value):
      object.__setattr__(self, name, value)
      if name == "payload":
        raise ValueError("synthetic post-write setter failure")
  if failure == "payload_setter":
    sp.modelDataV2SP.modelHeldDelayBinaryReference = WriteThenRaise()
  else:
    def fail(*args):
      raise ValueError("synthetic commit failure")
    capture.delay.commit_stage = fail
  ids = tuple(row["readId"] for row in consumer.pending)
  assert capture.stage(sp, native, driving, model) is None
  target = sp.modelDataV2SP.modelHeldDelayBinaryReference
  assert target.codecVersion != 1 or len(target.payload) != 976
  assert tuple(row["readId"] for row in consumer.pending) == ids and consumer.held.pending
  assert capture.staged is capture.delay.staged is capture.delay.draft is None
  assert sp.valid is native.valid is False


def test_packet_bounds_reject_before_reference_scan_or_private_copy():
  m = helper()
  class NoScan(list):
    def __iter__(self):
      pytest.fail("unbounded reference scan")
  raw = m._nested(m.empty_packet())["parent"]
  for row in m.PARENT:
    if row["source_type"].startswith("Float"):
      raw[row["source_path"].rsplit(".", 1)[1]] = 0.
  packet = m.AssetPacket(**raw)
  packet.references = NoScan([object()]*5)
  with pytest.raises(m.LayoutError):
    packet.packet()


def test_semantic_validation_precedes_detached_copy_and_nested_recovery():
  m = helper()
  nested = m._nested(m.empty_packet())
  nested["parent"]["present"] = object()
  m._from_nested = lambda *a: pytest.fail("invalid private scalar copied into nested recovery")
  with pytest.raises(m.LayoutError):
    m.encode_semantic(nested["parent"], m.empty_anchor())


@pytest.mark.parametrize("mismatch", ["owner", "instance", "site"])
def test_foreign_read_owner_instance_or_site_cannot_bind_colliding_read_id(mismatch):
  m, owner, sm = helper(), Owner(), SM()
  tracker = m.HeldCacheTracker(owner)
  token = tracker.begin(sm)
  tracker.offer(token, .22)
  owner.write()
  if mismatch == "owner":
    owner.last_read.owner = Owner()
  elif mismatch == "instance":
    owner.last_read.row["instanceId"] = 2
  else:
    owner.last_read.row["site"] = "delayCache"
  tracker.finish(token, sm)
  assert not tracker.snapshot["associatedReadKnown"]
  assert tracker.snapshot["associationStatus"] == "invalid"


@pytest.mark.parametrize("runner", ["stock", "tinygrad"])
def test_actual_refresh_native_exception_identity_and_assignment_order(runner):
  _, _, _, capture, model, consumer = composition(runner)
  sm, params = PublishedSM(), ParamsFixture()
  sm.values["frame"] = 60
  failure = KeyboardInterrupt("synthetic native getter exception")
  def fail(key):
    params.calls.append(("bool", key))
    raise failure
  params.get_bool = fail
  model.lat_delay = .9
  rel = "openpilot/selfdrive/modeld/modeld.py" if runner == "stock" else "openpilot/sunnypilot/modeld_v2/modeld.py"
  with pytest.raises(KeyboardInterrupt) as actual:
    refresh_fragment(rel, {
      'asset_observation': capture,
      'model': model,
      'sm': sm,
      'params': params,
      'get_lat_delay': helper_fragment(),
      'camera_offset_helper': SimpleNamespace(set_offset=lambda value: pytest.fail("later offset")),
    })
  assert actual.value is failure and model.lat_delay == .9 and sm.data_reads == 1
  assert params.calls == [("bool", "LagdToggle")]
  assert len(sm.accesses) == (14 if runner == "stock" else 15)  # BEFORE only, no fabricated successful AFTER.
  assert consumer.held.inflight is None and consumer.loss and not consumer.held.snapshot["present"]


def test_actual_cached_helper_read_site_and_offered_argument_are_distinct():
  _, _, _, capture, model, consumer = composition()
  sm, params = PublishedSM(), ParamsFixture(toggle=False)
  refresh_fragment("openpilot/selfdrive/modeld/modeld.py", {
    'asset_observation': capture,
    'model': model,
    'sm': sm,
    'params': params,
    'get_lat_delay': helper_fragment(),
  })
  anchor = consumer.held.snapshot
  assert model.lat_delay == .11 and anchor["argumentValue"] == .22
  assert anchor["branch"] == "cachedHelper" and anchor["associationStatus"] == "unsupported"
  assert consumer.last_read.owner is consumer and consumer.last_read.row["site"] == "delayCache"
  assert anchor["cacheReadId"] == consumer.last_read.row["readId"] and anchor["associatedReadKnown"]
  assert not anchor["sourceReferenceKnown"] and not anchor["valid"]
  assert params.calls == [("bool", "LagdToggle"), ("get", "LagdValueCache", True)]


@pytest.mark.parametrize("runner", ["stock", "tinygrad"])
def test_actual_off_and_legacy_refresh_have_no_new_metadata_reads(runner):
  _, _, _, capture, model, consumer = composition(runner, binary=False)
  assert consumer.held is None
  rel = "openpilot/selfdrive/modeld/modeld.py" if runner == "stock" else "openpilot/sunnypilot/modeld_v2/modeld.py"
  results = []
  for observed in (None, capture):
    sm, params = PublishedSM(), ParamsFixture()
    sm.values["frame"] = 60
    refresh_fragment(rel, {
      'asset_observation': observed,
      'model': model,
      'sm': sm,
      'params': params,
      'get_lat_delay': helper_fragment(),
      'camera_offset_helper': SimpleNamespace(set_offset=lambda value: None),
    })
    results.append((model.lat_delay, params.calls, sm.data_reads, sm.accesses))
  assert results[0] == results[1]
  assert results[0][2] == 1 and results[0][3] == ([] if runner == "stock" else ["frame"])


def test_two_actual_consumers_unique_tokens_exhaustion_and_lost_anchor_history():
  _, _, _, capture, model, consumer = composition()
  second = capture.reserve(True).delay_consumer()
  assert second.identity == 2 and second.held is not consumer.held
  assert capture.reserve(False) is None
  sm = PublishedSM()
  tracker = consumer.held
  first = tracker.begin(sm)
  value = object()
  assert tracker.offer(dict(first), value) is value  # Equal-looking token is not the real token.
  tracker.failed()
  tracker.sequence = 2**64-1
  snapshot = tracker.snapshot
  assert tracker.begin(sm) is None and tracker.snapshot is snapshot
  assert tracker.offer(None, .4) == .4 and consumer.loss
  # A second source-owned cache write before retirement keeps latest observation
  # but reports lost earlier history; it never coalesces that history as complete.
  owner, plain_sm = Owner(), SM()
  plain = helper().HeldCacheTracker(owner)
  refresh(plain, owner, plain_sm)
  first_snapshot = plain.snapshot
  refresh(plain, owner, plain_sm)
  assert plain.snapshot is not first_snapshot and owner.loss and plain.pending
  plain.retire(first_snapshot)
  assert plain.pending
  plain.retire(plain.snapshot)
  assert not plain.pending


def test_immediate_before_disabled_native_projection_and_new_overlay_registration():
  # Execute only the existing pure AST helper definitions, without the old module's
  # schema plugin/imports. These constants bind this patch's immediate-before
  # sources, not the older Tinygrad original-baseline literal (still mismatches).
  path = ROOT / ".elantra/test_model_asset_observation.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  wanted = {"normalized", "ast_hash", "DisabledProjection"}
  nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
  ns = {
    'ast': ast,
    'hashlib': hashlib,
    'json': json,
    'STOCK': "openpilot/selfdrive/modeld/modeld.py",
    'V2': "openpilot/sunnypilot/modeld_v2/modeld.py",
    'parsed': lambda rel: ast.parse((ROOT / rel).read_text(encoding="utf-8")),
  }
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
  # Byte bindings: stock e1c36a26..., Tinygrad f2dfb3d6..., verified independently
  # in source-before.json and disabled-projection-diagnosis-v1.json, no D: reads here.
  expected = {"openpilot/selfdrive/modeld/modeld.py": "5ab9c5a921a7932472f0cee0a6a0f32112f09bfde619fce67a61cf4bc1346cba",
              "openpilot/sunnypilot/modeld_v2/modeld.py": "14b0cb2b6af0751ecb58271eadb4092eb5b1085fc792ea4342f60f8e8025abfd"}
  for rel, original in expected.items():
    assert ns["ast_hash"](ns["DisabledProjection"]().visit(ns["parsed"](rel))) == original
  tree = ast.parse((ROOT / ".elantra/sync.py").read_text(encoding="utf-8"))
  added = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "OVERLAY_ADDED")
  assert "openpilot/selfdrive/modeld/model_observation_binary.py" in added
