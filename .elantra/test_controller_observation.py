"""Observer source/serialization contracts; fixtures do not establish timing or car performance."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("controller_observation", ROOT / "openpilot/selfdrive/controls/lib/controller_observation.py")
obs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(obs)


def observer(producer="controls", **kwargs):
  ticks = iter(range(1_000_000, 10_000_000, 100))
  return obs.Observer(producer, clock=lambda: next(ticks), epoch=b"e" * 16, start_worker=False, **kwargs)


def sm(services):
  maps = {name: dict.fromkeys(services, value) for name, value in
          (("logMonoTime", 900_000), ("recv_frame", 1), ("recv_time", .0009),
           ("valid", True), ("seen", True), ("updated", True), ("alive", True), ("freq_ok", True))}
  return SimpleNamespace(**maps)


def records(observer):
  return list(observer._pending)


@pytest.mark.parametrize("value", [None, "", "0", "true", "01"])
def test_disabled_has_no_observer_constructor_thread_publisher_clock_or_params(monkeypatch, value):
  if value is None:
    monkeypatch.delenv("ELANTRA_CONTROLLER_OBSERVER", raising=False)
  else:
    monkeypatch.setenv("ELANTRA_CONTROLLER_OBSERVER", value)
  monkeypatch.setattr(obs, "Observer", lambda *a, **k: pytest.fail("disabled allocated observer"))
  assert obs.create_observer("controls") is None


def test_source_refs_preserve_uint64_and_receive_identity_without_payload_access():
  o = observer()
  source = sm(obs.CONTROLS_INPUTS)
  source.logMonoTime["carState"] = 2**53 + 1
  o.begin_loop()
  o.capture("updateEnd", sm=source)
  ref = next(r for r in records(o)[-1]["refs"] if r["service"] == "carState")
  assert ref["event_ns"] == 2**53 + 1 and ref["recv_frame"] == 1
  assert ref["recv_time"] == .0009 and ref["valid"] is True
  assert ref["identity_kind"] == "timestampReference"
  assert next(r for r in records(o)[-1]["refs"] if r["service"] == "liveLocationKalman")["status"] == "notSubscribed"
  assert records(o)[-1]["policy_complete"] is False


def test_reheld_reference_allowed_but_new_receive_at_duplicate_or_regressing_timestamp_incomplete():
  o = observer()
  source = sm(obs.CONTROLS_INPUTS)
  o.begin_loop()
  o.capture("updateEnd", sm=source)
  o.capture("computeBegin", sm=source)
  assert records(o)[-1]["capture_complete"]
  source.recv_frame["carState"] += 1
  o.capture("computeEnd", sm=source)
  assert "timestampTie" in records(o)[-1]["faults"]
  source.logMonoTime["carState"] -= 1
  source.recv_frame["carState"] += 1
  o.capture("updateEnd", sm=source)
  assert "timestampRegression" in records(o)[-1]["faults"]


@pytest.mark.parametrize("change", ["invalid", "nonfinite", "missing", "wrong_type"])
def test_source_invalid_unknown_and_nonfinite_are_explicit(change):
  o, source = observer(), sm(obs.CONTROLS_INPUTS)
  if change == "invalid":
    source.valid["carState"] = False
  elif change == "nonfinite":
    source.recv_time["carState"] = float("nan")
  elif change == "missing":
    del source.logMonoTime["carState"]
  else:
    source.logMonoTime["carState"] = "900000"
  o.begin_loop()
  o.capture("updateEnd", sm=source)
  assert not records(o)[-1]["capture_complete"]
  assert records(o)[-1]["faults"]


def test_unseen_and_invalid_publications_or_distinct_sends_at_tied_time_remain_incomplete():
  o, source = observer(), sm(obs.CONTROLS_INPUTS)
  source.seen["carState"] = False
  o.begin_loop()
  o.capture("updateEnd", sm=source)
  assert not records(o)[-1]["capture_complete"]
  o.capture("publishBegin", service="carControl", event=SimpleNamespace(logMonoTime=99, valid=False))
  assert "sourceInvalid" in records(o)[-1]["faults"]
  o.capture("publishBegin", service="carControl", event=SimpleNamespace(logMonoTime=99, valid=True))
  assert "timestampTie" in records(o)[-1]["faults"]


def test_bounds_precede_can_copy_and_queue_overflow_is_visible():
  class TooMany(list):
    def __iter__(self):
      pytest.fail("oversize CAN was traversed")
  o = observer("card", capacity=2)
  o.begin_loop()
  o.capture("applyEnd", tx=TooMany([None] * (obs.MAX_CAN + 1)))
  assert "txOversize" in records(o)[-1]["faults"] and not records(o)[-1]["tx"]
  o.capture("computeBegin")
  assert o.dropped == 1 and len(records(o)) == 2
  o._pending.popleft()
  o.capture("computeEnd")
  assert records(o)[-1]["sequence"] == 4 and records(o)[-1]["dropped"] == 1


def test_loop_budget_is_bounded_and_serialization_or_send_is_never_in_capture():
  o = observer()
  o.begin_loop()
  for _ in range(obs.MAX_LOOP_RECORDS + 20):
    o.capture("computeBegin")
  assert len(records(o)) <= obs.MAX_LOOP_RECORDS
  assert o.dropped > 0
  assert all(r["capture_bytes"] <= obs.MAX_RECORD_CAPTURE_BYTES for r in records(o))


def test_oversize_loses_coverage_for_epoch_and_input_copy_budget_is_bounded():
  o = observer("card")
  o.begin_loop()
  o.capture("applyEnd", tx=[(0x340, b"x" * 65, 0)])
  o.capture("computeBegin")
  assert not records(o)[-1]["capture_complete"] and "priorLoss" in records(o)[-1]["faults"]
  source, o = sm(obs.CONTROLS_INPUTS), observer()
  o.begin_loop()
  for _ in range(obs.MAX_LOOP_RECORDS):
    o.capture("computeBegin", sm=source)
  assert sum(r["capture_bytes"] for r in records(o)) <= obs.MAX_LOOP_CAPTURE_BYTES
  assert o.dropped > 0


def test_worker_failure_stops_without_retry_and_future_capture_is_ineligible():
  calls = []
  o = observer()
  o.begin_loop()
  def broken(service, record):
    calls.append(service)
    raise RuntimeError("private data must not be logged")
  assert o.flush_one(broken) is False
  assert o.failed and o.errors == 1 and calls == ["controlsObservation"]
  assert o.flush_one(broken) is False and len(calls) == 1
  o.capture("computeBegin")
  assert records(o)[-1]["capture_complete"] is False


def test_capture_clock_ties_and_errors_do_not_escape_controller():
  o = obs.Observer("controls", clock=lambda: 1, epoch=b"e" * 16, start_worker=False)
  o.begin_loop()
  o.capture("computeBegin")
  assert "clockTie" in records(o)[-1]["faults"]
  o.clock = lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))
  o.capture("computeEnd")
  assert o.errors == 1


def test_tap_preserves_original_send_and_exception_and_only_known_fields():
  o, sent = observer(), []
  class PM:
    sock = {}
    def send(self, service, event):
      sent.append((service, event))
      return 41
  tap = obs.PublisherTap(PM(), o)
  event = SimpleNamespace(logMonoTime=123, valid=True, secret="VIN_OR_PARAM", to_dict=lambda: pytest.fail("payload dump"))
  o.begin_loop()
  assert tap.send("carControl", event) == 41
  assert sent == [("carControl", event)]
  assert [r["stage"] for r in records(o)][-2:] == ["publishBegin", "publishEnd"]
  assert records(o)[-1]["event_ns"] == 123
  assert "VIN_OR_PARAM" not in repr(records(o))
  def broken(*args):
    raise OSError("original error")
  tap.parent.send = broken
  with pytest.raises(OSError, match="original error"):
    tap.send("carControl", event)
  assert records(o)[-1]["stage"] == "publishFailed"


def method(path, cls, name, namespace):
  tree = ast.parse((ROOT / path).read_text())
  fn = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
            for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
  fn.decorator_list, fn.returns = [], None
  for arg in fn.args.args:
    arg.annotation = None
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
  return namespace[name]


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_controlsd_run_keeps_compute_publish_refresh_sp_order(enabled):
  calls, o = [], observer() if enabled else None
  class RK:
    def __init__(self, *args, **kwargs):
      pass
    def monitor_time(self):
      raise StopIteration
  source = sm(obs.CONTROLS_INPUTS)
  target = SimpleNamespace(observer=o, sm=source)
  def update():
    calls.append("update")
  def compute():
    calls.append("compute")
    return "CC", "log"
  class PM:
    sock = {}
    def send(self, service, data):
      calls.append(service)
  target.pm = obs.PublisherTap(PM(), o) if o else PM()
  target.update, target.state_control = update, compute
  target.publish = lambda *args: target.pm.send("carControl", SimpleNamespace(logMonoTime=101, valid=True))
  target.get_params_sp = lambda source: calls.append("refresh")
  target.run_ext = lambda source, pm: pm.send("carControlSP", SimpleNamespace(logMonoTime=102, valid=True))
  run = method("openpilot/selfdrive/controls/controlsd.py", "Controls", "run", {"Ratekeeper": RK})
  with pytest.raises(StopIteration):
    run(target)
  assert calls == ["update", "compute", "carControl", "refresh", "carControlSP"]
  if o:
    stages = [r["stage"] for r in records(o)]
    assert stages.index("computeBegin") < stages.index("computeEnd") < stages.index("paramsRefreshBegin")
    sp = next(i for i, r in enumerate(records(o)) if r["stage"] == "publishBegin" and r["service"] == "carControlSP")
    assert stages.index("paramsRefreshEnd") < sp


@pytest.mark.parametrize("enabled,alive,passive", [(False, True, False), (True, True, False), (True, False, False), (True, True, True)])
def test_actual_card_old_consumption_prior_output_and_apply_or_skip(enabled, alive, passive):
  o, calls = (observer("card") if enabled else None), []
  source = sm(obs.CARD_INPUTS)
  source.frame = 1
  values = {"carControl": "OLD", "carControlSP": "OLD_SP", "onroadEvents": []}
  class SM(SimpleNamespace):
    def __getitem__(self, key):
      return values[key]
    def all_checks(self, services):
      return True
    def all_alive(self, services):
      return alive
  source = SM(**vars(source))
  cs = SimpleNamespace(canValid=True)
  target = SimpleNamespace(observer=o, sm=source, CP=SimpleNamespace(passive=passive), initialized_prev=True,
                           last_actuators_output="PRIOR", can_log_mono_time=0)
  class PM:
    sock = {}
    def send(self, service, event):
      calls.append((service, getattr(getattr(event, "carOutput", None), "actuatorsOutput", None)))
  target.pm = obs.PublisherTap(PM(), o) if o else PM()
  def updated():
    if o:
      o.begin_loop()
      o.capture("consumeEnd", sm=source)
    calls.append(("consumed", values["carControl"]))
    return cs, "CS_SP", None
  target.state_update = updated
  def publish(*args):
    for service in ("carOutput", "carState"):
      event = SimpleNamespace(logMonoTime=110 if service == "carOutput" else 111, valid=True,
                              carOutput=SimpleNamespace(actuatorsOutput=target.last_actuators_output))
      target.pm.send(service, event)
  target.state_publish = publish
  target.CI = SimpleNamespace(apply=lambda cc, sp, t: (calls.append(("apply", cc, sp, t)) or ("NEW", [(0x340, b"12345678", 0)])))
  target.can_callbacks = ()
  ns = {"time": SimpleNamespace(monotonic=lambda: 2.), "REPLAY": False, "convert_carControlSP": lambda x: x,
        "can_list_to_can_capnp": lambda *a, **kw: b"serialized", "EventName": SimpleNamespace(selfdriveInitializing=1)}
  target.controls_update = method("openpilot/selfdrive/car/card.py", "Car", "controls_update", ns).__get__(target)
  step = method("openpilot/selfdrive/car/card.py", "Car", "step", ns)
  step(target)
  assert calls[:3] == [("consumed", "OLD"), ("carOutput", "PRIOR"), ("carState", "PRIOR")]
  if alive and not passive:
    assert calls[3] == ("apply", "OLD", "OLD_SP", 2_000_000_000)
    assert target.last_actuators_output == "NEW"
  else:
    assert len(calls) == 3
  if o:
    stages = [r["stage"] for r in records(o)]
    assert stages.index("consumeEnd") < stages.index("publishBegin")
    assert ("applyBegin" in stages) == (alive and not passive)
    assert ("applySkipped" in stages) == (not alive or passive)


@pytest.fixture(scope="module")
def isolated_schema(tmp_path_factory):
  import capnp
  import shutil
  directory = tmp_path_factory.mktemp("observer-schema")
  for path in (ROOT / "openpilot/cereal").glob("*.capnp"):
    shutil.copyfile(path, directory / path.name)
  shutil.copyfile(ROOT / "opendbc_repo/opendbc/car/car.capnp", directory / "car.capnp")
  (directory / "include").mkdir()
  shutil.copyfile(ROOT / "openpilot/cereal/include/c++.capnp", directory / "include/c++.capnp")
  return capnp.load(str(directory / "log.capnp"), imports=[str(directory), str(directory / "include")])


@pytest.mark.parametrize("producer,service", [("controls", "controlsObservation"), ("card", "cardObservation")])
def test_actual_isolated_schema_roundtrip_and_constant_ineligibility(isolated_schema, producer, service):
  o = observer(producer)
  o.begin_loop()
  o.capture("updateEnd", sm=sm(obs.CONTROLS_INPUTS if producer == "controls" else obs.CARD_INPUTS))
  raw = obs.encode_record(records(o)[-1], isolated_schema, publisher_ns=9_000_000)
  assert len(raw) <= obs.MAX_SERIALIZED_BYTES
  with isolated_schema.Event.from_bytes(raw) as event:
    result = getattr(event, service).observation
    assert event.which() == service and event.logMonoTime == 9_000_000
    assert result.captureNs == records(o)[-1]["capture_ns"]
    assert not result.policyComplete and not result.sourceAuthenticated and not result.predictiveEligible
    assert str(result.rawBoolPresence) == "unknown" and str(result.txEchoCorrelation) == "unsupported"
    assert str(result.calibratedRollUncertainty) == "unknown"


def test_actual_tx_header_clock_and_order_without_payload_dump(isolated_schema):
  o = observer("card")
  o.begin_loop()
  frames = [(0x340, b"VINSECRET", 0), (0x7e0, b"ECUFIRMWARE", 0)]
  o.capture("applyEnd", tx=frames)
  raw = obs.encode_record(records(o)[-1], isolated_schema, publisher_ns=9_000_000)
  assert b"VINSECRET" not in raw and b"ECUFIRMWARE" not in raw
  with isolated_schema.Event.from_bytes(raw) as event:
    entries = event.cardObservation.observation.tx
    assert [(x.address, x.src, x.payloadLength) for x in entries] == [(0x340, 0, 9), (0x7e0, 0, 11)]
    assert not any(x.payloadCaptured for x in entries)
  source = isolated_schema.Event.new_message(logMonoTime=2**53+19, valid=False)
  source.init("sendcan", 0)
  o.capture("publishEnd", service="sendcan", event=source.to_bytes())
  raw = obs.encode_record(records(o)[-1], isolated_schema, publisher_ns=9_000_100)
  with isolated_schema.Event.from_bytes(raw) as event:
    entry = event.cardObservation.observation
    assert entry.eventNs == 2**53+19 and not entry.eventValid and entry.eventPresent


def test_raw_tx_validity_future_and_distinct_publication_ties_checked_in_worker(isolated_schema):
  o, previous = observer("card"), {}
  o.begin_loop()
  source = isolated_schema.Event.new_message(logMonoTime=900_000, valid=True)
  source.init("sendcan", 0)
  for iteration in range(2):
    o.capture("publishBegin", service="sendcan", event=source.to_bytes())
    raw = obs.encode_record(records(o)[-1], isolated_schema, publisher_ns=9_000_000,
                            previous_publications=previous)
    with isolated_schema.Event.from_bytes(raw) as event:
      result = event.cardObservation.observation
      assert result.captureComplete == (iteration == 0)
      assert ("timestampTie" in [str(v) for v in result.faults]) == (iteration == 1)
  source.logMonoTime, source.valid = 20_000_000, False
  o.capture("publishBegin", service="sendcan", event=source.to_bytes())
  raw = obs.encode_record(records(o)[-1], isolated_schema, publisher_ns=9_000_000, previous_publications=previous)
  with isolated_schema.Event.from_bytes(raw) as event:
    result = event.cardObservation.observation
    assert not result.captureComplete
    assert {"sourceInvalid", "futureReference"} <= {str(v) for v in result.faults}


def test_overlay_registration_covers_every_runtime_schema_service_and_hook():
  tree = ast.parse((ROOT / ".elantra/sync.py").read_text())
  values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"OVERLAY_ADDED", "OVERLAY_MODIFIED"}}
  assert "openpilot/selfdrive/controls/lib/controller_observation.py" in values["OVERLAY_ADDED"]
  assert {"openpilot/cereal/custom.capnp", "openpilot/cereal/log.capnp", "openpilot/cereal/services.py",
          "openpilot/selfdrive/controls/controlsd.py", "openpilot/selfdrive/car/card.py"} <= set(values["OVERLAY_MODIFIED"])


def drain(o):
  result = []
  while o._pending and not o.failed:
    o.flush_one(lambda service, record: result.append(record))
  return result


def test_startup_exact_bytes_are_worker_hashed_released_and_never_exported(monkeypatch, isolated_schema):
  import hashlib
  o = obs.Observer("controls", clock=iter(range(2**53+1, 2**53+1000, 7)).__next__, epoch=b"e"*16, start_worker=False)
  private = b"VIN-and-firmware-private-raw-buffer"
  expected = hashlib.sha256(private).digest()
  token = o.begin_startup_reference("paramsCarParams")
  real_hash, calls = hashlib.sha256, []
  monkeypatch.setattr(obs.hashlib, "sha256", lambda raw: (calls.append(raw), real_hash(raw))[1])
  o.end_startup_reference(token, private, "complete")
  assert calls == [] and o._startup_reserved_bytes == len(private)
  output = drain(o)
  assert calls == [private] and o._startup_released_bytes == len(private)
  begin, end = (r["startup_reference"] for r in output)
  assert begin["id"] == end["id"] == 1 and begin["begin_ns"] == end["begin_ns"] == 2**53+1
  assert end["end_ns"] > begin["begin_ns"] and end["digest_ready_ns"] > end["end_ns"]
  assert end["sha256"] == expected and end["status"] == "ready" and end["byte_count"] == len(private)
  for record in output:
    assert "raw_bytes" not in record["startup_reference"] and private not in repr(record).encode()
    raw = obs.encode_record(record, isolated_schema, publisher_ns=2**53+900)
    assert len(raw) <= 1024 and private not in raw
    with isolated_schema.Event.from_bytes(raw) as event:
      ref = event.controlsObservation.observation.startupReference
      assert ref.present and ref.source == "paramsCarParams" and ref.decoder == "carParams"
      assert not event.controlsObservation.observation.policyComplete and not event.controlsObservation.observation.sourceAuthenticated


@pytest.mark.parametrize("raw,outcome,status", [(b"x"*65537, "complete", "oversize"), (bytearray(b"x"), "complete", "nonbytes"),
  (None, "readFailed", "readFailed"), (b"invalid", "parseFailed", "parseFailed"), (None, "serializationFailed", "serializationFailed")],
  ids=["oversize", "mutable", "read_failure", "parse_failure", "serialize_failure"])
def test_startup_invalid_buffers_and_original_failure_outcomes_never_get_valid_digest(raw, outcome, status):
  o = observer()
  token = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(token, raw, outcome)
  end = drain(o)[-1]["startup_reference"]
  assert end["status"] == status and end["sha256"] == b"" and end["digest_ready_ns"] == 0
  assert o._startup_reserved_bytes == o._startup_released_bytes == 0


def test_startup_cp_sp_independent_lifetime_and_queue_loss_have_no_valid_fallback():
  o = observer()
  cp = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(cp, b"CP", "complete")
  sp = o.begin_startup_reference("paramsCarParamsSP")
  o.end_startup_reference(sp, b"SP", "complete")
  assert cp != sp and len(o._pending) == 4
  assert o.begin_startup_reference("paramsCarParams") is None
  output = drain(o)
  assert len(output) == 4 and o._startup_jobs == 2
  assert {r["startup_reference"]["source"] for r in output} == {"paramsCarParams", "paramsCarParamsSP"}
  tiny = observer(capacity=1)
  token = tiny.begin_startup_reference("paramsCarParams")
  tiny.end_startup_reference(token, b"secret", "complete")
  assert tiny.dropped and tiny.loss
  assert tiny._startup_reserved_bytes == tiny._startup_released_bytes
  assert all("raw_bytes" not in r["startup_reference"] for r in tiny._pending)


def test_startup_digest_and_publisher_failure_release_all_buffers_without_retry(monkeypatch):
  o = observer()
  token = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(token, b"CP", "complete")
  monkeypatch.setattr(obs.hashlib, "sha256", lambda raw: (_ for _ in ()).throw(ValueError("secret failure")))
  output = drain(o)
  assert output[-1]["startup_reference"]["status"] == "digestFailed"
  assert output[-1]["startup_reference"]["sha256"] == b"" and o._startup_released_bytes == 2
  stopped = observer()
  cp = stopped.begin_startup_reference("paramsCarParams")
  stopped.end_startup_reference(cp, b"CP-private", "complete")
  sp = stopped.begin_startup_reference("paramsCarParamsSP")
  stopped.end_startup_reference(sp, b"SP-private", "complete")
  assert not stopped.flush_one(lambda *a: (_ for _ in ()).throw(RuntimeError("publisher failed")))
  assert stopped.failed and stopped._startup_reserved_bytes == stopped._startup_released_bytes
  assert all("raw_bytes" not in r["startup_reference"] for r in stopped._pending)


def startup_fragment(path, cls, targets, namespace):
  tree = ast.parse((ROOT/path).read_text())
  constructor = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                     for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
  selected = []
  for node in constructor.body:
    assigned = [target for n in ast.walk(node) if isinstance(n, ast.Assign) for target in n.targets]
    if any((isinstance(t, ast.Name) and t.id in targets) or (isinstance(t, ast.Attribute) and t.attr in targets) for t in assigned):
      selected.append(node)
  wrapper = ast.parse("def run(self):\n  return locals()\n").body[0]
  wrapper.body = [*selected, *wrapper.body]
  exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)
  return namespace["run"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", ["cp", "sp"])
def test_actual_startup_one_get_parse_order_values_and_disabled_no_capture(enabled, source):
  calls, raw, parsed = [], b"exact-original-private", object()
  o = observer() if enabled else None
  target = SimpleNamespace(observer=o, params=SimpleNamespace(get=lambda key, **kw: (calls.append(("get", key, kw)), raw)[1]))
  def parse(value, decoder):
    assert value is raw
    calls.append(("parse", decoder))
    return parsed
  path, cls, field, key, decoder = (("openpilot/selfdrive/controls/controlsd.py", "Controls", "CP", "CarParams", "CP") if source == "cp" else
    ("openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py", "ControlsExt", "CP_SP", "CarParamsSP", "SP"))
  run = startup_fragment(path, cls, {field}, {"messaging": SimpleNamespace(log_from_bytes=parse),
      "car": SimpleNamespace(CarParams="CP"), "custom": SimpleNamespace(CarParamsSP="SP"), "params": target.params})
  run(target)
  assert getattr(target, field) is parsed and calls == [("get", key, {"block": True}), ("parse", decoder)]
  if enabled:
    assert [r["stage"] for r in o._pending] == ["startupBegin", "startupEnd"]
    assert drain(o)[-1]["startup_reference"]["status"] == "ready"


@pytest.mark.parametrize("failure", ["read", "parse"])
def test_original_startup_exception_is_preserved_and_failure_reference_unresolved(failure):
  exception, calls = RuntimeError("private original error"), []
  def get(*a, **k):
    calls.append("get")
    if failure == "read":
      raise exception
    return b"CP"
  def parse(*a):
    calls.append("parse")
    raise exception
  o = observer()
  target = SimpleNamespace(observer=o, params=SimpleNamespace(get=get))
  run = startup_fragment("openpilot/selfdrive/controls/controlsd.py", "Controls", {"CP"},
      {"messaging": SimpleNamespace(log_from_bytes=parse), "car": SimpleNamespace(CarParams="CP")})
  with pytest.raises(RuntimeError) as captured:
    run(target)
  assert captured.value is exception and calls == (["get"] if failure == "read" else ["get", "parse"])
  reference = drain(o)[-1]["startup_reference"]
  assert reference["status"] == ("readFailed" if failure == "read" else "parseFailed") and reference["sha256"] == b""


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_card_existing_cp_sp_serialize_once_and_independent_brackets(enabled):
  calls, cp, sp = [], b"CP-original", b"SP-original"
  target = SimpleNamespace(observer=observer("card") if enabled else None,
      CP=SimpleNamespace(to_bytes=lambda: (calls.append("cp_bytes"), cp)[1]), CP_SP="SP-input")
  def convert(value):
    assert value == "SP-input"
    calls.append("sp_convert")
    return SimpleNamespace(to_bytes=lambda: (calls.append("sp_bytes"), sp)[1])
  run = startup_fragment("openpilot/selfdrive/car/card.py", "Car", {"cp_bytes", "CP_SP_capnp", "cp_sp_bytes"}, {"convert_to_capnp": convert})
  result = run(target)
  assert result["cp_bytes"] is cp and result["cp_sp_bytes"] is sp and calls == ["cp_bytes", "sp_convert", "sp_bytes"]
  if enabled:
    output = drain(target.observer)
    assert len(output) == 4 and {r["startup_reference"]["source"] for r in output} == {"serializedCarParams", "serializedCarParamsSP"}


def test_startup_extension_hook_is_overlay_registered():
  tree = ast.parse((ROOT/".elantra/sync.py").read_text())
  value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id == "OVERLAY_MODIFIED")
  assert "openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py" in value


def test_startup_tokens_begin_snapshot_and_cross_epoch_or_duplicate_end_are_refused():
  o, other = observer(), observer()
  token = o.begin_startup_reference("paramsCarParams")
  first = dict(o._pending[0]["startup_reference"])
  cross = other.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(cross, b"wrong", "complete")
  assert o._startup_jobs == 0
  o.end_startup_reference(token, b"right", "complete")
  o.end_startup_reference(token, b"duplicate", "complete")
  assert o._startup_jobs <= 1 and len(o._pending) == 2
  assert o._pending[0]["startup_reference"] == first


@pytest.mark.parametrize("ticks", [[100, 99], [True, 101], [100, None]])
def test_startup_bad_bracket_clock_is_unavailable_never_replaced_or_hashed(ticks):
  o = obs.Observer("controls", clock=iter(ticks).__next__, epoch=b"e"*16, start_worker=False)
  token = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(token, b"CP", "complete")
  output = drain(o)
  assert all(r["startup_reference"]["sha256"] == b"" for r in output)
  assert o._startup_jobs == 0 and (o.errors or o.loss)


def test_startup_worker_failure_before_new_enqueue_releases_the_racing_buffer():
  o = observer()
  token = o.begin_startup_reference("paramsCarParams")
  capture = o.capture
  def fail_before_enqueue(stage, **kwargs):
    if stage == "startupEnd":
      o.failed = True
      o._discard_startup_bytes()  # worker failure cleanup runs before the new raw job is appended
    return capture(stage, **kwargs)
  o.capture = fail_before_enqueue
  o.end_startup_reference(token, b"private-racing-buffer", "complete")
  assert o._startup_reserved_bytes == o._startup_released_bytes
  assert all("raw_bytes" not in r["startup_reference"] for r in o._pending)


def test_startup_worker_import_failure_releases_both_queued_jobs(monkeypatch):
  import builtins
  o = observer()
  for source in ("paramsCarParams", "paramsCarParamsSP"):
    token = o.begin_startup_reference(source)
    o.end_startup_reference(token, b"x"*65536, "complete")
  original_import = builtins.__import__
  def fail_worker_import(name, *args, **kwargs):
    if name == "openpilot.cereal":
      raise ImportError("fixture stops BEFORE actual publisher/import/runtime")
    return original_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", fail_worker_import)
  o._worker()
  assert o.failed and o._startup_reserved_bytes == o._startup_released_bytes == 131072
  assert all("raw_bytes" not in r["startup_reference"] for r in o._pending)


def test_startup_digest_ready_clock_failure_is_clock_unavailable_not_digest_failure():
  o = obs.Observer("controls", clock=iter([100, 200]).__next__, epoch=b"e"*16, start_worker=False)
  token = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(token, b"CP", "complete")
  end = drain(o)[-1]["startup_reference"]
  assert end["status"] == "clockInvalid" and end["sha256"] == b"" and end["digest_ready_ns"] == 0
  assert o._startup_reserved_bytes == o._startup_released_bytes == 2


@pytest.mark.parametrize("begin_ns", [200, 100])
def test_startup_begin_clock_tie_or_regression_cannot_become_ready_at_valid_end(begin_ns):
  o = obs.Observer("controls", clock=iter([200, begin_ns, 300, 400]).__next__, epoch=b"e"*16, start_worker=False)
  o.capture("initializeBegin")
  token = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(token, b"CP", "complete")
  references = [r["startup_reference"] for r in drain(o) if "startup_reference" in r]
  assert references[0]["begin_status"] == references[-1]["begin_status"] == "invalid"
  assert references[-1]["status"] == "clockInvalid" and references[-1]["sha256"] == b""
  assert o._startup_jobs == 0


def test_startup_maximum_bytes_empty_known_count_and_worker_publication_clock(isolated_schema):
  o = observer()
  cp = o.begin_startup_reference("paramsCarParams")
  o.end_startup_reference(cp, b"x"*65536, "complete")
  sp = o.begin_startup_reference("paramsCarParamsSP")
  o.end_startup_reference(sp, b"", "complete")
  output = drain(o)
  assert o._startup_reserved_bytes == o._startup_released_bytes == 65536
  empty = output[-1]["startup_reference"]
  assert empty["byte_count_known"] and empty["byte_count"] == 0 and len(empty["sha256"]) == 32
  end = output[1]["startup_reference"]
  assert end["begin_status"] == end["end_status"] == "available"
  raw = obs.encode_record(output[1], isolated_schema, publisher_ns=end["end_ns"])
  with isolated_schema.Event.from_bytes(raw) as event:
    ref = event.controlsObservation.observation.startupReference
    assert ref.status == "clockInvalid" and ref.sha256 == b"" and not event.valid
    assert ref.wire == "capnpUnpackedMessage"


@pytest.mark.parametrize("failure", ["cp", "sp"])
def test_actual_card_serialize_exception_identity_preserved(failure):
  exception, calls = RuntimeError("private serializer error"), []
  def cp_bytes():
    calls.append("cp_bytes")
    if failure == "cp":
      raise exception
    return b"CP"
  def convert(value):
    calls.append("sp_convert")
    raise exception
  target = SimpleNamespace(observer=observer("card"), CP=SimpleNamespace(to_bytes=cp_bytes), CP_SP="SP")
  run = startup_fragment("openpilot/selfdrive/car/card.py", "Car", {"cp_bytes", "CP_SP_capnp", "cp_sp_bytes"}, {"convert_to_capnp": convert})
  with pytest.raises(RuntimeError) as captured:
    run(target)
  assert captured.value is exception and calls == (["cp_bytes"] if failure == "cp" else ["cp_bytes", "sp_convert"])
  end = drain(target.observer)[-1]["startup_reference"]
  assert end["status"] == "serializationFailed" and end["sha256"] == b""
