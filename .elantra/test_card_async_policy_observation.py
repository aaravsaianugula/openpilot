"""Public scalar/source fixtures only; no card, Params or native runtime imports."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_controller_observation import method, obs, observer, records, sm

pytest_plugins = ("test_controller_observation",)
ROOT = Path(__file__).resolve().parents[1]
CARD = "openpilot/selfdrive/car/card.py"


def rows(o):
  return [r["card_policy"] for r in records(o) if "card_policy" in r]


def refresh(cap, values=(False, True, False), cp=False):
  with cap.refresh():
    for site, value in zip(("metricRefresh", "experimentalRefresh", "dynamicRefresh"), values, strict=True):
      with cap.read(site) as read:
        effective = read.returned(value)
        if site == "experimentalRefresh":
          effective = effective and read.cp_operand(cp)
        read.assigned(effective)


def test_read_return_assignment_and_use_are_separate_no_origin_inference():
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  refresh(cap)
  assert not records(o) and len(cap.pending) == 4
  cap.drain()
  assert len(cap.pending) == len(rows(o)) == 2
  row = rows(o)[1]
  assert row["returned_known"] and row["returned_value"] is True
  assert row["cp_known"] and row["cp_value"] is False and row["effective_value"] is False
  assert row["begin_ns"] < row["end_ns"] < row["assignment_ns"]
  before = len(rows(o))
  assert cap.use("metricSpeedLimitUse").returned(True) is True
  assert cap.use("metricCruiseUse").returned(False) is False
  assert [r["offered_value"] for r in rows(o)[before:]] == [True, False]
  assert all(not r["origin_read_known"] and r["origin_read_id"] == 0 for r in rows(o))


def test_private_pending_overflow_prefix_and_native_exception_identity():
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  for _ in range(4):
    refresh(cap)
  before = tuple(cap.pending)
  refresh(cap)
  assert len(cap.pending) == 16 and tuple(cap.pending) == before and cap.loss and cap.dropped == 4
  failure = KeyboardInterrupt("PRIVATE_DO_NOT_EXPORT")
  with pytest.raises(KeyboardInterrupt) as caught:
    with cap.refresh():
      with cap.read("metricRefresh"):
        raise failure
  assert caught.value is failure and "PRIVATE" not in repr(cap.pending)
  o.begin_loop()
  cap.drain()
  assert len(cap.pending) == 14 and not any(r["complete"] for r in rows(o))


class Params:
  def __init__(self, values):
    self.values, self.calls = iter(values), []

  def get_bool(self, key):
    self.calls.append(key)
    value = next(self.values)
    if isinstance(value, BaseException):
      raise value
    return value


@pytest.mark.parametrize("enabled,raw", [(False, False), (True, False), (True, True)])
def test_actual_async_source_short_circuit_once_getters_and_native_order(enabled, raw):
  o = observer("card") if enabled else None
  if o is not None:
    o.card_async.transfer()
  calls = []

  class CP:
    @property
    def openpilotLongitudinalControl(self):
      calls.append("cpRead")
      return False

  target = SimpleNamespace(observer=o, params=Params([True, raw, False]), CP=CP(),
                           v_cruise_helper=SimpleNamespace(read_custom_set_speed_params=lambda: calls.append("nested")))
  event = SimpleNamespace(is_set=iter((False, True)).__next__)
  namespace = {"time": SimpleNamespace(sleep=lambda value: calls.append(("sleep", value)))}
  method(CARD, "Car", "params_thread", namespace)(target, event)
  assert target.params.calls == ["IsMetric", "ExperimentalMode", "DynamicExperimentalControl"]
  assert target.is_metric is True and target.experimental_mode is False and target.dynamic_experimental_control is False
  assert calls == (["cpRead"] if raw else []) + ["nested", ("sleep", .1)]
  if enabled:
    assert len(o.card_async.pending) == 4 and not records(o)


def test_disabled_card_native_AST_matches_portable_frozen_original():
  class Disabled(ast.NodeTransformer):
    def visit_If(self, node):
      if ast.unparse(node.test) == "self.observer is None" and "self.observer.card_async" in ast.unparse(node):
        return [self.visit(n) for n in node.body]
      node = self.generic_visit(node)
      return node if node.body else None

    def visit_Expr(self, node):
      if "self.observer.card_async" in ast.unparse(node):
        return None
      return self.generic_visit(node)

  tree = Disabled().visit(ast.parse((ROOT / CARD).read_text()))
  assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == (
    "3197c8ab9971c0558aaf1bc0b8118f3c1826d404f327f3c5fe15c9dc5739d329")


def test_misused_read_site_and_cycle_entries_cannot_grow_or_complete():
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  with cap.refresh():
    for _ in range(25):
      with cap.read("metricCruiseUse") as read:
        read.returned(True)
        read.assigned(True)
      assert len(cap.active_cycle) <= 3
  assert cap.loss and len(cap.pending) <= 16
  assert all(dict(row)["site"] == "none" for row in cap.pending)


def test_duplicate_initial_and_counter_exhaustion_never_complete_or_alias():
  o = observer("card")
  cap = o.card_async
  for site in ("dynamicInitial", "metricInitial", "experimentalInitial", "metricInitial"):
    with cap.read(site) as read:
      read.returned(False)
      read.assigned(False)
  cap.initial_complete()
  assert cap.loss and not dict(cap.pending[-1])["complete"]
  cap.pending.clear()  # fixture only, no concurrent producer
  cap.transfer()
  cap.read_sequence = 2**64-1
  refresh(cap)
  assert len(cap.pending) == 1 and dict(cap.pending[0])["kind"] == "directSubsetComplete"
  assert not dict(cap.pending[0])["complete"]
  other = observer("card").card_async
  other.returned = 2**64-1
  with other.read("metricInitial") as read:
    read.returned(False)
    read.assigned(False)
  assert other.loss


def test_duplicate_CP_operand_keeps_first_observation_and_loss():
  cap = observer("card").card_async
  cap.transfer()
  with cap.read("experimentalRefresh") as read:
    read.returned(True)
    assert read.cp_operand(False) is False
    assert read.cp_operand(True) is True
    read.assigned(False)
  assert cap.loss and dict(cap.pending[0])["cp_value"] is False


def test_observed_error_after_loss_sample_cannot_leave_completion_valid():
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  refresh(cap)
  payload = dict(cap.pending[-1])
  assert payload["valid"] and payload["complete"]

  class Interposed(type(cap)):
    def __getattribute__(self, name):
      if name == "errors":
        object.__setattr__(self, "loss", True)
        object.__setattr__(self, "errors", 1)
      return super().__getattribute__(name)

  cap.__class__ = Interposed  # fixture: error lands AFTER emit samples loss but BEFORE errors
  cap.emit(payload)
  assert rows(o)[-1]["errors"] == 1
  assert not rows(o)[-1]["valid"] and not rows(o)[-1]["complete"]


@pytest.mark.parametrize("clock", [iter((100, 100, 200)).__next__, iter((100, 99, 200)).__next__, lambda: -1])
def test_clock_invalidity_sticky_and_uint64_preserved(clock):
  cap = observer("card").card_async
  cap.clock = clock
  with cap.read("metricInitial") as read:
    read.returned(False)
    read.assigned(False)
  assert cap.loss and not dict(cap.pending[0])["valid"]
  cap = observer("card").card_async
  cap.clock = iter((2**53+1, 2**53+2, 2**53+3)).__next__
  with cap.read("metricInitial") as read:
    read.returned(False)
    read.assigned(False)
  assert dict(cap.pending[0])["begin_ns"] == 2**53+1


def test_unknown_private_values_not_copied_coerced_or_rendered_and_failures_remain_original():
  class Forbidden(BaseException):
    pass

  class Private:
    def __bool__(self):
      raise Forbidden("observer coerced private value")

    def __repr__(self):
      raise Forbidden("observer rendered private value")

  cap = observer("card").card_async
  private = Private()
  with cap.read(private) as read:
    assert read.returned(private) is private
    read.assigned(private)
  assert cap.loss and "Private" not in repr(cap.pending)
  o = observer("card")
  failure = RuntimeError("PRIVATE_NATIVE_EXCEPTION")
  o.card_async.transfer()
  target = SimpleNamespace(observer=o, params=Params([True, failure]), CP=SimpleNamespace(openpilotLongitudinalControl=True))
  target.v_cruise_helper = SimpleNamespace(read_custom_set_speed_params=lambda: pytest.fail("ran after getter failure"))
  with pytest.raises(RuntimeError) as actual:
    method(CARD, "Car", "params_thread", {"time": None})(target, SimpleNamespace(is_set=lambda: False))
  assert actual.value is failure and target.is_metric is True
  assert not hasattr(target, "experimental_mode") and len(o.card_async.pending) == 3
  assert "PRIVATE_NATIVE_EXCEPTION" not in repr(o.card_async.pending)


def test_main_queue_failure_and_worker_failure_keep_source_thread_independent():
  o = observer("card", capacity=1)
  cap = o.card_async
  cap.transfer()
  refresh(cap)
  assert o.sequence == o.dropped == 0 and not records(o)
  o.begin_loop()
  cap.drain()
  assert cap.main_loss and cap.main_dropped == 2 and len(cap.pending) == 2
  assert o.flush_one(lambda *args: (_ for _ in ()).throw(RuntimeError("private"))) is False
  cap.drain()
  assert o.failed and all(not r["capture_complete"] for r in records(o))


def test_schema_roundtrip_privacy_unknown_links_and_wire_bound(isolated_schema, record_property):
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  refresh(cap)
  cap.drain()
  cap.use("experimentalInitializeUse").returned(False)
  sizes = []
  for record in records(o):
    raw = obs.encode_record(record, isolated_schema, publisher_ns=9_000_000)
    sizes.append(len(raw))
    assert len(raw) <= 1024 and record["capture_bytes"] == 1024 and not record["refs"]
    with isolated_schema.Event.from_bytes(raw) as event:
      r = event.cardObservation.observation
      p = r.cardAsyncPolicyObservation
      assert p.present and p.registryType == "bool" and not p.originReadKnown and p.originReadId == 0
      assert p.rawPresence == p.defaultUsed == "unknown"
      assert not any((p.policyComplete, p.sourceAuthenticated, p.predictiveEligible, r.policyComplete, r.sourceAuthenticated))
      assert r.processEpoch == b"e"*16 and p.consumerId == 1
      if p.kind == "use":
        assert p.offeredKnown and not p.offeredValue and p.method == "none"
  record_property("max_new_policy_Event_bytes", max(sizes))
  off = observer("card")
  off.capture("updateBegin")
  off_raw = obs.encode_record(records(off)[0], isolated_schema, publisher_ns=9_000_000)
  record_property("new_schema_unused_policy_Event_bytes", len(off_raw))


def test_max_fixed_policy_and_all_bounded_faults_fit_one_kib(isolated_schema, record_property):
  o = observer("card")
  with o.card_async.read("metricInitial") as read:
    read.returned(False)
    read.assigned(False)
  o.card_async.drain()
  record = records(o)[0]
  # Structural maximum: policy records forbid refs/TX/raw buffers; the only
  # variable list is bounded by the fourteen declared unique fault enums.
  record["faults"] = ("unknownField", "nonfinite", "sourceInvalid", "futureReference", "timestampTie", "timestampRegression",
                      "clockTie", "clockRegression", "txOversize", "eventOversize", "publisherFailed", "unseenSource",
                      "priorLoss", "identityUnchecked")
  raw = obs.encode_record(record, isolated_schema, publisher_ns=9_000_000)
  assert len(raw) <= 1024
  with isolated_schema.Event.from_bytes(raw) as event:
    row = event.cardObservation.observation
    assert len(row.faults) == 14 and len(row.references) == len(row.tx) == 0
    assert len(row.cardAsyncPolicyObservation.to_dict()) == 42
    assert row.cardAsyncPolicyObservation.present and not event.valid
  record_property("max_fixed_policy_all_faults_Event_bytes", len(raw))


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_constructor_three_reads_distinct_ungated_and_one_car_consumer(enabled):
  tree = ast.parse((ROOT / CARD).read_text())
  ctor = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Car"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
  blocks = [n for n in ctor.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.observer is None"
            and any(site in ast.unparse(n) for site in ("dynamicInitial", "metricInitial"))]
  assert len(blocks) == 2
  fn = ast.parse("def initial(self):\n  pass").body[0]
  fn.body = blocks
  namespace = {}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "card-initial-fragment", "exec"), namespace)
  o = observer("card") if enabled else None
  target = SimpleNamespace(observer=o, params=Params([False, True, True]))
  namespace["initial"](target)
  assert target.params.calls == ["DynamicExperimentalControl", "IsMetric", "ExperimentalMode"]
  assert target.experimental_mode is True and not target.dynamic_experimental_control and target.is_metric is True
  if enabled:
    assert len(o.card_async.pending) == 4 and dict(o.card_async.pending[-1])["complete"]
    assert all(dict(row)["consumer_id"] == 1 and dict(row)["cycle_id"] == 0 for row in o.card_async.pending)


def test_actual_argument_reads_are_separate_and_never_associated_by_equal_value():
  tree = ast.parse((ROOT / CARD).read_text())
  update = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Car"
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "state_update")
  blocks = [n for n in update.body if isinstance(n, ast.If) and
            ("update_speed_limit_assist(" in ast.unparse(n) or "initialize_v_cruise(" in ast.unparse(n))]
  assert len(blocks) == 2
  fn = ast.parse("def used(self, CS):\n  pass").body[0]
  fn.body = blocks
  namespace = {}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "card-use-fragment", "exec"), namespace)
  calls = []
  class Target:
    observer = observer("card")
    sm = {"longitudinalPlanSP": "LP", "carControl": SimpleNamespace(enabled=True)}
    CC_prev, CS_prev = SimpleNamespace(enabled=False), "OLD_CS"
    experimental_mode, dynamic_experimental_control = True, False
    metrics = iter((True, False))

    @property
    def is_metric(self):
      value = next(self.metrics)
      calls.append(("attribute", value))
      return value

    v_cruise_helper = SimpleNamespace(update_speed_limit_assist=lambda *a: calls.append(("sla", a)),
      update_v_cruise=lambda *a: calls.append(("cruise", a)), initialize_v_cruise=lambda *a: calls.append(("initialize", a)))

  target = Target()
  namespace["used"](target, "CS")
  assert calls == [("attribute", True), ("sla", (True, "LP")), ("attribute", False), ("cruise", ("CS", True, False)),
                   ("initialize", ("OLD_CS", True, False))]
  assert [r["offered_value"] for r in rows(target.observer)] == [True, False, True, False]
  assert not any(r["origin_read_known"] for r in rows(target.observer))


def test_actual_CP_read_exception_preserves_returned_getter_and_unknown_assignment():
  failure = RuntimeError("private CP operand")
  class CP:
    @property
    def openpilotLongitudinalControl(self):
      raise failure

  o = observer("card")
  o.card_async.transfer()
  target = SimpleNamespace(observer=o, params=Params([True, True]), CP=CP())
  with pytest.raises(RuntimeError) as actual:
    method(CARD, "Car", "params_thread", {"time": None})(target, SimpleNamespace(is_set=lambda: False))
  assert actual.value is failure and target.params.calls == ["IsMetric", "ExperimentalMode"]
  assert not hasattr(target, "experimental_mode")
  row = dict(o.card_async.pending[1])
  assert row["getter_outcome"] == "returned" and row["returned_known"] and row["returned_value"]
  assert row["operation_raised"] and not row["effective_known"] and not row["cp_known"]
  assert not dict(o.card_async.pending[-1])["complete"]


def test_unknown_payload_refuses_before_private_key_hash_or_copy():
  o = observer("card")
  cap = o.card_async
  cap.transfer()
  refresh(cap)
  payload = dict(cap.pending[0], _queued=False)
  payload["origin_read_known"] = True
  o.capture("cardAsyncPolicy", _card_policy=payload)
  assert o.errors == 1 and not records(o)
  payload = dict(cap.pending[0], _queued=False)
  payload["unknown"] = object()
  o.capture("cardAsyncPolicy", _card_policy=payload)
  assert o.errors == 2 and not records(o)


@pytest.mark.parametrize("raw_size,accepted", [(7936, True), (8192, False)])
def test_actual_worst_source_path_two_raw_charges_full_new_rows_and_bounds(raw_size, accepted):
  o, sent = observer("card"), []
  source = sm(obs.CARD_INPUTS)
  source.frame = 0
  values = {"carControl": SimpleNamespace(enabled=True), "carControlSP": object(), "onroadEvents": [], "longitudinalPlanSP": object()}

  class SM(SimpleNamespace):
    def __getitem__(self, key):
      return values[key]

    def update(self, timeout):
      pass

    def all_checks(self, services):
      return True

    def all_alive(self, services):
      return True

  source = SM(**vars(source))
  cs = SimpleNamespace(canValid=True)
  rd = SimpleNamespace(errors=SimpleNamespace(to_dict=dict))
  cruise = SimpleNamespace(v_cruise_kph=80., v_cruise_cluster_kph=79., update_speed_limit_assist=lambda *a: None,
                           update_v_cruise=lambda *a: None, initialize_v_cruise=lambda *a: None)
  target = SimpleNamespace(observer=o, sm=source, can_sock=None, can_rcv_cum_timeout_counter=0, CS_prev=object(),
                           CC_prev=SimpleNamespace(enabled=False), rk=SimpleNamespace(remaining=0.), v_cruise_helper=cruise,
                           is_metric=True, experimental_mode=True, dynamic_experimental_control=False,
                           initialized_prev=False, last_actuators_output="PRIOR", CP=SimpleNamespace(passive=False), CP_SP=object(),
                           CP_SP_capnp=object(), can_callbacks=(), params=SimpleNamespace(put_bool=lambda *a: None))
  target.CI = SimpleNamespace(update=lambda can: (cs, object()), init=lambda *a: None,
                              apply=lambda *a: ("NEXT", [(0x340, b"12345678", 0)]*64))
  target.RI = SimpleNamespace(update=lambda can: rd)

  class PM:
    sock = {}
    def send(self, service, event):
      sent.append(service)

  target.pm = obs.PublisherTap(PM(), o)
  def message(service):
    return SimpleNamespace(logMonoTime=900_000, valid=True, carOutput=SimpleNamespace(), carState=SimpleNamespace())
  namespace = {"messaging": SimpleNamespace(drain_sock_raw=lambda *a, **kw: [b"can"], new_message=message),
               "can_capnp_to_list": lambda x: x, "convert_to_capnp": lambda x: x, "convert_carControlSP": lambda x: x,
               "can_list_to_can_capnp": lambda *a, **kw: b"x"*raw_size, "REPLAY": False, "DT_CTRL": .01,
               "time": SimpleNamespace(monotonic=lambda: 2.), "EventName": SimpleNamespace(selfdriveInitializing=1)}
  for name in ("state_update", "state_publish", "controls_update"):
    setattr(target, name, method(CARD, "Car", name, namespace).__get__(target))
  o.card_async.transfer()
  refresh(o.card_async)
  method(CARD, "Car", "step", namespace)(target)
  assert sent == ["carParams", "carOutput", "carState", "radarTracks", "carParamsSP", "carStateSP", "sendcan"]
  assert o.sequence == 28 and len(rows(o)) == 6
  raw_records = [r for r in records(o) if r["service"] == "sendcan"]
  if accepted:
    assert len(records(o)) == 28 and o.dropped == 0 and len(raw_records) == 2
    assert [r["stage"] for r in raw_records] == ["publishBegin", "publishEnd"]
    assert all(r["capture_bytes"] == 256+raw_size for r in raw_records)
  else:
    assert o.dropped == 2 and not raw_records  # independent per-record cap; do not widen it
  assert sum(r["capture_bytes"] for r in records(o)) <= 32768
  assert 22*256+2*8*112+64*24+2*8192+6*1024 == 31488


def test_actual_send_failure_still_attempts_two_raw_captures_preserving_exception():
  o = observer("card")
  failure = RuntimeError("native")
  def broken(*args):
    raise failure
  tap = obs.PublisherTap(SimpleNamespace(sock={}, send=broken), o)
  with pytest.raises(RuntimeError) as actual:
    tap.send("sendcan", b"x"*7936)
  assert actual.value is failure and [r["stage"] for r in records(o)] == ["publishBegin", "publishFailed"]
  assert sum(r["capture_bytes"] for r in records(o)) == 2*8192
