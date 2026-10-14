"""Three direct card fields only; no read-to-use, atomic config or policy admission.

Constructor then params_thread is the sole private deque appender; card main is
the sole poplefter and existing Observer producer. Immutable scalar rows cross
that boundary. This prospective CPython SPSC protocol is not a runtime/timing
proof. No callback waits, retries, polls Params, creates a worker or holds locks.
"""
from collections import deque

MAX_PENDING = 16
MAX_DRAIN = 2
MAX_CARD_POLICY_BYTES = 1024  # TOTAL capture charge and worker wire bound
U64 = 2**64 - 1
SITES = {
  "dynamicInitial": "dynamicExperimentalControl", "metricInitial": "isMetric", "experimentalInitial": "experimentalMode",
  "metricRefresh": "isMetric", "experimentalRefresh": "experimentalMode", "dynamicRefresh": "dynamicExperimentalControl",
  "metricSpeedLimitUse": "isMetric", "metricCruiseUse": "isMetric",
  "experimentalInitializeUse": "experimentalMode", "dynamicInitializeUse": "dynamicExperimentalControl",
}
DEFAULTS = {
  "kind": "read", "site": "none", "key": "none", "consumer_id": 1, "read_sequence": 0, "cycle_id": 0,
  "begin_ns": 0, "end_ns": 0, "assignment_ns": 0, "begin_status": "unavailable", "end_status": "unavailable",
  "assignment_status": "unavailable", "getter_outcome": "unknown", "operation_raised": False,
  "returned_known": False, "returned_value": False, "effective_known": False, "effective_value": False,
  "cp_conditional": False, "cp_known": False, "cp_value": False, "offered_known": False, "offered_value": False,
  "origin_read_known": False, "origin_read_id": 0, "attempted": 0, "returned": 0, "raised": 0,
  "queued": 0, "dropped": 0, "errors": 0, "loss": False, "valid": False, "complete": False,
}
ENUMS = {"kind": ("read", "directSubsetComplete", "use"), "site": ("none", *SITES),
         "key": ("none", "isMetric", "experimentalMode", "dynamicExperimentalControl"),
         "getter_outcome": ("unknown", "returned", "raised")}
for _name in ("begin_status", "end_status", "assignment_status"):
  ENUMS[_name] = ("unavailable", "known", "tie", "regression")


def _uint(value):
  return type(value) is int and 0 <= value <= U64


def snapshot(payload, now, faults):
  """Validate fixed names/types BEFORE copy; no formatting of unknown inputs."""
  if type(payload) is not dict or len(payload) != len(DEFAULTS) + 1:
    raise ValueError("unknown card policy record")
  for key, value in payload.items():
    if type(key) is not str or key not in (*DEFAULTS, "_queued"):
      raise ValueError("unknown card policy field")
    if key == "_queued":
      if type(value) is not bool:
        raise ValueError("unknown card policy queue flag")
    elif key in ENUMS:
      if type(value) is not str or value not in ENUMS[key]:
        raise ValueError("unknown card policy enum")
    elif type(DEFAULTS[key]) is bool:
      if type(value) is not bool:
        raise ValueError("unknown card policy bool")
    elif not _uint(value):
      raise ValueError("unknown card policy clock/counter")
  if payload["origin_read_known"] or payload["origin_read_id"] != 0 or payload["consumer_id"] != 1:
    raise ValueError("unproved card read-to-use association")
  result = {k: v for k, v in payload.items() if k != "_queued"}
  if not _uint(now) or any(result[key] > now for key in ("begin_ns", "end_ns", "assignment_ns")):
    faults.append("futureReference")
  if result["loss"] or result["dropped"] or result["errors"] or faults:
    result["loss"] = True
    result["valid"] = result["complete"] = False
  return result


def encode(target, row):
  target.present, target.method, target.registryType = True, "getBool" if row["kind"] == "read" else "none", "bool"
  for name, value in row.items():
    field = name.split("_")
    setattr(target, field[0] + "".join(part.capitalize() for part in field[1:]), value)
  target.rawPresence, target.defaultUsed = "unknown", "unknown"
  target.policyComplete, target.sourceAuthenticated, target.predictiveEligible = False, False, False


class CardAsyncPolicyCapture:
  def __init__(self, observer):
    self.observer, self.clock = observer, observer.clock
    self.pending = deque()
    self.transferred = self.loss = self.main_loss = False
    self.read_sequence = self.cycle_id = self.attempted = self.returned = self.raised = self.queued = self.dropped = self.errors = 0
    self.main_dropped = self.read_last = self.use_last = 0
    self.active_cycle = None
    self.initial_sites = []

  def error(self):
    self.loss = True
    self.errors = min(U64, self.errors + 1)

  def bump(self, name):
    value = getattr(self, name)
    if value == U64:
      self.error()
    else:
      setattr(self, name, value + 1)

  def tick(self, use=False):
    try:
      value = self.clock()
      if not _uint(value) or value == 0:
        raise ValueError("unknown clock")
      previous = self.use_last if use else self.read_last
      status = "known" if value > previous else "tie" if value == previous else "regression"
      if use:
        self.use_last = value
        self.main_loss |= status != "known"
      else:
        self.read_last = value
        if status != "known":
          self.error()
      return value, status
    except Exception:
      if use:
        self.main_loss = True
      else:
        self.error()
      return 0, "unavailable"

  def transfer(self):
    # Called before the existing t.start; no clock/read/worker or ownership inference.
    if self.transferred:
      self.error()
    self.transferred = True

  def read(self, site):
    return _Read(self, site)

  def refresh(self):
    return _Refresh(self)

  def initial_complete(self):
    good = self.initial_sites == ["dynamicInitial", "metricInitial", "experimentalInitial"] and not self.transferred
    self.enqueue(dict(DEFAULTS, kind="directSubsetComplete", valid=good, complete=good))

  def enqueue(self, row):
    if len(self.pending) >= MAX_PENDING:
      self.dropped = min(U64, self.dropped + 1)
      self.error()
      return
    self.bump("queued")
    row.update(attempted=self.attempted, returned=self.returned, raised=self.raised, queued=self.queued,
               dropped=self.dropped, errors=self.errors, loss=self.loss)
    if self.loss:
      row["valid"] = row["complete"] = False
    self.pending.append(tuple(row.items()))  # bounded immutable scalar row; never private getter objects

  def emit(self, row):
    row.update(loss=row["loss"] or self.loss or self.main_loss,
               dropped=min(U64, self.dropped + self.main_dropped), errors=self.errors, _queued=False)
    self.observer.capture("cardAsyncPolicy", _card_policy=row)
    if not row["_queued"]:
      self.main_loss = True
      self.main_dropped = min(U64, self.main_dropped + 1)

  def drain(self):
    for _ in range(MAX_DRAIN):
      try:
        row = self.pending.popleft()
      except IndexError:
        break
      self.emit(dict(row))

  def use(self, site):
    return _Use(self, site)


class _Read:
  def __init__(self, capture, site):
    self.capture = capture
    self.row = dict(DEFAULTS)
    self.done = self.has_return = self.has_assignment = False
    allowed = ("metricRefresh", "experimentalRefresh", "dynamicRefresh") if capture.transferred else (
      "dynamicInitial", "metricInitial", "experimentalInitial")
    if type(site) is str and site in allowed:
      self.row.update(site=site, key=SITES[site])
    else:
      capture.error()

  def __enter__(self):
    c = self.capture
    if c.read_sequence == U64:
      c.error()
    else:
      c.read_sequence += 1
      self.row["read_sequence"] = c.read_sequence
    c.bump("attempted")
    self.row["cycle_id"] = c.cycle_id if c.transferred else 0
    self.row["begin_ns"], self.row["begin_status"] = c.tick()
    return self

  def returned(self, value):
    if self.has_return or self.done:
      self.capture.error()
      return value
    self.has_return = True
    self.row.update(getter_outcome="returned", returned_known=type(value) is bool, returned_value=value if type(value) is bool else False)
    self.row["end_ns"], self.row["end_status"] = self.capture.tick()
    if type(value) is not bool:
      self.capture.error()
    return value

  def cp_operand(self, value):
    if self.row["cp_conditional"] or self.done:
      self.capture.error()
      return value
    if self.row["site"] != "experimentalRefresh" or not self.has_return or self.row["returned_value"] is not True:
      self.capture.error()
    self.row.update(cp_conditional=True, cp_known=type(value) is bool, cp_value=value if type(value) is bool else False)
    if type(value) is not bool:
      self.capture.error()
    return value

  def assigned(self, value):
    if self.has_assignment or self.done or not self.has_return:
      self.capture.error()
      return
    self.has_assignment = True
    self.row.update(effective_known=type(value) is bool, effective_value=value if type(value) is bool else False)
    self.row["assignment_ns"], self.row["assignment_status"] = self.capture.tick()
    if type(value) is not bool:
      self.capture.error()

  def __exit__(self, typ, value, traceback):
    c, r = self.capture, self.row
    if self.done:
      c.error()
      return False
    self.done = True
    r["operation_raised"] = typ is not None
    if not self.has_return:
      r["getter_outcome"] = "raised" if typ is not None else "unknown"
      r["end_ns"], r["end_status"] = c.tick()
    if self.has_return:
      c.bump("returned")
    elif typ is not None:
      c.bump("raised")
    r["valid"] = (r["site"] != "none" and r["read_sequence"] > 0 and r["begin_status"] == r["end_status"] == "known"
                  and r["begin_ns"] < r["end_ns"] and (typ is not None or self.has_assignment))
    if self.has_assignment:
      r["valid"] &= r["effective_known"] and r["assignment_status"] == "known" and r["assignment_ns"] > r["end_ns"]
    if not r["valid"]:
      c.error()
    if not c.transferred and r["valid"]:
      if r["site"] in c.initial_sites or len(c.initial_sites) >= 3:
        c.error()
      else:
        c.initial_sites.append(r["site"])
    if c.active_cycle is not None:
      if len(c.active_cycle) >= 3:
        c.error()
      else:
        c.active_cycle.append(r["site"])
    if r["read_sequence"]:
      c.enqueue(r)
    return False  # preserve native BaseException object/order; never hold exception text


class _Refresh:
  def __init__(self, capture):
    self.capture = capture

  def __enter__(self):
    c = self.capture
    if not c.transferred or c.active_cycle is not None or c.cycle_id == U64:
      c.error()
    else:
      c.cycle_id += 1
    c.active_cycle = []
    return self

  def __exit__(self, typ, value, traceback):
    c = self.capture
    good = typ is None and c.active_cycle == ["metricRefresh", "experimentalRefresh", "dynamicRefresh"] and not c.loss
    c.active_cycle = None
    c.enqueue(dict(DEFAULTS, kind="directSubsetComplete", cycle_id=c.cycle_id, valid=good, complete=good))
    return False


class _Use:
  def __init__(self, capture, site):
    self.capture, self.done = capture, False
    self.row = dict(DEFAULTS, kind="use")
    if type(site) is str and site in SITES and site.endswith("Use"):
      self.row.update(site=site, key=SITES[site])
    else:
      capture.main_loss = True
    self.row["begin_ns"], self.row["begin_status"] = capture.tick(use=True)

  def returned(self, value):
    c, r = self.capture, self.row
    if self.done:
      c.main_loss = True
      return value
    self.done = True
    r["end_ns"], r["end_status"] = c.tick(use=True)
    r.update(offered_known=type(value) is bool, offered_value=value if type(value) is bool else False,
             valid=type(value) is bool and r["site"] != "none" and r["begin_status"] == r["end_status"] == "known")
    if not r["valid"]:
      c.main_loss = True
    c.emit(r)
    return value
