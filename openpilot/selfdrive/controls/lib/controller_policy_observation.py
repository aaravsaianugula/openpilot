"""Fixed controls-only returned-value evidence; never raw Params or policy admission.

Tokens belong to one existing Observer/producer. No publisher, polling, locks or
global sink lives here. Torqued/modeld/card-thread policy remains unobserved.
"""
import math

MAX_CONSUMERS = 64
MAX_GROUP_RECORDS = 16
MAX_POLICY_BYTES = 1024  # TOTAL capture charge and independently checked wire limit
GROUPS = ("controlsInit", "stockConstructor", "v0Constructor", "selection", "overrideRefresh", "controlsRefresh", "learnerWrite")
CONSUMERS = ("controls", "blinker", "stockTorque", "stockExtension", "v0Torque", "v0Extension")
KEYS = ("none", "EnforceTorqueControl", "TorqueControlTune", "LateralJerkTorqueController", "NeuralNetworkLateralControl",
        "TorqueParamsOverrideEnabled", "TorqueParamsOverrideLatAccelFactor", "TorqueParamsOverrideFriction", "LagdToggle",
        "LagdValueCache", "BlinkerPauseLateralControl", "IsMetric", "BlinkerMinLateralControlSpeed", "BlinkerLateralReengageDelay",
        "CustomTorqueParams", "LiveTorqueParamsToggle", "LiveTorqueParamsRelaxedToggle")
SITES = {
  "selectionEnforce": ("EnforceTorqueControl", "getBool", "bool", False, ("controls",)),
  "selectionTune": ("TorqueControlTune", "get", "float", False, ("controls",)),
  "jerkEnable": ("LateralJerkTorqueController", "getBool", "bool", False, ("stockExtension", "v0Extension")),
  "nnEnable": ("NeuralNetworkLateralControl", "getBool", "bool", False, ("stockExtension", "v0Extension")),
  "overrideEnforce": ("EnforceTorqueControl", "getBool", "bool", False, ("stockExtension", "v0Extension")),
  "overrideInitial": ("TorqueParamsOverrideEnabled", "getBool", "bool", False, ("stockExtension", "v0Extension")),
  "overrideRefresh": ("TorqueParamsOverrideEnabled", "getBool", "bool", False, ("stockExtension", "v0Extension")),
  "overrideFactor": ("TorqueParamsOverrideLatAccelFactor", "get", "float", True, ("stockExtension", "v0Extension")),
  "overrideFriction": ("TorqueParamsOverrideFriction", "get", "float", True, ("stockExtension", "v0Extension")),
  "delayInitial": ("LagdValueCache", "get", "float", True, ("controls",)),
  "lagdToggle": ("LagdToggle", "getBool", "bool", False, ("controls",)),
  "delayCache": ("LagdValueCache", "get", "float", True, ("controls",)),
  "blinkerInitial": ("BlinkerPauseLateralControl", "getBool", "int", False, ("blinker",)),
  "metricInitial": ("IsMetric", "getBool", "bool", False, ("blinker",)),
  "blinkerRefresh": ("BlinkerPauseLateralControl", "getBool", "int", False, ("blinker",)),
  "metricRefresh": ("IsMetric", "getBool", "bool", False, ("blinker",)),
  "blinkerMinSpeed": ("BlinkerMinLateralControlSpeed", "get", "int", True, ("blinker",)),
  "blinkerReengage": ("BlinkerLateralReengageDelay", "get", "int", True, ("blinker",)),
}


def _uint(value, maximum=2**64-1):
  return type(value) is int and 0 <= value <= maximum


def _scalar(value):
  if value is None:
    return {"scalar_kind": "none"}
  if type(value) is bool:
    return {"scalar_kind": "bool", "bool_value": value}
  if type(value) is int and -2**63 <= value < 2**63:
    return {"scalar_kind": "int", "int_value": value}
  if type(value) is float and math.isfinite(value):
    return {"scalar_kind": "float", "float_value": value}
  return {"scalar_kind": "unknown", "valid": False}  # no str/repr/coercion of private values


DEFAULTS = {
  "kind": "consumerCreated", "group": "controlsInit", "group_id": 0, "consumer_id": 0, "parent_id": 0,
  "consumer_kind": "controls", "site": "none", "key": "none", "method": "none", "registry_type": "unknown",
  "default_requested": False, "raw_presence": "unknown", "default_used": "unknown", "read_sequence": 0,
  "scalar_kind": "unknown", "bool_value": False, "int_value": 0, "float_value": 0.,
  "begin_ns": 0, "end_ns": 0, "begin_known": False, "valid": True,
  "write_source": "none", "factor": 0., "offset": 0., "friction": 0., "source_ns": 0, "source_frame": 0,
  "selected_id": 0, "selected_known": False, "selected_class": "unknown", "selection_reason": "notInstrumented",
  "attempts": 0, "returned": 0, "raised": 0, "accepted": 0, "dropped": 0, "complete": False,
}
ENUMS = {
  "kind": ("consumerCreated", "readReturned", "readRaised", "controllerSelected", "torqueParametersWritten", "delayCacheWritten", "groupComplete"),
  "group": GROUPS, "consumer_kind": CONSUMERS, "site": ("none", *SITES), "key": KEYS,
  "method": ("none", "get", "getBool"), "registry_type": ("unknown", "bool", "int", "float"),
  "raw_presence": ("unknown",), "default_used": ("unknown",), "scalar_kind": ("unknown", "none", "bool", "int", "float"),
  "write_source": ("none", "constructor", "learner", "override", "cachedDelay", "publishedDelay"),
  "selected_class": ("unknown", "angle", "curvature", "pid", "stockTorque", "v0Torque"),
  "selection_reason": ("observed", "nonTorqueNotInstrumented", "notInstrumented"),
}


def snapshot(payload, now, faults):
  """Validate BEFORE any copy. Worker receives fixed scalars only, never user text."""
  if type(payload) is not dict or len(payload) > len(DEFAULTS) + 1:
    raise ValueError("unknown policy payload")
  for key, value in payload.items():
    if type(key) is not str or key not in (*DEFAULTS, "_queued"):
      raise ValueError("unknown policy field")
    if key == "_queued":
      continue
    default = DEFAULTS[key]
    if key in ENUMS:
      if type(value) is not str or value not in ENUMS[key]:
        raise ValueError("unknown policy enum")
    elif type(default) is bool:
      if type(value) is not bool:
        raise ValueError("unknown policy bool")
    elif type(default) is float:
      if type(value) is not float or not math.isfinite(value):
        raise ValueError("unknown policy scalar")
    elif key == "int_value":
      if type(value) is not int or not -2**63 <= value < 2**63:
        raise ValueError("unknown policy integer")
    elif not _uint(value, 2**32-1 if key not in ("begin_ns", "end_ns", "source_ns", "source_frame") else 2**64-1):
      raise ValueError("unknown policy identity")
  result = dict(DEFAULTS)
  result.update((k, v) for k, v in payload.items() if k != "_queued")
  result["end_ns"] = now
  if result["kind"] in ("readReturned", "readRaised"):
    result["valid"] &= result["begin_known"] and result["begin_ns"] < now
  if faults:
    result["valid"] = result["complete"] = False
  if result["source_ns"] > now:
    result["valid"] = False
  return result


class PolicyCapture:
  def __init__(self, observer):
    self.observer = observer
    self.next_consumer = self.next_group = 0
    self.stack = []  # at most two actual nested constructor groups, hard cap four

  def error(self):
    self.observer.errors += 1
    self.observer.dropped += 1
    self.observer.loss = True

  def group(self, kind):
    return _Group(self, kind)

  def emit(self, consumer, **fields):
    group = self.stack[-1] if self.stack else None
    if group is None or not group.active or group.count >= MAX_GROUP_RECORDS:
      self.error()
      return False
    group.count += 1
    payload = {"group": group.kind, "group_id": group.id, "consumer_id": consumer.id if consumer else 0,
               "parent_id": consumer.parent if consumer else 0, "consumer_kind": consumer.kind if consumer else "controls",
               **fields, "_queued": False}
    self.observer.capture("policy", _policy=payload)
    accepted = payload["_queued"]
    group.accepted += accepted
    group.invalid |= not accepted or not payload.get("valid", True)
    if not accepted:
      self.observer.loss = True  # later groups must not repair an earlier missing record
    return accepted


class _Group:
  def __init__(self, manager, kind):
    self.manager, self.kind = manager, kind
    self.id = self.count = self.attempts = self.returned = self.raised = self.accepted = self.open_reads = 0
    self.active = self.invalid = False

  def __enter__(self):
    m = self.manager
    if self.kind not in GROUPS or m.next_group >= 2**32-1 or len(m.stack) >= 4:
      m.error()
      return self
    m.next_group += 1
    self.id, self.active = m.next_group, True
    self.initial_drops, self.initial_errors = m.observer.dropped, m.observer.errors
    m.stack.append(self)
    return self

  def __exit__(self, exc_type, exc, tb):
    m = self.manager
    if self.active:
      complete = not (exc_type or self.invalid or self.open_reads or m.observer.failed or m.observer.loss
                      or m.observer.dropped != self.initial_drops or m.observer.errors != self.initial_errors)
      m.emit(None, kind="groupComplete", attempts=self.attempts, returned=self.returned, raised=self.raised,
             accepted=self.accepted, dropped=m.observer.dropped-self.initial_drops, complete=complete)
      m.stack.pop()
      self.active = False
    return False  # original getter/control exception always propagates

  def consumer(self, kind, parent=None):
    m = self.manager
    if kind not in CONSUMERS or m.next_consumer >= MAX_CONSUMERS or not self.active:
      m.error()
      return Consumer(m, 0, "controls", 0)
    m.next_consumer += 1
    consumer = Consumer(m, m.next_consumer, kind, parent.id if isinstance(parent, Consumer) and parent.manager is m else 0)
    m.emit(consumer, kind="consumerCreated")
    return consumer


class Consumer:
  def __init__(self, manager, identity, kind, parent):
    self.manager, self.id, self.kind, self.parent = manager, identity, kind, parent
    self.sequence = 0

  def group(self, kind):
    return self.manager.group(kind)

  def child(self, kind):
    if not self.manager.stack:
      self.manager.error()
      return Consumer(self.manager, 0, "controls", 0)
    return self.manager.stack[-1].consumer(kind, self)

  def read(self, site):
    return _Read(self, site)

  def selected(self, controller, kind):
    token = getattr(controller, "_policy_capture", None)
    known = isinstance(token, Consumer) and token.manager is self.manager and token.id != 0 and token.kind in ("stockTorque", "v0Torque")
    self.manager.emit(self, kind="controllerSelected", selected_known=known, selected_id=token.id if known else 0,
                      selected_class=token.kind if known else kind if kind in ("angle", "curvature", "pid") else "unknown",
                      selection_reason="observed" if known else "nonTorqueNotInstrumented" if kind in ("angle", "curvature", "pid") else "notInstrumented")

  def torque_written(self, source, params, *, source_ns=0, source_frame=0):
    try:
      values = (params.latAccelFactor, params.latAccelOffset, params.friction)
      if any(type(v) is not float or not math.isfinite(v) for v in values):
        raise ValueError("unknown effective torque")
      self.manager.emit(self, kind="torqueParametersWritten", write_source=source,
                        factor=values[0], offset=values[1], friction=values[2], source_ns=source_ns, source_frame=source_frame)
    except Exception:
      self.manager.error()

  def delay_written(self, value, source):
    self.manager.emit(self, kind="delayCacheWritten", write_source=source, **_scalar(value))


class _Read:
  def __init__(self, consumer, site):
    self.consumer, self.site, self.group, self.begin = consumer, site, None, None
    self.value, self.has_return, self.closed = None, False, False

  def __enter__(self):
    c, m = self.consumer, self.consumer.manager
    try:
      if self.site not in SITES or c.kind not in SITES[self.site][4] or not c.id or c.sequence >= 2**32-1 or not m.stack:
        raise ValueError("unknown policy read")
      self.group = m.stack[-1]
      self.group.attempts += 1
      self.group.open_reads += 1
      c.sequence += 1
      self.begin = m.observer.clock()
      if not _uint(self.begin) or self.begin <= m.observer._last_clock:
        self.group.invalid = True
        self.begin = None
    except Exception:
      m.error()
    return self

  def returned(self, value):
    # Never retain a raw/private/custom result. Scalar validation does not coerce it.
    if self.has_return or self.closed:
      self.consumer.manager.error()
      return
    self.value = _scalar(value)
    self.has_return = True

  def __exit__(self, exc_type, exc, tb):
    c, m = self.consumer, self.consumer.manager
    if self.closed:
      m.error()
      return False
    self.closed = True
    if self.group is not None:
      self.group.open_reads -= 1
      raised = exc_type is not None
      self.group.raised += raised
      self.group.returned += not raised and self.has_return
      if not raised and not self.has_return:
        self.group.invalid = True
      key, method, registry, default, _ = SITES[self.site]
      fields = self.value if self.has_return and not raised else {}
      fields = {**fields, "valid": fields.get("valid", True) and self.begin is not None and (raised or self.has_return)}
      m.emit(c, kind="readRaised" if raised else "readReturned", site=self.site, key=key, method=method,
             registry_type=registry, default_requested=default, read_sequence=c.sequence,
             begin_ns=self.begin if self.begin is not None else 0, begin_known=self.begin is not None, **fields)
    self.value = None
    return False


def encode_policy(target, source):
  """Worker-only fixed schema; no dynamic/raw field names exported."""
  target.present = True
  for key in DEFAULTS:
    camel = key.split('_')[0] + ''.join(word.title() for word in key.split('_')[1:])
    value = source[key]
    if key == "key":
      value = value[0].lower() + value[1:]
    setattr(target, camel, value)
