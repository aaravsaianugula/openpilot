"""Bounded torqued-only typed-return evidence on the existing publication.

No publisher/thread/Params lives here. Presence is not completeness; local send
return is not delivery. Event-time/monotonic clock equivalence and enabled
overhead remain unproved. Full policy/source/predictive eligibility stays false.
"""
import math
import os
import time

MAX_READS = 8
MAX_INSTANCES = 64
MAX_WIRE_BYTES = 1024
U64 = 2**64-1
SITES = {
  "constructorLive": ("liveTorqueParamsToggle", "live"),
  "constructorCustom": ("customTorqueParams", "custom"),
  "initializeRelaxed": ("liveTorqueParamsRelaxedToggle", "relaxed"),
  "initializeCustom": ("customTorqueParams", "initial_custom"),
  "refreshLive": ("liveTorqueParamsToggle", "live"),
  "refreshCustom": ("customTorqueParams", "custom"),
}


def _uint(value):
  return type(value) is int and 0 <= value <= U64


def _frame(value):
  return type(value) is int and -2**63 <= value < 2**63


class TorquedPolicyCapture:
  """One source thread owns this fixed journal; there is no shared worker queue."""
  def __init__(self, *, clock=time.monotonic_ns, epoch=None):
    self.clock = clock
    self.epoch = os.urandom(16) if epoch is None else epoch
    if type(self.epoch) is not bytes or len(self.epoch) != 16:
      raise ValueError("unknown torqued epoch")
    self.pending = []
    self.attempts = self.dropped = self.errors = self.instances = 0
    self.loss = False
    self.last_clock = None

  def error(self):
    self.errors = min(U64, self.errors+1)
    self.dropped = min(U64, self.dropped+1)
    self.loss = True

  def now(self):
    try:
      value = self.clock()
      if not _uint(value) or (self.last_clock is not None and value <= self.last_clock):
        raise ValueError("unknown observation clock")
      self.last_clock = value
      return value
    except Exception:
      self.error()
      return None

  def instance(self, decimated):
    if self.instances >= MAX_INSTANCES:
      self.error()
      return None
    self.instances += 1
    return _Instance(self, self.instances, decimated)


class _Instance:
  def __init__(self, capture, identity, decimated):
    self.capture, self.id = capture, identity
    self.decimated = decimated if type(decimated) is bool else None
    self.latest = dict.fromkeys(("live", "custom", "initial_custom", "relaxed"), 0)
    self.open_read = self.staged = None
    self.publications = 0
    self.retired_sequence = capture.attempts

  def read(self, site, frame):
    return _Read(self, site, frame)

  def stage(self, msg, estimator):
    """Attach only at main publication, never from native get_msg/cache code."""
    c, header = self.capture, None
    try:
      if self.staged is not None or self.publications >= U64:
        raise ValueError("overlapping publication")
      self.publications += 1
      native = msg.lateralTorqueParameters
      header = native.policyObservation
      header.present = False  # final commit marker, remains absent after partial failure
      rows = [row for row in c.pending if row["instance_id"] == self.id]
      if len(rows) > MAX_READS:
        raise ValueError("unknown pending prefix")
      appended = c.now()
      event_ns = msg.logMonoTime
      frame = getattr(estimator, "frame", None)
      live = getattr(estimator, "use_live_torque_params", None)
      custom = getattr(estimator, "custom_torque_params", None)
      factor = getattr(estimator, "factor_sanity", None)
      friction = getattr(estimator, "friction_sanity", None)
      valid_factor = type(factor) is float and math.isfinite(factor)
      valid_friction = type(friction) is float and math.isfinite(friction)
      if not (_uint(event_ns) and _frame(frame) and type(live) is bool and type(custom) is bool
              and valid_factor and valid_friction and type(native.useParams) is bool and self.decimated is not None):
        c.error()
      header.processEpoch, header.instanceId = c.epoch, self.id
      header.readAttemptSequence, header.publicationAttemptId = c.attempts, self.publications
      header.windowStartSequence = self.retired_sequence
      header.sourceEventNs, header.sourceEventKnown = event_ns if _uint(event_ns) else 0, _uint(event_ns)
      header.appendObservationNs, header.appendKnown = appended if appended is not None else 0, appended is not None
      header.estimatorFrame, header.frameKnown = frame if _frame(frame) else 0, _frame(frame)
      header.publishedUseParams, header.publishedUseKnown = native.useParams if type(native.useParams) is bool else False, type(native.useParams) is bool
      header.cachedLive, header.cachedLiveKnown = live if type(live) is bool else False, type(live) is bool
      header.cachedCustom, header.cachedCustomKnown = custom if type(custom) is bool else False, type(custom) is bool
      header.factorSanity, header.factorSanityKnown = factor if valid_factor else 0., valid_factor
      header.frictionSanity, header.frictionSanityKnown = friction if valid_friction else 0., valid_friction
      header.decimated, header.decimatedKnown = self.decimated if self.decimated is not None else False, self.decimated is not None
      header.cachedLiveReadId, header.cachedCustomReadId = self.latest["live"], self.latest["custom"]
      header.initializeCustomReadId, header.relaxedReadId = self.latest["initial_custom"], self.latest["relaxed"]
      header.rawPresence, header.defaultUsed = "unknown", "unknown"
      header.readClock, header.publicationClock = "pythonMonotonicNs", "eventLogMonoTime"
      header.policyComplete, header.sourceAuthenticated, header.predictiveEligible = False, False, False
      header.bootIdentityKnown, header.defaultOriginKnown, header.clockDomainsVerified = False, False, False
      header.fullUseHistory, header.learnerThresholdsComplete, header.cacheRestoreProven = False, False, False
      target = header.init("reads", len(rows))
      for item, row in zip(target, rows, strict=True):
        item.readId, item.site, item.key = row["id"], row["site"], SITES[row["site"]][0]
        item.estimatorFrame = row["frame"]
        item.beginNs, item.endNs = row["begin_ns"], row["end_ns"]
        item.beginKnown, item.endKnown = row["begin_known"], row["end_known"]
        item.outcome, item.returnedKnown, item.returnedBool = row["outcome"], row["returned_known"], row["returned_bool"]
        item.valid = row["valid"]
      header.dropped, header.errors, header.epochLoss = c.dropped, c.errors, c.loss
      header.readCaptureComplete = not (c.loss or self.open_read is not None or appended is None)
      token = (c.epoch, self.id, self.publications, tuple(row["id"] for row in rows))
      header.present = True
      self.staged = token
      return token
    except Exception:
      c.error()
      if header is not None:
        try:
          header.present = False
        except Exception:
          pass  # unavailable header; original native message/exception is never replaced
      return None

  def retire(self, token):
    if token is None:
      return  # failed attachment is sticky; native send still occurred once
    try:
      if token is not self.staged:
        raise ValueError("unknown staged prefix")
      ids = token[3]
      self.capture.pending = [row for row in self.capture.pending if not (row["instance_id"] == self.id and row["id"] in ids)]
      if ids:
        self.retired_sequence = ids[-1]
      self.staged = None
    except Exception:
      self.capture.error()

  def send_failed(self, token):
    self.capture.error()
    if token is self.staged:
      self.staged = None  # no retirement and no retry; future publication is not guaranteed


class _Read:
  def __init__(self, instance, site, frame):
    self.instance, self.site, self.frame = instance, site, frame
    self.id = 0
    self.begin = None
    self.returned_known = self.returned_bool = self.has_return = self.closed = False

  def __enter__(self):
    i, c = self.instance, self.instance.capture
    if type(self.site) is not str or self.site not in SITES or not _frame(self.frame) or i.open_read is not None or c.attempts >= U64:
      c.error()
      return self
    c.attempts += 1
    self.id = c.attempts
    i.open_read = self
    self.begin = c.now()
    return self

  def returned(self, value):
    if self.has_return or self.closed:
      self.instance.capture.error()
      return
    self.has_return, self.returned_known = True, type(value) is bool
    self.returned_bool = value if self.returned_known else False  # never coerce/store private raw values

  def __exit__(self, exc_type, exc, tb):
    i, c = self.instance, self.instance.capture
    if self.closed or i.open_read is not self:
      c.error()
      return False
    self.closed, i.open_read = True, None
    end = c.now()
    outcome = "raised" if exc_type is not None else "returned" if self.has_return else "unavailable"
    valid = outcome == "returned" and self.returned_known and self.begin is not None and end is not None and self.begin < end
    if not valid:
      c.error()
    if len(c.pending) >= MAX_READS:
      c.error()  # refuse new row BEFORE copying; no eviction or key coalescing
    else:
      c.pending.append({"instance_id": i.id, "id": self.id, "site": self.site, "frame": self.frame,
                        "begin_ns": self.begin if self.begin is not None else 0, "end_ns": end if end is not None else 0,
                        "begin_known": self.begin is not None, "end_known": end is not None, "outcome": outcome,
                        "returned_known": self.returned_known if outcome == "returned" else False,
                        "returned_bool": self.returned_bool if outcome == "returned" else False, "valid": valid})
    if outcome == "returned":
      i.latest[SITES[self.site][1]] = self.id  # local initialization custom is deliberately separate from cache custom
    return False  # preserve original getter/constructor exception identity


def create_torqued_policy_capture():
  if os.environ.get("ELANTRA_TORQUED_POLICY_OBSERVER") != "1":
    return None
  try:
    return TorquedPolicyCapture()
  except Exception:
    return None  # unavailable, not proof of disabled or complete policy
