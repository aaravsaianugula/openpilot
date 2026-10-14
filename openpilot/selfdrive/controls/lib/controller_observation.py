"""Bounded first-stage clock evidence, disabled unless the environment value is exactly 1.

This is not a plant-admission or policy-authentication witness. The publisher worker
does not provide real-time isolation: pinned msgq's Cython send holds the GIL and
has no wall-time guarantee. Enabled control-thread p95/p99/max and backpressure
remain unmeasured. Static logged service registration also creates logger sockets
when these controller observers are off. No controller path waits or joins here.
"""
import math
import os
import threading
import time
from collections import deque

MAX_REFS = 24
MAX_CAN = 64
MAX_CAN_BYTES = 1024
MAX_LOOP_RECORDS = 32
MAX_LOOP_CAPTURE_BYTES = 32768
MAX_RECORD_CAPTURE_BYTES = 8192
MAX_SERIALIZED_BYTES = 8192
MAX_EVENT_BYTES = 8192
U64 = 2**64 - 1
CONTROLS_INPUTS = ("lateralDelay", "vehicleParameters", "lateralTorqueParameters", "modelV2", "selfdriveState",
                   "extrinsicsCalibration", "deviceMotion", "longitudinalPlan", "lateralManeuverPlan", "carState",
                   "carOutput", "driverMonitoringState", "onroadEvents", "driverAssistance", "radarState", "selfdriveStateSP")
CARD_INPUTS = ("pandaStates", "carControl", "onroadEvents", "carControlSP", "longitudinalPlanSP")
SERVICES = ("none", *dict.fromkeys((*CONTROLS_INPUTS, *CARD_INPUTS, "carParams", "carParamsSP", "carStateSP", "controlsState",
                                 "sendcan", "radarTracks", "liveLocationKalman")))
STAGES = ("loopStart", "updateBegin", "updateEnd", "consumeEnd", "computeBegin", "computeEnd", "paramsRefreshBegin",
          "paramsRefreshEnd", "publishBegin", "publishEnd", "publishFailed", "applyBegin", "applyEnd", "applySkipped",
          "initializeBegin", "initializeEnd")
REASONS = ("none", "passive", "notInitialized", "ccNotAlive")


def _uint(value, maximum=U64):
  return type(value) is int and 0 <= value <= maximum


def _tx_snapshot(frames, faults):
  if not isinstance(frames, (tuple, list)) or len(frames) > MAX_CAN:
    faults.append("txOversize")
    return ()
  result, size = [], 0
  for frame in frames:
    if not isinstance(frame, (tuple, list)) or len(frame) != 3:
      faults.append("unknownField")
      return ()
    address, data, bus = frame
    if not _uint(address, 2**32 - 1) or not _uint(bus, 255) or not isinstance(data, (bytes, bytearray)):
      faults.append("unknownField")
      return ()
    size += len(data)
    if len(data) > 64 or size > MAX_CAN_BYTES:
      faults.append("txOversize")
      return ()
    # Ordered header identity only. CAN payloads may contain VIN/firmware/diagnostic
    # data; existing sendcan logs own them. No raw payload is copied or exported.
    result.append({"address": address, "src": bus, "payload_length": len(data)})
  return tuple(result)


class Observer:
  """One source producer, a bounded CPython deque, and one source-owned publisher."""
  def __init__(self, producer, *, clock=time.monotonic_ns, epoch=None, capacity=128, start_worker=True):
    if producer not in ("controls", "card") or type(capacity) is not int or not 1 <= capacity <= 128:
      raise ValueError("invalid observer configuration")
    self.producer, self.clock = producer, clock
    self.service = "controlsObservation" if producer == "controls" else "cardObservation"
    self.epoch = os.urandom(16) if epoch is None else epoch
    if type(self.epoch) is not bytes or len(self.epoch) != 16:
      raise ValueError("invalid process epoch")
    self.capacity, self._pending = capacity, deque()
    self.sequence = self.loop = self.dropped = self.errors = 0
    self.apply_id = self.last_completed_apply = 0
    self._loop_records = self._loop_bytes = self._last_clock = 0
    self._last_refs = {}
    self._last_published = {}
    self.failed = False
    self.loss = False
    if start_worker:
      threading.Thread(target=self._worker, name=f"{producer}_observation", daemon=True).start()

  def begin_loop(self):
    self.loop += 1
    self._loop_records = self._loop_bytes = 0
    self.capture("loopStart")

  def _refs(self, sm, now, faults):
    names = CONTROLS_INPUTS if self.producer == "controls" else CARD_INPUTS
    result = []
    for service in (*names, "liveLocationKalman", "carParams", "carParamsSP"):
      ref = {"service": service, "status": "notSubscribed", "identity_kind": "unavailable", "event_ns": 0,
             "recv_frame": 0, "recv_time": 0., "seen": False, "updated": False, "valid": False, "alive": False, "freq_ok": False}
      if service not in names:
        result.append(ref)
        continue
      try:
        ns, frame, received = sm.logMonoTime[service], sm.recv_frame[service], sm.recv_time[service]
        flags = {name: getattr(sm, name)[service] for name in ("seen", "updated", "valid", "alive", "freq_ok")}
        if not _uint(ns) or not _uint(frame) or any(type(v) is not bool for v in flags.values()):
          raise ValueError("unknown source field")
        if type(received) not in (int, float) or not math.isfinite(received) or received < 0:
          faults.append("nonfinite")
          raise ValueError("unknown receive clock")
        ref.update(event_ns=ns, recv_frame=frame, recv_time=float(received), **flags,
                   status="observed" if flags["seen"] else "unseen", identity_kind="timestampReference")
        previous = self._last_refs.get(service)
        if flags["seen"]:
          if not flags["valid"]:
            faults.append("sourceInvalid")
          if ns > now:
            faults.append("futureReference")
          if previous is not None and frame != previous[1]:
            if ns == previous[0]:
              faults.append("timestampTie")
            elif ns < previous[0]:
              faults.append("timestampRegression")
          self._last_refs[service] = (ns, frame)
        else:
          faults.append("unseenSource")
      except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        faults.append("unknownField")
        ref["status"] = "unknown"
      result.append(ref)
    return tuple(result)

  def capture(self, stage, *, sm=None, service="none", event=None, begin_ns=None, apply_clock_ns=None, apply_id=None,
              reason="none", tx=None):
    """Copy only bounded clocks/flags/CAN. Never serialize, hash full payloads, send or wait."""
    self.sequence += 1
    try:
      now = self.clock()
      if not _uint(now) or stage not in STAGES or service not in SERVICES or reason not in REASONS:
        raise ValueError("unknown observation claim")
      faults = []
      if now <= self._last_clock:
        faults.append("clockTie" if now == self._last_clock else "clockRegression")
      self._last_clock = now
      refs = () if sm is None else self._refs(sm, now, faults)
      if len(refs) > MAX_REFS:
        raise ValueError("too many references")
      frames = () if tx is None else _tx_snapshot(tx, faults)
      event_ns, event_valid, event_present, raw = 0, False, False, None
      if event is not None:
        if type(event) is bytes:
          if service != "sendcan" or len(event) > MAX_EVENT_BYTES:
            faults.append("eventOversize")
          else:
            raw = event  # immutable, bounded; header parsing happens only in publisher worker
        else:
          try:
            ns, valid = event.logMonoTime, event.valid
            if not _uint(ns) or type(valid) is not bool:
              raise ValueError("unknown event identity")
            event_ns, event_valid, event_present = ns, valid, True
            if not valid:
              faults.append("sourceInvalid")
            if ns > now:
              faults.append("futureReference")
            if stage == "publishBegin":
              previous = self._last_published.get(service)
              if previous is not None and ns <= previous:
                faults.append("timestampTie" if ns == previous else "timestampRegression")
              self._last_published[service] = ns
          except (AttributeError, TypeError, ValueError, OverflowError):
            faults.append("unknownField")
      begin = now if begin_ns is None else begin_ns
      applied_clock = 0 if apply_clock_ns is None else apply_clock_ns
      applied_id = self.last_completed_apply if apply_id is None else apply_id
      if not _uint(begin) or begin > now or not _uint(applied_clock) or not _uint(applied_id):
        raise ValueError("unknown source clock")
      size = 256 + 112 * len(refs) + 24 * len(frames) + (0 if raw is None else len(raw))
      if (size > MAX_RECORD_CAPTURE_BYTES or self._loop_records >= MAX_LOOP_RECORDS
          or self._loop_bytes + size > MAX_LOOP_CAPTURE_BYTES or len(self._pending) >= self.capacity):
        self.dropped += 1
        return now
      self._loop_records += 1
      self._loop_bytes += size
      if self.failed:
        faults.append("publisherFailed")
      if "txOversize" in faults or "eventOversize" in faults:
        self.loss = True
      if self.loss:
        faults.append("priorLoss")
      record = {"producer": self.producer, "service_name": self.service, "epoch": self.epoch, "sequence": self.sequence,
                "loop": self.loop, "stage": stage, "service": service, "capture_ns": now, "begin_ns": begin,
                "apply_clock_ns": applied_clock, "apply_id": applied_id, "event_ns": event_ns,
                "event_valid": event_valid, "event_present": event_present, "event_bytes": raw,
                "reason": reason, "refs": refs, "tx": frames, "faults": tuple(dict.fromkeys(faults)),
                "capture_complete": not faults and not self.dropped and not self.errors,
                "policy_complete": False, "dropped": self.dropped, "errors": self.errors, "capture_bytes": size}
      self._pending.append(record)  # sole producer appends; worker only poplefts, no capacity eviction
      return now
    except Exception:
      # Observation failure must not replace a controller exception or alter its values.
      # No exception text/payload is exported. Counter remains sticky for this process epoch.
      self.errors += 1
      self.dropped += 1
      self.loss = True
      return None

  def begin_apply(self, source_clock, sm):
    self.apply_id += 1
    self.capture("applyBegin", sm=sm, apply_id=self.apply_id, apply_clock_ns=source_clock)
    return self.apply_id

  def end_apply(self, apply_id, source_clock, frames):
    self.last_completed_apply = apply_id
    self.capture("applyEnd", apply_id=apply_id, apply_clock_ns=source_clock, tx=frames)

  def flush_one(self, publish):
    if self.failed:
      return False
    try:
      record = self._pending.popleft()
    except IndexError:
      return True
    try:
      publish(self.service, record)
      return True
    except Exception:
      self.errors += 1
      self.failed = True
      return False  # no reconnect/retry; bounded future capture records the sticky failure

  def _worker(self):
    try:
      from openpilot.cereal import log
      from openpilot.cereal.messaging import PubMaster
      publisher = PubMaster([self.service])
    except Exception:
      self.errors += 1
      self.failed = True
      return
    previous_publications = {}
    def publish(service, record):
      publisher.send(service, encode_record(record, log, publisher_ns=time.monotonic_ns(),
                                            previous_publications=previous_publications))
    while not self.failed:
      if self._pending:
        self.flush_one(publish)
      else:
        time.sleep(.005)  # worker only; no control-path wait/join


class PublisherTap:
  """Forward original arguments once; observe only fixed service/Event header fields."""
  def __init__(self, parent, observer):
    self.parent, self.observer, self.sock = parent, observer, parent.sock

  def send(self, service, event):
    if service not in SERVICES:
      return self.parent.send(service, event)
    before = self.observer.capture("publishBegin", service=service, event=event)
    try:
      result = self.parent.send(service, event)
    except Exception:
      self.observer.capture("publishFailed", service=service, event=event, begin_ns=before)
      raise
    self.observer.capture("publishEnd", service=service, event=event, begin_ns=before)
    return result


def create_observer(producer):
  if os.environ.get("ELANTRA_CONTROLLER_OBSERVER") != "1":
    return None
  try:
    return Observer(producer)
  except Exception:
    return None  # enabled startup failure cannot change controller initialization; no eligibility claim


def encode_record(record, log, *, publisher_ns, previous_publications=None):
  """Worker-only bounded capnp serialization; actual sendcan header is parsed, not reconstructed."""
  event_ns, event_valid, event_present = record["event_ns"], record["event_valid"], record["event_present"]
  faults = list(record["faults"])
  if record["event_bytes"] is not None:
    with log.Event.from_bytes(record["event_bytes"], traversal_limit_in_words=4096) as original:
      if original.which() != "sendcan":
        raise ValueError("unexpected bounded TX event")
      event_ns, event_valid, event_present = original.logMonoTime, original.valid, True
    if not event_valid:
      faults.append("sourceInvalid")
    if event_ns > record["capture_ns"]:
      faults.append("futureReference")
    if previous_publications is None:
      faults.append("identityUnchecked")
    elif record["stage"] == "publishBegin":
      previous = previous_publications.get(record["service"])
      if previous is not None and event_ns <= previous:
        faults.append("timestampTie" if event_ns == previous else "timestampRegression")
      previous_publications[record["service"]] = event_ns
  complete = record["capture_complete"] and not faults
  event = log.Event.new_message(logMonoTime=publisher_ns, valid=complete)
  target = event.init(record["service_name"]).init("observation")
  target.version, target.producer, target.processEpoch = 1, record["producer"], record["epoch"]
  for field, key in (("sequence", "sequence"), ("loop", "loop"), ("captureNs", "capture_ns"), ("beginNs", "begin_ns"),
                     ("applyClockNs", "apply_clock_ns"), ("applyId", "apply_id"), ("dropped", "dropped"), ("errors", "errors")):
    setattr(target, field, record[key])
  target.stage, target.service, target.reason = record["stage"], record["service"], record["reason"]
  target.eventNs, target.eventValid, target.eventPresent = event_ns, event_valid, event_present
  target.captureComplete, target.policyComplete = complete, False
  target.sourceAuthenticated, target.predictiveEligible = False, False
  target.rawBoolPresence, target.txEchoCorrelation, target.calibratedRollUncertainty = "unknown", "unsupported", "unknown"
  target.bootIdentityKnown, target.semanticDigestAvailable, target.callbackTxObserved = False, False, False
  refs = target.init("references", len(record["refs"]))
  for item, source in zip(refs, record["refs"], strict=True):
    item.service, item.status, item.identityKind = source["service"], source["status"], source["identity_kind"]
    item.eventNs, item.recvFrame, item.recvTime = source["event_ns"], source["recv_frame"], source["recv_time"]
    item.seen, item.updated, item.valid, item.alive, item.freqOk = (source[k] for k in ("seen", "updated", "valid", "alive", "freq_ok"))
  tx = target.init("tx", len(record["tx"]))
  for item, source in zip(tx, record["tx"], strict=True):
    item.address, item.src, item.payloadLength = source["address"], source["src"], source["payload_length"]
    item.payloadCaptured = False
  target.faults = tuple(dict.fromkeys(faults))
  raw = event.to_bytes()
  if len(raw) > MAX_SERIALIZED_BYTES:
    raise ValueError("bounded observer record exceeded wire budget")
  return raw
