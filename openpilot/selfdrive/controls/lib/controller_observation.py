"""Bounded first-stage clock evidence, disabled unless the environment value is exactly 1.

This is not a plant-admission or policy-authentication witness. The publisher worker
does not provide real-time isolation: pinned msgq's Cython send holds the GIL and
has no wall-time guarantee. Enabled control-thread p95/p99/max and backpressure
remain unmeasured. Static logged service registration also creates logger sockets
when these controller observers are off. No controller path waits or joins here.
"""
import hashlib
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
MAX_STARTUP_BYTES = 65536
MAX_STARTUP_JOBS = 2  # lifetime cap also bounds pending/in-flight jobs and total raw retention to128KiB
MAX_STARTUP_RECORDS = 4
MAX_STARTUP_WIRE_BYTES = 1024
STARTUP_SOURCES = {"controls": ("paramsCarParams", "paramsCarParamsSP"),
                   "card": ("serializedCarParams", "serializedCarParamsSP")}
U64 = 2**64 - 1
CONTROLS_INPUTS = ("lateralDelay", "vehicleParameters", "lateralTorqueParameters", "modelV2", "selfdriveState",
                   "extrinsicsCalibration", "deviceMotion", "longitudinalPlan", "lateralManeuverPlan", "carState",
                   "carOutput", "driverMonitoringState", "onroadEvents", "driverAssistance", "radarState", "selfdriveStateSP")
CARD_INPUTS = ("pandaStates", "carControl", "onroadEvents", "carControlSP", "longitudinalPlanSP")
SERVICES = ("none", *dict.fromkeys((*CONTROLS_INPUTS, *CARD_INPUTS, "carParams", "carParamsSP", "carStateSP", "controlsState",
                                 "sendcan", "radarTracks", "liveLocationKalman")))
STAGES = ("loopStart", "updateBegin", "updateEnd", "consumeEnd", "computeBegin", "computeEnd", "paramsRefreshBegin",
          "paramsRefreshEnd", "publishBegin", "publishEnd", "publishFailed", "applyBegin", "applyEnd", "applySkipped",
          "initializeBegin", "initializeEnd", "startupBegin", "startupEnd")
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
    self._startup_tokens, self._startup_sources = {}, set()
    self._startup_records = self._startup_jobs = self._startup_reserved_bytes = self._startup_released_bytes = 0
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
              reason="none", tx=None, _startup=None):
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
      startup = None
      if _startup is not None:
        startup = dict(_startup)
        startup.pop("queued", None)
        if stage == "startupBegin":
          startup.update(begin_ns=now, begin_status="invalid" if faults else "available")
          _startup["begin_status"] = startup["begin_status"]
        else:
          startup.update(end_ns=now, end_status="available")
          if startup["begin_status"] != "available" or now <= startup["begin_ns"] or faults:
            startup["status"] = "clockInvalid"
            if now <= startup["begin_ns"] or faults:
              startup["end_status"] = "invalid"
            startup.pop("raw_bytes", None)
            _startup["raw_discarded"] = True
          _startup["status"] = startup["status"]
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
      size = 256 + 112 * len(refs) + 24 * len(frames) + (0 if raw is None else len(raw)) + (256 if startup else 0)
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
      if startup is not None:
        record["startup_reference"] = startup
        _startup["queued"] = True
      self._pending.append(record)  # sole producer appends; worker only poplefts, no capacity eviction
      return now
    except Exception:
      # Observation failure must not replace a controller exception or alter its values.
      # No exception text/payload is exported. Counter remains sticky for this process epoch.
      self.errors += 1
      self.dropped += 1
      self.loss = True
      return None

  def begin_startup_reference(self, source):
    """At most two independent fixed-source brackets; no raw reads or hashing."""
    try:
      if source not in STARTUP_SOURCES[self.producer] or source in self._startup_sources or len(self._startup_sources) >= MAX_STARTUP_JOBS:
        raise ValueError("unknown/duplicate startup source")
      self._startup_sources.add(source)
      reference = {"id": len(self._startup_sources), "source": source, "begin_ns": 0, "end_ns": 0,
                   "begin_status": "unknown", "end_status": "unknown", "outcome": "pending", "status": "pending",
                   "byte_count": 0, "byte_count_known": False, "sha256": b"", "digest_ready_ns": 0, "queued": False}
      self._startup_records += 1
      begin = self.capture("startupBegin", service="carParamsSP" if source.endswith("SP") else "carParams", _startup=reference)
      token = (self.epoch, reference["id"], source, begin, reference["queued"], reference["begin_status"])
      self._startup_tokens[token[1]] = token  # identity-checked opaque token; no raw bytes
      if not reference["queued"] or reference["begin_status"] != "available":
        self.loss = True
      return token
    except Exception:
      self.errors += 1
      self.dropped += 1
      self.loss = True
      return None

  def end_startup_reference(self, token, raw, outcome):
    """Retain only an original immutable bounded buffer; every observer exception stays local."""
    reference = None
    try:
      if (type(token) is not tuple or len(token) != 6 or self._startup_tokens.get(token[1]) is not token
          or token[0] != self.epoch or outcome not in ("complete", "readFailed", "parseFailed", "serializationFailed")
          or self._startup_records >= MAX_STARTUP_RECORDS):
        raise ValueError("unknown/duplicate startup completion")
      del self._startup_tokens[token[1]]
      self._startup_records += 1
      reference = {"id": token[1], "source": token[2], "begin_ns": token[3] if _uint(token[3]) else 0,
                   "end_ns": 0, "begin_status": token[5] if _uint(token[3]) else "unknown", "end_status": "unknown",
                   "outcome": outcome, "status": outcome, "byte_count": len(raw) if type(raw) is bytes else 0,
                   "byte_count_known": type(raw) is bytes, "sha256": b"", "digest_ready_ns": 0, "queued": False}
      if outcome == "complete":
        reference["status"] = ("nonbytes" if type(raw) is not bytes else "oversize" if len(raw) > MAX_STARTUP_BYTES else
                               "publisherFailed" if self.failed else "priorLoss" if self.loss or not token[4] else "pending")
      if reference["status"] == "pending" and reference["begin_status"] == "available" and self._startup_jobs < MAX_STARTUP_JOBS:
        reference["raw_bytes"] = raw
        reference["raw_reserved"] = True
        self._startup_jobs += 1
        self._startup_reserved_bytes += len(raw)
      else:
        self.loss = True
      self.capture("startupEnd", service="carParamsSP" if token[2].endswith("SP") else "carParams", _startup=reference)
      # Capture snapshots metadata. Remove this local duplicate immediately; queue owns the job.
      reference.pop("raw_bytes", None)
      if not reference["queued"] or reference.get("raw_discarded"):
        if reference.get("raw_reserved"):
          self._startup_released_bytes += reference["byte_count"]
          self._startup_jobs -= 1
        self.loss = True
      # Failed worker/enqueue race: if failure preceded append, release the newly queued job here;
      # if failure follows this check, worker cleanup sees the already appended record.
      if self.failed:
        self._discard_startup_bytes()
    except Exception:
      self.errors += 1
      self.dropped += 1
      self.loss = True
    finally:
      if reference is not None:
        reference.pop("raw_bytes", None)

  def _release_startup_bytes(self, reference):
    raw = reference.pop("raw_bytes", None)
    if raw is not None:
      self._startup_released_bytes += len(raw)
    return raw

  def _discard_startup_bytes(self):
    for record in tuple(self._pending):  # worker/error cleanup only; at most128 records and2 raw buffers
      reference = record.get("startup_reference")
      if reference is not None and "raw_bytes" in reference:
        self._release_startup_bytes(reference)
        reference["status"], reference["sha256"] = "publisherFailed", b""

  def _prepare_startup(self, record):
    reference = record.get("startup_reference")
    if reference is None:
      return
    raw = reference.pop("raw_bytes", None)
    if raw is None:
      return
    raw_size = len(raw)
    try:
      if reference["status"] != "pending" or reference["end_status"] != "available":
        return
      digest = hashlib.sha256(raw).digest()
      raw = None  # release BEFORE digest-ready and worker Event construction clocks
      if type(digest) is not bytes or len(digest) != 32:
        raise ValueError("unknown digest result")
      try:
        ready = self.clock()
      except Exception:
        reference["status"] = "clockInvalid"
        self.loss = True
        self.errors += 1
        return
      if not _uint(ready) or ready < reference["end_ns"]:
        reference["status"] = "clockInvalid"
        self.loss = True
        return
      reference.update(status="ready", sha256=digest, digest_ready_ns=ready)
    except Exception:
      reference["status"] = "digestFailed"
      self.loss = True
      self.errors += 1
    finally:
      raw = None
      self._startup_released_bytes += raw_size

  def begin_apply(self, source_clock, sm):
    self.apply_id += 1
    self.capture("applyBegin", sm=sm, apply_id=self.apply_id, apply_clock_ns=source_clock)
    return self.apply_id

  def end_apply(self, apply_id, source_clock, frames):
    self.last_completed_apply = apply_id
    self.capture("applyEnd", apply_id=apply_id, apply_clock_ns=source_clock, tx=frames)

  def flush_one(self, publish):
    if self.failed:
      self._discard_startup_bytes()
      return False
    try:
      record = self._pending.popleft()
    except IndexError:
      return True
    try:
      self._prepare_startup(record)
      publish(self.service, record)
      return True
    except Exception:
      self.errors += 1
      self.failed = True
      self._discard_startup_bytes()
      return False  # no reconnect/retry; bounded future capture records the sticky failure

  def _worker(self):
    try:
      from openpilot.cereal import log
      from openpilot.cereal.messaging import PubMaster
      publisher = PubMaster([self.service])
    except Exception:
      self.errors += 1
      self.failed = True
      self._discard_startup_bytes()
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
  startup = record.get("startup_reference")
  if startup is not None:
    startup = dict(startup)  # metadata only; internal raw job has already been popped in flush_one
    if startup["status"] == "ready" and (not _uint(publisher_ns) or publisher_ns < startup["digest_ready_ns"]):
      startup["status"], startup["sha256"] = "clockInvalid", b""
    complete = complete and startup["status"] == "ready"
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
  if startup is not None:
    ref = target.startupReference
    ref.present, ref.referenceId, ref.source = True, startup["id"], startup["source"]
    ref.decoder = "carParamsSP" if startup["source"].endswith("SP") else "carParams"
    ref.wire = "capnpUnpackedMessage"
    ref.beginNs, ref.endNs = startup["begin_ns"], startup["end_ns"]
    ref.beginStatus, ref.endStatus, ref.outcome, ref.status = (startup[k] for k in ("begin_status", "end_status", "outcome", "status"))
    ref.byteCount, ref.byteCountKnown = startup["byte_count"], startup["byte_count_known"]
    ref.sha256, ref.digestReadyNs = startup["sha256"], startup["digest_ready_ns"]
  raw = event.to_bytes()
  if len(raw) > (MAX_STARTUP_WIRE_BYTES if startup is not None else MAX_SERIALIZED_BYTES):
    raise ValueError("bounded observer record exceeded wire budget")
  return raw
