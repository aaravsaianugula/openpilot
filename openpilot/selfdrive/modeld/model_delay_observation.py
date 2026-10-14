"""Default-off model delay read/use references; no graph/source/predictive proof.

Fixed scalar capture only. Enabled overhead/GIL/backpressure and all clock domains
remain unvalidated. No Params/models/files/PM/worker/extra native calls live here.
"""
import math
from types import MappingProxyType

import numpy as np

U64, U32, U16 = 2**64-1, 2**32-1, 2**16-1
USED_MASK, RESERVED_MASK = 0x01ffffff, 0xfe000000


class PackedReadError(ValueError):
  pass


def _codes(**values):
  return MappingProxyType(values)


LAYOUT = (
  ("site", 0, 2, _codes(unknown=0, delayInitial=1, lagdToggle=2, delayCache=3)),
  ("key", 2, 2, _codes(unknown=0, lagdValueCache=1, lagdToggle=2)),
  ("method", 4, 2, _codes(unknown=0, get=1, getBool=2)),
  ("registryType", 6, 2, _codes(unknown=0, floating=1, boolean=2)),
  ("defaultRequested", 8, 1, None),
  ("rawPresence", 9, 1, _codes(unknown=0)),
  ("defaultOrigin", 10, 1, _codes(unknown=0)),
  ("beginStatus", 11, 3, _codes(unavailable=0, known=1, tie=2, regression=3, invalid=4)),
  ("endStatus", 14, 3, _codes(unavailable=0, known=1, tie=2, regression=3, invalid=4)),
  ("outcome", 17, 2, _codes(unavailable=0, returned=1, raised=2)),
  ("valueKind", 19, 2, _codes(unknown=0, knownBool=1, knownFloat=2, knownNone=3)),
  ("boolValue", 21, 1, None),
  ("castOutcome", 22, 2, _codes(notObserved=0, succeeded=1, raised=2, notApplied=3)),
  ("valid", 24, 1, None),
)
SCALARS = ("readId", "instanceId", "beginNs", "endNs", "floatValue")
ORIGINAL_KEYS = frozenset(SCALARS) | frozenset(field for field, *_ in LAYOUT)
PACKED_KEYS = frozenset(SCALARS) | {"statusBits"}


def _shape(row, expected):
  if type(row) is not dict:
    raise PackedReadError("row_type")
  if len(row) != len(expected):
    raise PackedReadError("row_shape")
  # Check types before set/hash comparison, lookup, or unknown-key formatting.
  if any(type(key) is not str for key in row):
    raise PackedReadError("key_type")
  if frozenset(row) != expected:
    raise PackedReadError("row_shape")


def _scalars(row):
  for name in SCALARS[:-1]:
    value = row[name]
    if type(value) is not int or not 0 <= value <= (U16 if name == "instanceId" else U64):
      raise PackedReadError("scalar_unsigned_type_or_range")
  if type(row["floatValue"]) is not float:
    raise PackedReadError("scalar_float64_type")
  # No arithmetic/coercion/normalization: preserve even -0 exactly. Finite/known
  # eligibility is the original collector's separate contract, never this codec.
  return {name: row[name] for name in SCALARS}


def pack_read(row):
  _shape(row, ORIGINAL_KEYS)
  result = _scalars(row)
  word = 0
  for name, shift, _, values in LAYOUT:
    value = row[name]
    if values is None:
      if type(value) is not bool:
        raise PackedReadError("boolean_type")
      code = 1 if value else 0
    else:
      if type(value) is not str or value not in values:
        raise PackedReadError("enum_state")
      code = values[value]
    word |= code << shift
  result["statusBits"] = word
  return result


def unpack_read(row):
  _shape(row, PACKED_KEYS)
  result = _scalars(row)
  word = row["statusBits"]
  if type(word) is not int or not 0 <= word <= U32:
    raise PackedReadError("status_word_type_or_range")
  if word & RESERVED_MASK:
    raise PackedReadError("reserved_status_bits")
  for name, shift, width, values in LAYOUT:
    code = (word >> shift) & ((1 << width)-1)
    if values is None:
      result[name] = code == 1
    else:
      inverse = {value: label for label, value in values.items()}
      if code not in inverse:
        raise PackedReadError("invalid_enum_code")
      result[name] = inverse[code]
  return result


def _payload():
  # Exactly the reviewed100-field schema; zero values never mean known defaults.
  return {'present': False,
   'preparedStatus': 'unavailable',
   'preparedAttemptId': 0,
   'preparedInstanceId': 0,
   'preparedNs': 0,
   'preparedClockStatus': 'unavailable',
   'preparationComplete': False,
   'mainFrameId': 0,
   'extraFrameId': 0,
   'mainTimestampSof': 0,
   'mainTimestampEof': 0,
   'extraTimestampSof': 0,
   'extraTimestampEof': 0,
   'frameMetadataKnown': False,
   'singleCameraAlias': False,
   'singleCameraAliasKnown': False,
   'smFrame': 0,
   'publishedDelay': 0.0,
   'publishedDelayKnown': False,
   'publishedEventNs': 0,
   'publishedRecvFrame': 0,
   'publishedSeen': False,
   'publishedUpdated': False,
   'publishedValid': False,
   'publishedAlive': False,
   'publishedReferenceKnown': False,
   'cachedValue': 0.0,
   'cachedValueKind': 'unknown',
   'cachedReadId': 0,
   'cachedReadIdKnown': False,
   'cachedWriteNs': 0,
   'cachedWriteClockStatus': 'unavailable',
   'cachedOrigin': 'unknown',
   'effectiveLatOffset': 0.0,
   'effectiveLatOffsetKnown': False,
   'publishedLatOffset': 0.0,
   'publishedLatOffsetKnown': False,
   'constructorCacheValue': 0.0,
   'constructorValueKind': 'unknown',
   'constructorReadId': 0,
   'constructorReadIdKnown': False,
   'constructorWriteNs': 0,
   'constructorWriteClockStatus': 'unavailable',
   'cpLongitudinalActuatorDelay': 0.0,
   'cpLongitudinalActuatorDelayKnown': False,
   'cpOrigin': 'unknown',
   'longDelay': 0.0,
   'longDelayKnown': False,
   'initialLongSmoothing': 0.0,
   'initialLongSmoothingKnown': False,
   'initialLongOwnerId': 0,
   'initialLongOwnerKnown': False,
   'currentLatSmoothing': 0.0,
   'currentLongSmoothing': 0.0,
   'currentLatSmoothingKnown': False,
   'currentLongSmoothingKnown': False,
   'currentSmoothingOrigin': 'unknown',
   'postprocessSmoothingBranch': 'unknown',
   'pythonLatActionT': 0.0,
   'pythonLongActionT': 0.0,
   'pythonActionOffsetsKnown': False,
   'frameDelay': 0.0,
   'actionDelay': 0.0,
   'heuristicOffsetsKnown': False,
   'preparedActionLat': 0.0,
   'preparedActionLong': 0.0,
   'preparedActionStatus': 'unavailable',
   'preparedLateralVEgo': 0.0,
   'preparedLateralDelay': 0.0,
   'preparedLateralStatus': 'unavailable',
   'preparationVEgo': 0.0,
   'preparationVEgoKnown': False,
   'postprocessLatActionT': 0.0,
   'postprocessLongActionT': 0.0,
   'postprocessActionOffsetsKnown': False,
   'readCaptureComplete': False,
   'reads': [],
   'preparedAttemptCount': 0,
   'preparationUnavailableCount': 0,
   'outputNoneObservedCount': 0,
   'failedFinalSpSendCount': 0,
   'discardedBeforeFinalSpReturnCount': 0,
   'lastDiscardedAttemptId': 0,
   'lastDiscardedInstanceId': 0,
   'lastDiscardReason': 'unknown',
   'counterOverflowKnown': False,
   'epochLoss': False,
   'runCallKnown': False,
   'graphCopyKnown': False,
   'graphConsumptionKnown': False,
   'rawPresenceKnown': False,
   'defaultOriginKnown': False,
   'publishedCpVersusLearnerOriginKnown': False,
   'sourceAuthenticated': False,
   'fullAssetKnown': False,
   'policyComplete': False,
   'predictiveEligible': False,
   'clockDomainsVerified': False,
   'captureOriginVerified': False,
   'durableAckKnown': False}


def _uint(value, limit=U64):
  return type(value) is int and 0 <= value <= limit


def _float(value):
  return type(value) is float and math.isfinite(value)


class _Clock:
  def __init__(self, clock):
    self.clock, self.last_clock = clock, None

  def now(self):
    try:
      value = self.clock()
      if not _uint(value):
        return 0, "invalid"
      if self.last_clock is not None and value <= self.last_clock:
        return value, "tie" if value == self.last_clock else "regression"
      self.last_clock = value
      return value, "known"
    except Exception:
      return 0, "unavailable"  # no replacement observation clock


class _Read:
  def __init__(self, owner, site, read_id):
    self.owner, self.return_seen, self.invalid = owner, False, site == "unknown" or read_id == 0
    self.completed = self.retained = self.write_seen = False
    self.row = {"readId": read_id, "instanceId": owner.identity, "site": site,
                "key": "lagdToggle" if site == "lagdToggle" else "lagdValueCache",
                "method": "getBool" if site == "lagdToggle" else "get",
                "registryType": "boolean" if site == "lagdToggle" else "floating",
                "defaultRequested": site != "lagdToggle", "rawPresence": "unknown", "defaultOrigin": "unknown",
                "beginNs": 0, "endNs": 0, "beginStatus": "unavailable", "endStatus": "unavailable",
                "outcome": "unavailable", "valueKind": "unknown", "boolValue": False, "floatValue": 0.,
                "castOutcome": "notObserved", "valid": False}
    if self.invalid:
      self.row.update(key="unknown", method="unknown", registryType="unknown", defaultRequested=False)

  def __enter__(self):
    self.row["beginNs"], self.row["beginStatus"] = self.owner.now()
    return self

  def returned(self, value):
    if self.return_seen:
      self.owner.error()
      self.invalid = True
      return
    self.return_seen = True
    self.row["outcome"] = "returned"
    if self.row["site"] == "lagdToggle" and type(value) is bool:
      self.row["valueKind"], self.row["boolValue"] = "knownBool", value
    elif self.row["site"] in ("delayInitial", "delayCache") and value is None:
      self.row["valueKind"] = "knownNone"
    elif self.row["site"] in ("delayInitial", "delayCache") and _float(value):
      self.row["valueKind"], self.row["floatValue"] = "knownFloat", value
    else:
      self.owner.error()  # no private/raw value retained or converted

  def __exit__(self, exc_type, exc, tb):
    self.row["endNs"], self.row["endStatus"] = self.owner.now()
    if exc_type is not None:
      self.row["outcome"], self.row["valueKind"] = "raised", "unknown"
    self.row["valid"] = (not self.invalid and self.row["beginStatus"] == self.row["endStatus"] == "known"
                         and self.row["outcome"] == "returned" and self.row["valueKind"] != "unknown")
    if not self.row["valid"]:
      self.owner.error()
    if len(self.owner.pending) < 4 and self.row["readId"]:
      self.owner.pending.append(self.row)
      self.retained = True
    else:
      self.owner.error()  # native getter still ran; no eviction/coalescing
    self.completed = True
    return False  # preserve the native exception object, including BaseException


class DelayConsumer(_Clock):
  """One already-reserved token; only its constructor/then owning main touches it."""
  def __init__(self, identity, epoch, clock):
    if type(identity) is not int or identity not in (1, 2) or type(epoch) is not bytes or len(epoch) != 16:
      raise ValueError("unknown reserved consumer identity")
    super().__init__(clock)
    self.identity, self.epoch = identity, epoch
    self.pending, self.sequence, self.dropped, self.errors = [], 0, 0, 0
    self.loss, self.delay_origin = False, "unknown"
    self.last_read, self.constructor_written = None, False
    self.constructor = self.cached = MappingProxyType({"value": 0., "kind": "unknown", "read_id": 0,
      "read_known": False, "write_ns": 0, "clock_status": "unavailable", "origin": "unknown"})

  def error(self):
    self.dropped, self.errors = min(U64, self.dropped+1), min(U64, self.errors+1)
    self.loss = True

  def read(self, site):
    if type(site) is not str or site not in ("delayInitial", "lagdToggle", "delayCache"):
      self.error()
      site = "unknown"  # never substitute a known getter site
    if self.sequence < U64:
      self.sequence += 1
      read_id = self.sequence
    else:
      self.error()
      read_id = 0  # no unique lineage; never reuse an old pending U64 ID
    self.last_read = _Read(self, site, read_id)
    return self.last_read

  def delay_written(self, value, origin):
    if type(origin) is str and origin == "constructor" and self.constructor_written:
      self.error()
      return
    stamp, status = self.now()
    token = self.last_read
    linked = (token is not None and token.completed and token.retained and not token.write_seen
              and token.row["outcome"] == "returned")
    latest = token.row if linked else None
    kind = "knownFloat" if _float(value) else "knownNone" if value is None else "unknown"
    known_origin = {"constructor": "constructorCache", "publishedDelay": "publishedHelper", "cachedDelay": "cachedHelper"}
    origin = known_origin.get(origin, "unknown") if type(origin) is str else "unknown"
    if kind == "unknown" or status != "known" or origin == "unknown":
      self.error()
    if latest is not None:
      latest["castOutcome"] = "succeeded" if origin == "cachedHelper" else "notApplied"
      token.write_seen = True
    else:
      self.error()
    self.cached = MappingProxyType({"value": value if kind == "knownFloat" else 0., "kind": kind,
      "read_id": latest["readId"] if latest is not None else 0, "read_known": latest is not None, "write_ns": stamp,
      "clock_status": status, "origin": origin})
    if origin == "constructorCache":
      self.constructor_written = True
      self.constructor = self.cached  # immutable even when first captured value is unknown

  def cache_failed(self):
    token = self.last_read
    if (token is not None and token.completed and token.retained and not token.write_seen
        and token.row["site"] == "delayCache" and token.row["outcome"] == "returned"):
      token.row["castOutcome"] = "raised"
      token.write_seen = True


def _vector(inputs, key):
  if type(inputs) is not dict or len(inputs) > 4 or any(type(name) is not str or len(name) > 128 for name in inputs):
    return "unavailable", 0., 0.
  if key not in inputs:
    return "absentOriginalBranch", 0., 0.
  value = inputs[key]
  if type(value) is not np.ndarray or value.shape != (2,) or value.dtype != np.dtype(np.float32):
    return "unavailable", 0., 0.
  first, second = value[0], value[1]
  if type(first) is not np.float32 or type(second) is not np.float32:
    return "unavailable", 0., 0.
  first, second = first.item(), second.item()
  return ("prepared", first, second) if _float(first) and _float(second) else ("unavailable", 0., 0.)


class DelayJournal(_Clock):
  """Main-only fixed snapshot; no loader callback mutates this clock/journal."""
  def __init__(self, epoch, clock):
    super().__init__(clock)
    self.epoch, self.consumers = epoch, {}
    self.current = self.staged = None
    self.final_started = False
    self.dropped = self.errors = self.last_attempt = 0
    self.loss, self.overflow_known = False, True
    self.counts = dict.fromkeys(("preparedAttemptCount", "preparationUnavailableCount", "outputNoneObservedCount",
      "failedFinalSpSendCount", "discardedBeforeFinalSpReturnCount"), 0)
    self.last_discard = (0, 0, "unknown")
    self.long = {}

  def error(self):
    self.loss = True
    self.dropped, self.errors = min(U64, self.dropped+1), min(U64, self.errors+1)

  def _bump(self, name):
    if self.counts[name] == U64:
      self.overflow_known = False
      self.error()
    else:
      self.counts[name] += 1

  def _consumer(self, capture, model):
    consumer = capture.consumer_for(model)
    if (not isinstance(consumer, DelayConsumer) or consumer.epoch is not self.epoch
        or type(consumer.identity) is not int or consumer.identity not in (1, 2)
        or (consumer.identity in self.consumers and self.consumers[consumer.identity] is not consumer)):
      self.error()
      return None
    self.consumers[consumer.identity] = consumer
    return consumer

  def _discard(self, reason):
    if self.current is not None and not self.current["discarded"]:
      self._bump("discardedBeforeFinalSpReturnCount")
      self.last_discard = (self.current["attempt"], self.current["instance"], reason)
      self.current["discarded"] = True
      self.error()

  def initial_long(self, capture, model, cp_delay, delay, smoothing, origin):
    if self.long:
      self.error()
      return
    consumer = self._consumer(capture, model)
    self.long = {"cpLongitudinalActuatorDelay": cp_delay if _float(cp_delay) else 0.,
      "cpLongitudinalActuatorDelayKnown": _float(cp_delay), "cpOrigin": origin if type(origin) is str and origin in ("parsed", "demo") else "unknown",
      "longDelay": delay if _float(delay) else 0., "longDelayKnown": _float(delay),
      "initialLongSmoothing": smoothing if _float(smoothing) else 0., "initialLongSmoothingKnown": _float(smoothing),
      "initialLongOwnerId": consumer.identity if consumer is not None else 0, "initialLongOwnerKnown": consumer is not None}

  def prepare(self, capture, model, sm, main, extra, inputs, *, v_ego, effective_lat, published_lat,
              lat_action, long_action, frame_delay, action_delay, lat_smoothing, long_smoothing, smoothing_origin):
    self._discard(self.current["failure"] if self.current is not None else "unknown")
    self._bump("preparedAttemptCount")
    data = _payload()
    consumer = self._consumer(capture, model)
    attempt, identity = capture.attempts, capture.attempted_id
    if consumer is not None and consumer.identity != identity:
      self.error()
      consumer = None
    self.current = {"payload": data, "consumer": consumer, "attempt": attempt, "instance": identity,
                    "discarded": False, "failure": "replacedUnreturned"}
    try:
      stamp, status = self.now()
      data.update(self.long)
      data.update(preparedAttemptId=attempt, preparedInstanceId=identity, preparedNs=stamp, preparedClockStatus=status,
                  preparedStatus="preparedForRun", singleCameraAlias=main is extra, singleCameraAliasKnown=True)
      good = (consumer is not None and not consumer.loss and _uint(attempt) and attempt > self.last_attempt and status == "known"
              and data["initialLongOwnerKnown"] and data["cpLongitudinalActuatorDelayKnown"] and data["longDelayKnown"])
      self.last_attempt = attempt
      frame_fields = (("mainFrameId", main.frame_id, U32), ("extraFrameId", extra.frame_id, U32),
        ("mainTimestampSof", main.timestamp_sof, U64), ("mainTimestampEof", main.timestamp_eof, U64),
        ("extraTimestampSof", extra.timestamp_sof, U64), ("extraTimestampEof", extra.timestamp_eof, U64))
      data["frameMetadataKnown"] = all(_uint(value, bound) for _, value, bound in frame_fields)
      for name, value, bound in frame_fields:
        data[name] = value if _uint(value, bound) else 0
      good = good and data["frameMetadataKnown"]
      for name, value, known in (("publishedDelay", sm["lateralDelay"].lateralDelay, "publishedDelayKnown"),
          ("effectiveLatOffset", effective_lat, "effectiveLatOffsetKnown"), ("publishedLatOffset", published_lat, "publishedLatOffsetKnown"),
          ("preparationVEgo", v_ego, "preparationVEgoKnown"), ("currentLatSmoothing", lat_smoothing, "currentLatSmoothingKnown"),
          ("currentLongSmoothing", long_smoothing, "currentLongSmoothingKnown")):
        data[name], data[known] = (value, True) if _float(value) else (0., False)
        # Tinygrad has no native raw-published+smoother counterfactual offset.
        # Its actual raw helper input is publishedDelay; do not invent math/use.
        optional_published_offset = name == "publishedLatOffset" and capture.runner == "tinygrad" and value is None
        good = good and (data[known] or optional_published_offset)
      data["currentSmoothingOrigin"] = (smoothing_origin if type(smoothing_origin) is str
                                        and smoothing_origin in ("stockLiteral", "declaredProcessing") else "unknown")
      refs = (sm.frame, sm.logMonoTime["lateralDelay"], sm.recv_frame["lateralDelay"])
      data["publishedReferenceKnown"] = all(_uint(value) for value in refs)
      data["smFrame"], data["publishedEventNs"], data["publishedRecvFrame"] = [value if _uint(value) else 0 for value in refs]
      for field, name in (("publishedSeen", "seen"), ("publishedUpdated", "updated"), ("publishedValid", "valid"), ("publishedAlive", "alive")):
        value = getattr(sm, name)["lateralDelay"]
        data[field] = value if type(value) is bool else False
        data["publishedReferenceKnown"] = data["publishedReferenceKnown"] and type(value) is bool
      data["publishedReferenceKnown"] = (data["publishedReferenceKnown"] and data["publishedSeen"]
                                          and data["publishedEventNs"] > 0)
      data["pythonLatActionT"], data["pythonLongActionT"] = lat_action if _float(lat_action) else 0., long_action if _float(long_action) else 0.
      data["pythonActionOffsetsKnown"] = _float(lat_action) and _float(long_action)
      data["frameDelay"], data["actionDelay"] = frame_delay if _float(frame_delay) else 0., action_delay if _float(action_delay) else 0.
      data["heuristicOffsetsKnown"] = _float(frame_delay) and _float(action_delay)
      data["preparedActionStatus"], data["preparedActionLat"], data["preparedActionLong"] = _vector(inputs, "action_t")
      data["preparedLateralStatus"], data["preparedLateralVEgo"], data["preparedLateralDelay"] = _vector(inputs, "lateral_control_params")
      good = (good and data["publishedReferenceKnown"] and data["pythonActionOffsetsKnown"] and data["heuristicOffsetsKnown"]
              and data["preparedActionStatus"] != "unavailable" and data["preparedLateralStatus"] != "unavailable")
      if consumer is not None:
        for prefix, snapshot in (("cached", consumer.cached), ("constructor", consumer.constructor)):
          data["cachedValue" if prefix == "cached" else "constructorCacheValue"] = snapshot["value"]
          data["cachedValueKind" if prefix == "cached" else "constructorValueKind"] = snapshot["kind"]
          data[prefix+"ReadId"], data[prefix+"ReadIdKnown"] = snapshot["read_id"], snapshot["read_known"]
          data[prefix+"WriteNs"], data[prefix+"WriteClockStatus"] = snapshot["write_ns"], snapshot["clock_status"]
        data["cachedOrigin"] = consumer.cached["origin"]
      data["preparationComplete"] = good
      if not good:
        raise ValueError("unavailable prepared reference")
    except Exception:
      data["preparationComplete"] = False
      self._bump("preparationUnavailableCount")
      self.error()

  def output_none(self, observed):
    if observed is True:
      self._bump("outputNoneObservedCount")
      if self.current is None:
        self.error()
      self._discard("outputNone")

  def postprocess(self, lat_action, long_action, branch):
    if self.current is None or self.current["discarded"]:
      self.error()
      return
    data = self.current["payload"]
    data["postprocessLatActionT"] = lat_action if _float(lat_action) else 0.
    data["postprocessLongActionT"] = long_action if _float(long_action) else 0.
    data["postprocessActionOffsetsKnown"] = _float(lat_action) and _float(long_action)
    data["postprocessSmoothingBranch"] = branch if type(branch) is str and branch in ("used", "bypassed") else "unknown"

  def stage(self, header, token):
    current = self.current
    if (current is None or current["discarded"] or self.staged is not None or current["consumer"] is None
        or current["attempt"] != header.attemptSequence or current["instance"] != header.activeInstanceId):
      self.error()
      raise ValueError("unavailable current prepared reference")
    consumer = current["consumer"]
    if len(consumer.pending) > 4:
      self.error()
      raise ValueError("read prefix exceeds fixed bound")
    data = dict(current["payload"])
    if (data["mainFrameId"] != header.modelFrameId or data["extraFrameId"] != header.modelFrameIdExtra
        or data["mainTimestampEof"] != header.modelTimestampEof):
      data["preparationComplete"] = False
      self.error()
    data.update(self.counts)
    data["lastDiscardedAttemptId"], data["lastDiscardedInstanceId"], data["lastDiscardReason"] = self.last_discard
    data["counterOverflowKnown"], data["epochLoss"] = self.overflow_known, self.loss or consumer.loss
    data["readCaptureComplete"] = not consumer.loss
    data["reads"] = [pack_read(row) for row in consumer.pending]  # hard bound4, fixed scalars only
    data["present"] = True
    header.modelDelayReference = data
    self.staged = (token, consumer, tuple(row["readId"] for row in consumer.pending), current["attempt"])
    self.final_started = False
    header.dropped = min(U64, header.dropped+self.dropped+sum(c.dropped for c in self.consumers.values()))
    header.errors = min(U64, header.errors+self.errors+sum(c.errors for c in self.consumers.values()))
    header.epochLoss = header.epochLoss or data["epochLoss"]
    header.referenceCaptureComplete = header.referenceCaptureComplete and not data["epochLoss"]

  def final_send_started(self, token):
    if self.staged is None or token is not self.staged[0] or self.final_started:
      self.error()
      return
    self.final_started = True

  def send_failed(self, token):
    if self.staged is not None and token is self.staged[0]:
      if self.final_started:
        self._bump("failedFinalSpSendCount")
        if self.current is not None:
          self.current["failure"] = "finalSendRaised"
      self.staged = None
      self.final_started = False
    self.error()

  def retire(self, token):
    if self.staged is None or token is not self.staged[0] or not self.final_started:
      self.error()
      return False
    _, consumer, ids, attempt = self.staged
    consumer.pending = [row for row in consumer.pending if row["readId"] not in ids]
    if self.current is not None and self.current["attempt"] == attempt:
      self.current = None
    self.staged, self.final_started = None, False
    return True
