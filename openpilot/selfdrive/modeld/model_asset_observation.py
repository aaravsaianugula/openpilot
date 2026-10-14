"""Default-off instance references, not compiled bytes or authenticated assets.

No files/Params/models/publishers live here. A loader owns its private token until
the existing native object handoff; only main writes the bounded journal. Enabled
overhead, clock domains, source identity and delivery are unvalidated.
"""
import math
import os
import time
from types import MappingProxyType

MAX_INSTANCES, MAX_REFERENCES, MAX_WIRE_BYTES = 2, 4, 1024
U64 = 2**64-1
ORIGINS = ("unknown", "envOverride", "bundleLookup", "stockBuiltin")
MODES = {"stock": "stock", "supercombo": "supercombo", "split": "split", "multi_policy": "multiPolicy"}
DEVICES = {"CPU": "cpu", "QCOM": "qcom", "AMD": "amd"}


def _uint(value, maximum=U64):
  return type(value) is int and 0 <= value <= maximum


def _finite(value):
  return type(value) is float and math.isfinite(value)


class _LoadReference:
  """Private single-loader token; completed scalar snapshot is immutable."""
  def __init__(self, identity, chestnut, clock, epoch, *, delay_enabled=False):
    self.id, self.chestnut, self.clock, self.epoch = identity, chestnut, clock, epoch
    self.begin_ns = None
    self.selection = "unknown"
    self.completed = None
    self.invalid = False
    self.delay_enabled, self._delay_consumer, self._delay_requested = delay_enabled, None, False

  def delay_consumer(self):
    if not self.delay_enabled:
      return None
    if self._delay_requested:
      self.invalid = True
      if self._delay_consumer is not None:
        self._delay_consumer.error()
      return None
    self._delay_requested = True
    try:
      from openpilot.selfdrive.modeld.model_delay_observation import DelayConsumer
      self._delay_consumer = DelayConsumer(self.id, self.epoch, self.clock)
      return self._delay_consumer
    except Exception:
      self.invalid = True
      return None

  def begin(self):
    try:
      value = self.clock()
      if self.begin_ns is not None or not _uint(value):
        raise ValueError("unknown load begin")
      self.begin_ns = value
    except Exception:
      self.invalid = True

  def origin(self, selection):
    if type(selection) is str and selection in ORIGINS and self.selection == "unknown" and self.completed is None:
      self.selection = selection
    else:
      self.invalid = True

  def loaded(self, model, width, height, mode):
    try:
      end = self.clock()
      if self.completed is not None or self.begin_ns is None or not _uint(end) or end <= self.begin_ns:
        raise ValueError("unknown load completion")
      generation = getattr(model, "generation", None)  # declared processing metadata, not compiled-asset identity
      lat, long = getattr(model, "LAT_SMOOTH_SECONDS", None), getattr(model, "LONG_SMOOTH_SECONDS", None)
      device = getattr(model, "DEV", getattr(model, "model_device", None))
      geometry = _uint(width, 16384) and _uint(height, 16384) and width > 0 and height > 0
      known_mode = type(mode) is str and mode in MODES
      known_device = type(device) is str and device in DEVICES
      valid = (not self.invalid and geometry and known_mode and self.selection != "unknown"
               and type(self.chestnut) is bool and type(model.chestnut) is bool and model.chestnut == self.chestnut)
      self.completed = MappingProxyType({"load_begin": self.begin_ns, "load_end": end, "load_clock_known": True,
        "origin": self.selection, "chestnut": self.chestnut if type(self.chestnut) is bool else False,
        "role_known": type(self.chestnut) is bool, "generation": generation if _uint(generation, 2**32-1) else 0,
        "generation_known": _uint(generation, 2**32-1), "width": width if geometry else 0, "height": height if geometry else 0,
        "geometry_known": geometry, "mode": MODES[mode] if known_mode else "unknown", "device": DEVICES[device] if known_device else "unknown",
        "lat": lat if _finite(lat) else 0., "lat_known": _finite(lat), "long": long if _finite(long) else 0.,
        "long_known": _finite(long), "valid": valid})
    except Exception:
      self.invalid = True  # no replacement clock, native constructor/exception unaffected


class ModelAssetCapture:
  def __init__(self, runner, *, clock=time.monotonic_ns, epoch=None, delay_enabled=False):
    if runner not in ("stock", "tinygrad"):
      raise ValueError("unknown model runner")
    self.runner, self.clock = runner, clock
    self.epoch = os.urandom(16) if epoch is None else epoch
    if type(self.epoch) is not bytes or len(self.epoch) != 16:
      raise ValueError("unknown model epoch")
    self.instances = self.reference_attempts = self.attempts = self.publications = 0
    self.dropped = self.errors = self.active_id = self.attempted_id = self.fallbacks = 0
    self.loss = False
    self.pending, self.loaded_ids = [], set()
    self.last_clock = self.staged = None
    self.delay = None
    if type(delay_enabled) is not bool:
      raise ValueError("unknown delay observation flag")
    if delay_enabled:
      from openpilot.selfdrive.modeld.model_delay_observation import DelayJournal
      self.delay = DelayJournal(self.epoch, self.clock)

  def error(self):
    self.dropped, self.errors = min(U64, self.dropped+1), min(U64, self.errors+1)
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

  def reserve(self, chestnut):
    if self.instances >= MAX_INSTANCES or type(chestnut) is not bool:
      self.error()
      return None
    self.instances += 1
    return _LoadReference(self.instances, chestnut, self.clock, self.epoch, delay_enabled=self.delay is not None)

  def consumer_for(self, model):
    if self.delay is None:
      return None
    try:
      ref = self._identity(model)
      if ref is None or ref.invalid:
        return None
      return ref._delay_consumer
    except Exception:
      self.error()
      return None

  def _identity(self, model):
    ref = getattr(model, "_model_asset_reference", None)
    if not isinstance(ref, _LoadReference) or ref.epoch is not self.epoch or not 1 <= ref.id <= self.instances:
      self.error()
      return None
    return ref

  def _record(self, kind, from_id, to_id, *, ref=None, failed_attempt=0):
    self.reference_attempts = min(U64, self.reference_attempts+1)
    if self.reference_attempts > MAX_REFERENCES:
      self.error()  # lifetime cap, never reset when prefix retires
      return
    observed = self.now()
    data = dict(ref.completed) if ref is not None and ref.completed is not None else {}
    valid = observed is not None and (kind != "load" or bool(data.get("valid", False)))
    if ref is not None and (ref.invalid or ref.completed is None or (observed is not None and data.get("load_end", U64) > observed)):
      valid = False
    if not valid:
      self.error()
    self.pending.append({"sequence": self.reference_attempts, "kind": kind, "from_id": from_id, "to_id": to_id,
                         "failed_attempt": failed_attempt, "observed": observed if observed is not None else 0,
                         "clock_known": observed is not None, "metadata": data, "valid": valid})

  def unavailable(self, reserved):
    if not isinstance(reserved, _LoadReference) or reserved.epoch is not self.epoch:
      self.error()
      return
    self.error()  # unavailable at selection; failure/timeout/late completion cause unknown
    self._record("bigUnavailable", reserved.id, 0)

  def _load(self, model):
    ref = self._identity(model)
    if ref is not None and ref.id not in self.loaded_ids:
      self.loaded_ids.add(ref.id)
      self._record("load", 0, ref.id, ref=ref)
    return ref

  def activate(self, model, standby=None):
    ref = self._load(model)
    if standby is not None and standby is not model:
      self._load(standby)
    if self.active_id or ref is None:
      self.error()
    identity = ref.id if ref is not None else 0
    self._record("initialSelection", self.active_id, identity)
    self.active_id = identity

  def attempt(self, model):
    ref = self._identity(model)
    self.attempts = min(U64, self.attempts+1)
    self.attempted_id = ref.id if ref is not None else 0
    if self.attempts == U64 or not self.active_id or self.attempted_id != self.active_id:
      self.error()

  def fallback(self, model):
    ref = self._identity(model)
    self.fallbacks += 1
    if self.fallbacks > 1 or ref is None or self.attempted_id != self.active_id:
      self.error()
    identity = ref.id if ref is not None else 0
    self._record("runtimeFallback", self.active_id, identity, failed_attempt=self.attempts)
    self.active_id = identity  # actual assignment, equal IDs allowed; no retroactive output

  def stage(self, sp, model_event, driving_event, model):
    header = None
    try:
      if self.staged is not None or self.publications >= U64:
        raise ValueError("overlapping model publication")
      self.publications += 1
      ref = self._identity(model)
      header = sp.modelDataV2SP.modelAssetReference
      header.present = False
      observed = self.now()
      times = (model_event.logMonoTime, driving_event.logMonoTime, sp.logMonoTime)
      native = model_event.modelV2
      if not all(_uint(value) for value in times) or not times[0] < times[1] < times[2]:
        self.error()
      if ref is None or ref.id != self.active_id or self.attempted_id != self.active_id or not self.attempts:
        self.error()
      header.processEpoch, header.runner = self.epoch, self.runner
      header.activeInstanceId, header.attemptedInstanceId = self.active_id, self.attempted_id
      header.attemptSequence, header.publicationAttemptId = self.attempts, self.publications
      header.appendNs, header.appendKnown = observed if observed is not None else 0, observed is not None
      header.modelEventNs, header.drivingEventNs, header.spEventNs = times
      header.modelEventValid, header.drivingEventValid, header.spEventValid = model_event.valid, driving_event.valid, sp.valid
      header.modelFrameId, header.modelFrameIdExtra, header.modelTimestampEof = native.frameId, native.frameIdExtra, native.timestampEof
      header.drivingFrameId, header.drivingFrameIdExtra = driving_event.drivingModelData.frameId, driving_event.drivingModelData.frameIdExtra
      header.referenceAttempts, header.dropped, header.errors, header.epochLoss = self.reference_attempts, self.dropped, self.errors, self.loss
      header.streamLayout, header.consumedDigest, header.fullAssetDigest = "unknown", "unknown", "unknown"
      header.fullAssetKnown, header.sourceAuthenticated, header.policyComplete, header.predictiveEligible = False, False, False, False
      header.bootIdentityKnown, header.clockDomainsVerified, header.exposureOriginVerified = False, False, False
      target = header.init("references", len(self.pending))
      for item, row in zip(target, self.pending, strict=True):
        item.sequence, item.kind, item.fromInstanceId, item.toInstanceId = row["sequence"], row["kind"], row["from_id"], row["to_id"]
        item.failedAttemptId, item.observationNs, item.clockKnown, item.valid = row["failed_attempt"], row["observed"], row["clock_known"], row["valid"]
        data = row["metadata"]
        item.selectionBeginNs, item.constructorEndNs = data.get("load_begin", 0), data.get("load_end", 0)
        item.constructorClockKnown = data.get("load_clock_known", False)
        item.origin, item.chestnut, item.roleKnown = data.get("origin", "unknown"), data.get("chestnut", False), data.get("role_known", False)
        item.declaredGeneration, item.declaredGenerationKnown = data.get("generation", 0), data.get("generation_known", False)
        item.camWidth, item.camHeight, item.geometryKnown = data.get("width", 0), data.get("height", 0), data.get("geometry_known", False)
        item.mode, item.device = data.get("mode", "unknown"), data.get("device", "unknown")
        item.latSmoothSeconds, item.latSmoothKnown = data.get("lat", 0.), data.get("lat_known", False)
        item.longSmoothSeconds, item.longSmoothKnown = data.get("long", 0.), data.get("long_known", False)
      header.referenceCaptureComplete = not self.loss and observed is not None
      token = (self.epoch, self.publications, tuple(row["sequence"] for row in self.pending))
      if self.delay is not None:
        self.delay.stage(header, token)
      header.present = True
      self.staged = token
      return token
    except Exception:
      self.error()
      if header is not None:
        try:
          header.present = False
        except Exception:
          pass  # unavailable header; native Event and exception are untouched
      return None

  def retire(self, token):
    if token is None:
      return
    if token is not self.staged:
      self.error()
      return
    if self.delay is not None and not self.delay.retire(token):
      self.error()
      return
    self.pending = [row for row in self.pending if row["sequence"] not in token[2]]
    self.staged = None

  def send_failed(self, token):
    self.error()
    if self.delay is not None:
      self.delay.send_failed(token)
    if token is self.staged:
      self.staged = None


def create_model_asset_capture(runner):
  delay_enabled = os.environ.get("ELANTRA_MODEL_DELAY_REFERENCE_OBSERVER") == "1"
  if os.environ.get("ELANTRA_MODEL_ASSET_OBSERVER") != "1" and not delay_enabled:
    return None
  try:
    return ModelAssetCapture(runner, delay_enabled=delay_enabled)
  except Exception:
    return None  # unavailable, never authenticated/complete
