"""Authenticated camera bytes over TCP, suitable for a separately configured USB IP link.

No device discovery, USB setup, decoding or actuation is performed here. Capture stamps
must be actual camera SOF nanoseconds in the publisher's declared clock domain. A clock
exchange measures an offset interval; its caller-supplied drift bound is an external
contract, not an oscillator calibration. HMAC keys must be provisioned out of band.
TCP peers never unpickle input, resynchronize a damaged stream, or reconnect silently.
"""

import collections
import hashlib
import hmac
import json
import math
import select
import socket
import struct
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vision_long.native_context import MAX_CONTEXT_BYTES, NativeContextRecord

Camera = Literal["narrow", "wide"]
PixelFormat = Literal["rgb24", "nv12", "hevc"]
_CAMERAS: tuple[Camera, ...] = ("narrow", "wide")
_FORMATS: tuple[PixelFormat, ...] = ("rgb24", "nv12", "hevc")
_PREFIX = struct.Struct("!4sBBHII")
_HEADER_SIZE = _PREFIX.size + 32
_FRAME = struct.Struct("!16sBBHQQQIII")
_HELLO, _WELCOME, _CLOCK_REQUEST, _CLOCK_RESPONSE, _CAMERA = range(1, 6)
_CONTEXT = 6
_PROPOSAL = 7
_PROPOSAL_METADATA = b'{"schema":"signed_proposal_v1"}'
_CLOCK_EXCHANGES = 3  # Independent four-timestamp exchanges per synchronization; the tightest is kept.
_CONTEXT_METADATA = b'{"schema":"native_context_v1"}'


class TransportError(ValueError):
    """The caller must remove transport authority and close a failed connection."""


def _integer(value: int, name: str, minimum: int = 0, maximum: int = 2**64 - 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")


def _session(value: str) -> None:
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("invalid session_id")


def _clock_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("invalid clock_id")


@dataclass(frozen=True)
class CameraFrame:
    session_id: str
    camera: Camera
    sequence: int  # Per-camera transport sequence, strictly increasing within a session.
    frame_id: int  # Actual source camera identity; narrow/wide IDs need not be equal.
    capture_ns: int  # Actual SOF, in the publisher clock domain, never send/receive time.
    width: int
    height: int
    stride: int  # Bytes; HEVC uses 0. NV12 requires contiguous Y then interleaved UV.
    pixel_format: PixelFormat
    payload: bytes

    def __post_init__(self) -> None:
        _session(self.session_id)
        if self.camera not in _CAMERAS or self.pixel_format not in _FORMATS:
            raise ValueError("invalid camera or pixel_format")
        for name in ("sequence", "frame_id", "capture_ns"):
            _integer(getattr(self, name), name)
        for name in ("width", "height"):
            _integer(getattr(self, name), name, 1, 8192)
        _integer(self.stride, "stride", 0, 65536)
        if type(self.payload) is not bytes or not self.payload:
            raise ValueError("payload must be nonempty bytes")
        if self.pixel_format == "hevc":
            if self.stride != 0:
                raise ValueError("HEVC stride must be zero")
        else:
            minimum = self.width * (3 if self.pixel_format == "rgb24" else 1)
            expected = self.stride * self.height
            if self.pixel_format == "nv12":
                if self.width % 2 or self.height % 2 or self.stride % 2:
                    raise ValueError("NV12 dimensions and stride must be even")
                expected = expected * 3 // 2
            if self.stride < minimum or len(self.payload) != expected:
                raise ValueError("invalid pixel buffer layout")


class FrameCodec:
    """Fixed, bounded network-order framing with HMAC over header and both bodies."""

    def __init__(
        self,
        key: bytes,
        max_payload_bytes: int = 16 * 1024 * 1024,
        *,
        with_context: bool = False,
        with_return: bool = False,
    ):
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("a provisioned key of at least 32 bytes is required")
        _integer(max_payload_bytes, "max_payload_bytes", 1, 64 * 1024 * 1024)
        self._key = key
        self.max_payload_bytes = max_payload_bytes
        if type(with_context) is not bool:
            raise ValueError("explicit context protocol selection required")
        if type(with_return) is not bool or (with_return and not with_context):
            raise ValueError("return_protocol_requires_native_context")
        self.with_context = with_context
        self.with_return = with_return

    def _header(self, prefix: bytes | memoryview) -> tuple[int, int, int]:
        if len(prefix) != _PREFIX.size:
            raise TransportError("truncated_header")
        magic, version, kind, reserved, metadata_size, payload_size = _PREFIX.unpack(prefix)
        if (
            magic != b"VLC1"
            or version != (3 if self.with_return else 2 if self.with_context else 1)
            or kind not in range(1, 8 if self.with_return else 7 if self.with_context else 6)
            or reserved
        ):
            raise TransportError("invalid_header")
        if metadata_size > 2048 or metadata_size == 0:
            raise TransportError("metadata_limit")
        if payload_size > self.max_payload_bytes:
            raise TransportError("payload_limit")
        if kind == _CAMERA and metadata_size != _FRAME.size:
            raise TransportError("invalid_frame_header")
        if kind == _CONTEXT and (metadata_size != len(_CONTEXT_METADATA) or not 1 <= payload_size <= MAX_CONTEXT_BYTES):
            raise TransportError("context_packet_limit")
        if kind == _PROPOSAL and (metadata_size != len(_PROPOSAL_METADATA) or not 1 <= payload_size <= 16384):
            raise TransportError("proposal_packet_limit")
        if kind not in (_CAMERA, _CONTEXT, _PROPOSAL) and payload_size:
            raise TransportError("unexpected_control_payload")
        return kind, metadata_size, payload_size

    def encode_packet(self, kind: int, metadata: bytes, payload: bytes = b"") -> bytes:
        if type(metadata) is not bytes or type(payload) is not bytes:
            raise ValueError("packet metadata and payload must be immutable bytes")
        prefix = _PREFIX.pack(
            b"VLC1", 3 if self.with_return else 2 if self.with_context else 1, kind, 0, len(metadata), len(payload)
        )
        self._header(prefix)
        authentication = hmac.new(self._key, digestmod=hashlib.sha256)
        authentication.update(prefix)
        authentication.update(metadata)
        authentication.update(payload)
        return b"".join((prefix, authentication.digest(), metadata, payload))

    def decode_packet(self, wire: bytes) -> tuple[int, bytes, bytes]:
        # Zero-copy authentication requires immutable storage through verification
        # and extraction; read-only views over mutable storage are insufficient.
        if type(wire) is not bytes:
            raise TransportError("packet_bytes_required")
        view = memoryview(wire)
        kind, metadata_size, payload_size = self._header(view[: _PREFIX.size])
        if len(wire) != _HEADER_SIZE + metadata_size + payload_size:
            raise TransportError("packet_length")
        authentication = hmac.new(self._key, digestmod=hashlib.sha256)
        authentication.update(view[: _PREFIX.size])
        authentication.update(view[_HEADER_SIZE:])
        if not hmac.compare_digest(authentication.digest(), view[_PREFIX.size : _HEADER_SIZE]):
            raise TransportError("integrity_failure")
        split = _HEADER_SIZE + metadata_size
        return kind, view[_HEADER_SIZE:split].tobytes(), view[split:].tobytes()

    @staticmethod
    def _read_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
        result = bytearray()
        try:
            while len(result) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError("packet_deadline")
                if not select.select([connection], [], [], remaining)[0]:
                    raise TransportError("packet_deadline")
                part = connection.recv(min(size - len(result), 65536))
                if not part:
                    raise TransportError("link_closed")
                result.extend(part)
        except TimeoutError as exc:
            raise TransportError("socket_timeout") from exc
        except (OSError, ValueError) as exc:
            if isinstance(exc, TransportError):
                raise
            raise TransportError("socket_error") from exc
        return bytes(result)

    def read_packet(self, connection: socket.socket, deadline: float | None = None) -> tuple[int, bytes, bytes]:
        timeout = connection.gettimeout()
        packet_deadline = time.monotonic() + (timeout if timeout is not None else 5.0)
        deadline = packet_deadline if deadline is None else min(deadline, packet_deadline)
        header = self._read_exact(connection, _HEADER_SIZE, deadline)
        _, metadata_size, payload_size = self._header(header[: _PREFIX.size])
        body = self._read_exact(connection, metadata_size + payload_size, deadline)
        return self.decode_packet(header + body)

    def encode_frame(self, frame: CameraFrame) -> bytes:
        metadata = _FRAME.pack(
            uuid.UUID(frame.session_id).bytes,
            _CAMERAS.index(frame.camera),
            _FORMATS.index(frame.pixel_format),
            0,
            frame.sequence,
            frame.frame_id,
            frame.capture_ns,
            frame.width,
            frame.height,
            frame.stride,
        )
        return self.encode_packet(_CAMERA, metadata, frame.payload)

    @staticmethod
    def frame_from_packet(packet: tuple[int, bytes, bytes]) -> CameraFrame:
        kind, metadata, payload = packet
        if kind != _CAMERA or len(metadata) != _FRAME.size:
            raise TransportError("expected_camera_frame")
        session, camera, pixel_format, reserved, sequence, frame_id, capture, width, height, stride = _FRAME.unpack(
            metadata
        )
        if camera >= len(_CAMERAS) or pixel_format >= len(_FORMATS) or reserved:
            raise TransportError("invalid_frame_metadata")
        try:
            return CameraFrame(
                str(uuid.UUID(bytes=session)),
                _CAMERAS[camera],
                sequence,
                frame_id,
                capture,
                width,
                height,
                stride,
                _FORMATS[pixel_format],
                payload,
            )
        except ValueError as exc:
            raise TransportError("invalid_frame_metadata") from exc

    def decode_frame(self, wire: bytes) -> CameraFrame:
        return self.frame_from_packet(self.decode_packet(wire))

    def read_frame(self, connection: socket.socket) -> CameraFrame:
        return self.frame_from_packet(self.read_packet(connection))

    def encode_context(self, record: NativeContextRecord) -> bytes:
        return self.encode_packet(_CONTEXT, _CONTEXT_METADATA, record.to_bytes())

    @staticmethod
    def context_from_packet(packet: tuple[int, bytes, bytes]) -> NativeContextRecord:
        if packet[0] != _CONTEXT or packet[1] != _CONTEXT_METADATA:
            raise TransportError("expected_native_context")
        try:
            return NativeContextRecord.from_bytes(packet[2])
        except ValueError as exc:
            raise TransportError("invalid_native_context:" + str(exc)) from exc


@dataclass(frozen=True)
class ClockMapping:
    session_id: str
    clock_id: str
    offset_low_ns: int  # local - remote, including unknown asymmetric network delay.
    offset_high_ns: int
    measured_local_ns: int
    drift_ppm: float  # Must be a separately justified upper bound, not a fitted estimate.
    max_age_ns: int

    def __post_init__(self) -> None:
        _session(self.session_id)
        _clock_id(self.clock_id)
        for name in ("offset_low_ns", "offset_high_ns"):
            _integer(getattr(self, name), name, -(2**64 - 1))
        _integer(self.measured_local_ns, "measured_local_ns")
        _integer(self.max_age_ns, "max_age_ns", 1)
        if self.offset_low_ns > self.offset_high_ns:
            raise ValueError("invalid clock interval")
        _integer(self.measured_local_ns - self.offset_low_ns, "clock anchor")
        if (
            type(self.drift_ppm) not in (int, float)
            or not math.isfinite(self.drift_ppm)
            or not 0 <= self.drift_ppm <= 10000
        ):
            raise ValueError("invalid drift bound")

    @classmethod
    def from_exchange(
        cls,
        *,
        session_id: str,
        clock_id: str,
        local_send_ns: int,
        remote_receive_ns: int,
        remote_send_ns: int,
        local_receive_ns: int,
        drift_ppm: float,
        max_age_ns: int,
    ) -> "ClockMapping":
        for name, value in (
            ("l0", local_send_ns),
            ("r1", remote_receive_ns),
            ("r2", remote_send_ns),
            ("l3", local_receive_ns),
        ):
            _integer(value, name)
        if local_receive_ns < local_send_ns or remote_send_ns < remote_receive_ns:
            raise ValueError("clock_exchange_regression")
        low, high = local_send_ns - remote_receive_ns, local_receive_ns - remote_send_ns
        if low > high:
            raise ValueError("clock_exchange_inconsistent")
        result = cls(session_id, clock_id, low, high, local_receive_ns, drift_ppm, max_age_ns)
        padding = math.ceil((local_receive_ns - local_send_ns) * result.drift_ppm / 1_000_000)
        return cls(session_id, clock_id, low - padding, high + padding, local_receive_ns, drift_ppm, max_age_ns)

    def local_bounds(self, capture_ns: int, now_ns: int) -> tuple[int, int]:
        _integer(capture_ns, "capture_ns")
        _integer(now_ns, "now_ns")
        if now_ns < self.measured_local_ns:
            raise TransportError("local_clock_regression")
        elapsed = now_ns - self.measured_local_ns
        if elapsed > self.max_age_ns:
            raise TransportError("clock_mapping_expired")
        # Cover drift since the exchange and capture stamps older than that exchange.
        span = max(
            elapsed,
            *(
                abs(capture_ns + offset - self.measured_local_ns)
                for offset in (self.offset_low_ns, self.offset_high_ns)
            ),
        )
        drift = math.ceil(span * self.drift_ppm / 1_000_000)
        return capture_ns + self.offset_low_ns - drift, capture_ns + self.offset_high_ns + drift


@dataclass(frozen=True)
class SynchronizationConfig:
    queue_capacity: int = 3
    max_capture_skew_ns: int = 5_000_000
    max_pair_age_ns: int = 100_000_000
    max_clock_uncertainty_ns: int = 5_000_000
    link_timeout_ns: int = 150_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), name, 1)
        if self.queue_capacity > 64:
            raise ValueError("queue_capacity exceeds 64")


@dataclass(frozen=True)
class CameraPair:
    session_id: str
    narrow: CameraFrame
    wide: CameraFrame
    capture_local_lower_ns: int
    capture_local_upper_ns: int
    receive_ns: int

    @property
    def capture_skew_ns(self) -> int:
        return abs(self.narrow.capture_ns - self.wide.capture_ns)


@dataclass(frozen=True)
class TransportHealth:
    eligible: bool
    reason: str
    session_id: str | None
    oldest_camera_age_upper_ns: int | None
    counters: dict[str, int]


class DualCameraSynchronizer:
    """Single-consumer, bounded chronological pairing; health must be rechecked at use."""

    def __init__(self, config: SynchronizationConfig):
        self.config = config
        self.session_id: str | None = None
        self.mapping: ClockMapping | None = None
        self.counters: collections.Counter[str] = collections.Counter()
        self.begin_session(None)

    def begin_session(self, session_id: str | None, mapping: ClockMapping | None = None) -> None:
        """Start a fresh authenticated handshake session; never use for clock refresh."""
        if session_id is not None:
            _session(session_id)
        if mapping is not None and mapping.session_id != session_id:
            raise ValueError("mapping_session_mismatch")
        self.session_id, self.mapping = session_id, mapping
        self._queues: dict[str, collections.deque[CameraFrame]] = {camera: collections.deque() for camera in _CAMERAS}
        self._previous: dict[str, CameraFrame] = {}
        self._last_pair: CameraPair | None = None
        self._last_receive: int | None = None
        self._last_now: int | None = None
        self._reason = "awaiting_pair"
        self._invalidated = False

    def invalidate(self, reason: str) -> None:
        """Withdraw authority immediately on the caller's socket/decode failure."""
        if not isinstance(reason, str) or not reason or len(reason) > 128:
            raise ValueError("invalid transport failure reason")
        self._fail(reason, True)

    def update_clock_mapping(self, mapping: ClockMapping) -> None:
        """Refresh measured bounds without resetting per-camera anti-replay state."""
        if mapping.session_id != self.session_id or self._invalidated:
            raise TransportError("clock_refresh_requires_valid_session")
        if self.mapping is not None:
            old = self.mapping
            if mapping.clock_id != old.clock_id or mapping.measured_local_ns <= old.measured_local_ns:
                self._fail("clock_mapping_identity_regression", True)
                raise TransportError("clock_mapping_identity_regression")
            padding = math.ceil((mapping.measured_local_ns - old.measured_local_ns) * old.drift_ppm / 1_000_000)
            if (
                mapping.offset_low_ns > old.offset_high_ns + padding
                or mapping.offset_high_ns < old.offset_low_ns - padding
            ):
                self._fail("clock_mapping_discontinuity", True)
                raise TransportError("clock_mapping_discontinuity")
        self.mapping = mapping

    def _fail(self, reason: str, invalidate_clock: bool = False) -> None:
        self._reason = reason
        self.counters[reason] += 1
        self._last_pair = None
        for queue in self._queues.values():
            queue.clear()
        if invalidate_clock:
            self.mapping = None
            self._invalidated = True

    def _clock(self, capture_ns: int, now_ns: int) -> tuple[int, int] | None:
        if self.mapping is None:
            if not self._invalidated:
                self._reason = "clock_unmapped"
            return None
        try:
            bounds = self.mapping.local_bounds(capture_ns, now_ns)
        except TransportError as exc:
            self._fail(str(exc), str(exc) == "local_clock_regression")
            return None
        if bounds[1] - bounds[0] > self.config.max_clock_uncertainty_ns:
            self._fail("clock_uncertainty")
            return None
        return bounds

    def push(self, frame: CameraFrame, now_ns: int) -> CameraPair | None:
        _integer(now_ns, "now_ns")
        if self._last_now is not None and now_ns < self._last_now:
            self._fail("local_clock_regression", True)
            return None
        self._last_now = now_ns
        if frame.session_id != self.session_id:
            self._fail("session_mismatch")
            return None
        bounds = self._clock(frame.capture_ns, now_ns)
        if bounds is None:
            return None
        previous = self._previous.get(frame.camera)
        if previous is not None:
            for field, reason in (
                ("capture_ns", "capture_time_regression"),
                ("sequence", "sequence_regression"),
                ("frame_id", "frame_id_regression"),
            ):
                if getattr(frame, field) <= getattr(previous, field):
                    self._fail(reason, True)
                    return None
            self.counters["sequence_gap"] += frame.sequence - previous.sequence - 1
        self._previous[frame.camera] = frame
        if bounds[0] > now_ns or now_ns - bounds[0] > self.config.max_pair_age_ns:
            self._fail("future_frame" if bounds[0] > now_ns else "stale_frame")
            return None
        self._last_receive = now_ns
        queue = self._queues[frame.camera]
        if len(queue) == self.config.queue_capacity:
            queue.popleft()
            self.counters["queue_overflow"] += 1
            self._last_pair, self._reason = None, "queue_overflow"
        queue.append(frame)
        narrow, wide = self._queues["narrow"], self._queues["wide"]
        while narrow and wide:
            delta = narrow[0].capture_ns - wide[0].capture_ns
            if abs(delta) > self.config.max_capture_skew_ns:
                (narrow if delta < 0 else wide).popleft()
                self.counters["sync_drop"] += 1
                self._last_pair, self._reason = None, "sync_loss"
                continue
            left, right = narrow.popleft(), wide.popleft()
            left_bounds, right_bounds = self._clock(left.capture_ns, now_ns), self._clock(right.capture_ns, now_ns)
            if left_bounds is None or right_bounds is None:
                return None
            low, high = min(left_bounds[0], right_bounds[0]), max(left_bounds[1], right_bounds[1])
            if now_ns - low > self.config.max_pair_age_ns:
                self._fail("stale_pair")
                return None
            self._last_pair = CameraPair(frame.session_id, left, right, low, high, now_ns)
            self._reason = "ok"
            self.counters["pairs"] += 1
            return self._last_pair
        return None

    def health(self, now_ns: int) -> TransportHealth:
        _integer(now_ns, "now_ns")
        if self._last_now is not None and now_ns < self._last_now:
            self._fail("local_clock_regression", True)
        self._last_now = max(now_ns, self._last_now or now_ns)
        reason, age = self._reason, None
        if self.mapping is not None:
            capture = (
                self._last_pair.narrow.capture_ns
                if self._last_pair
                else self.mapping.measured_local_ns - self.mapping.offset_low_ns
            )
            if self._clock(capture, now_ns) is None:
                reason = self._reason
        if self._last_pair is not None:
            # Recompute mapping drift, so the original receive-time interval cannot stay fresh forever.
            bounds = [self._clock(item.capture_ns, now_ns) for item in (self._last_pair.narrow, self._last_pair.wide)]
            if all(item is not None for item in bounds):
                age = now_ns - min(item[0] for item in bounds if item is not None)
                if self._last_receive is not None and now_ns - self._last_receive > self.config.link_timeout_ns:
                    reason = "link_timeout"
                elif age > self.config.max_pair_age_ns:
                    reason = "stale_pair"
            else:
                reason = self._reason
        elif self.mapping is None and reason == "awaiting_pair":
            reason = "clock_unmapped"
        return TransportHealth(
            reason == "ok" and self._last_pair is not None, reason, self.session_id, age, dict(self.counters)
        )


def _json(packet: tuple[int, bytes, bytes], kind: int, keys: set[str]) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_key")
            value[key] = item
        return value

    if packet[0] != kind:
        raise TransportError("unexpected_packet_kind")
    try:
        value = json.loads(packet[1], object_pairs_hook=unique)
        if type(value) is not dict or set(value) != keys:
            raise ValueError("control_keys")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TransportError("invalid_control_metadata") from exc


class _Connection:
    def __init__(self, connection: socket.socket, codec: FrameCodec, clock: Callable[[], int]):
        self.socket, self.codec, self.clock = connection, codec, clock
        self._write_lock = threading.Lock()
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, wire: bytes) -> None:
        try:
            with self._write_lock:
                self.socket.sendall(wire)
        except OSError as exc:
            raise TransportError("send_failed") from exc

    def _control(self, kind: int, metadata: dict) -> None:
        self._send(self.codec.encode_packet(kind, json.dumps(metadata, separators=(",", ":")).encode()))

    def close(self) -> None:
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # shutdown on an already disconnected socket is expected; close still runs.
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CameraPublisher(_Connection):
    """One publisher connection; a bounded control thread services clock exchanges."""

    def __init__(
        self,
        connection: socket.socket,
        codec: FrameCodec,
        clock: Callable[[], int],
        clock_id: str,
        on_proposal: Callable[[str, bytes], None] | None = None,
    ):
        super().__init__(connection, codec, clock)
        hello = _json(codec.read_packet(connection), _HELLO, {"client_nonce", "clock_id"})
        _session(hello["client_nonce"])
        _clock_id(hello["clock_id"])
        self.session_id = str(uuid.uuid4())
        self._control(
            _WELCOME, {"client_nonce": hello["client_nonce"], "session_id": self.session_id, "clock_id": clock_id}
        )
        self._on_proposal = on_proposal
        self._closed, self._synced = threading.Event(), threading.Event()
        self.error: str | None = None
        self._thread = threading.Thread(target=self._controls, name="camera-clock-control", daemon=True)
        self._thread.start()

    def _controls(self) -> None:
        try:
            while not self._closed.is_set():
                incoming = self.codec.read_packet(self.socket)
                if incoming[0] == _PROPOSAL:
                    if incoming[1] != _PROPOSAL_METADATA or self._on_proposal is None:
                        raise TransportError("proposal_return_not_configured")
                    self._on_proposal(self.session_id, incoming[2])
                    continue
                request = _json(incoming, _CLOCK_REQUEST, {"session_id", "nonce", "local_send_ns"})
                remote_receive = self.clock()
                if request["session_id"] != self.session_id:
                    raise TransportError("session_mismatch")
                _session(request["nonce"])
                _integer(request["local_send_ns"], "local_send_ns")
                self._control(
                    _CLOCK_RESPONSE, dict(**request, remote_receive_ns=remote_receive, remote_send_ns=self.clock())
                )
                self._synced.set()
        except (TransportError, ValueError) as exc:
            if not self._closed.is_set():
                self.error = str(exc)
        finally:
            self._closed.set()
            super().close()

    def send_frame(self, frame: CameraFrame) -> None:
        if self._closed.is_set():
            raise TransportError(self.error or "link_closed")
        if frame.session_id != self.session_id:
            raise TransportError("session_mismatch")
        self._send(self.codec.encode_frame(frame))

    def send_context(self, record: NativeContextRecord) -> None:
        if self._closed.is_set():
            raise TransportError(self.error or "link_closed")
        if record.session_id != self.session_id:
            raise TransportError("session_mismatch")
        self._send(self.codec.encode_context(record))

    def wait_for_clock_sync(self, timeout: float) -> bool:
        return self._synced.wait(timeout)

    def wait_closed(self, timeout: float) -> bool:
        return self._closed.wait(timeout)

    def close(self) -> None:
        self._closed.set()
        super().close()
        if threading.current_thread() is not self._thread:
            self._thread.join(2)
            if self._thread.is_alive():
                raise TransportError("control_thread_shutdown_failed")


class CameraTransportServer:
    def __init__(
        self,
        host: str,
        port: int,
        key: bytes,
        *,
        clock_id: str,
        clock: Callable[[], int],
        timeout_s: float = 5,
        with_context: bool = False,
        with_return: bool = False,
        on_proposal: Callable[[str, bytes], None] | None = None,
    ):
        _clock_id(clock_id)
        self.codec, self.clock, self.clock_id = (
            FrameCodec(key, with_context=with_context, with_return=with_return),
            clock,
            clock_id,
        )
        self._on_proposal = on_proposal
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind((host, port))
            self.socket.listen(1)
            self.socket.settimeout(timeout_s)
            self.address = self.socket.getsockname()
            self.timeout_s = timeout_s
        except BaseException:
            self.socket.close()
            raise

    def accept(self) -> CameraPublisher:
        connection, _ = self.socket.accept()
        connection.settimeout(self.timeout_s)
        try:
            return CameraPublisher(connection, self.codec, self.clock, self.clock_id, self._on_proposal)
        except BaseException:
            connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.socket.close()


@dataclass(frozen=True)
class ReceivedFrame:
    frame: CameraFrame
    receive_ns: int


@dataclass(frozen=True)
class ReceivedContext:
    record: NativeContextRecord
    receive_ns: int


class CameraTransportClient(_Connection):
    """One reader; clock exchanges can buffer at most three interleaved camera frames."""

    session_id: str
    remote_clock_id: str
    _pending: collections.deque[ReceivedFrame | ReceivedContext]
    dropped_during_sync: int
    _last_context_sequence: int

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        key: bytes,
        *,
        clock_id: str,
        clock: Callable[[], int] = time.monotonic_ns,
        timeout_s: float = 5,
        with_context: bool = False,
        with_return: bool = False,
    ):
        _clock_id(clock_id)
        codec = FrameCodec(key, with_context=with_context, with_return=with_return)
        connection = socket.create_connection((host, port), timeout_s)
        client = cls(connection, codec, clock)
        try:
            nonce = str(uuid.uuid4())
            client._control(_HELLO, {"client_nonce": nonce, "clock_id": clock_id})
            welcome = _json(codec.read_packet(connection), _WELCOME, {"client_nonce", "session_id", "clock_id"})
            if welcome["client_nonce"] != nonce:
                raise TransportError("handshake_nonce_mismatch")
            _session(welcome["session_id"])
            _clock_id(welcome["clock_id"])
            client.session_id, client.remote_clock_id = welcome["session_id"], welcome["clock_id"]
            client._pending = collections.deque()
            client.dropped_during_sync = 0
            client._last_context_sequence = 0
            return client
        except BaseException:
            client.close()
            raise

    def _frame(self, packet: tuple[int, bytes, bytes], received: int) -> ReceivedFrame:
        frame = self.codec.frame_from_packet(packet)
        if frame.session_id != self.session_id:
            raise TransportError("session_mismatch")
        return ReceivedFrame(frame, received)

    def receive_frame(self) -> ReceivedFrame:
        received = self.receive_observation()
        if not isinstance(received, ReceivedFrame):
            raise TransportError("context_requires_receive_observation")
        return received

    def _observation(self, packet: tuple[int, bytes, bytes], received: int) -> ReceivedFrame | ReceivedContext:
        if packet[0] != _CONTEXT:
            return self._frame(packet, received)
        record = self.codec.context_from_packet(packet)
        if record.session_id != self.session_id:
            raise TransportError("session_mismatch")
        if record.sequence <= self._last_context_sequence:
            raise TransportError("context_sequence_regression")
        self._last_context_sequence = record.sequence
        return ReceivedContext(record, received)

    def send_proposal(self, packet: bytes) -> None:
        if not self.codec.with_return:
            raise TransportError("proposal_return_not_enabled")
        self._send(self.codec.encode_packet(_PROPOSAL, _PROPOSAL_METADATA, packet))

    def receive_observation(self) -> ReceivedFrame | ReceivedContext:
        if self._pending:
            return self._pending.popleft()
        packet = self.codec.read_packet(self.socket)
        return self._observation(packet, self.clock())

    def synchronize_clock(
        self, *, drift_ppm: float, max_age_ns: int, target_uncertainty_ns: int | None = None
    ) -> ClockMapping:
        """Measure the clock relation with four-timestamp exchanges.

        Without ``target_uncertainty_ns`` this is exactly one exchange. With it,
        up to ``_CLOCK_EXCHANGES`` independent exchanges run until one is at most
        that wide, and the tightest is returned. Each exchange is a conservative
        interval on its own, so the narrowest is still a bound; a delayed one
        (scheduling on either host) only widens its own interval. The caller's
        synchronizer still rejects a returned mapping wider than its limit.
        """
        if target_uncertainty_ns is not None:
            _integer(target_uncertainty_ns, "target_uncertainty_ns", 1)
        best = self._clock_exchange(drift_ppm=drift_ppm, max_age_ns=max_age_ns)
        for _ in range(_CLOCK_EXCHANGES - 1):
            if target_uncertainty_ns is None or best.offset_high_ns - best.offset_low_ns <= target_uncertainty_ns:
                break
            mapping = self._clock_exchange(drift_ppm=drift_ppm, max_age_ns=max_age_ns)
            if mapping.offset_high_ns - mapping.offset_low_ns <= best.offset_high_ns - best.offset_low_ns:
                best = mapping
        return best

    def _clock_exchange(self, *, drift_ppm: float, max_age_ns: int) -> ClockMapping:
        nonce, sent = str(uuid.uuid4()), self.clock()
        timeout = self.socket.gettimeout()
        deadline = time.monotonic() + (timeout if timeout is not None else 5.0)
        self._control(_CLOCK_REQUEST, {"session_id": self.session_id, "nonce": nonce, "local_send_ns": sent})
        for _ in range(128):
            packet, received = self.codec.read_packet(self.socket, deadline), self.clock()
            if packet[0] in (_CAMERA, _CONTEXT):
                capacity = 8 if self.codec.with_context else 3
                if len(self._pending) == capacity:
                    if self.codec.with_context:
                        raise TransportError("context_clock_buffer_overflow")
                    self._pending.popleft()
                    self.dropped_during_sync += 1
                self._pending.append(self._observation(packet, received))
                continue
            reply = _json(
                packet, _CLOCK_RESPONSE, {"session_id", "nonce", "local_send_ns", "remote_receive_ns", "remote_send_ns"}
            )
            if reply["session_id"] != self.session_id or reply["nonce"] != nonce or reply["local_send_ns"] != sent:
                raise TransportError("clock_reply_identity_mismatch")
            return ClockMapping.from_exchange(
                session_id=self.session_id,
                clock_id=self.remote_clock_id,
                local_send_ns=sent,
                local_receive_ns=received,
                remote_receive_ns=reply["remote_receive_ns"],
                remote_send_ns=reply["remote_send_ns"],
                drift_ppm=drift_ppm,
                max_age_ns=max_age_ns,
            )
        raise TransportError("clock_reply_starved")
