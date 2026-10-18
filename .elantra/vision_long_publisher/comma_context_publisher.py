"""Read-only, source-pinned v2 camera/native-context publisher.

Installation and native execution require separate deployment authorization. This
tool never changes params/services, publishes native messages, or transmits CAN.
The native receive queue has no loss counter; its history is always unqualified.

Two entry points share one session body. The standalone CLI (``main``) serves one
bounded diagnostic session. The embedded worker (``start_embedded_publisher``) lives
for the whole card process: it listens, serves a laptop session until that peer
disconnects, and listens again; each connection is a fresh authenticated session with
fresh legal-history state in the bridge. A crash restarts the worker after a bounded,
growing backoff; every restart and failure is counted and exposed in its status.
"""

import argparse
import hashlib
import logging
import math
import select
import socket
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .extract_companion_context import REQUIRED, extract_record
from .native_source_identity import verify_native_source_identity
from vision_long.camera_io import copy_nv12
from vision_long.device_applied import (
    DRIVES_ROOT,
    STATUS_FILE,
    DriveLog,
    StatusFile,
    drive_directory_name,
    from_handoff,
)
from vision_long.integration import CameraReference
from vision_long.native_bridge import NativeReturnBridge
from vision_long.native_context import (
    SOURCE_PINS,
    CameraIdentity,
    NativeContextRecord,
    NativeEvent,
    canonical_json,
)
from vision_long.transport import CameraFrame, CameraPublisher, CameraTransportServer, TransportError

MAX_NATIVE_MESSAGE_BYTES = 1024 * 1024
# How long the listening worker waits on the socket before checking stop and draining the outbox.
ACCEPT_POLL_S = 0.25
# The laptop must complete its first clock exchange this soon after connecting.
CLOCK_SYNC_TIMEOUT_S = 5
# Worker restart backoff: doubles per consecutive failure from the base, capped; a run that
# lasted the reset span starts the streak again.
RESTART_BACKOFF_BASE_S = 1.0
RESTART_BACKOFF_MAX_S = 30.0
RESTART_STREAK_RESET_S = 60.0
# opendbc.car.carlog's logger name: card.py forwards it to cloudlog in the car process.
CAR_PROCESS_LOGGER = "carlog"


def read_camera_frame(client: Any, camera: Literal["narrow", "wide"], session: str, sequence: int) -> CameraFrame:
    buffer = client.recv(100)
    if buffer is None or not client.valid:
        raise RuntimeError("Missing or invalid native camera frame")
    frame_id, sof = client.frame_id, client.timestamp_sof
    if buffer.frame_id != frame_id:
        raise RuntimeError("Native camera buffer generation changed before copy")
    payload = copy_nv12(buffer.data, buffer.width, buffer.height, buffer.stride, buffer.uv_offset)
    if buffer.frame_id != frame_id:
        raise RuntimeError("Native camera buffer generation changed during copy")
    # Stable generation is a completed-reuse check, not a lock against a writer
    # beginning its next generation while copying. Physical ownership is unknown.
    return CameraFrame(
        session, camera, sequence, frame_id, sof, buffer.width, buffer.height, buffer.width, "nv12", payload
    )


def verify_native_source(root: Path) -> None:
    verify_native_source_identity(root)
    for name, digest in SOURCE_PINS:
        source = root / name
        if (
            source.stat().st_size > MAX_NATIVE_MESSAGE_BYTES
            or hashlib.sha256(source.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("Native schema/API source bytes changed: " + name)


def receive_events(sockets: dict[str, Any], messaging: Any, sequence: int, limit: int = 8):
    """A bounded poll, without conflation or an unbounded drain-to-latest loop."""
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("native_receive_limit")
    events = []
    limited = False
    for service, subscriber in sockets.items():
        count = 0
        for _ in range(limit):
            raw = subscriber.receive(non_blocking=True)
            if raw is None:
                break
            if type(raw) is not bytes or not 0 < len(raw) <= MAX_NATIVE_MESSAGE_BYTES:
                raise ValueError("native_message_byte_limit")
            # Keep the installed current schema reader alive through extraction.
            with messaging.log.Event.from_bytes(raw, traversal_limit_in_words=128 * 1024) as message:
                if message.which() != service:
                    raise ValueError("native_service_socket_mismatch")
                sequence += 1
                selected = extract_record(message, "live", sequence)
            if selected is None:
                raise ValueError("native_service_extractor_missing")
            events.append(
                NativeEvent(
                    service, selected["log_mono_ns"], selected["valid"], canonical_json(selected["data"]), sequence
                )
            )
            count += 1
        # This does not consume a ninth event or assert that the queue is empty.
        limited |= count == limit
    return tuple(events), sequence, limited


class NativeSource(Protocol):
    """The device's road cameras and native services, opened once per worker run."""

    boot: str

    def read_frames(self, session_id: str, sequences: dict[str, int]) -> tuple[CameraFrame, CameraFrame]: ...

    def receive_events(self, sequence: int) -> tuple[tuple[NativeEvent, ...], int, bool]: ...


class DeviceSource:
    """camerad's narrow/wide road streams and the REQUIRED native services, on the pinned device."""

    def __init__(self) -> None:
        from msgq.visionipc import VisionIpcClient  # type: ignore[import-not-found]  # Device-only module.
        from openpilot.cereal import messaging  # type: ignore[import-not-found]
        from openpilot.cereal.visionipc import VisionStreamType  # type: ignore[import-not-found]

        self._clients: list[tuple[Literal["narrow", "wide"], Any]] = [
            ("narrow", VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, False)),
            ("wide", VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)),
        ]
        for _, client in self._clients:
            if not client.connect(False):
                raise RuntimeError("Native road-camera stream unavailable")
        self._messaging = messaging
        self._subscribers = {service: messaging.sub_sock(service, conflate=False) for service in REQUIRED}
        self.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def read_frames(self, session_id: str, sequences: dict[str, int]) -> tuple[CameraFrame, CameraFrame]:
        frames = []
        for camera, client in self._clients:
            sequences[camera] += 1
            frames.append(read_camera_frame(client, camera, session_id, sequences[camera]))
        return frames[0], frames[1]

    def receive_events(self, sequence: int) -> tuple[tuple[NativeEvent, ...], int, bool]:
        return receive_events(self._subscribers, self._messaging, sequence)


def require_pinned_device() -> None:
    if not (3, 12, 3) <= sys.version_info < (3, 13):
        raise RuntimeError("Pinned native project requires Python >=3.12.3,<3.13")
    if socket.gethostname() != "comma-6f0745e3":
        raise RuntimeError("Pinned comma identity required")
    verify_native_source(Path("/data/openpilot"))


def boottime_ns() -> int:
    """The camera SOF clock domain, read on the device."""
    if sys.platform != "linux":
        raise RuntimeError("CLOCK_BOOTTIME requires the Linux device")
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-ip", required=True)
    parser.add_argument("--port", type=int, default=45556)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2400)
    parser.add_argument("--max-duration-s", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 100000 or not math.isfinite(args.max_duration_s) or not 1 <= args.max_duration_s <= 6000:
        parser.error("Explicit bounded pair count and duration required")
    with args.key_file.open("rb") as stream:
        key = stream.read(4097)
    return run_publisher(
        listen_ip=args.listen_ip, port=args.port, key=key, pairs=args.pairs, max_duration_s=args.max_duration_s
    )


def publish_pair(publisher, frames, sequence, clock, bridge=None):
    """Register exact publisher-owned originals before either camera is sent."""
    if len(frames) != 2 or {frame.camera for frame in frames} != {"narrow", "wide"}:
        raise ValueError("actual_dual_camera_pair_required")
    if any(frame.session_id != publisher.session_id or frame.sequence != sequence for frame in frames):
        raise ValueError("publisher_pair_identity_mismatch")
    if bridge is not None:
        bridge.dispatch(
            publisher.session_id,
            sequence,
            tuple(CameraReference(frame.camera, frame.frame_id, frame.capture_ns) for frame in frames),
            clock(),
        )
    for frame in frames:
        publisher.send_frame(frame)


def _key(key: bytes) -> None:
    if type(key) is not bytes or not 32 <= len(key) <= 4096:
        raise ValueError("bounded_authentication_key_required")


class AppliedForwarder:
    """Worker side of the device-applied outbox: validate, send to the same session, log the drive."""

    def __init__(
        self,
        bridge: NativeReturnBridge,
        drive_log: DriveLog | None,
        *,
        status_file: StatusFile | None = None,
        restarts: Callable[[], int] = lambda: 0,
        boot_clock: Callable[[], int] | None = None,
    ):
        self.bridge = bridge
        self.drive_log = drive_log
        self.status_file = status_file
        self._restarts = restarts
        self._boot_clock = boot_clock
        self._status_line: str | None = None
        self.sent = 0

    def drain(self, publisher: CameraPublisher | None) -> None:
        for session, item in self.bridge.take_applied():
            # A record the strict validation refuses is a device defect: it raises and the
            # supervisor restarts the worker, which closes the session (exact native output).
            record = from_handoff(item)
            if publisher is not None and session == publisher.session_id:
                publisher.send_device_applied(record)
                self.sent += 1
            if self.drive_log is not None:
                self.drive_log.record(record)
        if self.drive_log is not None:
            line = self.bridge.status_line
            if line is not None and line != self._status_line:
                self._status_line = line
                self.drive_log.status(line)
            self.drive_log.tick()
        self._write_status(publisher is not None)

    def _write_status(self, laptop_session: bool) -> None:
        """About 1 Hz, worker thread only; the controller never reads this file."""
        status, state = self.status_file, self.bridge.status_fields
        if status is None or state is None or self._boot_clock is None:
            return
        mode, authority, reason, toggle = state
        returned = self.bridge.last_return_ns
        overruns, worst_ns = self.bridge.completion_stats()
        status.offer(
            {
                "boot_ns": self._boot_clock(),
                "mode": mode,
                "authority": authority,
                "reason": reason,
                "toggle": "on" if toggle else "off",
                "laptop_session": "up" if laptop_session else "down",
                "last_packet_age_ms": None if returned is None else (time.monotonic_ns() - returned) // 1_000_000,
                "publisher_restarts": self._restarts(),
                "drive_log_dir": None if self.drive_log is None else str(self.drive_log.folder),
                "drive_log_error": None if self.drive_log is None else self.drive_log.error,
                "completion_overruns": overruns,
                "worst_completion_us": worst_ns // 1000,
            }
        )
        self.bridge.status_file_unwritable = status.persistently_failing


def serve_session(
    publisher: CameraPublisher,
    source: NativeSource,
    clock: Callable[[], int],
    *,
    bridge: NativeReturnBridge | None,
    forwarder: AppliedForwarder | None,
    stop: threading.Event,
    pairs: int | None = None,
    deadline: float | None = None,
    between_pairs: Callable[[], None] | None = None,
) -> None:
    """One authenticated laptop session: pairs, native context, receiver echoes and applied records.

    Ends when the peer disconnects (a ``TransportError`` from the connection), on ``stop``, or
    at the optional diagnostic bounds. The bridge session is always closed on the way out.
    """
    camera_clock = source.boot + ":CLOCK_BOOTTIME"
    if bridge is not None:
        bridge.open_session(publisher.session_id, camera_clock)
    try:
        if not publisher.wait_for_clock_sync(CLOCK_SYNC_TIMEOUT_S):
            raise TransportError("peer_failed_initial_clock_exchange")
        sequences = {"narrow": 0, "wide": 0}
        event_sequence = 0
        pair_sequence = 0
        while not stop.is_set() and (pairs is None or pair_sequence < pairs):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("native_observation_duration_exceeded")
            pair_sequence += 1
            frames = source.read_frames(publisher.session_id, sequences)
            identities = tuple(CameraIdentity(f.camera, f.frame_id, f.capture_ns, f.sequence) for f in frames)
            publish_pair(publisher, frames, pair_sequence, clock, bridge)
            events, event_sequence, limited = source.receive_events(event_sequence)
            context = NativeContextRecord(
                publisher.session_id,
                source.boot,
                pair_sequence,
                source.boot + ":CLOCK_MONOTONIC",
                camera_clock,
                (identities[0], identities[1]),
                events,
                drain_limited=limited,
                receiver_echo=bridge.receiver_echo(clock()) if bridge is not None else None,
            )
            publisher.send_context(context)
            if forwarder is not None:
                forwarder.drain(publisher)
            if between_pairs is not None:
                between_pairs()
    finally:
        if bridge is not None:
            bridge.close_session(publisher.session_id)


def run_publisher(*, listen_ip, port, key, pairs, max_duration_s, stop=None):
    """Standalone diagnostic CLI: one bounded session, no bridge, no return protocol."""
    _key(key)
    if (
        type(pairs) is not int
        or not 1 <= pairs <= 100000
        or not math.isfinite(max_duration_s)
        or not 1 <= max_duration_s <= 6000
    ):
        raise ValueError("bounded_publisher_run_required")
    deadline = time.monotonic() + max_duration_s
    require_pinned_device()
    source = DeviceSource()
    stop = stop or threading.Event()
    with CameraTransportServer(
        listen_ip,
        port,
        key,
        clock_id=source.boot + ":CLOCK_BOOTTIME",
        clock=boottime_ns,
        with_context=True,
        timeout_s=1,
    ) as server:
        while not select.select([server.socket], [], [], min(ACCEPT_POLL_S, max(0, deadline - time.monotonic())))[0]:
            if stop.is_set():
                return 0
            if time.monotonic() >= deadline:
                raise TimeoutError("native_peer_wait_duration_exceeded")
        with server.accept() as publisher:
            serve_session(
                publisher, source, boottime_ns, bridge=None, forwarder=None, stop=stop, pairs=pairs, deadline=deadline
            )
    return 0


class SessionCounters:
    def __init__(self) -> None:
        self.sessions = 0
        self.session_ends = 0
        self.handshake_failures = 0
        self.refused_while_active = 0
        self.last_session_end: str | None = None


def serve_sessions(
    server: CameraTransportServer,
    source: NativeSource,
    clock: Callable[[], int],
    bridge: NativeReturnBridge,
    forwarder: AppliedForwarder,
    stop: threading.Event,
    counters: SessionCounters,
) -> None:
    """Listen, serve one laptop session at a time, listen again; never busy-loops.

    A peer that disconnects, fails the handshake or breaks the protocol ends only its own
    session (counted). Anything else, such as a camera stream failure, propagates to the
    supervisor, which restarts the worker after its backoff.

    Policy for strangers and health probes: while no session is active, a bare connect, a
    silent peer (closed after the handshake read deadline) or a failed authentication only
    closes that connection; none of it reaches the bridge, so nothing latches. There is at most
    one active session: any further connection is accepted and closed at once between pairs,
    unread, and counted in ``refused_while_active``; the active session is untouched.
    """

    def refuse_extra() -> None:
        while select.select([server.socket], [], [], 0)[0]:
            try:
                connection, _ = server.socket.accept()
            except OSError:
                # The stranger left between readiness and accept: nothing to close.
                counters.refused_while_active += 1
                return
            connection.close()
            counters.refused_while_active += 1

    while not stop.is_set():
        ready = select.select([server.socket], [], [], ACCEPT_POLL_S)[0]
        forwarder.drain(None)
        if not ready:
            continue
        try:
            publisher = server.accept()
        except (TransportError, OSError) as error:
            counters.handshake_failures += 1
            counters.last_session_end = "handshake:" + type(error).__name__
            continue
        counters.sessions += 1
        with publisher:
            try:
                serve_session(
                    publisher,
                    source,
                    clock,
                    bridge=bridge,
                    forwarder=forwarder,
                    stop=stop,
                    between_pairs=refuse_extra,
                )
                counters.last_session_end = "stopped"
            except TransportError as error:
                counters.last_session_end = "peer:" + str(error)[:64]
            finally:
                counters.session_ends += 1


class EmbeddedPublisher:
    """Supervised card-lifetime worker; stop never waits on the controller.

    ``run`` serves until ``stop`` is set. When it raises (or returns early) the failure is
    counted, logged once on carlog, and the run restarts after a backoff that doubles per
    consecutive failure up to ``RESTART_BACKOFF_MAX_S``. Only ``stop`` ends the worker.
    """

    def __init__(
        self,
        run: Callable[[threading.Event, SessionCounters], None],
        *,
        backoff_base_s: float = RESTART_BACKOFF_BASE_S,
        backoff_max_s: float = RESTART_BACKOFF_MAX_S,
        log: logging.Logger | None = None,
    ):
        if not 0 < backoff_base_s <= backoff_max_s <= RESTART_BACKOFF_MAX_S:
            raise ValueError("bounded_restart_backoff_required")
        self._stop = threading.Event()
        self._run_once = run
        self._base, self._max = backoff_base_s, backoff_max_s
        self._log = log or logging.getLogger(CAR_PROCESS_LOGGER)
        self.counters = SessionCounters()
        self.status = "starting"
        self.error: str | None = None
        self.failures = 0
        self.restarts = 0
        self._thread = threading.Thread(target=self._run, name="vision-long-publisher", daemon=True)
        self._thread.start()

    def backoff_s(self, streak: int) -> float:
        return min(self._max, self._base * 2 ** max(0, streak - 1))

    def _run(self) -> None:
        streak = 0
        while not self._stop.is_set():
            started = time.monotonic()
            self.status = "running"
            try:
                self._run_once(self._stop, self.counters)
                if self._stop.is_set():
                    break
                raise RuntimeError("publisher_returned_without_stop")
            except Exception as error:
                # Counted, logged and restarted after a backoff: never silent, never a busy loop.
                self.failures += 1
                self.error = (type(error).__name__ + ":" + str(error))[:120]
                streak = 1 if time.monotonic() - started >= RESTART_STREAK_RESET_S else streak + 1
                delay = self.backoff_s(streak)
                self.status = "restarting"
                self._log.warning(
                    "vision_long_publisher failure=%d restart_in_s=%.1f error=%s", self.failures, delay, self.error
                )
                if self._stop.wait(delay):
                    break
                self.restarts += 1
        self.status = "stopped"

    @property
    def summary(self) -> str:
        """Status-reason suffix: restarts and failures, and the last error kind."""
        if not self.failures:
            return ""
        kind = (self.error or "").split(":", 1)[0]
        return f"publisher_failures={self.failures}:restarts={self.restarts}:{kind}"

    def stop(self) -> None:
        self._stop.set()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float) -> None:
        if not math.isfinite(timeout) or not 0 <= timeout <= 10:
            raise ValueError("bounded_off_controller_join_required")
        self._thread.join(timeout)


def start_embedded_publisher(
    bridge: NativeReturnBridge,
    *,
    listen_ip: str,
    port: int,
    key: bytes,
    drives_root: str = DRIVES_ROOT,
    status_path: str = STATUS_FILE,
) -> EmbeddedPublisher:
    """The card-lifetime device worker: pinned identity, camera source, server, then sessions."""
    if not isinstance(bridge, NativeReturnBridge):
        raise ValueError("native_owned_bridge_required")
    _key(key)
    drive_log: list[DriveLog] = []
    status_file = StatusFile(status_path)
    worker: list[EmbeddedPublisher] = []

    def run(stop: threading.Event, counters: SessionCounters) -> None:
        require_pinned_device()
        source = DeviceSource()
        if not drive_log:
            # One log per card start (one drive), kept across worker restarts.
            drive_log.append(
                DriveLog(Path(drives_root), drive_directory_name(datetime.now(UTC).timestamp(), source.boot))
            )
        forwarder = AppliedForwarder(
            bridge,
            drive_log[0],
            status_file=status_file,
            restarts=lambda: worker[0].restarts if worker else 0,
            boot_clock=boottime_ns,
        )
        with CameraTransportServer(
            listen_ip,
            port,
            key,
            clock_id=source.boot + ":CLOCK_BOOTTIME",
            clock=boottime_ns,
            with_context=True,
            with_return=True,
            on_proposal=bridge.return_packet,
            timeout_s=1,
        ) as server:
            serve_sessions(server, source, boottime_ns, bridge, forwarder, stop, counters)

    worker.append(EmbeddedPublisher(run))
    return worker[0]


if __name__ == "__main__":
    raise SystemExit(main())
