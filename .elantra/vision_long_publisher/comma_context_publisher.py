"""Read-only, source-pinned v2 camera/native-context publisher.

Installation and native execution require separate deployment authorization. This
tool never changes params/services, publishes native messages, or transmits CAN.
The native receive queue has no loss counter; its history is always unqualified.
"""

import argparse
import hashlib
import json
import math
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from .extract_companion_context import REQUIRED, extract_record
from vision_long.camera_io import copy_nv12
from vision_long.integration import CameraReference
from vision_long.native_bridge import NativeReturnBridge
from vision_long.native_context import (
    BUILD_PINS,
    SOURCE_PINS,
    CameraIdentity,
    NativeContextRecord,
    NativeEvent,
    canonical_json,
)
from vision_long.transport import CameraFrame, CameraTransportServer

MAX_NATIVE_MESSAGE_BYTES = 1024 * 1024


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
    directories = {"openpilot": root, "opendbc": root / "opendbc_repo", "msgq": root / "msgq_repo"}
    for name, commit in BUILD_PINS:
        actual = subprocess.run(
            ["git", "-C", str(directories[name]), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        if actual.stdout.strip() != commit:
            raise RuntimeError("Native integration source identity changed: " + name)
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


def run_publisher(*, listen_ip, port, key, pairs, max_duration_s, bridge=None, stop=None):
    if type(key) is not bytes or not 32 <= len(key) <= 4096:
        raise ValueError("bounded_authentication_key_required")
    if (
        type(pairs) is not int
        or not 1 <= pairs <= 100000
        or not math.isfinite(max_duration_s)
        or not 1 <= max_duration_s <= 6000
    ):
        raise ValueError("bounded_publisher_run_required")
    deadline = time.monotonic() + max_duration_s
    if not (3, 12, 3) <= sys.version_info < (3, 13):
        raise RuntimeError("Pinned native project requires Python >=3.12.3,<3.13")
    if socket.gethostname() != "comma-6f0745e3":
        raise RuntimeError("Pinned comma identity required")
    verify_native_source(Path("/data/openpilot"))
    from msgq.visionipc import VisionIpcClient  # type: ignore[import-not-found]  # Device-only module.
    from openpilot.cereal import messaging  # type: ignore[import-not-found]
    from openpilot.cereal.visionipc import VisionStreamType  # type: ignore[import-not-found]

    clients: list[tuple[Literal["narrow", "wide"], VisionIpcClient]] = [
        ("narrow", VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, False)),
        ("wide", VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)),
    ]
    for _, client in clients:
        if not client.connect(False):
            raise RuntimeError("Native road-camera stream unavailable")
    subscribers = {service: messaging.sub_sock(service, conflate=False) for service in REQUIRED}
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def clock():
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    with CameraTransportServer(
        listen_ip,
        port,
        key,
        clock_id=boot + ":CLOCK_BOOTTIME",
        clock=clock,
        with_context=True,
        with_return=bridge is not None,
        on_proposal=bridge.return_packet if bridge is not None else None,
        timeout_s=1,
    ) as server:
        # Wait for one explicit peer without blocking the native control thread
        # or repeatedly attempting failed connections. No automatic reconnect.
        while not select.select([server.socket], [], [], min(0.25, max(0, deadline - time.monotonic())))[0]:
            if stop is not None and stop.is_set():
                return 0
            if time.monotonic() >= deadline:
                raise TimeoutError("native_peer_wait_duration_exceeded")
        with server.accept() as publisher:
            try:
                if bridge is not None:
                    bridge.open_session(publisher.session_id, boot + ":CLOCK_BOOTTIME")
                if not publisher.wait_for_clock_sync(5):
                    raise RuntimeError("Peer failed initial measured clock exchange")
                sequences = dict(narrow=0, wide=0)
                event_sequence = 0
                for pair_sequence in range(1, pairs + 1):
                    if stop is not None and stop.is_set():
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native_observation_duration_exceeded")
                    identities = []
                    frames = []
                    for camera, client in clients:
                        sequences[camera] += 1
                        frame = read_camera_frame(client, camera, publisher.session_id, sequences[camera])
                        identities.append(CameraIdentity(camera, frame.frame_id, frame.capture_ns, frame.sequence))
                        frames.append(frame)
                    publish_pair(publisher, tuple(frames), pair_sequence, clock, bridge)
                    events, event_sequence, limited = receive_events(subscribers, messaging, event_sequence)
                    context = NativeContextRecord(
                        publisher.session_id,
                        boot,
                        pair_sequence,
                        boot + ":CLOCK_MONOTONIC",
                        boot + ":CLOCK_BOOTTIME",
                        (identities[0], identities[1]),
                        events,
                        drain_limited=limited,
                        receiver_echo=bridge.receiver_echo(clock()) if bridge is not None else None,
                    )
                    publisher.send_context(context)
                    if pair_sequence % 20 == 0:
                        print(
                            json.dumps(
                                dict(
                                    session_id=publisher.session_id,
                                    sent=sequences,
                                    context=pair_sequence,
                                    native_events=event_sequence,
                                    native_queue_history="unknown_possible_loss",
                                    source_clock_relation="unknown",
                                    physical_capture_delay="unknown",
                                    control_qualified=False,
                                )
                            ),
                            flush=True,
                        )
            finally:
                if bridge is not None:
                    bridge.close_session(publisher.session_id)
    return 0


class EmbeddedPublisher:
    """Explicit bounded worker lifecycle; stop never waits on the controller."""

    def __init__(self, bridge, settings):
        self._stop = threading.Event()
        self.status = "starting"
        self.error = None
        self._thread = threading.Thread(
            target=self._run, args=(bridge, settings), name="vision-long-publisher", daemon=True
        )
        self._thread.start()

    def _run(self, bridge, settings):
        try:
            self.status = "running"
            run_publisher(bridge=bridge, stop=self._stop, **settings)
            self.status = "stopped" if self._stop.is_set() else "complete"
        except Exception as error:
            self.error = type(error).__name__ + ": " + str(error)
            self.status = "failed"

    def stop(self):
        self._stop.set()

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout):
        if not math.isfinite(timeout) or not 0 <= timeout <= 10:
            raise ValueError("bounded_off_controller_join_required")
        self._thread.join(timeout)


def start_embedded_publisher(bridge: NativeReturnBridge, *, listen_ip, port, key, pairs, max_duration_s):
    if not isinstance(bridge, NativeReturnBridge):
        raise ValueError("native_owned_bridge_required")
    return EmbeddedPublisher(
        bridge, dict(listen_ip=listen_ip, port=port, key=key, pairs=pairs, max_duration_s=max_duration_s)
    )


if __name__ == "__main__":
    raise SystemExit(main())
