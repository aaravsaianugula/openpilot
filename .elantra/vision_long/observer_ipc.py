"""Private observer pipes: bounded JSON metadata only, never pickle or camera pixels.

One outstanding request per child. A parent supervisor independently bounds partial
reads and writes and kills/reaps a stuck child before releasing pipe resources.
"""

import os
import struct
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import BinaryIO, Protocol

from vision_long.device_applied import from_handoff
from vision_long.integration import CameraReference
from vision_long.native_context import ReceiverStateEcho, canonical_json, identity, integer, read_json

MAX_MESSAGE_BYTES = 64 * 1024
DISPATCH_MAX_AGE_NS = 50_000_000
VERSION = 1


class ObserverError(RuntimeError):
    pass


def fields(value: object, names: set[str]) -> dict:
    if type(value) is not dict or value.keys() != names:
        raise ObserverError("observer_fields")
    return value


def envelope(incarnation: str, sequence: int, session: str | None, kind: str, body: dict) -> dict:
    return {
        "version": VERSION,
        "incarnation": incarnation,
        "sequence": sequence,
        "session": session,
        "kind": kind,
        "body": body,
    }


def validate(message: object, incarnation: str, sequence: int) -> dict:
    row = fields(message, {"version", "incarnation", "sequence", "session", "kind", "body"})
    if type(row["version"]) is not int or row["version"] != VERSION or row["incarnation"] != incarnation:
        raise ObserverError("observer_incarnation_or_version")
    integer(row["sequence"], "request_sequence", 1, 2**63 - 1)
    if row["sequence"] != sequence:
        raise ObserverError("observer_request_order")
    if row["session"] is not None:
        identity(row["session"])
    if type(row["kind"]) is not str or type(row["body"]) is not dict:
        raise ObserverError("observer_envelope_types")
    return row


def read_message(stream: BinaryIO, started: Callable[[], None] = lambda: None) -> dict:
    # Idle EOF is observed even without an outstanding child RPC. Once the first
    # byte arrives, the caller's total-operation deadline never resets on progress.
    first = stream.read(1)
    if not first:
        raise EOFError("observer_pipe_EOF")
    started()

    def exact(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = stream.read(size - len(chunks))
            if not part:
                raise EOFError("observer_truncated_frame")
            chunks.extend(part)
        return bytes(chunks)

    size = struct.unpack("!I", first + exact(3))[0]
    if not 1 <= size <= MAX_MESSAGE_BYTES:
        raise ObserverError("observer_frame_limit")
    message = read_json(exact(size))
    if type(message) is not dict:
        raise ObserverError("observer_object_required")
    return message


def write_message(stream: BinaryIO, message: dict) -> None:
    payload = canonical_json(message)
    if not 1 <= len(payload) <= MAX_MESSAGE_BYTES:
        raise ObserverError("observer_frame_limit")
    data = memoryview(struct.pack("!I", len(payload)) + payload)
    while data:
        written = stream.write(data)
        if written is None or written <= 0:
            raise EOFError("observer_pipe_write_EOF")
        data = data[written:]
    stream.flush()


class PublisherBridge(Protocol):
    status_line: str | None
    status_fields: tuple[str, str, str, bool] | None
    last_return_ns: int | None
    status_file_unwritable: bool

    def open_session(self, session_id: str, camera_clock_id: str) -> bool: ...
    def dispatch(
        self, session_id: str, input_sequence: int, cameras: tuple[CameraReference, ...], dispatched_ns: int
    ) -> bool: ...
    def return_packet(self, session_id: str, packet: bytes) -> bool: ...
    def close_session(self, session_id: str) -> bool: ...
    def receiver_echo(self, now_ns: int) -> ReceiverStateEcho | None: ...
    def take_applied(self) -> list[tuple[str, tuple]]: ...
    def completion_stats(self) -> tuple[int, int]: ...


class RemoteBridge:
    """Child-only serial RPC proxy; no authority object exists in this process.

    An IPC defect poisons the proxy permanently. The parent watchdog kills this
    incarnation if a complete RPC cannot finish; no retry can refresh old evidence.
    """

    def __init__(self, reader: BinaryIO, writer: BinaryIO, incarnation: str, clock: Callable[[], int]):
        identity(incarnation)
        self.reader, self.writer, self.incarnation, self.clock = reader, writer, incarnation, clock
        self._lock = threading.RLock()
        self._sequence = 0
        self._failed = False
        self._session: str | None = None
        self._clock_id: str | None = None
        self._build: str | None = None
        self.status_line: str | None = None
        self.status_fields: tuple[str, str, str, bool] | None = None
        self.last_return_ns: int | None = None
        self.status_file_unwritable = False
        self._stats = (0, 0)

    @contextmanager
    def _operation(self):
        with self._lock:
            if self._failed:
                raise ObserverError("observer_proxy_poisoned")
            try:
                yield
            except (OSError, EOFError, ValueError, TypeError, ObserverError):
                self._failed = True
                raise

    def _call(self, kind: str, session: str | None, body: dict) -> dict:
        with self._lock:
            if self._failed:
                raise ObserverError("observer_proxy_poisoned")
            self._sequence += 1
            try:
                write_message(self.writer, envelope(self.incarnation, self._sequence, session, kind, body))
                response = validate(read_message(self.reader), self.incarnation, self._sequence)
                if response["session"] != session or response["kind"] != kind + "_ack":
                    raise ObserverError("observer_response_binding")
                return response["body"]
            except (OSError, EOFError, ValueError, ObserverError):
                self._failed = True
                raise

    def hello(self, boot_id: str, build_id: str) -> dict:
        with self._operation():
            if self._sequence != 0:
                raise ObserverError("observer_repeated_hello")
            body = self._call(
                "hello", None, {"boot_id": boot_id, "build_id": build_id, "pid": os.getpid(), "clock_ns": self.clock()}
            )
            fields(body, {"clock_ns", "configuration"})
            integer(body["clock_ns"], "parent_clock", 0, 2**63 - 1)
            if not 0 <= self.clock() - body["clock_ns"] <= DISPATCH_MAX_AGE_NS:
                raise ObserverError("observer_parent_clock_binding")
            if type(body["configuration"]) is not dict:
                raise ObserverError("observer_configuration")
            self._clock_id, self._build = boot_id + ":CLOCK_BOOTTIME", build_id
            return body["configuration"]

    def _accepted(self, kind: str, session: str, body: dict) -> bool:
        response = fields(self._call(kind, session, body), {"accepted"})
        if response["accepted"] is not True:
            self._failed = True
            raise ObserverError("observer_enqueue_rejected")
        return True

    def open_session(self, session_id: str, camera_clock_id: str) -> bool:
        with self._operation():
            if self._clock_id is None or self._session is not None or camera_clock_id != self._clock_id:
                raise ObserverError("observer_open_binding")
            self._accepted("open", session_id, {"clock_id": camera_clock_id})
            self._session = session_id
            return True

    def dispatch(
        self, session_id: str, input_sequence: int, cameras: tuple[CameraReference, ...], dispatched_ns: int
    ) -> bool:
        with self._operation():
            if session_id != self._session:
                raise ObserverError("observer_dispatch_session")
            self._accepted(
                "dispatch",
                session_id,
                {
                    "sequence": input_sequence,
                    "dispatched_ns": dispatched_ns,
                    "cameras": [
                        {"camera": c.camera, "frame_id": c.frame_id, "capture_ns": c.capture_ns} for c in cameras
                    ],
                },
            )
            if not 0 <= self.clock() - dispatched_ns <= DISPATCH_MAX_AGE_NS:
                self._failed = True
                raise ObserverError("observer_dispatch_ack_stale")
            return True

    def return_packet(self, session_id: str, packet: bytes) -> bool:
        with self._operation():
            # A late network callback for an already closed transport cannot affect
            # the next session. Every in-session packet retains its original bytes.
            if session_id != self._session:
                return False
            if type(packet) is not bytes or not 1 <= len(packet) <= 16384:
                raise ObserverError("observer_proposal_limit")
            return self._accepted("return", session_id, {"packet": packet.hex()})

    def close_session(self, session_id: str) -> bool:
        with self._operation():
            try:
                return self._accepted("close", session_id, {})
            finally:
                self._session = None

    def receiver_echo(self, now_ns: int) -> ReceiverStateEcho | None:
        with self._operation():
            body = fields(self._call("echo", self._session, {}), {"echo"})
            if body["echo"] is None:
                return None
            echo = ReceiverStateEcho.from_dict(body["echo"])
            if echo.session_id != self._session or echo.clock_id != self._clock_id or echo.build_id != self._build:
                raise ObserverError("observer_echo_binding")
            # Keep the parent-generated timestamp. An expired reply is unknown,
            # never a newly generated echo at the child's receipt time.
            current = self.clock()
            return echo if echo.echo_ns <= current < echo.next_handoff_ns else None

    def take_applied(self) -> list[tuple[str, tuple]]:
        with self._operation():
            body = fields(
                self._call("snapshot", self._session, {"status_file_unwritable": self.status_file_unwritable}),
                {"applied", "status_line", "status_fields", "last_return_ns", "completion_stats"},
            )
            rows = body["applied"]
            if type(rows) is not list or len(rows) > 64:
                raise ObserverError("observer_applied_bound")
            result = []
            for row in rows:
                if type(row) is not list or len(row) != 2 or type(row[1]) is not list or len(row[1]) != 12:
                    raise ObserverError("observer_applied_shape")
                identity(row[0])
                item = tuple(row[1])
                from_handoff(item)
                result.append((row[0], item))
            line, state = body["status_line"], body["status_fields"]
            if line is not None and (type(line) is not str or len(line) > 4096):
                raise ObserverError("observer_status_line")
            if state is not None and (
                type(state) is not list
                or len(state) != 4
                or any(type(x) is not str or len(x) > 1024 for x in state[:3])
                or type(state[3]) is not bool
            ):
                raise ObserverError("observer_status_fields")
            returned = body["last_return_ns"]
            if returned is not None:
                integer(returned, "last_return", 0, 2**63 - 1)
            stats = body["completion_stats"]
            if type(stats) is not list or len(stats) != 2:
                raise ObserverError("observer_completion_stats")
            for value in stats:
                integer(value, "completion_stat", 0, 2**63 - 1)
            self.status_line, self.status_fields = line, None if state is None else tuple(state)
            self.last_return_ns, self._stats = returned, (stats[0], stats[1])
            return result

    def completion_stats(self) -> tuple[int, int]:
        return self._stats
