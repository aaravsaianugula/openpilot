"""Card-side observer supervision. No camera imports, pixels, or controller waits.

Only this background supervisor launches/polls/reaps children. Its pipe worker
registers metadata on the existing bounded bridge; the controller reads one cached
readiness boolean. A fresh process never reconstructs the card's fault latch.
"""

import logging
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import BinaryIO

from vision_long.integration import CameraReference
from vision_long.native_bridge import NativeReturnBridge
from vision_long.native_context import identity, integer
from vision_long.observer_ipc import (
    DISPATCH_MAX_AGE_NS,
    ObserverError,
    envelope,
    fields,
    read_message,
    validate,
    write_message,
)


class ParentBroker:
    """One incarnation, one active transport session, one ordered pipe worker."""

    def __init__(
        self,
        bridge: NativeReturnBridge,
        *,
        incarnation: str,
        pid: int,
        boot_id: str,
        build_id: str,
        clock: Callable[[], int],
        configuration: dict,
    ):
        self.bridge, self.incarnation, self.pid = bridge, incarnation, pid
        self.boot_id, self.build_id, self.clock = boot_id, build_id, clock
        self.configuration = configuration
        self.sequence = 0
        self.session: str | None = None
        self._sessions: deque[str] = deque(maxlen=64)
        self.ready = False

    def handle(self, raw: dict) -> dict:
        request = validate(raw, self.incarnation, self.sequence + 1)
        self.sequence += 1
        kind, session, body = request["kind"], request["session"], request["body"]
        if kind == "hello":
            fields(body, {"boot_id", "build_id", "pid", "clock_ns"})
            integer(body["pid"], "child_pid", 1)
            integer(body["clock_ns"], "child_clock", 0, 2**63 - 1)
            now = self.clock()
            if (
                self.sequence != 1
                or session is not None
                or body["boot_id"] != self.boot_id
                or body["build_id"] != self.build_id
                or body["pid"] != self.pid
                or not 0 <= now - body["clock_ns"] <= DISPATCH_MAX_AGE_NS
            ):
                raise ObserverError("observer_hello_binding")
            self.ready = True
            result = {"clock_ns": now, "configuration": self.configuration}
        elif not self.ready:
            raise ObserverError("observer_hello_required")
        elif kind == "open":
            fields(body, {"clock_id"})
            if session is None or self.session is not None or session in self._sessions:
                raise ObserverError("observer_session_reused_or_active")
            if body["clock_id"] != self.boot_id + ":CLOCK_BOOTTIME":
                raise ObserverError("observer_session_clock")
            self._sessions.append(session)
            self.session = session
            result = {"accepted": self.bridge.open_session(session, body["clock_id"])}
        else:
            if session != self.session:
                raise ObserverError("observer_session_binding")
            result = self._session_request(kind, body)
        return envelope(self.incarnation, self.sequence, session, kind + "_ack", result)

    def _session_request(self, kind: str, body: dict) -> dict:
        if kind == "snapshot":
            fields(body, {"status_file_unwritable"})
            if type(body["status_file_unwritable"]) is not bool:
                raise ObserverError("observer_status_type")
            self.bridge.status_file_unwritable = body["status_file_unwritable"]
            state = self.bridge.status_fields
            return {
                "applied": [[session, list(item)] for session, item in self.bridge.take_applied()],
                "status_line": self.bridge.status_line,
                "status_fields": None if state is None else list(state),
                "last_return_ns": self.bridge.last_return_ns,
                "completion_stats": list(self.bridge.completion_stats()),
            }
        if kind == "echo":
            fields(body, set())
            echo = None if self.session is None else self.bridge.receiver_echo(self.clock())
            if echo is not None and echo.session_id != self.session:
                echo = None  # Open is queued but not staged: never disclose the preceding session.
            return {"echo": None if echo is None else echo.to_dict()}
        if self.session is None:
            raise ObserverError("observer_active_session_required")
        if kind == "close":
            fields(body, set())
            accepted = self.bridge.close_session(self.session)
            self.session = None
            return {"accepted": accepted}
        if kind == "dispatch":
            fields(body, {"sequence", "dispatched_ns", "cameras"})
            integer(body["sequence"], "pair_sequence", 1, 2**63 - 1)
            integer(body["dispatched_ns"], "dispatch_clock", 0, 2**63 - 1)
            rows = body["cameras"]
            if type(rows) is not list or len(rows) != 2:
                raise ObserverError("observer_dual_camera_required")
            cameras = tuple(CameraReference(**fields(row, {"camera", "frame_id", "capture_ns"})) for row in rows)
            if (
                {c.camera for c in cameras} != {"narrow", "wide"}
                or any(c.capture_ns > body["dispatched_ns"] for c in cameras)
                or not 0 <= self.clock() - body["dispatched_ns"] <= DISPATCH_MAX_AGE_NS
            ):
                raise ObserverError("observer_dispatch_clock_or_camera")
            return {"accepted": self.bridge.dispatch(self.session, body["sequence"], cameras, body["dispatched_ns"])}
        if kind == "return":
            fields(body, {"packet"})
            value = body["packet"]
            if type(value) is not str or not 2 <= len(value) <= 32768 or len(value) % 2:
                raise ObserverError("observer_proposal_limit")
            packet = bytes.fromhex(value)
            if packet.hex() != value:
                raise ObserverError("observer_proposal_encoding")
            return {"accepted": self.bridge.return_packet(self.session, packet)}
        raise ObserverError("observer_unknown_request")

    def revoke(self) -> None:
        self.ready = False
        if self.session is not None:
            self.bridge.close_session(self.session)
            self.session = None


class IsolatedPublisher:
    """Cached handle for the controller; all process work is asynchronous.

    Pipes have one worker and a separate watchdog. Partial reads and full reply
    writes share a single total deadline. Cleanup kills/reaps before joining the
    worker; if either fails this handle permanently stops, rather than leaking
    another worker on every restart. A normal laptop disconnect stays in-child.
    """

    def __init__(
        self,
        bridge: NativeReturnBridge,
        *,
        command: list[str],
        boot_id: str,
        build_id: str,
        clock: Callable[[], int],
        configuration: dict | None = None,
        startup_timeout_s: float = 5,
        operation_timeout_s: float = 1,
        restart: bool = True,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        identity(boot_id)
        if not command or startup_timeout_s <= 0 or operation_timeout_s <= 0:
            raise ValueError("observer_supervisor_configuration")
        self.bridge, self.command, self.boot_id, self.build_id, self.clock = bridge, command, boot_id, build_id, clock
        self.configuration = {} if configuration is None else configuration
        self.startup_timeout_s, self.operation_timeout_s, self.restart = startup_timeout_s, operation_timeout_s, restart
        self.cwd, self.env = cwd, env
        self.failures = 0
        self.restarts = 0
        self.last_pid: int | None = None
        self.summary = "observer_starting"
        self._ready = False
        self._stop = threading.Event()
        self._revoked = threading.Event()
        self._revoked.set()
        self._thread = threading.Thread(target=self._run, name="vision-long-observer-supervisor", daemon=True)
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        # Revocation is authoritative even if a pipe worker evaluated readiness
        # before revocation and publishes that old True afterward.
        return self._ready and not self._stop.is_set() and not self._revoked.is_set()

    def stop(self) -> None:
        self._ready = False
        self._stop.set()

    def join(self, timeout: float = 5) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        streak = 0
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                clean = self._incarnation()
            except (OSError, ValueError, ObserverError) as error:
                clean = True  # Spawn failed before any worker/child existed.
                self.summary = "observer_spawn_failed:" + type(error).__name__
            self._ready = False
            if self._stop.is_set():
                break
            self.failures += 1
            logging.getLogger("carlog").error("vision_long %s", self.summary)
            if not clean or not self.restart:
                break
            streak = 0 if time.monotonic() - began >= 60 else streak + 1
            if self._stop.wait(min(30, 2 ** min(max(streak - 1, 0), 5))):
                break
            self.restarts += 1

    def _incarnation(self) -> bool:
        incarnation = str(uuid.uuid4())
        command = [*self.command, "--incarnation", incarnation, "--boot-id", self.boot_id, "--build-id", self.build_id]
        broker = ParentBroker(
            self.bridge,
            incarnation=incarnation,
            pid=0,  # Filled from the actual exec result before the pipe worker starts.
            boot_id=self.boot_id,
            build_id=self.build_id,
            clock=self.clock,
            configuration={**self.configuration, "restarts": self.restarts},
        )
        done = threading.Event()
        revoked = threading.Event()
        self._revoked = revoked
        deadline: list[float | None] = [time.monotonic() + self.startup_timeout_s]

        def started() -> None:
            operation_limit = time.monotonic() + self.operation_timeout_s
            deadline[0] = operation_limit if deadline[0] is None else min(deadline[0], operation_limit)

        def exchange(reader: BinaryIO, writer: BinaryIO) -> None:
            try:
                while not self._stop.is_set() and not revoked.is_set():
                    message = read_message(reader, started)
                    response = broker.handle(message)
                    write_message(writer, response)
                    if deadline[0] is not None and time.monotonic() > deadline[0]:
                        raise ObserverError("observer_RPC_timeout")
                    deadline[0] = None
                    self._ready = broker.ready and not self._stop.is_set() and not revoked.is_set()
                    self.summary = "observer_ready"
            except Exception as error:
                # Optional adapter boundary: all implementation/protocol failures
                # withdraw and report; no failure is converted to a valid message.
                if not revoked.is_set():
                    self.summary = "observer_failed:" + type(error).__name__ + ":" + str(error)[:256]
            finally:
                self._ready = False
                broker.revoke()
                done.set()

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            cwd=self.cwd,
            env=self.env,
            close_fds=True,
        )
        worker = None
        try:
            self.last_pid = broker.pid = process.pid
            assert process.stdin is not None and process.stdout is not None
            worker = threading.Thread(
                target=exchange, args=(process.stdout, process.stdin), name="vision-long-observer-pipe", daemon=True
            )
            worker.start()
            while not done.wait(0.02):
                if self._stop.is_set():
                    break
                if process.poll() is not None:
                    self.summary = "observer_failed:EOF:child_exited"
                    break
                limit = deadline[0]
                if limit is not None and time.monotonic() > limit:
                    self.summary = "observer_failed:RPC_timeout"
                    break
        except Exception as error:
            self.summary = "observer_setup_failed:" + type(error).__name__
        finally:
            # Withdraw before any blocking cleanup, including failed thread.start.
            revoked.set()
            self._ready = False
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            self.summary = "observer_cleanup_failed:process"
            return False
        if worker is not None and worker.ident is not None:
            worker.join(timeout=2)
            if worker.is_alive():
                self.summary = "observer_cleanup_failed:pipe_worker"
                return False
        try:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            self.summary = "observer_cleanup_failed:pipe_close"
            return False
        return True
