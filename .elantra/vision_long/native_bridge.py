"""Bounded in-process return bridge; only the native controller mutates gates.

The publisher registers its own camera identities before sending pixels. The
peer can return proposal bytes (braking-only caps), never issued-camera evidence.
A trusted same-domain clock contract is required separately; names do not prove
physical clock alignment. No socket, camera decode or wait runs in stage_cycle.
"""

import logging
import threading
from collections.abc import Callable

from .integration import CameraReference, ControlGate
from .native_consumer import NativeCycle, NativeScc12Consumer, ReceiverStateRecorder
from .native_context import ReceiverStateEcho


class NativeReturnBridge:
    def __init__(
        self,
        gate_factory: Callable[[str], ControlGate],
        *,
        completion_clock: Callable[[], int] | None = None,
        trusted_clock_id: str | None = None,
        capacity: int = 32,
        receiver_contract: tuple[int, int] | None = None,
        fault_log: logging.Logger | None = None,
    ):
        """``fault_log`` is handed to each session's consumer for its fault-latch error line."""
        if not callable(gate_factory) or type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("bounded_bridge_configuration_required")
        if completion_clock is not None and not callable(completion_clock):
            raise ValueError("completion_clock_callable_required")
        if trusted_clock_id is not None and (
            not isinstance(trusted_clock_id, str) or not 1 <= len(trusted_clock_id) <= 128
        ):
            raise ValueError("invalid_trusted_clock_identity")
        # Declared (effect delay ns, SCC12 handoff period ns); without it no receiver echo exists.
        if receiver_contract is not None and (
            type(receiver_contract) is not tuple
            or len(receiver_contract) != 2
            or any(type(value) is not int or not 0 < value <= 1_000_000_000 for value in receiver_contract)
        ):
            raise ValueError("invalid_declared_receiver_contract")
        if fault_log is not None and not isinstance(fault_log, logging.Logger):
            raise ValueError("fault_log_logger_required")
        self.fault_log = fault_log
        self.receiver_contract = receiver_contract
        self._recorder: ReceiverStateRecorder | None = None
        self.gate_factory = gate_factory
        self.completion_clock = completion_clock
        self.trusted_clock_id = trusted_clock_id
        self.capacity = capacity
        self._lock = threading.Lock()
        self._events: list[tuple] = []
        self._overflow = False
        self._consumer: NativeScc12Consumer | None = None
        self._session: str | None = None
        self._sessions: set[str] = set()
        self._configuration: tuple | None = None
        self._owner: int | None = None
        self._last_now = -1
        self.status = "awaiting_session"

    def _put(self, event: tuple) -> None:
        with self._lock:
            if self._overflow:
                return
            if len(self._events) == self.capacity:
                self._events.clear()
                self._overflow = True
                return
            self._events.append(event)

    def open_session(self, session_id: str, camera_clock_id: str) -> None:
        # UUID/session syntax is checked by the authenticated transport handshake.
        self._put(("open", session_id, camera_clock_id))

    def dispatch(
        self, session_id: str, input_sequence: int, cameras: tuple[CameraReference, ...], dispatched_ns: int
    ) -> None:
        if (
            type(input_sequence) is not int
            or input_sequence < 1
            or type(dispatched_ns) is not int
            or dispatched_ns < 0
            or type(cameras) is not tuple
            or len(cameras) != 2
            or any(not isinstance(item, CameraReference) for item in cameras)
        ):
            raise ValueError("invalid_native_camera_dispatch")
        self._put(("dispatch", session_id, input_sequence, cameras, dispatched_ns))

    def return_packet(self, session_id: str, packet: bytes) -> None:
        if type(packet) is not bytes or not 0 < len(packet) <= 16384:
            raise ValueError("bounded_signed_proposal_required")
        self._put(("return", session_id, packet))

    def close_session(self, session_id: str) -> None:
        self._put(("close", session_id))

    def receiver_echo(self, now_ns: int) -> ReceiverStateEcho | None:
        """Publisher-worker read of the current session's receiver state, if exactly known."""
        recorder = self._recorder
        return None if recorder is None else recorder.echo(now_ns)

    def _invalidate(self, reason: str) -> None:
        self._recorder = None
        self._consumer = None
        self._session = None
        self._configuration = None
        self.status = reason

    def stage_cycle(self, now_ns: int) -> NativeScc12Consumer | None:
        owner = threading.get_ident()
        if self._owner is None:
            self._owner = owner
        if self._owner != owner:
            raise RuntimeError("native_bridge_controller_thread_changed")
        if type(now_ns) is not int or not 0 <= now_ns < 2**63 or now_ns <= self._last_now:
            self._invalidate("native_cycle_clock_regression")
            return None
        self._last_now = now_ns
        configuration = (self.gate_factory, self.completion_clock, self.trusted_clock_id, self.receiver_contract)
        if self._consumer is not None and configuration != self._configuration:
            self._invalidate("native_clock_or_gate_configuration_changed")
        if not self._lock.acquire(blocking=False):
            self._invalidate("native_mailbox_busy")
            return None
        try:
            events, self._events = self._events, []
            overflow, self._overflow = self._overflow, False
        finally:
            self._lock.release()
        if overflow:
            self._invalidate("native_mailbox_overflow")
            return None
        try:
            for event in events:
                kind, session = event[:2]
                if kind == "open":
                    self._invalidate("new_session_legal_history_unavailable")
                    if not isinstance(session, str) or session in self._sessions or len(self._sessions) >= 64:
                        raise ValueError("native_session_reused_or_lifetime_capacity")
                    self._sessions.add(session)
                    if (
                        self.completion_clock is None
                        or self.trusted_clock_id is None
                        or event[2] != self.trusted_clock_id
                    ):
                        self.status = "native_camera_completion_clock_contract_unavailable"
                        continue
                    gate = self.gate_factory(session)
                    if not isinstance(gate, ControlGate) or gate.session_id != session:
                        raise ValueError("native_gate_session_mismatch")
                    if self.receiver_contract is not None and gate.command_lag_envelope_ns is None:
                        # Echoed sessions anchor command paths; the gate must enforce the anchor.
                        raise ValueError("native_gate_lag_envelope_required")
                    self._session = session
                    recorder = (
                        None
                        if self.receiver_contract is None
                        else ReceiverStateRecorder(
                            session_id=session,
                            build_id=gate.build_id,
                            clock_id=self.trusted_clock_id,
                            effect_delay_ns=self.receiver_contract[0],
                            handoff_period_ns=self.receiver_contract[1],
                        )
                    )
                    self._consumer = NativeScc12Consumer(
                        gate, completion_clock=self.completion_clock, recorder=recorder, fault_log=self.fault_log
                    )
                    self._recorder = recorder
                    self._configuration = configuration
                elif session != self._session:
                    continue
                elif kind == "close":
                    self._invalidate("native_session_closed")
                elif kind == "dispatch":
                    _, _, sequence, cameras, dispatched = event
                    if not 0 <= now_ns - dispatched <= 50_000_000 or any(c.capture_ns > dispatched for c in cameras):
                        raise ValueError("native_dispatch_stale_or_future")
                    assert self._consumer is not None
                    self._consumer.gate.issue(sequence, cameras, now_ns)
                elif kind == "return" and self._consumer is not None:
                    # Every return is offered in arrival order; the consumer stages it
                    # strictly by authenticated sequence and counts unknown bytes.
                    self._consumer.offer_packet(event[2])
            if self._consumer is not None:
                self._consumer.stage(NativeCycle(now_ns, None))
                fault = self._consumer.fault_reason
                self.status = (
                    "staged_without_invented_native_evidence"
                    if fault is None
                    else "companion_disabled_after_fault:" + fault
                )
            return self._consumer
        except (ValueError, TypeError) as error:
            self._invalidate(str(error))
            return None
