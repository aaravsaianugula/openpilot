"""Bounded in-process return bridge; only the native controller mutates gates.

The publisher registers its own camera identities before sending pixels. The
peer can return proposal bytes (braking-only caps), never issued-camera evidence.
A trusted same-domain clock contract is required separately; names do not prove
physical clock alignment. No socket, camera decode or wait runs in stage_cycle.

The bridge lives for the whole card process and serves laptop sessions one after
another. A session fault latches the companion to exact native output for the rest
of the card lifetime: a later session, or turning the braking toggle off and on,
never clears it; the next card start does. The controller thread hands each
device-applied tuple to a bounded outbox the publisher worker drains.
"""

import collections
import logging
import threading
import time
from collections.abc import Callable

from .integration import COMPANION_DISABLED_AFTER_FAULT, CameraReference, ControlGate
from .native_consumer import NativeCycle, NativeScc12Consumer, ReceiverStateRecorder
from .native_context import MAX_ECHO_COMMANDS, ReceiverStateEcho

# Recently seen session ids refused on reuse (the device mints a fresh UUID per connection).
RECENT_SESSIONS = 64
# Device-applied tuples awaiting the worker: about 3 s at the 20 Hz companion rate.
APPLIED_CAPACITY = 64


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
        braking_enabled: Callable[[], bool],
    ):
        """``fault_log`` is handed to each session's consumer for its fault-latch error line.

        ``braking_enabled`` is the device braking toggle each session's consumer reads."""
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
        if receiver_contract is not None and receiver_contract[0] >= MAX_ECHO_COMMANDS * receiver_contract[1]:
            # The echo keeps MAX_ECHO_COMMANDS handoffs; a longer delay would leave none applied, silently.
            raise ValueError("declared_effect_delay_outside_echo_window")
        if fault_log is not None and not isinstance(fault_log, logging.Logger):
            raise ValueError("fault_log_logger_required")
        if not callable(braking_enabled):
            raise ValueError("braking_toggle_callable_required")
        self.braking_enabled = braking_enabled
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
        self._recent_sessions: collections.deque[str] = collections.deque(maxlen=RECENT_SESSIONS)
        # First session fault of this card lifetime; set once, never cleared.
        self._latched: str | None = None
        self._applied_lock = threading.Lock()
        self._applied: list[tuple[str, tuple]] = []
        self.applied_dropped = 0
        # Completion overruns and the worst completion from consumers this bridge has dropped.
        self._retired_overruns = 0
        self._retired_worst_ns = 0
        # The latest companion status line, set by the controller-side binding on change and
        # read by the worker for the drive log (one reference assignment, no lock).
        self.status_line: str | None = None
        # (mode, authority, reason, toggle on) behind that line, for the worker's status.json.
        self.status_fields: tuple[str, str, str, bool] | None = None
        # Set by the worker when status.json writes keep failing; only the reason shows it.
        self.status_file_unwritable = False
        # Monotonic arrival of the latest returned packet (publisher thread), for status.json.
        self.last_return_ns: int | None = None
        self._configuration: tuple | None = None
        self._owner: int | None = None
        self._last_now = -1
        self.status = "awaiting_session"

    def _put(self, event: tuple) -> bool:
        with self._lock:
            if self._overflow:
                return False
            if len(self._events) == self.capacity:
                self._events.clear()
                self._overflow = True
                return False
            self._events.append(event)
            return True

    def open_session(self, session_id: str, camera_clock_id: str) -> bool:
        # UUID/session syntax is checked by the authenticated transport handshake.
        return self._put(("open", session_id, camera_clock_id))

    def dispatch(
        self, session_id: str, input_sequence: int, cameras: tuple[CameraReference, ...], dispatched_ns: int
    ) -> bool:
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
        return self._put(("dispatch", session_id, input_sequence, cameras, dispatched_ns))

    def return_packet(self, session_id: str, packet: bytes) -> bool:
        if type(packet) is not bytes or not 0 < len(packet) <= 16384:
            raise ValueError("bounded_signed_proposal_required")
        self.last_return_ns = time.monotonic_ns()
        return self._put(("return", session_id, packet))

    def close_session(self, session_id: str) -> bool:
        return self._put(("close", session_id))

    @property
    def latched_fault(self) -> str | None:
        """The card-lifetime latch, including a fault the current consumer latched this cycle."""
        if self._latched is not None:
            return self._latched
        return None if self._consumer is None else self._consumer.fault_reason

    def _offer_applied(self, item: tuple) -> None:
        """Controller thread: never blocks, never raises; a busy or full outbox drops and counts."""
        session = self._session
        if session is None or not self._applied_lock.acquire(blocking=False):
            self.applied_dropped += 1
            return
        try:
            if len(self._applied) >= APPLIED_CAPACITY:
                self.applied_dropped += 1
            else:
                self._applied.append((session, item))
        finally:
            self._applied_lock.release()

    def take_applied(self) -> list[tuple[str, tuple]]:
        """Publisher worker: every pending (session, device-applied tuple), oldest first."""
        with self._applied_lock:
            items, self._applied = self._applied, []
        return items

    def receiver_echo(self, now_ns: int) -> ReceiverStateEcho | None:
        """Publisher-worker read of the current session's receiver state, if exactly known."""
        recorder = self._recorder
        return None if recorder is None else recorder.echo(now_ns)

    def completion_stats(self) -> tuple[int, int]:
        """(controller completions later than the declared bound, worst completion ns) this card lifetime."""
        overruns, worst = self._retired_overruns, self._retired_worst_ns
        consumer = self._consumer
        if consumer is not None:
            overruns += consumer.gate.completion_overruns
            worst = max(worst, consumer.gate.worst_completion_ns)
        return overruns, worst

    def _invalidate(self, reason: str) -> None:
        if self._consumer is not None:
            self._retired_overruns += self._consumer.gate.completion_overruns
            self._retired_worst_ns = max(self._retired_worst_ns, self._consumer.gate.worst_completion_ns)
        if self._consumer is not None and self._consumer.fault_reason is not None and self._latched is None:
            self._latched = self._consumer.fault_reason
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
                    if not isinstance(session, str) or session in self._recent_sessions:
                        raise ValueError("native_session_reused")
                    self._recent_sessions.append(session)
                    if self._latched is not None:
                        self.status = COMPANION_DISABLED_AFTER_FAULT + self._latched
                        continue
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
                        gate,
                        completion_clock=self.completion_clock,
                        recorder=recorder,
                        fault_log=self.fault_log,
                        braking_enabled=self.braking_enabled,
                        applied_sink=self._offer_applied,
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
                    # Fold on the validated parent cycle clock before a queued close/open
                    # can retire this consumer. The existing count/window/latch is unchanged.
                    self._consumer.settle_return_faults(now_ns)
            if self._consumer is not None:
                fault = self._consumer.fault_reason
                if fault is not None:
                    # Card-lifetime latch: this session and every later one stay exact native.
                    self._invalidate(COMPANION_DISABLED_AFTER_FAULT + fault)
                    return None
                self._consumer.stage(NativeCycle(now_ns, None))
                self.status = "staged_without_invented_native_evidence"
            return self._consumer
        except (ValueError, TypeError) as error:
            self._invalidate(str(error))
            return None
