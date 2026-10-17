"""Adapter for the pinned CN7 final SCC12 and automatic-resume source hooks.

A trusted bridge must explicitly stage each native cycle. This module creates
neither camera references nor a clock. The companion is a braking-only cap; it
never raises a native field. There is no transport, hook registration, CAN
sender, or device installation.
"""

import collections
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .integration import (
    COMPANION_DISABLED_AFTER_FAULT,
    ControlDecision,
    ControlGate,
    NativeScc12FieldError,
    PreparedScc12,
    prepare_scc12,
)
from .native_context import MAX_ECHO_COMMANDS, EchoedCommand, ReceiverStateEcho, echo_active_age_ns

# SCC12 alternatives the native packer builds per tick: exact native, companion cap.
SCC12_ALTERNATIVES = 2
# DECLARED injection bound: this many unauthenticated or foreign return packets inside the
# window latch the companion off for the session. Each one already gives native output; the
# latch stops an injector from toggling companion braking indefinitely.
UNAUTHENTICATED_LATCH_COUNT = 3
UNAUTHENTICATED_WINDOW_NS = 1_000_000_000


@dataclass(frozen=True)
class NativeCycle:
    """One consumer-owned cycle; its packet (if any) is staged until the next arbitration."""

    now_ns: int
    packet: bytes | None

    def __post_init__(self):
        if type(self.now_ns) is not int or not 0 <= self.now_ns < 2**63:
            raise ValueError("Explicit native monotonic cycle timestamp required")
        if self.packet is not None and not isinstance(self.packet, bytes):
            raise ValueError("Authenticated wire bytes or no new packet required")


class ReceiverStateRecorder:
    """Bookkeeping of every final SCC12 acceleration this consumer handed off.

    The controller thread records; the publisher worker reads echoes. A busy
    lock on the controller side would lose a command, so it poisons the
    recorder instead: no further echo exists for this session, never a guess.
    The effect delay and handoff period are declared receiver contracts that a
    separately verified bridge supplies; nothing here measures them.
    """

    def __init__(self, *, session_id: str, build_id: str, clock_id: str, effect_delay_ns: int, handoff_period_ns: int):
        for name, value in (("effect_delay_ns", effect_delay_ns), ("handoff_period_ns", handoff_period_ns)):
            if type(value) is not int or not 0 < value <= 1_000_000_000:
                raise ValueError("Declared receiver " + name + " required")
        self.session_id, self.build_id, self.clock_id = session_id, build_id, clock_id
        self.effect_delay_ns, self.handoff_period_ns = effect_delay_ns, handoff_period_ns
        self._commands: collections.deque[EchoedCommand] = collections.deque(maxlen=MAX_ECHO_COMMANDS)
        self._lock = threading.Lock()
        self._sequence = 0
        self.poisoned: str | None = None

    def record_handoff(self, decision_ns: int, accel_mps2: float, companion: bool) -> None:
        command = EchoedCommand(decision_ns, decision_ns + self.effect_delay_ns, float(accel_mps2), companion)
        if not self._lock.acquire(blocking=False):
            self.poisoned = "receiver_echo_recorder_busy"
            return
        try:
            if self._commands and decision_ns <= self._commands[-1].decision_ns:
                self.poisoned = "receiver_echo_handoff_regressed"
                return
            self._commands.append(command)
        finally:
            self._lock.release()

    def poison(self, reason: str) -> None:
        """The receiver state is no longer exactly representable for this session."""
        self.poisoned = reason

    def echo(self, now_ns: int) -> ReceiverStateEcho | None:
        """The receiver state at ``now_ns``, or None when it is not exactly known."""
        with self._lock:
            commands = tuple(self._commands)
            if self.poisoned is not None or not commands or now_ns < commands[-1].decision_ns:
                return None
            next_handoff = commands[-1].decision_ns + self.handoff_period_ns
            if now_ns >= next_handoff:
                # The declared next handoff is due or overdue: the queue may have moved.
                return None
            accepted = tuple(item for item in commands if item.apply_ns <= now_ns)
            if not accepted:
                return None
            self._sequence += 1
            return ReceiverStateEcho(
                self.session_id,
                self.build_id,
                self.clock_id,
                self._sequence,
                now_ns,
                commands[0].decision_ns,
                self.effect_delay_ns,
                self.handoff_period_ns,
                next_handoff,
                accepted,
                tuple(item for item in commands if item.apply_ns > now_ns),
                echo_active_age_ns(accepted, now_ns),
            )


@dataclass(frozen=True)
class NativeFieldFallback:
    """Exact native output for a cycle whose native fields broke the pinned contract.

    No gate attempt exists: the fields pass through untouched and the owning
    consumer latches the companion off for the session.
    """

    values: tuple[dict, dict]
    decisions: tuple[ControlDecision, ControlDecision]

    @classmethod
    def of(cls, values: dict, reason: str) -> "NativeFieldFallback":
        # Accel is unknown: the native fields themselves failed validation.
        decision = ControlDecision(math.nan, False, True, False, reason)
        return cls((dict(values), dict(values)), (decision, decision))

    def finalize(self, completion_clock: Callable[[], int] | None) -> tuple[int, ControlDecision]:
        return 0, self.decisions[0]


class _FinalScc12Boundary:
    def __init__(
        self,
        consumer: "NativeScc12Consumer",
        cycle: NativeCycle,
        long_active: bool,
        driver_override: bool,
        standstill: bool,
    ):
        self.consumer = consumer
        self.recorder = consumer.recorder
        self.gate = consumer.gate
        self.cycle = cycle
        self.long_active = long_active
        self.driver_override = driver_override
        self.standstill = standstill
        self.completion_clock = consumer.completion_clock
        self.decision: ControlDecision | None = None
        self.output: dict | None = None
        self._used = False
        self.prepared: PreparedScc12 | NativeFieldFallback | None = None
        self.messages: tuple | None = None
        self._handoffs: tuple | None = None
        self._finished = False

    def prepare_values(self, values: dict) -> tuple[dict, dict]:
        """Arbitrate the CURRENT native fields with the consumer's next staged packet.

        Returns the (exact native, companion cap) alternatives for native packing.
        An invalid native field returns exact native output, latches the companion
        off for the session and records the fault; it never raises into the controller.
        """
        if self._used:
            raise ValueError("Final SCC12 boundary already consumed this native cycle")
        self._used = True
        packet = self.consumer.take_packet()
        try:
            self.prepared = prepare_scc12(
                values,
                self.gate,
                packet,
                now_ns=self.cycle.now_ns,
                long_active=self.long_active,
                driver_override=self.driver_override,
                standstill=self.standstill,
            )
        except NativeScc12FieldError as fault:
            # Recorded and authority removed, never swallowed: exact native output.
            reason = "native_scc12_field_fault:" + str(fault)
            self.consumer.latch_fault(reason)
            self.prepared = NativeFieldFallback.of(values, reason)
        return self.prepared.values

    def record_messages(self, messages: tuple) -> tuple:
        if self.prepared is None or self.messages is not None or len(messages) != SCC12_ALTERNATIVES:
            raise ValueError("Exactly the native-packed SCC12 alternatives required once")
        if any(row[0] != 0x421 or not isinstance(row[1], bytes) or len(row[1]) != 8 for row in messages):
            raise ValueError("Actual pinned SCC12 messages required")
        self.messages = messages
        return messages[0]

    @staticmethod
    def _resume_ids(native_messages: list, automatic_resume_messages: tuple) -> set[int]:
        if type(automatic_resume_messages) is not tuple or any(
            row[0] != 0x4F1 or not any(row is native for native in native_messages) for row in automatic_resume_messages
        ):
            raise ValueError("Exact native-generated CLU11 resume messages required")
        return {id(row) for row in automatic_resume_messages}

    def prepare_handoff(self, native_actuators, native_messages: list, automatic_resume_messages: tuple = ()) -> None:
        if self.messages is None or self.prepared is None or self._handoffs is not None:
            raise ValueError("Native packed alternatives required before handoff preparation")
        positions = [index for index, row in enumerate(native_messages) if row is self.messages[0]]
        if len(positions) != 1:
            raise ValueError("The prepared native SCC12 message must appear exactly once")
        resume_ids = self._resume_ids(native_messages, automatic_resume_messages)
        alternatives = []
        for index, message in enumerate(self.messages):
            actuators = native_actuators.as_reader().as_builder()
            rows = list(native_messages)
            rows[positions[0]] = message
            if self.prepared.decisions[index].companion_eligible:
                actuators.accel = self.prepared.values[index]["aReqValue"]
                if not self.prepared.decisions[index].allow_resume:
                    rows = [row for row in rows if id(row) not in resume_ids]
            alternatives.append((actuators, rows))
        self._handoffs = tuple(alternatives)

    def prepare_resume_handoff(self, native_actuators, native_messages: list, automatic_resume_messages: tuple) -> None:
        """Odd native tick: filter only explicit resume messages, never send SCC12.

        prepare_values must receive the same current native field builder used
        on SCC12 ticks, including current tuning/ESCC. Resume is filtered only
        while an accepted cap carries the companion's own stop obligation.
        """
        if self.messages is not None or self.prepared is None or self._handoffs is not None:
            raise ValueError("Current native fields required once for a button-only handoff")
        resume_ids = self._resume_ids(native_messages, automatic_resume_messages)
        self._handoffs = tuple(
            (
                native_actuators,
                [
                    row
                    for row in native_messages
                    if not (decision.companion_eligible and not decision.allow_resume and id(row) in resume_ids)
                ],
            )
            for decision in self.prepared.decisions
        )

    def finalize_handoff(self) -> tuple:
        if self._finished or self._handoffs is None or self.prepared is None:
            raise ValueError("Final native handoff must be prepared and consumed exactly once")
        self._finished = True
        index, decision = self.prepared.finalize(self.completion_clock)
        self.consumer._absorb_gate_fault()
        self.decision = decision
        self.output = self.prepared.values[index]
        if self.messages is not None and self.recorder is not None:
            # An SCC12 tick: the selected alternative is what the receiver gets.
            if isinstance(self.prepared, NativeFieldFallback):
                self.recorder.poison("native_scc12_field_fault")
            else:
                self.recorder.record_handoff(self.cycle.now_ns, self.output["aReqValue"], decision.companion_eligible)
        return self._handoffs[index]


class NativeScc12Consumer:
    """Explicit same-process binding; unavailable cycles restore native behavior.

    The gate must be independently initialized with verified physical bounds,
    session/build identity, secret and actually issued dual-camera references.
    An upstream trusted bridge owns stage(), on the native controller thread.
    It must not replace capture times with receipt time.
    The bridge and physical qualification remain absent from this candidate.
    """

    def __init__(
        self,
        gate: ControlGate,
        *,
        completion_clock: Callable[[], int] | None = None,
        recorder: ReceiverStateRecorder | None = None,
        fault_log: logging.Logger | None = None,
    ):
        """``fault_log`` receives the fault-latch error line. The car process passes
        opendbc's ``carlog`` (card.py forwards it to cloudlog); elsewhere it is this
        module's standard logger."""
        if not isinstance(gate, ControlGate):
            raise ValueError("Canonical authenticated control gate required")
        if recorder is not None and (
            not isinstance(recorder, ReceiverStateRecorder)
            or (recorder.session_id, recorder.build_id) != (gate.session_id, gate.build_id)
        ):
            raise ValueError("Receiver recorder must belong to this gate's session and build")
        if fault_log is not None and not isinstance(fault_log, logging.Logger):
            raise ValueError("A standard logging.Logger is required for fault reporting")
        self.recorder = recorder
        self.gate = gate
        self.fault_log = logging.getLogger(__name__) if fault_log is None else fault_log
        # Only a trusted bridge can qualify this reader's domain. Never infer it
        # from native now_nanos or substitute Python/laptop wall-clock time.
        self.completion_clock = completion_clock
        self._pending: NativeCycle | None = None
        self._last_now_ns: int | None = None
        self.last_cycle_status = "native_cycle_unavailable"
        # Staged proposals survive SCC12-less ticks until the next arbitration.
        self._unknown: bytes | None = None
        self._newest: bytes | None = None
        # Highest authenticated sequence ever staged: newest is by sequence, not arrival.
        self._staged_sequence = 0
        self.regression_count = 0
        # Unauthenticated or foreign return bytes: session total, those not yet folded onto
        # the native clock, and the native-clock arrival times inside the latch window.
        self.unauthenticated_count = 0
        self._unauthenticated_unfolded = 0
        self._unauthenticated_arrivals: collections.deque[int] = collections.deque()
        # Session latch: a native-field fault removes companion authority for good.
        self.fault_count = 0
        self.fault_reason: str | None = None

    def offer_packet(self, data: bytes) -> None:
        """Stage one returned packet, strictly by authenticated sequence.

        The highest authenticated sequence wins, whether it is a cap or a withdrawal:
        a withdrawal is never superseded by an OLDER packet, and a newer cap supersedes
        an older withdrawal. An authenticated packet at or below the highest sequence
        already staged (older or duplicate) is refused as a regression and counted; it
        never replaces or cancels anything. Unauthenticated or foreign bytes are unknown,
        and unknown is native: they cancel every authenticated packet staged before them
        and are consumed by the next arbitration (exact native output); a packet arriving
        after them waits behind them. Each one is counted and logged, and
        ``UNAUTHENTICATED_LATCH_COUNT`` inside ``UNAUTHENTICATED_WINDOW_NS`` latch the
        companion off (folded at the next ``begin_cycle``, on the native clock).
        """
        if type(data) is not bytes:
            raise ValueError("Authenticated wire bytes required")
        sequence, _ = self.gate.classify_packet(data)
        if sequence is None:
            self.unauthenticated_count += 1
            self._unauthenticated_unfolded += 1
            self._unknown, self._newest = data, None
            if self.fault_reason is None:
                self.fault_log.warning(
                    "vision_long unauthenticated or foreign return packet for session %s (count=%d)",
                    self.gate.session_id,
                    self.unauthenticated_count,
                )
            return
        if sequence <= self._staged_sequence:
            self.regression_count += 1
            return
        self._staged_sequence = sequence
        self._newest = data

    def take_packet(self) -> bytes | None:
        """The packet for this arbitration: pending unknown bytes first, else the newest by sequence."""
        if self._unknown is not None:
            unknown, self._unknown = self._unknown, None
            return unknown
        newest, self._newest = self._newest, None
        return newest

    def latch_fault(self, reason: str) -> None:
        self.fault_count += 1
        self.fault_reason = reason
        self._unknown = self._newest = None
        # Visible, never swallowed: one explicit error line on the platform log.
        self.fault_log.error(
            "vision_long companion disabled for session %s after native fault %s (fault_count=%d)",
            self.gate.session_id,
            reason,
            self.fault_count,
        )

    def _absorb_gate_fault(self) -> None:
        """A clock fault the gate latched is the session's fault: latch here too and log it once."""
        if self.fault_reason is None and self.gate.fault_reason is not None:
            self.latch_fault(self.gate.fault_reason)

    def _fold_unauthenticated(self, now_ns: int) -> None:
        arrivals = self._unauthenticated_arrivals
        arrivals.extend([now_ns] * self._unauthenticated_unfolded)
        self._unauthenticated_unfolded = 0
        while arrivals and now_ns - arrivals[0] >= UNAUTHENTICATED_WINDOW_NS:
            arrivals.popleft()
        if self.fault_reason is None and len(arrivals) >= UNAUTHENTICATED_LATCH_COUNT:
            self.latch_fault(f"unauthenticated_return_packets:{len(arrivals)}_within_{UNAUTHENTICATED_WINDOW_NS}_ns")

    def stage(self, cycle: NativeCycle) -> None:
        if not isinstance(cycle, NativeCycle):
            raise ValueError("Explicit typed native cycle required")
        if cycle.packet is not None:
            self.offer_packet(cycle.packet)
        self._pending = cycle

    def begin_cycle(
        self, now_ns: int, *, long_active: bool, driver_override: bool, standstill: bool
    ) -> _FinalScc12Boundary | None:
        cycle, self._pending = self._pending, None
        if type(now_ns) is not int or not 0 <= now_ns < 2**63:
            raise ValueError("Actual native monotonic timestamp required")
        if any(type(value) is not bool for value in (long_active, driver_override, standstill)):
            raise ValueError("Actual native control and measured standstill state required")
        if self._last_now_ns is not None and now_ns <= self._last_now_ns and self.fault_reason is None:
            # A repeated or regressed native cycle clock is a clock fault: latch and log.
            self.latch_fault("native_cycle_repeated_or_regressed")
        self._absorb_gate_fault()
        if self.fault_reason is None:
            self._last_now_ns = now_ns
            self._fold_unauthenticated(now_ns)
        if self.fault_reason is not None:
            self.last_cycle_status = COMPANION_DISABLED_AFTER_FAULT + self.fault_reason
            return None
        if cycle is None:
            self.last_cycle_status = "native_cycle_unavailable"
            return None
        if cycle.now_ns != now_ns:
            self.last_cycle_status = "native_cycle_clock_mismatch"
            return None
        self.last_cycle_status = "native_cycle_bound"
        return _FinalScc12Boundary(self, cycle, long_active, driver_override, standstill)
