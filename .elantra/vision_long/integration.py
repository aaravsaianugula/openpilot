"""Authenticated proposal boundary. No imports, sockets, CAN writes or vehicle side effects.

The consumer supplies CURRENT native acceleration after downstream tuning, control
enable/override state, measured standstill and the exact input camera references
issued this session. Every wire timestamp is in the consumer's (native) clock
domain: laptop time is converted with its clock-relation uncertainty before
signing and cannot establish freshness. This release is BRAKING-ONLY: a proposal
is a cap that can only lower native fields; it never raises one and never
resumes the car. Companion positive acceleration is a future, separately
qualified feature. Deployment still requires a verified native hook.
"""

import hashlib
import hmac
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import TypeGuard

from .control_authority import ForwardMotionClearance

# Extra command-path lag the control's closed-loop sweep proves plans robust to:
# command_stage.PROVEN_EXTRA_LAG_S (0.10 s, 132/132 at 0/50/100 ms). Restated in
# integer ns because this support module ships to the consumer without the
# planner; tests pin it to the control constant. It is also the APPLICATION
# validity of an accepted braking cap at the consumer, measured from its anchor.
PROVEN_COMMAND_LAG_ENVELOPE_NS = 100_000_000
# DECISION-time camera age: the native side issues only identities this fresh, and
# the laptop decides only on inputs this fresh. It is not the cap's application validity.
CAMERA_DECISION_AGE_NS = 50_000_000
# DECLARED controller completion bound. The template's CarController.update arbitrates SCC12 with
# the frame's now_nanos and reads the trusted completion clock at finalize_handoff, at the end of
# the same 100 Hz (DT_CTRL = 10 ms) frame. It is declared at half that frame, not measured here.
# A completion later than this is a late clock: exact native output, and the session latches off.
CONTROLLER_COMPLETION_BOUND_NS = 5_000_000
# The companion's decision and packet period (20 Hz; ReceiverContract.decision_interval_ns).
COMPANION_PACKET_PERIOD_NS = 50_000_000
# Laptop return contract, enforced when it signs (return_budget_fault). A cap must become usable at
# the consumer (the later of its worst-case arrival and its receiver anchor) within this much of its
# older capture. Its predecessor, captured one period earlier, is then still inside the 100 ms
# application validity through the successor's takeover plus the declared completion: no gap.
RETURN_AGE_BUDGET_NS = PROVEN_COMMAND_LAG_ENVELOPE_NS - COMPANION_PACKET_PERIOD_NS - CONTROLLER_COMPLETION_BOUND_NS
# Prefix of every decision reason once a fault has latched the companion off for the session.
COMPANION_DISABLED_AFTER_FAULT = "companion_disabled_after_fault:"
SCC12_GRID = Decimal("0.01")


class NativeScc12FieldError(ValueError):
    """The native SCC12 fields handed to the boundary are outside the pinned contract."""


def _number(value: object, lo: float, hi: float) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi
    except OverflowError:
        return False


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128 and value.isascii() and value.isprintable()


def _integer(value: object) -> bool:
    return type(value) is int and 0 <= value < 2**63


@dataclass(frozen=True)
class CameraReference:
    camera: str
    frame_id: int
    capture_ns: int

    def __post_init__(self):
        if self.camera not in ("narrow", "wide") or not _integer(self.frame_id) or not _integer(self.capture_ns):
            raise ValueError("Invalid camera reference")


@dataclass(frozen=True)
class ProposalPacket:
    session_id: str
    build_id: str
    sequence: int
    input_sequence: int
    cameras: tuple[CameraReference, ...]
    accel: float | None
    should_stop: bool
    confidence: float
    healthy: bool
    faults: tuple[str, ...]
    forward_clearance: ForwardMotionClearance | None = None
    # Consumer-clock SCC12 handoff the companion's command path was anchored to
    # (the first handoff after the echo plus the declared minimum return latency).
    receiver_anchor_ns: int | None = None
    # Inclusive native-domain bounds of the clearance time; the clearance's own
    # timestamp_s is then the lower bound in native seconds (see native_domain_packet).
    # The laptop's report only: the braking-only consumer grants nothing from it.
    clearance_native_ns: tuple[int, int] | None = None
    # Typed planner flag: the companion has an active stop obligation OF ITS OWN (red
    # signal, stop sign before completion or clearance, a stopped lead or obstacle in
    # path inside the stop target, a legal boundary). Only this holds a standstill.
    stop_obligation: bool = False

    def __post_init__(self):
        if not _identifier(self.session_id) or not _identifier(self.build_id):
            raise ValueError("Invalid session/build identity")
        if type(self.stop_obligation) is not bool:
            raise ValueError("Explicit typed stop obligation required")
        if self.stop_obligation and (not self.healthy or not _number(self.accel, -15, 0)):
            raise ValueError("A stop obligation needs a healthy, nonpositive braking cap")
        if self.clearance_native_ns is not None:
            bounds = self.clearance_native_ns
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(_integer(value) for value in bounds)
                or bounds[0] > bounds[1]
                or self.forward_clearance is None
                or self.forward_clearance.timestamp_s != bounds[0] / 1e9
            ):
                raise ValueError("Native clearance bounds must restate the clearance stamp")
        if self.receiver_anchor_ns is not None and not _integer(self.receiver_anchor_ns):
            raise ValueError("Invalid receiver anchor")
        if not _integer(self.sequence) or not _integer(self.input_sequence):
            raise ValueError("Invalid packet sequence")
        if len(self.cameras) != 2 or {c.camera for c in self.cameras} != {"narrow", "wide"}:
            raise ValueError("Two distinct camera identities required")
        if self.accel is not None and not _number(self.accel, -15, 10):
            raise ValueError("Invalid acceleration")
        if type(self.should_stop) is not bool or type(self.healthy) is not bool or not _number(self.confidence, 0, 1):
            raise ValueError("Invalid health/confidence")
        if len(self.faults) > 32 or any(not _identifier(reason) for reason in self.faults):
            raise ValueError("Invalid bounded fault list")
        if self.forward_clearance is not None and not isinstance(self.forward_clearance, ForwardMotionClearance):
            raise ValueError("Typed forward clearance or unknown required")


def _withdraws(packet: ProposalPacket) -> bool:
    """An authenticated report that carries no usable cap: it withdraws companion authority."""
    return not packet.healthy or packet.confidence < 0.8 or bool(packet.faults) or packet.accel is None


def return_budget_fault(packet: ProposalPacket, arrival_high_ns: int | None) -> str | None:
    """Laptop side, before signing: why this cap could not cover its successor, or None to send it.

    ``arrival_high_ns`` is the native-clock upper bound of the packet's arrival at the consumer
    under the declared return contract (decision upper bound plus the declared maximum return
    latency), or None when it is unknown. A withdrawal only removes authority and is always sent.
    """
    if _withdraws(packet):
        return None
    if arrival_high_ns is None or not _integer(arrival_high_ns):
        return "return_age_unbounded"
    capture = min(camera.capture_ns for camera in packet.cameras)
    anchor = packet.receiver_anchor_ns
    if anchor is not None and anchor < capture:
        # The anchor window would close before the camera window: coverage is not guaranteed.
        return "receiver_anchor_before_capture"
    usable = arrival_high_ns if anchor is None else max(arrival_high_ns, anchor)
    if usable - capture > RETURN_AGE_BUDGET_NS:
        return "return_age_outside_budget"
    return None


def _canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("Explicit secret of at least 32 bytes required")


def native_domain_packet(packet: ProposalPacket, clearance_native_ns: tuple[int, int] | None) -> ProposalPacket:
    """Restate the clearance stamp in the consumer clock domain before signing.

    ``clearance_native_ns`` is the inclusive native-clock interval containing the
    laptop clearance instant, from the established clock relation and its uncertainty.
    """
    if packet.forward_clearance is None:
        if clearance_native_ns is not None:
            raise ValueError("Native clearance bounds without a clearance")
        return packet
    if clearance_native_ns is None or not _integer(clearance_native_ns[0]):
        raise ValueError("A clearance needs its native-domain bounds")
    return replace(
        packet,
        forward_clearance=replace(packet.forward_clearance, timestamp_s=clearance_native_ns[0] / 1e9),
        clearance_native_ns=clearance_native_ns,
    )


def wire_packet(
    packet: ProposalPacket, *, stop_obligation: bool, clearance_native_ns: tuple[int, int] | None
) -> ProposalPacket:
    """Final laptop-side assembly: native-domain clearance plus the typed stop obligation.

    ``stop_obligation`` is the planner's own flag (LongitudinalProposal.stop_obligation).
    It is carried only on a healthy packet with a cap; a withdrawal carries none and
    gives exact native output anyway. Nothing here infers an obligation.
    """
    if type(stop_obligation) is not bool:
        raise ValueError("Explicit typed stop obligation required")
    packet = native_domain_packet(packet, clearance_native_ns if packet.forward_clearance is not None else None)
    return replace(packet, stop_obligation=stop_obligation and packet.healthy and packet.accel is not None)


def sign_packet(packet: ProposalPacket, key: bytes) -> bytes:
    _key(key)
    if packet.forward_clearance is not None and packet.clearance_native_ns is None:
        raise ValueError("Clearance time must be converted to the native domain before signing")
    body = asdict(packet)
    mac = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
    return _canonical({"body": body, "mac": mac})


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_packet(data: bytes, key: bytes) -> ProposalPacket:
    _key(key)
    if not isinstance(data, bytes) or not 0 < len(data) <= 16384:
        raise ValueError("Invalid bounded packet")
    raw = json.loads(data, object_pairs_hook=_unique)
    if not isinstance(raw, dict) or set(raw) != {"body", "mac"} or not isinstance(raw["mac"], str):
        raise ValueError("Invalid packet schema")
    body = raw["body"]
    if not isinstance(body, dict) or set(body) != set(ProposalPacket.__dataclass_fields__):
        raise ValueError("Invalid body schema")
    expected = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, raw["mac"]):
        raise ValueError("Authentication failed")
    cameras = body["cameras"]
    if not isinstance(cameras, list) or len(cameras) != 2 or not isinstance(body["faults"], list):
        raise ValueError("Invalid collection schema")
    body["cameras"] = tuple(CameraReference(**camera) for camera in cameras)
    body["faults"] = tuple(body["faults"])
    clearance = body["forward_clearance"]
    if clearance is not None:
        if not isinstance(clearance, dict) or set(clearance) != set(ForwardMotionClearance.__dataclass_fields__):
            raise ValueError("Invalid forward clearance schema")
        if not isinstance(clearance["reference_ids"], list):
            raise ValueError("Invalid clearance references")
        clearance["reference_ids"] = tuple(clearance["reference_ids"])
        body["forward_clearance"] = ForwardMotionClearance(**clearance)
    bounds = body["clearance_native_ns"]
    if bounds is not None:
        if not isinstance(bounds, list):
            raise ValueError("Invalid native clearance bounds")
        body["clearance_native_ns"] = tuple(bounds)
    return ProposalPacket(**body)


@dataclass(frozen=True)
class ControlDecision:
    accel: float
    should_stop: bool
    allow_resume: bool
    companion_eligible: bool
    reason: str


@dataclass(frozen=True)
class PreparedControl:
    """One gate-owned attempt; preparation is not accepted control authority."""

    generation: int
    now_ns: int
    packet: ProposalPacket | None
    configuration: tuple
    fallback: ControlDecision
    decision: ControlDecision


# Gate outcome when an accepted cap meets a standstill it has no obligation to hold.
STANDSTILL_WITHOUT_OBLIGATION = "standstill_without_companion_obligation"
# Gate outcome while native is disabled or the driver overrides: exact native output.
NATIVE_DISABLED_OR_DRIVER_OVERRIDE = "native_disabled_or_driver_override"


class ControlGate:
    """Strongest braking wins; the signed packet is a braking-only cap on native.

    A cap lowers native fields only. Resume belongs to native: the companion holds
    it only while its packet carries a typed stop obligation of its own.
    """

    def __init__(
        self,
        key: bytes,
        session_id: str,
        build_id: str,
        min_accel: float,
        max_accel: float,
        *,
        command_lag_envelope_ns: int | None = None,
    ):
        """``command_lag_envelope_ns`` enforces the companion's anchored command path.

        When set (a receiver-echo session), a proposal is used only at a cycle
        within [anchor, anchor + envelope] of its ``receiver_anchor_ns``, measured
        on this consumer's own clock; without an anchor it is refused. It can only
        tighten the proven envelope, never widen it. A newer cap whose anchor is
        still ahead is staged and takes over at the first arbitration at or after
        its anchor; the cap in force keeps applying until then, within its own validity.
        """
        _key(key)
        if not _identifier(session_id) or not _identifier(build_id):
            raise ValueError("Explicit session and pinned build required")
        if not _number(min_accel, -15, 0) or not _number(max_accel, 0, 10):
            raise ValueError("Verified physical acceleration bounds required")
        if command_lag_envelope_ns is not None and (
            type(command_lag_envelope_ns) is not int
            or not 0 <= command_lag_envelope_ns <= PROVEN_COMMAND_LAG_ENVELOPE_NS
        ):
            raise ValueError("Command lag envelope must be an integer within the proven envelope")
        self.command_lag_envelope_ns = command_lag_envelope_ns
        self.key, self.session_id, self.build_id = key, session_id, build_id
        self.min_accel, self.max_accel = min_accel, max_accel
        self._issued: dict[int, tuple[CameraReference, ...]] = {}
        self._last_sequence = 0
        self._last_now = 0
        self._active: ProposalPacket | None = None
        self._active_configuration: tuple | None = None
        # Authenticated, valid caps newer than the one in force whose receiver anchor is still
        # ahead, in ascending sequence, and the configuration they were received under.
        self._staged: list[ProposalPacket] = []
        self._staged_configuration: tuple | None = None
        self._generation = 0
        self._pending: PreparedControl | None = None
        # Authenticated packets refused because their sequence was not the newest.
        self.regression_count = 0
        # Session latch: a regressed, late, invalid or failed consumer clock removes authority.
        self.fault_reason: str | None = None

    def classify_packet(self, data: bytes) -> tuple[int | None, bool]:
        """(authenticated sequence or None, withdraws) for this session's bytes.

        Read-only: no sequence, cache or attempt state changes. Unauthenticated or
        foreign bytes have no trusted sequence and withdraw; an authenticated report
        withdraws when unhealthy, faulted, low-confidence or without acceleration.
        """
        try:
            packet = read_packet(data, self.key)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
            return None, True
        if packet.session_id != self.session_id or packet.build_id != self.build_id:
            return None, True
        return packet.sequence, _withdraws(packet)

    def _application_expired(self, packet: ProposalPacket, now_ns: int) -> bool:
        """An accepted cap applies only inside the proven lag envelope from its anchor.

        The anchor is the older camera capture; a receiver-echo session also needs
        its anchored handoff, inside its configured (tighter or equal) envelope.
        """
        capture = min(camera.capture_ns for camera in packet.cameras)
        if not 0 <= now_ns - capture <= PROVEN_COMMAND_LAG_ENVELOPE_NS:
            return True
        envelope = self.command_lag_envelope_ns
        return envelope is not None and (
            packet.receiver_anchor_ns is None or not 0 <= now_ns - packet.receiver_anchor_ns <= envelope
        )

    def _anchor_ahead(self, packet: ProposalPacket, now_ns: int) -> bool:
        """An echo-session cap whose receiver anchor this consumer clock has not reached yet."""
        anchor = packet.receiver_anchor_ns
        return self.command_lag_envelope_ns is not None and anchor is not None and anchor > now_ns

    def _refusal(self, packet: ProposalPacket) -> str | None:
        """Time-independent reasons an authenticated packet cannot be a cap here, or None."""
        if packet.cameras != self._issued.get(packet.input_sequence):
            return "unissued_or_conflicting_input_identity"
        if _withdraws(packet) or packet.accel is None:
            return "companion_ineligible"
        if not self.min_accel <= packet.accel <= self.max_accel:
            return "companion_acceleration_out_of_bounds"
        if packet.should_stop and packet.accel > 0:
            return "contradictory_stop_and_acceleration"
        return None

    def _configuration(self) -> tuple:
        return (
            hashlib.sha256(self.key).digest(),
            self.session_id,
            self.build_id,
            self.min_accel,
            self.max_accel,
            self.command_lag_envelope_ns,
        )

    def _latch(self, reason: str) -> str:
        """Remove companion authority for the session and record why; returns the decision reason."""
        if self.fault_reason is None:
            self.fault_reason = reason
        self._active = self._active_configuration = None
        self._staged = []
        return COMPANION_DISABLED_AFTER_FAULT + self.fault_reason

    def issue(self, input_sequence: int, cameras: tuple[CameraReference, ...], now_ns: int) -> None:
        if not _integer(input_sequence) or not _integer(now_ns):
            raise ValueError("Invalid input sequence/clock")
        if self._issued and input_sequence <= max(self._issued):
            raise ValueError("Input sequence regression")
        if len(cameras) != 2 or {c.camera for c in cameras} != {"narrow", "wide"}:
            raise ValueError("Both camera references required")
        if any(not 0 <= now_ns - c.capture_ns <= CAMERA_DECISION_AGE_NS for c in cameras):
            raise ValueError("Inputs already stale or future")
        if abs(cameras[0].capture_ns - cameras[1].capture_ns) > 5_000_000:
            raise ValueError("Camera synchronization failed")
        self._issued[input_sequence] = cameras
        while len(self._issued) > 8:
            del self._issued[min(self._issued)]

    def prepare(
        self,
        data: bytes | None,
        *,
        now_ns: int,
        baseline_accel: float,
        baseline_stop: bool,
        enabled: bool,
        override: bool,
        standstill: bool = False,
    ) -> PreparedControl:
        """Authenticate once and prepare the companion outcome on the owning thread.

        Strictly by sequence: the highest authenticated sequence wins, whether it is a
        cap or a withdrawal. An older or duplicate packet is refused as a regression and
        never replaces or cancels anything. A newer withdrawal, or a newer packet that
        cannot be a cap, gives exact native output and cancels every older cap, in force
        or staged. A newer valid cap whose receiver anchor is still ahead is staged
        without disturbing the cap in force; it takes over at the first arbitration at or
        after its anchor, by highest sequence. Unauthenticated or foreign bytes are
        unknown, and unknown is native: they cancel every cap. While native is disabled
        or the driver overrides, the output is exact native, but packets are still
        authenticated and ordered and a valid cap stays in force, within its own
        validity, for when native control returns. An earlier accepted packet cannot
        remain active while this attempt is pending. The returned object is accepted
        only once, by finalize().
        """
        if not _number(baseline_accel, self.min_accel, self.max_accel):
            raise ValueError("Native baseline must already satisfy verified physical bounds")
        if any(type(value) is not bool for value in (baseline_stop, enabled, override, standstill)):
            raise ValueError("Explicit native control state required")
        if not _integer(now_ns):
            raise ValueError("Explicit native control state required")

        cached, self._active = self._active, None
        cached_configuration, self._active_configuration = self._active_configuration, None
        self._generation += 1
        native = ControlDecision(
            baseline_accel, baseline_stop, enabled and not override and not baseline_stop, False, "native_fallback"
        )

        def prepared(packet: ProposalPacket | None, decision: ControlDecision) -> PreparedControl:
            result = PreparedControl(self._generation, now_ns, packet, self._configuration(), native, decision)
            self._pending = result
            return result

        def fallback(reason: str) -> PreparedControl:
            return prepared(None, replace(native, reason=reason))

        def cancel(reason: str) -> PreparedControl:
            # The newest input is not a usable cap: nothing older may apply after it.
            self._staged = []
            return fallback(reason)

        if self.fault_reason is not None:
            return fallback(COMPANION_DISABLED_AFTER_FAULT + self.fault_reason)
        if now_ns < self._last_now:
            return fallback(self._latch("consumer_clock_regression"))
        self._last_now = now_ns
        if self._staged and self._staged_configuration != self._configuration():
            self._staged = []
        fresh = None
        regressed = False
        if data is not None:
            try:
                packet = read_packet(data, self.key)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
                return cancel("invalid_authenticated_packet")
            if packet.session_id != self.session_id or packet.build_id != self.build_id:
                return cancel("session_or_build_mismatch")
            if packet.sequence > self._last_sequence:
                # Consume sequence even when a new authenticated report has lost health.
                self._last_sequence = packet.sequence
                refusal = self._refusal(packet)
                if refusal is not None:
                    return cancel(refusal)
                anchor = packet.receiver_anchor_ns
                if anchor is not None and self._anchor_ahead(packet, now_ns):
                    if anchor - min(camera.capture_ns for camera in packet.cameras) > PROVEN_COMMAND_LAG_ENVELOPE_NS:
                        # Its camera validity ends before its anchor: it can never apply.
                        return cancel("companion_cap_outside_lag_envelope")
                    self._staged.append(packet)
                    self._staged_configuration = self._configuration()
                else:
                    self._staged = []  # Every staged cap is older than this usable one.
                    fresh = packet
            else:
                # Not the newest: refused without dropping the cap in force or any staged cap.
                self.regression_count += 1
                regressed = True
        if fresh is None:
            due = [packet for packet in self._staged if not self._anchor_ahead(packet, now_ns)]
            if due:
                fresh = max(due, key=lambda packet: packet.sequence)
                self._staged = [packet for packet in self._staged if packet.sequence > fresh.sequence]
        if fresh is None and cached is not None and cached_configuration != self._configuration():
            return fallback("cached_consumer_configuration_changed")
        active = cached if fresh is None else fresh
        if active is None:
            if not enabled or override:
                return fallback(NATIVE_DISABLED_OR_DRIVER_OVERRIDE)
            return fallback("packet_replay_or_regression" if regressed else "no_companion_proposal")
        if self._application_expired(active, now_ns):
            return fallback("companion_cap_outside_lag_envelope")
        refusal = self._refusal(active)
        if refusal is not None or active.accel is None:
            return fallback(refusal or "companion_ineligible")
        if not enabled or override:
            # Exact native while native is disabled or the driver overrides. The cap is
            # not applied, but stays in force for when native control returns.
            return prepared(active, replace(native, reason=NATIVE_DISABLED_OR_DRIVER_OVERRIDE))
        if standstill and not active.stop_obligation:
            # No stop obligation of its own: at rest the cap is no constraint and
            # native owns its resume exactly. The cap stays in force for motion.
            return prepared(active, replace(native, reason=STANDSTILL_WITHOUT_OBLIGATION))
        # The packet value is a CAP that only ever lowers native; it holds resume
        # only while it carries its own typed stop obligation.
        decision = ControlDecision(
            min(baseline_accel, active.accel),
            baseline_stop or active.should_stop,
            not active.stop_obligation,
            True,
            "strict_final_braking_min",
        )
        return prepared(active, decision)

    def _refine(
        self, attempt: PreparedControl, *, fallback: ControlDecision, decision: ControlDecision
    ) -> PreparedControl:
        """Bind precomputed final SCC12 outcomes to the same one-shot attempt."""
        if self._pending is not attempt:
            raise ValueError("Cannot refine an unowned or consumed control attempt")
        result = replace(attempt, fallback=fallback, decision=decision)
        self._pending = result
        return result

    def finalize(self, attempt: PreparedControl, *, completion_clock: Callable[[], int] | None) -> ControlDecision:
        """Read trusted consumer-domain time at the final prepared-output handoff.

        The clock must be supplied by a qualified same-domain bridge. There is
        deliberately no wall-clock default or reuse of the entry timestamp.
        A clock that fails (``OSError``, the error a monotonic clock read raises),
        returns an invalid value, regresses, or completes later than the declared
        ``CONTROLLER_COMPLETION_BOUND_NS`` gives exact native output and latches the
        companion off for the session (``fault_reason``); the consumer logs it.
        Any other exception from the clock is a defect and propagates.
        No transport publication or physical CAN-delivery timing is proved here.
        """
        if not isinstance(attempt, PreparedControl):
            raise ValueError("A gate-prepared control attempt is required")
        if self._pending is not attempt or attempt.generation != self._generation:
            raise ValueError("Control attempt is unowned, consumed or superseded")

        def fallback(reason: str) -> ControlDecision:
            return replace(attempt.fallback, reason=reason)

        self._pending = None
        if completion_clock is None or not callable(completion_clock):
            return fallback("consumer_completion_clock_unavailable")
        try:
            completed = completion_clock()
        except OSError as error:
            return fallback(self._latch("consumer_completion_clock_failed:" + type(error).__name__))
        if attempt.generation != self._generation:
            return fallback("control_attempt_consumed_or_superseded")
        if not _integer(completed):
            return fallback(self._latch("consumer_completion_clock_invalid"))
        if completed < max(attempt.now_ns, self._last_now):
            return fallback(self._latch("consumer_completion_clock_regressed"))
        self._last_now = completed
        if completed - attempt.now_ns > CONTROLLER_COMPLETION_BOUND_NS:
            return fallback(self._latch("consumer_completion_clock_late"))
        if attempt.configuration != self._configuration():
            return fallback("consumer_configuration_changed")
        active = attempt.packet
        if active is None:
            return attempt.decision
        if active.cameras != self._issued.get(active.input_sequence):
            return fallback("unissued_or_conflicting_input_identity")
        if self._application_expired(active, completed):
            # The proven lag envelope must hold at the instant the command is used.
            return fallback("companion_cap_outside_lag_envelope")
        # Accepted: the cap stays in force (also through a standstill it does not hold).
        self._active = active
        self._active_configuration = attempt.configuration
        return attempt.decision

    def evaluate(
        self,
        data: bytes | None,
        *,
        now_ns: int,
        baseline_accel: float,
        baseline_stop: bool,
        enabled: bool,
        override: bool,
        standstill: bool = False,
        completion_clock: Callable[[], int] | None = None,
    ) -> ControlDecision:
        """Convenience boundary when no downstream preparation remains."""
        attempt = self.prepare(
            data,
            now_ns=now_ns,
            baseline_accel=baseline_accel,
            baseline_stop=baseline_stop,
            enabled=enabled,
            override=override,
            standstill=standstill,
        )
        return self.finalize(attempt, completion_clock=completion_clock)


def _scc12_emergency_clear(values: dict) -> bool:
    # The pinned native CANPacker encodes omitted signals as zero. These are
    # final outbound commands, not a qualified incoming-AEB state producer.
    for name in ("AEB_CmdAct", "CF_VSM_DecCmdAct", "AEB_StopReq"):
        value = values.get(name, 0)
        if value is False:
            continue
        if not _number(value, 0, 0):
            return False
    command = values.get("CR_VSM_DecCmd", 0)
    # Positive command with clear active flags is inconsistent/unknown; it never
    # lets the companion release a resume. Units are g.
    return _number(command, 0, 2.55) and command == 0


@dataclass(frozen=True)
class PreparedScc12:
    """Native and companion alternatives prepared before the clock read."""

    gate: ControlGate
    attempt: PreparedControl
    values: tuple[dict, dict]
    decisions: tuple[ControlDecision, ControlDecision]

    def finalize(self, completion_clock: Callable[[], int] | None) -> tuple[int, ControlDecision]:
        decision = self.gate.finalize(self.attempt, completion_clock=completion_clock)
        # Fallback reasons may differ, but the selected native fields never do.
        index = 1 if decision is self.decisions[1] else 0
        return index, decision


def _grid(value: float, rounding: str) -> float:
    return float(Decimal(str(value)).quantize(SCC12_GRID, rounding=rounding))


def prepare_scc12(
    values: dict,
    gate: ControlGate,
    active: bytes | None,
    *,
    now_ns: int,
    long_active: bool,
    driver_override: bool,
    standstill: bool = False,
) -> PreparedScc12:
    """Prepare immediately before SCC12 checksum/packing, AFTER tuning and ESCC.

    Braking-only, per field: the UNROUNDED cap is compared with each native field.
    A field the cap does not undercut stays bit-for-bit native; a field it
    undercuts takes the cap quantized toward braking on the pinned CN7 0.01 m/s²
    grid, clamped into the bounds, and only while that is still below the field.
    No path raises a native field. StopReq stays owned by final native tuning and
    stopping state. At a standstill the companion holds (no field above 0, resume
    filtered) only under its own typed stop obligation; otherwise native is exact.
    Invalid native fields raise ``NativeScc12FieldError``.
    This pure helper does not register a live hook or transmit a CAN message.
    """
    if not isinstance(values, dict) or not {"aReqRaw", "aReqValue", "StopReq"} <= values.keys():
        raise NativeScc12FieldError("Native SCC12 fields required")
    raw, actual, stop = (values[key] for key in ("aReqRaw", "aReqValue", "StopReq"))
    if type(stop) not in (int, bool) or stop not in (0, 1):
        raise NativeScc12FieldError("Native StopReq must be a bit")
    lower, upper = max(gate.min_accel, -10.23), min(gate.max_accel, 10.24)
    if not _number(raw, lower, upper) or not _number(actual, lower, upper):
        raise NativeScc12FieldError("Native SCC12 acceleration outside pinned bounds")
    if type(standstill) is not bool:
        raise ValueError("Explicit measured native standstill required")
    # The strongest braking that is both on the grid and inside every bound.
    braking_limit = _grid(lower, ROUND_CEILING)
    emergency_clear = _scc12_emergency_clear(values)
    attempt = gate.prepare(
        active,
        now_ns=now_ns,
        baseline_accel=min(raw, actual),
        baseline_stop=bool(stop),
        enabled=long_active,
        override=driver_override,
        standstill=standstill,
    )
    packet = attempt.packet

    def toward_braking(field: float, cap: float) -> float:
        if not cap < field:
            return field  # The companion does not win this field: bit-exact native.
        quantized = max(braking_limit, _grid(cap, ROUND_FLOOR))
        return quantized if quantized < field else field

    def outcome(decision: ControlDecision) -> tuple[dict, ControlDecision]:
        output = dict(values)
        if not emergency_clear:
            decision = replace(decision, allow_resume=False)
        if not decision.companion_eligible:
            return output, decision
        if packet is None or packet.accel is None:
            raise ValueError("An eligible decision requires its accepted packet")
        cap = packet.accel
        if standstill:
            # Eligible at rest only under the companion's own stop obligation: hold.
            cap = min(cap, 0.0)
            decision = replace(decision, reason=decision.reason + ":companion_stop_obligation_holds")
        output["aReqRaw"] = toward_braking(raw, cap)
        output["aReqValue"] = toward_braking(actual, cap)
        return output, replace(decision, accel=min(output["aReqRaw"], output["aReqValue"]))

    native, native_decision = outcome(attempt.fallback)
    companion, companion_decision = outcome(attempt.decision)
    refined = gate._refine(attempt, fallback=native_decision, decision=companion_decision)
    return PreparedScc12(gate, refined, (native, companion), (native_decision, companion_decision))


def arbitrate_scc12(
    values: dict,
    gate: ControlGate,
    active: bytes | None,
    *,
    now_ns: int,
    long_active: bool,
    driver_override: bool,
    completion_clock: Callable[[], int] | None = None,
    standstill: bool = False,
) -> tuple[dict, ControlDecision]:
    """Synchronous field-only helper; native packing requires prepare_scc12()."""
    pending = prepare_scc12(
        values,
        gate,
        active,
        now_ns=now_ns,
        long_active=long_active,
        driver_override=driver_override,
        standstill=standstill,
    )
    index, decision = pending.finalize(completion_clock)
    return pending.values[index], decision
