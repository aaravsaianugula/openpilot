"""Bounded native observations; no movement permission or control authority.

Transport authentication identifies a provisioned peer, not a trustworthy physical
measurement. Native msgq can lose unread history without a loss counter. This
module therefore never qualifies even a correctly clock-joined observation.
"""

import collections
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any

from vision_long_publisher.extract_companion_context import MAX_AGE_NS, REQUIRED, payload_issue

MAX_CONTEXT_BYTES = 128 * 1024
BUILD_PINS = (
    ("openpilot", "13aedc522518fe1b62651d9a8a320e931efc335c"),
    ("opendbc", "b45eb78def29582bcce43c68cfb1d55f19224d1a"),
    ("msgq", "e7396e76dadbb49e374d4b664ff6bbb43a39bcb0"),
)
SOURCE_PINS = (
    ("openpilot/cereal/log.capnp", "0b1f601818dc8967b5131b1706a1db4fcae175baed83440ac51d0277cda4ff24"),
    ("openpilot/cereal/custom.capnp", "d439fdd23e665ec2d74039619c903219386ac6ce3ddadfbc279ae669b9dd1cd6"),
    ("openpilot/cereal/deprecated.capnp", "b3aa5c13de40001ed7376cef858500fb003f7698f96131cd0db18d878fa1af0d"),
    ("openpilot/cereal/include/c++.capnp", "fb306076cd38c27af1aed20fac6395e9a46fbe5b5df6a248b5f1b6845a079c44"),
    ("opendbc_repo/opendbc/car/car.capnp", "b971083a0cb3aed287ca9c54572d470fb798e633bc47611717f697b9e735f5a1"),
    ("opendbc_repo/opendbc/car/include/c++.capnp", "fb306076cd38c27af1aed20fac6395e9a46fbe5b5df6a248b5f1b6845a079c44"),
    ("openpilot/cereal/messaging/__init__.py", "ef101de88d76a55f5724c6142d7f073a21b9024eb5b2ffce49ca7b9e5ea929f0"),
    ("openpilot/cereal/visionipc.py", "ec6eb707b73dcbd44b8b5482f2b0d20af73520f2238af35d6cc0ffae6f7c7b54"),
    ("msgq_repo/msgq/ipc_pyx.pyx", "02cf21d0232967eed7bd68ed0209e55f5f1e3edef3675671bc6524ca64976fa4"),
    ("msgq_repo/msgq/visionipc/visionipc_pyx.pyx", "346e7a53985132855954e62f4274984c2df39abe88ae2a3d050906327d9d0b83"),
)


def integer(value: Any, name: str, minimum: int = 0, maximum: int = 2**64 - 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid_" + name)


def identity(value: Any) -> None:
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("invalid_uuid")


def _tree(value: Any) -> None:
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 24 or nodes > 32768:
            raise ValueError("context_structure_limit")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("context_json_key")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("context_nonfinite_json")
        elif type(item) not in (str, int, bool, type(None)):
            raise ValueError("context_json_type")


def canonical_json(value: Any) -> bytes:
    _tree(value)
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(result) > MAX_CONTEXT_BYTES:
        raise ValueError("context_byte_limit")
    return result


def read_json(payload: bytes) -> Any:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_CONTEXT_BYTES:
        raise ValueError("context_byte_limit")

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("context_duplicate_key")
            result[key] = item
        return result

    def constant(_):
        raise ValueError("context_nonfinite_json")

    try:
        value = json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
        _tree(value)
        return value
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("context_invalid_json") from exc


@dataclass(frozen=True)
class CameraIdentity:
    camera: str
    frame_id: int
    sof_ns: int
    transport_sequence: int

    def __post_init__(self):
        if self.camera not in ("narrow", "wide"):
            raise ValueError("invalid_camera")
        for name in ("frame_id", "sof_ns", "transport_sequence"):
            integer(getattr(self, name), name, 1 if name == "transport_sequence" else 0)

    def to_dict(self) -> dict:
        return dict(
            camera=self.camera, frame_id=self.frame_id, sof_ns=self.sof_ns, transport_sequence=self.transport_sequence
        )


@dataclass(frozen=True)
class NativeEvent:
    service: str
    log_mono_ns: int
    valid: bool
    data_json: bytes
    observed_sequence: int

    def __post_init__(self):
        if self.service not in REQUIRED or type(self.valid) is not bool:
            raise ValueError("invalid_native_service_or_valid")
        integer(self.log_mono_ns, "native_timestamp")
        integer(self.observed_sequence, "event_sequence", 1)
        if type(read_json(self.data_json)) is not dict:
            raise ValueError("native_data_object_required")

    @property
    def data(self) -> dict:
        return read_json(self.data_json)

    def to_dict(self) -> dict:
        return dict(
            service=self.service,
            log_mono_ns=self.log_mono_ns,
            valid=self.valid,
            data=self.data,
            observed_sequence=self.observed_sequence,
        )


MAX_ECHO_COMMANDS = 32
MAX_ECHO_WINDOW_NS = 60_000_000_000
MAX_ECHO_ACCEL_MPS2 = 15.0
_ECHO_KEYS = frozenset(
    {
        "schema",
        "session_id",
        "build_id",
        "clock_id",
        "echo_sequence",
        "echo_ns",
        "window_start_ns",
        "effect_delay_ns",
        "handoff_period_ns",
        "next_handoff_ns",
        "accepted",
        "pending",
        "active_age_ns",
    }
)
_ECHO_COMMAND_KEYS = frozenset({"decision_ns", "apply_ns", "accel_mps2", "companion"})


def _bounded_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not 0 < len(value) <= 128 or not value.isascii() or not value.isprintable():
        raise ValueError("invalid_" + name)


@dataclass(frozen=True)
class EchoedCommand:
    """One final SCC12 acceleration the consumer handed to the native sender.

    ``decision_ns`` is the native cycle that arbitrated and handed it off;
    ``apply_ns`` is that handoff plus the receiver's declared effect delay.
    Both are integer ns on the consumer's trusted clock.
    """

    decision_ns: int
    apply_ns: int
    accel_mps2: float
    companion: bool

    def __post_init__(self):
        integer(self.decision_ns, "echo_decision_ns", 0, 2**63 - 1)
        integer(self.apply_ns, "echo_apply_ns", 0, 2**63 - 1)
        if self.apply_ns < self.decision_ns:
            raise ValueError("echo_apply_before_decision")
        if (
            type(self.accel_mps2) is not float
            or not math.isfinite(self.accel_mps2)
            or abs(self.accel_mps2) > MAX_ECHO_ACCEL_MPS2
        ):
            raise ValueError("invalid_echo_accel")
        if type(self.companion) is not bool:
            raise ValueError("invalid_echo_source")

    def to_dict(self) -> dict:
        return dict(
            decision_ns=self.decision_ns, apply_ns=self.apply_ns, accel_mps2=self.accel_mps2, companion=self.companion
        )


def echo_active_age_ns(accepted: tuple[EchoedCommand, ...], echo_ns: int) -> int:
    """How long the applied value has held at ``echo_ns``, from apply timestamps.

    The run of equal values ending at the newest applied command starts at its
    first retained member. Older history is outside the echo window, so this is
    a lower bound whenever the run reaches the oldest retained command.
    """
    if not accepted:
        raise ValueError("echo_has_no_applied_command")
    start = accepted[-1].apply_ns
    for command in reversed(accepted):
        if command.accel_mps2 != accepted[-1].accel_mps2:
            break
        start = command.apply_ns
    return echo_ns - start


@dataclass(frozen=True)
class ReceiverStateEcho:
    """The native consumer's receiver state, echoed inside one context record.

    ``accepted`` are handed-off commands already applied by ``echo_ns``;
    ``pending`` are handed off but not yet applied. ``next_handoff_ns`` is the
    handoff phase: the earliest next SCC12 handoff under the declared period.
    This is bookkeeping of what was sent; it proves nothing physical.
    """

    session_id: str
    build_id: str
    clock_id: str
    echo_sequence: int
    echo_ns: int
    window_start_ns: int
    effect_delay_ns: int
    handoff_period_ns: int
    next_handoff_ns: int
    accepted: tuple[EchoedCommand, ...]
    pending: tuple[EchoedCommand, ...]
    active_age_ns: int

    def __post_init__(self):
        identity(self.session_id)
        _bounded_text(self.build_id, "echo_build_id")
        _bounded_text(self.clock_id, "echo_clock_id")
        integer(self.echo_sequence, "echo_sequence", 1)
        integer(self.echo_ns, "echo_ns", 0, 2**63 - 1)
        integer(self.window_start_ns, "echo_window_start_ns", 0, 2**63 - 1)
        integer(self.effect_delay_ns, "echo_effect_delay_ns", 1, 1_000_000_000)
        integer(self.handoff_period_ns, "echo_handoff_period_ns", 1, 1_000_000_000)
        integer(self.next_handoff_ns, "echo_next_handoff_ns", 0, 2**63 - 1)
        if not self.window_start_ns <= self.echo_ns <= self.window_start_ns + MAX_ECHO_WINDOW_NS:
            raise ValueError("echo_window_invalid")
        if not self.echo_ns < self.next_handoff_ns <= self.echo_ns + self.handoff_period_ns:
            raise ValueError("echo_handoff_phase_invalid")
        for name in ("accepted", "pending"):
            commands = getattr(self, name)
            if (
                type(commands) is not tuple
                or len(commands) > MAX_ECHO_COMMANDS
                or any(not isinstance(item, EchoedCommand) for item in commands)
            ):
                raise ValueError("echo_queue_bound")
        everything = (*self.accepted, *self.pending)
        for earlier, later in zip(everything[:-1], everything[1:], strict=True):
            if later.decision_ns <= earlier.decision_ns or later.apply_ns <= earlier.apply_ns:
                raise ValueError("echo_commands_out_of_order")
        if any(
            item.apply_ns - item.decision_ns != self.effect_delay_ns or item.decision_ns < self.window_start_ns
            for item in everything
        ):
            raise ValueError("echo_command_outside_contract_or_window")
        if any(item.apply_ns > self.echo_ns for item in self.accepted) or any(
            item.apply_ns <= self.echo_ns or item.decision_ns > self.echo_ns for item in self.pending
        ):
            raise ValueError("echo_applied_pending_split_invalid")
        if self.active_age_ns != echo_active_age_ns(self.accepted, self.echo_ns):
            raise ValueError("echo_active_age_mismatch")

    def to_dict(self) -> dict:
        return dict(
            schema="receiver_state_echo_v1",
            session_id=self.session_id,
            build_id=self.build_id,
            clock_id=self.clock_id,
            echo_sequence=self.echo_sequence,
            echo_ns=self.echo_ns,
            window_start_ns=self.window_start_ns,
            effect_delay_ns=self.effect_delay_ns,
            handoff_period_ns=self.handoff_period_ns,
            next_handoff_ns=self.next_handoff_ns,
            accepted=[item.to_dict() for item in self.accepted],
            pending=[item.to_dict() for item in self.pending],
            active_age_ns=self.active_age_ns,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ReceiverStateEcho":
        if type(value) is not dict or set(value) != _ECHO_KEYS or value["schema"] != "receiver_state_echo_v1":
            raise ValueError("receiver_echo_schema")
        queues: dict[str, Any] = {}
        for name in ("accepted", "pending"):
            items = value[name]
            if type(items) is not list or len(items) > MAX_ECHO_COMMANDS:
                raise ValueError("echo_queue_bound")
            if any(type(item) is not dict or set(item) != _ECHO_COMMAND_KEYS for item in items):
                raise ValueError("receiver_echo_command_schema")
            queues[name] = tuple(EchoedCommand(**item) for item in items)
        fields = {key: item for key, item in value.items() if key not in ("schema", "accepted", "pending")}
        return cls(**fields, **queues)


@dataclass(frozen=True)
class NativeContextRecord:
    session_id: str
    boot_id: str
    sequence: int
    event_clock_id: str
    camera_clock_id: str
    cameras: tuple[CameraIdentity, CameraIdentity]
    events: tuple[NativeEvent, ...]
    source_pins: tuple[tuple[str, str], ...] = SOURCE_PINS
    build_pins: tuple[tuple[str, str], ...] = BUILD_PINS
    drain_limited: bool = False
    receiver_echo: ReceiverStateEcho | None = None

    def __post_init__(self):
        identity(self.session_id)
        identity(self.boot_id)
        integer(self.sequence, "context_sequence", 1)
        if self.event_clock_id != self.boot_id + ":CLOCK_MONOTONIC":
            raise ValueError("invalid_event_clock_identity")
        if self.camera_clock_id != self.boot_id + ":CLOCK_BOOTTIME":
            raise ValueError("invalid_camera_clock_identity")
        if type(self.cameras) is not tuple or tuple(item.camera for item in self.cameras) != ("narrow", "wide"):
            raise ValueError("both_original_cameras_required")
        if type(self.events) is not tuple or len(self.events) > 128:
            raise ValueError("native_event_count_limit")
        if self.source_pins != SOURCE_PINS or self.build_pins != BUILD_PINS:
            raise ValueError("native_source_identity_changed")
        if type(self.drain_limited) is not bool:
            raise ValueError("invalid_drain_status")
        if self.receiver_echo is not None and not isinstance(self.receiver_echo, ReceiverStateEcho):
            raise ValueError("invalid_receiver_echo")

    def to_dict(self) -> dict:
        return dict(
            schema="native_context_v1",
            session_id=self.session_id,
            boot_id=self.boot_id,
            sequence=self.sequence,
            event_clock_id=self.event_clock_id,
            camera_clock_id=self.camera_clock_id,
            cameras=[item.to_dict() for item in self.cameras],
            events=[item.to_dict() for item in self.events],
            source_pins=dict(self.source_pins),
            build_pins=dict(self.build_pins),
            drain_limited=self.drain_limited,
            receiver_echo=self.receiver_echo.to_dict() if self.receiver_echo is not None else None,
            native_queue_history="unknown_possible_loss",
            physical_capture_delay="unknown",
            source_clock_relation="unknown",
            shared_camera_buffer_ownership="unknown_stable_generation_only",
            compiled_native_binary_verified=False,
            control_qualified=False,
        )

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def from_bytes(cls, payload: bytes):
        value = read_json(payload)
        keys = set(cls.__dataclass_fields__) | {
            "schema",
            "native_queue_history",
            "physical_capture_delay",
            "source_clock_relation",
            "shared_camera_buffer_ownership",
            "compiled_native_binary_verified",
            "control_qualified",
        }
        if type(value) is not dict or set(value) != keys:
            raise ValueError("native_context_schema")
        if (
            value.pop("schema") != "native_context_v1"
            or value.pop("native_queue_history") != "unknown_possible_loss"
            or value.pop("physical_capture_delay") != "unknown"
            or value.pop("source_clock_relation") != "unknown"
            or value.pop("shared_camera_buffer_ownership") != "unknown_stable_generation_only"
            or value.pop("compiled_native_binary_verified") is not False
            or value.pop("control_qualified") is not False
        ):
            raise ValueError("native_context_qualification_claim")
        if value["source_pins"] != dict(SOURCE_PINS) or value["build_pins"] != dict(BUILD_PINS):
            raise ValueError("native_source_identity_changed")
        try:
            if (
                type(value["events"]) is not list
                or len(value["events"]) > 128
                or type(value["cameras"]) is not list
                or len(value["cameras"]) != 2
            ):
                raise ValueError("native_context_count_limit")
            value["cameras"] = tuple(CameraIdentity(**item) for item in value["cameras"])
            events = []
            for item in value["events"]:
                if type(item) is not dict or set(item) != {
                    "service",
                    "log_mono_ns",
                    "valid",
                    "data",
                    "observed_sequence",
                }:
                    raise ValueError("native_event_schema")
                events.append(
                    NativeEvent(
                        item["service"],
                        item["log_mono_ns"],
                        item["valid"],
                        canonical_json(item["data"]),
                        item["observed_sequence"],
                    )
                )
            value["events"] = tuple(events)
            echo = value["receiver_echo"]
            value["receiver_echo"] = None if echo is None else ReceiverStateEcho.from_dict(echo)
            value["source_pins"], value["build_pins"] = SOURCE_PINS, BUILD_PINS
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise ValueError("native_context_schema") from exc


@dataclass(frozen=True)
class ClockRelation:
    """External logical timestamp contract, never a measured physical delay bound."""

    event_clock_id: str
    camera_clock_id: str
    boot_id: str
    offset_low_ns: int  # camera clock minus event clock
    offset_high_ns: int
    anchor_event_ns: int
    valid_span_ns: int
    evidence_id: str

    def __post_init__(self):
        identity(self.boot_id)
        if (
            self.event_clock_id != self.boot_id + ":CLOCK_MONOTONIC"
            or self.camera_clock_id != self.boot_id + ":CLOCK_BOOTTIME"
        ):
            raise ValueError("clock_relation_identity")
        integer(self.offset_low_ns, "offset_low", -(2**63), 2**63 - 1)
        integer(self.offset_high_ns, "offset_high", -(2**63), 2**63 - 1)
        integer(self.anchor_event_ns, "clock_anchor")
        integer(self.valid_span_ns, "clock_span", 1, 60_000_000_000)
        if self.offset_low_ns > self.offset_high_ns or not isinstance(self.evidence_id, str):
            raise ValueError("clock_relation_interval")
        if not self.evidence_id.strip() or len(self.evidence_id) > 256:
            raise ValueError("clock_relation_evidence")

    @classmethod
    def from_bytes(cls, payload: bytes):
        try:
            return cls(**read_json(payload))
        except TypeError as exc:
            raise ValueError("clock_relation_schema") from exc


class NativeHistory:
    def __init__(
        self, per_service_capacity: int = 32, max_bytes: int = 2 * 1024 * 1024, max_age_ns: dict[str, int] | None = None
    ):
        integer(per_service_capacity, "history_capacity", 1, 128)
        integer(max_bytes, "history_bytes", MAX_CONTEXT_BYTES, 8 * 1024 * 1024)
        self.capacity, self.max_bytes = per_service_capacity, max_bytes
        self.max_age = dict(MAX_AGE_NS)
        for service, age in (max_age_ns or {}).items():
            if service not in REQUIRED:
                raise ValueError("unknown_freshness_service")
            integer(age, "freshness_age", 1, 300_000_000_000)
            self.max_age[service] = age
        self.begin_session(None, None)

    def begin_session(self, session_id: str | None, boot_id: str | None):
        if session_id is not None:
            identity(session_id)
            identity(boot_id)
        self.session_id, self.boot_id = session_id, boot_id
        self.history: dict[str, collections.deque[NativeEvent]] = {key: collections.deque() for key in REQUIRED}
        self.evicted_through: dict[str, int | None] = dict.fromkeys(REQUIRED)
        self.last_sequence = self.last_event_sequence = self.bytes = 0
        self.fault: str | None = None

    def accept(self, record: NativeContextRecord):
        if record.session_id != self.session_id or record.boot_id != self.boot_id:
            self.fault = "session_mismatch"
            raise ValueError(self.fault)
        if record.sequence <= self.last_sequence:
            self.fault = "context_sequence_regression"
            raise ValueError(self.fault)
        if record.sequence != self.last_sequence + 1:
            self.fault = "context_sequence_gap"
        self.last_sequence = record.sequence
        if record.drain_limited:
            self.fault = "bounded_native_drain_limit"
        for event in record.events:
            if event.observed_sequence <= self.last_event_sequence:
                self.fault = "event_sequence_regression"
                raise ValueError(self.fault)
            if event.observed_sequence != self.last_event_sequence + 1:
                self.fault = "event_sequence_gap"
            self.last_event_sequence = event.observed_sequence
            queue = self.history[event.service]
            if queue and event.log_mono_ns <= queue[-1].log_mono_ns:
                self.fault = "native_timestamp_ambiguous_or_regressed"
                raise ValueError(self.fault)
            if len(queue) == self.capacity:
                evicted = queue.popleft()
                self.bytes -= len(evicted.data_json)
                self.evicted_through[event.service] = evicted.log_mono_ns
            if self.bytes + len(event.data_json) > self.max_bytes:
                self.fault = "history_byte_limit"
                raise ValueError(self.fault)
            queue.append(event)
            self.bytes += len(event.data_json)

    def join(self, cameras: tuple[CameraIdentity, CameraIdentity], relation: ClockRelation | None = None) -> dict:
        cutoff = min(item.sof_ns for item in cameras)
        result = {}
        for service, queue in self.history.items():
            item: dict[str, Any] = dict(
                status="unknown",
                reason=self.fault or "source_clock_relation_unknown",
                source=None,
                data=None,
                temporal_selection_status="unknown",
                physical_capture_delay="unknown",
                control_qualified=False,
                history_evicted_through_ns=self.evicted_through[service],
                history_retained_from_ns=queue[0].log_mono_ns if queue else None,
            )
            result[service] = item
            if self.fault or relation is None:
                continue
            if relation.boot_id != self.boot_id:
                item["reason"] = "clock_relation_identity"
                continue
            start = relation.anchor_event_ns + relation.offset_high_ns
            end = relation.anchor_event_ns + relation.valid_span_ns + relation.offset_low_ns
            if not start <= cutoff <= end:
                item["reason"] = "source_clock_relation_expired"
                continue
            selected = None
            ambiguous = False
            for event in queue:
                low = event.log_mono_ns + relation.offset_low_ns
                high = event.log_mono_ns + relation.offset_high_ns
                if low <= cutoff < high:
                    ambiguous = True
                elif high <= cutoff:
                    selected = event
            if ambiguous:
                item["reason"] = "ambiguous_clock_order"
                continue
            if selected is None:
                # Normal bounded retention is not source loss. A retained
                # causal latest event supersedes every earlier eviction, but
                # a cutoff before it cannot reconstruct an evicted value.
                item["reason"] = (
                    "required_history_evicted" if self.evicted_through[service] is not None else "no_causal_message"
                )
                continue
            if (
                not relation.anchor_event_ns
                <= selected.log_mono_ns
                <= relation.anchor_event_ns + relation.valid_span_ns
            ):
                item["reason"] = "source_clock_relation_expired"
                continue
            age = cutoff - (selected.log_mono_ns + relation.offset_low_ns)
            data = selected.data
            item.update(
                data=data,
                source=dict(
                    service=service,
                    log_mono_ns=selected.log_mono_ns,
                    valid=selected.valid,
                    observed_sequence=selected.observed_sequence,
                ),
                age_upper_ns=age,
            )
            missing = [field for field in REQUIRED[service] if field not in data]
            if not selected.valid:
                item["reason"] = "event_invalid"
            elif age > self.max_age[service]:
                item["reason"] = "message_stale"
            elif missing:
                item.update(reason="missing_native_fields", missing_fields=missing)
            else:
                try:
                    issue = payload_issue(service, data)
                except (KeyError, TypeError, IndexError):
                    issue = "malformed_native_nested_fields"
                item["reason"] = issue or "native_queue_history_unknown"
                if issue is None:
                    item["temporal_selection_status"] = "known_under_declared_clock_relation"
        return result


class ContextFrameBinder:
    """Require context references to equal actual camera packets already received."""

    def __init__(self, session_id: str, capacity: int = 8):
        identity(session_id)
        integer(capacity, "camera_reference_capacity", 1, 32)
        self.session_id, self.capacity = session_id, capacity
        self.frames: dict[str, collections.deque[CameraIdentity]] = {
            camera: collections.deque() for camera in ("narrow", "wide")
        }

    def add(self, session_id: str, camera: CameraIdentity):
        if session_id != self.session_id:
            raise ValueError("camera_reference_session_mismatch")
        queue = self.frames[camera.camera]
        if queue and (
            camera.transport_sequence <= queue[-1].transport_sequence
            or camera.frame_id <= queue[-1].frame_id
            or camera.sof_ns <= queue[-1].sof_ns
        ):
            raise ValueError("camera_reference_regression")
        if len(queue) == self.capacity:
            queue.popleft()
        queue.append(camera)

    def verify(self, record: NativeContextRecord):
        if record.session_id != self.session_id:
            raise ValueError("camera_reference_session_mismatch")
        if any(camera not in self.frames[camera.camera] for camera in record.cameras):
            raise ValueError("camera_reference_unobserved")
