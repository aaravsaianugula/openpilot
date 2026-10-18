"""Explicit native startup for the road runtime: one config file, one toggle, declared contracts.

``native_runtime_v2`` starts the card-lifetime publisher worker and an authority provider whose
gate may apply the braking-only companion cap only while the device braking toggle is ON. The
provider supplies the clock bridge (``boot:CLOCK_BOOTTIME``, the camera SOF domain, read by this
same process at cycle start and at controller completion) and the declared receiver contract, so
the device produces real receiver echoes. The declared constants below are labelled
``declared_not_measured``. JSON selects addresses and files only; it cannot widen any bound.
The retired ``native_diagnostic_return_v1`` schema is refused with a clear status.
All network/camera work belongs to the publisher worker, never stage_cycle.
"""

import importlib.util
import ipaddress
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .braking_toggle import BrakingToggle
from .integration import CONTROLLER_COMPLETION_BOUND_NS, PROVEN_COMMAND_LAG_ENVELOPE_NS, ControlGate
from .native_bridge import NativeReturnBridge
from .native_consumer import UNAUTHENTICATED_LATCH_COUNT, UNAUTHENTICATED_WINDOW_NS, NativeScc12Consumer

# opendbc.car.carlog's logger name: card.py forwards it to cloudlog in the car process.
CAR_PROCESS_LOGGER = "carlog"
RUNTIME_SCHEMA = "native_runtime_v2"
RETIRED_SCHEMA = "native_diagnostic_return_v1"
BOOT_ID_FILE = "/proc/sys/kernel/random/boot_id"

# ---- Declared constants: status declared_not_measured (no measurement backs these numbers) ----
DECLARED_STATUS = "declared_not_measured"
# SCC12 leaves on even 100 Hz frames (DT_CTRL 10 ms): one handoff every 20 ms.
DECLARED_SCC12_HANDOFF_PERIOD_NS = 20_000_000
# Handoff to effect: the platform's own declared longitudinalActuatorDelay for HKG (0.5 s).
DECLARED_SCC12_EFFECT_DELAY_NS = 500_000_000
# Companion floor: the planner's max_braking and Hyundai CarControllerParams.ACCEL_MIN.
COMPANION_MIN_ACCEL = -3.5
# Hyundai CarControllerParams.ACCEL_MAX: the bound native fields are validated against. The
# gate is braking-only, so a cap never raises a field; this is not a companion acceleration.
COMPANION_MAX_ACCEL = 2.0
# The build identity both sides sign and echo: the laptop's proposals must carry exactly this.
ROAD_BUILD_ID = "vision-long-road-v1"
# The anchored command-path envelope the gate enforces (the proven 100 ms, never wider).
DECLARED_COMMAND_LAG_ENVELOPE_NS = PROVEN_COMMAND_LAG_ENVELOPE_NS
DECLARED_CONSTANTS = {
    "status": DECLARED_STATUS,
    "build_id": ROAD_BUILD_ID,
    "scc12_handoff_period_ns": DECLARED_SCC12_HANDOFF_PERIOD_NS,
    "scc12_effect_delay_ns": DECLARED_SCC12_EFFECT_DELAY_NS,
    "companion_min_accel_mps2": COMPANION_MIN_ACCEL,
    "native_max_accel_mps2": COMPANION_MAX_ACCEL,
    "command_lag_envelope_ns": DECLARED_COMMAND_LAG_ENVELOPE_NS,
    # integration.CONTROLLER_COMPLETION_BOUND_NS: half the 10 ms frame; a later completion latches.
    "controller_completion_bound_ns": CONTROLLER_COMPLETION_BOUND_NS,
    # native_consumer: this many unauthenticated return packets inside the window latch the session.
    "unauthenticated_latch_count": UNAUTHENTICATED_LATCH_COUNT,
    "unauthenticated_window_ns": UNAUTHENTICATED_WINDOW_NS,
}
# Status-line reasons are one token: anything else becomes "_".
_REASON_UNSAFE = re.compile(r"[^A-Za-z0-9_:.=;-]")
MAX_REASON_CHARS = 160
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


class PublisherHandle(Protocol):
    @property
    def is_alive(self) -> bool: ...
    @property
    def summary(self) -> str: ...
    @property
    def failures(self) -> int: ...
    @property
    def restarts(self) -> int: ...
    def stop(self) -> None: ...


def _data_path(value: object, error: str) -> str:
    if type(value) is not str or not 1 <= len(value) <= 1024 or not value.isascii() or not value.isprintable():
        raise ValueError(error)
    path = PurePosixPath(value)
    if not path.is_absolute() or path.parts[:2] != ("/", "data") or ".." in path.parts or len(path.parts) < 3:
        raise ValueError(error)
    return value


def runtime_config(data: bytes) -> dict:
    """The one bounded ``native_runtime_v2`` schema; the retired v1 schema is refused by name."""
    if type(data) is not bytes or not 0 < len(data) <= 4096:
        raise ValueError("native_runtime_config_size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("native_runtime_config_duplicate")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=pairs)
    if type(value) is dict and value.get("schema") == RETIRED_SCHEMA:
        raise ValueError("native_runtime_v1_retired_install_native_runtime_v2")
    keys = {"schema", "listen_ip", "port", "key_file", "toggle_file"}
    if type(value) is not dict or set(value) != keys or value["schema"] != RUNTIME_SCHEMA:
        raise ValueError("native_runtime_config_schema")
    if type(value["listen_ip"]) is not str:
        raise ValueError("native_runtime_address")
    if not isinstance(ipaddress.ip_address(value["listen_ip"]), ipaddress.IPv4Address):
        raise ValueError("native_runtime_ipv4_required")
    if type(value["port"]) is not int or not 1024 <= value["port"] <= 65535:
        raise ValueError("native_runtime_port")
    _data_path(value["key_file"], "native_runtime_key_must_be_provisioned_under_data")
    _data_path(value["toggle_file"], "native_runtime_toggle_must_be_under_data")
    return value


@dataclass(frozen=True)
class AuthorityProvider:
    """Everything the bridge needs to admit a session, from this process and the declared block."""

    key: bytes
    build_id: str
    boot_id: str
    completion_clock: Callable[[], int]
    braking_enabled: Callable[[], bool]

    def __post_init__(self) -> None:
        if type(self.key) is not bytes or not 32 <= len(self.key) <= 4096:
            raise ValueError("bounded_provisioned_transport_key_required")
        if not _IDENTIFIER.fullmatch(self.build_id) or not _IDENTIFIER.fullmatch(self.boot_id):
            raise ValueError("authority_identity_required")
        if not callable(self.completion_clock) or not callable(self.braking_enabled):
            raise ValueError("authority_clock_and_toggle_required")

    @property
    def trusted_clock_id(self) -> str:
        # The publisher stamps camera SOF in this domain, in this same process.
        return self.boot_id + ":CLOCK_BOOTTIME"

    def gate(self, session_id: str) -> ControlGate:
        return ControlGate(
            self.key,
            session_id,
            self.build_id,
            COMPANION_MIN_ACCEL,
            COMPANION_MAX_ACCEL,
            command_lag_envelope_ns=DECLARED_COMMAND_LAG_ENVELOPE_NS,
        )

    def bridge(self) -> NativeReturnBridge:
        return NativeReturnBridge(
            self.gate,
            completion_clock=self.completion_clock,
            trusted_clock_id=self.trusted_clock_id,
            receiver_contract=(DECLARED_SCC12_EFFECT_DELAY_NS, DECLARED_SCC12_HANDOFF_PERIOD_NS),
            fault_log=logging.getLogger(CAR_PROCESS_LOGGER),
            braking_enabled=self.braking_enabled,
        )


def _reason(text: str) -> str:
    return _REASON_UNSAFE.sub("_", text)[:MAX_REASON_CHARS] or "unknown"


class NativeBinding:
    def __init__(
        self,
        bridge: NativeReturnBridge,
        worker: PublisherHandle | None = None,
        *,
        braking_enabled: Callable[[], bool],
        cycle_clock: Callable[[], int],
    ):
        if not isinstance(bridge, NativeReturnBridge) or not callable(braking_enabled) or not callable(cycle_clock):
            raise ValueError("typed_native_bridge_required")
        self.bridge = bridge
        self.worker = worker
        self.braking_enabled = braking_enabled
        self.cycle_now = cycle_clock
        self.stopped = False
        self.last_status = "awaiting_native_cycle"
        # The status line is rebuilt only when one of its inputs changes, never per cycle.
        self._state_key: tuple | None = None
        self._state: tuple[str, str, str] = ("braking_only", "disabled", "awaiting_native_cycle")

    def stage_cycle(self, now_ns: int) -> NativeScc12Consumer | None:
        """Bounded mailbox only; nothing here blocks the controller thread.

        ``now_ns`` must come from ``cycle_now`` (the camera clock domain), never card's now_nanos.
        """
        if self.stopped:
            self.last_status = "native_publisher_stopped"
            return None
        try:
            consumer = self.bridge.stage_cycle(now_ns)
            if self.worker is not None and not self.worker.is_alive:
                # Still fold parent-queued return faults/close on this actual
                # controller clock while the observer is down. Never expose a cap.
                self.last_status = "native_publisher_stopped"
                return None
            self.last_status = self.bridge.status
            return consumer
        except Exception as error:
            # This is an optional adapter fault boundary. Preserve native
            # control, expose the fault, and never retain a prior consumer.
            self.last_status = "native_bridge_failed:" + type(error).__name__
            self.stopped = True
            if self.worker is not None:
                self.worker.stop()
            return None

    def status_state(self, consumer: NativeScc12Consumer | None) -> tuple[str, str, str]:
        """(mode, authority, reason) for the one status line; OFF always wins.

        Per cycle this only compares a small tuple of current inputs; the reason is built,
        sanitised and published only when that tuple changes.
        """
        toggle = self.braking_enabled()
        latched = consumer.fault_reason if consumer is not None else None
        latched = latched or self.bridge.latched_fault
        worker = (None, None) if self.worker is None else (self.worker.failures, self.worker.restarts)
        key = (toggle, consumer is None, latched, self.last_status, worker, self.bridge.status_file_unwritable)
        if key == self._state_key:
            return self._state
        self._state_key = key
        if not toggle:
            state = ("observe", "disabled", "braking_toggle_off")
        else:
            if latched is not None:
                state = ("braking_only", "latched_native", _reason(latched))
            elif consumer is not None:
                state = ("braking_only", "enabled", "braking_toggle_on")
            else:
                summary = "" if self.worker is None else self.worker.summary
                state = ("braking_only", "disabled", _reason(self.last_status + (";" + summary if summary else "")))
        if self.bridge.status_file_unwritable:
            # Notification-only file: its failure is reported, never acted on.
            state = (state[0], state[1], state[2] + ";status_json_unwritable")
        mode, authority, reason = state
        self.bridge.status_fields = (mode, authority, reason, mode != "observe")
        self.bridge.status_line = f"vision_long_companion mode={mode} authority={authority} reason={reason}"
        self._state = state
        return state

    def stop(self) -> None:
        self.stopped = True
        if self.worker is not None:
            self.worker.stop()  # Event only; no join on the controller thread.


class NativeController(Protocol):
    vision_long_consumer: NativeScc12Consumer | None
    vision_long_bridge: NativeBinding | None


def bind_native_controller(
    controller: NativeController,
    provider: AuthorityProvider,
    cycle_clock: Callable[[], int],
    worker: PublisherHandle | None = None,
) -> NativeBinding:
    """Bind a typed authority provider's bridge to the controller's reviewed hook."""
    if not hasattr(controller, "vision_long_bridge"):
        raise ValueError("reviewed_native_bootstrap_hook_required")
    if not isinstance(provider, AuthorityProvider):
        raise ValueError("typed_authority_provider_required")
    return _attach(
        controller,
        NativeBinding(provider.bridge(), worker, braking_enabled=provider.braking_enabled, cycle_clock=cycle_clock),
    )


def _attach(controller: NativeController, binding: NativeBinding) -> NativeBinding:
    previous = controller.vision_long_bridge
    if previous is not None:
        previous.stop()
    controller.vision_long_consumer = None
    controller.vision_long_bridge = binding
    return binding


def boottime_ns() -> int:
    """CLOCK_BOOTTIME: the camera SOF domain, used for cycle start and controller completion."""
    if sys.platform != "linux":
        raise RuntimeError("CLOCK_BOOTTIME requires the Linux device")
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def start_native_runtime(controller: NativeController, config_path: Path) -> NativeBinding:
    """Constructor-only bounded file reads; the worker owns native imports and I/O."""
    with config_path.open("rb") as stream:
        config = runtime_config(stream.read(4097))
    with Path(config["key_file"]).open("rb") as stream:
        key = stream.read(4097)
    boot_id = Path(BOOT_ID_FILE).read_text().strip()
    publisher_root = Path(__file__).resolve().parents[1] / "vision_long_publisher"
    for name, module in tuple(sys.modules.items()):
        if name == "vision_long_publisher" or name.startswith("vision_long_publisher."):
            filename = getattr(module, "__file__", None)
            if not isinstance(filename, str) or not Path(filename).resolve().is_relative_to(publisher_root):
                raise RuntimeError("native_publisher_tools_package_collision")
    spec = importlib.util.find_spec("vision_long_publisher")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != publisher_root / "__init__.py":
        raise RuntimeError("native_publisher_package_unavailable")
    from vision_long.device_applied import DRIVES_ROOT, STATUS_FILE
    from vision_long.observer_supervisor import IsolatedPublisher

    if not hasattr(controller, "vision_long_bridge"):
        raise ValueError("reviewed_native_bootstrap_hook_required")
    toggle = BrakingToggle(config["toggle_file"])
    toggle.start()
    try:
        provider = AuthorityProvider(key, ROAD_BUILD_ID, boot_id, boottime_ns, toggle.enabled)
        bridge = provider.bridge()
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(publisher_root.parent) + os.pathsep + child_env.get("PYTHONPATH", "")
        worker = IsolatedPublisher(
            bridge,
            command=[sys.executable, "-m", "vision_long_publisher.comma_observer"],
            boot_id=boot_id,
            build_id=ROAD_BUILD_ID,
            clock=boottime_ns,
            configuration={
                "listen_ip": config["listen_ip"],
                "port": config["port"],
                "key_hex": key.hex(),
                "drives_root": DRIVES_ROOT,
                "status_path": STATUS_FILE,
            },
            cwd=str(publisher_root.parent),
            env=child_env,
        )
    except BaseException:
        toggle.stop()  # Nothing is bound: no reader thread outlives a failed start.
        raise
    return _attach(controller, NativeBinding(bridge, worker, braking_enabled=toggle.enabled, cycle_clock=boottime_ns))
