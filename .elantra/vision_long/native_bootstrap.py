"""Explicit native startup; JSON can enable observation transport, never authority.

The optional in-process binding requires an independently verified gate/clock.
The companion is a braking-only cap, so no native movement evidence is consumed.
This package supplies none of those qualifications.
All network/camera work belongs to the publisher worker, never stage_cycle.
"""

import importlib.util
import ipaddress
import json
import logging
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Protocol

from .native_bridge import NativeReturnBridge
from .native_consumer import NativeScc12Consumer

# opendbc.car.carlog's logger name: card.py forwards it to cloudlog in the car process.
CAR_PROCESS_LOGGER = "carlog"


class PublisherHandle(Protocol):
    @property
    def is_alive(self) -> bool: ...
    @property
    def error(self) -> str | None: ...
    def stop(self) -> None: ...


def runtime_config(data: bytes) -> dict:
    """One bounded exact diagnostic schema; no qualification/configurable code."""
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
    keys = {"schema", "listen_ip", "port", "key_file", "pairs", "max_duration_s"}
    if type(value) is not dict or set(value) != keys or value["schema"] != "native_diagnostic_return_v1":
        raise ValueError("native_runtime_config_schema")
    if type(value["listen_ip"]) is not str:
        raise ValueError("native_runtime_address")
    if not isinstance(ipaddress.ip_address(value["listen_ip"]), ipaddress.IPv4Address):
        raise ValueError("native_runtime_ipv4_required")
    if type(value["port"]) is not int or not 1024 <= value["port"] <= 65535:
        raise ValueError("native_runtime_port")
    if type(value["pairs"]) is not int or not 1 <= value["pairs"] <= 100000:
        raise ValueError("native_runtime_pairs")
    duration = value["max_duration_s"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= 6000:
        raise ValueError("native_runtime_duration")
    key = value["key_file"]
    if type(key) is not str or not 1 <= len(key) <= 1024 or not key.isascii() or not key.isprintable():
        raise ValueError("native_runtime_key_path")
    path = PurePosixPath(key)
    if not path.is_absolute() or path.parts[:2] != ("/", "data") or ".." in path.parts:
        raise ValueError("native_runtime_key_must_be_provisioned_under_data")
    return value


class NativeBinding:
    def __init__(self, bridge: NativeReturnBridge, worker: PublisherHandle | None = None, *, diagnostic: bool):
        """``diagnostic``: the bridge's gate factory refuses every session (no qualification)."""
        if not isinstance(bridge, NativeReturnBridge) or type(diagnostic) is not bool:
            raise ValueError("typed_native_bridge_required")
        self.bridge = bridge
        self.worker = worker
        self.stopped = False
        self.last_status = "diagnostic_transport_authority_unavailable" if diagnostic else "awaiting_native_cycle"

    def stage_cycle(self, now_ns: int) -> NativeScc12Consumer | None:
        """Bounded mailbox only; nothing here blocks the controller thread."""
        if self.stopped or (self.worker is not None and (self.worker.error is not None or not self.worker.is_alive)):
            self.last_status = "native_publisher_stopped_or_failed"
            return None
        try:
            consumer = self.bridge.stage_cycle(now_ns)
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

    def stop(self) -> None:
        self.stopped = True
        if self.worker is not None:
            self.worker.stop()  # Event only; no join on the controller thread.


class NativeController(Protocol):
    vision_long_consumer: NativeScc12Consumer | None
    vision_long_bridge: NativeBinding | None


def bind_native_controller(controller: NativeController, bridge: NativeReturnBridge) -> NativeBinding:
    """Explicit trusted in-process opt-in; no qualification is inferred here."""
    if not hasattr(controller, "vision_long_bridge"):
        raise ValueError("reviewed_native_bootstrap_hook_required")
    binding = NativeBinding(bridge, diagnostic=False)
    previous = controller.vision_long_bridge
    if previous is not None:
        previous.stop()
    controller.vision_long_consumer = None
    controller.vision_long_bridge = binding
    return binding


def start_diagnostic_runtime(config_path: Path) -> NativeBinding:
    """Constructor-only bounded file reads; worker owns native imports and I/O."""
    with config_path.open("rb") as stream:
        config = runtime_config(stream.read(4097))
    with Path(config["key_file"]).open("rb") as stream:
        key = stream.read(4097)
    if not 32 <= len(key) <= 4096:
        raise ValueError("bounded_provisioned_transport_key_required")

    def unavailable_gate(session_id: str):
        raise RuntimeError("live_physical_and_clock_qualification_unavailable")

    bridge = NativeReturnBridge(
        unavailable_gate,
        completion_clock=None,
        trusted_clock_id=None,
        fault_log=logging.getLogger(CAR_PROCESS_LOGGER),
    )
    publisher_root = Path(__file__).resolve().parents[1] / "vision_long_publisher"
    for name, module in tuple(sys.modules.items()):
        if name == "vision_long_publisher" or name.startswith("vision_long_publisher."):
            filename = getattr(module, "__file__", None)
            if not isinstance(filename, str) or not Path(filename).resolve().is_relative_to(publisher_root):
                raise RuntimeError("native_publisher_tools_package_collision")
    spec = importlib.util.find_spec("vision_long_publisher")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != publisher_root / "__init__.py":
        raise RuntimeError("native_publisher_package_unavailable")
    from vision_long_publisher.comma_context_publisher import (  # type: ignore[import-not-found]  # Device-only package.
        start_embedded_publisher,
    )

    worker = start_embedded_publisher(
        bridge,
        listen_ip=config["listen_ip"],
        port=config["port"],
        key=key,
        pairs=config["pairs"],
        max_duration_s=config["max_duration_s"],
    )
    return NativeBinding(bridge, worker, diagnostic=True)
