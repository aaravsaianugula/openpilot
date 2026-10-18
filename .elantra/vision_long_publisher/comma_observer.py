"""Private, source-verified camera observer subprocess; stdin/stdout are JSON IPC only."""

import argparse
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from vision_long.device_applied import DriveLog, StatusFile, drive_directory_name
from vision_long.native_bootstrap import ROAD_BUILD_ID
from vision_long.observer_ipc import RemoteBridge, fields
from vision_long.transport import CameraTransportServer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incarnation", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()
    # Native libraries may print directly to fd 1. Reserve the inherited pipe
    # before importing any camera code, and route all ordinary stdout to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as response:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        return run_observer(args, response)


def run_observer(args: argparse.Namespace, response: BinaryIO) -> int:
    from .comma_context_publisher import (
        AppliedForwarder,
        DeviceSource,
        SessionCounters,
        boottime_ns,
        require_pinned_device,
        serve_sessions,
    )

    require_pinned_device()
    source = DeviceSource()
    if source.boot != args.boot_id or ROAD_BUILD_ID != args.build_id:
        raise RuntimeError("observer_source_boot_or_build_mismatch")
    bridge = RemoteBridge(sys.stdin.buffer, response, args.incarnation, boottime_ns)
    config = bridge.hello(source.boot, ROAD_BUILD_ID)
    fields(config, {"listen_ip", "port", "key_hex", "drives_root", "status_path", "restarts"})
    key = bytes.fromhex(config["key_hex"])
    # A process restart begins a new log folder. Names label UTC only; retention
    # still uses last write. Parent completion counters and latch remain intact.
    drive_log = DriveLog(Path(config["drives_root"]), drive_directory_name(datetime.now(UTC).timestamp(), source.boot))
    forwarder = AppliedForwarder(
        bridge,
        drive_log,
        status_file=StatusFile(config["status_path"]),
        restarts=lambda: config["restarts"],
        boot_clock=boottime_ns,
    )

    def returned(session_id: str, packet: bytes) -> None:
        bridge.return_packet(session_id, packet)

    with CameraTransportServer(
        config["listen_ip"],
        config["port"],
        key,
        clock_id=source.boot + ":CLOCK_BOOTTIME",
        clock=boottime_ns,
        with_context=True,
        with_return=True,
        on_proposal=returned,
        timeout_s=1,
    ) as server:
        serve_sessions(server, source, boottime_ns, bridge, forwarder, threading.Event(), SessionCounters())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
