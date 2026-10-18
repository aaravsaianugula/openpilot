"""What the device actually applied, per companion cycle, and its bounded on-device drive log.

The controller thread hands the bridge one plain tuple per companion packet, at the first SCC12
handoff that arbitrated it (``from_handoff`` documents the layout). Everything that can raise,
validation and encoding included, runs on the publisher worker, never on the controller thread.
The worker sends each record to the laptop of the same session (transport kind 8) and feeds the
drive log, which writes on change plus a 1 Hz summary, never past its caps, and never while
``/data`` has less than ``MIN_FREE_BYTES`` free.
"""

import errno
import io
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SCHEMA = "device_applied_v1"
MAX_RECORD_BYTES = 2048
DRIVES_ROOT = "/data/vision_long/drives"
DRIVE_LOG_NAME = "applied.jsonl"
DRIVE_BYTES_CAP = 20 * 1024 * 1024
TOTAL_BYTES_CAP = 200 * 1024 * 1024
# /data runs about 90% full: below this much free space the log writes nothing.
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
SUMMARY_PERIOD_NS = 1_000_000_000
# The free-space query is a syscall; it is asked at most this often.
FREE_SPACE_CHECK_NS = 1_000_000_000
DRIVE_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")
BOOT_PREFIX = re.compile(r"[0-9a-f]{8}")
_REASON = re.compile(r"[\x21-\x7e]{1,160}")
_REASON_UNSAFE = re.compile(r"[^\x21-\x7e]")
MAX_REASON_CHARS = 160


def reason_token(text: str) -> str:
    """One printable token, like the status line's reason: a fault's free text keeps its words."""
    return _REASON_UNSAFE.sub("_", text)[:MAX_REASON_CHARS] or "unknown"


Mode = Literal["observe", "braking_only"]
Authority = Literal["disabled", "enabled", "latched_native"]
BRAKING_TOGGLE_OFF = "braking_toggle_off"


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):  # type: ignore[arg-type]
        raise ValueError("device_applied_" + name)
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DeviceApplied:
    pair_id: int | None
    boot_ns: int
    native_areq: float
    companion_cap: float | None
    final_areq: float
    # What final would have been with authority: equals final while the toggle is ON.
    shadow_final_areq: float
    stop_req: int
    mode: Mode
    authority: Authority
    reason: str
    override: bool
    aeb_active: bool
    toggle: bool

    def __post_init__(self) -> None:
        if self.pair_id is not None and (type(self.pair_id) is not int or not 1 <= self.pair_id < 2**63):
            raise ValueError("device_applied_pair_id")
        if type(self.boot_ns) is not int or not 0 <= self.boot_ns < 2**63:
            raise ValueError("device_applied_boot_ns")
        native = _finite(self.native_areq, "native_areq")
        final = _finite(self.final_areq, "final_areq")
        shadow = _finite(self.shadow_final_areq, "shadow_final_areq")
        if self.companion_cap is not None:
            _finite(self.companion_cap, "companion_cap")
        if type(self.stop_req) is not int or self.stop_req not in (0, 1):
            raise ValueError("device_applied_stop_req")
        if any(type(value) is not bool for value in (self.override, self.aeb_active, self.toggle)):
            raise ValueError("device_applied_flags")
        if type(self.reason) is not str or not _REASON.fullmatch(self.reason):
            raise ValueError("device_applied_reason")
        if (self.mode, self.authority) not in (
            ("observe", "disabled"),
            ("braking_only", "enabled"),
            ("braking_only", "latched_native"),
        ) or (self.mode == "observe") == self.toggle:
            raise ValueError("device_applied_mode_authority")
        if final > native:
            raise ValueError("device_applied_final_above_native")
        if shadow > native:
            raise ValueError("device_applied_shadow_above_native")
        if not self.toggle and final != native:
            raise ValueError("device_applied_toggle_off_not_native")
        if (self.authority == "latched_native" or self.override) and final != native:
            raise ValueError("device_applied_native_owner_not_native")

    def to_bytes(self) -> bytes:
        body = {"schema": SCHEMA, **asdict(self)}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

    @classmethod
    def from_bytes(cls, data: bytes) -> "DeviceApplied":
        if type(data) is not bytes or not 0 < len(data) <= MAX_RECORD_BYTES:
            raise ValueError("device_applied_size")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("device_applied_duplicate_key")
                result[key] = value
            return result

        value = json.loads(data, object_pairs_hook=unique)
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields | {"schema"} or value.pop("schema") != SCHEMA:
            raise ValueError("device_applied_schema")
        return cls(**value)

    def change_key(self) -> tuple:
        """What a reader of the drive log needs to see the moment it changes."""
        return (
            self.mode,
            self.authority,
            self.toggle,
            self.override,
            self.aeb_active,
            self.stop_req,
            self.final_areq < self.native_areq,
        )


# (pair_id, boot_ns, native_areq, companion_cap, final_areq, shadow_final_areq, stop_req,
#  toggle, latched_fault, decision_reason, override, aeb_active): built on the controller thread.
HandoffTuple = tuple[int | None, int, float, float | None, float, float, int, bool, str | None, str, bool, bool]


def from_handoff(item: HandoffTuple) -> DeviceApplied:
    """Validate a controller-thread tuple on the worker; mode and authority follow the toggle and latch."""
    pair_id, boot_ns, native, cap, final, shadow, stop, toggle, latched, reason, override, aeb = item
    if not toggle:
        mode: Mode = "observe"
        authority: Authority = "disabled"
        reason = BRAKING_TOGGLE_OFF
    elif latched is not None:
        mode, authority, reason = "braking_only", "latched_native", latched
    else:
        mode, authority = "braking_only", "enabled"
    return DeviceApplied(
        pair_id,
        boot_ns,
        native,
        cap,
        final,
        shadow,
        int(stop),
        mode,
        authority,
        reason_token(reason),
        override,
        aeb,
        toggle,
    )


def drive_directory_name(utc_seconds: float, boot_id: str) -> str:
    """``<UTC yyyymmddThhmmssZ>-<boot8>``, the same shape as the laptop's drive folders."""
    prefix = boot_id[:8]
    if not BOOT_PREFIX.fullmatch(prefix):
        raise ValueError("drive_boot_id")
    stamp = datetime.fromtimestamp(int(utc_seconds), UTC).strftime("%Y%m%dT%H%M%SZ")
    return stamp + "-" + prefix


def _folder_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


class DriveLog:
    """One drive's ``applied.jsonl``, written by the publisher worker only."""

    def __init__(
        self,
        root: Path,
        name: str,
        *,
        free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
        clock: Callable[[], int] = time.monotonic_ns,
        drive_cap: int = DRIVE_BYTES_CAP,
        total_cap: int = TOTAL_BYTES_CAP,
    ):
        if not DRIVE_NAME.fullmatch(name) or not 0 < drive_cap <= total_cap:
            raise ValueError("drive_log_configuration")
        self.root = root
        self.folder = root / name
        self.path = self.folder / DRIVE_LOG_NAME
        self._free_bytes = free_bytes
        self._clock = clock
        self.drive_cap = drive_cap
        self.written = 0
        self.dropped = 0
        self.suppressed_low_space = 0
        # The first /data error (EIO, ENOSPC, ...) disables the log for this drive; never control.
        self.write_errors = 0
        self.error: str | None = None
        self._free_checked: tuple[int, bool] | None = None
        self._stream: io.BufferedWriter | None = None
        self._last_key: tuple | None = None
        self._last: DeviceApplied | None = None
        self._pending = 0
        self._reasons: Counter[str] = Counter()
        self._summary_ns = clock()
        try:
            self._prune(total_cap - drive_cap)
        except OSError as error:
            self._fail(error)

    def _prune(self, budget: int) -> None:
        """Delete whole drive folders, oldest first, until the others fit beside this drive's cap."""
        if not self.root.is_dir():
            return
        drives = [
            path
            for path in self.root.iterdir()
            if path.is_dir() and DRIVE_NAME.fullmatch(path.name) and path != self.folder
        ]
        # By last write only: the name's UTC stamp comes from the device wall clock, which may be unsynced
        # at boot, so it is a label and never an order.
        drives.sort(key=lambda path: path.stat().st_mtime)
        sizes = {path: _folder_bytes(path) for path in drives}
        total = sum(sizes.values())
        for path in drives:
            if total <= budget:
                break
            shutil.rmtree(path)
            total -= sizes[path]

    def _space(self) -> bool:
        now = self._clock()
        cached = self._free_checked
        if cached is None or now - cached[0] >= FREE_SPACE_CHECK_NS:
            probe = self.root if self.root.is_dir() else self.root.parent
            cached = (now, self._free_bytes(probe) >= MIN_FREE_BYTES)
            self._free_checked = cached
        return cached[1]

    def _fail(self, error: OSError) -> None:
        """Count the error, name its cause and stop writing for this drive; the worker carries on."""
        self.write_errors += 1
        self.error = type(error).__name__ + ":" + errno.errorcode.get(error.errno or 0, "unknown")
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                self.write_errors += 1  # Closing a failed stream can fail again; counted, already disabled.

    def _write(self, line: dict) -> None:
        if self.error is not None:
            self.dropped += 1
            return
        if not self._space():
            self.suppressed_low_space += 1
            return
        data = (json.dumps(line, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if self.written + len(data) > self.drive_cap:
            self.dropped += 1
            return
        try:
            if self._stream is None:
                self.folder.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("ab")
            self._stream.write(data)
        except OSError as error:
            self.dropped += 1
            self._fail(error)
            return
        self.written += len(data)

    def status(self, line: str) -> None:
        self._write({"kind": "status", "t_ns": self._clock(), "line": line})

    def record(self, applied: DeviceApplied) -> None:
        key = applied.change_key()
        if key != self._last_key:
            self._last_key = key
            self._write({"kind": "change", "t_ns": self._clock(), "record": asdict(applied)})
        else:
            self._pending += 1
            self._reasons[applied.reason] += 1
        self._last = applied

    def tick(self) -> None:
        """1 Hz summary of the records not written as changes; flushes the file."""
        now = self._clock()
        if now - self._summary_ns < SUMMARY_PERIOD_NS:
            return
        self._summary_ns = now
        if self._pending and self._last is not None:
            self._write(
                {
                    "kind": "summary",
                    "t_ns": now,
                    "records": self._pending,
                    "reasons": dict(self._reasons),
                    "last": asdict(self._last),
                    "dropped": self.dropped,
                }
            )
        self._pending = 0
        self._reasons.clear()
        if self._stream is not None:
            try:
                self._stream.flush()
            except OSError as error:
                self._fail(error)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError as error:
                self._fail(error)
            self._stream = None


STATUS_FILE = "/data/vision_long/status.json"
# The notification monitor's view is refreshed about once a second, never per cycle.
STATUS_PERIOD_NS = 1_000_000_000
# Consecutive failed writes before the status reason says so (a single miss is only counted).
STATUS_FAILURES_PERSIST = 5


class StatusFile:
    """Atomic (tmp, fsync, rename) status for a notification monitor; worker thread only.

    Nothing reads it back: authority never depends on it. A failed write is counted and
    reported as persistent after ``STATUS_FAILURES_PERSIST`` consecutive failures.
    """

    def __init__(self, path: str = STATUS_FILE, *, clock: Callable[[], int] = time.monotonic_ns):
        self.path = Path(path)
        self._clock = clock
        self._last_ns: int | None = None
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error: str | None = None

    @property
    def persistently_failing(self) -> bool:
        return self.consecutive_failures >= STATUS_FAILURES_PERSIST

    def offer(self, fields: dict) -> bool:
        """Write when a period has passed since the last attempt; True when written."""
        now = self._clock()
        if self._last_ns is not None and now - self._last_ns < STATUS_PERIOD_NS:
            return False
        self._last_ns = now
        data = (json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(temporary, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            # Counted and surfaced when it persists; control never waits on or reads this file.
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = type(error).__name__
            return False
        self.consecutive_failures = 0
        return True
