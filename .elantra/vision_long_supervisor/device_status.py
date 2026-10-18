#!/usr/bin/env python3
"""
road-ready's /data/vision_long/status.json, read defensively, and the notices it gives.

The writer is the companion's publisher worker (vl-road 69227b0, tools/native_overlay/README.md):
atomic, about 1 Hz, and only while the worker runs, so a stale file means the worker is not
running (it is waiting out a restart backoff, or it is gone). Its fields, by their exact names:
boot_ns (CLOCK_BOOTTIME ns), mode, authority, reason, toggle (on|off), laptop_session (up|down),
last_packet_age_ms, publisher_restarts, drive_log_dir, drive_log_error, completion_overruns,
worst_completion_us. Nothing on the device reads this file for authority; neither does this.

Notice keys:
  companion_latched       authority latched_native (warn, with the reason)
  drive_log_error         the companion's drive log was disabled (warn, with the cause)
  publisher_storm         publisher_restarts rose 3 times within 5 min (warn)
  companion_session_down  laptop session down for 10 s while onroad with the braking toggle on (critical)
  publisher_down          the file exists but is stale for 10 s while onroad (warn)
Unknown (missing, garbage, stale) is never clearance: while the file is not fresh, the conditions
it last gave are held, so they never "recover" because the writer went quiet.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from notices import clip

FIELDS = ("boot_ns", "mode", "authority", "reason", "toggle", "laptop_session", "last_packet_age_ms",
          "publisher_restarts", "drive_log_dir", "drive_log_error", "completion_overruns", "worst_completion_us")
STALE_NS = 5_000_000_000
STORM_RESTARTS = 3
STORM_WINDOW_NS = 5 * 60 * 1_000_000_000
SESSION_DOWN_HOLD_NS = 10_000_000_000
PUBLISHER_DOWN_HOLD_NS = 10_000_000_000
KEYS = ("companion_latched", "drive_log_error", "publisher_storm", "companion_session_down")


@dataclass(frozen=True)
class DeviceStatus:
    kind: str                       # missing | garbage | stale | fresh
    session_up: bool | None = None
    toggle_on: bool | None = None
    authority: str | None = None
    reason: str | None = None
    drive_log_error: str | None = None
    publisher_restarts: int | None = None
    problem: str | None = None      # why it is not fully usable, for the log (None when fine or absent)


UNKNOWN = DeviceStatus("missing")


def boottime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _text(value) -> str | None:
    return value if type(value) is str and value else None


def read_status_from(raw, now_boot_ns: int) -> DeviceStatus:
    if type(raw) is not dict:
        return DeviceStatus("garbage", problem="not a JSON object")
    stamp = raw.get("boot_ns")
    if type(stamp) is not int:
        return DeviceStatus("garbage", problem="boot_ns missing or not an integer")
    if not 0 <= now_boot_ns - stamp <= STALE_NS:
        return DeviceStatus("stale", problem="stale: the publisher worker is not writing it")
    missing = sorted(name for name in FIELDS if name not in raw)
    restarts = raw.get("publisher_restarts")
    return DeviceStatus(
        "fresh",
        session_up={"up": True, "down": False}.get(raw.get("laptop_session")),
        toggle_on={"on": True, "off": False}.get(raw.get("toggle")),
        authority=_text(raw.get("authority")),
        reason=_text(raw.get("reason")),
        drive_log_error=_text(raw.get("drive_log_error")),
        publisher_restarts=restarts if type(restarts) is int else None,
        problem="missing keys: " + ", ".join(missing) if missing else None,
    )


def read_status(path: Path, now_boot_ns: int) -> DeviceStatus:
    """Missing is unknown and quiet (not installed yet); unreadable or garbage says why."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return UNKNOWN
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, ValueError):
        return DeviceStatus("garbage", problem="not JSON")
    return read_status_from(raw, now_boot_ns)


class StatusConditions:
    def __init__(self) -> None:
        self._held: dict[str, tuple[str, str]] = {}
        self._restarts: deque[int] = deque()
        self._last_restarts: int | None = None
        self._since: dict[str, int] = {}
        self._down = False

    def _for(self, key: str, on: bool, now: int, hold_ns: int) -> bool:
        if not on:
            self._since.pop(key, None)
            return False
        return now - self._since.setdefault(key, now) >= hold_ns

    def assess(self, status: DeviceStatus, now: int, onroad: bool | None) -> dict[str, tuple[str, str]]:
        present: dict[str, tuple[str, str]] = {}
        if status.kind == "fresh" or onroad is False:
            self._down = self._for("publisher_down", False, now, PUBLISHER_DOWN_HOLD_NS)
        elif status.kind == "stale" and onroad is True:
            self._down = self._for("publisher_down", True, now, PUBLISHER_DOWN_HOLD_NS)
        # Otherwise (missing, garbage, unreadable, or onroad unknown) what was known is held.
        if self._down:
            present["publisher_down"] = ("warn", "companion publisher not running (status.json stale)")
        if status.kind != "fresh":
            present.update(self._held)
            return present
        self._held = self._fresh(status, now, onroad)
        present.update(self._held)
        return present

    def _fresh(self, status: DeviceStatus, now: int, onroad: bool | None) -> dict[str, tuple[str, str]]:
        found: dict[str, tuple[str, str]] = {}
        if status.authority == "latched_native":
            found["companion_latched"] = ("warn", clip(f"companion latched to native: {status.reason or 'no reason given'}"))
        if status.drive_log_error is not None:
            found["drive_log_error"] = ("warn", clip(f"companion drive log off: {status.drive_log_error}"))
        restarts = status.publisher_restarts
        if restarts is not None:
            if self._last_restarts is not None and restarts > self._last_restarts:
                self._restarts.extend([now] * min(restarts - self._last_restarts, STORM_RESTARTS))
            self._last_restarts = restarts
        while self._restarts and now - self._restarts[0] > STORM_WINDOW_NS:
            self._restarts.popleft()
        if len(self._restarts) >= STORM_RESTARTS:
            found["publisher_storm"] = ("warn", f"companion publisher: {len(self._restarts)} restarts in 5 min")
        down = onroad is True and status.toggle_on is True and status.session_up is False
        if self._for("companion_session_down", down, now, SESSION_DOWN_HOLD_NS):
            found["companion_session_down"] = ("critical", "laptop session down while braking toggle is on")
        return found
