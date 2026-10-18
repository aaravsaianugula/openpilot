#!/usr/bin/env python3
"""
openpilot state for the Vision Long laptop monitor, read and never written.

Onroad comes from the IsOffroad param file (manager writes it from deviceState.started), so it
costs no msgq reader. Engaged and the driver causes need the 100 Hz streams, so selfdriveState
and carState are drained, never conflated: a cancel press lives in carState.buttonEvents for one
frame only, and the time engagement dropped has to be exact to the frame to be compared with it.

msgq caps a queue at 25 reader slots, a dead reader's slot is never handed back, and when the
slots run out msgq resets *every* reader of that queue, controlsd's included. So this process:
  - opens its two sockets at most once per process, and only once onroad and up for 30 s, so a
    crash loop at start never touches msgq;
  - keeps a per-boot budget of opens in /data/vision_long (keyed by the kernel boot_id). Once it
    is spent, engaged and the driver causes are reported unknown until the next boot.

Field names are this fork's: selfdriveState{enabled, alertType, alertText1} (log.capnp) and
carState{brakePressed, gasPressed, buttonEvents[{type}]} (opendbc car.capnp).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

WINDOW_NS = 500_000_000      # a driver cause within 0.5 s before the drop explains it
SETTLE_NS = 100_000_000      # carState published just after the drop still counts
DECIDE_AFTER_NS = 1_500_000_000
GAP_NS = 150_000_000         # 100 Hz stream: a gap over 150 ms means frames were missed
RESET_GAP_NS = 1_000_000_000  # no selfdriveState for over 1 s: engagement is unknown, never a drop
CAUSE_BUTTONS = frozenset({"cancel", "mainCruise"})
OPEN_BUDGET = 4
OPEN_AFTER_UPTIME_S = 30.0


@dataclass(frozen=True)
class SelfdriveSample:
    mono_ns: int
    enabled: bool
    alert_type: str
    alert_text: str


@dataclass(frozen=True)
class CarSample:
    mono_ns: int
    cause: str  # "" when this frame carries no driver cause


@dataclass(frozen=True)
class Disengage:
    mono_ns: int
    alert: str
    verified: bool  # carState covered the whole window, so "no driver cause" is a fact


def selfdrive_sample(event) -> SelfdriveSample:
    state = event.selfdriveState
    return SelfdriveSample(int(event.logMonoTime), bool(state.enabled), str(state.alertType), str(state.alertText1))


def car_sample(event) -> CarSample:
    state = event.carState
    causes = []
    if state.brakePressed:
        causes.append("brake")
    if state.gasPressed:
        causes.append("gas")
    causes.extend(str(button.type) for button in state.buttonEvents if str(button.type) in CAUSE_BUTTONS)
    return CarSample(int(event.logMonoTime), ",".join(causes))


def read_onroad(params_dir: Path) -> bool | None:
    """IsOffroad: b"1" offroad, b"0" onroad. Absent (manager not started) or anything else is unknown."""
    try:
        value = (params_dir / "IsOffroad").read_bytes()
    except FileNotFoundError:
        return None
    return {b"1": False, b"0": True}.get(value.strip())


class DisengageDetector:
    """Finds engaged true -> false with no brake, gas, cancel or main within 0.5 s before it."""

    def __init__(self) -> None:
        self._enabled: bool | None = None
        self._last_sd_ns: int | None = None
        self._pending: list[tuple[int, bool]] = []  # (drop time, selfdriveState was continuous)
        self._alerts: deque[tuple[int, str]] = deque(maxlen=400)
        self._car: deque[CarSample] = deque(maxlen=400)

    @property
    def engaged(self) -> bool | None:
        return self._enabled

    def reset(self) -> None:
        """The streams were lost: forget the engagement state rather than infer a drop across the gap."""
        self._enabled, self._last_sd_ns = None, None
        self._pending.clear()

    def feed(self, selfdrive: list[SelfdriveSample], car: list[CarSample]) -> None:
        self._car.extend(car)
        for sample in selfdrive:
            if self._last_sd_ns is not None and sample.mono_ns - self._last_sd_ns > RESET_GAP_NS:
                self._enabled = None
            if sample.alert_type:
                self._alerts.append((sample.mono_ns, f"{sample.alert_type} {sample.alert_text}".strip()))
            continuous = self._last_sd_ns is not None and sample.mono_ns - self._last_sd_ns <= GAP_NS
            if self._enabled and not sample.enabled:
                self._pending.append((sample.mono_ns, continuous))
            self._enabled, self._last_sd_ns = sample.enabled, sample.mono_ns

    def decide(self, now_ns: int, onroad: bool | None) -> list[Disengage]:
        latest_car = self._car[-1].mono_ns if self._car else None
        found: list[Disengage] = []
        waiting: list[tuple[int, bool]] = []
        for drop, continuous in self._pending:
            covered = latest_car is not None and latest_car >= drop + SETTLE_NS
            if not covered and now_ns < drop + DECIDE_AFTER_NS:
                waiting.append((drop, continuous))
                continue
            if onroad is not True:
                continue  # the car went offroad; a drop at ignition-off is not unexpected
            window = [s for s in self._car if drop - WINDOW_NS <= s.mono_ns <= drop + SETTLE_NS]
            if any(s.cause for s in window):
                continue
            found.append(Disengage(drop, self._alert_near(drop), continuous and self._covers(window, drop)))
        self._pending = waiting
        return found

    def _covers(self, window: list[CarSample], drop: int) -> bool:
        times = [s.mono_ns for s in window if s.mono_ns <= drop]
        if not times or times[0] > drop - WINDOW_NS + GAP_NS or drop - times[-1] > GAP_NS:
            return False
        return all(b - a <= GAP_NS for a, b in zip(times, times[1:], strict=False))

    def _alert_near(self, drop: int) -> str:
        after = [text for t, text in self._alerts if drop <= t <= drop + WINDOW_NS]
        before = [text for t, text in self._alerts if drop - WINDOW_NS <= t < drop]
        return after[0] if after else (before[-1] if before else "")


def kernel_boot_id(proc: Path) -> str:
    return (proc / "sys/kernel/random/boot_id").read_text().strip()


def claim_open(budget_file: Path, boot_id: str) -> bool:
    """Take one msgq open from this boot's budget. False once OPEN_BUDGET opens were taken."""
    try:
        record = json.loads(budget_file.read_text())
    except (FileNotFoundError, ValueError):
        record = {}
    opens = record.get("opens") if type(record) is dict and record.get("boot_id") == boot_id else 0
    if type(opens) is not int or opens >= OPEN_BUDGET:
        return False
    temporary = budget_file.with_name("." + budget_file.name + ".tmp")
    temporary.write_text(json.dumps({"boot_id": boot_id, "opens": opens + 1}))
    os.replace(temporary, budget_file)
    return True


def preload_messaging() -> None:
    """Import cereal messaging (pycapnp, the schema, msgq) once, at monitor start."""
    from openpilot.cereal import messaging  # type: ignore[import-not-found]  # noqa: F401


class MsgqStreams:
    """The two drained subscriptions. Device-only: importing cereal.messaging needs built msgq."""

    def __init__(self) -> None:
        from openpilot.cereal import messaging  # type: ignore[import-not-found]
        self._messaging = messaging
        self._selfdrive = messaging.sub_sock("selfdriveState", conflate=False)
        self._car = messaging.sub_sock("carState", conflate=False)

    def drain(self) -> tuple[list[SelfdriveSample], list[CarSample]]:
        drain = self._messaging.drain_sock
        return ([selfdrive_sample(e) for e in drain(self._selfdrive)], [car_sample(e) for e in drain(self._car)])
