#!/usr/bin/env python3
"""
What the comma can see of the laptop's health, turned into notice conditions.

Inputs, each of which may be unknown (None): the laptop's heartbeat (through wire.Receiver),
the NCM carrier, onroad, and road-ready's status (device_status.py). Unknown is never ok:
`laptop_view.ok` is true only for a fresh frame that says the supervisor is ok, the system is
running and the cycle is inside its budget.

Notice keys raised here (level conditions; notices.py owns the raise/fold/recover rules):
  laptop_lost        no laptop frame for > 3 s, while onroad, while road-ready reports a laptop
                     session up, or when the laptop's last word was that it was running
  laptop_unhealthy   supervisor_ok not true or state fault while running, the laptop not
                     running (starting included) for 30 s while onroad, or starting for 60 s
  laptop_cycle_slow  cycle p99 over its budget while running
  display_down       the display child is reported and not alive while running
  laptop.<key>       each active notice the laptop itself carries, mirrored
A condition other than laptop_lost must hold for HOLD_NS before it counts, so a start-up blip
or one late frame does not raise anything. While the laptop is lost or was never heard, the
conditions from its last frame are held as they were: silence never reads as recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from device_status import DeviceStatus
from notices import MIRROR_PREFIX, SEVERITIES, clip
from wire import Receiver

HOLD_NS = 5_000_000_000
NOT_RUNNING_ONROAD_NS = 30_000_000_000
STUCK_STARTING_NS = 60_000_000_000
RUNNING = frozenset({"running", "degraded"})
ACTIVE = frozenset({"starting", "running", "degraded", "fault"})
STATES = frozenset({"idle", "waiting", "starting", "running", "degraded", "stopping", "sleeping_soon", "fault"})
MIRROR_MAX = 8

def read_carrier(net: Path, ifname: str | None) -> bool | None:
    """/sys/class/net/<if>/carrier. No interface yet, or unreadable, is unknown."""
    if not ifname:
        return None
    try:
        return (net / ifname / "carrier").read_text().strip() == "1"
    except OSError:
        return None


def _text(value) -> str | None:
    return value if type(value) is str and value else None


def _number(value) -> float | None:
    return float(value) if type(value) in (int, float) else None


@dataclass(frozen=True)
class Assessment:
    present: dict[str, tuple[str, str]]
    view: dict


class Held:
    """A condition counts only after it has held continuously for its hold time."""

    def __init__(self) -> None:
        self._since: dict[str, int] = {}

    def check(self, key: str, on: bool, now: int, hold_ns: int) -> bool:
        if not on:
            self._since.pop(key, None)
            return False
        since = self._since.setdefault(key, now)
        return now - since >= hold_ns


class HealthEvaluator:
    def __init__(self, started_ns: int) -> None:
        self._started_ns = started_ns
        self._held = Held()
        # What the last frame said, held while the laptop is silent: losing the laptop is never
        # its conditions clearing, so nothing it reported can "recover" until it speaks again.
        self._derived: dict[str, tuple[str, str]] = {}
        self._lost_present = False

    def assess(self, receiver: Receiver, now: int, onroad: bool | None, carrier: bool | None,
               status: DeviceStatus) -> Assessment:
        present: dict[str, tuple[str, str]] = {}
        freshness = receiver.freshness(now)
        age = receiver.age_s(now)
        body = receiver.last.body if receiver.last is not None else None
        # The state in the last accepted frame is the laptop's last word, however old it is.
        state = body.get("state") if body is not None and body.get("state") in STATES else None

        lost = freshness == "lost" or (freshness == "never" and now - self._started_ns > 3_000_000_000)
        # With onroad unknown (IsOffroad unreadable), a lost laptop that was raised stays raised.
        gate = onroad is True or status.session_up is True or state in ACTIVE or (onroad is None and self._lost_present)
        if lost and gate:
            present["laptop_lost"] = ("critical", self._lost_text(age, carrier, status))
        self._lost_present = "laptop_lost" in present

        if body is not None and freshness in ("fresh", "stale"):
            derived: dict[str, tuple[str, str]] = {}
            self._assess_body(body, state, now, onroad, derived)
            self._derived = derived
        present.update(self._derived)
        return Assessment(present, self._view(freshness, age, body, state))

    def _lost_text(self, age: float | None, carrier: bool | None, status: DeviceStatus) -> str:
        silent = "laptop silent" if age is None else f"laptop silent {age:.0f} s"
        if carrier is False:
            cause = "USB cable unplugged or laptop off"
        elif carrier is None:
            cause = "USB link interface missing"
        else:
            cause = "link up, no heartbeat"
        if status.session_up is True:
            cause += "; companion session still up"
        return clip(f"{silent}: {cause}")

    def _assess_body(self, body: dict, state: str | None, now: int, onroad: bool | None,
                     present: dict[str, tuple[str, str]]) -> None:
        reason = _text(body.get("reason")) or "no reason given"
        unhealthy = state in ACTIVE - {"starting"} and (body.get("supervisor_ok") is not True or state == "fault")
        if self._held.check("laptop_unhealthy", unhealthy, now, HOLD_NS):
            present["laptop_unhealthy"] = ("critical" if state == "fault" else "warn", f"laptop {state}: {reason}")
        if self._held.check("stuck_starting", state == "starting", now, STUCK_STARTING_NS):
            present.setdefault("laptop_unhealthy", ("warn", "laptop stuck starting"))
        # `starting` counts as not running: a laptop that never gets past it is not running.
        not_running = onroad is True and state not in RUNNING
        if self._held.check("not_running", not_running, now, NOT_RUNNING_ONROAD_NS):
            present.setdefault("laptop_unhealthy", ("warn", f"laptop {state or 'state unknown'} while onroad"))

        cycle = body.get("cycle") if type(body.get("cycle")) is dict else {}
        p99, budget = _number(cycle.get("p99_ms")), _number(cycle.get("budget_ms"))
        slow = state in RUNNING and p99 is not None and budget is not None and p99 > budget
        if self._held.check("laptop_cycle_slow", slow, now, HOLD_NS):
            present["laptop_cycle_slow"] = ("warn", f"laptop cycle p99 {p99:.0f} ms over {budget:.0f} ms budget")

        children = body.get("children") if type(body.get("children")) is dict else {}
        display = children.get("display")
        down = state in RUNNING and type(display) is dict and display.get("alive") is not True
        if self._held.check("display_down", down, now, HOLD_NS):
            present["display_down"] = ("warn", "laptop display is not running")

        mirrored = body.get("notices") if type(body.get("notices")) is list else []
        for item in mirrored[:MIRROR_MAX]:
            if type(item) is not dict or item.get("state") != "active":
                continue
            key, severity, text = item.get("key"), item.get("severity"), item.get("text")
            if type(key) is str and key and severity in SEVERITIES and type(text) is str:
                present[MIRROR_PREFIX + key[:40]] = (severity, "laptop: " + text)

    def _view(self, freshness: str, age: float | None, body: dict | None, state: str | None) -> dict:
        age_ms = None if age is None else int(age * 1000)
        if freshness == "never":
            return {"last_rx_age_ms": None, "ok": False, "reason": "no_frame_yet"}
        if freshness != "fresh" or body is None:
            return {"last_rx_age_ms": age_ms, "ok": False, "reason": freshness}
        if state != "running":
            return {"last_rx_age_ms": age_ms, "ok": False, "reason": f"state_{state or 'unknown'}"}
        if body.get("supervisor_ok") is not True:
            return {"last_rx_age_ms": age_ms, "ok": False, "reason": "supervisor_not_ok"}
        cycle = body.get("cycle") if type(body.get("cycle")) is dict else {}
        p99, budget = _number(cycle.get("p99_ms")), _number(cycle.get("budget_ms"))
        if p99 is None or budget is None:
            return {"last_rx_age_ms": age_ms, "ok": False, "reason": "cycle_unknown"}
        if p99 > budget:
            return {"last_rx_age_ms": age_ms, "ok": False, "reason": "cycle_slow"}
        return {"last_rx_age_ms": age_ms, "ok": True, "reason": "ok"}
