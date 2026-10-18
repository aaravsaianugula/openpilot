#!/usr/bin/env python3
"""
The comma side of the Vision Long notice model (lifecycle contract, "Notices").

    {key, severity: info|warn|critical, text (<= 80 chars), first_mono_ns, count, state: active|recovered}

- A key raises once, on its first onset. While it is active a new onset only adds to `count`.
- At most one notification per key per 10 min: a key that raises again within 10 min of its last
  notification folds into the earlier notice (same first_mono_ns, count + 1) and is not shown again.
- A worse severity while active (warn to critical) is notified once more; a further escalation
  within 10 min of that one folds.
- A key that has been clear for 30 s gets one "recovered" notification and is closed. A folded
  episode that was never shown recovers silently, so one notice is followed by exactly one recovered.

Two kinds of input. `update()` takes level conditions, evaluated every cycle: a key present now is
on, a key absent is clear. `event()` takes an occurrence with no duration (an unexpected
disengagement): it is an onset at that instant and is clear from the same instant.

Times are CLOCK_MONOTONIC ns. The file the comma screen reads carries `notified_mono_ns` and
`recovered_mono_ns`, the moments a notification was issued, so the banner can show each one for
a fixed time without keeping state of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

SEVERITIES = ("info", "warn", "critical")
TEXT_MAX = 80
FOLD_NS = 10 * 60 * 1_000_000_000
CLEAR_NS = 30 * 1_000_000_000
# How long a recovered notice stays in what is exported (heartbeat and notice.json).
RECENT_NS = 60 * 1_000_000_000
MIRROR_PREFIX = "laptop."


def rank(severity: str) -> int:
    return SEVERITIES.index(severity)


def clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= TEXT_MAX else text[:TEXT_MAX - 3] + "..."


@dataclass
class Notice:
    key: str
    severity: str
    text: str
    first_mono_ns: int
    count: int
    state: str
    last_notified_ns: int
    notified_mono_ns: int | None
    recovered_mono_ns: int | None = None
    shown: bool = True
    present: bool = False
    clear_since_ns: int | None = None
    closed_ns: int | None = None
    escalated_ns: int | None = None

    def export(self) -> dict:
        return {
            "key": self.key, "severity": self.severity, "text": self.text,
            "first_mono_ns": self.first_mono_ns, "count": self.count, "state": self.state,
            "notified_mono_ns": self.notified_mono_ns, "recovered_mono_ns": self.recovered_mono_ns,
        }


class NoticeBook:
    def __init__(self) -> None:
        self._notices: dict[str, Notice] = {}

    def _onset(self, key: str, severity: str, text: str, now: int, issued: list[str]) -> Notice:
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}")
        text = clip(text)
        notice = self._notices.get(key)
        if notice is not None and notice.state == "active":
            notice.count += 1
            notice.text = text
            self._escalate(notice, severity, now, issued)
        elif notice is not None and now - notice.last_notified_ns < FOLD_NS:
            # A flap inside the 10 min window: the same notice again, silently, unless it is worse.
            notice.state, notice.count, notice.shown = "active", notice.count + 1, False
            notice.text = text
            notice.closed_ns = notice.recovered_mono_ns = None
            if rank(severity) > rank(notice.severity):
                self._escalate(notice, severity, now, issued)
            else:
                notice.severity = severity
        else:
            notice = Notice(key, severity, text, now, 1, "active", now, now)
            self._notices[key] = notice
            issued.append(f"raised {key} {severity}: {text}")
        notice.clear_since_ns = None
        return notice

    def _escalate(self, notice: Notice, severity: str, now: int, issued: list[str]) -> None:
        """A worse severity is shown once; a further escalation within 10 min of that folds."""
        if rank(severity) <= rank(notice.severity):
            return
        notice.severity = severity
        if notice.escalated_ns is not None and now - notice.escalated_ns < FOLD_NS:
            return
        notice.escalated_ns = notice.notified_mono_ns = notice.last_notified_ns = now
        notice.shown = True
        issued.append(f"escalated {notice.key} {severity}: {notice.text}")

    def update(self, present: dict[str, tuple[str, str]], now: int) -> list[str]:
        """Apply this cycle's level conditions {key: (severity, text)}. Returns what was issued."""
        issued: list[str] = []
        for key, (severity, text) in present.items():
            notice = self._notices.get(key)
            if notice is not None and notice.state == "active" and notice.present:
                notice.clear_since_ns = None
                notice.text = clip(text)
                self._escalate(notice, severity, now, issued)
                continue
            self._onset(key, severity, text, now, issued).present = True
        for key, notice in self._notices.items():
            if key not in present and notice.present:
                notice.present = False
                notice.clear_since_ns = now
        self._advance(now, issued)
        return issued

    def event(self, key: str, severity: str, text: str, now: int) -> list[str]:
        """One occurrence with no duration. It is clear from the instant it happened."""
        issued: list[str] = []
        notice = self._onset(key, severity, text, now, issued)
        notice.present = False
        notice.clear_since_ns = now
        return issued

    def _advance(self, now: int, issued: list[str]) -> None:
        for key in list(self._notices):
            notice = self._notices[key]
            if notice.state == "active" and notice.clear_since_ns is not None and now - notice.clear_since_ns >= CLEAR_NS:
                notice.state, notice.closed_ns = "recovered", now
                if notice.shown:
                    notice.recovered_mono_ns = now
                    issued.append(f"recovered {key}")
            if notice.state == "recovered" and now - notice.last_notified_ns >= FOLD_NS and now - (notice.closed_ns or now) >= RECENT_NS:
                del self._notices[key]

    def active(self) -> list[Notice]:
        return [n for n in self._notices.values() if n.state == "active"]

    def snapshot(self, now: int) -> list[dict]:
        """Active notices and those recovered within the last 60 s, worst first."""
        recent = [n for n in self._notices.values()
                  if n.state == "active" or (n.closed_ns is not None and now - n.closed_ns < RECENT_NS)]
        recent.sort(key=lambda n: (n.state != "active", -rank(n.severity), n.first_mono_ns))
        return [n.export() for n in recent]

    def heartbeat(self, now: int) -> list[dict]:
        """The heartbeat's `notices` field: {key, severity, text, state} only. Notices mirrored from
        the laptop (`laptop.*`) are never sent back to it."""
        return [{k: n[k] for k in ("key", "severity", "text", "state")} for n in self.snapshot(now)
                if not n["key"].startswith(MIRROR_PREFIX)]
