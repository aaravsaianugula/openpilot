"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure logic behind the Elantra port update panel.

Kept free of pyray and ui_state on purpose: this is the part with actual decisions in it
(is this build verified, may it be installed), and it needs to be testable without standing
up a UI. `.elantra/test_port_panel.py` imports these directly, so the tests exercise the
code that ships rather than a copy of it that can drift.
"""

from __future__ import annotations

import datetime
import json

CI_SUCCESS = "success"


def load_manifest(raw: str | bytes | None) -> dict:
    """Parse a build manifest out of a param value.

    Anything unparseable comes back as an empty dict, which callers read as "this is not one
    of our builds" rather than as an error. A malformed manifest must never be able to wedge
    the update flow.
    """
    if not raw:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ci_is_green(manifest: dict) -> bool:
    return manifest.get("upstream_ci_conclusion") == CI_SUCCESS


def pending_allowed(verified_only: bool, pending_raw: str | bytes | None) -> bool:
    """May the staged build be installed?

    With the gate off, always. With it on, only if the staged build's upstream CI passed --
    except when there is no manifest at all, which means the user deliberately switched to
    some other branch. Blocking that would strand them on a branch they chose, so it doesn't.
    """
    if not verified_only:
        return True
    pending = load_manifest(pending_raw)
    if not pending:
        return True
    return ci_is_green(pending)


def install_offered(update_available: bool, verified_only: bool,
                    pending_raw: str | bytes | None) -> bool:
    return update_available and pending_allowed(verified_only, pending_raw)


def update_blocked(update_available: bool, verified_only: bool,
                   pending_raw: str | bytes | None) -> bool:
    return update_available and not pending_allowed(verified_only, pending_raw)


def age(iso: str | None, now: datetime.datetime | None = None) -> str:
    """Coarse "how long ago" for the last sync. Weekly cadence, so days is the useful unit."""
    if not iso:
        return "unknown"
    try:
        when = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "unknown"
    now = now or datetime.datetime.now(datetime.UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)
    delta = now - when
    if delta.total_seconds() < 0:
        return "just now"
    days = delta.days
    if days >= 2:
        return f"{days} days ago"
    hours = int(delta.total_seconds() // 3600)
    if hours >= 2:
        return f"{hours} hours ago"
    return "just now"
