"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure logic behind the Elantra port panel: the update gate and the Vision Long braking toggle.

Kept free of pyray and ui_state on purpose: this is the part with actual decisions in it
(is this build verified, may it be installed, is braking switched on), and it needs to be
testable without standing up a UI. `.elantra/test_port_panel.py` and
`.elantra/test_vision_long_toggle.py` import these directly, so the tests exercise the code
that ships rather than a copy of it that can drift.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile

CI_SUCCESS = "success"

# Shared with the Vision Long companion (road contract): exactly the byte "1" is on; absent,
# unreadable or anything else is off. A plain file rather than a Params key, because a new key
# means a params_keys.h change and a native rebuild on the device.
VISION_LONG_BRAKING_FILE = "/data/vision_long/braking"
BRAKING_ON = b"1"
BRAKING_OFF = b"0"
BRAKING_DIR_MODE = 0o755
BRAKING_FILE_MODE = 0o644


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


def braking_enabled(path: str | os.PathLike[str] = VISION_LONG_BRAKING_FILE) -> bool:
    """True only when the state file holds exactly "1".

    Absent, unreadable, a directory, or any other content is off: off is the state in which
    the companion only watches, so an unreadable file can never switch braking on.
    """
    try:
        with open(path, "rb") as stream:
            content = stream.read(len(BRAKING_ON) + 1)
    except OSError:
        return False
    return content == BRAKING_ON


class BrakingFlushError(OSError):
    """The new state is in place (the rename landed) but flushing its folder failed."""


def set_braking(enabled: bool, path: str | os.PathLike[str] = VISION_LONG_BRAKING_FILE) -> None:
    """Write the state atomically: a temporary file in the same folder, then a rename.

    A reader sees either the old complete file or the new one, never a partial write. A
    failure before the rename leaves the previous state; a failure flushing the folder after
    it raises BrakingFlushError with the new state already in place.
    """
    target = os.fspath(path)
    folder = os.path.dirname(target)
    os.makedirs(folder, mode=BRAKING_DIR_MODE, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".braking.", dir=folder)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(BRAKING_ON if enabled else BRAKING_OFF)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, BRAKING_FILE_MODE)
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    try:
        _sync_directory(folder)
    except OSError as error:
        raise BrakingFlushError(error.errno, error.strerror) from error


def _sync_directory(folder: str) -> None:
    """Persist the rename itself, so the state survives the car losing power right after."""
    if not hasattr(os, "O_DIRECTORY"):
        return  # Windows, where the tests run, cannot open a directory for fsync.
    directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def apply_braking(enabled: bool, path: str | os.PathLike[str] = VISION_LONG_BRAKING_FILE) -> tuple[bool, str]:
    """Switch braking and report what is actually stored, plus a note when something failed.

    The note is empty on a clean write, "saved, flush failed: ..." when the rename landed but the
    folder flush did not, and "not saved: ..." when the write never landed. The returned state
    is always read back from the file, never assumed from the request.
    """
    try:
        set_braking(enabled, path)
    except BrakingFlushError as error:
        return braking_enabled(path), "saved, flush failed: " + (error.strerror or type(error).__name__)
    except OSError as error:
        return braking_enabled(path), "not saved: " + (error.strerror or type(error).__name__)
    return braking_enabled(path), ""


def braking_summary(enabled: bool) -> str:
    """The toggle's one-line state, in plain words."""
    return "on: may brake harder" if enabled else "off: watching only"
