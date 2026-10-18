"""The device's one braking toggle, read off the controller thread.

``/data/vision_long/braking`` holding exactly the one byte ``1`` is ON. Absent, unreadable, or
anything else (``1`` plus a newline, or any whitespace, included) is OFF. The sunnypilot settings
toggle writes this file atomically and the SSH command is ``printf 1 > /data/vision_long/braking``,
so the state stays as set across drives.

A daemon reader thread reads the file every ``READ_INTERVAL_NS`` and publishes one immutable
(state, read time) pair. The controller thread only reads that pair: no I/O, no lock, no wait.
A pair older than ``STALE_AFTER_NS`` counts as OFF, so a reader stalled on a hung ``/data`` read,
or a dead reader, turns braking OFF within that bound and never holds the 100 Hz loop. An OFF
written to the file is seen within one read interval.
"""

import threading
import time
from collections.abc import Callable

TOGGLE_FILE = "/data/vision_long/braking"
# Reads happen at most this far apart; four reads fit inside the staleness bound.
READ_INTERVAL_NS = 25_000_000
# OFF lands within this bound of the file changing, whatever the reader is doing.
STALE_AFTER_NS = 100_000_000
# Larger than every ON content, so trailing bytes are seen and refused.
MAX_TOGGLE_BYTES = 8
ON_CONTENT = b"1"

ToggleRead = Callable[[str], tuple[bool, str | None]]


def read_toggle_file(path: str) -> tuple[bool, str | None]:
    """(ON, error) for one bounded read; a missing file is OFF without an error."""
    try:
        with open(path, "rb") as stream:
            data = stream.read(MAX_TOGGLE_BYTES)
    except FileNotFoundError:
        return False, None
    except OSError as error:
        # Unreadable is OFF by contract; the reason is kept and counted, never hidden.
        return False, "unreadable:" + type(error).__name__
    return data == ON_CONTENT, None


class BrakingToggle:
    def __init__(
        self,
        path: str = TOGGLE_FILE,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        read: ToggleRead = read_toggle_file,
    ):
        self.path = path
        self._clock = clock
        self._read = read
        # (ON, monotonic read time); replaced whole, so a reader never sees a torn pair.
        self._sample: tuple[bool, int] | None = None
        self.read_errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> None:
        """One read on the calling thread (the reader thread, or startup)."""
        # Stamped before the read: a slow read ages its own result and cannot stretch OFF past the bound.
        started = self._clock()
        state, error = self._read(self.path)
        if error is not None:
            self.read_errors += 1
            self.last_error = error
        self._sample = (state, started)

    def start(self) -> None:
        """First read on the constructor's thread, then the periodic reader."""
        if self._thread is not None:
            raise RuntimeError("braking_toggle_already_started")
        self.refresh()
        self._thread = threading.Thread(target=self._run, name="vision-long-braking-toggle", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(READ_INTERVAL_NS / 1e9):
            self.refresh()

    def stop(self) -> None:
        self._stop.set()

    def enabled(self) -> bool:
        """Controller-thread check: ON only from a read no older than ``STALE_AFTER_NS``."""
        sample = self._sample
        if sample is None:
            return False
        state, read_ns = sample
        return state and 0 <= self._clock() - read_ns <= STALE_AFTER_NS
