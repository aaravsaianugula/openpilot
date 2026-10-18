#!/usr/bin/env python3
"""
The monitor's log: /data/vision_long/supervisor.log, rotated at 2 MB with one backup kept.

/data runs about 90% full, so the cap is hard. RotatingFileHandler rolls over before a record
would take the file to maxBytes, and every record is clipped to RECORD_MAX characters first, so
no single record can carry the file past the cap either. Repeating failures are logged once per
key (`once`), so a broken input writes one line, not one line per cycle.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
BACKUPS = 1
RECORD_MAX = 1000


class _Clip(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if len(message) > RECORD_MAX:
            message = message[:RECORD_MAX - 3] + "..."
        record.msg, record.args = message, None
        record.exc_info = record.exc_text = None
        return True


class Log:
    def __init__(self, path: Path, name: str = "vision_long_supervisor", max_bytes: int = MAX_BYTES) -> None:
        self._logger = logging.getLogger(f"{name}:{path}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for old in list(self._logger.handlers):
            self._logger.removeHandler(old)
            old.close()
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=BACKUPS, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler.addFilter(_Clip())
        self._logger.addHandler(handler)
        self._handler = handler
        self._seen: dict[str, str] = {}

    def info(self, message: str) -> None:
        self._logger.info(message)

    def once(self, key: str, message: str) -> None:
        """Log `message` the first time `key` reports it, and again only when it changes."""
        if self._seen.get(key) != message:
            self._seen[key] = message
            self._logger.warning(f"{key}: {message}")

    def cleared(self, key: str) -> None:
        """The failure behind `key` stopped; the next one is logged again."""
        if self._seen.pop(key, None) is not None:
            self._logger.info(f"{key}: ok again")

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()
