#!/usr/bin/env python3
"""
The Vision Long heartbeat codec (lifecycle contract, "Heartbeat channel").

    datagram = b"VLHB1" + HMAC-SHA256(key, b"vl-heartbeat-v1\\0" + body) + body
    body     = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

The laptop's codec must reproduce the same bytes; heartbeat-vectors.json pins both.

A frame is dropped for: a datagram over the size cap, a missing magic, a bad tag, a body that is
not a UTF-8 JSON object, a wrong schema or sender, a missing or ill-typed boot_id or seq, or a
seq that is not above the last one accepted from the same boot_id. A new boot_id starts over.

Freshness is measured only on the receiver's own CLOCK_MONOTONIC at receive time. The sender's
mono_ns and wall_utc are carried for logs and never compared with anything here, so a laptop
whose clock is hours off cannot make a frame look fresh, and a replayed frame cannot either.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"VLHB1"
TAG_BYTES = 32
TAG_CONTEXT = b"vl-heartbeat-v1\0"
SCHEMA = "vl_heartbeat_v1"
KEY_BYTES = 32
MAX_BODY_BYTES = 4096
MAX_DATAGRAM_BYTES = len(MAGIC) + TAG_BYTES + MAX_BODY_BYTES
SENDERS = ("comma", "laptop")
BOOT_ID = re.compile(r"[0-9a-f]{16}")

STALE_S = 1.0
LOST_S = 3.0
RETIRED_BOOTS = 256


class Drop(Exception):
    """A datagram that must be ignored. The message is a short reason key for the log."""


def load_key(path: Path) -> bytes:
    """The 32-byte link key. Anything else is refused rather than used."""
    data = path.read_bytes()
    if len(data) != KEY_BYTES:
        raise ValueError(f"{path} is {len(data)} bytes, expected {KEY_BYTES}")
    return data


def encode_body(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def tag(key: bytes, body: bytes) -> bytes:
    return hmac.new(key, TAG_CONTEXT + body, hashlib.sha256).digest()


def encode(body: dict, key: bytes) -> bytes:
    raw = encode_body(body)
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError(f"heartbeat body is {len(raw)} bytes, over the {MAX_BODY_BYTES}-byte cap")
    return MAGIC + tag(key, raw) + raw


def decode(datagram: bytes, key: bytes, sender: str) -> dict:
    """Verify and parse one datagram from `sender`. Raises Drop with the reason."""
    if len(datagram) > MAX_DATAGRAM_BYTES:
        raise Drop("oversize")
    if not datagram.startswith(MAGIC) or len(datagram) < len(MAGIC) + TAG_BYTES:
        raise Drop("magic")
    received = datagram[len(MAGIC):len(MAGIC) + TAG_BYTES]
    raw = datagram[len(MAGIC) + TAG_BYTES:]
    if not hmac.compare_digest(received, tag(key, raw)):
        raise Drop("tag")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise Drop("json") from error
    if type(body) is not dict:
        raise Drop("json")
    if body.get("schema") != SCHEMA:
        raise Drop("schema")
    if body.get("sender") != sender:
        raise Drop("sender")
    boot_id, seq = body.get("boot_id"), body.get("seq")
    if type(boot_id) is not str or BOOT_ID.fullmatch(boot_id) is None:
        raise Drop("boot_id")
    if type(seq) is not int or seq < 1:
        raise Drop("seq")
    return body


@dataclass
class Accepted:
    body: dict
    rx_mono_ns: int


class Receiver:
    """Replay and freshness state for one peer. Times are the receiver's CLOCK_MONOTONIC ns."""

    def __init__(self, key: bytes, sender: str):
        if sender not in SENDERS:
            raise ValueError(f"unknown sender {sender!r}")
        self._key = key
        self._sender = sender
        self._boot_id: str | None = None
        self._seq = 0
        # Boot ids that were replaced by a newer one. A frame from one of them is a replay of a
        # dead process, never a live peer, so it is refused however high its seq is.
        self._retired: dict[str, None] = {}
        self.last: Accepted | None = None
        self.accepted = 0

    def offer(self, datagram: bytes, rx_mono_ns: int) -> dict:
        """Accept one datagram or raise Drop. Only an accepted frame refreshes freshness."""
        body = decode(datagram, self._key, self._sender)
        boot_id, seq = body["boot_id"], body["seq"]
        if boot_id in self._retired:
            raise Drop("retired_boot")
        if boot_id == self._boot_id:
            if seq <= self._seq:
                raise Drop("replay")
        elif self._boot_id is not None:
            self._retired[self._boot_id] = None
            while len(self._retired) > RETIRED_BOOTS:
                self._retired.pop(next(iter(self._retired)))
        self._boot_id, self._seq = boot_id, seq
        self.last = Accepted(body, rx_mono_ns)
        self.accepted += 1
        return body

    def age_s(self, now_mono_ns: int) -> float | None:
        """Seconds since the last accepted frame, on this machine's monotonic clock; None if never."""
        if self.last is None:
            return None
        return max(0.0, (now_mono_ns - self.last.rx_mono_ns) / 1e9)

    def freshness(self, now_mono_ns: int) -> str:
        """never | fresh | stale (>1 s) | lost (>3 s)."""
        age = self.age_s(now_mono_ns)
        if age is None:
            return "never"
        if age > LOST_S:
            return "lost"
        if age > STALE_S:
            return "stale"
        return "fresh"


def new_boot_id() -> str:
    return os.urandom(8).hex()
