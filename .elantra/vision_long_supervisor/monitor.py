#!/usr/bin/env python3
"""
The comma-side Vision Long laptop monitor. Notification only: it is never in the authority path.

    /usr/local/venv/bin/python3 /data/openpilot/.elantra/vision_long_supervisor/monitor.py

At 2 Hz it trades heartbeats with the laptop over the USB link (UDP 45557, 192.168.250.1 <->
192.168.250.2), reads openpilot state, judges the laptop's health and keeps the notices: the
comma screen reads /data/vision_long/notice.json and the laptop screen gets them in the comma
frame. Authority never depends on any of this. The companion withdraws to native output on its
own when live's signed proposals stop, and the laptop supervisor kills live on its own checks.

It never interferes with openpilot: it publishes on no openpilot service, writes no params,
signals no process, and reads msgq under a per-boot budget (op_state.py says why). The unit
runs it at Nice=19 with a 5% CPU quota and a 64 MB memory cap.

Every input is read inside a guard. A failing input is reported unknown and logged once per key,
and the loop carries on at its own pace: it never dies silently and never spins.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_status
import health
import op_state
import wire
from logcap import Log
from notices import NoticeBook, clip

PERIOD_S = 0.5
PORT = 45557
COMMA_IP = "192.168.250.1"
LAPTOP_IP = "192.168.250.2"
RX_PER_CYCLE = 16
PROC_EVERY = 4                       # scan /proc for manager and selfdrived every 2 s
WRITE_ONROAD_NS = 2_000_000_000      # notice.json keepalive while onroad
BANNER_STALE_NS = 6_000_000_000      # the banner shows nothing for an older file (vision_long_notice.STALE_NS)
WRITE_OFFROAD_NS = 30_000_000_000
SELFDRIVE_FRESH_NS = 1_000_000_000
NOTICE_SCHEMA = "vl_notices_v1"
MANAGER_TITLES = ("manager.py", "openpilot.system.manager.manager", "openpilot.system.manager")
SELFDRIVED_TITLE = "openpilot.selfdrive.selfdrived.selfdrived"
GADGET_IFNAME = "sys/kernel/config/usb_gadget/vision_long/functions/ncm.usb0/ifname"


@dataclass(frozen=True)
class Paths:
    """Every path the monitor reads or writes. On the device the root is `/`."""

    root: Path

    @property
    def state(self) -> Path:
        return self.root / "data/vision_long"

    @property
    def key(self) -> Path:
        return self.state / "link.key"

    @property
    def notice(self) -> Path:
        return self.state / "notice.json"

    @property
    def status(self) -> Path:
        return self.state / "status.json"

    @property
    def log(self) -> Path:
        return self.state / "supervisor.log"

    @property
    def budget(self) -> Path:
        return self.state / "msgq_opens.json"

    @property
    def params(self) -> Path:
        return self.root / "data/params/d"

    @property
    def net(self) -> Path:
        return self.root / "sys/class/net"

    @property
    def backlight(self) -> Path:
        return self.root / "sys/class/backlight/panel0-backlight"

    @property
    def ifname(self) -> Path:
        return self.root / GADGET_IFNAME

    @property
    def proc(self) -> Path:
        return self.root / "proc"


def write_atomic(path: Path, data: bytes) -> None:
    """Temp file then rename, so a reader sees the old file or the new one, never half of one.
    No fsync: the file is rewritten every 2 s onroad and is worthless after a power cut, and
    /data is flash that should not take a sync this often."""
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
    os.replace(temporary, path)


def read_screen(backlight: Path) -> bool | None:
    """bl_power 0 and a non-zero brightness is on. Unreadable is unknown, never on."""
    try:
        power = (backlight / "bl_power").read_text().strip()
        brightness = int((backlight / "brightness").read_text().strip())
    except (OSError, ValueError):
        return None
    return power == "0" and brightness > 0


def process_titles(proc: Path) -> list[list[str]]:
    titles = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue  # the process exited between the listing and the read
        titles.append([part for chunk in raw.split(b"\0") for part in chunk.decode("utf-8", "replace").split(" ") if part])
    return titles


def openpilot_alive(titles: list[list[str]], onroad: bool | None) -> bool:
    """manager alive, and selfdrived alive when onroad (manager runs it only onroad)."""
    manager = any(any(p == t or p.endswith("/" + t) for p in parts for t in MANAGER_TITLES) for parts in titles)
    selfdrived = any(SELFDRIVED_TITLE in parts for parts in titles)
    return manager and (selfdrived or onroad is not True)


def next_delay(deadline: float, now: float) -> tuple[float, float]:
    """The next tick on a fixed 2 Hz schedule. After an overrun the schedule restarts from now and
    the loop still sleeps a tenth of a period, so it can never spin."""
    deadline += PERIOD_S
    if deadline - now < PERIOD_S / 10:
        return now, PERIOD_S / 10
    return deadline, round(deadline - now, 9)


class Monitor:
    def __init__(self, paths: Paths, local: tuple[str, int] = (COMMA_IP, PORT), peer: tuple[str, int] = (LAPTOP_IP, PORT),
                 clock=time.monotonic_ns, open_streams=op_state.MsgqStreams, boot_clock=device_status.boottime_ns,
                 preload=op_state.preload_messaging) -> None:
        self.paths = paths
        self.local, self.peer = local, peer
        self.clock = clock
        self.boot_clock = boot_clock  # CLOCK_BOOTTIME, the clock road-ready stamps status.json with
        self._open_streams = open_streams
        paths.state.mkdir(parents=True, exist_ok=True)
        self.log = Log(paths.log)
        self.started_ns = clock()
        self.boot_id = wire.new_boot_id()
        self.seq = 0
        self.cycles = 0
        self.book = NoticeBook()
        self.health = health.HealthEvaluator(self.started_ns)
        self.status_conditions = device_status.StatusConditions()
        self.detector = op_state.DisengageDetector()
        self.receiver: wire.Receiver | None = None
        self.key: bytes | None = None
        self.sock: socket.socket | None = None
        self.streams = None
        self.streams_tried = False
        self.streams_refused = False
        self.selfdrive_seen_ns: int | None = None
        self.openpilot_ok = False
        self.last_written: bytes | None = None
        self.last_write_ns: int | None = None
        self.frame: dict | None = None
        self.view: dict = {"last_rx_age_ms": None, "ok": False, "reason": "no_frame_yet"}
        self.log.info(f"monitor start boot_id={self.boot_id} pid={os.getpid()}")
        # cereal and pycapnp load on their own thread at start, so the first heartbeat is never
        # late. Until they are loaded the openpilot fields are null; msgq opens only after.
        self.preloaded = threading.Event()
        self.preload_error: str | None = None
        threading.Thread(target=self._preload, args=(preload,), name="cereal-preload", daemon=True).start()

    def _preload(self, preload) -> None:
        try:
            preload()
        except Exception as error:  # reported by the loop; the monitor runs on without msgq
            self.preload_error = f"{type(error).__name__}: {error}"
        finally:
            self.preloaded.set()

    # --- guarded inputs ------------------------------------------------------------------------

    def guard(self, key: str, read, unknown=None):
        try:
            value = read()
        except Exception as error:  # an input failure is reported unknown, never fatal
            self.log.once(key, f"{type(error).__name__}: {error}")
            return unknown
        self.log.cleared(key)
        return value

    def _load_key(self) -> None:
        if self.key is None:
            self.key = self.guard("link_key", lambda: wire.load_key(self.paths.key))
            if self.key is not None:
                self.receiver = wire.Receiver(self.key, "laptop")

    def _bind(self) -> None:
        if self.sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setblocking(False)
            sock.bind(self.local)
        except OSError as error:
            sock.close()
            # No cable yet means no 192.168.250.1: keep retrying, quietly.
            quiet = error.errno in (errno.EADDRNOTAVAIL, errno.EADDRINUSE)
            self.log.once("bind_address", ("waiting for " if quiet else "") + f"{self.local[0]}:{self.local[1]}: {error}")
            return
        self.log.cleared("bind_address")
        self.log.info(f"bound {self.local[0]}:{self.local[1]}")
        self.sock = sock

    def _receive(self, now_ns: int) -> None:
        if self.sock is None or self.receiver is None:
            return
        for _ in range(RX_PER_CYCLE):
            try:
                datagram, source = self.sock.recvfrom(wire.MAX_DATAGRAM_BYTES + 1)
            except BlockingIOError:
                return
            if source[0] != self.peer[0]:
                self.log.once("drop:source", f"datagram from {source[0]}")
                continue
            try:
                self.receiver.offer(datagram, self.clock())
            except wire.Drop as drop:
                self.log.once("drop:" + str(drop), f"laptop frame dropped ({drop})")

    def _streams_drain(self, now_ns: int, onroad: bool | None) -> None:
        if self.preload_error is not None:
            self.log.once("cereal_import", self.preload_error)
        ready = self.preloaded.is_set() and self.preload_error is None
        uptime_ok = now_ns - self.started_ns >= op_state.OPEN_AFTER_UPTIME_S * 1e9
        if self.streams is None and not self.streams_tried and ready and onroad is True and uptime_ok:
            self.streams_tried = True
            boot = op_state.kernel_boot_id(self.paths.proc)
            if op_state.claim_open(self.paths.budget, boot):
                self.streams = self._open_streams()
                self.log.info("subscribed selfdriveState, carState")
            else:
                self.streams_refused = True
                self.log.once("msgq_budget", f"{op_state.OPEN_BUDGET} msgq opens already used this boot; engaged unknown")
        if self.streams is None:
            return
        selfdrive, car = self.streams.drain()
        if selfdrive:
            self.selfdrive_seen_ns = now_ns
        self.detector.feed(selfdrive, car)

    # --- one cycle -----------------------------------------------------------------------------

    def step(self) -> None:
        now = self.clock()
        self.cycles += 1
        self._load_key()
        self.guard("bind", self._bind)
        self.guard("receive", lambda: self._receive(now))
        onroad = self.guard("onroad", lambda: op_state.read_onroad(self.paths.params))
        drained = self.guard("openpilot_streams", lambda: self._streams_drain(now, onroad), unknown=False)
        if drained is False or onroad is not True:
            # A lost stream or an offroad car ends what is known about engagement; never infer a drop.
            self.detector.reset()
        ifname = self.guard("ifname", lambda: self.paths.ifname.read_text().strip() or None)
        carrier = self.guard("carrier", lambda: health.read_carrier(self.paths.net, ifname))
        status = self.guard("status", lambda: device_status.read_status(self.paths.status, self.boot_clock()),
                            unknown=device_status.UNKNOWN)
        if status.problem is None:
            self.log.cleared("status_schema")
        else:
            self.log.once("status_schema", f"{self.paths.status}: {status.problem}")
        screen = self.guard("screen", lambda: read_screen(self.paths.backlight))
        if self.cycles % PROC_EVERY == 1:
            self.openpilot_ok = self.guard("openpilot_ok", lambda: openpilot_alive(process_titles(self.paths.proc), onroad),
                                           unknown=False)
        self.guard("notices", lambda: self._notices(now, onroad, carrier, status))
        self.guard("notice_file", lambda: self._write_notices(now, onroad))
        self.guard("send", lambda: self._send(now, onroad, screen))

    def _engaged(self, now: int) -> bool | None:
        if self.selfdrive_seen_ns is None or now - self.selfdrive_seen_ns > SELFDRIVE_FRESH_NS:
            return None
        return self.detector.engaged

    def _notices(self, now: int, onroad: bool | None, carrier: bool | None, status: device_status.DeviceStatus) -> None:
        own: dict[str, tuple[str, str]] = {}
        if self.receiver is not None:
            # Set first, so an exception in assess can never leave an old ok:true behind.
            self.view = {"last_rx_age_ms": None, "ok": False, "reason": "assess_error"}
            assessment = self.health.assess(self.receiver, now, onroad, carrier, status)
            own.update(assessment.present)
            self.view = assessment.view
        else:
            self.view = {"last_rx_age_ms": None, "ok": False, "reason": "no_link_key"}
            own["monitor_degraded"] = ("critical", f"Vision Long link key unusable: {self.paths.key}")
        own.update(self.status_conditions.assess(status, now, onroad))
        if self.streams_refused and onroad is True:
            own.setdefault("monitor_degraded", ("info", "laptop monitor cannot read engagement this boot"))
        issued = self.book.update(own, now)
        for drop in self.detector.decide(now, onroad):
            cause = drop.alert or "no driver input seen"
            text = f"openpilot disengaged: {cause}" + ("" if drop.verified else " (carState gap)")
            issued += self.book.event("disengage_unexpected", "warn", clip(text), now)
        for line in issued:
            self.log.info("notice " + line)

    def _write_notices(self, now: int, onroad: bool | None) -> None:
        notices = self.book.snapshot(now)
        content = json.dumps(notices, sort_keys=True).encode()
        keepalive = WRITE_ONROAD_NS if onroad is True else WRITE_OFFROAD_NS
        if content == self.last_written and self.last_write_ns is not None and now - self.last_write_ns < keepalive:
            return
        document = {"schema": NOTICE_SCHEMA, "written_mono_ns": now, "notices": notices}
        write_atomic(self.paths.notice, json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
        self.last_written, self.last_write_ns = content, now

    def _send(self, now: int, onroad: bool | None, screen: bool | None) -> None:
        if self.sock is None or self.key is None:
            return
        self.seq += 1
        notices = self.book.heartbeat(now)
        body = {
            "schema": wire.SCHEMA, "sender": "comma", "boot_id": self.boot_id, "seq": self.seq, "mono_ns": now,
            "wall_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "onroad": onroad, "engaged": self._engaged(now), "screen_on": screen, "openpilot_ok": self.openpilot_ok,
            "monitor_uptime_s": round((now - self.started_ns) / 1e9, 1),
            "laptop_view": self.view,
            "notices": notices,
        }
        while len(wire.encode_body(body)) > wire.MAX_BODY_BYTES and body["notices"]:
            body["notices"] = body["notices"][:-1]
        self.frame = body
        try:
            self.sock.sendto(wire.encode(body, self.key), self.peer)
        except OSError:
            # The address went away (cable out, link re-armed). Bind again next cycle.
            self.sock.close()
            self.sock = None
            raise

    def run(self) -> None:
        deadline = time.monotonic()
        while True:
            try:
                self.step()
            except Exception as error:  # step guards every input; this is the last line
                self.log.once("loop", f"{type(error).__name__}: {error}")
            deadline, delay = next_delay(deadline, time.monotonic())
            time.sleep(delay)


def main() -> int:
    Monitor(Paths(Path("/"))).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
