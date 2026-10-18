#!/usr/bin/env python3
"""
Tests for the comma-side Vision Long laptop monitor's loop and log
(.elantra/vision_long_supervisor/monitor.py, logcap.py).

The monitor runs over real UDP sockets on loopback: it sends signed comma frames the laptop can
verify, reads laptop frames, writes notice.json atomically, survives a failing input with one log
line, retries a bind that cannot succeed yet, and keeps its log under the cap. msgq is
device-only, so the monitor is handed a stream object with op_state.MsgqStreams' drain().

    python .elantra/test_vision_long_supervisor_monitor.py
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vision_long_supervisor"))
import monitor
import op_state
import wire
from logcap import MAX_BYTES, Log
from test_vision_long_supervisor import KEY, MS, S, VECTORS, laptop_body, signed

class Streams:
    """Stands in for op_state.MsgqStreams (msgq needs its built extension): same drain()."""

    def __init__(self):
        self.batches: list[tuple[list, list]] = []
        self.fail = False

    def drain(self):
        if self.fail:
            raise RuntimeError("msgq reader reset")
        return self.batches.pop(0) if self.batches else ([], [])


class Clock:
    def __init__(self, t: int = 1000 * S):
        self.t = t

    def __call__(self) -> int:
        return self.t


def free_udp() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    return sock


class MonitorLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = monitor.Paths(self.root)
        self.paths.state.mkdir(parents=True)
        self.paths.key.write_bytes(KEY)
        self.paths.params.mkdir(parents=True)
        (self.paths.params / "IsOffroad").write_bytes(b"0")
        (self.root / "proc/sys/kernel/random").mkdir(parents=True)
        (self.root / "proc/sys/kernel/random/boot_id").write_text("boot-1\n")
        self.laptop = free_udp()
        probe = free_udp()
        self.port = probe.getsockname()[1]
        probe.close()
        self.clock = Clock()
        self.streams = Streams()
        self.opens = 0

        def open_streams():
            self.opens += 1
            return self.streams
        self.mon = monitor.Monitor(self.paths, local=("127.0.0.1", self.port), peer=self.laptop.getsockname(),
                                   clock=self.clock, open_streams=open_streams, boot_clock=self.clock, preload=lambda: None)
        self.mon.preloaded.wait(5)
        self.seq = 0

    def tearDown(self):
        if self.mon.sock is not None:
            self.mon.sock.close()
        self.mon.log.close()
        self.laptop.close()
        self.tmp.cleanup()

    def send_laptop(self, **changes) -> None:
        self.seq += 1
        self.laptop.sendto(signed(laptop_body(self.seq, **changes)), ("127.0.0.1", self.port))

    def tick(self, seconds: float = 0.5) -> None:
        self.mon.step()
        self.clock.t += int(seconds * S)

    def notice_file(self) -> dict:
        return json.loads(self.paths.notice.read_bytes())

    def test_it_sends_signed_comma_frames_that_the_laptop_can_verify(self):
        self.tick()
        self.tick()
        frames = [wire.decode(self.laptop.recvfrom(8192)[0], KEY, "comma") for _ in range(2)]
        self.assertEqual([f["seq"] for f in frames], [1, 2])
        self.assertEqual(frames[0]["boot_id"], frames[1]["boot_id"])
        self.assertIs(frames[0]["onroad"], True)
        self.assertIsNone(frames[0]["engaged"])
        self.assertIsNone(frames[0]["screen_on"])
        self.assertEqual(frames[0]["laptop_view"]["reason"], "no_frame_yet")
        self.assertEqual(set(frames[0]), set(VECTORS["vectors"][0]["body"]))

    def test_laptop_frames_make_the_view_ok_and_silence_raises_one_lost_then_one_recovered(self):
        self.mon.step()
        self.send_laptop()
        self.tick()
        self.assertTrue(self.mon.view["ok"])
        for _ in range(8):
            self.tick()
        lost = [n for n in self.notice_file()["notices"] if n["key"] == "laptop_lost"]
        self.assertEqual((len(lost), lost[0]["state"], lost[0]["severity"]), (1, "active", "critical"))
        self.assertIn("laptop_lost", [n["key"] for n in self.mon.frame["notices"]])
        for _ in range(70):
            self.send_laptop()
            self.tick()
        lost = [n for n in self.notice_file()["notices"] if n["key"] == "laptop_lost"]
        self.assertEqual((lost[0]["state"], lost[0]["count"]), ("recovered", 1))
        log = self.paths.log.read_text()
        self.assertEqual((log.count("notice raised laptop_lost"), log.count("notice recovered laptop_lost")), (1, 1))

    def test_tampered_and_foreign_frames_never_refresh_the_laptop(self):
        self.mon.step()
        frame = bytearray(signed(laptop_body(1)))
        frame[-2] ^= 1
        self.laptop.sendto(bytes(frame), ("127.0.0.1", self.port))
        self.laptop.sendto(signed(laptop_body(2), key=bytes(32)), ("127.0.0.1", self.port))
        self.tick()
        self.assertIsNone(self.mon.receiver.last)
        self.assertEqual(self.paths.log.read_text().count("laptop frame dropped (tag)"), 1)

    def test_notice_file_is_atomic_and_carries_the_schema(self):
        self.tick()
        document = self.notice_file()
        self.assertEqual(document["schema"], "vl_notices_v1")
        self.assertEqual(document["written_mono_ns"], 1000 * S)
        self.assertEqual([p.name for p in self.paths.state.iterdir() if p.name.startswith(".")], [])

    def test_a_failing_input_is_logged_once_and_the_loop_goes_on(self):
        (self.paths.params / "IsOffroad").unlink()
        (self.paths.params / "IsOffroad").mkdir()
        for _ in range(10):
            self.tick()
        log = self.paths.log.read_text()
        self.assertEqual(log.count("onroad: "), 1)
        self.assertEqual(self.mon.seq, 10)
        self.assertIsNone(self.mon.frame["onroad"])

    def test_engagement_and_an_unexpected_disengagement_reach_the_screen(self):
        self.clock.t += 31 * S
        t0 = self.clock.t
        on = [op_state.SelfdriveSample(t0 - 400 * MS + i * 10 * MS, True, "", "") for i in range(40)]
        car = [op_state.CarSample(t0 - 1000 * MS + i * 10 * MS, "") for i in range(100)]
        self.streams.batches.append((on, car))
        self.tick(0.1)
        self.assertIs(self.mon.frame["engaged"], True)
        t1 = self.clock.t
        off = [op_state.SelfdriveSample(t1 + i * 10 * MS, False, "controlsMismatch/immediateDisable", "Controls Mismatch")
               for i in range(50)]
        car = [op_state.CarSample(t1 + i * 10 * MS, "") for i in range(50)]
        self.streams.batches.append((off, car))
        self.tick(0.5)
        for _ in range(3):
            t = self.clock.t
            self.streams.batches.append(([op_state.SelfdriveSample(t, False, "", "")], [op_state.CarSample(t, "")]))
            self.tick(0.5)
        self.assertIs(self.mon.frame["engaged"], False)
        found = [n for n in self.notice_file()["notices"] if n["key"] == "disengage_unexpected"]
        self.assertEqual(len(found), 1)
        self.assertIn("controlsMismatch", found[0]["text"])
        self.assertEqual(json.loads(self.paths.budget.read_text()), {"boot_id": "boot-1", "opens": 1})

    def test_going_offroad_resets_engagement_and_is_never_a_disengagement(self):
        self.clock.t += 31 * S
        t0 = self.clock.t
        self.streams.batches.append(([op_state.SelfdriveSample(t0 - 100 * MS + i * 10 * MS, True, "", "") for i in range(10)],
                                     [op_state.CarSample(t0 - 600 * MS + i * 10 * MS, "") for i in range(60)]))
        self.tick(0.2)
        self.assertIs(self.mon.frame["engaged"], True)
        (self.paths.params / "IsOffroad").write_bytes(b"1")
        self.tick(0.2)  # well inside the 1 s gap rule: only the offroad reset can prevent a drop
        (self.paths.params / "IsOffroad").write_bytes(b"0")
        t1 = self.clock.t
        self.streams.batches.append(([op_state.SelfdriveSample(t1 + i * 10 * MS, False, "", "") for i in range(50)],
                                     [op_state.CarSample(t1 + i * 10 * MS, "") for i in range(50)]))
        for _ in range(5):
            self.tick()
        self.assertNotIn("disengage_unexpected", [n["key"] for n in self.notice_file()["notices"]])

    def test_a_status_file_with_missing_keys_or_a_stale_stamp_is_logged_once(self):
        self.paths.status.write_text(json.dumps({"boot_ns": self.clock.t, "authority": "enabled"}))
        for _ in range(4):
            self.tick()
        self.paths.status.write_text(json.dumps({"boot_ns": 0}))
        for _ in range(4):
            self.tick()
        log = self.paths.log.read_text()
        self.assertEqual(log.count("status_schema: "), 2)
        self.assertEqual(log.count("missing keys: completion_overruns, drive_log_dir"), 1)
        self.assertEqual(log.count("stale: the publisher worker is not writing it"), 1)
        self.assertNotIn("status: ", log)

    def test_a_failing_stream_makes_engaged_unknown_and_logs_once(self):
        self.clock.t += 31 * S
        self.streams.batches.append(([op_state.SelfdriveSample(self.clock.t, True, "", "")], []))
        self.tick()
        self.streams.fail = True
        for _ in range(6):
            self.tick()
        self.assertIsNone(self.mon.frame["engaged"])
        self.assertEqual(self.paths.log.read_text().count("msgq reader reset"), 1)

    def test_msgq_is_opened_once_per_process_even_if_the_open_fails(self):
        self.clock.t += 31 * S

        def failing():
            self.opens += 1
            raise RuntimeError("msgq open failed")
        self.mon._open_streams = failing
        for _ in range(6):
            self.tick()
        self.assertEqual(self.opens, 1)
        self.mon._open_streams = lambda: self.streams
        self.tick()
        self.assertIsNone(self.mon.streams)

    def test_a_successful_open_is_not_repeated(self):
        self.clock.t += 31 * S
        for _ in range(6):
            self.tick()
        self.assertEqual(self.opens, 1)

    def test_a_slow_cereal_import_never_holds_up_the_heartbeat(self):
        import threading
        import time as real_time
        release = threading.Event()
        mon = monitor.Monitor(self.paths, local=("127.0.0.1", self.port + 0), peer=self.laptop.getsockname(),
                              clock=self.clock, open_streams=lambda: self.streams, boot_clock=self.clock,
                              preload=lambda: release.wait(5))
        try:
            self.clock.t += 31 * S
            for _ in range(4):
                began = real_time.monotonic()
                mon.step()
                self.assertLess(real_time.monotonic() - began, 0.5)
                self.clock.t += S // 2
            self.assertEqual(mon.seq, 4)
            self.assertIsNone(mon.frame["engaged"])
            self.assertIsNone(mon.streams)
            release.set()
            mon.preloaded.wait(5)
            mon.step()
            self.assertIsNotNone(mon.streams)
        finally:
            release.set()
            if mon.sock is not None:
                mon.sock.close()
            mon.log.close()

    def test_frames_from_another_address_are_dropped(self):
        other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            other.bind(("127.0.0.2", 0))
            self.mon.step()
            other.sendto(signed(laptop_body(1)), ("127.0.0.1", self.port))
            self.tick()
        finally:
            other.close()
        self.assertIsNone(self.mon.receiver.last)
        self.assertIn("drop:source", self.paths.log.read_text())

    def test_notice_file_never_goes_stale_for_the_banner_while_onroad(self):
        for _ in range(40):
            self.tick()
            written = self.notice_file()["written_mono_ns"]
            self.assertLessEqual(self.clock.t - written, monitor.BANNER_STALE_NS - S)

    def test_an_assess_failure_never_leaves_a_stale_ok(self):
        self.mon.step()
        self.send_laptop()
        self.tick()
        self.assertTrue(self.mon.view["ok"])

        def broken(*args):
            raise RuntimeError("assess bug")
        self.mon.health.assess = broken
        self.send_laptop()
        self.tick()
        self.assertEqual(self.mon.view, {"last_rx_age_ms": None, "ok": False, "reason": "assess_error"})
        self.assertEqual(self.mon.frame["laptop_view"]["reason"], "assess_error")

    def test_the_loop_never_spins_after_an_overrun(self):
        deadline, delay = monitor.next_delay(100.0, 103.0)
        self.assertEqual(deadline, 103.0)
        self.assertGreaterEqual(delay, monitor.PERIOD_S / 10)
        self.assertEqual(monitor.next_delay(100.0, 100.2), (100.5, 0.3))

    def test_offroad_or_early_never_opens_msgq(self):
        opened = []
        self.mon._open_streams = lambda: opened.append(1) or self.streams
        self.tick()  # onroad but up for under 30 s
        (self.paths.params / "IsOffroad").write_bytes(b"1")
        self.clock.t += 60 * S
        self.tick()
        self.assertEqual(opened, [])
        self.assertFalse(self.paths.budget.exists())

    def test_an_exhausted_budget_refuses_and_says_so(self):
        self.paths.budget.write_text(json.dumps({"boot_id": "boot-1", "opens": op_state.OPEN_BUDGET}))
        self.clock.t += 31 * S
        self.tick()
        self.tick()
        self.assertIsNone(self.mon.streams)
        self.assertIn("monitor_degraded", [n["key"] for n in self.notice_file()["notices"]])

    def test_a_bind_that_cannot_succeed_is_retried_quietly(self):
        mon = monitor.Monitor(self.paths, local=("192.0.2.1", self.port), peer=self.laptop.getsockname(), clock=self.clock)
        try:
            for _ in range(5):
                mon.step()
            self.assertIsNone(mon.sock)
            self.assertEqual(self.paths.log.read_text().count("bind_address: "), 1)
        finally:
            mon.log.close()

    def test_a_missing_key_raises_monitor_degraded_and_sends_nothing(self):
        self.paths.key.unlink()
        self.tick()
        self.assertIsNone(self.mon.frame)
        self.assertEqual(self.notice_file()["notices"][0]["key"], "monitor_degraded")
        self.paths.key.write_bytes(KEY)
        self.tick()
        self.assertIsNotNone(self.mon.frame)


class LogCap(unittest.TestCase):
    def test_the_log_never_grows_past_2_mb_and_keeps_one_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "supervisor.log"
            log = Log(path)
            try:
                for i in range(6000):
                    log.info(f"{i} " + "x" * 5000)
            finally:
                log.close()
            files = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(files, ["supervisor.log", "supervisor.log.1"])
            for name in files:
                self.assertLessEqual((Path(tmp) / name).stat().st_size, MAX_BYTES)

    def test_a_single_huge_record_cannot_carry_the_file_past_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "supervisor.log"
            log = Log(path)
            try:
                log.once("status", "x" * (3 * MAX_BYTES))
                log.info("y" * (3 * MAX_BYTES))
            finally:
                log.close()
            for name in ("supervisor.log", "supervisor.log.1"):
                if (Path(tmp) / name).exists():
                    self.assertLessEqual((Path(tmp) / name).stat().st_size, MAX_BYTES)

    def test_once_logs_a_repeating_failure_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "supervisor.log"
            log = Log(path)
            try:
                for _ in range(50):
                    log.once("carrier", "OSError: gone")
                log.cleared("carrier")
                log.once("carrier", "OSError: gone")
            finally:
                log.close()
            self.assertEqual(path.read_text().count("carrier: OSError: gone"), 2)


if __name__ == "__main__":
    unittest.main()
