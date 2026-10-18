#!/usr/bin/env python3
"""
Tests for the comma-side Vision Long laptop monitor (.elantra/vision_long_supervisor/).

What these pin:
  * the heartbeat codec reproduces the contract's golden vectors byte for byte, and drops a
    tampered, replayed, wrongly keyed, oversize or off-schema frame;
  * freshness is the receiver's own monotonic clock, so a sender's clock cannot make a frame
    fresh, and a replay never refreshes anything;
  * notices raise once, fold flaps inside 10 min, and recover once after 30 s clear;
  * the laptop health rules, the disengagement rule (driver cause within 0.5 s), and the
    defensive status.json parse;
  * the monitor itself over real UDP sockets on loopback: it sends signed comma frames, reads
    laptop frames, writes notice.json atomically, survives a failing input with one log line,
    retries a bind that cannot succeed yet, and keeps its log under the cap.

The msgq subscriptions are device-only (msgq needs its built extension). The decoders they feed
are tested here against real capnp Events built from this fork's cereal schema; the monitor
test hands it a stream object with the same drain() as op_state.MsgqStreams.

    python .elantra/test_vision_long_supervisor.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vision_long_supervisor"))
import device_status
import health
import notices
import op_state
import wire

VECTORS = json.loads((HERE / "vision_long_supervisor/heartbeat-vectors.json").read_text())
KEY = bytes.fromhex(VECTORS["key_hex"])
S = 1_000_000_000


def laptop_body(seq: int, boot_id: str = "fedcba9876543210", **changes) -> dict:
    body = json.loads(json.dumps(VECTORS["vectors"][1]["body"]))
    body.update(seq=seq, boot_id=boot_id, **changes)
    return body


def signed(body: dict, key: bytes = KEY) -> bytes:
    return wire.encode(body, key)


def status_doc(**changes) -> dict:
    """road-ready's status.json, with its writer's exact field names (vl-road 69227b0)."""
    doc = {"boot_ns": 0, "mode": "braking_only", "authority": "enabled", "reason": "braking_toggle_on", "toggle": "on",
           "laptop_session": "up", "last_packet_age_ms": 12, "publisher_restarts": 0,
           "drive_log_dir": "/data/vision_long/drives/20261017T000000Z-01234567", "drive_log_error": None,
           "completion_overruns": 0, "worst_completion_us": 840.0}
    doc.update(changes)
    return doc


class WireVectors(unittest.TestCase):
    def test_both_golden_vectors_are_reproduced_byte_for_byte(self):
        for vector in VECTORS["vectors"]:
            with self.subTest(sender=vector["body"]["sender"]):
                self.assertEqual(wire.encode(vector["body"], KEY).hex(), vector["datagram_hex"])

    def test_the_golden_vectors_decode_to_their_bodies(self):
        for vector in VECTORS["vectors"]:
            datagram = bytes.fromhex(vector["datagram_hex"])
            self.assertEqual(wire.decode(datagram, KEY, vector["body"]["sender"]), vector["body"])


class WireDrops(unittest.TestCase):
    def assertDrop(self, datagram: bytes, reason: str, sender: str = "laptop", key: bytes = KEY):
        with self.assertRaises(wire.Drop) as caught:
            wire.decode(datagram, key, sender)
        self.assertEqual(str(caught.exception), reason)

    def test_a_flipped_body_byte_or_tag_byte_is_a_bad_tag(self):
        good = bytearray(signed(laptop_body(1)))
        body_flip, tag_flip = bytearray(good), bytearray(good)
        body_flip[-3] ^= 0x01
        tag_flip[len(wire.MAGIC) + 4] ^= 0x80
        self.assertDrop(bytes(body_flip), "tag")
        self.assertDrop(bytes(tag_flip), "tag")

    def test_a_frame_signed_with_another_key_is_dropped(self):
        self.assertDrop(signed(laptop_body(1), key=bytes(32)), "tag")

    def test_oversize_datagrams_are_dropped_before_any_hmac_and_never_encoded(self):
        self.assertDrop(wire.MAGIC + bytes(32) + b"x" * (wire.MAX_BODY_BYTES + 1), "oversize")
        with self.assertRaises(ValueError):
            wire.encode(laptop_body(1, reason="x" * wire.MAX_BODY_BYTES), KEY)

    def test_a_body_at_the_cap_is_accepted(self):
        body = laptop_body(1, reason="")
        body["reason"] = "x" * (wire.MAX_BODY_BYTES - len(wire.encode_body(body)))
        self.assertEqual(len(wire.encode_body(body)), wire.MAX_BODY_BYTES)
        self.assertEqual(wire.decode(signed(body), KEY, "laptop")["seq"], 1)

    def test_signed_but_off_schema_frames_are_dropped(self):
        def raw(body_bytes: bytes) -> bytes:
            return wire.MAGIC + wire.tag(KEY, body_bytes) + body_bytes
        self.assertDrop(raw(b"[1,2]"), "json")
        self.assertDrop(raw(b"\xff\xfe"), "json")
        self.assertDrop(raw(b"{not json"), "json")
        self.assertDrop(signed(laptop_body(1, schema="vl_heartbeat_v2")), "schema")
        self.assertDrop(signed(laptop_body(1, sender="comma")), "sender")
        self.assertDrop(signed(laptop_body(1, boot_id="XYZ")), "boot_id")
        self.assertDrop(signed(laptop_body(0)), "seq")
        boolean_seq = laptop_body(1)
        boolean_seq["seq"] = True
        self.assertDrop(signed(boolean_seq), "seq")
        self.assertDrop(b"VLHB0" + bytes(40), "magic")


class WireReplayAndFreshness(unittest.TestCase):
    def test_seq_must_increase_within_a_boot_and_a_new_boot_resets(self):
        rx = wire.Receiver(KEY, "laptop")
        rx.offer(signed(laptop_body(5)), 1 * S)
        for stale in (5, 4, 1):
            with self.assertRaises(wire.Drop) as caught:
                rx.offer(signed(laptop_body(stale)), 2 * S)
            self.assertEqual(str(caught.exception), "replay")
        rx.offer(signed(laptop_body(6)), 2 * S)
        rx.offer(signed(laptop_body(1, boot_id="1111111111111111")), 3 * S)
        self.assertEqual(rx.last.body["boot_id"], "1111111111111111")

    def test_a_retired_boot_never_comes_back_even_with_a_higher_seq(self):
        rx = wire.Receiver(KEY, "laptop")
        rx.offer(signed(laptop_body(9)), 1 * S)
        rx.offer(signed(laptop_body(1, boot_id="1111111111111111")), 2 * S)
        with self.assertRaises(wire.Drop) as caught:
            rx.offer(signed(laptop_body(1000)), 3 * S)
        self.assertEqual(str(caught.exception), "retired_boot")
        self.assertEqual(rx.last.rx_mono_ns, 2 * S)

    def test_freshness_is_the_receivers_clock_whatever_the_sender_claims(self):
        rx = wire.Receiver(KEY, "laptop")
        self.assertEqual(rx.freshness(0), "never")
        # A laptop clock decades ahead: its stamps are carried and never used.
        rx.offer(signed(laptop_body(1, mono_ns=10**19, wall_utc="2099-01-01T00:00:00Z")), 100 * S)
        self.assertEqual(rx.freshness(100 * S + 1 * S), "fresh")
        self.assertEqual(rx.freshness(100 * S + 1 * S + 1), "stale")
        self.assertEqual(rx.freshness(100 * S + 3 * S), "stale")
        self.assertEqual(rx.freshness(100 * S + 3 * S + 1), "lost")

    def test_a_replayed_frame_does_not_refresh_a_lost_peer(self):
        rx = wire.Receiver(KEY, "laptop")
        frame = signed(laptop_body(1))
        rx.offer(frame, 1 * S)
        with self.assertRaises(wire.Drop):
            rx.offer(frame, 9 * S)
        self.assertEqual(rx.freshness(9 * S), "lost")

    def test_a_short_or_long_key_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "link.key"
            for size in (0, 31, 33):
                path.write_bytes(bytes(size))
                with self.assertRaises(ValueError):
                    wire.load_key(path)
            path.write_bytes(KEY)
            self.assertEqual(wire.load_key(path), KEY)


class Notices(unittest.TestCase):
    def test_a_key_raises_once_and_repeats_only_count(self):
        book = notices.NoticeBook()
        self.assertEqual(len(book.update({"k": ("warn", "a")}, 0)), 1)
        for t in range(1, 20):
            self.assertEqual(book.update({"k": ("warn", "a")}, t * S), [])
        self.assertEqual(book.snapshot(20 * S)[0]["count"], 1)
        book.update({}, 21 * S)
        self.assertEqual(book.update({"k": ("warn", "a")}, 22 * S), [])  # back before 30 s clear
        notice = book.snapshot(22 * S)[0]
        self.assertEqual((notice["count"], notice["state"], notice["notified_mono_ns"]), (2, "active", 0))

    def test_recovered_comes_once_after_30_s_clear_then_the_key_closes(self):
        book = notices.NoticeBook()
        book.update({"k": ("critical", "lost")}, 0)
        book.update({}, 10 * S)
        self.assertEqual(book.update({}, 10 * S + 30 * S - 1), [])
        self.assertEqual(book.update({}, 40 * S), ["recovered k"])
        self.assertEqual(book.update({}, 41 * S), [])
        self.assertEqual(book.snapshot(41 * S)[0]["recovered_mono_ns"], 40 * S)
        self.assertEqual(book.snapshot(41 * S)[0]["state"], "recovered")

    def test_a_flap_inside_10_min_folds_and_recovers_silently(self):
        book = notices.NoticeBook()
        book.update({"k": ("warn", "a")}, 0)
        book.update({}, 1 * S)
        book.update({}, 31 * S)
        self.assertEqual(book.update({"k": ("warn", "b")}, 120 * S), [])
        notice = book.snapshot(120 * S)[0]
        self.assertEqual((notice["first_mono_ns"], notice["count"], notice["state"]), (0, 2, "active"))
        book.update({}, 121 * S)
        self.assertEqual(book.update({}, 151 * S), [])
        self.assertIsNone(book.snapshot(151 * S)[0]["recovered_mono_ns"])

    def test_after_10_min_a_new_onset_notifies_again(self):
        book = notices.NoticeBook()
        book.update({"k": ("warn", "a")}, 0)
        book.update({}, 1 * S)
        book.update({}, 31 * S)
        self.assertEqual(len(book.update({"k": ("warn", "a")}, 600 * S)), 1)
        self.assertEqual(book.snapshot(600 * S)[0]["first_mono_ns"], 600 * S)

    def test_an_event_raises_counts_and_recovers_like_a_level(self):
        book = notices.NoticeBook()
        self.assertEqual(len(book.event("disengage_unexpected", "warn", "x", 0)), 1)
        self.assertEqual(book.event("disengage_unexpected", "warn", "y", 5 * S), [])
        self.assertEqual(book.update({}, 34 * S), [])
        self.assertEqual(book.update({}, 35 * S), ["recovered disengage_unexpected"])
        self.assertEqual(book.snapshot(35 * S)[0]["count"], 2)

    def test_export_is_worst_first_clipped_and_drops_old_recovered(self):
        book = notices.NoticeBook()
        book.update({"a": ("info", "i"), "b": ("critical", "c" * 200)}, 0)
        snap = book.snapshot(0)
        self.assertEqual([n["key"] for n in snap], ["b", "a"])
        self.assertEqual(len(snap[0]["text"]), notices.TEXT_MAX)
        book.update({}, 1 * S)
        book.update({}, 31 * S)
        self.assertEqual(len(book.snapshot(90 * S)), 2)
        self.assertEqual(book.snapshot(91 * S), [])
        fresh = notices.NoticeBook()
        fresh.update({"c": ("warn", "t")}, 0)
        self.assertEqual(fresh.heartbeat(0), [{"key": "c", "severity": "warn", "text": "t", "state": "active"}])

    def test_an_escalation_notifies_once_and_further_ones_fold(self):
        book = notices.NoticeBook()
        book.update({"k": ("info", "a")}, 0)
        self.assertEqual(book.update({"k": ("warn", "b")}, 5 * S), ["escalated k warn: b"])
        self.assertEqual(book.snapshot(5 * S)[0]["notified_mono_ns"], 5 * S)
        self.assertEqual(book.update({"k": ("critical", "c")}, 60 * S), [])
        notice = book.snapshot(60 * S)[0]
        self.assertEqual((notice["severity"], notice["notified_mono_ns"], notice["count"]), ("critical", 5 * S, 1))
        self.assertEqual(book.update({"k": ("warn", "d")}, 61 * S), [])
        self.assertEqual(book.snapshot(61 * S)[0]["severity"], "critical")

    def test_a_folded_episode_that_escalates_is_shown_and_recovers_loudly(self):
        book = notices.NoticeBook()
        book.update({"k": ("warn", "a")}, 0)
        book.update({}, 1 * S)
        book.update({}, 31 * S)
        self.assertEqual(book.update({"k": ("warn", "a")}, 60 * S), [])
        self.assertEqual(book.update({"k": ("critical", "b")}, 61 * S), ["escalated k critical: b"])
        book.update({}, 62 * S)
        self.assertEqual(book.update({}, 92 * S), ["recovered k"])

    def test_mirrored_laptop_keys_are_never_sent_back(self):
        book = notices.NoticeBook()
        book.update({"laptop.gpu_hot": ("warn", "laptop: GPU"), "laptop_lost": ("critical", "x")}, 0)
        self.assertEqual([n["key"] for n in book.heartbeat(0)], ["laptop_lost"])
        self.assertEqual(len(book.snapshot(0)), 2)


class Health(unittest.TestCase):
    def setUp(self):
        self.rx = wire.Receiver(KEY, "laptop")
        self.eval = health.HealthEvaluator(0)
        self.status = device_status.UNKNOWN

    def frame(self, seq: int, t: int, **changes) -> None:
        self.rx.offer(signed(laptop_body(seq, **changes)), t)

    def assess(self, t: int, onroad=True, carrier=True, status=None):
        return self.eval.assess(self.rx, t, onroad, carrier, status or self.status)

    def test_lost_while_onroad_names_the_cable_and_offroad_idle_stays_quiet(self):
        self.assertNotIn("laptop_lost", self.assess(3 * S).present)
        lost = self.assess(3 * S + 1, carrier=False).present["laptop_lost"]
        self.assertEqual(lost[0], "critical")
        self.assertIn("unplugged", lost[1])
        self.assertNotIn("laptop_lost", self.assess(10 * S, onroad=False).present)
        self.assertNotIn("laptop_lost", self.assess(10 * S, onroad=None).present)

    def test_lost_offroad_counts_when_a_session_was_up_or_the_laptop_was_running(self):
        up = device_status.read_status_from(status_doc(boot_ns=9 * S), 10 * S)
        self.assertIn("laptop_lost", self.assess(10 * S, onroad=False, status=up).present)
        self.frame(1, 20 * S)
        self.assertIn("laptop_lost", self.assess(24 * S, onroad=False).present)
        rx2, ev2 = wire.Receiver(KEY, "laptop"), health.HealthEvaluator(0)
        rx2.offer(signed(laptop_body(1, state="sleeping_soon", supervisor_ok=False)), 20 * S)
        ev2.assess(rx2, 20 * S, False, True, device_status.UNKNOWN)
        self.assertNotIn("laptop_lost", ev2.assess(rx2, 30 * S, False, True, device_status.UNKNOWN).present)

    def test_unhealthy_must_hold_5_s_and_starting_is_not_unhealthy(self):
        self.frame(1, 0, state="starting", supervisor_ok=False)
        self.assertNotIn("laptop_unhealthy", self.assess(0).present)
        for i, t in enumerate(range(1, 8)):
            self.frame(2 + i, t * S, state="running", supervisor_ok=False, reason="live_missing")
            present = self.assess(t * S).present
            self.assertEqual("laptop_unhealthy" in present, t >= 6, t)
        self.assertIn("live_missing", present["laptop_unhealthy"][1])
        self.frame(20, 8 * S, state="running", supervisor_ok=True)
        self.assertNotIn("laptop_unhealthy", self.assess(8 * S).present)

    def test_a_fault_state_is_critical(self):
        for t in range(7):
            self.frame(t + 1, t * S, state="fault", supervisor_ok=True)
            present = self.assess(t * S).present
        self.assertEqual(present["laptop_unhealthy"][0], "critical")

    def test_cycle_over_budget_and_display_down_and_mirrored_notices(self):
        mirrored = [{"key": "gpu_hot", "severity": "warn", "text": "GPU 92 C", "state": "active"},
                    {"key": "old", "severity": "warn", "text": "x", "state": "recovered"},
                    {"key": 5, "severity": "warn", "text": "bad"}, "garbage"]
        for t in range(7):
            self.frame(t + 1, t * S, cycle={"p50_ms": 40.0, "p99_ms": 61.0, "budget_ms": 50.0, "misses_10s": 3},
                       children={"live": {"alive": True}, "display": {"alive": False}}, notices=mirrored)
            present = self.assess(t * S).present
        self.assertIn("61 ms over 50 ms", present["laptop_cycle_slow"][1])
        self.assertIn("display_down", present)
        self.assertEqual(present["laptop.gpu_hot"], ("warn", "laptop: GPU 92 C"))
        self.assertEqual(sorted(k for k in present if k.startswith("laptop.")), ["laptop.gpu_hot"])

    def test_laptop_not_running_for_30_s_while_onroad(self):
        for t in range(32):
            self.frame(t + 1, t * S, state="waiting", supervisor_ok=False)
            present = self.assess(t * S).present
        self.assertIn("waiting while onroad", present["laptop_unhealthy"][1])

    def test_losing_the_laptop_never_recovers_what_it_last_reported(self):
        # The verifier's sequence: unhealthy raised at 5 s, frames stop at 10 s (lost at 13 s), and
        # at 43 s the old code issued "recovered" while the laptop was still silent.
        book = notices.NoticeBook()
        issued = []
        for half in range(120):
            t = half * S // 2
            if t <= 10 * S:
                self.frame(half + 1, t, state="running", supervisor_ok=False, reason="live_missing",
                           notices=[{"key": "gpu_hot", "severity": "warn", "text": "GPU 92 C", "state": "active"}])
            present = self.assess(t).present
            issued += book.update(present, t)
            if t >= 13 * S + S // 2:
                self.assertIn("laptop_unhealthy", present, t)
                self.assertIn("laptop.gpu_hot", present, t)
        self.assertEqual([line for line in issued if line.startswith("recovered")], [])
        self.assertEqual(sorted(line.split()[1] for line in issued), ["laptop.gpu_hot", "laptop_lost", "laptop_unhealthy"])
        # The laptop comes back healthy: now, and only now, do they recover.
        for half in range(120, 220):
            t = half * S // 2
            self.frame(half + 1, t, state="running", supervisor_ok=True)
            issued += book.update(self.assess(t).present, t)
        self.assertEqual(sorted(line for line in issued if line.startswith("recovered")),
                         ["recovered laptop.gpu_hot", "recovered laptop_lost", "recovered laptop_unhealthy"])

    def test_a_laptop_that_comes_back_stuck_starting_onroad_is_unhealthy(self):
        # The verifier's sequence: running, 10 s silent, then 4 min of `starting` while onroad.
        book, issued = notices.NoticeBook(), []
        seq = 0
        for half in range(520):
            t = half * S // 2
            if t <= 10 * S or t >= 20 * S:
                seq += 1
                self.frame(seq, t, state="running" if t <= 10 * S else "starting", supervisor_ok=t <= 10 * S)
            present = self.assess(t).present
            issued += book.update(present, t)
            if t == 50 * S + S // 2:
                self.assertIn("starting while onroad", present["laptop_unhealthy"][1])
            if t == 81 * S:
                self.assertEqual(present["laptop_unhealthy"], ("warn", "laptop stuck starting"))
        self.assertIn("laptop_unhealthy", present)
        self.assertIn("raised laptop_lost critical", " ".join(issued))
        self.assertTrue(any(line.startswith("raised laptop_unhealthy") for line in issued))
        self.assertNotIn("recovered laptop_unhealthy", issued)

    def test_stuck_starting_counts_offroad_too_after_60_s(self):
        for t in range(62):
            self.frame(t + 1, t * S, state="starting", supervisor_ok=False)
            present = self.assess(t * S, onroad=False).present
            if t < 60:
                self.assertNotIn("laptop_unhealthy", present, t)
        self.assertEqual(present["laptop_unhealthy"], ("warn", "laptop stuck starting"))

    def test_lost_holds_while_onroad_is_unknown(self):
        present = self.assess(4 * S).present
        self.assertIn("laptop_lost", present)
        for t in range(5, 60):
            self.assertIn("laptop_lost", self.assess(t * S, onroad=None).present, t)
        self.assertNotIn("laptop_lost", self.assess(61 * S, onroad=False).present)

    def test_view_is_ok_only_for_a_fresh_running_in_budget_frame(self):
        self.assertEqual(self.assess(0).view["reason"], "no_frame_yet")
        self.frame(1, 0)
        self.assertEqual(self.assess(S // 2).view, {"last_rx_age_ms": 500, "ok": True, "reason": "ok"})
        self.assertEqual(self.assess(2 * S).view["reason"], "stale")
        self.frame(2, 3 * S, cycle={"p99_ms": None, "budget_ms": 50.0})
        self.assertEqual(self.assess(3 * S).view["reason"], "cycle_unknown")
        self.frame(3, 4 * S, state="degraded")
        self.assertFalse(self.assess(4 * S).view["ok"])


class Status(unittest.TestCase):
    def test_the_writers_fields_are_read_by_their_exact_names(self):
        reason = "session_fault;publisher_failures=2:restarts=1:crash"
        status = device_status.read_status_from(status_doc(boot_ns=9 * S, authority="latched_native", reason=reason), 10 * S)
        self.assertEqual((status.kind, status.session_up, status.toggle_on, status.authority, status.problem),
                         ("fresh", True, True, "latched_native", None))
        self.assertEqual(status.reason, reason)

    def test_missing_garbage_stale_or_future_is_unknown_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            missing = device_status.read_status(path, 10 * S)
            self.assertEqual((missing.kind, missing.problem), ("missing", None))
            for garbage in (b"", b"{", b"[]", bytes([0xff])):
                path.write_bytes(garbage)
                status = device_status.read_status(path, 10 * S)
                self.assertEqual(status.kind, "garbage", garbage)
                self.assertIsNotNone(status.problem)
        stale = device_status.read_status_from(status_doc(boot_ns=4 * S), 10 * S)
        self.assertEqual((stale.kind, stale.session_up), ("stale", None))
        self.assertIn("stale", stale.problem)
        self.assertEqual(device_status.read_status_from(status_doc(boot_ns=11 * S), 10 * S).kind, "stale")
        self.assertEqual(device_status.read_status_from(status_doc(boot_ns="9"), 10 * S).kind, "garbage")
        partial = status_doc(boot_ns=9 * S)
        del partial["toggle"], partial["drive_log_error"]
        status = device_status.read_status_from(partial, 10 * S)
        self.assertEqual((status.kind, status.toggle_on), ("fresh", None))
        self.assertEqual(status.problem, "missing keys: drive_log_error, toggle")
        odd = device_status.read_status_from(status_doc(boot_ns=9 * S, laptop_session="maybe"), 10 * S)
        self.assertIsNone(odd.session_up)

    def run_status(self, docs, onroad=True):
        evaluator, book, issued, present = device_status.StatusConditions(), notices.NoticeBook(), [], {}
        for t, doc in docs:
            if doc is None:
                status = device_status.UNKNOWN
            elif isinstance(doc, device_status.DeviceStatus):
                status = doc
            else:
                status = device_status.read_status_from(doc, t)
            present = evaluator.assess(status, t, onroad)
            issued += book.update(present, t)
        return present, issued

    def test_latched_and_drive_log_error_raise_warnings_with_the_cause(self):
        doc = status_doc(boot_ns=10 * S, authority="latched_native", reason="regressed_clock", drive_log_error="ENOSPC")
        present, _ = self.run_status([(10 * S, doc)])
        self.assertEqual(present["companion_latched"], ("warn", "companion latched to native: regressed_clock"))
        self.assertEqual(present["drive_log_error"], ("warn", "companion drive log off: ENOSPC"))

    def test_a_publisher_restart_storm_is_three_in_five_minutes(self):
        docs = [(t * S, status_doc(boot_ns=t * S, publisher_restarts=n)) for t, n in ((0, 0), (60, 1), (120, 2), (200, 3))]
        present, _ = self.run_status(docs)
        self.assertIn("3 restarts", present["publisher_storm"][1])
        docs = [(t * S, status_doc(boot_ns=t * S, publisher_restarts=n)) for t, n in ((0, 0), (60, 1), (200, 2), (400, 3))]
        self.assertNotIn("publisher_storm", self.run_status(docs)[0])

    def test_session_down_onroad_with_the_toggle_on_is_critical_after_10_s(self):
        docs = [(t * S, status_doc(boot_ns=t * S, laptop_session="down")) for t in range(12)]
        present, _ = self.run_status(docs)
        self.assertEqual(present["companion_session_down"][0], "critical")
        self.assertNotIn("companion_session_down", self.run_status(docs[:10])[0])
        off = [(t * S, status_doc(boot_ns=t * S, laptop_session="down", toggle="off")) for t in range(12)]
        self.assertNotIn("companion_session_down", self.run_status(off)[0])
        self.assertNotIn("companion_session_down", self.run_status(docs, onroad=False)[0])

    def test_a_stale_file_onroad_is_publisher_down_and_unknown_never_recovers(self):
        docs = [(t * S, status_doc(boot_ns=t * S, authority="latched_native", reason="x")) for t in range(3)]
        stale = docs + [(t * S, status_doc(boot_ns=2 * S, authority="latched_native", reason="x")) for t in range(3, 60)]
        present, issued = self.run_status(stale)
        self.assertIn("companion_latched", present)
        self.assertIn("publisher_down", present)
        self.assertEqual([line for line in issued if line.startswith("recovered")], [])
        after_stale = stale + [(t * S, None) for t in range(60, 120)]
        after_stale += [(t * S, device_status.DeviceStatus("garbage", problem="not JSON")) for t in range(120, 180)]
        present, issued = self.run_status(after_stale)
        self.assertIn("publisher_down", present)
        self.assertEqual([line for line in issued if line.startswith("recovered")], [])
        missing = docs + [(t * S, None) for t in range(3, 60)]
        present, issued = self.run_status(missing)
        self.assertIn("companion_latched", present)
        self.assertNotIn("publisher_down", present)
        self.assertEqual([line for line in issued if line.startswith("recovered")], [])


def cereal_log():
    from openpilot.cereal import log
    return log


def event(service: str, mono_ns: int, **fields):
    log = cereal_log()
    message = log.Event.new_message(logMonoTime=mono_ns)
    body = message.init(service)
    for name, value in fields.items():
        if name == "buttonEvents":
            buttons = body.init("buttonEvents", len(value))
            for slot, (kind, pressed) in zip(buttons, value, strict=True):
                slot.type, slot.pressed = kind, pressed
        else:
            setattr(body, name, value)
    return log.Event.from_bytes(message.to_bytes()).__enter__()


class OpenpilotDecode(unittest.TestCase):
    def test_selfdrive_and_car_samples_come_from_this_forks_field_names(self):
        sd = op_state.selfdrive_sample(event("selfdriveState", 7, enabled=True, alertType="steerSaturated/warning",
                                             alertText1="Take Control"))
        self.assertEqual(sd, op_state.SelfdriveSample(7, True, "steerSaturated/warning", "Take Control"))
        car = op_state.car_sample(event("carState", 8, brakePressed=True, buttonEvents=[("cancel", True), ("setCruise", True)]))
        self.assertEqual(car, op_state.CarSample(8, "brake,cancel"))
        self.assertEqual(op_state.car_sample(event("carState", 9, buttonEvents=[("accelCruise", True)])).cause, "")
        self.assertEqual(op_state.car_sample(event("carState", 9, gasPressed=True)).cause, "gas")
        self.assertEqual(op_state.car_sample(event("carState", 9, buttonEvents=[("mainCruise", False)])).cause, "mainCruise")

    def test_onroad_reads_is_offroad_and_anything_else_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp)
            self.assertIsNone(op_state.read_onroad(params))
            for raw, expected in ((b"0", True), (b"1", False), (b"", None), (b"yes", None)):
                (params / "IsOffroad").write_bytes(raw)
                self.assertEqual(op_state.read_onroad(params), expected)


MS = 1_000_000


def drive(detector, start_ms: int, end_ms: int, drop_ms: int | None, causes: dict[int, str] | None = None,
          car_gap: tuple[int, int] | None = None, alert: tuple[int, str] | None = None):
    causes = causes or {}
    sd, car = [], []
    for t in range(start_ms, end_ms, 10):
        enabled = drop_ms is None or t < drop_ms
        alert_type = alert[1] if alert is not None and t >= alert[0] else ""
        sd.append(op_state.SelfdriveSample(t * MS, enabled, alert_type, ""))
        if car_gap is None or not car_gap[0] <= t < car_gap[1]:
            car.append(op_state.CarSample(t * MS, causes.get(t, "")))
    detector.feed(sd, car)


class Disengagement(unittest.TestCase):
    def test_a_drop_with_no_driver_cause_is_reported_with_the_alert(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 3000, 2000, alert=(2000, "steerTempUnavailable/softDisable"))
        found = d.decide(3000 * MS, True)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].mono_ns, found[0].verified), (2000 * MS, True))
        self.assertEqual(found[0].alert, "steerTempUnavailable/softDisable")
        self.assertEqual(d.decide(3500 * MS, True), [])

    def test_brake_gas_cancel_or_main_inside_half_a_second_explains_it(self):
        for cause, at in (("brake", 1990), ("gas", 1600), ("cancel", 1510), ("mainCruise", 2050)):
            d = op_state.DisengageDetector()
            drive(d, 0, 3000, 2000, causes={at: cause})
            self.assertEqual(d.decide(3000 * MS, True), [], cause)

    def test_a_cause_older_than_half_a_second_does_not(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 3000, 2000, causes={1490: "brake"})
        self.assertEqual(len(d.decide(3000 * MS, True)), 1)

    def test_missing_carstate_frames_make_it_unverified(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 3000, 2000, car_gap=(1700, 1900))
        self.assertFalse(d.decide(3000 * MS, True)[0].verified)

    def test_a_drop_waits_for_carstate_to_cover_it_and_offroad_is_ignored(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 2050, 2000)
        self.assertEqual(d.decide(2050 * MS, True), [])  # carState only to 2040 ms
        drive(d, 2050, 2200, 2000)
        self.assertEqual(len(d.decide(2200 * MS, True)), 1)
        d2 = op_state.DisengageDetector()
        drive(d2, 0, 3000, 2000)
        self.assertEqual(d2.decide(3000 * MS, False), [])

    def test_a_reset_forgets_engagement_instead_of_inferring_a_drop(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 1000, None)
        d.reset()
        self.assertIsNone(d.engaged)
        drive(d, 5000, 6000, 4000)
        self.assertEqual(d.decide(6000 * MS, True), [])

    def test_a_selfdrive_gap_over_a_second_is_never_a_drop(self):
        d = op_state.DisengageDetector()
        drive(d, 0, 1000, None)
        drive(d, 2500, 4000, 0)
        self.assertEqual(d.decide(4000 * MS, True), [])
        self.assertIs(d.engaged, False)

    def test_the_msgq_open_budget_is_per_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Path(tmp) / "msgq_opens.json"
            self.assertEqual([op_state.claim_open(budget, "boot-a") for _ in range(6)], [True] * 4 + [False] * 2)
            self.assertTrue(op_state.claim_open(budget, "boot-b"))
            budget.write_text("garbage")
            self.assertTrue(op_state.claim_open(budget, "boot-b"))


if __name__ == "__main__":
    unittest.main()
