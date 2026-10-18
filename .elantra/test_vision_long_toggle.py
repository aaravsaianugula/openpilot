#!/usr/bin/env python3
"""
Tests for the Vision Long braking toggle's file helpers (comma 4 / mici settings).

The on-screen toggle and the companion share one state file, /data/vision_long/braking:
exactly the byte "1" is on, absent or anything else is off. The UI side lives in
port_manifest.py next to the port panel's other pure logic, so it can be imported here by
path without pyray. These tests exercise the shipped module, not a copy of it.

    python .elantra/test_vision_long_toggle.py
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

MODULE = (Path(__file__).resolve().parent.parent
          / "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_manifest.py")


def load_module():
    spec = importlib.util.spec_from_file_location("port_manifest", MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = load_module()


class BrakingToggleFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.state = self.root / "vision_long" / "braking"

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_contract_path_is_the_shared_state_file(self):
        self.assertEqual(pm.VISION_LONG_BRAKING_FILE, "/data/vision_long/braking")

    def test_absent_file_reads_as_off(self):
        self.assertFalse(pm.braking_enabled(self.state))

    def test_exactly_one_reads_as_on(self):
        self.state.parent.mkdir(parents=True)
        self.state.write_bytes(b"1")
        self.assertTrue(pm.braking_enabled(self.state))

    def test_anything_else_reads_as_off(self):
        self.state.parent.mkdir(parents=True)
        for content in (b"", b"0", b"1\n", b" 1", b"11", b"on", b"true", b"\x00", b"1" * 4096, b"\xff\xfe"):
            with self.subTest(content=content[:8]):
                self.state.write_bytes(content)
                self.assertFalse(pm.braking_enabled(self.state))

    def test_a_directory_in_place_of_the_file_reads_as_off(self):
        self.state.mkdir(parents=True)
        self.assertFalse(pm.braking_enabled(self.state))

    def test_write_creates_the_folder_and_round_trips(self):
        pm.set_braking(True, self.state)
        self.assertEqual(self.state.read_bytes(), b"1")
        self.assertTrue(pm.braking_enabled(self.state))
        pm.set_braking(False, self.state)
        self.assertEqual(self.state.read_bytes(), b"0")
        self.assertFalse(pm.braking_enabled(self.state))

    def test_write_replaces_garbage_and_leaves_no_temporary_file(self):
        self.state.parent.mkdir(parents=True)
        self.state.write_bytes(b"garbage")
        pm.set_braking(True, self.state)
        self.assertEqual(sorted(os.listdir(self.state.parent)), ["braking"])

    def test_write_is_a_rename_never_an_in_place_rewrite(self):
        # A reader must see complete old bytes or complete new bytes: the new state arrives as
        # a different file, never as a truncate-then-write of the same one.
        self.state.parent.mkdir(parents=True)
        self.state.write_bytes(b"0")
        before = self.state.stat().st_ino
        pm.set_braking(True, self.state)
        self.assertNotEqual(self.state.stat().st_ino, before)
        self.assertEqual(self.state.read_bytes(), b"1")

    def test_a_failed_write_raises_and_keeps_the_previous_state(self):
        self.state.parent.mkdir(parents=True)
        self.state.write_bytes(b"1")
        blocker = self.root / "not_a_dir"
        blocker.write_bytes(b"x")
        with self.assertRaises(OSError):
            pm.set_braking(False, blocker / "braking")
        self.assertTrue(pm.braking_enabled(self.state))

    def test_the_summary_says_what_each_state_does(self):
        self.assertIn("watch", pm.braking_summary(False))
        self.assertIn("brake", pm.braking_summary(True))
        self.assertNotEqual(pm.braking_summary(True), pm.braking_summary(False))


if __name__ == "__main__":
    unittest.main()
