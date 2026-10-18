#!/usr/bin/env python3
"""
Tests for the onroad Vision Long banner's decisions
(openpilot/selfdrive/ui/sunnypilot/mici/onroad/vision_long_notice.py).

The widget itself (vision_long_banner.py) needs pyray and a running UI, which this desktop does
not have, so it is not rendered here. What it shows is decided by the pure module, loaded here by
path so the shipped code is what is tested: the worst active notice for 10 s after it was
notified, "recovered" for 3 s, nothing for a missing, garbage or stale file. The monitor's own
NoticeBook produces the documents in the round-trip test, so the two ends cannot drift apart.
An import check pins that neither banner file reaches a control path.

    python .elantra/test_vision_long_notice_banner.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONROAD = HERE.parent / "openpilot/selfdrive/ui/sunnypilot/mici/onroad"
spec = importlib.util.spec_from_file_location("vision_long_notice", ONROAD / "vision_long_notice.py")
vln = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vln
spec.loader.exec_module(vln)
sys.path.insert(0, str(HERE / "vision_long_supervisor"))
import notices

S = 1_000_000_000


def document(items: list[dict], written: int = 100 * S) -> dict:
    return {"schema": "vl_notices_v1", "written_mono_ns": written, "notices": items}


def item(key="laptop_lost", severity="critical", text="laptop silent", state="active", notified=100 * S, recovered=None):
    return {"key": key, "severity": severity, "text": text, "state": state, "first_mono_ns": notified, "count": 1,
            "notified_mono_ns": notified, "recovered_mono_ns": recovered}


class Choose(unittest.TestCase):
    def test_nothing_for_garbage_wrong_schema_or_a_stale_file(self):
        for doc in (None, [], "x", {}, {"schema": "vl_notices_v2", "written_mono_ns": 100 * S, "notices": [item()]},
                    {"schema": "vl_notices_v1", "written_mono_ns": "100", "notices": [item()]},
                    {"schema": "vl_notices_v1", "written_mono_ns": 100 * S, "notices": {"a": 1}}):
            self.assertIsNone(vln.choose(doc, 100 * S), doc)
        self.assertIsNone(vln.choose(document([item()]), 106 * S + 1))
        self.assertIsNone(vln.choose(document([item()], written=102 * S), 100 * S))  # from the future

    def test_an_active_notice_shows_for_10_s_after_it_was_notified(self):
        doc = document([item(notified=95 * S)])
        self.assertEqual(vln.choose(doc, 100 * S), vln.Banner("laptop silent", "critical", False))
        doc = document([item(notified=90 * S - 1)])
        self.assertIsNone(vln.choose(doc, 100 * S))

    def test_the_worst_wins_then_the_newest(self):
        doc = document([item("a", "warn", "w", notified=99 * S), item("b", "critical", "c", notified=95 * S),
                        item("c", "critical", "c2", notified=96 * S), item("d", "info", "i", notified=99 * S)])
        self.assertEqual(vln.choose(doc, 100 * S).text, "c2")

    def test_recovered_shows_for_3_s_and_an_active_notice_beats_it(self):
        back = item(state="recovered", notified=50 * S, recovered=98 * S)
        self.assertEqual(vln.choose(document([back]), 100 * S), vln.Banner("recovered: laptop silent", "info", True))
        self.assertIsNone(vln.choose(document([back], written=101 * S + 1), 101 * S + 1))
        self.assertEqual(vln.choose(document([back, item("x", "warn", "w", notified=99 * S)]), 100 * S).text, "w")

    def test_malformed_entries_are_skipped_not_fatal(self):
        doc = document(["x", {"severity": "loud", "text": "t"}, {"severity": "warn", "text": 5}, item(text="  ")])
        self.assertIsNone(vln.choose(doc, 100 * S))


class Visible(unittest.TestCase):
    def test_nothing_is_drawn_while_openpilot_shows_an_alert(self):
        banner = vln.Banner("t", "critical", False)
        self.assertIs(vln.visible(banner, True), banner)
        self.assertIsNone(vln.visible(banner, False))
        self.assertIsNone(vln.visible(None, True))

    def test_the_widget_paints_with_the_huds_can_draw(self):
        # The widget's render is one guarded call; the HUD's can_draw has to reach it.
        tree = ast.parse((ONROAD / "vision_long_banner.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "paint"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].args[3], "id", None), "can_draw")

    def test_nothing_is_painted_while_openpilot_shows_an_alert(self):
        drawn = []
        painter = vln.BannerPainter(FixedReader(vln.Banner("t", "critical", False)), lambda t: (1.0, 1.0),
                                    lambda *a: drawn.append(a), lambda m: None)
        self.assertFalse(painter.paint(0, 0, 536, False))
        self.assertEqual(drawn, [])


class FixedReader:
    """The reader interface (current()) with a fixed answer, or an exception to raise."""

    def __init__(self, banner=None, error=None):
        self.banner, self.error = banner, error

    def current(self):
        if self.error is not None:
            raise self.error
        return self.banner


class Painting(unittest.TestCase):
    def setUp(self):
        self.measured, self.drawn, self.reports = [], [], []

    def measure(self, text):
        self.measured.append(text)
        return 10.0 * len(text), 26.0

    def painter(self, reader, measure=None, draw=None):
        return vln.BannerPainter(reader, measure or self.measure, draw or (lambda *args: self.drawn.append(args)),
                                 self.reports.append)

    def test_a_notice_is_drawn_top_right_inside_the_width(self):
        painter = self.painter(FixedReader(vln.Banner("laptop silent", "critical", False)))
        self.assertTrue(painter.paint(0.0, 0.0, 536.0, True))
        x, y, w, h, severity, text = self.drawn[0][:6]
        self.assertEqual((severity, text), ("critical", "laptop silent"))
        self.assertLessEqual(x + w, 536.0)
        self.assertLessEqual(w, 536.0 * vln.MAX_WIDTH_FRACTION)

    def test_an_unknown_severity_is_never_drawn(self):
        self.assertIsNone(vln.visible(vln.Banner("t", "loud", False), True))
        painter = self.painter(FixedReader(vln.Banner("t", "loud", False)))
        self.assertFalse(painter.paint(0.0, 0.0, 536.0, True))
        self.assertEqual(self.drawn, [])

    def test_a_raising_measure_hides_the_banner_and_reports_once_per_type(self):
        def broken(text):
            raise ValueError("font atlas gone")
        painter = self.painter(FixedReader(vln.Banner("t", "warn", False)), measure=broken)
        for _ in range(10):
            self.assertFalse(painter.paint(0.0, 0.0, 536.0, True))
        self.assertEqual(self.drawn, [])
        self.assertEqual(len(self.reports), 1)
        self.assertIn("ValueError", self.reports[0])

    def test_a_raising_draw_or_reader_never_escapes(self):
        def broken_draw(*args):
            raise KeyError("severity")
        self.assertFalse(self.painter(FixedReader(vln.Banner("t", "warn", False)), draw=broken_draw).paint(0, 0, 536, True))
        self.assertFalse(self.painter(FixedReader(error=RuntimeError("disk"))).paint(0, 0, 536, True))
        self.assertEqual(sorted(r.split(":")[1].strip() for r in self.reports), ["KeyError", "RuntimeError"])

    def test_a_steady_notice_is_measured_once(self):
        painter = self.painter(FixedReader(vln.Banner("laptop cycle p99 61 ms over 50 ms budget", "warn", False)))
        painter.paint(0.0, 0.0, 300.0, True)
        first = len(self.measured)
        self.assertGreater(first, 1)  # it had to be shortened
        for _ in range(100):
            painter.paint(0.0, 0.0, 300.0, True)
        self.assertEqual(len(self.measured), first)
        self.assertTrue(self.drawn[-1][5].endswith("..."))

    def test_a_garbage_file_hides_the_banner_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notice.json"
            for garbage in (b"[" * 60000, b"\x00\xff", b'{"schema": "vl_notices_v1", "written_mono_ns": 1, "notices": [[[]]]}'):
                path.write_bytes(garbage)
                reader = vln.NoticeReader(path, clock=lambda: 1)
                self.assertIsNone(reader.current(), garbage[:10])

    def test_the_file_is_read_at_most_once_a_second_whatever_the_frame_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notice.json"
            path.write_text(json.dumps(document([item()])))
            now = [100 * S]
            reads = []
            real = vln.read_document
            vln.read_document = lambda p: reads.append(p) or real(p)
            try:
                reader = vln.NoticeReader(path, clock=lambda: now[0])
                for _ in range(120):  # two seconds at 60 fps
                    reader.current()
                    now[0] += S // 60
            finally:
                vln.read_document = real
            self.assertLessEqual(len(reads), 3)


class Staleness(unittest.TestCase):
    def test_the_monitor_keeps_the_file_fresher_than_the_banner_needs(self):
        import monitor
        self.assertEqual(monitor.BANNER_STALE_NS, vln.STALE_NS)


class RoundTrip(unittest.TestCase):
    def test_the_monitors_notices_show_once_and_a_fold_does_not_show_again(self):
        book = notices.NoticeBook()
        book.update({"laptop_lost": ("critical", "laptop silent 4 s: USB cable unplugged or laptop off")}, 100 * S)
        doc = document(book.snapshot(101 * S), written=101 * S)
        self.assertEqual(vln.choose(doc, 101 * S).severity, "critical")
        self.assertIsNone(vln.choose(document(book.snapshot(111 * S), written=111 * S), 111 * S))
        book.update({}, 112 * S)
        book.update({}, 142 * S)
        self.assertTrue(vln.choose(document(book.snapshot(142 * S), written=142 * S), 143 * S).recovered)
        book.update({"laptop_lost": ("critical", "again")}, 200 * S)  # folds: within 10 min
        self.assertIsNone(vln.choose(document(book.snapshot(200 * S), written=200 * S), 200 * S))


class Reader(unittest.TestCase):
    def test_reads_at_most_once_a_second_and_survives_a_missing_or_huge_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notice.json"
            now = [100 * S]
            reader = vln.NoticeReader(path, clock=lambda: now[0])
            self.assertIsNone(reader.current())
            path.write_text(json.dumps(document([item()])))
            now[0] += S // 2
            self.assertIsNone(reader.current())
            now[0] += S // 2
            self.assertIsNotNone(reader.current())
            path.write_bytes(b"{" + b" " * vln.MAX_FILE_BYTES + b"}")
            now[0] += S
            self.assertIsNone(reader.current())


class NoControlPath(unittest.TestCase):
    def imports(self, name: str) -> set[str]:
        tree = ast.parse((ONROAD / name).read_text())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                found.add(node.module or "")
        return found

    def test_the_decision_module_imports_only_the_standard_library(self):
        self.assertEqual(self.imports("vision_long_notice.py"),
                         {"__future__", "importlib", "json", "time", "dataclasses", "pathlib"})

    def test_the_widget_imports_no_control_path(self):
        for module in self.imports("vision_long_banner.py"):
            # swaglog is logging only (cloudlog for a hidden banner); nothing here reaches a control path.
            allowed = module in ("pyray", "openpilot.common.swaglog")
            self.assertTrue(allowed or module.startswith(("openpilot.system.ui.", "openpilot.selfdrive.ui.")), module)

    def test_the_hud_never_imports_the_widget_directly(self):
        # A failed widget import (say an upstream rename of a pyray helper) must hide the banner,
        # not take the onroad HUD down, so the HUD reaches the widget only through load_banner.
        hud = self.imports("hud_renderer.py")
        self.assertNotIn(vln.BANNER_MODULE, hud)
        self.assertIn("openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_long_notice", hud)
        self.assertEqual(vln.BANNER_MODULE, "openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_long_banner")

    def test_the_hud_loads_the_banner_guarded_and_renders_it_only_when_loaded(self):
        source = (ONROAD / "hud_renderer.py").read_text()
        tree = ast.parse(source)
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_banner"]
        self.assertEqual(len(loads), 1)
        renders = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                   and any(getattr(c.func, "attr", "") == "render" and getattr(c.func.value, "attr", "") == "_vision_long_banner"
                           for c in ast.walk(n) if isinstance(c, ast.Call))]
        self.assertEqual(len(renders), 1)
        self.assertEqual(ast.unparse(renders[0].test), "self._vision_long_banner is not None")


class LoadBanner(unittest.TestCase):
    def setUp(self):
        self.reports: list[str] = []
        self.folder = tempfile.TemporaryDirectory()
        sys.path.insert(0, self.folder.name)

    def tearDown(self):
        sys.path.remove(self.folder.name)
        for name in [n for n in sys.modules if n.startswith("vl_fake_banner_")]:
            del sys.modules[name]
        self.folder.cleanup()

    def module(self, name: str, body: str) -> str:
        (Path(self.folder.name) / f"{name}.py").write_text(body)
        return name

    def test_a_working_widget_is_built(self):
        name = self.module("vl_fake_banner_ok", "class VisionLongBanner:\n    pass\n")
        widget = vln.load_banner(self.reports.append, name)
        self.assertEqual(type(widget).__name__, "VisionLongBanner")
        self.assertEqual(self.reports, [])

    def test_a_missing_widget_module_hides_the_banner_and_reports_it(self):
        self.assertIsNone(vln.load_banner(self.reports.append, "vl_fake_banner_absent"))
        self.assertEqual(len(self.reports), 1)
        self.assertIn("ModuleNotFoundError", self.reports[0])

    def test_a_widget_module_that_fails_to_import_hides_the_banner(self):
        name = self.module("vl_fake_banner_broken", "from os import no_such_helper\n")
        self.assertIsNone(vln.load_banner(self.reports.append, name))
        self.assertIn("ImportError", self.reports[0])

    def test_a_raising_constructor_hides_the_banner(self):
        name = self.module("vl_fake_banner_raises", "class VisionLongBanner:\n    def __init__(self):\n        raise RuntimeError('no font')\n")
        self.assertIsNone(vln.load_banner(self.reports.append, name))
        self.assertIn("RuntimeError: no font", self.reports[0])

    def test_a_failing_logger_still_hides_the_banner(self):
        def report(message: str) -> None:
            raise OSError("log gone")
        self.assertIsNone(vln.load_banner(report, "vl_fake_banner_absent"))


if __name__ == "__main__":
    unittest.main()
