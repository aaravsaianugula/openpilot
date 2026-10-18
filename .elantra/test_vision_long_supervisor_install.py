#!/usr/bin/env python3
"""
Tests for the laptop monitor's unit install, rollback and boot-time re-establish
(.elantra/vision_long_supervisor/persist.py).

The device is a temporary directory standing in for `/` plus a fake that answers sudo, mount,
install, rm and systemctl the way the comma would, with `/` read-only outside a remount window,
and records every command. What these pin: a failed preflight changes nothing; every /etc write
is inside the window and the window always closes; a failed start is undone; the rollback
removes only the monitor's units; `reestablish` re-installs exactly the sets that were installed
before (the monitor's and R4's USB link) when an AGNOS update removed them, and does nothing,
not even a remount, when they are in place.

    python .elantra/test_vision_long_supervisor_install.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vision_long_supervisor"))
import persist

persist.SETTLE_S = 0.0
ALL_UNITS = (*persist.SUPERVISOR.units, *persist.LINK.units)


class FakeComma(persist.System):
    def __init__(self, layout: persist.Layout):
        self.layout = layout
        self.host = persist.HOSTNAME
        self.procs = ["/usr/local/venv/bin/python3 ./manager.py"]
        self.free = 4 << 30
        self.sudo_ok = True
        self.version = "systemd 245 (245.4-4ubuntu3.15)\n+PAM +AUDIT"
        self.verify = persist.Result(0, "", "")
        self.active = "active"
        self.fail_on: str | None = None
        self.root_writable = False
        self.calls: list[list[str]] = []

    def hostname(self) -> str:
        return self.host

    def processes(self) -> list[str]:
        return self.procs

    def free_bytes(self, path: Path) -> int:
        return self.free

    def changes(self) -> list[list[str]]:
        reads = (["sudo", "-n", "true"], ["systemctl", "--version"])
        return [c for c in self.calls if c[:3] not in reads and c[:2] not in (["systemctl", "--version"], ["systemctl", "is-active"])
                and c[0] != "systemd-analyze"]

    def run(self, argv, cwd=None, env=None) -> persist.Result:
        argv = list(argv)
        self.calls.append(argv)
        ok = persist.Result(0, "", "")
        if self.fail_on is not None and self.fail_on in " ".join(argv):
            return persist.Result(1, "", "simulated failure")
        if argv[:3] == ["sudo", "-n", "true"]:
            return ok if self.sudo_ok else persist.Result(1, "", "a password is required")
        if argv == ["systemctl", "--version"]:
            return persist.Result(0, self.version, "")
        if argv[0] == "systemd-analyze":
            return self.verify
        if argv[:2] == ["systemctl", "is-active"]:
            return persist.Result(0 if self.active == "active" else 3, self.active + "\n", "")
        if argv[:3] == ["sudo", "mount", "-o"]:
            self.root_writable = argv[3] == "remount,rw"
            return ok
        if self._writes_etc(argv) and not self.root_writable:
            return persist.Result(1, "", "Read-only file system")
        if argv[:2] == ["sudo", "install"]:
            shutil.copyfile(argv[4], argv[5])
        elif argv[:3] == ["sudo", "rm", "-f"]:
            Path(argv[3]).unlink(missing_ok=True)
        elif argv[:3] == ["sudo", "systemctl", "enable"]:
            self.layout.wants(argv[3]).parent.mkdir(exist_ok=True)
            self.layout.wants(argv[3]).write_text(argv[3])
        elif argv[:3] == ["sudo", "systemctl", "disable"]:
            self.layout.wants(argv[3]).unlink(missing_ok=True)
        return ok

    def _writes_etc(self, argv: list[str]) -> bool:
        if argv[:3] in (["sudo", "systemctl", "enable"], ["sudo", "systemctl", "disable"]):
            return True
        return argv[:2] in (["sudo", "install"], ["sudo", "rm"]) and str(self.layout.systemd) in argv[-1]


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None for p in sorted(root.rglob("*"))}


class Device(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = persist.Layout(Path(self.tmp.name))
        for units in (persist.SUPERVISOR, persist.LINK):
            target = self.layout.repo / units.source
            target.mkdir(parents=True, exist_ok=True)
            for unit in units.units:
                shutil.copyfile(HERE.parent / units.source / unit, target / unit)
        shutil.copyfile(HERE / "vision_long_supervisor/monitor.py", self.layout.repo / persist.SOURCE / "monitor.py")
        shutil.copyfile(HERE / "vision_long_link/usb_link.py", self.layout.repo / persist.LINK_SCRIPT)
        self.layout.systemd.mkdir(parents=True)
        self.layout.state.mkdir(parents=True)
        self.layout.key.write_bytes(bytes(32))
        self.layout.is_offroad.parent.mkdir(parents=True)
        self.layout.is_offroad.write_bytes(b"1")
        self.layout.python.parent.mkdir(parents=True)
        self.layout.python.write_text("")
        self.comma = FakeComma(self.layout)

    def tearDown(self):
        self.tmp.cleanup()

    def installed(self) -> list[str]:
        return sorted(p.name for p in self.layout.systemd.iterdir() if p.is_file())


class Install(Device):
    def test_install_puts_the_units_in_enables_them_and_closes_the_window(self):
        receipt = persist.install(self.comma, self.layout)
        self.assertEqual(self.installed(), sorted(persist.SUPERVISOR.units))
        for unit in persist.SUPERVISOR.enable:
            self.assertTrue(self.layout.wants(unit).exists(), unit)
        self.assertTrue(self.layout.marker.is_file())
        self.assertFalse(self.comma.root_writable)
        self.assertIn(["sudo", "systemctl", "start", persist.SERVICE], self.comma.calls)
        self.assertIn("systemd      245", receipt)

    def test_a_rerun_is_safe_and_leaves_the_same_state(self):
        persist.install(self.comma, self.layout)
        before = snapshot(self.layout.root)
        persist.install(self.comma, self.layout)
        self.assertEqual(snapshot(self.layout.root), before)

    def test_every_preflight_failure_changes_nothing_and_all_are_reported(self):
        cases = {
            "onroad": lambda: self.layout.is_offroad.write_bytes(b"0"),
            "host": lambda: setattr(self.comma, "host", "iq-6f0745e3"),
            "rsync": lambda: self.comma.procs.append("rsync --server -vlogDtpre.iLsfxC"),
            "sudo": lambda: setattr(self.comma, "sudo_ok", False),
            "old systemd": lambda: setattr(self.comma, "version", "systemd 229\n"),
            "verify": lambda: setattr(self.comma, "verify", persist.Result(0, "", "Unknown key name 'CPUQuota'")),
            "no key": lambda: self.layout.key.unlink(),
            "short key": lambda: self.layout.key.write_bytes(b"x"),
            "full disk": lambda: setattr(self.comma, "free", 1 << 20),
            "no python": lambda: self.layout.python.unlink(),
            "no monitor": lambda: (self.layout.repo / persist.SOURCE / "monitor.py").unlink(),
        }
        for name, breaks in cases.items():
            with self.subTest(name):
                self.tearDown()
                self.setUp()
                breaks()
                before = snapshot(self.layout.root)
                with self.assertRaises(persist.PersistError) as caught:
                    persist.install(self.comma, self.layout)
                self.assertIn("nothing was changed", str(caught.exception))
                self.assertEqual(snapshot(self.layout.root), before)
                self.assertEqual(self.comma.changes(), [])

    def test_a_monitor_that_does_not_stay_up_is_undone(self):
        self.comma.active = "activating"
        with self.assertRaises(persist.PersistError) as caught:
            persist.install(self.comma, self.layout)
        self.assertIn("Undone", str(caught.exception))
        self.assertEqual(self.installed(), [])
        self.assertFalse(any(self.layout.wants(u).exists() for u in persist.SUPERVISOR.enable))
        self.assertFalse(self.layout.marker.exists())
        self.assertFalse(self.comma.root_writable)

    def test_a_failed_enable_still_closes_the_window_and_is_undone(self):
        self.comma.fail_on = "enable " + persist.WATCH_TIMER
        with self.assertRaises(persist.PersistError):
            persist.install(self.comma, self.layout)
        self.assertFalse(self.comma.root_writable)
        self.assertEqual(self.installed(), [])

    def test_rollback_removes_only_the_monitor_and_keeps_state(self):
        persist.install(self.comma, self.layout)
        for unit in persist.LINK.units:
            shutil.copyfile(self.layout.source(persist.LINK, unit), self.layout.systemd / unit)
        self.layout.notice.write_text("{}")
        persist.rollback(self.comma, self.layout)
        self.assertEqual(self.installed(), sorted(persist.LINK.units))
        self.assertFalse(self.layout.marker.exists())
        self.assertFalse(self.layout.notice.exists())
        self.assertTrue(self.layout.key.exists())
        self.assertFalse(self.comma.root_writable)

    def test_rollback_stops_the_watch_before_the_monitor(self):
        persist.install(self.comma, self.layout)
        persist.rollback(self.comma, self.layout)
        stops = [c[3] for c in self.comma.calls if c[:3] == ["sudo", "systemctl", "stop"]]
        self.assertEqual(stops, [persist.WATCH_TIMER, persist.WATCH_SERVICE, persist.SERVICE])

    def test_a_key_the_comma_user_cannot_read_is_refused(self):
        real = persist.os.access
        persist.os.access = lambda path, mode: Path(path) != self.layout.key and real(path, mode)
        try:
            with self.assertRaises(persist.PersistError) as caught:
                persist.install(self.comma, self.layout)
        finally:
            persist.os.access = real
        self.assertIn("not readable", str(caught.exception))
        self.assertEqual(self.comma.changes(), [])

    def test_a_failure_inside_the_window_still_remounts_read_only(self):
        self.comma.fail_on = "daemon-reload"
        self.comma.calls.clear()
        with self.assertRaises(persist.PersistError):
            persist.install_units(self.comma, self.layout, persist.SUPERVISOR)
        self.assertFalse(self.comma.root_writable)
        self.assertEqual(self.comma.calls[-1], ["sudo", "mount", "-o", "remount,ro", "/"])

    def test_a_sigterm_inside_the_window_still_remounts_read_only(self):
        import signal
        previous = persist.install_term_handler()
        try:
            real_run = self.comma.run

            def run(argv, cwd=None, env=None):
                if argv[:3] == ["sudo", "systemctl", "enable"]:
                    signal.raise_signal(signal.SIGTERM)
                return real_run(argv, cwd, env)
            self.comma.run = run
            with self.assertRaises(persist.Terminated):
                persist.install_units(self.comma, self.layout, persist.SUPERVISOR)
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertFalse(self.comma.root_writable)

    def test_rollback_removes_the_marker_before_the_units(self):
        persist.install(self.comma, self.layout)
        self.comma.fail_on = "disable"
        with self.assertRaises(persist.PersistError):
            persist.rollback(self.comma, self.layout)
        self.assertFalse(self.layout.marker.exists())

    def test_rollback_onroad_changes_nothing(self):
        persist.install(self.comma, self.layout)
        self.layout.is_offroad.write_bytes(b"0")
        before, changes = snapshot(self.layout.root), len(self.comma.changes())
        with self.assertRaises(persist.PersistError):
            persist.rollback(self.comma, self.layout)
        self.assertEqual(snapshot(self.layout.root), before)
        self.assertEqual(len(self.comma.changes()), changes)


class Reestablish(Device):
    def agnos_update(self) -> None:
        """A slot flip: every unit and wants link in /etc/systemd/system is gone."""
        shutil.rmtree(self.layout.systemd)
        self.layout.systemd.mkdir()

    def provision_link(self) -> None:
        self.layout.runtime_config.parent.mkdir(parents=True, exist_ok=True)
        self.layout.runtime_config.write_text("{}")
        self.comma.root_writable = True
        persist.install_units(self.comma, self.layout, persist.LINK)
        self.comma.root_writable = False

    def test_after_an_agnos_update_both_sets_come_back(self):
        persist.install(self.comma, self.layout)
        self.provision_link()
        self.agnos_update()
        lines = persist.reestablish(self.comma, self.layout)
        self.assertEqual(self.installed(), sorted(ALL_UNITS))
        for unit in (*persist.SUPERVISOR.enable, *persist.LINK.enable):
            self.assertTrue(self.layout.wants(unit).exists(), unit)
        self.assertFalse(self.comma.root_writable)
        self.assertTrue(all("re-installed" in line for line in lines), lines)
        self.assertIn(["sudo", "systemctl", "start", persist.LINK.units[0]], self.comma.calls)

    def test_in_place_units_mean_no_remount_and_no_change(self):
        persist.install(self.comma, self.layout)
        self.provision_link()
        before, count = snapshot(self.layout.root), len(self.comma.calls)
        lines = persist.reestablish(self.comma, self.layout)
        self.assertEqual(snapshot(self.layout.root), before)
        self.assertEqual(self.comma.calls[count:], [])
        self.assertEqual(lines, ["monitor: in place", "usb link: in place"])

    def test_a_changed_or_unenabled_unit_is_put_right(self):
        persist.install(self.comma, self.layout)
        (self.layout.systemd / persist.SERVICE).write_text("[Service]\nExecStart=/bin/true\n")
        self.layout.wants(persist.WATCH_TIMER).unlink()
        persist.reestablish(self.comma, self.layout)
        self.assertEqual((self.layout.systemd / persist.SERVICE).read_bytes(),
                         self.layout.source(persist.SUPERVISOR, persist.SERVICE).read_bytes())
        self.assertTrue(self.layout.wants(persist.WATCH_TIMER).exists())

    def test_nothing_that_was_never_installed_or_was_rolled_back_comes_back(self):
        self.assertEqual(persist.reestablish(self.comma, self.layout), ["nothing expected: no Vision Long units were installed"])
        persist.install(self.comma, self.layout)
        persist.rollback(self.comma, self.layout)
        self.agnos_update()
        persist.reestablish(self.comma, self.layout)
        self.assertEqual(self.installed(), [])

    def test_without_sudo_it_reports_and_changes_nothing(self):
        persist.install(self.comma, self.layout)
        self.agnos_update()
        self.comma.sudo_ok = False
        before = snapshot(self.layout.root)
        with self.assertRaises(persist.PersistError):
            persist.reestablish(self.comma, self.layout)
        self.assertEqual(snapshot(self.layout.root), before)

    def test_the_boot_entry_point_logs_a_failure_instead_of_raising(self):
        persist.install(self.comma, self.layout)
        self.agnos_update()
        self.comma.sudo_ok = False
        self.assertEqual(persist.main(["reestablish"], root=self.layout.root, system=self.comma), 1)
        self.assertIn("FAILED", self.layout.reestablish_log.read_text())
        self.comma.sudo_ok = True
        self.assertEqual(persist.main(["reestablish"], root=self.layout.root, system=self.comma), 0)
        self.assertIn("monitor: re-installed", self.layout.reestablish_log.read_text())

    def test_any_exception_at_boot_is_logged_not_raised(self):
        persist.install(self.comma, self.layout)
        self.agnos_update()

        def broken(argv, cwd=None, env=None):
            raise RuntimeError("fork failed")
        self.comma.run = broken
        self.assertEqual(persist.main(["reestablish"], root=self.layout.root, system=self.comma), 1)
        self.assertIn("FAILED: RuntimeError: fork failed", self.layout.reestablish_log.read_text())

    def test_the_link_unit_names_match_r4s_installer(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("r4_install", HERE / "vision_long_link/install.py")
        r4 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = r4
        spec.loader.exec_module(r4)
        self.assertEqual((persist.LINK.source, persist.LINK.units, persist.LINK_SCRIPT, persist.RUNTIME_CONFIG),
                         (r4.UNIT_SOURCE, tuple(r4.UNITS), r4.LINK_SCRIPT, r4.RUNTIME_CONFIG))
        self.assertEqual((persist.LINK.enable, persist.HOSTNAME, persist.PYTHON), ((r4.TIMER,), r4.HOSTNAME, r4.PYTHON))

    def test_the_boot_hook_script_is_detached_and_time_limited(self):
        script = (HERE / "vision_long_supervisor/reestablish.sh").read_text()
        self.assertIn("setsid nohup timeout -k 30 120", script)
        self.assertTrue(script.rstrip().endswith("exit 0"))
        self.assertIn("persist.py\" reestablish >/dev/null 2>&1 </dev/null &", script)


class Units(unittest.TestCase):
    def test_the_monitor_unit_carries_the_limits_and_never_gives_up(self):
        text = (HERE / "vision_long_supervisor" / persist.SERVICE).read_text()
        lines = {line.strip() for line in text.splitlines()}
        for required in ("Nice=19", "CPUQuota=5%", "MemoryMax=64M", "IOSchedulingClass=idle", "Restart=always",
                         "RestartSec=5", "StartLimitIntervalSec=0", "User=comma", "Environment=PYTHONPATH=/data/openpilot",
                         "NoNewPrivileges=yes",
                         "ExecStart=/usr/local/venv/bin/python3 /data/openpilot/.elantra/vision_long_supervisor/monitor.py"):
            self.assertIn(required, lines, required)
        self.assertNotIn("StartLimitBurst", text)

    def test_the_watch_timer_rearms(self):
        timer = (HERE / "vision_long_supervisor" / persist.WATCH_TIMER).read_text()
        self.assertIn("OnBootSec=20s", timer)
        self.assertIn("OnUnitActiveSec=60s", timer)
        watch = (HERE / "vision_long_supervisor" / persist.WATCH_SERVICE).read_text()
        self.assertIn("Type=oneshot", watch)
        self.assertIn("ExecStart=/bin/systemctl start " + persist.SERVICE, watch)


if __name__ == "__main__":
    unittest.main()
