#!/usr/bin/env python3
"""
Tests for the Vision Long installer and rollback (.elantra/vision_long_link/install.py).

The device is a temporary directory standing in for `/` plus a fake that answers git, sudo
and systemctl the way the comma would and records every command. What these pin: every
preflight failure stops the run before any file or command changes anything; the key and the
toggle are created once and never overwritten; a failure after the first change is undone;
the rollback restores the saved SHAs and removes only what the install added.

    python .elantra/test_vision_long_install.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vision_long_link"))
import install as inst

BASE_OP, BASE_DBC = "13aedc522518fe1b62651d9a8a320e931efc335c", "b45eb78def29582bcce43c68cfb1d55f19224d1a"
TARGET_OP, TARGET_DBC = "c765b7a227f04e7963c7d221ee28fea54e931578", "ef16146a76b5e6abc614e2a8b1edba842f30c4e5"
SHAS = inst.Shas(BASE_OP, BASE_DBC, TARGET_OP, TARGET_DBC)
TEMPLATE = (HERE / "vision_long_link/native-runtime.template.json").read_bytes()
READ_ONLY = (("rev-parse",), ("status",), ("ls-tree",), ("show",), ("symbolic-ref",), ("cat-file",))


class FakeComma(inst.System):
    def __init__(self, layout: inst.Layout):
        self.layout = layout
        self.host = inst.HOSTNAME
        self.procs = ["/usr/bin/python3 -m openpilot.system.manager"]
        self.free = 20 << 30
        self.sudo_ok = True
        self.head, self.opendbc_head = BASE_OP, BASE_DBC
        self.fetch_head: str | None = TARGET_OP
        self.fetched_gitlink = TARGET_DBC
        self.template = TEMPLATE
        self.branch: str | None = None
        self.dirty: list[str] = []
        self.guards_code = 0
        self.link = (inst.LINK_NO_CARRIER, "link waiting: usb0 configured but no carrier")
        self.calls: list[list[str]] = []

    def hostname(self) -> str:
        return self.host

    def processes(self) -> list[str]:
        return self.procs

    def free_bytes(self, path: Path) -> int:
        return self.free

    def mutations(self) -> list[list[str]]:
        def read_only(argv: list[str]) -> bool:
            if argv[:3] == ["sudo", "-n", "true"]:
                return True
            return argv[0] == "git" and any(tuple(argv[3:3 + len(p)]) == p for p in READ_ONLY)
        return [argv for argv in self.calls if not read_only(argv)]

    def run(self, argv, cwd=None, env=None) -> inst.Result:
        self.calls.append(list(argv))
        ok = inst.Result(0, "", "")
        if argv[:3] == ["sudo", "-n", "true"]:
            return ok if self.sudo_ok else inst.Result(1, "", "a password is required")
        if argv[0] == "git":
            return self._git(Path(argv[2]), argv[3:])
        if argv[:2] == ["sudo", "install"]:
            shutil.copyfile(argv[4], argv[5])
            return ok
        if argv[:3] == ["sudo", "rm", "-f"]:
            Path(argv[3]).unlink(missing_ok=True)
            return ok
        if argv[0] == inst.PYTHON and argv[1] == ".elantra/guards.py":
            return inst.Result(self.guards_code, "PASSED" if self.guards_code == 0 else "FAILED: torque", "")
        if argv[0] == inst.PYTHON and argv[1] == "-c":
            return inst.Result(0, "overlay\n", "")
        if argv[0] == inst.PYTHON and argv[-1] == "check":
            return inst.Result(self.link[0], self.link[1] + "\n", "")
        return ok

    def _git(self, repo: Path, args: list[str]) -> inst.Result:
        opendbc = repo == self.layout.opendbc
        if args == ["rev-parse", "HEAD"]:
            return inst.Result(0, (self.opendbc_head if opendbc else self.head) + "\n", "")
        if args == ["rev-parse", "FETCH_HEAD"]:
            return inst.Result(0, self.fetch_head + "\n", "") if self.fetch_head else inst.Result(128, "", "bad")
        if args[0] == "ls-tree":
            sha = self.fetched_gitlink if args[1] == "FETCH_HEAD" else self.opendbc_head
            return inst.Result(0, f"160000 commit {sha}\topendbc_repo\n", "")
        if args[0] == "show":
            return inst.Result(0, self.template.decode(), "")
        if args == ["status", "--porcelain"]:
            lines = list(self.dirty)
            if self.layout.runtime_config.exists():
                lines.append("?? " + inst.RUNTIME_CONFIG)
            return inst.Result(0, "".join(line + "\n" for line in lines), "")
        if args[0] == "symbolic-ref":
            return inst.Result(0, self.branch + "\n", "") if self.branch else inst.Result(1, "", "")
        if args[0] == "cat-file":
            return inst.Result(0, "", "")
        if args[:3] == ["checkout", "-q", "-B"]:
            self.head = self.fetch_head or ""
            self._materialise_branch()
        elif args[:3] == ["checkout", "-q", "--detach"]:
            self.head = args[3]
        elif args[0] == "submodule":
            self.opendbc_head = TARGET_DBC if self.head == TARGET_OP else BASE_DBC
        return inst.Result(0, "", "")

    def _materialise_branch(self) -> None:
        source = HERE / "vision_long_link"
        target = self.layout.repo / inst.UNIT_SOURCE
        target.mkdir(parents=True, exist_ok=True)
        for name in ("native-runtime.template.json", *inst.UNITS, "usb_link.py"):
            shutil.copyfile(source / name, target / name)
        (self.layout.repo / ".elantra/vision_long").mkdir(parents=True, exist_ok=True)


def snapshot(root: Path) -> dict[str, bytes | None]:
    """Every file with its bytes and every directory (as None): an empty new folder is a change too."""
    return {p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None for p in sorted(root.rglob("*"))}


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.layout = inst.Layout(Path(self._tmp.name))
        self.layout.opendbc.mkdir(parents=True)
        self.layout.disable_updates.parent.mkdir(parents=True)
        self.layout.disable_updates.write_bytes(b"1")
        self.layout.systemd.mkdir(parents=True)
        self.comma = FakeComma(self.layout)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_refused_without_change(self, reason: str):
        before = snapshot(self.layout.root)
        with self.assertRaisesRegex(inst.InstallError, reason) as caught:
            inst.install(self.comma, self.layout, SHAS)
        self.assertIn("nothing was changed", str(caught.exception))
        self.assertEqual(snapshot(self.layout.root), before)
        self.assertEqual(self.comma.mutations(), [])

    # --- preflight ----------------------------------------------------------------------------

    def test_wrong_hostname_is_refused_before_any_change(self):
        self.comma.host = "iq-6f0745e3"
        self.assert_refused_without_change("hostname")

    def test_a_running_rsync_is_refused(self):
        self.comma.procs.append("rsync --server --sender -logDtpre.iLsfxC . /data/media")
        self.assert_refused_without_change("rsync")

    def test_a_dirty_checkout_is_refused(self):
        self.comma.dirty = [" M openpilot/selfdrive/ui/ui.py"]
        self.assert_refused_without_change("not clean")

    def test_an_unknown_checkout_is_refused(self):
        self.comma.head = "0" * 40
        self.assert_refused_without_change("neither the base")

    def test_updates_must_be_disabled(self):
        self.layout.disable_updates.write_bytes(b"0")
        self.assert_refused_without_change("DisableUpdates")
        self.layout.disable_updates.unlink()
        self.assert_refused_without_change("DisableUpdates")

    def test_low_free_space_is_refused(self):
        self.comma.free = 100 << 20
        self.assert_refused_without_change("MiB free")

    def test_sudo_without_a_password_is_required(self):
        self.comma.sudo_ok = False
        self.assert_refused_without_change("sudo")

    def test_the_fetched_branch_must_be_the_target(self):
        self.comma.fetch_head = "1" * 40
        self.assert_refused_without_change("fetched")
        self.comma.fetch_head = None
        self.assert_refused_without_change("missing")

    def test_the_fetched_branch_must_pin_the_target_opendbc(self):
        self.comma.fetched_gitlink = "2" * 40
        self.assert_refused_without_change("does not pin opendbc")

    def test_a_template_that_breaks_the_road_contract_is_refused(self):
        self.comma.template = TEMPLATE.replace(b"45556", b"45557")
        self.assert_refused_without_change("port")

    def test_a_malformed_sha_argument_is_refused(self):
        before = snapshot(self.layout.root)
        with self.assertRaisesRegex(inst.InstallError, "40-character"):
            inst.install(self.comma, self.layout, inst.Shas("13aedc52", BASE_DBC, TARGET_OP, TARGET_DBC))
        self.assertEqual(snapshot(self.layout.root), before)
        self.assertEqual(self.comma.calls, [])

    def test_a_wrong_sized_key_is_refused_never_overwritten(self):
        self.layout.state.mkdir(parents=True)
        self.layout.key.write_bytes(b"short")
        self.assert_refused_without_change("never overwritten")

    def test_every_failed_check_is_reported_at_once(self):
        self.comma.host = "elsewhere"
        self.layout.disable_updates.write_bytes(b"0")
        with self.assertRaises(inst.InstallError) as caught:
            inst.install(self.comma, self.layout, SHAS)
        self.assertIn("hostname", str(caught.exception))
        self.assertIn("DisableUpdates", str(caught.exception))

    def test_a_rerun_needs_the_saved_rollback_shas(self):
        self.comma.head, self.comma.opendbc_head = TARGET_OP, TARGET_DBC
        self.assert_refused_without_change("rollback SHAs")

    # --- install ------------------------------------------------------------------------------

    def test_a_fresh_install_provisions_everything_and_restarts_last(self):
        receipt = inst.install(self.comma, self.layout, SHAS)
        self.assertEqual(self.layout.rollback_openpilot.read_text().strip(), BASE_OP)
        self.assertEqual(self.layout.rollback_opendbc.read_text().strip(), BASE_DBC)
        self.assertEqual(self.layout.rollback_branch.read_text().strip(), inst.DETACHED)
        self.assertEqual(len(self.layout.key.read_bytes()), inst.KEY_BYTES)
        self.assertEqual(self.layout.toggle.read_bytes(), b"0")
        self.assertEqual(self.layout.runtime_config.read_bytes(), TEMPLATE)
        for unit in inst.UNITS:
            self.assertTrue((self.layout.systemd / unit).is_file())
        self.assertEqual(self.comma.calls[-1], ["sh", "-c", inst.RESTART])
        self.assertTrue(any(line.startswith("link key") and "sha256" in line for line in receipt))
        self.assertIn(["sudo", "systemctl", "enable", inst.TIMER], self.comma.calls)

    def test_an_existing_key_and_toggle_are_kept_byte_for_byte(self):
        self.layout.state.mkdir(parents=True)
        key = bytes(range(32))
        self.layout.key.write_bytes(key)
        self.layout.key.chmod(0o600)
        self.layout.toggle.write_bytes(b"1")
        receipt = inst.install(self.comma, self.layout, SHAS)
        self.assertEqual(self.layout.key.read_bytes(), key)
        self.assertEqual(self.layout.toggle.read_bytes(), b"1")
        self.assertIn("braking      ON", "\n".join(receipt))

    def test_a_failure_after_checkout_restores_the_base_and_keeps_the_key(self):
        self.comma.guards_code = 1
        with self.assertRaisesRegex(inst.InstallError, "Undone: restored the base"):
            inst.install(self.comma, self.layout, SHAS)
        self.assertEqual((self.comma.head, self.comma.opendbc_head), (BASE_OP, BASE_DBC))
        self.assertFalse(self.layout.runtime_config.exists())
        self.assertNotIn(["sh", "-c", inst.RESTART], self.comma.calls)

    def test_a_link_that_cannot_come_up_undoes_the_units_and_config(self):
        self.comma.link = (inst.LINK_OK + 1, "link failed: UDC a600000.dwc3 is bound to another gadget: g1")
        with self.assertRaisesRegex(inst.InstallError, "bound to another gadget"):
            inst.install(self.comma, self.layout, SHAS)
        for unit in inst.UNITS:
            self.assertFalse((self.layout.systemd / unit).exists())
        self.assertFalse(self.layout.runtime_config.exists())
        self.assertTrue(self.layout.key.is_file())
        self.assertEqual(self.comma.head, BASE_OP)

    def test_a_rerun_after_an_agnos_update_reinstalls_the_units(self):
        inst.install(self.comma, self.layout, SHAS)
        for unit in inst.UNITS:
            (self.layout.systemd / unit).unlink()
        key = self.layout.key.read_bytes()
        receipt = inst.install(self.comma, self.layout, SHAS)
        self.assertIn("mode         rerun", receipt)
        self.assertTrue((self.layout.systemd / inst.TIMER).is_file())
        self.assertEqual(self.layout.key.read_bytes(), key)
        self.assertEqual(self.layout.runtime_config.read_bytes(), TEMPLATE)

    # --- rollback -----------------------------------------------------------------------------

    def test_rollback_restores_the_base_and_keeps_key_toggle_and_logs(self):
        inst.install(self.comma, self.layout, SHAS)
        (self.layout.state / "link.log").write_text("x\n")
        key = self.layout.key.read_bytes()
        inst.rollback(self.comma, self.layout)
        self.assertEqual((self.comma.head, self.comma.opendbc_head), (BASE_OP, BASE_DBC))
        self.assertFalse(self.layout.runtime_config.exists())
        for unit in inst.UNITS:
            self.assertFalse((self.layout.systemd / unit).exists())
        self.assertEqual(self.layout.key.read_bytes(), key)
        self.assertTrue(self.layout.toggle.is_file())
        self.assertTrue((self.layout.state / "link.log").is_file())
        self.assertIn(["sudo", inst.PYTHON, str(self.layout.repo / inst.LINK_SCRIPT), "down"], self.comma.calls)
        self.assertEqual(self.comma.calls[-1], ["sh", "-c", inst.RESTART])
        self.assertFalse(any("pkill" in " ".join(c) for c in self.comma.calls))

    def test_rollback_without_saved_shas_changes_nothing(self):
        before = snapshot(self.layout.root)
        with self.assertRaisesRegex(inst.InstallError, "rollback preflight failed, nothing was changed"):
            inst.rollback(self.comma, self.layout)
        self.assertEqual(snapshot(self.layout.root), before)
        self.assertEqual(self.comma.mutations(), [])

    def test_rollback_returns_to_a_saved_branch(self):
        self.comma.branch = "master"
        inst.install(self.comma, self.layout, SHAS)
        self.comma.head = BASE_OP  # `git checkout master` lands on the saved SHA
        inst.rollback(self.comma, self.layout)
        self.assertIn(["git", "-C", str(self.layout.repo), "checkout", "-q", "master"], self.comma.calls)


class TemplateTests(unittest.TestCase):
    def test_the_shipped_template_carries_the_road_contract(self):
        self.assertEqual(inst.validate_template(TEMPLATE), TEMPLATE)

    def test_extra_fields_are_allowed_but_contract_fields_are_not_negotiable(self):
        self.assertTrue(inst.validate_template(b'{"schema": "native_runtime_v2", "listen_ip": "192.168.250.1", '
                                               + b'"port": 45556, "key_file": "/data/vision_long/link.key", '
                                               + b'"toggle_file": "/data/vision_long/braking", "pairs": 1}'))
        for broken in (b"[]", b"{", b'{"schema": "native_runtime_v2"}'):
            with self.subTest(broken=broken), self.assertRaises(inst.InstallError):
                inst.validate_template(broken)


if __name__ == "__main__":
    unittest.main()
