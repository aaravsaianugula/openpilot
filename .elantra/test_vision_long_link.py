#!/usr/bin/env python3
"""
Tests for the Vision Long USB link (CDC-NCM gadget on the comma 4's a600000.dwc3 UDC).

configfs and sysfs are plain directories here. The kernel's side of configfs (the attribute
files a mkdir creates, the ifname the bind assigns) is written by the test where the kernel
would write it; the link module itself only ever mkdirs, writes, reads and runs `ip`. What
these pin is the module's decisions: it never binds a UDC other than a600000.dwc3, never
touches another gadget, is a no-op when the link is already up, and its check names the
exact failure.

    python .elantra/test_vision_long_link.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vision_long_link"))
import usb_link

UDC = "a600000.dwc3"


class FakeIp:
    """Records `ip` and `nmcli` invocations and answers `ip -4 -o addr show` from its state."""

    def __init__(self, addresses: list[str] | None = None):
        self.addresses = list(addresses or [])
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> str:
        self.calls.append(argv)
        if argv[:4] == ["ip", "-4", "-o", "addr"]:
            return "".join(f"5: usb0    inet {a} brd 192.168.250.255 scope global usb0\n" for a in self.addresses)
        if argv[:3] == ["ip", "addr", "add"]:
            self.addresses.append(argv[3])
        return ""

    def mutating(self) -> list[list[str]]:
        return [c for c in self.calls if c[:4] != ["ip", "-4", "-o", "addr"]]


def link_marker(target: Path, member: Path) -> None:
    """Stands in for configfs's function symlink where the test machine cannot make symlinks."""
    member.write_text("-> " + str(target))


def configfs_rmdir(directory: Path) -> None:
    """configfs drops a directory's attribute files and empty default groups (strings, configs,
    functions) itself on rmdir; a plain directory does not. A non-empty child still fails."""
    for child in directory.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    directory.rmdir()


class LinkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.paths = usb_link.Paths(configfs=base / "config", udc_class=base / "udc", net=base / "net")
        (self.paths.configfs / "usb_gadget").mkdir(parents=True)
        for udc in (UDC, "a800000.dwc3"):
            (self.paths.udc_class / udc).mkdir(parents=True)
        self.gadget = self.paths.configfs / "usb_gadget" / usb_link.GADGET_NAME

    def tearDown(self):
        self._tmp.cleanup()

    def kernel_assigns_interface(self, name: str = "usb0", operstate: str = "up", flags: str = "0x1003"):
        function = self.gadget / "functions" / usb_link.FUNCTION
        function.mkdir(parents=True, exist_ok=True)
        (function / "ifname").write_text(name + "\n")
        iface = self.paths.net / name
        iface.mkdir(parents=True, exist_ok=True)
        (iface / "operstate").write_text(operstate + "\n")
        (iface / "flags").write_text(flags + "\n")

    def bring_up(self, ip: FakeIp) -> str:
        return usb_link.bring_up(self.paths, ip, nmcli_present=True, link=link_marker)

    def test_a_fresh_device_gets_the_ncm_gadget_bound_to_a600000_only(self):
        self.kernel_assigns_interface()
        ip = FakeIp()
        self.assertEqual(self.bring_up(ip), "usb0")
        self.assertEqual((self.gadget / "UDC").read_text().strip(), UDC)
        function = self.gadget / "functions" / usb_link.FUNCTION
        self.assertEqual((function / "dev_addr").read_text().strip(), usb_link.DEV_ADDR)
        self.assertEqual((function / "host_addr").read_text().strip(), usb_link.HOST_ADDR)
        self.assertEqual((self.gadget / "configs" / usb_link.CONFIG / usb_link.FUNCTION).read_text(),
                         "-> " + str(function))
        self.assertIn(["ip", "addr", "add", "192.168.250.1/24", "dev", "usb0"], ip.calls)
        self.assertIn(["ip", "link", "set", "usb0", "up"], ip.calls)
        self.assertIn(["nmcli", "device", "set", "usb0", "managed", "no"], ip.calls)

    def test_a_rerun_on_a_live_link_changes_nothing(self):
        self.kernel_assigns_interface()
        self.bring_up(FakeIp())
        before = {p: p.read_bytes() for p in self.gadget.rglob("*") if p.is_file()}
        ip = FakeIp(["192.168.250.1/24"])
        self.bring_up(ip)
        self.assertEqual({p: p.read_bytes() for p in self.gadget.rglob("*") if p.is_file()}, before)
        self.assertNotIn(["ip", "addr", "add", "192.168.250.1/24", "dev", "usb0"], ip.calls)

    def test_an_address_flushed_by_someone_else_is_put_back(self):
        self.kernel_assigns_interface()
        self.bring_up(FakeIp())
        ip = FakeIp([])
        self.bring_up(ip)
        self.assertIn(["ip", "addr", "add", "192.168.250.1/24", "dev", "usb0"], ip.calls)

    def test_a_udc_bound_to_another_gadget_is_refused_and_left_alone(self):
        other = self.paths.configfs / "usb_gadget" / "g1"
        other.mkdir()
        (other / "UDC").write_text(UDC + "\n")
        ip = FakeIp()
        with self.assertRaisesRegex(usb_link.LinkError, "bound to another gadget: g1"):
            self.bring_up(ip)
        self.assertFalse(self.gadget.exists())
        self.assertEqual((other / "UDC").read_text(), UDC + "\n")
        self.assertEqual(ip.calls, [])

    def test_a_gadget_on_the_other_udc_is_not_touched(self):
        modem = self.paths.configfs / "usb_gadget" / "g1"
        modem.mkdir()
        (modem / "UDC").write_text("a800000.dwc3\n")
        self.kernel_assigns_interface()
        self.bring_up(FakeIp())
        self.assertEqual((modem / "UDC").read_text(), "a800000.dwc3\n")
        self.assertEqual((self.gadget / "UDC").read_text().strip(), UDC)

    def test_missing_configfs_or_udc_fails_with_the_cause(self):
        missing = usb_link.Paths(configfs=self.paths.configfs / "nope", udc_class=self.paths.udc_class,
                                 net=self.paths.net)
        with self.assertRaisesRegex(usb_link.LinkError, "configfs"):
            usb_link.bring_up(missing, FakeIp(), nmcli_present=True, link=link_marker)
        (self.paths.udc_class / UDC).rmdir()
        with self.assertRaisesRegex(usb_link.LinkError, UDC):
            self.bring_up(FakeIp())

    def test_no_interface_after_binding_is_a_failure_not_a_success(self):
        ip = FakeIp()
        with self.assertRaisesRegex(usb_link.LinkError, "ifname"):
            self.bring_up(ip)
        self.assertEqual(ip.mutating(), [])

    def test_without_networkmanager_no_nmcli_call_is_made(self):
        self.kernel_assigns_interface()
        ip = FakeIp()
        usb_link.bring_up(self.paths, ip, nmcli_present=False, link=link_marker)
        self.assertFalse(any(c[0] == "nmcli" for c in ip.calls))

    def test_check_reports_ok_only_with_bind_address_and_carrier(self):
        self.kernel_assigns_interface()
        self.bring_up(FakeIp())
        self.assertEqual(usb_link.check(self.paths, FakeIp(["192.168.250.1/24"])),
                         (usb_link.CHECK_OK, "link ok 192.168.250.1 usb0 up"))

    def test_check_names_each_failure(self):
        code, text = usb_link.check(self.paths, FakeIp())
        self.assertEqual((code, text), (usb_link.CHECK_FAILED, "link failed: gadget vision_long is not configured"))
        self.kernel_assigns_interface(operstate="down", flags="0x1002")
        self.bring_up(FakeIp())
        (self.gadget / "UDC").write_text("\n")
        self.assertEqual(usb_link.check(self.paths, FakeIp(["192.168.250.1/24"]))[1],
                         "link failed: gadget vision_long is not bound to a600000.dwc3")
        (self.gadget / "UDC").write_text(UDC + "\n")
        self.assertEqual(usb_link.check(self.paths, FakeIp([]))[1],
                         "link failed: usb0 has no address 192.168.250.1/24")
        self.assertEqual(usb_link.check(self.paths, FakeIp(["192.168.250.1/24"]))[1],
                         "link failed: usb0 is administratively down")
        (self.paths.net / "usb0" / "flags").write_text("0x1003\n")
        self.assertEqual(usb_link.check(self.paths, FakeIp(["192.168.250.1/24"])),
                         (usb_link.CHECK_NO_CARRIER,
                          "link waiting: usb0 configured but no carrier (cable unplugged or laptop not enumerated)"))

    def test_teardown_removes_only_this_gadget(self):
        other = self.paths.configfs / "usb_gadget" / "g1"
        other.mkdir()
        (other / "UDC").write_text("a800000.dwc3\n")
        self.kernel_assigns_interface()
        self.bring_up(FakeIp())
        usb_link.teardown(self.paths, remove_dir=configfs_rmdir)
        self.assertFalse(self.gadget.exists())
        self.assertEqual((other / "UDC").read_text(), "a800000.dwc3" + chr(10))
        usb_link.teardown(self.paths, remove_dir=configfs_rmdir)


if __name__ == "__main__":
    unittest.main()
