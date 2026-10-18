#!/usr/bin/env python3
"""
Vision Long USB link: a CDC-NCM gadget on the comma 4's aux port, 192.168.250.1/24.

    usb_link.py up      configure and bind the gadget, address the interface (root; idempotent)
    usb_link.py check   print "link ok 192.168.250.1 usb0 up" or the exact failure
    usb_link.py down    unbind and remove this gadget only (root; used by the rollback)

The gadget is built through configfs, following the kernel's
Documentation/usb/gadget_configfs.txt and Documentation/ABI/testing/configfs-usb-gadget-ncm
(4.9): create usb_gadget/<name>, set the IDs and strings, create functions/ncm.<instance>,
set dev_addr/host_addr, create configs/c.1, symlink the function into it, then write the UDC
name to UDC. On 4.9 `ifname` is read-only and only meaningful after the bind, so it is read,
never written.

Only UDC a600000.dwc3 is ever written. A UDC already bound to another gadget is refused, and
no other gadget directory is ever written or removed: the modem and anything else on the
other controller are left exactly as they are. Every rerun is a no-op on a live link, which is
what lets a systemd timer re-assert it after a cable re-plug or an address flush.
"""

from __future__ import annotations

import errno
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GADGET_NAME = "vision_long"
UDC = "a600000.dwc3"
FUNCTION = "ncm.usb0"
CONFIG = "c.1"
LANGUAGE = "0x409"
# Linux Foundation "Multifunction Composite Gadget": the IDs the kernel's own gadget docs use.
ID_VENDOR = "0x1d6b"
ID_PRODUCT = "0x0104"
BCD_DEVICE = "0x0100"
BCD_USB = "0x0200"
MANUFACTURER = "comma.ai"
PRODUCT = "Vision Long link"
SERIAL = "vision-long"
CONFIGURATION = "NCM"
MAX_POWER_MA = "250"
# Locally administered, fixed, so the laptop sees the same adapter every boot and keeps its
# static 192.168.250.2 configuration.
DEV_ADDR = "02:76:6c:00:00:01"
HOST_ADDR = "02:76:6c:00:00:02"
ADDRESS = "192.168.250.1"
PREFIX = "24"
IFF_UP = 0x1
LOG_FILE = Path("/data/vision_long/link.log")
# /data runs ~90% full and the timer reruns this every 30 s: the log is capped at two files of
# this size, and an outcome is written only when it differs from the last one.
LOG_MAX_BYTES = 256 * 1024
LOG_BACKUPS = 1
LOGGER = "vision_long_link"
log = logging.getLogger(LOGGER)

CHECK_OK = 0
CHECK_FAILED = 1
CHECK_NO_CARRIER = 3

Runner = Callable[[list[str]], str]


class LinkError(RuntimeError):
    """The link cannot be brought up; the message names the cause."""


@dataclass(frozen=True)
class Paths:
    configfs: Path = Path("/sys/kernel/config")
    udc_class: Path = Path("/sys/class/udc")
    net: Path = Path("/sys/class/net")
    params: Path = Path("/data/params/d")

    @property
    def gadgets(self) -> Path:
        return self.configfs / "usb_gadget"

    @property
    def gadget(self) -> Path:
        return self.gadgets / GADGET_NAME

    @property
    def function(self) -> Path:
        return self.gadget / "functions" / FUNCTION

    @property
    def config(self) -> Path:
        return self.gadget / "configs" / CONFIG


def run_command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise LinkError(f"{' '.join(argv)} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _read(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else ""


def _write(path: Path, value: str) -> None:
    if _read(path) != value:
        path.write_text(value + "\n")
        log.info("wrote %s = %s", path, value)


def _write_mac(path: Path, value: str) -> None:
    # The comma 4's 4.9 kernel answers every ncm attribute with ENODEV ("Gadget function do not
    # bind yet") until the gadget is bound, and refuses writes once it is. The MACs are optional
    # (the kernel assigns its own), so only that refusal is skipped; any other error still fails.
    try:
        _write(path, value)
    except OSError as error:
        if error.errno != errno.ENODEV:
            raise
        log.info("%s unavailable before bind on this kernel; the kernel assigns the address", path.name)


def _mkdir(path: Path) -> None:
    if not path.is_dir():
        path.mkdir()
        log.info("created %s", path)


def udc_holder(paths: Paths) -> str | None:
    """Whatever else holds the UDC: another configfs gadget (AGNOS ADB is one), or a gadget
    driver bound outside configfs, which the UDC's own `function` attribute names."""
    if _read(paths.gadget / "UDC") == UDC:
        return None
    if paths.gadgets.is_dir():
        for gadget in sorted(paths.gadgets.iterdir()):
            if gadget.name != GADGET_NAME and _read(gadget / "UDC") == UDC:
                return gadget.name
    function = _read(paths.udc_class / UDC / "function")
    return function if function and function != GADGET_NAME else None


def blocked_reason(paths: Paths) -> str | None:
    holder = udc_holder(paths)
    if holder is None:
        return None
    adb = _read(paths.params / "AdbEnabled") or "absent"
    return f"link blocked: {UDC} held by {holder}; AdbEnabled={adb}"


def _configure(paths: Paths, link: Callable[[Path, Path], None]) -> None:
    # Each step is skipped when already done, so a run interrupted half way completes cleanly.
    # Attributes are never rewritten while bound: the kernel refuses that with EBUSY.
    gadget = paths.gadget
    _mkdir(gadget)
    for name, value in (("idVendor", ID_VENDOR), ("idProduct", ID_PRODUCT),
                        ("bcdDevice", BCD_DEVICE), ("bcdUSB", BCD_USB)):
        _write(gadget / name, value)
    strings = gadget / "strings"
    _mkdir(strings)
    _mkdir(strings / LANGUAGE)
    for name, value in (("manufacturer", MANUFACTURER), ("product", PRODUCT), ("serialnumber", SERIAL)):
        _write(strings / LANGUAGE / name, value)
    _mkdir(gadget / "functions")
    _mkdir(paths.function)
    for name, value in (("dev_addr", DEV_ADDR), ("host_addr", HOST_ADDR)):
        _write_mac(paths.function / name, value)
    _mkdir(gadget / "configs")
    _mkdir(paths.config)
    _mkdir(paths.config / "strings")
    _mkdir(paths.config / "strings" / LANGUAGE)
    _write(paths.config / "strings" / LANGUAGE / "configuration", CONFIGURATION)
    _write(paths.config / "MaxPower", MAX_POWER_MA)
    member = paths.config / FUNCTION
    if not member.exists() and not member.is_symlink():
        link(paths.function, member)
        log.info("linked %s into %s", FUNCTION, CONFIG)


def _addresses(interface: str, run: Runner) -> set[str]:
    output = run(["ip", "-4", "-o", "addr", "show", "dev", interface])
    found = set()
    for line in output.splitlines():
        fields = line.split()
        if "inet" in fields:
            found.add(fields[fields.index("inet") + 1])
    return found


def interface_name(paths: Paths) -> str:
    name = _read(paths.function / "ifname")
    if not name or not (paths.net / name).is_dir():
        raise LinkError(f"no network interface after binding: {paths.function / 'ifname'} reads {name!r}")
    return name


def bring_up(paths: Paths, run: Runner, nmcli_present: bool,
             link: Callable[[Path, Path], None] = os.symlink) -> str:
    """Configure, bind and address the link. Returns the interface name."""
    if not paths.gadgets.is_dir():
        raise LinkError(f"configfs is not mounted: {paths.gadgets} is missing")
    if not (paths.udc_class / UDC).exists():
        raise LinkError(f"UDC {UDC} is not present in {paths.udc_class}")
    blocked = blocked_reason(paths)
    if blocked is not None:
        raise LinkError(blocked)
    if _read(paths.gadget / "UDC") != UDC:
        _configure(paths, link)
        _write(paths.gadget / "UDC", UDC)
    interface = interface_name(paths)
    if nmcli_present:
        # NetworkManager would otherwise run DHCP on the new interface and flush our address.
        run(["nmcli", "device", "set", interface, "managed", "no"])
    if f"{ADDRESS}/{PREFIX}" not in _addresses(interface, run):
        run(["ip", "addr", "add", f"{ADDRESS}/{PREFIX}", "dev", interface])
        log.info("addressed %s with %s/%s", interface, ADDRESS, PREFIX)
    run(["ip", "link", "set", interface, "up"])
    return interface


def check(paths: Paths, run: Runner) -> tuple[int, str]:
    blocked = blocked_reason(paths)
    if blocked is not None:
        return CHECK_FAILED, blocked
    if not paths.function.is_dir():
        return CHECK_FAILED, f"link failed: gadget {GADGET_NAME} is not configured"
    if _read(paths.gadget / "UDC") != UDC:
        return CHECK_FAILED, f"link failed: gadget {GADGET_NAME} is not bound to {UDC}"
    try:
        interface = interface_name(paths)
    except LinkError as error:
        return CHECK_FAILED, f"link failed: {error}"
    if f"{ADDRESS}/{PREFIX}" not in _addresses(interface, run):
        return CHECK_FAILED, f"link failed: {interface} has no address {ADDRESS}/{PREFIX}"
    if not int(_read(paths.net / interface / "flags") or "0", 16) & IFF_UP:
        return CHECK_FAILED, f"link failed: {interface} is administratively down"
    if _read(paths.net / interface / "operstate") != "up":
        return (CHECK_NO_CARRIER,
                f"link waiting: {interface} configured but no carrier (cable unplugged or laptop not enumerated)")
    return CHECK_OK, f"link ok {ADDRESS} {interface} up"


def teardown(paths: Paths, remove_dir: Callable[[Path], None] = os.rmdir) -> None:
    """Unbind and remove this gadget, in the reverse order configfs requires. Others untouched."""
    gadget = paths.gadget
    if not gadget.is_dir():
        return
    if _read(gadget / "UDC"):
        (gadget / "UDC").write_text("\n")
        log.info("unbound %s", GADGET_NAME)
    member = paths.config / FUNCTION
    if member.is_symlink() or member.exists():
        member.unlink()
    for directory in (paths.config / "strings" / LANGUAGE, paths.config, paths.function,
                      gadget / "strings" / LANGUAGE):
        if directory.is_dir():
            remove_dir(directory)
    remove_dir(gadget)
    log.info("removed gadget %s", GADGET_NAME)


def _log_outcome(log_file: Path, outcome: str) -> None:
    """Write the outcome line only when it differs from the last one written."""
    last = log_file.with_suffix(".last")
    if _read(last) == outcome:
        return
    log.info("%s", outcome)
    last.write_text(outcome + "\n")


def execute(command: str, paths: Paths, log_file: Path, run: Runner, nmcli_present: bool,
            link: Callable[[Path, Path], None] = os.symlink, remove_dir: Callable[[Path], None] = os.rmdir,
            max_bytes: int = LOG_MAX_BYTES) -> int:
    """Run `up` or `down`, logging steps taken and the outcome as one line each."""
    log_file.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=LOG_BACKUPS)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    try:
        try:
            if command == "up":
                bring_up(paths, run, nmcli_present, link)
                code, outcome = check(paths, run)
            else:
                teardown(paths, remove_dir)
                code, outcome = CHECK_OK, f"gadget {GADGET_NAME} removed"
        except (LinkError, OSError) as error:
            code, outcome = CHECK_FAILED, " ".join(str(error).split())
            if not outcome.startswith("link "):
                outcome = f"link failed: {outcome}"
        _log_outcome(log_file, outcome)
    finally:
        log.removeHandler(handler)
        handler.close()
    print(outcome, file=sys.stderr if code == CHECK_FAILED else sys.stdout)
    return code


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("up", "check", "down"):
        print("usage: usb_link.py up|check|down", file=sys.stderr)
        return 2
    paths = Paths()
    if argv[1] == "check":
        code, text = check(paths, run_command)
        print(text)
        return code
    return execute(argv[1], paths, LOG_FILE, run_command, nmcli_present=shutil.which("nmcli") is not None)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
