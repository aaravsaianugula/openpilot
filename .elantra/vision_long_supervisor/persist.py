#!/usr/bin/env python3
"""
Install, roll back and re-establish the Vision Long systemd units on the comma 4.

    persist.py install       the laptop monitor's units (owner, over SSH as comma, parked)
    persist.py rollback      remove them again
    persist.py reestablish   put back any Vision Long unit an AGNOS update removed (boot hook)

`/` is read-only ext4 and A/B on AGNOS: an update flips the slot and every unit we put in
/etc/systemd/system is gone on the next boot. `reestablish` runs from the openpilot launch path
(reestablish.sh, started detached by launch_chffrplus.sh) and re-installs, from the shipped copies
in the checkout, every unit set that is expected but missing or different:
  - the monitor's three units, expected while /data/vision_long/supervisor.installed exists (this
    install writes it, this rollback removes it);
  - the USB link's two units (R4, vision_long_link), expected while the companion's runtime config
    exists (R4's install writes it, R4's rollback and undo remove it).
When everything is in place it changes nothing and never remounts. It never installs a set that
was not installed before, so a rollback stays rolled back.

Install runs every check before it changes anything and undoes its own changes on a failure,
in R4's style (vision_long_link/install.py; its link unit names are pinned here by a test).
Every write under /etc, `enable`'s wants links included, happens inside one remount,rw ...
remount,ro window, and the window always closes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from logcap import Log

# R4's USB link units (vision_long_link/install.py). Copied rather than imported, so the boot
# hook can never fail at import because of another file; a test pins every value equal to R4's.
HOSTNAME = "comma-6f0745e3"
PYTHON = "/usr/local/venv/bin/python3"
LINK_SOURCE = ".elantra/vision_long_link"
LINK_SCRIPT = ".elantra/vision_long_link/usb_link.py"
LINK_SERVICE = "vision-long-link.service"
LINK_TIMER = "vision-long-link.timer"
RUNTIME_CONFIG = ".elantra/vision_long/native-runtime.json"

SOURCE = ".elantra/vision_long_supervisor"
SERVICE = "vision-long-supervisor.service"
WATCH_SERVICE = "vision-long-supervisor-watch.service"
WATCH_TIMER = "vision-long-supervisor-watch.timer"
MIN_SYSTEMD = 231  # MemoryMax= (231); CPUQuota= (213) and StartLimitIntervalSec= in [Unit] (230) are older
MIN_FREE_BYTES = 16 << 20
REESTABLISH_LOG_BYTES = 256 * 1024
SETTLE_S = 3.0  # a crash at import shows within this; the unit's RestartSec is 5 s
WANTS = {".timer": "timers.target.wants", ".service": "multi-user.target.wants"}


@dataclass(frozen=True)
class Result:
    code: int
    out: str
    err: str


class System:
    """The device: host name, processes, free space and commands (R4's System, same shape)."""

    def hostname(self) -> str:
        return socket.gethostname()

    def processes(self) -> list[str]:
        return self.run(["ps", "-eo", "args"]).out.splitlines()

    def free_bytes(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> Result:
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
        return Result(done.returncode, done.stdout, done.stderr)


class Terminated(Exception):
    """SIGTERM (timeout 120 at boot) arrived. Raised so every `finally` runs and / goes back to ro."""


def _on_term(signum, frame):
    raise Terminated("SIGTERM")


def install_term_handler():
    """Turn SIGTERM into Terminated. Returns the previous handler."""
    return signal.signal(signal.SIGTERM, _on_term)


class PersistError(Exception):
    """A check failed. Nothing was changed, or the change was undone; the message says which."""


@dataclass(frozen=True)
class UnitSet:
    name: str
    source: str
    units: tuple[str, ...]
    enable: tuple[str, ...]
    start: tuple[str, ...]
    stop: tuple[str, ...]  # the timer that would start the service again goes first


SUPERVISOR = UnitSet("monitor", SOURCE, (SERVICE, WATCH_SERVICE, WATCH_TIMER), (SERVICE, WATCH_TIMER),
                     (WATCH_TIMER, SERVICE), (WATCH_TIMER, WATCH_SERVICE, SERVICE))
LINK = UnitSet("usb link", LINK_SOURCE, (LINK_SERVICE, LINK_TIMER), (LINK_TIMER,), (LINK_TIMER, LINK_SERVICE),
               (LINK_TIMER, LINK_SERVICE))


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def repo(self) -> Path:
        return self.root / "data/openpilot"

    @property
    def state(self) -> Path:
        return self.root / "data/vision_long"

    @property
    def key(self) -> Path:
        return self.state / "link.key"

    @property
    def marker(self) -> Path:
        return self.state / "supervisor.installed"

    @property
    def notice(self) -> Path:
        return self.state / "notice.json"

    @property
    def reestablish_log(self) -> Path:
        return self.state / "reestablish.log"

    @property
    def is_offroad(self) -> Path:
        return self.root / "data/params/d/IsOffroad"

    @property
    def systemd(self) -> Path:
        return self.root / "etc/systemd/system"

    @property
    def runtime_config(self) -> Path:
        return self.repo / RUNTIME_CONFIG

    @property
    def python(self) -> Path:
        return self.root / PYTHON.lstrip("/")

    def source(self, units: UnitSet, unit: str) -> Path:
        return self.repo / units.source / unit

    def wants(self, unit: str) -> Path:
        return self.systemd / WANTS[Path(unit).suffix] / unit


def _require(system: System, argv: list[str], what: str) -> Result:
    result = system.run(argv)
    if result.code != 0:
        tail = "\n".join((result.out + result.err).strip().splitlines()[-5:])
        raise PersistError(f"{what} failed (exit {result.code}):\n{tail}")
    return result


def unit_problems(layout: Layout, units: UnitSet) -> list[str]:
    """Why this set is not installed as shipped: a missing or different unit, or a missing wants link."""
    problems = []
    for unit in units.units:
        installed = layout.systemd / unit
        if not installed.is_file():
            problems.append(f"{unit} missing")
        elif installed.read_bytes() != layout.source(units, unit).read_bytes():
            problems.append(f"{unit} differs from the shipped copy")
    problems += [f"{unit} not enabled" for unit in units.enable if not layout.wants(unit).exists()]
    return problems


def install_units(system: System, layout: Layout, units: UnitSet) -> None:
    _require(system, ["sudo", "mount", "-o", "remount,rw", "/"], "remount / read-write")
    try:
        for unit in units.units:
            _require(system, ["sudo", "install", "-m", "0644", str(layout.source(units, unit)), str(layout.systemd / unit)],
                     "install " + unit)
        _require(system, ["sudo", "systemctl", "daemon-reload"], "systemctl daemon-reload")
        for unit in units.enable:
            _require(system, ["sudo", "systemctl", "enable", unit], "enable " + unit)
    finally:
        _require(system, ["sudo", "mount", "-o", "remount,ro", "/"], "remount / read-only")
    for unit in units.start:
        _require(system, ["sudo", "systemctl", "start", unit], "start " + unit)


def remove_units(system: System, layout: Layout, units: UnitSet) -> None:
    present = [unit for unit in units.units if (layout.systemd / unit).exists()]
    enabled = [unit for unit in units.enable if layout.wants(unit).exists()]
    if not present and not enabled:
        return
    for unit in units.stop:
        _require(system, ["sudo", "systemctl", "stop", unit], "stop " + unit)
    _require(system, ["sudo", "mount", "-o", "remount,rw", "/"], "remount / read-write")
    try:
        for unit in enabled:
            _require(system, ["sudo", "systemctl", "disable", unit], "disable " + unit)
        for unit in present:
            _require(system, ["sudo", "rm", "-f", str(layout.systemd / unit)], "remove " + unit)
    finally:
        _require(system, ["sudo", "mount", "-o", "remount,ro", "/"], "remount / read-only")
    _require(system, ["sudo", "systemctl", "daemon-reload"], "systemctl daemon-reload")


def _common(system: System, layout: Layout, failures: list[str]) -> None:
    offroad = layout.is_offroad.read_bytes() if layout.is_offroad.is_file() else b""
    if offroad != b"1":
        failures.append(f"the car is not offroad (IsOffroad is {offroad!r}); park and turn the ignition off")
    if system.hostname() != HOSTNAME:
        failures.append(f"hostname is {system.hostname()!r}, expected {HOSTNAME!r}")
    if any("rsync --server" in line for line in system.processes()):
        failures.append("an rsync transfer is running; wait for it to finish")
    if system.run(["sudo", "-n", "true"]).code != 0:
        failures.append("sudo needs a password; the units cannot be installed")


def _raise_if(failures: list[str], action: str) -> None:
    if failures:
        raise PersistError(f"{action} preflight failed, nothing was changed:\n  - " + "\n  - ".join(failures))


def systemd_version(system: System) -> int | None:
    result = system.run(["systemctl", "--version"])
    match = re.match(r"systemd (\d+)", result.out) if result.code == 0 else None
    return int(match.group(1)) if match else None


def preflight_install(system: System, layout: Layout) -> int:
    failures: list[str] = []
    _common(system, layout, failures)
    missing = [str(layout.source(SUPERVISOR, u)) for u in (*SUPERVISOR.units, "monitor.py") if not layout.source(SUPERVISOR, u).is_file()]
    if missing:
        failures.append("the checkout does not carry the monitor: " + ", ".join(missing))
    if not layout.python.is_file():
        failures.append(f"{PYTHON} is missing")
    if not layout.state.is_dir():
        failures.append(f"{layout.state} is missing; run the Vision Long install first")
    elif not layout.key.is_file() or layout.key.is_symlink() or layout.key.stat().st_size != 32:
        failures.append(f"{layout.key} is not a 32-byte link key; run the Vision Long install first")
    elif not os.access(layout.key, os.R_OK):
        failures.append(f"{layout.key} is not readable by this user; the monitor runs as comma (chown comma, chmod 600)")
    elif system.free_bytes(layout.state) < MIN_FREE_BYTES:
        failures.append(f"under {MIN_FREE_BYTES >> 20} MiB free on /data")
    version = systemd_version(system)
    if version is None or version < MIN_SYSTEMD:
        failures.append(f"systemd {version} is older than {MIN_SYSTEMD} (MemoryMax=)")
    if not missing:
        sources = [str(layout.source(SUPERVISOR, u)) for u in SUPERVISOR.units]
        verify = system.run(["systemd-analyze", "verify", *sources])
        if verify.code != 0 or "Unknown" in verify.out + verify.err:
            failures.append("systemd-analyze verify rejects the units: " + (verify.out + verify.err).strip()[-300:])
    _raise_if(failures, "install")
    return version or 0


def install(system: System, layout: Layout) -> list[str]:
    version = preflight_install(system, layout)
    try:
        install_units(system, layout, SUPERVISOR)
        time.sleep(SETTLE_S)
        state = system.run(["systemctl", "is-active", SERVICE]).out.strip()
        if state != "active":
            raise PersistError(f"{SERVICE} is {state or 'unknown'} after start")
        layout.marker.write_text("installed\n")
    except (PersistError, OSError) as error:
        try:
            remove_units(system, layout, SUPERVISOR)
            layout.marker.unlink(missing_ok=True)
        except (PersistError, OSError) as undo_error:
            raise PersistError(f"install failed: {error}\nUNDO ALSO FAILED: {undo_error}\n"
                               + "Run: bash .elantra/vision_long_supervisor/rollback.sh") from error
        raise PersistError(f"install failed: {error}\nUndone: the monitor units are removed.") from error
    return [
        f"systemd      {version}",
        f"units        {', '.join(SUPERVISOR.units)}",
        f"enabled      {', '.join(SUPERVISOR.enable)}",
        f"monitor      {SERVICE} active",
        f"marker       {layout.marker} (reestablish puts the units back after an AGNOS update)",
        f"log          {layout.state / 'supervisor.log'} (2 MB cap, one backup)",
    ]


def rollback(system: System, layout: Layout) -> list[str]:
    failures: list[str] = []
    _common(system, layout, failures)
    _raise_if(failures, "rollback")
    # The marker goes first: if removing the units fails half way, the boot hook must not put
    # the monitor back.
    layout.marker.unlink(missing_ok=True)
    remove_units(system, layout, SUPERVISOR)
    layout.notice.unlink(missing_ok=True)
    return ["units        removed", f"marker       {layout.marker} removed", f"kept         {layout.state} (key, logs)"]


def expected_sets(layout: Layout) -> list[UnitSet]:
    expected = []
    if layout.marker.is_file() and layout.source(SUPERVISOR, "monitor.py").is_file():
        expected.append(SUPERVISOR)
    if layout.runtime_config.is_file() and (layout.repo / LINK_SCRIPT).is_file():
        expected.append(LINK)
    return expected


def reestablish(system: System, layout: Layout) -> list[str]:
    """Re-install each expected unit set that is not in place. Returns one line per set."""
    lines = []
    for units in expected_sets(layout):
        problems = unit_problems(layout, units)
        if not problems:
            lines.append(f"{units.name}: in place")
            continue
        if system.run(["sudo", "-n", "true"]).code != 0:
            raise PersistError(f"{units.name}: {'; '.join(problems)}; sudo needs a password, cannot re-install")
        install_units(system, layout, units)
        lines.append(f"{units.name}: re-installed ({'; '.join(problems)})")
    return lines or ["nothing expected: no Vision Long units were installed"]


def main(argv: list[str], root: Path = Path("/"), system: System | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision Long systemd units (comma 4)")
    parser.add_argument("command", choices=("install", "rollback", "reestablish"))
    args = parser.parse_args(argv)
    layout, system = Layout(root), system or System()
    install_term_handler()
    if args.command == "reestablish":
        # The log opens first, so whatever goes wrong after it, at boot, is written down.
        log = Log(layout.reestablish_log, name="vision_long_reestablish", max_bytes=REESTABLISH_LOG_BYTES)
        try:
            for line in reestablish(system, layout):
                log.info(line)
        except Exception as error:  # the boot hook must record every failure, of any kind
            log.info(f"FAILED: {type(error).__name__}: {error}")
            return 1
        finally:
            log.close()
        return 0
    try:
        receipt = install(system, layout) if args.command == "install" else rollback(system, layout)
    except PersistError as error:
        print(f"\nVISION LONG MONITOR {args.command.upper()} STOPPED\n{error}", file=sys.stderr)
        return 1
    print(f"\nVISION LONG MONITOR {args.command.upper()} DONE\n  " + "\n  ".join(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
