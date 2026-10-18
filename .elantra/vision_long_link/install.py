#!/usr/bin/env python3
"""
Vision Long install and rollback on the comma 4, run by the owner over SSH as `comma`.

    install.py install --base-openpilot SHA --base-opendbc SHA --target-openpilot SHA --target-opendbc SHA
    install.py rollback

Entry points are .elantra/vision_long_install.sh and .elantra/vision_long_rollback.sh.

Install runs every check before it changes anything (the preflight of OWNER_CHECKLIST.md plus
the fetched branch, its template, sudo and the laptop monitor's own unit preflight, run on the
fetched copies). A device already on the target SHAs is a rerun: it re-verifies everything and
re-installs the units an AGNOS update may have removed. The USB link units and the laptop
monitor's units (.elantra/vision_long_supervisor, notification only) are installed through the
monitor's own installer (persist.py) inside one read-write window of `/`.
Any failure after the first change undoes what this run did: the checkout goes back to the
base and the runtime config and both unit sets are removed. The link key, the braking toggle and
the logs are never removed and never overwritten.

Rollback restores the saved SHAs, removes the runtime config, both unit sets, the monitor's
marker and the gadget, and keeps the key and the logs. Neither command uses `pkill -f`: the comma service is
restarted detached, which drops this SSH session by design.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

HOSTNAME = "comma-6f0745e3"
BRANCH = "vision-long-companion"
PYTHON = "/usr/local/venv/bin/python3"
DETACHED = "DETACHED"
MIN_FREE_BYTES = 1 << 30
KEY_BYTES = 32
KEY_MODE = 0o600
STATE_DIR_MODE = 0o755
FILE_MODE = 0o644
SHA = re.compile(r"[0-9a-f]{40}")
RUNTIME_CONFIG = ".elantra/vision_long/native-runtime.json"
# The runtime config is written from this template. The companion reads exactly the five
# native_runtime_v2 fields below and refuses any other, so the template must be exactly them.
TEMPLATE = ".elantra/vision_long_link/native-runtime.template.json"
# runtime_config's own bound (vision_long/native_bootstrap.py).
MAX_TEMPLATE_BYTES = 4096
UNIT_SOURCE = ".elantra/vision_long_link"
LINK_SCRIPT = ".elantra/vision_long_link/usb_link.py"
SERVICE = "vision-long-link.service"
TIMER = "vision-long-link.timer"
UNITS = (SERVICE, TIMER)
# The laptop monitor ships its own installer; its unit sets (persist.LINK pins the two above) are
# installed and removed through it. Its import needs logcap beside it.
MONITOR_SOURCE = ".elantra/vision_long_supervisor"
MONITOR_INSTALLER = MONITOR_SOURCE + "/persist.py"
MONITOR_MODULES = ("persist.py", "logcap.py")
CONTRACT = {
    "schema": "native_runtime_v2",
    "listen_ip": "192.168.250.1",
    "port": 45556,
    "key_file": "/data/vision_long/link.key",
    "toggle_file": "/data/vision_long/braking",
}
# Same values as usb_link.py (a test pins them equal). The installer runs from stdin on the
# base checkout, where usb_link.py does not exist yet, so it cannot import them.
UDC = "a600000.dwc3"
GADGET_NAME = "vision_long"
LINK_OK = 0
LINK_NO_CARRIER = 3
RESTART = 'setsid nohup sh -c "sleep 2; sudo systemctl restart comma" >/dev/null 2>&1 </dev/null &'
IDENTITY = ("from pathlib import Path; "
            + "from vision_long_publisher.native_source_identity import verify_native_source_identity as v; "
            + "print(v(Path({repo!r})))")


class InstallError(RuntimeError):
    """A check failed. Nothing was changed, or the change was undone; the message says which."""


@dataclass(frozen=True)
class Layout:
    """Every path the installer reads or writes. On the device the root is `/`."""

    root: Path

    @property
    def repo(self) -> Path:
        return self.root / "data/openpilot"

    @property
    def opendbc(self) -> Path:
        return self.repo / "opendbc_repo"

    @property
    def params(self) -> Path:
        return self.root / "data/params/d"

    @property
    def disable_updates(self) -> Path:
        return self.params / "DisableUpdates"

    @property
    def is_offroad(self) -> Path:
        return self.params / "IsOffroad"

    @property
    def configfs(self) -> Path:
        return self.root / "sys/kernel/config"

    @property
    def udc_class(self) -> Path:
        return self.root / "sys/class/udc"

    @property
    def state(self) -> Path:
        return self.root / "data/vision_long"

    @property
    def key(self) -> Path:
        return self.state / "link.key"

    @property
    def toggle(self) -> Path:
        return self.state / "braking"

    @property
    def rollback_openpilot(self) -> Path:
        return self.root / "data/vision-long-rollback-openpilot.sha"

    @property
    def rollback_opendbc(self) -> Path:
        return self.root / "data/vision-long-rollback-opendbc.sha"

    @property
    def rollback_branch(self) -> Path:
        return self.root / "data/vision-long-rollback-branch.txt"

    @property
    def runtime_config(self) -> Path:
        return self.repo / RUNTIME_CONFIG

    @property
    def systemd(self) -> Path:
        return self.root / "etc/systemd/system"


@dataclass(frozen=True)
class Result:
    code: int
    out: str
    err: str


class System:
    """The device: host name, processes, free space and commands."""

    def hostname(self) -> str:
        return socket.gethostname()

    def processes(self) -> list[str]:
        return self.run(["ps", "-eo", "args"]).out.splitlines()

    def free_bytes(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def run(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> Result:
        merged = None if env is None else {**os.environ, **env}
        done = subprocess.run(argv, cwd=cwd, env=merged, capture_output=True, text=True, check=False)
        return Result(done.returncode, done.stdout, done.stderr)


@dataclass(frozen=True)
class Shas:
    base_openpilot: str
    base_opendbc: str
    target_openpilot: str
    target_opendbc: str


# --- shared helpers ---------------------------------------------------------------------------

def _git(system: System, directory: Path, *args: str) -> str:
    result = system.run(["git", "-C", str(directory), *args])
    if result.code != 0:
        raise InstallError(f"git {' '.join(args)} failed in {directory}: {result.err.strip()}")
    return result.out.strip()


def _require(system: System, argv: list[str], what: str, cwd: Path | None = None,
             env: dict[str, str] | None = None) -> Result:
    result = system.run(argv, cwd=cwd, env=env)
    if result.code != 0:
        tail = "\n".join((result.out + result.err).strip().splitlines()[-5:])
        raise InstallError(f"{what} failed (exit {result.code}):\n{tail}")
    return result


def _status_lines(system: System, layout: Layout) -> list[str]:
    return [line for line in _git(system, layout.repo, "status", "--porcelain").splitlines() if line]


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _common_checks(system: System, layout: Layout, failures: list[str]) -> None:
    # Both commands restart the comma service, which must never happen while driving.
    offroad = layout.is_offroad.read_bytes() if layout.is_offroad.is_file() else b""
    if offroad != b"1":
        failures.append(f"the car is not offroad (IsOffroad is {offroad!r}); park and turn the ignition off")
    host = system.hostname()
    if host != HOSTNAME:
        failures.append(f"hostname is {host!r}, expected {HOSTNAME!r}")
    if any("rsync --server" in line for line in system.processes()):
        failures.append("an rsync transfer is running; wait for it to finish")
    if system.run(["sudo", "-n", "true"]).code != 0:
        failures.append("sudo needs a password; the link units cannot be installed")


def _read_text(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else ""


def link_blocked_reason(layout: Layout) -> str | None:
    """The same rule as usb_link.blocked_reason: another gadget or driver holds the UDC."""
    if _read_text(layout.configfs / "usb_gadget" / GADGET_NAME / "UDC") == UDC:
        return None
    holder = None
    gadgets = layout.configfs / "usb_gadget"
    if gadgets.is_dir():
        holder = next((g.name for g in sorted(gadgets.iterdir())
                       if g.name != GADGET_NAME and _read_text(g / "UDC") == UDC), None)
    if holder is None:
        function = _read_text(layout.udc_class / UDC / "function")
        holder = function if function and function != GADGET_NAME else None
    if holder is None:
        return None
    adb = _read_text(layout.params / "AdbEnabled") or "absent"
    return f"link blocked: {UDC} held by {holder}; AdbEnabled={adb}"


def _raise_if(failures: list[str], action: str) -> None:
    if failures:
        raise InstallError(f"{action} preflight failed, nothing was changed:\n  - " + "\n  - ".join(failures))


def _unique_pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise InstallError(f"runtime config template repeats {key!r} (duplicate keys are refused)")
        result[key] = value
    return result


def validate_template(raw: bytes) -> bytes:
    """The template must be a JSON object holding exactly the road-contract values, parsed as strictly as
    native_runtime_v2's runtime_config: bounded size, no duplicate key, and each value of the contract's own
    type (so 45556.0 is not the port). The parser refuses any extra field, so one would install a config
    that fails closed."""
    if not 0 < len(raw) <= MAX_TEMPLATE_BYTES:
        raise InstallError(f"runtime config template size {len(raw)} is outside 1..{MAX_TEMPLATE_BYTES} bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs)
    except ValueError as error:
        raise InstallError(f"runtime config template is not JSON: {error}") from error
    if type(value) is not dict:
        raise InstallError("runtime config template is not a JSON object")
    wrong = sorted(name for name, expected in CONTRACT.items()
                   if type(value.get(name)) is not type(expected) or value.get(name) != expected)
    if wrong:
        raise InstallError("runtime config template disagrees with the road contract on: " + ", ".join(wrong))
    extra = sorted(set(value) - set(CONTRACT))
    if extra:
        raise InstallError("runtime config template has fields native_runtime_v2 refuses: " + ", ".join(extra))
    return raw


def _identity(system: System, layout: Layout) -> None:
    env = {"PYTHONPATH": str(layout.repo / ".elantra")}
    result = _require(system, [PYTHON, "-c", IDENTITY.format(repo=str(layout.repo))], "identity check", env=env)
    if result.out.strip() != "overlay":
        raise InstallError(f"identity check printed {result.out.strip()!r}, expected 'overlay'")


def _restart(system: System) -> None:
    _require(system, ["sh", "-c", RESTART], "detached comma restart")


def _load_monitor_installer(directory: Path) -> ModuleType:
    """The laptop monitor's own installer (persist.py) from `directory`, which also holds logcap.py."""
    spec = importlib.util.spec_from_file_location("vision_long_monitor_installer", directory / "persist.py")
    if spec is None or spec.loader is None:
        raise InstallError(f"cannot load the monitor installer from {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module through sys.modules
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)  # persist.py puts its own folder on sys.path to import logcap
    finally:
        sys.path[:] = saved
    return module


def _fetched_monitor(system: System, layout: Layout, staging: Path, failures: list[str]) -> ModuleType | None:
    """Load the fetched branch's monitor installer from a scratch copy and run its unit preflight there.
    The fresh install runs on the base checkout, where none of it is on disk yet; nothing on the device
    changes. Returns the loaded installer, or None with the reason appended to `failures`."""
    source = staging / "data/openpilot" / MONITOR_SOURCE
    source.mkdir(parents=True)
    for name in MONITOR_MODULES:
        shown = system.run(["git", "-C", str(layout.repo), "show", f"FETCH_HEAD:{MONITOR_SOURCE}/{name}"])
        if shown.code != 0:
            failures.append(f"fetched {BRANCH} has no monitor installer ({MONITOR_SOURCE}/{name})")
            return None
        (source / name).write_text(shown.out)
    monitor = _load_monitor_installer(source)
    for name in (*monitor.SUPERVISOR.units, "monitor.py"):
        shown = system.run(["git", "-C", str(layout.repo), "show", f"FETCH_HEAD:{MONITOR_SOURCE}/{name}"])
        if shown.code == 0:
            (source / name).write_text(shown.out)
    monitor.unit_preflight(system, monitor.Layout(staging), failures)
    return monitor


# --- install ----------------------------------------------------------------------------------

def preflight_install(system: System, layout: Layout, shas: Shas) -> tuple[str, ModuleType]:
    """Every install check, before any change. Returns "fresh" or "rerun" and the fetched monitor
    installer; raises on failure."""
    failures: list[str] = []
    monitor: ModuleType | None = None
    for name, value in vars(shas).items():
        if SHA.fullmatch(value) is None:
            failures.append(f"{name} {value!r} is not a full 40-character commit SHA")
    _raise_if(failures, "install")
    _common_checks(system, layout, failures)

    head = _git(system, layout.repo, "rev-parse", "HEAD")
    opendbc_head = _git(system, layout.opendbc, "rev-parse", "HEAD")
    if (head, opendbc_head) == (shas.base_openpilot, shas.base_opendbc):
        mode = "fresh"
    elif (head, opendbc_head) == (shas.target_openpilot, shas.target_opendbc):
        mode = "rerun"
    else:
        mode = ""
        failures.append(f"checkout is {head[:8]} / {opendbc_head[:8]}, neither the base "
                        + f"{shas.base_openpilot[:8]} / {shas.base_opendbc[:8]} nor the target")

    allowed = {"?? " + RUNTIME_CONFIG} if mode == "rerun" else set()
    dirty = [line for line in _status_lines(system, layout) if line not in allowed]
    if dirty:
        failures.append("git status is not clean: " + "; ".join(dirty[:5]))

    fetched = system.run(["git", "-C", str(layout.repo), "rev-parse", "FETCH_HEAD"])
    if fetched.code != 0 or fetched.out.strip() != shas.target_openpilot:
        failures.append(f"fetched {BRANCH} is {fetched.out.strip()[:8] or 'missing'}, expected {shas.target_openpilot[:8]}")
    else:
        gitlink = _git(system, layout.repo, "ls-tree", "FETCH_HEAD", "opendbc_repo").split()
        if len(gitlink) < 3 or gitlink[2] != shas.target_opendbc:
            failures.append(f"fetched {BRANCH} does not pin opendbc {shas.target_opendbc[:8]}")
        template = system.run(["git", "-C", str(layout.repo), "show", "FETCH_HEAD:" + TEMPLATE])
        if template.code != 0:
            failures.append(f"fetched {BRANCH} has no {TEMPLATE}")
        else:
            try:
                validate_template(template.out.encode())
            except InstallError as error:
                failures.append(str(error))
        with tempfile.TemporaryDirectory(prefix="vision-long-monitor-") as staging:
            monitor = _fetched_monitor(system, layout, Path(staging), failures)

    if mode == "rerun":
        saved = (_read_sha(layout.rollback_openpilot), _read_sha(layout.rollback_opendbc))
        if saved != (shas.base_openpilot, shas.base_opendbc):
            failures.append("rerun on the target, but the saved rollback SHAs do not name the base")

    updates = layout.disable_updates.read_bytes() if layout.disable_updates.is_file() else b""
    if updates != b"1":
        failures.append(f"DisableUpdates is {updates!r}, must be b'1' or the updater could replace the checkout")
    free = system.free_bytes(layout.root / "data")
    if free < MIN_FREE_BYTES:
        failures.append(f"/data has {free >> 20} MiB free, need {MIN_FREE_BYTES >> 20} MiB")
    if layout.state.exists() and not layout.state.is_dir():
        failures.append(f"{layout.state} exists and is not a directory")
    if layout.key.exists() or layout.key.is_symlink():
        failures.extend(_key_problems(layout.key))
    blocked = link_blocked_reason(layout)
    if blocked is not None:
        failures.append(blocked + " (never taken over; turn ADB off in Developer settings and reboot first)")
    _raise_if(failures, "install")
    if monitor is None:
        raise InstallError("install preflight failed, nothing was changed: the monitor installer was not checked")
    return mode, monitor


def _read_sha(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else ""


def _key_problems(key: Path) -> list[str]:
    if key.is_symlink() or not key.is_file():
        return [f"{key} exists but is not a regular file; it is never overwritten"]
    problems = []
    if key.stat().st_size != KEY_BYTES:
        problems.append(f"{key} is {key.stat().st_size} bytes, expected {KEY_BYTES}; it is never overwritten")
    # Windows, where the tests run, has no Unix permission bits.
    if os.name == "posix" and key.stat().st_mode & 0o077:
        problems.append(f"{key} is readable by others (mode {key.stat().st_mode & 0o777:o}); chmod 600 it")
    return problems


def _provision_state(layout: Layout) -> tuple[str, bool]:
    """Create the key and the toggle only when absent. Returns the key's sha256 and the toggle."""
    layout.state.mkdir(mode=STATE_DIR_MODE, parents=True, exist_ok=True)
    if not layout.key.exists():
        descriptor = os.open(layout.key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, KEY_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(os.urandom(KEY_BYTES))
            stream.flush()
            os.fsync(stream.fileno())
    if not layout.toggle.exists():
        _write_atomic(layout.toggle, b"0", FILE_MODE)
    digest = hashlib.sha256(layout.key.read_bytes()).hexdigest()
    return digest, layout.toggle.read_bytes() == b"1"


def _install_units(system: System, layout: Layout, monitor: ModuleType) -> None:
    # `/` is read-only ext4 on AGNOS. The monitor's installer writes both sets (unit files and
    # `enable`'s wants links) inside one read-write window, then starts the link and the monitor.
    try:
        monitor.install_units(system, monitor.Layout(layout.root), monitor.LINK, monitor.SUPERVISOR)
    except monitor.PersistError as error:
        raise InstallError(str(error)) from error


def _remove_units(system: System, layout: Layout, monitor: ModuleType) -> None:
    # The marker goes first, so a half-done removal is never put back by `persist.py reestablish`.
    units = monitor.Layout(layout.root)
    units.marker.unlink(missing_ok=True)
    try:
        monitor.remove_units(system, units, monitor.SUPERVISOR, monitor.LINK)
    except monitor.PersistError as error:
        raise InstallError(str(error)) from error
    units.notice.unlink(missing_ok=True)


def _confirm_monitor(system: System, layout: Layout, monitor: ModuleType) -> None:
    try:
        monitor.confirm_monitor(system, monitor.Layout(layout.root))
    except monitor.PersistError as error:
        raise InstallError(str(error)) from error


def _link_check(system: System, layout: Layout) -> str:
    result = system.run([PYTHON, str(layout.repo / LINK_SCRIPT), "check"])
    if result.code not in (LINK_OK, LINK_NO_CARRIER):
        raise InstallError("USB link check: " + (result.out + result.err).strip())
    return result.out.strip()


def _verify_checkout(system: System, layout: Layout, openpilot: str, opendbc: str, allowed: set[str]) -> None:
    head = _git(system, layout.repo, "rev-parse", "HEAD")
    opendbc_head = _git(system, layout.opendbc, "rev-parse", "HEAD")
    gitlink = _git(system, layout.repo, "ls-tree", "HEAD", "opendbc_repo").split()
    if head != openpilot or opendbc_head != opendbc or len(gitlink) < 3 or gitlink[2] != opendbc:
        raise InstallError(f"checkout is {head[:8]} / {opendbc_head[:8]}, expected {openpilot[:8]} / {opendbc[:8]}")
    dirty = [line for line in _status_lines(system, layout) if line not in allowed]
    if dirty:
        raise InstallError("git status is not clean: " + "; ".join(dirty[:5]))


def _apply_install(system: System, layout: Layout, shas: Shas, mode: str, monitor: ModuleType) -> list[str]:
    env = {"PYTHONPATH": str(layout.repo)}
    if mode == "fresh":
        branch = system.run(["git", "-C", str(layout.repo), "symbolic-ref", "-q", "--short", "HEAD"])
        layout.rollback_openpilot.write_text(shas.base_openpilot + "\n")
        layout.rollback_opendbc.write_text(shas.base_opendbc + "\n")
        layout.rollback_branch.write_text((branch.out.strip() if branch.code == 0 else DETACHED) + "\n")
        _git(system, layout.repo, "checkout", "-q", "-B", BRANCH, "FETCH_HEAD")
        _git(system, layout.repo, "submodule", "update", "--init", "opendbc_repo")
    # A rerun keeps the config it finds (the companion test never reads it); it is rewritten below.
    present = {"?? " + RUNTIME_CONFIG} if mode == "rerun" else set()
    _verify_checkout(system, layout, shas.target_openpilot, shas.target_opendbc, present)
    _require(system, [PYTHON, ".elantra/guards.py", "--opendbc", "opendbc_repo", "--repo", "."],
             "guards", cwd=layout.repo, env=env)
    _require(system, [PYTHON, ".elantra/test_vision_long_companion.py"], "companion test", cwd=layout.repo, env=env)
    _identity(system, layout)

    key_digest, braking = _provision_state(layout)
    template = validate_template((layout.repo / TEMPLATE).read_bytes())
    _write_atomic(layout.runtime_config, template, FILE_MODE)
    _identity(system, layout)
    _install_units(system, layout, monitor)
    link = _link_check(system, layout)
    _confirm_monitor(system, layout, monitor)
    _verify_checkout(system, layout, shas.target_openpilot, shas.target_opendbc, {"?? " + RUNTIME_CONFIG})
    return [
        f"mode         {mode}",
        f"openpilot    {shas.target_openpilot}",
        f"opendbc      {shas.target_opendbc}",
        "guards       passed",
        "companion    test passed",
        "identity     overlay",
        f"runtime cfg  {layout.runtime_config}",
        f"link key     {layout.key} sha256 {key_digest}",
        f"braking      {'ON' if braking else 'off'} ({layout.toggle}, left as it was)",
        f"link units   {SERVICE}, {TIMER} enabled",
        f"usb link     {link}",
        f"monitor      {', '.join(monitor.SUPERVISOR.units)} installed, {monitor.SERVICE} active (notification only)",
        f"rollback     {layout.rollback_openpilot.parent}/vision-long-rollback-*.sha",
    ]


def _undo_install(system: System, layout: Layout, shas: Shas, mode: str, monitor: ModuleType) -> str:
    """Undo this run's changes after a failure. Key, toggle and logs stay."""
    _remove_units(system, layout, monitor)
    if layout.runtime_config.exists():
        layout.runtime_config.unlink()
    if mode == "rerun":
        return "companion disabled (no runtime config); checkout left on the target. Run the rollback to leave it."
    _checkout_saved(system, layout, shas.base_openpilot, _read_branch(layout))
    _git(system, layout.repo, "submodule", "update", "--init", "opendbc_repo")
    _verify_checkout(system, layout, shas.base_openpilot, shas.base_opendbc, set())
    return f"restored the base {shas.base_openpilot[:8]} / {shas.base_opendbc[:8]}."


def install(system: System, layout: Layout, shas: Shas) -> list[str]:
    mode, monitor = preflight_install(system, layout, shas)
    try:
        receipt = _apply_install(system, layout, shas, mode, monitor)
    except (InstallError, OSError) as error:
        try:
            undone = _undo_install(system, layout, shas, mode, monitor)
        except (InstallError, OSError) as undo_error:
            raise InstallError(f"install failed: {error}\nUNDO ALSO FAILED: {undo_error}\n"
                               + "Run .elantra/vision_long_rollback.sh, or the checklist's manual rollback.") from error
        raise InstallError(f"install failed: {error}\nUndone: {undone}") from error
    _restart(system)
    return receipt


# --- rollback ---------------------------------------------------------------------------------

def _read_branch(layout: Layout) -> str:
    return layout.rollback_branch.read_text().strip() if layout.rollback_branch.is_file() else DETACHED


def _checkout_saved(system: System, layout: Layout, sha: str, branch: str) -> None:
    if branch == DETACHED:
        _git(system, layout.repo, "checkout", "-q", "--detach", sha)
        return
    _git(system, layout.repo, "checkout", "-q", branch)
    if _git(system, layout.repo, "rev-parse", "HEAD") != sha:
        raise InstallError(f"branch {branch} no longer points at {sha[:8]}; check out {sha} by hand")


def preflight_rollback(system: System, layout: Layout) -> tuple[str, str, str, ModuleType]:
    failures: list[str] = []
    _common_checks(system, layout, failures)
    installer = layout.repo / MONITOR_INSTALLER
    if not installer.is_file():
        failures.append(f"{installer} is missing; the monitor units cannot be removed through it")
    openpilot, opendbc = _read_sha(layout.rollback_openpilot), _read_sha(layout.rollback_opendbc)
    for path, value, repo in ((layout.rollback_openpilot, openpilot, layout.repo),
                              (layout.rollback_opendbc, opendbc, layout.opendbc)):
        if SHA.fullmatch(value) is None:
            failures.append(f"{path} is missing or does not hold a commit SHA")
        elif system.run(["git", "-C", str(repo), "cat-file", "-e", value + "^{commit}"]).code != 0:
            failures.append(f"commit {value[:8]} from {path} is not in {repo}")
    dirty = [line for line in _status_lines(system, layout) if line != "?? " + RUNTIME_CONFIG]
    if dirty:
        failures.append("git status is not clean: " + "; ".join(dirty[:5]))
    _raise_if(failures, "rollback")
    return openpilot, opendbc, _read_branch(layout), _load_monitor_installer(installer.parent)


def rollback(system: System, layout: Layout) -> list[str]:
    openpilot, opendbc, branch, monitor = preflight_rollback(system, layout)
    _remove_units(system, layout, monitor)
    if (layout.repo / LINK_SCRIPT).is_file():
        _require(system, ["sudo", PYTHON, str(layout.repo / LINK_SCRIPT), "down"], "remove the USB gadget")
    if layout.runtime_config.exists():
        layout.runtime_config.unlink()
    _checkout_saved(system, layout, openpilot, branch)
    _git(system, layout.repo, "submodule", "update", "--init", "opendbc_repo")
    _verify_checkout(system, layout, openpilot, opendbc, set())
    _restart(system)
    return [
        f"openpilot    {openpilot} ({branch})",
        f"opendbc      {opendbc}",
        "runtime cfg  removed",
        "link units   removed, gadget removed",
        "monitor      units and marker removed",
        f"kept         {layout.key}, {layout.toggle}, logs",
    ]


# --- entry point ------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Vision Long install and rollback (comma 4)")
    commands = parser.add_subparsers(dest="command", required=True)
    install_parser = commands.add_parser("install")
    for name in ("base-openpilot", "base-opendbc", "target-openpilot", "target-opendbc"):
        install_parser.add_argument("--" + name, required=True)
    commands.add_parser("rollback")
    args = parser.parse_args(argv)
    layout = Layout(Path("/"))
    try:
        if args.command == "install":
            shas = Shas(args.base_openpilot, args.base_opendbc, args.target_openpilot, args.target_opendbc)
            receipt = install(System(), layout, shas)
        else:
            receipt = rollback(System(), layout)
    except InstallError as error:
        print(f"\nVISION LONG {args.command.upper()} STOPPED\n{error}", file=sys.stderr)
        return 1
    print(f"\nVISION LONG {args.command.upper()} DONE\n  " + "\n  ".join(receipt))
    print("\nThe comma service restarts in 2 s and this SSH session will drop. Reconnect after the UI is back.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
