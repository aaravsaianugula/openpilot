"""Source identity of the native checkout a publisher reads from (read-only Git queries).

Exactly two layouts are accepted:

* ``base``: the pinned openpilot and opendbc commits are ancestors of HEAD and the tracked
  native tree is byte-identical to them (changes elsewhere under ``.elantra/`` are allowed).
* ``overlay``: the same ancestry, every installed file the overlay receipt lists matches its
  SHA256 on disk, and every tracked difference from the pinned base is one of those files,
  the receipt itself, the ``opendbc_repo`` gitlink, or an ``.elantra/`` path outside the two
  companion packages.

msgq is not overlaid and stays an exact commit pin. A file cannot hold the hash of the
commit that contains it, so the receipt's per-file hashes are the identity, never HEAD.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Literal

from vision_long.native_context import BUILD_PINS

OVERLAY_RECEIPT = ".elantra/vision_long/overlay-receipt.json"
RECEIPT_SCHEMA = "vision_long_native_overlay_v1"
OPENDBC_DIRECTORY = "opendbc_repo"
COMPANION_PACKAGES = (".elantra/vision_long/", ".elantra/vision_long_publisher/")
INSTALLED_PREFIXES = (".elantra/", OPENDBC_DIRECTORY + "/")
# The only untracked file a companion package may hold: the explicit diagnostic opt-in.
UNTRACKED_RUNTIME_CONFIG = ".elantra/vision_long/native-runtime.json"
MAX_RECEIPT_BYTES = 64 * 1024
MAX_OVERLAY_FILE_BYTES = 4 * 1024 * 1024
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
GIT_NOT_ANCESTOR = 1


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, check=False)


def _git_output(directory: Path, *args: str) -> bytes:
    result = _git(directory, *args)
    if result.returncode != 0:
        raise RuntimeError(f"Native source query failed: git {args[0]} exit {result.returncode}")
    return result.stdout


def _paths(output: bytes) -> set[str]:
    return {item.decode() for item in output.split(b"\0") if item}


def _require_repository_root(directory: Path, name: str) -> None:
    # An uninitialized submodule directory would silently answer for the superproject instead.
    top = _git(directory, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.decode().strip()).resolve() != directory.resolve():
        raise RuntimeError("Native source is not its own Git checkout: " + name)


def _require_ancestor(directory: Path, commit: str, name: str) -> None:
    result = _git(directory, "merge-base", "--is-ancestor", commit, "HEAD")
    if result.returncode == GIT_NOT_ANCESTOR:
        raise RuntimeError("Native integration base is not an ancestor of HEAD: " + name)
    if result.returncode != 0:
        raise RuntimeError(f"Native integration base unavailable: {name} (git exit {result.returncode})")


def _tracked_changes(directory: Path, commit: str) -> set[str]:
    # Base tree against the working tree, so committed, staged and unstaged edits all count.
    # A submodule counts as changed when its checked-out commit differs from the base gitlink.
    return _paths(_git_output(directory, "diff", "--name-only", "-z", "--ignore-submodules=dirty", commit, "--"))


def _untracked_companion_files(root: Path) -> set[str]:
    # No --exclude-standard: an ignored file in a package the car process imports still counts.
    listed = _git_output(root, "ls-files", "--others", "-z", "--", *COMPANION_PACKAGES)
    return {
        path
        for path in _paths(listed)
        if path != UNTRACKED_RUNTIME_CONFIG
        and not (PurePosixPath(path).parent.name == "__pycache__" and path.endswith(".pyc"))
    }


def _relative_path(value: object) -> str:
    if type(value) is not str or not value.isascii() or "\\" in value:
        raise RuntimeError("Overlay receipt path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RuntimeError("Overlay receipt path is invalid")
    return value


def _installed_files(root: Path) -> dict[str, str]:
    receipt_path = root / OVERLAY_RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError("Overlay receipt missing: " + OVERLAY_RECEIPT)
    with receipt_path.open("rb") as stream:
        data = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(data) > MAX_RECEIPT_BYTES:
        raise RuntimeError("Overlay receipt exceeds its size bound")
    receipt = json.loads(data)
    pins = dict(BUILD_PINS)
    if (
        type(receipt) is not dict
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("reference_openpilot_commit") != pins["openpilot"]
        or receipt.get("reference_opendbc_commit") != pins["opendbc"]
        or type(receipt.get("output_files")) is not dict
    ):
        raise RuntimeError("Overlay receipt does not name the pinned base")
    installed = {}
    for path, digest in receipt["output_files"].items():
        relative = _relative_path(path)
        if type(digest) is not str or SHA256_HEX.fullmatch(digest) is None:
            raise RuntimeError("Overlay receipt digest is invalid: " + relative)
        if relative.startswith(INSTALLED_PREFIXES):
            installed[relative] = digest
    if not installed:
        raise RuntimeError("Overlay receipt lists no installed files")
    return installed


def _require_overlay_bytes(root: Path, installed: dict[str, str]) -> None:
    for relative, digest in installed.items():
        source = root / relative
        if not source.is_file() or source.stat().st_size > MAX_OVERLAY_FILE_BYTES:
            raise RuntimeError("Overlay file missing or oversized: " + relative)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Overlay file bytes differ from the receipt: " + relative)


def verify_native_source_identity(root: Path) -> Literal["base", "overlay"]:
    """Raise RuntimeError unless ``root`` is the pinned base or the pinned base plus the overlay."""
    pins = dict(BUILD_PINS)
    opendbc = root / OPENDBC_DIRECTORY
    for name, directory in (("openpilot", root), ("opendbc", opendbc), ("msgq", root / "msgq_repo")):
        _require_repository_root(directory, name)
    if _git_output(root / "msgq_repo", "rev-parse", "HEAD").decode().strip() != pins["msgq"]:
        raise RuntimeError("Native integration source identity changed: msgq")
    _require_ancestor(root, pins["openpilot"], "openpilot")
    _require_ancestor(opendbc, pins["opendbc"], "opendbc")
    untracked = _untracked_companion_files(root)
    if untracked:
        raise RuntimeError("Untracked file in a companion package: " + sorted(untracked)[0])
    opendbc_changes = _tracked_changes(opendbc, pins["opendbc"])
    native_changes = {
        path
        for path in _tracked_changes(root, pins["openpilot"])
        if path.startswith(COMPANION_PACKAGES) or not path.startswith(".elantra/")
    }
    if not native_changes and not opendbc_changes:
        return "base"
    installed = _installed_files(root)
    _require_overlay_bytes(root, installed)
    opendbc_prefix = OPENDBC_DIRECTORY + "/"
    allowed_opendbc = {path.removeprefix(opendbc_prefix) for path in installed if path.startswith(opendbc_prefix)}
    allowed_native = {path for path in installed if not path.startswith(opendbc_prefix)}
    allowed_native |= {OPENDBC_DIRECTORY, OVERLAY_RECEIPT}
    unexpected = sorted(native_changes - allowed_native) + sorted(
        opendbc_prefix + path for path in opendbc_changes - allowed_opendbc
    )
    if unexpected:
        raise RuntimeError("Native change outside the reviewed overlay: " + unexpected[0])
    return "overlay"
