#!/usr/bin/env python3
"""
Every path this branch changes outside .elantra/ must be registered in the sync overlay.

The weekly rebuild hard-resets to upstream sunnypilot and replays only what .elantra/sync.py
lists: OVERLAY_ADDED is restored wholesale, OVERLAY_MODIFIED is replayed as a diff. A file
that differs from upstream but is in neither list is deleted (or reverted) on the next
rebuild with no error at all. This compares the working tree, plus untracked files, against
the upstream commit named in build-manifest.json, so committed, staged, unstaged and new
files all count.

    python .elantra/test_overlay_registration.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The submodule gitlink is pinned by sync.py itself, never replayed through the overlay.
PINNED_BY_SYNC = {"opendbc_repo"}


def load_sync():
    spec = importlib.util.spec_from_file_location("elantra_sync", HERE / "sync.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load .elantra/sync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_paths(*args: str) -> set[str]:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return {item.decode() for item in result.stdout.split(b"\0") if item}


def upstream_commit() -> str:
    manifest = json.loads((HERE / "build-manifest.json").read_text(encoding="utf-8"))
    commit = manifest["sunnypilot_upstream_sha"]
    probe = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit + "^{commit}"],
                           capture_output=True, check=False)
    if probe.returncode != 0:
        raise RuntimeError(f"upstream base {commit} is not in this clone; fetch it before running this guard")
    return commit


def is_registered(path: str, added: list[str], modified: list[str]) -> bool:
    if path in modified or path in PINNED_BY_SYNC:
        return True
    return any(path == entry or path.startswith(entry + "/") for entry in added)


def unregistered_paths() -> list[str]:
    sync = load_sync()
    changed = git_paths("diff", "--name-only", "-z", upstream_commit(), "--")
    changed |= git_paths("ls-files", "--others", "--exclude-standard", "-z")
    return sorted(path for path in changed
                  if not path.startswith(".elantra/")
                  and not is_registered(path, sync.OVERLAY_ADDED, sync.OVERLAY_MODIFIED))


class OverlayRegistrationTests(unittest.TestCase):
    def test_every_changed_path_outside_elantra_is_registered(self):
        self.assertEqual(unregistered_paths(), [],
                         "these paths would be deleted or reverted by the next sync rebuild; "
                         + "add them to OVERLAY_ADDED or OVERLAY_MODIFIED in .elantra/sync.py")

    def test_a_registered_directory_covers_its_files_but_not_a_sibling_prefix(self):
        added = [".github/workflows/elantra-sync.yaml", "openpilot/a"]
        self.assertTrue(is_registered("openpilot/a/b.py", added, []))
        self.assertFalse(is_registered("openpilot/ab.py", added, []))
        self.assertTrue(is_registered("opendbc_repo", [], []))
        self.assertFalse(is_registered("openpilot/c.py", added, ["openpilot/c.pyx"]))


if __name__ == "__main__":
    unittest.main()
