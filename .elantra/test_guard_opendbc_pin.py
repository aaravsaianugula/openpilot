#!/usr/bin/env python3
"""
Tests for guard_opendbc_pin -- the guard that ties a superproject branch to the opendbc
commit it claims to guard.

Why this test exists. Every other guard reads files out of --opendbc; none of them look at
the gitlink. That gap was not theoretical. The elantra-torque-test pair carried 48 green
guards asserting a 409 low-speed torque schedule while the superproject pinned 69e2e548,
which contains none of it -- `git diff master elantra-torque-test -- opendbc_repo` was
empty. Every gate passed and the car would have run stock code.

A guard that only ever passes is not a guard, so the cases below build real throwaway git
repos and prove it FAILS when the pin diverges, when the gitlink is a plain file instead of
a submodule, and when --opendbc is not a checkout at all.

    python .elantra/test_guard_opendbc_pin.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

GUARDS = Path(__file__).resolve().parent / "guards.py"


def load_guards():
    spec = importlib.util.spec_from_file_location("elantra_guards", GUARDS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def make_repo(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    return repo


def commit(repo: Path, text: str) -> str:
    (repo / "f.txt").write_text(text, encoding="utf-8")
    git(repo, "add", "f.txt")
    git(repo, "commit", "-q", "-m", text)
    return subprocess.run(("git", "rev-parse", "HEAD"), cwd=str(repo),
                          capture_output=True, text=True, check=True).stdout.strip()


def pin(repo: Path, sha: str) -> None:
    """Write a mode-160000 gitlink without needing a real submodule checkout."""
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},opendbc_repo")


def run_case(g, label: str, repo: Path, opendbc: Path, want_fail: bool,
             failures: list[str]) -> None:
    g._failures.clear()
    g._passes.clear()
    g.guard_opendbc_pin(repo, opendbc)
    failed = bool(g._failures)
    if failed != want_fail:
        failures.append(f"{label}: expected {'FAIL' if want_fail else 'PASS'}, got " +
                        f"{'FAIL' if failed else 'PASS'}")
        print("  FAIL  " + label)
    else:
        print("  ok    " + label)


def main() -> int:
    g = load_guards()
    failures: list[str] = []
    print("guard_opendbc_pin")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        odbc = make_repo(root, "opendbc")
        sha_a = commit(odbc, "a")
        sha_b = commit(odbc, "b")

        # The only shape that may pass: the pin is the checkout the guards just read.
        matched = make_repo(root, "super_matched")
        commit(matched, "super")
        pin(matched, sha_b)
        run_case(g, "matching pin passes", matched, odbc, False, failures)

        # The real defect: guards validated one tree, the build ships another.
        stale = make_repo(root, "super_stale")
        commit(stale, "super")
        pin(stale, sha_a)
        run_case(g, "stale pin fails", stale, odbc, True, failures)

        # A regular file at opendbc_repo is mode 100644, not a gitlink. Treating that as
        # "no pin recorded" rather than reading field 1 blindly is the point.
        notlink = make_repo(root, "super_notlink")
        (notlink / "opendbc_repo").write_text("not a submodule\n", encoding="utf-8")
        git(notlink, "add", "opendbc_repo")
        git(notlink, "commit", "-q", "-m", "file where a gitlink should be")
        run_case(g, "plain file instead of gitlink fails", notlink, odbc, True, failures)

        # No gitlink at all.
        empty = make_repo(root, "super_empty")
        commit(empty, "super")
        run_case(g, "missing gitlink fails", empty, odbc, True, failures)

        # --opendbc pointing somewhere that is not a checkout must not pass by accident,
        # and must not raise either.
        plain = root / "not_a_repo"
        plain.mkdir()
        run_case(g, "non-checkout --opendbc fails", matched, plain, True, failures)

    print("\n" + "-" * 58)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: the pin guard fails on every divergence it is meant to catch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
