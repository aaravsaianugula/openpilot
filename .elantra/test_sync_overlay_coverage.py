#!/usr/bin/env python3
"""Negative tests for sync.py's overlay-coverage check.

The weekly rebuild hard-resets to upstream and then restores ONLY what OVERLAY_ADDED and
OVERLAY_MODIFIED name. Anything else that the previous build had changed is not deleted with an
error -- it simply never comes back, with no output naming it. Before this check existed the
superproject had no detector for that at all: the opendbc half has had a `stray` allowlist since
the beginning, and the superproject half, which is where every production file lives, had none.
guards.py only ever checked six specific paths somebody had hand-written.

So this proves the detector actually fires, on real git repositories rather than on a mock, for
each way a file can go missing. A check nobody has watched fail is decoration.

    python .elantra/test_sync_overlay_coverage.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sync

FAILS: list[str] = []


def check(label, cond, detail=""):
    if cond:
        print("  ok    " + label)
    else:
        print("  FAIL  " + label + (": " + detail if detail else ""))
        FAILS.append(label)


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_repo(tmp, base_files, prev_files):
    """A throwaway repo with two commits: the upstream base, then the 'previous build'."""
    repo = Path(tmp) / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    for rel, text in base_files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    for rel, text in prev_files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    # --allow-empty: some cases add only a gitlink afterwards, and git refuses an empty commit
    git(repo, "commit", "-q", "--allow-empty", "-m", "previous build")
    prev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    return repo, base, prev


def coverage_raises(repo, base, prev):
    try:
        sync.assert_overlay_covers_everything(repo, base, prev)
        return None
    except sync.SyncError as ex:
        return str(ex)


# --------------------------------------------------------------------- the unit
def test_overlay_covers():
    print("_overlay_covers")
    check("an exactly registered upstream file is covered",
          sync._overlay_covers("openpilot/selfdrive/controls/lib/drive_helpers.py"))
    check("a file under the .elantra directory entry is covered",
          sync._overlay_covers(".elantra/guards.py"))
    check("a file in a NEW subdirectory of .elantra is covered",
          sync._overlay_covers(".elantra/probes/whatever.py"))
    check("the submodule gitlink is exempt, not a stray",
          sync._overlay_covers("opendbc_repo"))
    check("an unregistered upstream file is NOT covered",
          not sync._overlay_covers("openpilot/selfdrive/controls/controlsd.py"))
    check("a prefix near-miss is NOT covered",
          not sync._overlay_covers(".elantranew/guards.py"),
          "startswith without the separator would wrongly cover this")
    check("a sibling of a registered file is NOT covered",
          not sync._overlay_covers("openpilot/selfdrive/controls/lib/latcontrol.py"))


# --------------------------------------------------------- the check, on real repos
def test_registered_only_passes():
    print("a build that changed only registered paths")
    with tempfile.TemporaryDirectory() as tmp:
        repo, base, prev = build_repo(
            tmp,
            {"openpilot/selfdrive/controls/lib/drive_helpers.py": "x = 1\n",
             ".elantra/guards.py": "y = 1\n"},
            {"openpilot/selfdrive/controls/lib/drive_helpers.py": "x = 2\n",
             ".elantra/guards.py": "y = 2\n",
             ".elantra/probes/new_tool.py": "z = 1\n"})
        err = coverage_raises(repo, base, prev)
        check("passes when every changed path is registered", err is None, err or "")


def test_unregistered_upstream_edit_is_caught():
    print("an unregistered UPSTREAM file (the silent revert)")
    with tempfile.TemporaryDirectory() as tmp:
        repo, base, prev = build_repo(
            tmp,
            {"openpilot/selfdrive/controls/controlsd.py": "x = 1\n"},
            {"openpilot/selfdrive/controls/controlsd.py": "x = 2\n"})
        err = coverage_raises(repo, base, prev)
        check("the check goes red", err is not None,
              "an edit to an unregistered upstream file would be reverted silently")
        check("and it names the file", err is not None and "controlsd.py" in err, err or "")


def test_unregistered_new_file_is_caught():
    print("an unregistered NEW file (the silent delete)")
    with tempfile.TemporaryDirectory() as tmp:
        repo, base, prev = build_repo(
            tmp,
            {"openpilot/selfdrive/controls/lib/drive_helpers.py": "x = 1\n"},
            {"openpilot/selfdrive/controls/lib/drive_helpers.py": "x = 2\n",
             "openpilot/sunnypilot/selfdrive/controls/lib/brand_new_module.py": "q = 1\n"})
        err = coverage_raises(repo, base, prev)
        check("the check goes red", err is not None,
              "a new file outside .elantra would simply never come back")
        check("and it names the file", err is not None and "brand_new_module.py" in err, err or "")


def test_the_exact_pairing_failure_this_exists_for():
    """The schedule module registered, its call site not: the fix LOOKS installed."""
    print("half a coupled change (constants kept, call site reverted)")
    with tempfile.TemporaryDirectory() as tmp:
        repo, base, prev = build_repo(
            tmp,
            {"openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py": "g = 1\n",
             "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_base.py": "b = 1\n"},
            {"openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py": "g = 2\n",
             "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_base.py": "b = 2\n"})
        err = coverage_raises(repo, base, prev)
        check("an unregistered call site is caught", err is not None,
              "the constants survive and the only line that reads them is reverted -- "
              + "which looks exactly like the fix is still installed")
        check("and it names the call site",
              err is not None and "latcontrol_torque_ext_base.py" in err, err or "")


def test_the_gitlink_alone_does_not_trip_it():
    print("a build whose only change is the submodule pin")
    with tempfile.TemporaryDirectory() as tmp:
        # a gitlink cannot be made with plain file writes; use update-index directly
        repo, base, _ = build_repo(tmp, {".elantra/guards.py": "y = 1\n"}, {})
        git(repo, "update-index", "--add", "--cacheinfo",
            "160000,0000000000000000000000000000000000000001,opendbc_repo")
        git(repo, "commit", "-qm", "pin opendbc")
        prev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True).stdout.strip()
        err = coverage_raises(repo, base, prev)
        check("the gitlink is not reported as a stray", err is None, err or "")


def main():
    for fn in (test_overlay_covers, test_registered_only_passes,
               test_unregistered_upstream_edit_is_caught, test_unregistered_new_file_is_caught,
               test_the_exact_pairing_failure_this_exists_for,
               test_the_gitlink_alone_does_not_trip_it):
        fn()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
        return 1
    print("PASSED: the overlay-coverage check fires on every way a file can go missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
