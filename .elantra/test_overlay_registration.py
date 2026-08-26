#!/usr/bin/env python3
"""
Prove the overlay registry still describes this branch exactly.

The sync rebuilds master by hard-resetting to an upstream commit and putting back only what
.elantra/overlay.py names. So the registry is not documentation -- it is the definition of
what survives. The invariant:

    OVERLAY_ADDED + OVERLAY_MODIFIED + OVERLAY_GITLINKS is *exactly* the set of paths by
    which this branch differs from the upstream commit it was built from.

Checked in both directions, because the two ways to get it wrong fail very differently:

  * a file on the branch that nobody registered is deleted next Monday, silently -- there is
    no error, because from the rebuild's point of view the file never existed
  * a path registered that does not exist blocks the sync *entirely*, because restore_paths()
    cannot check out what was never committed. Louder, and therefore less dangerous.

Runs on a bare Python 3 in a git checkout. No network, no dependencies.

Usage:
    python .elantra/test_overlay_registration.py [--repo <superproject>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import OVERLAY_ADDED, OVERLAY_GITLINKS, OVERLAY_MODIFIED

_failures: list[str] = []
_passes: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    suffix = ": " + detail if detail else ""
    if condition:
        _passes.append(label)
        print("  ok    " + label)
    else:
        _failures.append(label + suffix)
        print("  FAIL  " + label + suffix)


def git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(["git"] + args, cwd=repo, text=True, capture_output=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit("git " + " ".join(args) + " failed:\n" + (proc.stderr or "").strip())
    return proc.stdout.strip()


def covered_by(path: str, entries: list[str]) -> bool:
    """Is this file named by a registry entry, directly or via a directory entry?"""
    return any(path == e or path.startswith(e.rstrip("/") + "/") for e in entries)


def changed(repo: Path, base: str, status: str) -> list[str]:
    out = git(["diff", "--name-only", "--diff-filter=" + status, base, "HEAD"], repo)
    return [p for p in out.splitlines() if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="superproject checkout (default: the one containing this script)")
    args = ap.parse_args()
    repo = args.repo.resolve()

    manifest_path = repo / ".elantra/build-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("no .elantra/build-manifest.json in " + str(repo)
                         + " -- this branch was not produced by sync.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest["sunnypilot_upstream_sha"]

    print("overlay registration")
    print("  repo:  " + str(repo))
    print("  base:  " + base[:9] + "  (the upstream commit this branch was built from)")

    # If the manifest and the branch disagree about the base, every comparison below is
    # meaningless -- so establish that first rather than reporting a hundred bogus failures.
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"],
                              cwd=repo, capture_output=True)
    if ancestor.returncode != 0:
        raise SystemExit("the manifest base " + base[:9] + " is not an ancestor of HEAD. "
                         + "Fetch upstream, or rebuild -- the branch and its manifest disagree "
                         + "about what this was built from.")

    added = changed(repo, base, "A")
    modified = changed(repo, base, "M")
    deleted = changed(repo, base, "D")
    print("  diff:  " + str(len(added)) + " added, " + str(len(modified)) + " modified, "
          + str(len(deleted)) + " deleted")
    print()

    # 1. every added file is registered -- the headline failure mode
    strays = [p for p in added if not covered_by(p, OVERLAY_ADDED)]
    extra = " (+" + str(len(strays) - 6) + " more)" if len(strays) > 6 else ""
    check("every added file is registered", not strays, ", ".join(strays[:6]) + extra)
    if strays:
        print()
        print("  These files are on this branch but not in OVERLAY_ADDED. The sync rebuilds")
        print("  master by resetting to upstream and restoring only what OVERLAY_ADDED names,")
        print("  so they will be deleted -- with no error, because from the sync's point of")
        print("  view they were never there.")
        print()
        print("  Fix: add each path, or a directory containing it, to OVERLAY_ADDED in")
        print("  .elantra/overlay.py, in the same commit. See CLAUDE.md.")
        print()

    # 2. every registered added path actually exists -- a ghost path blocks the sync entirely
    known = set(added) | set(git(["ls-tree", "-r", "--name-only", "HEAD"], repo).splitlines())
    ghosts = [e for e in OVERLAY_ADDED
              if not any(f == e or f.startswith(e.rstrip("/") + "/") for f in known)]
    check("every registered path exists on the branch", not ghosts, ", ".join(ghosts))

    # 3. every modified upstream file is registered
    unregistered = [p for p in modified
                    if p not in OVERLAY_MODIFIED and p not in set(OVERLAY_GITLINKS)
                    and not covered_by(p, OVERLAY_ADDED)]
    check("every modified upstream file is registered", not unregistered,
          ", ".join(unregistered))

    # 4. every registered modification is still present in the diff -- a registered file whose
    #    edit was lost shrinks the derived overlay patch and the hook silently disappears
    lost = [e for e in OVERLAY_MODIFIED if e not in modified]
    check("every registered modification is still applied", not lost, ", ".join(lost))

    # Deleting an upstream file is not something the overlay supports: the rebuild starts from
    # upstream, so a deletion would be silently undone every week.
    check("no upstream files are deleted", not deleted, ", ".join(deleted))

    # Registry self-consistency -- cheap, and catches copy-paste mistakes in overlay.py itself
    overlap = sorted(set(OVERLAY_ADDED) & set(OVERLAY_MODIFIED))
    check("no path is both added and modified", not overlap, ", ".join(overlap))
    nested = sorted({b for a in OVERLAY_ADDED for b in OVERLAY_ADDED
                     if a != b and b.startswith(a.rstrip("/") + "/")})
    check("no OVERLAY_ADDED entry nests inside another", not nested, ", ".join(nested))
    check("gitlinks are not also in OVERLAY_MODIFIED",
          not (set(OVERLAY_GITLINKS) & set(OVERLAY_MODIFIED)),
          "a gitlink is an index entry, not text the overlay patch can carry")

    print("\n" + "-" * 60)
    if _failures:
        print("FAILED: " + str(len(_failures)) + " check(s) failed, "
              + str(len(_passes)) + " passed\n")
        for f in _failures:
            print("  - " + f)
        print("\nThe registry does not describe this branch. Fix it before pushing, or the")
        print("next sync will silently drop code.")
        return 1
    print("PASSED: all " + str(len(_passes)) + " checks green -- the registry matches the branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
