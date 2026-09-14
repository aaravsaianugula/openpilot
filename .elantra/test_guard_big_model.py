#!/usr/bin/env python3
"""
Negative tests for guard_big_model -- the guard over the cinque terre big driving model.

Why this test exists. Every way this change reverts leaves a tree that looks right:

  * the weekly rebuild hard-resets to upstream and an unregistered path never comes back. The
    file is still there, still 134 bytes, still a valid pointer -- just naming the old model.
    modeld loads it without comment. On master's sync.py there is not even a coverage detector
    to notice, so a literal-oid guard is the only thing standing between a silent revert and a
    road test of the wrong model;
  * the registration survives but .gitattributes stops routing .onnx through LFS, after which
    the next commit tries to put 766 MB into git itself;
  * the pointer is perfect and nothing can resolve it, because the seeder that fetches the
    object out of comma's store was dropped. Under filter.lfs.required=true that is not a
    degraded model, it is a failed checkout and a device that can no longer update.

A guard that only ever passes is not a guard. Each case builds a throwaway git repo, applies
exactly one mutation, and requires the guard to go RED for it.

    python .elantra/test_guard_big_model.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

GUARDS = Path(__file__).resolve().parent / "guards.py"

GITATTRIBUTES = "* text=auto\n*.onnx filter=lfs diff=lfs merge=lfs -text\n"
VERSION_LINE = "version https://git-lfs.github.com/spec/v1\n"
SYNC_STUB = ("OVERLAY_MODIFIED = [\n"
             + '    "openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx",\n'
             + "]\n")


def load_guards():
    spec = importlib.util.spec_from_file_location("guards", GUARDS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(repo: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=str(repo), check=True,
                   capture_output=True, text=True)


def pointer(oid: str, size: int) -> str:
    return VERSION_LINE + "oid sha256:" + oid + "\n" + "size " + str(size) + "\n"


def make_repo(root: Path, name: str, g, *, oid=None, size=None,
              gitattributes=GITATTRIBUTES, sync=SYNC_STUB, seeder=True) -> Path:
    """A minimal tree carrying only what guard_big_model reads."""
    repo = root / name
    (repo / ".elantra").mkdir(parents=True)
    (repo / "openpilot/selfdrive/modeld/models").mkdir(parents=True)

    (repo / ".gitattributes").write_text(gitattributes, encoding="utf-8")
    (repo / ".elantra/sync.py").write_text(sync, encoding="utf-8")
    if seeder:
        (repo / ".elantra/fetch_lfs_object.py").write_text("# seeder\n", encoding="utf-8")
    (repo / g.BIG_MODEL_PATH).write_text(
        pointer(oid or g.BIG_MODEL_OID, g.BIG_MODEL_SIZE if size is None else size),
        encoding="utf-8")

    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    # A pointer survives git-lfs's clean filter untouched, but do not depend on git-lfs being
    # installed at all: this test must run on a bare Python 3 box like the guards themselves.
    git(repo, "config", "filter.lfs.clean", "cat")
    git(repo, "config", "filter.lfs.smudge", "cat")
    git(repo, "config", "--bool", "filter.lfs.required", "false")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "tree")
    return repo


def run_case(g, label: str, repo: Path, should_fail: bool, failures: list) -> None:
    g._failures.clear()
    g._passes.clear()
    try:
        g.guard_big_model(repo)
    # Guards must never raise: sync.py reads a non-zero exit as "do not publish", so an
    # exception here would be indistinguishable from a real divergence.
    except Exception as e:
        failures.append(label + ": guard raised " + type(e).__name__ + ": " + str(e))
        print("  FAIL  " + label + ": guard raised " + type(e).__name__)
        return

    failed = bool(g._failures)
    if failed == should_fail:
        print("  ok    " + label)
        return
    want = "fail" if should_fail else "pass"
    failures.append(label + ": expected the guard to " + want)
    print("  FAIL  " + label + ": expected the guard to " + want)


def main() -> int:
    g = load_guards()
    failures: list = []
    print("guard_big_model")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        run_case(g, "an intact tree passes",
                 make_repo(root, "good", g), False, failures)

        # The headline failure: the rebuild dropped the overlay entry and restored upstream's
        # model. Same path, same size, valid pointer, wrong car behaviour.
        run_case(g, "pointer reverted to the previous model fails",
                 make_repo(root, "reverted", g, oid=g.PREV_BIG_MODEL_OID), True, failures)

        run_case(g, "pointer naming some other object fails",
                 make_repo(root, "other", g, oid="b" * 64), True, failures)

        run_case(g, "wrong size fails",
                 make_repo(root, "size", g, size=g.BIG_MODEL_SIZE - 1), True, failures)

        notptr = root / "notpointer"
        (notptr / ".elantra").mkdir(parents=True)
        (notptr / "openpilot/selfdrive/modeld/models").mkdir(parents=True)
        (notptr / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
        (notptr / ".elantra/sync.py").write_text(SYNC_STUB, encoding="utf-8")
        (notptr / ".elantra/fetch_lfs_object.py").write_text("# seeder\n", encoding="utf-8")
        (notptr / g.BIG_MODEL_PATH).write_bytes(b"\x08\x20\n\x12\x07pytorch raw onnx bytes")
        git(notptr, "init", "-q")
        git(notptr, "config", "user.email", "test@example.com")
        git(notptr, "config", "user.name", "test")
        git(notptr, "config", "filter.lfs.clean", "cat")
        git(notptr, "config", "--bool", "filter.lfs.required", "false")
        git(notptr, "add", "-A")
        git(notptr, "commit", "-q", "-m", "raw onnx committed outside lfs")
        run_case(g, "raw ONNX committed outside LFS fails", notptr, True, failures)

        run_case(g, "unregistered in OVERLAY_MODIFIED fails",
                 make_repo(root, "unregistered", g,
                           sync="OVERLAY_MODIFIED = [\n]\n"), True, failures)

        run_case(g, ".onnx no longer routed through LFS fails",
                 make_repo(root, "noattr", g, gitattributes="* text=auto\n"), True, failures)

        run_case(g, "missing LFS seeder fails",
                 make_repo(root, "noseeder", g, seeder=False), True, failures)

        # An empty directory is not a checkout. The guard must report that, not explode.
        bare = root / "bare"
        bare.mkdir()
        run_case(g, "a non-repo fails without raising", bare, True, failures)

    print("\n" + "-" * 58)
    if failures:
        print("FAILED: " + str(len(failures)) + " case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: the big model guard goes red on every way this change silently reverts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
