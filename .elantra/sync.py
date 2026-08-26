#!/usr/bin/env python3
"""
Rebuild this branch from sunnypilot master with the Elantra 2024-25 port on top.

The branch is derived, never merged. Every run hard-resets to a chosen upstream commit and
replays a small overlay, so there is no merge history to rot -- which is precisely how the
community elantra-2024-port branch ends up months behind.

What the overlay is:
  * two files copied wholesale from the previous build (they are ours, they never conflict)
  * a ~26-line diff against four upstream files, applied three-way
  * the opendbc submodule URL and gitlink, pointed at our Elantra-enabled opendbc

Nothing is published unless every gate passes. If the overlay will not apply, or the guards
fail, or opendbc's own tests fail, the run aborts with the branch untouched. Losing Elantra
support quietly is the one outcome this script exists to prevent.

Usage:
    python .elantra/sync.py --dry-run          # build and validate, publish nothing
    python .elantra/sync.py                    # build, validate, publish
    python .elantra/sync.py --ref <sha>        # pin a specific upstream commit
    python .elantra/sync.py --allow-red-ci     # publish from a commit whose CI is not green
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import (
    FORK_TINYGRAD, NV_DELTA_PATHS, OVERLAY_ADDED, OVERLAY_GITLINKS, OVERLAY_MODIFIED,
    TINYGRAD_BRANCH, TINYGRAD_PORT_BRANCH, UPSTREAM_TINYGRAD,
)

UPSTREAM_REPO = "sunnypilot/sunnypilot"
FORK_REPO = "aaravsaianugula/openpilot"
UPSTREAM_OPENDBC = "sunnypilot/opendbc"
FORK_OPENDBC = "aaravsaianugula/opendbc"

# The community port branch. Its delta is recomputed from opendbc master on every run rather
# than replayed from a frozen patch, so upstream fixes to the port arrive automatically.
OPENDBC_PORT_BRANCH = "elantra-2024-port"

MAIN_BRANCH = "master"
ROLLBACK_BRANCH = "master-previous"
OPENDBC_BRANCH = "elantra"

# The platforms this branch exists to support. Used by the test gate so a selector that
# stops matching is caught rather than silently reducing coverage.
PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")


# check-run conclusions that mean "this commit is not safe to ship".
BAD_CONCLUSIONS = {"failure", "timed_out", "cancelled", "action_required", "stale"}
# A commit with barely any checks probably has not finished starting them.
MIN_CHECK_RUNS = 5
# How far back to look for a green commit before giving up.
COMMIT_SEARCH_DEPTH = 80

# opendbc's own tests need its dependency tree installed. CI always runs them; a local
# --dry-run on a machine without that environment does not, and says so rather than
# pretending the gate ran.
RUN_OPENDBC_TESTS = False


class SyncError(RuntimeError):
    """Anything that must abort the run with nothing published."""


def log(msg: str) -> None:
    print(msg, flush=True)


def run(args: list[str], cwd: Path | None = None, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, cwd=cwd, text=True,
                          capture_output=capture, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()
        raise SyncError(f"command failed ({proc.returncode}): {' '.join(args)}\n{detail}")
    return proc


def git(args: list[str], cwd: Path, check: bool = True) -> str:
    return run(["git"] + args, cwd=cwd, check=check).stdout.strip()


def gh_json(path: str):
    proc = run(["gh", "api", path], check=False)
    if proc.returncode != 0:
        raise SyncError(f"GitHub API call failed: {path}\n{(proc.stderr or '').strip()}")
    return json.loads(proc.stdout)


# --- upstream CI gate ---------------------------------------------------------------------

def ci_verdict(repo: str, sha: str) -> tuple[str, int]:
    """Return (conclusion, number_of_checks) for one commit.

    sunnypilot reports through check-runs, not legacy commit statuses -- the combined status
    endpoint returns "pending" with zero entries forever, so using it would make the gate a
    no-op. Skipped and neutral runs are fine; only real failures disqualify a commit.
    """
    data = gh_json(f"repos/{repo}/commits/{sha}/check-runs?per_page=100")
    runs = data.get("check_runs", [])
    if len(runs) < MIN_CHECK_RUNS:
        return "insufficient", len(runs)
    if any(r.get("status") != "completed" for r in runs):
        return "in_progress", len(runs)
    if any((r.get("conclusion") or "") in BAD_CONCLUSIONS for r in runs):
        return "failure", len(runs)
    return "success", len(runs)


def pick_upstream_commit(repo: str, pinned: str | None, allow_red: bool) -> dict:
    """Newest upstream commit that is safe to build from."""
    if pinned:
        info = gh_json(f"repos/{repo}/commits/{pinned}")
        conclusion, n = ci_verdict(repo, info["sha"])
        log(f"pinned {info['sha'][:9]} -- upstream CI: {conclusion} ({n} checks)")
        if conclusion != "success" and not allow_red:
            raise SyncError(
                f"pinned commit {info['sha'][:9]} has upstream CI '{conclusion}'. Pass --allow-red-ci to build from it anyway.")
        return {"sha": info["sha"],
                "date": info["commit"]["committer"]["date"],
                "message": info["commit"]["message"].split("\n")[0],
                "ci_conclusion": conclusion, "ci_checks": n}

    commits = gh_json(f"repos/{repo}/commits?sha={MAIN_BRANCH}&per_page={COMMIT_SEARCH_DEPTH}")
    log(f"scanning {len(commits)} upstream commits for a green build...")
    for c in commits:
        conclusion, n = ci_verdict(repo, c["sha"])
        marker = "GREEN" if conclusion == "success" else conclusion
        log(f"  {c['sha'][:9]}  {c['commit']['committer']['date'][:10]}  {marker} ({n})")
        if conclusion == "success" or allow_red:
            return {"sha": c["sha"],
                    "date": c["commit"]["committer"]["date"],
                    "message": c["commit"]["message"].split("\n")[0],
                    "ci_conclusion": conclusion, "ci_checks": n}
    raise SyncError(
        f"no commit in the last {COMMIT_SEARCH_DEPTH} on {repo}:{MAIN_BRANCH} has green CI. Not syncing -- the branch stays where it is.")


# --- opendbc side -------------------------------------------------------------------------

def opendbc_tests(repo: Path) -> None:
    """Run opendbc's own tests against the rebuilt Elantra tree.

    The guards prove the port's text is still there; this proves it still *works* -- that the
    platform instantiates, its torque data resolves, and its fingerprints are well formed.
    Structural checks cannot catch a platform that parses but blows up on construction.
    """
    log("  installing opendbc (editable) + test deps")
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[testing]", "pytest"], cwd=repo)

    # Both platforms, spelled out: "ELANTRA_2024" does not substring-match
    # "ELANTRA_HEV_2024", so the obvious single-term filter silently tests only the ICE car.
    selector = " or ".join(PLATFORMS)
    log(f"  pytest: car interfaces for {len(PLATFORMS)} Elantra platforms")
    proc = run([sys.executable, "-m", "pytest",
                "opendbc/car/tests/test_car_interfaces.py",
                "-k", selector, "-q", "--no-header", "-p", "no:cacheprovider"],
               cwd=repo, check=False)
    out = (proc.stdout or "")
    log(out[-4000:])
    if proc.returncode != 0:
        raise SyncError("opendbc car-interface tests failed for the Elantra platforms:\n"
                        + (proc.stderr or "")[-2000:]
                        + "\n\nNothing published.")
    # A filter that matches nothing exits 0 with "no tests ran" -- that is a gate that
    # silently stopped gating, which is worse than a failing one.
    matched = re.search(r"(\d+) passed", out)
    passed = int(matched.group(1)) if matched else 0
    if passed < len(PLATFORMS):
        raise SyncError(
            f"expected at least {len(PLATFORMS)} car-interface tests to run (one per Elantra platform), but only {passed} passed. The selector "
            + f"{selector!r} no longer matches -- the gate is not testing what it claims.")

    log("  pytest: car list / docs consistency")
    docs = run([sys.executable, "-m", "pytest", "opendbc/car/tests/test_docs.py",
                "-q", "--no-header", "-p", "no:cacheprovider"], cwd=repo, check=False)
    log((docs.stdout or "")[-3000:])
    if docs.returncode != 0:
        raise SyncError("opendbc docs/car-list tests failed -- the new platforms are "
                        + "inconsistent with opendbc's own metadata rules. Nothing published.")


def build_opendbc(workdir: Path, dry_run: bool) -> str:
    """Rebuild <fork>/opendbc:elantra as opendbc master + the recomputed port delta."""
    repo = workdir / "opendbc"
    log("\n=== opendbc ===")
    log(f"cloning {UPSTREAM_OPENDBC} (blobless, two branches)")
    # Blobless and single-branch: opendbc carries hundreds of stale port branches and we only
    # ever need master and the community Elantra branch.
    run(["git", "clone", "-q", "--filter=blob:none", "--single-branch",
         "--branch", MAIN_BRANCH, f"https://github.com/{UPSTREAM_OPENDBC}.git", str(repo)])
    git(["remote", "rename", "origin", "upstream"], repo)
    # In CI the workflow's GITHUB_TOKEN is scoped to the openpilot repo only, so pushing the
    # opendbc fork goes over SSH with a deploy key that has write access to that one repo --
    # narrower than a personal access token, which would carry account-wide repo scope.
    git(["remote", "add", "fork",
         os.environ.get("OPENDBC_PUSH_URL", f"https://github.com/{FORK_OPENDBC}.git")], repo)
    git(["config", "--add", "remote.upstream.fetch",
         f"+refs/heads/{OPENDBC_PORT_BRANCH}:refs/remotes/upstream/{OPENDBC_PORT_BRANCH}"], repo)
    git(["fetch", "-q", "upstream"], repo)

    master_sha = git(["rev-parse", f"refs/remotes/upstream/{MAIN_BRANCH}"], repo)
    port_sha = git(["rev-parse", f"refs/remotes/upstream/{OPENDBC_PORT_BRANCH}"], repo)
    base = git(["merge-base", master_sha, port_sha], repo)
    log(f"  master {master_sha[:9]}  port {port_sha[:9]}  base {base[:9]}")

    patch = repo.parent / "elantra-delta.patch"
    diff = run(["git", "diff", "--binary", base, port_sha], cwd=repo).stdout
    if not diff.strip():
        raise SyncError(
            f"{UPSTREAM_OPENDBC}:{OPENDBC_PORT_BRANCH} carries no delta over master. "
            + "Either the port was merged upstream or the branch was reset -- refusing to "
            + "publish an opendbc with no Elantra support.")
    patch.write_text(diff, encoding="utf-8", newline="\n")
    log(f"  recomputed port delta: {len(diff.splitlines())} lines")

    git(["checkout", "-q", "-B", OPENDBC_BRANCH, master_sha], repo)
    apply = run(["git", "apply", "-3", str(patch)], cwd=repo, check=False)
    if apply.returncode != 0:
        raise SyncError(
            "the Elantra opendbc delta no longer applies to opendbc master:\n"
            + (apply.stderr or "").strip()
            + "\n\nRefusing to publish. Refresh the delta by hand, then re-run.")

    local_extras = Path(__file__).resolve().parent / "local-extras.patch"
    # Only treat it as a patch if it actually contains a diff -- a file of nothing but
    # comments is how someone leaves themselves a note, not a reason to abort the sync.
    if local_extras.is_file() and "diff --git" in local_extras.read_text(encoding="utf-8"):
        log("  applying local-extras.patch (your car's firmware, tuning)")
        extra = run(["git", "apply", "-3", str(local_extras)], cwd=repo, check=False)
        if extra.returncode != 0:
            raise SyncError("local-extras.patch no longer applies:\n"
                            + (extra.stderr or "").strip())

    git(["add", "-A"], repo)
    # Pin the commit timestamp to opendbc master's so the result is reproducible: same inputs
    # give the same sha, and an unchanged week does not churn the gitlink for no reason.
    stamp = git(["show", "-s", "--format=%cI", master_sha], repo)
    env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    commit = subprocess.run(
        ["git", "-c", "user.name=elantra-sync",
         "-c", "user.email=elantra-sync@users.noreply.github.com",
         "commit", "-q", "-m",
         "Elantra 2024-25 support, rebased onto opendbc master\n\n"
         + f"opendbc master     {master_sha}\n"
         + f"{OPENDBC_PORT_BRANCH}  {port_sha}\n\n"
         + "Delta recomputed on every sync so upstream fixes to the port flow through."],
        cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if commit.returncode != 0:
        raise SyncError("opendbc commit failed:\n" + (commit.stderr or "").strip())
    new_sha = git(["rev-parse", "HEAD"], repo)

    guards = Path(__file__).resolve().parent / "guards.py"
    log("  running structural guards on opendbc")
    proc = run([sys.executable, str(guards), "--opendbc", str(repo)], check=False, capture=True)
    log(proc.stdout)
    if proc.returncode != 0:
        raise SyncError("opendbc guards failed -- Elantra support is not intact. "
                        + "Nothing published.")

    if RUN_OPENDBC_TESTS:
        opendbc_tests(repo)

    if dry_run:
        log(f"  [dry-run] would push {new_sha[:9]} to {FORK_OPENDBC}:{OPENDBC_BRANCH}")
    else:
        # Keep the fork's master mirroring upstream too, so the elantra branch always has a
        # meaningful "ahead by" and the delta stays reviewable on GitHub.
        git(["push", "--no-verify", "--force", "fork",
             f"{master_sha}:refs/heads/{MAIN_BRANCH}"], repo, check=False)
        git(["push", "--no-verify", "--force", "fork", f"{OPENDBC_BRANCH}:refs/heads/{OPENDBC_BRANCH}"], repo)
        log(f"  pushed {new_sha[:9]} -> {FORK_OPENDBC}:{OPENDBC_BRANCH}")

    return new_sha


# --- tinygrad side ------------------------------------------------------------------------

def upstream_gitlink(repo: str, ref: str, path: str) -> tuple[str, str]:
    """(gitlink sha, submodule url) for one submodule at one commit, over the API.

    build_tinygrad has to know which tinygrad the *target* upstream commit pins, and it runs
    before the superproject is cloned. The contents endpoint answers both halves in one call,
    and reports the url as git resolved it -- so a sunnypilot that repoints tinygrad at a
    different fork is visible here rather than three steps later.
    """
    data = gh_json(f"repos/{repo}/contents/{path}?ref={ref}")
    if data.get("type") != "submodule":
        raise SyncError(
            f"{repo}@{ref[:9]}:{path} is no longer a submodule (type={data.get('type')!r}). "
            + "Upstream restructured the tree and this script's assumptions no longer hold.")
    return data["sha"], data.get("submodule_git_url") or ""


def build_tinygrad(workdir: Path, pinned: str, dry_run: bool) -> str:
    """Rebuild <fork>/tinygrad:nv-usb3-built as sunnypilot's pinned tinygrad + the NV-USB delta.

    Deliberately *not* rebased onto tinygrad master. sunnypilot pins a snapshot of its own
    fork and the whole of modeld is written against it; rebasing onto real upstream would give
    a tinygrad that carries the eGPU patch and no longer runs the models.
    """
    repo = workdir / "tinygrad"
    log("\n=== tinygrad ===")
    log(f"cloning {UPSTREAM_TINYGRAD} (blobless)")
    run(["git", "clone", "-q", "--filter=blob:none", "--single-branch", "--branch", MAIN_BRANCH,
         f"https://github.com/{UPSTREAM_TINYGRAD}.git", str(repo)])
    git(["remote", "rename", "origin", "upstream"], repo)
    git(["remote", "add", "fork",
         os.environ.get("TINYGRAD_PUSH_URL", f"https://github.com/{FORK_TINYGRAD}.git")], repo)

    # By sha, not by branch: sunnypilot's own tools/release/check-submodules.sh explicitly
    # skips tinygrad_repo from its "hash must be on master" check, so the pin is not promised
    # to be reachable from any branch.
    fetched = run(["git", "fetch", "-q", "--filter=blob:none", "upstream", pinned],
                  cwd=repo, check=False)
    if fetched.returncode != 0:
        raise SyncError(
            f"cannot fetch the tinygrad commit sunnypilot pins ({pinned[:9]}) from "
            + f"{UPSTREAM_TINYGRAD}:\n" + (fetched.stderr or "").strip()
            + "\n\nIt may live on a branch that was deleted, or in a different fork. "
            + "Nothing published.")
    git(["fetch", "-q", "--filter=blob:none", "fork", TINYGRAD_PORT_BRANCH], repo)
    port_sha = git(["rev-parse", "FETCH_HEAD"], repo)
    base = git(["merge-base", pinned, port_sha], repo)
    log(f"  pin {pinned[:9]}  port {port_sha[:9]}  base {base[:9]}")

    # The branch invariant. sunnypilot's tinygrad and upstream tinygrad are separate lineages,
    # so a merge-base computed across them can be months back -- and the "delta" would then
    # carry every upstream tinygrad change in between, silently dragging the pin past the
    # snapshot modeld is written against. That failure *succeeds*, which is worse than a
    # conflict, so check it explicitly rather than trusting the diff to be small.
    touched = [p for p in git(["diff", "--name-only", base, port_sha], repo).splitlines()
               if p.strip()]
    stray = sorted(set(touched) - set(NV_DELTA_PATHS))
    if stray:
        raise SyncError(
            f"{FORK_TINYGRAD}:{TINYGRAD_PORT_BRANCH} reaches outside the NV-USB delta:\n  "
            + "\n  ".join(stray)
            + f"\n\nIts merge-base with sunnypilot's pin is {base[:9]}. Either the branch was "
            + "cut from tinygrad/tinygrad instead of sunnypilot/tinygrad -- in which case this "
            + "'delta' is weeks of unrelated tinygrad churn and applying it would break modeld "
            + "-- or the rebased PR genuinely touches a new file, in which case add it to "
            + "NV_DELTA_PATHS in .elantra/overlay.py and give it a sentinel. Nothing published.")

    patch = repo.parent / "nv-usb3-delta.patch"
    diff = run(["git", "diff", "--binary", base, port_sha, "--"] + list(NV_DELTA_PATHS),
               cwd=repo).stdout
    if not diff.strip():
        raise SyncError(
            f"{FORK_TINYGRAD}:{TINYGRAD_PORT_BRANCH} carries no delta over sunnypilot's "
            + "tinygrad. Either the NV-USB work landed upstream or the branch was reset -- "
            + "refusing to publish a build whose eGPU support has silently gone away.")
    patch.write_text(diff, encoding="utf-8", newline="\n")
    log(f"  recomputed NV-USB delta: {len(diff.splitlines())} lines")

    git(["checkout", "-q", "-B", TINYGRAD_BRANCH, pinned], repo)
    applied = run(["git", "apply", "-3", str(patch)], cwd=repo, check=False)
    if applied.returncode != 0:
        raise SyncError(
            "the NV-USB eGPU delta no longer applies to the tinygrad sunnypilot pins:\n"
            + (applied.stderr or "").strip()
            + f"\n\n  sunnypilot pins  {UPSTREAM_TINYGRAD}@{pinned[:9]}\n"
            + f"  our delta        {FORK_TINYGRAD}:{TINYGRAD_PORT_BRANCH}@{port_sha[:9]}"
            + f" (base {base[:9]})\n\n"
            + "sunnypilot moved its tinygrad snapshot under the patch. This is expected: it is\n"
            + "an unmerged PR against a fast-moving tree, not a stable API.\n\n"
            + "To fix:\n"
            + f"  1. rebase {TINYGRAD_PORT_BRANCH} onto {UPSTREAM_TINYGRAD}:master (see CLAUDE.md)\n"
            + f"  2. python .elantra/sync.py --dry-run --tinygrad-pin {pinned}\n"
            + "  3. re-run the sync\n\n"
            + "Nothing published. The car keeps the last build, eGPU and all.")

    # A three-way apply can return 0 and still leave conflict markers behind. A tree that is a
    # SyntaxError on the device is not a better outcome than a clean abort here.
    for rel in NV_DELTA_PATHS:
        target = repo / rel
        if target.is_file() and re.search(r"^<{7} ", target.read_text(encoding="utf-8",
                                                                     errors="replace"), re.M):
            raise SyncError(f"three-way apply left conflict markers in {rel}. Nothing published.")

    git(["add", "-A"], repo)
    stamp = git(["show", "-s", "--format=%cI", pinned], repo)
    env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    commit = subprocess.run(
        ["git", "-c", "user.name=elantra-sync",
         "-c", "user.email=elantra-sync@users.noreply.github.com",
         "commit", "-q", "-m",
         "NV-USB eGPU support, replayed onto sunnypilot's tinygrad\n\n"
         + f"sunnypilot pin  {UPSTREAM_TINYGRAD}@{pinned}\n"
         + f"{TINYGRAD_PORT_BRANCH}          {port_sha}\n\n"
         + "Delta recomputed on every sync, so rebases of tinygrad PR #17369 flow through."],
        cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if commit.returncode != 0:
        raise SyncError("tinygrad commit failed:\n" + (commit.stderr or "").strip())
    new_sha = git(["rev-parse", "HEAD"], repo)

    # "git apply returned 0" and "the patch actually landed" are different claims.
    detector = Path(__file__).resolve().parent / "test_egpu_tinygrad.py"
    log("  checking the NV-USB patch is structurally intact")
    proc = run([sys.executable, str(detector), "--tinygrad", str(repo)], check=False)
    log(proc.stdout)
    if proc.returncode != 0:
        raise SyncError("the rebuilt tinygrad does not carry working eGPU support. "
                        + "Nothing published.")

    if dry_run:
        log(f"  [dry-run] would push {new_sha[:9]} to {FORK_TINYGRAD}:{TINYGRAD_BRANCH}")
    else:
        # Mirror the pin too, so the built branch has a meaningful "ahead by 1" on GitHub and
        # the delta stays reviewable.
        git(["push", "--no-verify", "--force", "fork",
             f"{pinned}:refs/heads/sunnypilot-base"], repo, check=False)
        git(["push", "--no-verify", "--force", "fork",
             f"{TINYGRAD_BRANCH}:refs/heads/{TINYGRAD_BRANCH}"], repo)
        log(f"  pushed {new_sha[:9]} -> {FORK_TINYGRAD}:{TINYGRAD_BRANCH}")

    return new_sha


# --- superproject -------------------------------------------------------------------------

def restore_paths(repo: Path, ref: str, paths: list[str]) -> None:
    """Check paths out of `ref`, working around blobless clones.

    `git checkout <ref> -- <paths>` does not go through the promisor, so in a partial clone it
    fails with "unable to read sha1 file" rather than fetching what it needs. `git cat-file`
    does fetch, so walk the blobs once to pull them local, then let checkout do its job.
    """
    # Check each path separately first. `git checkout` with a bad pathspec dies with a raw
    # git error that does not say which registered path is wrong or what to do about it, and
    # one missing entry out of four blocks the whole sync.
    missing = [p for p in paths
               if not git(["ls-tree", "-r", "--name-only", ref, "--", p], repo).strip()]
    if missing:
        raise SyncError(
            f"these paths are registered in OVERLAY_ADDED but do not exist at {ref[:9]}:\n  "
            + "\n  ".join(missing)
            + "\n\nThe sync cannot restore a file that was never committed. Either commit it "
            + f"to {MAIN_BRANCH}, or take it out of the registry in .elantra/overlay.py. "
            + "Nothing published.")

    listing = git(["ls-tree", "-r", "--name-only", ref, "--"] + paths, repo)
    files = [f for f in listing.splitlines() if f.strip()]
    wanted = "\n".join(f"{ref}:{f}" for f in files)
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, text=True, input=wanted,
                          capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SyncError("could not fetch overlay blobs from the previous build:\n"
                        + (proc.stderr or "").strip())

    git(["checkout", ref, "--"] + paths, repo)
    log(f"  restored {len(files)} overlay files from {ref[:9]}")


def previous_build(repo: Path) -> tuple[str, dict]:
    """The commit our last build sits on, plus the manifest describing it."""
    prev = git(["rev-parse", f"refs/remotes/fork/{MAIN_BRANCH}"], repo)
    raw = run(["git", "show", f"{prev}:.elantra/build-manifest.json"], cwd=repo, check=False)
    if raw.returncode != 0:
        raise SyncError(
            f"{FORK_REPO}:{MAIN_BRANCH} has no .elantra/build-manifest.json. This branch was "
            + "not produced by this script -- bootstrap it once by hand before syncing.")
    return prev, json.loads(raw.stdout)


def build_superproject(repo: Path, target: dict, opendbc_sha: str, tinygrad_sha: str,
                       tinygrad_meta: dict, dry_run: bool) -> tuple[str, str]:
    log("\n=== superproject ===")
    prev, prev_manifest = previous_build(repo)
    prev_base = prev_manifest["sunnypilot_upstream_sha"]
    log(f"  previous build {prev[:9]} was built from upstream {prev_base[:9]}")
    log(f"  target upstream {target['sha'][:9]} ({target['date'][:10]})")

    # opendbc moves on its own schedule, and local-extras.patch (your car's firmware) lands
    # there too -- so an unchanged upstream commit is not on its own a reason to skip. The
    # tinygrad pin has to be in this test too, or an unchanged upstream with a moved tinygrad
    # would silently keep the stale pin and the eGPU would run against the wrong tree.
    if (prev_base == target["sha"]
            and prev_manifest.get("opendbc_sha") == opendbc_sha
            and prev_manifest.get("tinygrad_sha") == tinygrad_sha):
        log("  already current: same upstream commit, same opendbc, same tinygrad.")
        return prev, ""
    if prev_base == target["sha"]:
        log("  upstream unchanged, but opendbc moved "
            + f"{(prev_manifest.get('opendbc_sha') or '')[:9]} -> {opendbc_sha[:9]}; rebuilding")

    # The overlay diff is derived from the last build rather than stored as a static patch,
    # so hand edits to the overlay are picked up automatically and there is only ever one
    # source of truth.
    overlay = run(["git", "diff", "--binary", prev_base, prev, "--"] + OVERLAY_MODIFIED,
                  cwd=repo).stdout
    if not overlay.strip():
        raise SyncError("the previous build carries no overlay diff against its upstream base. "
                        + "That means the UI, params and updater changes are gone. Not syncing.")
    patch = repo.parent / "overlay.patch"
    patch.write_text(overlay, encoding="utf-8", newline="\n")
    log(f"  overlay diff: {len([l for l in overlay.splitlines() if l.startswith('+') and not l.startswith('+++')])} added lines "
        + f"across {len(OVERLAY_MODIFIED)} files")

    git(["checkout", "-q", "--force", "-B", "sync-work", target["sha"]], repo)

    log("  restoring overlay files that are entirely ours")
    restore_paths(repo, prev, OVERLAY_ADDED)

    log("  replaying overlay diff (three-way)")
    applied = run(["git", "apply", "-3", str(patch)], cwd=repo, check=False)
    if applied.returncode != 0:
        raise SyncError(
            "the overlay no longer applies to upstream master:\n"
            + (applied.stderr or "").strip()
            + "\n\nUpstream changed one of:\n  " + "\n  ".join(OVERLAY_MODIFIED)
            + "\n\nNothing published. Resolve by hand, commit to master, then re-run.")

    log(f"  pinning opendbc submodule -> {opendbc_sha[:9]}")
    git(["update-index", "--cacheinfo", f"160000,{opendbc_sha},opendbc_repo"], repo)
    log(f"  pinning tinygrad submodule -> {tinygrad_sha[:9]}")
    git(["update-index", "--cacheinfo", f"160000,{tinygrad_sha},tinygrad_repo"], repo)

    manifest = {
        "sunnypilot_upstream_sha": target["sha"],
        "sunnypilot_upstream_date": target["date"][:10],
        "sunnypilot_upstream_subject": target["message"][:120],
        "upstream_ci_conclusion": target["ci_conclusion"],
        "upstream_ci_checked": target["ci_checks"],
        "upstream_ci_url": f"https://github.com/{UPSTREAM_REPO}/commit/{target['sha']}/checks",
        "opendbc_sha": opendbc_sha,
        "opendbc_repo": FORK_OPENDBC,
        "tinygrad_sha": tinygrad_sha,
        "tinygrad_repo": FORK_TINYGRAD,
        "tinygrad_upstream_sha": tinygrad_meta["pin"],
        "tinygrad_upstream_ci_conclusion": tinygrad_meta["ci_conclusion"],
        "tinygrad_upstream_ci_checked": tinygrad_meta["ci_checks"],
        "tinygrad_patch": "nv-usb3 (tinygrad#17369)",
        "egpu": True,
        "elantra_platforms": ["HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024"],
        "previous_master_sha": prev,
        "rollback_branch": ROLLBACK_BRANCH,
        "synced_at_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = repo / ".elantra/build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8", newline="\n")

    git(["add", "-A"], repo)

    # `git add -A` restages an *initialised* submodule at its own HEAD, which would silently
    # undo the pins above. In CI the clone is --no-checkout and submodules are never
    # initialised, so this never bites there -- but it bites hard when rehearsing with --repo
    # against a working tree that has ever seen `git submodule update --init`. Re-assert, then
    # prove it stuck: a gitlink that quietly reverted to sunnypilot's pin is exactly the
    # failure this design exists to prevent.
    pinned_gitlinks = {"opendbc_repo": opendbc_sha, "tinygrad_repo": tinygrad_sha}
    assert set(pinned_gitlinks) == set(OVERLAY_GITLINKS), "OVERLAY_GITLINKS and the pins here have drifted"
    for path in OVERLAY_GITLINKS:
        sha = pinned_gitlinks[path]
        git(["update-index", "--cacheinfo", f"160000,{sha},{path}"], repo)
        staged = git(["ls-files", "-s", "--", path], repo)
        if not staged.startswith(f"160000 {sha} "):
            raise SyncError(f"the {path} gitlink did not stick: staged {staged!r}, "
                            + f"wanted {sha[:9]}. Nothing published.")

    subject = f"Sync to sunnypilot {target['sha'][:9]} with Elantra 2024-25 port"
    body = (f"{subject}\n\n"
            + f"upstream   {UPSTREAM_REPO}@{target['sha']}\n"
            + f"           {target['message'][:100]}\n"
            + f"           CI {target['ci_conclusion']} ({target['ci_checks']} checks)\n"
            + f"opendbc    {FORK_OPENDBC}@{opendbc_sha}\n"
            + f"tinygrad   {FORK_TINYGRAD}@{tinygrad_sha}\n"
            + f"rollback   {ROLLBACK_BRANCH} -> {prev}\n\n"
            + "Rebuilt by .elantra/sync.py: reset to upstream, replay overlay. Not a merge.\n")
    git(["-c", "user.name=elantra-sync",
         "-c", "user.email=elantra-sync@users.noreply.github.com",
         "commit", "-q", "-m", body], repo)
    new = git(["rev-parse", "HEAD"], repo)
    log(f"  built {new[:9]}")
    return new, prev


def publish(repo: Path, new: str, prev: str, dry_run: bool) -> None:
    log("\n=== publish ===")
    if dry_run:
        log(f"  [dry-run] would move {ROLLBACK_BRANCH} -> {prev[:9]}")
        log(f"  [dry-run] would force-push {MAIN_BRANCH} -> {new[:9]}")
        return
    # Rollback pointer first: if the master push fails, the previous build is still parked.
    git(["push", "--no-verify", "--force", "fork", f"{prev}:refs/heads/{ROLLBACK_BRANCH}"], repo)
    log(f"  {ROLLBACK_BRANCH} -> {prev[:9]}")
    git(["push", "--no-verify", "--force", "fork", f"{new}:refs/heads/{MAIN_BRANCH}"], repo)
    log(f"  {MAIN_BRANCH} -> {new[:9]}")


def open_issue(title: str, body: str) -> None:
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    run(["gh", "issue", "create", "--repo", FORK_REPO, "--title", title,
         "--body", body, "--label", "sync-failure"], check=False)


# --- entrypoint ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build and validate, publish nothing")
    ap.add_argument("--ref", default=None, help="pin a specific upstream commit")
    ap.add_argument("--allow-red-ci", action="store_true",
                    help="build even from a commit whose upstream CI is not green")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="scratch directory (default: a temp dir, removed afterwards)")
    ap.add_argument("--repo", type=Path, default=None,
                    help="existing superproject checkout to build in (default: fresh clone)")
    ap.add_argument("--opendbc-tests", action="store_true",
                    help="also run opendbc's own tests (CI does; needs its dependency tree)")
    ap.add_argument("--tinygrad-pin", default=None,
                    help="override the tinygrad commit to replay the eGPU delta onto "
                         + "(default: whatever the target sunnypilot commit pins)")
    ap.add_argument("--require-tinygrad-ci", action="store_true",
                    help="abort if sunnypilot's tinygrad pin does not have green CI")
    args = ap.parse_args()

    global RUN_OPENDBC_TESTS
    RUN_OPENDBC_TESTS = args.opendbc_tests
    if not RUN_OPENDBC_TESTS:
        log("note: opendbc's own tests are NOT running this pass (--opendbc-tests off). "
            + "Structural guards still apply.")

    tmp = None
    if args.workdir:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="elantra-sync-")
        workdir = Path(tmp)

    try:
        target = pick_upstream_commit(UPSTREAM_REPO, args.ref, args.allow_red_ci)
        log(f"\ntarget: {target['sha'][:9]} {target['date'][:10]} "
            + f"CI={target['ci_conclusion']} ({target['ci_checks']} checks)")
        log(f"        {target['message'][:100]}")

        opendbc_sha = build_opendbc(workdir, args.dry_run)

        # Which tinygrad does the target sunnypilot commit pin, and from whose fork?
        tinygrad_pin, tinygrad_url = upstream_gitlink(UPSTREAM_REPO, target["sha"],
                                                      "tinygrad_repo")
        if args.tinygrad_pin:
            log(f"\noverriding tinygrad pin {tinygrad_pin[:9]} -> {args.tinygrad_pin[:9]}")
            tinygrad_pin = args.tinygrad_pin
        elif UPSTREAM_TINYGRAD not in tinygrad_url:
            # A red pin is a maybe; a *different fork* is a certainty. Our delta is cut
            # against sunnypilot's tinygrad and would be replayed onto a tree it was never
            # written for.
            raise SyncError(
                f"sunnypilot now pins tinygrad from {tinygrad_url!r}, not {UPSTREAM_TINYGRAD}. "
                + "The NV-USB delta would be replayed onto a tree it was never written for. "
                + "Nothing published.")

        tg_ci, tg_checks = ci_verdict(UPSTREAM_TINYGRAD, tinygrad_pin)
        log(f"tinygrad pin {tinygrad_pin[:9]} -- CI: {tg_ci} ({tg_checks} checks)")
        if tg_ci != "success":
            if args.require_tinygrad_ci:
                raise SyncError(f"tinygrad pin {tinygrad_pin[:9]} has CI '{tg_ci}' and "
                                + "--require-tinygrad-ci is set. Nothing published.")
            # Not fatal by default: we do not choose this pin, and the sunnypilot commit we
            # already gated on green CI was itself built against this exact tinygrad. That is
            # stronger evidence than the fork's own CI, which exercises a hundred backends we
            # never touch. sunnypilot's own check-submodules.sh declines to gate it too.
            log("note: building on it anyway -- the sunnypilot commit we picked is green and "
                + "its CI built modeld against this exact pin. Recorded in the manifest; "
                + "--require-tinygrad-ci makes this fatal.")
        tinygrad_meta = {"pin": tinygrad_pin, "ci_conclusion": tg_ci, "ci_checks": tg_checks}
        tinygrad_sha = build_tinygrad(workdir, tinygrad_pin, args.dry_run)

        if args.repo:
            repo = args.repo.resolve()
        else:
            repo = workdir / "openpilot"
            log(f"\ncloning {FORK_REPO}")
            run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                 f"https://github.com/{FORK_REPO}.git", str(repo)])
            git(["remote", "add", "upstream", f"https://github.com/{UPSTREAM_REPO}.git"], repo)
            git(["remote", "rename", "origin", "fork"], repo)
            # GITHUB_TOKEN can never push a commit that touches .github/workflows -- that is a
            # GitHub App restriction with no permissions flag to unlock it, and mirroring
            # upstream changes workflow files constantly. So CI pushes over SSH with a deploy
            # key scoped to this one repo.
            push_url = os.environ.get("OPENPILOT_PUSH_URL")
            if push_url:
                git(["remote", "set-url", "--push", "fork", push_url], repo)
        git(["fetch", "-q", "--filter=blob:none", "upstream", MAIN_BRANCH], repo)
        git(["fetch", "-q", "--filter=blob:none", "fork", MAIN_BRANCH], repo)

        new, prev = build_superproject(repo, target, opendbc_sha, tinygrad_sha,
                                       tinygrad_meta, args.dry_run)
        if not prev:
            return 0

        guards = repo / ".elantra/guards.py"
        log("\n=== guards on the assembled tree ===")
        proc = run([sys.executable, str(guards), "--opendbc", str(workdir / "opendbc"),
                    "--tinygrad", str(workdir / "tinygrad"), "--repo", str(repo)], check=False)
        log(proc.stdout)
        if proc.returncode != 0:
            raise SyncError("guards failed on the assembled tree. Nothing published.")

        publish(repo, new, prev, args.dry_run)
        log("\nDone." + ("  (dry run -- nothing was published)" if args.dry_run else ""))
        return 0

    except SyncError as e:
        log(f"\nSYNC ABORTED\n{e}")
        open_issue("Elantra sync aborted", f"```\n{e}\n```\n\n"
                   + f"`{MAIN_BRANCH}` was left untouched. The car keeps running the last "
                   + "published build.")
        return 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
