#!/usr/bin/env python3
"""
Rebuild this branch from sunnypilot master with the Elantra 2024-25 port on top.

The branch is derived, never merged. Every run hard-resets to a chosen upstream commit and
replays a small overlay, so there is no merge history to rot -- which is precisely how the
community elantra-2024-port branch ends up months behind.

What the overlay is:
  * the .elantra directory, the sync workflow and four UI files of our own, copied wholesale
    from the previous build (they are ours, they never conflict)
  * a ~31-line diff against five upstream files, applied three-way
  * the opendbc submodule URL and gitlink, pointed at our Elantra-enabled opendbc, which is
    itself rebuilt the same way: our opendbc master's delta, replayed onto upstream's

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
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "sunnypilot/sunnypilot"
FORK_REPO = "aaravsaianugula/openpilot"
UPSTREAM_OPENDBC = "sunnypilot/opendbc"
FORK_OPENDBC = "aaravsaianugula/opendbc"

MAIN_BRANCH = "master"
ROLLBACK_BRANCH = "master-previous"

# Our opendbc lives on its own master, and that branch is the SOURCE OF TRUTH this rebuild
# derives next week's delta from -- not a disposable mirror of upstream. It used to be rebuilt
# from sunnypilot/opendbc:elantra-2024-port, the 83-line community port, which carries no
# raised ceiling, no per-platform 0x485 split and no RAISED_LIMITS flag; the guards correctly
# refused the result, so every scheduled sync since 2026-08-31 aborted. The delta is now
# recomputed from our own master instead, which is the same mechanism build_superproject()
# already uses for the openpilot overlay.
OPENDBC_BRANCH = MAIN_BRANCH
OPENDBC_ROLLBACK = ROLLBACK_BRANCH

# Every path our opendbc delta is allowed to touch. Anything outside it aborts BY NAME rather
# than turning up as an unexplained three-way conflict weeks later. The fork's own scheduled
# "Update CARS.md" job is the concrete case this exists for: it commits straight to the default
# branch, which is now the delta source.
OPENDBC_DELTA_PATHS = (
    "docs/CARS.md",
    "opendbc/car/hyundai/fingerprints.py",
    "opendbc/car/hyundai/hyundaican.py",
    "opendbc/car/hyundai/interface.py",
    "opendbc/car/hyundai/tests/test_hyundai.py",
    "opendbc/car/hyundai/values.py",
    "opendbc/car/tests/routes.py",
    "opendbc/car/torque_data/substitute.toml",
    "opendbc/dbc/generator/hyundai/_hyundai_can_common.dbc",
    "opendbc/dbc/generator/hyundai/hyundai_can.dbc",
    "opendbc/dbc/generator/hyundai/hyundai_can_cn7.dbc",
    "opendbc/safety/modes/hyundai.h",
    "opendbc/safety/modes/hyundai_common.h",
    "opendbc/safety/tests/common.py",
    "opendbc/safety/tests/test_hyundai.py",
    "opendbc/sunnypilot/car/car_list.json",
)

# Tokens that must survive INTO the recomputed delta. An empty delta is only the loudest way
# to lose Elantra support; one reduced to a docs refresh would sail past an emptiness check.
REQUIRED_IN_DELTA = (
    "HYUNDAI_ELANTRA_2024",
    "HYUNDAI_ELANTRA_HEV_2024",
    "RAISED_LIMITS",
    "HYUNDAI_PARAM_RAISED_LIMITS",
    "LFAHDA_MFC_8",
    "hyundai_can_cn7",
)

# The platforms this branch exists to support. Used by the test gate so a selector that
# stops matching is caught rather than silently reducing coverage.
PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")

# Overlay files that are entirely ours. Restored wholesale, so they cannot conflict.
OVERLAY_ADDED = [
    ".elantra",
    ".github/workflows/elantra-sync.yaml",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_updates.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_manifest.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/onroad/steer_headroom.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/onroad/steer_headroom_bar.py",
]

# Upstream files we modify. Kept deliberately tiny -- this is the only conflict surface in
# the whole design, so every line here has to earn its place.
OVERLAY_MODIFIED = [
    ".gitmodules",
    "openpilot/common/params_keys.h",
    "openpilot/system/updated/updated.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/settings.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/onroad/hud_renderer.py",
]

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

    # The guards read values.py, hyundai.h, hyundai_common.h, interface.py and
    # carcontroller.py -- never opendbc's own Elantra tests. A delta that kept the 409 but
    # dropped the test changes would pass every other gate in this file.
    log("  pytest: the Elantra tests themselves")
    own = run([sys.executable, "-m", "pytest",
               "opendbc/car/hyundai/tests/test_hyundai.py",
               "-q", "--no-header", "-p", "no:cacheprovider"], cwd=repo, check=False)
    log((own.stdout or "")[-3000:])
    if own.returncode != 0:
        raise SyncError("opendbc's own Hyundai tests failed against the rebuilt tree -- the "
                        + "ceiling, the 0x485 split or the platform flags did not survive the "
                        + "delta. Nothing published.")

    log("  pytest: car list / docs consistency")
    docs = run([sys.executable, "-m", "pytest", "opendbc/car/tests/test_docs.py",
                "-q", "--no-header", "-p", "no:cacheprovider"], cwd=repo, check=False)
    log((docs.stdout or "")[-3000:])
    if docs.returncode != 0:
        raise SyncError("opendbc docs/car-list tests failed -- the new platforms are "
                        + "inconsistent with opendbc's own metadata rules. Nothing published.")


def build_opendbc(workdir: Path, dry_run: bool) -> tuple[str, str]:
    """Rebuild our opendbc as upstream master + our own recomputed Elantra delta.

    Same mechanism as build_superproject: the delta is DERIVED from the last published build
    rather than stored as a frozen patch, so upstream fixes flow through automatically, a fix
    committed by hand to our master is picked up on the next run, and nothing ever carries a
    stale conflict resolution forward.

    Returns (new_sha, prev_sha). It does NOT push -- see publish_opendbc(). Our opendbc master
    is the source of truth this function reads next week, so overwriting it before the
    superproject is known good would destroy that on a run that later aborts.
    """
    repo = workdir / "opendbc"
    log("\n=== opendbc ===")
    log(f"cloning {UPSTREAM_OPENDBC} (blobless)")
    run(["git", "clone", "-q", "--filter=blob:none", "--single-branch",
         "--branch", MAIN_BRANCH, f"https://github.com/{UPSTREAM_OPENDBC}.git", str(repo)])
    git(["remote", "rename", "origin", "upstream"], repo)
    # In CI the workflow's GITHUB_TOKEN is scoped to the openpilot repo only, so pushing the
    # opendbc fork goes over SSH with a deploy key that has write access to that one repo --
    # narrower than a personal access token, which would carry account-wide repo scope.
    git(["remote", "add", "fork",
         os.environ.get("OPENDBC_PUSH_URL", f"https://github.com/{FORK_OPENDBC}.git")], repo)
    git(["fetch", "-q", "--filter=blob:none", "fork", OPENDBC_BRANCH], repo)

    master_sha = git(["rev-parse", f"refs/remotes/upstream/{MAIN_BRANCH}"], repo)
    prev_sha = git(["rev-parse", f"refs/remotes/fork/{OPENDBC_BRANCH}"], repo)
    base = git(["merge-base", master_sha, prev_sha], repo)
    log(f"  upstream {master_sha[:9]}  ours {prev_sha[:9]}  base {base[:9]}")

    # (1) Our master must descend from an upstream commit and carry something of its own.
    if base == prev_sha:
        raise SyncError(
            f"{FORK_OPENDBC}:{OPENDBC_BRANCH} is an ancestor of "
            + f"{UPSTREAM_OPENDBC}:{MAIN_BRANCH} -- it carries no Elantra delta at all. Either "
            + "it was force-mirrored to upstream, or the port was merged upstream. Refusing to "
            + f"publish an opendbc with no Elantra support. Recover from {OPENDBC_ROLLBACK}.")

    names = [n for n in git(["diff", "--name-only", base, prev_sha], repo).splitlines() if n]
    # (2) A delta that vanished entirely.
    if not names:
        raise SyncError(
            f"{FORK_OPENDBC}:{OPENDBC_BRANCH} carries no delta over "
            + f"{UPSTREAM_OPENDBC}:{MAIN_BRANCH}. Refusing to publish an opendbc with no "
            + "Elantra support.")

    # (3) A delta that grew paths the port has no business touching. The fork's own scheduled
    #     CARS.md job commits straight to this branch; this names it rather than letting it
    #     ride along or surface as an unexplained three-way conflict weeks later.
    stray = [n for n in names if n not in OPENDBC_DELTA_PATHS]
    if stray:
        raise SyncError(
            f"{FORK_OPENDBC}:{OPENDBC_BRANCH} carries changes outside the Elantra port: "
            + f"{stray}. Disable the scheduled workflows on the opendbc fork, or add the path "
            + "to OPENDBC_DELTA_PATHS on purpose.")

    diff = run(["git", "diff", "--binary", base, prev_sha], cwd=repo).stdout
    # (4) A delta that still exists but no longer contains the port.
    missing = [t for t in REQUIRED_IN_DELTA if t not in diff]
    if missing:
        raise SyncError(
            f"the recomputed Elantra delta no longer mentions {missing}. The raised ceiling, "
            + "the 0x485 split or the platforms themselves are gone from "
            + f"{FORK_OPENDBC}:{OPENDBC_BRANCH}. Refusing to publish.")

    patch = repo.parent / "elantra-delta.patch"
    patch.write_text(diff, encoding="utf-8", newline="\n")
    log(f"  recomputed delta: {len(names)} files, {len(diff.splitlines())} lines")

    git(["checkout", "-q", "-B", OPENDBC_BRANCH, master_sha], repo)
    apply = run(["git", "apply", "-3", str(patch)], cwd=repo, check=False)
    if apply.returncode != 0:
        raise SyncError(
            "the Elantra opendbc delta no longer applies to opendbc master:\n"
            + (apply.stderr or "").strip()
            + "\n\nRefusing to publish. Resolve it by hand and commit the fix to "
            + f"{FORK_OPENDBC}:{OPENDBC_BRANCH}; next week's delta is recomputed from there, "
            + "so the resolution is not carried forward.")

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
    #
    # prev_sha is deliberately NOT named in this message. Naming it would make each run depend
    # on the previous run's hash, so an unchanged week would still produce a new commit, the
    # gitlink would move, and the car would rebuild for nothing. master_sha plus a digest of
    # the patch content identifies the inputs exactly, and is a fixed point.
    digest = hashlib.sha256(diff.encode()).hexdigest()[:12]
    stamp = git(["show", "-s", "--format=%cI", master_sha], repo)
    env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    commit = subprocess.run(
        ["git", "-c", "user.name=elantra-sync",
         "-c", "user.email=elantra-sync@users.noreply.github.com",
         "commit", "-q", "-m",
         "Elantra 2024-25 support, rebased onto opendbc master\n\n"
         + f"opendbc master  {master_sha}\n"
         + f"delta           {len(names)} files, sha256 {digest}\n\n"
         + f"Recomputed from {FORK_OPENDBC}:{OPENDBC_BRANCH} on every sync, so upstream fixes "
         + "flow through\nand a hand fix committed there is picked up automatically. "
         + "Not a merge."],
        cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if commit.returncode != 0:
        raise SyncError("opendbc commit failed:\n" + (commit.stderr or "").strip())
    new_sha = git(["rev-parse", "HEAD"], repo)

    here = Path(__file__).resolve().parent
    log("  running structural guards on opendbc")
    proc = run([sys.executable, str(here / "guards.py"), "--opendbc", str(repo)],
               check=False, capture=True)
    log(proc.stdout)
    if proc.returncode != 0:
        raise SyncError("opendbc guards failed -- Elantra support is not intact. "
                        + "Nothing published.")

    # A guard suite that has never been shown to fail is decoration. This one needs the opendbc
    # checkout, so it belongs here, before the opendbc push, rather than after it.
    log("  proving those guards can still fail")
    proc = run([sys.executable, str(here / "test_guard_torque_chain.py"),
                "--opendbc", str(repo)], check=False, capture=True)
    log(proc.stdout)
    if proc.returncode != 0:
        raise SyncError("test_guard_torque_chain.py failed -- the opendbc guards can no "
                        + "longer detect a real divergence, so a green guard run proves "
                        + "nothing. Nothing published.")

    if RUN_OPENDBC_TESTS:
        opendbc_tests(repo)

    return new_sha, prev_sha


def publish_opendbc(repo: Path, new_sha: str, prev_sha: str, dry_run: bool) -> None:
    """Push opendbc -- rollback pointer first, and only once everything else is green.

    This branch is what next week's delta is derived from, so it gets the same rollback pointer
    master has. Without it, a superproject rollback to master-previous pins an opendbc sha that
    only ever existed on a branch we have since force-pushed away.
    """
    if dry_run:
        log(f"  [dry-run] would push {new_sha[:9]} to {FORK_OPENDBC}:{OPENDBC_BRANCH}")
        return
    git(["push", "--no-verify", "--force", "fork",
         f"{prev_sha}:refs/heads/{OPENDBC_ROLLBACK}"], repo, check=False)
    git(["push", "--no-verify", "--force", "fork",
         f"{new_sha}:refs/heads/{OPENDBC_BRANCH}"], repo)
    log(f"  pushed {new_sha[:9]} -> {FORK_OPENDBC}:{OPENDBC_BRANCH}")


# --- superproject -------------------------------------------------------------------------

def restore_paths(repo: Path, ref: str, paths: list[str]) -> None:
    """Check paths out of `ref`, working around blobless clones.

    `git checkout <ref> -- <paths>` does not go through the promisor, so in a partial clone it
    fails with "unable to read sha1 file" rather than fetching what it needs. `git cat-file`
    does fetch, so walk the blobs once to pull them local, then let checkout do its job.
    """
    listing = git(["ls-tree", "-r", "--name-only", ref, "--"] + paths, repo)
    files = [f for f in listing.splitlines() if f.strip()]
    if not files:
        raise SyncError(f"none of the overlay paths exist at {ref[:9]}: {paths}")
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


def build_superproject(repo: Path, target: dict, opendbc_sha: str,
                       dry_run: bool) -> tuple[str, str]:
    log("\n=== superproject ===")
    prev, prev_manifest = previous_build(repo)
    prev_base = prev_manifest["sunnypilot_upstream_sha"]
    log(f"  previous build {prev[:9]} was built from upstream {prev_base[:9]}")
    log(f"  target upstream {target['sha'][:9]} ({target['date'][:10]})")

    # opendbc moves on its own schedule, and local-extras.patch (your car's firmware) lands
    # there too -- so an unchanged upstream commit is not on its own a reason to skip.
    if prev_base == target["sha"] and prev_manifest.get("opendbc_sha") == opendbc_sha:
        log("  already current: same upstream commit, same opendbc. Nothing to do.")
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

    manifest = {
        "sunnypilot_upstream_sha": target["sha"],
        "sunnypilot_upstream_date": target["date"][:10],
        "sunnypilot_upstream_subject": target["message"][:120],
        "upstream_ci_conclusion": target["ci_conclusion"],
        "upstream_ci_checked": target["ci_checks"],
        "upstream_ci_url": f"https://github.com/{UPSTREAM_REPO}/commit/{target['sha']}/checks",
        "opendbc_sha": opendbc_sha,
        "opendbc_repo": FORK_OPENDBC,
        "elantra_platforms": ["HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024"],
        "previous_master_sha": prev,
        "rollback_branch": ROLLBACK_BRANCH,
        "synced_at_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = repo / ".elantra/build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8", newline="\n")

    git(["add", "-A"], repo)
    subject = f"Sync to sunnypilot {target['sha'][:9]} with Elantra 2024-25 port"
    body = (f"{subject}\n\n"
            + f"upstream   {UPSTREAM_REPO}@{target['sha']}\n"
            + f"           {target['message'][:100]}\n"
            + f"           CI {target['ci_conclusion']} ({target['ci_checks']} checks)\n"
            + f"opendbc    {FORK_OPENDBC}@{opendbc_sha}\n"
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

        opendbc_sha, odbc_prev = build_opendbc(workdir, args.dry_run)

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

        new, prev = build_superproject(repo, target, opendbc_sha, args.dry_run)
        if not prev:
            return 0

        guards = repo / ".elantra/guards.py"
        log("\n=== guards on the assembled tree ===")
        proc = run([sys.executable, str(guards), "--opendbc", str(workdir / "opendbc"),
                    "--repo", str(repo)], check=False)
        log(proc.stdout)
        if proc.returncode != 0:
            raise SyncError("guards failed on the assembled tree. Nothing published.")

        # A guard suite that has never been shown to fail is decoration. These scripts run
        # the same guards against deliberately broken copies of the tree and require them to
        # go red. Without this the guards could quietly stop checking anything -- a regex that
        # matches nothing, a check whose subject moved -- and every sync would still pass.
        # They were written for this branch and, until now, ran nowhere.
        log("\n=== negative tests: prove the guards can still fail ===")
        for script, extra in (("test_guard_opendbc_pin.py", []),
                              ("test_lateral_report.py", []),
                              # cross-checks that all four scanners decode the same bytes the
                              # same way; they deploy separately and each carries its own copy
                              ("test_scanner_decoders.py", []),
                              # the onroad arc's decision logic, which nothing else exercises
                              ("test_steer_headroom.py", []),
                              # pins the override-yield arithmetic (-242 -> -254.5 counts) that
                              # the road-test document cites as confirmed by executable test
                              ("test_torque_projection.py", [])):
            # Three of these import opendbc (test_scanner_decoders through ceiling_replay,
            # test_torque_projection through CarControllerParams). Point PYTHONPATH at the
            # opendbc we just BUILT rather than relying on opendbc_tests() having pip-installed
            # it: that only happens under --opendbc-tests, so without this a plain run aborts
            # here on an ImportError and reports it as "the guards can no longer detect a real
            # divergence", which is not what happened.
            env = dict(os.environ)
            odbc = str(workdir / "opendbc")
            env["PYTHONPATH"] = odbc + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run([sys.executable, str(repo / ".elantra" / script), *extra],
                                  env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            log(proc.stdout)
            if proc.returncode != 0:
                raise SyncError(script + " failed -- the guards can no longer detect a real " +
                                "divergence, so a green guard run proves nothing. " +
                                "Nothing published.\n" + (proc.stderr or "")[-1500:])

        publish_opendbc(workdir / "opendbc", opendbc_sha, odbc_prev, args.dry_run)
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
