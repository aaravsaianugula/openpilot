#!/usr/bin/env python3
"""
Verify the branch that is actually on GitHub, not the one we built locally.

The sync script validates a working tree and then pushes it. This checks the result from
the outside -- fetching master fresh, resolving the opendbc gitlink through the GitHub API,
and confirming the Elantra platforms are really there. It catches the failure the in-process
guards structurally cannot: a push that landed somewhere other than where we thought.

Exits non-zero if the published branch would not support the car.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import NV_IFACE_REGISTRY, NV_SENTINELS

FORK_REPO = "aaravsaianugula/openpilot"
MAIN_BRANCH = "master"
ROLLBACK_BRANCH = "master-previous"
PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    # detail describes the consequence of failing, so it only makes sense on a failure
    if ok:
        print("  ok    " + label)
        return
    suffix = ": " + detail if detail else ""
    print("  FAIL  " + label + suffix)
    failures.append(label + suffix)


def gh(path: str):
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"GitHub API call failed: {path}\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def contents(repo: str, path: str, ref: str) -> str:
    blob = gh(f"repos/{repo}/contents/{path}?ref={ref}")
    return base64.b64decode(blob["content"]).decode("utf-8", "replace")


def commit_exists(repo: str, sha: str) -> bool:
    """Does this commit exist in this repo? gh() raises on 404, and here 404 is an answer."""
    proc = subprocess.run(["gh", "api", f"repos/{repo}/commits/{sha}"],
                          capture_output=True, text=True)
    return proc.returncode == 0


def verify_tinygrad(head: str, manifest: dict) -> None:
    """The published superproject really pins a tinygrad that carries the NV-USB patch.

    Four claims, and any three can hold while the fourth fails:
      * .gitmodules names our fork         -- else the device clones stock tinygrad
      * the gitlink matches the manifest   -- else we published something we did not build
      * that commit exists in that fork    -- else `git submodule update` dies on the car with
                                              "reference is not a tree": offroad, but stopped
      * that commit defines the NV classes -- else it clones fine and the eGPU never inits

    Checked by AST over the fetched source, not by matching source lines. tinygrad reformats
    constantly, and a whitespace-exact literal is a check that goes red for a reason that has
    nothing to do with the car.
    """
    print("\n[tinygrad] the published pin carries eGPU support")
    if "egpu" not in manifest:
        print("  note  this build predates eGPU support; nothing to verify")
        return

    tg_repo = manifest.get("tinygrad_repo") or ""
    pinned = manifest.get("tinygrad_sha") or ""
    check("manifest names a tinygrad fork and pin", bool(tg_repo and pinned))
    if not (tg_repo and pinned):
        return

    gitmodules = contents(FORK_REPO, ".gitmodules", head)
    check(".gitmodules points tinygrad at " + tg_repo, tg_repo in gitmodules,
          "the device would clone stock tinygrad and the eGPU would never come up")

    entry = gh(f"repos/{FORK_REPO}/contents/tinygrad_repo?ref={head}")
    check("tinygrad_repo is still a submodule", entry.get("type") == "submodule")
    check("tinygrad gitlink matches the manifest", entry.get("sha") == pinned,
          f"gitlink {(entry.get('sha') or '')[:9]} vs manifest {pinned[:9]}")

    if not commit_exists(tg_repo, pinned):
        check(f"{pinned[:9]} exists in {tg_repo}", False,
              "git submodule update would fail on the car with 'reference is not a tree'")
        return
    check(f"{pinned[:9]} exists in {tg_repo}", True)

    for rel, wanted in NV_SENTINELS.items():
        try:
            tree = ast.parse(contents(tg_repo, rel, pinned))
        except (SyntaxError, SystemExit):
            check(rel + " parses in the pinned tinygrad", False)
            continue
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.add(("class", node.name))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found.add(("def", node.name))
        for kind, name in wanted:
            check(f"{rel}: {kind} {name} defined", (kind, name) in found,
                  "the published pin does not carry the NV-USB delta")

    rel, cls, attr, want = NV_IFACE_REGISTRY
    ifaces: list[str] = []
    for node in ast.walk(ast.parse(contents(tg_repo, rel, pinned))):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == attr for t in stmt.targets):
                    if isinstance(stmt.value, ast.List | ast.Tuple):
                        ifaces = [e.id for e in stmt.value.elts if isinstance(e, ast.Name)]
    check(f"{want} is registered in {cls}.{attr}", want in ifaces,
          "the USB backend is defined but nothing will ever select it; found " + str(ifaces))


def verify_rollback_target(sha: str) -> None:
    """The rollback target has to be a working Elantra build in its own right.

    A rollback branch that exists but does not support the car is worse than none: it is a
    button that looks like an escape hatch and leaves you with an unrecognised vehicle. This
    resolves master-previous the whole way through its own gitlink, same as master.
    """
    print(f"\n  rollback target {sha[:9]}:")
    try:
        m = json.loads(contents(FORK_REPO, ".elantra/build-manifest.json", sha))
    except Exception as e:
        check("rollback target has a build manifest", False, str(e)[:120])
        return
    check("rollback target has a build manifest", True)

    gm = contents(FORK_REPO, ".gitmodules", sha)
    check("rollback target points at the Elantra opendbc",
          m.get("opendbc_repo", "aaravsaianugula/opendbc") in gm)

    pinned = gh(f"repos/{FORK_REPO}/contents/opendbc_repo?ref={sha}").get("sha")
    check("rollback target gitlink matches its manifest", pinned == m.get("opendbc_sha"))

    values = contents(m.get("opendbc_repo", "aaravsaianugula/opendbc"),
                      "opendbc/car/hyundai/values.py", pinned)
    for platform in PLATFORMS:
        check(f"rollback target supports {platform}", platform in values,
              "rolling back would leave the car unrecognised")


def main() -> int:
    print(f"Verifying {FORK_REPO}:{MAIN_BRANCH} as published\n")

    head = gh(f"repos/{FORK_REPO}/branches/{MAIN_BRANCH}")["commit"]["sha"]
    print(f"  head: {head}")

    manifest = json.loads(contents(FORK_REPO, ".elantra/build-manifest.json", head))
    check("build manifest present and parseable", True)

    # The rollback branch has to exist and be something other than master, or the panel's
    # rollback button leads nowhere.
    bootstrap = bool(manifest.get("bootstrap"))
    try:
        rb = gh(f"repos/{FORK_REPO}/branches/{ROLLBACK_BRANCH}")["commit"]["sha"]
        check(f"{ROLLBACK_BRANCH} exists", True)
        if bootstrap:
            # The very first build has nothing older worth rolling back to -- the branch it
            # replaced was a stale mirror with no Elantra support. The pointer exists so the
            # car's rollback button is wired up; it starts meaning something after sync #2.
            print(f"  note  {ROLLBACK_BRANCH} == {MAIN_BRANCH} on the bootstrap build")
        else:
            check(f"{ROLLBACK_BRANCH} differs from {MAIN_BRANCH}", rb != head,
                  "rollback would be a no-op")
            check(f"{ROLLBACK_BRANCH} matches the manifest",
                  rb == manifest.get("previous_master_sha"),
                  f"branch {rb[:9]} vs manifest {(manifest.get('previous_master_sha') or '')[:9]}")
            verify_rollback_target(rb)
    except SystemExit:
        check(f"{ROLLBACK_BRANCH} exists", False, "rollback from the car would fail")

    # .gitmodules must point at our opendbc, or the device builds stock opendbc and the car
    # is simply unsupported.
    gitmodules = contents(FORK_REPO, ".gitmodules", head)
    check("opendbc submodule points at the Elantra fork",
          manifest["opendbc_repo"] in gitmodules,
          "device would build stock opendbc")

    # Resolve the gitlink and read the real files out of the submodule commit.
    tree = gh(f"repos/{FORK_REPO}/contents/opendbc_repo?ref={head}")
    pinned = tree.get("sha")
    check("gitlink matches the manifest", pinned == manifest["opendbc_sha"],
          f"gitlink {(pinned or '')[:9]} vs manifest {manifest['opendbc_sha'][:9]}")

    odbc = manifest["opendbc_repo"]
    values = contents(odbc, "opendbc/car/hyundai/values.py", pinned)
    for platform in PLATFORMS:
        check(f"{platform} present in the pinned opendbc", platform in values)

    safety = contents(odbc, "opendbc/safety/modes/hyundai.h", pinned)
    dbc = contents(odbc, "opendbc/dbc/generator/hyundai/hyundai_can.dbc", pinned)
    # The same regexes guards.py uses. This was a whitespace-exact literal, which meant any
    # upstream reformat of hyundai.h would break the published-branch check for a reason with
    # nothing to do with the car -- and would disagree with guards.py, which still passed.
    safety_ok = re.search(r"\{\s*0x485\s*,\s*0\s*,\s*8\s*,", safety) is not None
    dbc_ok = re.search(r"^BO_\s+1157\s+LFAHDA_MFC:\s*8\s", dbc, re.MULTILINE) is not None
    check("panda safety allows 8 bytes on 0x485", safety_ok)
    check("dbc declares LFAHDA_MFC as 8 bytes", dbc_ok)
    check("safety and dbc agree on 0x485", safety_ok == dbc_ok and safety_ok,
          "one half of the CN7 frame widening is missing")

    car_list = json.loads(contents(odbc, "opendbc/sunnypilot/car/car_list.json", pinned))
    check("'Hyundai Elantra 2024-25' in the sunnypilot car list",
          "Hyundai Elantra 2024-25" in car_list)

    verify_tinygrad(head, manifest)

    print()
    if failures:
        print(f"PUBLISHED BRANCH IS BAD: {len(failures)} check(s) failed")
        for f in failures:
            print("  - " + f)
        print(f"\nRoll the car back to {ROLLBACK_BRANCH} from the Elantra port settings panel.")
        return 1
    print("Published branch verified: the car is supported by this build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
