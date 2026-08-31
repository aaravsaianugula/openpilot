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

import base64
import json
import subprocess
import sys

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

    # 0x485 is 8 bytes on the CN7 bus and 4 on every other Hyundai CAN platform. panda's
    # allow-list matches on EXACT length, so it must carry both entries and hyundai_tx_hook
    # picks by flag. Checking only the CN7 half would stay green while the SHARED dbc was
    # widened for all ~79 of them, which is the divergence this file exists to catch.
    safety = contents(odbc, "opendbc/safety/modes/hyundai.h", pinned)
    cn7_dbc = contents(odbc, "opendbc/dbc/generator/hyundai/hyundai_can_cn7.dbc", pinned)
    shared_dbc = contents(odbc, "opendbc/dbc/generator/hyundai/hyundai_can.dbc", pinned)
    flat = safety.replace(chr(9), " ")
    safety_8 = "{0x485, 0,       8," in flat
    safety_4 = "{0x485, 0,       4," in flat
    cn7_ok = "BO_ 1157 LFAHDA_MFC: 8" in cn7_dbc
    shared_ok = "BO_ 1157 LFAHDA_MFC: 4" in shared_dbc
    check("panda allow-list carries the 8-byte 0x485 (for the CN7)", safety_8)
    check("panda allow-list still carries the 4-byte 0x485 (everyone else)", safety_4,
          "a single entry silently blocks whichever platform uses the other length")
    check("hyundai_can_cn7.dbc declares LFAHDA_MFC as 8 bytes", cn7_ok)
    check("the SHARED hyundai_can.dbc is still 4 bytes", shared_ok,
          "widening the shared dbc changes every Hyundai CAN platform, not just the CN7")
    check("hyundai_tx_hook ties 0x485 length to the flag",
          "hyundai_lfahda_mfc_8 ? 8U : 4U" in flat,
          "without this both lengths are accepted for every platform")
    check("safety and dbc agree on 0x485", safety_8 and safety_4 and cn7_ok and shared_ok,
          "one half of the CN7 frame widening is missing")

    car_list = json.loads(contents(odbc, "opendbc/sunnypilot/car/car_list.json", pinned))
    check("'Hyundai Elantra 2024-25' in the sunnypilot car list",
          "Hyundai Elantra 2024-25" in car_list)

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
