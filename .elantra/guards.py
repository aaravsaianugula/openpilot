#!/usr/bin/env python3
"""
Structural guards for the Elantra 2024-25 overlay.

These run before anything is ever pushed. They are deliberately text/AST level so they work
on any machine with a bare Python 3 and no opendbc dependencies installed -- the point is to
prove the port survived a rebase onto new upstream code, not to replace opendbc's own test
suite. CI runs that suite separately; both gates must pass.

Every check raises on failure. Nothing here warns and continues: a guard that can be skipped
is not a guard.

Usage:
    python .elantra/guards.py --opendbc <path-to-opendbc-checkout> [--repo <superproject>]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")

# The two halves of the CN7 2024 LFAHDA_MFC widening. These must always agree: the dbc says
# how many bytes openpilot packs into 0x485, the safety code says how many bytes panda will
# allow out. A mismatch is not cosmetic drift -- it is either a car that refuses to steer or
# a safety allow-list wider than the message it guards.
LFAHDA_MFC_LEN = 8

EXPECTED_FLAGS = {
    "HYUNDAI_ELANTRA_2024": {"CHECKSUM_CRC8", "CAMERA_SCC"},
    "HYUNDAI_ELANTRA_HEV_2024": {"CHECKSUM_CRC8", "CAMERA_SCC", "HYBRID"},
}

EXPECTED_CAR_LIST = {
    "Hyundai Elantra 2024-25": "HYUNDAI_ELANTRA_2024",
    "Hyundai Elantra Hybrid 2024-25": "HYUNDAI_ELANTRA_HEV_2024",
    "Hyundai i30 Hybrid 2024": "HYUNDAI_ELANTRA_HEV_2024",
}

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


def read(path: Path) -> str:
    if not path.is_file():
        raise SystemExit("guard setup error: expected file is missing: " + str(path))
    return path.read_text(encoding="utf-8", errors="replace")


def _flags_in_assignment(source: str, platform: str) -> set[str] | None:
    """Pull the HyundaiFlags.* names out of a platform's flags= argument via AST.

    Returns None when the platform is absent entirely, so the caller can tell "not defined"
    apart from "defined with the wrong flags".
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if platform not in targets:
            continue
        if not isinstance(node.value, ast.Call):
            return set()
        for kw in node.value.keywords:
            if kw.arg != "flags":
                continue
            found: set[str] = set()
            for sub in ast.walk(kw.value):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                    if sub.value.id == "HyundaiFlags":
                        found.add(sub.attr)
            return found
        return set()
    return None


def guard_values(opendbc: Path) -> None:
    print("\n[values.py] platform definitions")
    source = read(opendbc / "opendbc/car/hyundai/values.py")
    for platform in PLATFORMS:
        flags = _flags_in_assignment(source, platform)
        check(platform + " is defined", flags is not None,
              "platform vanished from CAR -- the port did not survive the rebase")
        if flags is None:
            continue
        expected = EXPECTED_FLAGS[platform]
        check(platform + " carries " + " | ".join(sorted(expected)),
              expected.issubset(flags),
              "expected " + str(sorted(expected)) + ", found " + str(sorted(flags)))
    # CAMERA_SCC is what makes this port work at all on the CN7: SCC lives on the camera, not
    # a separate radar module. Losing it silently gives a car that fingerprints but never
    # engages.
    check("both platforms are CAMERA_SCC",
          all((_flags_in_assignment(source, p) or set()) >= {"CAMERA_SCC"} for p in PLATFORMS))
    check("Hyundai K harness referenced for the 2024 platforms",
          source.count("hyundai_k") >= 2)


def guard_fingerprints(opendbc: Path) -> None:
    print("\n[fingerprints.py] firmware fingerprints")
    source = read(opendbc / "opendbc/car/hyundai/fingerprints.py")
    for platform in PLATFORMS:
        check(platform + " has a fingerprint block", "CAR." + platform + ":" in source)
    # Without a camera FW entry the car cannot be recognised at all.
    check("2024 camera firmware present (99210-AA500/AA510)",
          "99210-AA500" in source or "99210-AA510" in source)


def guard_hyundaican(opendbc: Path) -> None:
    print("\n[hyundaican.py] LKAS11 LDWS mode list")
    source = read(opendbc / "opendbc/car/hyundai/hyundaican.py")
    for platform in PLATFORMS:
        check(platform + " in the LDWS-active-mode list", "CAR." + platform in source)


def guard_lfahda_pair(opendbc: Path) -> None:
    """The safety-critical one. dbc byte count and panda TX length must agree."""
    print("\n[dbc <-> safety] LFAHDA_MFC 0x485 length agreement")

    dbc_src = read(opendbc / "opendbc/dbc/generator/hyundai/hyundai_can.dbc")
    dbc_match = re.search(r"^BO_\s+1157\s+LFAHDA_MFC:\s*(\d+)\s", dbc_src, re.MULTILINE)
    check("LFAHDA_MFC message found in hyundai_can.dbc", dbc_match is not None)
    dbc_len = int(dbc_match.group(1)) if dbc_match else -1
    check("hyundai_can.dbc declares LFAHDA_MFC as " + str(LFAHDA_MFC_LEN) + " bytes",
          dbc_len == LFAHDA_MFC_LEN, "found " + str(dbc_len))

    safety_src = read(opendbc / "opendbc/safety/modes/hyundai.h")
    safety_match = re.search(r"\{\s*0x485\s*,\s*0\s*,\s*(\d+)\s*,", safety_src)
    check("0x485 TX entry found in safety/modes/hyundai.h", safety_match is not None)
    safety_len = int(safety_match.group(1)) if safety_match else -1
    check("panda safety allows " + str(LFAHDA_MFC_LEN) + " bytes on 0x485",
          safety_len == LFAHDA_MFC_LEN, "found " + str(safety_len))

    check("dbc and safety lengths agree",
          dbc_len == safety_len and dbc_len == LFAHDA_MFC_LEN,
          "dbc=" + str(dbc_len) + " safety=" + str(safety_len) + " -- these must never diverge")


def guard_car_list(opendbc: Path) -> None:
    print("\n[car_list.json] sunnypilot vehicle list")
    raw = read(opendbc / "opendbc/sunnypilot/car/car_list.json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        check("car_list.json parses as JSON", False, str(e))
        return
    check("car_list.json parses as JSON", True)
    for name, platform in EXPECTED_CAR_LIST.items():
        entry = data.get(name)
        check(repr(name) + " listed", entry is not None)
        if entry is not None:
            check(repr(name) + " -> " + platform, entry.get("platform") == platform,
                  "found " + repr(entry.get("platform")))


def guard_torque(opendbc: Path) -> None:
    print("\n[substitute.toml] torque parameters")
    source = read(opendbc / "opendbc/car/torque_data/substitute.toml")
    for platform in PLATFORMS:
        check(platform + " has a torque substitute", '"' + platform + '"' in source)


def guard_superproject(repo: Path) -> None:
    print("\n[superproject] submodule wiring")
    gitmodules = read(repo / ".gitmodules")
    check("opendbc submodule points at the Elantra-enabled fork",
          "aaravsaianugula/opendbc" in gitmodules,
          "the .gitmodules overlay did not apply -- the build would use stock opendbc and the car would not be supported")
    check("opendbc submodule path is still opendbc_repo", "path = opendbc_repo" in gitmodules)

    # Every file the panel needs must have survived the rebuild. A dropped overlay file is
    # an ImportError on the car's settings screen, and the sync would otherwise publish it
    # happily because the *diff* still applied cleanly.
    ui_dir = repo / "openpilot/selfdrive/ui/sunnypilot/mici/layouts"
    for name in ("port_updates.py", "port_manifest.py"):
        check(f"UI overlay file {name} present", (ui_dir / name).is_file(),
              "the settings panel would fail to import on the device")
    settings = ui_dir / "settings.py"
    if settings.is_file():
        text = settings.read_text(encoding="utf-8", errors="replace")
        check("port panel is registered in the mici settings",
              "ElantraPortLayoutMici" in text and "port_btn" in text,
              "the panel exists but nothing opens it")

    manifest_path = repo / ".elantra/build-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(read(manifest_path))
        for key in ("sunnypilot_upstream_sha", "opendbc_sha", "synced_at_utc",
                    "upstream_ci_conclusion", "elantra_platforms"):
            check("manifest carries " + key, key in manifest)
        check("manifest lists both Elantra platforms",
              set(PLATFORMS).issubset(set(manifest.get("elantra_platforms", []))))


def _git(cwd: Path, *args: str) -> str | None:
    """Run git and return stripped stdout, or None if it fails for any reason.

    Guards must never raise: sync.py reads a non-zero exit as "do not publish", and an
    exception here would be indistinguishable from a real divergence.
    """
    try:
        out = subprocess.run(("git", *args), cwd=str(cwd), capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _gitlink_sha(repo: Path, path: str) -> str | None:
    """The commit the superproject's index pins for a submodule path, or None."""
    line = _git(repo, "ls-files", "-s", "--", path)
    if not line:
        return None
    # "160000 <sha> 0\t<path>" -- any mode other than 160000 is not a gitlink.
    fields = line.split()
    if len(fields) < 2 or fields[0] != "160000":
        return None
    return fields[1]


def guard_opendbc_pin(repo: Path, opendbc: Path) -> None:
    """The tree the other guards just read must be the tree the superproject actually ships.

    Every guard above opens files under --opendbc. Not one of them looks at the gitlink. So a
    superproject branch can pass all of them while pinning an opendbc commit that has never
    heard of the change being guarded -- which is not hypothetical: elantra-torque-test carries
    the guards for a 409 schedule and pins 69e2e548, which contains none of it, and
    `git diff master elantra-torque-test -- opendbc_repo` is empty. Every guard was green and
    the car would have run stock code.

    Equality is the whole check. If the pinned commit is the checkout the guards just verified,
    then whatever they proved is what actually gets built.
    """
    print("\n[superproject <-> opendbc] the guarded tree is the pinned tree")
    pinned = _gitlink_sha(repo, "opendbc_repo")
    check("superproject records an opendbc_repo gitlink", pinned is not None,
          "git ls-files -s opendbc_repo returned no mode-160000 entry")
    if pinned is None:
        return

    head = _git(opendbc, "rev-parse", "HEAD")
    check("the guarded opendbc checkout resolves to a commit", head is not None,
          "--opendbc is not a git checkout, so the pin cannot be proven")
    if head is None:
        return

    check("pinned opendbc is the opendbc these guards just checked", pinned == head,
          "superproject pins " + pinned[:12] + " but --opendbc is at " + head[:12]
          + " -- these guards passed against a tree the build will not use")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opendbc", required=True, type=Path,
                    help="path to the opendbc checkout carrying the Elantra delta")
    ap.add_argument("--repo", type=Path, default=None,
                    help="path to the sunnypilot superproject (optional)")
    args = ap.parse_args()

    opendbc = args.opendbc.resolve()
    if not (opendbc / "opendbc/car/hyundai/values.py").is_file():
        raise SystemExit("not an opendbc checkout: " + str(opendbc))

    print("Elantra 2024-25 overlay guards")
    print("  opendbc: " + str(opendbc))
    if args.repo:
        print("  repo:    " + str(args.repo.resolve()))

    guard_values(opendbc)
    guard_fingerprints(opendbc)
    guard_hyundaican(opendbc)
    guard_lfahda_pair(opendbc)
    guard_car_list(opendbc)
    guard_torque(opendbc)
    if args.repo:
        guard_superproject(args.repo.resolve())
        guard_opendbc_pin(args.repo.resolve(), opendbc)

    print("\n" + "-" * 60)
    if _failures:
        print("FAILED: " + str(len(_failures)) + " guard(s) failed, "
              + str(len(_passes)) + " passed\n")
        for f in _failures:
            print("  - " + f)
        print("\nNothing will be published. Elantra support is not intact in this tree.")
        return 1
    print("PASSED: all " + str(len(_passes)) + " guards green -- Elantra support is intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
