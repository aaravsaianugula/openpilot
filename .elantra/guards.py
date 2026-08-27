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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import (
    FORK_TINYGRAD, NV_IFACE_REGISTRY, NV_SENTINELS, OVERLAY_ADDED, OVERLAY_HOOKS,
    OVERLAY_MODIFIED,
)

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
    check("opendbc submodule path is still opendbc_repo", "path = opendbc_repo" in gitmodules)
    check("tinygrad submodule path is still tinygrad_repo", "path = tinygrad_repo" in gitmodules)

    manifest_path = repo / ".elantra/build-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(read(manifest_path))
        for key in ("sunnypilot_upstream_sha", "opendbc_sha", "synced_at_utc",
                    "upstream_ci_conclusion", "elantra_platforms"):
            check("manifest carries " + key, key in manifest)
        check("manifest lists both Elantra platforms",
              set(PLATFORMS).issubset(set(manifest.get("elantra_platforms", []))))

        # The tinygrad keys only exist in manifests written after eGPU support landed. A
        # checkout of an older build is not a regression, so note it rather than failing --
        # but once a manifest claims eGPU support, the keys that describe it are mandatory,
        # so a later sync cannot quietly stop recording which tinygrad it pinned.
        if "egpu" not in manifest:
            print("  note  this manifest predates eGPU support; tinygrad keys not checked")
        else:
            for key in ("tinygrad_sha", "tinygrad_repo", "tinygrad_upstream_sha"):
                check("manifest carries " + key, key in manifest)


def _overlay_present(target: Path) -> bool:
    """A registered path counts as present if it is a file, or a directory holding one.

    The empty-directory case is the interesting one: `git checkout <ref> -- <dir>` can leave
    one behind after a rename upstream, and "the directory exists" would call that a pass.
    """
    if target.is_file():
        return True
    if target.is_dir():
        return any(p.is_file() for p in target.rglob("*") if "__pycache__" not in p.parts)
    return False


def guard_overlay_present(repo: Path) -> None:
    """Every path sync.py restores really is in the rebuilt tree.

    Generic on purpose. The named guards say what particular files *mean*; this one says the
    registry is not lying. Without it, adding a path to OVERLAY_ADDED and never checking it is
    how an overlay file gets restored one Monday and silently dropped the Monday after a
    rename upstream -- the sync would publish it happily, because the diff still applied.
    """
    print("\n[overlay] every registered overlay path survived the rebuild")
    check("the overlay registry is not empty", bool(OVERLAY_ADDED),
          "OVERLAY_ADDED is empty -- the sync would restore nothing at all")
    for path in OVERLAY_ADDED:
        check("overlay path " + path + " present", _overlay_present(repo / path),
              "registered in OVERLAY_ADDED, restored by sync.py, and not in the built tree")


def guard_overlay_hooks(repo: Path) -> None:
    """Every upstream file we modify still carries the modification.

    `git apply -3` can succeed and land a change somewhere useless if upstream moved the code
    around it, and more mundanely someone can hand-edit master and drop a line. This asserts
    the *effect* of the overlay rather than the fact that a patch applied.
    """
    print("\n[overlay] upstream files we modify still carry our hooks")
    # If OVERLAY_MODIFIED grows and nobody adds a hook, this guard quietly stops covering the
    # new file. Assert the two registries are the same set so that cannot happen unnoticed.
    drift = sorted(set(OVERLAY_MODIFIED) ^ set(OVERLAY_HOOKS))
    check("every overlay-modified file has a hook", not drift,
          "OVERLAY_MODIFIED and OVERLAY_HOOKS have drifted: " + ", ".join(drift))
    for path in sorted(OVERLAY_HOOKS):
        target = repo / path
        if not target.is_file():
            check(path + " exists", False, "an overlay-modified file is missing entirely")
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        for hook in OVERLAY_HOOKS[path]:
            check(path + ": " + hook, hook in text,
                  "the overlay applied but our change is not in the file")


def _defined(source: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            out.add(("class", node.name))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.add(("def", node.name))
    return out


def _class_list_attr(source: str, cls: str, attr: str) -> list[str] | None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == attr for t in stmt.targets):
                    if isinstance(stmt.value, ast.List | ast.Tuple):
                        return [e.id for e in stmt.value.elts if isinstance(e, ast.Name)]
                    return []
            return []
    return None



def _function_names_used(source: str, func: str) -> set[str]:
    """Every name called inside one top-level function. AST, so reflowing changes nothing."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return {n.attr if isinstance(n, ast.Attribute) else n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Attribute) or isinstance(n, ast.Name)}
    return set()


def guard_egpu_asics(repo: Path) -> None:
    """A card tinygrad cannot drive must not be handed the driving model.

    tinygrad's AM driver is RDNA3/RDNA4 only. An RDNA2 card is still an AMD card, so the
    vendor gate waves it through, modeld commits to DEV=USB+AMD:LLVM, the device fails to
    open, and the 60s loader timeout puts modeld in a restart loop -- the car cannot engage
    until the dock is unplugged. The table is only worth anything if enabled() consults it,
    which is the claim that can rot silently, so it is checked rather than assumed.
    """
    print("\n[egpu] cards tinygrad cannot drive")
    asics_src = read(repo / "openpilot/sunnypilot/egpu/asics.py")
    detect_src = read(repo / "openpilot/sunnypilot/egpu/detect.py")

    defined = _defined(asics_src)
    check("asics.py defines am_supports", ("def", "am_supports") in defined)
    check("asics.py defines asic_for", ("def", "asic_for") in defined)
    check("the blocklist is present", "UNSUPPORTED_AMD" in asics_src)

    # The card actually sitting in this dock. Losing this entry is the specific regression
    # this branch exists to prevent, so it is named rather than counted.
    check("the RX 6600 XT (Navi 23, 0x73ff) is blocked", "0x73FF" in asics_src,
          "the gate would let an RDNA2 card take the model and hang modeld")

    used = _function_names_used(detect_src, "enabled")
    check("enabled() actually consults the blocklist", "am_supports" in used,
          "asics.py would be decoration; every card would still be waved through")
    check("enabled() resolves the device id", "resolve_device" in used)


def guard_tinygrad(repo: Path, tinygrad: Path | None) -> None:
    """The eGPU half of the overlay: the submodule points at our tinygrad, and it is patched.

    Three separate claims, because any two can hold while the third fails:
      * .gitmodules names our fork    -- else the device clones stock tinygrad
      * the tree is actually here     -- else there is nothing to check
      * the NV classes are defined    -- else it clones fine and the eGPU never initialises
    """
    print("\n[tinygrad] NV-USB eGPU patch")
    gitmodules = read(repo / ".gitmodules")
    check("tinygrad submodule points at the patched fork", FORK_TINYGRAD in gitmodules,
          "the device would clone stock tinygrad and the eGPU would never come up")

    if tinygrad is None:
        print("  note  no tinygrad checkout given; the pin's contents were not inspected")
        return

    for rel, wanted in NV_SENTINELS.items():
        defined = _defined(read(tinygrad / rel))
        for kind, name in wanted:
            check(rel + ": " + kind + " " + name + " defined", (kind, name) in defined,
                  "the NV-USB delta did not survive the replay onto sunnypilot's tinygrad pin")

    # A class that exists and is never registered is a class that never runs.
    rel, cls, attr, want = NV_IFACE_REGISTRY
    ifaces = _class_list_attr(read(tinygrad / rel), cls, attr)
    check(cls + "." + attr + " exists", ifaces is not None,
          "the NV device class was restructured upstream -- this guard no longer checks anything")
    if ifaces is not None:
        check(want + " is registered in " + cls + "." + attr, want in ifaces,
              "the USB backend is defined but nothing will ever select it; found " + str(ifaces))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opendbc", required=True, type=Path,
                    help="path to the opendbc checkout carrying the Elantra delta")
    ap.add_argument("--repo", type=Path, default=None,
                    help="path to the sunnypilot superproject (optional)")
    ap.add_argument("--tinygrad", type=Path, default=None,
                    help="path to the rebuilt tinygrad checkout (optional)")
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
        repo = args.repo.resolve()
        guard_superproject(repo)
        guard_overlay_present(repo)
        guard_overlay_hooks(repo)
        guard_egpu_asics(repo)
        guard_tinygrad(repo, args.tinygrad.resolve() if args.tinygrad else None)

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
