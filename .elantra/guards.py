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

PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")

# The two halves of the CN7 2024 LFAHDA_MFC widening. These must always agree: the dbc says
# how many bytes openpilot packs into 0x485, the safety code says how many bytes panda will
# allow out. A mismatch is not cosmetic drift -- it is either a car that refuses to steer or
# a safety allow-list wider than the message it guards.
LFAHDA_MFC_LEN = 8

# The two halves of the CN7 2024 speed-scheduled steering torque: [m/s] -> [CAN counts].
# opendbc multiplies every command by this; panda accepts frames up to it. Below the top
# breakpoint the car gets more authority to reach the lateral acceleration the planner
# already asks for; at and above it, both numbers are the stock 384 and nothing changes.
STEER_MAX_SCHEDULE = ([8.0, 16.0], [409, 384])

EXPECTED_FLAGS = {
    "HYUNDAI_ELANTRA_2024": {"CHECKSUM_CRC8", "CAMERA_SCC", "DYNAMIC_LIMITS"},
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


def guard_dynamic_torque_pair(opendbc: Path) -> None:
    """The other safety-critical pair: opendbc's torque gain and panda's ceiling.

    opendbc multiplies every steering command by the value it reads out of
    STEER_MAX_LOOKUP_DYNAMIC; panda accepts frames up to the value it reads out of
    HYUNDAI_LIMITS_DYNAMIC. If the two drift apart, panda starts dropping LKAS11 frames
    openpilot is legitimately sending, the stream to the MDPS stops, and the EPS faults.
    Same class of coupled edit as LFAHDA_MFC above, and just as invisible until it bites.
    """
    print("\n[values.py <-> safety] speed-scheduled steering torque agreement")

    values_src = read(opendbc / "opendbc/car/hyundai/values.py")
    m = re.search(r"STEER_MAX_LOOKUP_DYNAMIC\s*=\s*(\[[^\]]*\])\s*,\s*(\[[^\]]*\])", values_src)
    check("STEER_MAX_LOOKUP_DYNAMIC found in values.py", m is not None)
    if m is None:
        return
    op_speeds = [float(v) for v in ast.literal_eval(m.group(1))]
    op_torques = [int(v) for v in ast.literal_eval(m.group(2))]
    check("opendbc schedule is " + str(STEER_MAX_SCHEDULE),
          (op_speeds, op_torques) == (list(STEER_MAX_SCHEDULE[0]), list(STEER_MAX_SCHEDULE[1])),
          "found " + str((op_speeds, op_torques)))

    safety_src = read(opendbc / "opendbc/safety/modes/hyundai.h")
    use = re.search(r"HYUNDAI_LIMITS_DYNAMIC\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", safety_src)
    check("HYUNDAI_LIMITS_DYNAMIC is used in safety/modes/hyundai.h", use is not None)
    if use is None:
        return
    pa_torques = [int(use.group(1)), int(use.group(2))]
    check("panda torques match opendbc's", pa_torques == op_torques,
          "panda=" + str(pa_torques) + " opendbc=" + str(op_torques)
          + " -- these must never diverge")

    bp = re.search(r"\.max_torque_lookup\s*=\s*\{[^{]*\{([^}]*)\}", safety_src)
    check("panda breakpoints found", bp is not None)
    if bp is not None:
        pa_speeds = [float(v) for v in bp.group(1).replace("\\", "").split(",") if v.strip()]
        # lookup_t holds exactly 3 points; the schedule repeats the top one
        check("panda breakpoints match opendbc's",
              pa_speeds[:2] == op_speeds and pa_speeds[2] == op_speeds[-1],
              "panda=" + str(pa_speeds) + " opendbc=" + str(op_speeds))

    # the flag has to mean the same number on both sides, or the limit silently never applies
    py_flag = re.search(r"^\s*DYNAMIC_LIMITS\s*=\s*(\d+)\s*$", values_src, re.MULTILINE)
    c_flag = re.search(r"HYUNDAI_PARAM_DYNAMIC_LIMITS\s*=\s*(\d+)",
                       read(opendbc / "opendbc/safety/modes/hyundai_common.h"))
    check("HyundaiSafetyFlags.DYNAMIC_LIMITS is defined", py_flag is not None)
    check("HYUNDAI_PARAM_DYNAMIC_LIMITS is defined", c_flag is not None)
    if py_flag and c_flag:
        check("safety flag values agree", py_flag.group(1) == c_flag.group(1),
              "python=" + py_flag.group(1) + " c=" + c_flag.group(1))

    # rate limits stay stock: max_rt_delta 112 over a 250 ms interval at 100 Hz caps the ramp
    # at 4.48 counts/frame anyway, so anything above 4 would be unreachable dead code
    check("steer ramp rates are still 3/7",
          (int(use.group(3)), int(use.group(4))) == (3, 7),
          "found " + str((int(use.group(3)), int(use.group(4)))))

    # Matching numbers are not enough. HyundaiFlags.DYNAMIC_LIMITS (CP.flags, gates the opendbc
    # gain) and HyundaiSafetyFlags.DYNAMIC_LIMITS (safetyParam, gates the panda ceiling) are two
    # different flags in two different words, and the block in interface.py is the only bridge
    # between them. Delete that block and every number checked above still agrees -- but opendbc
    # keeps applying the 409 gain while panda keeps enforcing 384, so every LKAS11 frame below
    # 16 m/s is rejected, the stream to the MDPS stops, and the EPS faults mid-turn. That is the
    # exact failure this guard exists to prevent, so guard the whole chain and not just the two
    # numbers at its ends.
    iface_src = read(opendbc / "opendbc/car/hyundai/interface.py")
    bridge = r"ret\.flags\s*&\s*HyundaiFlags\.DYNAMIC_LIMITS\s*:\s*ret\.safetyConfigs\[-1\]\.safetyParam\s*\|=\s*HyundaiSafetyFlags\.DYNAMIC_LIMITS"
    check("interface.py carries the car flag into safetyParam",
          re.search(bridge, iface_src) is not None,
          "the car flag and the panda flag are different words -- without this bridge opendbc "
          + "commands 409 while panda still enforces 384, and the low-speed frames are dropped")

    check("panda selects the dynamic limits on that flag",
          re.search(r"hyundai_dynamic_limits\s*\?\s*HYUNDAI_STEERING_LIMITS_DYNAMIC", safety_src) is not None,
          "HYUNDAI_LIMITS_DYNAMIC is defined but never reached from the tx hook")

    cc_src = read(opendbc / "opendbc/car/hyundai/carcontroller.py")
    check("carcontroller applies the schedule to the command",
          re.search(r"np\.interp\(\s*CS\.out\.vEgoRaw\s*,\s*\*self\.params\.STEER_MAX_LOOKUP\s*\)",
                    cc_src) is not None,
          "STEER_MAX_LOOKUP exists but nothing reads it, so the schedule is inert")

    check("carcontroller passes the scheduled ceiling to the rate limiter",
          re.search(r"apply_driver_steer_torque_limits\([^)]*steer_max\s*\)", cc_src) is not None,
          "without it the driver-torque clamp stays anchored to the static STEER_MAX while the "
          + "command is scaled by the scheduled one")


def _const_int(node: ast.AST) -> int | None:
    """Evaluate a constant integer expression from the AST, or return None.

    Deliberately not eval() and not literal_eval(): the first is unsafe on a file we are
    auditing, the second cannot see `2 ** 27`. Only these node types are understood, so
    anything referring to a name or calling anything is simply not a value we check.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_int(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp):
        left, right = _const_int(node.left), _const_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow):
            return left ** right if 0 <= right <= 64 else None
        if isinstance(node.op, ast.LShift):
            return left << right if 0 <= right <= 64 else None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def guard_no_aliased_flags(opendbc: Path) -> None:
    """No two members of HyundaiFlags or HyundaiSafetyFlags may share a value.

    Python's IntFlag does not reject a duplicate -- it makes the second member a silent alias
    of the first. Two flags that collide therefore test True for each other, so a platform
    carrying one silently acquires the behaviour of the other, and nothing raises anywhere.

    Not hypothetical: the community elantra-2024-port lineage, which is what is installed on
    the car, defines LFAHDA_MFC_8 as 1024 in HyundaiSafetyFlags and 2**27 in HyundaiFlags, and
    sets it on both HYUNDAI_ELANTRA_2024 and HYUNDAI_ELANTRA_HEV_2024. DYNAMIC_LIMITS
    originally took both of those values. Landing it there would have handed the Elantra
    Hybrid a raised low-speed steering torque ceiling that nobody opted it into.
    """
    print("\n[values.py] no two flags share a value")

    src = read(opendbc / "opendbc/car/hyundai/values.py")
    tree = ast.parse(src)
    for enum_name in ("HyundaiFlags", "HyundaiSafetyFlags"):
        node = None
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == enum_name:
                node = n
                break
        check(enum_name + " is defined", node is not None)
        if node is None:
            continue
        seen = {}
        aliases = []
        members = 0
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            # NOT ast.literal_eval: these enums are written as `2 ** 27`, which is a BinOp and
            # not a literal, so literal_eval skips every one of them and this check passes
            # vacuously -- which is exactly how it was first written, and it caught nothing.
            value = _const_int(stmt.value)
            if value is None:
                continue
            members += 1
            if value in seen:
                aliases.append(target.id + " aliases " + seen[value] + " (" + str(value) + ")")
            else:
                seen[value] = target.id
        # a count of zero means the parser stopped understanding how the enum is written and
        # the alias check silently became a no-op, which is how the first version of this
        # guard "passed" against a genuine collision
        check(enum_name + " members were actually parsed", members > 5,
              "only " + str(members) + " parsed -- the alias check is not testing anything")
        check(enum_name + " has no aliased members (" + str(members) + " checked)",
              not aliases, "; ".join(aliases))


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
    guard_dynamic_torque_pair(opendbc)
    guard_no_aliased_flags(opendbc)
    guard_car_list(opendbc)
    guard_torque(opendbc)
    if args.repo:
        guard_superproject(args.repo.resolve())

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
