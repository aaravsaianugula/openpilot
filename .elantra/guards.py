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

# The CN7 speed-scheduled steering torque ceiling, stated here as the third opinion.
# opendbc decides what the car commands, panda decides what it will let out, and these
# literals are what both are checked against -- a guard that only compared the two against
# each other would stay green while they moved together.
STEER_MAX_SCHEDULE = ([8.0, 16.0], [409, 384])
STEER_RATES = (3, 7)

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


def _int_expr(node) -> int | None:
    """Evaluate the integer forms these enums are written in, and nothing else.

    Deliberately neither eval nor ast.literal_eval: the former is an arbitrary code path in a
    guard that is supposed to be inert, and the latter cannot handle `2 ** 27`, which is how
    most of HyundaiFlags is written. An earlier version of this guard used literal_eval,
    silently parsed zero members, and passed against a real flag collision. Anything outside
    this grammar returns None and the caller reports it as a failure.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift)):
        left, right = _int_expr(node.left), _int_expr(node.right)
        if left is None or right is None or right < 0 or right > 64:
            return None
        return left ** right if isinstance(node.op, ast.Pow) else left << right
    return None


def _enum_members(source: str, enum_name: str) -> dict:
    """{member: value} for an IntFlag class; None where a value is not an integer expression.
    Callers must assert the dict is non-empty and fully evaluated."""
    out: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name != enum_name:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                out[stmt.targets[0].id] = _int_expr(stmt.value)
    return out


def _lookup_assigned_in_else(source: str) -> bool:
    """Is STEER_MAX_LOOKUP assigned inside the `else` of the chain that picks STEER_MAX?

    Precedence is the whole point. Assigned after the chain instead of inside its last branch,
    the schedule silently outranks every other ceiling: a car carrying DYNAMIC_LIMITS together
    with ALT_LIMITS_2 would command 409 at 5 m/s while panda enforced 170. Text matching cannot
    see the difference, so this walks the tree.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.If):
                continue
            # the tail `else` of an if/elif chain, i.e. one whose orelse is not another If
            if not stmt.orelse or isinstance(stmt.orelse[0], ast.If):
                continue
            body = "\n".join(ast.dump(x) for x in stmt.orelse)
            if "STEER_MAX_LOOKUP" in body and "STEER_MAX_LOOKUP_DYNAMIC" in body:
                return True
    return False


def guard_panda_enforcement(opendbc: Path) -> None:
    """The four details in lateral.h that make panda's schedule usable as a backstop.

    Every other check in this file reads a DECLARATION -- the macro body, the lookup row, the
    flag word. None of them reads the code that consumes those declarations. lateral.h decides
    what the numbers mean, and all four details below exist to keep panda strictly more
    permissive than what openpilot commands. Change any one of them upstream and every other
    guard here stays green while the two halves silently stop agreeing.
    """
    print("\n" + "[lateral.h] what panda actually enforces")
    lat = read(opendbc / "opendbc/safety/lateral.h")

    m = re.search(r"if\s*\(limits\.dynamic_max_torque\)\s*\{(.*?)\n\s*\}", lat, re.S)
    check("lateral.h has a dynamic_max_torque branch", m is not None)
    if m is None:
        return
    body = m.group(1)

    # vehicle_speed.min, not .max and not .values[0]. The minimum over the sample window is the
    # slowest recent reading, and on a descending schedule slower means MORE torque allowed --
    # the permissive direction. .max would let panda reject what openpilot legitimately commands.
    check("panda interpolates on vehicle_speed.min",
          re.search(r"vehicle_speed\.min\s*/\s*VEHICLE_SPEED_FACTOR", body) is not None)

    # The -1 m/s shift and the +1 count are the entire margin between the two halves. openpilot
    # interpolates on instantaneous vEgoRaw with no fudge at all.
    check("panda shifts the speed down by 1 m/s before interpolating",
          re.search(r"-\s*1\.", body) is not None)
    check("panda adds one count to the interpolated ceiling",
          re.search(r"safety_interpolate\([^)]*\)\s*\+\s*1", body) is not None)

    # Without the clamp the +1 could push the low-speed end past .max_torque, which is the only
    # absolute bound anywhere in this path.
    check("panda clamps the interpolated ceiling to +/- max_torque",
          re.search(r"SAFETY_CLAMP\(\s*max_torque\s*,\s*-\s*limits\.max_torque\s*," +
                    r"\s*limits\.max_torque\s*\)", body) is not None)

    # That slack is dynamic-path-only. A write-up on this branch claimed it was present on the
    # static limits too; it is not, and if it ever moves outside the branch every static Hyundai
    # silently gains a count of headroom.
    before = lat[:m.start()]
    static_init = re.search(r"int\s+max_torque\s*=\s*limits\.max_torque\s*;", before)
    check("the +1 slack is inside the dynamic branch, not on the static path",
          static_init is not None and "+ 1" not in before[static_init.end():])


def guard_dynamic_torque_pair(opendbc: Path) -> None:
    """The CN7 low-speed torque schedule, end to end.

    The schedule lives in five files that cannot see each other, and every link has a failure
    mode that looks exactly like success. Each check below exists because breaking that link
    alone leaves every other check green.
    """
    print("\n[values.py <-> safety] speed-scheduled steering torque")
    values = read(opendbc / "opendbc/car/hyundai/values.py")
    carctl = read(opendbc / "opendbc/car/hyundai/carcontroller.py")
    iface = read(opendbc / "opendbc/car/hyundai/interface.py")
    safety = read(opendbc / "opendbc/safety/modes/hyundai.h")
    common_h = read(opendbc / "opendbc/safety/modes/hyundai_common.h")

    bps, torques = STEER_MAX_SCHEDULE
    low, high = torques
    rate_up, rate_down = STEER_RATES

    # --- the opendbc half -------------------------------------------------------------
    m = re.search(r"STEER_MAX_LOOKUP_DYNAMIC\s*=\s*(\[\[.*?\]\])", values, re.S)
    check("opendbc declares STEER_MAX_LOOKUP_DYNAMIC", m is not None)
    if m is not None:
        try:
            got = ast.literal_eval(m.group(1))
        except (SyntaxError, ValueError):
            got = None
        check("opendbc schedule is " + str(STEER_MAX_SCHEDULE),
              got == [list(bps), list(torques)], "found " + str(got))

    check("the schedule is assigned inside the else branch, not after the chain",
          _lookup_assigned_in_else(values),
          "assigned outside it, DYNAMIC_LIMITS outranks ALT_LIMITS/ALT_LIMITS_2/CANFD and the "
          "car commands " + str(low) + " where panda enforces 270 or 170")

    check("carcontroller interpolates the schedule on vEgoRaw",
          "np.interp(CS.out.vEgoRaw, *self.params.STEER_MAX_LOOKUP)" in carctl)
    check("carcontroller hands the scheduled ceiling to the rate limiter",
          re.search(r"apply_driver_steer_torque_limits\([^)]*self\.params,\s*steer_max\)",
                    carctl, re.S) is not None,
          "the rate limiter would bound the driver-override envelope at the static STEER_MAX "
          "while the command used the scheduled one")
    check("carcontroller normalises the feedback by the same ceiling",
          "new_actuators.torque = apply_torque / steer_max" in carctl,
          "dividing by the static STEER_MAX reports a fraction the controller never asked for")

    # --- the bridge -------------------------------------------------------------------
    check("interface.py carries the car flag into safetyParam",
          re.search(r"HyundaiFlags\.DYNAMIC_LIMITS[\s\S]{0,120}?"
                    r"HyundaiSafetyFlags\.DYNAMIC_LIMITS\.value", iface) is not None,
          "without this the car commands the raised torque and panda rejects every frame")

    # --- the panda half ---------------------------------------------------------------
    macro = re.search(r"#define HYUNDAI_LIMITS_DYNAMIC\(([^)]*)\)\s*\{(.*?)\n\}",
                      safety, re.S)
    check("panda defines HYUNDAI_LIMITS_DYNAMIC", macro is not None)
    if macro is not None:
        body = macro.group(2)
        # .max_torque doubles as the SAFETY_CLAMP bound on the interpolated value, so it has
        # to be the LOW-speed number. Set to the high one, the schedule is capped at 384 and
        # the whole change is a no-op that still passes an arguments-only check.
        check("panda's max_torque is the low-speed value",
              re.search(r"\.max_torque\s*=\s*\(steer_low\)", body) is not None,
              "the schedule would be clamped to the high-speed ceiling and do nothing")
        check("panda enables the dynamic lookup",
              re.search(r"\.dynamic_max_torque\s*=\s*true", body) is not None)
        # Order, not just contents: reversed, panda rejects the raised torque at low speed and
        # accepts it on the highway, and a set-comparison cannot tell the two apart.
        check("panda's torque row runs low-speed first",
              re.search(r"\{\s*\(steer_low\)\s*,\s*\(steer_high\)\s*,\s*\(steer_high\)\s*,?\s*\}",
                        body) is not None,
              "inverted, panda rejects " + str(low) + " at low speed and accepts it on the highway")
        check("panda's breakpoints match opendbc's",
              re.search(r"\{\s*%g\.\s*,\s*%g\.\s*,\s*%g\.\s*,?\s*\}" % (bps[0], bps[1], bps[1]),
                        body) is not None,
              "expected {%g., %g., %g.}" % (bps[0], bps[1], bps[1]))

    inst = re.search(r"HYUNDAI_STEERING_LIMITS_DYNAMIC\s*=\s*HYUNDAI_LIMITS_DYNAMIC\(([^)]*)\)",
                     safety)
    check("panda instantiates the dynamic limits", inst is not None)
    if inst is not None:
        args = [a.strip() for a in inst.group(1).split(",")]
        check("panda torques match opendbc's",
              args[:2] == [str(low), str(high)], "found " + str(args[:2]))
        check("steer ramp rates are unchanged at %d/%d" % (rate_up, rate_down),
              args[2:4] == [str(rate_up), str(rate_down)], "found " + str(args[2:4]))

    check("panda selects the dynamic limits on the flag",
          re.search(r"hyundai_dynamic_limits\s*\?\s*HYUNDAI_STEERING_LIMITS_DYNAMIC", safety)
          is not None)
    # The ternary mentioning the flag is not enough -- nothing has to ever set it.
    check("hyundai_dynamic_limits is actually assigned from the safety param",
          re.search(r"hyundai_dynamic_limits\s*=\s*GET_FLAG\(param,\s*HYUNDAI_PARAM_DYNAMIC_LIMITS\)",
                    common_h) is not None,
          "the flag would be false forever and the schedule would never apply")
    check("panda has a vehicle speed to interpolate on",
          "UPDATE_VEHICLE_SPEED(" in safety,
          "without a speed sample the lookup is evaluated at 0 forever")

    # --- the two flag words must agree ------------------------------------------------
    sf = _enum_members(values, "HyundaiSafetyFlags")
    cf = _enum_members(values, "HyundaiFlags")
    check("HyundaiSafetyFlags parsed", len(sf) > 0)
    check("HyundaiFlags parsed", len(cf) > 0)
    param = re.search(r"HYUNDAI_PARAM_DYNAMIC_LIMITS\s*=\s*(\d+)", common_h)
    check("panda declares HYUNDAI_PARAM_DYNAMIC_LIMITS", param is not None)
    if param is not None and sf.get("DYNAMIC_LIMITS") is not None:
        check("safety flag values agree",
              int(param.group(1)) == sf["DYNAMIC_LIMITS"],
              "opendbc says " + str(sf["DYNAMIC_LIMITS"]) + ", panda says " + param.group(1))

    # IntFlag turns a duplicate value into a silent alias, so a platform carrying one member
    # tests True for the other. That is how the hybrid could have inherited this ceiling.
    for name, members in (("HyundaiFlags", cf), ("HyundaiSafetyFlags", sf)):
        vals = [v for v in members.values() if v is not None]
        check(name + " has no unevaluable members",
              len(vals) == len(members), "some member is not a plain integer expression")
        dupes = {v for v in vals if vals.count(v) > 1}
        check(name + " has no aliased members (" + str(len(vals)) + " checked)",
              not dupes, "duplicate value(s) " + str(sorted(dupes)))

    # --- and only the car it was measured on gets it -----------------------------------
    hev = _flags_in_assignment(values, "HYUNDAI_ELANTRA_HEV_2024") or set()
    check("the Elantra Hybrid does NOT get the schedule",
          "DYNAMIC_LIMITS" not in hev,
          "heavier car, no fleet data, and it borrows HYUNDAI_SONATA torque params")


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
    guard_panda_enforcement(opendbc)
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
