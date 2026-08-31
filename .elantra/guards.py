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

# LFAHDA_MFC 0x485 is 8 bytes on the CN7 bus and 4 on every other Hyundai CAN platform.
# panda allow-list matching is on EXACT length, so it must carry both entries and
# hyundai_tx_hook picks by flag. A guard that only checked the CN7 half would stay green
# while the other 79 platforms were widened along with it.
LFAHDA_MFC_LEN_CN7 = 8
LFAHDA_MFC_LEN_STOCK = 4

# The CN7 flat raised steering torque ceiling, stated here as the third opinion.
# opendbc decides what the car commands, panda decides what it will let out, and these
# literals are what both are checked against -- a guard that only compared the two against
# each other would stay green while they moved together.
STEER_MAX_RAISED = 409
STEER_MAX_STOCK = 384
STEER_RATES = (3, 7)

EXPECTED_FLAGS = {
    "HYUNDAI_ELANTRA_2024": {"CHECKSUM_CRC8", "CAMERA_SCC", "RAISED_LIMITS", "LFAHDA_MFC_8"},
    "HYUNDAI_ELANTRA_HEV_2024": {"CHECKSUM_CRC8", "CAMERA_SCC", "HYBRID", "LFAHDA_MFC_8"},
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
    """The safety-critical one. Per platform, dbc byte count and panda TX length must agree."""
    print("\n[dbc <-> safety] LFAHDA_MFC 0x485 length agreement")

    def dbc_len(name: str) -> int:
        src = read(opendbc / ("opendbc/dbc/generator/hyundai/" + name))
        m = re.search(r"^BO_\s+1157\s+LFAHDA_MFC:\s*(\d+)\s", src, re.MULTILINE)
        check("LFAHDA_MFC message found in " + name, m is not None)
        return int(m.group(1)) if m else -1

    cn7 = dbc_len("hyundai_can_cn7.dbc")
    check("hyundai_can_cn7.dbc declares LFAHDA_MFC as " + str(LFAHDA_MFC_LEN_CN7) + " bytes",
          cn7 == LFAHDA_MFC_LEN_CN7, "found " + str(cn7))

    stock = dbc_len("hyundai_can.dbc")
    check("the SHARED hyundai_can.dbc is still " + str(LFAHDA_MFC_LEN_STOCK) + " bytes",
          stock == LFAHDA_MFC_LEN_STOCK,
          "found " + str(stock) + " -- widening the shared dbc changes every Hyundai CAN " +
          "platform, not just the CN7")

    values = read(opendbc / "opendbc/car/hyundai/values.py")
    for platform in PLATFORMS:
        block = re.search(re.escape(platform) + r"\s*=\s*HyundaiPlatformConfig\((.*?)\n  \)",
                          values, re.S)
        check(platform + " platform block parsed", block is not None)
        if block is not None:
            check(platform + " uses hyundai_can_cn7_generated",
                  "hyundai_can_cn7_generated" in block.group(1),
                  "it would pack 4 bytes into a message this bus carries as 8")

    safety_src = read(opendbc / "opendbc/safety/modes/hyundai.h")
    lens = sorted(int(m) for m in re.findall(r"\{\s*0x485\s*,\s*0\s*,\s*(\d+)\s*,", safety_src))
    check("panda allow-list carries BOTH 0x485 lengths",
          lens == [LFAHDA_MFC_LEN_STOCK, LFAHDA_MFC_LEN_CN7],
          "found " + str(lens) + " -- tx_msg_safety_check matches exact length, so a single " +
          "entry silently blocks every platform that uses the other one")

    check("hyundai_tx_hook gates 0x485 length on the flag",
          re.search(r"hyundai_lfahda_mfc_8\s*\?\s*8U\s*:\s*4U", safety_src) is not None,
          "without this both lengths are accepted for every platform")

    common_h = read(opendbc / "opendbc/safety/modes/hyundai_common.h")
    check("the flag is assigned from the safety param",
          re.search(r"hyundai_lfahda_mfc_8\s*=\s*GET_FLAG\(param,\s*HYUNDAI_PARAM_LFAHDA_MFC_8\)",
                    common_h) is not None,
          "the flag would be false forever and the CN7 could not send 0x485 at all")

    iface = read(opendbc / "opendbc/car/hyundai/interface.py")
    check("interface.py carries the LFAHDA_MFC_8 car flag into safetyParam",
          re.search(r"HyundaiFlags\.LFAHDA_MFC_8[\s\S]{0,160}?" +
                    r"HyundaiSafetyFlags\.LFAHDA_MFC_8\.value", iface) is not None,
          "panda would enforce 4 bytes on a car whose dbc packs 8")


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


def _module_ints(source: str, names: tuple[str, ...]) -> dict:
    """{name: value} for top-level `NAME = <int expr>` assignments. Missing names are absent."""
    out: dict = {}
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in names:
            out[stmt.targets[0].id] = _int_expr(stmt.value)
    return out


def _platforms_with_flag(source: str, flag: str) -> set[str]:
    """Every platform whose flags= argument mentions HyundaiFlags.<flag>."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for kw in node.value.keywords:
            if kw.arg != "flags":
                continue
            for sub in ast.walk(kw.value):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                        and sub.value.id == "HyundaiFlags" and sub.attr == flag:
                    found.update(names)
    return found


def guard_ui_headroom(repo: Path, opendbc: Path) -> None:
    """The onroad arc's two ceilings, and the chain that has to hold for it to mean anything.

    This is display only -- it cannot change what the car commands -- so what is guarded here
    is not safety but honesty. An arc that says "past the stock ceiling" while reading a stale
    constant, the wrong flag, or a field nothing populates any more is worse than no arc at
    all, because it is the instrument this build's raised ceiling gets judged by.
    """
    print("\n[ui] steering headroom indicator")
    onroad = repo / "openpilot/selfdrive/ui/sunnypilot/mici/onroad"
    logic, widget = onroad / "steer_headroom.py", onroad / "steer_headroom_bar.py"
    hud = onroad / "hud_renderer.py"
    for f in (logic, widget):
        check("overlay file " + f.name + " present", f.is_file(),
              "a dropped overlay file is an ImportError on the driving screen")
    if not (logic.is_file() and widget.is_file() and hud.is_file()):
        return

    consts = _module_ints(read(logic), ("STOCK_COUNTS", "RAISED_COUNTS", "RAISED_LIMITS_FLAG"))
    check("the UI's stock ceiling is " + str(STEER_MAX_STOCK),
          consts.get("STOCK_COUNTS") == STEER_MAX_STOCK,
          "found " + str(consts.get("STOCK_COUNTS"))
          + " -- the arc would mark the old line in the wrong place")
    check("the UI's raised ceiling is " + str(STEER_MAX_RAISED),
          consts.get("RAISED_COUNTS") == STEER_MAX_RAISED,
          "found " + str(consts.get("RAISED_COUNTS"))
          + " -- the arc would be scaled against a ceiling nothing commands")

    values = read(opendbc / "opendbc/car/hyundai/values.py")
    car_flags = _enum_members(values, "HyundaiFlags")
    check("HyundaiFlags parsed",
          bool(car_flags) and all(v is not None for v in car_flags.values()),
          "the enum could not be read, so the flag check below would be vacuous")
    # opendbc defines RAISED_LIMITS twice: 2**27 on HyundaiFlags, which is what CarParams.flags
    # carries, and 1024 on HyundaiSafetyFlags, which goes into the safety param. 1024 in
    # CarParams.flags is a different flag on a different platform, so reading the wrong one
    # both fails to arm here and arms on a car this arc was never measured against.
    check("the UI tests HyundaiFlags.RAISED_LIMITS, not the safety-param bit",
          consts.get("RAISED_LIMITS_FLAG") is not None
          and consts.get("RAISED_LIMITS_FLAG") == car_flags.get("RAISED_LIMITS"),
          "UI has " + str(consts.get("RAISED_LIMITS_FLAG"))
          + ", HyundaiFlags.RAISED_LIMITS is " + str(car_flags.get("RAISED_LIMITS")))
    # If another platform ever takes the raised ceiling, 384 stops being the line that platform
    # came from, and this arc has to be re-thought rather than silently inherited.
    raised_on = _platforms_with_flag(values, "RAISED_LIMITS")
    check("exactly the CN7 carries the raised ceiling",
          raised_on == {"HYUNDAI_ELANTRA_2024"}, "found " + str(sorted(raised_on)))

    # The arc reads the signed integer actually put on CAN, not the value normalised by
    # STEER_MAX. If opendbc stops populating it the bar reads a constant zero and simply never
    # lights: no error, no alert, just an indicator that has quietly stopped being one.
    carctl = read(opendbc / "opendbc/car/hyundai/carcontroller.py")
    check("carcontroller still reports the raw CAN counts the arc reads",
          "new_actuators.torqueOutputCan = apply_torque" in carctl,
          "the arc would read zero for ever and never light")
    check("the arc still reads torqueOutputCan rather than the normalised value",
          "torqueOutputCan" in read(widget),
          "normalising by STEER_MAX draws 409 exactly where 384 used to be, "
          + "which is the whole bug this exists to fix")

    hud_src = read(hud)
    check("SteerHeadroomBar is installed in the SP mici HUD",
          "steer_headroom_bar import SteerHeadroomBar" in hud_src
          and re.search(r"self\._torque_bar\s*=\s*SteerHeadroomBar\(", hud_src) is not None,
          "the widget exists but the arc on screen is still upstream's")

    # It subclasses upstream rather than copying it, so upstream's shape is a real dependency.
    # A rename there is an ImportError on the driving screen, which is not a place to find one.
    upstream = repo / "openpilot/selfdrive/ui/mici/onroad/torque_bar.py"
    if not upstream.is_file():
        check("upstream torque_bar.py present", False,
              "SteerHeadroomBar has nothing to subclass. In a sparse checkout, "
              + "git sparse-checkout add openpilot/selfdrive/ui/mici/onroad")
        return
    up = read(upstream)
    for name in ("class TorqueBar", "def arc_bar_pts", "TORQUE_ANGLE_SPAN"):
        check("upstream still provides " + name, name in up,
              "SteerHeadroomBar imports it by name")
    check("arc_bar_pts still takes cap_radius",
          re.search(r"def arc_bar_pts\([^)]*cap_radius", up, re.S) is not None)
    check("TorqueBar still takes demo, scale and always",
          re.search(r"class TorqueBar.*?def __init__\(self, demo[^)]*scale[^)]*always", up, re.S) is not None)


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

    # The manifest was checked for the PRESENCE of opendbc_sha and nothing else, so it drifted
    # three commits behind the gitlink and stayed green. A manifest that records a different
    # commit from the one being shipped is not provenance, it is decoration -- and it is read
    # by verify_published.py and by the port panel on the car.
    manifest_path = repo / ".elantra/build-manifest.json"
    if manifest_path.is_file():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("opendbc_sha")
        except (OSError, ValueError):
            recorded = None
        check("manifest opendbc_sha matches the pinned gitlink", recorded == pinned,
              "manifest says " + str(recorded)[:12] + ", gitlink says " + pinned[:12])


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


def _raised_assigned_in_else(source: str) -> bool:
    """Is STEER_MAX raised inside the `else` of the chain that picks STEER_MAX?

    Precedence is the whole point. Assigned after the chain instead of inside its last branch,
    the raised ceiling silently outranks every other one: a car carrying RAISED_LIMITS together
    with ALT_LIMITS_2 would command 409 while panda enforced 170. Text matching cannot see the
    difference between the two placements, so this walks the tree.
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
            for inner in ast.walk(ast.Module(body=stmt.orelse, type_ignores=[])):
                if not isinstance(inner, ast.If):
                    continue
                if "RAISED_LIMITS" not in ast.dump(inner.test):
                    continue
                for asn in ast.walk(ast.Module(body=inner.body, type_ignores=[])):
                    if isinstance(asn, ast.Assign) and _int_expr(asn.value) == STEER_MAX_RAISED \
                            and any(getattr(t, "attr", None) == "STEER_MAX" for t in asn.targets):
                        return True
    return False


def guard_panda_enforcement(opendbc: Path) -> None:
    """What lateral.h actually DOES with the limits hyundai.h declares.

    Every other check in this file reads a DECLARATION -- the limit struct, the flag word.
    None of them reads the code that consumes it. lateral.h decides what the numbers mean.

    The CN7 used to ride the `dynamic_max_torque` branch, which shifts the speed down 1 m/s
    and adds a count of slack on top of the interpolated ceiling. It does not any more: a flat
    limit takes the static path, where the declared number IS the number. These checks exist to
    keep that true, because the slack moving out of that branch would hand every static Hyundai
    -- not just this one -- a count of headroom nobody asked for.
    """
    print("\n" + "[lateral.h] what panda actually enforces")
    lat = read(opendbc / "opendbc/safety/lateral.h")

    m = re.search(r"if\s*\(limits\.dynamic_max_torque\)\s*\{(.*?)\n\s*\}", lat, re.S)
    check("lateral.h still has a dynamic_max_torque branch (rivian uses it)", m is not None)
    if m is None:
        return

    # The static path: max_torque is taken straight from the struct, with no fudge before it.
    before = lat[:m.start()]
    static_init = re.search(r"int\s+max_torque\s*=\s*limits\.max_torque\s*;", before)
    check("the static path takes max_torque straight from the limits struct",
          static_init is not None)
    if static_init is not None:
        check("no slack is added before the dynamic branch",
              "+ 1" not in before[static_init.end():],
              "a +1 here would widen every static limit in every brand, silently")

    # Both fudges must stay INSIDE the branch the CN7 no longer takes.
    body = m.group(1)
    check("the -1 m/s speed shift is inside the dynamic branch",
          re.search(r"-\s*1\.", body) is not None)
    check("the +1 count of slack is inside the dynamic branch",
          re.search(r"safety_interpolate\([^)]*\)\s*\+\s*1", body) is not None)

    # And the global ceiling check has to compare against that unfudged max_torque, symmetric.
    check("the global torque check bounds both signs by max_torque",
          re.search(r"safety_max_limit_check\(\s*desired_torque\s*,\s*max_torque\s*," +
                    r"\s*-\s*max_torque\s*\)", lat) is not None,
          "an asymmetric or wider bound here is the whole ceiling, in one line")


def guard_raised_torque_pair(opendbc: Path) -> None:
    """The CN7 flat raised torque ceiling, end to end.

    The ceiling lives in four files that cannot see each other, and every link has a failure
    mode that looks exactly like success. Each check below exists because breaking that link
    alone leaves every other check green.
    """
    print("\n[values.py <-> safety] flat raised steering torque")
    values = read(opendbc / "opendbc/car/hyundai/values.py")
    carctl = read(opendbc / "opendbc/car/hyundai/carcontroller.py")
    iface = read(opendbc / "opendbc/car/hyundai/interface.py")
    safety = read(opendbc / "opendbc/safety/modes/hyundai.h")
    common_h = read(opendbc / "opendbc/safety/modes/hyundai_common.h")

    rate_up, rate_down = STEER_RATES

    # --- the opendbc half -------------------------------------------------------------
    check(f"opendbc raises STEER_MAX to {STEER_MAX_RAISED} under the flag, inside the else branch",
          _raised_assigned_in_else(values),
          "assigned outside it, RAISED_LIMITS outranks ALT_LIMITS/ALT_LIMITS_2/CANFD and the " +
          f"car commands {STEER_MAX_RAISED} where panda enforces 270 or 170")
    check(f"the stock ceiling under it is still {STEER_MAX_STOCK}",
          re.search(rf"self\.STEER_MAX\s*=\s*{STEER_MAX_STOCK}\b", values) is not None)

    # carcontroller must be upstream's shape. The flat ceiling needs no per-frame ceiling at
    # all, so anything left of the schedule here means a half-applied change.
    check("carcontroller multiplies by the static STEER_MAX",
          "int(round(actuators.torque * self.params.STEER_MAX))" in carctl)
    check("carcontroller normalises the feedback by the same STEER_MAX",
          "new_actuators.torque = apply_torque / self.params.STEER_MAX" in carctl)
    for dead in ("STEER_MAX_LOOKUP", "steer_max"):
        check(f"no {dead} left in carcontroller", dead not in carctl,
              "a flat ceiling needs none of the schedule machinery; leftovers mean half-applied")

    # --- the bridge -------------------------------------------------------------------
    check("interface.py carries the car flag into safetyParam",
          re.search(r"HyundaiFlags\.RAISED_LIMITS[\s\S]{0,140}?" +
                    r"HyundaiSafetyFlags\.RAISED_LIMITS\.value", iface) is not None,
          f"without this the car commands {STEER_MAX_RAISED} and panda rejects " +
          f"every frame above {STEER_MAX_STOCK}")

    # --- the panda half ---------------------------------------------------------------
    inst = re.search(r"HYUNDAI_STEERING_LIMITS_RAISED\s*=\s*HYUNDAI_LIMITS\(([^)]*)\)", safety)
    check("panda instantiates the raised limits from the plain HYUNDAI_LIMITS macro",
          inst is not None,
          "anything else is a different enforcement path with different fudges")
    if inst is not None:
        args = [a.strip() for a in inst.group(1).split(",")]
        check(f"panda's ceiling is {STEER_MAX_RAISED}",
              args[:1] == [str(STEER_MAX_RAISED)], "found " + str(args[:1]))
        check(f"steer ramp rates are unchanged at {rate_up}/{rate_down}",
              args[1:3] == [str(rate_up), str(rate_down)], "found " + str(args[1:3]))

    # The flat ceiling exists to be speed-independent. If anything puts the CN7 back on the
    # dynamic path, panda silently regains the -1 m/s shift and the +1 count of slack, and
    # every "speed cannot move the ceiling" claim on this branch becomes false.
    for dead in ("dynamic_max_torque", "HYUNDAI_LIMITS_DYNAMIC", "max_torque_lookup"):
        check(f"no {dead} anywhere on the hyundai path",
              dead not in safety and dead not in common_h)

    check("panda selects the raised limits on the flag",
          re.search(r"hyundai_raised_limits\s*\?\s*HYUNDAI_STEERING_LIMITS_RAISED", safety)
          is not None)
    # The ternary mentioning the flag is not enough -- nothing has to ever set it.
    check("hyundai_raised_limits is actually assigned from the safety param",
          re.search(r"hyundai_raised_limits\s*=\s*GET_FLAG\(param,\s*HYUNDAI_PARAM_RAISED_LIMITS\)",
                    common_h) is not None,
          "the flag would be false forever and the ceiling would never be raised")

    # Order inside the ternary is precedence. ALT_LIMITS_2 (170) and ALT_LIMITS (270) are
    # LOWER ceilings; if raised were tested first, a car carrying both flags would be handed
    # 409 by panda. This is the panda-side twin of the else-branch check above.
    tern = re.search(r"const TorqueSteeringLimits limits =(.*?);", safety, re.S)
    check("panda's limit ternary parsed", tern is not None)
    if tern is not None:
        order = list(re.findall(r"hyundai_(alt_limits_2|alt_limits|raised_limits)",
                                       tern.group(1)))
        check("the raised ceiling is tested last, after both ALT_LIMITS",
              order and order[-1] == "raised_limits",
              "found order " + str(order) + "; a lower ceiling must always win")

    # --- the two flag words must agree ------------------------------------------------
    sf = _enum_members(values, "HyundaiSafetyFlags")
    cf = _enum_members(values, "HyundaiFlags")
    check("HyundaiSafetyFlags parsed", len(sf) > 0)
    check("HyundaiFlags parsed", len(cf) > 0)
    param = re.search(r"HYUNDAI_PARAM_RAISED_LIMITS\s*=\s*(\d+)", common_h)
    check("panda declares HYUNDAI_PARAM_RAISED_LIMITS", param is not None)
    if param is not None and sf.get("RAISED_LIMITS") is not None:
        check("safety flag values agree",
              int(param.group(1)) == sf["RAISED_LIMITS"],
              "opendbc says " + str(sf["RAISED_LIMITS"]) + ", panda says " + param.group(1))

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
    check("the Elantra Hybrid does NOT get the raised ceiling",
          "RAISED_LIMITS" not in hev,
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
    guard_raised_torque_pair(opendbc)
    guard_panda_enforcement(opendbc)
    guard_car_list(opendbc)
    guard_torque(opendbc)
    if args.repo:
        guard_superproject(args.repo.resolve())
        guard_ui_headroom(args.repo.resolve(), opendbc)
        guard_opendbc_pin(args.repo.resolve(), opendbc)

    print("\n" + "-" * 60)
    if _failures:
        print("FAILED: " + str(len(_failures)) + " guard(s) failed, "
              + str(len(_passes)) + " passed\n")
        for f in _failures:
            print("  - " + f)
        print("\nNothing will be published. Elantra support is not intact in this tree.")
        return 1
    if args.repo:
        print("PASSED: all " + str(len(_passes)) + " guards green -- Elantra support is intact.")
    else:
        # Without --repo, guard_superproject and guard_opendbc_pin never ran. Those are the
        # two that answer "is the tree I just checked the tree that actually ships?", so a
        # bare pass here is not the same claim and must not be printed as if it were.
        print("PASSED: " + str(len(_passes)) + " opendbc guards green.")
        print("SKIPPED: the superproject and opendbc-pin guards -- pass --repo to run them.")
        print("         Without them this says nothing about which opendbc the car ships.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
