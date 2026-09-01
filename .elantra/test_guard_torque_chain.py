#!/usr/bin/env python3
"""
Negative tests for guard_raised_torque_pair -- the CN7 flat raised steering torque chain.

The ceiling lives in four files that cannot see each other: values.py decides what openpilot
commands, interface.py carries the flag into the safety param, hyundai.h decides what panda
lets out, and hyundai_common.h maps the param bit to the flag panda selects on. Break any
single link and the other three still agree, so a guard that only compared the two endpoints
stays green while the car commands torque panda will reject -- or, worse, while panda allows a
ceiling nothing on the car side ever asked for.

A fifth file, lateral.h, decides what those declarations MEAN. The flat ceiling exists to ride
its static path, where the declared number is the number; the dynamic path shifts the speed
down 1 m/s and adds a count of slack. ENFORCEMENT_CASES below break that path instead.

Each case copies the real checkout, applies exactly one mutation, and asserts the guard FAILS.
A guard suite that has never been shown to fail is decoration.

    python .elantra/test_guard_torque_chain.py --opendbc <path-to-opendbc-checkout>
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

GUARDS = Path(__file__).resolve().parent / "guards.py"

FILES = (
    "opendbc/car/hyundai/values.py",
    "opendbc/car/hyundai/carcontroller.py",
    "opendbc/car/hyundai/interface.py",
    "opendbc/safety/modes/hyundai.h",
    "opendbc/safety/modes/hyundai_common.h",
    "opendbc/safety/lateral.h",
)


# (label, find, replace, substring the RIGHT failure must contain)
ENFORCEMENT_CASES = [
    ("slack added on the static path, widening every brand's limit",
     "    int max_torque = limits.max_torque;\n",
     "    int max_torque = limits.max_torque;\n    max_torque = max_torque + 1;\n",
     "no slack is added before the dynamic branch"),

    ("the static read of max_torque is replaced by a fudged one",
     "int max_torque = limits.max_torque;",
     "int max_torque = limits.max_torque + 1;",
     "straight from the limits struct"),

    ("the -1 m/s shift escapes the dynamic branch",
     ("(vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;\n" +
     "      max_torque = safety_interpolate"),
     ("(vehicle_speed.min / VEHICLE_SPEED_FACTOR);\n" +
     "      max_torque = safety_interpolate"),
     "-1 m/s speed shift is inside"),

    ("the +1 count of slack disappears from the dynamic branch",
     "safety_interpolate(limits.max_torque_lookup, fudged_speed) + 1;",
     "safety_interpolate(limits.max_torque_lookup, fudged_speed);",
     "+1 count of slack is inside"),

    ("the global ceiling check becomes asymmetric",
     "safety_max_limit_check(desired_torque, max_torque, -max_torque);",
     "safety_max_limit_check(desired_torque, max_torque, -max_torque - 1);",
     "bounds both signs by max_torque"),
]


# Each case: (label, relative file, find, replace). One link, one break.
CASES = [
    ("opendbc's static ceiling edited alone",
     "opendbc/car/hyundai/values.py",
     "        self.STEER_MAX = 409                                  # the high-speed end",
     "        self.STEER_MAX = 450                                  # the high-speed end"),

    ("the low-speed schedule is raised above what panda will pass",
     "opendbc/car/hyundai/values.py",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [520, 409]"),

    ("the schedule's high-speed end drifts off the static STEER_MAX (fail-safe broken)",
     "opendbc/car/hyundai/values.py",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 395]"),

    ("the schedule gains authority with speed",
     "opendbc/car/hyundai/values.py",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]",
     "self.STEER_MAX_LOOKUP = [8.94, 13.41], [409, 500]"),

    ("the schedule is dropped, leaving the flat fallback",
     "opendbc/car/hyundai/values.py",
     "        self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]     # 20 mph -> 30 mph\n",
     ""),

    ("the raise is moved outside the if/elif chain (precedence inverted)",
     "opendbc/car/hyundai/values.py",
     ("      if CP.flags & HyundaiFlags.RAISED_LIMITS:\n" +
     "        self.STEER_MAX = 409                                  # the high-speed end, and the fallback\n" +
     "        self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]     # 20 mph -> 30 mph\n"),
     ("    if CP.flags & HyundaiFlags.RAISED_LIMITS:\n" +
     "      self.STEER_MAX = 409\n" +
     "      self.STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]\n")),

    ("the stock ceiling under the raise is removed",
     "opendbc/car/hyundai/values.py",
     "      self.STEER_MAX = 384\n",
     "      self.STEER_MAX = 409\n"),

    ("panda ceiling lowered below the schedule's peak",
     "opendbc/safety/modes/hyundai.h",
     "HYUNDAI_LIMITS(512, 3, 7)",
     "HYUNDAI_LIMITS(409, 3, 7)"),

    ("panda ramp rates changed",
     "opendbc/safety/modes/hyundai.h",
     "HYUNDAI_LIMITS(512, 3, 7)",
     "HYUNDAI_LIMITS(512, 5, 9)"),

    ("panda put back on the speed-scheduled path (regains the fudges)",
     "opendbc/safety/modes/hyundai.h",
     "HYUNDAI_STEERING_LIMITS_RAISED = HYUNDAI_LIMITS(512, 3, 7);",
     ("HYUNDAI_STEERING_LIMITS_RAISED = { .max_torque = 512, .dynamic_max_torque = true, " +
     ".max_torque_lookup = { {8.94, 13.41, 13.41}, {500., 409., 409.} }, " +
     ".max_rate_up = 3, .max_rate_down = 7, .type = TorqueDriverLimited };")),

    ("panda tests the raised ceiling before the lower ALT limits",
     "opendbc/safety/modes/hyundai.h",
     ("    const TorqueSteeringLimits limits = hyundai_alt_limits_2 ? HYUNDAI_STEERING_LIMITS_ALT_2 :\n" +
     "                                        hyundai_alt_limits ? HYUNDAI_STEERING_LIMITS_ALT :\n" +
     "                                        hyundai_raised_limits ? HYUNDAI_STEERING_LIMITS_RAISED : HYUNDAI_STEERING_LIMITS;"),
     ("    const TorqueSteeringLimits limits = hyundai_raised_limits ? HYUNDAI_STEERING_LIMITS_RAISED :\n" +
     "                                        hyundai_alt_limits_2 ? HYUNDAI_STEERING_LIMITS_ALT_2 :\n" +
     "                                        hyundai_alt_limits ? HYUNDAI_STEERING_LIMITS_ALT : HYUNDAI_STEERING_LIMITS;")),

    ("panda never assigns the flag it selects on",
     "opendbc/safety/modes/hyundai_common.h",
     "  hyundai_raised_limits = GET_FLAG(param, HYUNDAI_PARAM_RAISED_LIMITS);\n",
     ""),

    ("safety param bit renumbered on the panda side only",
     "opendbc/safety/modes/hyundai_common.h",
     "HYUNDAI_PARAM_RAISED_LIMITS = 1024;",
     "HYUNDAI_PARAM_RAISED_LIMITS = 2048;"),

    ("interface.py stops bridging the flag into safetyParam",
     "opendbc/car/hyundai/interface.py",
     ("      if ret.flags & HyundaiFlags.RAISED_LIMITS:\n" +
     "        ret.safetyConfigs[0].safetyParam |= HyundaiSafetyFlags.RAISED_LIMITS.value\n"),
     ""),

    ("carcontroller drops the per-frame ceiling and goes back to the static one",
     "opendbc/car/hyundai/carcontroller.py",
     "    steer_max = self.params.steer_max_at(CS.out.vEgoRaw)",
     "    steer_max = self.params.STEER_MAX"),

    ("carcontroller stops handing the ceiling to the rate limiter",
     "opendbc/car/hyundai/carcontroller.py",
     "                                                    self.params, steer_max)",
     "                                                    self.params)"),

    ("carcontroller normalises the feedback by something else",
     "opendbc/car/hyundai/carcontroller.py",
     "new_actuators.torque = apply_torque / steer_max",
     "new_actuators.torque = apply_torque / self.params.STEER_MAX"),

    ("a flag is aliased onto an existing value",
     "opendbc/car/hyundai/values.py",
     "  RAISED_LIMITS = 2 ** 27",
     "  RAISED_LIMITS = 2 ** 26"),

    ("the Elantra Hybrid inherits the raised ceiling",
     "opendbc/car/hyundai/values.py",
     ("    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.HYBRID |\n" +
     "          HyundaiFlags.LFAHDA_MFC_8,"),
     ("    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.HYBRID |\n" +
     "          HyundaiFlags.LFAHDA_MFC_8 | HyundaiFlags.RAISED_LIMITS,")),
]


def load_guards():
    spec = importlib.util.spec_from_file_location("elantra_guards", GUARDS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage(src: Path, dst: Path) -> None:
    for rel in FILES:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out)


def mutate(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"setup error: {rel} does not contain exactly one {old!r} " +
                         f"(found {text.count(old)}) -- the mutation would not be meaningful")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def run(g, root: Path, which: str = "guard_raised_torque_pair") -> list[str]:
    g._failures.clear()
    g._passes.clear()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        getattr(g, which)(root)
    return list(g._failures)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opendbc", required=True, type=Path)
    args = ap.parse_args()
    src = args.opendbc.resolve()
    for rel in FILES:
        if not (src / rel).is_file():
            raise SystemExit("not an opendbc checkout with the CN7 ceiling: " + str(src))

    g = load_guards()
    failures: list[str] = []
    print("guard_raised_torque_pair")

    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "clean"
        stage(src, clean)
        base = run(g, clean)
        if base:
            print("  FAIL  the unmutated checkout is already failing")
            for f in base:
                print("          " + f)
            return 1
        print("  ok    unmutated checkout passes")

        for i, (label, rel, old, new) in enumerate(CASES):
            root = Path(td) / f"case{i:02d}"
            stage(src, root)
            mutate(root, rel, old, new)
            if run(g, root):
                print("  ok    caught: " + label)
            else:
                print("  FAIL  MISSED: " + label)
                failures.append(label)

        # guard_panda_enforcement: what lateral.h actually DOES with those declarations.
        print("\nguard_panda_enforcement")
        base = run(g, clean, "guard_panda_enforcement")
        if base:
            print("  FAIL  the unmutated checkout is already failing")
            for f in base:
                print("          " + f)
            return 1
        print("  ok    unmutated checkout passes")

        for i, (label, old, new, expect) in enumerate(ENFORCEMENT_CASES):
            root = Path(td) / ("enf" + str(i).zfill(2))
            stage(src, root)
            mutate(root, "opendbc/safety/lateral.h", old, new)
            got = run(g, root, "guard_panda_enforcement")
            # Not just "something went red" -- the RIGHT check has to be the one that did.
            # A mutation that trips an unrelated guard is not evidence this guard works.
            if any(expect in f for f in got):
                print("  ok    caught: " + label)
            elif got:
                print("  FAIL  WRONG CHECK fired for: " + label)
                print("          expected a failure mentioning " + repr(expect))
                print("          got: " + "; ".join(got))
                failures.append(label + " (wrong check)")
            else:
                print("  FAIL  MISSED: " + label)
                failures.append(label)

    print("\n" + "-" * 58)
    if failures:
        print(f"FAILED: {len(failures)} mutation(s) slipped past the guards")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"PASSED: every single-link break is caught ({len(CASES)} chain + " +
          f"{len(ENFORCEMENT_CASES)} enforcement cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
