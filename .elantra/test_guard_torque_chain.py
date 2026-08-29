#!/usr/bin/env python3
"""
Negative tests for guard_dynamic_torque_pair -- the CN7 speed-scheduled steering torque chain.

The schedule lives in five files that cannot see each other: values.py decides what openpilot
commands, interface.py carries the flag into the safety param, carcontroller.py applies it,
hyundai.h decides what panda lets out, and hyundai_common.h maps the param bit to the flag
panda selects on. Break any single link and the other four still agree, so a guard that only
compared the two endpoints stays green while the car commands torque panda will reject -- or,
worse, commands the raised torque on the highway and the stock ceiling in a turn.

Each case below copies the real checkout, applies exactly one mutation, and asserts the guard
FAILS. A guard suite that has never been shown to fail is decoration.

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


ENFORCEMENT_CASES = [
    ("panda interpolates on the MAX speed instead of the min",
     "(vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;\n"
     "      max_torque = safety_interpolate",
     "(vehicle_speed.max / VEHICLE_SPEED_FACTOR) - 1.;\n"
     "      max_torque = safety_interpolate",
     "vehicle_speed.min"),

    ("panda stops shifting the speed down by 1 m/s",
     "(vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;\n"
     "      max_torque = safety_interpolate",
     "(vehicle_speed.min / VEHICLE_SPEED_FACTOR);\n"
     "      max_torque = safety_interpolate",
     "shifts the speed down"),

    ("panda stops adding the one-count slack",
     "safety_interpolate(limits.max_torque_lookup, fudged_speed) + 1;",
     "safety_interpolate(limits.max_torque_lookup, fudged_speed);",
     "adds one count"),

    ("panda drops the clamp to +/- max_torque",
     "max_torque = SAFETY_CLAMP(max_torque, -limits.max_torque, limits.max_torque);",
     "max_torque = max_torque + 0;",
     "clamps the interpolated ceiling"),
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
        raise SystemExit(f"setup error: {rel} does not contain exactly one {old!r} "
                         f"(found {text.count(old)}) -- the mutation would not be meaningful")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Each case: (label, relative file, find, replace). One link, one break.
CASES = [
    ("opendbc schedule edited alone",
     "opendbc/car/hyundai/values.py",
     "STEER_MAX_LOOKUP_DYNAMIC = [[8., 16.], [409, 384]]",
     "STEER_MAX_LOOKUP_DYNAMIC = [[8., 16.], [450, 384]]"),

    ("schedule assigned outside the if/elif chain (precedence inverted)",
     "opendbc/car/hyundai/values.py",
     "      if CP.flags & HyundaiFlags.DYNAMIC_LIMITS:\n"
     "        self.STEER_MAX_LOOKUP = self.STEER_MAX_LOOKUP_DYNAMIC\n",
     "    if CP.flags & HyundaiFlags.DYNAMIC_LIMITS:\n"
     "      self.STEER_MAX_LOOKUP = self.STEER_MAX_LOOKUP_DYNAMIC\n"),

    ("panda max_torque set to the high-speed value (schedule becomes a no-op)",
     "opendbc/safety/modes/hyundai.h",
     "  .max_torque = (steer_low), \\\n  .dynamic_max_torque = true,",
     "  .max_torque = (steer_high), \\\n  .dynamic_max_torque = true,"),

    ("panda torque row inverted (rejects low, accepts highway)",
     "opendbc/safety/modes/hyundai.h",
     "{(steer_low), (steer_high), (steer_high)},",
     "{(steer_high), (steer_low), (steer_low)},"),

    ("panda instantiated with the torques swapped",
     "opendbc/safety/modes/hyundai.h",
     "HYUNDAI_LIMITS_DYNAMIC(409, 384, 3, 7)",
     "HYUNDAI_LIMITS_DYNAMIC(384, 409, 3, 7)"),

    ("panda ramp rates changed",
     "opendbc/safety/modes/hyundai.h",
     "HYUNDAI_LIMITS_DYNAMIC(409, 384, 3, 7)",
     "HYUNDAI_LIMITS_DYNAMIC(409, 384, 5, 9)"),

    ("panda breakpoints diverge from opendbc's",
     "opendbc/safety/modes/hyundai.h",
     "{8., 16., 16.},",
     "{4., 12., 12.},"),

    ("panda never assigns the flag it selects on",
     "opendbc/safety/modes/hyundai_common.h",
     "  hyundai_dynamic_limits = GET_FLAG(param, HYUNDAI_PARAM_DYNAMIC_LIMITS);\n",
     ""),

    ("safety param bit renumbered on the panda side only",
     "opendbc/safety/modes/hyundai_common.h",
     "HYUNDAI_PARAM_DYNAMIC_LIMITS = 1024;",
     "HYUNDAI_PARAM_DYNAMIC_LIMITS = 2048;"),

    ("interface.py stops bridging the flag into safetyParam",
     "opendbc/car/hyundai/interface.py",
     "      if ret.flags & HyundaiFlags.DYNAMIC_LIMITS:\n"
     "        ret.safetyConfigs[0].safetyParam |= HyundaiSafetyFlags.DYNAMIC_LIMITS.value\n",
     ""),

    ("carcontroller normalises the feedback by the static ceiling",
     "opendbc/car/hyundai/carcontroller.py",
     "new_actuators.torque = apply_torque / steer_max",
     "new_actuators.torque = apply_torque / self.params.STEER_MAX"),

    ("carcontroller stops handing the scheduled ceiling to the rate limiter",
     "opendbc/car/hyundai/carcontroller.py",
     "self.params, steer_max)",
     "self.params)"),

    ("a flag is aliased onto an existing value",
     "opendbc/car/hyundai/values.py",
     "  DYNAMIC_LIMITS = 2 ** 27",
     "  DYNAMIC_LIMITS = 2 ** 26"),

    ("the Elantra Hybrid inherits the schedule",
     "opendbc/car/hyundai/values.py",
     "    CarSpecs(mass=3017 * CV.LB_TO_KG, wheelbase=2.72, steerRatio=12.9, tireStiffnessFactor=0.65),\n"
     "    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.HYBRID,",
     "    CarSpecs(mass=3017 * CV.LB_TO_KG, wheelbase=2.72, steerRatio=12.9, tireStiffnessFactor=0.65),\n"
     "    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.HYBRID | HyundaiFlags.DYNAMIC_LIMITS,"),
]


def run(g, root: Path, which: str = "guard_dynamic_torque_pair") -> list[str]:
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
            raise SystemExit("not an opendbc checkout with the CN7 schedule: " + str(src))

    g = load_guards()
    failures: list[str] = []
    print("guard_dynamic_torque_pair")

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
            root = Path(td) / ("case%02d" % i)
            stage(src, root)
            mutate(root, rel, old, new)
            got = run(g, root)
            if got:
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
            root = Path(td) / ("enf%02d" % i)
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
        print("FAILED: %d mutation(s) slipped past the guards" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: every single-link break is caught (%d chain + %d enforcement cases)"
          % (len(CASES), len(ENFORCEMENT_CASES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
