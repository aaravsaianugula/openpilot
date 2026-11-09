#!/usr/bin/env python3
"""
Negative tests for guard_ff_lat_accel_schedule -- the guard over the CN7 low-speed feedforward
gain schedule.

Why this test exists. The guard checks three things that each fail SILENTLY in production and
none of which shows up in a diff of the schedule itself:

  * the constants drifting off torqued.py's MIN_VEL, which is the schedule's entire justification;
  * the single line that reads them being dropped, duplicated, or moved onto the error path, which
    would change the closed-loop P gain -- the one thing this change explicitly does not do;
  * either half falling out of the overlay lists, after which the weekly rebuild keeps the numbers
    and deletes the only line that uses them. That last one reads exactly like the fix is still
    installed.

A guard that only ever passes is not a guard. Each case below copies the real tree, applies
exactly one mutation, and requires the guard to go RED for it.

    python .elantra/test_guard_ff_schedule.py --repo .
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

GUARDS = Path(__file__).resolve().parent / "guards.py"

SCHED = "openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py"
JERK = "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_jerk_aware.py"
V0 = "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py"
TEST = "openpilot/sunnypilot/selfdrive/controls/lib/tests/test_lat_accel_factor_schedule.py"
TORQUED = "openpilot/selfdrive/locationd/torqued.py"
SYNC = ".elantra/sync.py"

# Only these are copied into the throwaway tree. A full copy of the superproject is gigabytes and
# the guard reads exactly six files.
NEEDED = (SCHED, JERK, V0, TEST, TORQUED, SYNC)


def load_guards():
  spec = importlib.util.spec_from_file_location("elantra_guards", GUARDS)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def build_tree(src: Path, dst: Path) -> None:
  for rel in NEEDED:
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / rel, target)


def edit(tree: Path, rel: str, old: str, new: str) -> None:
  path = tree / rel
  text = path.read_text(encoding="utf-8")
  if old not in text:
    raise SystemExit(f"mutation anchor not found in {rel}: {old!r}")
  path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def run_case(g, label: str, tree: Path, want_fail: bool, failures: list[str]) -> None:
  g._failures.clear()
  g._passes.clear()
  g.guard_ff_lat_accel_schedule(tree)
  failed = bool(g._failures)
  if failed != want_fail:
    want = "FAIL" if want_fail else "PASS"
    got = "FAIL" if failed else "PASS"
    failures.append(f"{label}: expected {want}, got {got}")
    print("  FAIL  " + label)
  else:
    print("  ok    " + label)


# Each mutation is (label, relative path, find, replace). One edit each, on purpose: a case that
# breaks two things at once cannot tell you which check caught it.
MUTATIONS = (
  ("last gain value 0.99 instead of 1.0", SCHED, "0.86, 1.00]", "0.86, 0.99]"),
  ("a gain above 1.0", SCHED, "0.86, 1.00]", "0.86, 1.10]"),
  ("a gain below the 0.38 floor", SCHED, "[0.38,", "[0.20,"),
  ("gain schedule not monotone", SCHED, "0.53, 0.64,", "0.64, 0.53,"),
  ("breakpoints not increasing", SCHED, "[3.0, 4.5,", "[4.5, 3.0,"),
  ("schedule floor moved off MIN_VEL", SCHED, "LEARNER_MIN_VEL = 15.0", "LEARNER_MIN_VEL = 12.0"),
  ("torqued MIN_VEL moved out from under it", TORQUED, "\nMIN_VEL = 15", "\nMIN_VEL = 20"),
  ("blend outside [0, 1]", SCHED, "LOW_SPEED_FF_BLEND = 1.0", "LOW_SPEED_FF_BLEND = 2.5"),
  ("the division removed", JERK, "    ) / ff_gain", "    )"),
  ("the schedule applied twice", JERK,
   "    ff_gain = lat_accel_factor_gain(CS.vEgo) if self._low_speed_ff_gain else 1.0",
   "    ff_gain = lat_accel_factor_gain(CS.vEgo) * lat_accel_factor_gain(CS.vEgo)"),
  ("the schedule moved onto the error path", JERK,
   "LatControlInputs(self._setpoint, roll_compensation, CS.vEgo, CS.aEgo)",
   "LatControlInputs(self._setpoint / lat_accel_factor_gain(CS.vEgo), roll_compensation, CS.vEgo, CS.aEgo)"),
  ("the fingerprint gate removed", JERK, "ff_gain_applies(CP.carFingerprint)", "True"),
  # The gate computed but not consumed: __init__ is untouched, both symbols are still present
  # exactly once, and the CN7-only correction goes out to every car on the fork.
  ("the fingerprint gate computed but not applied", JERK,
   "lat_accel_factor_gain(CS.vEgo) if self._low_speed_ff_gain else 1.0",
   "lat_accel_factor_gain(CS.vEgo)"),
  ("the gate widened to an unmeasured platform", SCHED,
   'MEASURED_PLATFORMS = ("HYUNDAI_ELANTRA_2024",)',
   'MEASURED_PLATFORMS = ("HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024")'),
  ("the error line rewritten", JERK,
   "self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)",
   "self._pid_log.error = float(torque_from_setpoint) - float(torque_from_measurement)"),
  ("a half-edit landed in the v0 path", V0, "import math\n",
   "import math\nfrom openpilot.sunnypilot.selfdrive.controls.lib.lat_accel_factor_schedule import lat_accel_factor_gain\n"),
  ("call site dropped from OVERLAY_MODIFIED", SYNC,
   '    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_jerk_aware.py",\n', ""),
  ("schedule module dropped from OVERLAY_ADDED", SYNC,
   '    "openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py",\n', ""),
  ("schedule test dropped from OVERLAY_ADDED", SYNC,
   '    "openpilot/sunnypilot/selfdrive/controls/lib/tests/test_lat_accel_factor_schedule.py",\n', ""),
  # --- the KP cap, the coupled half ---
  ("KP cap raised above the stock gain", SCHED, "0.70, 1.00]", "0.70, 1.30]"),
  ("KP cap deeper than the measurement supports", SCHED, "[0.26,", "[0.05,"),
  ("KP cap not monotone", SCHED, "0.40, 0.40, 0.47,", "0.47, 0.40, 0.40,"),
  ("KP cap breakpoints not increasing", SCHED, "[2.5, 3.5,", "[3.5, 2.5,"),
  ("KP blend outside [0, 1]", SCHED, "LOW_SPEED_KP_BLEND = 1.0", "LOW_SPEED_KP_BLEND = 1.5"),
  # The dangerous combination: cut the gain, keep no feedforward to replace it.
  ("KP cut applied with the feedforward switched off", SCHED,
   "LOW_SPEED_FF_BLEND = 1.0", "LOW_SPEED_FF_BLEND = 0.0"),
  ("the KP cap not gated on the fingerprint", SCHED,
   "if not ff_gain_applies(car_fingerprint):", "if False:"),
  ("the v0 PID still built from the stock table", V0,
   "scaled_kp_interp(INTERP_SPEEDS, KP_INTERP, CP.carFingerprint)", "KP_INTERP"),
  ("v0 dropped from OVERLAY_MODIFIED", SYNC,
   '    "openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py",\n', ""),
)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                  help="superproject checkout to copy the real files from")
  args = ap.parse_args()
  src = args.repo.resolve()

  for rel in NEEDED:
    if not (src / rel).is_file():
      print("FAILED: " + rel + " is missing from " + str(src))
      return 1

  g = load_guards()
  failures: list[str] = []
  print("guard_ff_lat_accel_schedule -- negative tests")

  with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    clean = root / "clean"
    build_tree(src, clean)
    run_case(g, "the real tree passes", clean, False, failures)

    for i, (label, rel, old, new) in enumerate(MUTATIONS):
      tree = root / f"case{i:02d}"
      build_tree(src, tree)
      edit(tree, rel, old, new)
      run_case(g, label, tree, True, failures)

    missing = root / "missing"
    build_tree(src, missing)
    (missing / SCHED).unlink()
    run_case(g, "the schedule module deleted", missing, True, failures)

    no_test = root / "no_test"
    build_tree(src, no_test)
    (no_test / TEST).unlink()
    run_case(g, "the schedule test deleted", no_test, True, failures)

  print("")
  if failures:
    print("FAILED: " + str(len(failures)) + " case(s)")
    for f in failures:
      print("  - " + f)
    return 1
  print("PASSED: the guard goes red for every mutation and green on the real tree.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
