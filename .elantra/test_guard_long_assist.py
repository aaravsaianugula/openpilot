"""
Negative tests for guard_long_assist. Each case copies the real files, applies one mutation, and
requires the guard to go RED: Long Assist defaulting on, its one planner hook dropped, duplicated or
moved after the min(), and a file falling out of the overlay.

    python .elantra/test_guard_long_assist.py --repo .
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

GUARDS = Path(__file__).resolve().parent / "guards.py"
KEYS = "openpilot/common/params_keys.h"
PLANNER = "openpilot/selfdrive/controls/lib/longitudinal_planner.py"
SP_PLANNER = "openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py"
SYNC = ".elantra/sync.py"
HOOK = "    output_a_target_e2e, output_should_stop_e2e = self.long_assist.apply(output_a_target_e2e, output_should_stop_e2e)\n"


def load_guards():
  spec = importlib.util.spec_from_file_location("elantra_guards", GUARDS)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def edit(tree: Path, rel: str, old: str, new: str) -> None:
  path = tree / rel
  text = path.read_text(encoding="utf-8")
  if old not in text:
    raise SystemExit(f"mutation anchor not found in {rel}: {old!r}")
  path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


CASES = [
  ("unmodified tree passes", None, False),
  ("default on", (KEYS, '{"LongAssistMode", {PERSISTENT | BACKUP, INT, "0"}}', '{"LongAssistMode", {PERSISTENT | BACKUP, INT, "1"}}'), True),
  ("hook dropped", (PLANNER, HOOK, ""), True),
  ("hook duplicated", (PLANNER, HOOK, HOOK + HOOK), True),
  ("hook moved after the min()", (PLANNER, HOOK, "", "after_min"), True),
  ("SP planner stops publishing", (SP_PLANNER, "longitudinalPlanSP.longAssist", "longitudinalPlanSP.nothing"), True),
  ("map module unregistered", (SYNC, '    "openpilot/sunnypilot/selfdrive/controls/lib/long_assist_map.py",\n', ""), True),
]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
  repo = ap.parse_args().repo.resolve()
  g = load_guards()
  failures = []
  for label, mutation, want_fail in CASES:
    with tempfile.TemporaryDirectory() as tmp:
      tree = Path(tmp)
      for rel in (KEYS, PLANNER, SP_PLANNER, SYNC):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / rel, tree / rel)
      if mutation:
        edit(tree, *mutation[:3])
        if mutation[3:] == ("after_min",):
          text = (tree / PLANNER).read_text(encoding="utf-8")
          i = text.index("    output_a_target, self.mpc.source, _ = min(")
          (tree / PLANNER).write_text(text[:i] + HOOK + text[i:], encoding="utf-8", newline="\n")
      g._failures.clear()
      g._passes.clear()
      g.guard_long_assist(tree)
      if bool(g._failures) != want_fail:
        failures.append(label)
      print(("  ok    " if bool(g._failures) == want_fail else "  FAIL  ") + label)
  print("PASSED" if not failures else "FAILED: " + ", ".join(failures))
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
