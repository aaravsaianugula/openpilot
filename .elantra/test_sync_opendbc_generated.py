#!/usr/bin/env python3
"""Tests for how sync.py replays the opendbc delta around GENERATED files.

docs/CARS.md is not a file anyone edits. opendbc's own docs.py writes it from the car list, and
its first line is a count of every supported car. So any week upstream adds a car and our delta
adds ours, both sides rewrite the same lines, and a three-way apply cannot merge two regenerations
of one table -- that is exactly how the 2026-10-05 scheduled sync died ("U docs/CARS.md"). A hand
resolution committed to our opendbc master only moves the conflict to the next car upstream adds.

The fix replays everything EXCEPT the generated files, then regenerates them from the merged
tree with opendbc's own generator. These tests build real git repositories -- an upstream base, a
newer upstream, and our delta on the old base -- and prove:

  * the exact conflict now replays cleanly, and CARS.md is the generator's output for the merged
    tree (upstream's new cars AND ours), not either side's stale copy;
  * a generator that fails aborts the run by name -- nothing falls back to a stale table;
  * the stray-path check still fires for a real stray, and a CARS.md-only change (the fork's own
    scheduled docs job) is not one.

    python .elantra/test_sync_opendbc_generated.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sync

FAILS: list[str] = []

VALUES = "opendbc/car/hyundai/values.py"
SAFETY = "opendbc/safety/modes/hyundai.h"
CARS_MD = "docs/CARS.md"

# A stand-in for each real generator, with the same contract that matters here: run from the repo
# root with no arguments, it rewrites its file in place, and the first line counts every car --
# so two sides adding cars always collide on it.
FIXTURE_GENERATOR = '''\
import os

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), {up!r}))
with open(os.path.join(root, "opendbc", "car", "hyundai", "values.py"), encoding="utf-8") as f:
    cars = sorted(line.split()[1] for line in f if line.startswith("CAR "))
with open(os.path.join(root, {out!r}), "w") as f:  # text mode on purpose: CRLF on Windows
    f.write("# %d Supported Cars\\n\\n" % len(cars))
    for car in cars:
        f.write("| %s |\\n" % car)
'''


def generators(broken: str | None = None) -> dict[str, str]:
    """A fixture generator at every path sync.OPENDBC_GENERATED names, writing its own file."""
    files = {}
    for out, gen in sync.OPENDBC_GENERATED.items():
        up = "/".join([".."] * (gen.count("/")))
        files[gen] = broken if broken is not None else FIXTURE_GENERATOR.format(up=up, out=out)
    return files


def tables(values: str) -> dict[str, str]:
    return {out: generate(values) for out in sync.OPENDBC_GENERATED}


FILLER = "".join(f"# filler {i}\n" for i in range(12))
BASE_VALUES = "# car list\nCAR HYUNDAI_SONATA\n" + FILLER + "# end\n"
# Upstream adds two cars near the top; we add ours near the bottom. The source edits are far
# apart and merge cleanly -- only the generated table collides.
UPSTREAM_VALUES = "# car list\nCAR KIA_NEW\nCAR KIA_OTHER\nCAR HYUNDAI_SONATA\n" + FILLER + "# end\n"
OURS_VALUES = ("# car list\nCAR HYUNDAI_SONATA\n" + FILLER
               + "CAR HYUNDAI_ELANTRA_2024\nCAR HYUNDAI_ELANTRA_HEV_2024\n# end\n")
# Carries the rest of REQUIRED_IN_DELTA so the delta passes the token gate on its own merits.
OURS_SAFETY = ("#define HYUNDAI_PARAM_RAISED_LIMITS 1024\n// RAISED_LIMITS\n"
               "// LFAHDA_MFC_8 hyundai_can_cn7\n")


def check(label, cond, detail=""):
    if cond:
        print("  ok    " + label)
    else:
        print("  FAIL  " + label + (": " + detail if detail else ""))
        FAILS.append(label)


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def generate(values: str) -> str:
    cars = sorted(line.split()[1] for line in values.splitlines() if line.startswith("CAR "))
    return f"# {len(cars)} Supported Cars\n\n" + "".join(f"| {c} |\n" for c in cars)


def write(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(text.encode("utf-8"))


def commit(repo: Path, files: dict[str, str], msg: str) -> str:
    write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


def build(tmp: str, ours: dict[str, str], broken: str | None = None):
    """base -> upstream on master; base -> ours on a side branch. Returns (repo, upstream, ours)."""
    repo = Path(tmp) / "opendbc"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "core.autocrlf", "false")
    base = commit(repo, {VALUES: BASE_VALUES, SAFETY: "// base\n", **generators(broken),
                         **tables(BASE_VALUES)}, "base")
    upstream = commit(repo, {VALUES: UPSTREAM_VALUES, **tables(UPSTREAM_VALUES)},
                      "upstream adds three cars")
    git(repo, "checkout", "-q", "-b", "ours", base)
    mine = commit(repo, ours, "our delta")
    git(repo, "checkout", "-q", "master")
    return repo, upstream, mine


def replay(repo, upstream, mine):
    try:
        names, _ = sync.replay_opendbc_delta(repo, upstream, mine)
        return names, None
    except sync.SyncError as ex:
        return None, str(ex)


OURS = {VALUES: OURS_VALUES, SAFETY: OURS_SAFETY, **tables(OURS_VALUES)}


def test_the_cars_md_conflict_replays_cleanly():
    print("upstream and our delta both regenerate CARS.md (the 2026-10-05 failure)")
    with tempfile.TemporaryDirectory() as tmp:
        repo, upstream, mine = build(tmp, OURS)
        names, err = replay(repo, upstream, mine)
        check("the delta replays without a conflict", err is None, err or "")
        if err is not None:
            return
        check("no generated file is part of the replayed delta",
              not set(names) & set(sync.OPENDBC_GENERATED), f"replayed {names}")
        check("the source edits are", VALUES in names and SAFETY in names, f"replayed {names}")
        merged_values = (repo / VALUES).read_text(encoding="utf-8")
        check("the merged car list has upstream's cars and ours",
              all(c in merged_values for c in ("KIA_NEW", "KIA_OTHER", "HYUNDAI_ELANTRA_2024")))
        for out in sync.OPENDBC_GENERATED:
            raw = (repo / out).read_bytes()
            check(f"{out} is regenerated from the merged tree",
                  raw.decode("utf-8") == generate(merged_values), "got " + repr(raw[:80]))
            check(f"{out} is written with LF line endings on every platform",
                  b"\r\n" not in raw)


def test_generator_failure_aborts_by_name():
    print("a generator that fails")
    broken = "import sys\nsys.exit('docs generator exploded')\n"
    with tempfile.TemporaryDirectory() as tmp:
        repo, upstream, mine = build(tmp, OURS, broken=broken)
        _, err = replay(repo, upstream, mine)
        check("the run aborts rather than keeping a stale table", err is not None)
        check("and names the generated file", err is not None and CARS_MD in err, err or "")
        check("and carries the generator's own error", err is not None and "exploded" in err,
              err or "")


def test_generator_that_writes_nothing_aborts():
    print("a generator that exits 0 without writing its file")
    with tempfile.TemporaryDirectory() as tmp:
        repo, upstream, mine = build(tmp, OURS, broken="pass\n")
        _, err = replay(repo, upstream, mine)
        check("the run aborts rather than shipping upstream's stale copy", err is not None,
              "upstream's table, without our platforms, would have been committed")


def test_cars_md_only_change_is_not_a_stray():
    print("our master picks up a CARS.md-only commit (the fork's scheduled docs job)")
    with tempfile.TemporaryDirectory() as tmp:
        repo, upstream, mine = build(tmp, OURS)
        git(repo, "checkout", "-q", "ours")
        mine = commit(repo, {CARS_MD: generate(OURS_VALUES) + "| docs job touch |\n"}, "docs job")
        git(repo, "checkout", "-q", "master")
        _, err = replay(repo, upstream, mine)
        check("a generated-file change is not reported as a stray", err is None, err or "")


def test_real_stray_still_caught():
    print("our master carries a path outside the port")
    with tempfile.TemporaryDirectory() as tmp:
        repo, upstream, mine = build(tmp, dict(OURS, **{"README.md": "unrelated\n"}))
        _, err = replay(repo, upstream, mine)
        check("the stray check still goes red", err is not None)
        check("and names the stray", err is not None and "README.md" in err, err or "")
        check("without naming the generated doc as one",
              err is not None and CARS_MD not in err.split("Disable")[0], err or "")


def main():
    for fn in (test_the_cars_md_conflict_replays_cleanly, test_generator_failure_aborts_by_name,
               test_generator_that_writes_nothing_aborts,
               test_cars_md_only_change_is_not_a_stray, test_real_stray_still_caught):
        fn()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
        return 1
    print("PASSED: generated opendbc docs are regenerated, never three-way merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
