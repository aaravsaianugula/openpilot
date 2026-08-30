#!/usr/bin/env python3
"""Tests for the BEAM cache's empty-schedule guard. No GPU needed.

What this protects. When every candidate in a BEAM round fails to time -- the device hiccups, or the
search caps exclude everything, which is what `infs from N -> 0 actions` in the log means -- the beam
is still the unoptimised Scheduler it was seeded with, and its applied_opts is []. Caching that is
strictly worse than caching nothing:

  * [] is a cache HIT on every later run, so the kernel is never re-searched;
  * apply_opts takes its `elif beam >= 1` branch and so never falls through to
    hand_coded_optimizations, which is an elif chained off it;
  * the kernel therefore ships with no LOCAL opt at all, which means local_size (1,1,1) -- one work
    item per workgroup, 1/32 of a wave32 on RDNA2.

That is not hypothetical. One 7x7 depthwise conv in the big driving model reached the car in exactly
that state and owned 38% of the frame at 12 GFLOPS, and a later wider-cap search reproduced it on two
more kernels in a single run.

Both directions are covered: nothing empty can be written, and nothing empty already on disk is
honoured on read.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCH = os.environ.get("ARCH", "gfx1032")

failures: list[str] = []
passes: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        passes.append(name)
    else:
        failures.append(f"{name}{': ' + detail if detail else ''}")


def a_real_scheduler(tinygrad_path: Path):
    """A Scheduler over a real reduce AST, retargeted at AMD, with the device never opened.

    Same route BEAM's own compile workers take: they render and compile under ALLOW_DEVICE_USAGE=0.
    """
    from dataclasses import replace

    from tinygrad import Tensor, dtypes
    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.helpers import Target
    from tinygrad.renderer.llvmir import AMDLLVMRenderer
    from tinygrad.uop.ops import Ops

    ren = AMDLLVMRenderer(Target(device="AMD", arch=ARCH))
    a = Tensor.empty(64, 256, dtype=dtypes.half, device="CPU")
    b = Tensor.empty(256, 256, dtype=dtypes.half, device="CPU")
    linear = a.matmul(b).schedule_linear()
    ast = next(u.src[0] for u in linear.toposort()
               if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK)
    ast = ast.substitute({p: p.replace(arg=replace(p.arg, device="AMD"))
                          for p in ast.toposort() if p.op is Ops.PARAM})
    s = Scheduler(ast, ren)
    s.convert_loop_to_global()
    return s


def test_an_empty_result_falls_back_to_the_heuristic(tinygrad_path: Path):
    from tinygrad.codegen.opt.search import _never_empty

    s = a_real_scheduler(tinygrad_path)
    check("the fresh scheduler really has no opts (the state we must never cache)",
          len(s.applied_opts) == 0, f"got {s.applied_opts}")

    out = _never_empty(s, s)
    check("an empty beam result is replaced", len(out.applied_opts) > 0,
          "hand_coded_optimizations produced nothing either")
    # The whole point is a usable workgroup, so assert the property rather than the mechanism.
    from tinygrad.codegen.opt import OptOps
    check("the replacement gives the kernel a LOCAL axis",
          any(o.op is OptOps.LOCAL for o in out.applied_opts),
          f"got {out.applied_opts}")


def test_a_real_result_is_returned_untouched(tinygrad_path: Path):
    """It must only rescue the empty case -- a searched schedule has to pass through unchanged."""
    from tinygrad.codegen.opt import Opt, OptOps
    from tinygrad.codegen.opt.search import _never_empty

    s = a_real_scheduler(tinygrad_path)
    opted = s.copy()
    opted.apply_opt(Opt(OptOps.LOCAL, 0, 8))
    out = _never_empty(s, opted)
    check("a non-empty beam result is returned as-is", out is opted,
          f"returned a different scheduler: {out.applied_opts}")


def test_the_read_side_treats_an_empty_row_as_a_miss(tinygrad_path: Path):
    """A cache written before the fix can still hold an empty row; reading it must not honour it."""
    src = (tinygrad_path / "tinygrad" / "codegen" / "opt" / "search.py").read_text(encoding="utf-8")
    line = next((ln for ln in src.splitlines() if 'diskcache_get("beam_search", key)' in ln), "")
    check("the cache read requires a non-empty value", "len(val)" in line, f"read line is: {line.strip()}")


def test_both_write_paths_are_guarded(tinygrad_path: Path):
    """beam_search caches in two places -- the normal exit and the device-died except arm."""
    src = (tinygrad_path / "tinygrad" / "codegen" / "opt" / "search.py").read_text(encoding="utf-8")
    puts = [ln for ln in src.splitlines() if 'diskcache_put("beam_search"' in ln]
    guards = [ln for ln in src.splitlines() if "_never_empty(" in ln and "def " not in ln]
    check("every diskcache_put is preceded by a guard", len(puts) == 2 and len(guards) == 2,
          f"{len(puts)} put sites, {len(guards)} guards")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()

    if not (args.tinygrad / "tinygrad" / "codegen" / "opt" / "search.py").is_file():
        print(f"no tinygrad at {args.tinygrad}; nothing to test")
        return 2

    sys.path.insert(0, str(args.tinygrad))
    os.environ.setdefault("ALLOW_DEVICE_USAGE", "0")
    print(f"tinygrad: {args.tinygrad}   arch: {ARCH}")

    test_an_empty_result_falls_back_to_the_heuristic(args.tinygrad)
    test_a_real_result_is_returned_untouched(args.tinygrad)
    test_the_read_side_treats_an_empty_row_as_a_miss(args.tinygrad)
    test_both_write_paths_are_guarded(args.tinygrad)

    print("\n" + "-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s) failed, {len(passes)} passed\n")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"PASSED: all {len(passes)} cases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
