#!/usr/bin/env python3
"""Run the driving model on fixed inputs and dump its output, so two builds can be compared exactly.

Every change that touches GPU command submission needs this before it needs a stopwatch. Eliding
the per-dispatch CS_PARTIAL_FLUSH removes the thing that was serialising the whole graph, and what
that was also doing -- silently -- was hiding any dependency DepsTracker fails to see. A missed
dependency does not announce itself with a timeout; it produces slightly wrong numbers, sometimes,
and a frame time that looks like a win. So: same inputs, both builds, compare the bits.

`model_bench.py` cannot be used for this. Its scalar inputs come from a seeded numpy generator, but
its camera frames come from `Tensor.randint`, which does not reproduce across processes -- so two
runs differ for reasons that have nothing to do with the change under test.

  usage: model_output.py                      # run and write the output
         model_output.py --against ref.npy    # run and compare against an earlier one
  env:   PKL, ITERS (default 3), SEED (default 0), OUT (default /root/runs/out.npy)
"""
import os
import sys
import time

import numpy as np

PKL = os.environ["PKL"]
ITERS = int(os.environ.get("ITERS", "3"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "/root/runs/out.npy")

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.common.file_chunker import open_file_chunked
from openpilot.sunnypilot.modeld_v2.compile_modeld import (
    WARP_INPUTS, POLICY_INPUTS, derive_frame_skip, make_supercombo_input_queues)


def main() -> int:
  ref_path = None
  if "--against" in sys.argv:
    ref_path = sys.argv[sys.argv.index("--against") + 1]

  with Context(DEV=os.environ.get("DEV", "USB+AMD:LLVM")):
    key = Device.DEFAULT
    dev = Device[key]
    jits = load_oob(open_file_chunked(PKL))
    md = jits["metadata"]
    md = md.get("model", md)
    shapes = md["input_shapes"]
    legacy = "run_policy" not in jits
    cam = (1928, 1208)
    run_policy = jits[cam]["run_policy"] if legacy else jits["run_policy"]
    warp = None if legacy else jits[cam]

    queues, npy = make_supercombo_input_queues(shapes, derive_frame_skip({}, shapes), key)
    warp_dev = os.environ.get("WARP_DEV", "CPU")
    rng = np.random.default_rng(SEED)

    # Frames go through numpy, not Tensor.randint: the point is that a second process produces the
    # identical bytes, and tinygrad's own RNG state does not carry across processes.
    nv12 = 1928 * 1208 * 3 // 2
    frames = {k: Tensor(rng.integers(0, 255, size=(nv12,), dtype=np.uint8), device=warp_dev).realize()
              for k in ("frame", "big_frame")}
    for name in sorted(npy):
      npy[name][:] = rng.standard_normal(npy[name].shape).astype(npy[name].dtype)
    dev.synchronize()

    def one():
      if legacy:
        return run_policy(**queues, frame=frames["frame"], big_frame=frames["big_frame"])
      w = warp(**{k: queues[k] for k in WARP_INPUTS},
               frame=frames["frame"], big_frame=frames["big_frame"])
      return run_policy(**{k: queues[k] for k in POLICY_INPUTS if k in queues}, warped=w)

    outs = []
    for i in range(ITERS):
      st = time.perf_counter()
      o = one()
      arrs = [t.numpy() for t in (o if isinstance(o, tuple) else (o,))]
      dev.synchronize()
      outs.append(np.concatenate([a.ravel() for a in arrs]))
      print(f"  iter {i}: {(time.perf_counter()-st)*1e3:7.1f} ms  "
            f"finite={np.isfinite(outs[-1]).all()}  sum={float(outs[-1].sum()):+.6e}", flush=True)

    got = outs[-1]
    # Replaying the same graph on the same inputs must give the same bits every time. If it does
    # not, the run is racy and comparing two builds is meaningless -- say so before anything else.
    stable = all(np.array_equal(o, outs[0]) for o in outs)
    print(f"\n  self-consistency across {ITERS} replays: "
          + ("bit-identical" if stable else "*** NOT STABLE -- this build is racy on its own ***"))
    if not np.isfinite(got).all():
      print(f"  *** {int((~np.isfinite(got)).sum())} of {got.size} outputs are not finite ***")
      return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, got)
    print(f"  wrote {OUT}  ({got.size} values)")

    if ref_path:
      ref = np.load(ref_path)
      if ref.shape != got.shape:
        print(f"  *** shape mismatch: {ref.shape} vs {got.shape} ***")
        return 1
      if np.array_equal(ref, got):
        print(f"  MATCH: bit-identical to {ref_path}")
        return 0
      d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
      rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
      print(f"  *** DIFFERS from {ref_path}: {int((d>0).sum())} of {d.size} values, "
            f"max abs {d.max():.3e}, max rel-to-rms {d.max()/rms:.3e} ***")
      return 1
    return 0 if stable else 1


if __name__ == "__main__":
  sys.exit(main())
