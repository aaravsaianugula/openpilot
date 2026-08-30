#!/usr/bin/env python3
"""How much of a frame is the eGPU actually executing, and how much is everything else?

The model runs at 208 ms/frame while SmuMetrics reports 20-30% GPU activity, which says most of
the frame is not compute. That is a claim worth settling with hardware timestamps rather than
with an SMU average sampled over an unknown window.

PROFILE=1 makes tinygrad record a ProfileRangeEvent per submission, with st/en taken from the
GPU's own clock counter (hcq.py: release_mem with send_gpu_clock_counter).

KNOWN LIMIT, and it is the reason this tool exists in this shape: a captured JIT graph emits no
compute ProfileRangeEvents. On the big model the only ranges that come back are AMD:SDMA:0, i.e.
the copy engine -- measured at 8.49 ms/frame. Reading that as "the GPU executes 8.49 ms/frame"
is wrong and was briefly believed here. For compute inside a graph the number to use is the
`batched N ... tm` line DEBUG>=2 prints, which is GPU-timed: 194 ms/frame for `batched 497`.
So this reports transfer cost honestly and says nothing about compute.

  ITERS=20 .elantra/gpu_utilization.py
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

os.environ["PROFILE"] = os.environ.get("PROFILE", "1")

PKL = os.environ.get("PKL", "/data/rdna2-tmp/big_driving_tinygrad_gfx1032.pkl")
ITERS = int(os.environ.get("ITERS", "20"))
WARMUP = int(os.environ.get("WARMUP", "3"))

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled
from tinygrad.helpers import Context, ProfileRangeEvent
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.common.file_chunker import open_file_chunked
from openpilot.sunnypilot.modeld_v2.compile_modeld import (
    WARP_INPUTS, POLICY_INPUTS, derive_frame_skip, make_supercombo_input_queues)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_bench import widen_affinity  # same little-cluster trap applies here


def main() -> int:
  widen_affinity()
  with Context(DEV=os.environ.get("DEV", "USB+AMD:LLVM")):
    key = Device.DEFAULT
    dev = Device[key]
    smu = dev.iface.dev_impl.smu

    jits = load_oob(open_file_chunked(PKL))
    md = jits["metadata"]
    md = md.get("model", md)
    shapes = md["input_shapes"]
    legacy = "run_policy" not in jits
    cam = (1928, 1208)
    run_policy = jits[cam]["run_policy"] if legacy else jits["run_policy"]
    warp = None if legacy else jits[cam]

    queues, npy = make_supercombo_input_queues(shapes, derive_frame_skip({}, shapes), key)
    warp_dev = os.environ.get("WARP_DEV", "QCOM")
    frames = {k: Tensor.randint((1928 * 1208 * 3 // 2,), low=0, high=255,
                                dtype=dtypes.uint8, device=warp_dev).realize()
              for k in ("frame", "big_frame")}
    for v in npy.values():
      v[:] = np.random.default_rng(0).standard_normal(v.shape).astype(v.dtype)
    dev.synchronize()

    def one():
      if legacy:
        return run_policy(**queues, frame=frames["frame"], big_frame=frames["big_frame"])
      w = warp(**{k: queues[k] for k in WARP_INPUTS},
               frame=frames["frame"], big_frame=frames["big_frame"])
      return run_policy(**{k: queues[k] for k in POLICY_INPUTS if k in queues}, warped=w)

    def force(out):
      for t in (out if isinstance(out, tuple) else (out,)):
        t.numpy()

    for _ in range(WARMUP):
      force(one())
      dev.synchronize()

    # Only the timed window counts, so warmup's compile and capture ranges are discarded.
    Compiled.profile_events.clear()
    st = time.perf_counter()
    for _ in range(ITERS):
      force(one())
      dev.synchronize()
    wall = time.perf_counter() - st

    # st/en are GPU clock microseconds. Ranges on one queue do not overlap, but ranges on
    # different devices do, so they are summed per device and never pooled.
    busy: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for e in Compiled.profile_events:
      if isinstance(e, ProfileRangeEvent) and e.en is not None:
        busy[e.device] += float(e.en - e.st) / 1e6
        counts[e.device] += 1

    print(f"\n  frames {ITERS}   wall {wall * 1e3:.1f} ms total   {wall / ITERS * 1e3:.1f} ms/frame")
    for d in sorted(busy):
      print(f"    {d:<12} device-busy {busy[d] * 1e3:8.1f} ms"
            + f"  ({busy[d] / wall * 100:5.1f}% of wall)   {counts[d]:>5} ranges"
            + f"   {busy[d] / ITERS * 1e3:7.2f} ms/frame")
    amd = sum(v for k, v in busy.items() if "AMD" in k)
    print(f"\n  eGPU executing      {amd / ITERS * 1e3:7.2f} ms/frame  ({amd / wall * 100:.1f}%)")
    print(f"  everything else     {(wall - amd) / ITERS * 1e3:7.2f} ms/frame  ({(wall - amd) / wall * 100:.1f}%)")
    try:
      m = smu.metrics()
      print(f"\n  SMU says: gfx {m['gfxclk']} MHz, uclk {m['uclk']} MHz, "
            + f"activity {m['gfx_activity']}%, {m['socket_power']} W")
    except Exception as e:
      print(f"\n  SMU unreadable: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
  sys.exit(main())
