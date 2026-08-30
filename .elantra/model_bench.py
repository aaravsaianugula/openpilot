#!/usr/bin/env python3
"""Time the compiled big driving model on the eGPU, and say what clocks it ran at.

This replays run_policy -- the single captured graph that dominates a frame -- out of the
artifact modeld actually loads, rather than re-JITing the ONNX. Budget is openpilot's own, from
selfdrive/test/test_onroad.py: ("modelV2", 0.06, 0.040).

Every timing line is printed next to the clocks it was taken at. On this card that is not a
nicety. The memory clock boots at 96 MHz of a possible 1000, and a frame time without a memory
clock beside it does not mean anything: 96 MHz is 24.6 GB/s, 675 MHz is 172.8 GB/s, and the
model's working set is 1.78 GB.

CLOCKS=target (default) pins the fastest configuration this card has been measured to survive.
CLOCKS=none measures it exactly as AM leaves it.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

PKL = os.environ.get("PKL", "/data/rdna2-tmp/big_driving_tinygrad_gfx1032.pkl")
ITERS = int(os.environ.get("ITERS", "30"))
WARMUP = int(os.environ.get("WARMUP", "5"))
CLOCKS = os.environ.get("CLOCKS", "target")
BUDGET_MAX, BUDGET_MEAN = 0.06, 0.040

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.common.file_chunker import open_file_chunked
from openpilot.sunnypilot.modeld_v2.compile_modeld import (
    WARP_INPUTS, POLICY_INPUTS, derive_frame_skip, make_supercombo_input_queues)


def parse_cpu_list(text):
  out = set()
  for part in text.strip().split(","):
    if not part:
      continue
    lo, _, hi = part.partition("-")
    out.update(range(int(lo), int(hi or lo) + 1))
  return out


def widen_affinity():
  """An ssh session here inherits the little cluster only, and cores 4-7 are hotplugged offline
  while openpilot is stopped. Left alone every number below is a little-core number. The
  isolcpus=6,7 cores are excluded rather than taken, so this cannot make the car's own model
  miss frames while it runs."""
  if not hasattr(os, "sched_setaffinity"):
    return
  try:
    online = parse_cpu_list(Path("/sys/devices/system/cpu/online").read_text())
    isolated = parse_cpu_list(Path("/sys/devices/system/cpu/isolated").read_text())
  except OSError as e:
    print(f"  cpu topology unreadable ({e}), leaving affinity alone")
    return
  if usable := online - isolated:
    try:
      os.sched_setaffinity(0, usable)
    except OSError as e:
      print(f"  could not widen cpu affinity: {e}")
  print(f"  cpu affinity {sorted(os.sched_getaffinity(0))}"
        + f" (online {sorted(online)}, isolated for openpilot {sorted(isolated)})")


def summarize(ts):
  a = np.asarray(ts, dtype=np.float64)
  return {"n": a.size, "min": a.min(), "p50": np.percentile(a, 50, method="inverted_cdf"),
          "p99": np.percentile(a, 99, method="inverted_cdf"), "max": a.max(), "mean": a.mean()}


def main() -> int:
  widen_affinity()
  with Context(DEV=os.environ.get("DEV", "USB+AMD:LLVM")):
    key = Device.DEFAULT
    dev = Device[key]
    smu = dev.iface.dev_impl.smu
    mod = smu.smu_mod
    from tinygrad.runtime.support.am.ip import SMUError

    def clocks(tag):
      try:
        m = smu.metrics()
      except (SMUError, TimeoutError) as e:
        print(f"  {tag:<14} clocks unreadable: {type(e).__name__}: {e}", flush=True)
        return None
      print(f"  {tag:<14} gfx {m['gfxclk']:>4}  soc {m['socclk']:>4}  uclk {m['uclk']:>4}"
            + f"  fclk {m['fclk']:>4} MHz | gfx {m['gfx_activity']:>3}%"
            + f" | {m['socket_power']:>3} W  {m['temp_hotspot']:>3} C", flush=True)
      return m

    def pin_clocks():
      """Ask for the fastest configuration this card has been measured to survive.

      676 is DIMGREY_CAVEFISH_UMD_PSTATE_PROFILING_MEMCLK, the memory clock amdgpu's own
      profiling pstate uses for this ASIC. Measured here it is also the *only* UCLK level this
      SMU will enter: 456 and 1000 are both accepted in about 1.5 ms and then never answer
      another message, under soft-min, hard-min, ceiling-then-floor, and with every deep-sleep
      and memory-voltage feature masked off.

      GFXCLK gets a ceiling and no floor so its governor still boosts under load. Pinning it to
      a fixed 1950 measured slower than leaving it free to reach 2340.

      FCLK is deliberately absent: smu_v11_0_set_performance_level touches GFXCLK, MCLK and
      SOCCLK only and leaves the Data Fabric clock to PMFW.
      """
      for name, clck, lo, hi in (("GFXCLK", mod.PPCLK_GFXCLK, 0, 2350),
                                 ("UCLK", mod.PPCLK_UCLK, 676, 676),
                                 ("SOCCLK", mod.PPCLK_SOCCLK, 960, 1371)):
        try:
          # Ceiling before floor, the order smu_v11_0_set_soft_freq_limited_range uses.
          smu._send_msg(mod.PPSMC_MSG_SetSoftMaxByFreq, clck << 16 | hi, timeout=8000)
          smu._send_msg(mod.PPSMC_MSG_SetSoftMinByFreq, clck << 16 | lo, timeout=8000)
        except (SMUError, TimeoutError) as e:
          print(f"  pinning {name} failed: {type(e).__name__}: {e}", flush=True)
          return False
      return True

    print(f"pkl: {PKL}")
    t0 = time.perf_counter()
    jits = load_oob(open_file_chunked(PKL))
    print(f"loaded in {time.perf_counter() - t0:.1f}s;  device {key}")
    clocks("as booted")

    if CLOCKS == "target":
      if not pin_clocks():
        print("\n  clock pinning failed; the SMU is wedged, so any number below is meaningless")
        return 2
      time.sleep(1.0)
      clocks("pinned")

    # This pkl carries a flat metadata dict (input_shapes/output_slices at the top level), not
    # the nested {'model': ...} form. Take whichever is present rather than assuming.
    md = jits["metadata"]
    md = md.get("model", md)
    shapes = md["input_shapes"]
    legacy = "run_policy" not in jits
    cam = (1928, 1208)
    run_policy = jits[cam]["run_policy"] if legacy else jits["run_policy"]
    warp = None if legacy else jits[cam]
    print(f"{'legacy' if legacy else 'split'} pkl;  inputs {sorted(shapes)}")

    queues, npy = make_supercombo_input_queues(shapes, derive_frame_skip({}, shapes), key)
    # Whatever WARP_DEV the pkl was compiled with is where its warp JIT expects its frames. The
    # old form collapsed everything that was not QCOM onto Device.DEFAULT, which put frames on the
    # eGPU for a CPU-warp build and failed with "args mismatch in JIT" -- the captured graph names
    # the device, so a stand-in host has to feed the same one it compiled against.
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
      """run_policy returns a Tensor or a tuple of them. Realize whichever it is, so the timing
      covers the whole graph instead of stopping at the enqueue."""
      for t in (out if isinstance(out, tuple) else (out,)):
        t.numpy()

    print(f"\nwarmup x{WARMUP} (eager, capture, first replay)", flush=True)
    for i in range(WARMUP):
      st = time.perf_counter()
      force(one())
      dev.synchronize()
      print(f"  warmup {i}: {(time.perf_counter() - st) * 1e3:.0f} ms", flush=True)

    print(f"timing x{ITERS}", flush=True)
    ts = []
    for i in range(ITERS):
      st = time.perf_counter()
      force(one())
      dev.synchronize()
      ts.append(time.perf_counter() - st)
      if i == ITERS // 2:
        clocks("under load")
    s = summarize(ts)
    print(f"\n  min {s['min'] * 1e3:7.1f}  p50 {s['p50'] * 1e3:7.1f}  p99 {s['p99'] * 1e3:7.1f}"
          + f"  max {s['max'] * 1e3:7.1f}  mean {s['mean'] * 1e3:7.1f}  ms")

    # Where the frame actually goes. The GPU reports well under half activity during the run, so
    # the frame is not bound by the eGPU's compute, and "5x over budget" without saying which
    # stage owns the time is not an answer anyone can act on. The warp runs on the SoC's Adreno
    # (WARP_DEV=QCOM); everything after it is the eGPU plus the USB link it sits behind.
    if not legacy:
      warp_ts, total_ts = [], []
      for _ in range(10):
        st = time.perf_counter()
        w = warp(**{k: queues[k] for k in WARP_INPUTS},
                 frame=frames["frame"], big_frame=frames["big_frame"])
        for t in (w if isinstance(w, tuple) else (w,)):
          t.realize()
        Device[warp_dev].synchronize()
        warp_ts.append(time.perf_counter() - st)
        force(run_policy(**{k: queues[k] for k in POLICY_INPUTS if k in queues}, warped=w))
        dev.synchronize()
        total_ts.append(time.perf_counter() - st)
      wm, tm = float(np.mean(warp_ts)), float(np.mean(total_ts))
      print("\n  stage split (mean of 10):")
      print(f"    warp on {warp_dev:<5}          {wm * 1e3:7.1f} ms  ({wm / tm * 100:4.1f}%)")
      print(f"    policy on eGPU + transfer {(tm - wm) * 1e3:7.1f} ms  ({(tm - wm) / tm * 100:4.1f}%)")
    ok = s["max"] < BUDGET_MAX and s["mean"] < BUDGET_MEAN
    print("\n  budget: max < 60 ms, mean < 40 ms  ->  "
          + ("HOLDS" if ok else f"MISSES by {s['max'] / BUDGET_MAX:.1f}x on max,"
             + f" {s['mean'] / BUDGET_MEAN:.1f}x on mean"))
    print(f"  effective rate: {1 / s['mean']:.1f} Hz (need 20)")
    clocks("after")
    return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
