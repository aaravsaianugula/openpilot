#!/usr/bin/env python3
"""Time one stage-3 GEMM of the big driving model on the eGPU, and show the ISA it compiled to.

Recompiling the whole model takes ~15 minutes, which is far too slow to iterate a codegen change
against. This isolates the shape that dominates the frame. Of the 93 distinct AMD kernels in the
policy graph, ~22 do exactly 128 x 1536 x 6144 = 1,207,959,552 multiply-accumulates each -- the
two matmuls of a ConvNeXt stage-3 block, which between them carry roughly three quarters of the
model's 208.5 GFLOP per frame. That count is not inferred from the ONNX: it is the product of the
shape dims in the compiled kernels' own names, in the compile cache this project already built.

Three numbers are reported together, because a change can win on one and lose on another:

  instruction mix   Is a multiply-accumulate one instruction or three? Today tinygrad emits
                    v_mul_f16 + v_cvt_f32_f16 + v_add_f32 -- 2.88 VALU ops per MAC measured over
                    the real kernels. v_fma_mix_f32 would be 1.25 and v_dot2c_f32_f16 0.5.
  throughput        Against this card's two real ceilings, at the 2340 MHz this project pins:
                      2048 lanes x 2 flop x 2.34 GHz  =  9.58 TFLOPS  scalar fp32 FMA, v_fma_mix
                      x2 for packed fp16              = 19.17 TFLOPS  v_pk_fma_f16, v_dot2c
  numerics          Max relative error against a float64 reference. The fast lowerings compute the
                    product in fp32 instead of rounding it to fp16, so this should IMPROVE; a
                    regression here means the rewrite is wrong, not that the tolerance is tight.

Wall time and device time are both printed. They differ by the graph launch, which over USB costs
several milliseconds of round trips whatever the kernel size -- REPS exists to amortise that, and
the per-GEMM figures come from device time so the launch neither flatters nor penalises a codegen
change.

Env: M/K/N override the shape, REPS the GEMMs per graph, ITERS/WARMUP the timing loop, BOTH=0 to
skip the transposed shape, ISA=0 to skip disassembly.
"""
import os
import pickle
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter

import numpy as np

# The card, at the clocks .elantra/model_bench.py pins.
LANES, CLOCK_HZ = 2048, 2.34e9
PEAK_FP32 = LANES * 2 * CLOCK_HZ          # scalar FMA, and v_fma_mix_f32
PEAK_FP16 = PEAK_FP32 * 2                 # v_pk_fma_f16, v_dot2c_f32_f16

M = int(os.environ.get("M", "128"))
K = int(os.environ.get("K", "1536"))
N = int(os.environ.get("N", "6144"))
REPS = int(os.environ.get("REPS", "16"))
ITERS = int(os.environ.get("ITERS", "10"))
WARMUP = int(os.environ.get("WARMUP", "3"))
BOTH = os.environ.get("BOTH", "1") == "1"
WANT_ISA = os.environ.get("ISA", "1") == "1"

CACHE_DB = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                        "tinygrad", "cache.db")

# Instruction families, ordered so the output reads as the story: what the MAC became, then what
# the loads became, then the overheads that are the other half of the frame.
ISA_FAMILIES = (
  ("v_dot2c_f32_f16", r"v_dot2c?_f32_f16"),
  ("v_pk_* packed", r"v_pk_"),
  ("v_fma_mix_f32", r"v_fma_mix"),
  ("v_fma/fmac_f32", r"v_(fma|fmac|mac|mad)_f32"),
  ("v_fma/fmac_f16", r"v_(fma|fmac|fmaak|mac|mad)_f16"),
  ("v_mul_f16", r"v_mul_f16"),
  ("v_add_f32", r"v_add_f32"),
  ("v_cvt_f32_f16", r"v_cvt_f32_f16"),
  ("global_load_ushort", r"global_load_ushort"),
  ("global_load_dwordxN", r"global_load_dwordx"),
  ("LDS ds_read/write", r"ds_(read|write)"),
  ("scratch spill", r"buffer_(load|store)_dword"),
  ("exec-mask branch", r"s_(and_saveexec|cbranch_execz)"),
  ("s_waitcnt", r"s_waitcnt"),
)


def find_llvm_objdump():
  for cand in ("llvm-objdump-18", "llvm-objdump-19", "llvm-objdump-20", "llvm-objdump"):
    if subprocess.run(["which", cand], capture_output=True).returncode == 0:
      return cand
  return None


def cache_tables(con, arch):
  return [r[0] for r in con.execute("select name from sqlite_master where type='table'")
          if r[0].startswith("compile_llvm_" + arch)]


def kernels_with_mac_count(con, arch, wanted):
  """Named reduce kernels in the compile cache whose shape dims multiply out to `wanted` MACs.

  tinygrad names a kernel from its full_shape, so the product of the numbers in the name is the
  kernel's total loop-iteration count. For a GEMM that is exactly M*K*N. Matching on it finds the
  real kernel this bench just built without having to guess how the shape was split into axes.
  """
  found = {}
  for table in cache_tables(con, arch):
    for (key,) in con.execute(f'select key from "{table}"'):
      m = re.match(r"define amdgpu_kernel void @(r_[A-Za-z0-9_]+)\(", key)
      if m is None:
        continue
      dims = [int(x) for x in re.findall(r"_(\d+)", m.group(1))]
      if not dims:
        continue
      prod = 1
      for d in dims:
        prod *= d
      if prod in wanted:
        found.setdefault(m.group(1), prod)
  return found


def disassemble_kernel(con, arch, name):
  """Disassemble this kernel's object straight out of tinygrad's own compile cache.

  Reading the cache rather than hooking the compiler keeps the bench independent of tinygrad
  internals, and it works whether the kernel was just compiled or came back as a cache hit.
  """
  objdump = find_llvm_objdump()
  if objdump is None:
    return None, "no llvm-objdump on PATH"
  needle = f"void @{name}("
  for table in cache_tables(con, arch):
    for key, val in con.execute(f'select key, val from "{table}"'):
      if needle not in key:
        continue
      # tinygrad's diskcache stores values pickled. This is that cache, written by tinygrad on
      # this machine minutes ago -- not an untrusted artifact -- and it is the only format the
      # compiled object is available in.
      elf = pickle.loads(val)
      if not isinstance(elf, (bytes, bytearray)) or elf[:4] != b"\x7fELF":
        continue
      path = f"/tmp/gemm_bench_{name}.elf"
      with open(path, "wb") as f:
        f.write(elf)
      out = subprocess.run([objdump, "-d", "--mcpu=" + arch, path],
                           capture_output=True, text=True).stdout
      ops = Counter()
      for line in out.splitlines():
        m = re.match(r"\s+(v_|s_|global_|ds_|buffer_|flat_|scratch_)([a-z0-9_]+)", line)
        if m:
          ops[m.group(1) + m.group(2)] += 1
      return ops, path
  return None, f"kernel {name!r} not in the compile cache"


def report_isa(ops):
  real = sum(ops.values()) - ops.get("s_code_end", 0)
  valu = sum(v for k, v in ops.items() if k.startswith("v_") and k != "v_mov_b32_e32")
  print(f"    {real} instructions, {valu} VALU (excluding v_mov)")
  for label, pat in ISA_FAMILIES:
    rx = re.compile(pat)
    n = sum(v for k, v in ops.items() if rx.match(k))
    if n:
      print(f"      {label:<22} {n:6d}")
  return real, valu


def clocks(tag):
  """Print the clocks, because every '% of peak' below is divided by an assumed 2340 MHz.

  A GEMM that is really running at 1500 MHz is not 20% of peak, it is 32%, and the two lead to
  opposite conclusions about how much headroom is left. Same accessor model_bench.py uses.
  """
  from tinygrad import Device
  try:
    smu = Device[Device.DEFAULT].iface.dev_impl.smu
    m = smu.metrics()
  except Exception as e:  # no AM device (a non-USB run), or a wedged SMU: say so and carry on
    print(f"  clocks         unreadable: {type(e).__name__}: {e}")
    return None
  print(f"  clocks {tag:<8} gfx {m['gfxclk']:4d}  soc {m['socclk']:4d}  uclk {m['uclk']:4d}"
        + f"  fclk {m['fclk']:4d} MHz | gfx {m['gfx_activity']:3d}%"
        + f" | {m['socket_power']:3d} W  {m['temp_hotspot']:3d} C")
  return m


def run_shape(m, k, n, label):
  from tinygrad import Device, Tensor
  from tinygrad.engine.jit import TinyJit
  from tinygrad.helpers import GlobalCounters

  dev = Device.DEFAULT
  macs = m * k * n
  flops = 2 * macs
  print(f"\n=== {label}  M={m} K={k} N={n}  fp16 in / fp32 acc  x{REPS} reps ===")
  print(f"  {macs} MACs per GEMM ({flops / 1e9:.3f} GFLOP), {REPS} GEMMs per graph")

  rng = np.random.default_rng(0)
  # Small magnitudes on purpose: the numerics check is about the precision of the product and the
  # accumulation order, not about fp16 overflowing at K=6144.
  a_np = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
  b_np = [(rng.standard_normal((k, n)) * 0.1).astype(np.float32) for _ in range(REPS)]
  a = Tensor(a_np.astype(np.float16), device=dev).realize()
  bs = [Tensor(x.astype(np.float16), device=dev).realize() for x in b_np]

  @TinyJit
  def run(a, *bs):
    outs = [a.matmul(b) for b in bs]
    Tensor.realize(*outs)
    return outs

  out = None
  for _ in range(WARMUP):
    out = run(a, *bs)
    Device[dev].synchronize()

  GlobalCounters.reset()
  wall = []
  for i in range(ITERS):
    st = time.perf_counter()
    out = run(a, *bs)
    Device[dev].synchronize()
    el = time.perf_counter() - st
    if i == ITERS // 2:
      # Sampling the SMU costs a USB round trip, so this iteration's wall time is not comparable.
      # Device time is unaffected -- GlobalCounters only accumulates measured kernel time.
      clocks("in-run")
    else:
      wall.append(el)
  dev_s = GlobalCounters.time_sum_s / ITERS
  wall_s = float(np.mean(wall))

  per_gemm = dev_s / REPS if dev_s > 0 else 0.0
  gflops = flops / per_gemm / 1e9 if per_gemm > 0 else 0.0

  print(f"  device time    {dev_s * 1e3:8.2f} ms / graph   {per_gemm * 1e3:8.3f} ms / GEMM")
  print(f"  wall time      {wall_s * 1e3:8.2f} ms / graph   launch+host overhead {(wall_s - dev_s) * 1e3:.2f} ms")
  if per_gemm > 0:
    print(f"  throughput     {gflops:8.0f} GFLOPS   {100 * gflops * 1e9 / PEAK_FP32:5.1f}% of fp32 peak"
          + f"   {100 * gflops * 1e9 / PEAK_FP16:5.1f}% of packed-fp16 peak")
  else:
    print("  throughput     device time unavailable -- run with DEBUG>=2 so tinygrad times kernels")

  got = out[0].numpy().astype(np.float64)
  ref = a_np.astype(np.float64) @ b_np[0].astype(np.float64)
  # A throughput number for a kernel that computes garbage is worse than no number. BEAM selects
  # purely on measured time and nothing validates the winner: tinygrad's own check is
  # compile_linear(validate=VALIDATE_WITH_CPU), and realize.py only applies it on the non-JIT
  # path, so a captured graph is never validated. Say so loudly rather than letting a nan reach
  # the summary line as if it were a tolerance to be widened.
  bad = int((~np.isfinite(got)).sum())
  if bad:
    print(f"  numerics       *** {bad} of {got.size} outputs are not finite"
          + " -- this kernel is WRONG ***")
  # Scale by the RMS of the reference, not per-element by |ref|. A K=1536 dot of zero-mean random
  # inputs produces outputs that pass through zero, and dividing by one of those reports a huge
  # relative error for an absolutely tiny one -- which would make this gate meaningless. RMS-
  # relative error is the standard GEMM accuracy measure and is what moves when the lowering
  # changes the precision of the product.
  rms = float(np.sqrt(np.mean(ref ** 2)))
  err = np.abs(got - ref)
  max_rel, mean_rel = float(err.max() / rms), float(err.mean() / rms)
  print(f"  numerics       max err/rms {max_rel:.3e}   mean err/rms {mean_rel:.3e}   (ref rms {rms:.4f}, vs float64)")

  return {"label": label, "macs": macs, "per_gemm": per_gemm, "gflops": gflops,
          "max_rel": max_rel, "mean_rel": mean_rel, "nonfinite": bad}


def main():
  from tinygrad import Device
  from tinygrad.helpers import getenv

  dev = Device.DEFAULT
  arch = getattr(Device[dev], "arch", "gfx1032")
  print(f"device {dev}   arch {arch}")
  print(f"peaks: fp32 FMA {PEAK_FP32 / 1e12:.2f} TFLOPS, packed fp16 {PEAK_FP16 / 1e12:.2f} TFLOPS"
        + f"  ({LANES} lanes @ {CLOCK_HZ / 1e9:.2f} GHz)")
  print(f"env: BEAM={getenv('BEAM', 0)} SUM_DTYPE={os.environ.get('SUM_DTYPE', 'float32')}"
        + f" ALLOW_HALF8={getenv('ALLOW_HALF8', 0)} BEAM_UPCAST_MAX={getenv('BEAM_UPCAST_MAX', 256)}"
        + f" BEAM_LOCAL_MAX={getenv('BEAM_LOCAL_MAX', 1024)} DEBUG={getenv('DEBUG', 0)}")

  shapes = [(M, K, N, "stage-3 fc1")]
  if BOTH and K != N:
    shapes.append((M, N, K, "stage-3 fc2"))

  results = [run_shape(m, k, n, label) for m, k, n, label in shapes]

  if WANT_ISA:
    print("\n=== compiled ISA ===")
    if not os.path.exists(CACHE_DB):
      print(f"  no compile cache at {CACHE_DB}")
    else:
      con = sqlite3.connect(CACHE_DB)
      names = kernels_with_mac_count(con, arch, {r["macs"] for r in results})
      if not names:
        print("  no reduce kernel in the cache has a shape product matching {}".format(sorted({r["macs"] for r in results})))
      for name in sorted(names):
        print(f"\n  kernel {name}")
        ops, meta = disassemble_kernel(con, arch, name)
        if ops is None:
          print(f"    {meta}")
          continue
        report_isa(ops)

  print("\nsummary")
  for r in results:
    warn = f"  *** {r['nonfinite']} NON-FINITE OUTPUTS ***" if r["nonfinite"] else ""
    print(f"  {r['label']:<14} {r['per_gemm'] * 1e3:7.3f} ms/GEMM  {r['gflops']:6.0f} GFLOPS"
          + f"  {100 * r['gflops'] * 1e9 / PEAK_FP32:5.1f}% fp32 peak"
          + f"  {100 * r['gflops'] * 1e9 / PEAK_FP16:5.1f}% packed-fp16 peak"
          + f"  err/rms max {r['max_rel']:.2e} mean {r['mean_rel']:.2e}{warn}")
  # Non-zero exit on a wrong kernel: a fast wrong answer must not read as a good result.
  return 1 if any(r["nonfinite"] for r in results) else 0


if __name__ == "__main__":
  sys.exit(main())
