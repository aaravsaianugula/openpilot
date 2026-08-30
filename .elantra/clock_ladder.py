#!/usr/bin/env python3
"""Which clock request does this Navi 23's SMU actually accept, and which one kills it?

One rung per process, because a wedged SMU poisons every later message in the same open. The
caller slot-cycles the card between rungs (run_on_card.sh does), so each rung starts from the
same hardware state and the rungs are comparable.

The question this answers: floor.log shows SetSoftMinByFreq(UCLK, 1000) being accepted in 1.5 ms
and the *next* message -- SetSoftMinByFreq(FCLK) -- never being answered. Every reading after that
came from a dead SMU, so nobody has ever seen what the UCLK request did. amdgpu never sends a
soft limit to FCLK on this ASIC: smu_v11_0_set_performance_level touches GFXCLK, MCLK and SOCCLK
only, and leaves the Data Fabric clock to PMFW. Rung 0 makes the UCLK request on its own.

Rungs, RUNG=n:
  0  UCLK soft-min 1000, nothing else            -- the untested one
  1  amdgpu profile_peak: max-then-min on GFXCLK/UCLK/SOCCLK, no FCLK
  2  AMD's own pstate for this chip: gfx 1950 / soc 960 / mem 676
  3  UCLK soft-min 456      4  UCLK soft-min 675      5  UCLK soft-min 1000
  6  UCLK *hard*-min 1000 (0x1B) -- what amdgpu's display path uses to pin memory
  7  FCLK soft-min 1801 alone
  8  mask DS_UCLK/DS_FCLK/DF_CSTATE at boot, then UCLK soft-min 1000
  9  as 8, plus mask the memory voltage-scaling features
 10  as 8, plus every deep-sleep and D-state feature AM cannot service

Rungs 8-10 exist because the running feature mask says DS_UCLK(17), DS_FCLK(14) and DF_CSTATE(45)
are all active. AM already refuses DS_GFXCLK(12) with the comment "an engine that powers itself
down between submissions never comes back" -- the memory and fabric side of that were never
masked, and a UCLK DPM switch out of a fabric C-state is exactly a transition needing the
handshake AM does not implement.

PPCLK_DCEFCLK is never touched: forcing it is a confirmed GPU hang on Sienna Cichlid.
"""
import os
import sys
import time

RUNG = int(os.environ.get("RUNG", "0"))

# Extra bits to add to AM_SMU.SMU11_UNSERVICEABLE_FEATURES before the card is opened, per rung.
# Patched on the class rather than in the file: the mask is read inside _upload_pptable, which
# runs at device open, so this has to be in place before the first Device[...] touch.
EXTRA_UNSERVICEABLE = {
  8:  {17: "DS_UCLK", 14: "DS_FCLK", 45: "DF_CSTATE"},
  9:  {17: "DS_UCLK", 14: "DS_FCLK", 45: "DF_CSTATE", 10: "MEM_VDDCI_SCALING", 11: "MEM_MVDD_SCALING"},
  10: {13: "DS_SOCCLK", 14: "DS_FCLK", 15: "DS_LCLK", 16: "DS_DCEFCLK", 17: "DS_UCLK",
       19: "FW_DSTATE", 45: "DF_CSTATE"},
}
N = int(os.environ.get("N", "2048"))
GEMM_ITERS = int(os.environ.get("GEMM_ITERS", "8"))

from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context


def main() -> int:
  if (extra := EXTRA_UNSERVICEABLE.get(RUNG)):
    from tinygrad.runtime.support.am.ip import AM_SMU
    AM_SMU.SMU11_UNSERVICEABLE_FEATURES = {**AM_SMU.SMU11_UNSERVICEABLE_FEATURES, **extra}
    print("  masking off at boot: " + ", ".join(sorted(extra.values())), flush=True)
  with Context(DEV=os.environ.get("DEV", "USB+AMD:LLVM")):
    key = Device.DEFAULT
    dev = Device[key]
    adev = dev.iface.dev_impl
    smu, mod = adev.smu, adev.smu.smu_mod
    from tinygrad.runtime.support.am.ip import SMUError

    # SmuMetrics_t and SmuMetrics_V2_t agree up to offset 96 and diverge after it. Everything read
    # here lives below 96, so the raw offsets are version-independent and there is no struct to
    # pick wrong. CurrClock is indexed by PPCLK_e: GFXCLK 0, SOCCLK 1, UCLK 2, FCLK 3.
    def metrics(tag):
      try:
        smu._send_msg(mod.PPSMC_MSG_TransferTableSmu2Dram, mod.TABLE_SMU_METRICS, timeout=2000)
        raw = bytes(adev.vram.view(smu.driver_table_paddr, 96)[:])
      except (SMUError, TimeoutError) as e:
        print(f"  {tag:<22} SMU DEAD: {type(e).__name__}: {e}", flush=True)
        return None
      def u16(o): return int.from_bytes(raw[o:o + 2], "little")

      def u32(o): return int.from_bytes(raw[o:o + 4], "little")

      gfx, soc, uclk, fclk = u32(0), u32(4), u32(8), u32(12)
      print(f"  {tag:<22} gfx {gfx:>4}  soc {soc:>4}  uclk {uclk:>4}  fclk {fclk:>4} MHz"
            + f" | gfx act {u16(64):>3}%  uclk act {u16(66):>3}%"
            + f" | {u16(72):>3} W  {u16(76):>3} C", flush=True)
      return (gfx, soc, uclk, fclk)

    def send(name, msg, param, timeout=8000):
      st = time.perf_counter()
      try:
        smu._send_msg(msg, param, timeout=timeout)
        print(f"  {name:<44} ok    {(time.perf_counter() - st) * 1e3:7.2f} ms", flush=True)
        return True
      except (SMUError, TimeoutError) as e:
        print(f"  {name:<44} {type(e).__name__}: {e}", flush=True)
        return False

    def soft_min(clk, mhz):
      return send(f"SetSoftMinByFreq({clk[0]}, {mhz})", mod.PPSMC_MSG_SetSoftMinByFreq, clk[1] << 16 | mhz)

    def soft_max(clk, mhz):
      return send(f"SetSoftMaxByFreq({clk[0]}, {mhz})", mod.PPSMC_MSG_SetSoftMaxByFreq, clk[1] << 16 | mhz)

    def hard_min(clk, mhz):
      return send(f"SetHardMinByFreq({clk[0]}, {mhz})", mod.PPSMC_MSG_SetHardMinByFreq, clk[1] << 16 | mhz)

    GFX = ("GFXCLK", mod.PPCLK_GFXCLK)
    SOC = ("SOCCLK", mod.PPCLK_SOCCLK)
    UCK = ("UCLK", mod.PPCLK_UCLK)
    FCK = ("FCLK", mod.PPCLK_FCLK)

    rungs = {
      0: ("UCLK soft-min 1000, nothing else", lambda: soft_min(UCK, 1000)),
      1: ("amdgpu profile_peak, no FCLK", lambda: all([soft_max(GFX, 2350), soft_min(GFX, 2350),
                                                       soft_max(UCK, 1000), soft_min(UCK, 1000),
                                                       soft_max(SOC, 1371), soft_min(SOC, 1371)])),
      2: ("AMD pstate gfx1950/soc960/mem676", lambda: all([soft_max(GFX, 1950), soft_min(GFX, 1950),
                                                           soft_max(UCK, 676), soft_min(UCK, 676),
                                                           soft_max(SOC, 960), soft_min(SOC, 960)])),
      3: ("UCLK soft-min 456", lambda: soft_min(UCK, 456)),
      4: ("UCLK soft-min 675", lambda: soft_min(UCK, 675)),
      5: ("UCLK soft-min 1000", lambda: soft_min(UCK, 1000)),
      6: ("UCLK hard-min 1000", lambda: hard_min(UCK, 1000)),
      7: ("FCLK soft-min 1801 alone", lambda: soft_min(FCK, 1801)),
      8: ("DS_UCLK/DS_FCLK/DF_CSTATE masked, UCLK 1000", lambda: soft_min(UCK, 1000)),
      9: ("as 8 plus memory VR scaling masked, UCLK 1000", lambda: soft_min(UCK, 1000)),
      10: ("every deep-sleep masked, UCLK 1000", lambda: soft_min(UCK, 1000)),
      # Rung 2 (gfx 1950 / soc 960 / mem 676, each pinned max-then-min) is the first thing that
      # moved memory. Rungs 0/3/6 asked for a floor with the ceiling left at 0xffff and wedged;
      # rung 1 used the same max-then-min shape as rung 2 but asked for 1000. So either pinning
      # max==min is what makes a UCLK switch survivable, or the top memory state is unreachable.
      # These four separate the two.
      11: ("UCLK pinned 676, nothing else", lambda: all([soft_max(UCK, 676), soft_min(UCK, 676)])),
      12: ("rung 2 shape, UCLK 1000", lambda: all([soft_max(GFX, 1950), soft_min(GFX, 1950),
                                                   soft_max(UCK, 1000), soft_min(UCK, 1000),
                                                   soft_max(SOC, 960), soft_min(SOC, 960)])),
      13: ("UCLK pinned 456, nothing else", lambda: all([soft_max(UCK, 456), soft_min(UCK, 456)])),
      # The practical target: memory pinned at the level AMD themselves validate for this chip,
      # SOCCLK up, and GFXCLK left a floor of 0 so its governor can still boost past 2282 under
      # load rather than being held at a pinned value.
      14: ("target: gfx<=2350 free, soc 1371, uclk 676", lambda: all([soft_max(GFX, 2350), soft_min(GFX, 0),
                                                                     soft_max(UCK, 676), soft_min(UCK, 676),
                                                                     soft_max(SOC, 1371), soft_min(SOC, 960)])),
    }
    if RUNG not in rungs:
      print(f"no rung {RUNG}")
      return 2
    title, action = rungs[RUNG]

    print(f"=== RUNG {RUNG}: {title} ===", flush=True)
    metrics("as booted")

    print("  --- request ---", flush=True)
    action()

    print("  --- settle ---", flush=True)
    for dt in (0.2, 1.0, 3.0):
      time.sleep(dt)
      after = metrics(f"+{dt}s")
      if after is None:
        break

    if after is None:
      print("\n  VERDICT: SMU wedged by this rung. Not running a kernel on a dead card.", flush=True)
      return 1

    print("  --- under load ---", flush=True)
    a = Tensor.rand(N, N, dtype=dtypes.half, device=key).realize()
    b = Tensor.rand(N, N, dtype=dtypes.half, device=key).realize()
    dev.synchronize()
    flop, best = 2.0 * N * N * N, 0.0
    for i in range(GEMM_ITERS):
      st = time.perf_counter()
      (a @ b).realize()
      dev.synchronize()
      best = max(best, flop / (time.perf_counter() - st) / 1e9)
      if i == GEMM_ITERS - 2:
        loaded = metrics("under load")
    print(f"\n  {best:.1f} GFLOPS ({flop / best / 1e6:.1f} ms/iter)", flush=True)

    uclk = (loaded or after)[2]
    moved = uclk >= 456
    print(f"\n  VERDICT: uclk {uclk} MHz -- " + ("MEMORY MOVED" if moved else "still at its floor"), flush=True)
    return 0 if moved else 3


if __name__ == "__main__":
  sys.exit(main())
