# Handoff: the 6600 XT's clocks are fixed, and what is left is kernel quality

Written 2026-08-28, at a hardware-swap pause. Everything below is measured on the bench comma 4
(`comma@192.168.12.238`) with the RX 6600 XT in the chestnut dock, unless it says otherwise.

## State: what changed and what it bought

The card boots at UCLK 96 MHz of a possible 1000. That is now fixed in the driver, and fixing it
is what made the big driving model run a frame at all — it had never completed one.

| | before | after |
|---|---|---|
| UCLK (memory) | 96 MHz, 24.6 GB/s | **675 MHz, 172.8 GB/s** |
| GFXCLK under load | 2282 MHz (never was stuck) | 2340 MHz |
| SOCCLK | 480–800 MHz | 800–960 MHz |
| big model | **hung**, every time | **208 ms/frame, 4.8 Hz** |

The model still misses openpilot's budget (`("modelV2", 0.06, 0.040)`) by 5.2x on mean.

## The one fact to carry forward

**Only UCLK level 2 (675 MHz) can be entered on this ASIC.** The DPM table is
`[96, 456, 675, 1000]`; 456 and 1000 are both ACKed in ~1.5 ms and then the SMU never answers
another message — every later message times out, including `TransferTableSmu2Dram`. Eleven rungs
of `clock_ladder.py` reproduced it under soft-min, hard-min (0x1B), ceiling-then-floor, with
GFXCLK/SOCCLK raised first and not, and with `DS_UCLK`(17), `DS_FCLK`(14), `DF_CSTATE`(45),
`MEM_VDDCI_SCALING`(10) and `MEM_MVDD_SCALING`(11) masked off. 675 is what amdgpu's own
`DIMGREY_CAVEFISH_UMD_PSTATE_PROFILING_MEMCLK 676` snaps down to.

That wedge **was** the model hang. A/B on one binary in one session: `CLOCKS=none` (uclk 96) dies
with `Wait timeout: 30000 ms (signal not set to 3933, but 3932)`, byte-identical to the old
`mb4.log` / `compile2.log` failures; `CLOCKS=target` (uclk 675) completes 30 frames. There was
never a separate GPU bug.

## Two things I got wrong, so nobody re-derives them

**GFXCLK was never stuck at 500 MHz.** That is the *idle* clock and it is correct behaviour. The
governor reaches 2282–2340 MHz under load on its own, and did so before any of this work. Every
"pinned at 500 MHz" note in the older handoffs is reading an idle sample.

**The GPU is not starved.** I read SmuMetrics `AverageGfxActivity` at 20–30% and concluded the
card sat idle two-thirds of the frame behind a dispatch/transport bottleneck. Wrong. That field
averages over its own window and reads `500 MHz / 0%` when it samples at idle. The graph's own
timing is GPU-measured:

```
batched 497   tm 194.49ms
batched 497   tm 191.78ms
```

**194 ms of a 208 ms frame is real execution — the card is ~93% busy.** Transport is 5.8 ms total
(2 KB + 384 KB copies). Trust the `batched N ... tm` line over the SMU average.

Related trap: `.elantra/gpu_utilization.py` cannot measure compute. A captured JIT graph emits no
compute `ProfileRangeEvent`s, so `PROFILE=1` returns only `AMD:SDMA:0` ranges — the copy engine,
8.49 ms/frame. The script says so in its own docstring now.

## What is actually left: kernel quality, and only that

202 GFLOP/frame in 194 ms = **1.04 TFLOPS = 5.4% of the card's 19.2 TFLOPS packed-fp16 peak.**
20 Hz needs 4.0 TFLOPS (~21% of peak); the 40 ms mean budget needs 5.1 TFLOPS (~26%). Well-tuned
RDNA2 kernels reach 30–60%, so the target sits below what this part should manage.

Why the kernels are poor: the pkl was compiled with `TC_OPT=2`, which is a **no-op on gfx10**
(`tc.py` returns `[]` for `gfx10*` — RDNA2 has no matrix unit), and with **no BEAM search
anywhere** — not in `openpilot/selfdrive/modeld/SConscript`, not in the compile env. BEAM is the
whole lever.

### BEAM over USB does not work well, for two separate reasons

1. **It fails outright by default.** `search.py:157-159` sets `early_stop` to 3x the best kernel
   time and, with `BEAM_DEV_TIMEOUT=1` (the default), enforces it as a *device-side* wait
   deadline — about 1 ms for these kernels. One SMU round trip over the chestnut measures
   1.47–1.75 ms, so every candidate times out with
   `Wait timeout: 1 ms! (the signal is not set to 130, but 130)`, a value that had already
   arrived. **`BEAM_DEV_TIMEOUT=0` fixes this** and the search then runs; `compile_beam.sh` on
   the device sets it.
2. **Even fixed it is impractically slow:** ~1.5 kernels/min with the comma's CPU 79.5% idle —
   pure USB round-trip latency, ~95 candidate compiles per kernel. ~500–700 kernels is 5–8 hours.

### The way out: the BEAM cache is portable

`search.py:115`:

```python
key = {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size,
       "device": s.ren.target.device, "suffix": s.ren.suffix}
```

`target.device` is `"AMD"`. The *interface* (USB vs PCIe) is a separate `Target` field and is
**not** in the key, and the cached value is `applied_opts` — a schedule, not machine code. So a
BEAM run with the card in a real PCIe slot transfers straight back to the dock.

- **Trap: `arch` is not in the key either.** A cache built on gfx1100/gfx1200 would be silently
  applied to gfx1032, with wrong wave-size and LDS assumptions. Same card, or same arch, only.
- **Requires Linux.** `tinygrad/runtime/ops_amd.py:4` is `assert sys.platform != 'win32'`.
  A live USB is enough; no AMD driver install, tinygrad drives the card itself.

Procedure: card into a PCIe slot on Linux -> compile the same ONNX with `BEAM=2` -> copy
`~/.cache/tinygrad/cache.db` back to `/data/rdna2-cache/tinygrad/` -> card back in the dock ->
recompile on the comma (every kernel hits the cache, no search, ~15 min) -> `model_bench.py`.

A partial run (26 tuned kernels, 2407 gfx1032 compiles, 56.7 MB) was harvested before the swap.
Merging a partial cache is safe — misses just get searched.

## Files

| File | What |
|---|---|
| `tinygrad_repo/.../am/ip.py` | `AM_SMU.metrics()` (SMU-version-aware) and `_set_clocks_smu11` |
| `openpilot/selfdrive/modeld/modeld.py` | `ChestnutState` now calls `smu.metrics()` |
| `.elantra/clock_ladder.py` | the 15-rung SMU experiment harness |
| `.elantra/model_bench.py` | times the model, always beside the clocks it ran at |
| `.elantra/gpu_utilization.py` | transfer cost only — see the compute caveat above |
| `.elantra/test_am_clocks.py` | 74 cases, incl. the clock policy and metrics decoding |

`set_clocks` on SMU 11 now: no FCLK (amdgpu never sets it on this part), ceiling before floor,
GFXCLK/SOCCLK left on `(0, 0xffff)` — amdgpu's AUTO encoding — and UCLK pinned to the ASIC's
profiling level. Teardown deliberately leaves memory alone: dropping UCLK back to 96 wedges the
SMU the same way raising it does. SMU 13/14 are untouched and a test pins that.

## Verification at the pause

All six `.elantra` suites exit 0 — 417 cases. `ruff` clean on every changed file. Last hardware
run: card boots `uclk 675` unaided, 30 frames, 208.9 ms mean.

Known open: `openpilot/sunnypilot/egpu/asics.py:69` still blocklists `0x73FF`, so the shipping
path will not select this card. That gate stays shut until the numbers justify opening it.

---

# Addendum, same day: the search moved to a Windows host, and what that cost

## The eGPU runs from a desktop via WSL2, with zero tinygrad changes

`ops_amd.py:4` asserts `sys.platform != 'win32'`, but WSL2 is Linux so it never fires. The dock
stays on USB; no PCIe slot is involved.

    winget install dorssel.usbipd-win
    usbipd bind --busid <n>          # elevated
    usbipd attach --wsl --busid <n>  # a WSL2 session must already be running or this fails

In the distro: `libusb-1.0-0 usbutils clang`, and a venv with `numpy onnx pycapnp zstandard pyzmq
requests tqdm Pillow psutil setproctitle crcmod pycryptodome sympy`. `opendbc_repo` is not checked
out locally -- tar it off the comma (11 MB). The card then boots:
`AMDDevice: opening 0 with target (10, 3, 2) arch gfx1032`, `uclk 675` straight from the driver.

Native Windows would be a ~20-50 line guard patch rather than a port (`USBIface` bypasses
`PCIIfaceBase.__init__`; `USBPCIDevice` overrides everything touching sysfs or mmap), but there is
no reason to do it while WSL2 works.

## Why the host matters and the interface does not

BEAM's cost is ~100% LLVM compile. Median compile of this model's real kernels: **642.8 ms on the
comma, 73.0 ms on an i7-11700K** -- 8.8x per thread, 16 threads against ~4. USB latency
contributes nothing to the search. An earlier reading of "80% CPU idle" as latency-bound was
wrong: LLVM is single-threaded, so one pegged core of eight looks idle.

Run everything from ext4, never `/mnt/d`: importing tinygrad measured **0.66 s over 9p vs 0.12 s
native**, and BEAM respawns its worker pool constantly.

## The memory ceiling is a driver gap, not the card

Every retail RX 6600 XT runs 2000 MHz memory (UCLK 1000) under amdgpu, and this card's DPM table
offers it. Under AM it is accepted and then the SMU dies -- **reproduced on the i7-11700K over a
different USB controller**, so it is not a comma artifact. DC-mode capping is ruled out:
`GetDcModeMaxDpmFreq(UCLK)` reads 1000, same as the AC ceiling, and `NotifyPowerSource(AC)`
succeeds and changes nothing. 456 failing while 675 works still argues that only the boot and
profiling memory straps are trained.

## Power, from primary sources rather than the number everyone repeats

The 2024 Elantra owner's manual states the accessory-outlet limit twice, verbatim: **"The devices
should draw less than 180 W (watts) with the engine running."** Socket fuse **20 A**, 40 A
upstream. Comma publishes **no** chestnut figure at all -- the widely-quoted 100 W traces to
Phoronix, not to comma. What comma does document: chestnut is comma-four-only, in-car power comes
from the cigarette lighter, one 8-pin cable, and the PCIe link is **Gen3 x2 by deliberate design**
(~1.66 GB/s; their firmware caps it because the completer saturates either way).

Measured at `AM_POWER_LIMIT=100`, 9000 samples: mean **21.6 W**, p99 40.9 W, max **85.8 W**,
fault byte 0 throughout -- 48% of the car's limit. The model draws only 30-37 W, so power is not
the binding constraint. The manual's own warning is about **plug contact resistance**, not
current: "The plug may overheat and the fuse may open." Hard-wire rather than use a barrel plug,
and note there is **no battery-saver on that outlet**.

## Three defects found here, two of them mine

1. **`AMDev.recover()` is broken on this ASIC** (upstream, not ours). A hung candidate kernel ends
   in `KeyError: 'regBIF_BX_PF0_RSMU_INDEX'` -- `amdev.py:487` falls back to an indirect RSMU
   window for registers above the mapped MMIO range, and Navi 23 has no such register. One hang
   kills a whole search. Survivable only because the BEAM cache is on disk and keyed by kernel
   AST, so a supervisor that restarts resumes rather than redoes.
2. **The teardown path queried the SMU** (mine). `_set_clocks_smu11(level=0)` called
   `read_clocks((gfx, soc))`; `read_clocks` is cached per clk_list tuple, so that key missed the
   cache the boot path filled and issued a real `GetDpmFreqByIndex` during `fini`. Against a
   wedged SMU that is a 10 s timeout, and `fini` suppresses `SMUError` but not `TimeoutError` --
   so it escaped and **buried the real failure**. Teardown now asks for `(0, 0)`, the firmware
   minimum, and touches the SMU not at all. Regression test:
   `test_smu11_teardown_asks_the_smu_nothing`.
3. **`BEAM_DEV_TIMEOUT=0` was the wrong fix** (mine). The stock deadline is 3x the best kernel
   time -- about 1 ms -- below the chestnut's 1.5-2.9 ms round trip, so every candidate timed out
   spuriously. Disabling it let a pathological candidate run to `HCQDEV_WAIT_TIMEOUT_MS` and look
   exactly like a GPU hang, which defect 1 cannot survive. The fix is a floor, patched into
   `codegen/opt/search.py` as `BEAM_DEV_TIMEOUT_MIN_MS` (0 preserves upstream behaviour).

## Shell traps that cost real time

- **`pkill -f compile_modeld` matches the invoking shell's own command line and kills it.** The
  command then returns empty with no error and reads as a crash. Put the pattern in a script file.
- `wsl.exe -- bash -lc` fails on this distro (broken systemd user session); use `bash -c`.
- Heredocs through `wsl.exe` silently truncate. Write the file on the Windows side and `cp` it in.
- `echo \$? > done` inside a double-quoted `bash -c` expands in the *outer* shell, so the file
  always reads 0 and every run looks successful.
