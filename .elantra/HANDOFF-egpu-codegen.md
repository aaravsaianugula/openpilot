# Handoff: the eGPU's bottleneck is how tinygrad lowers an fp16 MAC

Written 2026-08-28. Everything below is measured on the RX 6600 XT (Navi 23, gfx1032) in the
chestnut dock, driven over usbipd from WSL2 on the i7-11700K, unless it says otherwise.

This supersedes the "what is actually left: kernel quality, and only that" section of
`HANDOFF-egpu-clocks.md`. That section was right that the clocks are done and the kernels are the
problem. It was wrong about which part of the kernels.

## The headline, corrected by the end-to-end measurement

**One poisoned BEAM cache row was 38% of the frame.** Everything below about `v_fma_mix_f32` is
true and worth having, but it was the wrong target: measured end to end it moved the model from
120.4 ms to 120.2 ms mean (min 117.4 -> 113.1, p50 119.7 -> 116.0), because the GEMMs it speeds up
are 41% of the frame and something else was 38%.

A `DEBUG=2` compile prints one `*** AMD` line per dispatch during the eager pass before JIT
capture — 497 of them, the whole graph, individually timed. That measurement had never been taken.
It says:

| kernel | share of frame | MACs/launch | launches |
|---|---|---|---|
| `r_8_16_1536_7_7` | **38%** | 9.6 M (0.8% of model FLOPs) | 30 |
| `r_24_32_32_2_4_384_4_4` | 26% | 1.208 G | 29 |
| `r_24_8_16_2_8_4_4_384_4` | 15% | 1.208 G | 30 |

The top kernel is a 7x7 depthwise conv doing **0.8% of the arithmetic for 38% of the time**, at
~1.8 GFLOPS — 0.02% of peak. Its kernel descriptor says `.max_flat_workgroup_size: 1`:
**one thread per workgroup**, so 31 of every 32 wave32 lanes are masked off, with a scalar branch
and a 64-bit address computation per reduce tap (49 of each). Across the graph, **40% of all
kernel time was in kernels with a workgroup smaller than a single wavefront.**

The cause is a `beam_search_22` row with **zero applied_opts**, written by the `except Exception`
arm of `codegen/opt/search.py` when the device died mid-search. An empty row is worse than no row:
`apply_opts` takes the `elif beam >= 1` branch, returns a schedule with no LOCAL at all, and never
falls through to `hand_coded_optimizations`, which would have given a sane workgroup.

Find it by unpickling every `val` and looking for `len(applied_opts) == 0` — there was exactly one
AMD row (rowid 45) and one CPU row. Delete it and recompile; BEAM re-searches only that kernel and
found `[Opt(LOCAL, axis=1, arg=16)]` at **433 us against 5273 us, 12.2x on that kernel**.

The search then died on that same kernel again — it is the pathological one that has killed runs
before — but the patched `except` arm cached the good schedule first (now rowid 126), so the next
compile hits the cache and does not re-search. That is the patch working as designed.

**Still unmeasured**: the end-to-end frame time with that schedule, because the dock wedged before
the pkl was written. Projected from the profile share: 38% dropping ~12x takes the frame from
~120 ms to ~78 ms, about 12.8 Hz. Not 20 Hz, but the first fix that addresses the GPU actually
being idle rather than the code it runs.

## The v_fma_mix_f32 lowering

`v_fma_mix_f32` lowering is worth **1.23x** on the kernel shape that dominates the model, and it
makes the model *more* numerically accurate. It is one PatternMatcher rule.

| | fc1 (M=128 K=1536 N=6144) | fc2 (M=128 K=6144 N=1536) |
|---|---|---|
| before (`AMD_FMA_MIX=0`) | 1529 GFLOPS (1403-1631) | 1522 GFLOPS (1453-1558) |
| after (`AMD_FMA_MIX=1`) | **1885 GFLOPS** (1816-1929) | **1853 GFLOPS** (1621-2000) |
| speedup | **1.233x** | **1.218x** |
| max err/rms vs float64 | 2.39e-3 -> **2.17e-3** | 2.18e-3 -> **2.00e-3** |

Three runs per arm; `deps/fma_mix_ab.sh`. The fc1 ranges do not overlap.

## What was actually wrong

`sum_acc_dtype` ([dtype.py:213](../tinygrad_repo/tinygrad/dtype.py)) widens a half reduction's
accumulator to fp32, but the multiply stays in fp16. So every multiply-accumulate rendered as

```llvm
%m = fmul nsz arcp contract afn half %a, %b     ; v_mul_f16
%e = fpext half %m to float                      ; v_cvt_f32_f16   <- blocks all fusion
%s = fadd nsz arcp contract afn float %acc, %e   ; v_add_f32
```

and the convert sitting between the multiply and the add stops the backend contracting anything.
Counted over all 10,567 cached gfx1032 kernels: 188,324 `fmul half`, 156,308 `fpext half`,
187,322 `fadd float` — a 1:1:1 motif, and effectively the whole of the model's 208.5 GFLOP is
inside reductions.

Widening both operands *before* the multiply lets LLVM select a single `v_fma_mix_f32`, which
takes fp16 sources through op_sel and accumulates in fp32. Measured on the real hot kernel with
`.elantra/render_probe.py`:

| | AMD_FMA_MIX=0 | AMD_FMA_MIX=1 |
|---|---|---|
| v_mul_f16 / v_cvt_f32_f16 / v_add_f32 | 48 / 48 / 48 | 0 / 7 / 12 |
| v_fma_mix_f32 | 0 | 36 |
| MAC-path VALU per 48 MACs | **144** (3.00/MAC) | **55** (1.15/MAC) |
| whole loop body | 265 instructions | **173** |

It is also strictly more accurate: today's `fmul half` rounds every product to fp16 before adding
it, and this keeps the product in fp32. That is why the error goes *down*, not up.

The patch is in `AMDLLVMRenderer.__init__`, gated on `gfx10*` and `AMD_FMA_MIX` (default on).

## What is NOT the problem — measured, do not re-try these

- **The BEAM caps.** `BEAM_UPCAST_MAX` 64->256 and `BEAM_LOCAL_MAX` 256->1024, the workaround the
  old TODO wanted reverted, make **no difference**: 1838 vs 1906 GFLOPS mean across a 4-arm sweep,
  inside the noise. `ALLOW_HALF8=1` likewise. (`deps/gemm_sweep.sh`.)
- **Occupancy, register pressure, LDS, scratch.** The hot GEMM kernels use **57-76 VGPRs**, which
  is full occupancy on RDNA2, with `group_segment_fixed_size: 0` and
  `private_segment_fixed_size: 0` — no LDS, no spilling. (Extracted from NT_AMDGPU_METADATA with
  `llvm-readelf --notes`.)
- **Memory bandwidth.** The dominant GEMM has an arithmetic intensity of 116 FLOP/byte against a
  machine balance of ~55, and moves 20.8 MB per GEMM at 17 GB/s — 10% of the 172.8 GB/s available.
  UCLK 675 is not what is holding this back.
- **The "unknown number of baseline-only BEAM rows".** There is exactly **one** poisoned row
  (rowid 47). Exactly one run ever died with an exception, in `/root/beam.log`, and the
  `except Exception` arm writes its row last, so the poisoned row is the last one that run inserted.

## `v_dot2c_f32_f16`: real hardware, but not the 5.8x it looks like

gfx1030-gfx1036 **do** have `v_dot2c_f32_f16` — 2 fp16 MACs with fp32 accumulate in one
instruction. Confirmed by subtarget gating: `llvm.amdgcn.fdot2` selects on gfx1030/31/32/33/36 and
is rejected on gfx1010 (RDNA1) and gfx900. So "RDNA2 has no matrix unit, therefore no 2x path" is
only half right — there is no WMMA, but there is a dot product.

**LLVM will not form it on its own.** Tested left-associated chains, right-associated pairs, a
pairwise tree, scalar loads and `<2 x half>` loads with extractelement: every shape lowers to
`v_fma_mix_f32` instead. Getting `v_dot2c` requires emitting `llvm.amdgcn.fdot2` explicitly from
the renderer, which needs the two half operands paired into a real `<2 x half>` value — and
tinygrad emits **zero** vector ALU (`codegen/__init__.py` devectorizes every elementwise op and
`codegen/late/coalesce.py` re-vectorizes only loads and stores).

And the payoff is now small. After `v_fma_mix_f32` the MAC path is 66 of 173 loop instructions;
dot2 would take it to ~36, a 21% instruction reduction. The observed static-to-real ratio on this
rig is about 0.3, so that is roughly **5-7% end to end** — for a change that touches the WMMA
machinery, the Python reference emulator, and the codegen expansion passes. Not recommended
against the alternatives.

## Measurement discipline this rig requires

- **15-20% run-to-run spread.** The same cached schedule for the same shape produced 1980 and 1570
  GFLOPS on two boots. Always 3+ repeats per arm; report the range.
- **GFXCLK is not 2340 under load.** Sampled in-run it is 1825-2185 MHz. Any "% of peak" divided by
  2.34 GHz understates by up to 15%. `gemm_bench.py` prints the clock beside every number.
- **`BEAM_UPCAST_MAX`/`BEAM_LOCAL_MAX` are not in the beam_search cache key.** An arm with different
  caps silently reuses the previous arm's schedule unless `IGNORE_BEAM_CACHE=1` is in the
  environment — `search.py` binds it as a default argument at import, so `Context(...)` does nothing.
- **BEAM selects on time alone and nothing validates the winner.** tinygrad's check is
  `compile_linear(validate=VALIDATE_WITH_CPU)` and `realize.py` only applies it on the non-JIT path,
  so a captured graph is never validated. One sweep arm produced NaN outputs. It did not reproduce
  with the same cached schedule, so it is transient rather than a wrong kernel — but `gemm_bench.py`
  now fails loudly and exits non-zero on non-finite output.

## The three driver defects behind the unrecoverable hang

The old handoff blamed `AMDev.recover()` reaching "some register above the mapped MMIO range". That
is not what happens. The index is **data**, not an address.

1. **`ip.py` `AM_GMC.flush_hdp`** read `regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL` and used the whole
   dword `// 4` as a register index. Only bits 2..18 are the address (the autogen table declares
   `'address': (2, 18)`). When the upper bits read back non-zero — as they do here — the index
   lands outside the aperture, `rreg` falls through to the indirect RSMU window, and nbio 2.3 has
   no `regBIF_BX_PF0_RSMU_INDEX`. `flush_hdp` is on the recover() path, so recovery could never
   work. Now reads the field.
2. **No `GFX_10_*__SRCID__*` constants exist** in the autogen — only 9_0, 11_0_0 and 12_0_0 — so on
   RDNA1/RDNA2 every GFX interrupt resolved to `''`, missed the benign list and set `is_err_state`.
   gfx10 is now mapped onto the gfx9 table; all 24 values were checked against the kernel's own
   `ivsrcid/gfx/irqsrcs_gfx_10_1.h` and are identical.
3. **The benign list named `"CP_EOP_INTR"`**, which matches no SRCID constant on any generation —
   the real suffix is `CP_EOP_INTERRUPT`. End-of-pipe is the ordinary completion interrupt for
   every dispatch, so this was mis-flagged as an error on gfx9 and gfx11 too, not just here.

`indirect_rreg`/`indirect_wreg` now raise a named error instead of `KeyError` when an ASIC has no
RSMU window.

`.elantra/test_am_recover.py` covers all of it: **46 cases green**, and **32 of them fail** against
the unfixed driver.

## Transport batching: tried, reverted, and why

`AMDComputeQueue._submit` writes the ring one dword at a time, and on USB each element write is a
0xF0 control transfer plus a bulk OUT — so a bound graph's 4-dword `INDIRECT_BUFFER` packet costs
8 USB transactions where a contiguous write costs 2. `bind()` has the same shape and writes the
whole captured command buffer element-wise. `AMDCopyQueue._submit` already batches with
`ring[a:b] = array.array('I', ...)`, so the compute queue is simply inconsistent with it.

I implemented the batched form (with ring-wraparound splitting) and **reverted it**. Reasons, in
order: the first hardware run hung — fc1 replayed the graph 13 times cleanly and fc2 hung on its
first replay with `Wait timeout: 20000 ms! (signal is not set to 614, but 613)` — and that exact
signature has occurred on this rig **without** the change (`/root/beam.log`, and the
`3933/3932` failure in `HANDOFF-egpu-clocks.md`), so the evidence is genuinely ambiguous. Against
that, the measured upside is small: a graph submit is ~1.7 ms and there are a few per frame, so
this is worth **1-2 ms of a 127 ms frame**. An unverified change to GPU command submission, on a
car, for ~1%, is the wrong trade.

It is worth retrying when the dock is healthy, because the `bind()` half also speeds up JIT capture
(one-off, but it runs on every compile). Bisect it: revert `bind()` alone first, since `_submit`
demonstrably survived 13 replays. The bigger transport costs are elsewhere anyway and untouched by
this: every 8-byte host signal store is 8 USB control transfers (`usb.py` `write()` sends one
control transfer **per byte**), and `_apply_var_vals` writes each changed dword individually.

## Hardware state at handoff

The dock is **down and needs a physical replug**. After the hang above, the ASM2464 bridge stopped
answering (`libusb_get_string_descriptor_ascii: Operation timed out`), and a
`usbipd detach` removed VID:PID `3801:0001` from the Windows USB bus entirely — it did not
re-enumerate within 60 s. `slot_cycle.py` cannot recover this: it drives PCIe power *through* the
bridge, and it is the bridge that is unresponsive. Unplug and replug the dock, then
`usbipd bind --busid <n>` (elevated) and `usbipd attach --wsl --busid <n>` with a WSL session
already running.

Not yet measured because of this: the end-to-end model number with `v_fma_mix_f32`. The micro-bench
says 1.23x on the shape carrying ~73% of the FLOPs; the model needs a fresh BEAM compile
(`beam_wsl.sh`, ~40-60 min, the rewrite changes every AST so nothing hits the cache) and then
`model_bench.py`.

## Tools added

| File | What |
|---|---|
| `.elantra/gemm_bench.py` | Times the stage-3 GEMM on the card. Prints clocks, ISA histogram, and a correctness gate; exits non-zero on non-finite output. |
| `.elantra/render_probe.py` | **No GPU.** Renders a kernel for gfx1032 and reports the IR and ISA it produced. Turns a codegen experiment into two seconds. Runs `opt -passes=default<O2>` before `llc`, as tinygrad does — skipping it produces `flat_load` instead of `global_load` and is not representative. |
| `.elantra/test_am_recover.py` | The 46-case suite for the three driver defects above. 32 of them fail against the unfixed driver. |
| `.elantra/test_amd_codegen.py` | 15 cases for the `v_fma_mix_f32` lowering, at three levels: the UOp rewrite, the emitted IR, and the machine code the backend actually selects. Also pins that gfx1100/gfx1200/gfx942 are untouched and that `AMD_FMA_MIX=0` restores the old lowering. Six fail with the rule disabled. No GPU needed. |
| `deps/on_card.sh` | Standard environment + slot cycle + boot retry. Retries only failures that never reached the device. |
| `deps/fma_mix_ab.sh` | Repeated A/B with spread reporting. |
| `deps/gemm_sweep.sh` | The 4-arm BEAM knob sweep. |

Environment note: the venv moved from `/tmp/tgvenv` (cleared on restart) to **`/root/tgvenv`**, and
`pycapnp` must be pinned to 2.1.0 per the project's own `pyproject.toml`.

## Where this leaves 20 Hz

Honestly: not reachable by these means. The frame was 127.5 ms (105 ms policy graph + ~20.7 ms
transport and host). A graph-wide 1.23x puts the frame near **105 ms, about 9.5 Hz**. The measured
levers that remain — dot2 at ~5-7%, transport batching at ~1-2 ms — do not close a 2x gap.

What the evidence says the gap actually is: after `v_fma_mix_f32` the loop body is 173 instructions
for 48 MACs, of which only 66 are the MAC. The rest is 64-bit address arithmetic, output conversion
and stores, loads and waitcnt. And a 35% static instruction cut bought 23% real, so the kernel is
only partly issue-bound — the remainder is latency that full occupancy is already failing to hide.
Closing that means a genuinely tiled GEMM (LDS staging, register blocking, SADDR-form addressing),
which is a different and much larger piece of work than any single lowering rule.

For calibration: tinygrad's own CI gates this exact model on a comma 4 + chestnut at 50 ms
(`.github/workflows/benchmark.yml`, `ASSERT_MIN_STEP_TIME=50`) — on the RX 9060 comma ships, which
has WMMA matrix cores and reaches it with no BEAM at all.
