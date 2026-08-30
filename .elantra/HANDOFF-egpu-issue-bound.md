# Handoff: the kernels are issue-bound, and 80% of peak is already demonstrated in this model

Written 2026-08-28 evening. Supersedes the diagnosis in `HANDOFF-egpu-codegen.md`, which was built
on a profile that predated its own cache write. All measured on the RX 6600 XT (gfx1032) in the
chestnut dock over usbipd from WSL2 unless stated.

## The number

| state | policy graph | frame | rate |
|---|---|---|---|
| previous handoff | 91.1 ms | 115.8 ms | 8.6 Hz |
| **now** | **64.6 ms** | **89.6 ms** | **11.2 Hz** |
| target | ~31 ms | 50 ms | 20 Hz |

3 repeats per arm: 117.0 / 114.9 / 115.5 then 90.3 / 91.0 / 87.4 ms. Spread under 2% — much tighter
than the 15-20% older handoffs warn about; that spread was real for single-kernel GEMM benches, not
for the whole-model frame.


## Measured and REJECTED, 2026-08-29 — do not re-run these

Three structural levers were run on hardware, 3 repeats per arm, same session, same pkl. All three
are null or negative. They are listed here so nobody spends another day on them.

| lever | predicted | measured | verdict |
|---|---|---|---|
| `AMD_ACC_SEED=1` (seed the FMA chain from the accumulator) | 1.31x on kernel B, 1.06x on A | 90.4 ms vs control 88.7 ms (p50 of 3) | **no gain** |
| `BEAM_UPCAST_MAX` 64 -> 256 | 1.61x, from the 76.9% sibling's schedule | graph 65.5 -> **73.5 ms** | **worse** |
| `AMD_ELIDE_FLUSH=1` (drop 139 of 497 wave drains) | unknown | 89.3 ms vs 88.4 ms | **no gain**, but correct |

**The upcast result is the informative one.** The search worked (27 of 28 candidates timed; the
earlier `infs from 24 -> 0 actions` was the device already degrading, not the cap). It found genuinely
different schedules -- kernel B improved in isolation, 2258 -> 1957 us -- and the graph still got
slower. Cause: a 4x larger upcast cuts resident waves 1536 -> 384, and at IPC 0.93 there is no stall
to hide, so the occupancy loss beats the instruction-density gain. **The caps were protecting us.**
BEAM is at the ceiling of what its optimizer vocabulary can express for these shapes.

**Flush elision is correct** -- `model_output.py --against` reports bit-identical, and the graph is
self-consistent across replays -- it simply does not pay. Keep the patch; it costs nothing off and
the correctness evidence is banked if a future change makes overlap matter.

**Two methodology errors of mine, recorded so they are not repeated:**
- The first flush A/B set `DEBUG=2` on the elision arm only, to capture the drain count. DEBUG=2 adds
  per-dispatch timing work, so that comparison was invalid and had to be redone. Never vary anything
  but the variable.
- `AMD_ACC_SEED` looked like a clear win on `render_probe`'s proxy kernel (173 -> 155 instructions,
  every MAC fused) and did nothing on the real model. That is the **second** time the proxy pointed
  the wrong way. It answers "did my rewrite fire", never "is this faster".

**What this means for 20 Hz.** The graph is 65 ms and every scheduler lever in this stack is now
measured and exhausted. Transport is 16.0 ms serial with ~8.7 ms recoverable, which lands the frame
at ~80 ms = 12.5 Hz. 20 Hz needs the graph at ~35 ms. Nothing in tinygrad's optimizer reaches that,
so the remaining path is purpose-written kernels -- see the greenfield fork on branch
`rdna2-greenfield`. The existence proof still holds: `r_32_96_32_2_2_4_2_24_4_4` runs at 76.9% of
peak in this same model against the hot kernels' 40%.

## What 20 Hz actually costs

The model is **198.9 GFLOP/frame**, counted from the ONNX graph (99.4 GMAC: 136 MatMul, 44 Conv) by
`.elantra/model_flops.py`. That settled a 4.1x disagreement between two tinygrad estimates and
confirms the 202 GFLOP figure in the older handoffs to within 1.6%.

- 20 Hz = 198.9 x 20 = **3.98 TFLOPS sustained**
- card peak = 2048 lanes x 2 FLOP x 2.34 GHz = **9.58 TFLOPS** at the clock it actually holds
  (10.6 at its 2589 rated boost). That is the fp32 rate, which is what `v_fma_mix_f32` issues at;
  packed fp16 via `v_dot2c_f32_f16` would be 19.2.
- so 20 Hz needs **42% of peak**. We are at 3.08 TFLOPS = **32%**. The gap is **1.3x**.

For scale: a 1B-param LLM does ~2 GFLOP/token, so one frame here is ~100 tokens of arithmetic and
20 Hz is equivalent to ~2000 tok/s. This is not an LLM workload.

## Three corrections to the previous handoff

**1. The poisoned-row fix DID work.** `HANDOFF-egpu-codegen.md` says `r_8_16_1536_7_7` still ran at
workgroup 1 afterwards. It did not. `/root/profile.log` has mtime 17:06; the cache holding the fixed
schedule has mtime 17:09 — the profile predates the row it was supposed to reflect by three minutes.
Recompiling picks it up and the kernel becomes `r_8_1536_16_7_7` with `Opt(LOCAL, axis=1, arg=16)`;
the name changes because LOCAL splits the axis, which is also why it looked like it had vanished.
Measured **5273 us -> 435 us, 12.1x**. Graph 91.1 -> 64.6 ms. Sub-wavefront time 38.7% -> 6.5%.

**2. The kernels are issue-bound — not bandwidth-bound, not latency-bound.** Real loop bodies,
disassembled from the compile cache (not render_probe's proxy — see traps):

| kernel | loop body | MACs | static ceiling | measured | IPC |
|---|---|---|---|---|---|
| `r_24_32_32_2_4_384_4_4` | 153 instr | 64 | 41.8% | 39.0% | **0.93** |
| `r_24_8_16_2_8_4_4_384_4` | 140 instr | 64 | 45.7% | 40.8% | **0.89** |

Static instruction count predicts measured time within 7-11%: there is essentially no stall to
recover, and ~90% of the shortfall is non-MAC instructions occupying issue slots. DRAM traffic is
9 GB/s of 172 available; L0 traffic 934 GB/s of ~4.8 TB/s; VGPRs 33 and 60, both above the 16-wave
cap. **Every instruction removed from a loop body is a proportional speedup.**

**3. 80% of peak is not hypothetical.** `r_32_96_32_2_2_4_2_24_4_4`, in this same model, has the
same schedule template as kernel A, same 64 threads/wg, same CU mode — and runs at **76.9% of
peak**. The only structural difference is `UPCAST x UNROLL = 256` against A and B's 64.

## Where the frame goes, measured for the first time

Frame 92.3 ms instrumented (`USB_STATS=1` in `runtime/support/usb.py`; env-gated, below noise off):

| | per frame |
|---|---|
| total USB | 83.6 ms in 345 transactions |
| of which overlaps GPU compute (completion polling) | 67.5 ms |
| **serial USB, on the critical path** | **16.0 ms in ~64 transactions** |
| serial Python | ~1.1 ms |

**Every USB transaction costs ~0.24 ms regardless of payload.** A 4-byte and a 218-byte write cost
the same. Transaction count is the only thing that matters.

Ranked, measured, not yet implemented:

1. **Coalesce per-submit MMIO writes** — ring dwords, `write_ptr`, `_apply_var_vals` are separate
   writes that could be one. 19.2 -> ~8 writes/frame, **~5.2 ms**. Sites `ops_amd.py:436`,
   `ops_amd.py:696`, `support/hcq.py:222`. The doorbell must land after, so it stays separate.
2. **Signal stores off the per-byte path** — `USBIface.alloc` routes `host=True` into `sys_buf`
   (`pcimem=False`), so each 64-bit store is 8 `control_write(0xE5)` calls. **~3.3 ms.** Note the
   trade: moving signals to VRAM makes *reads* 2 transfers instead of 1, and there are 285
   reads/frame against 2 stores. Fix the write path; do not just relocate the buffer.
3. `HCQSignal.wait` reads the signal twice per iteration (`support/hcq.py:293` and `:295`). 34 ms of
   link time but only ~0.24 ms of frame time — the link is idle during compute. Low priority, and it
   is a correctness-sensitive loop.

`_apply_var_vals` mv_sints writes **zero** dwords/frame (kernarg pointers never change). Python is
not the problem: 0.72 ms in `HCQGraph.__call__`.

## Implemented, default OFF

Three of these have since been measured and rejected -- see the REJECTED section above. They
are kept because they cost nothing when off, the correctness evidence is banked, and a future
change may make one of them matter. `LLVM_ZEXT_INDEX=2` is the only one never run on hardware.

177 test cases green across configurations; `ruff check tinygrad/` clean.

| flag | what | static evidence |
|---|---|---|
| `AMD_ACC_SEED=1` | seeds the FMA chain from the accumulator | **173 -> 155 instr**, all 48 MACs fuse, no bare `v_mul_f32` |
| `AMD_ELIDE_FLUSH=1` | stops draining the GPU after all 497 dispatches | 84 cases, mutation-checked |
| `AMD_DOT2=1` | packed `v_dot2c_f32_f16`, now with free operand pairing | 173 -> 157, `v_alignbit` eliminated |
| `LLVM_ZEXT_INDEX=2` | the only form that selects SADDR | 189 instr vs 173, VALU flat at 134 |

**`AMD_ACC_SEED` is the highest-value one.** tinygrad renders an unrolled reduce as
`acc + (m0 + m1 + ... + mU)`, so the innermost node is `ADD(MUL, MUL)` with nothing to accumulate
into, and one MAC per accumulator per iteration cannot fuse. Cost is `2U+1` instructions per
iteration: 9 of 153 in kernel A (U=4), but **33 of 140 in kernel B (U=16), 43% of B's entire
shortfall**. Predicted 1.31x on B, 1.06x on A.

## The empty-schedule poisoning, root-caused and fixed

When every BEAM candidate fails to time — log line `infs from N -> 0 actions` — `beam` is still the
unoptimised Scheduler it was seeded with and `applied_opts` is empty. Caching that is strictly worse
than caching nothing: it is a cache HIT forever, `apply_opts` takes its `elif beam >= 1` branch and
never falls through to `hand_coded_optimizations` (an elif chained off it), and the kernel ships with
`local_size (1,1,1)` — one work item per workgroup.

**This reproduced live**: a wider-cap re-search poisoned two AMD rows in one run.

Fixed at three sites in `codegen/opt/search.py`: both `diskcache_put` calls go through
`_never_empty`, which falls back to `hand_coded_optimizations`; and the cache read requires
`len(val)` so rows written before the fix are treated as misses. `.elantra/test_beam_cache.py`,
6 cases, mutation-checked.

**Do NOT "fix" this by making `optimize_local_size` fire on (1,1,1).** It divides `global_size` by
the chosen local size, which is only sound for kernels the renderer left free (`local_size=None`,
marked by an `i` SPECIAL). A kernel with no LOCAL opt indexes purely off workgroup id, so doing that
would compute 1/32 of the outputs.

## Traps that cost real time

- **tinygrad embeds ANSI colour codes INSIDE kernel names.** A plain grep for a kernel in a DEBUG log
  returns 0 while the kernel is right there. Strip the escape sequences first. This is why two
  separate analyses concluded a kernel had disappeared.
- **`render_probe.py` compiles a fresh matmul with hand-coded opts.** That is NOT the model's kernel
  — the model's are BEAM-tuned, differently shaped, and the proxy has no loop at all. Use it for
  "did my rewrite fire", never for "is this faster".
- **Eager `*** AMD` per-dispatch times are ~3.7x the graphed frame.** Each eager dispatch pays a USB
  submit-and-wait the captured graph does not. The ranking is meaningful; the microseconds are not
  the frame.
- **`mem` vs `lds` in the DEBUG=2 line.** Format is `membw|ldsbw`. `mem` counts each distinct byte
  once (footprint); `lds` is actual load/store traffic. Reading `lds` as memory bandwidth turns an
  issue-bound kernel into a fictitious bandwidth-bound one. It did, for an hour.
- **`DEBUG=2` is not verbosity.** It populates `GlobalCounters.time_sum_s`. Without it
  `gemm_bench.py` prints `0.000 ms/GEMM` and `0 GFLOPS` while kernels still run and the accuracy
  gate still passes — a run that looks successful and measured nothing.
- **`PROFILE=1` hides the flush-elision win**: `prof_graph_entries` puts a `timestamp()` after every
  dispatch, re-serialising the queue exactly the way the flush did.
- **SIGKILL on a compile wedges the dock.** The killed process does not drop its libusb claim and the
  ASM2464 stops answering (`libusb_get_string_descriptor_ascii: Operation timed out`, then
  `Set Address Failed` on the Windows side). Neither `slot_cycle.py` nor `usbipd detach/attach`
  recovers it; it needs a physical replug. **Use SIGTERM and wait.**
- `wsl.exe -- bash -c` strips `$VAR` and flattens newlines; pipe scripts in on stdin instead. Git
  Bash also rewrites `/root/...` arguments into `C:/Program Files/Git/root/...` — set
  `MSYS_NO_PATHCONV=1`.

## The delivery gap, still open

`openpilot/selfdrive/modeld/SConscript:53` builds the big model with
`DEBUG=2 DEV=USB+AMD:LLVM FLOAT16=1 JIT_BATCH_SIZE=0 GMMU=0 TC_OPT=2` — **no `BEAM=`**, and
`TC_OPT=2` is a no-op on gfx10. The on-car build therefore gets `hand_coded_optimizations` and
compiles the 208 ms version however well the rig is tuned. Needs `BEAM=2` plus the tuned cache merged
(`harvest_cache.sh` already does `.backup` + `INSERT OR IGNORE`), and the flags must match exactly
between rig and car because `ast.key` is the cache key.

## Tools added

| file | what |
|---|---|
| `KERNEL_AUDIT=<path>` | One JSON line per compiled kernel: name, beam key, cache hit/miss, opts, resulting local_size. `beam_search_22` is keyed on an opaque 32-byte `ast.key` and the compile cache on IR text — without this join you cannot tell which cache row produced which kernel. It is what made a targeted re-search possible. |
| `.elantra/profile_report.py` | Joins a DEBUG=2 log to an audit: time-ranked kernels with workgroup size, schedule, bytes of load/store per FLOP. |
| `.elantra/model_flops.py` | Counts the model's arithmetic from the ONNX graph. |
| `.elantra/model_output.py` | Deterministic inputs, dumps output for bit-comparison. Required before timing any command-submission change. |
| `.elantra/test_beam_cache.py` | 6 cases for the empty-schedule guard. |
| `.elantra/test_flush_elision.py` | 84 cases for the drain schedule. |
| `.elantra/scripts/sync_tinygrad.sh` | repo -> WSL with a diff check. Two tinygrad trees exist and only `/root/src` executes. |
| `.elantra/scripts/compile_model.sh` | Compile with audit + full log kept, CACHEDB isolation, env recorded. |
| `.elantra/scripts/research_kernels.sh` | Re-search named kernels only by clearing their rows by ast_key. Minutes instead of hours; others keep known-good schedules. |
| `.elantra/scripts/ab_gemm.sh` | Repeated A/B with range reporting. |

## Next, in order

1. **Replug the dock.** Then `usbipd bind --busid 1-17` (elevated) and
   `usbipd attach --wsl --busid 1-17`.
2. `AMD_ACC_SEED=1`: compile, bench 3x, check `gemm_bench`'s err/rms gate and `model_output.py`
   bit-match, then flip the default in `llvmir.py` and the assertion in `test_amd_codegen.py`.
3. `AMD_ELIDE_FLUSH=1`: `model_output.py --against` FIRST (it removes what was hiding any dependency
   `DepsTracker` misses; a failure looks like slightly-wrong numbers, not a timeout), then time it.
4. The upcast cap with `BEAM_DEBUG=1` so candidate failures are visible. Three of eight kernels timed
   nothing at `BEAM_UPCAST_MAX=256` and the reason was never established — that is the open question
   between this model and the 76.9% its own sibling kernel achieves.
5. Transport coalescing (~8.7 ms).
