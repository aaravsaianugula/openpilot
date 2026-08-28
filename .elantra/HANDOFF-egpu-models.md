# Handoff: get a chestnut-class big model driving on the RX 6600 XT

Goal: BMRLNAP Model v3 (or the best big model that can actually run) on the RX 6600 XT in the
chestnut dock, in the car. The RDNA2 driver port is **done and working**. What remains is the
model, and one open question that decides whether BMRLNAP v3 specifically is possible at all.

Do not take this document on trust. Verify against the code and the device. Where you cannot
verify something, say so rather than inferring it. If something here is wrong, say so directly.

## Where things are

- **Superproject**: `D:\Coding\sunnypilot-elantra`, branch `rdna2`, HEAD **`90552919c`**, pushed to
  `fork` = aaravsaianugula/openpilot.
- **tinygrad**: `tinygrad_repo`, branch **`rdna2-am`**, HEAD **`7178c0b4b`**, pushed to
  `origin` = aaravsaianugula/tinygrad.
- **The superproject's tinygrad gitlink is still `66ee3cfb`** and has stayed there all along.
  Moving it is a deliberate act; nothing so far has needed it.
- **Device**: comma 4, `comma@192.168.12.238`, AGNOS 19.6, dock + 6600 XT attached, openpilot
  running. `DisableUpdates` has been **cleared** — the device updates normally again.
- **Device git**: `/data/openpilot` on branch `rdna2` at `769f5cc0b` with the Phase 1 changes as
  uncommitted working-tree edits. `origin` is the same fork, so `origin/rdna2` is now well ahead.
  A `git stash` + `git merge --ff-only origin/rdna2` brings it current; I verified a reset would
  discard nothing unique (4 of 5 dirty files already match origin, the 5th is older than origin's).

### Running things on the device

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && export PATH=/usr/comma/shims:/usr/local/venv/bin:$PATH && export PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo && python3 ...'
```

- **`scp` is flaky on this link.** Use `ssh comma@... 'cat > /path' < localfile` and verify with
  `md5sum`. It has never failed.
- Anything touching the card needs openpilot stopped: `sudo systemctl stop comma`. Restart with
  `sudo systemctl restart comma`.
- **Never `pkill -f` a pattern that appears in your own SSH command line** — it kills your session.
- **Never `import tinygrad.runtime.ops_amd` on the device** — it tears the SSH session down. The
  probe reads `GFX_TARGET_VERSION` with `ast` + `importlib.util.find_spec().origin` for this reason.
- Long jobs: `nohup sh -c "... ; echo \$? > /path/x.done"` and poll for the `.done` file.
- `/data/rdna2-tg` is the patched tinygrad the port runs against. **openpilot's own
  `/data/openpilot/tinygrad_repo` is stock `66ee3cfb` with no RDNA2 support**, so anything that must
  use the port needs `PYTHONPATH=/data/rdna2-tg` **first**.
- **`/data/rdna2-tmp/recover.py` power-cycles the card** through the dock's own firmware
  (`CustomASM24Controller.set_pcie_power`, control request `0xF3`). It brings a wedged card back
  with no physical intervention: LTSSM `0x78` → `0x00` → `0x78` in 0.5 s. Run it before every
  bring-up. `/data/rdna2-tmp/cycle.sh` does recover-then-bring-up in one shot.

## Verified facts — do not re-derive

**The port works.** All seven IP blocks come up on the 6600 XT over the dock: `AM_SOC`, `AM_GMC`,
`AM_IH`, `AM_PSP`, `AM_SMU`, `AM_GFX`, `AM_SDMA`, then `boot done`. And it runs kernels:

```
sum -> 6
matmul -> 262144.0      # 64x64 ones matmul, = 64^3 exactly
```

`/data/rdna2-tmp/kernel_test.sh` reproduces this. `ops_amd.py` now admits gfx10.3.

**Clock control is unavailable on this card, and that is not a bug you can fix cheaply.** SMU 11
will not release DPM tables until a PPTable has been uploaded, which needs VBIOS atom-table
parsing AM does not have. The card runs at the SMU's default clocks and says so once per boot.
This costs performance, not correctness. **Measure it before assuming it is fine** — the model has
a 20 Hz budget and nobody has timed a frame on this card yet.

**The RDNA2 entries stay in `openpilot/sunnypilot/egpu/asics.py`.** A kernel returning 6 proves the
driver, not that a 1.76 GB model produces correct output. `test_egpu_tinygrad.py` now enforces
this: blocking an architecture AM can drive is allowed but must be named in
`BLOCKED_PENDING_VALIDATION` with a reason. Emptying that map is what retires the entry, and
nothing should empty it before `validate_numerics.py` has passed on this card.

## The model problem, and the one open question

**The published chestnut bundles cannot run on this card as published.** Established by evidence,
not inference:

- `driving_models_usbgpu_v22.json` has 8 bundles; newest is **BMRLNAP Model v3 (August 26, 2026)**,
  `is_big: true`, `runner: tinygrad`, 38 chunks, 1.76 GB.
- Its embedded binaries contain **88 ELF objects targeting `gfx1200`** (RDNA4) and its embedded
  LLVM IR contains **224 `llvm.amdgcn.wmma` intrinsics**.
- **WMMA is a matrix instruction introduced with RDNA3.** gfx1032 has none. A compile against it
  fails with `LLVM ERROR: Cannot select: intrinsic llvm.amdgcn.wmma.f32.16x16x16.f16` — I hit
  exactly this, which is why `tc.get_amd()` now returns no tensor cores for `gfx10*`.
- **No ONNX for BMRLNAP v3 is published anywhere.** Checked `models/defaults`,
  `models/recompiled19` … `recompiled22` on the HuggingFace dataset, and the on-SoC
  `driving_models_v21.json` (77 bundles, zero BMRLNAP).

### THE OPEN QUESTION — start here

tinygrad's `CapturedJit.__reduce__` (`tinygrad/engine/jit.py:174`) pickles
`(ret, _linear, expected_names, expected_input_info)`, and **`_linear` is a UOp graph**, not just
compiled binaries. `pm_compile` (`engine/realize.py:250-260`) rewrites a `CALL` whose src is an
`Ops.SINK` (an AST) into one whose src is an `Ops.PROGRAM` (a compiled kernel) — and the patterns
handle **both** forms.

So the question that decides everything:

> In the published BMRLNAP v3 pickle, do the `CALL` nodes carry `Ops.SINK` (a recompilable AST) or
> only `Ops.PROGRAM` (a baked gfx1200 binary)?

- **If SINK survives**: re-run `compile_linear()` on the stored graph with the device set to
  `USB+AMD:LLVM` and tensor cores off. BMRLNAP v3 then runs on gfx1032, and the whole problem is
  solved. This is the single highest-value experiment left.
- **If only PROGRAM**: the kernel choice — including WMMA — is baked in, and BMRLNAP v3 is
  genuinely impossible on RDNA2 without its ONNX. Say so plainly and stop; do not fake it.

**How to test it.** Download all 38 chunks to the device (watch space — `/data` is at 92%, ~7 GB
free; the compile below also writes there), reassemble, then load with tinygrad from
`/data/rdna2-tg` and walk `jit._linear.toposort()` counting `Ops.SINK` vs `Ops.PROGRAM` on the
`CALL` nodes. Do **not** try to execute it first — inspect the graph. That is a cheap, decisive
answer and it needs no card.

Chunks live at:
`https://huggingface.co/datasets/sunnypilot/sunnypilot_models_v1/resolve/main/models/recompiled22/model-BMRLNAP%20Model%20v3%20%28August%2026%2C%202026%29-406/driving_bmrlnap_model_v3_august_26_2026_tinygrad.pkl.chunkNNof38`

### The fallback that is already in flight

`big_driving_supercombo.onnx` (1,757,355,221 bytes, a real file) is on the device, and a compile of
it for gfx1032 was **running when this handoff was written**:

```
/data/rdna2-tmp/compile_big.sh          # log: /data/rdna2-tmp/compile.log, rc: compile.done
```

It uses SConscript's `usbgpu_tg_flags` token for token, with `PYTHONPATH=/data/rdna2-tg` first.
Check `compile.done` first thing. If it produced
`/data/rdna2-tmp/big_driving_tinygrad_gfx1032.pkl` (or its `.chunk01ofNN` — `chunk_file` deletes
the original, see the blocking defect below), that is a chestnut-class big model that runs on this
card, just not BMRLNAP v3.

## Blocking defects found by review, not yet fixed

Four agents wrote the stage 6/7/8 tooling and four reviewers tore into it. Every one came back
`fix-first`. These are real and each was demonstrated, not guessed:

1. **`validate_numerics.py` reports NaN as a perfect pass.** `d = float(np.max(np.abs(a-b)))` is
   NaN when the eGPU emits NaN, and `NaN > max_diff` is False, so `max_diff` never leaves 0.0 and
   the tool prints `PASS  max abs diff 0 <= 0.2` and exits 0. **An all-NaN eGPU passes
   validation.** This is the single most important thing to fix in the tree — it is the exact
   failure the tool exists to catch. Fix: treat any non-finite output as an immediate hard fail.
2. **`validate_numerics.py` stage 2 never runs on the eGPU.** `OnnxRunner` is built outside the
   `Context(DEV=device)` block, so initializers land on `Device.DEFAULT` and it raises
   `all buffers must be on the same device` before the first comparison.
3. **`compile_egpu.py` fails on every successful compile.** `compile_modeld.py` ends with
   `chunk_file()`, which `os.remove()`s the output; the `if not Path(output).is_file()` check then
   fires on success and refuses to write the marker. Check for the chunks, not the file.
4. **`measure_power.py`**: a failed reconnect leaves `handle=None` forever (every later read
   short-circuits), and a run with zero samples but non-empty gaps exits 3 while printing
   "The samples are real and the CSV is good".
5. **`fallback_matrix.py`**: rows 3 and 4 share one induction so row 4 goes green off row 3; row 6's
   assertion reads *every* noEntry event, so `wrongGear`/`doorOpen` satisfy it; row 5 can never
   pass; row 8's trigger and assertion are the same signal so it cannot fail.

### From the code audit — latent, and they go live when the gate opens

A five-agent audit of every eGPU path found **no blocking defect in the drivability path**: all
three Phase 1 fixes are intact and were re-verified on the device (build gate fires, eGPU target
never declared, `scons -n` exits 0, both runners compute `USBGPU=False`, no latch). But four
things are waiting for the moment `asics.py` stops blocking this card:

- **`SConscript:97-113` `do_compile` guards `env.Execute`'s return value but not exceptions.** A
  raise inside a SCons Python action gives exit 2, which `build.py:55-64` turns into a blocking
  TextWindow and `exit(1)` — the boot failure Phase 1 was written to prevent. Reachable via
  `link_up()` (`flash.py:110-116` only catches OSError/RuntimeError, so a ValueError from a
  malformed sysfs read escapes) or `chunk_file()` on ENOSPC. **`/data` is at 92% with ~7 GB free
  and the big pkl plus chunks needs ~4.2 GB.** Fix: one try/except around do_compile's body.
- **`SConscript:86` runs `get_existing_chunks()` in the read phase**, outside any guard. It raises
  `FileNotFoundError` when the ONNX and its manifest are both absent, and a read-phase raise fails
  scons before any skip logic exists. This is the "missing pkl" row of the fallback matrix, and it
  is the one case where "no eGPU outcome can fail the build" does not hold.
- **`test_egpu.py:1054` overstates itself.** The check named "do_compile can only skip, never fail
  the build" asserts only that every `return` is `None`. It says nothing about exceptions, which
  is exactly why the hole above is invisible.
- **`models.py:105-113` `read_marker` catches OSError but not UnicodeDecodeError.** A `.egpu`
  marker with invalid UTF-8 propagates through `detect.enabled()` into both runners *before*
  params or `UsbGpuLoading` are touched, so modeld dies at start.

Also: `onnxruntime is not installed on the comma 4` (verified — `import onnxruntime` →
ModuleNotFoundError, `pip list | grep onnx` empty). Stage 7 cannot produce a number until that is
solved. That is a hard blocker on validation, and validation is what retires the blocklist.

## Order of work

1. Read `/data/rdna2-tmp/compile.done`. If the gfx1032 compile succeeded, you have a big model.
2. **Answer the SINK-vs-PROGRAM question.** It decides whether BMRLNAP v3 is possible.
3. Fix defect 1 (NaN passes) before anything is validated. Then 2 and 3.
4. Get onnxruntime onto the device, or find another reference implementation, and run
   `validate_numerics.py` over ≥1000 consecutive frames from a real route under
   `/data/media/0/realdata`. Report the **max**, not the mean. Pass ≤ 0.2, fail > 1.0.
5. Time a frame. The model runs at 20 Hz and this card has no clock control.
6. Only then consider retiring the `asics.py` entry, by emptying `BLOCKED_PENDING_VALIDATION`.

## Ground rules

- No mock data, stubbed returns, or "demo mode" in a driving path. A blocked task reported
  honestly beats a fake one.
- Tests first, and never weaken a test to get green.
- The car must stay drivable at every point. Phase 1 guarantees this and nothing may weaken it:
  the build gate skips instead of failing, both model runners check the blocklist, and
  `guard.loading()` always releases `UsbGpuLoading`.
- Keep the port on `rdna2-am`. After the gitlink moves, register a separate `AM_DELTA_PATHS` +
  `AM_SENTINELS` in `.elantra/overlay.py` — do not widen `NV_DELTA_PATHS`.
- Anything under `opendbc/safety/` gets extra scrutiny.

## Known-good gates

```bash
python .elantra/test_egpu.py                                    # 246/246
python .elantra/test_overlay_registration.py                    # 8/8
python .elantra/test_egpu_tinygrad.py --tinygrad tinygrad_repo   # 45 pass, 5 expected NV fails
ruff check .elantra/ openpilot/sunnypilot/egpu/
```

The five NV sentinel failures are the negative control: the tinygrad pin is stock
`sunnypilot/tinygrad @ 66ee3cfb`, which does not carry the NV-USB delta.

On the device:

```bash
ssh comma@192.168.12.238 'cd /data/rdna2 && PYTHONPATH=/data/rdna2-tg XDG_CACHE_HOME=/data/rdna2-cache TMPDIR=/data/rdna2-tmp /usr/local/venv/bin/python3 .elantra/probe_rdna2.py --expect 0x73ff --strict'
```

## The thing worth saying out loud

The 6600 XT is the wrong card for the published chestnut models. The dock is built for RDNA3/RDNA4,
and on such a card BMRLNAP v3 would simply run — no driver port, no on-device compile, none of
this work needed. The port is real and it works, but if the goal is "drive on the newest published
big model soon", swapping the card is a shorter path than anything in this document.
