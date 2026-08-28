# Handoff: finish the eGPU — power, validation, and the published models

Three jobs, in this order. The first is blocking and is not a software problem.

Do not take this document on trust. Verify against the code and the device. Where you cannot
verify something, say so rather than inferring it. If something here is wrong, say so directly.

## 0. THE CARD HAS NO POWER — fix this before touching any code

Measured on the dock's own INA231 (the same `0xC0` / wLength-5 read `modeld.py:122` uses):

```
   1.583 V     0.000 A     0.00 W   fault=0x01     <- should be ~12 V
   1.583 V     0.001 A     0.00 W   fault=0x01
   1.583 V     0.000 A     0.00 W   fault=0x01
```

1.58 V at **zero current** is a floating rail, and the fault byte is set on every sample.
Consequently `LTSSM = 0x00` and the PCIe link will not retrain, before or after
`set_pcie_power(False)` → `set_pcie_power(True)`. The dock itself is healthy: it enumerates over
USB as `3801:0001` at 5 Gb/s with `custom ed4e39b7-CLEAN` firmware.

This card **worked on the bench earlier the same day** — `LTSSM 0x78`, full driver bring-up,
kernels running, a 1.77 GB model compiled against it. It lost power when the dock was moved into
the car.

Check, in this order: the GPU's own 8-pin PCIe power connector; the dock's 12 V feed; whether the
card is seated. Zero current points at the supply side rather than the card.

**Do not debug software until `LTSSM` reads `0x78`.** Confirm with:

```bash
ssh comma@<ip> 'PYTHONPATH=/data/rdna2-tg /usr/local/venv/bin/python3 -u /data/rdna2-tmp/supply.py'
```

That prints volts/amps/fault and the link state, and re-attempts a slot power cycle.

This probably also explains a reboot the comma did on its own while parked with the dock
attached, which came back clean with no crash logs. A browning-out 12 V rail fits.

## 1. Validate the compiled model — everything is staged

The moment the card links, this is ~30 minutes of runtime. Already done for you:

- `/data/rdna2-tmp/big_driving_tinygrad_gfx1032.pkl` — **1,773,925,673 bytes**, compiled on this
  card from the device's own `big_driving_supercombo.onnx`. Verified present after a reboot.
- **onnxruntime 1.29.0** installed at `/data/rdna2-tmp/pypkgs` (CPUExecutionProvider), deliberately
  out-of-tree so AGNOS is untouched.
- `.elantra/validate_numerics.py` deployed to `/data/rdna2/.elantra/`, with both known bugs fixed:
  non-finite output now hard-fails instead of reporting a perfect pass, and `OnnxRunner` is built
  inside `Context(DEV=...)` so its initializers land on the device under test.
- `/data/rdna2-tmp/validate.sh` sets the whole environment (`/data/rdna2-tg` **first** on
  PYTHONPATH — openpilot's own `tinygrad_repo` is stock `66ee3cfb` with no RDNA2 support).
- Route **`000000b2--546a2ed1b3`** has full `fcamera.hevc` + `ecamera.hevc` + `rlog.zst` per
  segment, ~1200 frames each. 531 segments on the device.

Run it:

```bash
ssh comma@<ip> 'sudo systemctl stop comma'
ssh comma@<ip> '/usr/local/venv/bin/python3 -u /data/rdna2-tmp/recover.py'
ssh comma@<ip> '/data/rdna2-tmp/validate.sh --stage copyout --device AMD'
ssh comma@<ip> '/data/rdna2-tmp/validate.sh --stage model --device AMD --route 000000b2--546a2ed1b3/2:5 --frames 1000'
```

Stage 1 is a byte-exact `_copyout` round trip ×10,000 — it catches a broken transfer path before
you spend an hour comparing model outputs. Stage 2 reports the **max** abs difference per output
tensor. **Pass ≤ 0.2, fail > 1.0.** The model is fp16, which is why the thresholds are not 1e-5.

Then **time a frame**. This card has no clock control (SMU 11 will not release DPM tables without
a PPTable upload, which needs VBIOS atom parsing AM does not have), so it runs at the SMU's
default clocks and says so once per boot. Whether it holds 20 Hz is genuinely unknown and nobody
has measured it. Measure it.

## 2. Turn it on — only after 1 passes

Four things, none of them subtle:

1. Remove `0x73FF` from `UNSUPPORTED_AMD` in `openpilot/sunnypilot/egpu/asics.py`, and empty
   `BLOCKED_PENDING_VALIDATION` in `.elantra/test_egpu_tinygrad.py` — the test enforces that those
   two move together, which is the point.
2. Put the pkl where `modeld_pkl_path(True)` points, chunked (`usbgpu_compiled()` checks for the
   chunkmanifest, and `chunk_file()` deletes the original — see the compile_egpu defect below).
3. Write the marker beside it: `vendor=amd` and `target=gfx1032`, via `models.write_marker`.
4. Fix the stale panel string at
   `openpilot/selfdrive/ui/sunnypilot/mici/layouts/egpu_state.py:52-55`. It currently tells the
   driver "tinygrad's driverless AMD driver supports RDNA3 and RDNA4 only", which stopped being
   true the day RDNA2 was ported. Say what is actually gating it.

Known defect that will bite in step 2: `compile_egpu.py:164` checks `if not Path(output).is_file()`
after a compile, but `compile_modeld.py` ends with `chunk_file()`, which `os.remove()`s the output.
So a *successful* compile looks like a failure and no marker is written. Check for the chunks.

## 3. The published driving models — what is actually true

**They are compiled machine code for gfx1200 (RDNA4), not portable graphs.** Established by
loading BMRLNAP Model v3 itself (sha256 `502bd18e…`, matches the catalog), decoding openpilot's
container (`helpers.py:39 load_oob`: int64 length + pickle opcodes + protocol-5 out-of-band weight
buffers — the graph is only ~5.8 MB of 1.7 GB), and walking it:

| | |
|---|---|
| AMD kernels in `run_policy` | 88, target `USB+AMD:LLVM:gfx1200` |
| every PROGRAM retains | `src = [SINK, LINEAR, SOURCE, BINARY]` — the AST survives |
| WMMA-free, **would** re-render for gfx1032 | **70 of 88** |
| WMMA baked into both retained ASTs | **18 of 88** |
| of those 18, any WMMA-free AST retained | **0** |

`do_to_program()` (`codegen/__init__.py:456`) accepts an `Ops.PROGRAM` back as input and
re-renders it for whatever renderer it is given, and `to_program` caches on `renderer.target`, so
a different target genuinely recompiles. **70 of the 88 kernels would move to gfx1032 today.**

The other 18 are matmul kernels whose retained ASTs already contain `Ops.WMMA`. RDNA2 has no matrix
unit, and tinygrad has no WMMA→scalar lowering: every handler in `renderer/llvmir.py:279-315` and
`renderer/ptx.py:133` maps `Ops.WMMA` straight onto a hardware matrix instruction. 70/88 is not a
model.

### The three real options

**A — write the WMMA→scalar lowering.** This is the path that makes published models work on this
card, and it is ordinary engineering, not magic:

- Add a `PatternMatcher` rule rewriting `Ops.WMMA(a, b, c)` into an explicit multiply-accumulate
  over the tile, and register it in `AMDLLVMRenderer` when `self.tensor_cores` is empty.
- The shape you must unwind is the `TensorCore` in `codegen/opt/tc.py` — `amd_rdna4` has
  `dims=(16,16,16)`, `threads=32`, `elements_per_thread`, `opts` and `swizzle`. The swizzle is the
  hard part: it describes which lane holds which element, and getting it wrong produces
  *plausible* wrong numbers, which is the exact failure stage 7 exists to catch.
- `expand_wmma` (`codegen/__init__.py:77`) shows how the axes are contracted/unrolled today and is
  the right thing to read first.
- Expect it to be slow — WMMA exists because scalar matmul is slow. Time it before celebrating.
- Validate it against the reference with `validate_numerics.py` before it drives anything.

**B — put an RDNA3 or RDNA4 card in the dock.** Published models then run as published, with none
of this. This is a shorter path to "newest model driving" than A by a wide margin.

**C — keep compiling on-device from `big_driving_supercombo.onnx`**, which already works and
produced the 1.77 GB pkl. You get a chestnut-class big model on this card; you do not get the
specific named bundles.

Pick deliberately. A is real work with a real risk of being too slow to use; B is a card swap.

## Where things are

- superproject `D:\Coding\sunnypilot-elantra`, branch `rdna2`, pushed to `fork`
  = aaravsaianugula/openpilot
- tinygrad `tinygrad_repo`, branch `rdna2-am`, pushed to `origin` = aaravsaianugula/tinygrad
- the superproject's tinygrad gitlink is still `66ee3cfb` and has never moved
- device `/data/openpilot` on `rdna2` at `769f5cc0b` with Phase 1 as uncommitted edits; `origin`
  is the same fork and is well ahead. A `git stash` + `git merge --ff-only origin/rdna2` brings it
  current and discards nothing unique (verified).
- `DisableUpdates` is cleared; the device updates normally

### Device recipe

```bash
ssh comma@<ip> 'cd /data/openpilot && export PATH=/usr/comma/shims:/usr/local/venv/bin:$PATH && export PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo && python3 ...'
```

- **`scp` is flaky.** Use `ssh comma@... 'cat > /path' < localfile`, verify with `md5sum`.
- Stop openpilot before touching the card: `sudo systemctl stop comma`. **Restart it when you
  finish** — leaving it stopped leaves the car on the boot logo.
- **Never `pkill -f` a pattern that appears in your own SSH command line** — it kills your session.
- **Never `import tinygrad.runtime.ops_amd` on the device** — it tears the session down.
- Long jobs: `nohup sh -c "... ; echo \$? > /path/x.done"`, poll for the `.done`.
- `/data/rdna2-tmp/cycle.sh` = recover-then-bring-up. `/data/rdna2-tmp/kernel_test.sh` reproduces
  `sum -> 6` and a 64×64 ones matmul → `262144`.

## What is already done — do not redo it

The RDNA2 driver port works. All seven IP blocks initialize on the 6600 XT over the dock
(`AM_SOC`, `AM_GMC`, `AM_IH`, `AM_PSP`, `AM_SMU`, `AM_GFX`, `AM_SDMA`, then `boot done`) and it
runs correct kernels. `ops_amd.py` admits gfx10.3. Gates: `test_egpu.py` 246/246,
`test_overlay_registration.py` 8/8, `test_egpu_tinygrad.py` 45 pass + 5 expected NV negative
controls, ruff clean on both trees.

## Ground rules

- No mock data, stubs, or demo mode in a driving path. A blocked task reported honestly beats a
  fake one.
- Tests first, never weakened.
- The car must stay drivable at every point. Phase 1 guarantees it: the build gate skips instead
  of failing, both runners check the blocklist, `guard.loading()` always releases `UsbGpuLoading`.
- Nothing unvalidated drives. The blocklist comes off when `validate_numerics.py` passes on this
  card, and not before.
