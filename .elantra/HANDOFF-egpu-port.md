# Handoff: finish the RDNA2 eGPU port and drive on the big model

Goal: run chestnut-class big driving models on an RX 6600 XT in the chestnut dock, in the car.
That needs tinygrad's AM driver ported to RDNA2, which does not exist. Stage it. The car must
stay drivable at every point — never leave the device in a state where a failed eGPU path can
stop it engaging.

Do not take this document or the memory files on trust. Verify against the code, and against
the device where it matters. Where you cannot verify something, say so plainly instead of
inferring it. If something here is wrong, say so directly — the last session found four things
the previous brief had wrong, and saying so was more useful than working around them.

## Where things are

- **Superproject**: `D:\Coding\sunnypilot-elantra`, branch `rdna2`, HEAD **`c6d974212`**.
  Remote `fork` = aaravsaianugula/openpilot. **Nothing is pushed** — both new commits are
  local only.
- **tinygrad**: `D:\Coding\sunnypilot-elantra\tinygrad_repo`, branch **`rdna2-am`**, HEAD
  **`e90344205`**, cut from `66ee3cfb`. Remote = aaravsaianugula/tinygrad. Not pushed.
- **The superproject's tinygrad gitlink is still `66ee3cfb` and must stay there** until stage 4.
  The port lives on a standalone branch on purpose (see "Where the port lives" below). Do not
  `git add tinygrad_repo` in the superproject.
- **opendbc**: pinned `69e2e548`. A worktree of that exact commit is handy for `guards.py`;
  `D:\Coding\opendbc-elantra` is on a *different* branch (`elantra-torque-test @ 01db6437`) —
  do not point guards at it.
- **Device**: comma 4 (mici), `comma@192.168.12.238`, offroad, on a bench with the dock and
  card attached and powered. AGNOS **19.6**, boot slot `_b`.
- **Device repo**: `/data/openpilot` at git HEAD `769f5cc0b` with the Phase 1 changes present
  as **uncommitted working-tree edits** (they were scp'd, not pulled). Content matches
  `c6d974212` except `.elantra/probe_rdna2.py`, which was only deployed to `/data/rdna2`.
- **Device scratch**: `/data/rdna2` (probe checkout), `/data/rdna2-tg` (tinygrad **with the
  stage 1 patch**, this is what to run the port against), `/data/rdna2-cache` (1.7 GB of
  cached ROCK/rocm tarballs — keep it, stage 2 needs them), `/data/rdna2-tmp`.
- **Device params set deliberately**: `EgpuDevice=0x73ff`, `EgpuVendor=amd`,
  `DisableUpdates=1`, `AlphaLongitudinalEnabled=1`.

### How to run anything on the device

AGNOS bakes the Python env into a venv at `/usr/local/venv`, activated by PATH. Bare
`/usr/bin/python3` has no capnp. Non-interactive SSH skips the profile, so:

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && export PATH=/usr/comma/shims:/usr/local/venv/bin:$PATH && export PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo && python3 ...'
```

For anything that regenerates autogen, also set `XDG_CACHE_HOME=/data/rdna2-cache` and
`TMPDIR=/data/rdna2-tmp`. `~/.cache` is a **100 MB** overlay and `/tmp` is a **150 MB** tmpfs;
both are far too small for a kernel tarball and you will get `No space left on device`.

Stop openpilot before anything that touches the card (`tmux kill-session -t comma`); the probe
and AM take an exclusive USB flock. Restart with `sudo systemctl restart comma`. That triggers
a full `build.py` run, so it takes a few minutes.

## Verified facts — do not re-derive, but re-check any you rely on

**The card.** RX 6600 XT, `1002:73ff` rev 0xc1. LTSSM 0x78. VRAM 8176 MB, 8192 MB BAR (ReBAR
on). Does not advertise FLR. PSP bootloader reports ready (`0x80000000` at
`MP0_SMN_C2PMSG_35`) — bring-up over the chestnut works. Probe verdict **GO**, exit 0.

**IP versions from the card's own discovery table:** GC 10.3.4, MP0 11.0.12, MP1 11.0.12,
SDMA0/1 5.2.4, NBIO/NBIF 3.3.2, OSSSYS 5.0.3, MMHUB 2.1.1, ATHUB 2.1.1, SMUIO 11.0.10,
THM 11.0.8, DF 3.7.4, UMC 8.8.0, CLK 11.0.5, HDP 5.0.3, PCIE 6.5.0, JPEG 3.0.16.

**GC 10.3.4 is Dimgrey Cavefish = Navi 23 = gfx1032.** Confirmed from the kernel's own table
(`drivers/gpu/drm/amd/amdkfd/kfd_device.c` at the pinned ROCK commit). The mapping is
non-monotonic — 10.3.0→gfx1030, 10.3.2→gfx1031, **10.3.4→gfx1032**, 10.3.1→gfx1033,
10.3.5→gfx1034, 10.3.3→gfx1035, 10.3.6/7→gfx1036. Digit-packing is wrong for five of seven and
fails silently with a plausible target for a different ASIC. Already fixed in `ops_amd.py`
(`GFX_TARGET_VERSION`) and pinned by a test.

**tinygrad does not fail on the arch assert.** `AMDev.__init__` calls `_build_regs()`
(`amdev.py:153`) before `ops_amd.py:953` is reached. The register build is what rejects the
card. `_build_regs` (`amdev.py:327`) requires **mp, hdp, gc, mmhub, osssys, nbio** — and
`sdma` only for SDMA 4.4.2/4.4.4, so RDNA2's 5.2.4 needs none.

**The USB link is 5 Gb/s and that is the ceiling.** Every root hub on the comma 4 reports
480/5000/480/5000; there is no 10000 Mb/s root. Not a cable problem. Budget against 5 Gb/s.

**`probe_once()` is blind on a cold boot.** A trained link is not an enumerated one — before
`USBPCIDevice` bring-up only the ASM2464 bridge answers (bus 0, `1b21:2464`) and the GPU is on
no candidate bus, so `probe.probe_ids()` returns `None` and `EgpuDeviceDetected` stays empty.
Every failure path in `probe_pci_ids()` returns `None` silently. `EgpuDevice=0x73ff` is set to
work around this. **This is an unfixed defect** — worth fixing if the port succeeds, because
the gate should not depend on a manually set param.

**Models are branch-independent.** `sunnypilot/models/fetcher.py` `MODEL_URL_USBGPU` points at
`driving_models_usbgpu_v22.json`, swapped in when chestnut is present. Every published bundle
is compiled for RDNA3/RDNA4 and **will not run on gfx1032** — stage 6 must compile on-device.

**comma's chestnut branches** are master plus one build commit carrying
`big_driving_tinygrad.pkl.chunk*`; every eGPU safety source file is byte-identical to master.
Nothing to merge. `Offroad_ChestnutBranch` can never fire here (branch not in
`CHESTNUT_BRANCHES`, and `big_model_available` is separately always True).

## What is already done

**Phase 1 — safety baseline. Complete, committed `9e53cf325`, verified on hardware.** Three
defects, each of which could stop the car engaging, each fixed with a test that fails without
the fix. Full write-up in `.elantra/EGPU_SAFETY.md`. Summary:

1. `SConscript:46` had no ASIC check and a failed eGPU compile returned non-zero, which
   `build.py:63` turns into a blocking TextWindow + `exit(1)`. Now gated on `egpu_build_ok()`
   and `do_compile` skips instead of failing the build.
2. `selfdrive/modeld/modeld.py:221` — the **default** runner — had no ASIC gate at all.
3. `modeld_v2` latched `UsbGpuLoading=True` on load failure (NO_ENTRY every frame *plus*
   suppressed `commIssue`/`posenetInvalid`). Now held by `guard.loading()` whose `finally`
   always releases, and it falls back to the on-SoC model at load and at runtime.

New module `openpilot/sunnypilot/egpu/guard.py` holds the logic (it is `OVERLAY_ADDED`, so no
sync conflict surface); the two upstream files take a one-line hook each and are registered in
`OVERLAY_MODIFIED` with hooks.

Verified with the dock in and the RDNA2 card attached: build A (gate fires, target never
declared) exit 0; build B (gate blinded, target attempted, compile fails, skip) exit 0; direct
`DEV=USB+AMD:LLVM` compile exit 1. No pkl or chunkmanifest produced, `usbgpu_compiled()` stays
False, `detect.enabled()` False, no tombstones.

**Phase 2 Stage 1 — registers and version mapping. Complete, committed `e90344205` (tinygrad)
and `c6d974212` (probe/test), proven on the card.**

Added to `reg_files`: gc 10.3.0, hdp 5.0.0, mmhub 2.0.0, osssys 5.0.0, nbio 2.3.0. Added
`soc_10` (from `navi10_enum.h` — gfx10 predates the soc21 naming) and `smu_11_0_7` (Sienna
Cichlid PPSMC; Navi 21-24 all resolve to it with no override). Added the `nbio` version
override and a `header_renames` map. Replaced digit-packing with `GFX_TARGET_VERSION`.

Stop condition met, measured against the card's own IP versions:

```
gc 10.3.4 -> 1271 regs   mp 11.0.12 -> 144   hdp 5.0.3 -> 1
mmhub 2.1.1 -> 384       osssys 5.0.3 -> 120  nbio 3.3.2 -> 7
smu 11.0.12 -> smu_11_0_7        import_soc((10,3,4)) -> soc_10
probe stage 7: all eight present, exit 0
```

Two things learned in stage 1 that change later stages:

- **`smu_11_0_7` carries `PPSMC_MSG_Mode1Reset`**, which is the branch `AM_SMU.mode1_reset()`
  already takes for MP0 11.x. **Stage 5 needs no invented bridge-level reset** — AM's existing
  path is correct once `smu_11_*` exists, which it now does. `PCIDevice.reset()` is
  `echo 1 > /sys/bus/pci/devices/<bus>/reset`, meaningless for a USB-bridged device, and AM
  never calls it.
- **`nbio` register *names* differ on RDNA2** and this adds scope to stage 3. `nbio_2_3`
  predates the `BIF_BX0_`/`BIF_BX_PF0_` prefixes and the `reg*` rename, so what AM calls
  `regBIF_BX0_PCIE_INDEX2` is `mmPCIE_INDEX2`, `regBIF_BX0_BIF_DOORBELL_INT_CNTL` is
  `mmBIF_DOORBELL_INT_CNTL`, `regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL` is
  `mmREMAP_HDP_MEM_FLUSH_CNTL`, and `regBIF_BX_PF0_GPU_HDP_FLUSH_REQ` is
  `mmBIF_BX_PF_GPU_HDP_FLUSH_REQ` (PF with no digit). `regBIF_BX_PF0_RSMU_*` does not exist at
  all — RSMU is MI-series only. The seven RDNA2 registers are now *available* in
  `nbio_2_3_0`, but AM's code still refers to them by their gfx11 names, so **stage 3 needs a
  name-mapping layer**.

## The work remaining

### Stage 2 — PSP

SOS load, TMR setup, ring creation for MP0 11.0.x. The bootloader already answers, so this is
the first genuine unknown.

**Firmware naming is the known trap, and it is worse than a rename.** AM builds blob names from
the discovery IP version (`fmt_ver`, `amdev.py:27`) — `psp_11_0_12_sos.bin`, `smu_11_0_12.bin`,
`sdma_5_2_4.bin`, `gc_10_3_4_{pfp,me,mec,rlc}.bin`. linux-firmware ships discrete Navi 2x under
**codenames**: `dimgrey_cavefish_{ce,me,mec,mec2,pfp,rlc,sdma,sos,ta,smc}.bin`. Every blob is
**SHA-256 pinned** against a generated `fw.hashes`, built by globbing
`psp_*_sos.bin`, `smu_*.bin`, `sdma_*.bin`, `gc_*_{pfp,me,mec,imu,rlc}.bin` — a glob that can
never match codename files. So this needs an IP-version → codename alias scheme **and** new
hash entries, not a rename. `fw.hashes` currently has `gc_10_3_6_*`/`gc_10_3_7_*` (the APUs),
no `psp_11_*`, no `smu_11_*`, no `sdma_5_2_2`.

Also check `ip.py:602`: `if GC >= (11,0,0): _rlc_autoload_cmd() else: _load_ip_fw_cmd(REG_LIST)`.
gfx10.3 takes the `else`, which is the gfx9/MI-style path — verify it against the kernel rather
than assuming.

**Stop condition:** SOS reports loaded and the PSP ring accepts a command.

### Stage 3 — SOC / GMC / IH / SMU

SMU 11 has a different message ABI from SMU 13. Plus the nbio name-mapping layer described
above. Also check `ip.py:18` (IH client enum splits at `GC >= 11`, so gfx10.3 falls to
`enum_soc15_ih_clientid` — verify that is right for Navi 2x) and `amdev.py:47-53`
(`GC < (11,0,0)` takes the MI P2S-table branch for SMU firmware, which is almost certainly
wrong for RDNA2).

**Stop condition:** clocks readable, memory controller configured, no fault storms.

### Stage 4 — GFX

The hardest part. `ip.py:396` uses RS64 for all `GC >= (10,0,0)`, conflating gfx10 with gfx11;
gfx10.3's MEC is an **F32** engine driven by `CP_MEC_CNTL`, not `CP_MEC_RS64_CNTL`. Behind
that, `amdev.py:66-81` only populates `ucode_start['MEC']` for header-v2 (RS64) firmware, so an
F32 header-v1 blob gives a `KeyError` at `ip.py:385`. Expect to need a third branch keyed on
`>= (11,0,0)` for RS64 with 10.x taking F32. IMU is correctly gfx11+ gated already.

This is also where the `ops_amd.py:953` arch assert finally gets widened to admit `(10,3)` —
**not before**. Widening it early trips `test_egpu_tinygrad.py`'s blocklist tripwire, which is
working as designed.

**Stop condition:** `DEV=USB+AMD python3 -c "from tinygrad import Tensor; print(Tensor([1,2,3]).sum().item())"` prints `6`.

### Stage 5 — SDMA 5.2 and teardown

SDMA 5.2.x already resolves to `sdma_5_0_0` correctly. Check `ip.py:514` — the `<= (5,2,0)`
guard skips `utc_l1_enable` for 5.2.4, verify that is intended. Recovery is
`smu.mode1_reset()`, which stage 1 already enabled.

**Stop condition:** `kill -9` mid-run, then a clean re-open with no physical power cycle. A
path that needs a power cycle to recover is a stop-ship for in-car use.

### Stage 6 — the model

Published bundles are gfx12, so the pickle must be compiled on-device for gfx1032. Generalise
`openpilot/sunnypilot/egpu/compile_nv.py` rather than copying it. The `.egpu` marker currently
records a **vendor only** — it must also record the gfx target, and
`models.assert_pkl_matches` must check it, or a gfx12 pickle can still reach a gfx1032 card.
Note `assert_pkl_matches` returns early when `usbgpu` is False (`models.py:56`) and is wired
only into `modeld_v2`.

Compilation is heavy: watch `/data` free space (currently ~11 GB) and remember `taskset -c 7`
fails from an SSH session (`Invalid argument`) but works from the boot build, which calls
`os.sched_setaffinity(0, range(8))` first.

### Stage 7 — numerical validation. Do not skip.

A model that loads proves nothing. tinygrad #11705 documented openpilot outputs on some
renderers being silently wrong — max abs diff >50,000, no error raised. Validate `_copyout`
first (known pattern, round-trip, byte-exact, ×10,000), then compare against onnxruntime
CPUExecutionProvider on the same ONNX over **≥1000 consecutive frames from a real segment of
the car**. Report the **max**, not the mean. The model is fp16, so the default `rtol=1e-5`
fails even on known-good backends: **pass ≤ 0.2, fail > 1.0**.

### Stage 8 — power, before the car

The dock carries an INA231 (vendor `IN 0xC0`, `wLength 5`, `<H h B>` = mV, mA, fault). Measure
idle, then the model at 20 Hz for 10 minutes, then a saturating GEMM, then cold start. The
6600 XT is 160 W TBP on one 8-pin, so this should be comfortable — measure rather than assume.
Any `bob_flt` assertion is an immediate fail.

### Phase 3 — on road

Only after everything above passes. Re-run the fallback matrix in `.elantra/EGPU_SAFETY.md`
section 4 with the eGPU now working — rows 2-6 are currently code-traced or unit-tested only,
and rows 7 (dock unplugged mid-drive) and 8 (eGPU faults after the model was driving) have
never been exercised. A mid-drive eGPU failure must still degrade to the on-board model
cleanly. Then in-vehicle validation against segments from the car. Give a written go/no-go.

## Ground rules

- **The RDNA2 gate in `openpilot/sunnypilot/egpu/asics.py` stays until the port actually
  works.** Remove entries only when a card genuinely runs, never to unblock testing.
  `test_egpu_tinygrad.py` enforces the ordering: registers may land ahead of the arch assert,
  never behind it.
- No mock data, stubbed returns, or "demo mode" anywhere in a driving path.
- Tests first, and never weaken a test to get green.
- Anything under `opendbc/safety/` or touching panda's allow-list gets extra scrutiny. Safety
  now lives in `opendbc/safety/modes/hyundai*.h`, **not** `panda/board/safety/`.
- `safetyParam` for `HYUNDAI_ELANTRA_2024` on opendbc `69e2e548` is **8** without
  `alpha_long` and **12** with it. `AlphaLongitudinalEnabled=1` on this device, so it runs
  **12** = `LONG|CAMERA_SCC`, matching the C constants. There is no `DYNAMIC_LIMITS` bit at
  this pin (max flag is `ALT_LIMITS_2 = 512`).
- Keep the port on `rdna2-am`. After stage 4, add a **separate** `AM_DELTA_PATHS` +
  `AM_SENTINELS` registry in `.elantra/overlay.py` alongside the NV one — do not widen
  `NV_DELTA_PATHS`, which is deliberately restricted to five NV files and whose value comes
  from being small. `sync.py:346` enforces the path allowlist; the "exactly one non-merge
  commit" rule is a comment invariant in `overlay.py:20`, not enforced in code.

## Known-good gate commands

```bash
python .elantra/test_egpu.py                                   # 178/178
python .elantra/test_overlay_registration.py                   # 8/8
python .elantra/test_port_panel.py
python .elantra/test_egpu_tinygrad.py --tinygrad tinygrad_repo  # 34 pass, 5 expected NV fails
python .elantra/guards.py --opendbc <69e2e548 worktree> --repo . --tinygrad tinygrad_repo
ruff check .elantra/ openpilot/sunnypilot/egpu/
```

The five NV sentinel failures are **expected and correct**: the tinygrad pin is stock
`sunnypilot/tinygrad @ 66ee3cfb`, which does not carry the NV-USB delta. They are the
detector's negative control.

Probe, against the patched tinygrad:

```bash
ssh comma@192.168.12.238 'cd /data/rdna2 && PYTHONPATH=/data/rdna2-tg XDG_CACHE_HOME=/data/rdna2-cache TMPDIR=/data/rdna2-tmp /usr/local/venv/bin/python3 .elantra/probe_rdna2.py --expect 0x73ff'
```

## Loose ends to deal with

- **`DisableUpdates=1` is still set on the device** and must be reverted before daily driving.
  It was set so the updater could not swap the tree mid-work. `UpdaterTargetBranch` is
  `CLEAR_ON_MANAGER_START` and falls back to the checked-out branch, so it needs nothing.
- **Nothing is pushed.** Both superproject commits and the tinygrad branch are local only.
- **The device's `/data/openpilot` has Phase 1 as uncommitted edits** at an older HEAD, and its
  `.elantra/probe_rdna2.py` is the *old* one. Decide whether to pull properly or keep scp'ing.
- **Seven leftover directories** at the device repo root from rdf5's older layout — `cereal/`,
  `selfdrive/`, `sunnypilot/`, `common/`, `system/camerad/`, `system/loggerd/`,
  `tinygrad_rdf_repo/`, ~451 MB. Verified inert (the tracked tree has zero bare imports, and
  every SConscript path is explicit `openpilot/...`). Delete only with the owner's say-so.
- **Harness defects recorded but not fixed**: `guards.py:118` counts `hyundai_k` anywhere in
  stock `values.py` and passes with the port removed; `guards.py:191` silently skips five
  manifest checks when the manifest is absent; the shipped `build-manifest.json` has no `egpu`
  key so `guards.py:204` **and** `verify_published.py:79` skip tinygrad verification entirely;
  CI never passes `--tinygrad`; `probe_rdna2.py`'s `_failed` global is written but never read.
