# eGPU safety inventory

What exists to keep an eGPU failure from taking the car with it, whether each piece is
actually reachable on this branch, and what happens in each failure case.

Everything below was checked against the tree at `rdna2`, and against the device (comma 4,
mici, AGNOS 19.6) where the answer depends on runtime state. Claims that could not be
exercised say so rather than being inferred.

---

## 1. The two model runners are not equivalent

This is the fact the rest of the document turns on. `process_config.py:124` and `:172`
register two mutually exclusive modeld processes, selected by `get_active_model_runner()`
(`sunnypilot/models/helpers.py:149-161`), which returns `stock` whenever no sunnypilot
bundle is active:

| | `selfdrive/modeld/modeld.py` (stock) | `sunnypilot/modeld_v2/modeld.py` (tinygrad) |
|---|---|---|
| selected when | no bundle active -- **the default** | a sunnypilot bundle is active |
| runs chestnut-class bundles | no | **yes** -- this is the eGPU target |
| ASIC gate | added here (was absent) | `egpu.enabled()`, `modeld.py:331` |
| `.egpu` marker checked | no | `assert_pkl_matches`, `modeld.py:107` |
| small-model fallback | yes, load and runtime | added here (was absent on both) |

A device that has never selected a bundle runs the **stock** path. Any statement of the form
"the eGPU is gated" has to hold on both, and before this branch it held on neither in the
build and on only one at runtime.

---

## 2. Mechanisms, and whether they are reachable

| Mechanism | Where | Reachable here? |
|---|---|---|
| `bigModelLoading` -> `ET.NO_ENTRY` | `events.py:232-234`, set from `UsbGpuLoading` at `selfdrived.py:198-203` | Yes |
| `bigModelFailed` -> `ET.SOFT_DISABLE` + `ET.PERMANENT` | `events.py:236-239`, `selfdrived.py:205-211` and `:400` | Yes |
| `big_model_settling` load-window suppression | `selfdrived.py:433-435`, `:454` | Yes -- see section 3(c) |
| `Chestnut` firmware-flash class | `hardwared.py:42-77`, driven from `:303` when offroad | Yes |
| `chestnutPresent` on `deviceState` | `common/hardware/usb.py:69-87` | Yes. Matches USB IDs only, **not** the flashed-firmware product string, so it is true on stock ASMedia firmware too -- unlike `usbgpu_present()` |
| `ChestnutState` telemetry (`log.capnp:717-728`) | `modeld.py:76-128` (stock), `egpu/chestnut_state.py` (sunnypilot) | Yes when the eGPU is driving |
| SConscript link-up wait | `selfdrive/modeld/SConscript`, 10 x 1 s on `link_up()` | Yes |
| `compile_modeld.py` link-up wait | `sunnypilot/modeld_v2/compile_modeld.py:301-308` | Only under `DEV=USB+...`; **raises** where the SConscript skips |
| RDNA2 blocklist | `egpu/asics.py:54-76`, 21 device IDs | Yes -- now on all three paths |
| `.egpu` marker + `assert_pkl_matches` | `egpu/models.py:47-64` | Sunnypilot runner only. Records a **vendor only**, not a gfx target |
| `_USBGPU` catalog switch | `models/fetcher.py:151-158`, driven by `manager.py:281` | Yes |
| `guard.loading()` | `egpu/guard.py` | Yes -- new |
| `guard.egpu_build_ok()` | `egpu/guard.py` | Yes -- new |

### Present but unreachable

- **`Offroad_ChestnutBranch`** (`alerts_offroad.json:20-23`, gated at `hardwared.py:304-308`)
  is dead twice over. `CHESTNUT_BRANCHES` (`common/version.py:19-26`) has no `rdna2` key, so
  `chestnut_target` is None. Independently, `big_model_available` (`hardwared.py:242`) is True
  because `big_driving_supercombo.onnx` exists -- and it is a real 1.76 GB file on the device,
  not an LFS pointer, so that term is True regardless. Either alone kills the alert.
- **`NvChestnutState`** (`egpu/chestnut_state.py`) needs `egpu.enabled()` to resolve NVIDIA,
  which needs `EgpuUseNvidia`, a confirmed (not assumed) vendor, and a locally compiled NV
  pkl. None hold on a stock config.
- **`bigModelReadyDEPRECATED`** (`log.capnp:140`, ordinal 101) -- retired, no references.

### Checked and deliberately left alone

`bigModelFailed` carries `SOFT_DISABLE` and `PERMANENT` but **no `NO_ENTRY`**, while sitting
inside the block at `selfdrived.py:396-401` whose comment says every event there should have
both. That looks like a defect. It is not ours to fix: the block is byte-identical across
`upstream/master`, `upstream/dev-chestnut` and `upstream/staging-chestnut`. The design is
coherent -- `bigModelFailed` is a degradation notice, re-engaging on the small model is the
intent (the alert text says "small model is still available"), and a genuinely dead modeld is
blocked by `commIssue` and `processNotRunning`, which do carry `NO_ENTRY`
(`events.py:678-681`, `:693-696`).

---

## 3. What was actually broken

Three defects, all fixed on this branch, each with a test that fails without the fix
(`.elantra/test_egpu.py`, groups `[build gate]`, `[stock runner]`, `[loading flag]`,
`[sunnypilot runner]`).

**(a) The build could stop the car from starting.** `SConscript` set
`USBGPU = usbgpu_present()` with no ASIC check, then compiled the big model with
`DEV=USB+AMD:LLVM` as soon as `link_up()` saw LTSSM `0x78`. tinygrad refuses gfx1032, so the
target failed, and `build.py:63` turns a failed build into a blocking `TextWindow` plus
`exit(1)` -- an error screen that needs the touchscreen to clear. Now gated on
`egpu_build_ok()`, and `do_compile` skips instead of returning non-zero, so no eGPU compile
failure of any kind can fail the build.

**(b) The default runner had no ASIC gate.** `selfdrive/modeld/modeld.py:221` was
`usbgpu_present() and usbgpu_compiled()`. `asics.py` was consulted only from the sunnypilot
runner, so a fresh device -- which runs the stock path -- took the eGPU path with a
blocklisted card regardless. Now also `and egpu.enabled()`.

**(c) A failed eGPU load latched the car out of engagement.** The worst of the three.
`modeld_v2` set `UsbGpuLoading = True`, then raised on load failure, so its
`put_bool("UsbGpuLoading", False)` was never reached. The param is `CLEAR_ON_MANAGER_START`
and a modeld crash restarts *modeld*, not manager -- so it stayed True for the rest of the
ignition cycle. selfdrived then read it as `big_model_loading` (a `NO_ENTRY` every frame)
**and** as `big_model_settling`, which suppresses `commIssue`, `posenetInvalid` and
`locationdTemporaryError`. The car could not engage and was not told why. The flag is now held
by `guard.loading()`, whose `finally` releases it on every path, and `modeld_v2` falls back to
the on-SoC model at load and at runtime instead of raising.

---

## 4. Fallback matrix

`Engage` = can openpilot still engage. **Evidence** says how the row was established. Only
rows marked `exercised` were actually run on hardware; the rest are code-traced or covered by
a unit test, and are not claims about the device.

| # | Case | Driver sees | Engage | Evidence |
|---|---|---|---|---|
| 1 | No dock | Nothing. `usbgpu_present()` False, small model, normal boot | Yes | **exercised** -- AGNOS 19.6, build exit 0, manager stable, `UsbGpuLoading`/`UsbGpuActive` unset |
| 2 | Dock, unsupported card (RDNA2) | Nothing. Build skips the eGPU target; both runners gate off; small model | Yes | **exercised** -- see below |
| 3 | Dock, supported card, pkl missing | `usbgpu_compiled()` False on stock; sunnypilot load fails -> "Big Model Failed", soft-disable, then small model | Yes | code-traced only |
| 4 | Dock, pkl corrupt or wrong vendor | `assert_pkl_matches` raises inside the load -> same as 3 | Yes | code-traced only |
| 5 | eGPU load times out (60 s) | "Big Model Loading" NO_ENTRY during the load, then "Big Model Failed"; small model | Yes | unit-tested (`loading()` releases on raise); not run on hardware |
| 6 | modeld crash-loop | `commIssue` / `processNotRunning` NO_ENTRY while down; recovers when it returns | No while down -- correct | code-traced only |
| 7 | Dock unplugged mid-drive | `big_failed` via `big_model_active and not usbgpu_present` -> soft-disable, small model | -- | **not exercised** |
| 8 | eGPU faults after the model was driving | `model.run()` handler -> "big model failed, fall back to small", `UsbGpuActive` False | -- | **not exercised** |

Row 2 is the next thing to run and is what actually proves fix (a). Rows 3-6 need a *supported*
card, which we do not have -- they are reachable only once the RDNA2 port works, or with a
borrowed RDNA3 card. Rows 7 and 8 need a working eGPU and a moving car; they belong to Phase 3.

### Where the gate can see the card

`resolve_device()` prefers `EgpuDevice` (PERSISTENT override) over `EgpuDeviceDetected`
(`CLEAR_ON_MANAGER_START`, written by `probe_once()` in the model manager). The build runs
*before* manager (`launch_chffrplus.sh`: `build.py` then `manager.py`), so on the very first
boot with a new dock no probe result exists yet and `egpu_build_ok()` returns True -- the
skip-on-failure backstop is what carries that build. From the second boot on, the previous
boot's probe result is on disk at build time and the gate fires properly.

Setting `EgpuDevice=0x73ff` makes the gate deterministic from the first build with no probe
round-trip. That is the right configuration for this device while the port is unfinished --
and, given the next finding, currently the *only* reliable one.

### `probe_once()` cannot identify the card on a cold boot

`probe.probe_pci_ids()` reads PCIe config space through `CustomASM24Controller` and tries
`CANDIDATE_BUSES = (4, 2)`. Measured on the bench, straight after a boot with the dock
attached and the PCIe link trained (`link_up()` True, LTSSM `0x78`):

```
bus=0: dword=0x24641b21  vendor=0x1b21 device=0x2464   <- the ASM2464 bridge itself
bus=1: TLP completion status: Unsupported Request
bus=2: TLP completion status: Unsupported Request
bus=3: TLP completion status: Unsupported Request
bus=4: TLP completion status: Unsupported Request
```

`probe_ids()` therefore returns `None`. A trained link is not an enumerated one: the bridge's
secondary/subordinate bus registers are only programmed during `USBPCIDevice` bring-up. After
running `.elantra/probe_rdna2.py` (which does construct `USBPCIDevice`), the same call returns
`('amd', 0x73ff)` -- so the code is correct, it just needs enumeration to have happened first.

Consequence: `EgpuDeviceDetected` is `CLEAR_ON_MANAGER_START` and the probe that would
repopulate it runs offroad, before anything has enumerated the bridge. So it stays empty, the
gate stays blind, and the build-time backstop is what protects every boot rather than just the
first. Every failure path in `probe_pci_ids()` returns `None` silently, which is why this was
invisible. Not fixed here -- it is a probe defect, not a safety-path defect, and the backstop
covers it -- but it is why `EgpuDevice` is set explicitly on this device.

### Evidence for row 2

Three builds on the device, dock attached, card enumerating as `1002:73ff`:

| Build | Gate state | Result |
|---|---|---|
| A | `EgpuDevice=0x73ff` -> `egpu_build_ok()` **False** | `USBGPU` False, eGPU target never declared, **scons exit 0** |
| B | both device params cleared -> `egpu_build_ok()` **True** | `[USBGPU]` target declared and attempted, compile returned non-zero, `Big model build failed on the eGPU, skipping`, **scons exit 0** |
| C | direct `DEV=USB+AMD:LLVM` compile | exit 1 -- this is the real failure the backstop absorbs |

Build B is the case that used to brick the boot: before the fix, `do_compile` returned
`env.Execute`'s non-zero result, SCons failed the target, and `build.py:63` opened a blocking
`TextWindow` and `exit(1)`.

Afterwards: no `big_driving_tinygrad.pkl` and no chunkmanifest were produced, so
`usbgpu_compiled()` stays False; with the dock still attached, `usbgpu_present()` is True,
`detect.enabled()` is False, and both runners compute `USBGPU = False`. `UsbGpuLoading` and
`UsbGpuActive` are unset -- no latch. openpilot came up clean with no tombstones.

### What tinygrad actually fails on

Not the `Unsupported arch` assert. `AMDev.__init__` calls `_build_regs()` (`amdev.py:153`)
*before* `ops_amd.py:953` is reached, and that raises first:

```
ImportError: Failed to import regs.hdp 5.0.3
```

`_build_regs` (`amdev.py:327-334`) requires `mp`, **`hdp`**, `gc`, `mmhub`, `osssys`, and
`nbio` (GC < 12). `sdma` regs are required only for SDMA 4.4.2/4.4.4, so our 5.2.4 does not
need them. Against `reg_files`, this card is missing **gc 10.3.x, hdp 5.x, mmhub 2.x,
osssys 5.x, nbio 3.x**, plus package-level `smu 11.x` and `soc_10`.

`.elantra/probe_rdna2.py` stage 7 checks gc, mp, smu, nbio, osssys and mmhub -- **it does not
check `hdp`**, so it would report everything present while the build still failed. That has to
be corrected as part of Phase 2 Stage 1, along with making stage 7 a real gate.

---

## 5. Known gaps, not yet closed

- The `.egpu` marker records a **vendor only**. A gfx12-compiled pickle can still reach a
  gfx1032 card. Closing this is Phase 2 Stage 6 -- the marker has to carry the gfx target and
  `assert_pkl_matches` has to check it.
- `assert_pkl_matches` returns early when `usbgpu` is False (`models.py:56`), so a mismatched
  pkl on the on-SoC path is never checked.
- `chestnutPresent` and `usbgpu_present()` answer different questions (USB ID vs. USB ID plus
  flashed-firmware product string). `selfdrived.py:206` uses the looser one for `big_failed`.
  Benign today, but they are not interchangeable.
- `compile_modeld.py:301-308` raises where the SConscript skips. Only reachable from a manual
  `DEV=USB+...` invocation, not from the boot build.
