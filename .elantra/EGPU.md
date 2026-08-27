# eGPU support: AMD today, NVIDIA maybe

comma ships the **chestnut** dock with an AMD card, and openpilot assumes that everywhere.
This branch adds a vendor abstraction so a different card is *safe*, and carries a patched
tinygrad so an NVIDIA card is *possible*. Whether an NVIDIA card actually works is unproven —
see "What is not proven" below, and do not let anything in this repo suggest otherwise.

## The hardware chain

```
comma four  ──USB 3.1 Gen 2──▶  chestnut  ──PCIe Gen3 x2──▶  GPU
Snapdragon 845 MAX              ASM2464PD                    RX 9060 8GB (stock)
no USB4/Thunderbolt             open 8051 firmware
```

Three facts that shape everything:

- **There is no PCIe tunneling.** The comma four has no USB4/Thunderbolt. The ASM2464PD is a
  USB-to-PCIe *bridge*, driven over USB3 as a mass-storage device, with PCIe TLPs issued
  through vendor control transfers. Every config-space access is a USB round trip.
- **The link is Gen3 x2, not Gen4 x4.** comma's own firmware caps the speed and disables lanes
  2–3 deliberately: the ASM completer tops out around 1.68 GB/s either way, and x2 has better
  signal margin. Measured throughput on a comma four is ~323 MB/s.
- **tinygrad only binds its USB transport to AMD.** Upstream `ops_nv.py` registers
  `[NVKIface, PCIIface, MOCKIface]` and no `USBIface`, so `DEV=USB+NV` does not resolve at
  all. NVIDIA eGPU support exists upstream only for **macOS over USB4/Thunderbolt**
  (`DEV=PCI+NV`, signed DriverKit extension), which is a different mechanism and unavailable
  here.

## What this branch actually does

**The vendor abstraction (`openpilot/sunnypilot/egpu/`) fixes a real bug, independent of
NVIDIA.** Upstream conflates "a flashed chestnut is attached" with "run the big model on it".
Those coincide only because comma ships one vendor. With an NVIDIA card today, modeld resolves
`DEV='AMD'`, `load_oob` gets an AMD-compiled pickle, tinygrad fails to open the device, the
60-second loader timeout fires, and manager restarts the process forever. The car cannot
engage until the dock is unplugged.

Now `usbgpu_present()` still means "a chestnut is attached", and `egpu.enabled()` decides
whether the model uses it. **An eGPU we cannot drive looks like no eGPU, not a broken one.**

**Three gates keep a wrong-vendor model away from the planner:**

| Gate | When | What it does |
|---|---|---|
| `models/manager.py` catalog gate | offroad | an NV device never sees AMD bundles at all |
| `egpu.enabled()` | offroad | without an opted-in local NV model, `DEV` stays `QCOM` |
| `assert_pkl_matches()` | runtime | catches what the others cannot: the `COMBINED_MODEL_PKL` override, a hand-copied pickle, a bundle that survived a vendor change |

Gate 3 raises *before* `load_oob` unpickles anything. The only sanity check downstream is
`np.all(np.isfinite(...))`, which catches NaN and **not** wrong-but-finite numbers going into
the planner — which is why provenance is checked explicitly rather than trusted to fail.

**Telemetry is split honestly.** `pcieLtssm`, `supplyVoltage` and `supplyCurrent` come from
the ASM2464 bridge and are vendor-neutral. The other seven `ChestnutState` fields come from
AMD's SMU and have no NVIDIA equivalent over this link. On NVIDIA they are **left unwritten** —
not zeroed to look complete. `log.capnp` is untouched.

**The model is a pre-compiled pickle.** Every published eGPU bundle is built with
`DEV=USB+AMD:LLVM`, so on NVIDIA there is nothing to download; the pickle must be built on the
device with `openpilot/sunnypilot/egpu/compile_nv.py`, which writes a `.egpu` provenance
marker next to it.

Note the NV device string is `USB+NV` with **no renderer suffix**. AMD uses `:LLVM`; NV has no
LLVM renderer (`ops_nv.py` registers CUDA, PTX, NVCC and NAK), so `:LLVM` would fail to
resolve rather than fall back.

## Prior art

The tinygrad side is not ours. It is [PR #17369](https://github.com/tinygrad/tinygrad/pull/17369)
by **David Russell** (`russedavid:bounty/nv-usb3-wip`), replayed onto sunnypilot's tinygrad in
`aaravsaianugula/tinygrad:nv-usb3`. It is open, CI-green, developed on an actual chestnut, and
reports 49–61 TFLOPS on an RTX 3090 (same GA102 die as a 3080 Ti). geohot's objection is code
quality, not feasibility. There is an unclaimed $500 tinygrad bounty for exactly this.

The PR had to invent: the GPU sitting on PCIe bus 2 rather than AMD's bus 4; NV-specific ASM
register setup; a Gen1-downtrain-and-stream workaround because the GSP firmware image does not
fit the bridge's 512 KB SRAM; and a PCIe FLR plus `booter_unload` for teardown, because
`PCIDevice.reset()` shells out to a sysfs path that cannot exist for a USB-attached device.

## What is NOT proven

Nothing below has been tested, by us or (mostly) by anyone:

- Whether `DEV=USB+NV` opens a GA102 through an ASM2464PD **on a comma four**. The PR author
  tested on a bench host, not on an 845 running AGNOS with camerad and modeld at realtime
  priority. The GSP streaming path busy-waits on millisecond deadlines.
- Whether the model outputs are **numerically correct**. tinygrad issue #11705 documented
  openpilot model outputs on the CUDA/PTX renderers being silently wrong — max abs diff
  >50,000 against the ONNX reference, no error raised. Those are the exact renderers an
  NVIDIA card uses here. **A model that loads proves nothing.**
- Whether stale WPR2 recovery works over the bridge. If clearing it needs a physical power
  cycle, that is a car-stranding failure and the answer is no.
- Whether the dock's supply can carry the card. See below.

## Power: read this before buying anything

**An RTX 3080 Ti is 350 W TGP. The chestnut's in-car budget is reported around 100 W**, and
comma publishes no figure at all. Worse:

- **There is no software power cap available.** The watt-valued control
  (`NV2080_CTRL_CMD_PMGR_PWR_POLICY_IDX_LIMIT_ARB_INPUT_SET`) is stripped from NVIDIA's open
  kernel modules and therefore absent from tinygrad's generated headers. Even if hand-written,
  it does not persist: persistence needs an InfoROM that GeForce boards lack, and tinygrad
  cold-boots the card from GSP firmware at every process start, rebuilding RM state from the
  vBIOS power table.
- **The vBIOS route is closed** — GA102 vBIOS is signed, and tinygrad's driverless boot *is*
  the vBIOS verifier (it runs FWSEC in heavy-secure mode to build FRTS). Editing the power
  table breaks the boot you depend on. There is no crossflash target for 12 GB / 384-bit.
- **Limiting the rail is fault injection, not capping.** 3080-class cards pull ~2× average on
  sub-millisecond transients. A brownout mid-kernel means a PCIe link drop with GSP half-alive.
- **Cabling:** an AIB 3080 Ti needs 2×8-pin. The chestnut kit ships one 8-pin cable.

**But the framing may be wrong.** Nothing in this stack sets the AMD ~100 W figure either —
`AM_POWER_LIMIT` defaults to `0.0`, so tinygrad never sends `SetPptLimit`; it pins clocks to
*maximum*. That number is the RX 9060's own vBIOS default. So the real question is not "how do
I cap 350 W" but **"what does a 20 Hz model actually draw behind a 1.66 GB/s link?"** — and the
chestnut already carries an INA231 to measure it. Measure before engineering.

## Bring-up, in order, each with a stop condition

**Stage −1 — before buying. A normal Linux PC, ~30 minutes.**

```bash
sudo lspci -vv -d 10de:2208 | grep -A8 -i "Resizable BAR"   # want: supported up to 8192MB
nvidia-smi -q -d POWER                                       # default/min/max power limit
nvidia-smi -q | grep -i inforom                              # expect N/A on GeForce
```
Then count the PCIe power connectors. No ReBAR capability is a strong negative: BAR1 stays at
256 MB and that is the *entire* CPU-visible pool. 2×8-pin against a one-cable kit is a
purchase blocker.

**Stage 0 — bench, with the dock.**

| Step | Check | Pass |
|---|---|---|
| 0.1 | `cat /sys/bus/usb/devices/*/speed` | `10000`. `480` is the known ASM2464PD USB-2 fallback — power-cycle and retry. Do **not** proceed at 480; every later number is meaningless. |
| 0.2 | LTSSM via `flash.py:link_up()` | `0x78` (L0), retried 10×1 s |
| 0.3 | `pcie_cfg_req(0x00, bus=2)` | reads `0x2208_10de` |
| 0.4 | `bar_info(1)`, `vram_size`, `large_bar` | 8 GiB good; 256 MB tight; `size > vram_size` → stop |
| 0.5 | `DEV=USB+NV python -c "...Tensor(...).sum().item()"` | ~5.3 s startup |
| **0.6** | **run 0.5 ten times, then once after `kill -9` mid-run** | **10/10 clean, and the post-kill run recovers via FLR. Any run needing a physical power cycle is a stop-ship.** |

**Stage 1** — `test_tiny.py`, then a 4096 FP16 GEMM: ≥45 TFLOPS. Below ~20, measure power
before debugging software; you may simply be throttling.

**Stage 2** — numerical validation. Validate `_copyout` first (upload a known 400 KB pattern,
round-trip, assert byte-exact, ×10,000) — the download path is a race by construction and a
torn read raises nothing. Then compare against onnxruntime `CPUExecutionProvider` on the same
ONNX. **Do not use the default `rtol=1e-5`:** the model is fp16 and known-good backends already
fail it (LLVM 0.0769, GPU 0.0841, CUDA **149.5** — silently wrong). **Pass ≤ 0.2, fail > 1.0.**
Replay ≥1000 consecutive frames from a real segment and report the max, not the mean — #11705
only reproduced on specific real frames.

**Stage 3** — power, on the bench, before the car. The INA231 (vendor IN `0xC0`, `wLength 5`,
`<H h B>` = mV, mA, fault) at ≥100 Hz, **plus a DC current probe and scope at ≥100 kHz** — the
INA231 is configured AVG=1/CT=1.1 ms and cannot see a 1 ms transient. Measure idle, then the
model at 20 Hz for 10 minutes (**this is the number that decides the purchase**), then a
saturating GEMM, then cold-start. Any `bob_flt` assertion is an immediate fail.

**Stage 4** — in-vehicle, only after 0–3 pass. Re-run Stage 2 against segments from this car.

## RDNA2, and the RX 6600 XT

The card now in this dock is an **RX 6600 XT** -- Navi 23, `1002:73ff` rev C1, gfx1032. It does
not work, and note the reversal from the 3080 Ti above: at 160 W TBP on a single 8-pin it clears
the power blocker that kills the NVIDIA plan outright, and dies on software support instead.

**tinygrad's AM driver is RDNA3/RDNA4 only, and says so.** `tinygrad/runtime/ops_amd.py`:

```python
self.target = ((trgt:=self.iface.props['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
self.arch = "gfx%d%x%x" % self.target
assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), f"Unsupported arch: {self.arch}"
```

`gfx_target_version` is built from the GC IP version in the card's own discovery table, so a
Navi 23 arrives as `(10, 3, x)` and trips that assert before a kernel runs. Underneath it there
is nothing to fall back on either: `autogen/am/regs.py` ships `gc_9_4_3, gc_11_0_0, gc_11_0_3,
gc_11_5_0, gc_12_0_0` and no `gc_10_*`; `autogen/am/` has `soc_9, soc_11, soc_12` and no
`soc_10`; the SMU modules are `smu_13_0_*` and `smu_14_0_2`, where Navi 2x needs SMU 11.
`import_module` matches on major version, so each of those is an independent failure.

**There is no kernel-driver escape hatch.** The chestnut is a USB-to-PCIe bridge: the GPU never
appears to the kernel as a PCIe endpoint, so `amdgpu` cannot bind to it and ROCm is not an
option no matter what `HSA_OVERRIDE_GFX_VERSION` says. Every path to this card runs through
tinygrad's userspace AM driver. That is the whole option space.

### What this branch does about it

Not a driver port -- a gate, so the car is not harmed by a card it cannot use.

Upstream conflates "a chestnut is attached" with "run the big model on it". The vendor
abstraction split that by vendor; an RDNA2 card shows why vendor is the wrong granularity. An
RX 6600 XT and comma's RX 9060 are both `0x1002`, so before this change the 6600 XT resolved as
plain `amd`, took `DEV=USB+AMD:LLVM`, was served the gfx12-compiled `_USBGPU` bundles, failed to
open the device, and left modeld in the 60-second-timeout restart loop -- the car could not
engage until the dock was unplugged.

Now `egpu/asics.py` carries a table of ASICs AM refuses, `probe.py` keeps the device ID out of
the config dword it already reads, and `detect.enabled()` consults both. The dock stays attached
and powered, the panel names the card and says why, and the model runs on the SoC.

**The table is a blocklist on purpose.** It names only cards with positive evidence against
them; anything unrecognised takes exactly today's path. An allowlist would gate every AMD card
comma has not shipped yet, and every one whose device ID we typed wrong, on our being right.
`.elantra/test_egpu_tinygrad.py` re-derives the supported targets from that assert by AST on
every run, so if RDNA2 support ever lands upstream the blocklist fails loudly instead of quietly
switching off eGPUs that would now work.

### The probe, and what it decides

`.elantra/probe_rdna2.py`. Run it offroad, on the bench, with modeld stopped -- `USBPCIDevice`
takes an exclusive flock and must not be contended with a running model.

| Stage | Check | Stop condition |
|---|---|---|
| 1 | USB link speed | `480` is the ASM2464PD USB-2 fallback. Stop: nothing measured after it means anything |
| 2 | chestnut enumerated, LTSSM | not `0x78` -> stop |
| 3 | config `0x00` on bus 4 | expect `0x73ff1002`, rev `0xc1` |
| 4 | PCIe capability walk | does the endpoint advertise FLR? Recorded, not a stop |
| 5 | `mmRCC_CONFIG_MEMSIZE`, then the discovery table at `VRAM_SIZE - 64KB` | dumps every IP version: GC, MP0, MP1, SDMA, NBIO |
| **6** | **`regMP0_SMN_C2PMSG_35`, bit 31** | **the decisive one** |
| 7 | which register sets tinygrad already has | what a port would still have to generate |

Stage 5 replicates `AMDev._run_discovery` rather than calling it, because constructing `AMDev`
is exactly what cannot work -- it resolves `soc_10` and `gc_10` modules that do not exist. It
reads VRAM through `MM_INDEX`/`MM_DATA`, so a 256 MB BAR over 8 GB of VRAM is not a problem.

**Stage 6 is the go/no-go.** `AM_PSP._wait_for_bootloader` polls `regMP0_SMN_C2PMSG_35` for bit
31, and everything the driver does afterwards -- SOS, the TMR, every other IP's firmware -- is
downstream of it. tinygrad [#15636](https://github.com/tinygrad/tinygrad/issues/15636) reports
an RX 6900 XT (also RDNA2) sitting there at `0x0` forever with `TimeoutError: BL not ready`, on
a different bridge. If the 6600 XT does the same over the chestnut, the port is dead before any
of the register work matters and the answer is to stop.

### If stage 6 passes, what the port would cost

Listed so the size is not a surprise, not as a plan: `gc_10_3_0`, `soc_10` and `smu_11_0_x`
register sets; PSP v11 bring-up, whose C2PMSG layout differs from v13; gfx10 CP init, which uses
F32 ME/PFP/MEC against AM's RS64 path; Navi 2x firmware naming (`navi23_*.bin`, not the
IP-versioned `gc_11_0_0_*.bin` convention AM expects); a reset path that does not depend on
endpoint FLR; and lifting the `ops_amd.py` assert. Then on-device model compilation for gfx1032,
since every published `_USBGPU` bundle is gfx12. It would live in a tinygrad fork branch, which
`sync.py` requires to be exactly one non-merge commit on top of a commit in
`sunnypilot/tinygrad`.

A supported card remains the short path: the RX 9060 comma ships, or any RDNA3/RDNA4 board.

## Local checks (no hardware)

```bash
python .elantra/test_egpu.py
python .elantra/test_egpu_tinygrad.py --tinygrad <checkout>
python .elantra/guards.py --opendbc <path> --repo . --tinygrad <checkout>
```

With the dock, on the bench:

```bash
python .elantra/probe_rdna2.py --expect 0x73ff
```

The detector is proven in both directions: green on the patched tree, and correctly failing on
stock `sunnypilot/tinygrad@66ee3cf`, where every sentinel is genuinely absent. A detector that
has only ever seen good input has never been tested.
