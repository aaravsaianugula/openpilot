#!/usr/bin/env python3
"""
Tests for AM_SMU.set_clocks: one clock domain refusing must not abandon the others.

The bug this exists to prevent, measured on the RX 6600 XT in the chestnut:

    am usb:4-2: running at the SMU's default clocks, it refused to set them:
                SMU refused msg 0x1a (param 0x2ffff): CMD_FAIL [0xff]

Decoded, msg 0x1a is PPSMC_MSG_SetSoftMaxByFreq and param 0x2ffff is PPCLK_UCLK(2)<<16|0xffff.
Sienna Cichlid's SMU will not take UCLK without a PPTable upload AM does not do. UCLK is the
*first* domain in the loop and its max-set was the one call not wrapped in suppress(), so the
raise escaped set_clocks entirely and GFXCLK -- the one that decides whether the card runs at
500 MHz or 2.6 GHz -- was never asked. The model ran at 262 GFLOPS, 2.5% of the card's peak,
and the log said only "default clocks", which reads identically to the SMU being dead.

No hardware needed: AM_SMU's clock path touches only self.smu_mod, self.adev.ip_ver and
self._send_msg, so a stub answers the question. What is NOT tested here is whether the SMU
then honours a max it accepted -- that needs the card, and read_clocks/gpuClockMhz answer it.

Usage:
    python .elantra/test_am_clocks.py                    # uses ./tinygrad_repo
    python .elantra/test_am_clocks.py --tinygrad <path>  # e.g. /data/rdna2-tg on the device
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

failures: list[str] = []
passes: list[str] = []


def case(name: str, got, want) -> None:
    if got == want:
        passes.append(name)
    else:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def check(name: str, condition: bool, detail: str = "") -> None:
    case(name + ((": " + detail) if detail and not condition else ""), bool(condition), True)


import contextlib
import io


@contextlib.contextmanager
def capture():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class FakeVram:
    """adev.vram, recording what got written where instead of touching a card.

    Also serves reads from `blob`, so read_table can decode a synthetic SmuMetrics without a card.
    """

    def __init__(self, blob: bytes = b""):
        self.writes: list[tuple[int, bytes]] = []
        self.blob = blob

    def view(self, paddr, size):
        outer = self

        class _V:
            def __setitem__(self, _slice, data):
                assert len(bytes(data)) == size, f"view sized {size} written with {len(bytes(data))}"
                outer.writes.append((paddr, bytes(data)))

            def __getitem__(self, _slice):
                return outer.blob.ljust(size, bytes(1))[:size]
        return _V()


class FakeFw:
    def __init__(self, pptable):
        self.smu_pptable = pptable


class FakeAdev:
    devfmt = "usb:4-2"

    def __init__(self, ip_ver, pptable=None, rom=None, vram_blob=b""):
        self.ip_ver = ip_ver
        self.fw = FakeFw(pptable)
        self.vram = FakeVram(vram_blob)
        # No SMUIO register bases unless a test supplies a ROM: the default fixture models a card
        # whose VBIOS cannot be read, which must still boot with DPM tables and no clocks.
        self.regs_offset = {} if rom is None else {24: {0: (0x16800,)}}
        self.rom, self.rom_idx, self.reg_writes = rom, 0, []

    def rreg(self, reg):
        if reg == 0x168E5:
            return self.rom_idx
        if reg == 0x168E6 and self.rom is not None:
            v = int.from_bytes(self.rom[self.rom_idx:self.rom_idx + 4].ljust(4, bytes(1)), "little")
            self.rom_idx += 4
            return v
        return 0

    def wreg(self, reg, val):
        self.reg_writes.append((reg, val))
        if reg == 0x168E5:
            self.rom_idx = val

    def paddr2mc(self, paddr):
        return paddr | (1 << 40)


class Recorder:
    """An AM_SMU with _send_msg replaced: records every (msg, param) and refuses on demand.

    `refuse` is a set of (msg, clck) pairs the fake SMU answers CMD_FAIL for; `timeout` is the
    set it does not answer at all. Everything else succeeds.
    """

    def __init__(self, am_smu_cls, smu_mod, ip_ver, refuse=(), timeout=(), levels=None, pptable=None, rom=None,
                 vram_blob=b"", fw_version=None, clock_levels=None):
        self.smu = am_smu_cls.__new__(am_smu_cls)
        self.smu.smu_mod = smu_mod
        self.smu.adev = FakeAdev(ip_ver, pptable, rom, vram_blob)
        self.fw_version = fw_version
        self.smu.driver_table_paddr = 0x9a000
        self.smu._send_msg = self._send_msg
        if levels is not None:
            self.smu.read_clocks = lambda clks: dict.fromkeys(clks, levels)
        if clock_levels is not None:
            self.smu.read_clocks = lambda clks: {c: clock_levels[c] for c in clks if c in clock_levels}
        self.mod = smu_mod
        self.refuse = set(refuse)
        self.running = 0xFFFFFFFFFFFFFFFF   # by default every feature came up
        self.timeout = set(timeout)
        self.sent: list[tuple[int, int]] = []

    def _send_msg(self, msg, param, read_back_arg=False, timeout=10000, debug=False):
        from tinygrad.runtime.support.am.ip import SMUError
        clck = (param >> 16) & 0xFFFF
        self.sent.append((msg, param))
        if (msg, clck) in self.timeout:
            raise TimeoutError(f"SMU msg {msg:#x} timeout")
        if (msg, clck) in self.refuse:
            raise SMUError(f"SMU refused msg {msg:#x} (param {param:#x}): CMD_FAIL [0xff]")
        if msg == self.mod.PPSMC_MSG_GetRunningSmuFeaturesLow:
            return self.running & 0xFFFFFFFF
        if msg == self.mod.PPSMC_MSG_GetRunningSmuFeaturesHigh:
            return (self.running >> 32) & 0xFFFFFFFF
        if msg == self.mod.PPSMC_MSG_GetSmuVersion and self.fw_version is not None:
            return self.fw_version
        return 0

    def maxed(self) -> set[int]:
        """Clock domains a SetSoftMaxByFreq was actually issued for."""
        return {(p >> 16) & 0xFFFF for m, p in self.sent if m == self.mod.PPSMC_MSG_SetSoftMaxByFreq}


def build(tinygrad_path: Path):
    sys.path.insert(0, str(tinygrad_path))
    from tinygrad.runtime.autogen.am import am
    from tinygrad.runtime.autogen.am import smu_11_0_7
    from tinygrad.runtime.support.am.ip import AM_SMU, SMUError
    return am, smu_11_0_7, AM_SMU, SMUError


def navi23_ip_ver(am):
    """The IP versions the RX 6600 XT reports from its own discovery table."""
    return {am.GC_HWIP: (10, 3, 4), am.MP0_HWIP: (11, 0, 12), am.MP1_HWIP: (11, 0, 12),
            am.SMUIO_HWIP: (11, 0, 10)}


def test_uclk_refusal_does_not_abandon_gfxclk(am, mod, AM_SMU):
    """The measured failure: UCLK says no, and GFXCLK must still be asked."""
    r = smu11_recorder(am, mod, AM_SMU, refuse=[(mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPCLK_UCLK)])
    r.smu.set_clocks(level=None)
    check("UCLK refusal still reaches GFXCLK", mod.PPCLK_GFXCLK in r.maxed(),
          f"SetSoftMaxByFreq issued only for {sorted(r.maxed())}")
    check("UCLK refusal still reaches SOCCLK", mod.PPCLK_SOCCLK in r.maxed())


def test_reports_which_domains_took(am, mod, AM_SMU):
    """A card with GFXCLK pinned and UCLK at boot is not the same as neither being set."""
    r = smu11_recorder(am, mod, AM_SMU, refuse=[(mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPCLK_UCLK)])
    took = r.smu.set_clocks(level=None)
    check("set_clocks reports per-domain outcomes", isinstance(took, dict),
          f"returned {type(took).__name__}")
    if isinstance(took, dict):
        case("GFXCLK reported as taken", took.get("PPCLK_GFXCLK"), True)
        case("UCLK reported as refused", took.get("PPCLK_UCLK"), False)


def test_every_domain_refusing_is_visible(am, mod, AM_SMU):
    """Total refusal must stay distinguishable from success, not silently return."""
    every = [(mod.PPSMC_MSG_SetSoftMaxByFreq, c)
             for c in (mod.PPCLK_UCLK, mod.PPCLK_FCLK, mod.PPCLK_SOCCLK, mod.PPCLK_GFXCLK)]
    r = smu11_recorder(am, mod, AM_SMU, refuse=every)
    took = r.smu.set_clocks(level=None)
    check("all-refused reports no domain taken",
          isinstance(took, dict) and bool(took) and not any(took.values()), f"got {took!r}")


def test_timeout_still_propagates(am, mod, AM_SMU):
    """SMUError is the SMU answering no. A timeout is it not answering, and is a real fault."""
    r = smu11_recorder(am, mod, AM_SMU, timeout=[(mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPCLK_UCLK)])
    try:
        r.smu.set_clocks(level=None)
        check("a max-set timeout propagates", False, "set_clocks swallowed TimeoutError")
    except TimeoutError:
        check("a max-set timeout propagates", True)


def test_level_branch_is_equally_tolerant(am, mod, AM_SMU):
    """The level path has the same shape and had the same bug."""
    r = smu11_recorder(am, mod, AM_SMU, refuse=[(mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPCLK_UCLK)])
    r.smu.set_clocks(level=-1)
    check("level branch: UCLK refusal still reaches GFXCLK", mod.PPCLK_GFXCLK in r.maxed(),
          f"SetSoftMaxByFreq issued only for {sorted(r.maxed())}")


def test_healthy_card_sets_everything(am, mod, AM_SMU):
    """No regression for a card whose SMU takes every domain."""
    r = smu11_recorder(am, mod, AM_SMU)
    took = r.smu.set_clocks(level=None)
    for name, clck in (("GFXCLK", mod.PPCLK_GFXCLK), ("UCLK", mod.PPCLK_UCLK),
                       ("SOCCLK", mod.PPCLK_SOCCLK)):
        check(f"healthy card sets {name}", clck in r.maxed())
    check("healthy card reports every domain taken",
          isinstance(took, dict) and all(took.values()) and len(took) == 3, f"got {took!r}")


def make_pptable(features: int, size: int = 1668) -> bytes:
    """A stand-in PPTable_t: uint32 Version, uint32 FeaturesToRun[2], then don't-care bytes."""
    body = bytearray(size)
    body[0:4] = (6).to_bytes(4, "little")
    body[4:12] = features.to_bytes(8, "little")
    return bytes(body)


# What a Navi 23's dimgrey_cavefish_smc.bin actually carries.
NAVI23_FEATURES = 0x00003763_a77ffdff


def test_pptable_is_uploaded_before_features_are_enabled(am, mod, AM_SMU):
    """Order is the whole thing: table, BTC, mask, enable. amdgpu's smu_smc_hw_setup order."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES))
    r.smu.init_hw()
    order = [m for m, _ in r.sent]
    # High before Low is smu_v11_0_set_allowed_mask's order, and the running-feature read-back is
    # the only thing that distinguishes "the SMU started DPM" from "the SMU took the mask".
    want = [mod.PPSMC_MSG_SetDriverDramAddrHigh, mod.PPSMC_MSG_SetDriverDramAddrLow,
            mod.PPSMC_MSG_TransferTableDram2Smu, mod.PPSMC_MSG_RunDcBtc,
            mod.PPSMC_MSG_SetAllowedFeaturesMaskHigh, mod.PPSMC_MSG_SetAllowedFeaturesMaskLow,
            mod.PPSMC_MSG_EnableAllSmuFeatures, mod.PPSMC_MSG_NotifyPowerSource,
            mod.PPSMC_MSG_GetRunningSmuFeaturesLow, mod.PPSMC_MSG_GetRunningSmuFeaturesHigh]
    case("init_hw message order", order, want)
    case("the pptable reached the driver table", len(r.smu.adev.vram.writes), 1)
    if r.smu.adev.vram.writes:
        paddr, data = r.smu.adev.vram.writes[0]
        case("written to driver_table_paddr", paddr, r.smu.driver_table_paddr)
        case("whole table written", len(data), 1668)
    transfer = [p for m, p in r.sent if m == mod.PPSMC_MSG_TransferTableDram2Smu]
    case("transferred as TABLE_PPTABLE", transfer, [mod.TABLE_PPTABLE])


def test_power_gating_features_are_not_enabled(am, mod, AM_SMU):
    """AM has no handshake to wake a gated block, so it must not let the SMU gate one."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES))
    r.smu.init_hw()
    lo = [p for m, p in r.sent if m == mod.PPSMC_MSG_SetAllowedFeaturesMaskLow][0]
    hi = [p for m, p in r.sent if m == mod.PPSMC_MSG_SetAllowedFeaturesMaskHigh][0]
    mask = lo | (hi << 32)
    for bit, name in AM_SMU.SMU11_UNSERVICEABLE_FEATURES.items():
        check(f"{name} not enabled", not (mask >> bit) & 1)
    # ...and the clock DPM features, which are the entire point, survive.
    for bit, name in ((1, "DPM_GFXCLK"), (3, "DPM_UCLK"), (4, "DPM_FCLK"), (5, "DPM_SOCCLK")):
        check(f"{name} still enabled", bool((mask >> bit) & 1))
    check("mask is a subset of the table's FeaturesToRun", mask & ~NAVI23_FEATURES == 0,
          f"mask {mask:#x} adds bits the firmware did not ask for")


def test_smu13_path_is_untouched(am, mod, AM_SMU):
    """SMU 13/14 boot with a soft pptable already loaded; they must not gain messages."""
    r = Recorder(AM_SMU, mod, {am.GC_HWIP: (11, 0, 0), am.MP0_HWIP: (13, 0, 0), am.MP1_HWIP: (13, 0, 0)},
                 pptable=None)
    r.smu.init_hw()
    case("no pptable means the original three messages",
         [m for m, _ in r.sent],
         [mod.PPSMC_MSG_SetDriverDramAddrHigh, mod.PPSMC_MSG_SetDriverDramAddrLow,
          mod.PPSMC_MSG_EnableAllSmuFeatures])
    case("nothing written to vram", len(r.smu.adev.vram.writes), 0)


def test_a_refused_pptable_does_not_stop_the_device(am, mod, AM_SMU):
    """No DPM is a slow card. A raise here is no card at all, and modeld would lose the eGPU."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES),
                 refuse=[(mod.PPSMC_MSG_TransferTableDram2Smu, mod.TABLE_PPTABLE)])
    try:
        r.smu.init_hw()
        check("a refused pptable is survivable", True)
    except Exception as e:
        check("a refused pptable is survivable", False, f"init_hw raised {type(e).__name__}: {e}")
    check("features are still enabled after a refused table",
          mod.PPSMC_MSG_EnableAllSmuFeatures in [m for m, _ in r.sent])


def test_a_pptable_timeout_still_propagates(am, mod, AM_SMU):
    """An SMU that does not answer is a fact about the device, not about the table."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES),
                 timeout=[(mod.PPSMC_MSG_TransferTableDram2Smu, mod.TABLE_PPTABLE)])
    try:
        r.smu.init_hw()
        check("a pptable timeout propagates", False, "init_hw swallowed TimeoutError")
    except TimeoutError:
        check("a pptable timeout propagates", True)


def test_board_data_is_patched_into_the_uploaded_table(am, mod, AM_SMU):
    """The upload must carry the VBIOS board bytes, not the firmware's zeroed ones.

    This is the whole chain in one assertion: read the ROM through SMUIO, walk ATOM to
    smc_dpm_info, and blit its 292 bytes over PPTable_t's I2cControllers..BoardReserved before
    the table goes to the SMU. If this passes and the clock still does not move, the board data
    was not the limiter.
    """
    import test_vbios
    rom = test_vbios.build_rom()
    expect = rom[test_vbios.SMC_AT + 4:test_vbios.SMC_AT + test_vbios.SMC_SIZE]

    r = Recorder(AM_SMU, mod, navi23_ip_ver(am),
                 pptable=make_pptable(NAVI23_FEATURES), rom=rom)
    r.smu.init_hw()
    check("something was uploaded", len(r.smu.adev.vram.writes) == 1,
          f"{len(r.smu.adev.vram.writes)} vram writes")
    if r.smu.adev.vram.writes:
        _, sent = r.smu.adev.vram.writes[0]
        case("board section carries the VBIOS bytes", sent[1344:1636], expect)
        case("the rest of the table is untouched", sent[:1344], make_pptable(NAVI23_FEATURES)[:1344])
        check("board section is no longer zero", any(sent[1344:1636]))
    case("only ROM_INDEX was written to MMIO",
         sorted({reg for reg, _ in r.smu.adev.reg_writes}), [0x168E5])


def test_a_feature_that_did_not_start_is_named(am, mod, AM_SMU, capture):
    """A card that took the mask and started nothing must not look like one that worked."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES))
    r.running = 0                      # the SMU reports nothing running
    with capture() as out:
        r.smu.init_hw()
    check("DPM_GFXCLK named when it did not start", "DPM_GFXCLK" in out.getvalue(), out.getvalue())
    check("says the requests will be ignored", "accepted and ignored" in out.getvalue())


def test_all_features_running_is_quiet(am, mod, AM_SMU, capture):
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES))
    r.running = NAVI23_FEATURES        # everything asked for came up
    with capture() as out:
        r.smu.init_hw()
    check("no complaint when the features are running", "did not start" not in out.getvalue(),
          out.getvalue())


def test_the_smu_is_told_it_is_on_ac(am, mod, AM_SMU):
    """A card that booted believing DC applies a different limit set. amdgpu says so explicitly."""
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am), pptable=make_pptable(NAVI23_FEATURES))
    r.smu.init_hw()
    sent = [p for m, p in r.sent if m == mod.PPSMC_MSG_NotifyPowerSource]
    case("NotifyPowerSource sent once, as AC", sent, [mod.SMU_POWER_SOURCE_AC])
    case("and AC is 0", mod.SMU_POWER_SOURCE_AC, 0)


# --- SmuMetrics decoding ------------------------------------------------------------------
#
# The bug: modeld's ChestnutState read metrics.AvgTemperature[...] and metrics.AvgFanRpm, which
# are SMU 13 names. SMU 11 spells them TemperatureHotspot, TemperatureMem and CurrFanSpeed, so on
# a 6600 XT the read raised AttributeError, was swallowed by an `except Exception`, and
# chestnutState published nothing -- including gpuClockMhz, the one signal that says whether a
# clock request did anything. Every clock number in this port was measured blind because of it.


def smu11_metrics_blob(mod, *, v2: bool, gfx=2282, soc=960, uclk=1000, fclk=1801,
                       gfx_act=34, uclk_act=12, watts=68, hotspot=34, mem_temp=40, fan=1500):
    """A SmuMetrics_t / SmuMetrics_V2_t image with known values at the real offsets.

    The two layouts agree up to offset 96 and diverge after it: V1 has ThrottlerStatus there and
    CurrFanSpeed at 102, V2 has AccCnt plus a 20-byte ThrottlingPercentage and CurrFanSpeed at
    122. That divergence is the whole reason the version has to be chosen rather than assumed.
    """
    buf = bytearray(256)

    def u32(off, val):
        buf[off:off + 4] = int(val).to_bytes(4, "little")

    def u16(off, val):
        buf[off:off + 2] = int(val).to_bytes(2, "little")

    for clck, val in ((mod.PPCLK_GFXCLK, gfx), (mod.PPCLK_SOCCLK, soc),
                      (mod.PPCLK_UCLK, uclk), (mod.PPCLK_FCLK, fclk)):
        u32(clck * 4, val)                 # CurrClock[13] at offset 0
    u16(54, gfx)                           # AverageGfxclkFrequencyPostDs
    u16(62, uclk)                          # AverageUclkFrequencyPostDs
    u16(64, gfx_act)                       # AverageGfxActivity
    u16(66, uclk_act)                      # AverageUclkActivity
    u16(72, watts)                         # AverageSocketPower
    u16(76, hotspot)                       # TemperatureHotspot
    u16(78, mem_temp)                      # TemperatureMem
    u16(122 if v2 else 102, fan)           # CurrFanSpeed
    return bytes(buf)


def navi23_metrics(am, mod, AM_SMU, *, fw, v2):
    r = Recorder(AM_SMU, mod, navi23_ip_ver(am),
                 vram_blob=smu11_metrics_blob(mod, v2=v2), fw_version=fw)
    return r.smu.metrics()


def test_metrics_uses_smu11_field_names(am, mod, AM_SMU):
    """The actual defect: SMU 13 names on an SMU 11 card. Reading at all is the assertion."""
    m = navi23_metrics(am, mod, AM_SMU, fw=0x3B3100, v2=True)
    case("hotspot from TemperatureHotspot", m["temp_hotspot"], 34)
    case("memory temp from TemperatureMem", m["temp_mem"], 40)
    case("power from AverageSocketPower", m["socket_power"], 68)
    case("activity from AverageGfxActivity", m["gfx_activity"], 34)


def test_metrics_reports_every_clock_domain(am, mod, AM_SMU):
    """Memory clock is the number this card is actually short of; it has to be readable."""
    m = navi23_metrics(am, mod, AM_SMU, fw=0x3B3100, v2=True)
    case("gfxclk", m["gfxclk"], 2282)
    case("socclk", m["socclk"], 960)
    case("uclk", m["uclk"], 1000)
    case("fclk", m["fclk"], 1801)


def test_metrics_picks_v2_at_the_firmware_threshold(am, mod, AM_SMU):
    """0x3B2300 is amdgpu's threshold for IP 11.0.12. The card here reports 0x3B3100."""
    at = navi23_metrics(am, mod, AM_SMU, fw=0x3B2300, v2=True)
    case("fan speed at the threshold", at["fan_rpm"], 1500)
    above = navi23_metrics(am, mod, AM_SMU, fw=0x3B3100, v2=True)
    case("fan speed above the threshold", above["fan_rpm"], 1500)


def test_metrics_picks_v1_below_the_firmware_threshold(am, mod, AM_SMU):
    """One count below the threshold the layout is V1, and the fan moves back to offset 102."""
    m = navi23_metrics(am, mod, AM_SMU, fw=0x3B22FF, v2=False)
    case("fan speed below the threshold", m["fan_rpm"], 1500)


def test_metrics_version_thresholds_match_amdgpu(am, mod, AM_SMU):
    """sienna_cichlid_get_smu_metrics_data's table, verbatim. A wrong number here is silent."""
    case("v2 thresholds", AM_SMU.SMU11_METRICS_V2_MIN_FW,
         {(11, 0, 7): 0x3A4300, (11, 0, 11): 0x412D00, (11, 0, 12): 0x3B2300, (11, 0, 13): 0x491100})
    case("v3 thresholds", AM_SMU.SMU11_METRICS_V3_MIN_FW, {(11, 0, 7): 0x3A4900})


def test_metrics_does_not_ask_an_smu13_card_for_a_version(am, mod, AM_SMU):
    """SMU 13 is what comma ships and it works today; its read must not gain a round trip."""
    import tinygrad.runtime.autogen.am.smu_13_0_0 as mod13
    r = Recorder(AM_SMU, mod13, {am.MP0_HWIP: (13, 0, 0), am.MP1_HWIP: (13, 0, 0), am.GC_HWIP: (11, 0, 0)},
                 vram_blob=bytes(512))
    r.smu.metrics()
    asked = [m for m, _ in r.sent if m == mod13.PPSMC_MSG_GetSmuVersion]
    check("no GetSmuVersion on the SMU 13 path", not asked, f"sent {len(asked)}")


# --- the SMU 11 clock policy -------------------------------------------------------------------
#
# Measured on the RX 6600 XT with .elantra/clock_ladder.py, one fresh card open per rung:
#
#   UCLK 456  soft-min                accepted in 1.47 ms, SMU never answers again
#   UCLK 675  soft-min                accepted, memory moves to 675 MHz and stays
#   UCLK 1000 soft-min                accepted in 1.48 ms, SMU never answers again
#   UCLK 1000 hard-min (0x1B)         same
#   UCLK 1000 ceiling-then-floor      same, with and without GFXCLK/SOCCLK raised first
#   UCLK 1000 with DS_UCLK, DS_FCLK, DF_CSTATE and the memory VR scaling features masked off
#                                     same
#
# Exactly one memory state on this ASIC is enterable, and it is the one amdgpu's own profiling
# pstate names: DIMGREY_CAVEFISH_UMD_PSTATE_PROFILING_MEMCLK 676. Asking for the top DPM entry --
# which is what set_clocks did by default, because AM_POWER_LIMIT is unset everywhere except the
# bench harness -- wedges the SMU during boot.


def navi23_clock_levels(mod):
    """What GetDpmFreqByIndex actually returns on this card."""
    return {mod.PPCLK_GFXCLK: [500, 2350], mod.PPCLK_SOCCLK: [480, 1371],
            mod.PPCLK_UCLK: [96, 456, 675, 1000], mod.PPCLK_FCLK: [500, 1801]}


def smu11_recorder(am, mod, AM_SMU, ip_ver=None, **kw):
    return Recorder(AM_SMU, mod, ip_ver or navi23_ip_ver(am),
                    clock_levels=navi23_clock_levels(mod), **kw)


def freq_msgs(r, mod, clck):
    """(msg, MHz) for every soft limit issued against one domain, in order."""
    return [(m, p & 0xFFFF) for m, p in r.sent
            if m in (mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPSMC_MSG_SetSoftMinByFreq)
            and (p >> 16) & 0xFFFF == clck]


def test_smu11_never_asks_for_fclk(am, mod, AM_SMU):
    """smu_v11_0_set_performance_level touches GFXCLK, MCLK and SOCCLK. FCLK belongs to PMFW."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=None)
    check("FCLK is never asked", not freq_msgs(r, mod, mod.PPCLK_FCLK),
          f"issued {freq_msgs(r, mod, mod.PPCLK_FCLK)}")


def test_smu11_pins_memory_to_the_asic_profiling_level(am, mod, AM_SMU):
    """676 snapped down to the card's own DPM table is 675 -- the one level that works."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=None)
    case("memory ceiling and floor both at the profiling level",
         [mhz for _, mhz in freq_msgs(r, mod, mod.PPCLK_UCLK)], [675, 675])


def test_smu11_never_asks_for_the_top_memory_level(am, mod, AM_SMU):
    """The measured wedge. 1000 is in the DPM table and must still never be requested."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=None)
    r.smu.set_clocks(level=-1)
    asked = [mhz for _, mhz in freq_msgs(r, mod, mod.PPCLK_UCLK)]
    check("1000 MHz is never requested", 1000 not in asked, f"requested {asked}")


def test_smu11_leaves_the_gfxclk_floor_open(am, mod, AM_SMU):
    """Pinning GFXCLK measured slower than leaving its governor free: 1950 pinned against 2340
    free, on the same card in the same session."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=None)
    gfx = dict(freq_msgs(r, mod, mod.PPCLK_GFXCLK))
    case("GFXCLK ceiling is the firmware maximum", gfx.get(mod.PPSMC_MSG_SetSoftMaxByFreq), 0xffff)
    case("GFXCLK floor left at zero", gfx.get(mod.PPSMC_MSG_SetSoftMinByFreq), 0)


def test_smu11_raises_the_ceiling_before_the_floor(am, mod, AM_SMU):
    """smu_v11_0_set_soft_freq_limited_range sends SetSoftMaxByFreq then SetSoftMinByFreq."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=None)
    for name, clck in (("GFXCLK", mod.PPCLK_GFXCLK), ("UCLK", mod.PPCLK_UCLK),
                       ("SOCCLK", mod.PPCLK_SOCCLK)):
        case(f"{name} ceiling before floor", [m for m, _ in freq_msgs(r, mod, clck)],
             [mod.PPSMC_MSG_SetSoftMaxByFreq, mod.PPSMC_MSG_SetSoftMinByFreq])


def test_smu11_leaves_memory_alone_on_an_asic_with_no_known_pstate(am, mod, AM_SMU):
    """Navy Flounder has no UMD pstate constants in sienna_cichlid_ppt.h. A guessed memory level
    wedges an SMU, and slow memory beats a dead card, so it must not be guessed."""
    ip = dict(navi23_ip_ver(am))
    ip[am.MP1_HWIP] = (11, 0, 11)
    r = smu11_recorder(am, mod, AM_SMU, ip_ver=ip)
    r.smu.set_clocks(level=None)
    check("no memory request on an unknown ASIC", not freq_msgs(r, mod, mod.PPCLK_UCLK),
          f"issued {freq_msgs(r, mod, mod.PPCLK_UCLK)}")


def test_smu11_teardown_does_not_touch_memory(am, mod, AM_SMU):
    """fini drops to level 0. Putting UCLK back to its 96 MHz boot level is the same class of
    transition that wedges it on the way up, and there is nothing to reclaim at teardown."""
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=0)
    check("teardown leaves memory where it is", not freq_msgs(r, mod, mod.PPCLK_UCLK),
          f"issued {freq_msgs(r, mod, mod.PPCLK_UCLK)}")


def test_smu11_teardown_asks_the_smu_nothing(am, mod, AM_SMU):
    """fini runs after a hang as often as after a clean run, so teardown must not need an answer.

    The regression: the teardown branch looked up the lowest DPM entry with read_clocks((gfx,soc)).
    read_clocks is cached per clk_list tuple, so that teardown-shaped key missed the cache the boot
    path had filled and issued a real GetDpmFreqByIndex. Against a wedged SMU that is a 10 s
    timeout, and amdev.fini suppresses SMUError but not TimeoutError -- so it escaped and buried
    whatever had actually gone wrong. Every BEAM round here died reporting this instead of the
    hang that caused it.
    """
    r = smu11_recorder(am, mod, AM_SMU)
    r.smu.set_clocks(level=0)
    queried = [(m, p) for m, p in r.sent if m == mod.PPSMC_MSG_GetDpmFreqByIndex]
    check("teardown issues no GetDpmFreqByIndex", not queried, f"issued {queried}")
    lows = [mhz for _, mhz in freq_msgs(r, mod, mod.PPCLK_GFXCLK)]
    case("teardown asks for the firmware minimum", lows, [0, 0])


def test_smu11_profiling_pstates_match_amdgpu(am, mod, AM_SMU):
    """sienna_cichlid_ppt.h, verbatim: (GFXCLK, SOCCLK, MEMCLK) keyed by MP1 IP version."""
    case("profiling pstates", AM_SMU.SMU11_PROFILING_PSTATE,
         {(11, 0, 7): (1825, 960, 1000), (11, 0, 12): (1950, 960, 676),
          (11, 0, 13): (2200, 960, 1000)})


def test_smu13_clock_path_is_unchanged(am, mod, AM_SMU):
    """SMU 13 is what comma ships and works today: all four domains, floor then ceiling."""
    import tinygrad.runtime.autogen.am.smu_13_0_0 as mod13
    r = Recorder(AM_SMU, mod13,
                 {am.MP0_HWIP: (13, 0, 0), am.MP1_HWIP: (13, 0, 0), am.GC_HWIP: (11, 0, 0)})
    r.smu.set_clocks(level=None)
    asked = {(p >> 16) & 0xFFFF for _, p in r.sent}
    for name in ("GFXCLK", "UCLK", "SOCCLK", "FCLK"):
        check(f"SMU 13 still asks {name}", getattr(mod13, "PPCLK_" + name) in asked)
    order = [m for m, p in r.sent if (p >> 16) & 0xFFFF == mod13.PPCLK_UCLK]
    case("SMU 13 still sends the floor first", order[0], mod13.PPSMC_MSG_SetSoftMinByFreq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()

    if not (args.tinygrad / "tinygrad" / "runtime" / "support" / "am" / "ip.py").is_file():
        print(f"no tinygrad AM driver at {args.tinygrad}; nothing to test")
        return 2

    am, mod, AM_SMU, _ = build(args.tinygrad)
    print(f"tinygrad: {args.tinygrad}")

    test_uclk_refusal_does_not_abandon_gfxclk(am, mod, AM_SMU)
    test_reports_which_domains_took(am, mod, AM_SMU)
    test_every_domain_refusing_is_visible(am, mod, AM_SMU)
    test_timeout_still_propagates(am, mod, AM_SMU)
    test_level_branch_is_equally_tolerant(am, mod, AM_SMU)
    test_healthy_card_sets_everything(am, mod, AM_SMU)
    test_pptable_is_uploaded_before_features_are_enabled(am, mod, AM_SMU)
    test_power_gating_features_are_not_enabled(am, mod, AM_SMU)
    test_smu13_path_is_untouched(am, mod, AM_SMU)
    test_a_refused_pptable_does_not_stop_the_device(am, mod, AM_SMU)
    test_a_pptable_timeout_still_propagates(am, mod, AM_SMU)
    test_board_data_is_patched_into_the_uploaded_table(am, mod, AM_SMU)
    test_a_feature_that_did_not_start_is_named(am, mod, AM_SMU, capture)
    test_all_features_running_is_quiet(am, mod, AM_SMU, capture)
    test_the_smu_is_told_it_is_on_ac(am, mod, AM_SMU)
    test_metrics_uses_smu11_field_names(am, mod, AM_SMU)
    test_metrics_reports_every_clock_domain(am, mod, AM_SMU)
    test_metrics_picks_v2_at_the_firmware_threshold(am, mod, AM_SMU)
    test_metrics_picks_v1_below_the_firmware_threshold(am, mod, AM_SMU)
    test_metrics_version_thresholds_match_amdgpu(am, mod, AM_SMU)
    test_metrics_does_not_ask_an_smu13_card_for_a_version(am, mod, AM_SMU)
    test_smu11_never_asks_for_fclk(am, mod, AM_SMU)
    test_smu11_pins_memory_to_the_asic_profiling_level(am, mod, AM_SMU)
    test_smu11_never_asks_for_the_top_memory_level(am, mod, AM_SMU)
    test_smu11_leaves_the_gfxclk_floor_open(am, mod, AM_SMU)
    test_smu11_raises_the_ceiling_before_the_floor(am, mod, AM_SMU)
    test_smu11_leaves_memory_alone_on_an_asic_with_no_known_pstate(am, mod, AM_SMU)
    test_smu11_teardown_does_not_touch_memory(am, mod, AM_SMU)
    test_smu11_teardown_asks_the_smu_nothing(am, mod, AM_SMU)
    test_smu11_profiling_pstates_match_amdgpu(am, mod, AM_SMU)
    test_smu13_clock_path_is_unchanged(am, mod, AM_SMU)

    print("\n" + "-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s) failed, {len(passes)} passed\n")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"PASSED: all {len(passes)} cases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
