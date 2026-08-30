#!/usr/bin/env python3
"""
Tests for the three defects that made a hung kernel unrecoverable on Navi 23.

The failure they add up to: one pathological BEAM candidate hangs, tinygrad calls recover(), and
recover() dies with `KeyError: 'regBIF_BX_PF0_RSMU_INDEX'` -- taking the whole search with it. The
handoff attributed that to "some register above the mapped MMIO range", which is not what happens.
The index is data, not an address:

  1. AM_GMC.flush_hdp read regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL and used the WHOLE dword, divided
     by four, as a register index. Only bits 2..18 of that register are the address. When the
     upper bits read back non-zero -- as they do on this card -- the index lands outside the
     aperture, AMDev.rreg falls through to the indirect RSMU window, and nbio 2.3 has no such
     register. flush_hdp is on the recover() path, so recovery could never work here.

  2. AM_SOC built its GFX interrupt-source table from constants named GFX_<major>__SRCID__*, and
     the autogen carries 9, 11 and 12 but nothing for 10. Every GFX interrupt on RDNA1/RDNA2
     therefore resolved to '', missed the benign list, and set is_err_state -- so a healthy device
     looked like a faulted one.

  3. That benign list named "CP_EOP_INTR", which is not an SRCID constant on any generation. The
     real suffix is CP_EOP_INTERRUPT. End-of-pipe is the ordinary completion interrupt for every
     dispatch, so on gfx9 and gfx11 too it was being treated as an error.

Everything here runs without a card. Fakes stand in for adev exactly as test_am_clocks.py does.
"""
import argparse
import inspect
import re
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


def build(tinygrad_path: Path):
    sys.path.insert(0, str(tinygrad_path))
    from tinygrad.runtime.autogen.am import am
    from tinygrad.runtime.support.am import ip as ip_mod
    from tinygrad.runtime.support.am.amdev import AMDev, AMRegister
    return am, ip_mod, AMDev, AMRegister


def navi23_ip_ver(am):
    """The IP versions the RX 6600 XT reports from its own discovery table."""
    return {am.GC_HWIP: (10, 3, 4), am.MP0_HWIP: (11, 0, 12), am.MP1_HWIP: (11, 0, 12),
            am.SMUIO_HWIP: (11, 0, 10), am.NBIO_HWIP: (2, 3, 0), am.SDMA0_HWIP: (5, 0, 2)}


class FakeAdev:
    """adev for the register paths: records writes, serves one canned register read."""
    devfmt = "usb:4-2"

    def __init__(self, ip_ver, remap_value=0):
        self.ip_ver = ip_ver
        self.remap_value = remap_value
        self.reg_writes: list[tuple[int, int]] = []
        self.regs: dict[str, object] = {}

    def rreg(self, reg):
        return self.remap_value

    def wreg(self, reg, val):
        self.reg_writes.append((reg, val))

    def reg(self, name):
        return self.regs[name]


def remap_register(AMRegister, adev, fields):
    """The real AMRegister, carrying the real field layout, over a fake adev."""
    return AMRegister(name="regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL", offset=301, segment=0,
                      fields=fields, bases={0: (0,)}, adev=adev)


def nbio_remap_fields(tinygrad_path: Path) -> dict:
    """The field layout the autogen register table actually declares for this register."""
    import functools

    from tinygrad.runtime.support.amd import AMDReg, import_asic_regs
    regs = import_asic_regs("nbio", (2, 3, 0), cls=functools.partial(AMDReg, bases={0: (0,) * 8}))
    reg = regs["regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"]
    return dict(reg.fields)


# ---------------------------------------------------------------- flush_hdp

def test_the_remap_register_has_a_narrow_address_field(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """The premise of the fix: the address is bits 2..18, not the whole dword.

    Locked down against the autogen table so that if AMD's own field layout ever changes, this
    says so rather than the driver silently going back to writing a wrong index.
    """
    fields = nbio_remap_fields(tinygrad_path)
    case("remap register declares an 'address' field", "address" in fields, True)
    case("address field is bits 2..18", fields.get("address"), (2, 18))


def test_flush_hdp_uses_the_address_field_not_the_whole_dword(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """The measured failure: upper bits set, so raw//4 indexes far outside the aperture."""
    fields = nbio_remap_fields(tinygrad_path)
    # Address 0x1a000 in bits 2..18, and rubbish in the bits above it -- which is what this card
    # returns and what raw//4 would drag into the index.
    raw = (0x1a000 << 2) | (0x5 << 19)
    adev = FakeAdev(navi23_ip_ver(am), remap_value=raw)
    adev.regs["regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"] = remap_register(AMRegister, adev, fields)

    gmc = ip_mod.AM_GMC(adev)
    gmc.flush_hdp()

    case("flush_hdp wrote exactly once", len(adev.reg_writes), 1)
    case("flush_hdp wrote the address field", adev.reg_writes[0][0], 0x1a000)
    case("flush_hdp wrote zero", adev.reg_writes[0][1], 0x0)
    check("flush_hdp did not use the raw dword", adev.reg_writes[0][0] != raw // 4,
          f"used raw//4 = {raw // 4:#x}, which is outside any MMIO aperture")


def test_flush_hdp_is_unchanged_when_the_upper_bits_are_clean(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """On an ASIC whose upper bits read back zero the old and new code agree, and must keep agreeing."""
    fields = nbio_remap_fields(tinygrad_path)
    raw = 0x1a000 << 2
    adev = FakeAdev(navi23_ip_ver(am), remap_value=raw)
    adev.regs["regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"] = remap_register(AMRegister, adev, fields)
    ip_mod.AM_GMC(adev).flush_hdp()
    case("clean upper bits give the same index as before", adev.reg_writes[0][0], raw // 4)


# ---------------------------------------------------- interrupt source names

def soc_for(am, ip_mod, ip_ver):
    soc = ip_mod.AM_SOC(FakeAdev(ip_ver))
    soc.init_sw()
    return soc


def gfx_srcs(am, soc):
    """The source-id -> name table AM_SOC built for the GFX clients."""
    return soc.ih_srcs_names.get(soc.gfx_ih_clients[0], {})


def test_gfx10_resolves_interrupt_source_names(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """The bug: no GFX_10_* constants exist, so every gfx10 interrupt was nameless."""
    srcs = gfx_srcs(am, soc_for(am, ip_mod, navi23_ip_ver(am)))
    check("gfx10 has a non-empty GFX source table", len(srcs) > 0, f"got {len(srcs)} entries")
    names = set(srcs.values())
    for want in ("CP_EOP_INTERRUPT", "SQ_INTERRUPT_ID", "CP_PRIV_REG_FAULT", "UTCL2_FAULT"):
        # UTCL2_FAULT is not a GFX srcid on gfx9/10; only assert the three that are.
        if want == "UTCL2_FAULT":
            continue
        check(f"gfx10 resolves {want}", want in names)


def test_gfx10_source_ids_match_the_kernel_header(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """gfx10 is mapped onto the gfx9 table, so the numbers have to be the same numbers.

    Values from the kernel's own drivers/gpu/drm/amd/include/ivsrcid/gfx/irqsrcs_gfx_10_1.h.
    """
    kernel_gfx_10_1 = {
        "CP_RB_INTERRUPT_PKT": 176, "CP_IB1_INTERRUPT_PKT": 177, "CP_IB2_INTERRUPT_PKT": 178,
        "CP_PM4_PKT_RSVD_BIT_ERROR": 180, "CP_EOP_INTERRUPT": 181, "CP_BAD_OPCODE_ERROR": 183,
        "CP_PRIV_REG_FAULT": 184, "CP_PRIV_INSTR_FAULT": 185, "CP_WAIT_MEM_SEM_FAULT": 186,
        "CP_CTX_EMPTY_INTERRUPT": 187, "CP_CTX_BUSY_INTERRUPT": 188,
        "CP_ME_WAIT_REG_MEM_POLL_TIMEOUT": 192, "CP_SIG_INCOMPLETE": 193, "CP_PREEMPT_ACK": 194,
        "CP_GPF": 195, "CP_GDS_ALLOC_ERROR": 196, "CP_ECC_ERROR": 197,
        "CP_COMPUTE_QUERY_STATUS": 199, "CP_VM_DOORBELL": 200, "CP_FUE_ERROR": 201,
        "RLC_STRM_PERF_MONITOR_INTERRUPT": 202, "GRBM_RD_TIMEOUT_ERROR": 232,
        "GRBM_REG_GUI_IDLE": 233, "SQ_INTERRUPT_ID": 239,
    }
    srcs = gfx_srcs(am, soc_for(am, ip_mod, navi23_ip_ver(am)))
    by_name = {v: k for k, v in srcs.items()}
    for name, srcid in kernel_gfx_10_1.items():
        case(f"gfx10 srcid {name}", by_name.get(name), srcid)


def test_gfx11_source_names_are_unchanged(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """The gfx10 alias must not disturb the generation that already worked."""
    ip_ver = dict(navi23_ip_ver(am))
    ip_ver[am.GC_HWIP] = (11, 0, 0)
    srcs = gfx_srcs(am, soc_for(am, ip_mod, ip_ver))
    names = set(srcs.values())
    check("gfx11 still resolves CP_EOP_INTERRUPT", "CP_EOP_INTERRUPT" in names)
    check("gfx11 table is the larger gfx11 one", len(srcs) > 24, f"got {len(srcs)} entries")


def test_every_benign_interrupt_name_actually_exists(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """The typo that made end-of-pipe look like a fault.

    interrupt_handler skips a small set of source names as benign. A name that matches no SRCID
    constant silently skips nothing and falls through to `is_err_state = True`, which is exactly
    what "CP_EOP_INTR" did -- on every generation, not just this one. Read the set out of the
    source rather than restating it, so a future edit is checked too.
    """
    src = inspect.getsource(ip_mod.AM_IH.interrupt_handler)
    m = re.search(r"if src_name in \{([^}]*)\}", src)
    check("found the benign source-name set in interrupt_handler", m is not None)
    if m is None:
        return
    benign = {s.strip().strip('"\'') for s in m.group(1).split(",") if s.strip()}
    check("the benign set is not empty", len(benign) > 0)

    all_srcid_names = {k[k.find("__SRCID__") + 9:] for k in dir(am) if "__SRCID__" in k}
    for name in sorted(benign):
        check(f"benign name {name} is a real SRCID", name in all_srcid_names,
              "matches no SRCID constant of any generation, so it never skips anything")

    navi23 = set(gfx_srcs(am, soc_for(am, ip_mod, navi23_ip_ver(am))).values())
    check("end-of-pipe is skippable on navi23", "CP_EOP_INTERRUPT" in navi23 & benign)


# ------------------------------------------------------------ the RSMU window

def test_an_asic_with_no_rsmu_window_says_so(am, ip_mod, AMDev, AMRegister, tinygrad_path):
    """A bare KeyError several frames from the out-of-range access is not a diagnosis."""
    dev = AMDev.__new__(AMDev)
    dev.mmio = [0] * 0x20000
    dev.ip_ver = navi23_ip_ver(am)
    # nbio 2.3 has no RSMU registers, so nothing named regBIF_BX_PF0_RSMU_* is on the device.
    try:
        dev.indirect_rreg(0x30000)
        got = "no exception"
    except KeyError as e:
        got = f"KeyError {e}"
    except RuntimeError as e:
        got = "RuntimeError"
        check("the error names the offending register", "0x30000" in str(e), str(e))
        check("the error names the aperture size", "0x20000" in str(e), str(e))
        check("the error names the nbio version", "2.3.0" in str(e), str(e))
    case("indirect access on an ASIC with no RSMU raises RuntimeError", got, "RuntimeError")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()

    if not (args.tinygrad / "tinygrad" / "runtime" / "support" / "am" / "ip.py").is_file():
        print(f"no tinygrad AM driver at {args.tinygrad}; nothing to test")
        return 2

    am, ip_mod, AMDev, AMRegister = build(args.tinygrad)
    print(f"tinygrad: {args.tinygrad}")
    a = (am, ip_mod, AMDev, AMRegister, args.tinygrad)

    test_the_remap_register_has_a_narrow_address_field(*a)
    test_flush_hdp_uses_the_address_field_not_the_whole_dword(*a)
    test_flush_hdp_is_unchanged_when_the_upper_bits_are_clean(*a)
    test_gfx10_resolves_interrupt_source_names(*a)
    test_gfx10_source_ids_match_the_kernel_header(*a)
    test_gfx11_source_names_are_unchanged(*a)
    test_every_benign_interrupt_name_actually_exists(*a)
    test_an_asic_with_no_rsmu_window_says_so(*a)

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
