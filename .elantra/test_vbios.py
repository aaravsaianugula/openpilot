#!/usr/bin/env python3
"""
Tests for atom_board_data: pulling the 292 board bytes out of a VBIOS image.

Those bytes are the voltage-regulator mapping and the current-telemetry calibration. AMD ships
them zeroed in the SMU firmware and expects the driver to fill them from the card's own VBIOS.
On a Navi 23 that gap is why the SMU accepts every clock request and stays at 497 MHz: GFXCLK is
driven by a DFLL, a DFLL's frequency follows the voltage the VR supplies, and with no VR mapping
there is no voltage to raise.

So a wrong 292 bytes is worse than none -- it would command an unknown regulator. Every test here
is about refusing to produce a plausible answer. There is no "best effort" path under test because
there must not be one.

The traversal is amdgpu's, and the constants are checked against it:
  rom[0x00:0x02]     55 AA                     PCI option ROM signature
  rom[0x30:0x3A]     " 761295520"              ATI signature, leading space included
  rom[0x48]          u16 -> atom_rom_header    OFFSET_TO_ATOM_ROM_HEADER_POINTER
  header + 0x04      "ATOM"                    ATOM_ROM_MAGIC_PTR
  header + 0x20      u16 -> master data table  ATOM_ROM_DATA_PTR
  master + 4 + 2*2   u16 -> smc_dpm_info       +4 skips the master's own common header,
                                               index 2 from get_index_into_master_table
  smc + 0            u16 structuresize == 296  sizeof(atom_smc_dpm_info_v4_9)
  smc + 2, +3        frev 4, crev 9
  smc + 4 .. +296    the 292 bytes             -> PPTable_t + 1344

Usage:
    python .elantra/test_vbios.py                      # uses ./tinygrad_repo
    python .elantra/test_vbios.py --tinygrad <path>
"""

from __future__ import annotations

import argparse
import struct
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


def raises(name: str, fn, expect_substr: str) -> None:
    """The message matters as much as the refusal -- these are read by someone holding a card."""
    try:
        fn()
        failures.append(f"{name}: returned instead of raising")
    except ValueError as e:
        if expect_substr.lower() in str(e).lower():
            passes.append(name)
        else:
            failures.append(f"{name}: raised {e!r}, expected mention of {expect_substr!r}")
    except Exception as e:  # a non-ValueError is itself the failure
        failures.append(f"{name}: raised {type(e).__name__}: {e}, expected ValueError")


# Layout of the synthetic image. Nothing here overlaps, and every region is far enough apart that
# an off-by-one in the parser lands in filler rather than in another table.
HDR_AT, MASTER_AT, SMC_AT, ROM_LEN = 0x200, 0x400, 0x600, 0x1000
BOARD_LEN, SMC_SIZE, BOARD_OFF = 292, 296, 4


def build_rom(*, sig=b"\x55\xaa", ati=b" 761295520", hdr_ptr=HDR_AT, atom_magic=b"ATOM",
              master_ptr=MASTER_AT, smc_ptr=SMC_AT, smc_size=SMC_SIZE, frev=4, crev=9,
              board: bytes | None = None, truncate: int | None = None) -> bytes:
    """A minimal but structurally real VBIOS. Every knob is a thing the parser must reject."""
    if board is None:
        # Distinctive, non-zero, and position-encoding, so a shifted copy is visible at a glance.
        board = bytearray((i * 7 + 1) & 0xFF for i in range(BOARD_LEN))
        # Values this card actually reports, so the fixture matches the real thing: a zero
        # VddGfxVrMapping is a legal rail index (soc is 2, mem0 1, mem1 3), and GfxMaxCurrent
        # is the field whose absence is the bug being fixed.
        board[132] = 0x00                              # VddGfxVrMapping
        board[140:142] = (260).to_bytes(2, "little")   # GfxMaxCurrent, amps
        board[192:196] = (0xFF).to_bytes(4, "little")  # MemoryChannelEnabled
        board = bytes(board)
    assert len(board) == BOARD_LEN
    rom = bytearray(b"\x00" * ROM_LEN)
    rom[0:2] = sig
    rom[2] = ROM_LEN >> 9                                 # image size in 512-byte blocks
    rom[0x30:0x30 + len(ati)] = ati
    struct.pack_into("<H", rom, 0x48, hdr_ptr)
    if hdr_ptr:
        rom[hdr_ptr + 4:hdr_ptr + 8] = atom_magic
        struct.pack_into("<H", rom, hdr_ptr + 0x20, master_ptr)
    if master_ptr:
        struct.pack_into("<H", rom, master_ptr, 74)       # the master table's own structuresize
        rom[master_ptr + 2], rom[master_ptr + 3] = 2, 1   # frev/crev of the master table
        # index 0 and 1 get decoys: a parser that forgets the +4, or that uses index 0, lands here.
        struct.pack_into("<H", rom, master_ptr + 4 + 0 * 2, 0xDEAD)
        struct.pack_into("<H", rom, master_ptr + 4 + 1 * 2, 0xBEEF)
        struct.pack_into("<H", rom, master_ptr + 4 + 2 * 2, smc_ptr)
    if smc_ptr:
        struct.pack_into("<H", rom, smc_ptr, smc_size)
        rom[smc_ptr + 2], rom[smc_ptr + 3] = frev, crev
        rom[smc_ptr + BOARD_OFF:smc_ptr + BOARD_OFF + BOARD_LEN] = board
    return bytes(rom[:truncate] if truncate is not None else rom)


def reader(image):
    """`read(offset, nbytes)` over a bytes image, clamped like a real short read would be."""
    image = bytes(image)

    def read(off, n):
        return image[off:off + n]
    return read


def counting_reader(image):
    """Same, but records every (offset, nbytes) so the seek count can be asserted."""
    image = bytes(image)
    calls = []

    def read(off, n):
        calls.append((off, n))
        return image[off:off + n]
    read.calls = calls
    return read


def test_happy_path(atom_board_data):
    rom = build_rom()
    got = atom_board_data(reader(rom))
    case("returns exactly 292 bytes", len(got), BOARD_LEN)
    case("returns smc_dpm_info + 4 .. + 296", got, rom[SMC_AT + BOARD_OFF:SMC_AT + SMC_SIZE])
    case("GfxMaxCurrent survives the copy", int.from_bytes(got[140:142], "little"), 260)
    check("accepts a memoryview as well as bytes", atom_board_data(reader(memoryview(rom))) == got)


def test_index_two_not_zero_or_one(atom_board_data):
    """The decoys at index 0 and 1 point at nothing; landing on them must not look like success."""
    rom = build_rom()
    got = atom_board_data(reader(rom))
    check("did not follow the index-0 decoy", got != rom[0xDEAD + 4:0xDEAD + SMC_SIZE])
    check("did not follow the index-1 decoy", got != rom[0xBEEF + 4:0xBEEF + SMC_SIZE])


def test_signature_checks(atom_board_data):
    raises("bad option-ROM signature", lambda: atom_board_data(reader(build_rom(sig=b"\x4d\x5a"))), "55aa")
    raises("missing ATI signature", lambda: atom_board_data(reader(build_rom(ati=b"nonsense12"))), "ATI signature")
    raises("no ATOM magic", lambda: atom_board_data(reader(build_rom(atom_magic=b"XXXX"))), "ATOM magic")


def test_byte_swapped_image_names_the_read(atom_board_data):
    """MOTA on a PCIe card means the reader transposed u16s. Say that, do not silently cope."""
    raises("MOTA blames the read path", lambda: atom_board_data(reader(build_rom(atom_magic=b"MOTA"))),
           "byte-swapping")


def test_zero_pointers(atom_board_data):
    raises("zero ROM header pointer", lambda: atom_board_data(reader(build_rom(hdr_ptr=0))), "0x48")
    raises("zero master table pointer", lambda: atom_board_data(reader(build_rom(master_ptr=0))), "master data table")
    raises("absent smc_dpm_info", lambda: atom_board_data(reader(build_rom(smc_ptr=0))), "no smc_dpm_info")


def test_revision_gate(atom_board_data):
    """v4_10 has no I2cControllers at the front; reading it as v4_9 would be silent garbage."""
    raises("rejects crev 10", lambda: atom_board_data(reader(build_rom(crev=10))), "v4_9")
    raises("rejects frev 5", lambda: atom_board_data(reader(build_rom(frev=5))), "v4_9")
    raises("rejects a wrong structuresize", lambda: atom_board_data(reader(build_rom(smc_size=304))), "v4_9")


def test_refuses_empty_board_data(atom_board_data):
    """A table of zeros must not be blitted in and look like the fix landed."""
    raises("all-zero board data", lambda: atom_board_data(reader(build_rom(board=bytes(BOARD_LEN)))),
           "not the missing data")
    # But a zero VddGfxVrMapping on its own is fine. It is a rail index, and this card really does
    # report gfx 0 alongside soc 2, mem0 1, mem1 3 -- guarding on it rejected valid board data.
    b = bytearray(BOARD_LEN)
    b[140:142] = (260).to_bytes(2, "little")
    b[192:196] = (0xFF).to_bytes(4, "little")
    got = atom_board_data(reader(build_rom(board=bytes(b))))
    check("a zero VddGfxVrMapping is accepted", got[132] == 0)
    check("GfxMaxCurrent is what gates it", int.from_bytes(got[140:142], "little") == 260)


def test_truncation_is_an_error_not_a_crash(atom_board_data):
    # Every truncation now surfaces through the one reader guard, which is the point of routing
    # all access through rd(): there is no second path that can read past the end quietly.
    for n, at in (("before 0x49", 0x20), ("before the ROM header", 0x210),
                  ("before the master table", 0x410), ("mid smc_dpm_info", SMC_AT + 100)):
        raises(f"truncated {n}", lambda at=at: atom_board_data(reader(build_rom(truncate=at))),
               "wanted")


def test_matches_the_real_pptable_span(atom_board_data, amdev):
    """The kernel's BUILD_BUG_ON asserts both spans are equal; assert the same thing here."""
    case("PPTABLE_BOARD_LEN is the copy length", amdev.PPTABLE_BOARD_LEN, BOARD_LEN)
    case("source span is sizeof - offsetof", amdev.SMC_DPM_INFO_V4_9[0] - amdev.SMC_DPM_INFO_BOARD_OFF,
         amdev.PPTABLE_BOARD_LEN)
    case("destination starts at PPTable_t.I2cControllers", amdev.PPTABLE_BOARD_OFF, 1344)
    case("destination ends at end of BoardReserved",
         amdev.PPTABLE_BOARD_OFF + amdev.PPTABLE_BOARD_LEN, 1636)
    case("smc_dpm_info is master-list index 2", amdev.SMC_DPM_INFO_INDEX, 2)


def test_reads_are_few_and_bounded(atom_board_data):
    """The reason this takes a reader at all: on a USB card every access is a round trip, so
    pulling 64 KiB to reach four small tables would cost thousands of them per device open."""
    r = counting_reader(build_rom())
    atom_board_data(r)
    total = sum(n for _, n in r.calls)
    check("fewer than 16 reads", len(r.calls) < 16, f"made {len(r.calls)}")
    check("under 512 bytes moved", total < 512, f"moved {total} bytes")
    check("never seeks past 64 KiB", all(o + n <= 0x10000 for o, n in r.calls),
          f"max end {max(o + n for o, n in r.calls):#x}")


def test_a_short_read_is_an_error(atom_board_data):
    """A truncated USB read that silently returns short is a live failure mode on this path, and
    a short read gives a plausible-looking wrong offset rather than an obvious crash."""
    def short(_off, _n):
        return b""
    raises("an empty read is refused", lambda: atom_board_data(short), "got 0")

    full = build_rom()

    def truncating(off, n):
        return full[off:off + n][:max(0, n - 1)]
    raises("a one-byte-short read is refused", lambda: atom_board_data(truncating), "got")


class FakeRomAdev:
    """A card whose SMUIO ROM window behaves like the real one: reading DATA advances INDEX.

    Registers other than the ROM pair are modelled too, so a wrong version map reads and writes
    the WRONG register and the test can prove the guard catches it rather than the card doing so.
    """
    SMUIO_BASE = 0x16800

    def __init__(self, image: bytes, ver=(11, 0, 10), *, map_shift=1, broken=False):
        self.image, self.ip_ver = image, {24: ver}          # am.SMUIO_HWIP == 24
        self.regs_offset = {24: {0: (self.SMUIO_BASE,)}}
        # The real card's map: *_6 shifts ROM_INDEX/DATA one dword up from the *_0 map.
        self.rom_index = self.SMUIO_BASE + (0xe5 if map_shift else 0xe4)
        self.rom_data = self.SMUIO_BASE + (0xe6 if map_shift else 0xe5)
        self.broken, self.idx, self.writes = broken, 0, []

    def rreg(self, reg):
        if reg == self.rom_index:
            return self.idx
        if reg == self.rom_data and not self.broken:
            v = int.from_bytes(self.image[self.idx:self.idx + 4].ljust(4, bytes(1)), "little")
            self.idx += 4
            return v
        return 0                                            # any other register reads as zero

    def wreg(self, reg, val):
        self.writes.append((reg, val))
        if reg == self.rom_index:
            self.idx = val


def test_rom_reader_reads_the_image(vbios_reader, atom_board_data):
    rom = build_rom()
    dev = FakeRomAdev(rom)
    read = vbios_reader(dev)
    case("reads a dword-aligned range", read(0, 4), rom[0:4])
    case("reads an unaligned range", read(0x31, 7), rom[0x31:0x38])
    case("reads across many dwords", read(0x600, 296), rom[0x600:0x600 + 296])
    case("the full walk works through it", atom_board_data(vbios_reader(FakeRomAdev(rom))),
         atom_board_data(reader(rom)))


def test_rom_reader_refuses_an_unknown_smuio(vbios_reader):
    """An unknown SMUIO version must not be guessed at -- the two maps differ by one dword and the
    wrong one writes into the ROM clock-gating control."""
    try:
        vbios_reader(FakeRomAdev(build_rom(), ver=(11, 0, 99)))
        failures.append("unknown SMUIO version: returned instead of raising")
    except RuntimeError as e:
        check("unknown SMUIO version is refused", "11.0.99" in str(e), str(e))


def test_rom_reader_probes_before_it_writes(vbios_reader):
    """The identification probe is write-free by design: if the map is wrong we must find out
    before ROM_INDEX has been written to whatever register that actually is."""
    dev = FakeRomAdev(build_rom(), map_shift=0)              # card is *_0, code will assume *_6
    try:
        vbios_reader(dev)
        failures.append("wrong register map: returned instead of raising")
    except RuntimeError as e:
        check("a wrong register map is caught", "not streaming" in str(e), str(e))
    check("nothing was written before the probe failed", dev.writes == [], f"wrote {dev.writes}")


def test_rom_reader_catches_a_dead_window(vbios_reader):
    dev = FakeRomAdev(build_rom(), broken=True)
    try:
        vbios_reader(dev)
        failures.append("dead ROM window: returned instead of raising")
    except RuntimeError as e:
        check("a dead window is refused", "not streaming" in str(e), str(e))
    check("nothing written to a dead window", dev.writes == [], f"wrote {dev.writes}")


def test_navi23_lands_on_the_documented_registers(vbios_reader, amdev):
    """SMUIO 11.0.10 -> smuio_11_0_6_offset.h -> ROM_INDEX 0x168E5, ROM_DATA 0x168E6."""
    case("11.0.10 uses the shifted map", amdev.SMUIO_ROM_REGS[(11, 0, 10)], (0xe5, 0xe6))
    case("11.0.7 uses the unshifted map", amdev.SMUIO_ROM_REGS[(11, 0, 7)], (0xe4, 0xe5))
    dev = FakeRomAdev(build_rom())
    vbios_reader(dev)(0, 4)
    case("wrote only ROM_INDEX", [r for r, _ in dev.writes], [0x168E5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()
    if not (args.tinygrad / "tinygrad" / "runtime" / "support" / "am" / "amdev.py").is_file():
        print(f"no tinygrad AM driver at {args.tinygrad}; nothing to test")
        return 2
    sys.path.insert(0, str(args.tinygrad))
    from tinygrad.runtime.support.am import amdev
    print(f"tinygrad: {args.tinygrad}")

    # A test that raises is a failed test, not a failed run. Without this a regression in the
    # happy path aborts the suite before the refusal cases -- which are the ones that matter --
    # ever run, and the output looks like a crash rather than a result.
    def run(fn, *a):
        try:
            fn(*a)
        except Exception as e:  # reporting it is the point
            failures.append(f"{fn.__name__}: raised {type(e).__name__}: {e}")

    run(test_happy_path, amdev.atom_board_data)
    run(test_index_two_not_zero_or_one, amdev.atom_board_data)
    run(test_signature_checks, amdev.atom_board_data)
    run(test_byte_swapped_image_names_the_read, amdev.atom_board_data)
    run(test_zero_pointers, amdev.atom_board_data)
    run(test_revision_gate, amdev.atom_board_data)
    run(test_refuses_empty_board_data, amdev.atom_board_data)
    run(test_truncation_is_an_error_not_a_crash, amdev.atom_board_data)
    run(test_matches_the_real_pptable_span, amdev.atom_board_data, amdev)
    run(test_reads_are_few_and_bounded, amdev.atom_board_data)
    run(test_a_short_read_is_an_error, amdev.atom_board_data)
    run(test_rom_reader_reads_the_image, amdev.vbios_reader, amdev.atom_board_data)
    run(test_rom_reader_refuses_an_unknown_smuio, amdev.vbios_reader)
    run(test_rom_reader_probes_before_it_writes, amdev.vbios_reader)
    run(test_rom_reader_catches_a_dead_window, amdev.vbios_reader)
    run(test_navi23_lands_on_the_documented_registers, amdev.vbios_reader, amdev)

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
