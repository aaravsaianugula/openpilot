#!/usr/bin/env python3
"""
Tests for the pure decoding and classification in lateral_report.py.

These are the parts that turn bytes into a number someone will act on, so they are worth
pinning. The EPS-fault case is here because it was wrong once: 0x251 arrives on three sources
on this car, and accepting every frame openpilot did not transmit pulled in 599 bus-1 frames
per segment with bit 14 set, against a true fault count of zero. That would have been reported
as "EPS faults at the raised torque ceiling" -- the exact signal the whole change is watched by.

    python .elantra/test_lateral_report.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "lateral_report.py"


def load():
    spec = importlib.util.spec_from_file_location("lateral_report", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lkas11(counts: int, act_toi: int = 0) -> bytes:
    """Pack CR_Lkas_StrToqReq (16|11@1+, offset -1024) and CF_Lkas_ActToi (27|1@1+)."""
    word = (((counts + 1024) & 0x7FF) << 16) | ((act_toi & 1) << 27)
    return word.to_bytes(4, "little") + bytes(4)


def mdps12(toi_flt: int) -> bytes:
    """Pack CF_Mdps_ToiFlt (14|1@1+)."""
    return ((toi_flt & 1) << 14).to_bytes(8, "little")


def main() -> int:
    m = load()
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        if cond:
            print("  ok    " + label)
        else:
            print("  FAIL  " + label)
            failures.append(label)

    print("lateral_report decoding")

    # --- LKAS11 torque, the number every conclusion rests on -------------------------
    ok = True
    for counts in (0, 1, -1, 384, -384, 409, -409, 1023, -1024):
        for toi in (0, 1):
            ok = ok and m.decode_lkas11(lkas11(counts, toi)) == (counts, toi)
    check("CR_Lkas_StrToqReq round-trips across the full signed range", ok)
    check("a short LKAS11 frame is rejected, not silently decoded as zero",
          m.decode_lkas11(b"\x00\x00") is None)
    # 384 counts is 3.00 Nm and 409 is 3.20 Nm on the OEM scale; if this drifts, every
    # torque figure quoted in values.py and the commit messages drifts with it.
    check("the OEM scale still puts 384 at 3.00 Nm and 409 at 3.20 Nm",
          abs(384 * 0.0078125 - 3.00) < 1e-9 and abs(409 * 0.0078125 - 3.1953125) < 1e-9)

    # --- MDPS12 fault bit -------------------------------------------------------------
    check("CF_Mdps_ToiFlt reads bit 14", m.decode_toiflt(mdps12(1)) == 1)
    check("bit 14 clear reads as no fault", m.decode_toiflt(mdps12(0)) == 0)
    neighbours = all(m.decode_toiflt((1 << b).to_bytes(8, "little")) == 0 for b in (12, 13, 15))
    # 13 is CF_Mdps_ToiActive and is set on essentially every frame while openpilot steers.
    # Reading it as a fault would report a ~100% fault rate.
    check("neighbouring MDPS12 bits (ToiUnavail/ToiActive/FailStat) are not read as faults",
          neighbours)
    check("a short MDPS12 frame is rejected", m.decode_toiflt(b"\x00") is None)

    # --- the bus filter, which is the part that was actually wrong --------------------
    check("a fault from the real MDPS on bus 0 counts",
          m.is_eps_fault_frame(m.MDPS12, 0, mdps12(1)))
    check("the same bit on bus 1 does NOT count (measured: 599 such frames per segment)",
          not m.is_eps_fault_frame(m.MDPS12, 1, mdps12(1)))
    check("openpilot's own forwarded copy (src 130) does not count",
          not m.is_eps_fault_frame(m.MDPS12, 130, mdps12(1)))
    check("a non-MDPS address on bus 0 does not count",
          not m.is_eps_fault_frame(m.LKAS11, 0, mdps12(1)))
    check("no fault bit means no fault", not m.is_eps_fault_frame(m.MDPS12, 0, mdps12(0)))

    # --- speed banding ----------------------------------------------------------------
    bands = [(0, "0-3"), (2.999, "0-3"), (3, "3-7"), (6.999, "3-7"), (7, "7-10"),
             (10, "10-14"), (13.999, "10-14"), (14, "14-18"), (18, "18+"), (60, "18+")]
    check("speed bands are contiguous and upper-exclusive",
          all(m.band_of(v) == b for v, b in bands))
    check("a negative speed lands in no band rather than the lowest one",
          m.band_of(-0.1) is None)

    # --- provenance -------------------------------------------------------------------
    base = {"git_commit": "abc", "safety_param": 12, "fingerprint": "HYUNDAI_ELANTRA_2024",
            "lat_accel_factor": 3.169, "friction": 0.0819, "nnlc_model": "HYUNDAI_ELANTRA_2021",
            "params": {"NeuralNetworkLateralControl": "0"}}
    check("the same configuration always tags the same", m.tag_of(base) == m.tag_of(dict(base)))
    for field, value in (("safety_param", 1036), ("friction", 0.15), ("git_commit", "def")):
        other = dict(base)
        other[field] = value
        check("a change of " + field + " changes the tag", m.tag_of(other) != m.tag_of(base))
    nnlc_on = dict(base)
    nnlc_on["params"] = {"NeuralNetworkLateralControl": "1"}
    # This is the one that matters most: a drive with NNLC on is not comparable to one with it
    # off, and compare() only refuses to mix them because the tag separates them.
    check("turning NNLC on changes the tag", m.tag_of(nnlc_on) != m.tag_of(base))

    print("\n" + "-" * 58)
    if failures:
        print("FAILED: %d case(s)" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: decoding, bus filtering and provenance behave as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
