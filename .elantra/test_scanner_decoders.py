#!/usr/bin/env python3
"""The four scanners each carry their own copy of the CAN decoders. This pins them together.

`lateral_report.py`, `eps_census.py`, `demand_decomp.py` and `ceiling_replay.py` are deployed
separately -- scp'd to the device one at a time, run from `/data` with nothing else beside them --
so each one duplicates the LKAS11 and MDPS12 bit layouts rather than importing a shared helper.
That is a deliberate deployment choice and a real drift risk: a bit position corrected in one file
and not the others produces two scanners that disagree about the same recorded drive, silently,
and the numbers in .elantra/ROAD-TEST-cn7-lateral.md come from more than one of them.

`lateral_report.py` is the reference, because it is the one with decoder tests of its own
(test_lateral_report.py) and the one the deployed watcher runs. Everything here asserts the others
extract identical values from identical bytes.

Run: python .elantra/test_scanner_decoders.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# hyundai_can.dbc, restated once so the packers below are not written from the same expression
# the decoders use -- a shared packer would agree with a shared bug.
LKAS11_TORQUE_SHIFT = 16
LKAS11_ACT_TOI_BIT = 27
TORQUE_OFFSET = 1024
MDPS_BITS = {"def": 11, "unavail": 12, "active": 13, "flt": 14, "failstat": 15}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def lkas11(counts: int, act_toi: int = 0) -> bytes:
    word = (((counts + TORQUE_OFFSET) & 0x7FF) << LKAS11_TORQUE_SHIFT) | ((act_toi & 1) << LKAS11_ACT_TOI_BIT)
    return word.to_bytes(4, "little") + bytes(4)


def mdps12(bits: dict | None = None, out_tq_raw: int = 2048, str_tq_raw: int = 2048) -> bytes:
    word = 0
    for name, bit in MDPS_BITS.items():
        if (bits or {}).get(name):
            word |= 1 << bit
    word |= (str_tq_raw & 0xFFF) << 40
    word |= (out_tq_raw & 0xFFF) << 52
    return word.to_bytes(8, "little")


def main() -> int:
    ref = load("lateral_report")
    census = load("eps_census")
    demand = load("demand_decomp")
    replay = load("ceiling_replay")

    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        if cond:
            print("  ok    " + label)
        else:
            print("  FAIL  " + label)
            failures.append(label)

    print("scanner decoder agreement")

    # --- LKAS11 torque, across the full signed range and both ActToi states ------------
    torque_ok = act_ok = True
    for counts in (0, 1, -1, 236, -236, 384, 409, -409, 435, -435, 1023, -1024):
        for toi in (0, 1):
            frame = lkas11(counts, toi)
            r_counts, r_toi = ref.decode_lkas11(frame)
            c_counts, c_toi = census.decode_lkas11(frame)
            d_counts = demand.decode_lkas11(frame)
            torque_ok = torque_ok and r_counts == c_counts == d_counts == counts
            act_ok = act_ok and r_toi == c_toi == toi
    check("LKAS11 torque decodes identically in lateral_report, eps_census and demand_decomp",
          torque_ok)
    check("CF_Lkas_ActToi decodes identically in lateral_report and eps_census", act_ok)

    check("all three reject a short LKAS11 frame rather than decoding it as zero",
          ref.decode_lkas11(b"\x00") is None
          and census.decode_lkas11(b"\x00") is None
          and demand.decode_lkas11(b"\x00") is None)

    # --- MDPS12 status bits -----------------------------------------------------------
    bits_ok = True
    for name in MDPS_BITS:
        frame = mdps12({name: 1})
        r = ref.decode_mdps_status(frame)
        c = census.decode_mdps12(frame)
        for other in MDPS_BITS:
            bits_ok = bits_ok and r[other] == c[other] == (1 if other == name else 0)
    check("every MDPS12 status bit agrees between lateral_report and eps_census", bits_ok)

    # A frame with no bits set must read clean in both -- the failure mode that matters is a
    # decoder that reports a fault on quiet traffic.
    quiet = mdps12()
    check("a quiet MDPS12 frame reports no fault bits anywhere",
          ref.eps_fault_bits(ref.MDPS12, 0, quiet) == frozenset()
          and not any(census.decode_mdps12(quiet)[b] for b in ("def", "unavail", "flt", "failstat")))

    # --- MDPS12 torque signals --------------------------------------------------------
    # 0.1 scale, -204.8 offset on CR_Mdps_OutTq; 0.01 / -20.48 on CR_Mdps_StrTq.
    tq_ok = True
    for raw in (0, 1024, 2048, 2200, 4095):
        frame = mdps12(out_tq_raw=raw, str_tq_raw=raw)
        c = census.decode_mdps12(frame)
        d = demand.decode_mdps12(frame)
        tq_ok = tq_ok and abs(c["out_tq"] - d["out_tq"]) < 1e-9
        tq_ok = tq_ok and abs(c["str_tq"] - d["str_tq"]) < 1e-9
        tq_ok = tq_ok and abs(c["out_tq"] - (raw * 0.1 - 204.8)) < 1e-9
        tq_ok = tq_ok and abs(c["str_tq"] - (raw * 0.01 - 20.48)) < 1e-9
    check("CR_Mdps_OutTq and CR_Mdps_StrTq agree between eps_census and demand_decomp, " +
          "and match the DBC scaling", tq_ok)

    check("both reject a short MDPS12 frame",
          census.decode_mdps12(b"\x00") is None and demand.decode_mdps12(b"\x00") is None)

    # --- the bus discrimination, which is the part that was actually wrong once --------
    # 0x251 also arrives on src 1 carrying something else; accepting it invented 599 phantom
    # faults per segment. Every scanner that reads faults must pin src 0.
    faulty = mdps12({"flt": 1})
    check("a fault on the real MDPS (src 0) is accepted",
          ref.eps_fault_bits(ref.MDPS12, 0, faulty) == frozenset({"flt"}))
    check("the same bits on src 1 are refused",
          ref.eps_fault_bits(ref.MDPS12, 1, faulty) == frozenset())
    check("eps_census pins the same constants",
          (census.MDPS12, census.LKAS11, census.POWERTRAIN_BUS, census.OP_TX_SRC)
          == (ref.MDPS12, ref.LKAS11, ref.POWERTRAIN_BUS, ref.OP_TX_SRC))
    check("demand_decomp pins the same constants",
          (demand.MDPS12, demand.LKAS11, demand.POWERTRAIN_BUS, demand.OP_TX_SRC)
          == (ref.MDPS12, ref.LKAS11, ref.POWERTRAIN_BUS, ref.OP_TX_SRC))

    # --- the limit constants the two counterfactual tools share -----------------------
    # ceiling_replay prices a rate change; demand_decomp measures what the rate costs today.
    # If they disagree about the rate, one of the two tables in the road-test doc is wrong.
    check("demand_decomp and ceiling_replay agree on STEER_DELTA_UP",
          demand.STEER_DELTA_UP == replay.PANDA_RATE_UP == 3)
    # The two tools now model DIFFERENT configurations on purpose, and that is the thing to
    # pin. demand_decomp reads recorded drives, every one of which ran the stock 50-count driver
    # window, so 50 is what reproduces them. ceiling_replay prices alternatives against what the
    # car runs TODAY, which is 100 after the CN7 allowance raise. Asserting they are equal --
    # which this check used to do -- would force one of them to lie about its own data.
    check("demand_decomp models the driver window the archive was recorded under",
          demand.STEER_DRIVER_ALLOWANCE == 50)
    check("ceiling_replay's default Limits still models that same recorded window",
          replay.Limits(409, 3, 7).STEER_DRIVER_ALLOWANCE == 50)
    check("ceiling_replay's grid starts at the ceiling and rates the drives actually ran",
          replay.GRID[0][:3] == (409, 3, 7) and demand.STEER_MAX == 409)
    check("...at the driver window the car runs today, not the one the archive ran",
          replay.GRID[0][3] == 100)
    check("...and the grid still carries the archive's window, so the delta stays visible",
          any(row[3] == 50 for row in replay.GRID))

    # --- the fields the report lines actually read -------------------------------------
    # The summary tables in .elantra/ROAD-TEST-cn7-lateral.md are printed straight out of these
    # dicts, so a renamed key is a broken report that still parses and still passes every test
    # above. Cheap to pin, and the print paths only ever run on the device.
    row = replay.replay([(5.0, 1.0, 0.0)] * 400, 409, 3, 7)
    for key in ("frames", "mean_applied", "mean_asked", "pct_of_ask_delivered",
                "pct_frames_at_ceiling", "p95_applied"):
        check(f"ceiling_replay result carries {key}", key in row)
    check("a 400-frame full-torque trace climbs at the rate limit and stops at the ceiling",
          row["frames"] == 400 and row["mean_applied"] > 0 and row["p95_applied"] == 409)

    band = demand.new_band()
    band["frames"] = 10
    band["applied_when_pinned"] = [236.0] * 10
    band["gain_samples"] = [1.9] * 10
    summary = demand.summarise(band)
    for key in ("frames", "pct_pinned_norm", "pct_limiter_binding",
                "pct_driver_opposing_past_allowance", "median_counts_lost_when_limited",
                "applied_when_pinned", "empirical_gain_lataccel_per_unit_torque",
                "pct_integrator_frozen"):
        check(f"demand_decomp summary carries {key}", key in summary)
    check("an empty band summarises without raising", demand.summarise(demand.new_band()) ==
          {"frames": 0})

    print()
    print("-" * 58)
    if failures:
        print(f"FAILED: {len(failures)} decoder disagreement(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: all four scanners decode the same bytes the same way")
    return 0


if __name__ == "__main__":
    sys.exit(main())
