#!/usr/bin/env python3
"""Tests for torque_projection.py -- the tool that says what flat 409 costs.

The tool's output is a number that gets reported as evidence, so the parts that can quietly
produce a WRONG number are what is tested here: the CAN decode, the schedule interpolation the
BEFORE chain is built on, the two limiter shapes, the drift guard on the copied constants, and
the aggregation (which mixes counters that must be summed with maxima that must not be).

Not tested: reading rlogs. That needs the device and a real log, and a fake one would only
prove the fake matches the parser.

    python .elantra/test_torque_projection.py
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("torque_projection",
                                                  HERE / "torque_projection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tp = _load()


class TestDecode(unittest.TestCase):
    def test_zero_is_the_offset_not_zero_bits(self):
        # CR_Lkas_StrToqReq is 16|11@1+ with offset -1024. A raw field of 1024 means zero
        # torque; reading the raw bits as signed would report every idle frame as -1024.
        raw = 1024 << 16
        self.assertEqual(tp.decode_lkas11(raw.to_bytes(8, "little"))[0], 0)

    def test_signed_extremes(self):
        for counts in (-1024, -409, -1, 0, 1, 409, 1023):
            raw = (counts + 1024) << 16
            self.assertEqual(tp.decode_lkas11(raw.to_bytes(8, "little"))[0], counts)

    def test_act_toi_bit(self):
        self.assertEqual(tp.decode_lkas11((1 << 27).to_bytes(8, "little"))[1], 1)
        self.assertEqual(tp.decode_lkas11((0).to_bytes(8, "little"))[1], 0)

    def test_short_frame_is_rejected_not_guessed(self):
        # A truncated frame must not be silently decoded as a small torque.
        self.assertIsNone(tp.decode_lkas11(b"\x00\x00\x00"))


class TestScheduleCeiling(unittest.TestCase):
    """The BEFORE chain is only meaningful if it reproduces the schedule that is on the car."""

    def test_breakpoints_and_plateaus(self):
        self.assertEqual(tp.schedule_ceiling(0.0), 409)
        self.assertEqual(tp.schedule_ceiling(8.0), 409)
        self.assertEqual(tp.schedule_ceiling(16.0), 384)
        self.assertEqual(tp.schedule_ceiling(40.0), 384)

    def test_midpoint_interpolates(self):
        self.assertEqual(tp.schedule_ceiling(12.0), 396)   # halfway between 409 and 384

    def test_monotonically_descending(self):
        vals = [tp.schedule_ceiling(v / 2) for v in range(80)]
        self.assertEqual(vals, sorted(vals, reverse=True))


class TestBands(unittest.TestCase):
    def test_boundaries_are_half_open_and_total(self):
        self.assertEqual(tp.band_of(0.0), "0-3")
        self.assertEqual(tp.band_of(2.999), "0-3")
        self.assertEqual(tp.band_of(3.0), "3-7")
        self.assertEqual(tp.band_of(18.0), "18+")
        self.assertEqual(tp.band_of(300.0), "18+")

    def test_negative_speed_has_no_band(self):
        # Not a crash and not band 0-3: an implausible reading must be visibly excluded.
        self.assertIsNone(tp.band_of(-1.0))


class TestStep(unittest.TestCase):
    """One frame of the car's own limiter, under each shape."""

    def setUp(self):
        self.before = tp.Limits(409)
        self.after = tp.Limits(tp.AFTER_FLAT)

    def test_rate_limit_binds_from_zero(self):
        # STEER_DELTA_UP is 3, so a full-demand frame from rest cannot exceed 3 counts.
        self.assertEqual(tp.step(1.0, 0, 0.0, self.after, 409, False), 3)

    def test_ceiling_binds_when_the_rate_limit_does_not(self):
        self.assertEqual(tp.step(1.0, 409, 0.0, self.after, 409, False), 409)
        self.assertEqual(tp.step(1.0, 384, 0.0, self.before, 384, True), 384)

    def test_the_two_shapes_differ_only_where_the_ceiling_does(self):
        # At 25 m/s the schedule is 384 and flat is 409. Seeded at the old ceiling with full
        # demand, BEFORE stays pinned and AFTER climbs by one rate step.
        b = tp.step(1.0, 384, 0.0, self.before, tp.schedule_ceiling(25.0), True)
        a = tp.step(1.0, 384, 0.0, self.after, tp.AFTER_FLAT, False)
        self.assertEqual(b, 384)
        self.assertEqual(a, 387)

    def test_below_the_lower_breakpoint_the_shapes_agree(self):
        # Both ceilings are 409 under 8 m/s, so the change must be a no-op there.
        for prev in (0, 100, 300, 409):
            b = tp.step(0.9, prev, 0.0, self.before, tp.schedule_ceiling(5.0), True)
            a = tp.step(0.9, prev, 0.0, self.after, tp.AFTER_FLAT, False)
            self.assertEqual(b, a, f"seeded at {prev}")

    def test_driver_override_ramps_the_command_down_not_off(self):
        # driver_max_torque = STEER_MAX + (50 + driver)*2, clamped at 0, so heavy opposition
        # sets the target to 0. It does NOT get there in one frame: STEER_DELTA_DOWN is 7, and
        # the rate limiter floors each step at prev - 7. Asserting an instant zero here would
        # be asserting a car that does not exist, and would hide a broken rate limiter.
        self.assertEqual(tp.step(1.0, 409, -300.0, self.after, 409, False), 402)
        self.assertEqual(tp.step(1.0, 10, -300.0, self.after, 409, False), 3)
        self.assertEqual(tp.step(1.0, 3, -300.0, self.after, 409, False), 0)

    def test_full_yield_point_moves_with_the_ceiling(self):
        # The quantified cost of the raise, and the one number that actually moves.
        #
        # Override starts REDUCING authority at driver torque < -STEER_DRIVER_ALLOWANCE (-50)
        # for both ceilings -- that threshold does not scale. What scales is the point of FULL
        # yield, where driver_max_torque = M + (50 + d)*2 reaches zero: d <= -M/2 - 50, i.e.
        # -242 at 384 and -254.5 at 409.
        #
        # Seeded from 0 so the rate limiter is not what is being measured; from a saturated
        # previous frame the ramp-down floor hides the effect entirely.
        for steer_max, yield_at in ((384, -242.0), (409, -254.5)):
            lim = tp.Limits(steer_max)
            self.assertGreater(tp.step(1.0, 0, yield_at + 1.0, lim, steer_max, False), 0,
                               f"{steer_max}: should not have fully yielded at {yield_at + 1}")
            self.assertEqual(tp.step(1.0, 0, yield_at - 1.0, lim, steer_max, False), 0,
                             f"{steer_max}: should have fully yielded at {yield_at - 1}")

    def test_the_raise_keeps_steering_where_the_stock_ceiling_had_given_up(self):
        # The same test stated as the driver would feel it: at one opposing torque between the
        # two yield points, the 384 car has stopped steering and the 409 car has not.
        between = -243.0
        self.assertEqual(tp.step(1.0, 0, between, tp.Limits(384), 384, False), 0)
        self.assertGreater(tp.step(1.0, 0, between, tp.Limits(409), 409, False), 0)


class TestParamDriftGuard(unittest.TestCase):
    def test_it_passes_against_the_real_params(self):
        tp.assert_params_match()

    def test_it_fails_when_the_copy_drifts(self):
        # The error path is the whole point of the guard: a silent copy is worse than none.
        original = tp.RATE_UP
        tp.RATE_UP = 99
        try:
            with self.assertRaises(SystemExit):
                tp.assert_params_match()
        finally:
            tp.RATE_UP = original


class TestMerge(unittest.TestCase):
    """Counters must sum; maxima must not. Getting this backwards inflates the headline."""

    def _rec(self, frames, changed, max_delta, max_abs, factory_max):
        b = tp.new_band()
        b.update({"frames": frames, "changed": changed, "max_abs_delta": max_delta,
                  "before_max_abs": max_abs, "after_max_abs": max_abs,
                  "delta_hist": {"3": changed}})
        return {
            "seg": "x", "error": None,
            "bands": {name: (b if name == "0-3" else tp.new_band()) for name in tp.BAND_NAMES},
            "factory": {"frames": 10, "nonzero": 0, "max_abs": factory_max, "act_toi": 0},
            "tx": {"frames": 5, "max_abs": 409},
            "valid": {"paired": 7, "exact_lag0": 1, "exact_lag1": 6,
                      "within1_lag1": 6, "max_mismatch": 2, "sum_mismatch": 3},
        }

    def test_sums_and_maxima(self):
        bands, factory, tx, valid = tp.merge([self._rec(100, 40, 5, 300, 0),
                                              self._rec(50, 10, 9, 200, 7)])
        self.assertEqual(bands["0-3"]["frames"], 150)
        self.assertEqual(bands["0-3"]["changed"], 50)
        self.assertEqual(bands["0-3"]["max_abs_delta"], 9)      # max, not 14
        self.assertEqual(bands["0-3"]["before_max_abs"], 300)   # max, not 500
        self.assertEqual(bands["0-3"]["delta_hist"]["3"], 50)
        self.assertEqual(factory["frames"], 20)
        self.assertEqual(factory["max_abs"], 7)                 # max, not 7
        self.assertEqual(valid["paired"], 14)
        self.assertEqual(valid["max_mismatch"], 2)              # max, not 4
        self.assertEqual(tx["max_abs"], 409)

    def test_errored_segments_contribute_nothing(self):
        bad = self._rec(999, 999, 999, 999, 999)
        bad["error"] = "Boom: unreadable"
        bands, factory, _, _ = tp.merge([self._rec(100, 40, 5, 300, 0), bad])
        self.assertEqual(bands["0-3"]["frames"], 100)
        self.assertEqual(factory["frames"], 10)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
