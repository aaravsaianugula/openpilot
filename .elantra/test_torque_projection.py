#!/usr/bin/env python3
"""Tests for torque_projection.py -- the tool that prices a change to the steering ceiling.

The tool's output is a number that gets reported as evidence, so the parts that can quietly
produce a WRONG number are what is tested here: the CAN decode, the two limiter shapes, the
EPS refusal, the drift guard on the copied constants, and the aggregation (which mixes
counters that must be summed with maxima that must not be).

BEFORE is the flat 409 the recorded drives actually ran, which is what makes the BEFORE chain
a genuine one-frame prediction test. AFTER is a flat candidate, defaulting to the same 409.
The speed-schedule interpolation this file used to test is gone: the schedule was driven, the
EPS faulted at 500 and at 450, and the tool now refuses any candidate above 409.

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


class TestEpsCeiling(unittest.TestCase):
    """The refusal is the most important behaviour in the tool, so it is tested first.

    500 and 450 were both driven on this car and both threw an EPS fault. A projection above
    409 would be arithmetic about counts the MDPS will not deliver -- a confident-looking
    number that argues for a change already known to break steering. The tool has to REFUSE,
    not warn, because a warning in a long report is a number someone quotes later.
    """

    def test_the_default_candidate_is_the_measured_limit(self):
        self.assertEqual(tp.EPS_CEILING, 409)
        self.assertEqual(tp.DEFAULT_CANDIDATE, tp.EPS_CEILING)

    def test_a_candidate_above_the_eps_ceiling_is_refused(self):
        original = tp.CANDIDATE_CEILING
        for bad in (410, 450, 500):
            tp.CANDIDATE_CEILING = bad
            try:
                with self.assertRaises(SystemExit) as caught:
                    tp.assert_params_match()
                self.assertIn(str(tp.EPS_CEILING), str(caught.exception))
            finally:
                tp.CANDIDATE_CEILING = original

    def test_panda_is_not_the_check(self):
        # 450 sits comfortably under panda's 512, so a tool that gated on panda would have
        # projected it happily. That is exactly the mistake this refusal exists to prevent:
        # panda's ceiling is 103 counts above the band the car actually fails in.
        self.assertLess(tp.EPS_CEILING, tp.PANDA_CEILING)
        self.assertLess(450, tp.PANDA_CEILING)

    def test_a_lower_candidate_is_still_allowed(self):
        # Lowering is a live question -- the refusal must not be a blanket ban on the tool.
        original = tp.CANDIDATE_CEILING
        tp.CANDIDATE_CEILING = 384
        try:
            tp.assert_params_match()
        finally:
            tp.CANDIDATE_CEILING = original


class TestBands(unittest.TestCase):
    def test_boundaries_are_half_open_and_total(self):
        self.assertEqual(tp.band_of(0.0), "0-3")
        self.assertEqual(tp.band_of(2.999), "0-3")
        self.assertEqual(tp.band_of(3.0), "3-7")
        self.assertEqual(tp.band_of(18.0), "18+")
        self.assertEqual(tp.band_of(300.0), "18+")

    def test_the_low_speed_boundaries_are_band_boundaries(self):
        # "under 20 mph" has to be readable straight off the table. If 8.94 falls inside a
        # band, every number for the low-speed regime is contaminated by frames from above it.
        # These splits are inherited from the retired schedule's breakpoints and kept on
        # purpose, so reports from before and after that change stay comparable.
        self.assertEqual(tp.band_of(8.93), "7-8.94")
        self.assertEqual(tp.band_of(8.94), "8.94-13.41")
        self.assertEqual(tp.band_of(13.40), "8.94-13.41")
        self.assertEqual(tp.band_of(13.41), "13.41-18")

    def test_the_sub_20mph_bands_cover_the_low_speed_regime_without_gaps(self):
        below = [n for n in tp.BAND_NAMES if n in ("0-3", "3-7", "7-8.94")]
        self.assertEqual(len(below), 3)
        edges = [tuple(float(x) for x in n.split("-")) for n in below]
        self.assertEqual(edges[0][0], 0.0)
        self.assertEqual(edges[-1][1], 8.94)
        for (_, hi), (lo, _) in zip(edges, edges[1:], strict=False):
            self.assertEqual(hi, lo, "a gap here silently drops frames from the report")

    def test_negative_speed_has_no_band(self):
        # Not a crash and not band 0-3: an implausible reading must be visibly excluded.
        self.assertIsNone(tp.band_of(-1.0))


class TestStep(unittest.TestCase):
    """One frame of the car's own limiter, under each shape."""

    def setUp(self):
        self.before = tp.Limits(tp.BEFORE_FLAT)
        # A deliberately DIFFERENT candidate from the default, so the two chains are
        # distinguishable: with both at 409 every difference test would pass vacuously.
        self.candidate = 450
        self.after = tp.Limits(self.candidate)

    def test_rate_limit_binds_from_zero(self):
        # STEER_DELTA_UP is 3, so a full-demand frame from rest cannot exceed 3 counts.
        self.assertEqual(tp.step(1.0, 0, 0.0, self.before, 409, False), 3)

    def test_ceiling_binds_when_the_rate_limit_does_not(self):
        self.assertEqual(tp.step(1.0, 409, 0.0, self.before, 409, False), 409)
        self.assertEqual(tp.step(1.0, 450, 0.0, self.after, 450, True), 450)

    def test_a_higher_candidate_climbs_where_the_recorded_ceiling_is_pinned(self):
        # Seeded at 409 with full demand: BEFORE is at its ceiling and stays there, the
        # higher candidate climbs by one rate step. This is what a projection MEASURES --
        # note the tool would refuse to actually run this candidate, and rightly.
        b = tp.step(1.0, 409, 0.0, self.before, tp.BEFORE_FLAT, False)
        a = tp.step(1.0, 409, 0.0, self.after, self.candidate, True)
        self.assertEqual(b, 409)
        self.assertEqual(a, 412)

    def test_an_equal_candidate_is_a_no_op_at_every_seed(self):
        # The DEFAULT case, and the one the tool actually ships with: candidate == recorded
        # ceiling must produce an identical chain at every seed and both call shapes. If this
        # ever differs, the two limiter shapes have diverged and every delta is suspect.
        same = tp.Limits(tp.BEFORE_FLAT)
        for prev in (0, 100, 300, 409):
            for driver in (0.0, -100.0, -260.0):
                b = tp.step(0.9, prev, driver, self.before, tp.BEFORE_FLAT, False)
                a = tp.step(0.9, prev, driver, same, tp.BEFORE_FLAT, True)
                self.assertEqual(b, a, f"seeded at {prev}, driver {driver}")

    def test_driver_override_ramps_the_command_down_not_off(self):
        # driver_max_torque = STEER_MAX + (50 + driver)*2, clamped at 0, so heavy opposition
        # sets the target to 0. It does NOT get there in one frame: STEER_DELTA_DOWN is 7, and
        # the rate limiter floors each step at prev - 7. Asserting an instant zero here would
        # be asserting a car that does not exist, and would hide a broken rate limiter.
        self.assertEqual(tp.step(1.0, 409, -300.0, self.before, 409, False), 402)
        self.assertEqual(tp.step(1.0, 10, -300.0, self.before, 409, False), 3)
        self.assertEqual(tp.step(1.0, 3, -300.0, self.before, 409, False), 0)

    def test_full_yield_point_moves_with_the_ceiling(self):
        # The quantified cost of the raise, and the one number that actually moves.
        #
        # Override starts REDUCING authority at driver torque < -STEER_DRIVER_ALLOWANCE (-50)
        # for both ceilings -- that threshold does not scale. What scales is the point of FULL
        # yield, where driver_max_torque = M + (50 + d)*2 reaches zero: d <= -M/2 - 50, i.e.
        # -242 at 384, -254.5 at 409, and -275 at 450. The 450 row is kept as the arithmetic
        # for a ceiling this car cannot use -- it is what the driver would have had to fight,
        # and it stays here so the tradeoff is still legible if the question is ever reopened.
        #
        # Seeded from 0 so the rate limiter is not what is being measured; from a saturated
        # previous frame the ramp-down floor hides the effect entirely.
        for steer_max, yield_at in ((384, -242.0), (409, -254.5), (450, -275.0)):
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

    def test_a_higher_ceiling_would_keep_steering_where_409_gives_up(self):
        # The authority a raise would buy, kept as a measurement rather than a proposal: at
        # -260 the 409 car has fully yielded and a 450 car has not. This is what made 450 look
        # attractive on paper. The car then faulted its EPS at that ceiling, which is the
        # reminder that arithmetic about counts is not a promise about torque.
        between = -260.0
        self.assertEqual(tp.step(1.0, 0, between, tp.Limits(409), 409, False), 0)
        self.assertGreater(tp.step(1.0, 0, between, tp.Limits(450), 450, True), 0)


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


class TestCandidateProvenance(unittest.TestCase):
    """A report must name the ceiling the scan actually used, or refuse to name one.

    `--candidate` lives on the `scan` subcommand only, so a bare `report` never sees it. Before
    the candidate was stamped into each record, `scan --candidate 384` followed by `report`
    printed a header reading "what a flat 409 changes" -- the module default -- over data
    measured at 384. The resume path made it worse: two runs at different candidates append to
    the SAME jsonl and nothing recorded that they disagreed.

    This is the same failure this whole tool exists to avoid: a confident number describing a
    build that was never measured.
    """

    def _rec(self, seg, candidate):
        rec = {"seg": seg, "error": None,
               "bands": {name: tp.new_band() for name in tp.BAND_NAMES},
               "factory": {"frames": 0, "nonzero": 0, "max_abs": 0, "act_toi": 0},
               "tx": {"frames": 0, "max_abs": 0},
               "valid": {"paired": 0, "exact_lag0": 0, "exact_lag1": 0, "within1_lag1": 0,
                         "max_mismatch": 0, "sum_mismatch": 0}}
        if candidate is not None:
            rec["candidate"] = candidate
        return rec

    def _report(self, recs):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "scan.jsonl"
            out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

            class _Args:
                pass
            a = _Args()
            a.out = str(out)
            buf = io.StringIO()
            with redirect_stdout(buf):
                tp.cmd_report(a)
            return buf.getvalue()

    def test_the_header_names_the_scanned_ceiling_not_the_default(self):
        # The actual regression: scanned at 384, module default is 409.
        text = self._report([self._rec("a", 384), self._rec("b", 384)])
        self.assertIn("what a flat 384 changes", text)
        self.assertNotIn("what a flat 409 changes", text)

    def test_a_file_mixing_two_candidates_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self._report([self._rec("a", 384), self._rec("b", 409)])
        self.assertIn("mixes candidate ceilings", str(caught.exception))

    def test_an_unstamped_file_is_refused_rather_than_assumed(self):
        with self.assertRaises(SystemExit) as caught:
            self._report([self._rec("a", None)])
        self.assertIn("unrecorded", str(caught.exception))

    def test_scan_stamps_the_candidate_it_ran_with(self):
        # Proves the write side, not just the read side: the field has to actually be emitted.
        import inspect
        src = inspect.getsource(tp.cmd_scan)
        self.assertIn('"candidate": CANDIDATE_CEILING', src)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
