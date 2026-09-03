#!/usr/bin/env python3
"""Tests for turn_tracking.py. Plain main()-style harness, matching the .elantra convention:
run it directly and read the exit code. pytest collects nothing from this.

    python .elantra/test_turn_tracking.py

Three checks in the previous version pinned metrics that were themselves wrong and have been
deliberately rewritten, not deleted:

  * the old frame tuple had no `req` field, so decompose derived the commanded count as
    output * a module-level STEER_MAX of 409. That constant was wrong on every 384-count route
    in the archive. `req` is now carried explicitly, computed once per segment from the
    ceiling the log actually reports, and the tests construct it the same way.

  * the truncation and rate-limit checks still assert the same behaviour they always did; only
    the scale they assert it at is now honest.

Everything else is unchanged and must stay green.
"""
import sys

import turn_tracking as tt

FAIL = []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def frame(v=1.0, cmd=0.0, act=0.0, out=0.0, can=0.0, driver=0.0, pressed=False, sat=False,
          mdl=None, yaw=None, steer_max=409.0, req=None):
    """A Frame with req derived from out the way collect_segment derives it: the normalised
    output times the ceiling THIS segment reported."""
    return tt.Frame(v, cmd, act, out, out * steer_max if req is None else req, can,
                    driver, pressed, sat, mdl, yaw)


# --- band_of -----------------------------------------------------------------------------
def test_bands():
    print("band_of")
    check("lower edge is inclusive", tt.band_of(3.0), "3-7")
    check("upper edge belongs to the next band", tt.band_of(7.0), "7-10")
    check("zero lands in the first band", tt.band_of(0.0), "0-3")
    check("very fast lands in the last band", tt.band_of(40.0), "18+")


# --- turn_events -------------------------------------------------------------------------
def test_turn_events():
    print("turn_events")
    quiet = [frame(cmd=0.0)] * 50
    strong = [frame(cmd=1.0)] * 100          # v=1 so lat accel == curvature numerically

    evs = tt.turn_events(quiet + strong + quiet)
    check("one clean turn is found", len(evs), 1)
    check("the turn keeps its frames", len(evs[0]), 100)

    short = [frame(cmd=1.0)] * (tt.MIN_TURN_FRAMES - 1)
    check("a run under MIN_TURN_FRAMES is not a turn", len(tt.turn_events(quiet + short + quiet)), 0)

    dip = [frame(cmd=1.0)] * 50 + [frame(cmd=0.6)] * 20 + [frame(cmd=1.0)] * 50
    evs = tt.turn_events(quiet + dip + quiet)
    check("a dip above TURN_OFF does not split the turn", len(evs), 1)
    check("the whole dipped run is kept", len(evs[0]), 120)

    gapped = [frame(cmd=1.0)] * 60 + [None] + [frame(cmd=1.0)] * 60
    evs = tt.turn_events(quiet + gapped + quiet)
    check("a None frame splits the turn in two", len(evs), 2)
    check("no event spans the gap", all(all(f is not None for f in e) for e in evs), True)


# --- decompose: path deviation -----------------------------------------------------------
def test_deviation_magnitude():
    print("decompose / path deviation")
    e, n = 1.0, 100                      # 1.0 m/s^2 for 1.00 s
    ev = [frame(v=1.0, cmd=1.0 + e, act=1.0)] * n
    d = tt.decompose(ev, 409.0)
    check("0.5*e*T^2 for a constant error", d["dev_peak"], 0.5 * e * (n * tt.DT) ** 2, tol=0.02)


def test_deviation_is_signed():
    print("decompose / deviation uses SIGNED error")
    # The failure mode guarded here: cmd and act on OPPOSITE sides, which abs() would cancel
    # to ~zero when the true error is large.
    n = 100
    ev = [frame(v=1.0, cmd=1.0, act=-1.0)] * n
    d = tt.decompose(ev, 409.0)
    check("opposite-sign tracking is a LARGE error, not zero",
          d["dev_peak"], 0.5 * 2.0 * (n * tt.DT) ** 2, tol=0.02)


# --- decompose: rate limiting ------------------------------------------------------------
def test_rate_limited_detection():
    print("decompose / rate limiting")
    ev = [frame(v=1.0, cmd=1.0, out=1.0, can=3.0 * k) for k in range(1, 101)]
    d = tt.decompose(ev, 409.0)
    check("a full-torque ask served 3 counts/frame is rate limited", d["pct_rate_limited"] > 90.0, True)
    check("and the command is truncated", d["pct_cmd_truncated"] > 90.0, True)
    check("the unbroken up-run is the whole turn", d["longest_up_run_s"], 0.99, tol=0.02)

    ev = [frame(v=1.0, cmd=1.0, out=100 / 409.0, can=100.0)] * 100
    d = tt.decompose(ev, 409.0)
    check("a satisfied, settled command is not rate limited", d["pct_rate_limited"], 0.0)
    check("and is not truncated", d["pct_cmd_truncated"], 0.0)

    # Releasing torque at STEER_DELTA_DOWN is a rate limit but not an up-rate one, and the
    # original metric counted it. Regression guard.
    ev = [frame(v=1.0, cmd=1.0, out=0.0, can=max(0.0, 700 - 7.0 * k)) for k in range(1, 101)]
    d = tt.decompose(ev, 409.0)
    check("unwinding at STEER_DELTA_DOWN is not counted as rate limited", d["pct_rate_limited"] < 5.0, True)


def test_truncation_uses_the_segment_ceiling():
    print("decompose / truncation is scored at the segment ceiling, not a constant")
    # A 384-count build commanding full scale and receiving it. Scored at 384 this is not
    # truncated; scored at a hard-coded 409 every frame would read as 25 counts short.
    ev = [frame(v=1.0, cmd=1.0, out=1.0, can=384.0, steer_max=384.0)] * 100
    check("full scale delivered on a 384 build is not truncated",
          tt.decompose(ev, 384.0)["pct_cmd_truncated"], 0.0)
    ev409 = [frame(v=1.0, cmd=1.0, out=1.0, can=384.0, steer_max=409.0)] * 100
    check("the same 384 counts on a 409 build IS 25 counts short",
          tt.decompose(ev409, 409.0)["pct_cmd_truncated"], 100.0)


def test_ceiling_reporting():
    print("decompose / ceiling occupancy")
    ev = [frame(v=1.0, cmd=1.0, out=1.0, can=409.0)] * 50 + \
         [frame(v=1.0, cmd=1.0, out=0.5, can=200.0)] * 50
    d = tt.decompose(ev, 409.0)
    check("half the frames sit at the ceiling", d["pct_at_ceiling"], 50.0)
    check("peak applied is the ceiling", d["peak_can"], 409.0)
    ev384 = [frame(v=1.0, cmd=1.0, out=1.0, can=384.0, steer_max=384.0)] * 100
    check("384 counts is the ceiling on a 384 build",
          tt.decompose(ev384, 384.0)["pct_at_ceiling"], 100.0)


def test_pinned_and_hands():
    print("decompose / pinned + driver")
    ev = [frame(v=1.0, cmd=1.0, out=1.0, pressed=True)] * 50 + \
         [frame(v=1.0, cmd=1.0, out=0.2, pressed=False)] * 50
    d = tt.decompose(ev, 409.0)
    check("half the frames are at the output limit", d["pct_output_pinned"], 50.0)
    check("half the frames have the driver on the wheel", d["pct_driver_on_wheel"], 50.0)


def test_deficit_split_is_exhaustive():
    print("decompose / the deficit split accounts for the whole shortfall")
    # Ask 409, driver opposing with 150 counts so the clamp allows 209, EPS given 100.
    # Driver part = 409-209 = 200, rate part = 209-100 = 109, and 200+109 = 309 = 409-100.
    ev = [frame(v=1.0, cmd=1.0, out=1.0, can=100.0, driver=-150.0)] * 100
    d = tt.decompose(ev, 409.0)
    check("driver clamp takes its cut first", d["deficit_driver"], 200.0)
    check("the rate limiter works on what is left", d["deficit_rate"], 109.0)
    check("the two sum to the shortfall", d["deficit_total"], 309.0)
    check("and the shortfall is mean ask minus mean applied", d["mean_req"] - d["mean_can"], 309.0)
    check("and the frames are flagged driver-limited", d["pct_driver_limited"], 100.0)

    # THE REASON THESE ARE MEANS. The clamp bites on a minority of frames -- measured at 42% of
    # 3-7 m/s turn frames on 2.03M engaged frames -- so a median reports the driver as costing
    # nothing while the medians of ask and applied sit hundreds of counts apart. Medians do not
    # sum; a decomposition must.
    ev = ([frame(v=1.0, cmd=1.0, out=1.0, can=100.0, driver=-150.0)] * 40 +
          [frame(v=1.0, cmd=1.0, out=1.0, can=409.0, driver=0.0)] * 60)
    d = tt.decompose(ev, 409.0)
    check("a minority-incidence clamp is invisible to a median",
          tt.med([200.0] * 40 + [0.0] * 60), 0.0)
    check("but the mean carries it", d["deficit_driver"], 80.0, tol=1e-9)
    check("the decomposition still sums", d["deficit_total"],
          d["deficit_driver"] + d["deficit_rate"], tol=1e-9)
    check("and equals mean ask minus mean applied",
          d["deficit_total"], d["mean_req"] - d["mean_can"], tol=1e-9)

    ev = [frame(v=1.0, cmd=1.0, out=1.0, can=100.0, driver=0.0)] * 100
    d = tt.decompose(ev, 409.0)
    check("a passive driver costs nothing", d["deficit_driver"], 0.0)
    check("and the whole shortfall is the rate limiter", d["deficit_rate"], 309.0)


def test_clip_curvature_detection():
    print("decompose / clip_curvature")
    ev = [frame(v=1.0, cmd=0.8, act=0.0, mdl=1.0)] * 50 + \
         [frame(v=1.0, cmd=1.0, act=0.0, mdl=1.0)] * 50
    d = tt.decompose(ev, 409.0)
    check("half the frames had demand removed", d["pct_clipped"], 50.0)
    check("and 20% of it was removed", d["clip_deficit"], 0.2, tol=1e-9)

    ev = [frame(v=1.0, cmd=1.0, act=0.0, mdl=None)] * 100
    check("no model curvature means nothing is claimed", tt.decompose(ev, 409.0)["pct_clipped"], 0.0)


def test_gain_is_model_free_from_yaw():
    print("decompose / plant gain")
    # 4 m/s, yaw 0.5 rad/s -> 2.0 m/s^2 achieved. 204.5 counts = 0.5 normalised. gain = 4.0.
    ev = [frame(v=4.0, cmd=0.2, act=0.125, can=204.5, yaw=0.5)] * 100
    d = tt.decompose(ev, 409.0)
    check("gain_yaw is achieved lat accel per unit applied torque", d["gain_yaw"], 4.0, tol=1e-6)
    # controlsState.curvature 0.125 at 4 m/s is also 2.0 m/s^2, so the two agree here by
    # construction. They are reported separately because in the logs they need not.
    check("gain_vm is computed from the VehicleModel curvature", d["gain_vm"], 4.0, tol=1e-6)

    ev = [frame(v=4.0, cmd=0.2, act=0.125, can=204.5, yaw=None)] * 100
    check("an invalid yaw yields no model-free gain", tt.decompose(ev, 409.0)["gain_yaw"] != tt.decompose(ev, 409.0)["gain_yaw"], True)


# --- collection-layer helpers ------------------------------------------------------------
def test_steer_max_recovery():
    print("steer_max_of")
    s409 = [(409.0 * x, x) for x in [0.1 + 0.01 * i for i in range(40)]]
    check("409 is recovered exactly", tt.steer_max_of(s409), 409.0)
    s384 = [(384.0 * x, x) for x in [0.1 + 0.01 * i for i in range(40)]]
    check("384 is recovered exactly", tt.steer_max_of(s384), 384.0)
    check("too few usable samples is None, not a default", tt.steer_max_of(s409[:5]), None)
    near_zero = [(0.4, 0.001)] * 100
    check("frames with no real torque are not used", tt.steer_max_of(near_zero), None)
    odd = [(500.0 * x, x) for x in [0.1 + 0.01 * i for i in range(40)]]
    check("a ceiling far from any known one is reported as measured", tt.steer_max_of(odd), 500.0)
    # The snap must not bend a real measurement onto a familiar number. 380 is 4 counts from
    # 384; an 8-count snap window reported it as 384, which is the assumed-ceiling failure this
    # function exists to prevent.
    near = [(380.0 * x, x) for x in [0.1 + 0.01 * i for i in range(40)]]
    check("a near-miss ceiling is NOT snapped to a known one", tt.steer_max_of(near), 380.0)
    noisy = [(409.0 * x + 1e-6, x) for x in [0.1 + 0.01 * i for i in range(40)]]
    check("float noise still snaps to the known ceiling", tt.steer_max_of(noisy), 409.0)


def test_nearest_and_lag():
    print("nearest / learn_lag")
    t = [0, 10, 20, 30, 40]
    check("exact hit", tt.nearest(t, 20), 2)
    check("rounds to the closer neighbour", tt.nearest(t, 24), 2)
    check("rounds up past the midpoint", tt.nearest(t, 26), 3)
    check("past the end clamps", tt.nearest(t, 999), 4)

    # carOutput trailing controlsState by 15 ms. Both streams are 100 Hz, so every candidate
    # from 10 to 19 ms selects the SAME neighbour and scores an identical zero residual: the
    # lag is identifiable only to one sample interval. The estimator reports the centre of that
    # plateau (14) rather than its first edge (10), which is what a plain argmin would return
    # and would be a full frame of false precision. What has to be exact is the JOIN, and the
    # zero residual is what proves it.
    n = 1000
    cs_t = [i * 10_000_000 for i in range(n)]
    req = [((i * 37) % 400) - 200.0 for i in range(n)]
    co_t = [x + 15_000_000 for x in cs_t]
    lag, resid, contrast = tt.learn_lag(cs_t, req, co_t, list(req))
    check("the learned lag is within half a frame of the truth", abs(lag - 15) <= 5, True)
    check("it is the plateau centre, not its first edge", lag, 14)
    check("and the join it picks is exact", resid, 0.0)
    check("a sharp minimum scores contrast 0", contrast, 0.0)
    check("which the trust test accepts", contrast <= tt.LAG_MAX_CONTRAST, True)

    # A stream with NO lag must not be given one.
    lag0, resid0, _ = tt.learn_lag(cs_t, req, list(cs_t), list(req))
    check("a synchronous stream learns a near-zero lag", abs(lag0) <= 5, True)
    check("and joins exactly", resid0, 0.0)

    # THE CONFOUNDED CASE, observed on a real segment: the EPS received a median 82 counts
    # against an ask of 289, so no offset makes the two agree. The residual curve is flat and
    # an unguarded argmin slid to the edge of the sweep and reported a lag with full
    # confidence. What must happen instead is that the estimator says it found nothing.
    flat = [40.0 for _ in range(n)]           # applied is unrelated to the ask
    _, _, contrast_flat = tt.learn_lag(cs_t, req, co_t, flat)
    check("a flat residual curve scores contrast near 1", contrast_flat > 0.9, True)
    check("and neither route to trust accepts it",
          contrast_flat <= tt.LAG_MAX_CONTRAST, False)

    # The other direction, also from real segments: a quiet stretch joins to 0.27 counts --
    # proof by construction -- yet scores a weak contrast, because when the signal barely
    # moves, misaligning it costs little. Contrast alone rejected those good joins, so a
    # sub-count residual has to be sufficient on its own.
    ramp = [200.0 + 0.02 * i for i in range(n)]                    # barely moves frame to frame
    noisy = [v + (0.3 if i % 2 else -0.3) for i, v in enumerate(ramp)]  # small rounding floor
    _, resid_q, contrast_q = tt.learn_lag(cs_t, ramp, co_t, noisy)
    check("a quiet segment still joins to well under a count", resid_q <= tt.LAG_GOOD_RESIDUAL, True)
    check("even though its contrast is weak", contrast_q > tt.LAG_MAX_CONTRAST, True)


def test_rail_occupancy_needs_no_pairing():
    print("rail_occupancy")
    up, down, live = tt.rail_occupancy([3.0 * k for k in range(100)])
    check("a pure STEER_DELTA_UP ramp is all up-rail", up, 100.0)
    check("and none of it is down-rail", down, 0.0)
    check("live steps are counted", live, 99)

    up, down, _ = tt.rail_occupancy([700.0 - 7.0 * k for k in range(100)])
    check("a pure STEER_DELTA_DOWN unwind is all down-rail", down, 100.0)
    check("and none of it is up-rail", up, 0.0)

    up, down, live = tt.rail_occupancy([0.0] * 100)
    check("a parked segment contributes no steps", live, 0)
    check("and no rail occupancy", up + down, 0.0)

    up, _, live = tt.rail_occupancy([0.0, 3.0, 6.0, 20.0, 23.0])
    check("a jump larger than the rate limit is not on the rail", up, 75.0)
    check("every live step counts toward the denominator", live, 4)

    # The rate limits are what a ramp-rate change alters, so they are an INPUT. Scoring a
    # 4-count ramp against the old 3 would report it as off-rail and silently destroy the
    # before/after comparison the tool exists to make.
    ramp4 = [4.0 * k for k in range(100)]
    up3, _, _ = tt.rail_occupancy(ramp4)
    up4, _, _ = tt.rail_occupancy(ramp4, rate_up=4, rate_down=7)
    check("a 4-count ramp scored against 3 looks off-rail", up3, 0.0)
    check("scored against 4 it is fully on the rail", up4, 100.0)


def test_driver_ceiling_matches_opendbc():
    print("driver_ceiling")
    check("a passive driver leaves the full ceiling", tt.driver_ceiling(409.0, 0.0, 1.0), 409.0)
    check("150 counts against cuts it to 209", tt.driver_ceiling(409.0, -150.0, 1.0), 209.0)
    check("150 counts with does not raise it above STEER_MAX",
          tt.driver_ceiling(409.0, 150.0, 1.0), 409.0)
    check("the window is symmetric for a left command",
          tt.driver_ceiling(409.0, 150.0, -1.0), 209.0)
    check("full yield at -(STEER_MAX/2)-ALLOWANCE", tt.driver_ceiling(409.0, -254.5, 1.0), 0.0)
    check("the clamp scales with the build ceiling", tt.driver_ceiling(384.0, -150.0, 1.0), 184.0)


def test_reversals_count_retracements_not_dither():
    print("decompose / stability watch")
    # A steady hold with a couple of counts of quantisation dither is NOT a reversal of
    # anything. An earlier version compared per-frame deltas against 0.75% of peak and scored
    # this at ~21 reversals/s, which made the stability column meaningless.
    dither = [380.0 + (2.0 if k % 2 else -2.0) for k in range(300)]
    ev = [frame(v=5.0, cmd=0.1, out=0.95, can=c) for c in dither]
    check("quantisation dither is not a reversal", tt.decompose(ev, 409.0)["reversals_per_s"], 0.0)

    # A genuine build-collapse-rebuild is two reversals: peak 400, so the threshold is 60.
    up = [4.0 * k for k in range(100)]          # 0 -> 396
    down = [396.0 - 4.0 * k for k in range(50)]  # 396 -> 200, a 196-count retracement
    up2 = [200.0 + 4.0 * k for k in range(50)]   # 200 -> 396, a 196-count advance
    ev = [frame(v=5.0, cmd=0.1, out=0.95, can=c) for c in up + down + up2]
    d = tt.decompose(ev, 409.0)
    check("a real collapse and rebuild is two reversals", d["reversals_per_s"] * d["frames"] * tt.DT, 2.0)

    # A retracement smaller than REVERSAL_FRAC of the peak does not count.
    shallow = up + [396.0 - 1.0 * k for k in range(20)] + [376.0 + 1.0 * k for k in range(20)]
    ev = [frame(v=5.0, cmd=0.1, out=0.95, can=c) for c in shallow]
    check("a shallow wobble is not a reversal", tt.decompose(ev, 409.0)["reversals_per_s"], 0.0)


def test_clip_deficit_distinguishes_zero_from_unknown():
    print("decompose / clip_deficit reports unknown as unknown")
    ev = [frame(v=1.0, cmd=1.0, mdl=1.0)] * 100
    check("model present and nothing clipped is a measured zero",
          tt.decompose(ev, 409.0)["clip_deficit"], 0.0)
    ev = [frame(v=1.0, cmd=1.0, mdl=None)] * 100
    got = tt.decompose(ev, 409.0)["clip_deficit"]
    check("no usable model curvature is nan, not a silent zero", got != got, True)


def test_tail_statistics_do_not_hide_the_ceiling():
    print("pctl / _pooled_pct")
    # Nearest-rank, not interpolated: index round(0.9 * 9) = 8, so the p90 of 1..10 is 9, not
    # 10. The max is what q=1.0 returns.
    check("nearest-rank p90 of 1..10 is 9", tt.pctl([float(i) for i in range(1, 11)], 0.9), 9.0)
    check("q=1.0 is the maximum", tt.pctl([float(i) for i in range(1, 11)], 1.0), 10.0)
    check("p50 is the median-ish middle", tt.pctl([1.0, 2.0, 3.0], 0.5), 2.0)
    check("nan entries are dropped", tt.pctl([1.0, float("nan"), 3.0], 0.0), 1.0)
    check("an empty sample is nan", tt.pctl([], 0.5) != tt.pctl([], 0.5), True)

    # The failure this metric exists to prevent: ceiling contact concentrated in one hard
    # corner out of ten. A median over turns calls that 0.0%; the pooled figure does not.
    turns = [{"frames": 100, "pct_at_ceiling": 0.0} for _ in range(9)]
    turns.append({"frames": 100, "pct_at_ceiling": 50.0})
    check("a median over turns erases a real ceiling hit",
          tt.med([e["pct_at_ceiling"] for e in turns]), 0.0)
    check("the pooled share does not", tt._pooled_pct(turns, "pct_at_ceiling"), 5.0)
    check("pooling is frame-weighted, not turn-weighted",
          tt._pooled_pct([{"frames": 900, "pct_at_ceiling": 0.0},
                          {"frames": 100, "pct_at_ceiling": 50.0}], "pct_at_ceiling"), 5.0)


def test_median_cell_survives_one_unmeasurable_turn():
    print("report / a median cell drops nan instead of becoming nan")
    nan = float("nan")
    sel = [{"gain_yaw": 2.0, "frames": 100}, {"gain_yaw": nan, "frames": 100},
           {"gain_yaw": 4.0, "frames": 100}]
    got = tt._col(sel, "gain_yaw")
    check("one unmeasurable turn does not blank the cell", got, 3.0)
    allnan = [{"gain_yaw": nan, "frames": 10}] * 3
    g = tt._col(allnan, "gain_yaw")
    check("but if nothing was measurable the cell IS nan", g != g, True)


def test_profile_keys_survive_json():
    print("decompose / profile keys")
    import json
    ev = [frame(v=5.0, cmd=0.1, act=0.05, out=0.5, can=200.0)] * 250
    d = json.loads(json.dumps(tt.decompose(ev, 409.0)))
    check("t=0 sample present after a JSON round trip", "0" in d["profile"], True)
    check("t=2.0s sample present", "200" in d["profile"], True)
    check("achieved/commanded is computed in lat-accel units",
          d["profile"]["0"]["act"] / d["profile"]["0"]["cmd"], 0.5, tol=1e-9)


def main():
    for fn in (test_bands, test_turn_events, test_deviation_magnitude, test_deviation_is_signed,
               test_rate_limited_detection, test_truncation_uses_the_segment_ceiling,
               test_ceiling_reporting, test_pinned_and_hands, test_deficit_split_is_exhaustive,
               test_clip_curvature_detection, test_gain_is_model_free_from_yaw,
               test_steer_max_recovery, test_nearest_and_lag, test_tail_statistics_do_not_hide_the_ceiling,
               test_reversals_count_retracements_not_dither, test_clip_deficit_distinguishes_zero_from_unknown,
               test_median_cell_survives_one_unmeasurable_turn,
               test_rail_occupancy_needs_no_pairing, test_driver_ceiling_matches_opendbc,
               test_profile_keys_survive_json):
        fn()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
