#!/usr/bin/env python3
"""turn_tracking.py -- decompose lateral turn performance, one turn event at a time.

Answers "why does the car go wide" by measuring, for each discrete turn, how much of the
commanded curvature the car achieved and where the rest of it went. Runs against the local
archive or the device's own store.

  scan    [--routes DIR] [--out FILE] [--limit N] [--force]
  report  [--out FILE] [--group build|jerk|commit|none]

Deliberate choices, each of which was a real defect at some point in this project:

  * STEER_MAX is read PER SEGMENT from the log, never hard-coded. An earlier version assumed
    409 and scored a fleet that was mostly a 384-count build; every count it derived from the
    normalised output was 6.5% high, and "no turn ever reached 400" was vacuous because the
    ceiling was 384. steer_max_of() recovers it as median(torqueOutputCan / actuatorsOutput
    .torque), which is exact -- carcontroller computes the second by dividing by the first.

  * Builds are NEVER pooled. 384-count and 409-count routes are different cars for this
    purpose and the report refuses to average them.

  * The message streams are joined by two different methods, because they are two different
    problems. controlsState and carControl come from controlsd in the same cycle and pair by
    INDEX. carOutput comes from card, a different process, and pairs by a per-segment LEARNED
    TIME LAG -- a fixed index offset cannot work, ~40% of segments have different message
    counts. An earlier version latched "the last carOutput seen", which put the ask at frame k
    beside the applied value at frame k-1 and reported ~100% of frames truncated on a highway
    where the car tracks perfectly.

  * pct_rail_up / pct_rail_down need no cross-process join at all -- they read consecutive
    carOutput frames and count the ones sitting exactly on a rate limit. That makes them the
    primary rate-limit evidence, immune to every pairing bug above.

  * binned MEDIANS, never fits -- the torque response is convex below ~10 m/s and a fit
    reports a slope the car does not have.

  * gain_yaw uses the yaw rate, not controlsState.curvature. controlsState.curvature is
    VehicleModel output computed with paramsd live-fitted steerRatio and stiffnessFactor -- a
    tyre-slip model whose job is to agree with the measured path. Comparing it against yaw
    tests whether that model is calibrated, not whether the car is slipping. Both are
    reported; only the yaw one is model-free.

  * uncalibrated and "Big Model Failed" windows are excluded. Both corrupt a tune measurement.

  * a route provenance is read from its OWN segment 0, never from the live device.
"""
import argparse
import bisect
import glob
import json
import os
import sys
from collections import defaultdict, namedtuple

try:
    from openpilot.tools.lib.logreader import LogReader as _LR

    def _events(p):
        return _LR(p, sort_by_time=True)

    BACKEND = "openpilot.LogReader"
except Exception:
    # Local archive: oplog.py + schemas/ beside the script, or wherever $TURN_TRACKING_OPLOG
    # points. Imported LAZILY -- pycapnp segfaults on native Windows, and requiring it at
    # module scope would make the unit tests undecodable on the machine that writes them.
    def _events(p):
        for d in (os.environ.get("TURN_TRACKING_OPLOG"),
                  os.path.dirname(os.path.abspath(__file__))):
            if d and d not in sys.path:
                sys.path.insert(0, d)
        import oplog
        return oplog.events(p)

    BACKEND = "local oplog"

DT = 0.01                       # controlsState is 100 Hz
SPEED_BANDS = ((0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 999.0))
BAND_NAMES = [f"{lo:g}-{hi:g}" if hi < 999 else f"{lo:g}+" for lo, hi in SPEED_BANDS]
TURN_ON = 0.8                   # m/s^2 commanded lateral accel that opens a turn event
TURN_OFF = 0.4                  # hysteresis, so one turn does not split on noise
MIN_TURN_FRAMES = 40            # 0.4 s; shorter is noise, not a turn
# The rate limits this scan SCORES AGAINST. They are the opendbc defaults, and they are also
# exactly what a ramp-rate change would alter -- so they are inputs, not constants. A scan of
# post-change drives run against the old numbers would report the new rail as "not on a rail"
# and quietly invalidate the before/after comparison this tool exists to make. Override with
# --rate-up/--rate-down and the value used is recorded in every scan record.
RATE_UP = 3                     # STEER_DELTA_UP,   opendbc/car/hyundai/values.py
RATE_DOWN = 7                   # STEER_DELTA_DOWN, same
DRIVER_ALLOWANCE = 50           # STEER_DRIVER_ALLOWANCE
DRIVER_MULTIPLIER = 2           # STEER_DRIVER_MULTIPLIER
DRIVER_FACTOR = 1               # STEER_DRIVER_FACTOR
KNOWN_CEILINGS = (255, 270, 384, 409)
SNAP_TOLERANCE = 1.0            # counts; only wide enough for float noise, see steer_max_of
SAMPLE_OFFSETS = (0, 10, 25, 50, 75, 100, 150, 200)
RAMP_SPLIT_FRAMES = 150         # 1.5 s: ramp phase vs sustained phase
REVERSAL_FRAC = 0.15            # a retracement worth this much of the turn peak is a reversal
LAG_SWEEP_MS = range(41)        # candidate carOutput lags, milliseconds
LAG_GOOD_RESIDUAL = 1.0         # counts; agreement this close IS proof the join is right
LAG_MAX_CONTRAST = 0.5          # otherwise the minimum must at least be sharp

LATERAL_PARAMS = ("LateralJerkTorqueController", "NeuralNetworkLateralControl",
                  "EnforceTorqueControl", "TorqueControlTune", "LiveTorqueParamsToggle",
                  "LiveTorqueParamsRelaxedToggle", "TorqueParamsOverrideEnabled",
                  "LagdToggle", "Mads")

# v      vEgo                                cmd    desiredCurvature the controller was given
# act    controlsState.curvature (VehicleModel output, NOT model-free)
# out    torqueState.output, normalised      req    out * this segment STEER_MAX, in counts
# can    torqueOutputCan at the learned lag  driver carState.steeringTorque
# mdl    modelV2.action.desiredCurvature     yaw    yaw rate, rad/s (None when invalid)
Frame = namedtuple("Frame", "v cmd act out req can driver pressed sat mdl yaw")


def band_of(v):
    for (lo, hi), name in zip(SPEED_BANDS, BAND_NAMES, strict=True):
        if lo <= v < hi:
            return name
    return None


def med(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def rlog_of(segdir):
    for n in ("rlog.zst", "rlog"):
        f = os.path.join(segdir, n)
        if os.path.exists(f):
            return f
    return None


def routes_under(root):
    """{route: [segment dirs, numerically sorted]}. Skips underscore-prefixed names, which
    would collide with marker files, and anything with no rlog."""
    out = defaultdict(list)
    for p in glob.glob(os.path.join(root, "*--*--*")):
        base = os.path.basename(p)
        if base.startswith("_"):
            continue
        try:
            route, seg = base.rsplit("--", 1)
            int(seg)
        except ValueError:
            continue
        if rlog_of(p):
            out[route].append(p)
    for r in out:
        out[r].sort(key=lambda p: int(os.path.basename(p).rsplit("--", 1)[1]))
    return dict(out)


def provenance(seg0):
    """Commit plus the lateral params that decide which controller ran.

    A param ABSENT from initData was never written, which for a BOOL reads as off at runtime --
    so absent and "0" are the same configuration and both report as "0".

    latAccelFactor here is the STATIC CarParams value. It is not what the controller uses: the
    live learner overwrites it at 4 Hz and the two differ. Named so nobody confuses them again.
    """
    prov = {"commit": None, "version": None, "params": {}}
    f = rlog_of(seg0)
    if not f:
        return prov
    seen_init = seen_cp = False
    for e in _events(f):
        w = e.which()
        if w == "initData" and not seen_init:
            d = e.initData
            prov["commit"] = d.gitCommit[:9]
            prov["version"] = d.version
            pm = {}
            for ent in d.params.entries:
                if ent.key in LATERAL_PARAMS:
                    try:
                        pm[ent.key] = bytes(ent.value).decode("utf-8", "replace").strip("\x00").strip()
                    except Exception:
                        pass
            for k in LATERAL_PARAMS:
                prov["params"][k] = pm.get(k, "0")
            seen_init = True
        elif w == "carParams" and not seen_cp:
            cp = e.carParams
            prov["fingerprint"] = str(cp.carFingerprint)
            try:
                prov["safetyParam"] = cp.safetyConfigs[-1].safetyParam
            except Exception:
                pass
            try:
                t = cp.lateralTuning.torque
                prov["latAccelFactorStatic"] = round(float(t.latAccelFactor), 4)
                prov["frictionStatic"] = round(float(t.friction), 4)
            except Exception:
                pass
            seen_cp = True
        if seen_init and seen_cp:
            break
    return prov


def steer_max_of(samples):
    """This segment STEER_MAX, recovered from the log rather than assumed.

    carcontroller writes actuatorsOutput.torque = apply_torque / STEER_MAX and
    actuatorsOutput.torqueOutputCan = apply_torque, so their ratio IS the constant. Only
    samples with real torque are used; near zero the ratio is 0/0. Returns None when the
    segment never steered hard enough to tell, which is a legitimate answer and must not be
    silently replaced by a default -- an assumed ceiling is what broke the previous version.
    """
    ratios = [c / t for c, t in samples if abs(t) > 0.05 and abs(c) > 1.0]
    if len(ratios) < 20:
        return None
    m = med(ratios)
    # The ratio is exact to float precision, so the snap exists only to clean up float noise --
    # it must NOT be wide enough to bend a real measurement onto a familiar number. At 8 counts
    # a true 380-count build reported as 384, which is the assumed-ceiling failure this whole
    # function exists to prevent, wearing a different hat.
    best = min(KNOWN_CEILINGS, key=lambda k: abs(k - m))
    return float(best) if abs(best - m) <= SNAP_TOLERANCE else float(round(m))


def learn_lag(cs_t, cs_req, co_t, co_can):
    """Milliseconds by which carOutput trails controlsState, learned per segment.

    controlsd and card are separate processes. A fixed index offset cannot work -- the two
    streams do not even have the same message count in ~40% of segments -- so the join is
    nearest-neighbour in time under a swept offset, choosing the offset that minimises the
    median |ask - applied|.

    That objective is sharp and correct WHEN THE TWO SIGNALS AGREE, and on real segments it
    lands at 14-17 ms with a residual under half a count. It is also confounded with the very
    thing this tool exists to measure: on a segment where the EPS received a median 82 counts
    against an ask of 289, no offset makes them agree, the residual curve is flat from 86
    counts down to 77 with no minimum, and an unguarded argmin slides to the edge of the sweep
    and reports a lag with total confidence.

    So the estimator also reports how sharp its minimum is. CONTRAST is the best residual over
    the typical residual across the sweep: near 0 is a real, sharp minimum; near 1 means the
    curve is flat and nothing was found. A lag pinned to the edge of the sweep means the same.
    Callers must check, and this module refuses to pretend otherwise.

    (Correlation was tried as the objective instead and is worse: it is flat near its peak on
    these smooth signals and drifted to the sweep edge on 5 of 6 real segments where the
    residual objective was landing correctly at 14-17 ms.)

    Both streams run at 100 Hz, so the lag is identifiable only to within one sample: every
    candidate inside a ~10 ms window selects the SAME neighbour and scores identically. We
    report the centre of that plateau rather than its first edge.

    Returns (lag_ms, residual_counts, contrast).
    """
    if len(cs_t) < 200 or len(co_t) < 200:
        return 0, float("nan"), 1.0
    ts, xs = cs_t[::7], cs_req[::7]
    scored = []
    for lag_ms in LAG_SWEEP_MS:
        off = lag_ms * 1e6  # logMonoTime is nanoseconds
        scored.append((lag_ms, med([abs(q - co_can[nearest(co_t, t + off)])
                                    for t, q in zip(ts, xs, strict=True)])))
    resids = [r for _, r in scored]
    best, typical = min(resids), med(resids)
    plateau = [lag for lag, r in scored if r <= best + max(1e-9, 0.001 * best)]
    contrast = best / typical if typical > 0 else 1.0
    return int(round(med(plateau))), best, contrast


def nearest(sorted_t, target):
    """Index of the entry in sorted_t closest to target."""
    j = bisect.bisect_left(sorted_t, target)
    if j >= len(sorted_t):
        return len(sorted_t) - 1
    if j > 0 and target - sorted_t[j - 1] < sorted_t[j] - target:
        return j - 1
    return j


def rail_occupancy(co_can, rate_up=RATE_UP, rate_down=RATE_DOWN):
    """Fraction of consecutive carOutput steps sitting exactly on a rate limit.

    Needs no cross-process join, no STEER_MAX and no turn detection, so it is the one
    rate-limit statistic that cannot be broken by a pairing bug. Steps are counted only where
    the applied torque is live at one end.
    """
    up = down = live = 0
    for k in range(1, len(co_can)):
        a, b = co_can[k - 1], co_can[k]
        if a == 0.0 and b == 0.0:
            continue
        live += 1
        d = abs(b - a)
        if abs(d - rate_up) < 0.5:
            up += 1
        elif abs(d - rate_down) < 0.5:
            down += 1
    if not live:
        return 0.0, 0.0, 0
    return 100.0 * up / live, 100.0 * down / live, live


def driver_ceiling(steer_max, driver_torque, command_sign):
    """Magnitude of the stage-1 window in apply_driver_steer_torque_limits, in the direction
    the controller is asking. opendbc/car/lateral.py:70-90."""
    if command_sign >= 0:
        d = steer_max + (DRIVER_ALLOWANCE + driver_torque * DRIVER_FACTOR) * DRIVER_MULTIPLIER
        return max(min(steer_max, d), 0.0)
    d = -steer_max + (-DRIVER_ALLOWANCE + driver_torque * DRIVER_FACTOR) * DRIVER_MULTIPLIER
    return -min(max(-steer_max, d), 0.0)


def collect_segment(segdir, rate_up=RATE_UP, rate_down=RATE_DOWN):
    """One segment -> (frames, meta). A None frame marks engaged-but-excluded or not engaged,
    so a turn event can never span a gap."""
    f = rlog_of(segdir)
    meta = {"steer_max": None, "lag_ms": None, "lag_resid": None, "lag_contrast": None,
            "lag_trusted": False, "read_error": None,
            "rail_up": 0.0, "rail_down": 0.0, "rail_frames": 0,
            "excluded": {"uncalibrated": 0, "big_model_failed": 0}}
    if not f:
        return [], meta

    cs = []
    co_t, co_can, co_pairs = [], [], []
    v = driver = 0.0
    pressed = lat = False
    mdl = yaw = None
    ok = True
    try:
        for e in _events(f):
            w = e.which()
            if w == "carState":
                c = e.carState
                v, driver, pressed = c.vEgo, float(c.steeringTorque), bool(c.steeringPressed)
            elif w == "carControl":
                lat = bool(e.carControl.latActive)
            elif w == "carOutput":
                a = e.carOutput.actuatorsOutput
                co_t.append(e.logMonoTime)
                co_can.append(float(a.torqueOutputCan))
                co_pairs.append((float(a.torqueOutputCan), float(a.torque)))
            elif w == "modelV2":
                try:
                    mdl = float(e.modelV2.action.desiredCurvature)
                except Exception:
                    pass
            elif w == "liveLocationKalman":
                m = e.liveLocationKalman.angularVelocityCalibrated
                yaw = float(m.value[2]) if bool(m.valid) else None
            elif w == "selfdriveState":
                a1 = e.selfdriveState.alertText1 or ""
                if "Calibrat" in a1:
                    ok = False
                    meta["excluded"]["uncalibrated"] += 1
                elif "Big Model Failed" in a1:
                    ok = False
                    meta["excluded"]["big_model_failed"] += 1
                else:
                    ok = True
            elif w == "controlsState":
                if not lat or not ok:
                    cs.append(None)
                    continue
                c = e.controlsState
                ts = c.lateralControlState.torqueState
                cs.append((e.logMonoTime, v, float(c.desiredCurvature), float(c.curvature),
                           float(ts.output), driver, pressed, bool(ts.saturated), mdl, yaw))
    except Exception as ex:
        # A truncated rlog leaves partial data behind. Keep it -- it is still real -- but record
        # that it is partial, so a reader of the report alone can see it without the scan console.
        meta["read_error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
        print(f"    ! {os.path.basename(segdir)}: {meta['read_error']}", file=sys.stderr)

    meta["steer_max"] = steer_max_of(co_pairs)
    meta["rail_up"], meta["rail_down"], meta["rail_frames"] = rail_occupancy(co_can, rate_up, rate_down)
    sm = meta["steer_max"]
    if sm is None or not co_t:
        # No recoverable ceiling means no trustworthy count arithmetic. Report the segment as
        # unusable rather than inventing a default.
        return [None] * len(cs), meta

    live = [r for r in cs if r is not None]
    meta["lag_ms"], meta["lag_resid"], meta["lag_contrast"] = learn_lag(
        [r[0] for r in live], [r[4] * sm for r in live], co_t, co_can)
    # Two independent routes to trusting the join, because either alone rejects good segments.
    # DIRECT: the two series agree to within a count at the chosen offset, which is proof by
    # construction. INDIRECT: the residual curve has a sharp minimum, which is what has to
    # carry a segment where the car genuinely was not given what it asked for. A quiet segment
    # can join perfectly (0.27 counts) and still score a weak contrast, because when the signal
    # barely moves, misalignment costs little -- contrast alone rejected those. A lag pinned to
    # the edge of the sweep fails regardless: that is the estimator saying it found nothing.
    meta["lag_trusted"] = ((meta["lag_resid"] <= LAG_GOOD_RESIDUAL
                            or meta["lag_contrast"] <= LAG_MAX_CONTRAST)
                           and min(LAG_SWEEP_MS) < meta["lag_ms"] < max(LAG_SWEEP_MS))
    off = meta["lag_ms"] * 1e6
    frames = []
    for r in cs:
        if r is None:
            frames.append(None)
            continue
        t, vv, cmd, act, out, drv, pr, sat, m2, yw = r
        frames.append(Frame(vv, cmd, act, out, out * sm, co_can[nearest(co_t, t + off)],
                            drv, pr, sat, m2, yw))
    return frames, meta


def _la(fr):
    return abs(fr.cmd) * max(fr.v, 1.0) ** 2


def turn_events(frames):
    """Contiguous engaged runs whose commanded lateral accel opens above TURN_ON and stays
    above TURN_OFF."""
    evs = []
    i = 0
    n = len(frames)
    while i < n:
        fr = frames[i]
        if fr is not None and _la(fr) > TURN_ON:
            j = i
            while j < n and frames[j] is not None and _la(frames[j]) > TURN_OFF:
                j += 1
            if j - i >= MIN_TURN_FRAMES:
                evs.append(frames[i:j])
            i = max(j, i + 1)
        else:
            i += 1
    return evs


def decompose(ev, steer_max, rate_up=RATE_UP):
    """One turn event -> the numbers that say where the turn went."""
    prof = {}
    for off in SAMPLE_OFFSETS:
        if off < len(ev):
            f = ev[off]
            v2 = max(f.v, 1.0) ** 2
            prof[off] = {"cmd": abs(f.cmd) * v2, "act": abs(f.act) * v2,
                         "out": abs(f.out), "can": abs(f.can)}

    # Lateral path deviation: double-integrate the SIGNED lat-accel tracking error. Signed
    # matters -- with abs() a command and an actual on opposite sides cancel to ~zero, which is
    # the largest possible error reported as the smallest.
    vel = pos = peak = 0.0
    n_rate = n_trunc = n_pin = n_press = n_sat = n_drvlim = n_clip = 0
    d_driver, d_rate, clip_def, gain_vm, gain_yaw = [], [], [], [], []
    n_mdl = 0
    prev_can = ev[0].can
    up_run = up_best = reversals = 0
    peak_can = max(abs(f.can) for f in ev)
    ext, rising = abs(ev[0].can), 0
    for f in ev:
        v2 = max(f.v, 1.0) ** 2
        vel += (f.cmd - f.act) * v2 * DT
        pos += vel * DT
        peak = max(peak, abs(pos))

        if abs(f.out) >= 0.999:
            n_pin += 1
        if abs(f.req - f.can) > 1.0:
            n_trunc += 1
            # Rate-limited specifically while BUILDING torque: the ask is further from zero than
            # what is applied, and the applied moved exactly one STEER_DELTA_UP step. Unwinding
            # at STEER_DELTA_DOWN is also a rate limit but is not what makes a car miss a turn.
            if abs(f.req) > abs(prev_can) and abs(abs(f.can - prev_can) - rate_up) <= 0.5:
                n_rate += 1
        if abs(f.can) > abs(prev_can) and abs(abs(f.can - prev_can) - rate_up) <= 0.5:
            up_run += 1
            up_best = max(up_best, up_run)
        else:
            up_run = 0

        # Where the missing counts went: the driver clamp takes its cut first, the rate limiter
        # works on what is left. Per frame the two sum to the shortfall whenever the applied value
        # is at or below the ask, which is the only case that matters here; both floor at zero, so
        # a join artefact putting applied ABOVE the ask reports no deficit rather than a negative
        # one. They are reported as MEANS, not medians: the driver clamp binds on ~42% of
        # low-speed turn frames, so its median is 0 and a median-reported decomposition says the
        # driver costs nothing while the medians of ask and applied sit 234 counts apart. Medians
        # do not sum; means do, and this is a decomposition.
        ceil = driver_ceiling(steer_max, f.driver, 1.0 if f.req >= 0 else -1.0)
        if ceil < abs(f.req) - 0.5:
            n_drvlim += 1
        d_driver.append(max(0.0, abs(f.req) - ceil))
        d_rate.append(max(0.0, min(abs(f.req), ceil) - abs(f.can)))

        # clip_curvature: what the model asked for versus what the controller was handed.
        if f.mdl is not None and abs(f.mdl) > 1e-4:
            n_mdl += 1
            short = 1.0 - abs(f.cmd) / abs(f.mdl)
            if short > 1e-3:
                n_clip += 1
                clip_def.append(short)

        if abs(f.can) > 40.0 and f.v > 1.0:
            gain_vm.append(abs(f.act) * f.v ** 2 / (abs(f.can) / steer_max))
            if f.yaw is not None:
                gain_yaw.append(abs(f.yaw) * f.v / (abs(f.can) / steer_max))

        # Direction reversals, for the stability watch. A per-frame delta sign flip is NOT a
        # reversal: on a steady hold with a couple of counts of quantisation dither it fires
        # ~21 times a second, which is not a reversal of anything. Track the running extremum
        # and count one only when the applied torque RETRACES past REVERSAL_FRAC of the turn
        # peak -- that is what "changed direction" means for this signal.
        if peak_can > 1.0:
            a = abs(f.can)
            thresh = REVERSAL_FRAC * peak_can
            if rising >= 0:
                if a >= ext:
                    ext, rising = a, 1
                elif ext - a > thresh:
                    reversals += 1
                    ext, rising = a, -1
            elif a <= ext:
                ext = a
            elif a - ext > thresh:
                reversals += 1
                ext, rising = a, 1

        prev_can = f.can
        n_press += 1 if f.pressed else 0
        n_sat += 1 if f.sat else 0

    n = len(ev)
    ramp, sus = ev[:RAMP_SPLIT_FRAMES], ev[RAMP_SPLIT_FRAMES:]
    return {"v0": ev[0].v, "band": band_of(ev[0].v), "frames": n, "profile": prof,
            "steer_max": steer_max,
            "dev_end": abs(pos), "dev_peak": peak,
            "pct_rate_limited": 100.0 * n_rate / n,
            "pct_cmd_truncated": 100.0 * n_trunc / n,
            "pct_output_pinned": 100.0 * n_pin / n,
            "pct_driver_on_wheel": 100.0 * n_press / n,
            "pct_driver_limited": 100.0 * n_drvlim / n,
            "pct_saturated_flag": 100.0 * n_sat / n,
            "pct_clipped": 100.0 * n_clip / n,
            "peak_can": peak_can,
            "pct_at_ceiling": 100.0 * sum(1 for f in ev if abs(f.can) >= steer_max - 1.0) / n,
            "longest_up_run_s": up_best * DT,
            "med_req": med([abs(f.req) for f in ev]),
            "med_can": med([abs(f.can) for f in ev]),
            "deficit_driver": sum(d_driver) / n, "deficit_rate": sum(d_rate) / n,
            "deficit_total": (sum(d_driver) + sum(d_rate)) / n,
            "mean_req": sum(abs(f.req) for f in ev) / n,
            "mean_can": sum(abs(f.can) for f in ev) / n,
            "deficit_ramp": med([abs(f.req) - abs(f.can) for f in ramp]) if ramp else float("nan"),
            "deficit_sustained": med([abs(f.req) - abs(f.can) for f in sus]) if sus else float("nan"),
            # 0.0 means measured-and-none; nan means the model curvature was never usable.
            "clip_deficit": (med(clip_def) if clip_def else 0.0) if n_mdl else float("nan"),
            "gain_vm": med(gain_vm) if gain_vm else float("nan"),
            "gain_yaw": med(gain_yaw) if gain_yaw else float("nan"),
            "reversals_per_s": reversals / (n * DT)}


def scan_route(segs, rate_up=RATE_UP, rate_down=RATE_DOWN):
    """Walk a route segment by segment. Segment-level facts are aggregated weighted by the
    frames that produced them, never averaged flat."""
    frames, agg = [], {"rail_up_n": 0.0, "rail_down_n": 0.0, "rail_frames": 0,
                       "lags": [], "resids": [], "contrasts": [], "untrusted": 0, "steer_max": set(),
                       "excluded": {"uncalibrated": 0, "big_model_failed": 0},
                       "read_errors": [], "band_frames": defaultdict(int)}
    for sd in segs:
        fr, meta = collect_segment(sd, rate_up, rate_down)
        frames.extend(fr)
        frames.append(None)  # segments are not contiguous; never let a turn span the seam
        agg["rail_up_n"] += meta["rail_up"] * meta["rail_frames"] / 100.0
        agg["rail_down_n"] += meta["rail_down"] * meta["rail_frames"] / 100.0
        agg["rail_frames"] += meta["rail_frames"]
        for k in agg["excluded"]:
            agg["excluded"][k] += meta["excluded"][k]
        if meta["read_error"]:
            agg["read_errors"].append(f"{os.path.basename(sd)}: {meta['read_error']}")
        if meta["steer_max"] is not None:
            agg["steer_max"].add(meta["steer_max"])
            agg["lags"].append(meta["lag_ms"])
            agg["resids"].append(meta["lag_resid"])
            agg["contrasts"].append(meta["lag_contrast"])
            if not meta["lag_trusted"]:
                agg["untrusted"] += 1
    for f in frames:
        if f is not None:
            b = band_of(f.v)
            if b:
                agg["band_frames"][b] += 1
    return frames, agg


def cmd_scan(a):
    routes = routes_under(a.routes)
    names = sorted(routes)
    if a.limit:
        names = names[-a.limit:]
    done = set()
    if os.path.exists(a.out) and not a.force:
        with open(a.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["route"])
                except Exception:
                    pass
    print(f"backend: {BACKEND}   routes: {len(names)}   already scanned: {len(done)}")
    with open(a.out, "a") as fh:
        for k, r in enumerate(names, 1):
            if r in done and not a.force:
                continue
            segs = routes[r]
            prov = provenance(segs[0])
            frames, agg = scan_route(segs, a.rate_up, a.rate_down)
            sms = sorted(agg["steer_max"])
            # A route that changed ceiling mid-drive cannot be scored against one constant.
            steer_max = sms[0] if len(sms) == 1 else None
            evs = [decompose(e, steer_max, a.rate_up) for e in turn_events(frames)] if steer_max else []
            eng = sum(1 for f in frames if f is not None)
            rail_n = agg["rail_frames"]
            fh.write(json.dumps({
                "route": r, "segments": len(segs), "engaged_frames": eng,
                "excluded": agg["excluded"], "read_errors": agg["read_errors"], "prov": prov,
                "steer_max": steer_max, "steer_max_seen": sms,
                "rate_up": a.rate_up, "rate_down": a.rate_down,
                "lag_ms": med(agg["lags"]) if agg["lags"] else None,
                "lag_resid": med(agg["resids"]) if agg["resids"] else None,
                "lag_contrast": med(agg["contrasts"]) if agg["contrasts"] else None,
                "lag_untrusted_segments": agg["untrusted"],
                "rail_up": 100.0 * agg["rail_up_n"] / rail_n if rail_n else 0.0,
                "rail_down": 100.0 * agg["rail_down_n"] / rail_n if rail_n else 0.0,
                "rail_frames": rail_n,
                "band_frames": dict(agg["band_frames"]),
                "events": evs}) + "\n")
            fh.flush()
            sm_label = f"{steer_max:g}" if steer_max else (f"MIXED{sms}" if sms else "unknown")
            jerk = prov["params"].get("LateralJerkTorqueController", "?")
            print(f"[{k}/{len(names)}] {r} segs={len(segs):3d} engaged={eng:6d} turns={len(evs):3d} steer_max={sm_label} jerk={jerk} commit={prov['commit']}")
    return 0


def _group_key(rec, how):
    if how == "build":
        sm = rec.get("steer_max")
        return f"STEER_MAX={sm:g}" if sm else "STEER_MAX=unknown"
    if how == "jerk":
        return "LateralJerkTorqueController=" + str(rec["prov"]["params"].get("LateralJerkTorqueController", "?"))
    if how == "commit":
        return str(rec["prov"]["commit"])
    return "all routes"


def _ratio(sel, off):
    """Median achieved/commanded lateral accel `off` frames after turn onset."""
    vals = [e["profile"][str(off)]["act"] / max(e["profile"][str(off)]["cmd"], 1e-9)
            for e in sel
            if str(off) in e["profile"] and e["profile"][str(off)]["cmd"] > 0.3]
    return med(vals) if vals else float("nan")


def _col(sel, key):
    """Median across turns, dropping turns that could not measure this quantity.

    med() sorts, and sorting a list containing nan gives an arbitrary order, so a single
    unmeasurable turn can make the whole cell read nan -- which is how gain_yaw, one of the
    load-bearing numbers, came out blank in two speed bands on a 1.37M-frame scan."""
    vals = [e[key] for e in sel if e[key] == e[key]]
    return med(vals) if vals else float("nan")


def _wmean(sel, key):
    """Frame-weighted mean across turns. The right aggregate for an additive decomposition:
    a median over turns of a per-turn median silently drops any component that bites in a
    minority of frames, which is exactly what the driver-torque clamp does."""
    frames = sum(e["frames"] for e in sel)
    return sum(e[key] * e["frames"] for e in sel) / frames if frames else float("nan")


def pctl(xs, q):
    """Nearest-rank percentile. Used where the tail IS the finding and a median would erase it."""
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def _pooled_pct(sel, key):
    """A frame-weighted percentage across every turn in the band.

    The median of per-turn percentages is the wrong statistic for anything that happens in a
    minority of turns: ceiling contact shows up in a handful of hard corners, so a median over
    turns reports 0.0% on a build that demonstrably reaches its ceiling. Pool the frames.
    """
    frames = sum(e["frames"] for e in sel)
    if not frames:
        return float("nan")
    return sum(e[key] / 100.0 * e["frames"] for e in sel) / frames * 100.0


def _table(evs, band_frames):
    """Two tables. Every cell states the sample it came from -- a median over 3 turns is not
    the same kind of number as a median over 300, and pooling builds is refused upstream."""
    print("\n  DELIVERY            turnFrm = frames inside turn events | bandFrm = all engaged frames in the band")
    head = ["band".ljust(7), "turns".rjust(5), "turnFrm".rjust(8), "bandFrm".rjust(8),
            "t=0".rjust(6), "0.25s".rjust(6), "0.75s".rjust(6), "1.5s".rjust(6), "|",
            "devPk".rjust(6), "medReq".rjust(7), "medCan".rjust(7), "pkCAN".rjust(7),
            "pk90".rjust(6), "%ceil".rjust(6)]
    print("  " + " ".join(head))
    print("  " + "-" * 119)
    for b in BAND_NAMES:
        sel = [e for e in evs if e["band"] == b]
        if len(sel) < 3:
            continue
        print("  " + " ".join([
            f"{b:<7}", f"{len(sel):5d}", f"{sum(e['frames'] for e in sel):8d}",
            f"{band_frames.get(b, 0):8d}", f"{_ratio(sel, 0):6.2f}", f"{_ratio(sel, 25):6.2f}",
            f"{_ratio(sel, 75):6.2f}", f"{_ratio(sel, 150):6.2f}", "|",
            f"{_col(sel, 'dev_peak'):6.2f}", f"{_col(sel, 'med_req'):7.0f}",
            f"{_col(sel, 'med_can'):7.0f}", f"{_col(sel, 'peak_can'):7.0f}",
            f"{pctl([e['peak_can'] for e in sel], 0.9):6.0f}",
            f"{_pooled_pct(sel, 'pct_at_ceiling'):5.1f}%"]))

    print("\n  WHERE IT WENT       deficits in CAN counts; defDrv + defRate is the whole shortfall")
    head = ["band".ljust(7), "turns".rjust(5), "upRun".rjust(7), "defRamp".rjust(7),
            "defSust".rjust(7), "defDrv".rjust(7), "defRate".rjust(7), "drvLim%".rjust(7), "|",
            "pin%".rjust(6), "hand%".rjust(6), "clip%".rjust(6), "clipDef".rjust(7),
            "gainYaw".rjust(7), "rev/s".rjust(6)]
    print("  " + " ".join(head))
    print("  " + "-" * 118)
    for b in BAND_NAMES:
        sel = [e for e in evs if e["band"] == b]
        if len(sel) < 3:
            continue
        print("  " + " ".join([
            f"{b:<7}", f"{len(sel):5d}", f"{_col(sel, 'longest_up_run_s'):6.2f}s",
            f"{_col(sel, 'deficit_ramp'):7.0f}", f"{_col(sel, 'deficit_sustained'):7.0f}",
            f"{_wmean(sel, 'deficit_driver'):7.0f}", f"{_wmean(sel, 'deficit_rate'):7.0f}",
            f"{_col(sel, 'pct_driver_limited'):6.1f}%", "|",
            f"{_col(sel, 'pct_output_pinned'):5.1f}%", f"{_col(sel, 'pct_driver_on_wheel'):5.1f}%",
            f"{_col(sel, 'pct_clipped'):5.1f}%", f"{100.0 * _col(sel, 'clip_deficit'):6.1f}%",
            f"{_col(sel, 'gain_yaw'):7.2f}", f"{_col(sel, 'reversals_per_s'):6.1f}"]))


def cmd_report(a):
    recs = []
    with open(a.out) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

    groups = defaultdict(list)
    for r in recs:
        groups[_group_key(r, a.group)].append(r)

    for g in sorted(groups):
        rs = groups[g]
        evs = [e for r in rs for e in r["events"]]
        band_frames = defaultdict(int)
        for r in rs:
            for b, n in (r.get("band_frames") or {}).items():
                band_frames[b] += n
        rail_n = sum(r.get("rail_frames", 0) for r in rs)
        print("\n" + "=" * 118)
        engaged = sum(r["engaged_frames"] for r in rs)
        print(f"{g}   routes={len(rs)}  engaged_frames={engaged}  turn_events={len(evs)}")
        print(f"  commits={sorted({r['prov']['commit'] for r in rs})[:6]}")
        seen = sorted({r.get("steer_max") for r in rs}, key=lambda x: (x is None, x))
        print(f"  steer_max seen={seen}   (builds are never pooled; group with --group build)")
        errs = [e for r in rs for e in (r.get("read_errors") or [])]
        if errs:
            print(f"  WARNING: {len(errs)} segment(s) were read only partially; their numbers come")
            print(f"           from a truncated log. First: {errs[0]}")
        if rail_n:
            # The headline. No cross-process join, no STEER_MAX, no turn detection: nothing in
            # this line can be broken by a pairing bug.
            up = sum(r.get("rail_up", 0.0) * r.get("rail_frames", 0) for r in rs) / rail_n
            dn = sum(r.get("rail_down", 0.0) * r.get("rail_frames", 0) for r in rs) / rail_n
            ru = sorted({r.get("rate_up", RATE_UP) for r in rs})
            rd = sorted({r.get("rate_down", RATE_DOWN) for r in rs})
            print(f"  RAIL OCCUPANCY (pairing-free), over {rail_n} live carOutput steps:")
            print(f"    {up:.1f}% sit exactly on STEER_DELTA_UP={ru}, {dn:.1f}% on STEER_DELTA_DOWN={rd}")
            if len(ru) > 1 or len(rd) > 1:
                print("    WARNING: this group mixes scans taken against DIFFERENT rate limits, so the")
                print("    two numbers above are not comparable. Re-scan, or group differently.")
        lags = [r["lag_ms"] for r in rs if r.get("lag_ms") is not None]
        resids = [r["lag_resid"] for r in rs if r.get("lag_resid") is not None]
        if lags:
            cons = [r["lag_contrast"] for r in rs if r.get("lag_contrast") is not None]
            c = f", contrast {med(cons):.3f} (0 = sharp minimum, 1 = nothing found)" if cons else ""
            print(f"  carOutput join: lag median {med(lags):.0f} ms (range {min(lags):.0f}-{max(lags):.0f}){c}")
            print(f"    residual median {med(resids):.2f} counts -- a DIAGNOSTIC, not the join objective:")
            print("    a large residual at high correlation means the car was not given what it asked for.")
            bad = sum(r.get("lag_untrusted_segments", 0) for r in rs)
            if bad:
                print(f"    WARNING: {bad} segment(s) had no trustworthy join (residual above")
                print(f"    {LAG_GOOD_RESIDUAL} count AND contrast above {LAG_MAX_CONTRAST},")
                print("    or the lag pinned to the sweep edge). Their paired numbers are suspect;")
                print("    the rail-occupancy figures above are unaffected -- they need no join.")
        if not evs:
            print("  no turn events in this group")
            continue
        _table(evs, band_frames)
        print("\n  t=N    median achieved/commanded curvature N seconds after turn onset (1.0 = tracking)")
        print("  devPk  median peak lateral path deviation over the turn, metres")
        print("  medReq median commanded CAN counts | medCan median counts the EPS received")
        print("  pkCAN  median peak applied counts | pk90 the 90th-percentile turn, where the")
        print("         ceiling question actually lives -- a median hides it")
        print("  %ceil  FRAME-POOLED share within 1 count of this build STEER_MAX. Pooled, not a")
        print("         median over turns: ceiling contact happens in a few hard corners, and a")
        print("         median over turns reports 0.0% on a build that demonstrably reaches it.")
        print("  upRun  longest UNBROKEN run of STEER_DELTA_UP steps, seconds")
        print("  defDrv/defRate MEAN counts lost to the driver clamp and to the rate limiter.")
        print("         Means, because they are a decomposition and must sum; the clamp bites on a")
        print("         minority of frames, so a median reports it as costing nothing.")
        print("  clip%  frames where clip_curvature reduced the model demand | clipDef by how much")
        print("  gainYaw achieved lateral accel per unit APPLIED normalised torque, from the yaw")
        print("          rate -- model-free. Compare against the single latAccelFactor in use.")
        print("  rev/s  applied-torque direction reversals per second (stability watch)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.set_defaults(fn=cmd_scan)
    s.add_argument("--routes", default="/data/media/0/realdata")
    s.add_argument("--out", default="turn_tracking.jsonl")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--rate-up", type=int, default=RATE_UP, dest="rate_up",
                   help="STEER_DELTA_UP to score against; raise it when scanning post-change drives")
    s.add_argument("--rate-down", type=int, default=RATE_DOWN, dest="rate_down",
                   help="STEER_DELTA_DOWN to score against")
    r = sub.add_parser("report")
    r.set_defaults(fn=cmd_report)
    r.add_argument("--out", default="turn_tracking.jsonl")
    r.add_argument("--group", default="build", choices=["build", "jerk", "commit", "none"])
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
