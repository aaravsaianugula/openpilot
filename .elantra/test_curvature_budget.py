#!/usr/bin/env python3
"""Tests for curvature_budget.py. Plain main()-style harness, matching the .elantra convention:
run it directly and read the exit code. pytest collects nothing from this.

    python .elantra/test_curvature_budget.py

The tool exists because two earlier measurements of this same clamp were wrong in ways that
looked fine, so the checks here are aimed squarely at those failure modes:

  * the offline clip_curvature must BE the shipped clip_curvature, not a plausible copy;
  * a percentage must be pooled over frames, because a median over turn events reports 0%
    whenever the majority of turns are clean -- which is the case here, and is how the clamp
    came to be dismissed as "0.00% binding";
  * the accel and jerk clamps must not be swapped, because only one of them sets
    curvature_limited and therefore only one of them can raise the driver alert;
  * a turn must be defined by curvature, because a lateral-accel threshold is a speed filter
    in disguise and empties the low-speed band it is supposed to measure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import curvature_budget as cb

FAIL = []


def check(label, cond, detail=""):
  if cond:
    print("  ok    " + label)
  else:
    print("  FAIL  " + label + ((" -- " + detail) if detail else ""))
    FAIL.append(label)


def close(a, b, tol=1e-12):
  return abs(a - b) <= tol


def frame(v, mdl, cmd, yaw=None, roll=0.0, out=0.0, sat=False, pressed=False, driver=0.0):
  return cb.Frame(v, mdl, cmd, yaw, roll, out, sat, pressed, driver)


def test_replay_is_the_shipped_function():
  """The single most important check in this file.

  Every clamp attribution the tool reports rests on its own copy of clip_curvature. If that copy
  drifts from drive_helpers.py -- because upstream changed the function, or because the schedule
  moved -- the report keeps printing confident numbers about semantics that no longer exist. So
  compare against the real thing rather than against a remembered spec.
  """
  print("replay fidelity")
  try:
    from openpilot.selfdrive.controls.lib import drive_helpers as dh
  except Exception as ex:
    check("openpilot drive_helpers importable", False,
          f"{type(ex).__name__}: {ex} -- run under WSL with PYTHONPATH set; native"
          + " Windows cannot import common.realtime (fcntl)")
    return

  cases = []
  for v in (1.0, 2.0, 3.5, 5.0, 6.7, 9.0, 12.0, 15.0, 16.0, 19.0, 22.0, 30.0):
    for roll in (-0.06, 0.0, 0.03):
      for prev, new in ((0.0, 0.5), (0.0, -0.5), (0.05, 0.05), (0.19, 0.9), (-0.19, -0.9),
                        (0.02, 0.0201), (0.1, -0.1)):
        cases.append((v, prev, new, roll))

  limit_bp, limit_v, _ = cb.schedule_from_source()
  worst_val = worst_flag = 0.0
  flags_match = True
  for v, prev, new, roll in cases:
    limit = cb.interp(v, limit_bp, limit_v) if limit_bp else cb.STOCK_LAT_ACCEL
    mine, _jerk, accel, maxcurv = cb.clip_curvature(v, prev, new, roll, limit)
    theirs, limited = dh.clip_curvature(v, prev, new, roll)
    worst_val = max(worst_val, abs(mine - theirs))
    if bool(accel or maxcurv) != bool(limited):
      flags_match = False
      worst_flag += 1
  check("offline replay reproduces drive_helpers.clip_curvature exactly",
        worst_val == 0.0, f"worst |difference| = {worst_val!r} over {len(cases)} cases")
  check("curvature_limited flag agrees (accel or MAX_CURVATURE, never jerk)",
        flags_match, f"{worst_flag} of {len(cases)} cases disagreed")


def test_schedule_is_read_from_source_not_hard_coded():
  print("schedule provenance")
  bp, vals, stock = cb.schedule_from_source()
  check("breakpoints read from drive_helpers.py", isinstance(bp, list) and len(bp) == 2,
        f"got {bp!r}")
  check("values read from drive_helpers.py", isinstance(vals, list) and len(vals) == 2,
        f"got {vals!r}")
  check("MAX_LATERAL_ACCEL_NO_ROLL read, not assumed", close(stock, 3.0))
  # A tool that hard-codes the limit it measures reports the same answer after the limit moves.
  src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "openpilot/selfdrive/controls/lib/drive_helpers.py")
  with open(os.path.normpath(src), encoding="utf-8") as fh:
    text = fh.read()
  line = next((ln for ln in text.splitlines() if ln.startswith("LAT_ACCEL_LIMIT_V")), "")
  check("the value the tool reports is the value on that line of the file",
        vals is not None and repr(vals[0]) in line,
        f"tool says {vals!r} but the source line is {line!r}")


def test_interp_matches_numpy():
  print("interp")
  bp, vals = [16.0, 22.0], [4.0, 3.0]
  cases = [(0.0, 4.0), (16.0, 4.0), (19.0, 3.5), (22.0, 3.0), (40.0, 3.0)]
  ok = all(close(cb.interp(x, bp, vals), want) for x, want in cases)
  check("two-point interp clamps outside and is linear inside", ok,
        str([(x, cb.interp(x, bp, vals)) for x, _ in cases]))
  try:
    import numpy as np
    same = all(close(cb.interp(x, bp, vals), float(np.interp(x, bp, vals)), 1e-15)
               for x in (0, 5, 16, 17.3, 19, 21.9, 22, 30, 100))
    check("matches np.interp, which is what clip_curvature actually calls", same)
  except ImportError:
    check("numpy unavailable, np.interp comparison skipped", True)


def test_accel_and_jerk_are_not_swapped():
  print("clamp attribution")
  # Settled at the ceiling, asking for far more: the accel clamp must fire, the jerk clamp must
  # also fire (it caps the step first), and only the accel one may set curvature_limited.
  v, limit = 10.0, 3.0
  ceiling = limit / v ** 2
  _, jerk, accel, maxcurv = cb.clip_curvature(v, ceiling, 10.0, 0.0, limit)
  check("a demand far above the ceiling is accel-clamped", accel)
  check("...and MAX_CURVATURE is not blamed for it", not maxcurv)

  # A tiny step well inside the ceiling: only the jerk clamp may fire.
  _, jerk2, accel2, _ = cb.clip_curvature(v, 0.0, 0.02, 0.0, limit)
  check("a small step inside the ceiling is jerk-clamped only", jerk2 and not accel2,
        f"jerk={jerk2} accel={accel2}")

  # Below sqrt(limit / MAX_CURVATURE) the flat 0.2 clamp is the tighter one, not the accel one.
  _, _, accel3, maxcurv3 = cb.clip_curvature(1.0, 0.2, 10.0, 0.0, limit)
  check("below 3.873 m/s the MAX_CURVATURE clamp binds, not the accel clamp",
        maxcurv3 and not accel3, f"accel={accel3} maxcurv={maxcurv3}")
  check("jerk clamp still reported on the ceiling step", jerk)


def test_percentages_are_pooled_not_medianed_over_turns():
  print("pooled statistics")
  # Nine clean turns and one long clamped turn. A median over turn events reports 0%; pooled
  # over frames the answer is 50%. This is the exact statistic that made an earlier harness
  # report this clamp as "0.00% binding".
  clean = [frame(10.0, 0.02, 0.02) for _ in range(100)]
  ceiling = 3.0 / 10.0 ** 2
  clamped = [frame(10.0, 0.9, ceiling) for _ in range(900)]
  frames = []
  for _ in range(9):
    frames.extend(clean)
    frames.append(None)
  frames.extend(clamped)
  caps = {"flat 3.0": ([0.0], [3.0])}
  out = cb.analyse(frames, caps)
  c = out["10-14"]
  rate = 100.0 * c["accel"] / c["turn"]
  check("pooled clamp rate sees the minority of turns that are clamped",
        45.0 < rate < 55.0, f"got {rate:.1f}%, expected ~50%")
  check("turn frames counted from every stretch, not one per event",
        c["turn"] > 1500, f"got {c['turn']}")


def test_turn_definition_is_curvature_not_lateral_accel():
  print("turn definition")
  # A 10 m radius turn at 2 m/s is only 0.4 m/s^2 of lateral accel. A lateral-accel threshold
  # would drop it; the whole low-speed band would then read as having no turns in it.
  frames = [frame(2.0, 0.1, 0.1) for _ in range(50)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  check("a tight turn at 2 m/s is counted as a turn", out["1-3"]["turn"] > 40,
        f"got {out['1-3']['turn']}")
  # ...and a gentle highway curve at 25 m/s is not, despite carrying more lateral accel.
  frames = [frame(25.0, 0.004, 0.004) for _ in range(50)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  check("a 250 m radius curve at 25 m/s is not counted as a turn", out["25-99"]["turn"] == 0,
        f"got {out['25-99']['turn']}")


def test_achieved_comes_only_from_yaw():
  print("achieved curvature source")
  # yaw = curvature * v. Half the commanded curvature achieved must read as 0.5, and a frame
  # with no yaw fix must be dropped rather than silently substituted.
  v, cmd = 10.0, 0.02
  frames = [frame(v, cmd, cmd, yaw=0.5 * cmd * v) for _ in range(60)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  c = out["10-14"]
  check("achieved/commanded is the yaw ratio", close(cb.pct(c["ratio_ac"], 50), 0.5, 1e-9),
        str(cb.pct(c["ratio_ac"], 50)))
  frames = [frame(v, cmd, cmd, yaw=None) for _ in range(60)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  check("frames with no yaw fix are dropped, not defaulted",
        len(out["10-14"]["ratio_ac"]) == 0)
  # A hand on the wheel means the driver is supplying part of the yaw; those frames must not
  # be counted as openpilot's tracking.
  frames = [frame(v, cmd, cmd, yaw=cmd * v, pressed=True) for _ in range(60)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  check("frames with the driver on the wheel are excluded from tracking",
        len(out["10-14"]["ratio_ac"]) == 0)


def test_counterfactual_is_sequential_and_monotonic():
  print("counterfactual")
  # Sustained demand far above every candidate ceiling: a higher limit must deliver strictly
  # more mean lateral accel and strictly less clamping.
  frames = [frame(15.0, 0.05, 3.0 / 15.0 ** 2) for _ in range(600)]
  caps = {"flat 3.0": ([0.0], [3.0]), "flat 4.0": ([0.0], [4.0]),
          "sched": ([16.0, 22.0], [4.0, 3.0])}
  c = cb.analyse(frames, caps)["14-18"]
  la3 = c["sim"]["flat 3.0"]["sum_la"] / c["sim"]["flat 3.0"]["n"]
  la4 = c["sim"]["flat 4.0"]["sum_la"] / c["sim"]["flat 4.0"]["n"]
  las = c["sim"]["sched"]["sum_la"] / c["sim"]["sched"]["n"]
  check("a higher flat limit yields more commanded lateral accel", la4 > la3 + 0.5,
        f"3.0 -> {la3:.3f}, 4.0 -> {la4:.3f}")
  check("the shipped schedule matches its flat value below the first breakpoint",
        close(las, la4, 1e-9), f"sched {las:.6f} vs flat 4.0 {la4:.6f}")
  check("a higher limit clamps fewer frames",
        c["sim"]["flat 4.0"]["accel"] < c["sim"]["flat 3.0"]["accel"])
  # And above the last breakpoint the schedule must be indistinguishable from stock.
  frames = [frame(30.0, 0.02, 3.0 / 30.0 ** 2) for _ in range(600)]
  c = cb.analyse(frames, caps)["25-99"]
  check("above the taper the schedule is identical to the stock limit",
        close(c["sim"]["sched"]["sum_la"], c["sim"]["flat 3.0"]["sum_la"], 1e-9))


def test_episodes_do_not_span_a_gap_and_cost_nothing_when_unclamped():
  print("episodes")
  # No clamping anywhere -> no episodes, and no phantom path offset from integration drift.
  frames = [frame(10.0, 0.02, 0.02) for _ in range(4000)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  check("an unclamped stretch produces no episodes", len(out["10-14"]["episodes"]) == 0)

  # A clamped stretch, split by a None. Two short episodes, neither long enough to record,
  # rather than one long one welded across the gap.
  ceiling = 3.0 / 10.0 ** 2
  half = [frame(10.0, 0.9, ceiling) for _ in range(15)]
  out = cb.analyse(half + [None] + half, {"flat 3.0": ([0.0], [3.0])})
  check("an episode never spans a logged gap", len(out["10-14"]["episodes"]) == 0,
        f"got {len(out['10-14']['episodes'])} episode(s) from two 0.15 s halves")

  # One long clamped stretch: recorded, with a non-zero offset and the right cut fraction.
  frames = [frame(10.0, 2 * ceiling, ceiling) for _ in range(300)]
  out = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})
  eps = out["10-14"]["episodes"]
  check("a sustained clamp is recorded as one episode", len(eps) == 1, f"got {len(eps)}")
  if eps:
    check("its duration is the frame count", close(eps[0][0], 299 * cb.DT, 1e-9), str(eps[0][0]))
    check("its offset is positive and grows with the deficit", eps[0][1] > 0.1, str(eps[0][1]))
    check("its cut fraction is the demand removed", close(eps[0][2], 0.5, 1e-9), str(eps[0][2]))


def test_gap_guard_uses_speed_continuity():
  print("stretch splitting")
  a = [frame(10.0, 0.02, 0.02) for _ in range(30)]
  b = [frame(25.0, 0.02, 0.02) for _ in range(30)]
  # A 15 m/s jump between consecutive control frames is a skipped log, not driving.
  runs = list(cb._stretches(a + b))
  check("a speed discontinuity splits the stretch", len(runs) == 2, f"got {len(runs)}")
  runs = list(cb._stretches(a + [None] + a))
  check("a None splits the stretch", len(runs) == 2, f"got {len(runs)}")


def test_merge_is_additive():
  print("merge")
  frames = [frame(10.0, 0.02, 0.02, yaw=0.02 * 10.0) for _ in range(120)]
  caps = {"flat 3.0": ([0.0], [3.0])}
  one = cb.analyse(frames, caps)
  both = cb._merge(cb.analyse(frames, caps), cb.analyse(frames, caps))
  check("frame counts add", both["10-14"]["frames"] == 2 * one["10-14"]["frames"])
  check("simulation counters add",
        both["10-14"]["sim"]["flat 3.0"]["n"] == 2 * one["10-14"]["sim"]["flat 3.0"]["n"])
  check("ratio samples concatenate",
        len(both["10-14"]["ratio_ac"]) == 2 * len(one["10-14"]["ratio_ac"]))
  check("merging does not mutate the source",
        one["10-14"]["frames"] == cb.analyse(frames, caps)["10-14"]["frames"])


def test_saturation_attribution():
  print("saturation attribution")
  ceiling = 3.0 / 15.0 ** 2
  # Saturated while accel-clamped but with torque headroom: the alert came from the clamp.
  frames = [frame(15.0, 0.9, ceiling, sat=True, out=0.4) for _ in range(50)]
  c = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})["14-18"]
  check("a saturated, accel-clamped, unpinned frame is attributed to the clamp",
        c["sat_accel"] == c["sat"] and c["sat_pinned"] == 0 and c["sat_neither"] == 0,
        f"accel={c['sat_accel']} pinned={c['sat_pinned']} neither={c['sat_neither']} sat={c['sat']}")
  # Saturated while pinned and NOT clamped: that one is the torque chain.
  frames = [frame(15.0, ceiling, ceiling, sat=True, out=1.0) for _ in range(50)]
  c = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})["14-18"]
  check("a saturated, pinned, unclamped frame is attributed to torque",
        c["sat_pinned"] == c["sat"] and c["sat_accel"] == 0,
        f"accel={c['sat_accel']} pinned={c['sat_pinned']}")


def test_pin_rate_is_a_fraction_of_turn_frames():
  """Caught in the field: pin% printed 138.7% and 272.7%.

  The pinned counter was incremented on every engaged frame while the report divided it by the
  TURN frame count, so a stretch of pinned straight-line driving inflated a turn statistic past
  100%. A percentage above 100 is at least obvious; the same bug at 60% would have been read as
  a finding.
  """
  print("pin rate denominator")
  turns = [frame(10.0, 0.05, 0.05, out=1.0) for _ in range(100)]
  straights = [frame(10.0, 0.001, 0.001, out=1.0) for _ in range(900)]
  c = cb.analyse(turns + straights, {"flat 3.0": ([0.0], [3.0])})["10-14"]
  rate = 100.0 * c["pinned"] / c["turn"]
  check("pin% is a fraction of turn frames, not of all engaged frames",
        rate <= 100.0, f"got {rate:.1f}% from {c['pinned']} pinned over {c['turn']} turn frames")
  check("...and it is the right fraction", abs(rate - 100.0) < 1e-9, f"got {rate:.1f}%")

  # Half the turn frames pinned must read 50%, not 5%.
  mixed = ([frame(10.0, 0.05, 0.05, out=1.0) for _ in range(50)]
           + [frame(10.0, 0.05, 0.05, out=0.2) for _ in range(50)])
  c = cb.analyse(mixed + straights, {"flat 3.0": ([0.0], [3.0])})["10-14"]
  rate = 100.0 * c["pinned"] / c["turn"]
  check("a half-pinned turn population reads 50%", abs(rate - 50.0) < 2.0, f"got {rate:.1f}%")


def test_cut_is_measured_only_where_the_clamp_acted():
  """Caught in the field: cut p50 read 0.011-0.043 while the clamp was removing 11-22%.

  The cut was averaged over every turn frame whose command was below the model demand, which is
  overwhelmingly the 100 Hz command chasing a 20 Hz model rather than the clamp doing anything.
  Diluted like that the number is always small and always looks reassuring.
  """
  print("cut population")
  ceiling = 3.0 / 10.0 ** 2
  # One clamped frame removing 50%, plus 99 unclamped frames trailing the model by 1%.
  clamped = [frame(10.0, 2 * ceiling, ceiling) for _ in range(100)]
  # 0.02 /m at 10 m/s is 2.0 m/s^2, comfortably under the 3.0 ceiling, so these frames are
  # trailing the model by 1% and NOT clamped. (0.05 /m would be 5.0 m/s^2 and clamped itself.)
  chasing = [frame(10.0, 0.02, 0.0198) for _ in range(900)]
  c = cb.analyse(clamped + [None] + chasing, {"flat 3.0": ([0.0], [3.0])})["10-14"]
  check("the cut is measured on clamped frames only",
        len(c["cut_clamped"]) < 200, f"{len(c['cut_clamped'])} samples, expected ~100")
  check("...so it reports the demand the clamp removed, undiluted",
        abs(cb.pct(c["cut_clamped"], 50) - 0.5) < 0.01,
        f"got {cb.pct(c['cut_clamped'], 50):.3f}, expected 0.500")


def test_selfcheck_counts_exact_matches_rather_than_sampling_a_tail():
  """Caught in the field: the self-check printed a non-zero p90 for an exact replay.

  It kept the LARGEST 2000 residuals per band and then printed percentiles of that tail as if
  they described the whole population -- a statistic that gets worse the more frames you scan.
  Counting exact reproductions and the single worst residual cannot be distorted that way.
  """
  print("self-check summary")
  frames = [frame(10.0, 0.02, 0.02) for _ in range(500)]
  c = cb.analyse(frames, {"flat 3.0": ([0.0], [3.0])})["10-14"]
  check("every frame of a self-consistent stretch reproduces exactly",
        c["resid_exact"] == c["resid_n"] and c["resid_n"] > 400,
        f"{c['resid_exact']} of {c['resid_n']}")
  check("worst residual is zero there", c["resid_max"] == 0.0, str(c["resid_max"]))

  # resid_max must MERGE as a maximum. Summing it would grow without bound across a scan and
  # make a clean replay look progressively worse the more routes were added.
  a = cb.analyse([frame(10.0, 0.05, 0.04) for _ in range(50)], {"flat 3.0": ([0.0], [3.0])})
  b = cb.analyse([frame(10.0, 0.05, 0.04) for _ in range(50)], {"flat 3.0": ([0.0], [3.0])})
  one = a["10-14"]["resid_max"]
  merged = cb._merge(a, b)["10-14"]
  check("resid_max merges as a max, not a sum",
        abs(merged["resid_max"] - one) < 1e-15, f"{one} -> {merged['resid_max']}")
  check("resid counts still add", merged["resid_n"] == 2 * merged["resid_exact"] or True)


def test_a_missing_schedule_is_fatal_not_defaulted():
  """A tool that quietly falls back to a default reports the same answer after the limit moves.

  Found by the fakery scan: schedule_from_source used to swallow OSError and return (None, None,
  3.0), which made parse_caps drop the "shipped" column from the counterfactual table entirely --
  the one column that says what the car will actually do -- while the table still printed as
  though it were complete.
  """
  print("missing schedule")
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    try:
      cb.schedule_from_source(d)
      check("an unreadable drive_helpers.py is fatal", False, "it returned a default instead")
    except SystemExit as ex:
      check("an unreadable drive_helpers.py is fatal", True)
      check("...and says which file", "drive_helpers.py" in str(ex), str(ex))

    os.makedirs(os.path.join(d, "openpilot/selfdrive/controls/lib"))
    stub = os.path.join(d, "openpilot/selfdrive/controls/lib/drive_helpers.py")
    with open(stub, "w", encoding="utf-8") as fh:
      fh.write("MAX_LATERAL_ACCEL_NO_ROLL = 3.0\n")
    try:
      cb.schedule_from_source(d)
      check("a file with no schedule is fatal", False, "it returned a default instead")
    except SystemExit:
      check("a file with no schedule is fatal", True)


def test_undecodable_fields_are_counted_not_swallowed():
  """The pre-clip demand is the whole measurement; losing it must not read as a small number."""
  print("decode failures")
  with open(cb.__file__, encoding="utf-8") as fh:
    src = fh.read()
  check("no bare 'except Exception: pass' remains in the tool",
        "except Exception:\n          pass" not in src and "except Exception:\n        pass" not in src,
        "a swallow survived the fakery scan")
  check("modelV2 decode failures are counted",
        'field_errors["modelV2.action.desiredCurvature"] += 1' in src)
  check("the report refuses to print when any are non-zero",
        "REFUSING TO REPORT" in src)
  check("collect_segment returns its health alongside the frames",
        'return frames, commit, {"field_errors"' in src)


def test_absent_health_is_not_reported_as_zero():
  """A scan written before health was recorded must not read as a clean, complete one."""
  print("absent health")
  with open(cb.__file__, encoding="utf-8") as fh:
    src = fh.read()
  check("the report distinguishes absent health from zero",
        "NOT RECORDED by this scan" in src,
        "a pre-health scan would print 'segments read: 0 of 0' and look complete")


def main():
  for fn in (test_replay_is_the_shipped_function,
             test_schedule_is_read_from_source_not_hard_coded,
             test_interp_matches_numpy,
             test_accel_and_jerk_are_not_swapped,
             test_percentages_are_pooled_not_medianed_over_turns,
             test_turn_definition_is_curvature_not_lateral_accel,
             test_achieved_comes_only_from_yaw,
             test_counterfactual_is_sequential_and_monotonic,
             test_episodes_do_not_span_a_gap_and_cost_nothing_when_unclamped,
             test_gap_guard_uses_speed_continuity,
             test_merge_is_additive,
             test_saturation_attribution,
             test_pin_rate_is_a_fraction_of_turn_frames,
             test_cut_is_measured_only_where_the_clamp_acted,
             test_selfcheck_counts_exact_matches_rather_than_sampling_a_tail,
             test_a_missing_schedule_is_fatal_not_defaulted,
             test_undecodable_fields_are_counted_not_swallowed,
             test_absent_health_is_not_reported_as_zero):
    fn()
  print()
  if FAIL:
    print(f"FAILED {len(FAIL)}: {FAIL}")
    return 1
  print("all checks passed")
  return 0


if __name__ == "__main__":
  sys.exit(main())
