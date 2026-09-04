"""Tests for the demand-side limits in clip_curvature.

clip_curvature is the first amplitude gate on the model's plan: controlsd calls it at 100 Hz on
modelV2.action.desiredCurvature and hands the result to the lateral controller. Its lateral-accel
clamp is expressed as an ACCELERATION, so the tightest radius it will command scales as
v^2 / limit -- 15 m at 15 mph and 81 m at 35 mph under the stock 3.0 m/s^2, against real turn radii
of 7.5-20 m. Measured over 3.41M engaged frames on the CN7 that clamp is active on 8% of turn
frames at 11-13 mph rising to 80% at 31-36 mph, and it accounts for 92.3% of the frames that raise
"Turn Exceeds Steering Limit".

These tests pin the three things that must stay true after raising it:
  * the highway is bit-identical, because the clamp is measured never to bind above 18 m/s;
  * the city limit really did move, which is the whole point of the change;
  * nothing else in the function moved -- the jerk clamp, the MAX_CURVATURE clamp, the roll term
    and the curvature_limited semantics that the alert depends on.
"""
import numpy as np
import pytest

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib import drive_helpers
from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_CURVATURE,
  MAX_LATERAL_ACCEL_NO_ROLL,
  MAX_LATERAL_JERK,
  clip_curvature,
)

# Frames captured from the archive (D:/comma_four/routes). Each row is a real control frame that an
# exact offline replay of the STOCK clip_curvature reproduces to within 1e-9 of the logged
# controlsState.desiredCurvature, so it pins observed behaviour rather than a re-derivation of it.
# Stored at full float precision on purpose: rounding v_ego to 6 dp alone moves the accel-clamp
# ceiling by ~3e-9 /m, which is enough to break the 1e-9 assertion below.
# (v_ego, roll, prev_curvature, model_curvature, expected_output, which_clamp)
LOG_FIXTURE = [
  (5.127913951873779, 0.011340639553964138, 0.11871398240327835, 0.11871398240327835, 0.11831878125667572, "accel"),
  (5.129669189453125, 0.03155473619699478, -0.1023479774594307, -0.10264115035533905, -0.10224589705467224, "accel"),
  (5.6410017013549805, 0.011340639553964138, 0.09825432300567627, 0.1102876216173172, 0.09777384251356125, "accel"),
  (5.667961597442627, 0.022612497210502625, -0.0865059345960617, -0.09327679872512817, -0.08647792041301727, "accel"),
  (6.550593376159668, 0.022612497210502625, -0.06500363349914551, -0.08992813527584076, -0.06474373489618301, "accel"),
  (6.570066928863525, 0.010934718884527683, 0.07230232656002045, 0.07747391611337662, 0.07198455929756165, "accel"),
  (7.561275482177734, 0.022612497210502625, -0.04879145324230194, -0.058581870049238205, -0.0485924631357193, "accel"),
  (8.499967575073242, -0.00572592206299305, 0.04070746526122093, 0.04785415530204773, 0.04074534401297569, "accel"),
  (8.516148567199707, 0.012983284890651703, -0.03971865028142929, -0.062359344214200974, -0.039608996361494064, "accel"),
  (9.406407356262207, -0.00572592206299305, 0.03297134116292, 0.03375205770134926, 0.033270932734012604, "accel"),
  (9.31119441986084, 0.02726190723478794, 0.024234244599938393, 0.023558173328638077, 0.023657532408833504, "jerk"),
  (10.436959266662598, 0.03402227908372879, 0.007775590755045414, 0.008489072322845459, 0.00823460053652525, "jerk"),
  (11.428960800170898, 0.04038582742214203, 0.0016015938017517328, 0.0020068392623215914, 0.0019843801856040955, "jerk"),
  (11.864421844482422, 0.04347681254148483, -0.0001759091974236071, 0.00019775460532400757, 0.00017929398745764047, "jerk"),
  (12.18820571899414, 0.043991819024086, -0.0016792984679341316, -0.001323320553638041, -0.0013427168596535921, "jerk"),
  (12.780987739562988, 0.0429939366877079, -0.003603309625759721, -0.003278669435530901, -0.0032972253393381834, "jerk"),
  (15.048542022705078, 0.0296164657920599, -0.00044072006130591035, -0.0007278841803781688, -0.0006615109741687775, "jerk"),
  (16.001880645751953, 0.030468059703707695, 0.00210534012876451, 0.0018706772243604064, 0.001910073566250503, "jerk"),
  (21.310949325561523, 0.06193748489022255, 2.7179794415133074e-05, -2.4053299057413824e-05, -2.4053299057413824e-05, "none"),
  (21.4991397857666, 0.0851290225982666, 0.0015082847094163299, 0.0015082847094163299, 0.0015082847094163299, "none"),
  (21.635160446166992, 0.018461398780345917, 1.878792500065174e-05, -1.1096465641458053e-05, -1.1096465641458053e-05, "none"),
  (22.493680953979492, 0.052187222987413406, 0.001621016999706626, 0.001621016999706626, 0.001621016999706626, "none"),
  (23.49713134765625, 0.04985013231635094, 0.0016103677917271852, 0.0016103677917271852, 0.0016103677917271852, "none"),
  (24.497587203979492, 0.022731434553861618, 0.00016707397298887372, 0.00017357968317810446, 0.00017357968317810446, "none"),
  (25.394195556640625, 0.023203516378998756, 1.701877590676304e-05, 1.701877590676304e-05, 1.701877590676304e-05, "none"),
  (25.445541381835938, 0.021676843985915184, -0.00014174831449054182, -0.00014174831449054182, -0.00014174831449054182, "none"),
]

# Speeds where the archive shows the clamp binding on real turns, and where it never does.
CITY_SPEEDS = (5.0, 6.7, 8.0, 10.0, 12.0, 14.0, 15.6)
HIGHWAY_SPEEDS = (22.0, 25.0, 30.0, 40.0)

HUGE = 10.0  # a curvature demand far beyond every clamp, so only the clamps decide the output


def settled_curvature(v_ego, roll=0.0, sign=1.0):
  """The steady-state |curvature| clip_curvature permits at this speed.

  Run to convergence on purpose: a single step is rate-limited by the lateral-jerk clamp, so a
  one-shot call measures the jerk clamp and not the acceleration ceiling this change is about.
  """
  k = 0.0
  for _ in range(4000):
    k, _ = clip_curvature(v_ego, k, sign * HUGE, roll)
  return k


def settled_lat_accel(v_ego, roll=0.0, sign=1.0):
  return settled_curvature(v_ego, roll, sign) * max(v_ego, drive_helpers.MIN_SPEED) ** 2


class TestClipCurvatureSchedule:
  """The speed schedule itself: shape, endpoints and the bound that keeps it honest."""

  def test_schedule_returns_to_stock_before_highway(self):
    # The last breakpoint MUST be the stock constant. The archive has zero turn frames tight
    # enough to reach the clamp above 18 m/s, so there is nothing to buy up there and a raised
    # limit would be an unmeasured change to highway behaviour.
    assert drive_helpers.LAT_ACCEL_LIMIT_V[-1] == MAX_LATERAL_ACCEL_NO_ROLL

  def test_schedule_is_bounded(self):
    # 409 counts is the MDPS's acceptance limit and buys ~3.65 m/s^2 at 14-18 m/s. A limit above
    # that is headroom the car cannot deliver; it would only move the failure from "clamped" to
    # "saturated". This bound is what stops a future edit opening it wide.
    assert max(drive_helpers.LAT_ACCEL_LIMIT_V) <= 4.5

  def test_schedule_breakpoints_cover_the_complaint_band(self):
    bp = drive_helpers.LAT_ACCEL_LIMIT_BP
    assert list(bp) == sorted(bp), "np.interp requires increasing breakpoints"
    # The measured clamp rate peaks at 80% of turn frames at 14-16 m/s. If the taper starts below
    # that, the band that needs the change most sits on the ramp instead of the flat.
    assert bp[0] >= 14.0

  def test_schedule_is_non_increasing(self):
    v = drive_helpers.LAT_ACCEL_LIMIT_V
    assert list(v) == sorted(v, reverse=True)


class TestHighwayUnchanged:
  """Above the last breakpoint nothing may move, at all."""

  @pytest.mark.parametrize("v_ego", HIGHWAY_SPEEDS)
  def test_lateral_accel_ceiling_is_stock(self, v_ego):
    assert settled_lat_accel(v_ego) == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL, abs=1e-9)

  @pytest.mark.parametrize("v_ego", HIGHWAY_SPEEDS)
  def test_negative_direction_too(self, v_ego):
    assert settled_lat_accel(v_ego, sign=-1.0) == pytest.approx(-MAX_LATERAL_ACCEL_NO_ROLL, abs=1e-9)


class TestCityLimitRaised:
  """The regression the change exists for."""

  @pytest.mark.parametrize("v_ego", CITY_SPEEDS)
  def test_ceiling_is_above_stock(self, v_ego):
    assert settled_lat_accel(v_ego) > MAX_LATERAL_ACCEL_NO_ROLL + 1e-6

  def test_intersection_turn_at_15_mph(self):
    # 6.7 m/s is 15 mph. A 10 m radius intersection turn is 0.1 /m and needs 4.49 m/s^2. Stock
    # permits 3.0/6.7^2 = 0.0669 /m -- a 15 m radius, which is why the car ran wide. Assert the
    # new ceiling explicitly rather than "more than before", so a wrong schedule cannot pass.
    v_ego = 6.7
    expected = drive_helpers.LAT_ACCEL_LIMIT_V[0] / v_ego ** 2
    assert settled_curvature(v_ego) == pytest.approx(expected, rel=1e-9)
    assert settled_curvature(v_ego) > 3.0 / v_ego ** 2

  def test_taper_is_between_the_breakpoints(self):
    lo, hi = drive_helpers.LAT_ACCEL_LIMIT_BP
    mid = 0.5 * (lo + hi)
    want = 0.5 * (drive_helpers.LAT_ACCEL_LIMIT_V[0] + drive_helpers.LAT_ACCEL_LIMIT_V[1])
    assert settled_lat_accel(mid) == pytest.approx(want, rel=1e-6)


class TestUntouchedBehaviour:
  """Everything in clip_curvature that must not have moved."""

  def test_permitted_radius_is_monotonic_in_speed(self):
    # A step or inversion at a breakpoint would make the car turn tighter as it speeds up.
    speeds = np.arange(1.0, 40.01, 0.25)
    radii = [1.0 / settled_curvature(float(v)) for v in speeds]
    assert all(b >= a - 1e-9 for a, b in zip(radii, radii[1:], strict=False))

  @pytest.mark.parametrize("v_ego", (3.0, 7.0, 15.0, 30.0))
  def test_jerk_clamp_rate_is_unchanged(self, v_ego):
    # One step from a settled zero must move by exactly the lateral-jerk allowance.
    step, limited = clip_curvature(v_ego, 0.0, HUGE, 0.0)
    assert step == pytest.approx(MAX_LATERAL_JERK / v_ego ** 2 * DT_CTRL, rel=1e-9)
    # A frame limited only by the jerk clamp must NOT report curvature_limited: latcontrol.py
    # ORs that flag into the saturation timer behind "Turn Exceeds Steering Limit", and the jerk
    # clamp fires on 24-45% of turn frames purely smoothing the 20 Hz model staircase.
    # Compared as bool(): the flag is np.bool_ when the accel clamp fires (clamp() compares a
    # Python float against the np.float64 the jerk clip returned) and a plain bool otherwise.
    # That is pre-existing upstream behaviour and harmless -- latcontrol.py only tests truthiness
    # -- but it makes an `is True` assertion fail, so do not write one.
    assert bool(limited) is False

  def test_max_curvature_clamp_still_binds_and_reports(self):
    # Below sqrt(limit / MAX_CURVATURE) the flat 0.2 /m clamp is the tighter one. That is a 5.0 m
    # radius against the Elantra's ~5.3 m kerb radius, so it is the car, not a comfort choice.
    v_ego = 1.0
    assert settled_curvature(v_ego) == pytest.approx(MAX_CURVATURE, rel=1e-9)
    _, limited = clip_curvature(v_ego, MAX_CURVATURE, HUGE, 0.0)
    assert bool(limited) is True

  def test_accel_clamp_reports_curvature_limited(self):
    v_ego = 10.0
    settled = settled_curvature(v_ego)
    _, limited = clip_curvature(v_ego, settled, HUGE, 0.0)
    assert bool(limited) is True

  @pytest.mark.parametrize("roll", (-0.06, 0.0, 0.06))
  def test_roll_shifts_both_bounds_additively(self, roll):
    v_ego = 10.0
    comp = roll * ACCELERATION_DUE_TO_GRAVITY
    limit = float(np.interp(v_ego, drive_helpers.LAT_ACCEL_LIMIT_BP, drive_helpers.LAT_ACCEL_LIMIT_V))
    assert settled_lat_accel(v_ego, roll=roll) == pytest.approx(limit + comp, rel=1e-9)
    assert settled_lat_accel(v_ego, roll=roll, sign=-1.0) == pytest.approx(-limit + comp, rel=1e-9)


class TestBehaviourPreservingRefactor:
  """With the schedule forced back to the stock constant, every recorded frame must replay."""

  def test_reproduces_logged_frames(self, monkeypatch):
    monkeypatch.setattr(drive_helpers, "LAT_ACCEL_LIMIT_V",
                        [MAX_LATERAL_ACCEL_NO_ROLL, MAX_LATERAL_ACCEL_NO_ROLL])
    for v_ego, roll, prev, model, expected, which in LOG_FIXTURE:
      out, limited = clip_curvature(v_ego, prev, model, roll)
      assert out == pytest.approx(expected, abs=1e-9), f"{which} frame at {v_ego:.2f} m/s"
      assert bool(limited) == (which in ("accel", "maxcurv")), f"{which} frame at {v_ego:.2f} m/s"

  def test_fixture_covers_every_clamp_and_both_directions(self):
    kinds = {row[5] for row in LOG_FIXTURE}
    assert {"accel", "jerk", "none"} <= kinds
    for kind in ("accel", "jerk"):
      signs = {np.sign(row[3]) for row in LOG_FIXTURE if row[5] == kind}
      assert signs == {-1.0, 1.0}, f"{kind} rows must exercise both turn directions"

  def test_raised_schedule_changes_the_clamped_frames(self):
    # The mirror of the test above: with the real schedule in place the accel-clamped frames must
    # move and the others must not. A change that quietly did nothing would pass the replay test.
    moved = 0
    for v_ego, roll, prev, model, expected, which in LOG_FIXTURE:
      out, _ = clip_curvature(v_ego, prev, model, roll)
      if which == "accel":
        assert abs(out) >= abs(expected) - 1e-12
        moved += abs(out) > abs(expected) + 1e-9
      else:
        assert out == pytest.approx(expected, abs=1e-9)
    assert moved > 0, "no accel-clamped frame moved -- the schedule is not being applied"
