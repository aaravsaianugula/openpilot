#!/usr/bin/env python3
"""
Tests for the steering headroom indicator's decision logic.

The widget needs pyray and a running UI, which CI cannot stand up. Its *decisions* -- which
band a command falls in, how long a crossing is held, how the ceiling's intensity follows
dwell, where the peak mark sits -- are pure and live in steer_headroom.py precisely so they
can be tested. This imports that module by path, so it exercises the code that ships rather
than a copy of it that would quietly drift.

Two things here are worth more than the boundary arithmetic:

  * The envelope has to actually reach the value it is being coloured for. A symmetric 0.1 s
    filter, which is what upstream's arc uses, lags a short pin badly enough that the bar can
    say "at the ceiling" while its tip is still well short of it. That is tested against the
    symmetric filter directly, so the reason for the fast attack is written down in a form
    that fails if someone removes it.
  * White blends to cyan in RGB, not through blend_colors(). blend_colors walks HSV by the
    shortest hue path and white's hue reads as 0, so the HSV route sweeps backwards through
    magenta into pale lavender -- which is the colour this design reserves for the ceiling.
    The blend is asserted to stay cool across its whole range.

    python .elantra/test_steer_headroom.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = (Path(__file__).resolve().parent.parent
          / "openpilot/selfdrive/ui/sunnypilot/mici/onroad/steer_headroom.py")


def load_module():
    if not MODULE.is_file():
        raise SystemExit(
            f"steer_headroom.py is missing at {MODULE}.\n"
            + "The indicator's decision logic is gone, which means the arc no longer knows "
            + "where the stock ceiling was.")
    spec = importlib.util.spec_from_file_location("steer_headroom", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sh = load_module()

failures: list[str] = []
passes = 0


def case(name: str, got, want) -> None:
    global passes
    ok = got == want
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f": got {got!r}, want {want!r}"))
    if ok:
        passes += 1
    else:
        failures.append(name)


def near(name: str, got: float, lo: float, hi: float) -> None:
    global passes
    ok = lo <= got <= hi
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f": got {got!r}, want {lo}..{hi}"))
    if ok:
        passes += 1
    else:
        failures.append(name)


class Clock:
    """A clock the test drives, so timing is asserted rather than slept through."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


FPS = 60.0
DT = 1.0 / FPS


def make():
    """A primed state: one update so the next one sees a real dt."""
    clock = Clock()
    state = sh.HeadroomState(clock=clock)
    state.update(0.0, True)
    return state, clock


def hold(state, clock, counts: float, seconds: float, lat_active: bool = True,
         v_ego_raw: float | None = None) -> None:
    for _ in range(max(1, round(seconds * FPS))):
        clock.advance(DT)
        state.update(counts, lat_active, v_ego_raw)


def sweep(state, clock, values, lat_active: bool = True) -> None:
    for v in values:
        clock.advance(DT)
        state.update(v, lat_active)


def main() -> int:
    print(f"Steering headroom decision logic\n  module: {MODULE}\n")

    print("[schedule] the ceiling follows speed, and the arc has to follow it exactly")
    # The bug this section exists to prevent: with a fixed 409 the arc paints the red
    # at-the-limit tier at 409 counts while the car still has 91 counts of authority left,
    # in precisely the speed range the raise was made for. Every number here is what
    # CarControllerParams.steer_max_at() returns for the same speed.
    case("standstill is the full raised ceiling", sh.ceiling_at(0.0), 450)
    case("still full at 20 mph exactly", sh.ceiling_at(8.94), 450)
    case("halfway down the ramp is 430, banker's rounding", sh.ceiling_at((8.94 + 13.41) / 2), 430)
    case("back to 409 by 30 mph", sh.ceiling_at(13.41), 409)
    case("and stays there on the freeway", sh.ceiling_at(31.0), 409)
    case("monotonic across the whole range",
         [sh.ceiling_at(v / 10.0) for v in range(500)]
         == sorted([sh.ceiling_at(v / 10.0) for v in range(500)], reverse=True), True)
    case("never above what panda will pass",
         max(sh.ceiling_at(v / 10.0) for v in range(500)) <= 512, True)

    print("\n[schedule] which is the whole point: 409 is not the edge at low speed")
    state, clock = make()
    hold(state, clock, 409, 1.0, v_ego_raw=5.0)
    case("409 at 5 m/s is headroom, NOT the limit", state.tier, sh.TIER_HEADROOM)
    case("and it is not painted red", state.color() != sh.COLOR_LIMIT, True)
    state, clock = make()
    hold(state, clock, 445, 1.0, v_ego_raw=5.0)
    case("445 at 5 m/s is approaching the edge", state.tier, sh.TIER_NEAR)
    state, clock = make()
    hold(state, clock, 450, 2.0, v_ego_raw=5.0)
    case("450 at 5 m/s IS the edge", state.tier, sh.TIER_LIMIT)
    case("and it is painted red", state.color(), sh.COLOR_LIMIT)

    print("\n[schedule] and the freeway is untouched, bit for bit")
    state, clock = make()
    hold(state, clock, 409, 2.0, v_ego_raw=25.0)
    case("409 at 25 m/s is still the edge", state.tier, sh.TIER_LIMIT)
    case("still red", state.color(), sh.COLOR_LIMIT)
    state, clock = make()
    hold(state, clock, 404, 1.0, v_ego_raw=25.0)
    case("404 at 25 m/s is still purple", state.tier, sh.TIER_NEAR)

    print("\n[schedule] the purple band is the last 9 counts, wherever the edge is")
    case("purple starts at 400 when the edge is 409", sh.near_threshold(409), 400)
    case("and at 441 when the edge is 450", sh.near_threshold(450), 441)
    case("440 below the low-speed edge is only headroom",
         sh.tier_for(440, ceiling=450), sh.TIER_HEADROOM)
    case("441 is not", sh.tier_for(441, ceiling=450), sh.TIER_NEAR)

    print("\n[schedule] the ceiling tracks speed within one live state")
    state, clock = make()
    hold(state, clock, 409, 0.3, v_ego_raw=5.0)
    case("edge is 450 while slow", state.ceiling, 450)
    hold(state, clock, 409, 0.3, v_ego_raw=25.0)
    case("edge became 409 once up to speed", state.ceiling, 409)
    case("and the same 409 command is now AT the limit", state.tier, sh.TIER_LIMIT)

    print("\n[schedule] omitting the speed leaves the ceiling alone")
    # A car with no schedule -- any non-CN7 -- never passes a speed, and must keep the
    # ceiling it was constructed with rather than silently jumping to the raised value.
    state, clock = make()
    hold(state, clock, 300, 0.2)
    case("no speed given, ceiling unchanged", state.ceiling, sh.RAISED_COUNTS)

    print("\n[bands] 384 is the old ceiling, so being AT it is not past it")
    case("383 is normal", sh.tier_for(383), sh.TIER_NORMAL)
    case("384 is normal", sh.tier_for(384), sh.TIER_NORMAL)
    case("385 is headroom", sh.tier_for(385), sh.TIER_HEADROOM)
    case("399 is the extended band", sh.tier_for(399), sh.TIER_HEADROOM)
    case("400 is about to hit", sh.tier_for(400), sh.TIER_NEAR)
    case("408 is still about to hit", sh.tier_for(408), sh.TIER_NEAR)
    case("409 is the limit", sh.tier_for(409), sh.TIER_LIMIT)
    case("410 cannot happen but is still the limit", sh.tier_for(410), sh.TIER_LIMIT)
    case("0 is normal", sh.tier_for(0), sh.TIER_NORMAL)
    print("  -- and the car steers both ways")
    case("-385 is headroom", sh.tier_for(-385), sh.TIER_HEADROOM)
    case("-409 is the limit", sh.tier_for(-409), sh.TIER_LIMIT)
    case("-404 is about to hit", sh.tier_for(-404), sh.TIER_NEAR)
    case("-384 is normal", sh.tier_for(-384), sh.TIER_NORMAL)

    print("\n[car gate] 384 is comma's HKG default -- it means nothing on another brand")
    case("hyundai carrying the flag", sh.is_raised("hyundai", sh.RAISED_LIMITS_FLAG), True)
    case("hyundai, flag among others",
         sh.is_raised("hyundai", sh.RAISED_LIMITS_FLAG | 2 ** 3 | 2 ** 11), True)
    case("hyundai without the flag", sh.is_raised("hyundai", 0), False)
    case("hyundai with the neighbouring ALT_LIMITS_2 bit only", sh.is_raised("hyundai", 2 ** 26), False)
    case("another brand carrying the same bit", sh.is_raised("toyota", sh.RAISED_LIMITS_FLAG), False)
    # opendbc defines RAISED_LIMITS twice. 1024 is the safety-param bit, and in CarParams.flags
    # it is a different flag on a different platform -- reading it here would arm the wrong car.
    case("the safety-param bit is not the car flag", sh.is_raised("hyundai", 1024), False)
    case("the car flag is not the safety-param bit", sh.RAISED_LIMITS_FLAG != 1024, True)

    print("\n[bands] the colour boundary IS the 384 line, so it has to be exact")
    # There is no mark at 384 any more: with the arc only 268 px wide, a tick was saying the
    # same thing as the colour change twice over. So where the bar leaves white is now the
    # entire signal, which makes tier_for's boundary the feature rather than a detail.
    case("384 itself is still cyan", sh.tier_for(sh.STOCK_COUNTS), sh.TIER_NORMAL)
    case("385 leaves white", sh.tier_for(sh.STOCK_COUNTS + 1), sh.TIER_HEADROOM)
    case("and there is no gap between the two bands",
         sh.tier_for(sh.STOCK_COUNTS) != sh.tier_for(sh.STOCK_COUNTS + 1), True)

    print("\n[crossing] rare and short, so it is held before it may fade")
    state, clock = make()
    hold(state, clock, 392, 0.5)
    near("cyan is lit while past 384", state.headroom, 0.95, 1.0)
    hold(state, clock, 0, 0.5)
    near("still lit half a second after dropping back", state.headroom, 0.95, 1.0)
    hold(state, clock, 0, 1.5)
    near("faded once the hold expired", state.headroom, 0.0, 0.1)

    print("\n[crossing] a command that never passes 384 never lights it")
    state, clock = make()
    hold(state, clock, 384, 1.0)
    near("384 held for a second stays white", state.headroom, 0.0, 0.001)
    case("and reports the normal band", state.tier, sh.TIER_NORMAL)

    print("\n[envelope] the tip must reach the value it is being coloured for")
    # A real pin: rate limiting is 3 counts per 100 Hz frame, so ~5 counts per 60 fps frame.
    ramp = list(range(300, 409, 5)) + [409] * 4
    state, clock = make()
    sweep(state, clock, ramp)
    fast = abs(state.envelope)
    # The same input through the symmetric 0.1 s filter upstream's arc uses.
    slow = 300.0
    for v in ramp:
        slow = sh._toward(slow, float(v), 0.1, DT)
    near("attack/release envelope arrives at the ceiling", fast, 403.0, 409.0)
    case("the symmetric filter upstream uses does not", slow < 403.0, True)
    case("and the band is reported from the raw command, not the envelope",
         state.tier, sh.TIER_LIMIT)

    print("\n[limit] red arrives on contact; dwell drives how hard it breathes")
    # Hitting the limit is not a matter of degree -- purple already said "about to" -- so the
    # colour does not ease in with dwell the way it used to. Dwell only sets the pulse.
    state, clock = make()
    hold(state, clock, 409, 0.30)
    near("red is essentially full a third of a second in", state.limit, 0.90, 1.0)
    near("but a graze barely breathes", state.pulse_depth,
         sh.PULSE_DEPTH_MIN, sh.PULSE_DEPTH_MIN + 0.04)
    state, clock = make()
    hold(state, clock, 409, 1.5)
    near("a sustained pin breathes at full depth", state.pulse_depth,
         sh.PULSE_DEPTH_MAX - 1e-6, sh.PULSE_DEPTH_MAX)
    state, clock = make()
    hold(state, clock, 300, 1.0)
    case("never at the limit means no red at all", state.limit, 0.0)
    case("and no purple either, at 300", state.near, 0.0)

    print("\n[ceiling] dwell sheds slower than it accrues, so repeated taps build")
    state, clock = make()
    hold(state, clock, 409, 1.0)
    hold(state, clock, 200, 1.0)
    near("a second pinned survives a second released", state.dwell, 0.40, 0.50)
    state, clock = make()
    for _ in range(6):
        hold(state, clock, 409, 0.2)
        hold(state, clock, 200, 0.2)
    near("six taps at one intersection accumulate", state.dwell, 0.40, 0.70)
    case("and read as more than a single tap",
         state.pulse_depth > sh.PULSE_DEPTH_MIN + 0.04, True)

    print("\n[peak] sits still long enough to glance at, then walks back to the bar")
    state, clock = make()
    hold(state, clock, 409, 0.2)
    hold(state, clock, 100, 1.0)
    near("held at the peak a second later", state.peak, 405.0, 409.0)
    case("and is worth drawing", state.peak_visible, True)
    case("tinted by the band it reached", state.peak_color(), sh.COLOR_LIMIT)
    hold(state, clock, 100, 5.0)
    near("decaying afterwards", state.peak, 300.0, 385.0)
    hold(state, clock, 100, 60.0)
    near("floors on the live bar rather than falling past it", state.peak, 99.0, 101.0)
    case("and stops being drawn there", state.peak_visible, False)

    print("\n[peak] the mark fades on time, so a fast release cannot pop it in")
    state, clock = make()
    hold(state, clock, 409, 0.5)
    # The worst case the car can actually produce: coming off the ceiling the command falls at
    # 7 counts per 100 Hz frame, so the gap between the peak and the bar opens further in one
    # rendered frame than any gap-width ramp could cover. Time is what makes this smooth.
    clock.advance(DT)
    state.update(120, True)
    near("one frame after a full release it is still nearly invisible", state.peak_alpha, 0.0, 0.15)
    hold(state, clock, 120, 0.15)
    near("part way in after 150 ms", state.peak_alpha, 0.2, 0.8)
    hold(state, clock, 120, 0.9)
    near("fully visible about a second later", state.peak_alpha, 0.9, 1.0)
    hold(state, clock, 409, 0.7)
    near("and fades back out once the bar catches it up", state.peak_alpha, 0.0, 0.1)

    print("\n[peak] nothing survives lateral going inactive")
    state, clock = make()
    hold(state, clock, 409, 0.5)
    hold(state, clock, 0, 0.05, lat_active=False)
    case("peak cleared", state.peak, 0.0)
    case("dwell cleared", state.dwell, 0.0)

    print("\n[colour] four bands, four colours, and they have to stay apart")
    # Pink and crimson are the pair at risk of reading as one colour through a windscreen in
    # sunlight, and they are only ~17 degrees apart in hue -- they are separated by lightness
    # and chroma instead. Hue angle cannot see that, so measure perceptual distance (CIE76).
    # A just-noticeable difference is about 2.3; these need to survive glare and motion.
    def lab(c):
        def lin(u):
            u /= 255.0
            return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

        r, g, b = (lin(v) for v in c)
        x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

        def f(t):
            return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

        fx, fy, fz = f(x), f(y), f(z)
        return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))

    def delta_e(a, b):
        return sum((u - v) ** 2 for u, v in zip(lab(a), lab(b), strict=True)) ** 0.5

    trio = (sh.COLOR_BASE, sh.COLOR_HEADROOM, sh.COLOR_NEAR, sh.COLOR_LIMIT)
    for i, j, label in ((0, 1, "white vs cyan"), (1, 2, "cyan vs purple"), (2, 3, "purple vs red")):
        case(label + " are at least 25 apart perceptually", delta_e(trio[i], trio[j]) >= 25, True)
        case(label + " differ by at least 100 in RGB",
             sum(abs(x - y) for x, y in zip(trio[i], trio[j], strict=True)) >= 100, True)
    # The one that actually distinguishes the adjacent pair, so it cannot quietly erode.
    # White to cyan is the weakest adjacent step and it is a lightness one, L* 100 to 86 --
    # a hue measure cannot see it at all, which is why the pairs above use CIE76.
    case("the base really is upstream white, so the arc says nothing below 384",
         sh.COLOR_BASE, sh.COLOR_WHITE)
    case("each step down the sequence gets darker",
         [round(lab(c)[0]) for c in trio] == sorted((round(lab(c)[0]) for c in trio),
                                                    reverse=True), True)
    case("all four are pastel, not saturated",
         all(min(c) >= 120 and max(c) >= 230 for c in trio), True)

    print("\n[colour] the wash toward the middle of the bar keeps the band's own hue")
    # The centre of the bar is the tier colour washed halfway to white. Hue is the right
    # measure here -- unlike the pairwise check above, this is about a single colour not
    # drifting into another family as it lightens.
    def hue(c):
        r, g, b = (v / 255 for v in c)
        hi, lo = max(r, g, b), min(r, g, b)
        if hi == lo:
            return 0.0
        d = hi - lo
        if hi == r:
            h = ((g - b) / d) % 6
        elif hi == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        return h * 60

    def hue_apart(a, b):
        d = abs(hue(a) - hue(b))
        return min(d, 360 - d)

    ok = True
    for c in trio:
        for i in range(11):
            w = sh.lerp_rgb(c, sh.COLOR_WHITE, i / 10)
            ok = ok and (i == 10 or hue_apart(w, c) < 12)
    case("hue is preserved along every wash", ok, True)

    print("\n[colour] and the bands map to the right ones")
    state, clock = make()
    hold(state, clock, 0, 0.5)
    case("white below the stock ceiling", state.color(), sh.COLOR_BASE)
    state, clock = make()
    hold(state, clock, 300, 1.0)
    case("still white right up to it", state.color(), sh.COLOR_BASE)
    state, clock = make()
    hold(state, clock, 392, 1.0)
    case("cyan in the band we added", state.color(), sh.COLOR_HEADROOM)
    state, clock = make()
    hold(state, clock, 404, 1.0)
    case("purple approaching the edge", state.color(), sh.COLOR_NEAR)
    state, clock = make()
    hold(state, clock, 409, 2.0)
    case("red at the edge", state.color(), sh.COLOR_LIMIT)

    print("\n[time] the animation must survive a stalled UI and a first frame")
    state = sh.HeadroomState(clock=Clock())
    state.update(409, True)
    case("the first frame advances nothing", state.envelope, 0.0)
    case("but still reports the band", state.tier, sh.TIER_LIMIT)
    state, clock = make()
    hold(state, clock, 409, 1.0)
    clock.advance(5.0)
    state.update(0.0, True)
    near("a five second stall is one clamped frame, not a teleport", state.headroom, 0.5, 1.0)

    print("\n" + "-" * 62)
    if failures:
        print(f"FAILED: {len(failures)} case(s) of {len(failures) + passes}")
        for f in failures:
            print("  - " + f)
        return 1
    if passes < 50:
        print(f"FAILED: only {passes} cases ran. Cases were removed, not fixed.")
        return 1
    print(f"PASSED: {passes} cases -- the indicator's decisions behave as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
