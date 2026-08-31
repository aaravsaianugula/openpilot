"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Where the delivered steering command sits relative to the two ceilings that matter.

This build raises the CN7's steering ceiling from comma's HKG default of 384 counts to a flat
409 (HyundaiFlags.RAISED_LIMITS). The onroad torque arc normalises by STEER_MAX, so 409 draws
exactly where 384 used to and the extra authority is invisible from the driver's seat. This
module decides what the arc should say about it; steer_headroom_bar.py draws the decision.

The events are rare and short -- 0.49% of a measured drive was above 384, in bursts of a few
frames, almost all of it below 10 m/s -- and the band itself is 6.1% of the arc. So most of
what is here is about making a brief, eight-pixel event legible: a hold on the crossing so it
outlives the frames that caused it, dwell-weighted intensity at the ceiling, and a peak that
sits still long enough to glance at.

Everything in this file is pure: no pyray, no cereal, an injected clock. The widget needs a
running UI and CI cannot stand one up, but these decisions can be tested on their own -- see
.elantra/test_steer_headroom.py, which imports this file by path so it exercises the code that
ships rather than a copy of it.

Two invariants are relied on here and deliberately not re-derived:
  * RAISED_LIMITS implies STEER_MAX == 409, assigned inside the `else` branch of opendbc's
    STEER_MAX chain so it can never outrank CANFD / ALT_LIMITS / ALT_LIMITS_2.
  * The stock ceiling under it is still 384.
Both are already enforced by guards.guard_raised_torque_pair. The numbers are duplicated here
so the UI needs no import-time dependency on a brand module, and guards.guard_ui_headroom fails
the build if the copies ever disagree with opendbc.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Ceilings, in CAN counts. Pinned against opendbc by guards.guard_ui_headroom.
STOCK_COUNTS = 384
RAISED_COUNTS = 409
HYUNDAI_BRAND = "hyundai"

# HyundaiFlags.RAISED_LIMITS, which is what CarParams.flags carries and what opendbc's own
# CarControllerParams tests. NOT HyundaiSafetyFlags.RAISED_LIMITS: opendbc defines that name
# twice, 2**27 on the car flags and 1024 on the safety param, and 1024 in CarParams.flags is a
# different flag on a different platform. guards.guard_ui_headroom pins both halves.
RAISED_LIMITS_FLAG = 2 ** 27

TIER_NORMAL = 0
TIER_HEADROOM = 1
TIER_NEAR = 2
TIER_LIMIT = 3

# Four states, four colours, all pastel so they sit on a camera feed without shouting.
NEAR_COUNTS = 400
COLOR_BASE = (255, 255, 255)      # white           -- the bounds the car always had, to 384
COLOR_HEADROOM = (140, 230, 245)  # #8ce6f5  cyan   -- the extended bounds, 385 to 399
COLOR_NEAR = (186, 148, 240)      # #ba94f0  purple -- 400 to 408: approaching the edge
COLOR_LIMIT = (236, 124, 138)     # #ec7c8a  red    -- 409: at the edge

# White for the base is not a placeholder: it is what upstream's arc already draws, so below
# 384 this looks like the stock bar and says nothing. The indicator only speaks once there is
# something to say, which is the whole point of it.
#
# Every adjacent pair is far apart perceptually (CIE76 31, 64, 55 along the sequence), which
# matters more than hue angle here -- white to cyan is a lightness step, L* 100 to 86, that a
# hue measure cannot see at all. All four are held to the same pastel floor rather than letting
# red go saturated: the brief is pastel, and what makes the edge insistent is the pulse.
COLOR_WHITE = (255, 255, 255)     # the wash toward the middle of the bar, and the unarmed arc

# Each crossing is an event, so it rises fast and is held before it is allowed to fade. The
# bands above 384 are narrow -- 15 counts of pink, 9 of purple -- and at 3 counts per 100 Hz
# frame the car crosses them in a few hundred milliseconds, so without the hold they would be
# gone before they were seen.
HEADROOM_RISE = 0.10
HEADROOM_FADE = 0.40
HEADROOM_HOLD = 0.70

# Hitting the limit is not a matter of degree: red arrives on contact rather than easing in,
# because purple already says "about to". Dwell no longer drives the colour, only how hard the
# red breathes -- a graze pulses gently, a sustained pin insistently. Dwell sheds more slowly
# than it accrues, so a run of taps at one intersection builds rather than resetting.
PIN_DWELL_FULL = 1.20
DWELL_DECAY = 0.55
DWELL_MAX = 2.40

PULSE_HZ = 1.40
PULSE_DEPTH_MIN = 0.04
PULSE_DEPTH_MAX = 0.14

# The peak mark holds still long enough to look at after the corner, then walks back down to
# the live bar rather than vanishing.
PEAK_HOLD = 2.50
PEAK_DECAY = 20.0
# The mark is only worth drawing once it has separated from the bar's own tip, but it has to
# fade on TIME rather than on the size of that gap: coming off the ceiling the command falls at
# 7 counts per 100 Hz frame, so a gap-width fade opens fully inside a single rendered frame,
# which is a pop wearing a ramp's clothing. Hysteresis on the gap, first-order fade on the alpha.
PEAK_GAP_ON = 10.0
PEAK_GAP_OFF = 6.0
PEAK_RISE = 0.22
PEAK_FALL = 0.18

# Below this an element is not worth a draw call. A first-order fade never reaches zero, so
# something has to say where 'gone' is; the widget gates on the same number.
VISIBLE_EPS = 0.02

# Fast attack so the tip actually reaches the peak it is being coloured for, slower release so
# it does not twitch. Safe because the command is rate limited to 3 counts per 100 Hz frame on
# the way up, so the input is already smooth.
ENVELOPE_ATTACK = 0.03
ENVELOPE_RELEASE = 0.14

# A stalled UI must not teleport the animation. Longer gaps are treated as one long frame.
MAX_DT = 0.25


def is_raised(brand: str, flags: int) -> bool:
  """Does this car actually have the raised ceiling, so that 384 means anything to it?

  384 is comma's HKG default. On any other brand it is not a line the car has ever had, so the
  indicator stays off and the arc renders exactly as upstream draws it.
  """
  return brand == HYUNDAI_BRAND and bool(int(flags) & RAISED_LIMITS_FLAG)


def tier_for(counts: float, stock: int = STOCK_COUNTS, ceiling: int = RAISED_COUNTS) -> int:
  """Which band a command falls in. 384 itself is the old ceiling, not past it."""
  mag = abs(counts)
  if mag >= ceiling:
    return TIER_LIMIT
  if mag >= NEAR_COUNTS:
    return TIER_NEAR
  if mag > stock:
    return TIER_HEADROOM
  return TIER_NORMAL


def _clip01(x: float) -> float:
  return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], f: float) -> tuple[int, int, int]:
  """Straight RGB interpolation.

  Deliberately not blend_colors(): that walks HSV by the shortest hue path, and white has zero
  saturation so its hue reads as 0 (red). Blending white toward cyan through it sweeps backwards
  through magenta and lands on pale lavender mid-way -- which is the ceiling tier's colour, on
  a bar that is supposed to be nowhere near it. RGB has no hue to get wrong between colours
  this light, and every blend here has white or another pastel at one end.
  """
  f = _clip01(f)
  return (round(a[0] + (b[0] - a[0]) * f),
          round(a[1] + (b[1] - a[1]) * f),
          round(a[2] + (b[2] - a[2]) * f))


def _toward(value: float, target: float, tau: float, dt: float) -> float:
  """First-order approach. dt <= 0 leaves the value alone; tau <= 0 snaps to target."""
  if dt <= 0.0:
    return value
  if tau <= 0.0:
    return target
  return value + (dt / (tau + dt)) * (target - value)


class HeadroomState:
  """Everything the arc needs to know, advanced one frame at a time.

  Fed with the signed CAN counts actually sent to the car
  (carOutput.actuatorsOutput.torqueOutputCan), not the value normalised by STEER_MAX, so the
  thresholds stay true regardless of what the normalisation divides by.
  """

  def __init__(self, ceiling: int = RAISED_COUNTS, stock: int = STOCK_COUNTS,
               clock: Callable[[], float] = time.monotonic):
    self.ceiling = int(ceiling)
    self.stock = int(stock)
    self._clock = clock
    self._last_t: float | None = None
    self.reset()

  def reset(self) -> None:
    self.counts = 0.0       # raw signed counts, this frame
    self.envelope = 0.0     # signed counts, attack/release smoothed; drives the bar geometry
    self.headroom = 0.0     # 0..1, how far into pink
    self.near = 0.0         # 0..1, how far into purple
    self.limit = 0.0        # 0..1, how far into red
    self.dwell = 0.0        # seconds spent at the limit, net of decay
    self.pulse_depth = 0.0  # 0..1, amplitude of the red's breathing
    self.peak = 0.0         # highest |counts| still being remembered
    self.peak_sign = 1.0
    self.peak_alpha = 0.0   # 0..1, how visible the mark is
    self._peak_on = False
    self._headroom_hold = 0.0
    self._near_hold = 0.0
    self._limit_hold = 0.0
    self._peak_hold = 0.0

  @property
  def tier(self) -> int:
    return tier_for(self.counts, self.stock, self.ceiling)

  @property
  def peak_tier(self) -> int:
    return tier_for(self.peak, self.stock, self.ceiling)

  @property
  def peak_visible(self) -> bool:
    return self.peak_alpha > VISIBLE_EPS

  def update(self, counts: float, lat_active: bool) -> None:
    now = self._clock()
    dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), MAX_DT)
    self._last_t = now

    if not lat_active:
      # Nothing is being commanded, so there is nothing to remember. A peak that survived a
      # stop would be read as something this drive did.
      counts = 0.0
      self.peak = 0.0
      self._peak_on = False
      self.dwell = 0.0
      self._headroom_hold = 0.0
      self._near_hold = 0.0
      self._limit_hold = 0.0

    self.counts = float(counts)
    mag = abs(self.counts)

    tau = ENVELOPE_ATTACK if mag > abs(self.envelope) else ENVELOPE_RELEASE
    self.envelope = _toward(self.envelope, self.counts, tau, dt)

    if mag > self.stock:
      self._headroom_hold = now + HEADROOM_HOLD
      self.headroom = _toward(self.headroom, 1.0, HEADROOM_RISE, dt)
    elif now >= self._headroom_hold:
      self.headroom = _toward(self.headroom, 0.0, HEADROOM_FADE, dt)

    # ...then purple once it is close to the limit, then red once it is on it. Same shape each
    # time, so a fast run up through all three reads as a sequence rather than a jump.
    if mag >= NEAR_COUNTS:
      self._near_hold = now + HEADROOM_HOLD
      self.near = _toward(self.near, 1.0, HEADROOM_RISE, dt)
    elif now >= self._near_hold:
      self.near = _toward(self.near, 0.0, HEADROOM_FADE, dt)

    if mag >= self.ceiling:
      self._limit_hold = now + HEADROOM_HOLD
      self.limit = _toward(self.limit, 1.0, HEADROOM_RISE, dt)
    elif now >= self._limit_hold:
      self.limit = _toward(self.limit, 0.0, HEADROOM_FADE, dt)

    if mag >= self.ceiling:
      self.dwell = min(self.dwell + dt, DWELL_MAX)
    else:
      self.dwell = max(self.dwell - dt * DWELL_DECAY, 0.0)
    full = _clip01(self.dwell / PIN_DWELL_FULL)
    self.pulse_depth = PULSE_DEPTH_MIN + (PULSE_DEPTH_MAX - PULSE_DEPTH_MIN) * full

    gap = self.peak - abs(self.envelope)
    if gap > PEAK_GAP_ON:
      self._peak_on = True
    elif gap < PEAK_GAP_OFF:
      self._peak_on = False
    self.peak_alpha = _toward(self.peak_alpha, 1.0 if self._peak_on else 0.0,
                              PEAK_RISE if self._peak_on else PEAK_FALL, dt)

    if mag >= self.peak:
      self.peak = mag
      self.peak_sign = 1.0 if self.counts >= 0.0 else -1.0
      self._peak_hold = now + PEAK_HOLD
    elif now >= self._peak_hold:
      # Walks back down to the live command and stops there, rather than vanishing. The floor
      # is mag, not the envelope: the branch above re-latches on the next frame anyway, so an
      # envelope floor would read as load-bearing while never actually being reached.
      self.peak = max(mag, self.peak - dt * self.ceiling / PEAK_DECAY)

  def color(self) -> tuple[int, int, int]:
    """The bar's colour: cyan, warmed to pink, then purple, then red -- in that order.

    Composed rather than switched, so a crossing is a blend between two neighbours and never a
    jump between two colours that are far apart.
    """
    rgb = lerp_rgb(COLOR_BASE, COLOR_HEADROOM, self.headroom)
    rgb = lerp_rgb(rgb, COLOR_NEAR, self.near)
    return lerp_rgb(rgb, COLOR_LIMIT, self.limit)

  def peak_color(self) -> tuple[int, int, int]:
    """The peak mark carries the tier it is currently sitting in, so it cools as it decays."""
    return {TIER_LIMIT: COLOR_LIMIT, TIER_NEAR: COLOR_NEAR,
            TIER_HEADROOM: COLOR_HEADROOM}.get(self.peak_tier, COLOR_BASE)
