"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The onroad torque arc, told where the two ceilings are.

Upstream's TorqueBar normalises by STEER_MAX, so on this build 409 counts draws in exactly the
place 384 used to and the raised authority is invisible. This subclass keeps upstream's
geometry -- same arc, same radius, same growth -- and changes what it says above 384: white to
the stock ceiling exactly as upstream draws it, then pastel cyan through the band this build
added, pastel purple approaching the raised one, pastel red at it.

There is no background track and no mark at 384. The colour boundary is the line: the bar
leaves white at exactly the point the stock ceiling used to stop it, so a separate tick would
be saying the same thing twice on an arc that is 268 px wide.

It is a subclass rather than a patch so that upstream's torque_bar.py stays untouched: the arc
maths (arc_bar_pts, which is the subtle part) is imported, not copied, and the only file this
build modifies to install it is the SP mici hud renderer, two lines.

On any car without the raised ceiling this delegates to super()._render and draws upstream's
arc unchanged -- 384 is comma's HKG default and means nothing elsewhere.

Everything on screen moves continuously. There is no state in here that can pop: no boolean
gating a visible element on or off, and no quantised size that steps as the bar grows. Both were
in an earlier version and both were visible in the rendered frames. The peak mark is a flat quad
rather than an arc because over 9 px of a 1228 px radius the curvature is about four thousandths
of a pixel, and routing it through arc_bar_pts cost far more in cache misses than it costs to
draw.

The decisions live in steer_headroom.py, which is pure and tested by
.elantra/test_steer_headroom.py. What is here is drawing.
"""

import math
import time
from collections.abc import Callable

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.torque_bar import TORQUE_ANGLE_SPAN, TorqueBar, arc_bar_pts
from openpilot.selfdrive.ui.sunnypilot.mici.onroad.steer_headroom import (
  COLOR_WHITE,
  PULSE_HZ,
  VISIBLE_EPS,
  HeadroomState,
  is_raised,
  lerp_rgb,
)
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon

# Angular width of the peak mark, in degrees of the 12.7-degree arc -- about 9 px at this
# radius. A hairline, not a second bar.
PEAK_WIDTH_DEG = 0.45

PEAK_ALPHA = 0.80

# The bar is palest at its centre and fully saturated at the tip, in whatever tier colour is
# current. A fixed wash rather than one that ramps with the tier: the hue already carries which
# band we are in, and having the saturation move at the same time made the change read as two
# separate events instead of one.
CENTRE_WASH = 0.5

# Upstream switches its centre dot on and off with a bare comparison, so it blinks whenever the
# bar hovers at half deflection. Fade it across this much of the range instead.
DOT_FADE = 0.06

# Below this the bar is shorter than the centre dot is wide (4 counts is 1.4 px against the
# dot's 5 px radius), and arc_bar_pts floors the span at 0.001 deg, so what gets drawn is a
# smudge of tier colour under the dot rather than a bar. Measured on a rest frame: the dot
# came out (178,187,188) instead of neutral grey. Nothing visible is lost by not drawing it,
# and there is no pop because the shape it replaces is sub-pixel.
BAR_MIN = 0.01


def _mark_quad(mid_r: float, thickness: float, a_deg: float, w_deg: float) -> np.ndarray:
  """A radial mark as a 4-point ribbon: [outer0, outer1, inner1, inner0].

  draw_polygon interleaves two chains into a triangle strip, so four points in that order are
  a well-formed quad. Flat rather than curved on purpose -- see the note at the top.
  """
  a0, a1 = math.radians(a_deg - w_deg / 2), math.radians(a_deg + w_deg / 2)
  ro, ri = mid_r + thickness / 2, mid_r - thickness / 2
  c0, s0 = math.cos(a0), math.sin(a0)
  c1, s1 = math.cos(a1), math.sin(a1)
  return np.array([[c0 * ro, s0 * ro], [c1 * ro, s1 * ro],
                   [c1 * ri, s1 * ri], [c0 * ri, s0 * ri]], dtype=np.float32)


class SteerHeadroomBar(TorqueBar):
  def __init__(self, demo: bool = False, scale: float = 1.0, always: bool = False,
               clock: Callable[[], float] = time.monotonic):
    super().__init__(demo=demo, scale=scale, always=always)
    # The clock is injectable so .elantra/render_headroom_bar.py can drive the timing rather
    # than race it: offscreen frames run far faster than real time, and on the wall clock the
    # dwell would never accumulate and the ceiling would never light.
    self._state = HeadroomState(clock=clock)
    self._raised: bool | None = None

  @property
  def _armed(self) -> bool:
    """Resolved once the car is up. Cached only on a real answer, never on 'no CP yet'."""
    if self._raised is None:
      CP = ui_state.CP
      if CP is None:
        return False
      self._raised = is_raised(CP.brand, CP.flags)
    return self._raised

  def update_counts(self, counts: float, lat_active: bool = True) -> None:
    """Drive the indicator directly. Used by .elantra/render_headroom_bar.py."""
    self._state.update(counts, lat_active)

  def _update_state(self) -> None:
    # Keep upstream's filter fed so the unarmed path stays exactly upstream's.
    super()._update_state()
    if self._demo or not self._armed:
      return
    # The signed integer actually put on CAN, not the value normalised by STEER_MAX, so 384
    # and 409 stay true no matter what the normalisation divides by. Sign follows upstream.
    self._state.update(-ui_state.sm['carOutput'].actuatorsOutput.torqueOutputCan,
                       ui_state.sm['carControl'].latActive)

  def _render(self, rect: rl.Rectangle) -> None:
    if not self._demo and not self._armed:
      super()._render(rect)
      return

    if not self._demo:
      self._torque_line_alpha_filter.update(ui_state.status not in (UIStatus.DISENGAGED, UIStatus.LONG_ONLY))
    else:
      self._torque_line_alpha_filter.update(1.0)
    fade = self._torque_line_alpha_filter.x
    if fade < 1e-2:
      return

    # The tiers only mean something while lateral is actually commanding. Otherwise the bar
    # stays white, so the cyan base never suggests the car is steering when it is not.
    engaged = self._demo or ui_state.status in (UIStatus.ENGAGED, UIStatus.LAT_ONLY)

    st = self._state
    scale = self._scale
    x = float(np.clip(st.envelope / st.ceiling, -1.0, 1.0))
    mag = abs(x)

    line_offset = float(np.interp(mag, [0.5, 1], [22 * scale, 26 * scale]))
    height = float(np.interp(mag, [0.5, 1], [14 * scale, 56 * scale]))

    radius = 1200 * scale
    top_angle = -90.0
    span = fade * TORQUE_ANGLE_SPAN
    half = span / 2.0
    mid_r = radius + height / 2
    cap = 7 * scale

    cx = rect.x + rect.width / 2 + 8  # offset 8px to right of camera feed
    cy = rect.y + rect.height + radius - line_offset
    origin = np.array([cx, cy], dtype=np.float32)

    # No background track. The bar is the only persistent element: at rest there is just the
    # centre dot, and everything else is drawn only where the bar actually is.
    # The bar. At the ceiling it breathes in brightness rather than growing a second, thicker
    # halo behind itself: that halo needed a threshold to know where to start from, and a
    # threshold on a value that hovers is a flicker.
    if engaged:
      tip = st.color()
      breath = 1.0 - st.pulse_depth * (1.0 - math.cos(rl.get_time() * 2.0 * math.pi * PULSE_HZ)) / 2.0
      start_color = rl.Color(*lerp_rgb(tip, COLOR_WHITE, CENTRE_WASH),
                             int(255 * 0.9 * fade * breath))
      end_color = rl.Color(*tip, int(255 * fade * breath))
    else:
      start_color = end_color = rl.Color(255, 255, 255, int(255 * 0.35 * fade))

    lo, hi = (top_angle, top_angle + half * x) if x >= 0 else (top_angle + half * x, top_angle)
    bar_pts = arc_bar_pts(mid_r, height, lo, hi, cap_radius=cap) + origin if mag > BAR_MIN else None
    # Anchored where the full-scale arc would end, as upstream anchors it to the background
    # track, rather than to the bar's own tip: the endpoint then sits at a fixed place on
    # screen instead of collapsing onto the start as the bar shrinks toward zero. Computed
    # rather than measured off the track polygon, since there is no longer a track to measure.
    edge = cx + (1.0 if x >= 0 else -1.0) * (mid_r + height / 2) * math.cos(math.radians(top_angle + half))
    if bar_pts is not None:
      draw_polygon(rect, bar_pts, gradient=Gradient(
        start=(cx / rect.width, 0),
        end=((cx * (1 - 0.65) + edge * 0.65) / rect.width, 0),
        colors=[start_color, end_color],
        stops=[0.0, 1.0],
      ))

    # how close it got, still there when you look down after the corner
    peak_a = st.peak_alpha
    if engaged and peak_a > VISIBLE_EPS:
      at = top_angle + half * st.peak_sign * min(st.peak / st.ceiling, 1.0)
      draw_polygon(rect, _mark_quad(mid_r, height + 10 * scale, at, PEAK_WIDTH_DEG) + origin,
                   color=rl.Color(*st.peak_color(), int(255 * PEAK_ALPHA * peak_a * fade)))

    dot = float(np.clip((0.5 - mag) / DOT_FADE, 0.0, 1.0))
    if dot > 1e-3:
      dot_y = rect.y + rect.height - line_offset - height / 2
      rl.draw_circle(int(cx), int(dot_y), (10 // 2 * scale),
                     rl.Color(182, 182, 182, int(255 * 0.9 * fade * dot)))
