"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The onroad Vision Long banner: one line, top right, for a laptop-monitor notice.

UI only. All per-frame logic is vision_long_notice.BannerPainter, which is guarded (no exception
reaches the HUD; each failure type is logged once through cloudlog), reads notice.json at most
once a second through NoticeReader, and measures a steady notice once. This file only supplies the
pyray measure and draw functions. It gives way to openpilot: while openpilot has an alert on
screen (the HUD's top icons are hidden) nothing is drawn.
"""

import pyray as rl

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_long_notice import BannerPainter, NoticeReader
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached

FONT_SIZE = 26
BACKGROUND = {
  "critical": rl.Color(201, 34, 49, 235),
  "warn": rl.Color(218, 111, 37, 235),
  "info": rl.Color(40, 40, 40, 220),
}


class VisionLongBanner:
  def __init__(self, measure=None):
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._painter = BannerPainter(NoticeReader(), measure or self._measure, self._draw, cloudlog.error)

  def _measure(self, text: str) -> tuple[float, float]:
    size = measure_text_cached(self._font, text, FONT_SIZE)
    return size.x, size.y

  def _draw(self, x: float, y: float, w: float, h: float, severity: str, text: str, text_x: float, text_y: float) -> None:
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), 0.4, 8, BACKGROUND[severity])
    rl.draw_text_ex(self._font, text, rl.Vector2(text_x, text_y), FONT_SIZE, 0, rl.WHITE)

  def render(self, rect: rl.Rectangle, can_draw: bool) -> None:
    self._painter.paint(rect.x, rect.y, rect.width, can_draw)
