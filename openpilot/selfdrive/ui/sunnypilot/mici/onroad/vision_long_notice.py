"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What the onroad Vision Long banner shows, decided from /data/vision_long/notice.json.

The laptop monitor (.elantra/vision_long_supervisor/monitor.py) writes that file atomically:
{schema: "vl_notices_v1", written_mono_ns, notices: [{key, severity, text, state,
notified_mono_ns, recovered_mono_ns, ...}]}. Both clocks are CLOCK_MONOTONIC, which this UI
process shares with the monitor.

The banner shows the worst active notice for 10 s after it was notified, and "recovered" for 3 s
after a recovery was notified. A folded repeat carries its first notification time, so it is not
shown again. A missing, unreadable, garbage, wrong-schema or stale file shows nothing.

Pure on purpose: no pyray and no openpilot import, so .elantra/test_vision_long_notice_banner.py
can test it without a UI. Nothing here is on any control path; it only reads one file.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

NOTICE_FILE = Path("/data/vision_long/notice.json")
BANNER_MODULE = "openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_long_banner"
SCHEMA = "vl_notices_v1"
SHOW_NS = 10_000_000_000
RECOVERED_NS = 3_000_000_000
STALE_NS = 6_000_000_000       # the monitor rewrites the file at least every 2 s onroad
FUTURE_NS = 1_000_000_000
READ_EVERY_NS = 1_000_000_000
MAX_FILE_BYTES = 64 * 1024
TEXT_MAX = 80
SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}
MAX_WIDTH_FRACTION = 0.62
PAD_X = 14
PAD_Y = 8
MARGIN = 10
FIT_CACHE_MAX = 16


@dataclass(frozen=True)
class Banner:
  text: str
  severity: str
  recovered: bool


def _int(value) -> int | None:
  return value if type(value) is int else None


def choose(document, now_ns: int) -> Banner | None:
  """The banner for this moment, or None. `document` is the parsed file, or anything at all."""
  if type(document) is not dict or document.get("schema") != SCHEMA:
    return None
  written = _int(document.get("written_mono_ns"))
  if written is None or now_ns - written > STALE_NS or written - now_ns > FUTURE_NS:
    return None
  notices = document.get("notices")
  if type(notices) is not list:
    return None

  showing: list[tuple[int, int, Banner]] = []
  recovered: list[tuple[int, Banner]] = []
  for item in notices:
    if type(item) is not dict or item.get("severity") not in SEVERITY_RANK or type(item.get("text")) is not str:
      continue
    text = " ".join(item["text"].split())[:TEXT_MAX]
    if not text:
      continue
    notified, back = _int(item.get("notified_mono_ns")), _int(item.get("recovered_mono_ns"))
    if item.get("state") == "active" and notified is not None and 0 <= now_ns - notified <= SHOW_NS:
      showing.append((SEVERITY_RANK[item["severity"]], notified, Banner(text, item["severity"], False)))
    elif item.get("state") == "recovered" and back is not None and 0 <= now_ns - back <= RECOVERED_NS:
      recovered.append((back, Banner(("recovered: " + text)[:TEXT_MAX], "info", True)))
  if showing:
    return max(showing, key=lambda entry: (entry[0], entry[1]))[2]
  if recovered:
    return max(recovered, key=lambda entry: entry[0])[1]
  return None


def visible(banner: Banner | None, can_draw: bool) -> Banner | None:
  """Openpilot's own alert always wins: while it is on screen the banner is not drawn. A severity
  the banner has no colour for is never drawn either."""
  if banner is None or not can_draw or banner.severity not in SEVERITY_RANK:
    return None
  return banner


def read_document(path: Path):
  """The parsed file, or None when it is missing, too big, unreadable or not JSON."""
  try:
    with open(path, "rb") as stream:
      raw = stream.read(MAX_FILE_BYTES + 1)
  except OSError:
    return None
  if len(raw) > MAX_FILE_BYTES:
    return None
  try:
    return json.loads(raw)
  except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError; deep nesting recurses
    return None


def load_banner(report, module: str = BANNER_MODULE):
  """The banner widget, or None when it cannot be imported or built.

  The HUD reaches the widget only through here: the widget imports pyray and UI helpers, and an
  import or constructor failure (an upstream rename, a missing font) must hide the banner, never
  take the onroad HUD down. The failure is reported once, here, since the HUD never retries."""
  try:
    return importlib.import_module(module).VisionLongBanner()
  except Exception as error:  # the banner is optional; the HUD must come up without it
    try:
      report(f"vision long banner unavailable: {type(error).__name__}: {error}")
    except Exception:  # a failing logger must not take the HUD down either; there is nowhere left to report it
      pass
    return None


class NoticeReader:
  """Reads the file at most once a second; decides the banner on every frame from that read."""

  def __init__(self, path: Path = NOTICE_FILE, clock=time.monotonic_ns):
    self._path = path
    self._clock = clock
    self._document = None
    self._read_ns: int | None = None

  def current(self) -> Banner | None:
    now = self._clock()
    if self._read_ns is None or now - self._read_ns >= READ_EVERY_NS:
      self._document = read_document(self._path)
      self._read_ns = now
    return choose(self._document, now)


class BannerPainter:
  """Everything the banner does per frame, guarded. The widget hands in its measure and draw
  functions, so this runs, and is tested, without pyray.

  `paint` never raises: any exception (the reader, a measure, a draw, a missing colour) hides the
  banner for that frame and is reported once per exception type. The fitted text is cached per
  (text, width), so a steady notice is measured once, not once per frame."""

  def __init__(self, reader, measure, draw, report):
    self._reader = reader
    self._measure = measure  # text -> (width, height)
    self._draw = draw        # (x, y, w, h, severity, text, text_x, text_y) -> None
    self._report = report
    self._reported: set[str] = set()
    self._fits: dict[tuple[str, int], tuple[str, float, float]] = {}

  def _fit(self, text: str, max_width: float) -> tuple[str, float, float]:
    key = (text, int(max_width))
    fitted = self._fits.get(key)
    if fitted is None:
      width, height = self._measure(text)
      while width + 2 * PAD_X > max_width and len(text) > 4:
        text = text[:-4] + "..."
        width, height = self._measure(text)
      if len(self._fits) >= FIT_CACHE_MAX:
        self._fits.clear()
      fitted = self._fits[key] = (text, width, height)
    return fitted

  def paint(self, x: float, y: float, width: float, can_draw: bool) -> bool:
    """Draw the banner if there is one to show. Returns whether it was drawn."""
    try:
      banner = visible(self._reader.current(), can_draw)
      if banner is None:
        return False
      text, text_width, text_height = self._fit(banner.text, width * MAX_WIDTH_FRACTION)
      box_w, box_h = text_width + 2 * PAD_X, text_height + 2 * PAD_Y
      box_x, box_y = x + width - box_w - MARGIN, y + MARGIN
      self._draw(box_x, box_y, box_w, box_h, banner.severity, text, box_x + PAD_X, box_y + PAD_Y)
      return True
    except Exception as error:  # nothing may reach the onroad render loop; reported once per type
      name = type(error).__name__
      if name not in self._reported:
        self._reported.add(name)
        self._report_safely(f"vision long banner hidden: {name}: {error}")
      return False

  def _report_safely(self, message: str) -> None:
    try:
      self._report(message)
    except Exception:  # a failing logger must not take the HUD down either; there is nowhere left to report it
      return
