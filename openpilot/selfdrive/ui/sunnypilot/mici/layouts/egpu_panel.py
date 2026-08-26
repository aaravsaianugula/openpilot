"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

eGPU panel (comma 4 / mici).

comma ships the chestnut with an AMD card and openpilot assumes that everywhere. With an
NVIDIA card the correct behaviour is for the car to behave as though there were no eGPU at
all -- the model runs on the device, no alert fires, no icon goes red. That is right, and
from the outside it is indistinguishable from a dock that silently failed.

This panel is where the difference is written down: what is plugged in, which card we believe
is behind the bridge, and why the model is or is not running on it.

The decisions live in egpu_state.py, which has no pyray in it and is tested by
.elantra/test_egpu.py. This file only renders them.
"""

import pyray as rl

from openpilot.selfdrive.ui.mici.widgets.button import BigMultiToggle, BigParamControl, GreyBigButton
from openpilot.selfdrive.ui.sunnypilot.mici.layouts.egpu_state import (
  NVIDIA, idle_reason, status_rows, telemetry_note,
)
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller

# auto first, so the cycle starts by handing the decision back to the device.
VENDOR_OPTIONS = ["auto", "amd", "nvidia"]
MAX_ROWS = 5


def _egpu():
  """Imported lazily: the settings screen constructs this panel on every boot, and detect
  reaches into modeld helpers that touch /sys."""
  from openpilot.sunnypilot.egpu import detect
  return detect


def _facts() -> tuple[bool, str, bool, bool, bool, bool]:
  """Everything the panel renders, gathered in one place."""
  detect = _egpu()
  from openpilot.selfdrive.modeld.helpers import usbgpu_present
  vendor, assumed = detect.resolve()
  return (usbgpu_present(), vendor, assumed,
          ui_state.params.get_bool("EgpuUseNvidia"),
          detect.nv_model_available(),
          detect.enabled())


class EgpuStatusInfo(Widget):
  """Stacked label/value rows describing the dock."""

  ROW_H = 92

  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 360, self.ROW_H * MAX_ROWS))

    header_color = rl.Color(255, 255, 255, int(255 * 0.9))
    sub_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)

    self._rows = []
    for _ in range(MAX_ROWS):
      head = UnifiedLabel("", 44, max_width=max_width, text_color=header_color,
                          font_weight=FontWeight.DISPLAY, wrap_text=False)
      body = UnifiedLabel("", 30, max_width=max_width, text_color=sub_color,
                          font_weight=FontWeight.ROMAN, wrap_text=False, scroll=True)
      self._rows.append((head, body))
    self._visible_rows = 0

  def _update_state(self):
    bridge, vendor, assumed, use_nvidia, nv_model, enabled = _facts()
    rows = status_rows(bridge, vendor, assumed, use_nvidia, nv_model, enabled)[:MAX_ROWS]
    self._visible_rows = len(rows)
    for (head, body), (label, value) in zip(self._rows, rows, strict=False):
      head.set_text(tr(label))
      body.set_text(tr(value))

  def _render(self, _):
    for i in range(self._visible_rows):
      head, body = self._rows[i]
      top = self._rect.y + i * self.ROW_H
      head.set_position(self._rect.x + 20, top - 10)
      head.render()
      body.set_position(self._rect.x + 20, top + 44)
      body.render()


class EgpuLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._status = EgpuStatusInfo()

    self._reason = GreyBigButton(tr("model is not using the eGPU"), "")
    self._reason.set_visible(self._has_reason)

    self._telemetry = GreyBigButton(tr("gpu readings unavailable"), "")
    self._telemetry.set_visible(self._has_telemetry_note)

    # Offroad only. EgpuVendor is read when modeld starts, so changing it while driving would
    # either do nothing or restart the model process mid-drive.
    #
    # BigMultiToggle rather than BigMultiParamToggle: the latter stores an integer index, and
    # EgpuVendor is a string so that a params dump says "nvidia" rather than "2".
    self._vendor_toggle = BigMultiToggle(tr("eGPU vendor"), VENDOR_OPTIONS,
                                         select_callback=self._on_vendor)
    self._vendor_toggle.set_value(_egpu().configured())
    self._vendor_toggle.set_visible(lambda: not ui_state.started)

    # Off by default and deliberately awkward to reach: NVIDIA support is unmerged upstream
    # and has not been validated against a reference model on this car.
    self._nvidia_toggle = BigParamControl(tr("use NVIDIA eGPU"), "EgpuUseNvidia")
    self._nvidia_toggle.set_visible(self._nvidia_selected)

    self._scroller.add_widgets([
      self._status,
      self._reason,
      self._telemetry,
      self._vendor_toggle,
      self._nvidia_toggle,
    ])

  # --- state helpers ---------------------------------------------------------------------

  @staticmethod
  def _nvidia_selected() -> bool:
    return _egpu().resolve()[0] == NVIDIA and not ui_state.started

  def _has_reason(self) -> bool:
    bridge, vendor, assumed, use_nvidia, nv_model, _ = _facts()
    reason = idle_reason(bridge, vendor, assumed, use_nvidia, nv_model)
    if reason:
      self._reason.set_value(reason)
    return bool(reason)

  def _has_telemetry_note(self) -> bool:
    note = telemetry_note(_egpu().resolve()[0])
    if note:
      self._telemetry.set_value(note)
    return bool(note)

  # --- actions ---------------------------------------------------------------------------

  @staticmethod
  def _on_vendor(value: str) -> None:
    ui_state.params.put("EgpuVendor", value, block=True)
    # A cached probe result would otherwise override the choice just made.
    ui_state.params.remove("EgpuVendorDetected")
