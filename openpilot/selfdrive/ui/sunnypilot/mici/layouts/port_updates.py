"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Elantra 2024-25 port update panel (comma 4 / mici).

This branch is rebuilt weekly from sunnypilot master, but only from a commit whose own
CI went green. That gate lives in the sync job, not here -- this panel exists so the gate
is visible from the car instead of being folklore, and so a bad week is one slider away
from being undone.

Rollback works by pointing the stock updater at the master-previous branch, which the sync
job keeps parked on the last build that passed. No new code runs in the update or finalize
path; that path is what bricks devices and it stays untouched.

The panel also carries the Vision Long braking switch. Its state is the file the companion
reads, /data/vision_long/braking, not a Params key, so it stays as set across drives.
"""

import subprocess
import time

import pyray as rl

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl, BigToggle, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog
from openpilot.selfdrive.ui.sunnypilot.mici.layouts.port_manifest import (
  age, braking_enabled, braking_summary, install_offered, load_manifest, set_braking, update_blocked,
)
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller

MAIN_BRANCH = "master"
ROLLBACK_BRANCH = "master-previous"

UPDATED_PROC = "openpilot.system.updated.updated"

# The state file can also be switched over SSH, so the toggle re-reads it while it is visible.
BRAKING_REFRESH_S = 1.0


def _signal_updater(signal: str) -> None:
  """Poke the updater daemon the same way the stock software panel does."""
  subprocess.run(["pkill", signal, "-f", UPDATED_PROC], check=False)


def _manifest(param: str) -> dict:
  """Read one of the build manifests updated.py publishes."""
  return load_manifest(ui_state.params.get(param))


class PortStatusInfo(Widget):
  """Four stacked rows describing exactly which build is running and where it came from."""

  ROW_H = 92

  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, 360, self.ROW_H * 4))

    header_color = rl.Color(255, 255, 255, int(255 * 0.9))
    sub_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)

    def row(title: str):
      head = UnifiedLabel(title, 44, max_width=max_width, text_color=header_color,
                          font_weight=FontWeight.DISPLAY, wrap_text=False)
      body = UnifiedLabel("", 30, max_width=max_width, text_color=sub_color,
                          font_weight=FontWeight.ROMAN, wrap_text=False, scroll=True)
      return head, body

    self._upstream_head, self._upstream_body = row(tr("sunnypilot build"))
    self._ci_head, self._ci_body = row(tr("upstream ci"))
    self._synced_head, self._synced_body = row(tr("last synced"))
    self._car_head, self._car_body = row(tr("elantra support"))

  def _update_state(self):
    manifest = _manifest("ElantraBuildManifest")

    if not manifest:
      unknown = tr("not an Elantra port build")
      for body in (self._upstream_body, self._ci_body, self._synced_body, self._car_body):
        body.set_text(unknown)
      return

    sha = (manifest.get("sunnypilot_upstream_sha") or "")[:9] or tr("unknown")
    date = manifest.get("sunnypilot_upstream_date") or ""
    self._upstream_body.set_text(f"{sha} {date}".strip())

    conclusion = manifest.get("upstream_ci_conclusion") or tr("unknown")
    checked = manifest.get("upstream_ci_checked") or 0
    self._ci_body.set_text(tr("{} ({} checks)").format(conclusion, checked)
                           if checked else str(conclusion))

    self._synced_body.set_text(age(manifest.get("synced_at_utc")))

    platforms = manifest.get("elantra_platforms") or []
    opendbc = (manifest.get("opendbc_sha") or "")[:9]
    self._car_body.set_text(tr("{} platforms, opendbc {}").format(len(platforms), opendbc)
                            if platforms else tr("MISSING"))

  def _render(self, _):
    rows = (
      (self._upstream_head, self._upstream_body),
      (self._ci_head, self._ci_body),
      (self._synced_head, self._synced_body),
      (self._car_head, self._car_body),
    )
    for i, (head, body) in enumerate(rows):
      top = self._rect.y + i * self.ROW_H
      head.set_position(self._rect.x + 20, top - 10)
      head.render()
      body.set_position(self._rect.x + 20, top + 44)
      body.render()


class VisionLongBrakingToggle(BigToggle):
  """Switches Vision Long braking on or off by writing the file the companion reads."""

  def __init__(self):
    super().__init__(tr("vision long braking"), "", toggle_callback=self._on_toggle)
    self._next_refresh = 0.0
    self._write_error = ""
    self._refresh()

  def _refresh(self):
    enabled = braking_enabled()
    self.set_checked(enabled)
    self.set_value(self._write_error or tr(braking_summary(enabled)))
    self._next_refresh = time.monotonic() + BRAKING_REFRESH_S

  def show_event(self):
    super().show_event()
    self._write_error = ""
    self._refresh()

  def _update_state(self):
    super()._update_state()
    if time.monotonic() >= self._next_refresh:
      self._refresh()

  def _on_toggle(self, checked: bool):
    try:
      set_braking(checked)
    except OSError as error:
      # Shown on the toggle until the next successful switch; the pill re-reads the file, so
      # it keeps showing the state the companion actually sees.
      cloudlog.exception("vision long braking toggle write failed")
      self._write_error = tr("not saved: {}").format(error.strerror or type(error).__name__)
    else:
      self._write_error = ""
    self._refresh()


class ElantraPortLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._status = PortStatusInfo()

    self._vision_long_toggle = VisionLongBrakingToggle()
    # Two short notes rather than one long one: a GreyBigButton is a fixed height and wraps.
    self._vision_long_what = GreyBigButton(
      "", tr("Braking only. It can only slow the car more than openpilot, never speed it up."))
    self._vision_long_how = GreyBigButton(
      "", tr("Stays on until you turn it off. Needs the laptop linked by USB cable."))

    # Default on. The sync job already refuses to publish a build from a red upstream commit,
    # so this is a second, visible line of defence rather than the only one.
    self._verified_toggle = BigParamControl(tr("verified builds only"), "ElantraVerifiedUpdatesOnly")

    self._check_btn = BigButton(tr("check for update"))
    self._check_btn.set_click_callback(self._on_check)

    self._install_btn = BigButton(tr("install update"))
    self._install_btn.set_click_callback(self._on_install)
    self._install_btn.set_visible(self._install_offered)

    self._blocked_note = GreyBigButton(
      tr("update held back"),
      tr("The pending build's upstream CI did not pass. Turn off \"verified builds only\" to "
         + "install it anyway."),
    )
    self._blocked_note.set_visible(self._update_blocked)

    self._rollback_btn = BigButton(tr("roll back to previous build"))
    self._rollback_btn.set_click_callback(self._on_rollback)
    self._rollback_btn.set_visible(lambda: self._target_branch() != ROLLBACK_BRANCH)

    self._latest_btn = BigButton(tr("return to latest"))
    self._latest_btn.set_click_callback(self._on_return_to_latest)
    self._latest_btn.set_visible(lambda: self._target_branch() == ROLLBACK_BRANCH)

    self._scroller.add_widgets([
      self._status,
      self._vision_long_toggle,
      self._vision_long_what,
      self._vision_long_how,
      self._verified_toggle,
      self._check_btn,
      self._install_btn,
      self._blocked_note,
      self._rollback_btn,
      self._latest_btn,
    ])

  # --- state helpers -------------------------------------------------------------------

  @staticmethod
  def _target_branch() -> str:
    return ui_state.params.get("UpdaterTargetBranch") or ""

  @staticmethod
  def _verified_only() -> bool:
    return ui_state.params.get_bool("ElantraVerifiedUpdatesOnly")

  @classmethod
  def _install_offered(cls) -> bool:
    return install_offered(ui_state.params.get_bool("UpdateAvailable"), cls._verified_only(),
                           ui_state.params.get("ElantraNewBuildManifest"))

  @classmethod
  def _update_blocked(cls) -> bool:
    return update_blocked(ui_state.params.get_bool("UpdateAvailable"), cls._verified_only(),
                          ui_state.params.get("ElantraNewBuildManifest"))

  # --- actions -------------------------------------------------------------------------

  def _on_check(self):
    _signal_updater("-SIGUSR1")

  def _on_install(self):
    ui_state.params.put_bool("DoReboot", True, block=True)

  def _switch_branch(self, branch: str):
    ui_state.params.put("UpdaterTargetBranch", branch, block=True)
    _signal_updater("-SIGUSR1")

  def _on_rollback(self):
    dialog = BigConfirmationDialog(
      tr("roll back to the previous build"),
      gui_app.texture("icons_mici/settings/software.png", 64, 75),
      lambda: self._switch_branch(ROLLBACK_BRANCH),
      red=True,
    )
    gui_app.push_widget(dialog)

  def _on_return_to_latest(self):
    self._switch_branch(MAIN_BRANCH)
