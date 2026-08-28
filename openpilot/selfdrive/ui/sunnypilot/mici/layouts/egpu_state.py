"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pure logic behind the eGPU panel.

Kept free of pyray, ui_state and Params on purpose: the decisions here -- which card is
believed to be in the dock, and why the model is or is not running on it -- are exactly what
needs testing, and they need testing on a machine with no dock and no GPU.
`.elantra/test_egpu.py` imports these directly, so the tests exercise the shipping code.

The panel exists because the onroad icon cannot explain itself. With an NVIDIA card and no
compiled model the correct behaviour is for the car to look like it has no eGPU at all --
which is indistinguishable, from the outside, from a dock that failed to come up. This is
where that distinction is written down.
"""

from __future__ import annotations

AMD = "amd"
NVIDIA = "nvidia"

VENDOR_LABELS = {AMD: "AMD", NVIDIA: "NVIDIA"}


def vendor_label(vendor: str, assumed: bool) -> str:
  """How the vendor reads on screen.

  "assumed" is not decoration. Nothing confirmed the card: no explicit setting, no cached
  probe. It happens to be right for every AMD user, which is exactly why it has to be said
  out loud rather than presented as fact.
  """
  name = VENDOR_LABELS.get(vendor, vendor or "unknown")
  return name + " (assumed)" if assumed else name


def idle_reason(bridge_present: bool, vendor: str, assumed: bool,
                use_nvidia: bool, nv_model: bool,
                asic_name: str | None = None, asic_supported: bool = True) -> str | None:
  """Why the driving model is not running on the eGPU, or None when it is.

  Ordered by what the user would have to fix first, so the panel never tells someone to
  compile a model when the dock is not even plugged in. The card check comes before the
  vendor check because an unsupported card is an AMD card: it would otherwise fall straight
  through to "everything is fine" and the panel would have nothing to say about a dock that
  is plainly not being used.
  """
  if not bridge_present:
    return "No chestnut detected. Check the USB cable and that the dock has power."
  if not asic_supported:
    # Deliberately says nothing about which architectures the driver covers. The previous
    # wording ("supports RDNA3 and RDNA4 only") was true when it was written and false the day
    # RDNA2 was ported, and the driver is not what gates these cards anyway -- it brings them
    # up. What they are missing is a numerical validation on a recorded route from this car.
    return ((asic_name or "This card") + " is not enabled for driving: its model output has "
            + "not been validated on this car yet. The dock stays powered and the driving "
            + "model runs on the device.")
  if vendor == AMD:
    return None
  if assumed:
    return ("The card in the dock has not been identified. Set the eGPU vendor below -- "
            + "until then the model runs on the device to avoid loading a model built "
            + "for the wrong card.")
  if not use_nvidia:
    return ("NVIDIA eGPU support is off. It is unmerged upstream and unvalidated on this "
            + "car, so it has to be turned on deliberately.")
  if not nv_model:
    return ("No model compiled for this NVIDIA card. Published eGPU models are built for "
            + "AMD and cannot run here; build one with sunnypilot/egpu/compile_nv.py.")
  return None


def status_rows(bridge_present: bool, vendor: str, assumed: bool, use_nvidia: bool,
                nv_model: bool, egpu_enabled: bool,
                asic_name: str | None = None) -> list[tuple[str, str]]:
  """Label/value pairs describing the dock, top to bottom.

  When the card has actually been identified its own name is shown instead of the vendor:
  the chip name already implies the vendor, and "Navi 23" is the thing worth reading when
  the row below says the model is running on the device.
  """
  if not bridge_present:
    return [("chestnut", "not detected"),
            ("gpu", "-"),
            ("driving model", "on device")]

  rows = [("chestnut", "detected"),
          ("gpu", asic_name or vendor_label(vendor, assumed))]
  if egpu_enabled:
    rows.append(("driving model", "on the eGPU"))
  else:
    rows.append(("driving model", "on device"))
  if vendor == NVIDIA:
    rows.append(("nvidia support", "on" if use_nvidia else "off"))
    rows.append(("compiled model", "present" if nv_model else "none"))
  return rows


def telemetry_note(vendor: str) -> str | None:
  """Explain the missing GPU readings on NVIDIA rather than showing empty gauges.

  The temperature, power and clock fields come from AMD's SMU. There is no equivalent for
  NVIDIA over this link, so they are left unwritten -- not zeroed to look complete.
  """
  if vendor != NVIDIA:
    return None
  return ("Temperature, power and clock readings come from AMD's onboard controller and "
          + "have no NVIDIA equivalent over USB. Link state and supply voltage still "
          + "report.")
