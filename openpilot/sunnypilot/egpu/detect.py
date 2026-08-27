"""Which card is in the chestnut, and whether to route the driving model through it.

`usbgpu_present()` upstream answers "is a flashed chestnut attached". That is a question about
the *bridge*, and it is vendor-neutral -- it is true with an NVIDIA card behind it just as it
is with an AMD one. Upstream then treats that as "run the big model on it", which is only
correct because comma ships one vendor.

Those are two different questions and this module keeps them apart:

    usbgpu_present()  -- a flashed chestnut is attached          (upstream, unchanged)
    resolve()         -- which vendor's card is behind it
    enabled()         -- route the driving model through it

Everything openpilot-specific is imported lazily so this module stays importable on a machine
with no openpilot build, which is what lets .elantra/test_egpu.py exercise the shipping code
rather than a copy of it.
"""

from __future__ import annotations

import os
from pathlib import Path

from openpilot.sunnypilot.egpu import asics
from openpilot.sunnypilot.egpu.vendors import AMD, AUTO, NVIDIA, VALID_VENDORS, spec_for

# Where compile_nv.py puts a locally compiled NVIDIA model. The published eGPU bundles are
# all AMD-compiled (sunnypilot's model CI builds them with DEV=USB+AMD:LLVM), so on NVIDIA
# there is no catalog to download from and the pkl has to be built on the device.
NV_MODEL_PATH = Path(__file__).resolve().parents[1].parent / "selfdrive/modeld/models/big_driving_tinygrad_nv.pkl"


def _params(params=None):
  if params is not None:
    return params
  from openpilot.common.params import Params
  return Params()


def _get(params, key: str) -> str:
  raw = _params(params).get(key)
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8", errors="replace")
  return (raw or "").strip().lower()


def configured(params=None) -> str:
  """The vendor the user selected. Anything unrecognised reads as `auto`, never raises.

  A garbage param must not be able to stop the car from starting, so this is deliberately
  total: unknown input degrades to autodetection, which degrades to AMD.
  """
  value = _get(params, "EgpuVendor")
  return value if value in VALID_VENDORS else AUTO


def _parse_device_id(raw: str) -> int | None:
  """A 16-bit PCI device ID from a param, or None. Hex, 0x prefix optional.

  Total by construction, like configured(): a hand-edited param must never be able to raise
  on the path that decides whether the car gets a model.
  """
  text = raw.removeprefix("0x")
  # Hex digits only, and at most four of them. int(x, 16) accepts a sign and surrounding
  # space, so "+3ff" would otherwise parse as 1023 -- a real device ID, just not the one
  # that was written down. Four digits is also what bounds the result to 16 bits.
  if not text or len(text) > 4 or not all(c in "0123456789abcdefABCDEF" for c in text):
    return None
  return int(text, 16)


# Whether this process has already asked the hardware. The model manager is one process
# per manager start and the probe is a USB round trip, so at most one attempt per start --
# but only once a dock is actually there, so plugging one in later still gets probed.
_probe_attempted = False


def resolve_device(params=None) -> int | None:
  """The eGPU's PCI device ID: explicit override > cached probe > nothing known.

  None means "we do not know", and every caller treats that as "behave exactly as today".
  That is the safe direction: not knowing which card is attached must never switch off an
  eGPU that works.
  """
  for key in ("EgpuDevice", "EgpuDeviceDetected"):
    device_id = _parse_device_id(_get(params, key))
    if device_id is not None:
      return device_id
  return None


def resolve(params=None) -> tuple[str, bool]:
  """(vendor, assumed). `assumed` means nothing actually confirmed it.

  Order: explicit param > cached probe result > AMD. Reading only, never probing:
  `probe_once()` is the one place that touches hardware, and keeping that out of here is
  what lets this be called freely from a 1 Hz loop and from the UI.

  Falling back to AMD is deliberate rather than lazy: it is byte-for-byte today's behaviour,
  so a user with an AMD card can never regress because this code exists. `assumed` is what
  keeps that from being a silent guess -- the UI renders it, and an NVIDIA model is not
  allowed to arm on an assumed vendor.
  """
  explicit = configured(params)
  if explicit != AUTO:
    return explicit, False

  cached = _get(params, "EgpuVendorDetected")
  if cached in (AMD, NVIDIA):
    return cached, False

  return AMD, True


def probe_once(params=None) -> None:
  """Ask the dock what is in it, and cache the answer for the rest of this manager start.

  Called from the model manager, which `process_config.py` registers `only_offroad`. That is
  what makes it safe: probing borrows tinygrad's USB controller and USBPCIDevice takes an
  exclusive flock, so doing this underneath a running modeld would either fail outright or
  disturb a device that is driving.

  Triggered on the *card* being unknown rather than the vendor. Driving this off vendor
  resolution was wrong: resolve() short-circuits on an explicit EgpuVendor, so a user who
  picked "amd" in the panel -- an ordinary thing to do -- never got a device ID and the
  RDNA2 gate silently did not apply to them.

  Attempted at most once per manager start, and only once a dock is actually present, so a
  dock plugged in after boot is still identified. Both cached params are
  CLEAR_ON_MANAGER_START, so a card swapped while the device was off is picked up next boot
  rather than remembered forever.
  """
  global _probe_attempted
  if _probe_attempted:
    return

  from openpilot.selfdrive.modeld.helpers import usbgpu_present
  if not usbgpu_present():
    return

  _probe_attempted = True
  if resolve_device(params) is not None:
    return

  from openpilot.sunnypilot.egpu.probe import probe_ids
  if (probed := probe_ids()) is not None:
    store = _params(params)
    store.put("EgpuVendorDetected", probed[0])
    store.put("EgpuDeviceDetected", f"0x{probed[1]:04x}")


def vendor(params=None) -> str:
  return resolve(params)[0]


def queue_dev(params=None) -> str:
  """The tinygrad Device[...] key for the attached card."""
  return spec_for(resolve(params)[0]).tg_key


def apply_env(params=None) -> None:
  """Set tinygrad env that must exist before `import tinygrad`.

  Upstream sets GMMU=0 unconditionally at modeld import, describing it as a no-op for qcom.
  That is kept for every path except a confirmed NVIDIA eGPU, where GMMU is an AMD-MMU
  control with no defined meaning over USB+NV.
  """
  env = {"GMMU": "0"}
  try:
    from openpilot.selfdrive.modeld.helpers import usbgpu_present
    if usbgpu_present():
      resolved, assumed = resolve(params)
      if resolved == NVIDIA and not assumed:
        env = dict(spec_for(NVIDIA).env)
  except Exception:
    # This runs at modeld import, before anything else is set up. The line it replaces
    # (`os.environ['GMMU'] = '0'`) could not fail; reading params or globbing /sys can.
    # Falling back to exactly upstream's behaviour keeps a params problem from becoming
    # "modeld cannot even be imported", which would take the car off the road.
    env = {"GMMU": "0"}
  os.environ.update(env)


def nv_model_available() -> bool:
  """Is there a locally compiled NVIDIA model, and does its marker agree?

  The marker is what distinguishes "compiled here for NV" from "a pkl someone copied in".
  Without it we would be trusting a filename.
  """
  from openpilot.sunnypilot.egpu.models import pkl_vendor
  return NV_MODEL_PATH.is_file() and pkl_vendor(str(NV_MODEL_PATH)) == NVIDIA


def enabled(params=None) -> bool:
  """Route the driving model through the eGPU at all?

  AMD  -- yes, as today; sunnypilot publishes AMD-compiled bundles for exactly this, unless
          the card is one tinygrad's AM driver refuses (see asics.py).
  NV   -- only with EgpuUseNvidia set, a confirmed (not assumed) vendor, and a locally
          compiled NV model actually present.

  When this is False the dock stays attached and powered and the car simply runs the ordinary
  on-SoC model. That is the important property: an eGPU we cannot drive must look like no
  eGPU, not like a broken one.
  """
  resolved, assumed = resolve(params)
  if not asics.am_supports(resolved, resolve_device(params)):
    return False
  if resolved == AMD:
    return True
  if assumed:
    return False
  return bool(_params(params).get_bool("EgpuUseNvidia")) and nv_model_available()


def asic(params=None) -> asics.AsicSpec | None:
  """What we know about the attached card, or None when we know nothing specific about it."""
  return asics.asic_for(resolve(params)[0], resolve_device(params))


def uses_amd_catalog(params=None) -> bool:
  """Should the model manager serve the _USBGPU catalog?

  Only when an AMD card is actually going to run the model. Every published eGPU bundle is
  AMD-compiled, so offering them on an NVIDIA device would hand modeld a pickle it cannot
  load -- which upstream turns into a 60-second timeout and a restart loop.
  """
  return enabled(params) and resolve(params)[0] == AMD


def assert_pkl_matches(pkl_path: str, usbgpu: bool, params=None) -> None:
  """Refuse a model compiled for a different card. Resolves the vendor itself so the call
  site in modeld stays a single line -- every line we add to an upstream file is a conflict
  the weekly sync has to replay."""
  from openpilot.sunnypilot.egpu.models import assert_pkl_matches as _assert
  _assert(pkl_path, usbgpu, resolve(params)[0])
