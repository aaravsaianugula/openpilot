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

from openpilot.sunnypilot.egpu.vendors import AMD, AUTO, NVIDIA, VALID_VENDORS, spec_for

# Where compile_nv.py puts a locally compiled NVIDIA model. The published eGPU bundles are
# all AMD-compiled (sunnypilot's model CI builds them with DEV=USB+AMD:LLVM), so on NVIDIA
# there is no catalog to download from and the pkl has to be built on the device.
NV_MODEL_PATH = Path(__file__).resolve().parents[1].parent / "selfdrive/modeld/models/big_driving_tinygrad_nv.pkl"
MARKER_SUFFIX = ".egpu"


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


def resolve(params=None, allow_probe: bool = False) -> tuple[str, bool]:
  """(vendor, assumed). `assumed` means nothing actually confirmed it.

  Order: explicit param > cached probe result > live probe (opt-in) > AMD.

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

  if allow_probe:
    from openpilot.sunnypilot.egpu.probe import probe_vendor
    probed = probe_vendor()
    if probed in (AMD, NVIDIA):
      _params(params).put("EgpuVendorDetected", probed)
      return probed, False

  return AMD, True


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
  from openpilot.selfdrive.modeld.helpers import usbgpu_present
  if usbgpu_present():
    resolved, assumed = resolve(params)
    if resolved == NVIDIA and not assumed:
      env = dict(spec_for(NVIDIA).env)
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

  AMD  -- yes, as today; sunnypilot publishes AMD-compiled bundles for exactly this.
  NV   -- only with EgpuUseNvidia set, a confirmed (not assumed) vendor, and a locally
          compiled NV model actually present.

  When this is False the dock stays attached and powered and the car simply runs the ordinary
  on-SoC model. That is the important property: an eGPU we cannot drive must look like no
  eGPU, not like a broken one.
  """
  resolved, assumed = resolve(params)
  if resolved == AMD:
    return True
  if assumed:
    return False
  return bool(_params(params).get_bool("EgpuUseNvidia")) and nv_model_available()


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
