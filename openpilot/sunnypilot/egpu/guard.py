"""Keeping an eGPU we cannot drive from taking the car with it.

`detect.enabled()` answers "should the driving model run on this card". This module answers
the two questions that decide whether getting that answer wrong is survivable:

    egpu_build_ok()  -- may SCons even attempt the eGPU compile?
    loading()        -- hold UsbGpuLoading across a model load, on every exit path

Both exist because the failure they prevent is not "no big model", it is "no car".

A failed eGPU compile used to fail its SCons target, and `build.py` turns a failed build into
a blocking TextWindow plus exit(1) -- the device sits on an error screen that needs the
touchscreen to clear. A model load that raises used to leave `UsbGpuLoading` latched True,
and selfdrived reads that as a NO_ENTRY every frame while *also* suppressing the commIssue
and posenetInvalid events that would have explained it.

Nothing here decides whether the eGPU is good. `detect` already does that. This decides what
happens when it is not.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from openpilot.sunnypilot.egpu import detect


def egpu_build_ok(params=None) -> bool:
  """May the big model be compiled for the attached card?

  The same verdict as `detect.enabled()`, but total. SCons runs this before openpilot has
  been built, so Params may not be loadable at all -- on a clean tree `libparams_c.so` does
  not exist yet, and importing it raises OSError. An unreadable answer means "we do not
  know", and not knowing must never switch off an eGPU that works, so it reads as yes.

  That fail-open is only safe because it is not the last line of defence: the SConscript
  treats a failed eGPU compile as a skip, so a card that slips through this gate costs one
  wasted compile, not a boot.
  """
  try:
    return detect.enabled(params)
  except Exception as e:
    # Loud on purpose. This is a totality boundary, not an error being swallowed -- if it
    # fires anywhere except a pre-build tree, that is worth seeing in the build log.
    print(f"egpu: cannot tell whether this card is supported ({e}); attempting the build anyway")
    return True


@contextlib.contextmanager
def loading(params, usbgpu: bool) -> Iterator[None]:
  """Hold `UsbGpuLoading` across the model load and clear it on every exit path.

  The `finally` is the entire point of this function, so it is worth writing down why.
  `UsbGpuLoading` is CLEAR_ON_MANAGER_START, and a modeld crash restarts *modeld*, not
  manager. A load that raises therefore leaves the flag True for the rest of the ignition
  cycle. selfdrived reads it as `big_model_loading`, which adds NO_ENTRY every frame, and
  as `big_model_settling`, which suppresses `commIssue`, `posenetInvalid` and
  `locationdTemporaryError`. The result is a car that cannot engage and cannot say why --
  from a component whose whole contract is that losing it degrades to the on-SoC model.

  Upstream's `selfdrive/modeld/modeld.py` never had this bug because it falls back instead
  of raising, so its clear is always reached. This makes the property explicit rather than
  a consequence of statement order.
  """
  params.put_bool("UsbGpuLoading", usbgpu)
  params.remove("UsbGpuActive")
  try:
    yield
  finally:
    params.put_bool("UsbGpuLoading", False)
