"""Model provenance: which vendor a compiled tinygrad pickle was built for.

The eGPU "big model" is not an ONNX file interpreted at runtime. It is a pickle of a tinygrad
program already compiled for one device, and sunnypilot's model CI builds every published
eGPU bundle with DEV=USB+AMD:LLVM. So a bundle that is perfectly good on the card comma ships
is wrong for an NVIDIA card, and nothing in the file name says so.

Loading the wrong one is very likely to raise -- tinygrad would try to open Device["AMD"]
with no AMD device present. "Very likely" is not a guarantee, and the only sanity check
downstream is `np.all(np.isfinite(...))`, which catches NaN and not wrong-but-finite numbers
going into the planner. Hence an explicit marker rather than trusting the failure mode.
"""

from __future__ import annotations

from openpilot.sunnypilot.egpu.vendors import AMD, NVIDIA

MARKER_SUFFIX = ".egpu"
_KNOWN = (AMD, NVIDIA)


def pkl_vendor(pkl_path: str) -> str:
  """Which vendor the pickle at this path was compiled for.

  A marker written by our own NV compile is authoritative. Absent means the pickle came from
  the sunnypilot catalog, and every published eGPU bundle is AMD-compiled -- so absent reads
  as AMD. That is fail-closed in the direction that matters: an unmarked file is never
  mistaken for an NVIDIA build.
  """
  try:
    with open(pkl_path + MARKER_SUFFIX, encoding="utf-8") as f:
      raw = f.read().strip().lower()
  except OSError:
    return AMD
  return raw if raw in _KNOWN else AMD


def write_marker(pkl_path: str, vendor: str) -> None:
  """Record what a freshly compiled pickle was built for. Written next to the artifact, the
  same way fetcher.py writes its .chunkmanifest sidecar."""
  if vendor not in _KNOWN:
    raise ValueError("refusing to mark a model with unknown vendor " + repr(vendor))
  with open(pkl_path + MARKER_SUFFIX, "w", encoding="utf-8", newline="\n") as f:
    f.write(vendor + "\n")


def assert_pkl_matches(pkl_path: str, usbgpu: bool, vendor: str) -> None:
  """Refuse to hand a model to a card it was not compiled for.

  Raises before the pickle is unpickled, so no wrong-vendor model output can reach the
  planner. This is the last of three gates and should be unreachable in normal operation --
  the model-manager catalog gate and `enabled()` both act while the car is parked. It exists
  for what those cannot see: the COMBINED_MODEL_PKL override, a hand-copied pickle, or a
  bundle that survived a vendor change.
  """
  if not usbgpu:
    return
  found = pkl_vendor(pkl_path)
  if found != vendor:
    raise RuntimeError(
      "eGPU model vendor mismatch: " + pkl_path + " was compiled for " + repr(found)
      + ", but this device's eGPU is " + repr(vendor) + ".\n"
      + "Select a non-eGPU model in Settings > Models, or build one for this card with\n"
      + "  python -m openpilot.sunnypilot.egpu.compile_nv")
