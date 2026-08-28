"""Model provenance: which card a compiled tinygrad pickle was built for.

The eGPU "big model" is not an ONNX file interpreted at runtime. It is a pickle of a tinygrad
program already compiled for one device, and sunnypilot's model CI builds every published
eGPU bundle with DEV=USB+AMD:LLVM. So a bundle that is perfectly good on the card comma ships
is wrong for an NVIDIA card, and nothing in the file name says so.

Loading the wrong one is very likely to raise -- tinygrad would try to open Device["AMD"]
with no AMD device present. "Very likely" is not a guarantee, and the only sanity check
downstream is `np.all(np.isfinite(...))`, which catches NaN and not wrong-but-finite numbers
going into the planner. Hence an explicit marker rather than trusting the failure mode.

Vendor alone is not enough. tinygrad emits one ISA, and the published bundles are gfx12; an
RDNA2 card is gfx1032 and carries the same PCI vendor ID, 0x1002. That pickle sails through a
vendor-only check and reaches `load_oob`. So the marker also records the *target* -- the LLVM
arch the pickle was actually built for -- and `assert_pkl_matches` compares it.

Marker format, `<pkl>.egpu`, one `key=value` per line:

    vendor=amd
    target=gfx1032

A marker written before the target existed is a single bare vendor token ("amd\n"). That form
still parses, as vendor with UNKNOWN_TARGET, because those files are sitting next to models
already compiled on devices in the field and re-flashing them is not something this code gets
to require.
"""

from __future__ import annotations

from dataclasses import dataclass

from openpilot.sunnypilot.egpu.vendors import AMD, NVIDIA

MARKER_SUFFIX = ".egpu"

# "we do not know what this was built for", which is a real and common answer on both sides of
# the comparison -- see assert_pkl_matches.
UNKNOWN_TARGET = ""

_KNOWN = (AMD, NVIDIA)
_TARGET_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_TARGET_MAX = 24


@dataclass(frozen=True)
class Marker:
  vendor: str
  target: str


def _clean_target(raw: str | None) -> str:
  """A target token, or UNKNOWN_TARGET if it is not one.

  Deliberately narrow: a target ends up in an error message and is compared against a value
  derived from hardware, so anything with whitespace, punctuation or a path separator in it is
  not a target we should be pretending to understand.
  """
  text = (raw or "").strip().lower()
  if not text or len(text) > _TARGET_MAX or not all(c in _TARGET_CHARS for c in text):
    return UNKNOWN_TARGET
  return text


def _parse(text: str) -> Marker:
  fields: dict[str, str] = {}
  bare = ""
  for line in text.splitlines():
    line = line.strip().lower()
    if not line:
      continue
    if "=" in line:
      key, _, value = line.partition("=")
      fields.setdefault(key.strip(), value.strip())
    elif not bare:
      bare = line  # the pre-target format: the whole marker was one vendor token

  vendor = fields.get("vendor", bare)
  if vendor not in _KNOWN:
    vendor = AMD
  return Marker(vendor, _clean_target(fields.get("target")))


def read_marker(pkl_path: str) -> Marker:
  """What the marker beside this pickle claims.

  A marker written by our own compile is authoritative. Absent means the pickle came from the
  sunnypilot catalog, and every published eGPU bundle is AMD-compiled -- so absent reads as
  AMD, with no target. That is fail-closed in the direction that matters: an unmarked file is
  never mistaken for an NVIDIA build.
  """
  try:
    with open(pkl_path + MARKER_SUFFIX, encoding="utf-8") as f:
      text = f.read()
  except OSError:
    return Marker(AMD, UNKNOWN_TARGET)
  return _parse(text)


def pkl_vendor(pkl_path: str) -> str:
  """Which vendor the pickle at this path was compiled for."""
  return read_marker(pkl_path).vendor


def pkl_target(pkl_path: str) -> str:
  """Which LLVM target the pickle was compiled for, or UNKNOWN_TARGET if it does not say."""
  return read_marker(pkl_path).target


def write_marker(pkl_path: str, vendor: str, target: str = UNKNOWN_TARGET) -> None:
  """Record what a freshly compiled pickle was built for. Written next to the artifact, the
  same way fetcher.py writes its .chunkmanifest sidecar.

  An empty target is allowed and means "the compiler could not establish one" -- writing a
  guess would be worse than writing nothing, because a wrong target either refuses a good
  model or waves a bad one through.
  """
  if vendor not in _KNOWN:
    raise ValueError("refusing to mark a model with unknown vendor " + repr(vendor))
  cleaned = _clean_target(target)
  if target and not cleaned:
    raise ValueError("refusing to mark a model with an unreadable target " + repr(target))

  lines = ["vendor=" + vendor]
  if cleaned:
    lines.append("target=" + cleaned)
  with open(pkl_path + MARKER_SUFFIX, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")


def assert_pkl_matches(pkl_path: str, usbgpu: bool, vendor: str, target: str | None = None) -> None:
  """Refuse to hand a model to a card it was not compiled for.

  Raises before the pickle is unpickled, so no wrong-vendor model output can reach the
  planner. This is the last of three gates and should be unreachable in normal operation --
  the model-manager catalog gate and `enabled()` both act while the car is parked. It exists
  for what those cannot see: the COMBINED_MODEL_PKL override, a hand-copied pickle, or a
  bundle that survived a vendor change.

  `target` is the attached card's LLVM target, or None when the caller does not know it.

  An unknown target on *either* side passes, and that is a decision rather than an oversight.
  Knowledge here is asymmetric and mostly absent: published bundles carry no marker at all,
  markers written before this change carry no target, and asics.py only knows the gfx string
  of the cards it has an entry for. Failing closed on unknown would therefore refuse the
  ordinary AMD catalog model on the ordinary AMD card -- taking working eGPUs off the road for
  no reason except that this code exists, which is the one thing detect.py and asics.py are
  both written to never do. So this gate strengthens from "never checked" to "checked whenever
  both ends actually say", and the compiler always writes a target so the locally built path
  -- the one that can realistically produce a gfx mismatch -- is fully covered.
  """
  if not usbgpu:
    return

  found = read_marker(pkl_path)
  if found.vendor != vendor:
    raise RuntimeError(
      "eGPU model vendor mismatch: " + pkl_path + " was compiled for " + repr(found.vendor)
      + ", but this device's eGPU is " + repr(vendor) + ".\n"
      + "Select a non-eGPU model in Settings > Models, or build one for this card with\n"
      + "  python -m openpilot.sunnypilot.egpu.compile_egpu")

  expected = _clean_target(target)
  if expected and found.target and found.target != expected:
    raise RuntimeError(
      "eGPU model target mismatch: " + pkl_path + " was compiled for " + repr(found.target)
      + ", but this device's eGPU is " + repr(expected) + ".\n"
      + "Same vendor, different ISA -- tinygrad would load it and produce wrong numbers.\n"
      + "Select a non-eGPU model in Settings > Models, or build one for this card with\n"
      + "  python -m openpilot.sunnypilot.egpu.compile_egpu")
