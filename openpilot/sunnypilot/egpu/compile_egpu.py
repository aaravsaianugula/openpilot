"""Compile a driving model for the specific card that is in the chestnut.

Every published eGPU model bundle is AMD-compiled for gfx12 -- sunnypilot's model CI builds
them with DEV=USB+AMD:LLVM -- so anything that is not a gfx12 AMD card has nothing to download
and the pickle has to be built on the device. That is true of an NVIDIA card, and equally true
of an RDNA2 AMD card, which is why this is parameterised by the EgpuSpec rather than being
NVIDIA-only. `compile_nv.py` is now a shim onto it.

What it does: sets the tinygrad environment from the vendor descriptor, hands everything else
to the stock compiler, and writes the `.egpu` provenance marker that lets modeld refuse the
result on a card it does not belong to. The marker records the LLVM target as well as the
vendor, because same-vendor / different-ISA is the mismatch a vendor check cannot see.

No upstream change was needed to compile: compile_modeld.py already gates its chestnut
link_up() wait on the DEV string containing 'USB', which both of ours satisfy.

Usage mirrors compile_modeld.py, plus two options of our own:

    python -m openpilot.sunnypilot.egpu.compile_egpu \
        [--egpu-vendor amd|nvidia] [--egpu-target gfx1032] \
        --model-type supercombo --model-size 1440x960 \
        --camera-resolutions 1928x1208 --supercombo-onnx <path> --output <path>

`--egpu-vendor` defaults to the card detect.py has actually confirmed; an assumed vendor is
refused rather than guessed at, because a wrong guess here costs a multi-hour compile and
produces a pickle marked for a card that is not present. On NVIDIA `--output` defaults to
where detect.py looks for it. There is no AMD default: a locally built AMD model has to be
placed into a model bundle by the model manager, and inventing a path here would be guessing
at somebody else's layout.

`--egpu-target` overrides the target written into the marker. It is needed for cards asics.py
has no entry for -- notably every NVIDIA card, where the target is an sm_XX compute capability
and we have no table. Left off, the target comes from the probed PCI device ID, and if that
yields nothing the marker is written without a target, which reads back as "unknown".

Requires the dock and the card. It cannot be run on a machine without them, and it does not
pretend otherwise -- tinygrad will fail to open the device.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from openpilot.common.file_chunker import get_manifest_path
from openpilot.sunnypilot.egpu import asics, detect
from openpilot.sunnypilot.egpu.models import UNKNOWN_TARGET, write_marker
from openpilot.sunnypilot.egpu.vendors import AUTO, NVIDIA, EgpuSpec, compile_flags, spec_for

COMPILER = "openpilot.sunnypilot.modeld_v2.compile_modeld"
VENDOR_FLAG = "--egpu-vendor"
TARGET_FLAG = "--egpu-target"


def take_option(args: list[str], flag: str) -> tuple[list[str], str]:
  """Pull one `--flag value` / `--flag=value` pair out of argv.

  Ours are stripped rather than passed through: compile_modeld.py parses with argparse and
  errors out on anything it does not declare.
  """
  rest: list[str] = []
  value = ""
  i = 0
  while i < len(args):
    arg = args[i]
    if arg == flag and i + 1 < len(args):
      value = args[i + 1]
      i += 2
      continue
    if arg.startswith(flag + "="):
      value = arg.split("=", 1)[1]
      i += 1
      continue
    rest.append(arg)
    i += 1
  return rest, value.strip().lower()


def option_value(args: list[str], flag: str) -> str | None:
  """What argv says a compile_modeld option is set to, without consuming it."""
  for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args):
      return args[i + 1]
    if arg.startswith(flag + "="):
      return arg.split("=", 1)[1]
  return None


def default_output(vendor: str) -> Path | None:
  return detect.NV_MODEL_PATH if vendor == NVIDIA else None


def compile_env(spec: EgpuSpec, warp_dev: str, base: dict[str, str] | None = None) -> dict[str, str]:
  """The environment the stock compiler has to run under to target this card.

  compile_flags() produces the same shape SConscript uses for AMD. Split into env because
  tinygrad reads these from the environment, not from argv.
  """
  env = dict(os.environ if base is None else base)
  for token in compile_flags(spec, warp_dev).split():
    key, _, value = token.partition("=")
    env[key] = value
  return env


def resolve_target(vendor: str, override: str = "", params=None) -> str:
  """The LLVM target of the card we are compiling against.

  Derived from the probed PCI device ID rather than from tinygrad, because opening the card
  here would take the exclusive flock the compile subprocess is about to need. The ID names
  the chip, and the chip names the ISA, so this is a fact about the hardware and not a guess.
  UNKNOWN_TARGET when asics.py has no entry -- see the marker rules in models.py.
  """
  if override:
    return override
  asic = asics.asic_for(vendor, detect.resolve_device(params))
  return asic.gfx if asic is not None else UNKNOWN_TARGET


def resolve_vendor(override: str, params=None) -> str:
  """Which card to build for: the flag, else the one that was actually confirmed.

  An assumed vendor is refused. detect.resolve() falls back to AMD when nothing confirmed
  anything, which is the right default for deciding whether to *run* today's model and the
  wrong one for deciding what to spend hours compiling.
  """
  if override and override != AUTO:
    return override
  vendor, assumed = detect.resolve(params)
  if assumed:
    raise SystemExit("no eGPU vendor confirmed: plug the dock in so the model manager can probe it,"
                     + " or pass " + VENDOR_FLAG + " amd|nvidia")
  return vendor


def main(argv: list[str], params=None) -> int:
  args, vendor_arg = take_option(list(argv), VENDOR_FLAG)
  args, target_arg = take_option(args, TARGET_FLAG)

  spec = spec_for(resolve_vendor(vendor_arg, params))
  target = resolve_target(spec.name, target_arg, params)
  warp_dev = os.environ.get("WARP_DEV", "QCOM")

  output = option_value(args, "--output")
  if output is None:
    fallback = default_output(spec.name)
    if fallback is None:
      print("--output is required when compiling for " + spec.name
            + ": a locally built model has to be placed into a bundle by the model manager,"
            + " and this tool will not guess where")
      return 2
    fallback.parent.mkdir(parents=True, exist_ok=True)
    output = str(fallback)
    args += ["--output", output]

  print("compiling for " + spec.dev + " (WARP_DEV=" + warp_dev + ", target="
        + (target or "unknown") + ")")
  proc = subprocess.run([sys.executable, "-m", COMPILER] + args, env=compile_env(spec, warp_dev))
  if proc.returncode != 0:
    print("compile failed; no marker written -- the model will not be offered to modeld")
    return proc.returncode

  # A successful compile does not leave a pkl behind. compile_modeld.py ends in chunk_file(),
  # which writes <output>.chunkNNofMM plus <output>.chunkmanifest and then os.remove()s the
  # output -- and usbgpu_compiled() and open_file_chunked() both look for the manifest, not the
  # pkl. So the manifest is the evidence of success. Demanding the pkl fails every genuine
  # compile, skips write_marker, and leaves assert_pkl_matches with no target to enforce, which
  # is the one check standing between a gfx1200 pickle and a gfx1032 card.
  if not Path(output).is_file() and not Path(get_manifest_path(output)).is_file():
    print("compiler reported success but produced neither " + output
          + " nor its chunk manifest; refusing to write a provenance marker for a model"
          + " that is not there")
    return 1

  write_marker(output, spec.name, target)
  if not target:
    print("no target known for this card, so the marker records the vendor only and modeld"
          + " cannot tell this pickle from one built for another " + spec.name + " ISA."
          + " Pass " + TARGET_FLAG + " to close that.")
  print("wrote " + output + " and its " + spec.name + " provenance marker")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
