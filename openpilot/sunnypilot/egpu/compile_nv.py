"""Compile a driving model for an NVIDIA card in the chestnut.

Every published eGPU model bundle is AMD-compiled -- sunnypilot's model CI builds them with
DEV=USB+AMD:LLVM -- so on NVIDIA there is nothing to download and the pickle has to be built
on the device. This is a thin wrapper: it sets the tinygrad environment from the NV
descriptor, hands everything else to the stock compiler, and writes the provenance marker
that lets modeld refuse a mismatched model later.

No upstream change was needed to compile: compile_modeld.py already gates its chestnut
link_up() wait on `'USB' in os.getenv('DEV', '')`, which DEV=USB+NV satisfies.

Usage mirrors compile_modeld.py, with the output path defaulting to where detect.py looks:

    python -m openpilot.sunnypilot.egpu.compile_nv \\
        --model-type supercombo --model-size 1440x960 \\
        --camera-resolutions 1928x1208 --supercombo-onnx <path>

Requires the dock, the card, and a working DEV=USB+NV. It cannot be run on a machine without
them, and it does not pretend otherwise -- tinygrad will fail to open the device.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from openpilot.sunnypilot.egpu.detect import NV_MODEL_PATH
from openpilot.sunnypilot.egpu.models import write_marker
from openpilot.sunnypilot.egpu.vendors import NVIDIA, compile_flags, spec_for

COMPILER = "openpilot.sunnypilot.modeld_v2.compile_modeld"


def main(argv: list[str]) -> int:
  spec = spec_for(NVIDIA)
  warp_dev = os.environ.get("WARP_DEV", "QCOM")

  # compile_flags() produces the same shape SConscript uses for AMD. Split into env because
  # tinygrad reads these from the environment, not from argv.
  env = dict(os.environ)
  for token in compile_flags(spec, warp_dev).split():
    key, _, value = token.partition("=")
    env[key] = value

  args = list(argv)
  if not any(a == "--output" or a.startswith("--output=") for a in args):
    NV_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    args += ["--output", str(NV_MODEL_PATH)]

  output = None
  for i, a in enumerate(args):
    if a == "--output":
      output = args[i + 1]
    elif a.startswith("--output="):
      output = a.split("=", 1)[1]

  print("compiling for " + spec.dev + " (WARP_DEV=" + warp_dev + ")")
  proc = subprocess.run([sys.executable, "-m", COMPILER] + args, env=env)
  if proc.returncode != 0:
    print("compile failed; no marker written -- the model will not be offered to modeld")
    return proc.returncode

  if output is None or not Path(output).is_file():
    print("compiler reported success but produced no file at " + str(output)
          + "; refusing to write a provenance marker for a model that is not there")
    return 1

  write_marker(output, NVIDIA)
  print("wrote " + output + " and its " + NVIDIA + " provenance marker")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
