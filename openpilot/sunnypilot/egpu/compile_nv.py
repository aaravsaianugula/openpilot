"""Compile a driving model for an NVIDIA card in the chestnut.

Kept as an entry point because it is what .elantra/EGPU.md, the model panel's error string and
every note written during bring-up tell people to run. The work moved to compile_egpu.py once
RDNA2 made the same problem exist on AMD; all this does is pin the vendor.

    python -m openpilot.sunnypilot.egpu.compile_nv \
        --model-type supercombo --model-size 1440x960 \
        --camera-resolutions 1928x1208 --supercombo-onnx <path>
"""

from __future__ import annotations

import sys

from openpilot.sunnypilot.egpu import compile_egpu
from openpilot.sunnypilot.egpu.vendors import NVIDIA


def main(argv: list[str]) -> int:
  return compile_egpu.main([compile_egpu.VENDOR_FLAG, NVIDIA] + list(argv))


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
