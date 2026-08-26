"""What each eGPU vendor needs, as data.

The chestnut is a USB-to-PCIe bridge, not a GPU. Which card sits behind it decides the
tinygrad device string, the telemetry that exists, and which pre-compiled model bundles are
usable. Upstream hardcodes all of that to AMD because AMD is what comma ships.

Pure data, no imports beyond the stdlib. Importable in CI on a machine with no tinygrad, no
pyray and no GPU, which is what makes .elantra/test_egpu.py able to test it at all.

Every AMD literal here is copied verbatim from the upstream site it replaces, so the AMD
path stays bit-identical to today. .elantra/test_egpu.py re-derives them from upstream
source and fails if upstream changes them under us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

AMD = "amd"
NVIDIA = "nvidia"
AUTO = "auto"

VALID_VENDORS = (AUTO, AMD, NVIDIA)


@dataclass(frozen=True)
class EgpuSpec:
  name: str
  tg_key: str               # tinygrad Device[...] key, used at runtime to place tensors
  dev: str                  # tinygrad DEV= string for compiling against the chestnut
  pci_vendor_id: int        # PCIe config-space vendor ID at offset 0x00
  has_smu: bool             # can populate the SMU-derived ChestnutState fields
  catalog_suffix: str       # ModelCache suffix; "" means no published eGPU catalog exists
  env: dict[str, str] = field(default_factory=dict)   # tinygrad env, set before `import tinygrad`
  flags: str = ""           # remaining compile flags


# DEV=USB+AMD:LLVM and the flags come from selfdrive/modeld/SConscript; 'AMD' as the runtime
# device key comes from sunnypilot/modeld_v2/modeld.py. GMMU=0 is set at modeld import
# upstream, described there as "for usbgpu fast loading, noop for qcom".
AMD_SPEC = EgpuSpec(
  name=AMD,
  tg_key="AMD",
  dev="USB+AMD:LLVM",
  pci_vendor_id=0x1002,
  has_smu=True,
  catalog_suffix="_USBGPU",
  env={"GMMU": "0"},
  flags="DEBUG=2 FLOAT16=1 JIT_BATCH_SIZE=0 TC_OPT=2",
)

# No renderer suffix on purpose. tinygrad's NV backend registers
# [CUDARenderer, PTXRenderer, NVCCRenderer, NAKRenderer] (ops_nv.py) and *no* LLVM renderer,
# so the ':LLVM' that AMD uses is not a valid target here -- it would fail to resolve rather
# than fall back. Leaving it off takes tinygrad's default (CUDA).
#
# Worth knowing: CUDA and PTX are the two renderers tinygrad issue #11705 found producing
# silently wrong openpilot model outputs (max abs diff >50,000 against the ONNX reference,
# no error raised). That is why an NV model is not allowed to drive until it has been
# validated numerically -- see .elantra/EGPU.md.
#
# GMMU is an AMD-MMU control with no defined meaning over USB+NV, so it is not set here.
NV_SPEC = EgpuSpec(
  name=NVIDIA,
  tg_key="NV",
  dev="USB+NV",
  pci_vendor_id=0x10DE,
  has_smu=False,
  catalog_suffix="",
  env={},
  flags="DEBUG=2 FLOAT16=1 JIT_BATCH_SIZE=0 TC_OPT=2",
)

SPECS = {AMD: AMD_SPEC, NVIDIA: NV_SPEC}
BY_PCI_ID = {spec.pci_vendor_id: spec for spec in SPECS.values()}


def spec_for(vendor: str) -> EgpuSpec:
  """The descriptor for a vendor name. Unknown names are an error, not a silent default."""
  try:
    return SPECS[vendor]
  except KeyError:
    raise ValueError("unknown eGPU vendor " + repr(vendor)
                     + "; expected one of " + ", ".join(sorted(SPECS))) from None


def compile_flags(spec: EgpuSpec, warp_dev: str) -> str:
  """The full tinygrad flag string for compiling a model against this card.

  Reproduces SConscript's usbgpu_tg_flags for AMD, token for token.
  """
  env = " ".join(k + "=" + v for k, v in sorted(spec.env.items()))
  parts = ["DEV=" + spec.dev, "WARP_DEV=" + warp_dev, spec.flags, env]
  return " ".join(p for p in parts if p)
