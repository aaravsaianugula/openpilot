"""Which AMD ASIC is behind the chestnut, and whether tinygrad can drive it.

`vendors.py` answers "whose card is this". That turns out not to be enough. tinygrad's
driverless AM driver is RDNA3/RDNA4 only, and says so outright in `tinygrad/runtime/ops_amd.py`:

    assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), "Unsupported arch"

`target` is derived from the GC IP version in the card's own discovery table, so an RDNA2 card
(GC 10.3.x, gfx103x) trips that assert -- and it trips *after* modeld has resolved
DEV='USB+AMD:LLVM' and been handed a gfx12-compiled bundle. The 60-second loader timeout fires,
manager restarts modeld, and the car cannot engage until the dock is unplugged. Vendor alone
cannot see that coming: an RX 6600 XT and the RX 9060 comma ships are both 0x1002.

**This table is a blocklist, and that is deliberate.** It names only the ASICs we have positive
evidence AM refuses. A card that is not listed takes exactly today's path, unchanged. An
allowlist would read better and would be wrong: it would put every AMD card comma has not
shipped yet -- and every one whose device ID we typed wrong -- behind a gate that switches a
working eGPU off. That is the same reasoning already written into `detect.resolve()`: a user
with a working AMD card must never regress *because this code exists*.

Device IDs are the Navi 2x entries under vendor 1002 in pci.ids (https://pci-ids.ucw.cz),
verbatim, minus the `Navi 2x USB` functions -- those are the card's USB-C controller, not the
GPU. One ID covers several retail names (0x73FF is 6600, 6600 XT and 6600M alike), so the names
here are pci.ids' own and are deliberately not narrowed to a single SKU.

Pure data, no imports beyond the package, same discipline as vendors.py: .elantra/test_egpu.py
imports this on a machine with no tinygrad, no dock and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from openpilot.sunnypilot.egpu.vendors import AMD

RDNA2 = "rdna2"


@dataclass(frozen=True)
class AsicSpec:
  name: str          # pci.ids' description of the chip
  gfx: str           # LLVM target, e.g. "gfx1032" -- what AM's assert compares against
  arch: str          # architecture family, for anything user-facing
  am_supported: bool # can tinygrad's driverless AM driver open it


def _rdna2(name: str, gfx: str) -> AsicSpec:
  return AsicSpec(name=name, gfx=gfx, arch=RDNA2, am_supported=False)


# gfx10.3x. No gc_10_* in tinygrad/runtime/autogen/am/regs.py, no soc_10, no smu_11 -- and
# ops_amd.py rejects the target before any of that is reached.
UNSUPPORTED_AMD: dict[int, AsicSpec] = {
  0x73A1: _rdna2("Navi 21 [Radeon Pro V620]", "gfx1030"),
  0x73A2: _rdna2("Navi 21 Pro-XTA [Radeon Pro W6900X]", "gfx1030"),
  0x73A3: _rdna2("Navi 21 GL-XL [Radeon PRO W6800]", "gfx1030"),
  0x73A5: _rdna2("Navi 21 [Radeon RX 6950 XT]", "gfx1030"),
  0x73AB: _rdna2("Navi 21 Pro-XLA [Radeon Pro W6800X]", "gfx1030"),
  0x73AE: _rdna2("Navi 21 [Radeon Pro V620 MxGPU]", "gfx1030"),
  0x73AF: _rdna2("Navi 21 [Radeon RX 6900 XT]", "gfx1030"),
  0x73BF: _rdna2("Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]", "gfx1030"),
  0x73C3: _rdna2("Navi 22", "gfx1031"),
  0x73CE: _rdna2("Navi 22-XL SRIOV MxGPU", "gfx1031"),
  0x73DF: _rdna2("Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]", "gfx1031"),
  0x73E0: _rdna2("Navi 23", "gfx1032"),
  0x73E1: _rdna2("Navi 23 WKS-XM [Radeon PRO W6600M]", "gfx1032"),
  0x73E3: _rdna2("Navi 23 WKS-XL [Radeon PRO W6600]", "gfx1032"),
  0x73EF: _rdna2("Navi 23 [Radeon RX 6650 XT / 6700S / 6800S]", "gfx1032"),
  0x73FF: _rdna2("Navi 23 [Radeon RX 6600/6600 XT/6600M]", "gfx1032"),
  0x7421: _rdna2("Navi 24 [Radeon PRO W6500M]", "gfx1034"),
  0x7422: _rdna2("Navi 24 [Radeon PRO W6400]", "gfx1034"),
  0x7423: _rdna2("Navi 24 [Radeon PRO W6300/W6300M]", "gfx1034"),
  0x7424: _rdna2("Navi 24 [Radeon RX 6300]", "gfx1034"),
  0x743F: _rdna2("Navi 24 [Radeon RX 6400/6500 XT/6500M]", "gfx1034"),
}


def asic_for(vendor: str, device_id: int | None) -> AsicSpec | None:
  """The descriptor for a probed device ID, or None when we have no opinion.

  None is the ordinary case and means "behave exactly as before". Only AMD IDs are matched:
  a PCI device ID is unique within a vendor, not across vendors, and 0x73FF under 0x10DE
  would be an entirely different chip.
  """
  if vendor != AMD or device_id is None:
    return None
  return UNSUPPORTED_AMD.get(device_id)


def am_supports(vendor: str, device_id: int | None) -> bool:
  """Can tinygrad's driverless AM driver open this card?

  True unless there is positive evidence otherwise, so an unrecognised card is never gated
  off and a probe that failed to read anything costs nobody their eGPU.
  """
  spec = asic_for(vendor, device_id)
  return spec is None or spec.am_supported
