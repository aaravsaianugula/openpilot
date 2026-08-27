"""Read the eGPU's PCIe vendor ID through the chestnut bridge.

Deliberately an adapter, never a reimplementation. The register sequence that turns ASM
controller writes into a PCIe configuration-read TLP is tinygrad's, it is undocumented, and a
wrong write to a bridge control register is not a recoverable software error. So when
tinygrad is importable and the device is free we borrow its controller; otherwise we return
None.

None is not a failure. It means "ask the user", which is what the EgpuVendor param is for.
The param is the source of truth precisely because this cannot be: it cannot run in CI, it
cannot run on a machine without the dock, and so it can never be a tested path.

Callers must guarantee tinygrad does not currently hold the device -- USBPCIDevice takes an
exclusive flock, so probing underneath a running modeld would either fail or, worse, disturb
a device that is driving. Only offroad callers may use this.
"""

from __future__ import annotations

from openpilot.sunnypilot.egpu.vendors import BY_PCI_ID

# Where the GPU sits behind the bridge. AMD's USBIface enumerates it on bus 4; the NV work in
# tinygrad PR #17369 puts it on bus 2. We do not know which card is there yet -- that is the
# whole question -- so both are tried.
CANDIDATE_BUSES = (4, 2)


def probe_pci_ids() -> tuple[int, int] | None:
  """(vendor ID, device ID) from PCIe config offset 0x00, or None if it cannot be read safely.

  Both live in the one dword we already fetch -- vendor in the low half, device in the high
  half -- so the device ID costs no extra USB round trip. Which card it is matters as much as
  whose: tinygrad's AM driver refuses RDNA2, and an RX 6600 XT and comma's RX 9060 are both
  0x1002 (see asics.py).
  """
  try:
    from tinygrad.runtime.support.usb import USB3, CustomASM24Controller
  except Exception:
    return None

  # The custom-firmware IDs tinygrad's own USBIface enumerates. A chestnut still on stock
  # ASMedia firmware is not usable by tinygrad at all, so there is nothing to probe.
  try:
    devices = USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)
  except Exception:
    return None
  if not devices:
    return None

  try:
    usb = CustomASM24Controller(USB3(devices[0][0]))
  except Exception:
    # Includes the case where the PCIe link is not up, which CustomASM24Controller raises on.
    return None

  for bus in CANDIDATE_BUSES:
    try:
      word = usb.pcie_cfg_req(0x00, bus=bus, dev=0, fn=0, size=4)
    except Exception:
      continue
    vendor_id, device_id = word & 0xFFFF, (word >> 16) & 0xFFFF
    if vendor_id in BY_PCI_ID:
      return vendor_id, device_id
  return None


def probe_pci_vendor_id() -> int | None:
  """The vendor ID alone, or None if it could not be read safely."""
  ids = probe_pci_ids()
  return None if ids is None else ids[0]


def probe_ids() -> tuple[str, int] | None:
  """(vendor name, device ID) behind the bridge, or None when it could not be determined."""
  ids = probe_pci_ids()
  return None if ids is None else (BY_PCI_ID[ids[0]].name, ids[1])


def probe_vendor() -> str | None:
  """The vendor name behind the bridge, or None when it could not be determined."""
  probed = probe_ids()
  return None if probed is None else probed[0]
