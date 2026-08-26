"""chestnutState telemetry, split by where the numbers actually come from.

The cereal struct fuses two unrelated sources:

  * pcieLtssm, supplyVoltage, supplyCurrent come from the ASM2464 bridge. The bridge is the
    same silicon whichever card is behind it, so these are vendor-neutral. Upstream reaches
    them through Device["AMD"].iface.pci_dev.usb only because that handle happens to own the
    open USB device -- its own comment says the ASM runs on USB-C power and these still read
    with no GPU at all.

  * tempC, memoryTempC, powerDrawW, powerLimitW, gpuUsagePercent, gpuClockMhz, fanSpeedRpm
    come from AMD's SMU (PPSMC_MSG_GetPptLimit, TABLE_SMU_METRICS). There is no NVIDIA
    equivalent over this link: no NVML, no nvidia-smi, and tinygrad's NV backend exposes no
    SMU analogue. GA102 thermals live behind BAR0 therm/PMU registers, which is not something
    to hand-poke from a driving process.

So on NVIDIA those seven fields are left unwritten. Not zeroed to look tidy, not filled with a
plausible number -- unwritten. "SMU fields at 0, bridge fields populated" is already a wire
state upstream produces (it is what `big=False` emits), so this reuses an existing shape
rather than inventing one.

The vendor itself is a session constant, so it is logged once via cloudlog at modeld start
rather than added as a repeated field -- log.capnp is not in the overlay, churns weekly
upstream, and capnp field-id changes are a replay-compatibility hazard.
"""

from __future__ import annotations

import struct

from openpilot.sunnypilot.egpu.detect import resolve
from openpilot.sunnypilot.egpu.vendors import NVIDIA, spec_for

# ASM2464 register carrying PCIe LTSSM state; 0x78 is L0. Same address the chestnut flasher
# polls in system/hardware/chestnut/flash.py:link_up().
ASM_LTSSM_REG = 0xB450
# Vendor control IN on the bridge: <H h B> = bus millivolts, shunt milliamps, converter fault.
ASM_SUPPLY_REQUEST = 0xC0
ASM_SUPPLY_LEN = 5


def ChestnutState(pm, big):  # named for the class it substitutes, so the call site is one line
  """Vendor-dispatching factory, named like the class so the call site is a one-line swap.

  AMD dispatches to upstream's own class, imported lazily. Lazily on purpose twice over: it
  keeps this module importable without tinygrad (so CI can test it), and it means the AMD
  path is literally upstream's code and cannot drift from it.
  """
  if resolve()[0] == NVIDIA:
    return NvChestnutState(pm, big)
  from openpilot.selfdrive.modeld.modeld import ChestnutState as AmdChestnutState
  return AmdChestnutState(pm, big)


class NvChestnutState:
  """Bridge telemetry only. See the module docstring for why that is the whole of it."""

  def __init__(self, pm, big: bool):
    self.pm = pm
    self.big = big
    self.valid = True
    self.sends = 0

  def send(self) -> None:
    import messaging
    from tinygrad import Device

    msg = messaging.new_message('chestnutState')
    state = msg.chestnutState
    self.sends += 1

    tg_key = spec_for(NVIDIA).tg_key
    asm_valid = False
    try:
      if tg_key in Device._opened_devices:
        asm = Device[tg_key].iface.pci_dev.usb
        state.pcieLtssm = asm.read(ASM_LTSSM_REG, 1)[0]
        raw = bytes(asm.usb.control_read(ASM_SUPPLY_REQUEST, ASM_SUPPLY_LEN))[:4]
        state.supplyVoltage, state.supplyCurrent = struct.unpack('<Hh', raw)
        asm_valid = True
        self.valid = True
    except Exception:
      # Log the first failure only. A bridge that has stopped answering will keep not
      # answering at 10 Hz, and filling the log with it buries the one that mattered.
      if self.valid:
        # Imported here rather than at module scope: swaglog pulls in opendbc, which is a
        # submodule and is not checked out in CI. Keeping it lazy is what lets
        # .elantra/test_egpu.py import this module and assert which fields get written.
        from openpilot.common.swaglog import cloudlog
        cloudlog.exception("chestnut bridge read failed")
      self.valid = False

    # Upstream's `asm_valid and (not big or metrics_valid)` has no NV analogue, because there
    # are no metrics to be valid. The bridge read is the whole of the validity claim.
    msg.valid = asm_valid
    self.pm.send('chestnutState', msg)
