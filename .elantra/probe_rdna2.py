#!/usr/bin/env python3
"""
Can tinygrad's driverless AM driver ever drive an RDNA2 card over the chestnut?

tinygrad's AM driver is RDNA3/RDNA4 only. `tinygrad/runtime/ops_amd.py` says so outright:

    assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), "Unsupported arch"

so an RX 6600 XT (Navi 23, GC 10.3.x, gfx1032) is refused before a kernel runs, and below that
assert there is no gc_10_* register set, no soc_10 and no smu_11 either. Supporting the card
means porting AM to GC 10.3: new register sets, PSP v11 bring-up, gfx10 CP init, Navi 2x
firmware naming, and a reset path that does not need endpoint FLR. That is weeks of driver
work, and it is worth nothing if the card cannot even be brought out of reset over a USB
bridge -- which is a real possibility. tinygrad issue #15636 reports an RX 6900 XT (also
RDNA2) dying at exactly that point on the sibling TinyGPU path, with "BL not ready" and
C2PMSG_35 reading 0x0.

**Stage 6 is the whole point of this script.** If the PSP bootloader never reports ready, the
port is dead before any of the register work matters and the answer is to stop. Everything
before it exists to make stage 6's result trustworthy rather than to be interesting.

Nothing here runs a kernel, loads firmware, or resets the card. Stage 2 opens tinygrad's
USBPCIDevice, which programs the BARs -- the same thing tinygrad does on every open, and it
takes the exclusive flock. Stages 3-4 are config-space reads, stages 5-7 MMIO reads.

Run it offroad, on the bench, with the dock attached and modeld stopped: USBPCIDevice takes an
exclusive flock, so this cannot share the device with a running model.

Usage:
    python .elantra/probe_rdna2.py
    python .elantra/probe_rdna2.py --expect 0x73ff     # fail if a different card answers
"""

from __future__ import annotations

import argparse
import array
import ctypes
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ASM_LTSSM_REG = 0xB450
LTSSM_L0 = 0x78
GPU_BUS = 4              # where AMD's USBIface enumerates the card behind the bridge
MMIO_BAR = 5             # AMDev maps bar 5 as its register aperture
VRAM_BAR = 0
mmRCC_CONFIG_MEMSIZE = 0xDE3
MM_INDEX, MM_DATA, MM_INDEX_HI = 0x00, 0x01, 0x06
PSP_BL_READY = 0x80000000

# Stage 6 has three outcomes, not two. "We could not run the check" must never print as
# "the card failed the check" -- that is a wrong answer to the only question this tool asks.
GO, NO_GO, UNTESTED = "go", "no-go", "untested"

_failed = False


def head(title: str) -> None:
  print("")
  print("=" * 72)
  print(title)
  print("=" * 72)


def ok(label: str, detail: str = "") -> None:
  print("  ok    " + label + ((": " + detail) if detail else ""))


def info(label: str, detail: str = "") -> None:
  print("  --    " + label + ((": " + detail) if detail else ""))


def bad(label: str, detail: str = "") -> None:
  global _failed
  _failed = True
  print("  FAIL  " + label + ((": " + detail) if detail else ""))


def ver(v) -> str:
  """An IP version tuple as people write it."""
  return f"{v[0]}.{v[1]}.{v[2]}"


def stop(why: str) -> int:
  print("")
  print("STOP: " + why)
  return 1


def speed_verdict(speed: str | None) -> tuple[bool, str, str, str]:
  """(proceed, level, label, detail) for the dock's link speed.

  Pure, so it can be tested without a dock. The first version of this probe surveyed every
  USB device and stopped if *any* of them read 480 -- which on a comma four is always true,
  because the modem is a USB 2 device. It declared "only USB 2 devices present" on a machine
  whose own output listed three devices at 5000. Only the dock's own port means anything.
  """
  if speed is None:
    return True, "info", "no chestnut on the USB bus", "stage 2 will say so definitively"
  if speed == "10000":
    return True, "ok", "the dock is enumerated at 10000 Mb/s", ""
  if speed == "480":
    return False, "bad", "the dock enumerated at USB 2 (480 Mb/s)",            "the known ASM2464PD fallback -- power-cycle the dock and retry"
  return True, "info", "the dock enumerated at " + speed + " Mb/s",          "expected 10000; throughput past here is not comparable to the documented figures"


def _usb_helpers():
  """openpilot's own chestnut USB constants and sysfs readers, loaded by file path.

  `import openpilot.common.hardware.usb` runs the package __init__, which pulls in
  HardwareBase -> cereal -> capnp. This is a bench tool that has to work against a bare
  checkout with no openpilot build, and usb.py is itself stdlib-only -- so load that one
  file directly rather than duplicating the IDs here and letting them drift.
  """
  import importlib.util
  path = REPO / "openpilot/common/hardware/usb.py"
  spec = importlib.util.spec_from_file_location("chestnut_usb", path)
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot load " + str(path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _chestnut_port() -> tuple[str | None, bool]:
  """(link speed, is on custom firmware) for the dock, or (None, False) if it is not attached."""
  usb = _usb_helpers()
  CHESTNUT_USB_IDS, CHESTNUT_ROM_USB_IDS = usb.CHESTNUT_USB_IDS, usb.CHESTNUT_ROM_USB_IDS
  read, read_int, usb_devices = usb.read, usb.read_int, usb.usb_devices
  for device in usb_devices():
    ids = (read_int(device / "idVendor", 16), read_int(device / "idProduct", 16))
    if ids in CHESTNUT_USB_IDS:
      return read(device / "speed"), True
    if ids in CHESTNUT_ROM_USB_IDS:
      return read(device / "speed"), False
  return None, False


def stage1_usb_speed() -> bool:
  """The dock's own link speed. 480 there is the ASM2464PD USB-2 fallback, and makes every
  later number meaningless, so it is a stop rather than a warning."""
  head("Stage 1 -- dock USB link speed")
  speed, custom_fw = _chestnut_port()
  if speed is not None:
    info("chestnut firmware", "custom (tinygrad-usable)" if custom_fw
         else "stock ASMedia ROM -- tinygrad cannot drive this; reflash the dock")
  proceed, level, label, detail = speed_verdict(speed)
  {"ok": ok, "info": info, "bad": bad}[level](label, detail)
  return proceed


def stage2_bridge():
  """The chestnut, its PCIe link, and an opened device. The controller raises if it is down."""
  head("Stage 2 -- chestnut bridge and PCIe link")
  try:
    from tinygrad.runtime.support.system import USBPCIDevice
    from tinygrad.runtime.support.usb import USB3
  except Exception as e:
    bad("tinygrad is not importable here", str(e))
    return None

  # Not swallowed: reported. Without libusb there is no way to reach the bridge at all, and
  # that is a fact about the machine this is running on, not a fault of the card.
  try:
    devices = USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)
  except Exception as e:
    bad("cannot enumerate USB devices", str(e))
    info("this probe needs libusb and the dock", "run it on the comma four, not a laptop")
    return None

  if not devices:
    bad("no chestnut on custom firmware found",
        "tinygrad only drives the custom firmware; stock ASMedia firmware is not usable")
    return None
  ok("chestnut found", str(len(devices)) + " device(s)")

  # Let USBPCIDevice build the controller. It owns the ASM2464 endpoint numbers and the
  # custom-vs-stock firmware choice, and both have changed shape between tinygrad revisions
  # -- constructing USB3 by hand here broke against the tinygrad the car actually runs.
  # It also takes the exclusive flock, which is what keeps this off a device modeld is using.
  try:
    pci_dev = USBPCIDevice("AM", *devices[0])
  except Exception as e:
    bad("could not open the bridge", str(e))
    # 0x00 is not a software state. The bridge answered over USB, so its firmware is fine;
    # the PCIe side simply never trained, which on this dock is almost always the card.
    if "LTSSM=0x00" in str(e):
      info("LTSSM 0x00 means the link never came up at all",
           "check the card's 8-pin PCIe power lead, that the dock's supply is switched on, "
           + "and that the card is fully seated -- none of that is fixable from software")
    return None

  ltssm = pci_dev.usb.read(ASM_LTSSM_REG, 1)[0]
  if ltssm != LTSSM_L0:
    bad("LTSSM is not L0", hex(ltssm) + ", want " + hex(LTSSM_L0))
    return None
  ok("LTSSM is L0", hex(ltssm))
  return pci_dev


def stage3_identity(usb, expect: int | None) -> int | None:
  """Which card actually answered, read straight out of config space."""
  head("Stage 3 -- card identity")
  word = usb.pcie_cfg_req(0x00, bus=GPU_BUS, dev=0, fn=0, size=4)
  vendor_id, device_id = word & 0xFFFF, (word >> 16) & 0xFFFF
  rev = usb.pcie_cfg_req(0x08, bus=GPU_BUS, dev=0, fn=0, size=4) & 0xFF
  info("config 0x00", hex(word))
  ok("vendor", hex(vendor_id) + (" (AMD)" if vendor_id == 0x1002 else ""))
  ok("device", hex(device_id))
  ok("revision", hex(rev))

  from openpilot.sunnypilot.egpu.asics import asic_for
  from openpilot.sunnypilot.egpu.vendors import AMD
  spec = asic_for(AMD, device_id) if vendor_id == 0x1002 else None
  if spec is not None:
    ok("identified", spec.name + " -- " + spec.gfx + " (" + spec.arch + ")")
    info("driveable by AM today", "no, which is exactly why we are probing")
  else:
    info("not in the RDNA2 blocklist",
         "either a card AM already supports, or one we have no entry for")

  if expect is not None and device_id != expect:
    bad("a different card answered", "expected " + hex(expect) + ", got " + hex(device_id))
    return None
  return device_id


def stage4_flr(usb) -> None:
  """Does the endpoint advertise Function Level Reset?

  tinygrad issue #15636 stalled on RDNA2 partly because the GPU function does not support FLR
  and the reset path had nothing else to use. Not a stop condition -- a bridge-level or PSP
  mode1 reset may still work -- but it decides what a teardown path would have to be.
  """
  head("Stage 4 -- reset capability")
  status = (usb.pcie_cfg_req(0x04, bus=GPU_BUS, dev=0, fn=0, size=4) >> 16) & 0xFFFF
  if not status & 0x10:
    info("no PCI capability list", "cannot tell whether FLR is supported")
    return

  ptr = usb.pcie_cfg_req(0x34, bus=GPU_BUS, dev=0, fn=0, size=4) & 0xFF
  seen: set[int] = set()
  while ptr and ptr != 0xFF and ptr not in seen:
    seen.add(ptr)
    cap = usb.pcie_cfg_req(ptr & 0xFC, bus=GPU_BUS, dev=0, fn=0, size=4)
    cap_id, nxt = cap & 0xFF, (cap >> 8) & 0xFF
    if cap_id == 0x10:  # PCI Express capability
      devcap = usb.pcie_cfg_req((ptr + 4) & 0xFC, bus=GPU_BUS, dev=0, fn=0, size=4)
      info("PCIe capability at", hex(ptr))
      if devcap & (1 << 28):
        ok("endpoint advertises FLR")
      else:
        info("endpoint does NOT advertise FLR",
             "teardown would need a bridge or PSP mode1 reset -- see tinygrad #15636")
      return
    ptr = nxt
  info("no PCIe capability structure found", "unexpected for a GPU")


def stage5_discovery(pci_dev):
  """The card's own IP discovery table: the exact work list a port would have to write.

  This replicates AMDev._run_discovery rather than calling it, because constructing AMDev is
  precisely what cannot work here -- it resolves soc_10 and gc_10 modules that do not exist.
  VRAM is read through MM_INDEX/MM_DATA, so a 256MB BAR over 8GB of VRAM is not a problem.
  """
  head("Stage 5 -- IP discovery table")
  from tinygrad.runtime.autogen.am import am

  mmio = pci_dev.map_bar(MMIO_BAR, fmt="I")

  def rreg(reg: int) -> int:
    if reg >= len(mmio):
      raise IndexError("register " + hex(reg) + " is outside the MMIO aperture ("
                       + hex(len(mmio)) + " dwords); AM would reach it over RSMU")
    return mmio[reg]

  def wreg(reg: int, val: int) -> None:
    mmio[reg] = val

  vram_size = rreg(mmRCC_CONFIG_MEMSIZE) << 20
  if vram_size == 0 or vram_size > (128 << 30):
    bad("implausible VRAM size", hex(vram_size) + " -- MMIO reads are not working")
    return None
  ok("VRAM size", str(vram_size >> 20) + " MB")
  bar_addr, bar_size = pci_dev.bar_info(VRAM_BAR)
  ok("VRAM BAR", str(bar_size >> 20) + " MB at " + hex(bar_addr))
  info("large BAR", "yes" if bar_size >= vram_size else "no, the CPU-visible pool is the BAR")

  addr, size = vram_size - (64 << 10), (10 << 10)
  words = []
  for caddr in range(addr, addr + size, 4):
    wreg(MM_INDEX_HI, caddr >> 31)
    wreg(MM_INDEX, (caddr & 0x7FFFFFFF) | 0x80000000)
    words.append(rreg(MM_DATA))
  disc = bytearray(bytes(array.array("I", words)))

  bhdr = am.struct_binary_header.from_buffer(disc)
  if bhdr.binary_signature != am.BINARY_SIGNATURE:
    bad("discovery table signature mismatch", hex(bhdr.binary_signature))
    return None
  ihdr = am.struct_ip_discovery_header.from_address(
    ctypes.addressof(bhdr) + bhdr.table_list[am.IP_DISCOVERY].offset)
  if ihdr.signature != am.DISCOVERY_TABLE_SIGNATURE:
    bad("ip discovery signature mismatch", hex(ihdr.signature))
    return None
  ok("discovery table parsed", str(ihdr.num_dies) + " die(s)")

  names = {v: k.removesuffix("_HWIP") for k, v in vars(am).items()
           if k.endswith("_HWIP") and isinstance(v, int)}
  ip_ver: dict[int, tuple[int, int, int]] = {}
  regs_offset: dict[int, dict[int, tuple]] = {}
  for die in range(ihdr.num_dies):
    dhdr = am.struct_die_header.from_address(
      ctypes.addressof(bhdr) + ihdr.die_info[die].die_offset)
    off = ctypes.addressof(bhdr) + ctypes.sizeof(dhdr) + ihdr.die_info[die].die_offset
    for _ in range(dhdr.num_ips):
      ip = am.struct_ip_v4.from_address(off)
      base_t = ctypes.c_uint64 if ihdr.base_addr_64_bit else ctypes.c_uint32
      bases = (base_t * ip.num_base_address).from_address(off + 8)
      for hw_ip in range(1, am.MAX_HWIP):
        if hw_ip in am.hw_id_map and am.hw_id_map[hw_ip] == ip.hw_id:
          regs_offset.setdefault(hw_ip, {})[ip.instance_number] = tuple(bases)
          ip_ver[hw_ip] = (ip.major, ip.minor, ip.revision)
      off += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address

  print("")
  print("  IP block versions -- this is the port's work list:")
  for hw_ip in sorted(ip_ver, key=lambda k: names.get(k, str(k))):
    name = names.get(hw_ip, "HWIP" + str(hw_ip))
    print(f"      {name:<12} {ver(ip_ver[hw_ip])}")

  gc = ip_ver.get(am.GC_HWIP)
  if gc is not None:
    gfxver = int(f"{gc[0]:02d}{gc[1]:02d}{gc[2]:02d}")
    target = (gfxver // 10000, (gfxver // 100) % 100, gfxver % 100)
    print("")
    ok("gfx target", f"gfx{target[0]}{target[1]:x}{target[2]:x} {target}")
    if target[0] in (11, 12) or target in ((9, 4, 2), (9, 5, 0)):
      ok("ops_amd.py would accept this target", "AM can drive this card as it stands")
    else:
      info("ops_amd.py would reject this target",
           "assert (target in ((9,4,2),(9,5,0))) or target[0] in (11, 12)")
  return ip_ver, regs_offset, mmio


def stage6_psp(ip_ver, regs_offset, mmio, timeout_s: float) -> str:
  """The decisive check: does the PSP bootloader come up?

  AM_PSP._wait_for_bootloader polls regMP0_SMN_C2PMSG_35 for bit 31. Everything the driver
  does afterwards -- loading SOS, the TMR, every other IP's firmware -- is downstream of this
  one register. tinygrad #15636 is an RDNA2 card sitting here reading 0x0 forever.
  """
  head("Stage 6 -- PSP bootloader (THE DECISIVE CHECK)")
  from tinygrad.runtime.autogen.am import am
  from tinygrad.runtime.support.amd import import_module

  mp0 = ip_ver.get(am.MP0_HWIP)
  if mp0 is None:
    bad("no MP0 (PSP) block in the discovery table", "nothing to poll")
    return UNTESTED
  ok("MP0 (PSP) IP version", ver(mp0))

  # submod="regs" matters: without it import_module searches the autogen package, whose
  # __all__ lists soc_*/regs/pmc, not the per-ASIC register tables. import_asic_regs() is
  # the wrapper that gets this right, and the mp_11_0_0 table Navi 2x needs is present.
  try:
    regs = import_module("mp", mp0, submod="regs")
  except Exception as e:
    bad("no register set for this PSP version", str(e))
    return UNTESTED

  entry = regs.get("mmMP0_SMN_C2PMSG_35") or regs.get("regMP0_SMN_C2PMSG_35")
  if entry is None:
    bad("C2PMSG_35 is not defined for this PSP version", "cannot poll the bootloader")
    return UNTESTED
  offset, segment = entry[0], entry[1]
  addr = regs_offset[am.MP0_HWIP][0][segment] + offset
  info("regMP0_SMN_C2PMSG_35 at", hex(addr))
  if addr >= len(mmio):
    bad("C2PMSG_35 is outside the MMIO aperture",
        hex(addr) + " >= " + hex(len(mmio)) + "; AM reaches it over RSMU, which needs the "
        + "nbio register set this probe does not build")
    return UNTESTED

  deadline, value = time.monotonic() + timeout_s, 0
  while time.monotonic() < deadline:
    value = mmio[addr]
    if value & PSP_BL_READY:
      ok("PSP bootloader reports ready", hex(value))
      print("")
      print("  GO. The card comes out of reset over the chestnut. The RDNA2 port is worth")
      print("  attempting, and stage 5's IP versions are the modules it needs.")
      return GO
    time.sleep(0.05)

  bad("PSP bootloader never reported ready",
      "C2PMSG_35 = " + hex(value) + " after " + str(timeout_s) + "s, want bit 31 set")
  entry64 = regs.get("mmMP0_SMN_C2PMSG_64") or regs.get("regMP0_SMN_C2PMSG_64")
  if entry64 is not None:
    addr64 = regs_offset[am.MP0_HWIP][0][entry64[1]] + entry64[0]
    if addr64 < len(mmio):
      info("C2PMSG_64", hex(mmio[addr64]))
  return NO_GO


def stage7_registers(ip_ver) -> None:
  """What tinygrad would still be missing even if the card did come up."""
  head("Stage 7 -- register sets tinygrad already has for this card")
  from tinygrad.runtime.autogen.am import am
  from tinygrad.runtime.support.amd import import_module

  # smu_* are modules of the autogen package; the rest are tables inside am/regs.py. AM
  # itself makes the same distinction -- _build_regs uses import_asic_regs (submod="regs")
  # while AM_SMU resolves its module package-level.
  for prefix, hwip, submod in (("gc", am.GC_HWIP, "regs"), ("mp", am.MP0_HWIP, "regs"),
                               ("smu", am.MP1_HWIP, ""), ("nbio", am.NBIO_HWIP, "regs"),
                               ("osssys", am.OSSSYS_HWIP, "regs"),
                               ("mmhub", am.MMHUB_HWIP, "regs")):
    version = ip_ver.get(hwip)
    if version is None:
      continue
    try:
      import_module(prefix, version, submod=submod)
      ok(f"{prefix} {ver(version)}", "present")
    except Exception as e:
      info(f"{prefix} {ver(version)}", "MISSING -- " + str(e))

  # soc_* are generated lazily on the *package*, unlike the structs above which live in the
  # am.py module inside it. AM_SOC does `import_soc(ip)` -> getattr(package, f"soc_{major}"),
  # so a missing soc_10 is one of the concrete things an RDNA2 port would have to produce.
  import tinygrad.runtime.autogen.am as am_pkg
  major = ip_ver.get(am.GC_HWIP, (0,))[0]
  try:
    getattr(am_pkg, "soc_" + str(major))
    ok("soc_" + str(major), "present")
  except AttributeError as e:
    info("soc_" + str(major), "MISSING -- AM_SOC cannot be built: " + str(e))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--expect", default=None,
                  help="PCI device id the card must report, hex (e.g. 0x73ff)")
  ap.add_argument("--psp-timeout", type=float, default=5.0,
                  help="seconds to wait for the PSP bootloader (default 5)")
  args = ap.parse_args()
  expect = int(args.expect, 16) if args.expect else None

  print("RDNA2-on-chestnut bring-up probe")
  print("  repo: " + str(REPO))

  if not stage1_usb_speed():
    return stop("USB 2 fallback. Nothing measured past this point would mean anything.")

  pci_dev = stage2_bridge()
  if pci_dev is None:
    return stop("The bridge or its PCIe link is not up. Fix that before reading anything.")

  if stage3_identity(pci_dev.usb, expect) is None:
    return stop("Not the card we were told to probe.")

  stage4_flr(pci_dev.usb)

  discovered = stage5_discovery(pci_dev)
  if discovered is None:
    return stop("Could not read the card's discovery table. MMIO is not working.")
  ip_ver, regs_offset, mmio = discovered

  result = stage6_psp(ip_ver, regs_offset, mmio, args.psp_timeout)
  stage7_registers(ip_ver)

  head("Verdict")
  if result == GO:
    print("  GO -- the PSP bootloader came up. Porting AM to this card is worth doing.")
    print("  Stage 5 lists the IP versions and stage 7 the register sets still missing.")
    return 0
  if result == NO_GO:
    print("  NO-GO -- the PSP bootloader was polled and never reported ready. Everything a")
    print("  port would build sits downstream of that register, so do not start the port on")
    print("  this evidence. This is the failure tinygrad #15636 reports for RDNA2.")
    return 1
  print("  INCONCLUSIVE -- stage 6 could not be run, so the card was never actually asked.")
  print("  This is NOT a no-go: the decisive question is simply still unanswered. Fix what")
  print("  stage 6 reported above and run again.")
  return 2


if __name__ == "__main__":
  sys.exit(main())
