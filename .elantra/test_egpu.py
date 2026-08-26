#!/usr/bin/env python3
"""
Tests for the eGPU vendor logic.

None of this needs a dock, a GPU, or an openpilot build -- which is the point. The parts of
eGPU support that can be got wrong without hardware are: which vendor we believe is attached,
whether the model is allowed to run on it, and whether a model compiled for one card can
reach a device carrying the other. Those are all decidable here.

What is NOT testable here, and is not pretended to be:
  * whether DEV=USB+NV opens a GA102 through an ASM2464PD at all
  * whether the PCIe config probe returns 0x10DE
  * whether an NV-compiled model is numerically equivalent to the QCOM one
  * whether the dock's supply can carry the card
See .elantra/EGPU.md for the hardware bring-up those belong to.

The modules are imported from the tree, not copied, so the tests exercise shipping code.

Usage:
    python .elantra/test_egpu.py
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openpilot.sunnypilot.egpu import detect, models, vendors

EGPU_DIR = REPO / "openpilot/sunnypilot/egpu"
SCONSCRIPT = REPO / "openpilot/selfdrive/modeld/SConscript"
STATE_MODULE = REPO / "openpilot/selfdrive/ui/sunnypilot/mici/layouts/egpu_state.py"

failures: list[str] = []
passes: list[str] = []


def case(name: str, got, want) -> None:
    if got == want:
        passes.append(name)
        print("  ok    " + name)
    else:
        failures.append(name + ": got " + repr(got) + ", want " + repr(want))
        print("  FAIL  " + name + ": got " + repr(got) + ", want " + repr(want))


def check(name: str, condition: bool, detail: str = "") -> None:
    case(name + ((": " + detail) if detail and not condition else ""), bool(condition), True)


def load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeParams:
    """Just enough Params to drive the resolver. Values are str/bool like the real thing."""

    def __init__(self, **values):
        self._v = dict(values)

    def get(self, key):
        return self._v.get(key)

    def get_bool(self, key):
        return bool(self._v.get(key))

    def put(self, key, value, block=False):
        self._v[key] = value

    def remove(self, key):
        self._v.pop(key, None)


# --- vendor resolution --------------------------------------------------------------------

def test_configured():
    print("\n[configured] a garbage param must never raise")
    for raw, want in [("amd", "amd"), ("nvidia", "nvidia"), ("auto", "auto"),
                      ("AMD", "amd"), ("  NVIDIA  ", "nvidia"),
                      (b"amd", "amd"), ("", "auto"), (None, "auto"),
                      ("gibberish", "auto"), ("amd; rm -rf /", "auto")]:
        case("configured(" + repr(raw) + ")", detect.configured(FakeParams(EgpuVendor=raw)), want)


def test_resolve():
    print("\n[resolve] explicit > cache > probe > assumed AMD")
    case("explicit nvidia is not assumed",
         detect.resolve(FakeParams(EgpuVendor="nvidia")), ("nvidia", False))
    case("explicit amd is not assumed",
         detect.resolve(FakeParams(EgpuVendor="amd")), ("amd", False))
    case("auto with a cached probe uses it",
         detect.resolve(FakeParams(EgpuVendor="auto", EgpuVendorDetected="nvidia")),
         ("nvidia", False))
    case("auto with nothing falls back to AMD, marked assumed",
         detect.resolve(FakeParams()), ("amd", True))
    case("explicit beats a stale cache",
         detect.resolve(FakeParams(EgpuVendor="amd", EgpuVendorDetected="nvidia")),
         ("amd", False))
    case("a garbage cache is ignored, not trusted",
         detect.resolve(FakeParams(EgpuVendor="auto", EgpuVendorDetected="banana")),
         ("amd", True))
    case("queue_dev follows the vendor",
         detect.queue_dev(FakeParams(EgpuVendor="nvidia")), "NV")
    case("queue_dev on AMD is unchanged from upstream",
         detect.queue_dev(FakeParams(EgpuVendor="amd")), "AMD")


def test_enabled(tmp: Path):
    print("\n[enabled] the gate that decides whether the model touches the eGPU")
    case("AMD always runs on the eGPU, exactly as today",
         detect.enabled(FakeParams(EgpuVendor="amd")), True)
    case("an assumed vendor never runs an NV model",
         detect.enabled(FakeParams()), True)  # assumed -> amd -> today's behaviour
    case("NVIDIA without the opt-in stays off",
         detect.enabled(FakeParams(EgpuVendor="nvidia")), False)

    real = detect.NV_MODEL_PATH
    try:
        missing = tmp / "absent.pkl"
        detect.NV_MODEL_PATH = missing
        case("NVIDIA opted in but with no compiled model stays off",
             detect.enabled(FakeParams(EgpuVendor="nvidia", EgpuUseNvidia=True)), False)

        present = tmp / "big_driving_tinygrad_nv.pkl"
        present.write_bytes(b"not really a model")
        models.write_marker(str(present), vendors.NVIDIA)
        detect.NV_MODEL_PATH = present
        case("NVIDIA opted in with a marked model runs",
             detect.enabled(FakeParams(EgpuVendor="nvidia", EgpuUseNvidia=True)), True)

        Path(str(present) + models.MARKER_SUFFIX).write_text("amd\n", encoding="utf-8")
        case("a model marked for the other vendor does not count",
             detect.enabled(FakeParams(EgpuVendor="nvidia", EgpuUseNvidia=True)), False)
    finally:
        detect.NV_MODEL_PATH = real


def test_apply_env_is_total():
    """apply_env replaces `os.environ['GMMU'] = '0'`, which could not fail.

    It runs at modeld import, before anything else is set up. If a params read can make it
    raise, a params problem becomes "modeld cannot be imported", which takes the car off the
    road -- strictly worse than upstream. So it must fall back to upstream's exact behaviour.
    """
    print("\n[apply_env] never worse than the one-liner it replaces")

    class ExplodingParams:
        def get(self, key):
            raise RuntimeError("params unavailable")

        def get_bool(self, key):
            raise RuntimeError("params unavailable")

    before = os.environ.get("GMMU")
    try:
        os.environ.pop("GMMU", None)
        raised = None
        try:
            detect.apply_env(ExplodingParams())
        except Exception as e:  # catching broadly IS the assertion here
            raised = e
        case("a broken params store does not raise", raised, None)
        case("and GMMU still gets upstream's value", os.environ.get("GMMU"), "0")

        os.environ.pop("GMMU", None)
        detect.apply_env(FakeParams(EgpuVendor="amd"))
        case("the AMD path sets GMMU exactly as upstream did", os.environ.get("GMMU"), "0")
    finally:
        if before is None:
            os.environ.pop("GMMU", None)
        else:
            os.environ["GMMU"] = before


def test_catalog():
    print("\n[uses_amd_catalog] the offroad gate that keeps AMD bundles off an NV device")
    case("AMD gets the _USBGPU catalog", detect.uses_amd_catalog(FakeParams(EgpuVendor="amd")), True)
    case("NVIDIA never gets the _USBGPU catalog",
         detect.uses_amd_catalog(FakeParams(EgpuVendor="nvidia", EgpuUseNvidia=True)), False)
    case("an assumed vendor still gets today's AMD behaviour",
         detect.uses_amd_catalog(FakeParams()), True)


# --- model provenance ---------------------------------------------------------------------

def test_pkl_vendor(tmp: Path):
    print("\n[pkl_vendor] absent marker means AMD -- fail closed, never guess NVIDIA")
    pkl = tmp / "m.pkl"
    pkl.write_bytes(b"x")
    marker = Path(str(pkl) + models.MARKER_SUFFIX)

    case("no marker reads as amd", models.pkl_vendor(str(pkl)), "amd")
    for raw, want in [("nvidia", "nvidia"), ("NVIDIA\n", "nvidia"), ("  amd  ", "amd"),
                      ("", "amd"), ("garbage", "amd")]:
        marker.write_text(raw, encoding="utf-8")
        case("marker " + repr(raw), models.pkl_vendor(str(pkl)), want)
    marker.unlink()

    case("a path that does not exist reads as amd rather than raising",
         models.pkl_vendor(str(tmp / "nope.pkl")), "amd")


def test_assert_pkl(tmp: Path):
    print("\n[assert_pkl_matches] the last gate before load_oob unpickles anything")
    pkl = tmp / "amd.pkl"
    pkl.write_bytes(b"x")

    raised = None
    try:
        models.assert_pkl_matches(str(pkl), True, "nvidia")
    except RuntimeError as e:
        raised = str(e)
    check("an AMD pickle on an NV device raises", raised is not None)
    if raised:
        check("the message names both vendors", "nvidia" in raised and "amd" in raised)
        check("the message names the file", str(pkl) in raised)

    ok = True
    try:
        models.assert_pkl_matches(str(pkl), False, "nvidia")
    except RuntimeError:
        ok = False
    check("not routing through the eGPU means any pickle is fine", ok)

    ok = True
    try:
        models.assert_pkl_matches(str(pkl), True, "amd")
    except RuntimeError:
        ok = False
    check("a matching vendor passes", ok)

    bad = None
    try:
        models.write_marker(str(pkl), "intel")
    except ValueError as e:
        bad = str(e)
    check("refusing to mark a model with an unknown vendor", bad is not None)


# --- descriptor drift ---------------------------------------------------------------------

def test_descriptor_drift():
    """Re-derive the AMD literals from upstream source.

    If upstream changes a compile flag or the device string, this fails here rather than the
    AMD path silently diverging from what it used to be.
    """
    print("\n[drift] the AMD descriptor still matches upstream's own literals")
    src = SCONSCRIPT.read_text(encoding="utf-8", errors="replace")

    match = re.search(r"usbgpu_tg_flags\s*=\s*f?'([^']*)'", src)
    check("SConscript still defines usbgpu_tg_flags", match is not None)
    if match:
        upstream = set(match.group(1).replace("{tg_backend}", "QCOM").split())
        ours = set(vendors.compile_flags(vendors.AMD_SPEC, "QCOM").split())
        case("every upstream AMD compile flag is in our descriptor",
             sorted(upstream - ours), [])

    check("SConscript still selects AMD for the usbgpu queue",
          re.search(r"'usbgpu'\s*:\s*\{[^}]*'QUEUE_DEV'\s*:\s*'AMD'", src) is not None)
    case("our AMD tinygrad key matches", vendors.AMD_SPEC.tg_key, "AMD")
    case("our AMD device string matches SConscript", vendors.AMD_SPEC.dev, "USB+AMD:LLVM")

    # NV has no LLVM renderer -- ops_nv.py registers CUDA/PTX/NVCC/NAK. A ':LLVM' suffix here
    # would fail to resolve rather than fall back, so assert we never grow one by accident.
    check("the NV device string carries no renderer suffix", ":" not in vendors.NV_SPEC.dev)
    case("NV has no SMU", vendors.NV_SPEC.has_smu, False)
    case("NV has no published model catalog", vendors.NV_SPEC.catalog_suffix, "")


# --- telemetry discipline -------------------------------------------------------------------

def test_chestnut_state_fields():
    """The NV telemetry path must write the three bridge fields and touch no SMU field."""
    print("\n[chestnut_state] NVIDIA publishes bridge data only, never invented SMU numbers")

    class FakeState:
        def __init__(self):
            self.written = {}

        def __setattr__(self, key, value):
            if key == "written":
                object.__setattr__(self, key, value)
            else:
                self.written[key] = value

    class FakeMsg:
        def __init__(self):
            self.chestnutState = FakeState()
            self.valid = None

    class FakePm:
        def __init__(self):
            self.sent = []

        def send(self, name, msg):
            self.sent.append((name, msg))

    class FakeAsmUsb:
        @staticmethod
        def control_read(request, length):
            # 11.98 V, -250 mA, fault byte
            return struct.pack("<HhB", 11980, -250, 0)

    class FakeAsm:
        usb = FakeAsmUsb()

        @staticmethod
        def read(reg, size):
            return [0x78]

    fake_msg = FakeMsg()
    fake_messaging = type(sys)("messaging")
    fake_messaging.new_message = lambda name: fake_msg
    sys.modules["messaging"] = fake_messaging

    fake_tinygrad = type(sys)("tinygrad")

    class FakeDevice:
        _opened_devices = {"NV"}

        def __getitem__(self, key):
            iface = type("I", (), {"pci_dev": type("P", (), {"usb": FakeAsm()})()})()
            return type("D", (), {"iface": iface})()

    fake_tinygrad.Device = FakeDevice()
    sys.modules["tinygrad"] = fake_tinygrad

    try:
        from openpilot.sunnypilot.egpu.chestnut_state import NvChestnutState
        pm = FakePm()
        state = NvChestnutState(pm, big=True)
        state.send()

        written = fake_msg.chestnutState.written
        case("wrote exactly the three bridge fields", sorted(written),
             ["pcieLtssm", "supplyCurrent", "supplyVoltage"])
        case("LTSSM is the raw register byte", written.get("pcieLtssm"), 0x78)
        case("supply voltage decodes as unsigned mV", written.get("supplyVoltage"), 11980)
        case("supply current decodes as signed mA", written.get("supplyCurrent"), -250)
        case("the message is marked valid when the bridge answered", fake_msg.valid, True)
        case("the message was published once", len(pm.sent), 1)

        smu_fields = {"tempC", "memoryTempC", "powerDrawW", "powerLimitW",
                      "gpuUsagePercent", "gpuClockMhz", "fanSpeedRpm"}
        case("no SMU field was fabricated", sorted(smu_fields & set(written)), [])
    finally:
        sys.modules.pop("messaging", None)
        sys.modules.pop("tinygrad", None)


# --- panel logic ----------------------------------------------------------------------------

def test_panel_logic():
    print("\n[egpu_state] the panel explains itself correctly")
    state = load_by_path(STATE_MODULE, "egpu_state")

    case("no dock is reported as no dock",
         state.idle_reason(False, "amd", False, False, False).startswith("No chestnut"), True)
    case("a working AMD dock has nothing to explain",
         state.idle_reason(True, "amd", False, False, False), None)
    case("an unidentified card asks the user to set the vendor",
         "Set the eGPU vendor" in state.idle_reason(True, "nvidia", True, False, False), True)
    case("NVIDIA off explains the opt-in",
         "support is off" in state.idle_reason(True, "nvidia", False, False, False), True)
    case("NVIDIA on with no model explains the compile",
         "No model compiled" in state.idle_reason(True, "nvidia", False, True, False), True)
    case("NVIDIA fully set up has nothing to explain",
         state.idle_reason(True, "nvidia", False, True, True), None)

    case("assumed vendors are labelled as assumed",
         state.vendor_label("amd", True), "AMD (assumed)")
    case("confirmed vendors are not", state.vendor_label("nvidia", False), "NVIDIA")

    rows = state.status_rows(False, "amd", True, False, False, False)
    case("with no dock the model row says on device", dict(rows).get("driving model"), "on device")
    rows = dict(state.status_rows(True, "nvidia", False, True, True, True))
    case("an NVIDIA dock in use says so", rows.get("driving model"), "on the eGPU")
    case("the NVIDIA rows report the compiled model", rows.get("compiled model"), "present")

    case("the telemetry note only appears for NVIDIA", state.telemetry_note("amd"), None)
    check("the NVIDIA telemetry note explains the missing readings",
          "no NVIDIA equivalent" in (state.telemetry_note("nvidia") or ""))


# --- import purity ---------------------------------------------------------------------------

def test_import_purity():
    """Nothing in the egpu package may import tinygrad or pyray at module scope.

    Both are why upstream's modeld cannot be imported in CI. Keeping them lazy is the only
    reason these tests can import the shipping code at all, so it is worth asserting rather
    than remembering.
    """
    print("\n[purity] no heavy imports at module scope")
    banned = {"tinygrad", "pyray", "messaging"}
    for path in sorted(EGPU_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        top = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.append(node.module.split(".")[0])
        hits = sorted(banned & set(top))
        case(path.name + " has no heavy top-level import", hits, [])


def main() -> int:
    print("eGPU vendor logic")
    with tempfile.TemporaryDirectory(prefix="egpu-test-") as raw_tmp:
        tmp = Path(raw_tmp)
        test_configured()
        test_resolve()
        test_enabled(tmp)
        test_apply_env_is_total()
        test_catalog()
        test_pkl_vendor(tmp)
        test_assert_pkl(tmp)
        test_descriptor_drift()
        test_chestnut_state_fields()
        test_panel_logic()
        test_import_purity()

    print("\n" + "-" * 60)
    if failures:
        print("FAILED: " + str(len(failures)) + " case(s) failed, "
              + str(len(passes)) + " passed\n")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: all " + str(len(passes)) + " cases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
