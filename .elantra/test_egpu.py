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

from openpilot.sunnypilot.egpu import asics, compile_egpu, detect, models, vendors

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

    def put_bool(self, key, value, block=False):
        self._v[key] = bool(value)

    def remove(self, key):
        self._v.pop(key, None)


_REAL_PROBE_IDS = None


def monkey_probe(dock_present: bool, probe_ids) -> None:
    """Point detect's two lazy imports at fakes.

    probe_once imports usbgpu_present and probe_ids inside the function body, so both are
    replaced at their source module rather than as attributes of detect.
    """
    global _REAL_PROBE_IDS
    import types

    from openpilot.sunnypilot.egpu import probe as probe_mod
    if _REAL_PROBE_IDS is None:
        _REAL_PROBE_IDS = probe_mod.probe_ids

    helpers = types.ModuleType("openpilot.selfdrive.modeld.helpers")
    helpers.usbgpu_present = lambda: dock_present
    sys.modules["openpilot.selfdrive.modeld.helpers"] = helpers
    probe_mod.probe_ids = probe_ids


def unmonkey_probe() -> None:
    sys.modules.pop("openpilot.selfdrive.modeld.helpers", None)
    if _REAL_PROBE_IDS is not None:
        from openpilot.sunnypilot.egpu import probe as probe_mod
        probe_mod.probe_ids = _REAL_PROBE_IDS


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


def _raises(fn) -> str | None:
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    return None


def test_marker_target(tmp: Path):
    """The marker has to say which gfx target the pickle was built for, not just whose card.

    Vendor alone cannot separate a gfx12 bundle from a gfx1032 one, and both are 0x1002. The
    format therefore grew a `target=` field -- and it had to grow it without orphaning the
    vendor-only markers already sitting next to models compiled before this change.
    """
    print("\n[marker] vendor AND gfx target, with the old vendor-only form still readable")
    pkl = tmp / "target.pkl"
    pkl.write_bytes(b"x")
    marker = Path(str(pkl) + models.MARKER_SUFFIX)

    models.write_marker(str(pkl), vendors.AMD, "gfx1032")
    case("a written target reads back", models.pkl_target(str(pkl)), "gfx1032")
    case("and the vendor still reads back", models.pkl_vendor(str(pkl)), "amd")
    case("read_marker returns both at once",
         (models.read_marker(str(pkl)).vendor, models.read_marker(str(pkl)).target),
         ("amd", "gfx1032"))

    models.write_marker(str(pkl), vendors.NVIDIA, "SM_86")
    case("a target is normalised to lower case", models.pkl_target(str(pkl)), "sm_86")

    models.write_marker(str(pkl), vendors.NVIDIA)
    case("omitting the target writes no target field", models.pkl_target(str(pkl)),
         models.UNKNOWN_TARGET)
    case("and the vendor is still recorded", models.pkl_vendor(str(pkl)), "nvidia")

    marker.write_text("nvidia\n", encoding="utf-8")
    case("an old vendor-only marker still names its vendor", models.pkl_vendor(str(pkl)), "nvidia")
    case("an old vendor-only marker reports an unknown target",
         models.pkl_target(str(pkl)), models.UNKNOWN_TARGET)

    marker.unlink()
    case("an absent marker reports an unknown target",
         models.pkl_target(str(pkl)), models.UNKNOWN_TARGET)

    for raw in ["vendor=amd\ntarget=gfx 1032\n", "vendor=amd\ntarget=../../etc\n",
                "vendor=amd\ntarget=\n"]:
        marker.write_text(raw, encoding="utf-8")
        case("an unreadable target degrades to unknown rather than matching " + repr(raw),
             models.pkl_target(str(pkl)), models.UNKNOWN_TARGET)
    marker.unlink()

    bad = None
    try:
        models.write_marker(str(pkl), vendors.AMD, "gfx 1032")
    except ValueError as e:
        bad = str(e)
    check("refusing to mark a model with an unreadable target", bad is not None)


def test_assert_pkl_target(tmp: Path):
    """A gfx12 pickle reaching a gfx1032 card is the gap this closes.

    tinygrad compiles to one ISA. Handing an AMD-marked gfx12 pickle to an RDNA2 card gets
    past the vendor gate -- both are 0x1002 -- and lands in `load_oob`, where the only check
    downstream is `np.all(np.isfinite(...))`. Wrong-but-finite numbers reach the planner.
    """
    print("\n[assert_pkl_matches] the gfx target is checked too, whenever both sides know it")
    pkl = tmp / "gfx12.pkl"
    pkl.write_bytes(b"x")
    models.write_marker(str(pkl), vendors.AMD, "gfx1201")

    raised = _raises(lambda: models.assert_pkl_matches(str(pkl), True, "amd", "gfx1032"))
    check("a gfx12 pickle on a gfx1032 card raises", raised is not None)
    if raised:
        check("the message names both targets", "gfx1201" in raised and "gfx1032" in raised)
        check("the target message names the file", str(pkl) in raised)

    check("the same target passes",
          _raises(lambda: models.assert_pkl_matches(str(pkl), True, "amd", "gfx1201")) is None)

    check("a vendor mismatch is still rejected, target or no target",
          _raises(lambda: models.assert_pkl_matches(str(pkl), True, "nvidia", "gfx1201")) is not None)

    check("not routing through the eGPU still skips the whole check",
          _raises(lambda: models.assert_pkl_matches(str(pkl), False, "amd", "gfx1032")) is None)

    # An unknown target passes on purpose, and the reason is asymmetric knowledge: every
    # published bundle arrives with no marker at all, and asics.py only knows the gfx string
    # of cards it has an entry for. Failing closed here would refuse every AMD user's
    # catalog model -- a regression caused purely by this code existing.
    check("a device whose target we do not know accepts a targeted pickle",
          _raises(lambda: models.assert_pkl_matches(str(pkl), True, "amd")) is None)
    check("and None means the same as unknown",
          _raises(lambda: models.assert_pkl_matches(str(pkl), True, "amd", None)) is None)

    legacy = tmp / "legacy.pkl"
    legacy.write_bytes(b"x")
    Path(str(legacy) + models.MARKER_SUFFIX).write_text("amd\n", encoding="utf-8")
    check("an old vendor-only marker passes a targeted device",
          _raises(lambda: models.assert_pkl_matches(str(legacy), True, "amd", "gfx1032")) is None)
    check("but its vendor is still enforced",
          _raises(lambda: models.assert_pkl_matches(str(legacy), True, "nvidia", "gfx1032")) is not None)

    unmarked = tmp / "catalog.pkl"
    unmarked.write_bytes(b"x")
    check("an unmarked catalog bundle is not refused by the target check",
          _raises(lambda: models.assert_pkl_matches(str(unmarked), True, "amd", "gfx1032")) is None)


# --- which ASIC, not just which vendor ------------------------------------------------------

def test_asic_table():
    """The table is a blocklist. An id it does not know must change nothing.

    tinygrad's driverless AM driver is RDNA3/RDNA4 only -- ops_amd.py asserts
    `target[0] in (11, 12)` (plus two CDNA targets) and raises "Unsupported arch" for anything
    else. RDNA2 is gfx10.3, so a 6600 XT fails that assert after modeld has already committed
    to the eGPU. We cannot enumerate every card AMD will ever ship, so we enumerate the ones we
    have positive evidence AM refuses, and leave every other card on exactly today's path.
    """
    print("\n[asics] a blocklist of cards AM refuses, never an allowlist")
    spec = asics.asic_for(vendors.AMD, 0x73FF)
    check("the RX 6600 XT is in the table", spec is not None)
    if spec is not None:
        case("and it is Navi 23 / gfx1032", (spec.gfx, spec.arch), ("gfx1032", "rdna2"))
        case("and AM cannot drive it", spec.am_supported, False)

    case("an AMD id we do not know yields no opinion", asics.asic_for(vendors.AMD, 0x7550), None)
    case("a device id under the wrong vendor is not matched",
         asics.asic_for(vendors.NVIDIA, 0x73FF), None)
    case("no device id at all yields no opinion", asics.asic_for(vendors.AMD, None), None)

    check("every entry in the table is marked unsupported",
          all(not s.am_supported for s in asics.UNSUPPORTED_AMD.values()))
    check("every entry names a gfx10.3x target",
          all(s.gfx.startswith("gfx103") for s in asics.UNSUPPORTED_AMD.values()))
    check("the whole Navi 2x line is covered, not just the one card we own",
          {asics.asic_for(vendors.AMD, i).arch for i in (0x73BF, 0x73DF, 0x73FF, 0x743F)} == {"rdna2"})

    check("am_supports says yes to anything unknown", asics.am_supports(vendors.AMD, 0x7550))
    check("am_supports says yes when nothing was detected", asics.am_supports(vendors.AMD, None))
    check("am_supports says no to the 6600 XT", not asics.am_supports(vendors.AMD, 0x73FF))


def test_resolve_device():
    print("\n[resolve_device] explicit > cached > nothing known")
    case("an explicit override is used", detect.resolve_device(FakeParams(EgpuDevice="0x73ff")), 0x73FF)
    case("the 0x prefix is optional", detect.resolve_device(FakeParams(EgpuDevice="73ff")), 0x73FF)
    case("case does not matter", detect.resolve_device(FakeParams(EgpuDevice="0x73FF")), 0x73FF)
    case("a cached probe result is used",
         detect.resolve_device(FakeParams(EgpuDeviceDetected="0x73ff")), 0x73FF)
    case("explicit beats a stale cache",
         detect.resolve_device(FakeParams(EgpuDevice="0x743f", EgpuDeviceDetected="0x73ff")), 0x743F)
    case("nothing known is None, not a guess", detect.resolve_device(FakeParams()), None)
    case("a garbage id is ignored, not trusted",
         detect.resolve_device(FakeParams(EgpuDevice="banana")), None)
    case("an out-of-range id is ignored", detect.resolve_device(FakeParams(EgpuDevice="0x1ffff")), None)
    case("bytes decode like every other param", detect.resolve_device(FakeParams(EgpuDevice=b"0x73ff")), 0x73FF)


def test_device_id_parsing_is_strict():
    """A PCI device ID is 16 bits unsigned. Nothing else is an ID.

    int(x, 16) happily accepts a sign, so "+3ff" would otherwise parse as 1023 -- a real
    device ID that is not the one written down.
    """
    print("\n[device id] only a 16-bit unsigned hex value is an id")
    for raw in ("-1", "+3ff", "-73f", " ", "0x", "zzzz", "1ffff", ""):
        case("rejects " + repr(raw), detect._parse_device_id(raw), None)
    for raw, want in (("73ff", 0x73FF), ("0x73ff", 0x73FF), ("0", 0), ("ffff", 0xFFFF)):
        case("accepts " + repr(raw), detect._parse_device_id(raw), want)


def test_probe_once(monkey):
    """probe_once must learn the card even when the vendor is not in question.

    resolve() short-circuits on an explicit EgpuVendor, so hanging the probe off it meant a
    user who picked "amd" in the panel -- a perfectly ordinary thing to do -- never got a
    device ID, and the whole RDNA2 gate silently did not apply to them.
    """
    print("\n[probe_once] the dock is identified regardless of vendor")
    calls = []

    def fake_probe_ids():
        calls.append(1)
        return ("amd", 0x73FF)

    monkey(True, fake_probe_ids)

    detect._probe_attempted = False
    params = FakeParams(EgpuVendor="amd")          # vendor pinned: resolve() never probes
    detect.probe_once(params)
    case("probes even with the vendor explicitly set", len(calls), 1)
    case("and caches the device id", params.get("EgpuDeviceDetected"), "0x73ff")
    case("so the gate now applies to a pinned-vendor user", detect.enabled(params), False)

    detect.probe_once(params)
    case("but only once per manager start", len(calls), 1)

    calls.clear()
    detect._probe_attempted = False
    already = FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x7550")
    detect.probe_once(already)
    case("a known card is not re-probed", len(calls), 0)

    calls.clear()
    detect._probe_attempted = False
    monkey(False, fake_probe_ids)
    detect.probe_once(FakeParams())
    case("no dock means no probe", len(calls), 0)
    case("and no latch, so a dock plugged in later is still found",
         detect._probe_attempted, False)

    monkey(True, lambda: None)
    detect._probe_attempted = False
    unknown = FakeParams()
    detect.probe_once(unknown)
    case("a probe that reads nothing caches nothing",
         unknown.get("EgpuDeviceDetected"), None)
    detect._probe_attempted = False
    unmonkey_probe()


def test_asic_gate():
    """A card we cannot drive must look like no eGPU, not like a broken one.

    Without this, a 6600 XT resolves as plain "amd", takes DEV=USB+AMD:LLVM, is served the
    gfx12-compiled _USBGPU bundles, fails to open the device, and modeld restarts forever --
    the car cannot engage until the dock is unplugged.
    """
    print("\n[enabled] an eGPU AM cannot drive looks like no eGPU")
    case("an RDNA2 card does not get the model",
         detect.enabled(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x73ff")), False)
    case("and it gets no AMD catalog to download",
         detect.uses_amd_catalog(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x73ff")), False)
    case("a supported AMD card is untouched",
         detect.enabled(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x7550")), True)
    case("an unknown device id keeps today's behaviour exactly",
         detect.enabled(FakeParams(EgpuVendor="amd")), True)
    case("a garbage device id keeps today's behaviour exactly",
         detect.enabled(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="banana")), True)
    case("an explicit device override beats the cached probe",
         detect.enabled(FakeParams(EgpuVendor="amd", EgpuDevice="0x73ff", EgpuDeviceDetected="0x7550")),
         False)
    case("an assumed vendor with an RDNA2 id still refuses",
         detect.enabled(FakeParams(EgpuDeviceDetected="0x73ff")), False)


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

    print("\n[egpu_state] a card AM cannot drive is named and explained")
    unsupported = state.idle_reason(True, "amd", False, False, False,
                                    "Navi 23 [Radeon RX 6600/6600 XT/6600M]", False)
    check("an unsupported card is explained, not silently ignored", unsupported is not None)
    if unsupported:
        check("and it is named", "Navi 23" in unsupported)
        check("and the reason given is the driver, not the dock", "RDNA3 and RDNA4" in unsupported)
        check("and it says the car still drives", "runs on the device" in unsupported)
    case("an unsupported card outranks the vendor check, which would have said nothing",
         state.idle_reason(True, "amd", False, False, False, "Navi 23", False) is None, False)
    case("no dock still outranks an unsupported card",
         state.idle_reason(False, "amd", False, False, False, "Navi 23", False).startswith("No chestnut"),
         True)
    case("a supported card explains nothing",
         state.idle_reason(True, "amd", False, False, False, None, True), None)

    rows = dict(state.status_rows(True, "amd", True, False, False, False, "Navi 23"))
    case("an identified card is named in the gpu row", rows.get("gpu"), "Navi 23")
    case("and the model row says it is not being used", rows.get("driving model"), "on device")
    rows = dict(state.status_rows(True, "amd", True, False, False, True))
    case("without an identified card the vendor label is used as before",
         rows.get("gpu"), "AMD (assumed)")

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


def test_probe_tool():
    """The bring-up probe is run by hand on a bench, so nothing else would catch a typo.

    It may import tinygrad -- it is a hardware tool, not a test -- but only lazily, so that
    it can be read and checked on a machine that has neither tinygrad nor a dock.
    """
    print("\n[probe] the RDNA2 bring-up tool is intact")
    tool = REPO / ".elantra/probe_rdna2.py"
    check("the probe tool exists", tool.is_file())
    if not tool.is_file():
        return
    source = tool.read_text(encoding="utf-8")
    compiled = None
    try:
        compiled = compile(source, str(tool), "exec")
    except SyntaxError as e:
        check("the probe tool compiles", False, str(e))
    check("the probe tool compiles", compiled is not None)

    tree = ast.parse(source)
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module.split(".")[0])
    case("the probe tool imports tinygrad lazily", sorted({"tinygrad"} & set(top)), [])

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for stage in ("stage1_usb_speed", "stage2_bridge", "stage3_identity", "stage4_flr",
                  "stage5_discovery", "stage6_psp", "main"):
        check("the probe defines " + stage, stage in defined)

    # speed_verdict is the one piece of probe logic that is pure, and the one that was
    # already wrong once: the first version surveyed every USB device and stopped if any
    # read 480, which on a comma four is always true because the modem is USB 2. It fired
    # on real hardware whose own output listed three devices at 5000.
    probe = load_by_path(tool, "probe_rdna2")
    case("no dock: proceed and let stage 2 say so", probe.speed_verdict(None)[:2], (True, "info"))
    case("dock at 10 Gb/s: proceed", probe.speed_verdict("10000")[:2], (True, "ok"))
    case("dock at 480: stop", probe.speed_verdict("480")[:2], (False, "bad"))
    case("a 5000 Mb/s port is NOT the USB-2 fallback",
         probe.speed_verdict("5000")[:2], (True, "info"))
    check("the 480 stop names the ASM2464PD fallback",
          "ASM2464PD" in probe.speed_verdict("480")[3])

    # The probe once printed "NO-GO" for a stage 6 it had never been able to run -- a wrong
    # answer to the only question it exists to ask. Three outcomes, and the third says so.
    check("stage 6 has three outcomes, not two",
          len({probe.GO, probe.NO_GO, probe.UNTESTED}) == 3)
    verdict_src = source[source.index("def main("):]
    check("an unrunnable stage 6 is not reported as a no-go", "NOT a no-go" in verdict_src)
    check("a real no-go says the register was actually polled", "was polled" in verdict_src)


# --- compiling for the card that is actually there -------------------------------------------

class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


class FakeRun:
    """Stands in for the compile subprocess, which needs a dock, a card and about an hour.

    Records what it was asked to run and produces the file a real compiler would, so main()
    can be exercised end to end: option stripping, the tinygrad environment, and -- the point
    of stage 6 -- whether the marker beside the artifact names the gfx target.
    """

    def __init__(self, returncode: int = 0, produce: bool = True):
        self.returncode = returncode
        self.produce = produce
        self.cmd: list[str] = []
        self.env: dict[str, str] = {}

    def run(self, cmd, env=None):
        self.cmd = list(cmd)
        self.env = dict(env or {})
        if self.produce and self.returncode == 0:
            out = None
            for i, arg in enumerate(cmd):
                if arg == "--output":
                    out = cmd[i + 1]
                elif arg.startswith("--output="):
                    out = arg.split("=", 1)[1]
            if out is not None:
                Path(out).write_bytes(b"compiled")

        return _Completed(self.returncode)


def _with_fake_run(fake, fn):
    real = compile_egpu.subprocess
    compile_egpu.subprocess = fake
    try:
        return fn()
    finally:
        compile_egpu.subprocess = real


def test_compile_option_plumbing():
    """Our two options must never reach compile_modeld, which argparses and would abort."""
    print("\n[compile_egpu] our options are consumed, the compiler's are passed through")
    argv = ["--egpu-vendor", "nvidia", "--model-type", "supercombo",
            "--egpu-target=gfx1032", "--output", "/tmp/x.pkl"]
    rest, vendor = compile_egpu.take_option(argv, compile_egpu.VENDOR_FLAG)
    case("--egpu-vendor is read", vendor, "nvidia")
    rest, target = compile_egpu.take_option(rest, compile_egpu.TARGET_FLAG)
    case("--egpu-target=... is read", target, "gfx1032")
    case("and nothing of ours is left for the compiler", rest,
         ["--model-type", "supercombo", "--output", "/tmp/x.pkl"])

    case("a flag that is not there reads as empty",
         compile_egpu.take_option(["--model-type", "supercombo"], compile_egpu.TARGET_FLAG),
         (["--model-type", "supercombo"], ""))
    case("--output is found spaced",
         compile_egpu.option_value(["--output", "/a/b.pkl"], "--output"), "/a/b.pkl")
    case("--output is found joined",
         compile_egpu.option_value(["--output=/a/b.pkl"], "--output"), "/a/b.pkl")
    case("no --output reads as None",
         compile_egpu.option_value(["--model-type", "supercombo"], "--output"), None)


def test_compile_env_and_target():
    print("\n[compile_egpu] the environment and the target come from the vendor descriptor")
    amd = compile_egpu.compile_env(vendors.AMD_SPEC, "QCOM", base={})
    case("the AMD device string is what SConscript uses", amd.get("DEV"), "USB+AMD:LLVM")
    case("the AMD env still carries GMMU=0", amd.get("GMMU"), "0")
    case("WARP_DEV is passed through", amd.get("WARP_DEV"), "QCOM")
    case("every AMD compile flag reaches the environment",
         sorted(set(vendors.compile_flags(vendors.AMD_SPEC, "QCOM").split())
                - {k + "=" + v for k, v in amd.items()}), [])

    nv = compile_egpu.compile_env(vendors.NV_SPEC, "QCOM", base={})
    case("the NV device string carries no renderer suffix", nv.get("DEV"), "USB+NV")
    check("GMMU is not invented for NV, where it has no meaning", "GMMU" not in nv)

    case("the environment we inherit is not thrown away",
         compile_egpu.compile_env(vendors.NV_SPEC, "QCOM", base={"HOME": "/data"}).get("HOME"),
         "/data")

    case("the 6600 XT's target comes from its PCI id",
         compile_egpu.resolve_target(vendors.AMD, params=FakeParams(EgpuDevice="73ff")),
         "gfx1032")
    case("a card asics.py has no entry for has no known target",
         compile_egpu.resolve_target(vendors.AMD, params=FakeParams(EgpuDevice="1234")),
         models.UNKNOWN_TARGET)
    case("NVIDIA has no gfx table, so no target unless told",
         compile_egpu.resolve_target(vendors.NVIDIA, params=FakeParams(EgpuDevice="73ff")),
         models.UNKNOWN_TARGET)
    case("an explicit target wins over the table",
         compile_egpu.resolve_target(vendors.AMD, "sm_86", FakeParams(EgpuDevice="73ff")),
         "sm_86")

    case("NVIDIA compiles to where detect.py looks for it",
         compile_egpu.default_output(vendors.NVIDIA), detect.NV_MODEL_PATH)
    case("AMD has no invented default path", compile_egpu.default_output(vendors.AMD), None)


def test_compile_refuses_to_guess():
    """An assumed vendor is a guess, and this one costs hours and produces a mismarked pkl."""
    print("\n[compile_egpu] an unconfirmed card is refused, not guessed at")
    case("a confirmed vendor is used",
         compile_egpu.resolve_vendor("", FakeParams(EgpuVendor="nvidia")), "nvidia")
    case("a probed vendor is confirmed too",
         compile_egpu.resolve_vendor("", FakeParams(EgpuVendorDetected="amd")), "amd")
    case("the flag overrides everything",
         compile_egpu.resolve_vendor("nvidia", FakeParams(EgpuVendor="amd")), "nvidia")

    raised = None
    try:
        compile_egpu.resolve_vendor("", FakeParams())
    except SystemExit as e:
        raised = str(e)
    check("an assumed vendor refuses to compile", raised is not None)
    if raised:
        check("and says which flag to pass", compile_egpu.VENDOR_FLAG in raised)


def test_compile_writes_a_targeted_marker(tmp: Path):
    """The whole point of stage 6: the artifact carries the ISA it was built for."""
    print("\n[compile_egpu] the marker it writes names the target, not just the vendor")
    params = FakeParams(EgpuVendor="amd", EgpuDevice="73ff")
    out = tmp / "built.pkl"
    fake = FakeRun()
    rc = _with_fake_run(fake, lambda: compile_egpu.main(
        ["--model-type", "supercombo", "--output", str(out)], params))

    case("a successful compile exits zero", rc, 0)
    case("the marker records the vendor", models.pkl_vendor(str(out)), "amd")
    case("the marker records the gfx target", models.pkl_target(str(out)), "gfx1032")
    check("the compiler was invoked as a module", compile_egpu.COMPILER in fake.cmd)
    check("no eGPU option leaked into the compiler's argv",
          not any(a.startswith("--egpu-") for a in fake.cmd))
    case("the compiler ran against the AMD device", fake.env.get("DEV"), "USB+AMD:LLVM")

    # And the model it just wrote must survive the gate it exists to satisfy.
    ok = True
    try:
        models.assert_pkl_matches(str(out), True, "amd", "gfx1032")
    except RuntimeError:
        ok = False
    check("the model it produced passes assert_pkl_matches on that card", ok)
    check("and is refused on a gfx12 card",
          _raises(lambda: models.assert_pkl_matches(str(out), True, "amd", "gfx1201")) is not None)

    failed = tmp / "failed.pkl"
    rc = _with_fake_run(FakeRun(returncode=3), lambda: compile_egpu.main(
        ["--model-type", "supercombo", "--output", str(failed)], params))
    case("a failed compile propagates its exit code", rc, 3)
    check("and writes no marker", not Path(str(failed) + models.MARKER_SUFFIX).exists())

    missing = tmp / "missing.pkl"
    rc = _with_fake_run(FakeRun(produce=False), lambda: compile_egpu.main(
        ["--model-type", "supercombo", "--output", str(missing)], params))
    check("a compiler that produces no file is not marked as success", rc != 0)
    check("and writes no marker for a file that is not there",
          not Path(str(missing) + models.MARKER_SUFFIX).exists())

    rc = _with_fake_run(FakeRun(), lambda: compile_egpu.main(["--model-type", "supercombo"], params))
    check("AMD without --output is refused rather than guessed", rc != 0)


def test_compile_nv_is_a_shim():
    """compile_nv.py is still an entry point, but must not be a second copy of the logic."""
    print("\n[compile_nv] the old entry point delegates instead of duplicating")
    tree = ast.parse((EGPU_DIR / "compile_nv.py").read_text(encoding="utf-8"))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module.split(".")[0])
    check("it still exposes a main", _def_node(tree, "main") is not None)
    check("it does not spawn the compiler itself", "subprocess" not in top)
    source = (EGPU_DIR / "compile_nv.py").read_text(encoding="utf-8")
    check("it delegates to compile_egpu", "compile_egpu.main" in source)
    check("it does not write a marker of its own", "write_marker" not in source)

    case("it pins the vendor to nvidia, ahead of whatever the caller passed",
         _compile_nv_forwards(),
         [compile_egpu.VENDOR_FLAG, vendors.NVIDIA, "--model-type", "supercombo"])


def _compile_nv_forwards() -> list[str]:
    """The argv compile_nv.main hands to compile_egpu.main, with nothing else run."""
    from openpilot.sunnypilot.egpu import compile_nv
    seen: list[str] = []
    real = compile_egpu.main
    compile_egpu.main = lambda argv, params=None: (seen.extend(argv) or 0)
    try:
        compile_nv.main(["--model-type", "supercombo"])
    finally:
        compile_egpu.main = real
    return seen


# --- an eGPU we cannot drive must not take the car with it -----------------------------------

def _def_node(tree, name: str):
    """The FunctionDef called `name`, anywhere in the tree, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _names_called(node) -> set[str]:
    """Every bare or attribute call target under `node`, by last component."""
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            fn = child.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


class ExplodingParams:
    """Params that cannot be read at all -- a clean tree, where libparams_c.so is not built."""

    def get(self, key):
        raise OSError("libparams_c.so: cannot open shared object file")

    def get_bool(self, key):
        raise OSError("libparams_c.so: cannot open shared object file")


def test_build_gate():
    """SCons must not attempt an eGPU compile for a card tinygrad's AM driver refuses.

    The failure this prevents is not "no big model". do_compile returned env.Execute's
    non-zero result, which fails the SCons target, which makes build.py open a blocking
    TextWindow and exit(1) -- the device sits on an error screen needing the touchscreen.
    """
    print("\n[build gate] an unsupported card cannot fail the build")
    from openpilot.sunnypilot.egpu import guard

    case("an RDNA2 card is not built for",
         guard.egpu_build_ok(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x73ff")), False)
    case("a supported AMD card is built for",
         guard.egpu_build_ok(FakeParams(EgpuVendor="amd", EgpuDeviceDetected="0x744c")), True)
    case("an unknown card is still built for",
         guard.egpu_build_ok(FakeParams(EgpuVendor="amd")), True)
    # SCons reads this before openpilot is built, so Params may not load at all. Not knowing
    # must read as yes: the skip-on-failure below is the backstop, not this.
    case("unreadable params do not fail the build",
         guard.egpu_build_ok(ExplodingParams()), True)

    src = SCONSCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Primary gate: don't even declare the eGPU target for a card we cannot compile for.
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "USBGPU" for t in n.targets)]
    check("the SConscript assigns USBGPU exactly once", len(assigns) == 1)
    if len(assigns) == 1:
        expr = ast.get_source_segment(src, assigns[0].value) or ""
        check("the build's USBGPU consults the eGPU gate", "egpu_build_ok" in expr, expr)

    # Backstop: whatever slips through, a failed eGPU compile is a skip, not a build failure.
    do_compile = _def_node(tree, "do_compile")
    check("the SConscript still has a do_compile", do_compile is not None)
    if do_compile is not None:
        # Every exit is a skip. A `return ret` here is what bricked the boot.
        check("do_compile can only skip, never fail the build",
              all(n.value is None for n in ast.walk(do_compile) if isinstance(n, ast.Return)))


def test_stock_runner_gate():
    """The upstream runner is the default one, and it had no ASIC check at all.

    get_active_model_runner() returns `stock` whenever no bundle is active, so a fresh device
    with an RDNA2 card in the dock ran the *unguarded* path while asics.py sat unconsulted.
    """
    print("\n[stock runner] the default modeld consults the same gate")
    modeld = REPO / "openpilot/selfdrive/modeld/modeld.py"
    check("upstream modeld.py exists", modeld.is_file())
    if not modeld.is_file():
        return
    src = modeld.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "USBGPU" for t in n.targets)]
    check("modeld.py assigns USBGPU exactly once", len(assigns) == 1)
    if len(assigns) == 1:
        expr = ast.get_source_segment(src, assigns[0].value) or ""
        check("the stock runner's USBGPU consults the eGPU gate", "egpu" in expr.lower(), expr)


def test_loading_flag_is_released():
    """UsbGpuLoading must be false on every path out of the model load, raise included.

    It is CLEAR_ON_MANAGER_START, and a modeld crash restarts modeld, not manager. Left True
    it is a NO_ENTRY every frame *and* sets selfdrived's big_model_settling, which suppresses
    commIssue and posenetInvalid -- the car cannot engage and is not told why.
    """
    print("\n[loading flag] a failed load does not latch the car out of engagement")
    from openpilot.sunnypilot.egpu import guard

    p = FakeParams(UsbGpuActive=True)
    with guard.loading(p, True):
        case("the flag is held during the load", p.get("UsbGpuLoading"), True)
        case("any stale UsbGpuActive is cleared first", p.get("UsbGpuActive"), None)
    case("the flag is released on a clean load", p.get("UsbGpuLoading"), False)

    p = FakeParams()
    raised = False
    try:
        with guard.loading(p, True):
            raise RuntimeError("eGPU model load failed or timed out (60s)")
    except RuntimeError:
        raised = True
    check("the load failure still propagates", raised)
    case("the flag is released when the load raises", p.get("UsbGpuLoading"), False)

    p = FakeParams()
    with guard.loading(p, False):
        pass
    case("without an eGPU the flag is never set", p.get("UsbGpuLoading"), False)


def test_sunnypilot_runner_degrades():
    """modeld_v2 had no small-model fallback at all -- upstream's has one on both paths."""
    print("\n[sunnypilot runner] an eGPU fault degrades instead of killing modeld")
    v2 = REPO / "openpilot/sunnypilot/modeld_v2/modeld.py"
    check("modeld_v2 exists", v2.is_file())
    if not v2.is_file():
        return
    src = v2.read_text(encoding="utf-8")
    tree = ast.parse(src)

    main_fn = _def_node(tree, "main")
    check("modeld_v2 has a main", main_fn is not None)
    if main_fn is not None:
        check("modeld_v2 holds the loading flag through the guard",
              "loading" in _names_called(main_fn))

    # model.run() was unguarded: the only handler was the top-level re-raise, so an eGPU
    # fault mid-drive killed the process rather than falling back.
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "run" and node.handlers):
                guarded = True
    check("model.run() is inside a try with a handler", guarded)
    check("the handler names the fallback", "fall back to small" in src)


def main() -> int:
    print("eGPU vendor logic")
    with tempfile.TemporaryDirectory(prefix="egpu-test-") as raw_tmp:
        tmp = Path(raw_tmp)
        test_configured()
        test_resolve()
        test_enabled(tmp)
        test_apply_env_is_total()
        test_catalog()
        test_asic_table()
        test_resolve_device()
        test_device_id_parsing_is_strict()
        test_asic_gate()
        test_probe_once(monkey_probe)
        test_pkl_vendor(tmp)
        test_assert_pkl(tmp)
        test_marker_target(tmp)
        test_assert_pkl_target(tmp)
        test_descriptor_drift()
        test_chestnut_state_fields()
        test_panel_logic()
        test_import_purity()
        test_probe_tool()
        test_compile_option_plumbing()
        test_compile_env_and_target()
        test_compile_refuses_to_guess()
        test_compile_writes_a_targeted_marker(tmp)
        test_compile_nv_is_a_shim()
        test_build_gate()
        test_stock_runner_gate()
        test_loading_flag_is_released()
        test_sunnypilot_runner_degrades()

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
