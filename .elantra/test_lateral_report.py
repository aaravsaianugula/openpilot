#!/usr/bin/env python3
"""
Tests for the pure decoding and classification in lateral_report.py.

These are the parts that turn bytes into a number someone will act on, so they are worth
pinning. The EPS-fault case is here because it was wrong once: 0x251 arrives on three sources
on this car, and accepting every frame openpilot did not transmit pulled in 599 bus-1 frames
per segment with bit 14 set, against a true fault count of zero. That would have been reported
as "EPS faults at the raised torque ceiling" -- the exact signal the whole change is watched by.

    python .elantra/test_lateral_report.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "lateral_report.py"


def load():
    spec = importlib.util.spec_from_file_location("lateral_report", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lkas11(counts: int, act_toi: int = 0) -> bytes:
    """Pack CR_Lkas_StrToqReq (16|11@1+, offset -1024) and CF_Lkas_ActToi (27|1@1+)."""
    word = (((counts + 1024) & 0x7FF) << 16) | ((act_toi & 1) << 27)
    return word.to_bytes(4, "little") + bytes(4)


def mdps12(toi_flt: int) -> bytes:
    """Pack CF_Mdps_ToiFlt (14|1@1+)."""
    return ((toi_flt & 1) << 14).to_bytes(8, "little")


def main() -> int:
    m = load()
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        if cond:
            print("  ok    " + label)
        else:
            print("  FAIL  " + label)
            failures.append(label)

    print("lateral_report decoding")

    # --- LKAS11 torque, the number every conclusion rests on -------------------------
    ok = True
    for counts in (0, 1, -1, 384, -384, 409, -409, 1023, -1024):
        for toi in (0, 1):
            ok = ok and m.decode_lkas11(lkas11(counts, toi)) == (counts, toi)
    check("CR_Lkas_StrToqReq round-trips across the full signed range", ok)
    check("a short LKAS11 frame is rejected, not silently decoded as zero",
          m.decode_lkas11(b"\x00\x00") is None)
    # 384 counts is 3.00 Nm and 409 is 3.20 Nm on the OEM scale; if this drifts, every
    # torque figure quoted in values.py and the commit messages drifts with it.
    check("the OEM scale still puts 384 at 3.00 Nm and 409 at 3.20 Nm",
          abs(384 * 0.0078125 - 3.00) < 1e-9 and abs(409 * 0.0078125 - 3.1953125) < 1e-9)

    # --- the refusals: a check that has never been shown to fail is decoration ---------
    # Both of these guard a vacuous pass that really happened. summarise() used to compare
    # 0.0 == 0.0 on an empty dataset and print "estimators agree"; cmd_compare used to lose
    # a whole route silently when its segment 0 would not read. Each case below asserts the
    # refusal fires, and a positive control asserts it does NOT fire on good data -- a
    # refusal that always fires is as useless as one that never does.
    import contextlib
    import io
    import json as _json
    import tempfile

    def band_report(frames: int, pinned: int) -> dict:
        bands = {}
        for est in ("estimator_a", "estimator_b"):
            bands[est] = {b: m.new_band() for b in m.BAND_NAMES}
            first = m.BAND_NAMES[1]
            bands[est][first]["frames"] = frames
            bands[est][first]["pinned"]["384"] = pinned
        return {"bands": bands, "segments_total": 1, "segments_read": 1,
                "segments_skipped": []}

    def summarise_failures(report: dict) -> list:
        out: list = []
        with contextlib.redirect_stdout(io.StringIO()):
            m.summarise("t", [report], out)
        return out

    check("zero engaged frames is refused, not reported as agreement",
          len(summarise_failures(band_report(0, 0))) == 1)
    check("a real dataset with agreeing estimators still passes",
          summarise_failures(band_report(10000, 100)) == [])

    # 5% vs 0.1% is only 4.9 points apart, inside the 2pp-era absolute tolerance once the
    # rates are small -- the relative bound is what catches it.
    disagree = band_report(10000, 500)
    disagree["bands"]["estimator_b"][m.BAND_NAMES[1]]["pinned"]["384"] = 10
    check("estimators that disagree relatively are caught even when close in points",
          len(summarise_failures(disagree)) == 1)

    prov = {"git_branch": "b", "git_commit": "c", "safety_param": 12, "lat_accel_factor": 3.169,
            "friction": 0.0819, "params": {}, "nnlc_model": None, "dirty": False,
            "fingerprint": "HYUNDAI_ELANTRA_2024"}

    def compare_rc(with_marker: bool) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for name, tag in (("r1--aaaaaaaaaa", "before"), ("r2--bbbbbbbbbb", "after")):
                (out / (name + ".json")).write_text(
                    _json.dumps({**band_report(10000, 100), "tag": tag, "provenance": prov}),
                    encoding="utf-8")
            if with_marker:
                (out / m.UNREADABLE_FILE).write_text(_json.dumps({"badroute": "OSError"}),
                                                     encoding="utf-8")

            class Args:
                pass
            args = Args()
            args.out = str(out)
            args.before = "before"
            args.after = "after"
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = m.cmd_compare(args)
            return rc, buf.getvalue()

    rc_clean, text_clean = compare_rc(False)
    check("two clean sides compare successfully (positive control)", rc_clean == 0)
    check("the unreadable marker is not itself read as a route report",
          "_unreadable" not in text_clean)

    rc_bad, text_bad = compare_rc(True)
    check("an unreadable route makes compare REFUSE rather than quietly drop it",
          rc_bad == 1 and "badroute" in text_bad)

    # --- MDPS12 fault bit -------------------------------------------------------------
    check("CF_Mdps_ToiFlt reads bit 14", m.decode_toiflt(mdps12(1)) == 1)
    check("bit 14 clear reads as no fault", m.decode_toiflt(mdps12(0)) == 0)
    neighbours = all(m.decode_toiflt((1 << b).to_bytes(8, "little")) == 0 for b in (12, 13, 15))
    # 13 is CF_Mdps_ToiActive and is set on essentially every frame while openpilot steers.
    # Reading it as a fault would report a ~100% fault rate.
    check("neighbouring MDPS12 bits (ToiUnavail/ToiActive/FailStat) are not read as faults",
          neighbours)
    check("a short MDPS12 frame is rejected", m.decode_toiflt(b"\x00") is None)

    # --- every OTHER MDPS12 status bit ------------------------------------------------
    # Reading bit 14 alone is how this tool reported 3 fault frames in 1.59M engaged ones while
    # the two raised-ceiling routes carried 210 and 77 fault frames each. openpilot's own fault
    # condition is `ToiUnavail != 0 or ToiFlt != 0` (hyundai/carstate.py), and "the MDPS dropped
    # out rather than delivering" is most naturally ToiActive falling. All of it has to be here.
    for name, bit in (("def", 11), ("unavail", 12), ("active", 13), ("flt", 14), ("failstat", 15)):
        got = m.decode_mdps_status((1 << bit).to_bytes(8, "little"))
        only_this_one = got is not None and got[name] == 1 and sum(
            got[k] for k in ("def", "unavail", "active", "flt", "failstat")) == 1
        check(f"decode_mdps_status reads CF_Mdps_{name} at bit {bit} and nothing else",
              only_this_one)
    check("a short frame gives no status at all", m.decode_mdps_status(b"\x00") is None)

    # ToiActive is a STATUS bit, not a fault: it is set on essentially every steering frame, so
    # counting it would report a ~100% fault rate.
    check("ToiActive is not counted as a fault",
          m.eps_fault_bits(m.MDPS12, 0, (1 << 13).to_bytes(8, "little")) == frozenset())
    check("ToiUnavail counts as a fault -- openpilot faults on it too",
          m.eps_fault_bits(m.MDPS12, 0, (1 << 12).to_bytes(8, "little")) == frozenset({"unavail"}))
    check("FailStat counts as a fault",
          m.eps_fault_bits(m.MDPS12, 0, (1 << 15).to_bytes(8, "little")) == frozenset({"failstat"}))
    check("several fault bits at once are all reported",
          m.eps_fault_bits(m.MDPS12, 0, ((1 << 12) | (1 << 14)).to_bytes(8, "little"))
          == frozenset({"unavail", "flt"}))
    check("the bus filter applies to every fault bit, not just ToiFlt",
          m.eps_fault_bits(m.MDPS12, 1, ((1 << 12) | (1 << 14)).to_bytes(8, "little"))
          == frozenset())
    check("a non-MDPS address yields no fault bits",
          m.eps_fault_bits(m.LKAS11, 0, (1 << 14).to_bytes(8, "little")) == frozenset())

    # --- the bus filter, which is the part that was actually wrong --------------------
    check("a fault from the real MDPS on bus 0 counts",
          m.is_eps_fault_frame(m.MDPS12, 0, mdps12(1)))
    check("the same bit on bus 1 does NOT count (measured: 599 such frames per segment)",
          not m.is_eps_fault_frame(m.MDPS12, 1, mdps12(1)))
    check("openpilot's own forwarded copy (src 130) does not count",
          not m.is_eps_fault_frame(m.MDPS12, 130, mdps12(1)))
    check("a non-MDPS address on bus 0 does not count",
          not m.is_eps_fault_frame(m.LKAS11, 0, mdps12(1)))
    check("no fault bit means no fault", not m.is_eps_fault_frame(m.MDPS12, 0, mdps12(0)))

    # --- speed banding ----------------------------------------------------------------
    bands = [(0, "0-3"), (2.999, "0-3"), (3, "3-7"), (6.999, "3-7"), (7, "7-10"),
             (10, "10-14"), (13.999, "10-14"), (14, "14-18"), (18, "18+"), (60, "18+")]
    check("speed bands are contiguous and upper-exclusive",
          all(m.band_of(v) == b for v, b in bands))
    check("a negative speed lands in no band rather than the lowest one",
          m.band_of(-0.1) is None)

    # --- provenance -------------------------------------------------------------------
    base = {"git_commit": "abc", "safety_param": 12, "fingerprint": "HYUNDAI_ELANTRA_2024",
            "lat_accel_factor": 3.169, "friction": 0.0819, "nnlc_model": "HYUNDAI_ELANTRA_2021",
            "params": {"NeuralNetworkLateralControl": "0"}}
    check("the same configuration always tags the same", m.tag_of(base) == m.tag_of(dict(base)))
    for field, value in (("safety_param", 1036), ("friction", 0.15), ("git_commit", "def")):
        other = dict(base)
        other[field] = value
        check("a change of " + field + " changes the tag", m.tag_of(other) != m.tag_of(base))
    nnlc_on = dict(base)
    nnlc_on["params"] = {"NeuralNetworkLateralControl": "1"}
    # This is the one that matters most: a drive with NNLC on is not comparable to one with it
    # off, and compare() only refuses to mix them because the tag separates them.
    check("turning NNLC on changes the tag", m.tag_of(nnlc_on) != m.tag_of(base))


    # --- the scan loop itself, driven through a fake LogReader --------------------------
    # The decoders above are pure and easy to test; the ACCOUNTING is where the bug lived, and
    # it only runs inside scan_route's event loop. scan_route does its LogReader import inside
    # the function, so a fake module in sys.modules reaches it without touching the real one.
    import types

    class _Ev:
        def __init__(self, kind, **kw):
            self._kind = kind
            for k, v in kw.items():
                setattr(self, k, v)

        def which(self):
            return self._kind

    def _ns(**kw):
        return types.SimpleNamespace(**kw)

    def _cc(lat):
        return _Ev("carControl", carControl=_ns(
            latActive=lat, actuators=_ns(curvature=0.0, torque=0.0), currentCurvature=0.0))

    def _can(*frames):
        return _Ev("can", can=[_ns(address=a, src=s, dat=d) for a, s, d in frames])

    FLT = mdps12(1)
    UNAVAIL = (1 << 12).to_bytes(8, "little")

    events = [_Ev("carState", carState=_ns(vEgo=5.0)),
              _Ev("carOutput", carOutput=_ns(actuatorsOutput=_ns(torqueOutputCan=100)))]
    events += [_cc(True)] * 60                       # engaged, band 3-7
    events += [_can((m.MDPS12, 0, FLT))]             # a fault WHILE engaged
    events += [_cc(False)]                           # the disengagement
    events += [_can((m.MDPS12, 0, FLT))]             # ...and one immediately after it
    events += [_cc(False)] * 198                     # window still open (200 frames)
    events += [_can((m.MDPS12, 0, UNAVAIL))]         # ToiUnavail near the end of the window
    events += [_cc(False)] * 5                       # window now expired
    events += [_can((m.MDPS12, 0, FLT)),             # too late to be "at the disengagement"
               _can((m.MDPS12, 1, FLT))]             # and a bus-1 decoy that must never count

    fake = types.ModuleType("openpilot.tools.lib.logreader")
    fake.LogReader = lambda path, **kw: iter(events)
    for _name in ("openpilot", "openpilot.tools", "openpilot.tools.lib"):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    sys.modules["openpilot.tools.lib.logreader"] = fake

    rep = m.scan_route("test", ["seg0"])
    band = rep["bands"]["estimator_a"]["3-7"]

    check("a fault while steering lands in faults_engaged",
          band["faults_engaged"]["flt"] == 1)
    check("a fault just AFTER the disengagement is still counted -- the blind spot that hid " +
          "every real event on this car", band["faults_at_disengage"]["flt"] == 1)
    check("ToiUnavail is counted in its own right, not folded into ToiFlt",
          band["faults_at_disengage"]["unavail"] == 1 and band["faults"]["unavail"] == 1)
    check("a fault past the 2 s window counts overall but NOT at the disengagement",
          band["faults"]["flt"] == 3 and band["faults_at_disengage"]["flt"] == 1)
    check("toi_flt keeps its original meaning for continuity",
          band["toi_flt"] == 3 and band["toi_flt_engaged"] == 1)
    check("engaged frames are still counted normally", band["frames"] == 60)

    print("\n" + "-" * 58)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: decoding, bus filtering and provenance behave as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
