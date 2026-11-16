#!/usr/bin/env python3
"""
Tests for openpilot/sunnypilot/models/elantra_catalog.py -- the merge that puts a model we host
into sunnypilot's picker.

Why this test exists. sunnypilot's catalog is the ONLY way a model reaches the Models screen:
there is no local source, no sideload path, and no custom-bundle param. Carrying a model they
have not published yet therefore means adding a bundle to the list they serve. That merge sits
in front of every reader -- the manager resolves downloads from it, the UI lists from it, and
`validate_active_bundles` clears the active bundle if it is ever absent from it -- so the ways
it can go wrong are all quiet:

  * it must FAIL OPEN. If our catalog is unreachable, malformed, or simply not published yet,
    sunnypilot's list has to come through untouched. A model of ours must never be able to empty
    the picker or wipe someone's selection;
  * it must dedupe by ref. When sunnypilot eventually publishes this same model, ours has to
    drop out rather than show the driver two identical entries pointing at different hosts;
  * it must not collide on index. sunnypilot's chestnut list is 0..10 today and grows; an entry
    reusing a live index reorders or hides one of theirs.

The pure merge is tested here with no network and no cereal, so it runs on a bare Python 3.

    python .elantra/test_elantra_catalog.py
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "openpilot/sunnypilot/models/elantra_catalog.py"


def load_module():
    spec = importlib.util.spec_from_file_location("elantra_catalog", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case(name: str, failures: list, fn) -> None:
    try:
        fn()
    except AssertionError as e:
        failures.append(name + ": " + str(e))
        print("  FAIL  " + name + ": " + str(e))
    # An unexpected exception type is still a failed case, not a crash of the harness.
    except Exception as e:
        failures.append(name + ": unexpected " + type(e).__name__ + ": " + str(e))
        print("  FAIL  " + name + ": unexpected " + type(e).__name__ + ": " + str(e))
    else:
        print("  ok    " + name)


def bundle(ref, short="X", index=0, big=True):
    return {"short_name": short, "display_name": short + " model", "ref": ref, "index": index,
            "is_big": big, "is_20hz": True, "environment": "release", "runner": "tinygrad",
            "generation": "12", "minimum_selector_version": 19, "overrides": {}, "models": []}


def sunnypilot_catalog():
    return {"tinygrad_ref": "e837e367aac9e1a66e689f4f32ce20ca9367df13",
            "bundles": [bundle("aaa", "BMV6", 10), bundle("bbb", "SM", 8)]}


def main() -> int:
    m = load_module()
    failures: list = []
    print("elantra_catalog.merge_extra_bundles")

    ours = bundle("ccc", "CT1", 0)

    def appends_our_bundle():
        base = sunnypilot_catalog()
        out = m.merge_extra_bundles(base, {"bundles": [copy.deepcopy(ours)]})
        refs = [b["ref"] for b in out["bundles"]]
        assert refs == ["aaa", "bbb", "ccc"], refs
        assert out["tinygrad_ref"] == base["tinygrad_ref"], "dropped a top-level key"
    case("appends our bundle and keeps the rest of the document", failures, appends_our_bundle)

    def reindexes_above_theirs():
        base = sunnypilot_catalog()          # max index 10
        out = m.merge_extra_bundles(base, {"bundles": [copy.deepcopy(ours)]})
        got = [b for b in out["bundles"] if b["ref"] == "ccc"][0]["index"]
        assert got > 10, "index " + str(got) + " collides with sunnypilot's live range"
    case("reindexes our bundle above sunnypilot's highest", failures, reindexes_above_theirs)

    def dedupes_by_ref():
        base = sunnypilot_catalog()
        base["bundles"].append(bundle("ccc", "CT1-official", 11))
        out = m.merge_extra_bundles(base, {"bundles": [copy.deepcopy(ours)]})
        refs = [b["ref"] for b in out["bundles"]]
        assert refs.count("ccc") == 1, "duplicated a ref sunnypilot already publishes: " + str(refs)
        keep = [b for b in out["bundles"] if b["ref"] == "ccc"][0]
        assert keep["short_name"] == "CT1-official", "ours won over sunnypilot's published copy"
    case("drops ours once sunnypilot publishes the same ref", failures, dedupes_by_ref)

    def does_not_mutate_the_input():
        base = sunnypilot_catalog()
        before = copy.deepcopy(base)
        m.merge_extra_bundles(base, {"bundles": [copy.deepcopy(ours)]})
        assert base == before, "merge mutated the caller's document"
    case("does not mutate the fetched document", failures, does_not_mutate_the_input)

    # --- fail-open: every one of these must return sunnypilot's list untouched ---------------
    def fails_open():
        base = sunnypilot_catalog()
        expected = [b["ref"] for b in base["bundles"]]
        bad = {
            "not published yet (None)": None,
            "empty document": {},
            "bundles missing": {"tinygrad_ref": "x"},
            "bundles not a list": {"bundles": "nope"},
            "bundles is a dict": {"bundles": {"ref": "zzz"}},
            "entry not a dict": {"bundles": ["nope"]},
            "entry has no ref": {"bundles": [{"short_name": "NR"}]},
            "entry ref empty": {"bundles": [bundle("", "E")]},
            "document is a list": ["nope"],
            "document is a string": "nope",
        }
        for label, extra in bad.items():
            out = m.merge_extra_bundles(copy.deepcopy(base), extra)
            got = [b["ref"] for b in out.get("bundles", [])]
            assert got == expected, label + " changed the list: " + str(got)
    case("fails open on every malformed or absent catalog", failures, fails_open)

    def a_bad_entry_does_not_block_a_good_one():
        base = sunnypilot_catalog()
        out = m.merge_extra_bundles(base, {"bundles": ["junk", {"no": "ref"}, copy.deepcopy(ours)]})
        refs = [b["ref"] for b in out["bundles"]]
        assert refs == ["aaa", "bbb", "ccc"], refs
    case("skips junk entries but still adds the good one", failures, a_bad_entry_does_not_block_a_good_one)

    def base_document_garbage_is_survivable():
        # If sunnypilot's own document is unusable we must not raise -- the caller caches
        # whatever comes back and a raise here would take the whole fetch down with it.
        for base in (None, {}, {"bundles": "no"}, ["x"]):
            out = m.merge_extra_bundles(base, {"bundles": [copy.deepcopy(ours)]})
            assert isinstance(out, dict | list) or out is None, repr(out)
    case("a malformed sunnypilot document does not raise", failures, base_document_garbage_is_survivable)

    def url_is_declared_for_chestnut_only():
        assert "chestnut" in m.EXTRA_MODEL_URLS, "no chestnut source declared"
        assert "qcom" not in m.EXTRA_MODEL_URLS, \
            "the big model is chestnut-only; offering it in the qcom list would be undownloadable"
        url = m.EXTRA_MODEL_URLS["chestnut"]
        assert url.startswith("https://"), url
    case("declares a chestnut source only, over https", failures, url_is_declared_for_chestnut_only)

    print("\n" + "-" * 58)
    if failures:
        print("FAILED: " + str(len(failures)) + " case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: the catalog merge adds our model and can never break sunnypilot's list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
