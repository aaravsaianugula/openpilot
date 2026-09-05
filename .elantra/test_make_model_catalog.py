#!/usr/bin/env python3
"""
Tests for make_model_catalog.py -- it turns a compiled chunk set into the catalog entry that
puts the model in sunnypilot's picker.

Why this test exists. Everything this generator writes is a promise the download path checks
later, on the car, over a slow link. `_process_artifact` verifies EVERY chunk's sha256 and
raises "Hash validation failed for chunk N" if one disagrees; a wrong count silently changes the
`.chunkNNofMM` names, so every URL 404s. None of that is visible when the catalog is written --
only when a driver taps the model.

So the generator must never emit a promise it has not checked:

  * the chunk count comes from the manifest AND from the files on disk, and disagreement is
    fatal. Trusting either alone is how you publish a catalog for 17 chunks against 16 files;
  * a missing chunk aborts. A catalog listing an artifact that is not there is exactly the kind
    of plausible-looking placeholder that only fails on the car;
  * `minimum_selector_version` is READ from helpers.py, not typed in. The compat gate is exact
    equality, so drifting off it makes the bundle vanish from the picker with no error at all.

    python .elantra/test_make_model_catalog.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "make_model_catalog.py"


def load_module():
    spec = importlib.util.spec_from_file_location("make_model_catalog", MODULE)
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


NAME = "driving_cinqueterre_tinygrad.pkl"
BODIES = [b"chunk-one-bytes", b"chunk-two-bytes", b"three"]


def make_chunks(d: Path, n=3, name=NAME, manifest=None, drop=None):
    for i, body in enumerate(BODIES[:n]):
        if drop is not None and i == drop:
            continue
        (d / (name + f".chunk{i + 1:02d}of{n:02d}")).write_bytes(body)
    (d / (name + ".chunkmanifest")).write_text(str(n if manifest is None else manifest),
                                               encoding="utf-8")
    return d


def fake_helpers(d: Path, version=19) -> Path:
    p = d / "helpers.py"
    p.write_text("REQUIRED_JSON_VERSION = " + str(version) + "\n", encoding="utf-8")
    return p


def main() -> int:
    m = load_module()
    failures: list = []
    print("make_model_catalog")

    base = "https://example.test/rel/"

    def builds_a_correct_entry():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d)
            doc = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base,
                                  short_name="CT1", display_name="Cinque Terre",
                                  helpers_path=fake_helpers(d))
            b = doc["bundles"][0]
            assert b["short_name"] == "CT1"
            assert b["is_big"] is True, "chestnut sources are matched on is_big"
            # sunnypilot emits this as a string; _parse_bundle int()s it. Match their shape.
            assert b["minimum_selector_version"] == "19", b["minimum_selector_version"]
            art = b["models"][0]["artifact"]
            assert art["file_name"] == NAME, art["file_name"]
            assert art["download_uri"]["url"] == base + NAME, art["download_uri"]["url"]
            names = [c["file_name"] for c in art["chunks"]]
            assert names == [NAME + ".chunk01of03", NAME + ".chunk02of03",
                             NAME + ".chunk03of03"], names
            got = [c["sha256"] for c in art["chunks"]]
            want = [hashlib.sha256(x).hexdigest() for x in BODIES]
            assert got == want, "per-chunk hashes wrong"
            whole = hashlib.sha256(b"".join(BODIES)).hexdigest()
            assert art["download_uri"]["sha256"] == whole, "whole-artifact hash wrong"
    case("emits correct names, order and hashes", failures, builds_a_correct_entry)

    def chunk_names_match_what_the_downloader_derives():
        # manager.py builds each URL with get_chunk_name(base, i, len(chunks)). If our emitted
        # names disagree by even the zero padding, every chunk 404s on the car.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d)
            doc = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base,
                                  short_name="CT1", display_name="Cinque Terre",
                                  helpers_path=fake_helpers(d))
            chunks = doc["bundles"][0]["models"][0]["artifact"]["chunks"]
            n = len(chunks)
            derived = [f"{NAME}.chunk{i + 1:02d}of{n:02d}" for i in range(n)]
            assert [c["file_name"] for c in chunks] == derived
    case("names match get_chunk_name's derivation exactly", failures,
         chunk_names_match_what_the_downloader_derives)

    def manifest_disagreeing_with_disk_is_fatal():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d, manifest=17)          # manifest says 17, three files on disk
            try:
                m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=fake_helpers(d))
            except (ValueError, FileNotFoundError):
                return
            raise AssertionError("published a count the files on disk do not support")
    case("manifest/disk disagreement aborts", failures, manifest_disagreeing_with_disk_is_fatal)

    def a_missing_chunk_is_fatal():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d, drop=1)               # chunk02of03 absent
            try:
                m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=fake_helpers(d))
            except (ValueError, FileNotFoundError):
                return
            raise AssertionError("published a catalog for an artifact with a hole in it")
    case("a missing chunk aborts", failures, a_missing_chunk_is_fatal)

    def no_chunks_at_all_is_fatal():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            try:
                m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=fake_helpers(d))
            except (ValueError, FileNotFoundError):
                return
            raise AssertionError("emitted a catalog with nothing behind it")
    case("nothing compiled yet aborts", failures, no_chunks_at_all_is_fatal)

    def selector_version_is_read_not_typed():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d)
            doc = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                  display_name="X", helpers_path=fake_helpers(d, version=23))
            got = doc["bundles"][0]["minimum_selector_version"]
            assert int(got) == 23, "hard-coded the selector version (" + str(got) + ")"
    case("selector version tracks helpers.py", failures, selector_version_is_read_not_typed)

    def ref_is_deterministic_and_content_bound():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d)
            h = fake_helpers(d)
            a = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=h)["bundles"][0]["ref"]
            b = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=h)["bundles"][0]["ref"]
            assert a == b, "ref is not deterministic"
            (d / (NAME + ".chunk03of03")).write_bytes(b"different")
            c = m.build_catalog(chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                                display_name="X", helpers_path=h)["bundles"][0]["ref"]
            assert c != a, "ref did not change when the artifact did"
    case("ref is deterministic and bound to the bytes", failures, ref_is_deterministic_and_content_bound)

    def survives_the_real_parser_shape():
        # The document must round-trip through json and carry every key _parse_bundle reads.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_chunks(d)
            doc = json.loads(json.dumps(m.build_catalog(
                chunks_dir=d, file_name=NAME, base_url=base, short_name="CT1",
                display_name="X", helpers_path=fake_helpers(d))))
            b = doc["bundles"][0]
            for key in ("index", "short_name", "display_name", "generation", "environment",
                        "runner", "is_20hz", "minimum_selector_version", "overrides", "models",
                        "ref"):
                assert key in b, "missing key _parse_bundle reads: " + key
            assert isinstance(b["index"], int), "index must be an int for _parse_bundle"
            assert int(b["minimum_selector_version"]) > 0, "selector version must int() cleanly"
            assert b["runner"] == "tinygrad", b["runner"]
            assert doc["bundles"][0]["models"][0]["type"] == "chunked"
    case("carries every key the bundle parser reads", failures, survives_the_real_parser_shape)

    print("\n" + "-" * 58)
    if failures:
        print("FAILED: " + str(len(failures)) + " case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: the generator only publishes promises it has verified against the bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
