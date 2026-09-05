#!/usr/bin/env python3
"""
Tests for fetch_lfs_object.py -- the seeder for LFS objects our own store does not have.

Why this test exists. This tool exists to keep the device's updater alive. updated.py runs
`git checkout --force` with filter.lfs.required=true, so a pointer whose object cannot be
resolved does not degrade, it aborts the update. Every failure mode below therefore breaks the
car's ability to take a new build, and three of them do it quietly:

  * a pointer parsed loosely -- accepting a truncated or non-sha256 pointer seeds an object under
    the wrong name, which looks installed and resolves to nothing;
  * a download that is not verified end to end -- a truncated body still writes a plausible file,
    and git-lfs only notices later, on a checkout, far from here;
  * a non-atomic install -- a partial file at the final path is indistinguishable from a good one
    to every later have_object() check, so the damage looks repaired and is not.

The network is never touched: the HTTP layer is injected.

    python .elantra/test_fetch_lfs_object.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "fetch_lfs_object.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fetch_lfs_object", MODULE)
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


VERSION_LINE = "version https://git-lfs.github.com/spec/v1\n"


def pointer(oid: str, size: int) -> str:
    return VERSION_LINE + "oid sha256:" + oid + "\n" + "size " + str(size) + "\n"


def main() -> int:
    m = load_module()
    failures: list = []

    blob = b"cinque terre" * 977
    good_oid = hashlib.sha256(blob).hexdigest()
    good_size = len(blob)

    print("fetch_lfs_object")

    # --- pointer parsing --------------------------------------------------------------------
    def parses_a_real_pointer():
        oid, size = m.parse_pointer(pointer(good_oid, good_size))
        assert oid == good_oid, "oid " + str(oid)
        assert size == good_size, "size " + str(size)
    case("parses a well-formed pointer", failures, parses_a_real_pointer)

    def rejects_bad_pointers():
        bad = {
            "empty": "",
            "no version line": "oid sha256:" + good_oid + "\nsize " + str(good_size) + "\n",
            "not a pointer at all": "\x08\x20\n\x12\x07pytorch",
            "wrong hash algo": (VERSION_LINE + "oid sha1:" + ("a" * 40) + "\nsize 5\n"),
            "short oid": (VERSION_LINE + "oid sha256:deadbeef\nsize 5\n"),
            "non-numeric size": (VERSION_LINE + "oid sha256:" + good_oid + "\nsize huge\n"),
            "missing size": (VERSION_LINE + "oid sha256:" + good_oid + "\n"),
        }
        for label, text in bad.items():
            try:
                m.parse_pointer(text)
            except ValueError:
                continue
            raise AssertionError("accepted a bad pointer (" + label + ")")
    case("rejects every malformed pointer", failures, rejects_bad_pointers)

    # --- object store layout ----------------------------------------------------------------
    def builds_the_store_path():
        p = m.object_path(Path("/repo"), good_oid)
        assert p.parts[-4:] == ("objects", good_oid[0:2], good_oid[2:4], good_oid), "got " + str(p)
        assert ".git" in p.parts and "lfs" in p.parts, "got " + str(p)
    case("object path is .git/lfs/objects/aa/bb/<oid>", failures, builds_the_store_path)

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        (repo / ".git").mkdir(parents=True)

        def missing_object_is_missing():
            assert m.have_object(repo, good_oid, good_size) is False
        case("absent object reports missing", failures, missing_object_is_missing)

        def installs_and_verifies():
            m.install_object(repo, good_oid, good_size, io.BytesIO(blob))
            dest = m.object_path(repo, good_oid)
            assert dest.is_file(), "object was not written"
            assert dest.stat().st_size == good_size, "wrong size on disk"
            assert m.have_object(repo, good_oid, good_size) is True
        case("installs a good object and finds it again", failures, installs_and_verifies)

        def rejects_wrong_content():
            body = b"WRONG BYTES!!"
            other = hashlib.sha256(b"not the model").hexdigest()
            try:
                m.install_object(repo, other, len(body), io.BytesIO(body))
            except ValueError:
                pass
            else:
                raise AssertionError("accepted content whose sha256 is not the oid")
            assert not m.object_path(repo, other).exists(), \
                "left a corrupt object at the final path -- every later check would trust it"
        case("sha256 mismatch raises and leaves nothing behind", failures, rejects_wrong_content)

        def rejects_short_body():
            oid2 = hashlib.sha256(blob + b"tail").hexdigest()
            try:
                m.install_object(repo, oid2, len(blob) + 4, io.BytesIO(blob))
            except ValueError:
                pass
            else:
                raise AssertionError("accepted a truncated download")
            assert not m.object_path(repo, oid2).exists(), "left a truncated object behind"
        case("truncated body raises and leaves nothing behind", failures, rejects_short_body)

        def no_temp_files_survive():
            store = repo / ".git" / "lfs" / "objects"
            strays = [p.name for p in store.rglob("*") if p.is_file() and "tmp" in p.name.lower()]
            assert not strays, "temp files left in the object store: " + repr(strays)
        case("failed installs leave no temp files in the store", failures, no_temp_files_survive)

        def wrong_size_on_disk_is_not_have():
            stray = hashlib.sha256(b"stray").hexdigest()
            p = m.object_path(repo, stray)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stray")
            assert m.have_object(repo, stray, 999) is False, \
                "a wrong-sized file in the store must not count as present"
        case("wrong-sized cached file is not treated as present", failures,
             wrong_size_on_disk_is_not_have)

    # --- endpoint fallback ------------------------------------------------------------------
    def falls_through_to_the_fallback():
        seen = []

        def fake_post(url, body):
            seen.append(url)
            if "sunnypilot" in url:
                return {"objects": [{"oid": good_oid, "size": good_size,
                                     "error": {"code": 404, "message": "Object does not exist"}}]}
            return {"objects": [{"oid": good_oid, "size": good_size,
                                 "actions": {"download": {"href": "https://example/obj"}}}]}

        href, _ = m.resolve_href(["https://sunnypilot/info/lfs", "https://comma/info/lfs"],
                                 good_oid, good_size, post=fake_post)
        assert href == "https://example/obj", "got " + str(href)
        assert len(seen) == 2, "should have tried both endpoints, tried " + repr(seen)
    case("404 on our own store falls through to comma's", failures, falls_through_to_the_fallback)

    def all_404_raises():
        def fake_post(url, body):
            return {"objects": [{"oid": good_oid, "size": good_size,
                                 "error": {"code": 404, "message": "nope"}}]}
        try:
            m.resolve_href(["https://a/info/lfs", "https://b/info/lfs"],
                           good_oid, good_size, post=fake_post)
        except LookupError:
            return
        raise AssertionError("returned an href when no store had the object")
    case("no store has it -> raises rather than returning None", failures, all_404_raises)

    def a_dead_endpoint_does_not_stop_the_search():
        def fake_post(url, body):
            if "dead" in url:
                raise OSError("connection refused")
            return {"objects": [{"oid": good_oid, "size": good_size,
                                 "actions": {"download": {"href": "https://example/obj"}}}]}
        href, _ = m.resolve_href(["https://dead/info/lfs", "https://live/info/lfs"],
                                 good_oid, good_size, post=fake_post)
        assert href == "https://example/obj", "got " + str(href)
    case("an unreachable endpoint is skipped, not fatal", failures,
         a_dead_endpoint_does_not_stop_the_search)

    def a_size_mismatch_from_the_server_is_refused():
        def fake_post(url, body):
            return {"objects": [{"oid": good_oid, "size": good_size + 1,
                                 "actions": {"download": {"href": "https://example/obj"}}}]}
        try:
            m.resolve_href(["https://a/info/lfs"], good_oid, good_size, post=fake_post)
        except LookupError:
            return
        raise AssertionError("accepted an href for an object the server sized differently")
    case("server reporting a different size is not trusted", failures,
         a_size_mismatch_from_the_server_is_refused)

    print("\n" + "-" * 58)
    if failures:
        print("FAILED: " + str(len(failures)) + " case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: the LFS seeder verifies what it installs and never leaves a partial object")
    return 0


if __name__ == "__main__":
    sys.exit(main())
