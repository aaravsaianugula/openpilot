#!/usr/bin/env python3
"""
Seed a git-LFS object that this fork's own LFS store does not have.

Why this exists. .lfsconfig points LFS at gitlab.com/sunnypilot/public/sunnypilot-new-lfs.git,
which we have no write access to. When we carry a model blob from upstream comma before
sunnypilot has mirrored it, the pointer in our tree names an oid that store has never seen. That
is not a soft failure: filter.lfs.required is true, so `git lfs pull` fails, and -- the reason
this file exists -- updated.py's `git checkout --force` inside the staging overlay fails too, and
the device stops being able to take a build at all.

git-lfs resolves an oid out of .git/lfs/objects before it asks any server. So seeding the object
there makes the pointer resolve locally and offline, forever, without write access to anyone's
store. comma's own store is public and anonymous, which is where the bytes come from.

    # seed whatever oid the tracked pointer names
    python .elantra/fetch_lfs_object.py openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx

    # seed an oid before the pointer that names it exists in this tree (the safe ordering)
    python .elantra/fetch_lfs_object.py --oid e8d8...ff28 --size 765950064

    # report without downloading
    python .elantra/fetch_lfs_object.py --check <path>

Stdlib only, like guards.py: this has to run on a bare Python 3 on the device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# comma's LFS store. Public, anonymous, and the origin of every model blob we inherit through
# sunnypilot. Listed as a fallback rather than a replacement: our own store holds sunnypilot's
# own assets, which comma's does not have, so repointing lfs.url wholesale would break those.
FALLBACK_ENDPOINTS = (
    "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs",
)

POINTER_MAX_BYTES = 1024
LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"
_OID_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------------------------
# pointers


def parse_pointer(text: str) -> tuple[str, int]:
    """Parse a git-LFS pointer file. Raises ValueError on anything that is not exactly one.

    Deliberately strict. A loosely parsed pointer seeds an object under the wrong name, which
    looks installed and resolves to nothing on the next checkout.
    """
    if not text or "git-lfs.github.com/spec/v1" not in text:
        raise ValueError("not a git-lfs pointer")

    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key and value:
            fields[key] = value.strip()

    oid_field = fields.get("oid")
    if not oid_field:
        raise ValueError("pointer has no oid")
    algo, _, digest = oid_field.partition(":")
    if algo != "sha256":
        raise ValueError("unsupported oid algorithm: " + str(algo))
    if not _OID_RE.match(digest):
        raise ValueError("oid is not a 64-char sha256: " + repr(digest))

    size_field = fields.get("size")
    if size_field is None:
        raise ValueError("pointer has no size")
    try:
        size = int(size_field)
    except ValueError:
        raise ValueError("size is not an integer: " + repr(size_field)) from None
    if size < 0:
        raise ValueError("negative size: " + str(size))

    return digest, size


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True, check=True)
    return out.stdout


def read_pointer(repo: Path, path: str, ref: str | None = None) -> tuple[str, int]:
    """Read the pointer for `path`.

    Prefers the worktree when it still holds a pointer -- that is the case that matters, because
    we seed the NEW oid just after editing the pointer and before anything tries to resolve it.
    Falls back to the committed blob when the worktree file has already been smudged to the real
    object (766 MB of ONNX is not a pointer).
    """
    if ref is None:
        f = repo / path
        if f.is_file() and f.stat().st_size <= POINTER_MAX_BYTES:
            try:
                return parse_pointer(f.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                pass
        ref = "HEAD"
    return parse_pointer(_git(repo, "show", ref + ":" + path))


# --------------------------------------------------------------------------------------------
# the local object store


def object_path(repo: Path, oid: str) -> Path:
    """.git/lfs/objects/<oid[0:2]>/<oid[2:4]>/<oid> -- git-lfs's own layout."""
    return Path(repo) / ".git" / "lfs" / "objects" / oid[0:2] / oid[2:4] / oid


def have_object(repo: Path, oid: str, size: int) -> bool:
    """True only when the object is present AND the right length.

    The size check is not decoration: a partially written object is the failure mode this whole
    file is defending against, and it is indistinguishable from a good one by name alone.
    """
    p = object_path(repo, oid)
    try:
        return p.is_file() and p.stat().st_size == size
    except OSError:
        return False


def install_object(repo: Path, oid: str, size: int, stream) -> Path:
    """Stream `stream` into the object store, verifying sha256 and length before publishing.

    Writes to a temp file beside the destination -- same directory, so os.replace is atomic --
    and removes it on any failure. Nothing partial is ever visible at the final path.
    """
    dest = object_path(repo, oid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (oid + ".incomplete")

    digest = hashlib.sha256()
    written = 0
    try:
        with open(tmp, "wb") as f:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                written += len(chunk)
                f.write(chunk)

        if written != size:
            raise ValueError("expected " + str(size) + " bytes, got " + str(written)
                             + " -- truncated transfer")
        actual = digest.hexdigest()
        if actual != oid:
            raise ValueError("sha256 mismatch: expected " + oid + ", got " + actual)

        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


# --------------------------------------------------------------------------------------------
# the LFS batch API


def _post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Accept": LFS_CONTENT_TYPE, "Content-Type": LFS_CONTENT_TYPE},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def resolve_href(endpoints, oid: str, size: int, post=_post_json) -> tuple[str, dict]:
    """Ask each endpoint in turn for a download href. Raises LookupError if none has the object.

    An endpoint that 404s or is unreachable is skipped rather than fatal -- the whole point is
    that our own store is expected to 404 on exactly the objects we are seeding.
    """
    tried: list[str] = []
    body = {"operation": "download", "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}], "hash_algo": "sha256"}

    for endpoint in endpoints:
        url = endpoint.rstrip("/") + "/objects/batch"
        try:
            payload = post(url, body)
        except Exception as e:
            tried.append(endpoint + " (" + type(e).__name__ + ")")
            continue

        for obj in payload.get("objects", []):
            if obj.get("oid") != oid:
                continue
            if "error" in obj:
                tried.append(endpoint + " (" + str(obj["error"].get("code")) + ")")
                break
            # A store that agrees on the oid but not the length is describing a different
            # object, or lying. Either way it is not the one the pointer names.
            if obj.get("size") != size:
                tried.append(endpoint + " (size " + str(obj.get("size")) + ")")
                break
            action = obj.get("actions", {}).get("download")
            if not action or not action.get("href"):
                tried.append(endpoint + " (no download action)")
                break
            return action["href"], action.get("header") or {}
        else:
            tried.append(endpoint + " (oid absent from response)")

    raise LookupError("no LFS store has " + oid + "; tried: " + ", ".join(tried))


def lfs_endpoints(repo: Path) -> list[str]:
    """This repo's configured LFS endpoint first, then the public fallbacks."""
    endpoints: list[str] = []
    for args in (["config", "-f", ".lfsconfig", "lfs.url"], ["config", "lfs.url"]):
        try:
            url = _git(repo, *args).strip()
        except subprocess.CalledProcessError:
            continue
        if url and url not in endpoints:
            endpoints.append(url)
    for url in FALLBACK_ENDPOINTS:
        if url not in endpoints:
            endpoints.append(url)
    return endpoints


def fetch(repo: Path, oid: str, size: int, *, endpoints=None, post=_post_json) -> bool:
    """Ensure `oid` is in the local store. Returns True if it downloaded, False if already there."""
    if have_object(repo, oid, size):
        return False
    href, headers = resolve_href(endpoints or lfs_endpoints(repo), oid, size, post=post)
    req = urllib.request.Request(href, headers=headers or {})
    with urllib.request.urlopen(req, timeout=600) as resp:
        install_object(repo, oid, size, resp)
    return True


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="repo-relative paths of LFS-tracked files")
    ap.add_argument("--repo", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--ref", default=None, help="read pointers from this ref instead of the worktree")
    ap.add_argument("--oid", default=None, help="seed this oid directly, with --size")
    ap.add_argument("--size", type=int, default=None, help="byte length of --oid")
    ap.add_argument("--check", action="store_true", help="report only; download nothing")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit("not a git repository: " + str(repo))

    wanted: list[tuple[str, str, int]] = []
    if args.oid is not None:
        if args.size is None:
            raise SystemExit("--oid requires --size")
        if not _OID_RE.match(args.oid):
            raise SystemExit("--oid is not a 64-char sha256")
        wanted.append(("<explicit>", args.oid, args.size))
    for path in args.paths:
        oid, size = read_pointer(repo, path, args.ref)
        wanted.append((path, oid, size))

    if not wanted:
        raise SystemExit("nothing to do: pass a path, or --oid with --size")

    missing = 0
    for label, oid, size in wanted:
        present = have_object(repo, oid, size)
        print(label)
        print("  oid  " + oid)
        print("  size " + str(size))
        if present:
            print("  ->   already in the local store")
            continue
        missing += 1
        if args.check:
            print("  ->   MISSING")
            continue
        print("  ->   downloading...")
        fetch(repo, oid, size)
        print("  ->   installed at " + str(object_path(repo, oid)))

    if args.check and missing:
        print("\n" + str(missing) + " object(s) missing. A checkout of these would fail"
              + " under filter.lfs.required.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
