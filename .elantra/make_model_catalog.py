#!/usr/bin/env python3
"""
Turn a compiled chunk set into the catalog document that puts the model in sunnypilot's picker.

sunnypilot's catalog is the only route into the Models screen, so a model this port carries
ahead of them needs an entry of its own. `openpilot/sunnypilot/models/elantra_catalog.py` merges
that entry into the list they serve; this script writes it, from the artifact rather than from a
template.

Everything written here is a promise the car checks later, on a slow link:
`ModelManagerSP._process_artifact` verifies every chunk's sha256 and raises
"Hash validation failed for chunk N", and the chunk count decides the `.chunkNNofMM` names, so
one wrong number 404s every URL. Nothing is emitted that has not been read off disk.

    python .elantra/make_model_catalog.py \\
        --chunks /data/modelbuild \\
        --file-name driving_cinqueterre_tinygrad.pkl \\
        --base-url https://github.com/aaravsaianugula/openpilot/releases/download/elantra-models/ \\
        --short-name CT1 --display-name "Cinque Terre (September 04, 2026)" \\
        --out /data/modelbuild/elantra_models_chestnut.json

Stdlib only: this runs on the device next to the build, with no openpilot imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

READ_SIZE = 1024 * 1024

# The runtime gate is exact equality against helpers.REQUIRED_JSON_VERSION, so it is read from
# that file rather than typed here: a bundle whose selector version drifts is dropped by
# is_bundle_version_compatible and simply never appears, with no error anywhere.
HELPERS_REL = "openpilot/sunnypilot/models/helpers.py"

# comma's big supercombo lineage, same as the BMRLNAP bundles this sits beside. generation
# gates real behaviour in modeld_v2 -- >=10 enables curvature smoothing, >=11 mlsim -- so it
# tracks the bundles built from the same model family rather than being invented here.
GENERATION = "12"
DEFAULT_FOLDER = "Elantra Port"
# BMV6's smoothing. Matching it is what makes an A/B against BMV6 a comparison of weights
# rather than of smoothing constants.
DEFAULT_LAT = ".1"
DEFAULT_LONG = ".3"


def chunk_name(name: str, idx: int, num_chunks: int) -> str:
  """Mirror of common/file_chunker.get_chunk_name -- the names the downloader derives."""
  return f"{name}.chunk{idx + 1:02d}of{num_chunks:02d}"


def read_selector_version(helpers_path: Path) -> int:
  source = Path(helpers_path).read_text(encoding="utf-8", errors="replace")
  m = re.search(r"^REQUIRED_JSON_VERSION\s*=\s*(\d+)", source, re.M)
  if not m:
    raise ValueError("could not read REQUIRED_JSON_VERSION from " + str(helpers_path))
  return int(m.group(1))


def _sha256(path: Path) -> tuple[str, int]:
  digest = hashlib.sha256()
  size = 0
  with open(path, "rb") as f:
    while True:
      block = f.read(READ_SIZE)
      if not block:
        break
      digest.update(block)
      size += len(block)
  return digest.hexdigest(), size


def discover_chunks(chunks_dir: Path, file_name: str) -> list[Path]:
  """Resolve the chunk set, cross-checking the manifest against the files on disk.

  Both are required to agree. The manifest alone would happily describe 17 chunks against 16
  files; the files alone cannot tell a complete set from a partial copy.
  """
  chunks_dir = Path(chunks_dir)
  manifest = chunks_dir / (file_name + ".chunkmanifest")
  if not manifest.is_file():
    raise FileNotFoundError("no chunkmanifest at " + str(manifest)
                            + " -- has the model been compiled?")
  try:
    declared = int(manifest.read_text(encoding="utf-8").strip())
  except ValueError:
    raise ValueError("chunkmanifest is not an integer: " + str(manifest)) from None
  if declared < 1:
    raise ValueError("chunkmanifest declares " + str(declared) + " chunks")

  on_disk = sorted(chunks_dir.glob(file_name + ".chunk??of??"))
  if len(on_disk) != declared:
    raise ValueError("manifest declares " + str(declared) + " chunks but "
                     + str(len(on_disk)) + " are on disk in " + str(chunks_dir))

  paths = []
  for i in range(declared):
    p = chunks_dir / chunk_name(file_name, i, declared)
    if not p.is_file():
      raise FileNotFoundError("missing chunk " + p.name
                              + " -- refusing to publish a catalog with a hole in it")
    paths.append(p)
  return paths


def build_catalog(chunks_dir, file_name: str, base_url: str, short_name: str,
                  display_name: str, helpers_path, folder: str = DEFAULT_FOLDER,
                  lat: str = DEFAULT_LAT, long: str = DEFAULT_LONG,
                  tinygrad_ref: str | None = None) -> dict:
  paths = discover_chunks(Path(chunks_dir), file_name)
  num_chunks = len(paths)

  chunks = []
  whole = hashlib.sha256()
  total = 0
  for i, p in enumerate(paths):
    digest, size = _sha256(p)
    chunks.append({"file_name": chunk_name(file_name, i, num_chunks), "sha256": digest})
    total += size
    with open(p, "rb") as f:
      while True:
        block = f.read(READ_SIZE)
        if not block:
          break
        whole.update(block)

  artifact_sha = whole.hexdigest()
  # Deterministic and content-bound: recompiling identical bytes keeps the ref, so the device
  # does not see it as a different model; changed bytes always get a new one.
  ref = hashlib.sha1(artifact_sha.encode()).hexdigest()

  bundle = {
    "short_name": short_name,
    "display_name": display_name,
    "is_20hz": True,
    "is_big": True,
    "ref": ref,
    "environment": "release",
    "runner": "tinygrad",
    # The merge renumbers this above whatever sunnypilot's document uses, so the value here
    # only matters as a well-formed int for _parse_bundle.
    "index": 0,
    # Emitted as a string because that is the shape sunnypilot uses in their own
    # documents. _parse_bundle coerces with int() either way, but matching them means
    # anything that ever reads the raw JSON treats our entry exactly like theirs.
    "minimum_selector_version": str(read_selector_version(Path(helpers_path))),
    "generation": GENERATION,
    "build_time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "overrides": {"folder": folder, "lat": lat, "long": long},
    "models": [{
      "type": "chunked",
      "artifact": {
        "file_name": file_name,
        "download_uri": {"url": base_url.rstrip("/") + "/" + file_name, "sha256": artifact_sha},
        "chunks": chunks,
      },
    }],
  }

  doc = {"bundles": [bundle], "total_bytes": total}
  if tinygrad_ref:
    doc["tinygrad_ref"] = tinygrad_ref
  return doc


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--chunks", required=True, type=Path, help="directory holding the chunk set")
  ap.add_argument("--file-name", required=True, help="artifact name, e.g. driving_x_tinygrad.pkl")
  ap.add_argument("--base-url", required=True, help="URL prefix the chunks are served from")
  ap.add_argument("--short-name", required=True)
  ap.add_argument("--display-name", required=True)
  ap.add_argument("--out", required=True, type=Path)
  ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
  ap.add_argument("--folder", default=DEFAULT_FOLDER)
  ap.add_argument("--lat", default=DEFAULT_LAT)
  ap.add_argument("--long", default=DEFAULT_LONG)
  ap.add_argument("--tinygrad-ref", default=None,
                  help="tinygrad commit the pkl was compiled against, recorded for provenance")
  args = ap.parse_args()

  doc = build_catalog(chunks_dir=args.chunks, file_name=args.file_name,
                      base_url=args.base_url, short_name=args.short_name,
                      display_name=args.display_name,
                      helpers_path=args.repo / HELPERS_REL,
                      folder=args.folder, lat=args.lat, long=args.long,
                      tinygrad_ref=args.tinygrad_ref)

  args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

  b = doc["bundles"][0]
  art = b["models"][0]["artifact"]
  print("wrote " + str(args.out))
  print("  bundle     " + b["short_name"] + "  " + b["display_name"])
  print("  ref        " + b["ref"])
  print("  selector   " + str(b["minimum_selector_version"]) + " (read from helpers.py)")
  print("  chunks     " + str(len(art["chunks"])))
  print("  bytes      " + str(doc["total_bytes"]))
  print("  base url   " + art["download_uri"]["url"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
