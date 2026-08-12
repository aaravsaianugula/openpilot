"""Verified small-model additions sourced from official OpenPilot experiments."""

from copy import deepcopy


TINYGRAD_REF = "2fecac4e4ac32fe369c41f8400b6e7b9adb18683"
RESOURCE_BASE = "https://raw.githubusercontent.com/firestar5683/StarPilot-Resources/Models"

DEEP_MODEL16_REF = "f02d134f40f5e7be22b182af21b438915a47600e"
REBEL_LEGION_REF = "2895346746634d7eec0ee749f946c87039948a25"
GYHU_REF = "574735edc6e1aafdc2a69395f9a32e7f5cc4b62b"
RDF1_REF = "a95e2c25cae5fbf1afba7628bfb7acc4af59e0cc"

OFFICIAL_SMOOTHING = {
  DEEP_MODEL16_REF: (".0", ".3"),
  REBEL_LEGION_REF: (".1", ".1"),
}
MASTER_FOLDER_REFS = {GYHU_REF, RDF1_REF}


def _artifact(name: str, sha256: str) -> dict:
  return {
    "file_name": name,
    "download_uri": {
      "url": f"{RESOURCE_BASE}/{name}",
      "sha256": sha256,
    },
    "chunks": [
      {"file_name": f"{name}.p00", "sha256": ""},
      {"file_name": f"{name}.p01", "sha256": ""},
    ],
  }


def _bundle(*, index: int, short_name: str, display_name: str, ref: str, source_sha256: str,
            source_size: int, artifact_name: str, artifact_sha256: str, lat: str, long: str) -> dict:
  return {
    "short_name": short_name,
    "display_name": display_name,
    "is_20hz": True,
    "ref": ref,
    "source_sha256": source_sha256,
    "source_size": source_size,
    "environment": "development",
    "runner": "tinygrad",
    "index": index,
    "minimum_selector_version": "16",
    "generation": "12",
    "overrides": {
      "folder": "Master Models",
      "lat": lat,
      "long": long,
    },
    "models": [
      {
        "type": "chunked",
        "artifact": _artifact(artifact_name, artifact_sha256),
      }
    ],
  }


OPENPILOT_EXPERIMENTAL_BUNDLES = (
  _bundle(
    index=1001,
    short_name="NOPP",
    display_name="No-PP (OpenPilot c740fe5f)",
    ref="c740fe5f58faefcb9184c6f9f1fb45130b83cfe8",
    source_sha256="40bae078480fa5cc94cf0f90ec26c4295efcde8a3f07f1e110acf5c503e29e10",
    source_size=96_600_448,
    artifact_name="nopp3_driving_tinygrad.pkl",
    artifact_sha256="981705dbf416fcde3f61bd841a6d892a5b15c4e0156440968480023368081309",
    lat=".0",
    long=".3",
  ),
  _bundle(
    index=1002,
    short_name="RDF2",
    display_name="RDF Checkpoint 2 (OpenPilot 0f8b4248)",
    ref="0f8b4248a2e8bdd63b6ea4c6e1bbb32119cb2620",
    source_sha256="ea07a008696413e6ff402d3e2d2d4a3b9133ce9f1031b2efc50ffc524d8204d8",
    source_size=96_635_513,
    artifact_name="rdf23_driving_tinygrad.pkl",
    artifact_sha256="bfd72297d5194b6d178ca5f3a42686421e955845caea3d4cf5b26cb77868f709",
    lat=".1",
    long=".3",
  ),
  _bundle(
    index=1003,
    short_name="RDF3",
    display_name="RDF Checkpoint 3 (OpenPilot ea2151ba)",
    ref="ea2151ba4b82854277f37f03b949f15fe2733dc8",
    source_sha256="095c9145cd58bd7af723fce238bc4c2510e00a7d12f61407ecc436c12a550a47",
    source_size=96_635_514,
    artifact_name="rdf33_driving_tinygrad.pkl",
    artifact_sha256="a2ffcd3b3557a268926c6ae533c9d4b1696e99b565f669ac45e75ded6212abcc",
    lat=".1",
    long=".3",
  ),
  _bundle(
    index=1004,
    short_name="RDF4",
    display_name="RDF Checkpoint 4 (OpenPilot a5a6412d)",
    ref="a5a6412d08474cffb49a69afb910756afdee123e",
    source_sha256="b9550d192e32769f1fca82b2e2f2fae65e335b9f6fbbda2b455fcc8b2207bef6",
    source_size=96_636_386,
    artifact_name="rdf43_driving_tinygrad.pkl",
    artifact_sha256="a77db33c2e2d6a7570dc2a4a70c2b877429ee8bd9ca5dfeda74b5a41231aaff9",
    lat=".1",
    long=".3",
  ),
  _bundle(
    index=1005,
    short_name="RDF5",
    display_name="RDF Checkpoint 5 (OpenPilot 7fb03ca4)",
    ref="7fb03ca474f03e95e59ec0c8a6c5fba831bd5fd1",
    source_sha256="567e028c11989ceace7952c8efbcd60c10d5ba726a87f4b61d25632855a18b02",
    source_size=96_635_310,
    artifact_name="rdf53_driving_tinygrad.pkl",
    artifact_sha256="ef350dee3c5d74c213d6bbcc673f92f73036bb6b5111e6c4ce4b4df6c6a2c756",
    lat=".1",
    long=".3",
  ),
)


def apply_openpilot_experimental_overlay(catalog: dict) -> dict:
  if not isinstance(catalog, dict):
    raise TypeError("catalog must be a dictionary")
  if catalog.get("tinygrad_ref") != TINYGRAD_REF:
    raise ValueError("incompatible tinygrad ref")

  overlaid = deepcopy(catalog)
  bundles = overlaid.setdefault("bundles", [])

  for bundle in bundles:
    ref = bundle.get("ref")
    if ref in OFFICIAL_SMOOTHING:
      lat, long = OFFICIAL_SMOOTHING[ref]
      overrides = bundle.setdefault("overrides", {})
      overrides["lat"] = lat
      overrides["long"] = long
    if ref in MASTER_FOLDER_REFS:
      bundle.setdefault("overrides", {})["folder"] = "Master Models"

  existing_refs = {bundle.get("ref") for bundle in bundles}
  bundles.extend(deepcopy(bundle) for bundle in OPENPILOT_EXPERIMENTAL_BUNDLES if bundle["ref"] not in existing_refs)
  return overlaid
