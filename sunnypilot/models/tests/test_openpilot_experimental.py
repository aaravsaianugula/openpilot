import copy

import pytest

from openpilot.sunnypilot.models.openpilot_experimental import apply_openpilot_experimental_overlay


DEEP_REF = "f02d134f40f5e7be22b182af21b438915a47600e"
REBEL_REF = "2895346746634d7eec0ee749f946c87039948a25"
GYHU_REF = "574735edc6e1aafdc2a69395f9a32e7f5cc4b62b"
RDF1_REF = "a95e2c25cae5fbf1afba7628bfb7acc4af59e0cc"

EXPECTED_EXPERIMENTS = {
  "c740fe5f58faefcb9184c6f9f1fb45130b83cfe8": (
    "nopp3_driving_tinygrad.pkl",
    "981705dbf416fcde3f61bd841a6d892a5b15c4e0156440968480023368081309",
    "40bae078480fa5cc94cf0f90ec26c4295efcde8a3f07f1e110acf5c503e29e10",
    ".0",
    ".3",
  ),
  "0f8b4248a2e8bdd63b6ea4c6e1bbb32119cb2620": (
    "rdf23_driving_tinygrad.pkl",
    "bfd72297d5194b6d178ca5f3a42686421e955845caea3d4cf5b26cb77868f709",
    "ea07a008696413e6ff402d3e2d2d4a3b9133ce9f1031b2efc50ffc524d8204d8",
    ".1",
    ".3",
  ),
  "ea2151ba4b82854277f37f03b949f15fe2733dc8": (
    "rdf33_driving_tinygrad.pkl",
    "a2ffcd3b3557a268926c6ae533c9d4b1696e99b565f669ac45e75ded6212abcc",
    "095c9145cd58bd7af723fce238bc4c2510e00a7d12f61407ecc436c12a550a47",
    ".1",
    ".3",
  ),
  "a5a6412d08474cffb49a69afb910756afdee123e": (
    "rdf43_driving_tinygrad.pkl",
    "a77db33c2e2d6a7570dc2a4a70c2b877429ee8bd9ca5dfeda74b5a41231aaff9",
    "b9550d192e32769f1fca82b2e2f2fae65e335b9f6fbbda2b455fcc8b2207bef6",
    ".1",
    ".3",
  ),
  "7fb03ca474f03e95e59ec0c8a6c5fba831bd5fd1": (
    "rdf53_driving_tinygrad.pkl",
    "ef350dee3c5d74c213d6bbcc673f92f73036bb6b5111e6c4ce4b4df6c6a2c756",
    "567e028c11989ceace7952c8efbcd60c10d5ba726a87f4b61d25632855a18b02",
    ".1",
    ".3",
  ),
}


def base_catalog():
  return {
    "tinygrad_ref": "2fecac4e4ac32fe369c41f8400b6e7b9adb18683",
    "bundles": [
      {"ref": DEEP_REF, "short_name": "OPM16D", "overrides": {"folder": "Master Models", "lat": ".1", "long": ".3"}},
      {"ref": REBEL_REF, "short_name": "RLM", "overrides": {"folder": "Master Models", "lat": ".1", "long": ".3"}},
      {"ref": GYHU_REF, "short_name": "GYHUM", "overrides": {"folder": "2026 Deep RL Models", "lat": ".1", "long": ".3"}},
      {"ref": RDF1_REF, "short_name": "RDFM", "overrides": {"folder": "2026 Deep RL Models", "lat": ".1", "long": ".3"}},
    ],
  }


def test_overlay_adds_exact_published_experiments_without_mutating_remote_catalog():
  catalog = base_catalog()
  original = copy.deepcopy(catalog)

  overlaid = apply_openpilot_experimental_overlay(catalog)

  assert catalog == original
  by_ref = {bundle["ref"]: bundle for bundle in overlaid["bundles"]}
  assert len(overlaid["bundles"]) == len(original["bundles"]) + len(EXPECTED_EXPERIMENTS)
  for ref, (artifact_name, artifact_hash, source_hash, lat, long) in EXPECTED_EXPERIMENTS.items():
    bundle = by_ref[ref]
    artifact = bundle["models"][0]["artifact"]
    assert bundle["source_sha256"] == source_hash
    assert bundle["overrides"]["folder"] == "Master Models"
    assert bundle["overrides"]["lat"] == lat
    assert bundle["overrides"]["long"] == long
    assert artifact["file_name"] == artifact_name
    assert artifact["download_uri"]["sha256"] == artifact_hash
    assert [chunk["file_name"] for chunk in artifact["chunks"]] == [f"{artifact_name}.p00", f"{artifact_name}.p01"]


def test_overlay_corrects_official_smoothing_contracts():
  overlaid = apply_openpilot_experimental_overlay(base_catalog())
  by_ref = {bundle["ref"]: bundle for bundle in overlaid["bundles"]}

  assert by_ref[DEEP_REF]["overrides"] == {"folder": "Master Models", "lat": ".0", "long": ".3"}
  assert by_ref[REBEL_REF]["overrides"] == {"folder": "Master Models", "lat": ".1", "long": ".1"}
  assert by_ref[GYHU_REF]["overrides"]["folder"] == "Master Models"
  assert by_ref[RDF1_REF]["overrides"]["folder"] == "Master Models"


def test_overlay_is_idempotent_and_does_not_duplicate_rdf1():
  once = apply_openpilot_experimental_overlay(base_catalog())
  twice = apply_openpilot_experimental_overlay(once)
  refs = [bundle["ref"] for bundle in twice["bundles"]]

  assert twice == once
  assert len(refs) == len(set(refs))
  assert refs.count(RDF1_REF) == 1


@pytest.mark.parametrize("catalog", [None, [], "catalog"])
def test_overlay_rejects_non_dictionary_catalogs(catalog):
  with pytest.raises(TypeError, match="catalog must be a dictionary"):
    apply_openpilot_experimental_overlay(catalog)


@pytest.mark.parametrize("tinygrad_ref", [None, "", "wrong-ref"])
def test_overlay_rejects_catalogs_from_an_incompatible_tinygrad_abi(tinygrad_ref):
  catalog = base_catalog()
  catalog["tinygrad_ref"] = tinygrad_ref

  with pytest.raises(ValueError, match="incompatible tinygrad ref"):
    apply_openpilot_experimental_overlay(catalog)
