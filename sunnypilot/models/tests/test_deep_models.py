import asyncio
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import types

import pytest

if sys.platform == "win32":
  messaging_stub = types.ModuleType("cereal.messaging")
  messaging_stub.PubMaster = object
  messaging_stub.new_message = lambda *_args, **_kwargs: None
  sys.modules.setdefault("cereal.messaging", messaging_stub)

from openpilot.common import file_chunker
from openpilot.sunnypilot.models.fetcher import ModelParser
from openpilot.sunnypilot.models.helpers import _bundle_artifacts, _bundle_is_valid_locally, _compute_hash
from openpilot.sunnypilot.models import manager as model_manager
from openpilot.sunnypilot.models.openpilot_experimental import OPENPILOT_EXPERIMENTAL_BUNDLES


RDF_REF = "a95e2c25cae5fbf1afba7628bfb7acc4af59e0cc"
RDF_ARTIFACT = "driving_rdfm_tinygrad.pkl"
RDF_ARTIFACT_HASH = "410855ac549a543201be78ca1f7966a3c9feac3bdeb034c3ad4ff9595f733b1f"
RDF_CHUNK_HASHES = (
  "53a55730a7329620fb1d88e05716283c2c22058472fa34f67f158bfc22ecf694",
  "7cf83608588851bd4b95bd71e972c2b2cda987afd5280b728e457d0c98e58a1e",
  "a0b3e7ada4e64960c4ff86bc4c49df37704ee733a159cb4e50c614a28f784b75",
  "9a4ff35cf998797633dbaebf3df1c7c3320f98ed083a09a8dbedc99ecb725973",
  "96b5d8886a0ac2ff204a9a4182d90c517c60cfbac00d38ec2920db1fd839c6fd",
)


def rdf_catalog_fixture():
  chunks = [{"file_name": f"{RDF_ARTIFACT}.chunk{index:02d}of05", "sha256": sha256} for index, sha256 in enumerate(RDF_CHUNK_HASHES, start=1)]
  return {
    "tinygrad_ref": "2fecac4e4ac32fe369c41f8400b6e7b9adb18683",
    "bundles": [
      {
        "short_name": "RDFM",
        "display_name": "RDF Model (August 05, 2026)",
        "is_20hz": True,
        "ref": RDF_REF,
        "environment": "development",
        "runner": "tinygrad",
        "index": 73,
        "minimum_selector_version": "16",
        "generation": "12",
        "overrides": {"folder": "2026 Deep RL Models", "lat": ".1", "long": ".3"},
        "models": [
          {
            "type": "chunked",
            "artifact": {
              "file_name": RDF_ARTIFACT,
              "download_uri": {
                "url": "https://gitlab.com/sunnypilot/public/docs.sunnypilot.ai8/-/raw/main/models/recompiled18/model-RDF/driving_rdfm_tinygrad.pkl",
                "sha256": RDF_ARTIFACT_HASH,
              },
              "chunks": chunks,
            },
          }
        ],
      }
    ],
  }


def test_v18_rdf_bundle_parses_declared_chunks(tmp_path, monkeypatch):
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.Paths.model_root", lambda: str(tmp_path))
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.info", lambda *_args, **_kwargs: None)
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.warning", lambda *_args, **_kwargs: None)
  bundles = ModelParser.parse_models(rdf_catalog_fixture())

  assert len(bundles) == 6
  bundle = next(bundle for bundle in bundles if bundle.ref == RDF_REF)
  artifact = bundle.models[0].artifact
  assert bundle.index == 73
  assert bundle.ref == RDF_REF
  assert bundle.minimumSelectorVersion == 16
  assert artifact.fileName == RDF_ARTIFACT
  assert [chunk.sha256 for chunk in artifact.chunks] == list(RDF_CHUNK_HASHES)


def test_chunk_stream_reads_declared_chunks_in_order(tmp_path):
  open_file_chunked = getattr(file_chunker, "open_file_chunked", None)
  assert open_file_chunked is not None

  artifact_path = tmp_path / RDF_ARTIFACT
  chunks = (b"first-", b"second-", b"third")
  Path(file_chunker.get_manifest_path(artifact_path)).write_text(str(len(chunks)))
  for index, data in enumerate(chunks):
    Path(file_chunker.get_chunk_name(artifact_path, index, len(chunks))).write_bytes(data)

  with open_file_chunked(artifact_path) as stream:
    assert stream.read() == b"first-second-third"
  assert file_chunker.read_file_chunked(artifact_path) == b"first-second-third"


def test_chunk_url_uses_explicit_remote_filename_and_preserves_legacy_fallback():
  assert model_manager._chunk_url(
    "https://host/Models/rdf53_driving_tinygrad.pkl",
    "rdf53_driving_tinygrad.pkl.p00",
    0,
    2,
  ) == "https://host/Models/rdf53_driving_tinygrad.pkl.p00"
  assert model_manager._chunk_url("https://host/model.pkl", "", 0, 2) == "https://host/model.pkl.chunk01of02"


def test_chunked_download_reads_explicit_remote_parts_into_local_chunk_layout(tmp_path, monkeypatch):
  class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format, *_args):
      pass

  remote_dir = tmp_path / "remote"
  local_dir = tmp_path / "local"
  remote_dir.mkdir()
  local_dir.mkdir()
  artifact_name = "rdf53_driving_tinygrad.pkl"
  parts = (b"first-part-", b"second-part")
  for index, data in enumerate(parts):
    (remote_dir / f"{artifact_name}.p{index:02d}").write_bytes(data)

  server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(remote_dir)))
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.Paths.model_root", lambda: str(local_dir))
    monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.warning", lambda *_args, **_kwargs: None)
    artifact = ModelParser._parse_artifact({
      "file_name": artifact_name,
      "download_uri": {
        "url": f"http://127.0.0.1:{server.server_port}/{artifact_name}",
        "sha256": hashlib.sha256(b"".join(parts)).hexdigest(),
      },
      "chunks": [
        {"file_name": f"{artifact_name}.p00", "sha256": ""},
        {"file_name": f"{artifact_name}.p01", "sha256": ""},
      ],
    })

    class ActiveDownloadParams:
      @staticmethod
      def get(_key):
        return 1005

    manager = model_manager.ModelManagerSP.__new__(model_manager.ModelManagerSP)
    manager.params = ActiveDownloadParams()
    manager.selected_bundle = None
    manager._chunk_size = 4
    manager._download_start_times = {}
    manager._report_status = lambda: None
    local_artifact = local_dir / artifact_name

    asyncio.run(manager._process_artifact(artifact, str(local_dir)))

    assert file_chunker.read_file_chunked(local_artifact) == b"first-part-second-part"
    assert not (local_dir / f"{artifact_name}.p00").exists()
    assert Path(file_chunker.get_chunk_name(local_artifact, 0, 2)).is_file()
    assert Path(file_chunker.get_chunk_name(local_artifact, 1, 2)).is_file()
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def logical_chunked_bundle(tmp_path, monkeypatch, expected_hash):
  artifact_name = "logical_model.pkl"
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.Paths.model_root", lambda: str(tmp_path))
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.info", lambda *_args, **_kwargs: None)
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.warning", lambda *_args, **_kwargs: None)
  monkeypatch.setattr("openpilot.sunnypilot.models.helpers.Paths.model_root", lambda: str(tmp_path))
  bundle = ModelParser._parse_bundle({
    "short_name": "LOGICAL",
    "display_name": "Logical Model",
    "is_20hz": True,
    "ref": "logical-ref",
    "environment": "development",
    "runner": "tinygrad",
    "index": 2000,
    "minimum_selector_version": "16",
    "generation": "12",
    "overrides": {},
    "models": [{
      "type": "chunked",
      "artifact": {
        "file_name": artifact_name,
        "download_uri": {"url": "https://host/logical_model.pkl", "sha256": expected_hash},
        "chunks": [
          {"file_name": f"{artifact_name}.p00", "sha256": ""},
          {"file_name": f"{artifact_name}.p01", "sha256": ""},
        ],
      },
    }],
  })
  return bundle, tmp_path / artifact_name


def test_bundle_artifacts_uses_complete_logical_hash_for_chunked_artifact(tmp_path, monkeypatch):
  expected_hash = hashlib.sha256(b"first-second").hexdigest()
  bundle, _artifact_path = logical_chunked_bundle(tmp_path, monkeypatch, expected_hash)

  assert _bundle_artifacts(bundle) == [("logical_model.pkl", expected_hash)]


def test_bundle_validity_rejects_missing_and_corrupt_logical_parts(tmp_path, monkeypatch):
  data = (b"first-", b"second")
  expected_hash = hashlib.sha256(b"".join(data)).hexdigest()
  bundle, artifact_path = logical_chunked_bundle(tmp_path, monkeypatch, expected_hash)

  assert not _bundle_is_valid_locally(bundle)
  for index, part in enumerate(data):
    Path(file_chunker.get_chunk_name(artifact_path, index, len(data))).write_bytes(part)
  assert _bundle_is_valid_locally(bundle)

  Path(file_chunker.get_chunk_name(artifact_path, 1, len(data))).write_bytes(b"corrupt")
  assert not _bundle_is_valid_locally(bundle)


@pytest.mark.skipif(os.getenv("RUN_LIVE_OPENPILOT_MODEL_DOWNLOAD") != "RDF5", reason="126 MB live resource check is opt-in")
def test_live_rdf5_resource_download_matches_published_hash(tmp_path, monkeypatch):
  bundle = next(bundle for bundle in OPENPILOT_EXPERIMENTAL_BUNDLES if bundle["short_name"] == "RDF5")
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.Paths.model_root", lambda: str(tmp_path))
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.info", lambda *_args, **_kwargs: None)
  monkeypatch.setattr("openpilot.sunnypilot.models.fetcher.cloudlog.warning", lambda *_args, **_kwargs: None)
  monkeypatch.setattr("openpilot.sunnypilot.models.manager.cloudlog.error", lambda *_args, **_kwargs: None)
  artifact = ModelParser._parse_artifact(bundle["models"][0]["artifact"])

  class ActiveDownloadParams:
    @staticmethod
    def get(_key):
      return bundle["index"]

  manager = model_manager.ModelManagerSP.__new__(model_manager.ModelManagerSP)
  manager.params = ActiveDownloadParams()
  manager.selected_bundle = None
  manager._chunk_size = 128 * 1000
  manager._download_start_times = {}
  manager._report_status = lambda: None

  asyncio.run(manager._process_artifact(artifact, str(tmp_path)))

  logical_path = tmp_path / artifact.fileName
  assert _compute_hash(str(logical_path)) == "ef350dee3c5d74c213d6bbcc673f92f73036bb6b5111e6c4ce4b4df6c6a2c756"
  assert [Path(file_chunker.get_chunk_name(logical_path, index, 2)).stat().st_size for index in range(2)] == [99_614_720, 26_850_484]
