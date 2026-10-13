"""Offline adapter guards: provenance, state safety, and causal replay inputs."""

import hashlib

import numpy as np
import pytest

from rdf_bench import (
  BenchError, availability_times, bundle_payload, check_snapshot, effective_action_t, hash_chunks,
  production_warmup, validate_replay, validate_result_path, verify_packet_files,
)


def snapshot(**changes):
  value = {"offroad": True, "onroad": False, "battery_percent": 65, "external_power": False,
           "active_processes": [], "temperatures_c": {"cpu0-gold-usr": 40., "gpu0-usr": 39.}}
  value.update(changes)
  return value


def test_owner_wall_power_override_does_not_gate_on_sysfs_power():
  check_snapshot(snapshot(external_power=False), 80.)


@pytest.mark.parametrize("changes,reason", [
  ({"offroad": False}, "offroad"), ({"onroad": True}, "offroad"),
  ({"battery_percent": 34}, "battery"), ({"active_processes": ["camerad"]}, "active"),
  ({"temperatures_c": {"gpu0-usr": 80.}}, "thermal"), ({"temperatures_c": {}}, "thermal"),
])
def test_live_state_changes_stop_the_isolated_bench(changes, reason):
  with pytest.raises(BenchError, match=reason):
    check_snapshot(snapshot(**changes), 80.)


def test_chunk_and_whole_hashes_are_verified_before_runtime_loading(tmp_path):
  blocks = [b"first", b"second"]
  paths = [tmp_path / str(i) for i in range(2)]
  for path, block in zip(paths, blocks, strict=True):
    path.write_bytes(block)
  expected = [hashlib.sha256(block).hexdigest() for block in blocks]
  whole = hashlib.sha256(b"".join(blocks)).hexdigest()
  assert hash_chunks(paths, expected, whole)["whole_sha256"] == whole
  paths[1].write_bytes(b"tampered")
  with pytest.raises(BenchError, match="SHA256"):
    hash_chunks(paths, expected, whole)


def test_bundle_forces_actual_rdf_v2_runtime_semantics():
  value = bundle_payload()
  assert value["internalName"] == "RDFMV2"
  assert value["is20hz"] is True and value["generation"] == 12
  assert value["ref"] == "35703097905a122c9f3ddf0d12889b4873d7e2a2"
  assert {entry["key"]: entry["value"] for entry in value["overrides"]} == {"lat": ".1", "long": ".3"}


def replay():
  return {"t": np.array([10., 10.05, 10.1]), "wide_t": np.array([9.999, 10.049, 10.099]),
          "narrow_t": np.array([10., 10.05, 10.1]),
          "tfm": np.tile(np.eye(3), (3, 1, 1)), "big_tfm": np.tile(np.eye(3), (3, 1, 1)), "desire": np.zeros((3, 8)),
          "traffic_convention": np.tile([1., 0.], (3, 1)), "action_t": np.ones((3, 2)) * .3, "v_ego": np.ones(3) * 8,
          "frame_id": np.arange(3), "wide_frame_id": np.arange(3)}


def test_replay_requires_causal_two_camera_pairs_and_recorded_inputs():
  validate_replay(replay(), 3)
  data = replay()
  data["wide_t"][1] += .1
  with pytest.raises(BenchError, match="camera"):
    validate_replay(data, 3)
  data = replay()
  data["t"][1] = data["t"][0]
  with pytest.raises(BenchError, match="increasing"):
    validate_replay(data, 3)
  data = replay()
  data["tfm"][0, 1, 1] = np.nan
  with pytest.raises(BenchError, match="finite"):
    validate_replay(data, 3)


def test_compute_availability_preserves_serial_backlog_and_does_not_look_ahead():
  ready, serial = availability_times([10., 10.05, 10.1], [2600., 30., 30.])
  np.testing.assert_allclose(ready, [12.6, 10.08, 10.13])
  np.testing.assert_allclose(serial, [12.6, 12.63, 12.66])


def test_rdf_catalog_smoothing_is_added_to_model_action_t_exactly_once():
  np.testing.assert_allclose(effective_action_t([.284269, .575], False, .1, .3), [.384269, .875])
  np.testing.assert_allclose(effective_action_t([.384269, .875], True, .1, .3), [.384269, .875])


def test_replay_never_silently_drops_or_repeats_source_frames():
  data = replay()
  data["frame_id"][1] = data["frame_id"][0]
  with pytest.raises(BenchError, match="consecutive"):
    validate_replay(data, 3)


def test_result_cannot_overwrite_packet_metadata_or_arrays():
  from pathlib import Path

  root = Path("/data/rdf-controller-tuning")
  with pytest.raises(BenchError, match="collides"):
    validate_result_path(root / "packet" / "packet.json", Path("/data/media/0/models"), root / "packet")
  validate_result_path(root / "results.json", Path("/data/media/0/models"), root / "packet")


@pytest.mark.parametrize("suffix,source_name", [(".json", ".chunk01of03"), (".npz", ".chunkmanifest")])
def test_result_hardlink_to_pinned_model_is_rejected_before_any_write(tmp_path, monkeypatch, suffix, source_name):
  import rdf_bench

  bench, models = tmp_path / "bench", tmp_path / "models"
  bench.mkdir()
  models.mkdir()
  monkeypatch.setattr(rdf_bench, "BENCH_ROOT", bench)
  source = models / (rdf_bench.NAME + source_name)
  source.write_bytes(b"protected model")
  output = bench / "result.json"
  output.with_suffix(suffix).hardlink_to(source)
  with pytest.raises(BenchError, match="aliases"):
    validate_result_path(output, models, None)
  assert source.read_bytes() == b"protected model"


def test_transferred_packet_hash_mismatch_stops_replay(tmp_path):
  metadata = {"artifact_sha256": {}}
  for name in ("packet.npz", "road.nv12", "wide.nv12"):
    (tmp_path / name).write_bytes(name.encode())
    metadata["artifact_sha256"][name] = hashlib.sha256(name.encode()).hexdigest()
  assert len(verify_packet_files(tmp_path, metadata)) == 3
  (tmp_path / "wide.nv12").write_bytes(b"truncated")
  with pytest.raises(BenchError, match="SHA256"):
    verify_packet_files(tmp_path, metadata)


def test_kernel_warmup_cannot_silently_prime_temporal_state():
  class Queue:
    def __init__(self):
      self.value = np.zeros(3)

    def numpy(self):
      return self.value

  class Model:
    def __init__(self):
      self.input_queues = {"feat_q": Queue()}
      self.numpy_inputs = {"prev_feat": np.zeros(2)}
      self.prev_desire = np.zeros(8)

    def warmup(self):
      self.input_queues["feat_q"].value[0] = 1

  with pytest.raises(BenchError, match="advanced temporal state"):
    production_warmup(Model())
