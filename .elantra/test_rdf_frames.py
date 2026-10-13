"""Packet causality, real NV12 layout and missing-input regression checks."""
import numpy as np
import pytest
from collections import defaultdict
from pathlib import Path

import rdf_frames as rf


def frame(fid, t, index=None):
  return {"frame_id": fid, "t": t, "segment_id": fid if index is None else index, "valid": True}


def test_nv12_padding_preserves_every_pixel_and_only_adds_zero_storage():
  width, height = 1344, 760
  raw = np.arange(width * height * 3 // 2, dtype=np.uint8)
  packed = rf.pack_nv12(raw.tobytes(), width, height)
  stride, yh, uvh, size = rf.get_nv12_info(width, height)
  assert (stride, yh, uvh, size) == (1408, 768, 384, 2428928)
  assert len(packed) == size
  y = np.frombuffer(packed, dtype=np.uint8, count=stride * yh).reshape(yh, stride)
  uv = np.frombuffer(packed, dtype=np.uint8, count=stride * uvh, offset=stride * yh).reshape(uvh, stride)
  np.testing.assert_array_equal(y[:height, :width], raw[:width*height].reshape(height, width))
  np.testing.assert_array_equal(uv[:height//2, :width], raw[width*height:].reshape(height//2, width))
  assert not y[:, width:].any() and not y[height:].any()
  assert not uv[:, width:].any() and not uv[height//2:].any()
  assert not np.frombuffer(packed, dtype=np.uint8, offset=stride*(yh+uvh)).any()
  with pytest.raises(rf.PacketError, match="size"):
    rf.pack_nv12(b"short", width, height)


def test_pairing_anchors_at_both_cameras_available_without_clipping_timestamps():
  pairs = rf.pair_frames([frame(100, 10., 0), frame(101, 10.05, 1)],
                         [frame(100, 10.000013, 0), frame(101, 10.04999, 1)])
  assert pairs[0]["t"] == 10.000013
  assert pairs[0]["narrow_t"] == 10.
  assert pairs[0]["wide_t"] == 10.000013
  assert all(p["t"] >= p["narrow_t"] and p["t"] >= p["wide_t"] for p in pairs)
  assert pairs[1]["t"] == 10.05


@pytest.mark.parametrize("wide", [[frame(100, 9.97, 0)], [], [frame(101, 10., 0)]])
def test_missing_or_desynchronized_exact_camera_match_fails(wide):
  with pytest.raises(rf.PacketError):
    rf.pair_frames([frame(100, 10., 0)], wide)


def test_no_repeated_frames_or_dropped_frame_interpolation():
  with pytest.raises(rf.PacketError, match="contiguous"):
    rf.pair_frames([frame(100, 10, 0), frame(102, 10.1, 2)], [frame(100, 10, 0), frame(102, 10.1, 2)])
  with pytest.raises(rf.PacketError, match="duplicate"):
    rf.pair_frames([frame(100, 10, 0)], [frame(100, 10, 0), frame(100, 10, 0)])


def test_asof_rejects_future_stale_and_invalid_latest_state():
  rows = [{"t": 1., "valid": True, "value": 7}, {"t": 1.1, "valid": False, "value": 9}]
  with pytest.raises(rf.PacketError, match="future"):
    rf.asof(rows, .9, .2, "sample")
  value, age = rf.asof(rows, 1.05, .2, "sample")
  assert value["value"] == 7 and age == pytest.approx(.05)
  with pytest.raises(rf.PacketError, match="invalid"):
    rf.asof(rows, 1.15, .2, "sample")
  with pytest.raises(rf.PacketError, match="stale"):
    rf.asof(rows[:1], 1.3, .2, "sample")


def test_desire_is_explicit_recorded_state_not_probability_argmax():
  assert rf.desire_index("off", "none", "none") == 0
  assert rf.desire_index("preLaneChange", "left", "none") == 0
  assert rf.desire_index("laneChangeStarting", "left", "none") == 3
  assert rf.desire_index("laneChangeStarting", "right", "none") == 4
  assert rf.desire_index("off", "none", "turnLeft") == 1
  assert rf.desire_index("off", "none", "turnRight") == 2
  with pytest.raises(rf.PacketError, match="unknown"):
    rf.desire_index("unknown", "none", "none")
  with pytest.raises(rf.PacketError, match="unknown"):
    rf.desire_index("laneChangeStarting", "none", "none")


def test_canonical_warp_is_finite_and_uses_real_sensor_intrinsics():
  road, wide = rf.warps(np.array([0., .007, -.017]), "mici", "os04c10")
  assert road.shape == wide.shape == (3, 3)
  assert road.dtype == wide.dtype == np.float32
  assert np.isfinite(road).all() and np.isfinite(wide).all()
  assert not np.allclose(road, np.eye(3))
  with pytest.raises(rf.PacketError, match="camera"):
    rf.warps(np.zeros(3), "unknown_device", "unknown_sensor")


def test_packet_dynamic_inputs_are_causal_and_physical_action_offsets_explicit():
  streams = defaultdict(list, carState=[{"t": .95, "valid": True, "v": 8., "left_blinker": False, "right_blinker": False},
                                      {"t": 1.1, "valid": True, "v": 1000., "left_blinker": False, "right_blinker": False}],
    extrinsicsCalibration=[{"t": .9, "valid": True, "calibrated": True, "rpy": np.zeros(3)}],
    traffic=[{"t": .95, "valid": True, "is_rhd": False}],
    desire_state=[{"t": .95, "valid": True, "state": "off", "direction": "none"}],
    turn_state=[{"t": .95, "valid": True, "turn": "none"}],
    lateralDelay=[{"t": .9, "valid": True, "delay": .2, "status": "estimated"}])
  static = {"cp": {"steer_delay": .1, "long_delay": .5}, "device": "mici", "narrow_sensor": "os04c10"}
  pairs = rf.pair_frames([frame(0, 1.)], [frame(0, 1.000013)])
  result = rf.packet_inputs(pairs, streams, static)
  assert result["v_ego"][0] == 8
  np.testing.assert_allclose(result["action_t"][0], [.275, .575])
  np.testing.assert_array_equal(result["traffic_convention"][0], [1, 0])
  np.testing.assert_array_equal(result["desire"][0], [1, 0, 0, 0, 0, 0, 0, 0])
  streams["turn_state"][0]["valid"] = False
  result = rf.packet_inputs(pairs, streams, static)
  assert result["desire_proof"][0] == 2
  assert result["turn_state_valid"][0] == 0
  streams["carState"][0]["left_blinker"] = True
  with pytest.raises(rf.PacketError, match="invalid"):
    rf.packet_inputs(pairs, streams, static)
  streams["carState"][0]["left_blinker"] = False
  streams["extrinsicsCalibration"].append({"t": .99, "valid": True, "calibrated": False, "rpy": np.zeros(3)})
  with pytest.raises(rf.PacketError, match="uncalibrated"):
    rf.packet_inputs(pairs, streams, static)


def test_output_directory_cannot_alias_or_overwrite_source(tmp_path):
  segment = tmp_path / "raw"
  segment.mkdir()
  with pytest.raises(rf.PacketError):
    rf.prepare_output(segment / "packet", segment)
  target = tmp_path / "packet"
  target.mkdir()
  (target / "existing.txt").write_text("preserve")
  with pytest.raises(rf.PacketError, match="empty"):
    rf.prepare_output(target, segment)
  assert (target / "existing.txt").read_text() == "preserve"


def test_decoder_drains_large_error_output_and_preserves_real_exit(tmp_path, monkeypatch):
  import subprocess
  import sys

  original = subprocess.Popen

  def failing_decoder(_command, **kwargs):
    assert kwargs["stderr"] != subprocess.PIPE, "undrained stderr can deadlock frame reads"
    return original([sys.executable, "-c", "import sys; sys.stderr.write('baddecode' * 40000); sys.exit(7)"], **kwargs)

  monkeypatch.setattr(rf.shutil, "which", lambda _name: sys.executable)
  monkeypatch.setattr(rf.subprocess, "Popen", failing_decoder)
  with pytest.raises(rf.PacketError, match="exit7.*baddecode"):
    rf.decode_camera(tmp_path / "input.hevc", np.array([0]), tmp_path / "output.nv12")


@pytest.mark.parametrize("bytes_written", [12, 11, 13])
def test_decoder_uses_bounded_compact_file_and_exact_frame_count(tmp_path, monkeypatch, bytes_written):
  import subprocess

  def decode(command, **kwargs):
    assert command[-1] != "pipe:1"
    assert kwargs["stdout"] == subprocess.DEVNULL and 0 < kwargs["timeout"] <= 120
    Path(command[-1]).write_bytes(bytes(range(bytes_written)))
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "pack_nv12", lambda raw: b"p" + raw)
  monkeypatch.setattr(rf.subprocess, "run", decode)
  target = tmp_path / "output.nv12"
  if bytes_written == 12:
    rf.decode_camera(tmp_path / "input.hevc", np.array([7, 8]), target)
    assert target.read_bytes() == b"p" + bytes(range(6)) + b"p" + bytes(range(6, 12))
  else:
    with pytest.raises(rf.PacketError, match="byte count"):
      rf.decode_camera(tmp_path / "input.hevc", np.array([7, 8]), target)
    assert not target.exists()


def test_decoder_timeout_cannot_produce_a_completed_packet(tmp_path, monkeypatch):
  import subprocess

  def timeout(command, **kwargs):
    raise subprocess.TimeoutExpired(command, kwargs["timeout"])

  monkeypatch.setattr(rf.subprocess, "run", timeout)
  target = tmp_path / "output.nv12"
  with pytest.raises(rf.PacketError, match="120s bound"):
    rf.decode_camera(tmp_path / "input.hevc", np.array([0]), target)
  assert not target.exists()


@pytest.mark.parametrize("bytes_written", [12, 11, 13])
def test_ram_decode_compresses_exact_native_bytes_without_host_raw_file(tmp_path, monkeypatch, bytes_written):
  import hashlib
  import subprocess
  import sys
  import zstandard

  original = subprocess.Popen

  def decoder(command, **kwargs):
    assert command[-1] == "pipe:1"
    assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] != subprocess.PIPE
    return original([sys.executable, "-c", f"import sys; sys.stdout.buffer.write(bytes(range({bytes_written})))"], **kwargs)

  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "available_memory_bytes", lambda: 8 * 1024**3)
  monkeypatch.setattr(rf, "pack_nv12", lambda raw: b"p" + bytes(raw))
  monkeypatch.setattr(rf.subprocess, "Popen", decoder)
  target = tmp_path / "road.nv12.zst"
  if bytes_written == 12:
    receipt = rf.decode_camera_compressed(tmp_path / "input.hevc", np.array([7, 8]), target)
    with zstandard.ZstdDecompressor().stream_reader(target.open("rb")) as stream:
      native = stream.read()
    assert native == b"p" + bytes(range(6)) + b"p" + bytes(range(6, 12))
    assert receipt["native_sha256"] == hashlib.sha256(native).hexdigest()
    assert receipt["native_bytes"] == 14 and receipt["compact_bytes"] == 12
    assert receipt["decode_exit_code"] == 0 and receipt["kind"] == "ram_decode"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["road.nv12.zst"]
  else:
    with pytest.raises(rf.PacketError, match="byte count"):
      rf.decode_camera_compressed(tmp_path / "input.hevc", np.array([7, 8]), target)
    assert not target.exists()


@pytest.mark.parametrize("indices,frames,bytes_limit,available", [
  ([], 10, 100, 8 * 1024**3), ([0, 2], 10, 100, 8 * 1024**3),
  ([0, .5], 10, 100, 8 * 1024**3), ([-1], 10, 100, 8 * 1024**3),
  ([0, 1], 1, 100, 8 * 1024**3), ([0, 1], 10, 11, 8 * 1024**3),
  ([0, 1], 10, 100, 1),
])
def test_ram_decode_refuses_invalid_indices_limits_and_low_free_memory_before_process(
    tmp_path, monkeypatch, indices, frames, bytes_limit, available):
  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "available_memory_bytes", lambda: available)
  monkeypatch.setattr(rf.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("guard must run before decoding"))
  with pytest.raises(rf.PacketError):
    rf.decode_camera_compressed(tmp_path / "input.hevc", np.asarray(indices), tmp_path / "road.nv12.zst",
                                max_frames=frames, max_bytes=bytes_limit)


def test_ram_decode_failure_and_timeout_leave_no_completed_output(tmp_path, monkeypatch):
  import subprocess
  import sys

  original = subprocess.Popen
  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "available_memory_bytes", lambda: 8 * 1024**3)

  def failure(_command, **kwargs):
    return original([sys.executable, "-c", "import sys;sys.stderr.write('broken' * 40000);sys.exit(7)"], **kwargs)

  monkeypatch.setattr(rf.subprocess, "Popen", failure)
  target = tmp_path / "road.nv12.zst"
  with pytest.raises(rf.PacketError, match="exit7.*broken"):
    rf.decode_camera_compressed(tmp_path / "input.hevc", np.array([0]), target)
  assert not target.exists()

  def hang(_command, **kwargs):
    return original([sys.executable, "-c", "import time;time.sleep(30)"], **kwargs)

  monkeypatch.setattr(rf.subprocess, "Popen", hang)
  with pytest.raises(rf.PacketError, match="bound"):
    rf.decode_camera_compressed(tmp_path / "input.hevc", np.array([0]), target, timeout=.1)
  assert not target.exists()


def compact_reuse_fixture(tmp_path, monkeypatch):
  import hashlib
  import json

  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "pack_nv12", lambda raw: b"p" + bytes(raw))
  source, compact = tmp_path / "input.hevc", tmp_path / "compact.nv12"
  source.write_bytes(b"camera")
  compact.write_bytes(bytes(range(12)))
  context = {"source_camera_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_log_sha256": "a" * 64,
             "selection_sha256": "b" * 64, "camera_receipt_sha256": "c" * 64}
  receipt = {"exit_code": 0, "compact": str(compact), "compact_bytes": 12,
    "compact_sha256": hashlib.sha256(compact.read_bytes()).hexdigest(), "source_camera": str(source),
    "source_presentation_indices": [7, 8], "source_frame_ids": [123, 124], "frames": 2, "width": 2, "height": 2,
    "command": rf._compact_command(source, np.array([7, 8]), compact), **context}
  receipt_path = tmp_path / "decode-receipt.json"
  receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
  return source, compact, receipt_path, context, receipt


@pytest.mark.parametrize("corrupt", [False, True])
def test_compact_reuse_verifies_actual_pixel_hash_and_retains_failed_evidence(tmp_path, monkeypatch, corrupt):
  import json

  source, compact, receipt_path, context, _ = compact_reuse_fixture(tmp_path, monkeypatch)
  monkeypatch.setattr(rf.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("reused decode must not be rerun"))
  target = tmp_path / "road.nv12.zst"
  if corrupt:
    compact.write_bytes(b"\xff" + compact.read_bytes()[1:])
    with pytest.raises(rf.PacketError, match="SHA"):
      rf.reuse_camera_compact(source, np.array([7, 8]), np.array([123, 124]), target, receipt_path, context)
    assert not target.exists()
    failed = list(tmp_path.glob("*.failed-*.json"))
    assert len(failed) == 1 and json.loads(failed[0].read_text())["status"] == "failed_not_a_packet"
  else:
    result = rf.reuse_camera_compact(source, np.array([7, 8]), np.array([123, 124]), target, receipt_path, context)
    assert result["kind"] == "reused_predecoded_compact" and result["decode_exit_code"] == 0
    assert result["native_bytes"] == 14 and result["proof_receipt_sha256"] == rf.sha256(receipt_path)


@pytest.mark.parametrize("field,value", [("exit_code", 7), ("compact_sha256", None), ("source_log_sha256", "d" * 64),
                                        ("selection_sha256", "d" * 64), ("source_presentation_indices", [6, 7]),
                                        ("source_frame_ids", [122, 123]), ("command", ["ffmpeg", "wrong"])])
def test_compact_reuse_cannot_change_source_selection_or_executed_decode(tmp_path, monkeypatch, field, value):
  import json

  source, _compact, receipt_path, context, receipt = compact_reuse_fixture(tmp_path, monkeypatch)
  receipt[field] = value
  receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
  target = tmp_path / "road.nv12.zst"
  with pytest.raises(rf.PacketError):
    rf.reuse_camera_compact(source, np.array([7, 8]), np.array([123, 124]), target, receipt_path, context)
  assert not target.exists()



def test_compressed_only_packet_declares_virtual_raw_hashes_and_keeps_bench_contract(tmp_path, monkeypatch):
  segment = tmp_path / "route--0"
  segment.mkdir()
  for name in ("rlog.zst", "fcamera.hevc", "ecamera.hevc"):
    (segment / name).write_bytes(name.encode())
  schemas = tmp_path / "schemas"
  schemas.mkdir()
  (schemas / "log.capnp").write_text("schema")
  narrow = [frame(i, i * .05) for i in range(134)]
  wide = [frame(i, i * .05 + .000013) for i in range(134)]
  static = {"source_commit": "recorded", "source_dirty": False, "narrow_sensor": "os04c10", "cp": {}}
  monkeypatch.setattr(rf, "read_log", lambda *_args: ({"narrowRoadEncodeIdx": narrow, "wideRoadEncodeIdx": wide}, static))

  def inputs(pairs, *_args):
    return {"t": np.array([p["t"] for p in pairs]), "narrow_t": np.array([p["narrow_t"] for p in pairs]),
            "wide_t": np.array([p["wide_t"] for p in pairs]), "road_index": np.arange(len(pairs)),
            "wide_index": np.arange(len(pairs)), "frame_id": np.arange(len(pairs)),
            "desire_proof": np.zeros(len(pairs)), "turn_state_valid": np.ones(len(pairs))}

  def compressed(source, indices, target, **_kw):
    target.write_bytes(source.name.encode())
    return {"command": ["ffmpeg", str(source), "pipe:1"], "native_sha256": "a" * 64,
            "native_bytes": len(indices) * rf.get_nv12_info(rf.WIDTH, rf.HEIGHT)[3],
            "compressed_sha256": rf.sha256(target), "compressed_bytes": target.stat().st_size, "kind": "ram_decode"}

  monkeypatch.setattr(rf, "packet_inputs", inputs)
  monkeypatch.setattr(rf, "decode_camera_compressed", compressed)
  target = tmp_path / "packet"
  metadata = rf.build(segment, schemas, target, 0., 6.7, compressed_only=True)
  assert metadata["host_storage"]["mode"] == "compressed_only" and not metadata["host_storage"]["raw_files_present"]
  assert not (target / "road.nv12").exists() and not (target / "wide.nv12").exists()
  assert metadata["artifact_sha256"]["road.nv12"] == "a" * 64
  assert metadata["artifact_bytes"]["road.nv12"] == 134 * metadata["buffer_size"]
  assert metadata["artifact_sha256"]["road.nv12.zst"] == rf.sha256(target / "road.nv12.zst")
