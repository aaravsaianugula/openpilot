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
