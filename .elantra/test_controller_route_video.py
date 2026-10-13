"""Route video frame selection, input collision and decoder byte boundaries."""
from types import SimpleNamespace
import json
import sys
import time
import hashlib

import numpy as np
import pytest

from controller_model_overlay import RenderError
from controller_route_video import decode_rgb, display_schedule, encode_ram, ram_admission, read_reuse_rgb, run_phase, validate_targets


def test_display_schedule_retains_global_clock_across_local_index_reset():
  packet = {"narrow_t": np.array([1., 1.05, 1.1, 1.15])}
  times, rows = display_schedule(packet, 4, 10)
  np.testing.assert_allclose(times, [1., 1.1])
  np.testing.assert_array_equal(rows, [0, 2])


def test_output_receipt_hardlink_and_raw_root_are_rejected(tmp_path):
  raw = tmp_path / "raw"
  raw.mkdir()
  source = raw / "camera.hevc"
  source.write_bytes(b"keep")
  with pytest.raises(RenderError, match="raw"):
    validate_targets(raw / "clip.mp4", raw, [source])
  out = tmp_path / "clip.mp4"
  out.with_suffix(".json").hardlink_to(source)
  with pytest.raises(RenderError, match="collision|exists"):
    validate_targets(out, raw, [source])
  assert source.read_bytes() == b"keep"


@pytest.mark.parametrize("actual_frames", [1, 3])
def test_compact_decoder_refuses_partial_or_extra_frames(tmp_path, monkeypatch, actual_frames):
  source = tmp_path / "camera.hevc"
  source.write_bytes(b"source")
  target = tmp_path / "compact.rgb"
  def run(command, *args, **kwargs):
    target.write_bytes(b"\0" * (actual_frames * 4 * 2 * 3))
    return SimpleNamespace(returncode=0)
  monkeypatch.setattr("controller_route_video.run_phase", run)
  with pytest.raises(RenderError, match="byte count"):
    decode_rgb(source, np.array([10, 12]), {"width": 4, "height": 2, "nb_read_frames": 20}, target)


def test_compact_decoder_records_exact_selected_presentation_indices(tmp_path, monkeypatch):
  source, target = tmp_path / "camera.hevc", tmp_path / "compact.rgb"
  source.write_bytes(b"source")
  def run(command, *args, **kwargs):
    assert "select=eq(n\\,10)+eq(n\\,12)" in command
    assert args[2] == 120
    target.write_bytes(b"\0" * (2 * 4 * 2 * 3))
    return SimpleNamespace(returncode=0)
  monkeypatch.setattr("controller_route_video.run_phase", run)
  pixels, receipt = decode_rgb(source, np.array([10, 12]), {"width": 4, "height": 2, "nb_read_frames": 20}, target)
  assert pixels.shape == (2, 2, 4, 3)
  assert receipt["source_presentation_indices"] == [10, 12]
  assert json.loads(target.with_suffix(".decoder.json").read_text())["compact_sha256"] == receipt["compact_sha256"]
  pixels._mmap.close()


def test_phase_failure_persists_actual_exit_command_and_full_stderr(tmp_path):
  command = [sys.executable, "-c", "import sys;sys.stderr.write('x'*320000);sys.exit(7)"]
  errors, receipt = tmp_path / "stderr.log", tmp_path / "phase.json"
  with pytest.raises(RenderError, match="failed"):
    run_phase(command, errors, receipt, 5)
  proof = json.loads(receipt.read_text())
  assert proof["command"] == command and proof["exit_code"] == 7 and proof["status"] == "failed"
  assert errors.stat().st_size == 320000


def test_entire_phase_timeout_kills_child_and_retains_exact_receipt(tmp_path):
  command = [sys.executable, "-c", "import time;time.sleep(30)"]
  errors, receipt = tmp_path / "stderr.log", tmp_path / "phase.json"
  start = time.monotonic()
  with pytest.raises(RenderError, match="timeout"):
    run_phase(command, errors, receipt, .15)
  proof = json.loads(receipt.read_text())
  assert time.monotonic() - start < 5
  assert proof["command"] == command and proof["status"] == "timeout" and proof["exit_code"] is not None
  assert proof["timeout_s"] == .15


def test_ram_admission_refuses_low_memory_or_more_than_2gib():
  with pytest.raises(RenderError, match="memory"):
    ram_admission([48, 48], 96, 128)
  with pytest.raises(RenderError, match="cap"):
    ram_admission([2 * 1024**3], 48, 10 * 1024**3)
  assert ram_admission([48], 96, 1024**3)["allocated_array_bytes"] == 144


def reuse_fixture(tmp_path):
  source = tmp_path / "camera.hevc"
  source.write_bytes(b"camera")
  directory = tmp_path / "reuse"
  directory.mkdir()
  raw = directory / "source-0.rgb"
  raw.write_bytes(bytes(range(48)))
  errors = directory / "source-0.stderr.log"
  errors.write_bytes(b"")
  command = ["ffmpeg", "-i", str(source), "-vf", "select=eq(n\\,10)+eq(n\\,12)", "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw)]
  def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
  decoder = {"command": command, "source": str(source), "source_sha256": digest(source), "exit_code": 0, "status": "passed",
             "expected_bytes": 48, "actual_bytes": 48, "source_presentation_indices": [10, 12]}
  phase = {"command": command, "exit_code": 0, "status": "passed", "stderr": str(errors), "stderr_sha256": digest(errors)}
  (directory / "source-0.decoder.json").write_text(json.dumps(decoder))
  (directory / "source-0.phase.json").write_text(json.dumps(phase))
  return source, directory, decoder, phase


@pytest.mark.parametrize("mutation", [None, "source_sha", "indices", "size", "phase", "path", "buffer_sha", "select", "extra_filter"])
def test_reused_rgb_requires_exact_source_phase_indices_bytes_and_fresh_hash(tmp_path, mutation):
  source, directory, decoder, phase = reuse_fixture(tmp_path)
  if mutation == "source_sha":
    decoder["source_sha256"] = "0" * 64
  elif mutation == "indices":
    decoder["source_presentation_indices"] = [11, 13]
  elif mutation == "size":
    (directory / "source-0.rgb").write_bytes(b"short")
  elif mutation == "phase":
    phase["exit_code"] = 1
  elif mutation == "path":
    decoder["source"] = str(tmp_path / "other.hevc")
  elif mutation == "buffer_sha":
    decoder["compact_sha256"] = hashlib.sha256(bytes(range(48))).hexdigest()
    (directory / "source-0.rgb").write_bytes(bytes(reversed(range(48))))
  elif mutation == "select":
    decoder["command"][decoder["command"].index("-vf") + 1] = "select=eq(n\\,11)+eq(n\\,13)"
  elif mutation == "extra_filter":
    decoder["command"][-1:-1] = ["-filter:v", "scale=2:4"]
  (directory / "source-0.decoder.json").write_text(json.dumps(decoder))
  (directory / "source-0.phase.json").write_text(json.dumps(phase))
  if mutation:
    with pytest.raises(RenderError):
      read_reuse_rgb(directory, 0, source, np.array([10, 12]), 4, 2)
  else:
    pixels, proof = read_reuse_rgb(directory, 0, source, np.array([10, 12]), 4, 2)
    assert pixels.shape == (2, 2, 4, 3)
    assert proof["fresh_compact_sha256"] == hashlib.sha256(bytes(range(48))).hexdigest()
    assert (directory / "source-0.rgb").read_bytes() == bytes(range(48))


def test_ram_encoder_deadline_covers_real_pipe_backpressure(tmp_path):
  command = [sys.executable, "-c", "import sys,time;sys.stderr.write('x'*320000);sys.stderr.flush();time.sleep(30)"]
  errors, receipt = tmp_path / "stderr.log", tmp_path / "phase.json"
  start = time.monotonic()
  with pytest.raises(RenderError, match="timeout"):
    encode_ram(command, np.zeros(8 * 1024**2, dtype=np.uint8), errors, receipt, 1.)
  proof = json.loads(receipt.read_text())
  assert time.monotonic() - start < 5
  assert proof["status"] == "timeout" and proof["exit_code"] is not None and not proof["writer_alive"]
  assert proof["bytes_written_confirmed"] < proof["expected_bytes"]
  assert errors.stat().st_size == 320000


def test_ram_encoder_records_complete_pipe_writes(tmp_path):
  command = [sys.executable, "-c", "import sys;sys.stderr.write(str(len(sys.stdin.buffer.read())))"]
  errors, receipt = tmp_path / "stderr.log", tmp_path / "phase.json"
  pixels = np.arange(1024 * 1024, dtype=np.uint8)
  result = encode_ram(command, pixels, errors, receipt, 5)
  proof = json.loads(receipt.read_text())
  assert result.returncode == 0 and proof["bytes_written_confirmed"] == pixels.nbytes
  assert proof["status"] == "passed" and not proof["writer_alive"]
