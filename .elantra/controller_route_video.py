"""Render one hash-bound two-source RDF route; takeover remains display-only."""
import argparse
from contextlib import nullcontext
import faulthandler
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import sys

import numpy as np
from PIL import Image, ImageDraw

from controller_extract import sha256
from controller_model_overlay import RenderError, measured_forecasts, rdf_path
from controller_route_overlay import display_rdf_copy, load_route_evidence
from controller_video import camera_mapping, compose, frame_schedule, probe_video, projection_config, text
from rdf_frames import available_memory_bytes


def display_schedule(packet, frames, fps):
  return frame_schedule((np.arange(frames), packet["narrow_t"]), float(packet["narrow_t"][0]), frames / 20., fps)


def validate_targets(output, raw_root, inputs):
  targets = [output, output.with_suffix(".json"), output.with_suffix(".view.npz"), output.with_suffix(".view.json"),
             output.with_suffix(".controlled-mask.npz"), output.with_suffix(".encoder.log"), output.with_suffix(".encoder-phase.json"),
             output.with_suffix(".rgb-reuse.json"), output.with_suffix(".compose-phase.json")]
  if output.suffix.lower() != ".mp4":
    raise RenderError("route output must be MP4")
  for target in targets:
    if target.resolve().is_relative_to(raw_root.resolve()):
      raise RenderError("route outputs must stay outside raw source root")
    for source in inputs:
      if target.resolve() == source.resolve() or (target.exists() and source.exists() and target.samefile(source)):
        raise RenderError("route output collision with input")
    if target.exists():
      raise RenderError("route output/receipt already exists; preserve it")


def run_phase(command, stderr_path, receipt_path, timeout):
  """Bound the whole child lifetime, without stdin/stdout/stderr pipe backpressure."""
  result, status, detail = None, "launch_failed", None
  with stderr_path.open("xb") as errors:
    try:
      child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
      try:
        result = child.wait(timeout=timeout)
        status = "passed" if result == 0 else "failed"
      except subprocess.TimeoutExpired:
        child.kill()
        status = "timeout"
        try:
          result = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
          status = "termination_timeout"
    except OSError as exc:
      detail = repr(exc)
  proof = {"command": command, "exit_code": result, "status": status, "timeout_s": timeout,
           "termination_cleanup_timeout_s": 5, "stderr": str(stderr_path), "stderr_sha256": sha256(stderr_path), "detail": detail}
  with receipt_path.open("x", encoding="utf-8") as output:
    json.dump(proof, output, indent=2, allow_nan=False)
  if status != "passed":
    raise RenderError("FFmpeg/child phase " + status + "; command/exit/stderr/partial preserved")
  return subprocess.CompletedProcess(command, result)


def ram_admission(source_bytes, render_bytes, available):
  sizes = [*source_bytes, render_bytes]
  if any(type(value) is not int or value <= 0 for value in sizes) or type(available) is not int or available < 0:
    raise RenderError("invalid bounded RAM memory contract")
  total = sum(sizes)
  if total > 2 * 1024**3:
    raise RenderError("route RAM arrays exceed2GiB cap")
  required = 2 * total + 512 * 1024**2
  if available < required:
    raise RenderError(f"insufficient available memory: require{required} available{available}; no fallback")
  return {"allocated_array_bytes": total, "array_cap_bytes": 2 * 1024**3,
          "available_bytes": available, "required_available_bytes": required}


def read_reuse_rgb(directory, index, source, indices, width, height):
  """Read a complete owned decoded buffer once, with explicit fresh digest proof."""
  raw, decoder_path, phase_path = [directory / f"source-{index}{suffix}" for suffix in (".rgb", ".decoder.json", ".phase.json")]
  for path in (directory, raw, decoder_path, phase_path):
    if path.is_symlink() or path.resolve() != path.absolute():
      raise RenderError("reused RGB path must be regular and contain no symlink")
  decoder, phase = (json.loads(path.read_text()) for path in (decoder_path, phase_path))
  expected = len(indices) * width * height * 3
  command = decoder.get("command", [])
  try:
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
      raise TypeError("decoder argv must be a string list")
    declared_source = Path(command[command.index("-i") + 1])
    pixels = command[command.index("-pix_fmt") + 1]
    format_name = command[command.index("-f") + 1]
    selected_filter = command[command.index("-vf") + 1]
  except (ValueError, IndexError, TypeError) as exc:
    raise RenderError("reused RGB decoder command contract missing") from exc
  filter_options = [token for token in command if token.split("=", 1)[0] in ("-vf", "-filter", "-filter_complex", "-lavfi")
                    or token.startswith(("-vf:", "-filter:", "-filter_complex:", "-lavfi:"))]
  expected_filter = "select=" + "+".join(f"eq(n\\,{int(value)})" for value in indices)
  errors = Path(phase.get("stderr", ""))
  if (decoder.get("status") != "passed" or decoder.get("exit_code") != 0 or phase.get("status") != "passed" or phase.get("exit_code") != 0
      or command != phase.get("command") or declared_source.resolve() != source.resolve() or Path(decoder.get("source", "")).resolve() != source.resolve()
      or decoder.get("source_sha256") != sha256(source) or pixels != "rgb24" or format_name != "rawvideo"
      or filter_options != ["-vf"] or selected_filter != expected_filter
      or Path(command[-1]).resolve() != raw.resolve() or decoder.get("source_presentation_indices") != indices.astype(int).tolist()
      or decoder.get("expected_bytes") != expected or decoder.get("actual_bytes") != expected or not raw.is_file() or raw.stat().st_size != expected
      or errors.resolve().parent != directory.resolve() or not errors.is_file() or errors.is_symlink() or sha256(errors) != phase.get("stderr_sha256")):
    raise RenderError("reused RGB source/phase/local-index/size/path proof differs")
  before = raw.stat()
  array = np.empty((len(indices), height, width, 3), dtype=np.uint8)
  view, count, digest = memoryview(array).cast("B"), 0, hashlib.sha256()
  with raw.open("rb") as stream:
    while count < expected:
      block = view[count:min(expected, count + 8 * 1024**2)]
      read = stream.readinto(block)
      if not read:
        raise RenderError("reused RGB became incomplete while reading")
      digest.update(block[:read])
      count += read
    if stream.read(1):
      raise RenderError("reused RGB contains extra bytes")
  after = raw.stat()
  if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
    raise RenderError("reused RGB changed while reading")
  if "compact_sha256" in decoder and digest.hexdigest() != decoder["compact_sha256"]:
    raise RenderError("reused RGB differs from the persisted decoder compact SHA")
  proof = {**decoder, "reuse_source": str(raw.resolve()), "fresh_compact_sha256": digest.hexdigest(), "reused_read_once_bytes": count,
           "decoder_receipt_sha256": sha256(decoder_path), "phase_receipt_sha256": sha256(phase_path),
           "reuse_stat": {"inode": after.st_ino, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns},
           "fresh_hash_basis": "Bound on reuse; original failing run computed but did not persist compact SHA before timeout"}
  return array, proof


def encode_ram(command, array, stderr_path, receipt_path, timeout):
  """The parent deadline covers the entire writer and child, including backpressure."""
  if array.dtype != np.uint8 or not array.flags.c_contiguous or not 0 < array.nbytes <= 2 * 1024**3:
    raise RenderError("bounded contiguous uint8 RAM encoder input required")
  state = {"bytes": 0, "error": None, "inflight": False}
  started, code, status, alive = time.monotonic(), None, "launch_failed", False
  with stderr_path.open("xb") as errors:
    try:
      child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors, bufsize=0)
      def write():
        view = memoryview(array).cast("B")
        try:
          while state["bytes"] < len(view):
            begin = state["bytes"]
            state["inflight"] = True
            count = child.stdin.write(view[begin:min(len(view), begin + 1024**2)])
            if count is None or count <= 0:
              raise RuntimeError("encoder stdin accepted no bytes")
            state["bytes"] += count
            state["inflight"] = False
        except Exception as exc:
          state["error"] = repr(exc)
        finally:
          try:
            child.stdin.close()
          except OSError:
            pass
      writer = threading.Thread(target=write, daemon=True)
      writer.start()
      try:
        code = child.wait(timeout=max(.001, timeout - (time.monotonic() - started)))
        status = "passed" if code == 0 else "failed"
      except subprocess.TimeoutExpired:
        child.kill()
        status = "timeout"
        try:
          code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
          status = "termination_timeout"
      writer.join(5)
      alive = writer.is_alive()
      if status == "passed" and (alive or state["error"] is not None or state["bytes"] != array.nbytes):
        status = "writer_incomplete"
    except OSError as exc:
      state["error"] = repr(exc)
  proof = {"command": command, "exit_code": code, "status": status, "timeout_s": timeout, "wall_s": time.monotonic() - started,
           "cleanup_wait_s": 5, "writer_join_s": 5, "writer_alive": alive, "writer_error": state["error"],
           "bytes_written_confirmed": state["bytes"], "write_in_flight_partial_bytes_unknown": state["inflight"],
           "expected_bytes": array.nbytes, "stderr": str(stderr_path), "stderr_sha256": sha256(stderr_path)}
  with receipt_path.open("x", encoding="utf-8") as output:
    json.dump(proof, output, indent=2, allow_nan=False)
  if status != "passed":
    raise RenderError("RAM encoder " + status + "; writer/child deadline receipt preserved")
  return subprocess.CompletedProcess(command, code)


class RamWriter:
  """Write frame bytes into the explicitly admitted, fixed-size RGB array."""
  def __init__(self, array):
    self.view = memoryview(array).cast("B")
    self.bytes = 0

  def write(self, data):
    end = self.bytes + len(data)
    if end > len(self.view):
      raise RenderError("RAM composition exceeded exact preallocated frame bytes")
    self.view[self.bytes:end] = data
    self.bytes = end


def decode_rgb(source, indices, probe, target):
  indices = np.asarray(indices)
  if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu" or np.any(np.diff(indices) <= 0)
      or indices[0] < 0 or indices[-1] >= int(probe["nb_read_frames"]) or target.exists()):
    raise RenderError("invalid compact route decode indices/output")
  width, height = int(probe["width"]), int(probe["height"])
  expected = len(indices) * width * height * 3
  if shutil.disk_usage(target.parent).free < expected * 2:
    raise RenderError("insufficient host compact decode reserve")
  expression = "select=" + "+".join(f"eq(n\\,{int(index)})" for index in indices)
  command = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-xerror", "-threads", "2", "-filter_threads", "1", "-i", str(source),
             "-vf", expression, "-fps_mode", "passthrough", "-threads:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", str(target)]
  result = run_phase(command, target.with_suffix(".stderr.log"), target.with_suffix(".phase.json"), 120)
  receipt = {"command": command, "exit_code": result.returncode, "source": str(source), "source_sha256": sha256(source),
             "source_presentation_indices": indices.astype(int).tolist(), "expected_bytes": expected, "timeout_s": 120,
             "actual_bytes": target.stat().st_size if target.is_file() else None}
  receipt["status"] = "passed" if receipt["actual_bytes"] == expected else "byte_count_mismatch"
  if receipt["status"] == "passed":
    receipt["compact_sha256"] = sha256(target)
  with target.with_suffix(".decoder.json").open("x", encoding="utf-8") as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
  if result.returncode:
    raise RenderError("route compact decoder failed; stderr/partial preserved")
  if not target.is_file() or target.stat().st_size != expected:
    raise RenderError("route compact decoder byte count differs from exact selected frames")
  receipt.update(compact_bytes=expected)
  return np.memmap(target, mode="r", dtype=np.uint8, shape=(len(indices), height, width, 3)), receipt


def render(args):
  if any(shutil.which(name) is None for name in ("ffmpeg", "ffprobe")):
    raise RenderError("existing ffmpeg/ffprobe are required")
  spec, packet, data, metadata, rdf, masks, evidence = load_route_evidence(
    args.selection, args.selection_sha256, args.packet, args.raw_root, args.dataset_dir, args.schema_dir, args.rdf_output, args.transfer_receipt)
  inputs = [Path(path) for path in evidence["input_sha256"]]
  inputs.extend((Path(__file__), Path(__file__).with_name("controller_route_overlay.py")))
  validate_targets(args.out, args.raw_root, inputs)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  decode_root = args.out.parent / (args.out.stem + "-decode")
  if decode_root.exists() or not decode_root.resolve().is_relative_to(args.out.parent.resolve()):
    raise RenderError("route compact decode directory must be new inside artifact output")
  decode_root.mkdir()
  times, packet_rows = display_schedule(packet, spec["frames"], args.fps)
  replay = measured_forecasts(data)
  display_rdf = display_rdf_copy(rdf, masks)
  pixels, decodes, local_to_compact, probes, origins = [], [], [], [], []
  reuse = getattr(args, "ram_reuse", None)
  render_bytes = len(times) * 1600 * 1000 * 3
  memory_proof = None
  if reuse is not None:
    if reuse.is_symlink() or not reuse.is_dir() or not reuse.resolve().is_relative_to(args.out.parent.resolve()):
      raise RenderError("RAM reuse directory must be preserved inside the artifact output directory")
    packet_meta = json.loads((args.packet / "packet.json").read_text())
    width, height = int(packet_meta["width"]), int(packet_meta["height"])
    if (width, height) != (1344, 760):
      raise RenderError("RAM reuse requires the verified os04c10 source dimensions")
    sizes = [int(np.unique(packet["road_index"][packet_rows[packet["source_chunk_index"][packet_rows] == i]]).size * width * height * 3)
             for i in range(2)]
    memory_proof = ram_admission(sizes, render_bytes, available_memory_bytes())
  for i, chunk in enumerate(spec["source_chunks"]):
    source = args.raw_root / chunk["segment"] / "fcamera.hevc"
    wanted = packet_rows[packet["source_chunk_index"][packet_rows] == i]
    local = np.unique(packet["road_index"][wanted])
    if reuse is None:
      probe = probe_video(source)
      images, receipt = decode_rgb(source, local, probe, decode_root / f"source-{i}.rgb")
    else:
      probe = {"width": width, "height": height, "basis": "frozen packet/complete successful decoder byte count; no new source probe/decode"}
      images, receipt = read_reuse_rgb(reuse, i, source, local, width, height)
    pixels.append(images)
    decodes.append(receipt)
    local_to_compact.append({int(index): j for j, index in enumerate(local)})
    probes.append(probe)
    origins.append(float(camera_mapping(data, int(chunk["segment"].rsplit("--", 1)[1]))[1][0]))
  if any((p["width"], p["height"]) != (probes[0]["width"], probes[0]["height"]) for p in probes):
    raise RenderError("route source camera dimensions differ")
  size = (int(probes[0]["width"]), int(probes[0]["height"]))
  intrinsics, projection = projection_config(data, metadata, size)
  render_target = decode_root / "render.rgb"
  if reuse is None and shutil.disk_usage(decode_root).free < 2 * render_bytes:
    raise RenderError("insufficient host file-backed encode reserve")
  if reuse is not None:
    with args.out.with_suffix(".rgb-reuse.json").open("x", encoding="utf-8") as output:
      json.dump({"mode": "bounded_RAM_reuse", "memory_admission": memory_proof, "buffers": decodes,
                 "old_buffers_and_partial_preserved": True, "no_redecode_no_inference": True}, output, indent=2, allow_nan=False)
    render_array = np.empty((len(times), 1000, 1600, 3), dtype=np.uint8)
    ram_writer = RamWriter(render_array)
  encode_command = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-threads", "2", "-filter_threads", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1600x1000", "-r", str(args.fps), "-i",
                    "pipe:0" if reuse is not None else str(render_target),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads:v", "2", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.out)]
  frames, calibrated, fresh = [], 0, 0
  censor = evidence["raw_censor_t"]
  render_hash = hashlib.sha256()
  def compose_all():
    nonlocal calibrated, fresh
    try:
      with (nullcontext(ram_writer) if reuse is not None else render_target.open("xb")) as output:
        for display_t, row in zip(times, packet_rows, strict=True):
          source_num, local = int(packet["source_chunk_index"][row]), int(packet["road_index"][row])
          timestamp = float(packet["narrow_t"][row])
          image = Image.fromarray(np.asarray(pixels[source_num][local_to_compact[source_num][local]]))
          _, fresh_index = rdf_path(display_rdf, data, timestamp)
          canvas, horizon, is_calibrated = compose(image, data, replay, timestamp, origins[source_num],
                                                 spec["source_chunks"][source_num]["segment"], intrinsics, projection, 1.22, display_rdf)
          controlled = fresh_index is not None and bool(masks["controlled_admissible"][fresh_index]) and timestamp < censor
          post = timestamp >= censor
          label = ("POST-BRAKE TAKEOVER — DISPLAY ONLY; excluded from controlled metrics" if post else
                   ("PRE-BRAKE — RDF ready: availability only; future GT may include takeover" if controlled else
                    "HISTORY / stale or missing display motion; no visible admitted output"))
          draw = ImageDraw.Draw(canvas)
          text(draw, (24, 94), f"TWO-SOURCE ROUTE25+26 | {label}", 14, "#ff7272" if post else "#ffbd69")
          footer = "Measured future GT is display-only, including driver braking. No completed-stop, tuned-response or improvement claim."
          text(draw, (24, 987), footer, 11, "#99aaba")
          raw = canvas.tobytes()
          output.write(raw)
          render_hash.update(raw)
          fresh += fresh_index is not None
          calibrated += is_calibrated
          frames.append({"display_t": float(display_t), "source_t": timestamp, "source_chunk_index": source_num,
                         "segment": spec["source_chunks"][source_num]["segment"], "frame_id": int(packet["frame_id"][row]),
                         "source_presentation_index": local, "packet_row": int(row), "camera_age_ms": (display_t - timestamp) * 1000,
                         "rdf_index": None if fresh_index is None else int(fresh_index), "controlled_output_available": controlled,
                         "post_censor_display": post, "calibrated_flat_ground": bool(is_calibrated), "measured_gt_horizon_s": horizon})
          if reuse is not None:
            print(json.dumps({"stage": "compose_frame_done", "index": len(frames) - 1}), file=sys.stderr, flush=True)
    finally:
      for array in pixels:
        if hasattr(array, "_mmap"):
          array._mmap.close()
  if reuse is not None:
    state = {"error": None}
    def guarded_compose():
      try:
        compose_all()
      except BaseException as exc:
        state["error"] = exc
    started = time.monotonic()
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
    worker = threading.Thread(target=guarded_compose, daemon=True)
    worker.start()
    worker.join(180)
    faulthandler.cancel_dump_traceback_later()
    alive = worker.is_alive()
    status = "timeout" if alive else ("failed" if state["error"] is not None else "passed")
    with args.out.with_suffix(".compose-phase.json").open("x", encoding="utf-8") as output:
      json.dump({"status": status, "timeout_s": 180, "wall_s": time.monotonic() - started,
                 "completed_frames": len(frames), "error": repr(state["error"]), "worker_alive": alive}, output, indent=2)
    if alive:
      faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
      raise RenderError("RAM composition timeout; daemon worker exits with this bounded process; sources preserved")
    if state["error"] is not None:
      raise RenderError("RAM composition failed") from state["error"]
    if ram_writer.bytes != render_bytes:
      raise RenderError("RAM encoder input byte count differs")
    encoding = encode_ram(encode_command, render_array, args.out.with_suffix(".encoder.log"), args.out.with_suffix(".encoder-phase.json"), 120)
  else:
    compose_all()
    if render_target.stat().st_size != render_bytes:
      raise RenderError("file-backed encoder input byte count differs; partial preserved")
    encoding = run_phase(encode_command, args.out.with_suffix(".encoder.log"), args.out.with_suffix(".encoder-phase.json"), 120)
  probe = probe_video(args.out)
  if int(probe["nb_read_frames"]) != len(times):
    raise RenderError("route encoded frame count differs; no silent drops allowed")
  if any(sha256(Path(path)) != digest for path, digest in evidence["input_sha256"].items()):
    raise RenderError("route input changed during render")
  with args.out.with_suffix(".view.npz").open("xb") as output:
    np.savez_compressed(output, **data)
  with args.out.with_suffix(".controlled-mask.npz").open("xb") as output:
    np.savez_compressed(output, input_t=rdf["input_t"], output_ready_t=rdf["output_ready_t"],
                        frame_id=rdf["frame_id"], raw_censor_t=np.array(censor), **masks)
  view = {**evidence, "view_output_sha256": sha256(args.out.with_suffix(".view.npz")),
          "contract": "Both untouched source tapes merged by exact publication grid; ambiguous duplicate conflicts reject"}
  with args.out.with_suffix(".view.json").open("x", encoding="utf-8") as output:
    json.dump(view, output, indent=2, allow_nan=False)
  if reuse is None:
    for i, receipt in enumerate(decodes):
      compact = decode_root / f"source-{i}.rgb"
      if compact.resolve().parent != decode_root.resolve() or compact.is_symlink() or sha256(compact) != receipt["compact_sha256"]:
        raise RenderError("generated compact cleanup path/SHA changed; preserve it")
      compact.unlink()  # Only these two verified generated RGB files; raw sources preserved.
      receipt["generated_compact_removed_after_success"] = True
    if render_target.resolve().parent != decode_root.resolve() or render_target.is_symlink() or sha256(render_target) != render_hash.hexdigest():
      raise RenderError("generated encoder input cleanup path/SHA changed; preserve it")
    render_target.unlink()
  else:
    for receipt in decodes:
      original = Path(receipt["reuse_source"]).stat()
      current = {"inode": original.st_ino, "bytes": original.st_size, "mtime_ns": original.st_mtime_ns}
      if current != receipt["reuse_stat"]:
        raise RenderError("preserved RGB input changed after RAM read")
  sources = [Path(__file__), Path(__file__).with_name("controller_route_overlay.py"), Path(__file__).with_name("controller_video.py")]
  receipt = {**evidence, "output": str(args.out.resolve()), "video_sha256": sha256(args.out), "output_probe": probe,
             "output_fps": args.fps, "output_frames": len(frames), "distinct_source_frames": len({f["frame_id"] for f in frames}),
             "calibrated_frames": int(calibrated), "fresh_rdf_visible_frames": int(fresh), "frames": frames,
             "decode_receipts": decodes, "encode_command": encode_command, "encoder_exit_code": encoding.returncode,
             "encoder_phase_receipt_sha256": sha256(args.out.with_suffix(".encoder-phase.json")),
             "encoder_input": {"mode": "RAM" if reuse is not None else "file", "bytes": render_bytes,
                               "sha256": render_hash.hexdigest(), "verified_generated_file_removed": reuse is None},
             "memory_admission": memory_proof, "preserved_rgb_reuse": reuse is not None,
             "source_sha256": {str(p.resolve()): sha256(p) for p in sources},
             "controlled_mask_sha256": sha256(args.out.with_suffix(".controlled-mask.npz")),
             "source_alignment": "Explicit packet source_chunk_index/globalID/localpresentationindex/actualnarrowEOF; containerfps unused",
             "gt_semantics": "Measured future is display-only and may include driver takeover; no controlled score computed here",
             "display_seam_policy": "Original source26 unknown/NaN/calibration rows retained; no derived seam repair or silent fill",
             "scope": "Fresh RDF prediction on fixed recorded images/motion; postbrake visible diagnostic only; no native/tuned physical response"}
  with args.out.with_suffix(".json").open("x", encoding="utf-8") as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
  print(json.dumps({"video": str(args.out), "frames": len(frames), "fresh_visible": int(fresh),
                    "controlled_outputs": evidence["controlled_outputs"], "calibrated": int(calibrated)}))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--selection", type=Path, required=True)
  parser.add_argument("--selection-sha256", required=True)
  parser.add_argument("--packet", type=Path, required=True)
  parser.add_argument("--raw-root", type=Path, required=True)
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--schema-dir", type=Path, required=True)
  parser.add_argument("--rdf-output", type=Path, required=True)
  parser.add_argument("--transfer-receipt", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--fps", type=float, default=10.)
  parser.add_argument("--ram-reuse", type=Path, help="Explicit boundedRAM recovery from complete proven RGB buffers; never decodes or falls back")
  render(parser.parse_args())


if __name__ == "__main__":
  main()
