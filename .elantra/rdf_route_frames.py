#!/usr/bin/env python3
"""Build only a frozen, exact two-adjacent-segment RDF packet.

No merged rlog is fabricated. A single native stream/camera preserves chronological
model state across the boundary; every row retains its real segment/local indices.
All source proof is plural. Legacy source fields label the first scored segment
only and must never be used to bind the entire multi-source packet to one log.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from controller_extract import CP_FIELDS, decompress_log, load_schema, sha256
from controller_tune import cohort_key
import rdf_frames as rf


def _json(value):
  def encode(v):
    if isinstance(v, bytes):
      return {"capnp_data_hex": v.hex()}
    return v.tolist()
  return json.dumps(value, sort_keys=True, allow_nan=False, default=encode)


def _hash(value):
  return hashlib.sha256(_json(value).encode()).hexdigest()


def validate_selection(spec):
  chunks = spec.get("source_chunks", [])
  if len(chunks) != 2 or spec.get("partition") != "train":
    raise rf.PacketError("selection requires exactly two frozen training source chunks")
  if (spec.get("warmup_frames") != rf.WARMUP_FRAMES or spec.get("first_scored_packet_index") != rf.WARMUP_FRAMES or
      spec.get("frames", 0) <= rf.WARMUP_FRAMES or spec.get("scored_frames") != spec["frames"] - rf.WARMUP_FRAMES):
    raise rf.PacketError("selection requires132 real warmup frames and an explicit scored boundary")
  numbers = []
  for chunk in chunks:
    match = re.fullmatch(r"([0-9a-f]{8}--[0-9a-f]{10})--(\d+)", chunk.get("segment", ""))
    if match is None or match[1] != spec.get("route"):
      raise rf.PacketError("source chunks must belong to the same explicit route")
    numbers.append(int(match[2]))
    for field in ("road_presentation_indices", "wide_presentation_indices", "frame_ids"):
      bounds = chunk.get(field, [])
      if (len(bounds) != 2 or any(type(x) is not int or x < 0 for x in bounds) or
          bounds[1] - bounds[0] + 1 != chunk.get("frames")):
        raise rf.PacketError("chunk requires exact contiguous frame/index bounds")
    if not re.fullmatch(r"[0-9a-f]{64}", chunk.get("source_log_sha256", "")):
      raise rf.PacketError("chunk source log SHA256 missing")
  if numbers[1] != numbers[0] + 1 or chunks[1]["frame_ids"][0] != chunks[0]["frame_ids"][1] + 1:
    raise rf.PacketError("source chunks must be ordered adjacent segments and global frame IDs")
  if sum(c["frames"] for c in chunks) != spec["frames"]:
    raise rf.PacketError("selected total frame count differs from chunks")
  rf._compact_spec(np.arange(spec["frames"]), 1200, 2 * 1024**3)
  return chunks


def read_selection(path, expected_sha256):
  if sha256(path) != expected_sha256:
    raise rf.PacketError("frozen selection SHA256 differs")
  spec = json.loads(path.read_text(encoding="utf-8"))
  validate_selection(spec)
  return spec


def select_route_pairs(spec, stream_sets):
  chunks = validate_selection(spec)
  if len(stream_sets) != 2:
    raise rf.PacketError("exactly two source stream sets required")
  pairs = []
  for chunk_index, (chunk, streams) in enumerate(zip(chunks, stream_sets, strict=True)):
    narrow, wide = (sorted(streams[name], key=lambda r: r["segment_id"])
                    for name in ("narrowRoadEncodeIdx", "wideRoadEncodeIdx"))
    for rows in (narrow, wide):
      if (not rows or len({r["segment_id"] for r in rows}) != len(rows) or
          len({r["frame_id"] for r in rows}) != len(rows)):
        raise rf.PacketError("missing or ambiguous camera frame/presentation indices")
    lo, hi = chunk["road_presentation_indices"]
    selected = rf.pair_frames([r for r in narrow if lo <= r["segment_id"] <= hi], wide)
    if (len(selected) != chunk["frames"] or
        [selected[0]["frame_id"], selected[-1]["frame_id"]] != chunk["frame_ids"] or
        [selected[0]["wide_index"], selected[-1]["wide_index"]] != chunk["wide_presentation_indices"]):
      raise rf.PacketError("actual camera frames differ from frozen chunk bounds")
    if ((chunk_index == 0 and (selected[-1]["road_index"] != narrow[-1]["segment_id"] or
                              selected[-1]["wide_index"] != wide[-1]["segment_id"])) or
        (chunk_index == 1 and (selected[0]["road_index"] != 0 or selected[0]["wide_index"] != 0))):
      raise rf.PacketError("selected chunks do not cover the actual segment boundary")
    if "input_t" in chunk and not np.allclose([selected[0]["t"], selected[-1]["t"]], chunk["input_t"], atol=1e-6, rtol=0):
      raise rf.PacketError("actual chunk capture times differ from frozen selection")
    pairs.extend({**p, "source_num": int(chunk["segment"].rsplit("--", 1)[1]), "source_chunk_index": chunk_index} for p in selected)
  for a, b in zip(pairs[:-1], pairs[1:], strict=True):
    if b["frame_id"] != a["frame_id"] + 1 or not .025 <= b["t"] - a["t"] <= .075:
      raise rf.PacketError("global camera ID/time boundary has a duplicate or gap")
  if (pairs[rf.WARMUP_FRAMES]["frame_id"] != spec["first_scored_frame_id"] or
      pairs[-1]["frame_id"] != spec["last_scored_frame_id"] or
      not np.allclose([pairs[0]["t"], pairs[-1]["t"]], spec["capture_interval"], atol=1e-6, rtol=0)):
    raise rf.PacketError("actual warmup/scored capture boundary differs from frozen selection")
  return pairs


def merge_streams(stream_sets):
  merged, duplicates = defaultdict(list), Counter()
  for streams in stream_sets:
    for service, rows in streams.items():
      merged[service].extend(rows)
  for service, rows in merged.items():
    unique = {}
    for row in sorted(rows, key=lambda r: r["t"]):
      signature = _json(row)
      if row["t"] in unique:
        if unique[row["t"]][0] != signature:
          raise rf.PacketError(f"conflicting {service} payloads at one publication timestamp")
        duplicates[service] += 1
      else:
        unique[row["t"]] = signature, row
    merged[service] = [pair[1] for pair in unique.values()]
  return merged, dict(duplicates)


def same_contract(metadata):
  keys = ("car_params", "car_params_sp", "git_commit", "dirty", "schema_sha256", "params", "camera_sensor",
          "fingerprint", "op_long", "torque_factor", "torque_friction", "opendbc_gitlink")
  material = [{key: meta.get(key) for key in keys} for meta in metadata]
  if not material or any(_json(item) != _json(material[0]) for item in material[1:]):
    raise rf.PacketError("source CP/controller/schema/hardware contract changed across segments")


def route_output(out, segments):
  roots = {source.resolve().parent for source in segments}
  if len(roots) != 1:
    raise rf.PacketError("source segments must share one authoritative raw root")
  if out.resolve().is_relative_to(next(iter(roots))):
    raise rf.PacketError("packet output cannot alias the raw source root")
  return rf.prepare_output(out, next(iter(roots)))


def compress_camera_chunks(chunks, target):
  """Decode sequential bounded chunks into one native/compressed stream without resets."""
  total = sum(len(chunk["indices"]) for chunk in chunks)
  rf._compact_spec(np.arange(total), 1200, 2 * 1024**3)
  receipts = []

  def compact_frames():
    offset = 0
    for chunk in chunks:
      indices, frame_bytes, expected = rf._compact_spec(chunk["indices"], 1200, 2 * 1024**3)
      pixels, receipt = rf._decode_compact_ram(chunk["camera"], indices, expected, target.parent, 120)
      receipt.update(segment=chunk["segment"], source_num=chunk["source_num"], frames=len(indices),
                     source_presentation_indices=[int(indices[0]), int(indices[-1])], virtual_frame_offset=offset,
                     compact_sha256=hashlib.sha256(pixels).hexdigest())
      receipts.append(receipt)
      for i in range(len(indices)):
        # A copied single frame prevents the compressor's last memoryview from
        # retaining the entire prior chunk while the next buffer is allocated.
        yield bytes(memoryview(pixels)[i * frame_bytes:(i + 1) * frame_bytes])
      del pixels
      offset += len(indices)
  result = rf._compress_frames(compact_frames(), target)
  result.update(chunks=receipts, frames=total, kind="sequential_two_source_ram_decode")
  return result


def _raw_static(log, schema, meta):
  """Bind sidecar fields and hash every full recorded CP/SP semantic payload.

  Full payloads (including firmware/VIN fields) are never retained in receipts;
  canonical hashes encode Cap'n Proto Data bytes as hex and reject nonfinite data.
  """
  snapshots = {"car_params_snapshots": [], "car_params_sp_snapshots": []}
  init_seen = False
  expected_cp, expected_sp = _hash(meta["car_params"]), _hash(meta.get("car_params_sp", {}))
  for event in schema.Event.read_multiple_bytes(decompress_log(log), traversal_limit_in_words=2**31):
    service = event.which()
    if service == "initData":
      if not event.valid or str(event.initData.gitCommit) != meta["git_commit"] or bool(event.initData.dirty) != meta["dirty"]:
        raise rf.PacketError("actual recorded source differs from metadata contract")
      init_seen = True
    elif service == "carParams":
      cp = event.carParams.to_dict()
      if not event.valid or _hash({k: cp[k] for k in CP_FIELDS if k in cp}) != expected_cp:
        raise rf.PacketError("actual carParams sidecar fields changed or invalid")
      snapshots["car_params_snapshots"].append({"t": int(event.logMonoTime) * 1e-9, "semantic_sha256": _hash(cp)})
    elif service == "carParamsSP":
      full_sp_hash = _hash(event.carParamsSP.to_dict())
      if not event.valid or full_sp_hash != expected_sp:
        raise rf.PacketError("actual full carParamsSP changed or invalid")
      snapshots["car_params_sp_snapshots"].append({"t": int(event.logMonoTime) * 1e-9, "semantic_sha256": full_sp_hash})
  if not init_seen or not snapshots["car_params_snapshots"]:
    raise rf.PacketError("source lacks actual InitData/full CP snapshots")
  if "car_params_sp" in meta and not snapshots["car_params_sp_snapshots"]:
    raise rf.PacketError("source lacks required actual full carParamsSP snapshot")
  for rows in snapshots.values():
    if len({row["semantic_sha256"] for row in rows}) > 1:
      raise rf.PacketError("actual full CP/SP semantic payload changed within source")
    rows.sort(key=lambda row: row["t"])
  return snapshots


def same_full_contract(records):
  """Refuse cross-source CP changes hidden by the sidecar's selected fields."""
  for field in ("car_params_snapshots", "car_params_sp_snapshots"):
    present = [bool(record[field]) for record in records]
    hashes = {row["semantic_sha256"] for record in records for row in record[field]}
    if (not records or (field == "car_params_snapshots" and not all(present)) or
        len(set(present)) > 1 or len(hashes) > 1):
      raise rf.PacketError("actual full CP/SP semantic payload differs across sources")


def _camera_receipt(path, selection_sha, spec, root):
  receipt = json.loads(path.read_text(encoding="utf-8"))
  expected = {row["relative_path"]: row["inventory_bytes"] for row in spec["camera_allowlist"]}
  required = {chunk["segment"] + "/" + name for chunk in spec["source_chunks"] for name in ("fcamera.hevc", "ecamera.hevc")}
  verified = receipt.get("verified_files", [])
  if (set(expected) != required or len(expected) != 4 or receipt.get("status") != "complete" or
      receipt.get("rsync_exit") != 0 or receipt.get("dry_run_exit") != 0 or receipt.get("selection_sha256") != selection_sha or
      receipt.get("expected_files") != 4 or receipt.get("verified_target_files") != 4 or len(verified) != 4 or
      {row["relative_path"] for row in verified} != required or
      receipt.get("expected_bytes") != sum(expected.values()) or receipt.get("verified_target_bytes") != sum(expected.values())):
    raise rf.PacketError("complete exact four-camera selection-bound transfer receipt required")
  result = {}
  for row in verified:
    camera = root / row["relative_path"]
    if (Path(row["target"]).resolve() != camera.resolve() or not camera.is_file() or
        row["bytes"] != expected[row["relative_path"]] or camera.stat().st_size != row["bytes"] or sha256(camera) != row["sha256"]):
      raise rf.PacketError("actual camera hash/size/path differs from verified receipt")
    result[row["relative_path"]] = row
  return result


def build(selection_path, expected_selection_sha, camera_receipt, root, metadata_dir, schema_dir, splits_path, out):
  spec = read_selection(selection_path, expected_selection_sha)
  chunks = spec["source_chunks"]
  splits = json.loads(splits_path.read_text(encoding="utf-8"))
  if (sha256(splits_path) != spec["frozen_splits_sha256"] or
      spec["route"] not in splits["cohorts"][spec["cohort"]]["split"]["train"]):
    raise rf.PacketError("frozen training split/source reservation differs")
  sources = [root / chunk["segment"] for chunk in chunks]
  schema_hashes = {str(p.relative_to(schema_dir)): sha256(p) for p in sorted(schema_dir.rglob("*.capnp"))}
  inputs = [selection_path, camera_receipt, splits_path, *schema_dir.rglob("*.capnp")]
  inputs += [source / name for source in sources for name in ("rlog.zst", "fcamera.hevc", "ecamera.hevc")]
  inputs += [metadata_dir / (chunk["segment"] + ".json") for chunk in chunks]
  if any(not path.is_file() for path in inputs):
    raise rf.PacketError("source/receipt/schema input missing; no decoding fallback")
  before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in inputs}
  cameras = _camera_receipt(camera_receipt, expected_selection_sha, spec, root)
  stream_sets, metadata, statics, raw_contracts = [], [], [], []
  schema = load_schema(schema_dir)
  for chunk, source in zip(chunks, sources, strict=True):
    meta = json.loads((metadata_dir / (chunk["segment"] + ".json")).read_text(encoding="utf-8"))
    log = source / "rlog.zst"
    if (sha256(log) != chunk["source_log_sha256"] or meta.get("source_log_sha256") != chunk["source_log_sha256"] or
        Path(meta["source_log"]).resolve() != log.resolve() or meta.get("schema_sha256") != schema_hashes or cohort_key(meta) != spec["cohort"]):
      raise rf.PacketError("actual log/schema/cohort differs from frozen source contract")
    streams, static = rf.read_log(log, schema_dir, int(source.name.rsplit("--", 1)[1]))
    raw_contracts.append(_raw_static(log, schema, meta))
    stream_sets.append(streams)
    metadata.append(meta)
    statics.append(static)
  same_contract(metadata)
  same_full_contract(raw_contracts)
  hardware = [{k: s[k] for k in ("device", "narrow_sensor", "wide_sensor", "source_commit", "source_dirty")} for s in statics]
  if _json(hardware[0]) != _json(hardware[1]):
    raise rf.PacketError("actual camera/source hardware contract changed")
  pairs = select_route_pairs(spec, stream_sets)
  past_cp = [row["t"] for record in raw_contracts for row in record["car_params_snapshots"] if row["t"] <= pairs[0]["t"]]
  if not past_cp:
    raise rf.PacketError("no verified static CP snapshot preceding the first input frame")
  static = {**statics[0], "cp": {**statics[0]["cp"], "publication_t": max(past_cp)}}
  merged, duplicate_counts = merge_streams(stream_sets)
  arrays = rf.packet_inputs(pairs, merged, static)
  arrays.update(source_num=np.asarray([p["source_num"] for p in pairs], dtype=np.uint32),
                source_chunk_index=np.asarray([p["source_chunk_index"] for p in pairs], dtype=np.uint8),
                virtual_frame_index=np.arange(len(pairs), dtype=np.uint32), scored_mask=np.arange(len(pairs)) >= rf.WARMUP_FRAMES)
  output = route_output(out, sources)
  receipts = {}
  for name, key, camera in (("road.nv12", "road_index", "fcamera.hevc"), ("wide.nv12", "wide_index", "ecamera.hevc")):
    decode_chunks = [{"segment": chunk["segment"], "source_num": int(source.name.rsplit("--", 1)[1]),
      "camera": source / camera, "indices": arrays[key][arrays["source_chunk_index"] == i]}
      for i, (chunk, source) in enumerate(zip(chunks, sources, strict=True))]
    receipts[name] = compress_camera_chunks(decode_chunks, output / (name + ".zst"))
  np.savez_compressed(output / "packet.npz", **arrays)
  if any(before[path] != (path.stat().st_size, path.stat().st_mtime_ns) for path in before):
    raise rf.PacketError("input source/proof changed during packet build")
  stride, yh, uvh, size = rf.get_nv12_info(rf.WIDTH, rf.HEIGHT)
  if any(receipt["native_bytes"] != len(pairs) * size for receipt in receipts.values()):
    raise rf.PacketError("native concatenated byte count differs from selected frames")
  source_records = [{**chunk, "source_segment": str(source.resolve()), "metadata_sha256": sha256(metadata_dir / (chunk["segment"] + ".json")),
    "car_params_sha256": _hash(meta["car_params"]), "car_params_sp_sha256": _hash(meta.get("car_params_sp", {})),
    "full_static_snapshots": contract,
    "cameras": {name: cameras[chunk["segment"] + "/" + name] for name in ("fcamera.hevc", "ecamera.hevc")}}
    for chunk, source, meta, contract in zip(chunks, sources, metadata, raw_contracts, strict=True)]
  primary = int(arrays["source_chunk_index"][rf.WARMUP_FRAMES])
  hashes = {"packet.npz": sha256(output / "packet.npz")}
  sizes = {"packet.npz": (output / "packet.npz").stat().st_size}
  for name, receipt in receipts.items():
    hashes.update({name: receipt["native_sha256"], name + ".zst": receipt["compressed_sha256"]})
    sizes.update({name: receipt["native_bytes"], name + ".zst": receipt["compressed_bytes"]})
  repo = Path(__file__).resolve().parent.parent
  commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
  result = {"format_version": 2, "multi_source": True, "source_segments": source_records, "source_route": spec["route"],
    "source_segment": str(sources[primary].resolve()), "source_log_sha256": chunks[primary]["source_log_sha256"],
    "source_cameras_sha256": {name: source_records[primary]["cameras"][name]["sha256"] for name in ("fcamera.hevc", "ecamera.hevc")},
    "legacy_source_fields_basis": "primary first-scored segment label only; ALL frame provenance is in source_segments and per-row source_num/local indices",
    "consumer_admission": "single-source video/native inputs must reject or explicitly bind all source chunks; never claim one log covers this packet",
    "frames": len(pairs), "warmup_frames": rf.WARMUP_FRAMES, "warmup_seconds": 6.6, "recurrent_state_reset_at_boundary": False,
    "width": rf.WIDTH, "height": rf.HEIGHT, "stride": stride, "y_height": yh, "uv_height": uvh, "buffer_size": size,
    "frame_copy_size": stride * (yh + uvh), "source_commit": metadata[0]["git_commit"], "source_dirty": metadata[0]["dirty"],
    "source_opendbc_gitlink": metadata[0].get("opendbc_gitlink"), "repo_commit": commit,
    "selection_sha256": expected_selection_sha, "selection": str(selection_path.resolve()), "camera_receipt_sha256": sha256(camera_receipt),
    "schema_sha256": schema_hashes, "capture_interval": [pairs[0]["t"], pairs[-1]["t"]], "duplicate_message_counts": duplicate_counts,
    "static_car_params": static["cp"], "static_config_timing": "all full CP/SP semantic payloads identical; verified CP snapshot precedes first input",
    "full_static_hash_basis": "complete to_dict canonical sorted JSON; Data=hex; hashes/times bind each source_log_sha256",
    "max_wide_age_s": .025, "max_actual_camera_skew_s": float(np.max(abs(arrays["narrow_t"] - arrays["wide_t"]))),
    "calibration_status": "each source frame calibrated and causal-as-of, maximum age500ms",
    "camera_sensor": static["narrow_sensor"], "causal_join": "actual publication times; inputs as-of max camera EOF; conflicting duplicates rejected",
    "action_t_basis": "causal learned lateralDelay else verified CPsteer; CPlong; each+.075 production offset; no future GT or smoothing assumed",
    "frame_source_mapping": "source_num is real segment number; source_chunk_index selects ordered source_segments; road_index/wide_index remain LOCAL",
    "host_storage": {"mode": "compressed_only", "raw_files_present": False,
      "stored_files": ["packet.npz", "road.nv12.zst", "wide.nv12.zst"], "raw_artifact_receipt_basis": "computed during padding; device expansion verifies"},
    "decoder_receipts": receipts, "artifact_sha256": hashes, "artifact_bytes": sizes,
    "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(rf.__file__))},
    "diagnostic_only": True, "production_enabled": False,
    "scope": "one frozen training episode; no fabricated merged log, inferred capture origin, model reset, future GT input, or device/production change"}
  (output / "packet.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--selection", type=Path, required=True)
  parser.add_argument("--selection-sha256", required=True)
  parser.add_argument("--camera-receipt", type=Path, required=True)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--metadata-dir", type=Path, required=True)
  parser.add_argument("--schema-dir", type=Path, required=True)
  parser.add_argument("--splits", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  result = build(args.selection, args.selection_sha256, args.camera_receipt, args.root, args.metadata_dir, args.schema_dir, args.splits, args.out)
  print(json.dumps({"packet": str((args.out / "packet.json").resolve()), "frames": result["frames"], "multi_source": True}))


if __name__ == "__main__":
  main()
