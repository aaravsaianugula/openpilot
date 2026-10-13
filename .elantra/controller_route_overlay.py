"""Explicit plural-source route evidence, preserving single-source validators."""
import json
from pathlib import Path

import numpy as np

from controller_model_overlay import RenderError
from controller_extract import hold_calibration, load_schema, model_capture_times, sha256
import rdf_route_frames as route


def validate_sources(spec, packet_meta, metadata, raw_contracts, metadata_hashes, selection_sha, first_t):
  records = packet_meta.get("source_segments", [])
  chunks = spec["source_chunks"]
  if (packet_meta.get("multi_source") is not True or packet_meta.get("format_version") != 2
      or packet_meta.get("selection_sha256") != selection_sha or len(chunks) != 2
      or any(len(rows) != 2 for rows in (records, metadata, raw_contracts, metadata_hashes))):
    raise RenderError("route requires all two-source selection/packet/static contracts")
  try:
    route.same_contract(metadata)
    route.same_full_contract(raw_contracts)
  except route.rf.PacketError as exc:
    raise RenderError(str(exc)) from exc
  for chunk, record, meta, raw, meta_sha in zip(chunks, records, metadata, raw_contracts, metadata_hashes, strict=True):
    if (record.get("segment") != chunk["segment"] or record.get("source_log_sha256") != chunk["source_log_sha256"]
        or meta.get("source_log_sha256") != chunk["source_log_sha256"] or record.get("metadata_sha256") != meta_sha
        or record.get("full_static_snapshots") != raw):
      raise RenderError("plural route log/metadata/full static source binding differs")
    for name in ("fcamera.hevc", "ecamera.hevc"):
      selected = [row for row in spec["camera_allowlist"] if row["relative_path"] == chunk["segment"] + "/" + name]
      if len(selected) != 1 or record.get("cameras", {}).get(name, {}).get("sha256") != selected[0]["sha256"]:
        raise RenderError("plural route camera source binding differs")
  past_cp = [row["t"] for record in raw_contracts for row in record["car_params_snapshots"] if row["t"] <= first_t]
  if not past_cp:
    raise RenderError("missing actual full CP snapshot before first route history input")


def validate_mapping(spec, packet, pairs):
  n = spec["frames"]
  keys = ("t", "narrow_t", "wide_t", "frame_id", "wide_frame_id", "road_index", "wide_index", "source_num", "source_chunk_index")
  if len(pairs) != n or any(key not in packet or np.asarray(packet[key]).shape != (n,) for key in keys):
    raise RenderError("route packet mapping arrays/count missing")
  expected = {}
  chunks = spec["source_chunks"]
  for key in ("frame_id", "wide_frame_id"):
    expected[key] = np.concatenate([np.arange(c["frame_ids"][0], c["frame_ids"][1] + 1) for c in chunks])
  for key, bounds in (("road_index", "road_presentation_indices"), ("wide_index", "wide_presentation_indices")):
    expected[key] = np.concatenate([np.arange(c[bounds][0], c[bounds][1] + 1) for c in chunks])
  expected["source_num"] = np.concatenate([np.full(c["frames"], int(c["segment"].rsplit("--", 1)[1])) for c in chunks])
  expected["source_chunk_index"] = np.concatenate([np.full(c["frames"], i) for i, c in enumerate(chunks)])
  for key in keys:
    actual = np.asarray(packet[key])
    raw = np.asarray([p[key] for p in pairs])
    if (not np.isfinite(actual).all() or (key in expected and (actual.dtype.kind not in "iu" or not np.array_equal(actual, expected[key])))
        or not np.array_equal(actual, raw)):
      raise RenderError("route packet ID/local-index/capture mapping differs: " + key)
  if (not np.array_equal(packet.get("virtual_frame_index"), np.arange(n))
      or np.asarray(packet.get("scored_mask")).dtype.kind != "b"
      or not np.array_equal(packet["scored_mask"], np.arange(n) >= spec["warmup_frames"])):
    raise RenderError("route packet history/virtual mapping differs")


def validate_rdf_binding(rdf, bench, transfer, packet, packet_meta, hashes):
  """The frozen bench binds packet JSON; transport receipt binds returned JSON/NPZ."""
  proof = bench.get("packet", {})
  verified = proof.get("verified_file_sha256", {})
  adapter = "104d268c1bf946ce5b9c53d65fa066797d8c36371ff370b974504cced0e7a74d"
  if (bench.get("status") != "passed" or bench.get("no_actuation") is not True or bench.get("no_params_writes") is not True
      or bench.get("adapter_sha256") != adapter or hashes.get("executed_adapter") != adapter
      or transfer.get("sha256", {}).get(".executed-adapter.py") != adapter
      or bench.get("runtime_ref_verified") != "f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae" or bench.get("model_device") != "QCOM"
      or bench.get("production_warmup", {}).get("temporal_state_restored") is not True
      or bench.get("production_warmup", {}).get("state_arrays_compared") != 14
      or transfer.get("status") != "verified" or transfer.get("benchmark_exit_code") != 0 or transfer.get("transfer_exit_code") != 0
      or transfer.get("sha256", {}).get(".npz") != hashes["rdf_npz"]
      or transfer.get("sha256", {}).get(".json") != hashes["rdf_json"]
      or proof.get("metadata_sha256") != hashes["packet_json"]
      or packet_meta.get("artifact_sha256", {}).get("packet.npz") != hashes["packet_npz"]):
    raise RenderError("route RDF/packet/transport hash binding differs")
  for name in ("packet.npz", "road.nv12", "wide.nv12"):
    if verified.get(name) != packet_meta["artifact_sha256"].get(name):
      raise RenderError("route RDF native packet artifact binding differs")
  for key, source in (("input_t", "t"), ("narrow_t", "narrow_t"), ("wide_t", "wide_t"),
                      ("frame_id", "frame_id"), ("wide_frame_id", "wide_frame_id")):
    if key not in rdf or not np.array_equal(rdf[key], packet[source]):
      raise RenderError("route RDF capture/frame mapping differs: " + key)
  warmup = np.asarray(rdf.get("history_warmup"))
  if (not np.array_equal(rdf.get("scored_mask"), packet["scored_mask"])
      or not np.array_equal(rdf.get("history_scored_mask"), packet["scored_mask"])
      or warmup.dtype.kind != "b" or not np.array_equal(warmup, ~packet["scored_mask"]) or int(warmup.sum()) != 132
      or "serial_ready_t" not in rdf or not np.array_equal(rdf["output_ready_t"], rdf["serial_ready_t"])):
    raise RenderError("route RDF history or conservative serial availability differs")


def controlled_masks(rdf, censor):
  """Takeover outputs can be displayed, but never counted as controlled evidence."""
  capture, ready = np.asarray(rdf["narrow_t"]), np.asarray(rdf["output_ready_t"])
  if (not np.isfinite(censor) or capture.ndim != 1 or ready.shape != capture.shape
      or not np.isfinite(capture).all() or not np.isfinite(ready).all() or np.any(ready < capture)):
    raise RenderError("route controlled censor/availability clock invalid")
  display = np.ones(len(capture), bool)
  for key in ("scored_mask", "controller_admissible", "deadline_admissible"):
    values = np.asarray(rdf.get(key))
    if values.dtype.kind != "b" or values.shape != capture.shape:
      raise RenderError("route requires explicit RDF history/deadline flags")
    display &= values
  display &= (ready - capture) <= .2 + 1e-8
  return {"display_admissible": display, "controlled_admissible": display & (capture < censor) & (ready < censor),
          "post_censor_capture": capture >= censor}


def controlled_gt_mask(anchors, horizon, censor, output_admitted):
  """Every controlled outcome sample must precede takeover, even for earlier output."""
  anchors, horizon, admitted = np.asarray(anchors), np.asarray(horizon), np.asarray(output_admitted)
  if (anchors.ndim != 1 or horizon.ndim != 1 or not len(horizon) or admitted.shape != anchors.shape or admitted.dtype.kind != "b"
      or not np.isfinite(anchors).all() or not np.isfinite(horizon).all() or not np.isfinite(censor)
      or horizon[0] < 0 or np.any(np.diff(horizon) <= 0)):
    raise RenderError("controlled future GT horizon/censor contract invalid")
  return admitted[:, None] & (anchors[:, None] + horizon[None, :] < censor)


def display_rdf_copy(rdf, masks):
  admitted = np.asarray(masks["display_admissible"])
  if admitted.dtype.kind != "b" or admitted.shape != np.asarray(rdf["scored_mask"]).shape:
    raise RenderError("route display admission mask shape/type differs")
  return {**rdf, "scored_mask": admitted.copy()}


def merge_tapes(tapes):
  """Sort each real publication grid; equal duplicates collapse, conflicts reject."""
  if not tapes or any(tape.keys() != tapes[0].keys() for tape in tapes):
    raise RenderError("route display tape fields differ")
  groups = {"raw_encode_t": [], "raw_camera_t": [], "raw_learner_t": [], "raw_delay_t": [], "t": []}
  for key in tapes[0]:
    if key.startswith("raw_encode_"):
      group = "raw_encode_t"
    elif key.startswith("raw_camera_"):
      group = "raw_camera_t"
    elif key.startswith(("raw_learner_", "raw_learned_torque_")):
      group = "raw_learner_t"
    elif key.startswith(("raw_delay_", "raw_learned_lateral_delay")):
      group = "raw_delay_t"
    elif key.startswith("raw_"):
      raise RenderError("unknown route raw publication group: " + key)
    else:
      group = "t"
    groups[group].append(key)
  merged = {}
  for clock, keys in groups.items():
    if not keys:
      continue
    if clock not in keys:
      raise RenderError("route publication clock missing: " + clock)
    for tape in tapes:
      if any(np.asarray(tape[key]).ndim < 1 or len(tape[key]) != len(tape[clock]) for key in keys):
        raise RenderError("route publication shapes disagree")
      if any(tape[key].dtype != tapes[0][key].dtype or tape[key].shape[1:] != tapes[0][key].shape[1:] for key in keys):
        raise RenderError("route publication types/shapes differ")
    combined = {key: np.concatenate([tape[key] for tape in tapes]) for key in keys}
    order = np.argsort(combined[clock], kind="stable")
    times = combined[clock][order]
    if not np.isfinite(times).all():
      raise RenderError("route publication clock is not finite")
    keep = []
    for position in order:
      if keep and combined[clock][position] == combined[clock][keep[-1]]:
        if any(not np.array_equal(combined[key][position], combined[key][keep[-1]], equal_nan=True) for key in keys):
          raise RenderError("conflicting route display duplicate at one publication timestamp")
      else:
        keep.append(position)
    merged.update({key: values[keep] for key, values in combined.items()})
  if "t" not in merged or np.any(np.diff(merged["t"]) > .075 + 1e-8):
    raise RenderError("route measured motion has a gap")
  return merged


def load_route_evidence(selection_path, selection_sha, packet_dir, raw_root, dataset_dir, schema_dir, rdf_path, transfer_path):
  """Reconstruct camera/causal input proof from both untouched raw logs before display."""
  spec = route.read_selection(selection_path, selection_sha)
  packet_json, packet_npz = packet_dir / "packet.json", packet_dir / "packet.npz"
  packet_meta = json.loads(packet_json.read_text())
  packet = dict(np.load(packet_npz, allow_pickle=False))
  if sha256(packet_npz) != packet_meta.get("artifact_sha256", {}).get("packet.npz"):
    raise RenderError("route packet NPZ hash changed")
  schema_hashes = {str(p.relative_to(schema_dir)): sha256(p) for p in sorted(schema_dir.rglob("*.capnp"))}
  if schema_hashes != spec["schema_sha256"] or packet_meta.get("schema_sha256") != schema_hashes:
    raise RenderError("route plural schema proof differs")
  for module in (route, route.rf):
    path = Path(module.__file__)
    if packet_meta.get("source_sha256", {}).get(path.name) != sha256(path):
      raise RenderError("route packet builder source changed before consumer verification")
  metadata, raw_contracts, tapes, stream_sets, statics, input_hashes = [], [], [], [], [], {}
  executed = rdf_path.with_suffix(".executed-adapter.py")
  paths = [selection_path, packet_json, packet_npz, rdf_path, rdf_path.with_suffix(".json"), transfer_path, executed,
           *schema_dir.rglob("*.capnp"), Path(route.__file__), Path(route.rf.__file__)]
  schema = load_schema(schema_dir)
  for chunk in spec["source_chunks"]:
    source = raw_root / chunk["segment"]
    log, dataset = source / "rlog.zst", dataset_dir / (chunk["segment"] + ".npz")
    metadata_path = dataset.with_suffix(".json")
    meta = json.loads(metadata_path.read_text())
    if (sha256(log) != chunk["source_log_sha256"] or Path(meta["source_log"]).resolve() != log.resolve()
        or meta["schema_sha256"] != schema_hashes):
      raise RenderError("route raw source log/schema path binding differs")
    record_index = len(metadata)
    records = packet_meta.get("source_segments", [])
    if record_index >= len(records) or Path(records[record_index].get("source_segment", "")).resolve() != source.resolve():
      raise RenderError("route packet source path/order differs")
    paths.extend((log, dataset, metadata_path))
    for name in ("fcamera.hevc", "ecamera.hevc"):
      camera = source / name
      selected = next(row for row in spec["camera_allowlist"] if row["relative_path"] == chunk["segment"] + "/" + name)
      if camera.is_symlink() or source.is_symlink() or camera.stat().st_size != selected["inventory_bytes"] or sha256(camera) != selected["sha256"]:
        raise RenderError("route camera source size/SHA/path differs")
      paths.append(camera)
    streams, static = route.rf.read_log(log, schema_dir, int(source.name.rsplit("--", 1)[1]))
    raw_contracts.append(route._raw_static(log, schema, meta))
    stream_sets.append(streams)
    statics.append(static)
    metadata.append(meta)
    tape = hold_calibration(dict(np.load(dataset, allow_pickle=False)))
    tape["model_capture_t"] = model_capture_times(tape, meta)
    tapes.append(tape)
  pairs = route.select_route_pairs(spec, stream_sets)
  meta_hashes = [sha256(dataset_dir / (c["segment"] + ".json")) for c in spec["source_chunks"]]
  validate_sources(spec, packet_meta, metadata, raw_contracts, meta_hashes, selection_sha, pairs[0]["t"])
  validate_mapping(spec, packet, pairs)
  merged, duplicates = route.merge_streams(stream_sets)
  past = [r["t"] for c in raw_contracts for r in c["car_params_snapshots"] if r["t"] <= pairs[0]["t"]]
  static = {**statics[0], "cp": {**statics[0]["cp"], "publication_t": max(past)}}
  reconstructed = route.rf.packet_inputs(pairs, merged, static)
  if any(key not in packet or not np.array_equal(values, packet[key]) for key, values in reconstructed.items()):
    raise RenderError("route packet causal inputs differ from actual plural raw sources")
  data = merge_tapes(tapes)
  rdf = dict(np.load(rdf_path, allow_pickle=False))
  bench, transfer = (json.loads(path.read_text()) for path in (rdf_path.with_suffix(".json"), transfer_path))
  hashes = {"packet_json": sha256(packet_json), "packet_npz": sha256(packet_npz),
            "rdf_npz": sha256(rdf_path), "rdf_json": sha256(rdf_path.with_suffix(".json")), "executed_adapter": sha256(executed)}
  validate_rdf_binding(rdf, bench, transfer, packet, packet_meta, hashes)
  from controller_model_overlay import validate_rdf_output
  validate_rdf_output(rdf, data)
  censor = float(spec["controlled_prefix"]["raw_censor_t"])
  masks = controlled_masks(rdf, censor)
  gt_horizon = np.arange(61) * .05
  masks["controlled_3s_gt_valid"] = controlled_gt_mask(rdf["narrow_t"], gt_horizon, censor, masks["controlled_admissible"])
  masks["controlled_3s_gt_horizon"] = gt_horizon
  input_hashes.update({str(path.resolve()): sha256(path) for path in paths})
  proof = {"multi_source": True, "selection_sha256": selection_sha, "input_sha256": input_hashes,
           "source_segments": packet_meta["source_segments"], "packet_hashes": hashes,
           "raw_censor_t": censor, "controlled_capture_frames_max": spec["controlled_prefix"]["controlled_capture_frames"],
           "controlled_outputs": int(masks["controlled_admissible"].sum()), "raw_duplicate_counts": duplicates,
           "full_controlled_3s_gt_windows": int(np.all(masks["controlled_3s_gt_valid"], axis=1).sum()),
           "controlled_metrics": "None computed; availability mask is not an outcome score; future GT censored at raw brake",
           "view_motion_rows": len(data["t"]), "view_raw_capture_rows": len(data["raw_encode_t"]),
           "view_scope": "Plural original telemetry/captures; display-only measured future, no model/controller inputs or modified outcome"}
  return spec, packet, data, metadata[0], rdf, masks, proof
