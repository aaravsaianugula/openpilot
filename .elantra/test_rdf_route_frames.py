"""Exact two-segment packet boundaries, provenance and compressed stream regressions."""
from collections import defaultdict
import copy
import hashlib
import json

import numpy as np
import pytest

import rdf_frames as rf
import rdf_route_frames as route


def selection():
  return {"route": "000001b1--18b1c22fda", "partition": "train", "cohort": "cohort",
    "frames": 276, "warmup_frames": 132, "first_scored_packet_index": 132, "scored_frames": 144,
    "first_scored_frame_id": 30009, "last_scored_frame_id": 30152, "capture_interval": [53.55, 67.3],
    "source_chunks": [
      {"segment": "000001b1--18b1c22fda--24", "frames": 129, "frame_ids": [29877, 30005],
       "road_presentation_indices": [1071, 1199], "wide_presentation_indices": [1071, 1199],
       "source_log_sha256": "a" * 64},
      {"segment": "000001b1--18b1c22fda--25", "frames": 147, "frame_ids": [30006, 30152],
       "road_presentation_indices": [0, 146], "wide_presentation_indices": [0, 146],
       "source_log_sha256": "b" * 64}]}


def camera_streams():
  result = []
  for number, first_id in ((24, 28806), (25, 30006)):
    rows = [{"t": (number - 24) * 60 + i * .05, "valid": True, "frame_id": first_id + i, "segment_id": i}
            for i in range(1200)]
    result.append({"narrowRoadEncodeIdx": rows, "wideRoadEncodeIdx": copy.deepcopy(rows)})
  return result


def test_exact_boundary_preserves_local_index_reset_and132_real_warmup():
  pairs = route.select_route_pairs(selection(), camera_streams())
  assert len(pairs) == 276
  assert pairs[0]["frame_id"] == 29877 and pairs[-1]["frame_id"] == 30152
  assert pairs[128]["road_index"] == 1199 and pairs[129]["road_index"] == 0
  assert pairs[128]["source_num"] == 24 and pairs[129]["source_num"] == 25
  assert pairs[132]["frame_id"] == 30009
  assert np.diff([p["frame_id"] for p in pairs]).tolist() == [1] * 275
  assert pairs[132]["t"] - pairs[0]["t"] == pytest.approx(6.6)


@pytest.mark.parametrize("fault", ["wrong_route", "not_adjacent", "gap", "duplicate_index", "missing_wide", "time_gap", "wrong_bound"])
def test_route_rejects_unproven_source_or_camera_boundary(fault):
  spec, streams = selection(), camera_streams()
  if fault == "wrong_route":
    spec["source_chunks"][1]["segment"] = "000001b2--f5d83a61b3--25"
  elif fault == "not_adjacent":
    spec["source_chunks"][1]["segment"] = "000001b1--18b1c22fda--26"
  elif fault == "gap":
    streams[1]["narrowRoadEncodeIdx"][0]["frame_id"] += 1
  elif fault == "duplicate_index":
    streams[0]["narrowRoadEncodeIdx"][1072]["segment_id"] = 1071
  elif fault == "missing_wide":
    streams[1]["wideRoadEncodeIdx"].pop(0)
  elif fault == "time_gap":
    for rows in streams[1].values():
      for row in rows:
        row["t"] += .1
  elif fault == "wrong_bound":
    spec["source_chunks"][0]["frame_ids"][0] += 1
  with pytest.raises(rf.PacketError):
    route.select_route_pairs(spec, streams)


def test_dynamic_merge_deduplicates_only_identical_timestamp_payloads():
  row = {"t": 1., "valid": True, "rpy": np.array([0., .1, .2])}
  merged, counts = route.merge_streams([{ "extrinsicsCalibration": [row]}, {"extrinsicsCalibration": [copy.deepcopy(row)]}])
  assert len(merged["extrinsicsCalibration"]) == 1 and counts == {"extrinsicsCalibration": 1}
  changed = copy.deepcopy(row)
  changed["rpy"][0] = 1.
  with pytest.raises(rf.PacketError, match="conflicting"):
    route.merge_streams([{ "extrinsicsCalibration": [row]}, {"extrinsicsCalibration": [changed]}])
  assert isinstance(merged, defaultdict)


@pytest.mark.parametrize("field", ["car_params", "car_params_sp", "git_commit", "dirty", "schema_sha256", "params", "camera_sensor"])
def test_full_cp_controller_source_schema_and_hardware_must_match(field):
  first = {"car_params": {"longitudinalTuning": {"kiV": [0.]}}, "car_params_sp": {"flag": 1},
           "git_commit": "a" * 40, "dirty": True, "schema_sha256": {"log.capnp": "b" * 64},
           "params": {"EnforceTorqueControl": "1"}, "camera_sensor": "os04c10"}
  second = copy.deepcopy(first)
  second[field] = "changed"
  with pytest.raises(rf.PacketError, match="contract"):
    route.same_contract([first, second])
  assert route.same_contract([first, copy.deepcopy(first)]) is None


def test_selection_hash_and_score_boundary_cannot_drift(tmp_path):
  path = tmp_path / "selection.json"
  path.write_text(json.dumps(selection()), encoding="utf-8")
  digest = rf.sha256(path)
  assert route.read_selection(path, digest)["frames"] == 276
  path.write_text(json.dumps({**selection(), "warmup_frames": 131}), encoding="utf-8")
  with pytest.raises(rf.PacketError, match="SHA"):
    route.read_selection(path, digest)
  with pytest.raises(rf.PacketError, match="warmup"):
    route.validate_selection({**selection(), "warmup_frames": 131})


def test_outputs_cannot_alias_any_raw_source_root_or_existing_packet(tmp_path):
  raw = tmp_path / "routes"
  sources = [raw / "000001b1--18b1c22fda--24", raw / "000001b1--18b1c22fda--25"]
  for source in sources:
    source.mkdir(parents=True)
  for target in (raw / "packet", sources[0] / "packet", sources[1] / "packet"):
    with pytest.raises(rf.PacketError, match="source"):
      route.route_output(target, sources)
  output = tmp_path / "packet"
  output.mkdir()
  (output / "keep").write_text("existing")
  with pytest.raises(rf.PacketError, match="empty"):
    route.route_output(output, sources)
  assert (output / "keep").read_text() == "existing"


def test_two_compact_chunks_form_one_ordered_native_hash_and_stream(tmp_path, monkeypatch):
  import zstandard

  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "pack_nv12", lambda raw: b"p" + bytes(raw))
  calls = []

  def decode(source, indices, expected, directory, timeout):
    calls.append((str(source), indices.tolist(), expected))
    value = 1 if source.name == "first.hevc" else 2
    return bytearray([value] * expected), {"decode_exit_code": 0, "command": ["ffmpeg", str(source), "pipe:1"]}

  monkeypatch.setattr(rf, "_decode_compact_ram", decode)
  chunks = [{"segment": "first", "source_num": 24, "camera": tmp_path / "first.hevc", "indices": np.array([1198, 1199])},
            {"segment": "second", "source_num": 25, "camera": tmp_path / "second.hevc", "indices": np.array([0, 1, 2])}]
  target = tmp_path / "road.nv12.zst"
  receipt = route.compress_camera_chunks(chunks, target)
  with target.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as raw:
    pixels = raw.read()
  expected = (b"p" + b"\x01" * 6) * 2 + (b"p" + b"\x02" * 6) * 3
  assert pixels == expected and receipt["native_sha256"] == hashlib.sha256(expected).hexdigest()
  assert receipt["native_bytes"] == 35 and len(receipt["chunks"]) == 2
  assert receipt["chunks"][1]["virtual_frame_offset"] == 2
  assert calls == [(str(tmp_path / "first.hevc"), [1198, 1199], 12), (str(tmp_path / "second.hevc"), [0, 1, 2], 18)]
  assert sorted(p.name for p in tmp_path.iterdir()) == ["road.nv12.zst"]


@pytest.mark.parametrize("fault", ["hidden_cp_change", "invalid_cp", "missing_init", "missing_cp", "hidden_sp_change", "missing_sp"])
def test_raw_source_snapshots_cannot_hide_behind_matching_sidecars(tmp_path, monkeypatch, fault):
  from types import SimpleNamespace

  meta = {"git_commit": "a" * 40, "dirty": True, "car_params": {"mass": 1404.}, "car_params_sp": {"flag": 1}}

  def event(name, **values):
    return SimpleNamespace(which=lambda: name, valid=True, logMonoTime=1_000_000_000, **values)

  init = event("initData", initData=SimpleNamespace(gitCommit="a" * 40, dirty=True))
  cp = event("carParams", carParams=SimpleNamespace(to_dict=lambda: {"mass": 1404.}))
  sp = event("carParamsSP", carParamsSP=SimpleNamespace(to_dict=lambda: {"flag": 1}))
  messages = [init, cp, sp]
  if fault == "hidden_cp_change":
    messages.insert(1, event("carParams", carParams=SimpleNamespace(to_dict=lambda: {"mass": 1500.})))
  elif fault == "invalid_cp":
    cp.valid = False
  elif fault == "missing_init":
    messages.remove(init)
  elif fault == "missing_cp":
    messages.remove(cp)
  elif fault == "hidden_sp_change":
    messages.insert(1, event("carParamsSP", carParamsSP=SimpleNamespace(to_dict=lambda: {"flag": 2})))
  elif fault == "missing_sp":
    messages.remove(sp)
  monkeypatch.setattr(route, "decompress_log", lambda _path: b"recorded")
  schema = SimpleNamespace(Event=SimpleNamespace(read_multiple_bytes=lambda *_a, **_kw: iter(messages)))
  with pytest.raises(rf.PacketError):
    route._raw_static(tmp_path / "rlog.zst", schema, meta)


@pytest.mark.parametrize("where", ["within_source", "across_sources"])
def test_full_cp_field_outside_sidecar_subset_cannot_change(where, tmp_path, monkeypatch):
  from types import SimpleNamespace

  assert "steerLimitTimer" not in route.CP_FIELDS
  meta = {"git_commit": "a" * 40, "dirty": True, "car_params": {"mass": 1404.}}

  def schema(timer_values):
    messages = [SimpleNamespace(which=lambda: "initData", valid=True, logMonoTime=0,
                initData=SimpleNamespace(gitCommit="a" * 40, dirty=True))]
    for timer in timer_values:
      messages.append(SimpleNamespace(which=lambda: "carParams", valid=True, logMonoTime=1_000_000_000,
                      carParams=SimpleNamespace(to_dict=lambda value=timer: {"mass": 1404., "steerLimitTimer": value})))
    return SimpleNamespace(Event=SimpleNamespace(read_multiple_bytes=lambda *_a, **_kw: iter(messages)))

  monkeypatch.setattr(route, "decompress_log", lambda _p: b"recorded")
  if where == "within_source":
    with pytest.raises(rf.PacketError, match="full"):
      route._raw_static(tmp_path / "rlog.zst", schema([.5, .7]), meta)
  else:
    first = route._raw_static(tmp_path / "first.zst", schema([.5]), meta)
    second = route._raw_static(tmp_path / "second.zst", schema([.7]), meta)
    with pytest.raises(rf.PacketError, match="full"):
      route.same_full_contract([first, second])


def test_full_static_receipt_keeps_hashes_and_times_without_private_payload(tmp_path, monkeypatch):
  from types import SimpleNamespace

  meta = {"git_commit": "a" * 40, "dirty": False, "car_params": {"mass": 1404.}, "car_params_sp": {"flag": 1}}
  payload = {"mass": 1404., "steerLimitTimer": .5, "carVin": "private-vin", "carFw": [{"fwVersion": b"\x00\xff"}]}
  messages = [SimpleNamespace(which=lambda: "initData", valid=True, logMonoTime=0,
              initData=SimpleNamespace(gitCommit=meta["git_commit"], dirty=False))]
  for time in (2, 1):
    messages.append(SimpleNamespace(which=lambda: "carParams", valid=True, logMonoTime=time * 1_000_000_000,
                    carParams=SimpleNamespace(to_dict=lambda: payload)))
  messages.append(SimpleNamespace(which=lambda: "carParamsSP", valid=True, logMonoTime=1_000_000_000,
                  carParamsSP=SimpleNamespace(to_dict=lambda: {"flag": 1})))
  monkeypatch.setattr(route, "decompress_log", lambda _p: b"recorded")
  schema = SimpleNamespace(Event=SimpleNamespace(read_multiple_bytes=lambda *_a, **_kw: iter(messages)))
  result = route._raw_static(tmp_path / "rlog.zst", schema, meta)
  assert result["car_params_snapshots"] == [{"t": t, "semantic_sha256": route._hash(payload)} for t in (1., 2.)]
  assert result["car_params_sp_snapshots"] == [{"t": 1., "semantic_sha256": route._hash({"flag": 1})}]
  assert "private-vin" not in json.dumps(result) and "fwVersion" not in json.dumps(result)
  route.same_full_contract([result, result])


def test_source_chunks_stop_on_decode_error_and_preserve_failed_stream(tmp_path, monkeypatch):
  monkeypatch.setattr(rf, "WIDTH", 2)
  monkeypatch.setattr(rf, "HEIGHT", 2)
  monkeypatch.setattr(rf, "pack_nv12", lambda raw: b"p" + bytes(raw))

  def decode(source, _indices, expected, _directory, _timeout):
    if source.name == "second.hevc":
      raise rf.PacketError("second source corrupted")
    return bytearray(expected), {"decode_exit_code": 0, "command": ["ffmpeg", str(source)]}

  monkeypatch.setattr(rf, "_decode_compact_ram", decode)
  chunks = [{"segment": "first", "source_num": 24, "camera": tmp_path / "first.hevc", "indices": np.array([0])},
            {"segment": "second", "source_num": 25, "camera": tmp_path / "second.hevc", "indices": np.array([0])}]
  target = tmp_path / "road.nv12.zst"
  with pytest.raises(rf.PacketError, match="second source corrupted"):
    route.compress_camera_chunks(chunks, target)
  assert not target.exists()
  assert len(list(tmp_path.glob("*.failed-*.json"))) == 1


def camera_proof_fixture(tmp_path):
  spec = selection()
  rows, allowlist = [], []
  raw = tmp_path / "routes"
  for chunk in spec["source_chunks"]:
    source = raw / chunk["segment"]
    source.mkdir(parents=True, exist_ok=True)
    for name in ("fcamera.hevc", "ecamera.hevc"):
      path = source / name
      path.write_bytes((chunk["segment"] + name).encode())
      relative = chunk["segment"] + "/" + name
      rows.append({"relative_path": relative, "target": str(path), "bytes": path.stat().st_size, "sha256": rf.sha256(path)})
      allowlist.append({"relative_path": relative, "inventory_bytes": path.stat().st_size})
  spec["camera_allowlist"] = allowlist
  total = sum(row["bytes"] for row in rows)
  receipt = {"status": "complete", "rsync_exit": 0, "dry_run_exit": 0, "selection_sha256": "a" * 64,
             "expected_files": 4, "verified_target_files": 4, "expected_bytes": total, "verified_target_bytes": total,
             "verified_files": rows}
  path = tmp_path / "cameras.json"
  path.write_text(json.dumps(receipt), encoding="utf-8")
  return spec, raw, path, receipt


@pytest.mark.parametrize("fault", [None, "incomplete", "wrong_selection", "corrupt", "extra", "wrong_target"])
def test_verified_camera_allowlist_cannot_be_missing_mixed_or_changed(tmp_path, fault):
  spec, raw, path, receipt = camera_proof_fixture(tmp_path)
  if fault == "incomplete":
    receipt["status"] = "running"
  elif fault == "wrong_selection":
    receipt["selection_sha256"] = "b" * 64
  elif fault == "corrupt":
    camera = raw / receipt["verified_files"][0]["relative_path"]
    camera.write_bytes(b"x" + camera.read_bytes()[1:])
  elif fault == "extra":
    receipt["verified_files"].append(copy.deepcopy(receipt["verified_files"][0]))
  elif fault == "wrong_target":
    receipt["verified_files"][0]["target"] = str(tmp_path / "unrelated.hevc")
  path.write_text(json.dumps(receipt), encoding="utf-8")
  if fault is None:
    assert len(route._camera_receipt(path, "a" * 64, spec, raw)) == 4
  else:
    with pytest.raises(rf.PacketError):
      route._camera_receipt(path, "a" * 64, spec, raw)


def test_complete_packet_keeps_plural_proof_and_unchanged_bench_protocol(tmp_path, monkeypatch):
  from types import SimpleNamespace
  from rdf_bench import validate_replay

  spec, raw, camera_path, proof = camera_proof_fixture(tmp_path)
  schemas, metadata_dir = tmp_path / "schemas", tmp_path / "metadata"
  schemas.mkdir()
  metadata_dir.mkdir()
  (schemas / "log.capnp").write_text("test schema")
  schema_hashes = {"log.capnp": rf.sha256(schemas / "log.capnp")}
  streams = camera_streams()
  statics = []
  meta = {"git_commit": "a" * 40, "dirty": True, "schema_sha256": schema_hashes, "camera_sensor": "os04c10",
          "params": {}, "fingerprint": "HYUNDAI_ELANTRA_2024", "op_long": True, "car_params": {"mass": 1404.}}
  spec["cohort"] = route.cohort_key(meta)
  for i, chunk in enumerate(spec["source_chunks"]):
    source = raw / chunk["segment"]
    log = source / "rlog.zst"
    log.write_bytes(chunk["segment"].encode())
    chunk["source_log_sha256"] = rf.sha256(log)
    sidecar = {**meta, "source_log": str(log), "source_log_sha256": rf.sha256(log)}
    (metadata_dir / (chunk["segment"] + ".json")).write_text(json.dumps(sidecar), encoding="utf-8")
    statics.append({"device": "mici", "narrow_sensor": "os04c10", "wide_sensor": "os04c10",
                    "source_commit": meta["git_commit"], "source_dirty": True,
                    "cp": {"steer_delay": .1, "long_delay": .5, "fingerprint": meta["fingerprint"], "publication_t": i * 60}})
    times = [i * 60 + j * .05 for j in range(1200)]
    for key, extra in {
        "carState": {"v": 10., "left_blinker": False, "right_blinker": False},
        "extrinsicsCalibration": {"rpy": np.zeros(3), "calibrated": True},
        "traffic": {"is_rhd": False}, "desire_state": {"state": "off", "direction": "none"},
        "turn_state": {"turn": "none"}, "lateralDelay": {"delay": .1, "status": "estimated"}}.items():
      streams[i][key] = [{"t": t, "valid": True, **extra} for t in times]
  splits_path = tmp_path / "splits.json"
  splits_path.write_text(json.dumps({"cohorts": {spec["cohort"]: {"split": {"train": [spec["route"]]}}}}), encoding="utf-8")
  spec["frozen_splits_sha256"] = rf.sha256(splits_path)
  selection_path = tmp_path / "selection.json"
  selection_path.write_text(json.dumps(spec), encoding="utf-8")
  proof["selection_sha256"] = rf.sha256(selection_path)
  camera_path.write_text(json.dumps(proof), encoding="utf-8")
  monkeypatch.setattr(route, "load_schema", lambda _p: SimpleNamespace())
  monkeypatch.setattr(route, "_raw_static", lambda *_args: {
    "car_params_snapshots": [{"t": 0., "semantic_sha256": "a" * 64}],
    "car_params_sp_snapshots": [{"t": 0., "semantic_sha256": "b" * 64}]})
  monkeypatch.setattr(rf, "read_log", lambda _p, _s, number: (streams[number-24], statics[number-24]))
  calls = []

  def compress(chunks, target):
    calls.append(chunks)
    target.write_bytes(b"compressed")
    return {"native_sha256": "a" * 64, "native_bytes": 276 * rf.get_nv12_info(rf.WIDTH, rf.HEIGHT)[3],
            "compressed_sha256": rf.sha256(target), "compressed_bytes": target.stat().st_size}

  monkeypatch.setattr(route, "compress_camera_chunks", compress)
  output = tmp_path / "packet"
  packet = route.build(selection_path, rf.sha256(selection_path), camera_path, raw, metadata_dir, schemas, splits_path, output)
  with np.load(output / "packet.npz", allow_pickle=False) as data:
    validate_replay({key: data[key] for key in data.files}, 276)
    assert data["source_num"][128:130].tolist() == [24, 25]
    assert data["road_index"][128:130].tolist() == [1199, 0]
    assert not data["scored_mask"][:132].any() and data["scored_mask"][132:].all()
  assert packet["multi_source"] and len(packet["source_segments"]) == 2
  assert all(source["full_static_snapshots"]["car_params_snapshots"][0]["semantic_sha256"] == "a" * 64
             for source in packet["source_segments"])
  assert packet["source_segment"] == str((raw / spec["source_chunks"][1]["segment"]).resolve())
  assert packet["action_t_basis"] and packet["warmup_frames"] == 132
  assert not packet["recurrent_state_reset_at_boundary"] and not packet["production_enabled"]
  assert len(calls) == 2 and all(len(chunks) == 2 for chunks in calls)
  assert not (output / "road.nv12").exists() and not (output / "wide.nv12").exists()
