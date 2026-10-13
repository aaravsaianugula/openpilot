"""Plural-source identity and takeover censor regressions for route evidence."""
import copy

import numpy as np
import pytest

from controller_model_overlay import RenderError, rdf_path
from controller_route_overlay import (
  controlled_gt_mask, controlled_masks, display_rdf_copy, merge_tapes, validate_mapping, validate_sources, validate_rdf_binding,
)


@pytest.fixture
def episode():
  chunks = [{"segment": f"000001b7--1972f5d799--{num}", "source_log_sha256": str(num)[0] * 64,
             "metadata_sha256": str(num)[1] * 64, "frames": count, "frame_ids": [first, first + count - 1],
             "road_presentation_indices": [local, local + count - 1], "wide_presentation_indices": [local, local + count - 1]}
            for num, count, first, local in ((25, 166, 31038, 1034), (26, 154, 31204, 0))]
  camera = [{"relative_path": f"{c['segment']}/{name}", "sha256": name[0] * 64, "inventory_bytes": 20}
            for c in chunks for name in ("fcamera.hevc", "ecamera.hevc")]
  spec = {"source_chunks": chunks, "frames": 320, "warmup_frames": 132, "camera_allowlist": camera,
          "controlled_prefix": {"raw_censor_t": 108.5, "controlled_capture_frames": 37}}
  records = [{**c, "cameras": {name: {"sha256": name[0] * 64} for name in ("fcamera.hevc", "ecamera.hevc")},
              "full_static_snapshots": {"car_params_snapshots": [{"t": 99., "semantic_sha256": "a" * 64}],
                                        "car_params_sp_snapshots": [{"t": 99., "semantic_sha256": "b" * 64}]}}
             for c in chunks]
  meta = {"multi_source": True, "format_version": 2, "source_segments": records, "selection_sha256": "s" * 64}
  metadata = [{"car_params": {"stopAccel": -2}, "car_params_sp": {"enableGasInterceptor": False}, "git_commit": "c" * 40,
               "dirty": True, "source_log_sha256": c["source_log_sha256"]} for c in chunks]
  packet = {"t": 100 + np.arange(320) * .05, "narrow_t": 100 + np.arange(320) * .05,
            "wide_t": 100 + np.arange(320) * .05, "frame_id": np.arange(31038, 31358),
            "wide_frame_id": np.arange(31038, 31358), "source_chunk_index": np.r_[np.zeros(166, int), np.ones(154, int)],
            "source_num": np.r_[np.full(166, 25), np.full(154, 26)],
            "road_index": np.r_[np.arange(1034, 1200), np.arange(154)],
            "wide_index": np.r_[np.arange(1034, 1200), np.arange(154)], "virtual_frame_index": np.arange(320),
            "scored_mask": np.arange(320) >= 132}
  pairs = [{k: v[i].item() for k, v in packet.items() if k not in ("virtual_frame_index", "scored_mask")} for i in range(320)]
  return spec, meta, metadata, packet, pairs


def test_all_source_contracts_and_actual_past_cp_are_required(episode):
  spec, meta, metadata, packet, _ = episode
  validate_sources(spec, meta, metadata, [c["full_static_snapshots"] for c in meta["source_segments"]],
                   [c["metadata_sha256"] for c in spec["source_chunks"]], "s" * 64, packet["t"][0])


@pytest.mark.parametrize("mutation", ["swapped", "omitted", "camera", "log", "metadata", "full_cp", "future_cp"])
def test_plural_identity_cannot_be_replaced_by_primary_header(episode, mutation):
  spec, meta, metadata, packet, _ = episode
  raw = copy.deepcopy([c["full_static_snapshots"] for c in meta["source_segments"]])
  if mutation == "swapped":
    meta["source_segments"].reverse()
  elif mutation == "omitted":
    meta["source_segments"].pop()
  elif mutation == "camera":
    meta["source_segments"][1]["cameras"]["ecamera.hevc"]["sha256"] = "x" * 64
  elif mutation == "log":
    metadata[1]["source_log_sha256"] = "x" * 64
  elif mutation == "metadata":
    meta["source_segments"][1]["metadata_sha256"] = "x" * 64
  elif mutation == "full_cp":
    raw[1]["car_params_snapshots"][0]["semantic_sha256"] = "x" * 64
  else:
    for i, records in enumerate(raw):
      records["car_params_snapshots"][0]["t"] = 101.
      meta["source_segments"][i]["full_static_snapshots"] = copy.deepcopy(records)
  with pytest.raises(RenderError):
    validate_sources(spec, meta, metadata, raw, [c["metadata_sha256"] for c in spec["source_chunks"]], "s" * 64, packet["t"][0])


def test_exact_local_indices_and_global_ids_are_bound(episode):
  spec, _, _, packet, pairs = episode
  validate_mapping(spec, packet, pairs)


@pytest.mark.parametrize("key", ["road_index", "wide_index", "frame_id", "source_chunk_index", "source_num", "narrow_t"])
def test_swapped_or_changed_packet_mapping_rejected(episode, key):
  spec, _, _, packet, pairs = episode
  packet[key][200] += 1
  with pytest.raises(RenderError, match="mapping"):
    validate_mapping(spec, packet, pairs)


def test_merge_rejects_conflicting_duplicate_and_collapses_identical_rows():
  a = {"t": np.array([1., 1.05]), "v": np.array([5., 6.])}
  b = {"t": np.array([1.05, 1.1]), "v": np.array([6., 7.])}
  np.testing.assert_array_equal(merge_tapes([a, b])["v"], [5, 6, 7])
  b["v"][0] = 9
  with pytest.raises(RenderError, match="conflicting"):
    merge_tapes([a, b])


def test_output_ready_after_brake_is_not_controlled_even_for_earlier_exposure():
  rdf = {"narrow_t": np.array([1., 1.1, 1.2, 1.3]), "output_ready_t": np.array([1.05, 1.25, 1.26, 1.35]),
         "scored_mask": np.ones(4, bool), "controller_admissible": np.ones(4, bool), "deadline_admissible": np.ones(4, bool)}
  masks = controlled_masks(rdf, 1.2)
  np.testing.assert_array_equal(masks["controlled_admissible"], [True, False, False, False])
  np.testing.assert_array_equal(masks["post_censor_capture"], [False, False, True, True])
  assert masks["display_admissible"].all()  # Takeover remains visible, explicitly unscored.


def test_ready_before_brake_does_not_admit_gt_horizon_after_takeover():
  valid = controlled_gt_mask(np.array([1., 1.1]), np.array([0., 1., 2., 3.]), 2.5, np.array([True, False]))
  np.testing.assert_array_equal(valid, [[True, True, False, False], [False, False, False, False]])
  assert not np.any(np.all(valid, axis=1))  # No full3s controlled outcome windows.


def test_ready_output_with_rejected_display_mask_is_never_visible_or_mutated():
  data = {"t": np.array([1., 1.05, 1.1]), "v": np.ones(3), "yaw": np.zeros(3),
          **{key: np.ones(3) for key in ("car_state_valid", "location_valid", "location_measurement_valid")}}
  rdf = {"output_ready_t": np.array([1.01]), "narrow_t": np.array([1.]), "scored_mask": np.ones(1, bool),
         "position": np.array([[[0., 0, 0], [.1, 0, 0], [.2, 0, 0]]]), "horizon": np.array([0., .1, .2])}
  assert rdf_path(rdf, data, 1.05)[1] == 0
  display = display_rdf_copy(rdf, {"display_admissible": np.zeros(1, bool)})
  assert rdf_path(display, data, 1.05)[1] is None
  assert rdf["scored_mask"][0]


@pytest.mark.parametrize("mutation", ["metadata", "warmup_scalar", "warmup_nonboolean", "warmup_corrupt"])
def test_packet_rdf_binding_rejects_primary_only_or_changed_artifact(episode, mutation):
  _, meta, _, packet, _ = episode
  meta["artifact_sha256"] = {"packet.npz": "p" * 64, "road.nv12": "r" * 64, "wide.nv12": "w" * 64}
  rdf = {key: packet[source].copy() for key, source in (("input_t", "t"), ("narrow_t", "narrow_t"),
         ("wide_t", "wide_t"), ("frame_id", "frame_id"), ("wide_frame_id", "wide_frame_id"))}
  rdf.update(scored_mask=packet["scored_mask"], history_scored_mask=packet["scored_mask"], history_warmup=~packet["scored_mask"],
             output_ready_t=packet["t"] + .04, serial_ready_t=packet["t"] + .04)
  adapter = "104d268c1bf946ce5b9c53d65fa066797d8c36371ff370b974504cced0e7a74d"
  proof = {"status": "passed", "no_actuation": True, "no_params_writes": True,
           "adapter_sha256": adapter, "runtime_ref_verified": "f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae", "model_device": "QCOM",
           "production_warmup": {"temporal_state_restored": True, "state_arrays_compared": 14},
           "packet": {"metadata_sha256": "j" * 64, "verified_file_sha256": meta["artifact_sha256"].copy()}}
  transfer = {"status": "verified", "benchmark_exit_code": 0, "transfer_exit_code": 0,
              "sha256": {".npz": "o" * 64, ".json": "b" * 64, ".executed-adapter.py": adapter}}
  hashes = {"packet_json": "j" * 64, "packet_npz": "p" * 64, "rdf_npz": "o" * 64, "rdf_json": "b" * 64, "executed_adapter": adapter}
  validate_rdf_binding(rdf, proof, transfer, packet, meta, hashes)
  if mutation == "metadata":
    proof["packet"]["metadata_sha256"] = "primary-only"
  elif mutation == "warmup_scalar":
    rdf["history_warmup"] = np.array(132)
  elif mutation == "warmup_nonboolean":
    rdf["history_warmup"] = rdf["history_warmup"].astype(np.uint8)
  else:
    rdf["history_warmup"][-1] = True
  with pytest.raises(RenderError, match="binding|history"):
    validate_rdf_binding(rdf, proof, transfer, packet, meta, hashes)
