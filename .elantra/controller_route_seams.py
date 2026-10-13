#!/usr/bin/env python3
"""Explicit analysis-only repair of two proven TRAIN route prefix seams.

Original extraction and its20Hz grid remain immutable. Only absent-asof prefix
entries can be reconstructed from actual adjacent raw events; no observed invalid
event, stale value or unknown capture origin is replaced. No controller is tuned.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

import controller_extract as extract
import controller_stops as stops
from controller_tune import cohort_key, enrich_route_metadata
import rdf_route_frames as route_frames
import rdf_frames as frames

COHORT = "f9c0b765b31b"
ALLOWED_PAIRS = (("000001b1--18b1c22fda--24", "000001b1--18b1c22fda--25"),
                 ("000001b7--1972f5d799--25", "000001b7--1972f5d799--26"))


class SeamError(RuntimeError):
  pass


def value_hash(value):
  array = np.ascontiguousarray(value)
  return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def same_sample(a, b):
  return (a.valid == b.valid and a.values.keys() == b.values.keys() and
          all(np.array_equal(np.asarray(a.values[k]),np.asarray(b.values[k]),equal_nan=True) for k in a.values))


def repair_prefix(original, previous_samples, current_samples, max_prefix_s=1.):
  """Return a copy and exact raw-event field receipts, without adding fields.

  Absence requires original age/valid entries NaN AND no current-log service event
  at/before that exact row. Latest actual event must be valid and causal; LLK
  measurement/calibration state must also be valid. Dynamic freshness stays200ms;
  only explicit persistent calibration permits500ms. Existing finite age entries
  (including stale/invalid observations) and raw EncodeIdx arrays remain untouched.
  """
  if not np.isfinite(max_prefix_s) or not 0 < max_prefix_s <= 1:
    raise SeamError("prefix budget must be finite, positive and at most1s")
  out = {key: value.copy() for key, value in original.items()}
  t = out["t"]
  if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
    raise SeamError("original strictly increasing finite20Hz grid required")
  repairs, refusals, absent_fields, skipped = [], [], [], Counter()
  candidates = np.flatnonzero(t - t[0] <= max_prefix_s + 1e-9)
  for service, fields in extract.SERVICE_FIELDS.items():
    age_key, valid_key = "age_" + service, extract.VALID_FIELDS[service]
    if age_key not in out or valid_key not in out:
      absent_fields.append({"service": service, "reason": "original_age_or_valid_contract_absent"})
      continue
    previous = sorted((s for s in previous_samples if s.service == service), key=lambda s: s.t)
    current = sorted((s for s in current_samples if s.service == service), key=lambda s: s.t)
    merged = sorted([(s, "previous") for s in previous] + [(s, "current") for s in current], key=lambda pair: pair[0].t)
    current_times, source_times = [s.t for s in current], [pair[0].t for pair in merged]
    max_age = .5 if service == "extrinsicsCalibration" else .2
    for i in candidates:
      if not np.isnan(out[age_key][i]):
        skipped[service + ":existing_age"] += 1
        continue
      reason, source = None, None
      if np.isfinite(out[valid_key][i]):
        reason = "existing_validity_observation"
      elif bisect_right(current_times, t[i]):
        reason = "current_log_already_has_source_event"
      else:
        index = bisect_right(source_times, t[i]) - 1
        if index < 0:
          reason = "no_causal_source_event"
        else:
          source, source_label = merged[index]
          age = float(t[i] - source.t)
          equal_time = merged[bisect_left(source_times,source.t):index+1]
          if any(not same_sample(source,other) for other,_ in equal_time):
            reason = "conflicting_latest_source_events"
          elif not source.valid:
            reason = "latest_source_event_invalid"
          elif not 0 <= age <= max_age + 1e-9:
            reason = "latest_source_event_stale"
          elif service == "liveLocationKalman" and source.values.get("location_measurement_valid") != 1:
            reason = "latest_location_measurement_invalid"
          elif service == "extrinsicsCalibration" and source.values.get("calibrated") != 1:
            reason = "latest_calibration_state_invalid"
      if reason:
        refusals.append({"row": int(i), "t": float(t[i]), "service": service, "reason": reason,
                         "source_t": None if source is None else float(source.t)})
        continue
      values = {age_key: age, valid_key: float(source.valid)}
      for field in fields:
        if field not in out or field not in source.values:
          absent_fields.append({"row": int(i), "service": service, "field": field, "reason": "original_or_raw_field_absent"})
          continue
        raw = np.asarray(source.values[field])
        target = np.asarray(out[field][i])
        if target.ndim == 0:
          if raw.ndim != 0 or (np.issubdtype(raw.dtype, np.integer) and abs(int(raw)) > 2**53 - 1):
            raise SeamError("source scalar shape or exact integer representation differs")
          values[field] = raw
        else:
          if raw.ndim != 1 or target.ndim != 1:
            raise SeamError("source vector shape differs from canonical padded field")
          padded = np.full(target.shape, np.nan)
          count = min(len(raw), len(target))  # Same padding contract as align_samples.
          padded[:count] = raw[:count]
          values[field] = padded
      for field, value in values.items():
        before = np.asarray(out[field][i]).copy()
        raw = np.asarray(value)
        mask = np.isnan(before) & np.isfinite(raw)
        if not np.any(mask):
          continue
        after = np.where(mask, raw, before)
        out[field][i] = after
        repairs.append({"row": int(i), "t": float(t[i]), "service": service, "field": field,
          "source_t": float(source.t), "source_label": source_label, "source_age_s": age, "max_age_s": max_age,
          "event_valid": bool(source.valid), "measurement_valid": source.values.get("location_measurement_valid"),
          "before_sha256": value_hash(before), "after_sha256": value_hash(np.asarray(out[field][i])),
          "positions": np.flatnonzero(np.atleast_1d(mask)).tolist()})
  affected = sorted({row["row"] for row in repairs})
  if affected:
    subset = {key: value[affected].copy() for key, value in out.items() if value.ndim and value.shape[0] == len(t)}
    # max_age=.2 disables additional calibration holds. Reuse only its exact
    # validity formulas at the explicitly repaired rows, without copying fields.
    validity = extract.hold_calibration(subset, max_age=.2)
    for key in ("valid_lat", "valid_long", "valid"):
      out[key][affected] = validity[key]
  return out, {"affected_rows": affected, "repairs": repairs, "refusals": refusals, "absent_fields": absent_fields,
               "skipped": dict(skipped), "max_prefix_s": max_prefix_s, "dynamic_max_age_s": .2, "calibration_max_age_s": .5,
               "capture_origin_policy": "raw EncodeIdx mappings unchanged; unknown model capture origin remains unknown"}


def verify_pair(metadata, contracts, train_routes, first_t):
  if len(metadata) != 2 or len(contracts) != 2 or tuple(m["segment"] for m in metadata) not in ALLOWED_PAIRS:
    raise SeamError("only the two explicit adjacent training pairs are admitted")
  if (metadata[0]["route"] != metadata[1]["route"] or metadata[0]["route"] not in train_routes or
      any(m.get("fingerprint") != "HYUNDAI_ELANTRA_2024" or m.get("op_long") is not True for m in metadata)):
    raise SeamError("same frozen training route and exact car/control contract required")
  try:
    route_frames.same_contract(metadata)
    route_frames.same_full_contract(contracts)
  except frames.PacketError as error:
    raise SeamError(str(error)) from error
  for key in ("car_params_snapshots", "car_params_sp_snapshots"):
    rows = [row for proof in contracts for row in proof[key]]
    if rows and not any(np.isfinite(row["t"]) and row["t"] <= first_t for row in rows):
      raise SeamError("no actual full CP/SP snapshot preceding repair origin")


def output_directory(out, raw_root, dataset):
  out = out.resolve()
  if any(out.is_relative_to(path.resolve()) for path in (raw_root, dataset)):
    raise SeamError("derived output cannot alias raw sources or immutable dataset")
  if out.exists() and (not out.is_dir() or any(out.iterdir())):
    raise SeamError("derived output must be new or empty")
  out.mkdir(parents=True, exist_ok=True)
  return out


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def derive(dataset, raw_root, schema_dir, splits_path, out):
  splits = json.loads(splits_path.read_text(encoding="utf-8"))
  train = splits["cohorts"][COHORT]["split"]["train"]
  schema_hashes = {str(p.relative_to(schema_dir)): extract.sha256(p) for p in sorted(schema_dir.rglob("*.capnp"))}
  schema = extract.load_schema(schema_dir)
  inputs = {splits_path: extract.sha256(splits_path)}
  pending = []
  for pair in ALLOWED_PAIRS:
    metas, contracts, samples, proofs, target = [], [], [], [], None
    for number, name in enumerate(pair):
      sidecar, npz = dataset / (name + ".json"), dataset / (name + ".npz")
      meta = json.loads(sidecar.read_text(encoding="utf-8"))
      log = raw_root / name / "rlog.zst"
      if (cohort_key(meta) != COHORT or meta["schema_sha256"] != schema_hashes or
          Path(meta["source_log"]).resolve() != log.resolve() or extract.sha256(log) != meta["source_log_sha256"]):
        raise SeamError("actual source/schema/cohort differs from frozen metadata")
      if meta["route"] not in train:
        raise SeamError("heldout route refused before opening arrays")
      for path in (sidecar, npz, log):
        inputs[path] = extract.sha256(path)
      decoded, info = extract.decode_log(log, schema)
      if (info["source_log_sha256"] != meta["source_log_sha256"] or info["camera_sensor"] != meta["camera_sensor"] or
          info["init_data"].get("deviceType") != "mici" or info["init_data"].get("deviceType") != meta["init_data"].get("deviceType")):
        raise SeamError("source/camera hardware changed or metadata differs while decoding")
      samples.append(decoded)
      contracts.append(route_frames._raw_static(log, schema, meta))
      metas.append(meta)
      proofs.append({"segment": name, "source_log": str(log.resolve()), "source_log_sha256": inputs[log],
                     "original_npz": str(npz.resolve()), "original_npz_sha256": inputs[npz], "metadata_sha256": inputs[sidecar]})
      with np.load(npz, allow_pickle=False) as archive:
        if number == 0:
          previous_end = float(archive["t"][-1])
        else:
          target = {key: archive[key] for key in archive.files}
    verify_pair(metas, contracts, train,first_t=float(target["t"][0]))
    if not 0 < target["t"][0] - previous_end <= .075 + 1e-8:
      raise SeamError("actual grid join is not a contiguous<=75ms boundary")
    derived, receipt = repair_prefix(target, *samples)
    receipt.update(source_proof=[{**proof, "full_static_snapshots": contract} for proof, contract in zip(proofs, contracts, strict=True)],
                   previous_end_t=previous_end, first_current_t=float(target["t"][0]),
                   changed_array_keys=[key for key in target if not np.array_equal(target[key], derived[key], equal_nan=True)])
    pending.append((metas[1], derived, receipt))
  output = output_directory(out, raw_root, dataset)
  results = []
  for meta, data, receipt in pending:
    name = meta["segment"]
    npz, sidecar, proof_path = (output / (name + suffix) for suffix in (".npz", ".json", ".repair.json"))
    np.savez_compressed(npz, **data)
    receipt.update(derived_npz_sha256=extract.sha256(npz), original_grid_unchanged=True,
                   original_raw_arrays_unchanged=True, production_enabled=False)
    write_json(proof_path, receipt)
    derived_meta = {**meta, "diagnostic_only": True, "production_enabled": False, "original_summary": meta["summary"],
      "derived_route_seam": {"receipt": str(proof_path.resolve()), "receipt_sha256": extract.sha256(proof_path),
        "source_proof": receipt["source_proof"], "helper_sha256": extract.sha256(Path(__file__)),
        "scope": "existing missing-asof prefix fields only; unchanged original grid/raw capture mapping; no tune or trusted plant"},
      "summary": {**meta["summary"], **{key + "_rows": int(data[key].sum()) for key in ("valid", "valid_lat", "valid_long")}}}
    write_json(sidecar, derived_meta)
    results.append({"segment": name, "npz": str(npz.resolve()), "npz_sha256": extract.sha256(npz),
                    "metadata_sha256": extract.sha256(sidecar), "receipt_sha256": extract.sha256(proof_path),
                    "repaired_rows": len(receipt["affected_rows"]), "repaired_fields": len(receipt["repairs"]),
                    "refusals": len(receipt["refusals"])})
  if any(extract.sha256(path) != digest for path, digest in inputs.items()):
    raise SeamError("original source/data/split changed during derived analysis")
  manifest = {"status": "complete_derived_analysis", "results": results, "original_inputs_sha256": {str(p.resolve()): h for p,h in inputs.items()},
    "original_files_unchanged": True, "schema_sha256": schema_hashes, "splits_sha256": inputs[splits_path],
    "source_sha256": {p.name: extract.sha256(p) for p in (Path(__file__), Path(extract.__file__), Path(route_frames.__file__))},
    "diagnostic_only": True, "production_enabled": False, "plant_or_gain_identification": False,
    "scope": "only two frozen TRAIN seams; no old dataset, raw file, manifest or split rewritten"}
  write_json(output / "manifest.json", manifest)
  return manifest


def evaluation_members(paths):
  """Same static route-context step as the original census's training_paths."""
  enrich_route_metadata(paths)
  members = [(path,meta) for path,meta in paths if cohort_key(meta) == COHORT]
  excluded = [{"segment":meta["segment"],"cohort":cohort_key(meta),"reason":"different_or_unknown_cohort"}
              for _,meta in paths if cohort_key(meta) != COHORT]
  return members,excluded


def evaluate(dataset, derived_dir, splits_path, original_census, out):
  """Two-route amendment only; preserve original census and all interventions."""
  if out.exists() or out.resolve().is_relative_to(dataset.resolve()) or out.resolve().is_relative_to(derived_dir.resolve()):
    raise SeamError("amendment must be a new separate output; originals/derived receipts immutable")
  manifest_path = derived_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("status") != "complete_derived_analysis" or extract.sha256(splits_path) != manifest["splits_sha256"]:
    raise SeamError("complete exact derived receipt and unchanged split required")
  if any(extract.sha256(Path(path)) != digest for path,digest in manifest["original_inputs_sha256"].items()):
    raise SeamError("immutable original changed since repair")
  expected = {row["segment"]: row for row in manifest["results"]}
  if set(expected) != {pair[1] for pair in ALLOWED_PAIRS}:
    raise SeamError("exact two derived source segments required")
  census = json.loads(original_census.read_text(encoding="utf-8"))
  splits = json.loads(splits_path.read_text(encoding="utf-8"))
  routes = [pair[0].rsplit("--",1)[0] for pair in ALLOWED_PAIRS]
  amendments, evaluation_inputs = [], {original_census:extract.sha256(original_census),manifest_path:extract.sha256(manifest_path)}
  for route in routes:
    if route not in splits["cohorts"][COHORT]["split"]["train"]:
      raise SeamError("heldout route refused")
    paths = []
    for path in sorted(dataset.glob(route + "--*.npz")):
      selected = derived_dir / path.name if path.stem in expected else path
      meta_path = selected.with_suffix(".json")
      meta = json.loads(meta_path.read_text(encoding="utf-8"))
      evaluation_inputs[selected],evaluation_inputs[meta_path] = extract.sha256(selected),extract.sha256(meta_path)
      if path.stem in expected:
        row = expected[path.stem]
        if extract.sha256(selected) != row["npz_sha256"] or extract.sha256(meta_path) != row["metadata_sha256"]:
          raise SeamError("derived array/metadata changed since repair")
      paths.append((selected,meta))
    paths,excluded = evaluation_members(paths)
    if not paths:
      raise SeamError("no admitted same-cohort route data; empty results refused")
    blocks, coverage = stops.stitch_route(paths, calibration_hold=True)
    events = [event for block in blocks for event in stops.stop_events(block, route, True)]
    before = [event for event in census["events"] if event["route"] == route]
    amendments.append({"route": route, "source_segments": [path.stem for path,_ in paths], "coverage": coverage,"excluded_sources":excluded,
      "static_context_inheritance": [{"segment":meta["segment"],"source_segment":meta["route_provenance_source"],
        "fields":meta["route_context_inherited_fields"]} for _,meta in paths if "route_provenance_source" in meta],
      "original_events": before, "derived_events": events, "original_count": len(before), "derived_count": len(events),
      "original_active_count": sum(e["long_active_at_prediction"]==1 for e in before),
      "derived_active_count": sum(e["long_active_at_prediction"]==1 for e in events),
      "derived_comma_only_completed": sum(e["comma_only_eligible"] for e in events)})
  if any(extract.sha256(path) != digest for path,digest in evaluation_inputs.items()):
    raise SeamError("evaluation source/derived inputs changed")
  result = {"scope": "only two TRAIN routes rerun from explicit derived seams; original65-row census retained; no global recount or stopline GT",
    "derived_manifest_sha256": extract.sha256(manifest_path), "original_census_sha256": extract.sha256(original_census),
    "source_sha256": {p.name:extract.sha256(p) for p in (Path(__file__),Path(stops.__file__),Path(extract.__file__))},
    "evaluation_inputs_sha256":{str(path.resolve()):digest for path,digest in evaluation_inputs.items()},
    "amendments": amendments, "diagnostic_only": True, "production_enabled": False, "fit_or_tune_performed": False}
  write_json(out,result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command",required=True)
  for command in (sub.add_parser("derive"),sub.add_parser("evaluate")):
    command.add_argument("--dataset",type=Path,required=True)
    command.add_argument("--splits",type=Path,required=True)
    command.add_argument("--out",type=Path,required=True)
  sub.choices["derive"].add_argument("--raw-root",type=Path,required=True)
  sub.choices["derive"].add_argument("--schema-dir",type=Path,required=True)
  sub.choices["evaluate"].add_argument("--derived-dir",type=Path,required=True)
  sub.choices["evaluate"].add_argument("--original-census",type=Path,required=True)
  args = parser.parse_args()
  if args.command == "derive":
    result = derive(args.dataset,args.raw_root,args.schema_dir,args.splits,args.out)
  else:
    result = evaluate(args.dataset,args.derived_dir,args.splits,args.original_census,args.out)
  print(json.dumps({"command":args.command,"out":str(args.out.resolve()),"status":result.get("status","diagnostic_amendment")}))


if __name__ == "__main__":
  main()
