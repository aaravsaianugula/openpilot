"""Additive TRAIN-only longitudinal observations from retained Motion payloads.

No file loader/writer, joins, filling, fitting or physical admission. The reviewed
caller must verify the original NPZ/metadata/split files and load car_schema from
the exact recorded hash-verified schema snapshot. This pure helper binds those
claims to actual array/metadata content and validates the wire field contract;
opaque hash labels cannot independently authenticate a caller's schema module.
"""
from collections import Counter
import re

import numpy as np

from controller_motion_tape import SEGMENT_RE
from controller_multirate_lateral import identity_hash
from controller_native_lateral import decode_payload

KIND = "motion_longitudinal_payload_view_v1"
PRESERVED = ("t_ns", "event_seq", "frame_seq", "valid", "clock_duplicate", "clock_conflict", "row_usable")
REASONS = {"event_invalid": 1, "clock_duplicate": 2, "clock_conflict": 4, "can_invalid": 8,
           "nonfinite_value": 16, "actuator_struct_absent": 32, "original_row_unusable": 64}
SEMANTICS = {
  "carControl_accel_request": "CarControl.actuators.accel m/s^2; requested command at its own publication clock.",
  "carOutput_accel_reported": "CarOutput.actuatorsOutput.accel m/s^2; independent post-CarController report, not delivered force/acceleration.",
  "carState_can_valid": "Actual CarState.canValid boolean; independent Event.valid retained.",
  "clock": "Original raw uint64 logMonoTime publication/batch ns; acquisition timing unproven. No asof grid or resampling.",
  "invalid_rows": "Retained with reason bits; consumers must reject invalid interpositions across their complete consumed spans.",
  "scalar_presence": "Capnp scalars have no presence bit: actual decoded zero is retained; absent actuator pointer gives NaN/unusable.",
}
_SHA = re.compile(r"[0-9a-f]{64}")
_SERVICES = {"carState": ("CarState", None, "can_valid"), "carControl": ("CarControl", "actuators", "accel_request"),
             "carOutput": ("CarOutput", "actuatorsOutput", "accel_reported")}


class ViewError(ValueError):
  pass


def _hash(value):
  try:
    return identity_hash(value)
  except (TypeError, ValueError, AttributeError) as error:
    raise ViewError("noncanonical/object-array input content") from error


def _admit(arrays, metadata, split, proof, *, partition="train"):
  if partition not in ("train", "validation"):
    raise ViewError("only TRAIN or validation content admission is allowed")
  if not all(isinstance(v, dict) for v in (arrays, metadata, split, proof)):
    raise ViewError("explicit arrays/metadata/split/proof mappings required")
  if (type(metadata.get("source_eligible")) is not bool or not isinstance(metadata.get("eligibility_reasons"), list)
      or any(not isinstance(reason, str) for reason in metadata["eligibility_reasons"])):
    raise ViewError("explicit original source/policy eligibility metadata required; no defaults")
  if (set(split) != {"train", "validation", "test"} or any(not isinstance(v, list) or any(not isinstance(r, str) for r in v) for v in split.values())
      or any(len(v) != len(set(v)) for v in split.values())
      or any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))):
    raise ViewError("frozen disjoint TRAIN/validation/test route split required")
  segment = metadata.get("segment", "")
  match = SEGMENT_RE.fullmatch(segment) if isinstance(segment, str) else None
  if not match or match[1] not in split[partition] or metadata.get("route", match[1]) != match[1]:
    raise ViewError(f"exact flat segment/route must belong to reserved {partition.upper()}, without route aliases")
  for key in ("source_log_sha256", "output_npz_sha256", "splits_sha256"):
    if not isinstance(metadata.get(key), str) or not _SHA.fullmatch(metadata[key]):
      raise ViewError("missing source/Motion NPZ/frozen-split SHA256")
  hashes = metadata.get("schema_sha256")
  if (not isinstance(hashes, dict) or not {"car.capnp", "log.capnp"} <= set(hashes)
      or any(not isinstance(k, str) or not isinstance(v, str) or not _SHA.fullmatch(v) for k, v in hashes.items())
      or not isinstance(metadata.get("cohort"), str) or not metadata["cohort"]):
    raise ViewError("exact recorded schema-hash map and cohort required")
  if any(not isinstance(v, np.ndarray) or v.dtype.hasobject for v in arrays.values()):
    raise ViewError("original nonobject NPZ array content required")
  expected = {"cohort": metadata["cohort"], "segment": segment, "source_log_sha256": metadata["source_log_sha256"],
              "motion_npz_sha256": metadata["output_npz_sha256"], "arrays_sha256": _hash(arrays),
              "metadata_sha256": _hash(metadata), "schema_sha256": dict(hashes), "split_sha256": _hash(split),
              "motion_splits_sha256": metadata["splits_sha256"]}
  if any(proof.get(k) != v for k, v in expected.items()):
    raise ViewError("source/schema/split/Motion-file/content hash binding differs")
  return match[1], expected


def _schema_types(schema):
  """Check the recorded fork's required field ordinals/types before any decode."""
  try:
    types = {service: getattr(schema, spec[0]) for service, spec in _SERVICES.items()}
    can = types["carState"].schema.fields["canValid"]
    if can.proto.ordinal.explicit != 26 or can.proto.slot.type.which() != "bool":
      raise ViewError("recorded CarState.canValid wire contract differs")
    actuator_types = []
    for service, ordinal in (("carControl", 6), ("carOutput", 0)):
      field = types[service].schema.fields[_SERVICES[service][1]]
      accel = field.schema.fields["accel"]
      if (field.proto.ordinal.explicit != ordinal or field.proto.slot.type.which() != "struct"
          or accel.proto.ordinal.explicit != 4 or accel.proto.slot.type.which() != "float32"):
        raise ViewError("recorded actuator acceleration wire contract differs")
      actuator_types.append(field.schema.node.id)
    if actuator_types[0] != actuator_types[1] or any(not callable(t.from_bytes) for t in types.values()):
      raise ViewError("recorded shared actuator/decoder type differs")
  except (AttributeError, KeyError, TypeError) as error:
    raise ViewError("recorded schema lacks required payload types/fields") from error
  return types


def _stream(arrays, service):
  required = (*PRESERVED, "payload", "offsets")
  if any(service+"_"+key not in arrays for key in required):
    raise ViewError(service+" required raw stream/contract missing")
  fields = {key: np.asarray(arrays[service+"_"+key]) for key in required}
  t = fields["t_ns"]
  n = len(t) if t.ndim == 1 else 0
  if (not n or t.dtype != np.uint64 or np.any(t[1:] < t[:-1]) or any(fields[k].shape != t.shape for k in PRESERVED)
      or any(fields[k].dtype.kind != "b" for k in ("valid", "clock_duplicate", "clock_conflict", "row_usable"))
      or any(fields[k].dtype.kind not in "iu" for k in ("event_seq", "frame_seq"))):
    raise ViewError(service+" malformed raw clocks/order/validity")
  duplicates = np.r_[False, t[1:] == t[:-1]] | np.r_[t[1:] == t[:-1], False]
  if not np.array_equal(duplicates, fields["clock_duplicate"]) or np.any(fields["clock_conflict"] & ~duplicates):
    raise ViewError(service+" inconsistent duplicate/conflict markers")
  offsets, blob = fields["offsets"], fields["payload"]
  if (blob.ndim != 1 or blob.dtype != np.uint8 or offsets.shape != (n+1,) or offsets.dtype.kind not in "iu"
      or offsets[0] != 0 or offsets[-1] != len(blob) or np.any(offsets[1:] <= offsets[:-1])):
    raise ViewError(service+" malformed full-payload offsets")
  return fields


def derive_arrays(arrays, metadata, split, *, car_schema, proof):
  """Return new scalar columns + copied original stream markers and a receipt.

  proof keys: cohort,segment,source_log_sha256,motion_npz_sha256,arrays_sha256,
  metadata_sha256,schema_sha256,split_sha256,motion_splits_sha256. The caller
  verifies original files/module authenticity; this function has no file I/O.
  An exclusive new artifact may combine the original arrays with these columns;
  original keys returned here are copied exactly, never mutated or overwritten.
  """
  route, binding = _admit(arrays, metadata, split, proof)
  types = _schema_types(car_schema)
  streams = {service: _stream(arrays, service) for service in _SERVICES}
  output, summaries = {}, {}
  for service, (_, pointer, column) in _SERVICES.items():
    new_keys = (service+"_"+column, service+"_longitudinal_usable", service+"_longitudinal_reason_bits")
    if any(key in arrays for key in new_keys):
      raise ViewError("input already contains derived longitudinal columns")
    stream = streams[service]
    n = len(stream["t_ns"])
    values = np.zeros(n, bool) if pointer is None else np.full(n, np.nan)
    bits = np.zeros(n, np.uint16)
    for i in range(n):
      try:
        payload = decode_payload(stream["payload"], stream["offsets"], i)
        with types[service].from_bytes(payload, traversal_limit_in_words=2**20) as message:
          if pointer is None:
            values[i] = bool(message.canValid)
          elif not message._has(pointer):
            bits[i] |= REASONS["actuator_struct_absent"]
          else:
            values[i] = float(getattr(message, pointer).accel)
      except Exception as error:
        raise ViewError(f"{service} malformed payload at raw row{i}") from error
    masks = {"event_invalid": ~stream["valid"], "clock_duplicate": stream["clock_duplicate"],
             "clock_conflict": stream["clock_conflict"], "original_row_unusable": ~stream["row_usable"]}
    masks["can_invalid" if pointer is None else "nonfinite_value"] = ~values if pointer is None else ~np.isfinite(values)
    for reason, mask in masks.items():
      bits[mask] |= REASONS[reason]
    output.update({service+"_"+k: stream[k].copy() for k in PRESERVED})
    output.update(dict(zip(new_keys, (values, bits == 0, bits), strict=True)))
    counts = Counter({reason: int(np.count_nonzero(bits & flag)) for reason, flag in REASONS.items()})
    summaries[service] = {"rows": n, "usable_rows": int(np.count_nonzero(bits == 0)), "reasons": dict(counts),
      "rejected_rows": [{"index": int(i), "t_ns": int(stream["t_ns"][i]), "reasons": [k for k, v in REASONS.items() if bits[i] & v]}
                        for i in np.flatnonzero(bits)]}
  receipt = {"kind": KIND, "route": route, "cohort": metadata["cohort"], "segment": metadata["segment"],
             "source_log_sha256": metadata["source_log_sha256"], "input_proof": binding,
             "input_arrays_sha256": binding["arrays_sha256"], "output_view_arrays_sha256": _hash(output),
             "schema_module_boundary": "Reviewed caller must load exact hash-verified recorded schema; hash labels are not independent authenticity proof.",
             "input_source_eligible": metadata["source_eligible"], "eligibility_reasons": metadata["eligibility_reasons"].copy(),
             "source_eligible": False, "selection_eligible": False, "production_enabled": False, "candidate": None,
             "semantics": dict(SEMANTICS), "reason_bits": dict(REASONS), "streams": summaries}
  return output, receipt
