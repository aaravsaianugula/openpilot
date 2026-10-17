"""Extract only camera/model context from archived rlogs, retaining source clocks.

No VIN, GPS, cabin images or CAN payloads are exported. Events are sorted by
logMonoTime, never consumed in per-service file order. No missing field is filled
with a guessed value; the benchmark must validate coverage before inference.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def extract(event) -> dict | None:
    kind = event.which()
    fields = {
        "narrowRoadCameraState": ("frameId", "timestampSof", "timestampEof", "sensor", "transform"),
        "wideRoadCameraState": ("frameId", "timestampSof", "timestampEof", "sensor", "transform"),
        "narrowRoadEncodeIdx": ("frameId", "segmentNum", "segmentId", "timestampSof", "timestampEof"),
        "wideRoadEncodeIdx": ("frameId", "segmentNum", "segmentId", "timestampSof", "timestampEof"),
        "extrinsicsCalibration": ("rpyCalib", "calStatus"),
        "deviceState": ("deviceType",),
        "driverMonitoringState": ("isRHD",),
        "carState": ("vEgo", "leftBlinker", "rightBlinker", "steeringPressed"),
        "lateralDelay": ("lateralDelay",),
        "carParams": ("longitudinalActuatorDelay",),
        "lateralPlanDEPRECATED": ("desire",),
    }
    if kind == "modelV2":
        # Logged model lane-change state is provenance, not a replacement desire input.
        meta = event.modelV2.meta
        payload = {name: str(getattr(meta, name)) for name in ("laneChangeState", "laneChangeDirection")}
    elif kind == "modelDataV2SP":
        payload = {"laneTurnDirection": str(event.modelDataV2SP.laneTurnDirection)}
    elif kind in fields:
        payload = getattr(event, kind).to_dict()
        payload = {name: payload[name] for name in fields[kind] if name in payload}
    else:
        return None
    return {"log_mono_ns": int(event.logMonoTime), "service": kind, "valid": bool(event.valid), "data": payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlog", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--opendbc-car", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import capnp
    import zstandard

    capnp.remove_import_hook()
    schema = capnp.load(str(args.schema), imports=[str(args.opendbc_car)])
    compressed = args.rlog.read_bytes()
    raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=512 * 1024 * 1024)
    records = []
    counts: Counter[str] = Counter()
    archive_git_commit = None
    for event in schema.Event.read_multiple_bytes(raw):
        counts[event.which()] += 1
        if event.which() == "initData":
            if archive_git_commit is not None:
                raise ValueError("Ambiguous archive source commits")
            archive_git_commit = str(event.initData.gitCommit)
        record = extract(event)
        if record is not None:
            records.append(record)
    records.sort(key=lambda record: record["log_mono_ns"])
    result = {
        "segment": args.rlog.parent.name,
        "archive_git_commit": archive_git_commit,
        "rlog_sha256": hashlib.sha256(compressed).hexdigest(),
        "schema_sha256": hashlib.sha256(args.schema.read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "service_counts": dict(counts),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, separators=(",", ":"))
    selected = Counter(record["service"] for record in records)
    print(json.dumps({"output": str(args.output), "selected_counts": dict(selected), "service_counts": dict(counts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
