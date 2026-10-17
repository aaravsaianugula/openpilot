"""Export exact causal logged calibration heights for road-projected visualization only."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_heights(manifest: dict, calibrations: dict[int, dict]) -> list[dict]:
    result = []
    for frame in manifest["frames"]:
        timestamp = frame["context_sources"]["extrinsicsCalibration"]
        if timestamp > min(frame["narrow_timestamp_ns"], frame["wide_timestamp_ns"]):
            raise ValueError("Visualization calibration cannot use future information")
        calibration = calibrations[timestamp]
        height = calibration["height"]
        if (
            calibration["valid"] is not True
            or calibration["status"] != "calibrated"
            or calibration["rpy"] != list(frame["rpy_calib"])
            or len(height) != 1
            or type(height[0]) not in (int, float)
            or not math.isfinite(height[0])
            or height[0] <= 0
        ):
            raise ValueError("Exact valid logged height and matching calibration are required; no nominal fallback")
        result.append(
            {
                key: frame[key]
                for key in (
                    "source_id",
                    "ordinal",
                    "narrow_frame_id",
                    "wide_frame_id",
                    "narrow_timestamp_ns",
                    "wide_timestamp_ns",
                )
            }
            | {"calibration_timestamp_ns": timestamp, "height_m": height[0]}
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rlog", required=True, type=Path, action="append")
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--opendbc-car", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import capnp
    import zstandard

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    if digest(args.schema) != provenance["schema_sha256"]:
        raise ValueError("Archive schema differs from the inference context")
    actual_rlogs = {path.parent.name: digest(path) for path in args.rlog}
    if actual_rlogs != provenance["source_rlogs"] or len(actual_rlogs) != len(args.rlog):
        raise ValueError("Exact same source rlogs are required")
    capnp.remove_import_hook()
    schema = capnp.load(str(args.schema), imports=[str(args.opendbc_car)])
    calibrations = {}
    for path in args.rlog:
        raw = zstandard.ZstdDecompressor().decompress(path.read_bytes(), max_output_size=512 * 1024**2)
        for event in schema.Event.read_multiple_bytes(raw):
            if event.which() == "extrinsicsCalibration":
                timestamp = int(event.logMonoTime)
                if timestamp in calibrations:
                    raise ValueError("Ambiguous calibration event identity")
                data = event.extrinsicsCalibration
                calibrations[timestamp] = dict(
                    valid=bool(event.valid),
                    status=str(data.calStatus),
                    rpy=list(data.rpyCalib),
                    height=list(data.height),
                )
    frames = join_heights(manifest, calibrations)
    result = {
        "schema": "logged_camera_height_v1",
        "manifest_sha256": digest(args.manifest),
        "extractor_sha256": digest(Path(__file__)),
        "source_rlogs": actual_rlogs,
        "schema_sha256": digest(args.schema),
        "usage": "visualization_only_native_UI_model_z_plus_logged_height_no_bumper_or_default_height",
        "physical_calibration_verified": False,
        "frames": frames,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "frames": len(frames)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
