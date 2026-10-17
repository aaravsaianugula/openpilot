"""Export causal native context at each original dual-camera SOF for offline companion replay.

Logged model/radar leads are comparators, never ground truth. Logged calibration
is retained without asserting independent physical validation. No VIN, CAN, cabin
data, nominal camera height, future observation or renamed-service fallback is
exported. Invalid latest messages shadow earlier valid messages.

GNSS (gpsLocationExternal, and liveLocationKalman position/orientation with their
std) is exported because map look-ahead and trusted local time have no other
input. It is private location data: outputs stay under D:/comma_four, never in
``routes/``. Receiver accuracy and Kalman std are estimates, never hard bounds;
a consumer must declare its own bound and qualification before using them.
"""

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from .extract_camera_context import extract as extract_camera_record
from .extract_projection_height import digest as file_sha256

ENCODERS = ("narrowRoadEncodeIdx", "wideRoadEncodeIdx")
CAMERA_IDENTITIES = ("narrowRoadCameraState", "wideRoadCameraState", "deviceState")
ARCHIVE_CAMERA_COMMIT = "e748a4dda814dd0dd171fe76f2feab2399bd429d"
ARCHIVE_CAMERA_SHA256 = "1de3f9e6147f28195c71673ae5c8ec38e65ede997ac98d2e1e3b114eda873ec4"
CAMERA_PRODUCERS = {
    ARCHIVE_CAMERA_COMMIT: ARCHIVE_CAMERA_SHA256,
    # Exact current camera.py is byte-identical; device/sensor/raster checks still apply.
    "13aedc522518fe1b62651d9a8a320e931efc335c": ARCHIVE_CAMERA_SHA256,
}
MAX_STREAM_BYTES = 2 * 1024**3
PRIVATE_ROOT = Path("D:/comma_four")
ROUTES_ROOT = PRIVATE_ROOT / "routes"
# u-blox fix as published by ubloxd; units per cereal log.capnp GpsLocationData.
GNSS_FIELDS = (
    "latitude",  # degree
    "longitude",  # degree
    "speed",  # m/s
    "bearingDeg",  # degree, course over ground
    "horizontalAccuracy",  # m, receiver estimate
    "bearingAccuracyDeg",  # degree, receiver estimate
    "speedAccuracy",  # m/s, receiver estimate
    "unixTimestampMillis",  # receiver UTC, ms
    "hasFix",
    "source",
)
# LLK pose fields beyond the yaw contract: optional, so a build or recorded index
# without them keeps the yaw service known and leaves the pose unknown.
LLK_POSE_FIELDS = ("gpsOK", "positionGeodetic", "positionECEF", "orientationNED")
FIELDS = {
    "carState": (
        "vEgo",
        "aEgo",
        "leftBlinker",
        "rightBlinker",
        "gasPressed",
        "brakePressed",
        "steeringPressed",
        "standstill",
        "vCruise",
        "vCruiseCluster",
    ),
    "liveLocationKalman": (
        "angularVelocityCalibrated",
        "status",
        "inputsOK",
        "posenetOK",
        "gpsOK",
        "positionGeodetic",
        "positionECEF",
        "orientationNED",
    ),
    "gpsLocationExternal": GNSS_FIELDS,
    "modelV2": (
        "frameId",
        "frameIdExtra",
        "timestampEof",
        "position",
        "laneLines",
        "laneLineProbs",
        "laneLineStds",
        "roadEdges",
        "roadEdgeStds",
        "meta",
        "action",
        "leadsV3",
        "confidence",
    ),
    "extrinsicsCalibration": ("rpyCalib", "rpyCalibSpread", "calStatus", "height", "wideFromDeviceEuler"),
    "longitudinalPlan": ("aTarget", "shouldStop", "hasLead", "longitudinalPlanSource", "modelMonoTime"),
    "carControl": ("enabled", "longActive", "actuators", "cruiseControl"),
    "carOutput": ("actuatorsOutput",),
    "selfdriveState": ("enabled", "active", "state", "personality", "experimentalMode"),
    "radarState": ("leadOne", "leadTwo", "mdMonoTime", "radarErrors"),
    "radarTracks": ("points", "errors"),
    "carParams": (
        "carFingerprint",
        "flags",
        "radarUnavailable",
        "openpilotLongitudinalControl",
        "longitudinalActuatorDelay",
    ),
    "modelDataV2SP": ("laneTurnDirection", "leftLaneChangeEdgeBlock", "rightLaneChangeEdgeBlock"),
}
REQUIRED = {
    "carState": FIELDS["carState"][:8],
    "liveLocationKalman": FIELDS["liveLocationKalman"][:4],
    "gpsLocationExternal": GNSS_FIELDS[:9],
    "modelV2": (
        "frameId",
        "frameIdExtra",
        "timestampEof",
        "position",
        "laneLines",
        "laneLineProbs",
        "roadEdges",
        "meta",
        "action",
        "leadsV3",
    ),
    "extrinsicsCalibration": ("rpyCalib", "calStatus", "height"),
    "longitudinalPlan": ("aTarget", "shouldStop", "hasLead", "longitudinalPlanSource"),
    "carControl": FIELDS["carControl"],
    "carOutput": FIELDS["carOutput"],
    "selfdriveState": FIELDS["selfdriveState"],
    "radarState": FIELDS["radarState"],
    "radarTracks": FIELDS["radarTracks"],
    "carParams": FIELDS["carParams"],
    "modelDataV2SP": FIELDS["modelDataV2SP"],
    "onroadEvents": ("events",),
}
# Explicit export freshness policy, not a claim that these are road-qualified limits.
MAX_AGE_NS = dict.fromkeys(REQUIRED, 200_000_000)
MAX_AGE_NS.update(
    modelV2=250_000_000,
    radarState=250_000_000,
    modelDataV2SP=250_000_000,
    extrinsicsCalibration=2_000_000_000,
    onroadEvents=2_000_000_000,
    carParams=300_000_000_000,
)
XYZT = ("x", "y", "z", "t", "xStd", "yStd", "zStd")
LEAD = (
    "present",
    "dRel",
    "yRel",
    "vRel",
    "vLead",
    "vLeadK",
    "aLeadK",
    "aLeadTau",
    "modelProb",
    "radar",
    "radarTrackId",
)
MODEL_LEAD = ("prob", "probTime", "t", "x", "xStd", "y", "yStd", "v", "vStd", "a", "aStd")


def pick(data: dict, fields: tuple[str, ...]) -> dict:
    return {field: data[field] for field in fields if field in data}


def native_value(value):
    if type(value).__name__ == "_DynamicEnum":
        try:
            return str(value)
        except RuntimeError:
            return {"unknown_enum_raw": int(value.raw)}
    if type(value) in (str, int, float, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {key: native_value(child) for key, child in value.items()}
    if hasattr(value, "schema") and hasattr(value.schema, "fields"):
        return {key: native_value(getattr(value, key)) for key in value.schema.fields}
    return [native_value(child) for child in value]


def native_payload(reader, fields: tuple[str, ...]) -> dict:
    if hasattr(reader, "schema"):
        return {field: native_value(getattr(reader, field)) for field in fields if field in reader.schema.fields}
    return pick(reader.to_dict(), fields)


def json_native(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {key: json_native(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_native(child) for child in value]
    return value


def extract_record(event, segment: str, event_index: int) -> dict | None:
    service = event.which()
    if service in ENCODERS:
        selected = extract_camera_record(event)
        data = selected["data"]
    elif service in CAMERA_IDENTITIES:
        identity_fields = (
            ("deviceType",) if service == "deviceState" else ("frameId", "timestampSof", "timestampEof", "sensor")
        )
        data = native_payload(getattr(event, service), identity_fields)
    elif service == "onroadEvents":
        fields = ("name", "overrideLongitudinal", "userDisable", "softDisable", "immediateDisable", "noEntry")
        data = {"events": [native_payload(item, fields) for item in event.onroadEvents]}
    elif service in FIELDS:
        data = native_payload(getattr(event, service), FIELDS[service])
        if service == "modelV2":
            for field in ("position",):
                if field in data:
                    data[field] = pick(data[field], XYZT)
            for field in ("laneLines", "roadEdges"):
                if field in data:
                    data[field] = [pick(item, XYZT) for item in data[field]]
            if "meta" in data:
                data["meta"] = pick(
                    data["meta"],
                    ("desirePrediction", "desireState", "laneChangeState", "laneChangeDirection", "hardBrakePredicted"),
                )
            if "action" in data:
                data["action"] = pick(data["action"], ("desiredAcceleration", "shouldStop", "desiredCurvature"))
            if "leadsV3" in data:
                data["leadsV3"] = [pick(item, MODEL_LEAD) for item in data["leadsV3"]]
        elif service == "radarState":
            for field in ("leadOne", "leadTwo"):
                if field in data:
                    data[field] = pick(data[field], LEAD)
        elif service == "radarTracks" and "points" in data:
            points = []
            for original in data["points"]:
                point = pick(original, ("trackId", "dRel", "yRel", "vRel"))
                # Native CAMERA_SCC deliberately logs NaN lateral position. Preserve that
                # unavailability explicitly; never repair it to a numeric lane position.
                if isinstance(point.get("yRel"), float) and not math.isfinite(point["yRel"]):
                    point["unknown_fields"] = {"yRel": f"logged_nonfinite_{point['yRel']}"}
                    point["yRel"] = None
                points.append(point)
            data["points"] = points
        elif service in ("carControl", "carOutput"):
            for field in ("actuators", "actuatorsOutput"):
                if field in data:
                    data[field] = pick(data[field], ("accel", "longControlState"))
            if "cruiseControl" in data:
                data["cruiseControl"] = pick(data["cruiseControl"], ("cancel", "resume", "override"))
    else:
        return None
    return {
        "segment": segment,
        "event_index": event_index,
        "log_mono_ns": int(event.logMonoTime),
        "service": service,
        "valid": bool(event.valid),
        "data": json_native(data),
    }


def source_of(record: dict) -> dict:
    source = {field: record[field] for field in ("segment", "event_index", "log_mono_ns", "service", "valid")}
    if len(record.get("equivalent_sources", [])) > 1:
        source["equivalent_sources"] = record["equivalent_sources"]
    return source


def payload_issue(service: str, data: dict) -> str | None:
    def tagged_unknown(value, tag: str) -> bool:
        if isinstance(value, dict):
            return tag in value or any(tagged_unknown(child, tag) for child in value.values())
        return isinstance(value, list) and any(tagged_unknown(child, tag) for child in value)

    if tagged_unknown(data, "nonfinite_float"):
        return "nonfinite_native_payload"
    if tagged_unknown(data, "unknown_enum_raw"):
        return "unknown_native_enum"
    if service in ("carControl", "carOutput"):
        field = "actuators" if service == "carControl" else "actuatorsOutput"
        if any(key not in data[field] for key in ("accel", "longControlState")):
            return "missing_native_nested_fields"
    elif service == "radarState":
        if any(key not in data[field] for field in ("leadOne", "leadTwo") for key in LEAD):
            return "missing_native_nested_fields"
    elif service == "onroadEvents":
        if any("overrideLongitudinal" not in item for item in data["events"]):
            return "missing_native_nested_fields"
    elif service == "modelV2":
        curves = [data["position"], *data["laneLines"], *data["roadEdges"]]
        if not data["laneLines"] or not data["roadEdges"] or not data["leadsV3"]:
            return "model_geometry_or_leads_unavailable"
        if any(any(key not in curve for key in ("x", "y", "z", "t")) for curve in curves):
            return "missing_native_nested_fields"
        if any(len(curve["x"]) < 2 or len({len(curve[key]) for key in ("x", "y", "z", "t")}) != 1 for curve in curves):
            return "model_geometry_invalid"
        if len(data["laneLineProbs"]) != len(data["laneLines"]):
            return "model_geometry_invalid"
        if any(
            key not in data["meta"]
            for key in ("desireState", "desirePrediction", "laneChangeState", "laneChangeDirection")
        ):
            return "missing_native_nested_fields"
        if any(key not in data["action"] for key in ("desiredAcceleration", "shouldStop")):
            return "missing_native_nested_fields"
        if any(any(key not in lead for key in MODEL_LEAD) for lead in data["leadsV3"]):
            return "missing_native_nested_fields"
    return None


def snapshot(record: dict | None, service: str, timestamp: int) -> dict:
    result: dict
    if record is None:
        result = {"status": "unknown", "reason": "no_causal_message", "age_ns": None, "source": None, "data": None}
    else:
        data = record["data"]
        age = timestamp - record["log_mono_ns"]
        if age < 0:
            raise ValueError("Future observation cannot enter causal context")
        missing = [field for field in REQUIRED[service] if field not in data]
        reason = "known"
        if record["valid"] is not True:
            reason = "event_invalid"
        elif age > MAX_AGE_NS[service]:
            reason = "message_stale"
        elif missing:
            reason = "missing_native_fields"
        elif (issue := payload_issue(service, data)) is not None:
            reason = issue
        elif service == "extrinsicsCalibration":
            if data["calStatus"] != "calibrated" or len(data["rpyCalib"]) != 3:
                reason = "calibration_unavailable"
            elif len(data["height"]) != 1 or data["height"][0] <= 0:
                reason = "calibration_height_unavailable"
        elif service == "liveLocationKalman":
            measurement = data["angularVelocityCalibrated"]
            if (
                data["status"] != "valid"
                or data["inputsOK"] is not True
                or data["posenetOK"] is not True
                or measurement.get("valid") is not True
                or len(measurement.get("value", [])) != 3
            ):
                reason = "yaw_measurement_invalid"
        elif service == "gpsLocationExternal":
            if data["hasFix"] is not True:
                reason = "gnss_fix_unavailable"
        elif service == "modelV2":
            capture_age = timestamp - data["timestampEof"]
            if capture_age < 0:
                reason = "source_capture_in_future"
            elif capture_age > 300_000_000:
                reason = "source_capture_stale"
        result = {
            "status": "known" if reason == "known" else "unknown",
            "reason": reason,
            "age_ns": age,
            "source": source_of(record),
            "data": data,
            "missing_fields": missing,
        }
        if service == "modelV2" and "timestampEof" in data:
            result["source_capture_age_ns"] = timestamp - data["timestampEof"]
    if service == "liveLocationKalman":
        result["yaw_rate_radps"] = (
            result["data"]["angularVelocityCalibrated"]["value"][2] if result["status"] == "known" else None
        )
        result["yaw_source_field"] = "liveLocationKalman.angularVelocityCalibrated.value[2]"
    return result


def iter_context(manifest: dict, records: list[dict]) -> Iterator[dict]:
    by_service = defaultdict(list)
    identities: dict[tuple[str, int], dict] = {}
    for record in records:
        service = record["service"]
        if service not in REQUIRED:
            continue
        identity = (service, record["log_mono_ns"])
        if identity in identities:
            previous = identities[identity]
            if previous["data"] != record["data"] or previous["valid"] is not record["valid"]:
                raise ValueError(f"Ambiguous native message timestamp: {identity}")
            previous["equivalent_sources"].append(source_of(record))
            continue
        selected = dict(record, equivalent_sources=[source_of(record)])
        identities[identity] = selected
        by_service[service].append(selected)
    for entries in by_service.values():
        entries.sort(key=lambda record: record["log_mono_ns"])
    times = {service: [record["log_mono_ns"] for record in entries] for service, entries in by_service.items()}
    previous_timestamp = -1
    for ordinal, original in enumerate(manifest["frames"]):
        frame = pick(
            original,
            ("source_id", "ordinal", "narrow_frame_id", "wide_frame_id", "narrow_timestamp_ns", "wide_timestamp_ns"),
        )
        if frame["ordinal"] != ordinal:
            raise ValueError("Original frame ordinals must be complete and chronological")
        timestamp = min(frame["narrow_timestamp_ns"], frame["wide_timestamp_ns"])
        if timestamp <= previous_timestamp:
            raise ValueError("Original camera SOF is not chronological")
        previous_timestamp = timestamp
        frame["reference_timestamp_ns"] = timestamp
        frame["camera_skew_ns"] = abs(frame["narrow_timestamp_ns"] - frame["wide_timestamp_ns"])
        frame["context"] = {}
        for service in REQUIRED:
            index = bisect.bisect_right(times.get(service, []), timestamp) - 1
            selected_record = by_service[service][index] if index >= 0 else None
            frame["context"][service] = snapshot(selected_record, service, timestamp)
        yield frame


def join_context(manifest: dict, records: list[dict]) -> list[dict]:
    return list(iter_context(manifest, records))


def validate_camera_identity(manifest: dict, records: list[dict]) -> list[dict]:
    encodes = {}
    for record in records:
        if record["service"] not in ENCODERS:
            continue
        key = (record["segment"], record["service"], record["data"]["segmentId"])
        if key in encodes:
            raise ValueError("Ambiguous camera encode identity")
        encodes[key] = record
    identities = []
    for frame in manifest["frames"]:
        segments = [
            item
            for item in manifest["source_segments"]
            if item["global_start_ordinal"] <= frame["ordinal"] < item["global_start_ordinal"] + item["frame_count"]
        ]
        if len(segments) != 1:
            raise ValueError("Camera identity has no unique original segment")
        segment = segments[0]
        local_ordinal = frame["ordinal"] - segment["global_start_ordinal"]
        sources = {}
        for camera, service in zip(("narrow", "wide"), ENCODERS, strict=True):
            encode_record = encodes.get((segment["source_id"], service, local_ordinal))
            if (
                encode_record is None
                or encode_record["valid"] is not True
                or encode_record["data"]["frameId"] != frame[f"{camera}_frame_id"]
                or encode_record["data"]["timestampSof"] != frame[f"{camera}_timestamp_ns"]
            ):
                raise ValueError("Camera identity differs from the original manifest/source rlog")
            sources[camera] = source_of(encode_record)
        identities.append(sources)
    return identities


def probe_video(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required to verify original video dimensions and decoded frame counts")
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,codec_name,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    if completed.stderr.strip():
        raise ValueError(f"Original video decode diagnostics: {completed.stderr.strip()}")
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1 or streams[0]["codec_name"] != "hevc":
        raise ValueError("Expected one original HEVC video stream")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_read_frames"]),
        "command": command,
        "exit_code": completed.returncode,
    }


def camera_geometry(commit: str, records: list[dict], video_info: list[dict], source: Path) -> dict:
    valid = [record for record in records if record["valid"] is True]
    devices = {record["data"].get("deviceType") for record in valid if record["service"] == "deviceState"}
    sensors = {
        service: {record["data"].get("sensor") for record in valid if record["service"] == service}
        for service in CAMERA_IDENTITIES[:2]
    }
    result = {
        "status": "unknown",
        "reason": "unsupported_or_unverified_archive_camera_contract",
        "device_types": sorted(devices, key=str),
        "sensors": {key: sorted(value, key=str) for key, value in sensors.items()},
        "physical_calibration_verified": False,
        "intrinsics": None,
    }
    if not source.is_file():
        return result
    result["camera_source"] = {"path": str(source), "sha256": file_sha256(source), "commit": commit}
    if (
        CAMERA_PRODUCERS.get(commit) != file_sha256(source)
        or devices != {"mici"}
        or any(sensor != {"os04c10"} for sensor in sensors.values())
    ):
        return result
    # Exact pinned archive camera.py: _os_config lines45-48, CameraConfig.intrinsics lines20-25,
    # and DEVICE_CAMERAS mici/os04c10 mapping lines70-71. These are source nominal intrinsics.
    if any((info["width"], info["height"]) != (1344, 760) for info in video_info):
        result["reason"] = "encoded_geometry_differs_no_verified_resize_convention"
        return result
    result.update(
        status="source_supported_nominal_intrinsics",
        reason="pinned_native_device_sensor_mapping",
        intrinsics={
            "narrow": [[1141.5, 0.0, 672.0], [0.0, 1141.5, 380.0], [0.0, 0.0, 1.0]],
            "wide": [[425.25, 0.0, 672.0], [0.0, 425.25, 380.0], [0.0, 0.0, 1.0]],
        },
    )
    return result


def build_camera_manifest(
    segment_paths: list[Path],
    records: list[dict],
    commit: str,
    schema_hash: str,
    source_rlogs: dict[str, str],
    camera_source: Path,
) -> dict:
    if not segment_paths:
        raise ValueError("At least one video segment rlog is required")
    names = [path.parent.name.rsplit("--", 1) for path in segment_paths]
    if len({name[0] for name in names}) != 1 or [int(name[1]) for name in names] != list(
        range(int(names[0][1]), int(names[0][1]) + len(names))
    ):
        raise ValueError("Video segments must be consecutive and ordered within one route")
    route = names[0][0]
    source_id = f"{route}-{names[0][1]}to{names[-1][1]}--0"
    frames: list[dict] = []
    source_segments: list[dict] = []
    infos: list[dict] = []
    for path in segment_paths:
        selected = {
            service: sorted(
                [
                    record
                    for record in records
                    if record["segment"] == path.parent.name and record["service"] == service
                ],
                key=lambda item: item["data"]["segmentId"],
            )
            for service in ENCODERS
        }
        count = len(selected[ENCODERS[0]])
        if count == 0 or any(
            len(entries) != count or [record["data"]["segmentId"] for record in entries] != list(range(count))
            for entries in selected.values()
        ):
            raise ValueError("Complete unique original dual-camera encoded identities are required")
        source_segment = {"source_id": path.parent.name, "global_start_ordinal": len(frames), "frame_count": count}
        for camera, filename in (("narrow", "fcamera.hevc"), ("wide", "ecamera.hevc")):
            video = path.parent / filename
            info = probe_video(video)
            if info["frame_count"] != count:
                raise ValueError("Original decoded video count differs from encode identities")
            infos.append(info)
            source_segment.update(
                {
                    f"local_{camera}_path": str(video),
                    f"{camera}_path": str(video),
                    f"{camera}_sha256": file_sha256(video),
                    f"{camera}_video_info": info,
                }
            )
        source_segments.append(source_segment)
        for narrow, wide in zip(selected[ENCODERS[0]], selected[ENCODERS[1]], strict=True):
            if narrow["valid"] is not True or wide["valid"] is not True:
                raise ValueError("Invalid original camera encoder event")
            frame: dict = {"source_id": source_id, "ordinal": len(frames)}
            for camera, record in (("narrow", narrow), ("wide", wide)):
                frame[f"{camera}_frame_id"] = record["data"]["frameId"]
                frame[f"{camera}_timestamp_ns"] = record["data"]["timestampSof"]
                if frames:
                    previous = frames[-1]
                    if (
                        frame[f"{camera}_frame_id"] != previous[f"{camera}_frame_id"] + 1
                        or not 40_000_000
                        <= frame[f"{camera}_timestamp_ns"] - previous[f"{camera}_timestamp_ns"]
                        <= 60_000_000
                    ):
                        raise ValueError("Original camera identity or timing is discontinuous")
            if abs(frame["narrow_timestamp_ns"] - frame["wide_timestamp_ns"]) > 10_000_000:
                raise ValueError("Original dual-camera skew exceeds native model's diagnostic threshold")
            frames.append(frame)
    geometry = camera_geometry(commit, records, infos, camera_source)
    dimensions = {(info["width"], info["height"]) for info in infos}
    if len(dimensions) != 1:
        raise ValueError("Original encoded camera dimensions differ")
    width, height = dimensions.pop()
    intrinsics = geometry["intrinsics"]
    return {
        "schema": "companion_camera_manifest_v1",
        "source_id": source_id,
        "camera_width": width,
        "camera_height": height,
        "geometry": geometry,
        "narrow_intrinsics": intrinsics["narrow"] if intrinsics is not None else None,
        "wide_intrinsics": intrinsics["wide"] if intrinsics is not None else None,
        "provenance": {"archive_git_commit": commit, "schema_sha256": schema_hash, "source_rlogs": source_rlogs},
        "source_segments": source_segments,
        "frames": frames,
    }


def summary(frames: Iterable[dict]) -> dict:
    reasons: dict[str, Counter] = {service: Counter() for service in REQUIRED}
    recorded_ages: dict[str, list[int]] = {service: [] for service in REQUIRED}
    for frame in frames:
        for service in REQUIRED:
            item = frame["context"][service]
            reasons[service][item["reason"]] += 1
            if item["age_ns"] is not None:
                recorded_ages[service].append(item["age_ns"])
    result = {}
    for service in REQUIRED:
        ages = sorted(recorded_ages[service])
        result[service] = {
            "reasons": dict(reasons[service]),
            "message_age_ns": (
                {
                    "min": ages[0],
                    "p50": ages[int((len(ages) - 1) * 0.50)],
                    "p95": ages[int((len(ages) - 1) * 0.95)],
                    "p99": ages[int((len(ages) - 1) * 0.99)],
                    "max": ages[-1],
                }
                if ages
                else None
            ),
        }
    return result


def write_context_index(output: Path, metadata: dict, frames: Iterable[dict], *, target_bytes: int = 1024**2) -> dict:
    """Write exact frame objects once; publish the index only after complete bounded parts."""
    if output.exists():
        raise FileExistsError("Output already exists; preserve prior evidence")
    if metadata.get("schema") != "companion_context_v1" or "frames" in metadata:
        raise ValueError("Stream metadata must describe the existing frame schema without frames")
    if type(target_bytes) is not int or not 1 <= target_bytes <= 256 * 1024**2:
        raise ValueError("Invalid context part target")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    buffer = bytearray()
    part_count = 0
    count = 0
    total_bytes = 0

    def flush() -> None:
        nonlocal part_count
        if not buffer:
            return
        name = f"context-part-{len(parts):06d}.jsonl"
        with (output.parent / name).open("xb") as handle:
            handle.write(buffer)
        parts.append(
            {
                "path": name,
                "bytes": len(buffer),
                "sha256": hashlib.sha256(buffer).hexdigest(),
                "start_ordinal": count - part_count,
                "frame_count": part_count,
            }
        )
        buffer.clear()
        part_count = 0

    def writing_frames() -> Iterator[dict]:
        nonlocal count, part_count, total_bytes
        for frame in frames:
            if type(frame.get("ordinal")) is not int or frame["ordinal"] != count or count >= 500000:
                raise ValueError("Context frames must retain complete cumulative ordinals within bounded capacity")
            if "source_id" in metadata and frame.get("source_id") != metadata["source_id"]:
                raise ValueError("Context frame source identity differs from metadata")
            line = (json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if len(line) > 256 * 1024**2:
                raise ValueError("Single original context frame exceeds bounded part capacity")
            if total_bytes + len(line) > MAX_STREAM_BYTES:
                raise ValueError("Context stream exceeds bounded immutable input capacity")
            if buffer and len(buffer) + len(line) > target_bytes:
                flush()
            buffer.extend(line)
            part_count += 1
            count += 1
            total_bytes += len(line)
            yield frame
        flush()
        if count == 0:
            raise ValueError("Empty context stream")

    result = dict(metadata, schema="companion_context_index_v1", summary=summary(writing_frames()))
    result.update(frame_count=count, parts=parts)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))
    return result


def private_output(path: Path) -> Path:
    """The context carries GNSS: it stays in the drive-data tree, never inside routes/."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PRIVATE_ROOT.resolve()) or resolved.is_relative_to(ROUTES_ROOT.resolve()):
        raise ValueError("GNSS-bearing context must be written under D:/comma_four, outside routes/")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rlog", type=Path, action="append", default=[])
    parser.add_argument("--segment-rlog", type=Path, action="append", default=[])
    parser.add_argument("--prefix-rlog", type=Path, action="append", default=[])
    parser.add_argument("--camera-manifest-output", type=Path)
    parser.add_argument("--camera-source", type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--opendbc-car", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stream-context", action="store_true", help="Write bounded JSONL parts and a context index")
    args = parser.parse_args()
    import capnp  # type: ignore[import-untyped]  # replay environment only; no stubs
    import zstandard  # type: ignore[import-not-found]  # replay environment only

    if args.output.exists() or (args.camera_manifest_output is not None and args.camera_manifest_output.exists()):
        raise FileExistsError("Output already exists; preserve prior evidence")
    private_output(args.output)
    manifest = None
    schema_hash = file_sha256(args.schema)
    if args.manifest is not None:
        if args.segment_rlog or args.prefix_rlog or args.camera_manifest_output:
            raise ValueError("Existing manifest mode and standalone segment mode cannot be mixed")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest["schema"] not in ("mountain_dew_replay_v1", "companion_camera_manifest_v1"):
            raise ValueError("Unsupported original camera manifest")
        if schema_hash != manifest["provenance"]["schema_sha256"]:
            raise ValueError("Archive schema differs from original camera context")
        paths = args.rlog
        for segment in manifest["source_segments"]:
            for camera in ("narrow", "wide"):
                if file_sha256(Path(segment[f"local_{camera}_path"])) != segment[f"{camera}_sha256"]:
                    raise ValueError("Original video digest differs from manifest")
    else:
        if args.rlog or not args.segment_rlog or args.camera_manifest_output is None:
            raise ValueError("Standalone mode needs --segment-rlog and --camera-manifest-output")
        paths = [*args.prefix_rlog, *args.segment_rlog]
        if args.prefix_rlog:
            first_route, first_segment = args.segment_rlog[0].parent.name.rsplit("--", 1)
            if args.prefix_rlog[-1].parent.name != f"{first_route}--{int(first_segment) - 1}":
                raise ValueError("Prefix must immediately precede the first video segment")
    source_rlogs = {path.parent.name: file_sha256(path) for path in paths}
    if not paths or len(source_rlogs) != len(paths):
        raise ValueError("Unique original rlogs are required")
    if manifest is not None and source_rlogs != manifest["provenance"]["source_rlogs"]:
        raise ValueError("Exact original prefix and segment rlogs are required")
    capnp.remove_import_hook()
    schema = capnp.load(str(args.schema), imports=[str(args.opendbc_car)])
    expected_services = set(REQUIRED) | set(ENCODERS) | set(CAMERA_IDENTITIES) | {"initData"}
    missing_services = expected_services - set(schema.Event.schema.fields)
    if missing_services:
        raise ValueError(f"Archive schema lacks exact native services: {sorted(missing_services)}")
    records = []
    counts = {}
    archive_commits = set()
    for path in paths:
        raw = zstandard.ZstdDecompressor().decompress(path.read_bytes(), max_output_size=512 * 1024**2)
        service_counts: Counter[str] = Counter()
        commits = []
        for index, event in enumerate(schema.Event.read_multiple_bytes(raw)):
            service = event.which()
            service_counts[service] += 1
            if service == "initData":
                commits.append(str(event.initData.gitCommit))
            record = extract_record(event, path.parent.name, index)
            if record is not None:
                records.append(record)
        if len(commits) != 1:
            raise ValueError(f"Archive source commit mismatch/ambiguity in {path.parent.name}")
        archive_commits.add(commits[0])
        counts[path.parent.name] = dict(service_counts)
    if len(archive_commits) != 1:
        raise ValueError("Original rlogs span different source builds")
    commit = archive_commits.pop()
    if manifest is not None and commit != manifest["provenance"]["archive_git_commit"]:
        raise ValueError("Archive source commit differs from original camera manifest")
    records.sort(key=lambda record: record["log_mono_ns"])
    manifest_path = args.manifest
    if manifest is None:
        camera_source = args.camera_source or args.schema.parent.parent / "common" / "transformations" / "camera.py"
        manifest = build_camera_manifest(args.segment_rlog, records, commit, schema_hash, source_rlogs, camera_source)
        manifest["extractor_sha256"] = file_sha256(Path(__file__))
        manifest_path = args.camera_manifest_output
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, allow_nan=False, separators=(",", ":"))
    identities = validate_camera_identity(manifest, records)
    frames = (
        dict(frame, camera_identity_sources=identity)
        for frame, identity in zip(iter_context(manifest, records), identities, strict=True)
    )
    schema_paths = [*args.schema.parent.glob("*.capnp"), args.opendbc_car / "car.capnp"]
    result = {
        "schema": "companion_context_v1",
        "source_id": manifest["source_id"],
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(manifest_path),
        "archive_git_commit": commit,
        "source_rlogs": source_rlogs,
        "source_segments": manifest["source_segments"],
        "schema_sha256": file_sha256(args.schema),
        "schema_files": {str(path): file_sha256(path) for path in sorted(set(schema_paths))},
        "extractor_sha256": file_sha256(Path(__file__)),
        "helper_sha256": {
            str(Path("tools") / name): file_sha256(Path(__file__).parent / name)
            for name in ("extract_camera_context.py", "extract_projection_height.py", "prepare_big_replay_context.py")
        },
        "reader_versions": {name: importlib.metadata.version(name) for name in ("pycapnp", "zstandard")},
        "usage": "offline_replay_context_only_no_vehicle_actuation",
        "join_policy": "latest_logMonoTime_at_or_before_earliest_dual_camera_SOF_invalid_latest_shadows_good",
        "camera_identity_policy": "encode_events_only_identify_original_images_not_causal_perception_inputs",
        "model_prediction_policy": (
            "native_predicted_future_trajectories_available_at_message_time_not_future_observations"
        ),
        "lead_role": "native_radar_and_model_leads_are_comparators_not_ground_truth",
        "raw_track_role": "OEM_SCC_lead_range_rate_may_feed_radarTracks_not_independent_physical_radar_proof",
        "physical_calibration_verified": False,
        "calibration_role": "logged_calibrated_rpy_and_height_if_present_no_nominal_fallback",
        "freshness_policy_ns": MAX_AGE_NS,
        "gnss_policy": "private_location_receiver_estimates_not_hard_bounds",
        "service_counts": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stream_context:
        result = write_context_index(args.output, result, frames)
    else:
        materialized_frames = list(frames)
        result.update(summary=summary(materialized_frames), frames=materialized_frames)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, allow_nan=False, separators=(",", ":"))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "frames": len(manifest["frames"]),
                "summary": result["summary"],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
