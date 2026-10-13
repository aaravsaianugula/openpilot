"""Isolated RDFMV2 QCOM inference; no messaging publishers, params writes or actuation.

Probe mode uses fresh synthetic buffers and proves functionality only. Replay mode
reads packet.npz/packet.json and memory-mapped road.nv12/wide.nv12 from --frames-dir.
Those frames must use production get_nv12_info() padding, recorded warp transforms,
camera timestamps, desire, traffic convention, action_t and measured speed.
"""

import argparse
import hashlib
import inspect
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

NAME = "driving_rdf_model_v2_august_11_2026_tinygrad.pkl"
WHOLE_SHA = "52fcf48bfb991f327a8982037eb0855d9a63437d78e9f4828d2be54df0f32567"
CHUNK_SHA = (
  "daa108d1e79cc488e0df40a518ec57db5002418c7f3e90cff39c9f4033af6098",
  "08678a51a38bf7c6fe86ac69f13bedbc497093cc415e47ce81b0eab8ee2c5c76",
  "88ff671dc77085cb7d39f5ceda9da2c9f2b2c30ef011eac2a0a095a8c2412aa8",
)
INPUT_SHAPES = {"img": (1, 12, 128, 256), "big_img": (1, 12, 128, 256), "desire_pulse": (1, 33, 8),
                "traffic_convention": (1, 2), "action_t": (1, 2), "features_buffer": (1, 32, 512)}
HISTORY_WARMUP_FRAMES = 132
BENCH_ROOT = Path("/data/rdf-controller-tuning")


class BenchError(RuntimeError):
  pass


def hash_chunks(paths, expected_chunks, expected_whole):
  whole, entries = hashlib.sha256(), []
  if len(paths) != len(expected_chunks):
    raise BenchError("wrong artifact chunk count")
  for path, expected in zip(paths, expected_chunks, strict=True):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
      while block := handle.read(1024 * 1024):
        digest.update(block)
        whole.update(block)
    if digest.hexdigest() != expected:
      raise BenchError(f"artifact SHA256 mismatch: {path.name}")
    entries.append({"file": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
  if whole.hexdigest() != expected_whole:
    raise BenchError("whole artifact SHA256 mismatch")
  return {"chunks": entries, "whole_sha256": whole.hexdigest()}


def bundle_payload():
  return {"internalName": "RDFMV2", "displayName": "RDF Model V2 (August 11, 2026)",
          "ref": "35703097905a122c9f3ddf0d12889b4873d7e2a2", "index": 72, "generation": 12,
          "is20hz": True, "minimumSelectorVersion": 19, "runner": "tinygrad", "environment": "development",
          "overrides": [{"key": "lat", "value": ".1"}, {"key": "long", "value": ".3"}]}


def check_snapshot(snapshot, max_temp_c):
  if not snapshot["offroad"] or snapshot["onroad"]:
    raise BenchError("offroad guard failed")
  if snapshot["battery_percent"] < 35:
    raise BenchError("battery below 35 percent")
  if snapshot["active_processes"]:
    raise BenchError("modeld/camera process active: " + ", ".join(snapshot["active_processes"]))
  temps = snapshot["temperatures_c"]
  if not temps or not np.isfinite(list(temps.values())).all() or max(temps.values()) >= max_temp_c:
    raise BenchError("thermal guard failed")
  # Owner explicitly confirmed wall power despite sysfs reporting offline.
  # external_power remains evidence and is deliberately not an acceptance gate.


def read_snapshot():
  def read(path):
    try:
      return Path(path).read_text().strip()
    except FileNotFoundError:
      return None

  power = {name: read(f"/sys/class/power_supply/{name}/online") for name in ("usb", "dc")}
  battery = read("/sys/class/power_supply/battery/capacity")
  temps = {}
  for folder in Path("/sys/class/thermal").glob("thermal_zone*"):
    kind = read(folder / "type") or ""
    if (kind.startswith("cpu") and kind.endswith("-usr")) or kind in {"gpu0-usr", "gpu1-usr", "ddr-usr", "compute-hvx-usr", "bottom_soc"}:
      raw = read(folder / "temp")
      if raw is not None:
        temps[kind] = float(raw) / 1000.
  processes = []
  listing = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
  names = {"camerad", "camerad.py", "modeld", "modeld.py", "modeld_tinygrad", "openpilot.selfdrive.modeld.modeld",
           "openpilot.selfdrive.modeld.modeld_tinygrad", "openpilot.sunnypilot.modeld_v2.modeld",
           "openpilot.system.camerad.camerad", "openpilot.selfdrive.camerad.camerad"}
  for line in listing.splitlines()[1:]:
    parts = line.split()
    if parts and int(parts[0]) != os.getpid() and any(Path(token).name in names for token in parts[1:]):
      processes.append(" ".join(parts[:3]))
  return {"monotonic_t": time.monotonic(), "offroad": read("/data/params/d/IsOffroad") == "1",
          "onroad": read("/data/params/d/IsOnroad") == "1", "battery_percent": -1 if battery is None else int(battery),
          "external_power": any(value == "1" for value in power.values()), "power_online": power,
          "battery_status": read("/sys/class/power_supply/battery/status"),
          "active_processes": processes, "temperatures_c": temps}


def validate_replay(data, frames):
  shapes = {"tfm": (frames, 3, 3), "big_tfm": (frames, 3, 3), "desire": (frames, 8),
            "traffic_convention": (frames, 2), "action_t": (frames, 2)}
  for key in ("t", "narrow_t", "wide_t", "v_ego", "frame_id", "wide_frame_id", *shapes):
    if key not in data:
      raise BenchError(f"replay missing recorded {key}")
    shape = shapes.get(key, (frames,))
    if np.asarray(data[key]).shape != shape or not np.isfinite(data[key]).all():
      raise BenchError(f"replay {key} must be finite with shape {shape}")
  delta = np.diff(data["t"])
  if len(delta) and (np.any(delta <= 0) or np.any(delta > .15)):
    raise BenchError("replay timestamps must be increasing without gaps above 150 ms")
  if not np.array_equal(data["frame_id"], data["wide_frame_id"]) or np.any(np.diff(data["frame_id"]) != 1):
    raise BenchError("recorded two-camera frame IDs must match and be consecutive without repeats/drops")
  for key in ("wide_t", "narrow_t"):
    if key in data:
      age = data["t"] - data[key]
      if np.any(age < -1e-8) or np.any(age > .025):
        raise BenchError("camera timestamp must be causal and within 25 ms of the decision")
  if any(np.any(np.abs(np.linalg.det(data[key])) < 1e-10) for key in ("tfm", "big_tfm")):
    raise BenchError("recorded camera warp transform is singular")


def availability_times(input_t, compute_ms):
  ready = np.asarray(input_t) + np.asarray(compute_ms) / 1000.
  serial, previous = np.empty(len(ready)), -np.inf
  for i, (timestamp, duration) in enumerate(zip(input_t, compute_ms, strict=True)):
    previous = max(float(timestamp), previous) + float(duration) / 1000.
    serial[i] = previous
  return ready, serial


def effective_action_t(packet_values, includes_smoothing, lat_smoothing, long_smoothing):
  values = np.asarray(packet_values, dtype=float)
  return values.copy() if includes_smoothing else values + [lat_smoothing, long_smoothing]


def validate_result_path(output, model_root, frames_dir):
  if output.suffix != ".json" or not output.resolve().is_relative_to(BENCH_ROOT.resolve()):
    raise BenchError("results must be JSON under /data/rdf-controller-tuning")
  targets = (output.resolve(), output.with_suffix(".npz").resolve())
  model_sources = [model_root / NAME, model_root / (NAME + ".chunkmanifest")]
  model_sources += [model_root / (NAME + f".chunk{i:02d}of03") for i in range(1, 4)]
  for target in targets:
    if target.is_relative_to(model_root.resolve()):
      raise BenchError("results cannot overwrite the model artifact directory")
    for source in model_sources:
      if target.exists() and source.exists() and target.samefile(source):
        raise BenchError("result path aliases a pinned model chunk/manifest")
    if frames_dir is not None:
      for name in ("packet.json", "packet.npz", "road.nv12", "wide.nv12"):
        source = (frames_dir / name).resolve()
        if target == source or (target.exists() and source.exists() and target.samefile(source)):
          raise BenchError("result path collides with a replay packet input")


def verify_packet_files(directory, metadata):
  verified = {}
  for name in ("packet.npz", "road.nv12", "wide.nv12"):
    expected = metadata["artifact_sha256"][name]
    digest = hashlib.sha256()
    with (directory / name).open("rb") as handle:
      while block := handle.read(1024 * 1024):
        digest.update(block)
    if digest.hexdigest() != expected:
      raise BenchError(f"replay packet SHA256 mismatch: {name}")
    verified[name] = digest.hexdigest()
  return verified


def temporal_snapshot(model):
  values = {"queue:" + key: value.numpy().copy() for key, value in model.input_queues.items()}
  values.update({"numpy:" + key: value.copy() for key, value in model.numpy_inputs.items()})
  values["prev_desire"] = model.prev_desire.copy()
  return values


def production_warmup(model):
  before = temporal_snapshot(model)
  started = time.perf_counter()
  model.warmup()
  duration = (time.perf_counter() - started) * 1000
  after = temporal_snapshot(model)
  changed = [key for key in before if not np.array_equal(before[key], after[key])]
  if changed:
    if getattr(model, "_combined_model_type", None) != "supercombo" or getattr(model, "is_run_model", True):
      raise BenchError("canonical warmup advanced temporal state: " + ", ".join(changed))
    from openpilot.sunnypilot.modeld_v2.compile_modeld import make_supercombo_input_queues

    model.input_queues, model.numpy_inputs = make_supercombo_input_queues(model.input_shapes, model.frame_skip, device=model.QUEUE_DEV)
    model.prev_desire[:] = 0
    model.full_frames.clear()
    model._blob_cache.clear()
    reset = temporal_snapshot(model)
    remaining = [key for key in before if key not in reset or not np.array_equal(before[key], reset[key])]
    if remaining:
      raise BenchError("temporal reset after kernel warmup failed: " + ", ".join(remaining))
  return {"duration_ms": duration, "total_with_state_verification_ms": (time.perf_counter() - started) * 1000,
          "temporal_state_restored": True, "state_arrays_compared": len(before),
          "canonical_warmup_changed_arrays": changed,
          "reset_method": "canonical make_supercombo_input_queues factory + cache/prev_desire reset" if changed else "none needed",
          "method": "canonical ModelState.warmup(); exact all-state comparison after explicit reset if needed",
          "availability": "initialization and kernel warmup complete before first real replay input; fresh history still starts empty"}


class Monitor:
  def __init__(self, max_temp_c):
    self.max_temp_c, self.snapshots, self.error = max_temp_c, [], None
    self.done = threading.Event()
    self.thread = threading.Thread(target=self.run, daemon=True)

  def check(self):
    if self.error is not None:
      raise self.error
    snapshot = read_snapshot()
    self.snapshots.append(snapshot)
    check_snapshot(snapshot, self.max_temp_c)

  def check_error(self):
    if self.error is not None:
      raise self.error

  def run(self):
    while not self.done.wait(2):
      try:
        self.check()
      except Exception as exc:
        self.error = exc
        break


class FrameBuffer:
  def __init__(self, size):
    self.data = bytearray(size)
    self.array = np.frombuffer(self.data, dtype=np.uint8)


def construct_model(model_path):
  from openpilot.cereal import custom
  from openpilot.sunnypilot.modeld_v2.modeld import ModelState
  from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase

  bundle = custom.ModelManagerSP.ModelBundle()
  for key, value in bundle_payload().items():
    setattr(bundle, key, value)
  model = ModelState.__new__(ModelState)
  ModelStateBase.__init__(model)  # Reads LagdValueCache; never writes Params.
  model.generation = 12
  model.LAT_SMOOTH_SECONDS, model.LONG_SMOOTH_SECONDS = .1, .3
  model.MIN_LAT_CONTROL_SPEED, model.PLANPLUS_CONTROL, model.chestnut = .3, 1., False
  model._init_combined(str(model_path), 1344, 760, bundle)
  actual = {key: tuple(value) for key, value in model.input_shapes.items()}
  if actual != INPUT_SHAPES or model.constants.MODEL_FREQ != 20 or model.DEV != "QCOM":
    raise BenchError("loaded RDF v2 model/runtime contract mismatch")
  return model


def execute(args, result):
  if not 0 < args.max_seconds <= 600 or not 0 < args.limit <= 1200 or not 0 < args.max_temp_c <= 85:
    raise BenchError("bounded limits require seconds<=600, frames<=1200, thermal<=85 C")
  monitor = Monitor(args.max_temp_c)
  monitor.check()
  monitor.thread.start()
  result["monitor"] = monitor.snapshots
  try:
    base = args.model_root / NAME
    if (base.parent / (base.name + ".chunkmanifest")).read_text().strip() != "3":
      raise BenchError("RDF v2 chunkmanifest must declare exactly three")
    paths = [base.parent / (base.name + f".chunk{i:02d}of03") for i in range(1, 4)]
    result["artifact"] = hash_chunks(paths, CHUNK_SHA, WHOLE_SHA)
    result["bundle"] = bundle_payload()
    os.environ.update(DEV="QCOM", GMMU="0", PYTHONDONTWRITEBYTECODE="1", CACHELEVEL="0")
    import tinygrad
    from openpilot.cereal import log
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.common.transformations.model import get_warp_matrix
    from openpilot.sunnypilot.modeld_v2.constants import Plan

    result["tinygrad_module"] = str(Path(tinygrad.__file__).resolve())
    runtime = Path(tinygrad.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if revision != args.runtime_ref:
      raise BenchError("loaded tinygrad revision differs from requested isolated runtime")
    result["runtime_ref_declared"] = args.runtime_ref
    result["runtime_ref_verified"] = revision
    qcom_source = runtime / "tinygrad" / "runtime" / "ops_qcom.py"
    result["qcom_backend_source"] = str(qcom_source)
    result["qcom_backend_sha256"] = hashlib.sha256(qcom_source.read_bytes()).hexdigest()
    result["status"] = "loading"
    print(json.dumps({"stage": "loading", "whole_sha256": WHOLE_SHA, "runtime": result["tinygrad_module"]}), flush=True)
    started = time.perf_counter()
    model = construct_model(base)
    source_path = Path(inspect.getfile(type(model)))
    result["model_source"] = str(source_path.resolve())
    result["model_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    base_source = Path(inspect.getfile(type(model).__mro__[1]))
    result["model_state_base_source"] = str(base_source.resolve())
    result["model_state_base_sha256"] = hashlib.sha256(base_source.read_bytes()).hexdigest()
    result["initialization_ms"] = (time.perf_counter() - started) * 1000
    result["model_device"], result["warp_device"] = model.DEV, model.WARP_DEV
    result["input_shapes"] = {key: list(value) for key, value in model.input_shapes.items()}
    if args.production_warmup:
      result["production_warmup"] = production_warmup(model)
      print(json.dumps({"stage": "production_warmup", **result["production_warmup"]}), flush=True)
    monitor.check()
    size = get_nv12_info(1344, 760)[3]
    buffers = {key: FrameBuffer(size) for key in model.vision_input_names}
    rng = np.random.default_rng(20261012)
    replay, narrow, wide = None, None, None
    count = args.limit
    if args.mode == "replay":
      if args.frames_dir is None:
        raise BenchError("--frames-dir is required for chronological footage replay")
      if args.packet_action_t_includes_smoothing is None:
        raise BenchError("replay requires explicit --packet-action-t-includes-smoothing yes/no")
      replay = dict(np.load(args.frames_dir / "packet.npz", allow_pickle=False))
      metadata_path = args.frames_dir / "packet.json"
      packet_metadata = json.loads(metadata_path.read_text())
      verified_packet = verify_packet_files(args.frames_dir, packet_metadata)
      frames = len(replay["t"])
      validate_replay(replay, frames)
      if packet_metadata["buffer_size"] != size or packet_metadata["frames"] != frames:
        raise BenchError("packet frame count/buffer size differs from production NV12 layout")
      for name in ("road.nv12", "wide.nv12"):
        if (args.frames_dir / name).stat().st_size != frames * size:
          raise BenchError("camera packet file length does not match timestamp count and NV12 layout")
      narrow = np.memmap(args.frames_dir / "road.nv12", mode="r", dtype=np.uint8, shape=(frames, size))
      wide = np.memmap(args.frames_dir / "wide.nv12", mode="r", dtype=np.uint8, shape=(frames, size))
      result["packet"] = {"directory": str(args.frames_dir), "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                          "source_segment": packet_metadata["source_segment"], "frames": frames,
                          "verified_file_sha256": verified_packet,
                          "causal_join": packet_metadata["causal_join"], "action_t_basis": packet_metadata["action_t_basis"],
                          "calibration_status": packet_metadata["calibration_status"]}
      result["packet"]["action_t_includes_model_smoothing"] = args.packet_action_t_includes_smoothing == "yes"
      result["packet"]["rdf_smoothing_added"] = [.1, .3] if args.packet_action_t_includes_smoothing == "no" else [0., 0.]
      count = min(count, frames)
    camera = DEVICE_CAMERAS[("mici", "os04c10")]
    rows, plans, actions, desired_curvatures, desired_accels, should_stops, timing_inputs = [], [], [], [], [], [], []
    result["frames"] = rows
    previous_action = log.ModelDataV2.Action()
    for index in range(count):
      monitor.check_error()
      if replay is None:
        for buf in buffers.values():
          buf.array[:] = rng.integers(0, 256, size=size, dtype=np.uint8)
        rpy, desire, traffic, action_t = np.zeros(3), np.zeros(8), np.array([1., 0.]), np.array([.275, .875])
        transforms = {model._road_key: get_warp_matrix(rpy, camera.narrow_road.intrinsics, False).astype(np.float32),
                      model._wide_key: get_warp_matrix(rpy, camera.wide_road.intrinsics, True).astype(np.float32)}
        speed = 8.
      else:
        buffers[model._road_key].array[:] = narrow[index]
        buffers[model._wide_key].array[:] = wide[index]
        desire, traffic, action_t = (replay[key][index] for key in ("desire", "traffic_convention", "action_t"))
        action_t = effective_action_t(action_t, args.packet_action_t_includes_smoothing == "yes",
                                      model.LAT_SMOOTH_SECONDS, model.LONG_SMOOTH_SECONDS)
        transforms = {model._road_key: replay["tfm"][index], model._wide_key: replay["big_tfm"][index]}
        speed = float(replay["v_ego"][index])
      inputs = {model.desire_key: np.array(desire, dtype=np.float32, copy=True),
                "traffic_convention": np.array(traffic, dtype=np.float32), "action_t": np.array(action_t, dtype=np.float32)}
      timing_inputs.append(np.array(action_t, copy=True))
      previous = model.numpy_inputs["prev_feat"].copy()
      t0 = time.perf_counter()
      outputs = model.run(buffers, transforms, inputs)
      elapsed = (time.perf_counter() - t0) * 1000
      if not outputs or any(not np.isfinite(value).all() for value in outputs.values()):
        raise BenchError("model produced missing or nonfinite outputs")
      previous_action = model.get_action_from_model(outputs, previous_action, float(action_t[0]), float(action_t[1]), speed)
      plans.append(np.asarray(outputs["plan"])[0].copy())
      actions.append(np.asarray(outputs["action"])[0].copy())
      desired_curvatures.append(float(previous_action.desiredCurvature))
      desired_accels.append(float(previous_action.desiredAcceleration))
      should_stops.append(bool(previous_action.shouldStop))
      compute_ms = (time.perf_counter() - t0) * 1000
      changed = not np.array_equal(previous, model.numpy_inputs["prev_feat"])
      row = {"frame": index, "inference_ms": elapsed, "recurrent_features_changed": changed,
             "compute_ready_ms": compute_ms,
             "action": np.asarray(outputs["action"]).tolist(), "plan_shape": list(np.asarray(outputs["plan"]).shape),
             "history_warmup": replay is not None and index < HISTORY_WARMUP_FRAMES}
      if replay is not None:
        row["source_t"] = float(replay["t"][index])
      rows.append(row)
      print(json.dumps({"stage": "inference", **row}), flush=True)
      monitor.check_error()
    result.update(status="passed", mode=args.mode, frames=rows, total_frames=count,
                  footage_acceptance=False, recurrent_features_changed=any(row["recurrent_features_changed"] for row in rows),
                  scored_footage_frames=max(0, count - HISTORY_WARMUP_FRAMES) if replay is not None else 0,
                  semantics="fresh synthetic functional probe; no footage/road acceptance" if replay is None else
                            "chronological two-camera recorded replay; no vehicle actuation; initial 132 history frames excluded")
    if not result["recurrent_features_changed"]:
      raise BenchError("recurrent features did not advance")
    monitor.check()
    outputs_path = args.out.with_suffix(".npz")
    timestamps = replay["t"][:count] if replay is not None else np.arange(count) * .05
    compute_ms = np.asarray([row["compute_ready_ms"] for row in rows])
    ready, serial_ready = availability_times(timestamps, compute_ms)
    history_scored = np.arange(count) >= HISTORY_WARMUP_FRAMES
    deadline_admissible = serial_ready - timestamps <= .2
    plan_array = np.asarray(plans)
    outputs = {"t": timestamps, "input_t": timestamps, "output_ready_t": serial_ready, "serial_ready_t": serial_ready,
               "input_plus_compute_t": ready,
               "compute_ms": compute_ms, "inference_ms": np.asarray([row["inference_ms"] for row in rows]),
               "action_t": np.asarray(timing_inputs),
               "plan": plan_array, "action": np.asarray(actions), "position": plan_array[:, :, Plan.POSITION],
               "velocity": plan_array[:, :, Plan.VELOCITY], "acceleration": plan_array[:, :, Plan.ACCELERATION],
               "horizon": np.asarray(model.constants.T_IDXS), "scored_mask": history_scored,
               "history_scored_mask": history_scored, "deadline_admissible": deadline_admissible,
               "controller_admissible": history_scored & deadline_admissible, "ready_age_s": serial_ready - timestamps}
    if replay is not None:
      outputs.update(frame_id=replay["frame_id"][:count], wide_frame_id=replay["wide_frame_id"][:count],
                     wide_t=replay["wide_t"][:count], packet_action_t=replay["action_t"][:count])
      if "narrow_t" in replay:
        outputs["narrow_t"] = replay["narrow_t"][:count]
    np.savez_compressed(outputs_path, **outputs,
                        desired_curvature=np.asarray(desired_curvatures), desired_acceleration=np.asarray(desired_accels),
                        should_stop=np.asarray(should_stops), history_warmup=np.arange(count) < HISTORY_WARMUP_FRAMES,
                        model_sha256=np.asarray(WHOLE_SHA))
    result["outputs_npz"] = str(outputs_path)
    result["deadline_limit_s"] = .2
    result["deadline_admissible_scored_frames"] = int(np.count_nonzero(history_scored & deadline_admissible)) if replay is not None else 0
    result["availability_semantics"] = {"input_plus_compute_t": "replayed_input_t_plus_measured_compute",
                                        "output_ready_t": "max(replayed_input_t,previous_ready_t)+measured_compute; includes serial queue backlog",
                                        "serial_ready_t": "same conservative availability as output_ready_t"}
  finally:
    monitor.done.set()
    monitor.thread.join(timeout=3)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-root", type=Path, default=Path("/data/media/0/models"))
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--mode", choices=("probe", "replay"), default="probe")
  parser.add_argument("--frames-dir", type=Path)
  parser.add_argument("--limit", type=int, default=8)
  parser.add_argument("--max-seconds", type=int, default=120)
  parser.add_argument("--max-temp-c", type=float, default=80.)
  parser.add_argument("--runtime-ref", required=True)
  parser.add_argument("--production-warmup", action="store_true")
  parser.add_argument("--packet-action-t-includes-smoothing", choices=("yes", "no"))
  args = parser.parse_args()
  try:
    validate_result_path(args.out, args.model_root, args.frames_dir)
  except BenchError as exc:
    parser.error(str(exc))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  result = {"status": "starting", "wall_power_owner_override": True, "no_params_writes": True, "no_actuation": True,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  code = 0
  def timeout(_signum, _frame):
    raise BenchError("bounded inference probe timeout")
  signal.signal(signal.SIGALRM, timeout)
  signal.alarm(args.max_seconds)
  try:
    execute(args, result)
  except Exception as exc:
    result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    print(json.dumps({"stage": "failed", "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
    code = 1
  finally:
    signal.alarm(0)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return code


if __name__ == "__main__":
  raise SystemExit(main())
