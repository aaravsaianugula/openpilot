#!/usr/bin/env python3
"""
Price the CN7 low-speed feedforward gain schedule against recorded drives, before driving it.

What this answers: if the feedforward had been divided by lat_accel_factor_gain(vEgo), how many
more counts would actually have reached the EPS? Not how many more would have been *asked for* --
the ask is clipped to +-1.0, then scaled to counts, then run through the driver clamp and the
3-up / 7-down rate limiter, and any of those can eat the whole difference. That is the point of
running this on a laptop instead of learning it from the road.

Method. The controller's output is clip(p + i + d + f, -1, 1) and the schedule touches only `f`
(openpilot/common/pid.py, latcontrol_torque_jerk_aware.py). So for each recorded frame:

    f_lat  = f - friction          # the part that was divided by latAccelFactor
    f_new  = f_lat / g(vEgo) + friction
    ask    = -clip(pid_sum + f_new, -1, 1)
    counts = round(ask * STEER_MAX)

and `counts` goes through opendbc's own apply_driver_steer_torque_limits -- imported, never
reimplemented, because a reimplementation is a second copy of the limiter that can drift.

Two chains are reported and they answer different questions:

  * ONE-STEP, the validity check and a hard LOWER bound. Both chains are seeded each frame from the
    counts the car was ACTUALLY applying on the previous frame, then stepped once. Its exact-match
    rate against the recorded command is printed rather than smoothed away -- if that is poor,
    nothing else on the page is worth reading. But note what it can and cannot show: seeded from
    the real previous value and stepped once, the rate limiter caps the difference at 3 counts no
    matter how much larger the ask became, so a sustained change shows up here as about +1%. That
    is an artefact of the method, not a measurement of the effect.
  * FREE-RUNNING, the magnitude, and an UPPER bound. `last` carries forward, so the chain climbs to
    wherever the new ask sustains. Open loop overstates a change that gives the car what it was
    asking for, because a car that finally got it would have backed off and asked for less.

The truth is between the two, and both are printed for that reason.

The friction split is not exactly recoverable from the logs: get_friction_in_torque_space's input
mixes the lateral-accel error with a model-derived lookahead jerk. It saturates at +-friction
whenever the error exceeds FRICTION_THRESHOLD, which is most of a turn, so the saturated case is
the point estimate -- and the envelope from assuming zero friction instead is printed beside it, so
the uncertainty is visible rather than guessed.

Two processes, and this is not optional: oplog loads its own cereal schemas and opendbc registers
conflicting capnp schema IDs, so importing both aborts the interpreter with a duplicate-ID error
whichever order they come in.

    # 1. decode only -- no opendbc on the path
    python .elantra/ff_schedule_replay.py --routes-dir /data/media/0/realdata --dump-trace ff.bin

    # 2. replay only -- no oplog on the path
    python .elantra/ff_schedule_replay.py --trace ff.bin --out ff_schedule_replay.json
"""

from __future__ import annotations

import argparse
import array
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BANDS = ((3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 10.0),
         (10.0, 13.0), (13.0, 16.0), (16.0, 22.0), (22.0, 99.0))

# A new magic rather than extra columns behind the old one. ceiling_trace.bin is 3,420,923 frames
# of CRTR0001 and its reader assumes exactly three columns; widening that format in place would
# make the existing artifact decode as garbage without erroring.
TRACE_MAGIC = b"FFTR0002"
COLUMNS = ("vego", "torque", "driver", "f", "pid", "friction", "laf", "can", "pressed", "des_la")


def dump_trace(rows: list, path: str) -> None:
  """Write named float32 columns behind a header that records their names.

  The names are in the file so a later column can be added without every reader having to agree
  on position first. Flat binary rather than pickle: this only carries numbers between two
  processes, and a format that cannot execute anything is the right one for a file a tool reads
  back without checking where it came from.
  """
  with open(path, "wb") as fh:
    fh.write(TRACE_MAGIC + struct.pack("<QH", len(rows), len(COLUMNS)))
    for name in COLUMNS:
      raw = name.encode("utf-8")
      fh.write(struct.pack("<H", len(raw)) + raw)
    for i in range(len(COLUMNS)):
      col = array.array("f", [r[i] for r in rows])
      if sys.byteorder == "big":
        col.byteswap()
      col.tofile(fh)


def load_trace(path: str) -> dict:
  with open(path, "rb") as fh:
    head = fh.read(len(TRACE_MAGIC) + 10)
    if head[:len(TRACE_MAGIC)] != TRACE_MAGIC:
      raise SystemExit(f"{path} is not an ff_schedule_replay trace")
    n, ncol = struct.unpack("<QH", head[len(TRACE_MAGIC):])
    names = []
    for _ in range(ncol):
      (length,) = struct.unpack("<H", fh.read(2))
      names.append(fh.read(length).decode("utf-8"))
    cols = {}
    for name in names:
      col = array.array("f")
      col.fromfile(fh, n)
      if sys.byteorder == "big":
        col.byteswap()
      cols[name] = col
  missing = [c for c in COLUMNS if c not in cols]
  if missing:
    raise SystemExit(f"{path} is missing columns: {missing}")
  return cols


def _reader():
  """openpilot's LogReader on the device, the local schema loader off it."""
  try:
    from openpilot.tools.lib.logreader import LogReader
    return lambda p: LogReader(p, sort_by_time=True)
  except ImportError:
    import oplog
    return oplog.events


def collect(segs: list) -> list:
  """One row per engaged frame, joined on logMonoTime rather than by index.

  controlsd publishes carControl and controlsState together, card publishes carOutput from its own
  loop, and the k-th message of one is not the k-th of the other -- pairing by index silently
  shifts the applied counts by a frame or two, which is most of a rate-limited ramp.
  """
  import bisect

  events = _reader()
  rows: list = []
  for seg in segs:
    cc: list = []
    cs: list = []
    ct: list = []
    co: list = []
    tp: list = []
    try:
      for ev in events(seg):
        which = ev.which()
        t = ev.logMonoTime * 1e-9
        if which == "carControl":
          m = ev.carControl
          cc.append((t, float(m.latActive), float(m.actuators.torque)))
        elif which == "carState":
          m = ev.carState
          cs.append((t, float(m.vEgo), float(m.steeringTorque), float(m.steeringPressed)))
        elif which == "controlsState":
          m = ev.controlsState
          q = m.lateralControlState.torqueState
          ct.append((t, float(q.f), float(q.p) + float(q.i) + float(q.d), float(m.desiredCurvature)))
        elif which == "carOutput":
          co.append((t, float(getattr(ev.carOutput.actuatorsOutput, "torqueOutputCan", 0.0))))
        elif which == "lateralTorqueParameters":
          m = ev.lateralTorqueParameters
          tp.append((t, float(m.frictionCoefficientFiltered), float(m.latAccelFactorFiltered)))
    except Exception as exc:  # a truncated segment is normal in an archive; say so and move on
      print(f"  skip {os.path.basename(seg)}: {type(exc).__name__}: {exc}", file=sys.stderr)
      continue
    if not (cc and cs and ct and co and tp):
      continue
    for series in (cs, ct, co, tp):
      series.sort()
    cc.sort()

    stamps = {id(series): [x[0] for x in series] for series in (cs, ct, co, tp)}

    def hold(series, t, stamps=stamps):
      """Last value of `series` at or before t. Zero-order hold: carState and controlsState run at
      100 Hz beside carControl, but lateralTorqueParameters is 4 Hz and must be held, not
      interpolated -- it is a learner output, not a continuous signal."""
      i = bisect.bisect_right(stamps[id(series)], t) - 1
      return series[max(i, 0)][1:]

    for t, lat_active, torque in cc:
      if lat_active < 0.5:
        continue
      vego, driver, pressed = hold(cs, t)
      f, pid_sum, desired_curvature = hold(ct, t)
      (can,) = hold(co, t)
      friction, laf = hold(tp, t)
      rows.append((vego, torque, driver, f, pid_sum, friction, laf, can,
                   pressed, desired_curvature * vego ** 2))
  return rows


def _clip(x, lo, hi):
  return lo if x < lo else (hi if x > hi else x)


def replay(cols: dict, gain, limits, friction_share: float, turns_only: bool) -> dict:
  """Both chains, in one pass. friction_share in [0, 1] scales the part of f held out of the
  division: 1.0 is the saturated case (the point estimate), 0.0 is the no-friction envelope.

  The BEFORE command is the logged `actuators.torque`, not a reconstruction from p+i+f -- so the
  baseline is exactly what the car did, and only the delta is modelled. Since
  torque = -clip(p+i+d+f, -1, 1), adding (f_new - f) inside the same clip is exact whether or not
  the original command was saturated.
  """
  from opendbc.car.lateral import apply_driver_steer_torque_limits

  steer_max = limits.STEER_MAX
  out = {f"{lo:g}-{hi:g}": {"frames": 0, "one_step_before": 0, "one_step_after": 0,
                            "free_before": 0, "free_after": 0, "exact": 0, "recon_close": 0,
                            "at_ceiling_before": 0, "at_ceiling_after": 0,
                            "ask_grew_delivery_did_not": 0}
         for lo, hi in BANDS}

  free_before = 0
  free_after = 0
  n = len(cols["vego"])
  for i in range(n):
    v = cols["vego"][i]
    driver = cols["driver"][i]
    can = cols["can"][i]
    prev = cols["can"][i - 1] if i else 0.0

    f = cols["f"][i]
    pid_sum = cols["pid"][i]
    friction = cols["friction"][i] * friction_share
    held = friction if f >= 0.0 else -friction
    f_new = (f - held) / gain(v) + held

    before = cols["torque"][i]
    after = -_clip(-before + (f_new - f), -1.0, 1.0)

    one_before = apply_driver_steer_torque_limits(int(round(before * steer_max)), int(prev), driver, limits)
    one_after = apply_driver_steer_torque_limits(int(round(after * steer_max)), int(prev), driver, limits)
    free_before = apply_driver_steer_torque_limits(int(round(before * steer_max)), int(free_before), driver, limits)
    free_after = apply_driver_steer_torque_limits(int(round(after * steer_max)), int(free_after), driver, limits)

    band = next((f"{lo:g}-{hi:g}" for lo, hi in BANDS if lo <= v < hi), None)
    if band is None:
      continue
    if turns_only and not (abs(cols["des_la"][i]) > 1.5 and cols["pressed"][i] < 0.5):
      continue
    cell = out[band]
    cell["frames"] += 1
    cell["one_step_before"] += abs(one_before)
    cell["one_step_after"] += abs(one_after)
    cell["free_before"] += abs(free_before)
    cell["free_after"] += abs(free_after)
    cell["exact"] += int(one_before == int(round(can)))
    # Independent self-check: do the logged PID terms actually explain the logged command? If this
    # is low the trace columns are mis-joined and every number beside it is meaningless.
    cell["recon_close"] += int(abs(-_clip(pid_sum + f, -1.0, 1.0) - before) < 2e-3)
    cell["at_ceiling_before"] += int(abs(one_before) >= steer_max - 4)
    cell["at_ceiling_after"] += int(abs(one_after) >= steer_max - 4)
    if abs(after) > abs(before) + 1e-9 and abs(one_after) <= abs(one_before):
      cell["ask_grew_delivery_did_not"] += 1
  return out


def report(saturated: dict, envelope: dict, steer_max: int) -> None:
  print("")
  print(f"CN7 low-speed feedforward schedule, replayed at STEER_MAX={steer_max}")
  print("  1step:  seeded from the counts actually applied last frame -- a LOWER bound, because the")
  print("          rate limiter caps one frame's difference at 3 counts however much bigger the ask")
  print("  free:   open loop, an UPPER bound -- a car that got what it asked for would back off")
  print("  match%: how often the BEFORE chain reproduced the recorded command exactly")
  print("  recon%: how often the logged p+i+d+f explains the logged command (a join self-check)")
  print("")
  head = ("band", "frames", "match%", "recon%", "before", "after", "1step d%", "free d%", "pin%->", "no-gain%")
  print(" ".join(f"{h:>10}" for h in head))
  for band, cell in saturated.items():
    n = cell["frames"]
    if n < 100:
      continue
    before = cell["one_step_before"] / n
    after = cell["one_step_after"] / n
    fb = cell["free_before"] / n
    fa = cell["free_after"] / n
    env = envelope[band]
    env_after = env["one_step_after"] / max(env["frames"], 1)
    row = (band, n,
           f"{100.0 * cell['exact'] / n:.1f}",
           f"{100.0 * cell['recon_close'] / n:.1f}",
           f"{fb:.0f}",
           f"{fa:.0f}",
           f"{100.0 * (after - before) / max(before, 1e-9):+.1f}",
           f"{100.0 * (fa - fb) / max(fb, 1e-9):+.1f}",
           f"{100.0 * cell['at_ceiling_before'] / n:.0f}->{100.0 * cell['at_ceiling_after'] / n:.0f}",
           f"{100.0 * cell['ask_grew_delivery_did_not'] / n:.0f}")
    print(" ".join(f"{str(x):>10}" for x in row))
    if abs(env_after - after) > 0.5:
      print(f"{'':>21} friction envelope: after would be {env_after:.0f} counts"
            + " with the friction term excluded from the division")
  print("")
  print("  no-gain%: frames where the ask grew but the delivered counts did not -- the driver clamp")
  print("            or the rate limiter took all of it. A large number here means the change buys")
  print("            nothing in that band no matter what the plant gain says.")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("routes", nargs="*", help="route ids to decode (default: every route under --routes-dir)")
  ap.add_argument("--routes-dir", default="/data/media/0/realdata")
  ap.add_argument("--dump-trace", help="decode only, write the trace here and exit")
  ap.add_argument("--trace", help="replay only, read the trace from here")
  ap.add_argument("--out", help="write the per-band result as JSON")
  ap.add_argument("--turns-only", action="store_true",
                  help="restrict to hands-off frames commanding over 1.5 m/s^2 -- the frames the"
                       + " change is aimed at, not every engaged frame including straights")
  args = ap.parse_args()

  if bool(args.dump_trace) == bool(args.trace):
    ap.error("pass exactly one of --dump-trace (decode) or --trace (replay);"
             + " oplog and opendbc cannot be imported in the same process")

  if args.dump_trace:
    import glob
    segs = []
    for pattern in (args.routes or ["*"]):
      segs += sorted(glob.glob(os.path.join(args.routes_dir, f"{pattern}--*", "rlog.zst")))
    if not segs:
      raise SystemExit(f"no segments matched under {args.routes_dir}")
    print(f"decoding {len(segs)} segments")
    rows = collect(segs)
    if not rows:
      raise SystemExit("no engaged frames decoded -- nothing to replay")
    dump_trace(rows, args.dump_trace)
    print(f"wrote {len(rows)} frames to {args.dump_trace}")
    return 0

  from opendbc.car.hyundai.values import CarControllerParams, HyundaiFlags

  class _Probe:
    # RAISED_LIMITS must be here or CarControllerParams falls through to the 384 HKG default. This
    # is the CarParams.flags bit (2**27), NOT the safetyParam bit (1024) -- two different flags
    # with the same name, and reading the wrong one fails silently in both directions.
    carFingerprint = "HYUNDAI_ELANTRA_2024"
    flags = int(HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.CAMERA_SCC | HyundaiFlags.RAISED_LIMITS)

  # The real CarControllerParams, not a copy of its numbers -- there is nothing here to drift.
  # (ceiling_replay.py keeps its own copy and its drift check currently fires: that copy still says
  # STEER_DRIVER_ALLOWANCE = 50 while opendbc says 100. That is a real, separate divergence and not
  # this tool's to fix.)
  limits = CarControllerParams(_Probe())
  if limits.STEER_MAX != 409:
    raise SystemExit(f"STEER_MAX is {limits.STEER_MAX}; the MDPS accepts 409 and faults at 410,"
                     + " so a replay against anything else prices a build this car cannot run")

  # The schedule is loaded from the shipped module by path: .elantra is not a legal package name,
  # so this file cannot `import openpilot...` the ordinary way from every checkout, and a local
  # copy of the numbers is exactly the drift this replay exists to avoid.
  import importlib.util
  sched_path = Path(__file__).resolve().parents[1] / "openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py"
  spec = importlib.util.spec_from_file_location("lat_accel_factor_schedule", sched_path)
  sched = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(sched)

  cols = load_trace(args.trace)
  print(f"replaying {len(cols['vego'])} frames from {args.trace}")
  print(f"schedule: bp={sched.FF_LAT_ACCEL_GAIN_BP} v={sched.FF_LAT_ACCEL_GAIN_V}"
        + f" blend={sched.LOW_SPEED_FF_BLEND}")

  if args.turns_only:
    print("frame set: hands-off, |commanded lateral accel| > 1.5 m/s^2")
  saturated = replay(cols, sched.lat_accel_factor_gain, limits, 1.0, args.turns_only)
  envelope = replay(cols, sched.lat_accel_factor_gain, limits, 0.0, args.turns_only)
  report(saturated, envelope, limits.STEER_MAX)

  if args.out:
    payload = {"trace": args.trace, "steer_max": limits.STEER_MAX,
               "schedule": {"bp": list(sched.FF_LAT_ACCEL_GAIN_BP),
                            "v": list(sched.FF_LAT_ACCEL_GAIN_V),
                            "blend": sched.LOW_SPEED_FF_BLEND},
               "turns_only": args.turns_only,
               "saturated_friction": saturated, "no_friction_envelope": envelope}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
