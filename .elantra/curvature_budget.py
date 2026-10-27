#!/usr/bin/env python3
"""curvature_budget.py -- where the model's planned curvature goes before the car turns.

The CN7 runs wide in turns. The torque chain is only the second half of that story; the first
half is that openpilot clamps the COMMANDED CURVATURE before the steering controller ever sees
it, in clip_curvature (selfdrive/controls/lib/drive_helpers.py), and the clamp is expressed as a
lateral ACCELERATION -- so the tightest radius it permits scales as v^2 / limit and it bites
harder the faster you go, until the turns stop being tight enough to reach it at all.

This tool measures that budget, one frame at a time:

    model demand  ->  commanded  ->  achieved
    (modelV2)        (controlsState)  (yaw rate)

  * the first leg is clip_curvature and nothing else, because both ends come out of controlsd
    in the same message loop -- no cross-process join, nothing to align;
  * the second leg is measured from liveLocationKalman.angularVelocityCalibrated ONLY.

Deliberate choices, each of which was a real defect in an earlier tool on this problem:

  * ACHIEVED CURVATURE IS NEVER controlsState.curvature. That field is the steering angle over
    paramsd's live-fitted steerRatio -- a slip model whose job is to make the two agree. It
    matches the command to within a few percent at 0-7 m/s BY CONSTRUCTION and cannot observe a
    car running wide. Only the yaw rate can. (carState.yawRate is always 0 on this fork.)
  * EVERY PERCENTAGE IS POOLED OVER FRAMES, never a median over turn events. A median-over-turns
    reports 0% whenever most turns are clean, which is exactly the case here: it understated the
    clamp rate by about 12 points and is why this clamp was dismissed twice.
  * THE CLAMP ATTRIBUTION IS A REPLAY, NOT AN INFERENCE. clip_curvature is re-run on the logged
    inputs against the logged previous output, and the tool prints the residual against the
    logged result. If that residual is not ~0, nothing else in the report means anything, so it
    is printed first and loudly.
  * THE COUNTERFACTUAL IS SEQUENTIAL. Re-pricing a candidate limit feeds the simulation its own
    previous output, with the lateral-jerk clamp in the loop, because a one-step check against
    the logged previous value silently gives the candidate the benefit of the real limiter's
    ramp.
  * TURNS ARE DEFINED BY CURVATURE, NOT LATERAL ACCELERATION. A lateral-accel threshold is a
    speed filter in disguise: at 2 m/s a 10 m radius turn is only 0.4 m/s^2 and would be
    excluded, which is how the low-speed band ends up looking empty.

  scan   [--routes DIR] [--out FILE] [--limit N] [--force] [--caps 3.0,4.0,...]
  report [--out FILE]
"""
import argparse
import json
import os
import sys
from collections import defaultdict, namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the log access and route walking that already exist rather than growing a second copy.
# What is deliberately NOT reused is turn_tracking's metric layer: its delivery table is built
# on controlsState.curvature and four of its percentage columns are medians over turn events.
from turn_tracking import _events, rlog_of, routes_under

DT = 0.01                       # controlsState is 100 Hz
G = 9.81                        # ACCELERATION_DUE_TO_GRAVITY, common/constants.py
MAX_LATERAL_JERK = 5.0          # drive_helpers.py
MAX_CURVATURE = 0.2             # drive_helpers.py
MIN_SPEED = 1.0                 # drive_helpers.py
STOCK_LAT_ACCEL = 3.0           # MAX_LATERAL_ACCEL_NO_ROLL before this port touched it

# A turn is the model asking for a radius under 67 m, at any speed. Curvature, not lateral
# accel, so the definition does not quietly become a speed filter.
TURN_CURVATURE = 0.015
MIN_EPISODE_FRAMES = 20         # 0.2 s; shorter is the 20 Hz model staircase, not a turn
GAP_SPEED_JUMP = 1.5            # m/s between consecutive frames -> treat as a new stretch

BANDS = ((1, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, 18), (18, 25), (25, 99))

Frame = namedtuple("Frame", "v mdl cmd yaw roll out sat pressed driver")


def band_of(v):
  for lo, hi in BANDS:
    if lo <= v < hi:
      return f"{lo:g}-{hi:g}"
  return None


DASH = "-"


def row(*widths):
  """A fixed-width table row printer. None inserts a column separator.

  Header and data rows share one width list, so a column can never drift out of alignment
  with its own heading -- which is how a report ends up quietly mislabelled.
  """
  def fmt(*fields):
    out, it = ["  "], iter(fields)
    for w in widths:
      if w is None:
        out.append("| ")
        continue
      out.append(f"{next(it)!s:>{w}} ")
    return "".join(out).rstrip()
  return fmt


def pct(xs, p):
  """Percentile of a list, or nan. Explicit so no caller can accidentally get a mean."""
  if not xs:
    return float("nan")
  xs = sorted(xs)
  return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))]


def schedule_from_source(repo_root=None):
  """The lateral-accel schedule clip_curvature is actually compiled with.

  Read from drive_helpers.py rather than hard-coded, because a tool that hard-codes the limit
  it is measuring reports the same answer after the limit changes -- which is how an earlier
  harness on this problem got every derived count wrong by 6.5%.
  """
  if repo_root is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  path = os.path.join(repo_root, "openpilot/selfdrive/controls/lib/drive_helpers.py")
  try:
    with open(path, encoding="utf-8") as fh:
      lines = fh.readlines()
  except OSError as ex:
    # Fatal on purpose. Returning a default here would drop the one column of the
    # counterfactual that says what the car will actually do, and the table would still
    # print as though it were complete.
    raise SystemExit(f"cannot read {path}: {ex}") from ex

  bp = vals = None
  stock = STOCK_LAT_ACCEL
  for line in lines:
    head, sep, tail = line.partition("=")
    if not sep:
      continue
    name, body = head.strip(), tail.split("#")[0].strip()
    if name == "MAX_LATERAL_ACCEL_NO_ROLL":
      stock = float(body)
    elif name == "LAT_ACCEL_LIMIT_BP":
      bp = [float(x) for x in body.strip("[]").split(",") if x.strip()]
    elif name == "LAT_ACCEL_LIMIT_V":
      vals = [stock if x.strip() == "MAX_LATERAL_ACCEL_NO_ROLL" else float(x)
              for x in body.strip("[]").split(",") if x.strip()]
  if bp is None or vals is None:
    raise SystemExit(f"{path} defines no LAT_ACCEL_LIMIT_BP/LAT_ACCEL_LIMIT_V; this tool "
                     + "cannot price a schedule it cannot read")
  return bp, vals, stock


def interp(x, xp, fp):
  """np.interp for two-point schedules, without importing numpy on the device."""
  if x <= xp[0]:
    return fp[0]
  if x >= xp[-1]:
    return fp[-1]
  for i in range(1, len(xp)):
    if x <= xp[i]:
      span = xp[i] - xp[i - 1]
      t = 0.0 if span == 0 else (x - xp[i - 1]) / span
      return fp[i - 1] + t * (fp[i] - fp[i - 1])
  return fp[-1]


def clip_curvature(v_ego, prev, new, roll, limit):
  """clip_curvature, reimplemented for offline replay. Returns (out, jerk, accel, maxcurv).

  Kept in step with drive_helpers.py by the residual self-check, which compares this against
  the value controlsd actually published. If upstream changes the function, the residual moves
  off zero and the report says so instead of quietly reporting the old semantics.
  """
  v = max(v_ego, MIN_SPEED)
  rate = MAX_LATERAL_JERK / v ** 2 * DT
  a = min(max(new, prev - rate), prev + rate)
  jerk = a != new
  comp = roll * G
  hi, lo = (limit + comp) / v ** 2, (-limit + comp) / v ** 2
  b = min(max(a, lo), hi)
  accel = b != a
  c = min(max(b, -MAX_CURVATURE), MAX_CURVATURE)
  return c, jerk, accel, c != b


def collect_segment(segdir):
  """One segment -> [Frame or None]. None marks a frame that must not join a neighbour."""
  path = rlog_of(segdir)
  if not path:
    return [], None, {"field_errors": {}, "read_error": "no rlog"}
  frames = []
  commit = None
  field_errors: dict = defaultdict(int)
  read_error = None
  v = roll = driver = 0.0
  lat = ok = False
  ok = True
  mdl = yaw = None
  pressed = False
  try:
    for e in _events(path):
      w = e.which()
      if w == "carState":
        c = e.carState
        v, driver, pressed = c.vEgo, float(c.steeringTorque), bool(c.steeringPressed)
      elif w == "carControl":
        lat = bool(e.carControl.latActive)
      elif w == "initData":
        if commit is None:
          commit = e.initData.gitCommit[:9]
      elif w == "modelV2":
        try:
          mdl = float(e.modelV2.action.desiredCurvature)
        except (AttributeError, ValueError, TypeError):
          # Narrow, and counted: this field IS the pre-clip demand. A schema where it is
          # missing does not mean "measure a bit less", it means the tool is measuring the
          # previous frame forever. The report refuses to print if this is ever non-zero.
          field_errors["modelV2.action.desiredCurvature"] += 1
      elif w == "vehicleParameters":
        try:
          roll = float(e.vehicleParameters.roll)
        except (AttributeError, ValueError, TypeError):
          field_errors["vehicleParameters.roll"] += 1
      elif w == "liveLocationKalman":
        m = e.liveLocationKalman.angularVelocityCalibrated
        yaw = float(m.value[2]) if bool(m.valid) else None
      elif w == "selfdriveState":
        a1 = e.selfdriveState.alertText1 or ""
        ok = not ("Calibrat" in a1 or "Big Model Failed" in a1)
      elif w == "controlsState":
        c = e.controlsState
        if not lat or not ok or mdl is None:
          frames.append(None)
          continue
        ts = c.lateralControlState.torqueState
        frames.append(Frame(v, mdl, float(c.desiredCurvature), yaw, roll,
                            float(ts.output), bool(ts.saturated), pressed, driver))
  except Exception as ex:
    # A truncated rlog leaves real data behind and is worth keeping, but the report has to be
    # able to say how much of the archive it could not read.
    read_error = f"{type(ex).__name__}: {str(ex)[:60]}"
    print(f"    ! {os.path.basename(segdir)}: {read_error}", file=sys.stderr)
  return frames, commit, {"field_errors": dict(field_errors), "read_error": read_error}


def _stretches(frames):
  """Contiguous runs of usable frames. A speed jump means the log skipped; never span one."""
  run = []
  for f in frames:
    if f is None or (run and abs(f.v - run[-1].v) > GAP_SPEED_JUMP):
      if len(run) > 1:
        yield run
      run = [] if f is None else [f]
    else:
      run.append(f)
  if len(run) > 1:
    yield run


class _Episode:
  """One run of consecutive accel-clamped frames, and the path deficit it accumulated.

  This is the only part of analyse()'s loop with a lifetime of its own, and the only part whose
  correctness depends on being closed at the right moment -- at the end of a stretch, so an
  episode is never welded across a logged gap. Interleaved with the four per-frame accumulators
  it was also what pushed that loop to five levels of indentation.
  """

  def __init__(self, cell):
    self._cell = cell
    self._reset()

  def _reset(self):
    self.band = None
    self.n = 0
    self.vel = 0.0
    self.pos = 0.0
    self.peak = 0.0
    self.cut = []

  def feed(self, band, f, clamped):
    if not clamped:
      self.close()
      return
    if self.band is None:
      self.band = band
    # Double-integrated curvature deficit. Exactly zero outside a clamp, which is why it is
    # accumulated per episode rather than across a whole stretch: it carries no integration bias.
    self.n += 1
    self.vel += (f.mdl - f.cmd) * max(f.v, MIN_SPEED) ** 2 * DT
    self.pos += self.vel * DT
    self.peak = max(self.peak, abs(self.pos))
    if abs(f.mdl) > 1e-5:
      self.cut.append(1.0 - abs(f.cmd) / abs(f.mdl))

  def close(self):
    if self.band is not None and self.n >= MIN_EPISODE_FRAMES:
      mean_cut = sum(self.cut) / len(self.cut) if self.cut else 0.0
      self._cell(self.band)["episodes"].append([self.n * DT, self.peak, mean_cut])
    self._reset()


def analyse(frames, caps):
  """Per-band counters for one segment. Every percentage denominator is a frame count."""
  out = {}

  def cell(band):
    if band not in out:
      out[band] = {"frames": 0, "turn": 0, "jerk": 0, "accel": 0, "maxcurv": 0,
                   "sat": 0, "sat_accel": 0, "sat_pinned": 0, "sat_neither": 0,
                   "pinned": 0, "resid_n": 0, "resid_exact": 0, "resid_max": 0.0,
                   "cut_clamped": [], "ratio_cm": [], "ratio_ac": [], "ratio_am": [],
                   "sim": {k: {"n": 0, "accel": 0, "sum_la": 0.0} for k in caps},
                   "episodes": []}
    return out[band]

  for run in _stretches(frames):
    # --- one-step exact replay: what the shipped code did, and proof that it did ----------
    sim_prev = dict.fromkeys(caps, run[0].cmd)
    episode = _Episode(cell)
    for i in range(1, len(run)):
      f, prev = run[i], run[i - 1]
      band = band_of(f.v)
      if band is None:
        continue
      c = cell(band)
      c["frames"] += 1
      if f.sat:
        c["sat"] += 1
      pinned = abs(f.out) >= 0.999

      # The schedule in force when this frame was recorded is not knowable from the log, so the
      # replay uses the stock constant and the residual is what says whether that was right.
      pred, jerk, accel, maxcurv = clip_curvature(f.v, prev.cmd, f.mdl, f.roll, STOCK_LAT_ACCEL)
      # Summarised, never sampled: "how often is the replay exact, and how wrong does it ever
      # get" is the whole self-check, and unlike a percentile it cannot be skewed by which
      # residuals a bounded buffer happened to keep.
      resid = abs(pred - f.cmd)
      c["resid_n"] += 1
      c["resid_exact"] += 1 if resid == 0.0 else 0
      c["resid_max"] = max(c["resid_max"], resid)

      if f.sat:
        c["sat_accel"] += 1 if accel else 0
        c["sat_pinned"] += 1 if pinned else 0
        c["sat_neither"] += 1 if not accel and not pinned else 0

      is_turn = abs(f.mdl) > TURN_CURVATURE
      if is_turn:
        c["turn"] += 1
        c["jerk"] += 1 if jerk else 0
        c["accel"] += 1 if accel else 0
        c["maxcurv"] += 1 if maxcurv else 0
        c["pinned"] += 1 if pinned else 0
        if accel and abs(f.mdl) > 1e-5:
          # The demand the clamp removed, on the frames it actually acted. Measured over every
          # turn frame instead, this is dominated by the command chasing a 20 Hz model at
          # 100 Hz and reads as a couple of percent no matter how hard the clamp is biting.
          c["cut_clamped"].append(1.0 - abs(f.cmd) / abs(f.mdl))
        if abs(f.mdl) > 2e-3 and abs(f.cmd) > 2e-3:
          c["ratio_cm"].append(abs(f.cmd) / abs(f.mdl))
          if f.yaw is not None and not f.pressed and abs(f.driver) < 30.0:
            ach = abs(f.yaw) / max(f.v, MIN_SPEED)
            c["ratio_ac"].append(ach / abs(f.cmd))
            c["ratio_am"].append(ach / abs(f.mdl))

      # --- sequential counterfactual, jerk clamp in the loop -------------------------------
      for key, sched in caps.items():
        limit = interp(f.v, sched[0], sched[1])
        val, _, sacc, _ = clip_curvature(f.v, sim_prev[key], f.mdl, f.roll, limit)
        sim_prev[key] = val
        if is_turn:
          s = c["sim"][key]
          s["n"] += 1
          s["accel"] += 1 if sacc else 0
          s["sum_la"] += abs(val) * max(f.v, MIN_SPEED) ** 2

      # --- accel-clamp episodes, and what they cost in metres ------------------------------
      episode.feed(band, f, accel)
    # A stretch ends at a logged gap; an episode must never be welded across one.
    episode.close()
  return out


def _merge(dst, src):
  for band, c in src.items():
    d = dst.setdefault(band, None)
    if d is None:
      dst[band] = json.loads(json.dumps(c))
      continue
    for k, v in c.items():
      if k == "resid_max":
        d[k] = max(d[k], v)
      elif isinstance(v, (int, float)):
        d[k] += v
      elif isinstance(v, list):
        d[k].extend(v)
      elif isinstance(v, dict):
        for key, s in v.items():
          for kk, vv in s.items():
            d[k][key][kk] += vv
  return dst


def parse_caps(spec, repo_schedule):
  """{label: (breakpoints, values)} -- flat caps plus the schedule the tree is compiled with."""
  caps = {}
  for part in (spec or "").split(","):
    part = part.strip()
    if part:
      caps[f"flat {part}"] = ([0.0], [float(part)])
  bp, vals, _ = repo_schedule
  if bp and vals:
    caps["shipped {}".format("/".join(f"{v:g}" for v in vals))] = (bp, vals)
  return caps


def cmd_scan(a):
  caps = parse_caps(a.caps, schedule_from_source())
  routes = routes_under(a.routes)
  names = sorted(routes)
  if a.limit:
    names = names[-a.limit:]
  done = set()
  if os.path.exists(a.out) and not a.force:
    with open(a.out) as fh:
      for i, line in enumerate(fh, 1):
        try:
          done.add(json.loads(line)["route"])
        except (ValueError, KeyError) as ex:
          # Resuming past a corrupt line would silently skip a route and the scan would
          # report a smaller archive as though that were the whole archive.
          raise SystemExit(f"{a.out} line {i} is corrupt ({ex}); delete it or pass --force"
                           ) from ex
  print(f"routes: {len(names)}   already scanned: {len(done)}   caps: {list(caps)}")
  with open(a.out, "a") as fh:
    for k, route in enumerate(names, 1):
      if route in done and not a.force:
        continue
      merged, commit = {}, None
      health: dict = {"field_errors": defaultdict(int), "read_errors": 0, "segments": 0}
      for seg in routes[route]:
        frames, c, seg_health = collect_segment(seg)
        commit = commit or c
        health["segments"] += 1
        health["read_errors"] += 1 if seg_health["read_error"] else 0
        for key, n in seg_health["field_errors"].items():
          health["field_errors"][key] += n
        _merge(merged, analyse(frames, caps))
      health["field_errors"] = dict(health["field_errors"])
      fh.write(json.dumps({"route": route, "commit": commit, "caps": list(caps),
                           "health": health, "bands": merged}) + "\n")
      fh.flush()
      turn = sum(c["turn"] for c in merged.values())
      print(f"[{k}/{len(names)}] {route} segs={len(routes[route]):3d} turnFrames={turn:6d} commit={commit}")
  return 0


def cmd_report(a):
  recs = []
  with open(a.out) as fh:
    for i, line in enumerate(fh, 1):
      try:
        recs.append(json.loads(line))
      except ValueError as ex:
        raise SystemExit(f"{a.out} line {i} is corrupt ({ex})") from ex
  if not recs:
    print("no records in " + a.out)
    return 1

  bands = {}
  for r in recs:
    _merge(bands, r["bands"])
  caps = recs[0]["caps"]
  commits = sorted({r.get("commit") for r in recs if r.get("commit")})

  total = sum(c["frames"] for c in bands.values())
  print("=" * 100)
  print(f"routes={len(recs)}  engaged frames={total}  builds={commits[:8]}")
  print("Builds are pooled deliberately: every figure below is a CURVATURE, which does not")
  print("depend on STEER_MAX, so a 384-count route and a 409-count route are comparable here.")

  segs = sum(r.get("health", {}).get("segments", 0) for r in recs)
  read_errors = sum(r.get("health", {}).get("read_errors", 0) for r in recs)
  field_errors: dict = defaultdict(int)
  for r in recs:
    for key, n in r.get("health", {}).get("field_errors", {}).items():
      field_errors[key] += n
  if any("health" in r for r in recs):
    stale = sum(1 for r in recs if "health" not in r)
    print(f"segments read: {segs - read_errors} of {segs} "
          + f"({read_errors} truncated or unreadable)")
    if stale:
      print(f"  {stale} route(s) predate decode-health recording; their segment counts are")
      print("  NOT in the line above, and are unknown rather than zero.")
  else:
    # Absent is not zero. A scan from before this was recorded must not read as a clean one.
    print("segments read: NOT RECORDED by this scan -- rescan with --force to learn how much")
    print("  of the archive was actually decoded.")
  if field_errors:
    print("")
    print("REFUSING TO REPORT -- fields this tool depends on could not be decoded:")
    for key, n in sorted(field_errors.items()):
      print(f"    {key}: {n} messages")
    print("Every number below would be computed from whatever stale value was left behind.")
    return 1

  rn = sum(c["resid_n"] for c in bands.values())
  rexact = sum(c["resid_exact"] for c in bands.values())
  rmax = max((c["resid_max"] for c in bands.values()), default=0.0)
  print("")
  print("SELF-CHECK -- offline clip_curvature replay vs the logged controlsState.desiredCurvature")
  if rn:
    exact_pct = 100.0 * rexact / rn
    print(f"  reproduced EXACTLY on {rexact} of {rn} frames ({exact_pct:.3f}%)")
    print(f"  worst |predicted - logged| anywhere: {rmax:.3g} /m")
    print("  Healthy is ABOVE 90%, and 92.5% is what the 110-route archive gives today -- not")
    print("  100%, and it should not be. The log records when modelV2 was SENT, not when")
    print("  controlsd read it, so on the ~7.5% of frames adjacent to a 20 Hz model update the")
    print("  replay pairs the command with the neighbouring demand. That is a decode artefact,")
    print("  not a semantic one. Below 90% means the replay has drifted from the shipped")
    print("  function, and every clamp attribution in this report is then meaningless.")
  else:
    print("  no frames replayed")

  print("")
  print("CLAMP RATES, pooled over frames (never a median over turn events)")
  print(f"  turn frame = the model asking for a radius under {1.0 / TURN_CURVATURE:g} m")
  print(row(9, 9, 9, None, 7, 7, 7, None, 7, 7)(
      "band m/s", "frames", "turnFrm", "accel%", "jerk%", "maxCrv%", "pin%", "cut p50"))
  print("  every rate is a fraction of turnFrm; cut p50 is the demand removed on the frames")
  print("  the accel clamp actually acted, so it is nan where that clamp never fired")
  for lo, hi in BANDS:
    b = f"{lo:g}-{hi:g}"
    c = bands.get(b)
    if not c or c["turn"] < 200:
      continue
    cut = c["cut_clamped"]
    n = c["turn"]
    print(row(9, 9, 9, None, 7, 7, 7, None, 7, 7)(
        b, c["frames"], n, f"{100.0 * c['accel'] / n:.1f}%", f"{100.0 * c['jerk'] / n:.1f}%",
        f"{100.0 * c['maxcurv'] / n:.2f}%", f"{100.0 * c['pinned'] / n:.1f}%",
        f"{pct(cut, 50):.3f}"))

  print("")
  print("WHERE THE PLAN GOES, pooled, quiet wheel only (not pressed, |column torque| < 30)")
  print("  cmd/mdl = survives clip_curvature | ach/cmd = torque chain | ach/mdl = end to end")
  print("  achieved is yaw-rate only -- controlsState.curvature cannot see a car running wide")
  plan_row = row(9, 9, None, 7, 7, None, 7, 7, None, 7, 7)
  print(plan_row("band m/s", "n", "c/m p50", "p10", "a/c p50", "p10", "a/m p50", "p10"))
  for lo, hi in BANDS:
    b = f"{lo:g}-{hi:g}"
    c = bands.get(b)
    if not c or len(c["ratio_ac"]) < 200:
      continue
    print(plan_row(b, len(c["ratio_ac"]),
                   f"{pct(c['ratio_cm'], 50):.3f}", f"{pct(c['ratio_cm'], 10):.3f}",
                   f"{pct(c['ratio_ac'], 50):.3f}", f"{pct(c['ratio_ac'], 10):.3f}",
                   f"{pct(c['ratio_am'], 50):.3f}", f"{pct(c['ratio_am'], 10):.3f}"))

  print("")
  print("COUNTERFACTUAL -- the same logged model demand replayed sequentially at each limit")
  print("  jerk clamp in the loop, prev fed from the simulation's own output")
  print("  cell = mean commanded lateral accel on turn frames / %% of them still accel-clamped")
  band_label = "band m/s"
  head = f"  {band_label:>9}"
  for k in caps:
    head += f" | {k:>18}"
  print(head)
  for lo, hi in BANDS:
    b = f"{lo:g}-{hi:g}"
    c = bands.get(b)
    if not c or c["turn"] < 200:
      continue
    line = f"  {b:>9}"
    base = None
    for k in caps:
      s = c["sim"][k]
      if s["n"] == 0:
        line += f" | {DASH:>18}"
        continue
      mean = s["sum_la"] / s["n"]
      base = mean if base is None else base
      clamped = 100.0 * s["accel"] / s["n"]
      gain = 100.0 * (mean - base) / base if base else 0.0
      line += f" | {mean:6.3f} {clamped:5.1f}% {gain:+5.1f}%"
    print(line)

  print("")
  print(f"ACCEL-CLAMP EPISODES >= {MIN_EPISODE_FRAMES * DT:.1f} s, and the path they cost")
  print("  offset = the curvature deficit double-integrated over the episode. It is zero outside")
  print("  the clamp, so it carries no integration bias, but it is OPEN LOOP: the model re-plans")
  print("  every 50 ms and the driver intervenes, so treat it as an upper bound.")
  ep_row = row(9, 8, None, 7, 7, 7, None, 7, 7, 7, None, 7)
  print(ep_row("band m/s", "episodes", "dur p50", "p90", "max",
               "off p50", "p90", "max", "cut p50"))
  for lo, hi in BANDS:
    b = f"{lo:g}-{hi:g}"
    c = bands.get(b)
    if not c or len(c["episodes"]) < 5:
      continue
    dur = [e[0] for e in c["episodes"]]
    off = [e[1] for e in c["episodes"]]
    cut = [e[2] for e in c["episodes"]]
    print(ep_row(b, len(dur),
                 f"{pct(dur, 50):.2f}", f"{pct(dur, 90):.2f}", f"{max(dur):.2f}",
                 f"{pct(off, 50):.2f}", f"{pct(off, 90):.2f}", f"{max(off):.2f}",
                 f"{pct(cut, 50):.3f}"))

  sat = sum(c["sat"] for c in bands.values())
  print("")
  print("WHAT RAISES \"Turn Exceeds Steering Limit\"")
  sat_pct = 100.0 * sat / total if total else 0.0
  print(f"  latched torqueState.saturated frames: {sat} of {total} engaged ({sat_pct:.4f}%)")
  if sat:
    acc = sum(c["sat_accel"] for c in bands.values())
    pin = sum(c["sat_pinned"] for c in bands.values())
    nei = sum(c["sat_neither"] for c in bands.values())
    print(f"    accel-clamped on that frame: {100.0 * acc / sat:5.1f}%")
    print(f"    torque output pinned:        {100.0 * pin / sat:5.1f}%")
    print(f"    neither:                     {100.0 * nei / sat:5.1f}%")
    print("  latcontrol.py ORs curvature_limited into the saturation timer, and drive_helpers")
    print("  sets that flag from the accel and MAX_CURVATURE clamps only -- never the jerk clamp.")
    per = [(b, c["sat"]) for b, c in bands.items() if c["sat"]]
    print("  by band: " + "  ".join(f"{b}={n}" for b, n in sorted(per)))
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("scan")
  s.set_defaults(fn=cmd_scan)
  s.add_argument("--routes", default="/data/media/0/realdata")
  s.add_argument("--out", default="curvature_budget.jsonl")
  s.add_argument("--limit", type=int, default=0)
  s.add_argument("--force", action="store_true")
  s.add_argument("--caps", default="3.0,3.6,4.0,4.5",
                 help="flat lateral-accel limits to price, in m/s^2; the schedule compiled"
                      + " into drive_helpers.py is always added")
  r = sub.add_parser("report")
  r.set_defaults(fn=cmd_report)
  r.add_argument("--out", default="curvature_budget.jsonl")
  a = ap.parse_args()
  return a.fn(a)


if __name__ == "__main__":
  sys.exit(main())
