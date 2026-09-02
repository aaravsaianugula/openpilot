# Road test — CN7 Elantra lateral work

**Governing rule:** steering is commanded only through Hyundai's own LKAS11
`CR_Lkas_StrToqReq`. The MDPS remains the final arbiter — its boost curve, its fault logic, its
driver-override arbitration. We are adjusting a request it is free to refuse.
<https://blog.comma.ai/safer-control-of-steering/>

---

## Read this first — what is actually on the car

The speed schedule that used to be here is gone. The ceiling is now a **flat 409 counts at every
speed**, which is what carrotpilot ships for HKG. Everything below was rewritten for it; an
earlier revision told you to feel for a torque step at 8 and 16 m/s, and to expect the freeway to
be bit-identical. Both are now false.

### 409 is the MDPS's acceptance limit — it accepts 409 and trips at 410. Settled 2026-09-01.

Two builds went above 409 and both faulted the EPS. That much was known. What was *not* known was
whether the cause was the **value** or the **speed-scheduled mechanism** both builds shared, and
an earlier revision of this document said the question was open and untested. **It is now closed,
from the recorded drives, and the answer is the value.**

The two raised drives, identified from each route's **pinned opendbc gitlink**, not from a label:

| route | openpilot | opendbc pin | ceiling | engaged | frames > 409 | ToiFlt onsets / frames | driver saw `steerTempUnavailable` |
|---|---|---|---|---|---|---|---|
| `000000dc` | `4f9f205da` | `eb08f1481` | 500 schedule | 16,110 | **115** | **14 / 14** | **8** |
| `000000dd` | `ff8e23637` | `5a8ae2e83` | 450 schedule | 6,262 | **43** | **5 / 6** | **2** |
| `000000da` | `840b9ea8c` | `1ded2adf8` | flat 409 | 57,016 | 0 | 0 / 0 | 0 |
| `000000db` | `840b9ea8c` | `1ded2adf8` | flat 409 | 64,323 | 0 | 1 † / 1 | 0 |

An **onset** is a 0→1 transition of the bit; **frames** is how long it stayed set. One `dd` event
lasted two frames, which is why its two numbers differ — count onsets, not frames, when comparing
builds.

† `db`'s single onset is at `cmd = 0`, wheel at −451°, driver at full column saturation (−1024)
and 0.6 m/s: the driver hard against the steering stops. It is **not** a torque-ceiling fault and
is excluded from the 19 below — which is why the flat-409 side still reads zero for the thing
being measured.

**The commanded count at each of the 19 fault onsets:**

```
410  412  413  421  426  428  428  429  429  429
429  430  431  432  432  432  432  432  433
```

Minimum **410**, against ~1.59M engaged frames at ≤ 409 with none at all.

**Why this is the value and not the mechanism.** Read off the pinned source, not the commit
messages — `eb08f1481` has `STEER_MAX_LOOKUP = [8.94, 13.41], [500, 409]` and `5a8ae2e83` has
`[8.94, 13.41], [450, 409]`. `np.interp` **clamps below its first breakpoint**, so under 8.94 m/s
the ceiling was a *constant* (500 on `dc`, 450 on `dd`). Every one of the 19 onsets happened
between **2.1 and 8.6 m/s**, i.e. under a stationary ceiling. The flat-ceiling experiment believed
never to have run was run inside the schedule's own flat region, and it faulted. **A flat build
above 409 would fault the same way and must not be driven.**

One caveat stated precisely: the onset speeds above are `carState.vEgo`, while `steer_max_at()`
reads `vEgoRaw`. The two differ by well under 0.1 m/s, but the single highest onset (8.64 m/s)
sits only 0.30 m/s under the breakpoint, so that one alone cannot be called flat with certainty.
The other **18 are at ≤ 7.25 m/s**, and the conclusion rests on those.

**Why it was missed.** The scanner read `CF_Mdps_ToiFlt` (bit 14) alone, and only while
`latActive` was true. A fault that *disengages* the car sets the bit once `latActive` is already
false, so the engaged-only counter scored exactly the events it existed to catch as zero — which
is where "3 fault frames in 1.59M" came from. `CF_Mdps_ToiUnavail` (12) and `CF_Mdps_FailStat`
(15) were never read at all. `.elantra/lateral_report.py` now carries every status bit plus a 2 s
post-disengagement window; `.elantra/eps_census.py` is the full scanner.

- **panda will not protect you here.** It enforces 512, so a build that commands 450 sails
  straight through and the EPS is what stops you. What holds 409 is
  `CarControllerParams.STEER_MAX` and the guards around it, nothing downstream.
- **If you see an EPS fault or a steering dropout on this drive**, the first thing to check is
  what the build actually commands — and capture the route, because the log of a fault
  alongside its commanded counts is exactly the evidence that would settle this.

State as of 2026-08-30, read off the device:

| | |
|---|---|
| `/data/openpilot` | `elantra-lateral` @ `4196916071`, working tree clean |
| `opendbc_repo` | pinned at `b96b64af`, matching the gitlink and the manifest |
| panda firmware | rebuilt from the installed source, two clean builds byte-identical, signature verified on the live board |
| `NeuralNetworkLateralControl` | **0 — deliberately off, so the ceiling is the only live change** |
| `LateralJerkTorqueController` | 1 |
| lateral tune | live learner **valid**: `latAccelFactor = 2.947`, `friction = 0.100`, `decay = 250` (converged, 10204 bucket points). Offline seed is `3.169` / `0.0819`. |
| `DisableUpdates` | 1 — nothing will move the branch under you mid-test |

### Pre-flight — and why the obvious check lies

`CarParamsPersistent` is written **at the start of a drive**. Read it while parked and you get
the *previous* drive's value, which is how an older pre-flight reported the flag as False on a
car that was fully flashed and one ignition away from commanding 409.

Compute it from the installed source instead. This does not depend on drive state:

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo /usr/local/venv/bin/python3 -c "
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, CarControllerParams
CP = CarInterface.get_params(CAR.HYUNDAI_ELANTRA_2024, {0:{0x391:8},1:{},2:{}}, [], True, False, False)
sp = CP.safetyConfigs[-1].safetyParam
print(\"safetyParam\", sp, \"| RAISED_LIMITS:\", bool(sp & 1024), \"| LFAHDA_MFC_8:\", bool(sp & 2048))
print(\"STEER_MAX   \", CarControllerParams(CP).STEER_MAX)"'
```

Expect `safetyParam 3084 | RAISED_LIMITS: True | LFAHDA_MFC_8: True` and `STEER_MAX 409`.

3084 = LONG(4) | CAMERA_SCC(8) | RAISED_LIMITS(1024) | LFAHDA_MFC_8(2048). It was 1036 on both
the schedule build and the first flat build, because those shared the same bit values — which is
why an earlier revision of this document said the safetyParam could not tell the builds apart.
The 2048 bit is new as of 2026-08-31, so it does distinguish them again. Do not rely on that
staying true: **the check that cannot go stale is the panda signature**, and `STEER_MAX 409`
with no lookup for the opendbc half.

---

## Test 1 — the flat torque ceiling

### What it does

The LKAS11 torque ceiling is **409 counts at every speed**, replacing 384 (and replacing the
schedule that already gave 409 below 8 m/s).

**Panda is set to 512, not 409** — carrotpilot parity, and what let the ceiling move during the
500/450 experiments without a reflash. It accepts exactly 512 and rejects 513, at every speed,
including before it has received a single speed frame. So panda is *not* the thing holding the
command at 409, and the 103 counts between them are unenforced. That band is exactly where the
EPS faults, which is why the checks that pin opendbc's 409 are the ones that matter.

409 against 384 is **+6.5%**. This platform's DBC carries `CR_Lkas_StrToqReq` as raw counts —
`16|11@1+ (1.0,-1024.0)` — so the familiar "3.20 Nm vs 3.00 Nm" is an inference from the Nm
scaling in `hyundai_2015_ccan.dbc`, a file this car does not use. Treat the Nm figure as
indicative, not measured. The wire maximum is 1023 counts; nothing here goes near it.

### Where the change actually lands — and it is not where you think

Measured over **all 509 recorded segments, 1,672,068 engaged frames**, by re-running the real
limiter frame-by-frame under both ceilings seeded from the torque the car was actually applying
(`.elantra/torque_projection.py`):

| speed | frames | frames changed | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| 0–3 m/s | 117,741 | **0.00%** | — | 0 |
| 3–7 m/s | 253,136 | **0.00%** | — | 0 |
| 7–10 m/s | 184,012 | 4.92% | 1.30 | 6 |
| 10–14 m/s | 215,269 | 34.69% | 1.52 | 10 |
| 14–18 m/s | 289,502 | 55.58% | 1.96 | 10 |
| 18+ m/s | 612,408 | 53.73% | 1.92 | 10 |
| **all** | **1,672,068** | **34.31%** | ~1.9 | 10 |

**Read that table before you drive.** The car already had 409 below 8 m/s under the schedule, so
this change does **nothing at all** in the low-speed band the original shortfall measurement was
about. Its entire practical effect is at road and highway speed, where it moves about half of all
frames by roughly two counts. That WAS the regime with no road data; it no longer is (see
"What flat 409 has actually driven" below).

That inverted the old test plan: the freeway used to be the control and became the experiment.
It has since been driven -- see below -- so it is now an experiment with results.

For reference, how often the command is actually pinned at its ceiling on the current build:
0.279% of frames at 0–3 m/s, 0.346% at 3–7, 0.023% at 7–10, ~0 above. The schedule already
largely closed the low-speed deficit.

### The factory envelope: 157 counts, and it IS measured

An earlier revision of this document said your own logs could not answer this, because the
camera never actuates while openpilot is engaged, and set up a passive watcher that "may never
appear". That was wrong, and the answer was already in the logs.

Route `000000b9--0c339ed202` — 18 segments, **zero** LKAS11 frames transmitted by openpilot
across all of them — is a drive with stock LKAS in full control. In its 40,144 actuating frames
the factory camera requests up to **157 counts** with `CF_Lkas_ActToi` set. Across all 509
segments that is the maximum; the other 495 are zero for exactly the reason previously given.

**What it does not mean.** It is not an argument for a 157-count ceiling. Hyundai's LKAS is a
lane-centring feature: it nudges toward a lane line and hands back rather than pulling harder —
every peak request in that drive lands at **1.1–2.0 degrees of wheel angle**. openpilot steers
through turns and intersections at up to ~146 degrees. So 157 bounds what Hyundai's *feature*
asks for, not what the MDPS can deliver, and not a safety limit.

| | counts | wheel angle at peak |
|---|---|---|
| Hyundai's LKAS, measured | 157 | 1.1–2.0° |
| comma's HKG default (what you ran before) | 384 | — |
| flat, what the car runs | 409 | — |
| **the highest the EPS accepts** | **409** | measured; 410 trips `ToiFlt` |
| what panda would let out | 512 | unenforced, and 410–512 all faults |

`values.py` quotes comma's rule — *"find the maximum value that the stock LKAS will request"* —
and that rule is simply the wrong instrument here: it was written for cars whose stock LKAS
attempts the same manoeuvres openpilot does. The CN7's does not. 384 and 409 both rest on fleet
evidence instead, which is the honest basis to cite. **The MDPS is the arbiter either way** — its
boost curve, its fault logic, its override arbitration, and it is free to refuse.

### Two costs, stated plainly

- **STEER_MAX is a gain, not just a ceiling.** Every command scales by 409/384 = **+6.51%**, at
  all speeds. The live torque learner absorbs this: it has ±100% headroom here (`factor_sanity`
  is 1.0 because `EnforceTorqueControl` and `LiveTorqueParamsRelaxedToggle` are both set) and
  `STEER_MAX` is not in its cache key, so it does not reset. As of 2026-08-31 it is **valid and
  converged** — `latAccelFactor 2.947`, `friction 0.100`, `decay 250` (the ceiling), 10204 bucket
  points — so the gain is already being trimmed. **But the learner only samples above
  `MIN_VEL = 15 m/s`, so nothing compensates it below ~34 mph.**
- **Driver override yields later.** The ceiling anchors the override envelope:
  `driver_max_torque = STEER_MAX + (50 + driver)*2`. Override still *begins* reducing authority
  at driver torque −50 either way, but the point of **full yield** moves from **−242 to −254.5
  counts** (+5.2%). This is the path that has to work when you fight the wheel — test it
  deliberately. Confirmed by executable test, not arithmetic
  (`.elantra/test_torque_projection.py`).

### Drive it

Empty lot first. Hands on the wheel throughout.

1. **Full-lock turns at walking pace.** Expect **nothing to have changed** — the ceiling here was
   already 409. If something feels different, the build is not what this document says it is.
   Stop and re-run the pre-flight.
2. **Deliberately override mid-turn at ~5 m/s.** Also unchanged from the schedule build. It must
   wind down the way it does today.
3. Quiet streets, real intersections at 5–8 m/s. Still unchanged.
4. **Highway. This is the test.** ~55% of frames here move, by about two counts. This is the one
   band the learner does sample (>15 m/s), so it trims the +6.51% over time. Watch for tracking
   that feels sharper or twitchier than you are used to, and for any oscillation on a long
   constant-radius curve.

### What flat 409 has actually driven

Re-scanned 2026-09-01 with `.elantra/eps_census.py`, which reads every MDPS12 status bit rather
than `ToiFlt` alone and counts faults in a 2 s window after lateral drops out as well as during
it. Both changes matter: the previous counter's engaged-only filter is blind to a fault that
*causes* the disengagement, and that is the only kind this car actually produces.

| route | build | engaged frames | ToiFlt onsets while steering | max \|counts\| |
|---|---|---|---|---|
| 000000c9 | **flat 409** | 160,664 | **0** | 409 |
| 000000ca | **flat 409** | 172,952 | **0** | 409 |
| 000000da | **flat 409** | 57,016 | **0** | 409 |
| 000000db | **flat 409** | 64,323 | **0** † | 409 |
| 000000dc | 500 schedule | 16,110 | **14** | **435** |
| 000000dd | 450 schedule | 6,262 | **5** | **436** |

† the full-lock event described above, at `cmd = 0`.

Flat 409 has ~1.59M engaged frames behind it with **zero** ToiFlt onsets while steering and zero
`steerTempUnavailable`, and the ceiling is genuinely reached (2.2% of engaged frames sit in the
top 25-count bin). The two raised builds fault within 22,000 frames between them.

This half is no longer merely observational. The raised and flat builds differ in the one
variable, the fault fires only on the raised side, and it fires at a sharp, reproducible
threshold — see the onset list above.

### The car is not using the ceiling it already has

Measured with `.elantra/demand_decomp.py` over routes `000000e2`/`e3` — 158,969 engaged frames,
controller **v0** in every one of them:

| band | asking for 100% | **applied when asking for 100%** | reaches 409 | slew rate binding | counts lost |
|---|---|---|---|---|---|
| 0–3 m/s | 45.4% | **89** | 3.2% | 71.0% | 247 |
| 3–7 m/s | 38.7% | **236** | 20.2% | 72.8% | 124 |
| 7–10 m/s | 13.9% | **267** | 13.8% | 68.7% | 46 |
| 10–14 m/s | 1.9% | 210 | 1.5% | 42.8% | 7 |

**When the controller commands its full 409 at 3–7 m/s, the median that actually reaches the EPS
is 236 counts.** `STEER_DELTA_UP` is 3 counts/frame, so 0 → 409 takes 1.36 s and a tight low-speed
turn is over before the limiter arrives. This is the real low-speed shortfall, and the ceiling is
not what causes it.

Three losses sit upstream of the ceiling, in order of size:

1. **The slew rate.** 68–73% of low-speed frames are rate-limited, median 124 counts short at
   3–7 m/s. `apply_driver_steer_torque_limits` is doing this, and **panda enforces the same 3/7**
   (`HYUNDAI_LIMITS(512, 3, 7)`), so opendbc cannot move it alone.
2. **The feedforward model is calibrated out of domain.** `selfdrive/locationd/torqued.py` learns
   `latAccelFactor` only from `vego > MIN_VEL` (**15 m/s**) *and* `|lat_accel| <= LAT_ACC_THRESHOLD`
   (**1 m/s²**) — highway, gentle curves only. It is converged and in use (`calPerc 100`,
   `useParams True`) at **2.84** on `e2` and **3.49** on `e9`, and then applied unchanged at 4 m/s
   and 3 m/s². `torque_from_lateral_accel` divides by that single speed-independent constant, so
   the feedforward under-commands down low and the P term (KP 11.5–30 there) drives to the rail.
3. **The driver-allowance clip**, binding on **42%** of 3–7 m/s frames — column torque opposing
   past the 50-count allowance pulls `max_steer_allowed` below `STEER_MAX`.

A fourth, smaller one: the integrator is **frozen below 5 m/s** (`CS.vEgo < 5` in both
`latcontrol_torque.py` and `latcontrol_torque_v0.py`), and `steer_limited_by_safety` freezes it
again whenever the command is being clipped — so saturation suppresses the integral action that
would resolve it.

**What would actually recover it.** `.elantra/ceiling_replay.py` replays the recorded normalised
torque through opendbc's own limiter under a grid, over 64,256 frames at 3–14 m/s:

| ceiling | rate | what it needs | mean applied | % of ask | % at ceiling | vs today |
|---|---|---|---|---|---|---|
| 409 | 3/7 | *nothing — today* | 101.0 | 68.4 | 3.03 | — |
| 409 | 4/7 | panda rate | 106.8 | 72.4 | 3.90 | +5.7% |
| **409** | **10/10** | panda rate + `max_rt_delta` | **116.5** | 78.9 | 6.58 | **+15.3%** |
| 450 | 3/7 | opendbc only — **and the EPS refuses it** | 114.1 | 70.2 | 2.85 | +13.0% |

**Raising the slew rate alone beats raising the ceiling, and never commands a count above 409.**
Note the `% at ceiling` column: raising the ceiling makes it *fall* (3.03 → 2.85), because the ask
moves further out of reach; raising the rate more than doubles it.

Two caveats, both real. The replay is **open loop** — the recorded `actuators.torque` is held
fixed while the limits move, so it *overstates* every row; the ranking survives, the absolute
numbers do not. And a rate change is **a panda reflash, not an opendbc edit**: panda caps both
`max_rate_up` (3) and `max_rt_delta` (112 counts per 250 ms). At 100 Hz that second limit alone
caps sustained slew at 4.48 counts/frame, so raising `max_rate_up` to 10 without also raising
`max_rt_delta` buys a safety fault rather than a faster ramp. Fixing the tune (2) is the change
that needs no reflash at all.

### Stop if

- Any new EPS fault, or "Steering Assist Temporarily Unavailable". Both are logged per drive —
  but check the log with the **current** `lateral_report.py`, which reads every MDPS12 status bit
  and counts faults for 2 s after lateral drops out. The older counter read `CF_Mdps_ToiFlt`
  alone while `latActive` was true, and a fault that *disengages the car* sets its bit once
  `latActive` is already false, so it reported zero for every real event on this car.
- Any oscillation or hunting, **especially on the highway** — that is where the gain increase is
  compensated only above 15 m/s by the learner, and where the ceiling is genuinely reached.
- Overriding is harder than the +5.2% above would explain.
- Anything at all different below 8 m/s. Nothing changed there; a difference means the build is
  not what you think it is.

### Roll it back

**"The schedule build" is ambiguous and one of the three is not safe.** Three different speed
schedules have existed on this car, and only the first stays inside what the EPS accepts:

| opendbc | schedule | peak | verdict |
|---|---|---|---|
| `54267ecca` | `[[8, 16], [409, 384]]` | **409** | safe — ramps *down* from 409, never exceeds it |
| `eb08f1481` | `[[8.94, 13.41], [500, 409]]` | 500 | **faults the EPS** (route `000000dc`) |
| `5a8ae2e83` | `[[8.94, 13.41], [450, 409]]` | 450 | **faults the EPS** (route `000000dd`) |

The rollback below targets `bb90e6d94`, which pins `54267ecca` — the 409-peak schedule, with one
clean road test behind it. Confirm the pin before you flash anything:
`git ls-tree <commit> opendbc_repo`.

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && git checkout bb90e6d94 && git submodule update --init --recursive opendbc_repo && sudo reboot'
```

pandad reflashes on boot whenever the running signature differs from the built binary, so the
panda follows the source automatically. This round trip has been executed and verified in both
directions. Confirm it landed the way you expect with the pre-flight block above, which computes
the ceiling from the installed source rather than reading `CarParamsPersistent` — that param is
written at the *start* of a drive, so reading it while parked gives you the previous drive's
value.

---

## Test 2 — NNLC, on its own drive

Currently **off**. Turn it on only after Test 1 has a result, or you cannot attribute anything.

```bash
ssh comma@192.168.12.238 '/usr/local/venv/bin/python3 -c "from openpilot.common.params import Params; Params().put_bool(\"NeuralNetworkLateralControl\", True, block=True)"'
```

It takes effect on the next start, not immediately. Two things worth knowing before you do:

- Enabling NNLC also clears `EnforceTorqueControl` automatically. On this car that changes
  nothing — with both toggles off, `get_params` already produces the same
  `latAccelFactor`/`friction` that `configure_torque_tune` would — but it is not a no-op in
  general.
- Prove it is actually running rather than trusting the toggle, which has several silent no-op
  paths: `_initialize_neural_network_lateral_control(CP, CP_SP, params)` must return `True` and
  resolve a model that is not `MOCK`.

NNLC replaces the single linear feedforward with a neural one trained on CN7 data. Measured on
the device: **+14% at 4 m/s**, **+9% at 5 m/s**, **−26% at 25 m/s**. So expect more effort in
slow turns and less on the highway. The highway change is not a side effect to tolerate — it is
the same correction. One `latAccelFactor` cannot be right at both 5 and 25 m/s.

Same abort criteria as Test 1, except that a highway difference is *expected* here rather than a
stop signal.

---

## After either drive — the measurement

A **systemd timer** (`elantra-lateral-watch.timer`, every 30 min, offroad only) scans each new
route and writes a per-route report. This replaced a crontab entry, which never worked across a
power cycle: `/var` is a tmpfs on this device, so the crontab is wiped on every boot.

```bash
ssh comma@192.168.12.238 'systemctl list-timers elantra-lateral-watch.timer --no-pager'
```

To force a scan and list the configurations it has seen:

```bash
ssh comma@192.168.12.238 '/data/elantra-lateral/lateral_watch.sh; \
  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 /data/elantra-lateral/lateral_report.py \
  compare --out /data/elantra-lateral/reports'
```

Then pass two tags to compare them:

```bash
ssh comma@192.168.12.238 'PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 \
  /data/elantra-lateral/lateral_report.py compare <before-tag> <after-tag> \
  --out /data/elantra-lateral/reports'
```

It refuses to compare across different configurations, refuses if either side lost a segment or
a whole route, refuses if either side has **zero engaged frames** — that last case used to print
"Both sides clean" over an empty table — and computes the pinned fraction twice by independent
routes, reporting DISAGREE on either an absolute or a relative gap rather than averaging.

> **Check which version the device is running before you trust its reports.** The watcher runs
> from `/data/elantra-lateral/lateral_report.py`, which is a *copy* — it does not follow the repo.
> The fault-window and status-bit fixes above are in the repo version; if the device still has the
> older file, every report it writes is scored with the blind spot. Compare and update with:
>
> ```bash
> ssh comma@192.168.12.238 'grep -c DISENGAGE_WINDOW_FRAMES /data/elantra-lateral/lateral_report.py'
> # 0 means the device is running the OLD counter; push the current one across:
> scp .elantra/lateral_report.py comma@192.168.12.238:/data/elantra-lateral/lateral_report.py
> ```

For a question the per-route report does not answer, three scanners in `.elantra/` go deeper —
`eps_census.py` (what ceiling did a route actually run, and every fault channel),
`demand_decomp.py` (where the low-speed demand goes) and `ceiling_replay.py` (what a ceiling or
rate change would actually reach the car). See the README for how each is run.

What to look at, in order:

1. **The fault columns.** `faults_engaged` and `faults_at_disengage` should both stay at zero,
   for **every** bit and not just `flt`. Anything else is the finding.

   Two traps here, both of which have already produced a wrong answer on this car:
   - **`toi_flt_engaged` alone is not enough.** A fault that *disengages the car* sets its bit
     once `latActive` is already false, so an engaged-only counter scores it as zero. That is why
     `faults_at_disengage` exists, and why the raised-ceiling routes read "0 faults" for weeks.
   - **Bit 14 alone is not enough.** `CF_Mdps_ToiUnavail` (12) and `CF_Mdps_FailStat` (15) are
     separate channels, and openpilot's own fault condition is `ToiUnavail != 0 or ToiFlt != 0`.
   - Still true: bit 14 on `src 1` is *not* a fault — that bus carries something else entirely,
     and accepting it invents 599 phantom faults per segment.

   `CF_Mdps_FailStat` is **not** the torque-ceiling signature. It fires on flat-409 routes too,
   at high steering angle with driver input — a parking-manoeuvre signal. The clean discriminators
   are `flt` while steering and openpilot's own `steerTempUnavailable` event.
2. **`%` pinned in the 3–7 m/s band.** Baseline **4.56%**.
3. **`demand` vs `deliv`** where pinned. Closing this gap is the actual goal.
4. **`FACTORY_ENVELOPE_SEEN.txt`** in `/data/elantra-lateral/`. This was set up when the
   envelope was believed unmeasurable. It has since been measured at **157 counts** from route
   `000000b9--0c339ed202` (see Test 1). The watcher is still worth keeping — one drive is a
   floor, not a maximum — but it is no longer the only source.

---

## The honest limit of this work

The factory envelope is measured — **157 counts** — and it turns out not to be the deciding
number, because Hyundai's lane-centring feature and openpilot's job are not the same task.
carrotpilot ships 409 with no commit message, no comment and no measurement in its history; we
now know *why* that number is right for this car, which its own history never recorded.

**Answered since the first revision of this document:**

- *What the MDPS does at 409 in a tight turn.* It accepts 409 and trips `CF_Mdps_ToiFlt` at 410.
  Measured across 19 fault onsets, all under a stationary ceiling. The arbiter's line is found.
- *Whether the low-speed shortfall is the ceiling.* It is not. When the controller commands its
  full 409 at 3–7 m/s the median that reaches the EPS is **236 counts** — the 3 count/frame slew
  rate gets there first.

**What remains genuinely unknown:**

- **Sustained saturation at 409 above 8 m/s.** Reaching the ceiling there is measured, but only
  in short bursts — a few hundred frames out of ~580k. A long, continuously saturated highway
  curve has still not been driven.
- **Where the tune settles, and it will not settle low.** The learner re-converges against the
  gain only from samples above `MIN_VEL` (15 m/s) and below 1 m/s² of lateral acceleration, so
  the low-speed regime is never compensated at all — not slowly, never. The measured
  lateral-accel-per-unit-torque down there is far from the constant the controller divides by.
- **Whether raising the slew rate actually helps.** `ceiling_replay.py` says +15.3%, but it is
  an open-loop replay: the recorded `actuators.torque` is held fixed while the limits move, and
  a car that finally got what it asked for would back off. The ranking of the options is sound;
  the magnitude is an upper bound. It also needs a panda reflash of **two** numbers —
  `max_rate_up` and `max_rt_delta` — and raising only the first buys a safety fault.
- **Whether the driver-allowance clip is costing as much as it appears.** It binds on 42% of
  3–7 m/s frames, but that measurement does not separate "the driver is fighting the wheel" from
  "the driver is resting a hand on it", and those want opposite responses.
