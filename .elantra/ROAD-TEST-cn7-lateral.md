# Road test — CN7 Elantra lateral work

**Governing rule:** steering is commanded only through Hyundai's own LKAS11
`CR_Lkas_StrToqReq`. The MDPS remains the final arbiter — its boost curve, its fault logic, its
driver-override arbitration. We are adjusting a request it is free to refuse.
<https://blog.comma.ai/safer-control-of-steering/>

---

> **If you are about to drive this: go to Test 6.** It is the drive that is staged
> now -- all three lateral changes live on one drive -- and it supersedes Tests 3, 4 and 5.
> Findings and open questions: `D:/comma_four/tuning/lat-tracking/FINDINGS-2026-09-03.md`.

---

## Read this first — what is actually on the car

The LKAS11 steering torque ceiling is a **flat 409 counts at every speed** on
`HYUNDAI_ELANTRA_2024`, against the 384-count HKG default. `STEER_MAX` is a gain, not just a
clamp, so every command is 6.51% stronger than stock — at every speed, not only where the old
ceiling used to bind. panda enforces a flat 512 and does not police the difference. The MDPS is
the arbiter and is free to refuse any of it.

**Know which build you are coming from before you read the expectations below.** They differ,
and the difference is the whole content of the drive.

### 409 is the MDPS's acceptance limit — it accepts 409 and trips at 410

Two builds commanded above 409, and both faulted the EPS. The drives, identified from each
route's **pinned opendbc gitlink** rather than from a label:

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
between **2.1 and 8.6 m/s**, i.e. under a stationary ceiling. So the schedule is not what
faulted the car; the value is. **A flat build above 409 would fault the same way and must not be
driven** — the experiment has already run, inside the schedule's own flat region.

One caveat stated precisely: the onset speeds above are `carState.vEgo`, while `steer_max_at()`
reads `vEgoRaw`. The two differ by well under 0.1 m/s, but the single highest onset (8.64 m/s)
sits only 0.30 m/s under the breakpoint, so that one alone cannot be called flat with certainty.
The other **18 are at ≤ 7.25 m/s**, and the conclusion rests on those.

**How to count these, and it is not obvious.** Read every MDPS12 status bit —
`CF_Mdps_ToiUnavail` (12), `CF_Mdps_ToiFlt` (14), `CF_Mdps_FailStat` (15); openpilot's own
condition is `ToiUnavail != 0 or ToiFlt != 0`. And count for **2 s after `latActive` drops**, not
only while it is true: a fault that disengages the car sets its bit once lateral is already off,
so an engaged-only counter scores precisely the events it exists to catch as zero.
`.elantra/lateral_report.py` does both; `.elantra/eps_census.py` is the full scanner.

- **panda will not protect you here.** It enforces 512, so a build that commands 450 sails
  straight through and the EPS is what stops you. What holds 409 is
  `CarControllerParams.STEER_MAX` and the guards around it, nothing downstream.
- **If you see an EPS fault or a steering dropout on this drive**, the first thing to check is
  what the build actually commands — and capture the route, because the log of a fault
  alongside its commanded counts is exactly the evidence that would settle this.

Device state is **not recorded here** — a pasted snapshot goes stale silently, and this one
did. Run the pre-flight block below, which computes everything in this table from the installed
source. For reference, what it should look like:

| | |
|---|---|
| `/data/openpilot` | the published `master`, working tree clean |
| `opendbc_repo` | the gitlink, matching `.elantra/build-manifest.json` |
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

3084 = LONG(4) | CAMERA_SCC(8) | RAISED_LIMITS(1024) | LFAHDA_MFC_8(2048).

**`safetyParam` does not identify a build.** Two builds with different ceilings can carry the
same bits, and have. Identify by the opendbc gitlink; verify with the panda signature and with
`STEER_MAX` computed from the installed source, as the block above does.

---

## Test 6 — all three changes, one drive — **THIS IS THE DRIVE**

Supersedes Tests 3, 4 and 5, which staged these across three separate drives. The owner chose one
drive with everything live. Those tests are kept below because their per-change detail is still the
best description of each mechanism, and because if this drive goes wrong they are how you bisect it.

### What is on the car

| # | change | where | inert above |
|---|---|---|---|
| A | lateral-accel clamp 3.0 → **4.0 m/s²**, tapering back to 3.0 by 22 m/s | `drive_helpers.py` | 22 m/s (49 mph) |
| B | `STEER_DRIVER_ALLOWANCE` 50 → **100**, `RAISED_LIMITS` branch only | opendbc `values.py` | — |
| C | low-speed **feedforward schedule** + **KP cap**, `HYUNDAI_ELANTRA_2024` only | `lat_accel_factor_schedule.py` | **15 m/s (34 mph), bit-identically** |

`STEER_MAX` is untouched at a flat 409. No panda reflash: opendbc's raised driver window stays
inside panda's own (the bound is 101, derived; 100 is under it). Change B touches only `values.py`,
pure Python, so it needs no rebuild on the device either.

> **Correction — do not be misled by the opendbc commit message.** `f3492e02`, the commit that
> raises the driver window, ends with "Road-tested per ROAD-TEST-cn7-lateral.md Test 3". **It has
> not been road tested.** That line was written as an instruction and reads as a claim, and Test 3
> is superseded by this section. **None of the three changes on this branch has ever been driven.**
> The message was not rewritten because doing so means force-pushing a branch the device is pinned
> to; it is corrected here, where the person about to drive will actually see it.

### Expect, by speed band — this is how one drive still attributes three changes

They barely overlap, which is what makes a single drive readable.

| speed | what should change | which change owns it |
|---|---|---|
| **0–11 mph** (0–5 m/s) | more torque held through tight turns; less running wide at intersections | **B**, plus C's KP cap, which bites hardest here. A does nothing — the clamp never fires this low. |
| **11–29 mph** (5–13 m/s) | the biggest expected improvement: turn-in arrives sooner and holds | **C** (feedforward; bounded at +5% to +14.4% delivered counts), some B |
| **29–36 mph** (13–16 m/s) | more available curvature; fewer "Turn Exceeds Steering Limit" | **A**, with C tapering to nothing by 34 mph |
| **36–49 mph** (16–22 m/s) | mild; A only, tapering out | **A** alone — B and C are arithmetically inert |
| **above 49 mph** | **nothing whatsoever** | all three inert; the path is bit-identical |

### What would falsify this

- **Any change in behaviour above 49 mph.** All three are arithmetically inert there and the
  feedforward path is bit-identical (`x / 1.0` in IEEE-754). If highway feel changes, something
  other than these three changes is different — **stop and investigate before driving further.**
- **No change at all below 30 mph.** Then the feedforward/KP half is not executing. Run
  `verify_lat_fix.py` on the device: it checks `LateralJerkTorqueController` is on and
  `NeuralNetworkLateralControl` is off, which is the only combination in which this fix does
  anything at all.
- **Improvement only above 36 mph.** That says A is doing the work and C is not — the opposite of
  what the measurement predicts, and worth knowing.
- **Turns that now go in too tight, or overshoot and correct.** That is the feedforward asking for
  more than the car needs. The measurement says the shipped schedule is conservative in every band;
  this would refute that.

### Abort criteria — things you can feel, not numbers

Disengage and end the drive on any of these:

1. **Oscillation or hunting** at low speed — the wheel searching either side of centre. Implicates
   the **KP cap** (C): too little proportional gain, with the feedforward not carrying enough.
   Roll back `LOW_SPEED_KP_BLEND` first.
2. **A turn that goes in tighter than you asked for**, or overshoots and corrects back. Implicates
   the **feedforward** (C). Roll back `LOW_SPEED_FF_BLEND`.
3. **Steering that fights you on a straight**, or new resistance on centre.
4. **Any new fault, or "Turn Exceeds Steering Limit" where it did not appear before.** The clamp
   raise should make that alert *less* frequent, not more.
5. **Anything above 49 mph feeling different at all.**

There is no scenario in which the right response is to keep driving to gather more data.

### Rollback — one constant each, no rebuild, no reflash

Each is a single edit plus `sudo systemctl restart comma`:

```
LOW_SPEED_KP_BLEND = 0.0     # lat_accel_factor_schedule.py -- disables the KP cap alone
LOW_SPEED_FF_BLEND = 0.0     # lat_accel_factor_schedule.py -- disables the feedforward
                             #   (guards then require the KP blend to be 0 too)
LAT_ACCEL_LIMIT_V = [3.0, MAX_LATERAL_ACCEL_NO_ROLL]   # drive_helpers.py -- stock demand
```

B rolls back by pointing the `opendbc_repo` gitlink at `bc4fd936`. To revert everything at once:
`git checkout 4338acc5d` on the device, then restart.

### What was validated offline, and what could not be

**Validated offline.** The feedforward's effect on delivered counts, bounded above and below by a
replay through opendbc's own driver clamp and rate limiter over 3.42M recorded frames: +1.1% at
3–4 m/s rising to +14.4% at 8–13 m/s, and exactly 0.0% above 16 m/s. The plant gain the schedule
rests on, independently reproduced in 8 of 9 speed bands. And that the schedule is conservative —
every value at or above the reproduced ratio.

**Could not be, at all.** The **KP cap**. It is a closed-loop change: a gain change moves the
trajectory, so replaying it against a recorded trajectory answers nothing. This drive is the only
evidence there will ever be.

**Could not be, and it matters.** Whether **A buys anything below 36 mph**. On the reproduced plant
gain the EPS saturates before the stock 3.0 m/s² clamp binds at every speed below 16 m/s, which
would mean A mostly converts "clamped" into "saturated" down there. The one band with enough
genuinely settled frames measured 28% higher, which would reverse that. See
`FINDINGS-2026-09-03.md` §4 — **this drive is what decides it**, which is why the band attribution
above is worth recording carefully.

**No data at all.** Below 2 m/s: the archive holds no hands-off turn frames there, so the correction
in that regime is a bounded extrapolation rather than a measurement. Treat the first few
walking-pace turns as the most uncertain part of the drive.

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

**Read that table before you drive, and read what it is a delta from.** It compares flat 409
against the *speed-scheduled* build, which already gave 409 below 8 m/s — that is why its
low-speed rows are 0.00%. It is the right table if you are coming from the schedule build, and
the wrong one if you are coming from stock.

**From the stock 384 ceiling, every command changes, at every speed.** `STEER_MAX` is a
multiplier — `new_torque = round(actuators.torque * STEER_MAX)` — so the whole command curve
scales by 409/384, including in the low-speed band. There is no equivalent measured table for
that comparison yet: producing one needs `torque_projection.py` re-run against the routes with
`--candidate 384` as the baseline, which needs the car. Until that exists, treat the low-speed
band as **changed and untested from stock**, and drive it as the experiment rather than the
control.

For reference, how often the command is actually pinned at its ceiling **on the flat-409
build**: 0.279% of frames at 0–3 m/s, 0.346% at 3–7, 0.023% at 7–10, ~0 above. The comparable
figure measured **under the 384 ceiling** was 4.56% at 3–7 m/s. Label every one of these with
the ceiling it was measured at, or a pass reads as a regression.

### The factory envelope: 157 counts, and it IS measured

The factory camera almost never actuates while openpilot is engaged, which is why this looks
unmeasurable at first. It is not: one route has openpilot transmitting zero LKAS11 frames, so
stock LKAS was driving, and the answer was already in the logs.

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

Empty lot first. Hands on the wheel throughout. Expectations below are **from the stock 384
ceiling**; if you are coming from the schedule build instead, steps 1–3 should feel unchanged
and step 4 is the whole test.

1. **Full-lock turns at walking pace.** Expect *slightly* more authority than stock — every
   command here is 6.5% stronger. It should feel like more of the same, not like a different
   car. Anything abrupt, and the build is not what this document says it is: stop and re-run
   the pre-flight.
2. **Deliberately override mid-turn at ~5 m/s.** This is the path that must work when you fight
   the wheel, and it is the one the ceiling moves: full yield goes from −242 to −254.5 counts,
   so you push slightly further before it lets go. It must wind down smoothly, not step.
3. **Quiet streets, real intersections at 5–8 m/s.** This is the band the whole change was
   motivated by, and the band with the least data from stock. Watch for the car asking for more
   than it used to in tight turns.
4. **Highway.** ~55% of frames move by about two counts relative to the schedule build. This is
   the one band the torque learner samples (`MIN_VEL = 15 m/s`), so it trims the +6.51% out over
   time; below ~34 mph nothing compensates it. Watch for tracking that feels sharper or
   twitchier than you are used to, and for oscillation on a long constant-radius curve.

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
  but check the log with a `lateral_report.py` that reads every MDPS12 status bit and counts for
  2 s after lateral drops out. Freshness-check the device's copy first (see below); an
  engaged-only counter reports zero for every real event on this car.
- Any oscillation or hunting, **especially on the highway** — that is where the gain increase is
  compensated only above 15 m/s by the learner, and where the ceiling is genuinely reached.
- Overriding is harder than the +5.2% above would explain.
- Anything *abrupt* below 8 m/s. Coming from stock, more authority down there is expected —
  6.5% more — but it should be a smooth increase. A step, a snatch, or a fault is not.

### Roll it back

**Roll back to the last published build**, using the port panel's *roll back to previous build*,
or by pointing the device at `master-previous`. That is a build with a road test behind it and
the flat 409 the EPS accepts.

**Do not roll back to a schedule build.** Three speed schedules have existed on this car and two
of them command above 409, which is the band the EPS refuses. Use this to *recognise* what a car
is running — keyed on the opendbc gitlink, the one identifier that survives a branch being
rebuilt — not as a list of places to go:

| opendbc pin | ceiling | verdict |
|---|---|---|
| `54267ecca` | `[[8, 16], [409, 384]]` schedule, peak 409 | inside what the EPS accepts |
| `eb08f1481` | `[[8.94, 13.41], [500, 409]]` | **faults the EPS** (route `000000dc`, 14 onsets) |
| `5a8ae2e83` | `[[8.94, 13.41], [450, 409]]` | **faults the EPS** (route `000000dd`, 5 onsets) |
| `b96b64af`, `1ded2adf`, `f29553cb` | flat 409 | the current shape |

Confirm any pin before you flash anything: `git ls-tree <commit> opendbc_repo`.

**These shas are reachable only because they were tagged.** Both repos carry
`archive/pre-rebuild-elantra-lateral`, cut before the branches were rebuilt; without it none of
the pins above would resolve, because `master-previous` is a rotating pointer and not an archive.
Tag before rewriting a branch, or this table becomes unusable.

```bash
ssh comma@<device> 'cd /data/openpilot && git checkout <sha> && git submodule update --init --recursive opendbc_repo && sudo reboot'
```

pandad reflashes on boot whenever the running signature differs from the built binary, so the
panda follows the source automatically. Confirm it landed the way you expect with the pre-flight
block above, which computes the ceiling from the installed source rather than reading
`CarParamsPersistent` — that param is written at the *start* of a drive, so reading it while
parked gives you the previous drive's value.

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

## Test 3 — the lateral-acceleration clamp AND the driver window, one drive

**TWO CHANGES ARE ON THIS DRIVE, deliberately bundled.** They are independent, they act on
different halves of the problem, and — the part that makes bundling survivable — they barely
overlap in speed, so the after-scan can still tell them apart:

| | what it changes | where it acts |
|---|---|---|
| **A. lateral-accel clamp** | how much curvature openpilot will *ask for*, in `clip_curvature` | **0.0% below 9 mph**, growing to +18.8% at 31–36 mph, **0.0% above 49 mph** |
| **B. driver window** | how much of the 409 counts survives the column-torque sensor, `STEER_DRIVER_ALLOWANCE` 50 → 100 | +27.2% at 2–7 mph, falling to +5.0% at 31–40 mph |

So: **under 9 mph, only B can be doing anything** — A's clamp provably never fires down there, so
anything you feel at walking pace is the driver window. **At 27–36 mph A dominates** (+18.8%
against B's +5.0%). The middle band is genuinely mixed and you should not try to attribute it.

Neither touches the torque ceiling. `STEER_MAX` is still a flat 409, the rate is still 3/7, and
panda is not reflashed — B stays strictly inside the driver window panda already enforces.

### A — the lateral-acceleration clamp

**A is a demand-side change, not a torque change.** What it alters is how much curvature openpilot
is *willing to ask for* before the steering controller ever sees the number.

`clip_curvature` (`selfdrive/controls/lib/drive_helpers.py`) caps the commanded curvature at a
lateral **acceleration** of 3.0 m/s² — an ISO comfort guideline, not a limit of this car. Because
it is an acceleration, the tightest radius it permits is `v²/3.0`: 15 m at 15 mph, 33 m at 22 mph,
81 m at 35 mph. Real turns are 7.5–20 m. Measured over 3.41M engaged frames from the archive, that
clamp is active on **8% of turn frames at 11–13 mph rising to 80% at 31–36 mph**, removes a median
11–22% of what the model asked for, and accounts for **92.3%** of the frames that raise
*"Turn Exceeds Steering Limit"* — against 0.0% of frames above 40 mph, which is why the highway
has always been fine.

The change makes the limit speed-scheduled: **4.0 m/s² below 16 m/s**, tapering to the stock 3.0
by 22 m/s. Above 22 m/s it is arithmetically identical to today.

### What should change, and what should not

Priced by replaying the same recorded model demand through `clip_curvature` sequentially, with the
jerk clamp in the loop (`.elantra/curvature_budget.py`):

| where | mean commanded lateral accel | accel clamp active |
|---|---|---|
| 9–11 mph | +0.6% | 0.7% → 0.0% |
| 11–13 mph | +2.4% | 2.9% → 0.1% |
| 13–16 mph | +3.9% | 5.2% → 0.9% |
| 16–18 mph | **+7.9%** | 12.6% → 4.1% |
| 18–22 mph | **+8.8%** | 6.6% → 2.1% |
| 22–27 mph | +5.0% | 7.3% → 1.9% |
| 27–31 mph | **+9.6%** | 46.9% → 7.6% |
| 31–36 mph | **+18.2%** | 78.6% → 30.7% |
| 40 mph and up | **0.0%** | unchanged |

**Below about 11 mph, expect nothing.** Under 3.87 m/s the flat `MAX_CURVATURE = 0.2` clamp (a 5 m
radius, which is roughly this car's kerb radius) binds before the acceleration one, so there is
nothing there for this change to release. That band is torque-limited and is a separate problem.
**No change at walking pace is a predicted result, not a failed drive.**

### B — the driver window

`opendbc/car/hyundai/values.py`, inside the `RAISED_LIMITS` branch only:
`STEER_DRIVER_ALLOWANCE` 50 → 100. No reflash.

`CR_Mdps_StrColTq` is the **column** torque, and on this car it carries the EPS's own reaction to
road load, not just your hands. Measured on straight-line, hands-off, engaged frames it rises with
openpilot's own command — median 11 counts at 0–20 applied, 53 at 120–200, 83 at 200–300 — and it
**opposes** the command on 70–83% of frames above 120 counts, the opposing magnitude falling
monotonically with speed (114 counts at 2–3 m/s down to 26 at 18–25). That is road load.

`apply_driver_steer_torque_limits` then cuts the 409 ceiling by twice the excess over the
allowance. At 50, across 3.42M engaged frames, the effective ceiling falls **below 350 counts on
32–46% of hands-off frames under 10 m/s**, and to ~200 at the 10th percentile. The car has been
throttling itself back on a signal nobody generated.

Replayed through opendbc's own limiter over the whole archive, mean applied counts:

| band | 1–3 m/s | 3–5 | 5–7 | 7–10 | 10–14 | 14–18 |
|---|---|---|---|---|---|---|
| allowance 50 | 81.6 | 105.9 | 104.0 | 71.0 | 47.4 | 34.1 |
| allowance 100 | 103.8 | 128.9 | 122.3 | 81.4 | 51.0 | 35.8 |
| gain | **+27.2%** | **+21.7%** | **+17.6%** | **+14.6%** | +7.6% | +5.0% |

**Why 100 needs no reflash.** panda runs the same clamp shape (`TorqueDriverLimited`, allowance
50, multiplier 2) against `max_torque` 512 rather than opendbc's 409, so its window is 103 counts
wider at every driver torque. Solving `409 + (A + d)*2 <= 512 + (50 + d)*2` for all `d` gives
`A <= 101.5`. `.elantra/test_ceiling_replay.py` proves that bound exact against opendbc's own
limiter and panda's own constants: **101 never exceeds panda, 102 does.** 100 leaves a count of
margin.

**WHAT IT COSTS YOU, and this is the part to be awake to.** The car yields to a real hand later.
Full yield moves from −254.5 to −304.5 counts of driver torque — you have to push about 20% harder
before openpilot lets go completely. `STEER_THRESHOLD` is unchanged at 150, so it still *notices*
a held wheel at the same point; it just stops backing off so early. **Test this deliberately in
the empty lot before taking it near traffic.**

### The drive

Empty lot first, and this time the lot has a job to do rather than being a formality.

0. **In the lot, at walking pace, deliberately fight the wheel.** This is change B, and it is the
   one thing on this drive that alters how the car responds to *you*. Take over mid-turn the way
   you normally would and confirm it still gives way — it should feel like it holds on slightly
   longer, not like it fights you. If you cannot comfortably override it, stop and revert B. That
   is the whole abort criterion for this half.
1. **Walking pace to 9 mph turns.** Change A provably does nothing here, so anything you feel is
   change B. Expect the car to pull through a tight turn it used to give up on.
2. **11–16 mph turns.** Ordinary intersection right-turns. Expect a *small* difference — a few
   percent more curvature. If it feels dramatically different here, something is wrong: stop.
3. **16–22 mph turns.** Bigger intersections and slip lanes. Expect +8%.
4. **27–36 mph curves.** This is where the change actually lives, and where the alert used to
   fire. Expect the car to hold the line noticeably better, and expect *"Turn Exceeds Steering
   Limit"* to become rare or stop.
5. **Highway.** Confirm nothing moved. It should be indistinguishable. Any difference at all above
   40 mph means the schedule is not what this document says it is — stop and re-read
   `LAT_ACCEL_LIMIT_BP` / `LAT_ACCEL_LIMIT_V`.

Keep both hands ready throughout, as in Test 1. The failure mode this change could introduce is
not weakness, it is over-eagerness.

### Abort criteria — additional to Test 1's

Test 1's criteria all still apply: any new EPS fault on any MDPS bit, any oscillation, anything
abrupt below 8 m/s. In addition, stop the drive on:

- **Any oscillation or weave at 27–40 mph.** This is the band with the largest authority increase
  and it is the one place a comfort clamp was doing real work. It is the specific risk of this
  change.
- **Turn-in that feels snatchy rather than firmer.** More commanded curvature should read as the
  car committing to the corner earlier, not as a jerk. The lateral-jerk clamp is deliberately
  untouched, so if the *rate* feels different, something other than this change moved.
- **Any difference on the highway at all.**
- **The wheel feeling reluctant to give way.** This is change B and it is the criterion that
  matters most: you must be able to take over at any moment without effort you would not have
  expected. Revert B alone if so — the curvature change does not touch this.

### Rollback

**They roll back independently, and B is the one to drop first** if the car feels reluctant to
give way: it is the only change that touches your authority over the wheel. Set
`STEER_DRIVER_ALLOWANCE = 50` in `opendbc/car/hyundai/values.py`, inside the `RAISED_LIMITS`
branch.

For **A** — one constant, one file, no reflash. Set `LAT_ACCEL_LIMIT_V = [3.0, 3.0]` — or equivalently
`[MAX_LATERAL_ACCEL_NO_ROLL, MAX_LATERAL_ACCEL_NO_ROLL]` — in
`openpilot/selfdrive/controls/lib/drive_helpers.py` and restart:

```bash
ssh comma@192.168.12.238 'setsid nohup sh -c "sleep 2; sudo systemctl restart comma" >/dev/null 2>&1 </dev/null &'
```

That restores byte-identical stock behaviour at every speed. `.elantra/guards.py` will still pass
with the schedule flattened, deliberately: the guard bounds the schedule, it does not require the
change to be present.

### What to measure afterwards

```bash
python .elantra/curvature_budget.py scan --routes /data/media/0/realdata --out /data/lat-tracking/curvature_after.jsonl
python .elantra/curvature_budget.py report --out /data/lat-tracking/curvature_after.jsonl
```

Read the SELF-CHECK block first. It must say the offline replay reproduces the logged
`desiredCurvature` on essentially every frame; if it does not, the tool and the car have diverged
and no other number in the report means anything.

Then compare against the before-scan on **the accel-clamp rate**, not on achieved lateral accel.
The clamp rate is deterministic given the demand — it needs no cross-process join and no build
normalisation — whereas achieved accel varies with what roads you happened to drive. The
prediction is 78.6% → 30.7% at 31–36 mph and 46.9% → 7.6% at 27–31 mph.

**What would say this change is worthless:** the clamp rate falls as predicted but the achieved
lateral accel on those same frames does not rise. That would mean the torque chain was the binding
constraint all along and the clamp was innocent — revert. The archive says the car has the
headroom (on clamped frames `|output|` sits at p50 0.905 at 14–18 m/s, with only 33% of them
pinned), but that is an inference from before the change, and the drive is what tests it.

**Sample size, stated in advance.** The 14–18 m/s band holds only 1,341 turn frames in the entire
118-route archive. One drive will not resolve it. Plan on several before/after drives, and do not
read a single drive as a verdict.

---

---

## Test 4 — the low-speed feedforward gain, on its own drive

### What it does

`latAccelFactor` is the constant the controller divides a desired lateral acceleration by to get a
torque request. `torqued.py` fits it only on samples above `MIN_VEL` (15 m/s) **and** below 1 m/s²
of lateral acceleration, and emits one scalar for every speed. The section above already says the
low-speed regime is never compensated. This test is the measurement of how far off it is, and the
correction.

Measured over the 20 archived routes that ran a 409-count build — 344 segments, 1,992,979 frames,
engaged and hands-off, achieved lateral acceleration taken from
`liveLocationKalman.angularVelocityCalibrated` with `sin(roll)·g` subtracted the way `torqued.py`
does it, and the torque series lag-aligned by 0.2 s — the lateral acceleration this car produces at
a full 409-count command is:

| v (m/s) | 3–4 | 4–5 | 5–6 | 6–8 | 8–10 | 10–13 | 13–16 | 16–22 | 22–32 |
|---|---|---|---|---|---|---|---|---|---|
| **m/s² at 409 counts** | 1.08 | 1.57 | 1.90 | 1.99 | 2.20 | 2.40 | 2.81 | 3.08 | 3.04 |
| **÷ 3.157** | 0.34 | 0.50 | 0.60 | 0.63 | 0.70 | 0.76 | 0.89 | 0.98 | 0.96 |

The number to trust this on: at 16–22 m/s the estimate is 3.08 against the 3.157 the learner fits
from exactly that range. Two unrelated methods agreeing to 3% where they overlap is what makes the
2.9× divergence below 8 m/s a measurement rather than an artefact of the estimator.

`openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py` divides **the feedforward
and only the feedforward** by that ratio. The error path keeps the unscheduled factor, so the
closed-loop P gain is unchanged at every speed — this adds no new stability question. Above 15 m/s
the gain is exactly 1.0 and `x / 1.0` is bit-identical in IEEE-754, so the highway is provably
untouched: same feedforward, same anti-windup bound, same integrator, same counts.

### What this will and will not fix — read this before driving

**It will not fix intersection turns under 13 mph, and nothing in the steering chain will.** On
hands-off turn frames below 6 m/s the controller already requests full torque on 81–96% of frames
and already gets ≥405 of its 409 counts on 37–54% of them. It is at the rail. A feedforward
correction cannot help a loop that is already saturated, and the offline replay prices the gain
down there at +1.7%.

Where it should be visible is **13–29 mph**: the replay puts the sustained delivered torque up
8–14% at 6–13 m/s, and the torque now arrives from the feedforward rather than being built by the
proportional term after the car has already run wide.

If the sub-13 mph turns are unchanged, **that is the predicted result, not a failure.**

### Pre-flight, additional to Test 1's

From the first minute of the drive's own log, not from today's params:

- `LateralJerkTorqueController` is `1`. The edit lives in the jerk-aware path and is **inert** if
  that toggle moves. "The fix is not running" must never be read as "the fix did not work".
- `TorqueParamsOverrideEnabled` is `0`. With it and `EnforceTorqueControl` both on,
  `latcontrol_torque_v0.py` re-widens the PID limits every 300th frame without the extension
  re-narrowing them; the 409 ceiling still holds on the wire, but the integrator recovers slower.
- `NeuralNetworkLateralControl` is `0`. NNLC replaces this feedforward outright, so the two cannot
  be judged on the same drive.

### Drive it

Set `LOW_SPEED_FF_BLEND = 0.5` first. It scales the whole correction: `0.0` is arithmetically
today's car, `1.0` is the full measured schedule.

1. **Empty lot.** Hands on. Full-lock manoeuvres at walking pace, both directions. Nothing should
   feel abrupt, and the wheel should not snatch as it loads up.
2. **Walking pace to 13 mph.** Expect almost no difference — see above.
3. **13–29 mph corners.** This is the band under test. The car should begin the turn earlier and
   hold the line further through it.
4. **Highway.** Must be **indistinguishable**. Any difference at all is a bug, not a tuning
   observation: the gain is exactly 1.0 up there.

Then repeat with `LOW_SPEED_FF_BLEND = 1.0`.

Note the lateral-accel clamp (Test 3A, `LAT_ACCEL_LIMIT_V = [4.0, 3.0]`) is present on this build.
It acts on 1–3% of turn frames and is not perceptible from the seat, so it does not meaningfully
confuse this drive. If you want strict attribution, set both entries to
`MAX_LATERAL_ACCEL_NO_ROLL` for drives 1–2 — but its own tests in
`openpilot/selfdrive/controls/tests/test_drive_helpers.py` assert the raise, so they go red while
it is held flat, and that is expected rather than a regression.

### The prediction, made before driving

`.elantra/ff_schedule_replay.py`, over all 1,367,301 engaged frames of the 409-count archive,
restricted to hands-off frames commanding more than 1.5 m/s^2. Mean delivered counts, free-running
chain (an upper bound -- a car that finally got what it asked for would back off and ask for less):

| v (m/s) | frames | before | after | change | % pinned | ask grew, delivery did not |
|---|---|---|---|---|---|---|
| 3-4 | 419 | 367 | 371 | +1.1% | 55% | 2% |
| 4-5 | 2469 | 344 | 363 | +5.5% | 45% | 8% |
| 5-6 | 4931 | 357 | 375 | +5.0% | 41% | 8% |
| 6-8 | 7440 | 335 | 364 | **+8.6%** | 23% | 19% |
| 8-10 | 3256 | 284 | 325 | **+14.4%** | 7% | 31% |
| 10-13 | 4953 | 267 | 306 | **+14.4%** | 2% | 42% |
| 13-16 | 4009 | 285 | 298 | +4.7% | 12% | 33% |
| **16-22** | 7692 | 188 | 188 | **+0.0%** | 0% | 0% |
| **22-99** | 9238 | 131 | 131 | **+0.0%** | 0% | 0% |

Read the last two rows first: **exactly zero** above 16 m/s, which is the property the whole design
rests on. (13-16 moves because the schedule is still below 1.0 between 13 and 15 m/s; it reaches
exactly 1.0 at 15.) The tool's own join self-check -- does the logged `p+i+d+f` reproduce the logged
command -- comes back at 100.0% in every band, so the numbers above are not resting on a mis-joined
trace.

The "ask grew, delivery did not" column is the honest counterweight: at 10-13 m/s, 42% of the
frames where the request rose saw no extra counts reach the EPS at all, because the rate limiter or
the driver clamp took the difference. That is why the free-running column is an upper bound and not
a promise.

### Stop if

Additional to Test 1's abort criteria:

- **Turn-in overshoot with a corrective countersteer at 15–25 mph.** The most likely failure mode.
  The P path is untouched so this is overshoot, not a growing oscillation — but back `BLEND` off
  rather than pressing on.
- **A slow weave at low speed on a constant-radius curve.** A larger feedforward moves the
  anti-windup bound, so `torqueState.i` is the thing to look at afterwards.
- **Sticky-then-abrupt steering.** The new feedforward saturating, then the 3-up/7-down rate
  limiter becoming the binding constraint. The replay's `no-gain%` column predicts where this can
  happen.
- **Any highway difference whatsoever.**

No EPS fault is reachable from this change on its own: the command is clipped to ±409 counts by
`apply_driver_steer_torque_limits` inside opendbc, downstream of the controller, and 409 is the
measured MDPS acceptance limit. `test_lat_accel_factor_schedule.py` asserts that at the limiter
rather than at the PID, for exactly this reason.

### Roll it back

Set every element of `FF_LAT_ACCEL_GAIN_V` to `1.0` in
`openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py`, or
`LOW_SPEED_FF_BLEND = 0.0`. Either makes the gain exactly 1.0 at every speed, and `x / 1.0` is
bit-identical, so the result is not "close to today" but arithmetically today. Then
`systemctl restart comma`. The guards stay green through the rollback by design — the one test
that goes red is `test_low_speed_feedforward_actually_moved`, which is correct, because you
deliberately reverted it.

### What to measure afterwards

- `.elantra/ff_schedule_replay.py --dump-trace` then `--trace --turns-only` on the new drive, and
  compare its per-band delivered counts against the prediction below.
- The gain itself, re-measured on the new drive: if the delivered counts rose at 6–13 m/s but the
  achieved lateral acceleration did not rise with them, the gain curve was an artefact and the car
  is at a rack-force wall the measurement mistook for a gain. That is the one result that
  falsifies this change outright.
- `reversals_per_s` from `turn_tracking.py`, for the weave.

### What is still not compensated

The **deadband**. Refitting the same data with an intercept gives a stiction offset of ~50 counts
at 3–8.5 m/s falling to ~11–26 above 16 m/s, against a `friction` term of 0.0819 normalised =
33.5 counts that is itself speed-independent. It is second order next to a 2.9× gain error and it
points the same way. It is deliberately **not** part of this change — one variable per drive — and
it is the obvious follow-on once this one has been driven.

---

## Test 5 — the low-speed proportional gain, with Test 4

### Why this exists, and why it is the half that actually reaches the intersection turns

Test 4 corrects the feedforward, and the offline replay prices it at +8.6-14.4% delivered torque at
13-29 mph but only +1.1-5.5% below 13 mph. That is not the fix failing; it is the fix hitting a
different wall. Measured on the 409-count archive, hands-off turn frames below 13 mph:

| | |
|---|---|
| delivered counts | p10 135, **p50 353**, p90 **409**, mean 311 |
| frames at >=405 counts | **29%** |
| frames requesting max | **71%** |
| of the frames that asked max and got <405 | mean shortfall 100 counts: **83 to the rate limiter still climbing**, 15 to the driver clamp |
| how long the request holds at max | median **0.34 s**; 53% of bursts under 0.5 s; only **13%** last the 1.36 s the 3 counts/frame ramp needs to reach 409 |

So the ceiling is not the constraint -- the car demonstrably reaches 409 -- and the driver is not
the constraint. **The command does not stay still long enough for the rate limiter to arrive**, and
the delivered torque is a saw-tooth averaging 311 counts under a ceiling it touches.

The command will not hold because the loop gain down there is roughly nine times the highway's:

| v (m/s) | 3.5 | 4.5 | 5.5 | 7.0 | 9.0 | 11.5 | 14.5 | 19+ |
|---|---|---|---|---|---|---|---|---|
| KP vs highway | **25.4x** | 16.1x | 10.3x | 6.7x | 4.3x | 3.0x | 2.2x | 1.0x |
| plant gain deficit | 2.8x | 1.9x | 1.6x | 1.5x | 1.4x | 1.3x | 1.1x | 1.0x |
| over-compensation | **9x** | 8x | 6x | 4.5x | 3x | 2.3x | 2x | 1x |

`KP_INTERP` rises 25x from highway to 3.5 m/s while the plant only weakens 2.8x. The error it acts
on is neither large nor noisy -- median 0.17 m/s^2 at 1-4 m/s, moving 0.023 m/s^2 per frame -- but
at that gain a routine error asks for 576 counts, so the command slams to the rail, falls off it,
and slams back.

The cap is the largest KP for which the P term **alone** cannot exceed full scale at that band's
measured p90 tracking error. It still lets P ask for everything the actuator has at the worst error
actually observed; it stops it doing so at an ordinary one.

| v (m/s) | 2-3 | 3-4 | 4-5 | 5-6 | 6-8 | 8-10 | 10-13 | 13-16 |
|---|---|---|---|---|---|---|---|---|
| error p90 (m/s^2) | 0.316 | 0.331 | 0.537 | 0.582 | 0.634 | 0.548 | 0.346 | 0.379 |
| KP stock | 38.9 | 24.1 | 14.6 | 10.4 | 7.2 | 4.3 | 3.1 | 2.2 |
| KP cap | 10.3 | 9.6 | 5.9 | 4.9 | 5.0 | 5.8 | 9.1 | 7.5 |
| binds? | 3.8x | 2.5x | 2.5x | 2.1x | 1.4x | **no** | **no** | **no** |

It stops binding around 8 m/s, so **every speed at or above 20 mph keeps the stock gain
untouched** -- the whole highway and most of the 15-35 mph band.

### Read this before driving: what is different about this test

**This is a closed-loop change and no offline replay can validate it.** Test 4's feedforward
correction could be priced against recorded drives because it does not move the trajectory. A gain
change does: a calmer command changes where the car goes, which changes the error, which changes
the command. Replaying it against a recorded trajectory answers nothing. The only honest evidence
available before driving is that the open-loop command swing roughly halves (27 -> 13 counts/frame
at 4-13 mph, 7 -> 4 at 1-4 m/s). **Direction, not magnitude.** Everything else is the road test.

**It is only safe together with Test 4.** Cutting the proportional gain is affordable because the
corrected feedforward now carries the demand. Cutting it *without* that is a pure authority
reduction and the car will steer **less** than it does today. `.elantra/guards.py` refuses the
combination `LOW_SPEED_KP_BLEND > 0` with `LOW_SPEED_FF_BLEND == 0` for exactly this reason, and
that refusal is a negative-tested guard, not a comment.

**Below 2.5 m/s there is no measurement.** The archive holds no hands-off turn frames under 2 m/s,
so `np.interp` clamping the cap at 0.26 is an extrapolation. It is bounded on purpose -- a
constant-loop-gain target would have implied a 25x cut there instead of 3.8x -- but it is still
unmeasured, and the empty-lot stage exists to cover it.

### Drive it

Both blends together, and never the KP one alone:

1. **`LOW_SPEED_FF_BLEND = 0.5`, `LOW_SPEED_KP_BLEND = 0.0`** -- this is Test 4, unchanged. Drive
   it first and separately so the feedforward has its own verdict.
2. **`0.5` / `0.5`** -- empty lot first. Full-lock manoeuvres at walking pace, both directions.
   The wheel should feel *calmer*, not slower. Then walking pace, then intersections.
3. **`1.0` / `1.0`** -- the full measured correction. This is the configuration the numbers above
   describe.

What should change: the car should begin turns earlier and hold the line through them below 20 mph,
and the steering should feel less busy. What must not change at all: anything above 20 mph.

### Stop if

Additional to Test 1's and Test 4's:

- **The car turns in more slowly or runs wider than before.** This is the specific failure mode of
  cutting the gain, and it means the feedforward is not carrying what the P term used to. Back
  `LOW_SPEED_KP_BLEND` to 0 first, not both.
- **A slow weave or a lazy, drifting correction on a constant-radius low-speed curve.** Reduced gain
  buying sluggishness rather than calm.
- **Anything at all above 20 mph.** The cap does not bind there; a difference means the
  implementation is wrong, not the tuning.

### Roll it back

`LOW_SPEED_KP_BLEND = 0.0` in
`openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py`. That returns the stock
`KP_INTERP` table exactly -- the multiplier becomes 1.0 at every speed and `scaled_kp_interp`
reproduces the input list. Roll this back *before* the feedforward, never after: the feedforward
alone is safe, the gain cut alone is not.

### What to measure afterwards

The three numbers this change is aimed at, from the new drive's logs, hands-off turn frames below
13 mph:

- **how long the request holds at max** -- median 0.34 s today; if it does not lengthen, the chatter
  was not the loop gain and this diagnosis is wrong;
- **fraction of frames delivering >=405 counts** -- 29% today;
- **mean delivered counts** -- 311 today, against a ceiling of 409 that the car already touches.

If the hold time lengthens and the delivered mean rises toward 409 while achieved lateral
acceleration rises with it, the mechanism is confirmed. If the hold time lengthens but the car still
runs wide, then the car really is authority-limited at ~1.9 m/s^2 below 8 m/s and no steering-side
change will fix intersection turns -- which is the honest end of this line of work.

## After any drive — the measurement

A **systemd timer** (`elantra-lateral-watch.timer`, every 30 min, offroad only) scans each new
route and writes a per-route report. It has to be a systemd timer and not a crontab entry:
`/var` is a tmpfs on this device, so a crontab is wiped on every boot.

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
carrotpilot ships 409 as its default for all HKG, with no measurement recorded for it — and it
has no 2024-25 CN7 platform at all. Its CN7 is `HYUNDAI_ELANTRA_2021` ("Elantra 2021-23"),
pre-facelift, so the fleet behind that number is this generation but not this car. On this car
the number was an extrapolation across a facelift, and has since been measured directly.

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
