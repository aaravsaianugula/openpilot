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

### 409 is the car's limit, not ours

**A raised ceiling was driven and the EPS refused it.** Two builds went above 409 — a schedule
peaking at 500, then a flat-topped 450 — and **both threw an EPS fault**. The MDPS does not
deliver the extra authority; it drops out. Coming back down to 409 cleared the fault both times.

This is the single most important fact in this document, and it changes how you read the rest:

- **409 is a measured hardware boundary.** Not a policy number, not a compromise, not something
  to revisit if the car feels short of authority at low speed. It has been tested.
- **The remaining low-speed shortfall cannot be fixed by raising this number.** Whatever is
  left to win down there is in the tune, not the ceiling.
- **panda will not protect you here.** It enforces 512 — 103 counts above the boundary — so if
  a build ever commands 450 again, panda passes it and the EPS faults. What holds 409 is
  `CarControllerParams.STEER_MAX` and the guards around it, nothing downstream.

If you see an EPS fault or a steering dropout on this drive, **that is the signature of a
ceiling above 409**, and the first thing to check is what the build actually commands.

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
| flat, what you are testing | 409 | — |
| **what the EPS refuses** | **450, 500** | faulted on this car |
| what panda would let out | 512 | unenforced above 409 |

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

Measured 2026-08-31 from the recorded drives, re-scanned with the corrected EPS fault counter
(the old one counted MDPS fault frames whether or not openpilot was steering, which made any
"rate" built on it meaningless -- three routes in the store carry 1,271 / 938 / 556 fault frames
against ZERO engaged frames).

| route | build | engaged frames | EPS faults **while steering** | max \|counts\| at 18+ m/s | frames at 409, 18+ |
|---|---|---|---|---|---|
| 000000bd | speed schedule | 50,822 | 0 | 337 | 0 |
| 000000c5 | **flat 409** | 401,042 | **1** | **409** | 93 |
| 000000c6 | **flat 409** | 310,382 | **0** | **409** | 206 |

So flat 409 has ~711,000 engaged frames behind it with **one** EPS fault frame in total
(0.00014%), and the ceiling is demonstrably reached at highway speed -- which the schedule build
never did (its max above 18 m/s is 337, below even the old 384). Across every flat-409 route in
the store the totals are ~959,000 engaged frames, with zero EPS faults in every band above
3 m/s.

This is observational, not a controlled A/B: the routes are different drives on different roads.
It is evidence that the ceiling does not provoke the MDPS, not proof that it never will.

### Stop if

- Any new EPS fault, or "Steering Assist Temporarily Unavailable". `CF_Mdps_ToiFlt` is logged per
  drive for exactly this.
- Any oscillation or hunting, **especially on the highway** — that is where the gain increase is
  compensated only above 15 m/s by the learner, and where the ceiling is genuinely reached.
- Overriding is harder than the +5.2% above would explain.
- Anything at all different below 8 m/s. Nothing changed there; a difference means the build is
  not what you think it is.

### Roll it back

Back to the speed-schedule build, which has one clean road test behind it:

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && git checkout bb90e6d94 && git submodule update --init --recursive opendbc_repo && sudo reboot'
```

pandad reflashes on boot whenever the running signature differs from the built binary, so the
panda follows the source automatically. This round trip has been executed and verified in both
directions — see the rollback section of the handover.

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

What to look at, in order:

1. **`ToiFlt` column.** Should stay at zero. Measured zero on `src 0` across all 62 baseline
   segments. Anything else is the finding. (Bit 14 on `src 1` is *not* a fault — that bus
   carries something else on this car.)
2. **`%` pinned in the 3–7 m/s band.** Baseline **4.56%**.
3. **`demand` vs `deliv`** where pinned. Closing this gap is the actual goal.
4. **`FACTORY_ENVELOPE_SEEN.txt`** in `/data/elantra-lateral/`. This was set up when the
   envelope was believed unmeasurable. It has since been measured at **157 counts** from route
   `000000b9--0c339ed202` (see Test 1). The watcher is still worth keeping — one drive is a
   floor, not a maximum — but it is no longer the only source.

---

## The honest limit of this work

The factory envelope is measured — **157 counts** — and it turns out not to be the deciding
number, because Hyundai's lane-centring feature and openpilot's job are not the same task. What
is left is that 384 and 409 both rest on fleet evidence rather than on a derivation, and
carrotpilot ships 409 with no commit message, no comment and no measurement in its history. Fleet
evidence is real evidence; it just should not be reported as something it is not.

What remains genuinely unknown:

- **Sustained saturation at 409 above 8 m/s.** Reaching the ceiling there is now measured (see
  below) but only in short bursts -- a few hundred frames out of ~580k. A long, continuously
  saturated highway curve has still not been driven, and that is the case that remains open.
- **Where the tune settles.** The learner is valid and converged, but it re-converges against
  the new gain only from samples above 15 m/s. Below that the +6.51% stays uncompensated
  indefinitely. The car you drive tomorrow is not the car you drive in a week, and neither is
  wrong.
- **What the MDPS does at 409 in a tight turn.** It is the arbiter and it can refuse; nothing
  desk-side tells you where that line is.
