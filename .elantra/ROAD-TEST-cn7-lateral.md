# Road test — CN7 Elantra lateral work

Two changes, deliberately tested one at a time. They are independent, and the first needs no
flash at all.

**Governing rule for both:** steering is commanded only through Hyundai's own LKAS11
`CR_Lkas_StrToqReq`. The MDPS remains the final arbiter — its boost curve, its fault logic,
its driver-override arbitration. We are adjusting a request it is free to refuse.
<https://blog.comma.ai/safer-control-of-steering/>

---

## Test 1 — NNLC (live on the car now, no flash)

### What changed

`NeuralNetworkLateralControl` is on. Nothing else was touched: no torque limit, no safety
param, no flash. `LateralJerkTorqueController` stays off deliberately — sunnypilot force-
disables both if both are set, which is one of the ways this toggle silently does nothing.

The car was running a single linear feedforward, `latAccelFactor = 3.169` and
`friction = 0.0819`, borrowed from the 2021 Elantra via `substitute.toml`. NNLC replaces that
with a neural feedforward trained on real CN7 data (`HYUNDAI_ELANTRA_2021.json`, resolved on
the device and confirmed not to be the MOCK fallback), plus a low-speed curvature term that
only exists on the NNLC path.

Measured difference in the feedforward itself, on the device:

| condition | linear (latAccel / 3.169) | NNLC | change |
|---|---|---|---|
| 4 m/s, 1.6 m/s² | 0.505 | 0.575 | **+14%** |
| 5 m/s, 2.0 m/s² | 0.631 | 0.690 | **+9%** |
| 25 m/s, 1.0 m/s² | 0.316 | 0.232 | **−26%** |

So expect **more** steering effort in slow turns and **less** on the highway. The highway
change is not a side effect to tolerate — it is the same correction: a single latAccelFactor
that is right at 5 m/s is wrong at 25, and vice versa.

### Drive it

Quiet low-speed streets or an empty lot first. Hands on the wheel throughout.

1. Tight turns at roughly 5 m/s (11 mph) — the regime that was falling short. This is where
   you should feel the difference.
2. Normal city driving, a few intersections and turns.
3. Highway. Expect it to feel *slightly* softer, not looser. Lane keeping should still be
   centred and calm.

### Stop if

- Any new EPS fault, or "Steering Assist Temporarily Unavailable".
- Oscillation or hunting in the wheel at any speed.
- The car feels like it is fighting you, or overriding is harder than before.
- Highway lane keeping wanders or feels under-damped.

### Turn it off

Settings → Steering → Neural Network Lateral Control, off. It takes effect on the next start
of the car software — the param is read once at init, so toggling it mid-drive does nothing.

---

## Test 2 — the low-speed torque schedule (NOT on the car yet)

### Status: built and verified, not flashed

This lives on branch `elantra-lateral` in both repos. The car is still running `rdna2` with
opendbc `69e2e548` and `safetyParam = 12`, which has no schedule. **Getting it onto the car is
a separate, deliberate step.** Do Test 1 first and re-measure before deciding whether this is
even needed.

### What it does

Below 8 m/s the LKAS11 torque ceiling goes 384 → **409 counts**, ramping back to the stock 384
by 16 m/s. Above 16 m/s the command is bit-for-bit unchanged.

409 counts is **3.20 Nm** on the OEM scale against 3.00 Nm today. The wire maximum is 1023 =
7.99 Nm; nothing here goes anywhere near it.

Why a schedule and not the flat 409 that other forks ship: measured on your car, the command
sits at the 384 ceiling for 9.7% of frames between 3 and 7 m/s and **never once above 14 m/s**.
`STEER_MAX` is a gain, not a ceiling, so a flat raise would multiply every highway command too,
for a shortfall that does not exist there.

### Two costs, stated plainly

- **Driver override below 8 m/s.** The ceiling anchors both ends of the override envelope, so
  the car holds **25 counts (0.195 Nm) more** against you before winding down, and a full
  release moves about 5% further. This is the path that has to work when you fight the wheel —
  test it deliberately.
- **Above 16 m/s** the command is unchanged, but panda's *acceptance* threshold becomes 385
  rather than 384. That single count is upstream's existing slack, present on the stock limits
  too; nothing actually sends it.

### Before flashing

Read the safety param off the device and confirm it carries the bit:

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c "
from openpilot.common.params import Params
from openpilot.cereal import messaging
from opendbc.car import structs as car
cp = messaging.log_from_bytes(Params().get(\"CarParamsPersistent\"), car.CarParams)
p = cp.safetyConfigs[-1].safetyParam
print(\"safetyParam\", p, \"| DYNAMIC_LIMITS set:\", bool(p & 1024))"'
```

Expect **1036** (12 | 1024). If it still reads 12, the flag never reached panda and the change
is inert — do not go looking for a difference on the road.

### Drive it

Same order as Test 1, but the regime that matters is narrow: **tight turns at 3–8 m/s**.

1. Empty lot. Full-lock-ish turns at walking pace, hands on the wheel.
2. Deliberately override mid-turn at ~5 m/s. It must wind down the way it does today, just
   from a slightly higher starting point. If overriding feels notably heavier, stop.
3. Quiet streets, real intersections at 5–8 m/s.
4. Highway. **Nothing should feel different at all.** If it does, that is a finding and a
   reason to stop — the schedule is supposed to be inert up there.

### Stop if

- Any new EPS fault or "Steering Assist Temporarily Unavailable" — this is the one the raised
  ceiling could plausibly cause, and the whole reason `CF_Mdps_ToiFlt` is now logged per drive.
- Any torque step you can feel as the car crosses 8 or 16 m/s.
- Any oscillation or hunting in a turn.
- Anything at all that is different above 36 mph.
- Overriding is harder than the 25 counts above would explain.

---

## After either drive — the measurement runs itself

A crontab entry scans every new route offroad and writes a per-route report. You do not need to
start anything.

```bash
ssh comma@192.168.12.238 '/data/elantra-lateral/lateral_watch.sh; \
  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 /data/elantra-lateral/lateral_report.py \
  compare --out /data/elantra-lateral/reports'
```

That lists the configurations it has seen, one tag each. Pass two tags to compare them:

```bash
ssh comma@192.168.12.238 'PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 \
  /data/elantra-lateral/lateral_report.py compare <before-tag> <after-tag> \
  --out /data/elantra-lateral/reports'
```

It refuses to compare across different configurations, refuses if either side lost a segment,
and computes the pinned fraction twice by independent routes — reporting DISAGREE rather than
averaging if they part company.

What to look at, in order:

1. **`ToiFlt` column.** Should stay at zero. Anything else is the finding.
2. **`%` pinned in the 3–7 m/s band.** Was 9.7%. NNLC may move this either way — a correct
   feedforward can ask for *more*, not less — which is why the ceiling decision comes after.
3. **`demand` vs `deliv`.** Was 2.354 vs 2.048 where pinned. This closing is the actual goal.
4. **`FACTORY_ENVELOPE_SEEN.txt`** in `/data/elantra-lateral/`. If this file ever appears, the
   factory camera actuated during a drive and we finally have the one number nobody has
   measured — what Hyundai's own LKAS asks for on a CN7. It may never appear.

---

## One thing still open, and it is the honest limit of this work

384 is comma's default for most HKG cars, not a CN7 measurement. `values.py` says outright:
*"find the maximum value that the stock LKAS will request."* Nobody has ever done that for this
platform — not comma, not carrotpilot, which ships 409 with no commit message, no comment and
no measurement anywhere in its history.

Your own logs cannot answer it: the factory camera requested zero torque in every frame
screened, because it never actuates while openpilot is engaged. The passive collection above
watches for a window opportunistically, but the real answer needs a deliberate drive with
openpilot passive and factory LFA active, logging bus 2.

Until then, 409 rests on your own shortfall measurement (which implies 424–449, so 409 is
deliberately conservative) plus the fact that a large Korean CN7 fleet runs it without EPS
faults. That is a reasonable basis for a bounded first step. It is not the same thing as
knowing the envelope.
