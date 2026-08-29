# Road test — CN7 Elantra lateral work

**Governing rule:** steering is commanded only through Hyundai's own LKAS11
`CR_Lkas_StrToqReq`. The MDPS remains the final arbiter — its boost curve, its fault logic, its
driver-override arbitration. We are adjusting a request it is free to refuse.
<https://blog.comma.ai/safer-control-of-steering/>

---

## Read this first — what is actually on the car

An earlier revision of this document said the torque schedule was "not on the car yet" and told
you to test NNLC alone. **That was wrong**, and its pre-flight check agreed with it for a reason
worth understanding, because the same trap will catch you again.

State as of 2026-08-29, read off the device and verified after a reboot:

| | |
|---|---|
| `/data/openpilot` | `elantra-lateral`, working tree clean |
| `opendbc_repo` | pinned, matching |
| panda firmware | rebuilt from the installed source, byte-identical, signature verified on the live board |
| `NeuralNetworkLateralControl` | **0 — deliberately off, so the schedule is the only live change** |
| `LateralJerkTorqueController` | 0 |
| lateral tune | `latAccelFactor = 3.169`, `friction = 0.0819` |
| `DisableUpdates` | 1 — nothing will move the branch under you mid-test |

**The torque schedule is live and will arm on your next ignition.** Test it on its own. NNLC is
a separate change with its own drive, below.

### Pre-flight — and why the obvious check lies

`CarParamsPersistent` is written **at the start of a drive**. Read it while parked and you get
the *previous* drive's value, which is how the old pre-flight reported `DYNAMIC_LIMITS: False`
on a car that was fully flashed and one ignition away from commanding 409.

Compute it from the installed source instead. This does not depend on drive state:

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo /usr/local/venv/bin/python3 -c "
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, CarControllerParams
CP = CarInterface.get_params(CAR.HYUNDAI_ELANTRA_2024, {0:{0x391:8},1:{},2:{}}, [], True, False, False)
sp = CP.safetyConfigs[-1].safetyParam
print(\"safetyParam\", sp, \"| DYNAMIC_LIMITS:\", bool(sp & 1024))
print(\"schedule    \", CarControllerParams(CP).STEER_MAX_LOOKUP)"'
```

Expect `safetyParam 1036 | DYNAMIC_LIMITS: True` and `[[8.0, 16.0], [409, 384]]`.

Once the drive has started, `CarParamsPersistent` becomes trustworthy and should agree.

---

## Test 1 — the low-speed torque schedule

### What it does

Below 8 m/s the LKAS11 torque ceiling goes 384 → **409 counts**, ramping back to 384 by 16 m/s.
Above 16 m/s the command is bit-for-bit unchanged.

409 against 384 is **+6.5%**. This platform's DBC carries `CR_Lkas_StrToqReq` as raw counts —
`16|11@1+ (1.0,-1024.0)` — so the familiar "3.20 Nm vs 3.00 Nm" is an inference from the Nm
scaling in `hyundai_2015_ccan.dbc`, a file this car does not use. Treat the Nm figure as
indicative, not measured. The wire maximum is 1023 counts; nothing here goes near it.

### What the measurement actually says

Re-derived 2026-08-29 over 62 one-minute segments sampled evenly across the 496 then on the
device — 225,275 frames with lateral control active, command read from
`carOutput.actuatorsOutput.torqueOutputCan`:

| speed | frames | at the 384 ceiling |
|---|---|---|
| 0–3 m/s | 13,487 | **2.21%** |
| 3–7 m/s | 32,617 | **4.56%** |
| 7–10 m/s | 26,869 | 0.64% |
| 10–14 m/s | 36,682 | 0.25% |
| 14–18 m/s | 29,797 | 0.09% |
| 18+ m/s | 85,823 | 0.02% |

An earlier 40-segment sample put the 3–7 band at 9.7% and reported exactly zero outside
3–14 m/s. Neither survived the full population. The ceiling does bind at low speed, and it binds
**about half as often as first claimed**, with small but non-zero tails.

Note the shape: **0–3 m/s is the second-most-pinned band**, and the schedule is flat at its most
permissive value from 0 to 8 m/s. The extra authority is largest where the evidence for it is
weakest. Whether to raise the lower breakpoint is the open question this data leaves — which is
why step 1 of the drive below is a parking-lot test.

### Two costs, stated plainly

- **Driver override below 8 m/s.** The ceiling anchors both ends of the override envelope, so
  the car holds 25 counts more against you before winding down. Measured against the compiled
  safety, the driver torque that drives the command to zero moves from about −243 to about
  −255 counts: **+5.2%**. This is the path that has to work when you fight the wheel — test it
  deliberately.
- **Above 16 m/s the command is unchanged, but panda's acceptance threshold is 385, not 384.**
  Earlier text here claimed that single count was upstream slack "present on the stock limits
  too". It is not: the `+1` and the −1 m/s speed fudge both live inside
  `if (limits.dynamic_max_torque)` in `lateral.h`, so a stock Hyundai gets exactly 384. Nothing
  sends the extra count — openpilot commands at most 384 up there — but the claim was wrong and
  the difference is real. At exactly 16.0 m/s the speed fudge puts the threshold at **388**.

### Drive it

Empty lot first. Hands on the wheel throughout. The regime that matters is narrow: **tight turns
at 3–8 m/s**.

1. **Full-lock turns at walking pace.** This is the band the measurement says was believed clean
   and is not, and the MDPS boost curve is steepest here, so 409 does the most work. Most likely
   place for a surprise.
2. **Deliberately override mid-turn at ~5 m/s.** It must wind down the way it does today, from a
   slightly higher starting point. If overriding feels notably heavier, stop.
3. Quiet streets, real intersections at 5–8 m/s.
4. Highway. **Nothing should feel different at all.**

### Stop if

- Any new EPS fault, or "Steering Assist Temporarily Unavailable". This is the one the raised
  ceiling could plausibly cause, and the reason `CF_Mdps_ToiFlt` is logged per drive.
- Any torque step you can feel crossing 8 or 16 m/s.
- Any oscillation or hunting in a turn.
- **Anything at all different above 36 mph.** The command is bit-identical up there; a
  difference means the schedule is not doing what was measured. Stop everything.
- Overriding is harder than the +5% above would explain.

### Roll it back

```bash
ssh comma@192.168.12.238 'cd /data/openpilot && git checkout rdna2 && git submodule update --init --recursive opendbc_repo && sudo reboot'
```

This round trip has been executed and verified: the panda reflashes to the rollback firmware and
back, signature-checked both ways.

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
4. **`FACTORY_ENVELOPE_SEEN.txt`** in `/data/elantra-lateral/`. If it ever appears, the factory
   camera actuated during a drive and we finally have the number nobody has measured. It may
   never appear.

---

## The honest limit of this work

384 is comma's default for most HKG cars, not a CN7 measurement. `values.py` states the rule
outright: *"find the maximum value that the stock LKAS will request."* Nobody has done that for
this platform — not comma, not carrotpilot, which ships a flat 409 with no commit message, no
comment and no measurement anywhere in its history.

Your own logs cannot answer it. The factory camera requested zero torque in every frame
screened, because it never actuates while openpilot is engaged. The passive collection above
watches for a window opportunistically; the real answer needs a deliberate drive with openpilot
passive and factory LFA active, logging bus 2.

So both 384 and 409 are unverified against the factory envelope. This change moves from one
unverified number to another 6.5% higher, on the basis of your own shortfall measurement and the
fact that a large Korean CN7 fleet runs 409 without EPS faults. That is a reasonable basis for a
bounded, reversible first step. It is not the same thing as knowing the envelope, and no amount
of desk work closes it.
