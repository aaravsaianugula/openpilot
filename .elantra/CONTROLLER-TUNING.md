# Comma 4 / 2024 Elantra SEL Convenience controller experiment

Branch: `experimental/comma4-cn7-controller-tuning`, based on fork master
`13aedc522518fe1b62651d9a8a320e931efc335c`. The exact opendbc pin is
`b45eb78def29582bcce43c68cfb1d55f19224d1a`.

This branch adds desktop analysis, empirical controller experiments and recorded-video
comparisons. It does not change the shipping controller, safety firmware or device.
The existing flat **409** steering ceiling and **3/7 counts per 10 ms** rate limits
are fixed constraints, never optimization variables. The existing real controller/CAN
packing regression reaches 409 with matching normalized feedback at seven speeds.

## Existing recordings and generated artifacts

The current filesystem differs from the older `D:\comma_four\README.md`:

- `F:\routes`: 1,208 complete narrow/wide road camera + rlog segments.
- `D:\Openmayo_Archive\comma4_sidequest\recorded_routes\realdata`: another 225,
  with no duplicate segment names across these two sources.
- Together: 1,433 complete recorded segments, across 120 populated routes.
- At initial inventory `D:\comma_four\routes` was absent. The overnight refresh
  subsequently added 395 missing device rlogs there, keeping flat segment folders;
  selected cameras are transferred separately with integrity receipts. Existing
  desktop footage was preserved.
- Generated datasets, reports and videos belong under
  `D:\comma_four\tuning\controller-tuning-20261012`.

The inventory names missing files and extraction failures. Presence of a camera/log
triple does not guarantee that its compressed log is complete or decodable.

## Reproduce on this desktop

Run from `D:\Coding\sunnypilot-comma4-tuning`, using the existing Windows Python
with NumPy, pycapnp, zstandard, Pillow, pytest and Ruff; FFmpeg/FFprobe must be on PATH.
No GPU, model download, device connection or global package installation is needed.

```powershell
python .elantra/controller_extract.py extract `
  --root F:\routes `
  --root D:\Openmayo_Archive\comma4_sidequest\recorded_routes\realdata `
  --schema-dir D:\comma_four\tuning\lat-tracking\schemas `
  --out D:\comma_four\tuning\controller-tuning-20261012\dataset --workers 6

python .elantra/controller_metrics.py `
  --dataset D:\comma_four\tuning\controller-tuning-20261012\dataset `
  --out D:\comma_four\tuning\controller-tuning-20261012\archive-analysis.json

python .elantra/controller_tune.py `
  --dataset D:\comma_four\tuning\controller-tuning-20261012\dataset `
  --out D:\comma_four\tuning\controller-tuning-20261012\experiment
```

`controller_tune.py` selects a provenance cohort with at least five routes, preferring
recordings that actually reached 409 without exceeding it. `--cohort` selects another
eligible cohort explicitly. It reads whole routes into chronological training (60%),
validation (20%) and test (20%) partitions using recorded wall clock timestamps.
Changing a candidate after seeing test results requires a new held-out evaluation.

For a selected segment listed in the report, pass its original raw directory,
dataset NPZ and generated replay NPZ to `controller_video.py`, with `--out`,
`--start`, `--seconds` and `--fps`. The MP4's JSON receipt records source frame IDs,
actual camera times, projection status and output frame count. Requested viewing FPS
does not imply inference speed or a source camera frame rate.

## Analysis contract

Messages are sorted by monotonic time, then joined causally at 20 Hz. Dynamic
telemetry expires at 200 ms. The analysis may hold the *same valid calibration
event* for up to 500 ms, explicitly marked; it never fills a missing/new/invalid
calibration event or stale speed, yaw or command.

Measured motion uses `carState.vEgo`, `carState.aEgo` and
`liveLocationKalman.angularVelocityCalibrated.value[2]`. `carState.yawRate` and
model-predicted velocity are not ground truth. Steering delivery uses recorded
`carOutput.torqueOutputCan`, independently accompanied by decoded transmitted
LKAS11; MDPS faults are decoded only on bus 0. The normalized request multiplied
by 409 is labelled a reference, since historical builds used other ceilings.

Controller/model settings, car parameters and source commits remain separate cohorts.
Recorded source commits are resolved to their local gitlinks where possible;
unknown identities stay unknown, and a dirty recorded build is not certified by a
clean commit's gitlink. Learner outputs and delays are retained in the dataset.

The longitudinal baseline is acceleration feedforward with stopping behavior, not
a velocity PID. Identification excludes stops and driver inputs. Lateral and
longitudinal response tables use empirical binned medians with route/sample support;
unsupported cells are not extrapolated. Delay/dynamics and candidate gains are
compared on identical window populations, with missing support in the denominator.

## Interpreting the comparison

- **Measured future GT** is integrated recorded ego motion, not a certified safe
  path or lane-position ground truth. Road cam projection assumes locally flat ground.
- **Recorded model path** is its logged prediction on a nonuniform horizon.
- **Baseline simulation** replays recorded commands through the empirical vehicle
  model; it does not rerun the historical controller against altered vehicle state.
- **Tuned simulation** runs an experimental inverse-response feedback controller
  against frozen recorded targets. It does not rerun perception on changed images.
- Longitudinal speed predictions feed lateral dynamics when both axes are simulated;
  otherwise recorded speed is explicitly retained. Unsupported trajectories stop.
- Untrusted plant models are diagnostic only and are labelled. `candidate.json`
  includes identification status and always has `production_enabled: false`.

Model prediction errors and replay advantages are descriptive offline evidence.
They do not prove that a different controller would safely follow the same road,
avoid a lead vehicle or improve a live drive. No road test, deployment, service
restart, panda reflash or vehicle actuation is part of this desktop experiment.

All additions live in `.elantra/`, which the existing weekly overlay carries wholesale.

## Fresh RDF v2 and finite automatic search

See `OVERNIGHT-TUNING.md` for exact RDFMV2 identity, isolated device execution and
current receipts. `rdf_frames.py` preserves consecutive synchronized narrow/wide
images, recorded calibration and causal inputs. `rdf_bench.py` verifies the compiled
artifact hash before loading, runs without publishers or control changes, and saves
native plans/actions plus measured compute and conservative serial availability.
The initial 132 actual camera frames build temporal history and are excluded from
scoring. Model initialization/kernel warmup is reported separately.

`controller_search.py --dataset <dataset> --plant-report <tuning-report.json>
--out <search-directory>` reuses an identified same-cohort plant and immutable route
split. It searches separate acceleration, braking and lateral gains over the stated
finite grid, selecting validation target tracking after support and physical/comfort
constraints. It freezes selection before reading held-out test arrays. Low-speed
stops and untrusted plant axes remain unsupported. A partial axis result cannot
become a complete controller candidate; production stays disabled.

Pass `--rdf-output <rdf_bench.npz>` to `controller_video.py` for fresh RDF evidence.
This view shows logged model versus fresh RDF trajectories with measured future GT
and actual hardware traces. It labels the absence of a validated tune. Exact frame
identity, genuine history and causal output availability are required; paths move
from capture to the displayed ego frame using measured past motion. Capture-origin
model errors and stop distances now use exact EncodeIdx frame matches, never a
publication timestamp substituted for camera capture.

The selected historical research gains `0.4/0.4` have no established mapping to
native controller settings. The current native LongControl constructor fixes Kp
at zero, takes its integral schedule from `CP.longitudinalTuning.kiBP/kiV`, and
uses `CP.stopAccel` as its stopping floor. Its stopping ramp is a source literal,
not an existing CP parameter. Native parameter optimization must run the actual
controller against simulated vehicle feedback and preserve `shouldStop`; a replay
against frozen recorded feedback is only a command diagnostic.

Native feedforward acceleration at the model's future action time is distinct
from a physically aligned acceleration reference at the current simulation tick.
Tune selection requires a causally available raw plan with capture/horizon
alignment, a trusted same-cohort plant, identical supported validation windows
and explicit stopping coverage. Command-target tracking alone cannot select a
physical controller tune. See `native-parameter-contract.json` in the artifact
directory for the current source-bound parameter and runtime-overwrite contract.

`controller_native_search.py` now provides `load_native_runtime`, `evaluate_trial`
and `validation_search` for actual LongControl with simulated vehicle feedback.
Its declared default grid contains twelve Ki/stopAccel combinations, preserving
the recorded Ki breakpoints and baseline vector. It retains every failed trial,
uses one common validation population and constrains acceleration/braking errors,
jerk, saturation, native stopping completion and model-target distance feasibility.
Model targets are predictions, never stopline GT. A missing verified TRAIN-only
uncertainty margin, unsupported low-speed response or uncertain reset prevents
selection. The core refuses held-out windows and always disables production.

This native longitudinal core has actual POSIX API/regression evidence, including
the moving PID/Ki branch and future-GT invariance. It has not selected a native
current-car tune: the recent source/plant and low-speed stopping evidence remain
insufficient. Native lateral parameter optimization and a coupled native
longitudinal/lateral search remain unfinished.

The owner requires longitudinal and lateral tuning together, with independent
left/right turn acceptance. The combined observational selector now freezes turn
direction and entry/steady/exit labels from the same geometric curvature reference
for every candidate. Positive FRD curvature requests a right turn; negative requests
a left turn. Simulated longitudinal speed changes lateral demand and response, but
cannot change those direction/phase populations. Tracking, comfort and saturation
must pass separately on both sides and all three phases. Legacy pooled lateral
scores cannot select a candidate. `controller_turns.py` records the segmentation
thresholds and every missing or unsupported point; these thresholds describe the
research evaluation and do not change controller gains.

The two recent selected clips contain left-turn demand and no qualifying right-turn
demand under this contract. Their new TRAIN-only coverage receipt does not establish
balanced turn coverage. No historical search, fit or held-out evaluation was rerun.
The native lateral runtime now has actual POSIX API proof. The joint steering-rack/
yaw plant and native coupled search still need admission before a candidate can be
selected. `controller_native_lateral_runtime.py` supplies actual v0/jerk handles
with independent immutable Params, native learner/override ordering, simulated
angle/speed/acceleration feedback and separate prior delivered-torque feedback.
The only numeric candidates are finite declared factor/friction values; fixed
override policy is explicit and distinct from the recorded baseline. The source
409/3/7/driver/angle envelope is verified rather than replaced. Model delays remain
fixed. Native reset retains PID/filter history; cold restart remains uncertain.

Explicit acquisition durations now preserve the same 132 genuine history frames
and validity/active/driver/409 gates. A new TRAIN-only 16/24-second census found
right-turn footage, including a 24-second b8--9 clip with entry/steady/exit coverage.
This changes footage acquisition, not controller acceptance. No test data was read;
camera availability, packet construction and fresh inference remain separate gates.

`controller_motion_tape.py` now provides a separate selected-TRAIN raw motion tape.
It retains genuine uint64 publication clocks, rack angle, unsigned Hyundai rate,
full calibrated LLK yaw/roll uncertainty, individual CAN origins/request bits,
static payloads and typed policy snapshots. It never creates a 100Hz replay grid.
All sources must be regular unique files in the exact flat layout, with no
symlink/junction/hardlink alias, before any schema load or decode. Returned
LKAS11 echoes use CAN source128; outgoing sendcan uses bus0. Neither proves
physical actuator delivery. Nineteen tests and independent source review cover
these source and clock contracts.

A bounded actual run decoded only six selected TRAIN logs from b7 and b8, including
their own segment0 static sources, into `motion-recent-train-v1`. Original datasets
and the frozen chronological split remain unchanged; no held-out array was opened.
The raw carState cadence is approximately100Hz and calibrated LLK yaw approximately
20Hz. All six logs have finite roll but nonfinite roll uncertainty, and their
recorded Params omit required `CustomTorqueParams`. The producer preserves both
gaps and rejects LLK quality admission. Dirty source and unknown opendbc identity
also remain untrusted. These tapes do not yet establish a selectable joint plant.

`controller_multirate_lateral.py` implements a separate joint binned-median research
hypothesis: one rack-angle/calibrated-yaw/unsigned-rate transition after five
100Hz command/speed nodes. Feedback is held between those nominal50ms transitions;
this ZOH/Markov command-proxy assumption has not been physically validated. It
never divides a50ms delta into invented100Hz measurements or shifts CAN commands
by model delay. Static/policy anchors, invalid interpositions, request-false
prefixes, exact source grouping and cell bounds fail closed. At least100 samples
and two independent TRAIN routes are required per supported cell. Nineteen tests
passed. Its first real diagnostic retains6787 excluded windows in two distinct
recorded policy groups and selects no cell or candidate.

Full semantic CP/SP hashing now retains Cap'n Proto Data fields, including firmware
bytes. Three focused regressions and an actual17-firmware-field raw CP/SP example
passed; existing non-Data hash semantics and all prior test functions remain
unchanged. This is a typed CP/SP content contract, not a generic type-injective
serialization format. The moving native coupled runner has scoped host review
and still needs actual API evidence; standstill/hold coupling remains a separate gap.

`controller_native_joint.py` now supplies a moving-regime conditional joint search.
It caches actual native longitudinal trials, feeds their pre-transition simulated
speed/acceleration into the native lateral handle, and advances empirical joint
angle/yaw/rate after five valid commands. Fixed raw-plan current-time geometry,
path and left/right populations remain separate from future action feedforward.
Every finite trial retains missing/failed windows and independent direction,
acceleration/braking, comfort, saturation and stopping-feasibility rejections.

Full source/static/policy/model/cell hashes, actual LLK uncertainty, causal learner
snapshot, complete five-tick horizons and native model horizons must pass before
either controller runs. An independently extracted inference-time lateral-delay
witness binds the exact frame/capture/source, packet delay and action inputs; later
core published delay and extension cache are distinct. Thirty-seven host checks
and Ruff passed after independent review fixes. The one actual moving POSIX case
now passes in the isolated local runtime described below. These research checks
do not establish physical plant support or car performance.

This runner keeps complete candidates absent and production disabled. It supports
moving joint diagnostics; inactive lateral response, low-speed transition/hold,
100Hz physical feedback, initialization and frozen adaptation remain unsupported.
Stop/hold work must retain the continuous native handle and require its own
empirical request-false/mixed-pattern response, without inventing zero curvature
or a frozen rack state at standstill.

`controller_multirate_stop_hold.py` adds that separate empirical response hypothesis.
Its24 features retain initial angle/yaw/unsigned rate, physical roll, five speeds,
five unshifted returned-CAN commands and exact effective-active/request patterns.
Inactive zero torque is legitimate; active request-false and unsupported patterns
are refused. Contact, nonreversing hold and100Hz feedback validation remain false.
Full static identity,16 typed Params, causal anchors and every consumed raw row
must pass before support is counted. Thirty-five synthetic checks and Ruff passed.
Independent review found a phased prefix predecessor outside the original quality
span; first-red regressions and the corrected as-of quality checks are preserved.
No actual hold fit has run. Synthetic native stop/hold execution is recorded below.

The committed location publisher at both recorded base `e748a4d` and current base
`13aedc5` explicitly writes NaNs for calibrated orientation standard deviation.
Finite roll remains present, but its calibrated uncertainty is unknown by publisher
design. This is not evidence of damaged logs, nor proof of the dirty runtime source.
No uncalibrated covariance or zero uncertainty is substituted. The source receipt
`validation-llk-roll-uncertainty-source-v1.json` preserves both exact Git blob hashes.

`controller_native_stop_hold.py` extends conditional native joint diagnostics through
one approach, stop request and hold, using the same lateral handle and cached
longitudinal simulated feedback. Completed macros dispatch to the moving or separate
hold response according to actual effective-active/request patterns. Only successful
completed macros count toward hold duration and turn support; attempted commands
and partial failures remain visible separately. Undefined standstill geometry stays
unknown, with no re-entry after model-speed jitter. A finite mixed moving/stopping
search retains identical window IDs, independent turn/longitudinal constraints,
stop-target feasibility and hold-motion regressions. Thirty-nine host cases and Ruff
passed; one actual synthetic native case also passed. Physical contact/feedback
validation remains pending.
No complete candidate or production path is enabled.

A separate hash-bound raw carState witness preserves original uint64 publication
clocks, Event.valid, canValid, standstill and measured speed. The latest as-of
publication must match the existing tape clock exactly and be at most15ms old;
every consumed interposed row
must be valid. Missing evidence and hardware standstill above0.3m/s refuse before
either native controller runs. These measured states check source eligibility only;
they never replace simulated feedback. Simulated standstill remains an explicitly
unvalidated assumption and a hard selection rejection.
Nanosecond conversion uses the existing producers' exact integer-times1e-9
expression; division can differ by one floating ULP and falsely break source-clock
equality. Producer-clock and as-of boundary regressions preserve exact binding.

The moving joint and continuous stop/hold API cases now each pass once on an
isolated x86 WSL runtime with the exact 13a/b45/msgq source pins. The moving case
executes two native trials of 100 ticks with distinct synthetic initial speeds and
different resulting rack/yaw trajectories. The stop case executes 200 ticks and 40
successful macros through one actual V0 handle, native PID-to-stopping transition,
inactive zero-CAN/request-false behavior and a supported synthetic hold. Both
retain the 409 ceiling and native rate bounds, with no complete or production candidate.

The runtime verifies 44 native files/assets and both built ELF libraries. A missing
generated-header include, pytest temporary-file capture failure and missing locked
zstandard dependency were diagnosed separately; original failures and successful
phase outputs remain preserved. No controller/test assertions were weakened.
Final source/binary/snapshot checks, exact commands/exits and the two non-skipped
JUnit cases live under `native-posix-api-v1/phase3-api-v3` in the tuning artifact tree.
These synthetic CPU checks do not run RDF inference, identify a real plant or
validate 100Hz feedback, contact, frozen vision/adaptation or road performance.

The moving joint scorer now distinguishes attempted commands from successfully
supported complete macros. Failed/partial terminal macros contribute no turn
metrics, and failed windows cannot contribute turn or longitudinal scores to the
fixed common population. Cached native exceptions remain named rejected trials
without aborting the search. Final 41 host checks passed; two affected shared
stop/hold summary cases passed. The affected actual moving native case passed
once against the macro fix. The later final delta is one summary-only filter,
verified separately by regression and exact source/AST comparison.

`controller_motion_longitudinal.py` derives additive observations from retained
full payloads: requested CarControl acceleration, independently published
CarOutput acceleration report and actual CarState.canValid. It preserves raw
uint64 clocks, invalid/duplicate rows and original quality. Missing actuator
pointers remain unknown; real scalar zero remains zero. Twenty-five real-capnp
fixture cases passed. A single exclusive six-TRAIN-tape derivation completed in
4.676 seconds: 35,451 requested and 35,437 reported samples, with 42 and 30 invalid
rows respectively retained. All original hashes remained unchanged; the NPZ
allowlist recorded zero reserved-array opens. Neither channel proves delivered
force or sensor acquisition timing, and source eligibility remains false.

The current native joint runner is still explicitly one-way: its independent
longitudinal trajectory supplies speed to lateral control. Genuine two-way
response requires one empirical next-state update for speed, acceleration,
rack angle, calibrated yaw and unsigned vendor rate after each supported macro.
Both native controllers must consume that simulated state; long-only trajectory
caching cannot survive lateral parameter changes. Candidate post-controller
acceleration must come from pinned native processing or a validated causal
request-to-report model, never recorded CarOutput or assumed request equality.
No new fit, inference, physical acceptance or complete candidate follows from
these additive columns. Exact scoped evidence is in
`validation-joint-correctness-motion-view-checkpoint.json` under the tuning tree.

`controller_multirate_joint.py` adds a pure five-state response hypothesis for
speed, acceleration, rack angle, calibrated yaw and unsigned vendor rate. Its
21 features include physical LLK roll and five independent requested-acceleration,
published-acceleration-report and CAN-command nodes. One supported nominal50ms
macro advances all five states together. Actual recorded48â€“52ms transitions are
retained; no independent long trajectory or measured future state supplies feedback.
Original source/policy/lineage, immutable TRAIN split, consumed-prefix quality and
at least100 samples/two independent routes per cell are required. Unknown original
pins cannot be enriched into known identities. Unsupported predictions, including
negative speed/rate and entry into low speed, are rejected without extrapolation.
Thirty-seven host tests and Ruff passed. No real-data fit ran for this component;
100Hz ZOH, speed/acceleration consistency and median joint-state coherence remain
unvalidated, so selection and production are disabled.

`controller_native_longitudinal_output.py` isolates the pinned Hyundai native
request-to-software-report processing. It preserves publication of the previous
held report before optional apply, native update on every fifth actual apply,
skipped-apply holds and a cold independent handle per trial. Full native state,
configuration, source/bootstrap paths and adapter-owned phase/output are fenced.
Reported acceleration remains distinct from delivered force. The first actual
API case exposed an integer-versus-name enum mismatch in cold initialization;
the failure and post-run preservation proof remain saved. Strict schema-bound
integer/name/reader handling fixes the mismatch without accepting bool, unknown
ordinals or impostors. The corrected host suite passed53 cases with one opt-in
POSIX skip. The corrected actual case passed once in5.03s on the pinned isolated
x86 POSIX runtime. It exercises native apply/hold/flags/stopping with synthetic
inputs; it establishes API behavior, not car performance or a supported plant.

The two-way native runner is a separate unfinished integration. Both native
controllers and native report processing must consume the same simulated state.
Card consumes CarControl before publishing CarOutput/CarState and later applies
that consumed snapshot; a newer command arriving before apply is not substituted.
Steering hardware limits belong to apply, independently of native control compute.
The native lateral runtime now exposes separate `compute` and `apply` operations;
compute advances PID/learner/curvature/lag state, while only a successful apply
advances transmitted counts, angle counter and its separate chronological clock.
The legacy same-clock `step` wrapper retains its fields and behavior. Twenty-three
host cases passed with two opt-in skips; Ruff passed after a literal-only test
helper correction. One new actual pinned native split/queued/skipped-apply and
legacy-equivalence case passed in4.19s. Original native sources, Params, libraries
and snapshots were verified unchanged. These synthetic API checks do not identify
physical response or establish a correct recorded publication schedule.
No queued speculative CAN count, recorded output seed or assumed zero latency
can establish candidate feedback. Physical targets and left/right phase populations
must stay candidate-independent. No accepted full tune follows from these modules.

The shared runner `controller_native_two_way.py` now connects actual native
LongControl, lateral compute/apply and Hyundai acceleration-report processing.
Both controllers consume one simulated speed/acceleration/rack/yaw/rate state;
steering changes can therefore affect subsequent longitudinal feedback. The
empirical macro commits only at its50ms boundary, while candidate CS/CO/CC
publication queues preserve consumed snapshots, skipped applies and source clocks.
Fixed physical targets and left/right entry/steady/exit populations are shared
across the finite search. Missing or partial macros contribute no supported score.

Thirty host cases passed. Independent source/clock/bounds review closed concrete
publication, calibration, prefix, censor and state-boundary defects with first-red
regressions. One new actual three-adapter composition case passed on pinned x86
POSIX in21.10s, with source/assets/dependencies/libraries and all26 research files
unchanged before/after. It runs A/B/A trials, observes native CAN and all five state
channels change, and verifies that perturbed unused future measured feedback does
not change the simulated traces. The empirical cells in this API case are arbitrary
fixtures, not fitted vehicle response. Original failed preparation/proof attempts
remain preserved; no native execution was repeated for those host-only corrections.

No real five-state fit, accepted candidate or physical gain follows from this
checkpoint. Unknown recorded source/policy/roll uncertainty, cold prefix,100Hz ZOH,
joint median coherence, actuator-history sufficiency and low-speed/contact evidence
still prevent selection. Production remains false. Exact evidence is in
`validation-native-three-adapter-checkpoint.json` under the tuning tree.

`controller_multirate_joint_stop_hold.py` adds a separate pure five-state
low-speed/hold hypothesis. Its31 features preserve initial state, roll, published
requested acceleration, independent held reports, CAN counts and exact consumed
effective-active/request patterns. Initial and next hardware standstill are
independent categorical observations. Conflicting next categories are unsupported;
there is no speed-derived default, contact clamp, rack freeze or invented restart.
Explicit compute/consume/apply/echo witnesses must match source, original/view
hashes and latest causal CC/CS snapshots, including invalid interpositions. These
witnesses are caller-authenticated observations, not proof of delivered actuation.

Thirty-five synthetic host cases/Ruff and bounded independent clock/standstill/
admission review passed. Real-data fit and native recurrence integration remain
unperformed. The existing older stop runner still caches its longitudinal
trajectory; it cannot supply genuine two-way stopping. This new predictor must be
connected to the shared event recurrence without that cache before a native joint
stopping algorithm can be claimed. Source/contact/ZOH/history/uncertainty gates
continue to disable candidate selection and production.

The new24s right-turn RDF replay and video are complete:480 synchronized pairs,
132 genuine history frames and348 timing-admissible later outputs. Together the
six completed clips contain2,800 pairs/2,008 timely outputs. The right video has240
calibrated flat-ground display frames and173 fresh RDF overlays, plus measured/
command hardware traces. Source/model/runtime/reset/transfer/phase-storage proofs
and actual still/sequence inspection are saved in
`validation-right-rdf-video-checkpoint.json`. It freezes recorded images and motion;
no optimized current tune or physical improvement is shown.

### Shared native approach, stopping and hold

The 31-feature low-speed model now feeds the same native event recurrence as
LongControl, lateral compute/apply and Hyundai request/report processing. One
continuous handle per controller survives approach, stop and hold. Hardware
standstill is an independently predicted category; inactive zero torque and
request faults remain distinct. Mixed search windows require identical full
CP/SP and typed policy identities before factories. No scalar LongCache is used.

The native stop origin now uses the actual compute clock and consumed CS
feedback. Request distance integrates the covered simulated state under the
declared ZOH assumption; feedback speed, simulated physical speed and later
node time are separate. Stop/hold duration uses conservative supported nodes.
Model-target feasibility remains distinct from stopline ground truth.

Fifty-five host checks passed before the final timing amendment; 15 affected
checks passed on the corrected source, with scoped Ruff and independent review.
One corrected actual three-adapter stop/hold case passed in 10.45 s on pinned x86
POSIX. It checks 200 ticks and 40 macros per trial, native PID-to-stopping/ramp,
category publication, the 409 ceiling, signed 3-up/7-down bounds and
future-unused-GT invariance.
Its arbitrary response cells and route/support labels are synthetic fixtures.
The first run's artificial zero-to-post-warmup CAN assertion failure is retained;
only that assertion changed, with genuine prior published CAN now checked.
No gains, native source or response fixtures were retuned.

This path starts within the declared 3 m/s domain. A source-predicted category
transition from higher-speed approach is still unimplemented. Real source/policy,
roll uncertainty, contact/hold, ZOH and joint response coherence remain unvalidated.
No current full tune is selected; production is false. Exact evidence is in
`validation-native-stop-three-adapter-checkpoint.json` under the tuning tree.

### Categorical approach plant

The pure predictor now has named `fit_approach` and `predict_approach_macro`
entrypoints with a distinct model kind. One categorical plant can span observed
approach, crossing 3 m/s and hold cells. Its domain remains the retained TRAIN
global and local cell bounds. Initial and next hardware standstill remain
independent observations; missing or conflicting categories are unsupported.
The existing low-speed entrypoints keep their signatures, model kind and 3 m/s
refusal. No automatic regime switch or fallback was added.

All 52 pure synthetic cases passed, including all 35 unchanged earlier cases;
Ruff and bounded independent review were clean. First-red API failures and
source/test snapshots remain preserved. No actual-data fit, held-out array,
native controller job or device execution occurred. Connecting the new kind to
the shared native recurrence remains the next step. Current full tune selection
and production stay disabled. See `validation-categorical-approach-component-checkpoint.json`.

### Native categorical approach integration

An explicit `approach_stop_model` now selects one admitted 31-feature categorical
plant for the whole native approach, stop and hold recurrence. It spans supported
TRAIN speed cells across 3 m/s without switching models or resetting controllers.
The legacy low-speed option retains its 3 m/s refusal. Both options preserve
independent hardware standstill, continuous native handles, causal publications,
skipped applies, exact 50 ms response commits and fixed 409/native envelopes.

Seventy-six host cases passed, with the two older native opt-in cases skipped;
all earlier cases remain unchanged. Ruff, root review and bounded independent
source/clock review were clean. One new actual LongControl/V0/Hyundai output case
passed in 12.68 s on pinned x86 POSIX (child/outer exits 0, no skip). Its synthetic
3.5 m/s approach crosses 3 m/s and reaches hold using the same plant and handles.
Future-unused measured feedback does not affect the traces. Source/dependency/
snapshot equality, Params isolation and owned-process cleanup passed.

These are synthetic algorithm/API checks. No real joint plant, accepted current
tune or physical driving improvement follows. Model-target feasibility remains
separate from stopline ground truth; source, uncertainty and physical response
gates keep selection disabled. See `validation-native-approach-checkpoint.json`.

The search retains candidate objectives, traces and failures but no longer assigns
diagnostic ranks without predictive plant-validation evidence. Four affected
finite-search regression cases and Ruff passed. An unbound caller success label
cannot enable ranking. This change affects search reporting only; the verified
native control recurrence is unchanged.

### Frozen response validation

`controller_multirate_validation.py` now evaluates a frozen TRAIN plant on
explicit VALIDATION records. It reuses the exact 21/31-feature collectors and
predictors, preserving all source, queue, uncertainty and unsupported-cell gates.
Five state residuals and categorical standstill errors retain route coverage and
measured left/right yaw groups. Unknown labels, unsupported predictions and
quality exclusions stay separate. Conditional error summaries state their
denominators; no overall PASS, rank or tune acceptance follows.

All metadata and physical-source separation checks precede raw arrays. Existing
public fits remain TRAIN-only; TEST is never admitted. Historical training/helper
proof must be supplied and authenticated by a reviewed caller, not invented from
current files. Seventy-eight targeted checks passed; two review fixes then passed
26 affected checks, with Ruff and scoped independent review clean. The fixes
preserve observed category labels on unsupported predictions and reject malformed
array-valued model claims before data access. No real-data validation or fit ran.
See `validation-multirate-predictive-checkpoint.json` for exact proof.

Current publishers do not record the exact consumed CS/CC and compute/apply
correlations required by the categorical plant. Publication clocks and a wrapping
CAN counter cannot reconstruct them uniquely. Future observation traces need
those clocks, prior full CP/SP and typed policy values, plus verified source/build
provenance; calibrated roll uncertainty also remains unresolved. This source
finding does not authorize instrumentation deployment or a road test.

One later strict read-only probe succeeded. Five locally absent candidate rlogs
were identified, without transfer or decoding. Their model/source/policy identity
and route partition remain unknown. Available storage at that instant was below
the native floor plus 2 GiB margin; no allocation or cleanup followed. Fresh
storage guards remain mandatory before any new device allocation.

### Default-off timing observer

The first observation patch records producer input references and actual compute,
publication, apply and skip clocks through separate controlsd/card services. It
preserves original controller calls and the prior-output publication order.
Capture has bounded queues and budgets; ordered CAN headers retain address, bus
and length only. Raw payloads and Params are not exported. Outside-overlay files
are registered in `sync.py`.

Twenty-nine host contracts, including isolated schema roundtrips and enabled/off
hook ordering, passed. Ruff and all 115 overlay guards passed; root and bounded
independent review were clean. This is source/host evidence, not live controller
timing. Device-specific GIL scheduling, backpressure, logging overhead and durable
delivery remain unmeasured; activation and deployment remain gated. Static service entries
add passive logger subscriptions even while capture is off. Policy, source and
predictive eligibility stay false; roll uncertainty and TX/echo correlation stay
unknown or unsupported. See `validation-clock-observer-checkpoint.json`.

### Conditional diagnostic winner selection

`controller_predictive_selection.select_diagnostic` can now choose a strictly
better finite-grid diagnostic row when explicit predeclared quantitative limits
and frozen model, source, helper, validation and search bindings pass. All
incomplete/unsupported rows and original assumptions remain visible. Existing
common-window, left/right entry/steady/exit, acceleration/braking, path, comfort
and stopping constraints are reused. Hash checks are conditional integrity;
independent caller authentication and physical validation remain unperformed.

`controller_native_two_way.attach_predictive_diagnostics` adds these results to
an already frozen report without rerunning controllers, fits or search. The
original report hash binds evidence before attachment, and original rows,
selection and production fields stay unchanged. Forty-nine selector and eight
attachment synthetic cases passed, with Ruff and scoped reviews. See
`validation-predictive-selection-stage1-checkpoint.json` and
`validation-predictive-selection-attachment-checkpoint.json`. No current data
was admitted, no native case repeated and no installable/full tune selected.

### Local observer IPC measurements

One corrected isolated x86 run completed all 14 synthetic IPC cases and 4,200
ticks. The first schema-include abort is preserved; recovery added only the
exact pinned include to a separate package and changed artifact path/hash
literals. Native source/dependencies remained unchanged, Params stayed absent,
and all owned process groups and namespaces were gone after exact cleanup.

Card slow-reader tick start lateness peaked at 18.655 ms, with two samples at
least 10 ms late; its loop duration peaked at 3.322 ms. Controls pressure
recorded 2,700 observer drops and 2,682 timed sequence missing references,
without attributing them to transport or ring loss. Both actual worker failures
and oversize loss flags were observed. Per-case quantiles, denominators and
raw samples are retained in `validation-clock-observer-ipc-checkpoint.json`
and the bound compact-metrics artifact. This short desktop experiment is not
a comma timing, real-time, durable-logging or deployment acceptance result.
The observer remains default-off and has not been deployed.

### Startup CP/SP byte references

The default-off observer now brackets the existing controls CP/SP reads and
parses, and the card CP/SP serializations, independently. It retains at most two
original immutable buffers of 64 KiB each for worker hashing. Only fixed tags,
clocks, lengths, outcomes and 32-byte digests are exported; raw CP/SP, VIN and
firmware remain private. Disabled branches preserve the original reads, calls
and exceptions, and the extension hook is registered in the overlay.

Twenty-nine prior observer contracts remain unchanged. Twenty-nine new startup cases
cover hashing and release, privacy, bounded failure paths, actual hook fragments
and clocks. Review caught a tied/regressing begin clock that could later appear
valid; the first failing cases and its narrow fix are retained. Ruff and all 115
Elantra guards passed. Exact checks and source bindings are in
`validation-startup-static-reference-checkpoint.json`.

A byte digest identifies that startup buffer only. It does not identify current
mutable CP, an atomic CP/SP pair, runtime source, complete torque policy or a
qualified physical plant. All source/policy/predictive eligibility flags stay
false. The supplement has not been deployed or activated.

One isolated x86 run completed two new worker/wire cases and 600 timed loops.
All eight startup references arrived with the required envelope and clock fields,
the digests matched independent checks, and both producers released their 560-byte
synthetic buffer jobs. No drops or errors were recorded. Sources stayed unchanged,
Params stayed absent and owned groups/namespaces were cleaned. Every digest was
ready before the first scheduled tick; hashing contention remains unmeasured. These tiny
400/160-byte fixtures and short desktop measurements do not establish comma
timing, durable logging or driving performance. Exact proof is retained in
`validation-startup-static-ipc-checkpoint.json`; the prior 14 cases were not rerun.

### Controls policy read and write references

The default-off controls observer now records typed results at the existing
13-key read sites, with actual begin/end clocks and distinct consumer IDs.
Separate records identify controller selection and subsequent constructor,
learner, override and delay-cache writes. Raw Params, default origin and external
learner/model/card-thread reads remain unknown; full policy eligibility stays false.

Thirty-eight new host cases and 58 affected existing observer/startup cases passed
in separate processes, with Ruff and all 115 overlay guards. Fixtures exercise
the actual hook fragments and preserve disabled getter order and values. Maximum
constructor/normal schedules are 31/28 records within unchanged 32-record and
32 KiB capture-accounting budgets. The initial combined-process schema-ID abort is preserved.
Source reviews were clean. See `validation-policy-reads-checkpoint.json` for
commands and exact bindings. The first enabled worker/wire attempt is recorded in the later torqued section;
no activation, deployment or tune follows.

### Torqued three-key read references

A separate default-off supplement observes the existing LiveTorqueParamsToggle,
CustomTorqueParams and LiveTorqueParamsRelaxedToggle reads in torqued. It attaches
one bounded header only to the existing main lateralTorqueParameters publication;
it adds no publisher, thread, Params polling or cache instrumentation. Read-return
IDs remain distinct from cached values and the published useParams field. Pending
read evidence retires only after the original single send returns locally; this
is not subscriber delivery or durable acknowledgement.

Twenty new host cases, 58 existing observer/startup cases, scoped Ruff and all 115
overlay guards passed. Root and independent clock/privacy/source reviews were clean.
Separate schema processes measured a 504-byte maximum eight-read envelope and an
8-byte structural increase even when disabled. Original disabled operations and
cache/get_msg ASTs match the preserved baseline. Exact commands and bindings are in
`validation-torqued-policy-checkpoint.json` under the artifact root. Enabled IPC,
clock-domain equivalence, comma overhead and delivery remain unverified. Full policy,
source and predictive eligibility stay false; no activation or deployment occurred.

The controls-policy IPC attempt retained all 9,331 planned received messages and
300 loop samples but exited 1 because its validator expected incorrect service
labels on the four startup references. That failed run remains preserved. The corrected
validator passed the retained evidence in one separate Windows-host offline check:
9,331 received messages, including 4,527 typed policy rows, and 300 retained ticks.
The four expected startup service labels were the only validator change; six
independent retained-header regression cases passed. Reconstructed 400/160-byte
synthetic CP/SP fixtures matched independently known digests before the unchanged
full startup gates. Windows serializer 2.2.4 differs from the measured WSL 2.1.0;
byte equality is not timing equivalence. Original IPC exits remain 1, and no new
IPC/producer run occurred. Exact bindings are in
`validation-clock-observer-policy-ipc-checkpoint.json`.

The retained synthetic worst-case loop p99/max were 2.029/2.173 ms; tick-start
lateness peaked at 1.080 ms, with zero of 300 samples at least 10 ms late. No
per-call or causal overhead claim follows. Final publisher counter assignment
was unavailable after the original exception. The owned process was gone and
its namespace was later observed absent; removal timing/cause is unknown, with
no namespace cleanup by us. Source/input hashes and the original failure are
preserved; deployment and car timing remain unverified.

### Model load, activation and fallback references

A default-off reference supplement now attaches a bounded header to the existing
modelDataV2SP output. It distinguishes the actual attempted and active instances,
load origin and fallback from declared bundle generation. Late background completion
cannot activate a model or relabel earlier output. Model and driving Event/frame
identities are recorded independently; no driving EOF is invented. The original
load, warmup, run and send operations and disabled native ASTs are preserved.

Twenty-eight new host cases and six affected existing schema/startup cases passed,
with Ruff and all 115 overlay guards. Root and scoped independent source reviews
were clean. Separate schema processes measured a 488-byte maximum four-reference
header and an 8-byte default-off structural increase. Exact commands, first-red
failures and source bindings are in `validation-model-asset-reference-checkpoint.json`.

No model, weights, Params, IPC or device runtime executed for this supplement.
It adds no publisher, thread, polling, file probe or compiled-stream hashing.
Successful unpickling is not EOF proof; consumed/full asset digests, authenticated
source, exposure origin and complete policy/predictive eligibility remain unknown
or false. Enabled overhead and delivery are unverified; activation/deployment
remain gated.

### Logger-owned route and boot references

A default-off logger supplement now adds bounded references to existing InitData
and Sentinel records. It preserves the owner-built route/bootlog labels and
distinguishes the cached InitData construction clock from each fresh segment
boundary. Cached CurrentBootlog, declared Git/version strings and dirty state
remain claims; they do not authenticate loaded source or identify a kernel boot.
No extra Params/source/asset reads, hashes, publisher, service or thread were added.

Thirteen host contracts, two affected schema cases, scoped Ruff and all 115 overlay
guards passed. Root and independent clock/privacy/source reviews were clean.
Actual isolated x86 C++ compilation, three syntax units and twelve synthetic
serializer groups passed. The first attempt stopped on a missing development
header; a later fixture-only Text::Builder comparison failed compilation. Both
failures are preserved. The final fixture corrected two reader conversions and
reused the eight successful generated schemas without regeneration. Syntax used
an isolated pinned public Zstandard header, with no installation or library run.
This does not establish target-device SDK equivalence.

The host schema probe measured a 376-byte maximum added envelope and an 8-byte
default-off pointer increase for each InitData/Sentinel record. C++ cases checked
the actual added envelope against the 1 KiB bound. Source/provider/compiler and
reused-schema hashes stayed equal; all owned test processes are gone. Exact
commands, exits and failures are in `validation-logger-route-boot-checkpoint.json`.
No production logger, Params, Hardware, IPC or device execution occurred.
Authenticated source, kernel boot, clock-domain equivalence, complete policy,
predictive eligibility and durable recording remain false or unknown. No
activation/deployment, current plant admission or controller tune follows.

### Default-off model delay read and preparation references

`ELANTRA_MODEL_DELAY_REFERENCE_OBSERVER=1` adds bounded delay references to the
existing modelDataV2SP asset header. A private consumer uses the same already
reserved model token before the original base getter; reserve, load-begin,
getter/cast, refresh cadence, model timer and send order stay intact. Stock
published offsets, tinygrad cached-plus-smoothing offsets, initial longitudinal
delay ownership and later fallback metadata remain separate. Python action offsets
and exact two-element Float32 prepared inputs are recorded separately. Preparation
is not evidence that the model call, graph copy or graph consumption occurred.

The first full schema measured 1120 bytes and failed the unchanged 1024-byte header
limit. Lossless 25-bit status packing retains all nineteen read fields; the full
100-field snapshot, four asset references and four read rows measure exactly
1024 header bytes (1064 including the Event), with no spare space. Both-off bytes
remain identical. Thirty-three new source/host cases passed; after narrow style
changes, three affected cases and eighteen affected prior cases passed separately.
Scoped Ruff, all 115 overlay guards, root review and independent hook/lineage
review passed. Original first-red and lint failures are preserved, including
read-ID exhaustion, unknown constructor snapshot replacement and unseen references.
See `validation-model-delay-reference-checkpoint.json` for exact commands and hashes.

Only the exact entered final SP local send-return retires its staged prefix.
Earlier send failures, None output, replacement and fallback retain unresolved
getter evidence. No delivery or durable ACK is claimed. Current published-delay
references do not by themselves identify the publication used by an earlier
held-cache refresh. Tinygrad's unused published-plus-smoothing offset and the
actual postprocess smoothing branch remain unknown; no counterfactual math or
metadata-based branch inference fills those gaps. Raw presence/default origin,
authenticated source/assets, exposure/clock domains, complete policy/predictive
eligibility and graph consumption remain false or unknown. No model, weights,
Params, IPC, native/device runtime, activation or deployment occurred. Enabled
costs and physical response remain unvalidated; this is source preparation, not
a controller gain or accepted tune.

### Default-off card asynchronous policy observations

The existing controller observer now records the three direct card fields
IsMetric, ExperimentalMode and DynamicExperimentalControl. Constructor reads
remain distinct from the existing asynchronous refresh. Returned getter values,
the conditional CP operand and effective assignments are separate from each
actual argument offered to the cruise helper. Every read-to-use origin remains
unknown; equal values or nearby clocks do not establish an association. A failed
CP access has no returned-operand witness, not proof that no access was attempted.
Nested cruise settings, full mutable CP/SP and complete policy remain unobserved.

One private scalar journal holds at most sixteen immutable rows; card main drains
at most two each tick and remains the existing Observer's sole producer. There
are no extra Params reads, polls, publishers, threads, services, locks or waits.
Original getters, short-circuit reads, arguments, exceptions and disabled native
operations are preserved. A review found a loss/counter interposition that could
leave completion true; its regression and conservative observed-counter rejection
are retained. No atomic configuration or verified runtime handoff is claimed.

The initial 24-case host run had 23 passes and one disabled-AST fixture failure.
After correcting that fixture, it and two style-affected cases passed separately;
one added maximum-wire case also passed. Forty-three affected existing cases,
Ruff and all 115 overlay guards passed. Root and independent source reviews were
clean. Separate schema processes measured a 400-byte maximum new record and an
unused-pointer increase from 224 to 232 bytes. Conservative capture charges,
including both sendcan raw captures, total 31,488 bytes within the unchanged
32,768-byte/32-record loop bounds. Per-record limits still reject oversized input;
these charges are not heap, latency or delivery measurements. Exact commands,
first-reds and source hashes are in validation-card-async-policy-checkpoint.json.

No actual card, Params, model, IPC, native/device runtime, activation or deployment
occurred. CPython handoff/timing, delivery, read-to-use origin, full policy,
authenticated source, transport and calibrated-roll uncertainty remain unverified.
Source/policy/predictive eligibility stay false; no plant or tune is accepted.
