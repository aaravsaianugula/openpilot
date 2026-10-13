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
