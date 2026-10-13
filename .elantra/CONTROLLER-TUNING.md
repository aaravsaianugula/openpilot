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
