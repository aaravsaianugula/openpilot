# Overnight continuation: RDF v2 / CN7 controller optimization

Owner instruction, October 12, 2026: continue autonomously overnight; use an
algorithmic search, rather than manually selected gains, for **acceleration,
braking/stopping and lateral control**. Target driving model is **RDF v2**.
Isolated model execution on comma and recorded footage is authorized. Existing
live vehicle deployment/actuation, service-restart and firmware gates remain.

Heartbeat `rdf-v2-elantra-tuning-overnight` is ACTIVE every 30 minutes in this chat.
Continue useful work while running; the heartbeat is recovery/follow-up, not a
reason to wait. Stay quiet on unchanged state. Prepare a factual morning report
for October 13 in America/Los_Angeles; pause the heartbeat at actual completion.

## Workspace and existing evidence

- Worktree `D:\Coding\sunnypilot-comma4-tuning`, branch
  `experimental/comma4-cn7-controller-tuning`.
- Base master `13aedc522518fe1b62651d9a8a320e931efc335c`; exact nested opendbc
  `b45eb78def29582bcce43c68cfb1d55f19224d1a`. Do not reset or modify other checkouts.
- All new source is under `.elantra/`; weekly overlay preserves it wholesale.
- Artifacts `D:\comma_four\tuning\controller-tuning-20261012`.
- Full extraction finished: **1,433 available complete segments, 1,426 decoded,
  119 decoded routes / 120 available routes, seven named failed logs**.
  See `dataset/manifest.json`; exit 1 accurately reports those corrupt/truncated
  sources. Do not rerun the full archive, hide failures, or claim 1,433 decoded.
- Source footage is `F:\routes` and
  `D:\Openmayo_Archive\comma4_sidequest\recorded_routes\realdata`.
  `D:\comma_four\routes` does not exist; the older archive README is stale.
- Dataset schemas are the existing recorded-build snapshot in
  `D:\comma_four\tuning\lat-tracking\schemas`.
- `archive-analysis.json`: corrected exact frame-capture alignment, 28 provenance
  cohorts, 24,924 admitted three-second windows. Earlier publication-origin
  results are preserved in `archive-analysis-publication-baseline.json`.
- `stopping-analysis-capture.json`: corrected exact capture-origin stop errors,
  414 initial model-stop episodes; 36 initially comma-long-active / 378 inactive.
  Only five uninterrupted comma-controlled completed stops. The older
  `stopping-analysis.json` is retained as the publication-origin baseline.
  Initial model stop predictions are not certified stop-line locations.
- Existing comparison `videos/controller-comparison.mp4`: 30 s / 300 frames,
  frame-mapped via EncodeIdx, inspected frame and sequence. It visibly labels
  the untrusted plant as diagnostic. `videos/longitudinal-pilot.mp4` is a
  separate earlier pilot, not RDF v2 inference.

## Exact RDF v2 identity verified from the live official model catalog

- Name `RDF Model V2 (August 11, 2026)`, alias `RDFMV2`.
- Normal catalog `driving_models_v22.json`, not the Chestnut catalog.
- Ref `35703097905a122c9f3ddf0d12889b4873d7e2a2`, generation 12,
  `is20Hz=true`, `is_big=false`, selector version 19.
- Tinygrad commit `e837e367aac9e1a66e689f4f32ce20ca9367df13`.
- Artifact `driving_rdf_model_v2_august_11_2026_tinygrad.pkl`, three chunks;
  whole SHA256 `52fcf48bfb991f327a8982037eb0855d9a63437d78e9f4828d2be54df0f32567`.
- The old flat archive has v2 count manifests but lacks their referenced chunks.
  `rdf23` / `rdf33` aliases and RDF v1 chunk files are different artifacts.
- Verified v2 artifacts are now in `D:\comma_four\models\rdf-v2`: three
  chunks totaling 126,297,124 bytes, plus `rdf-v2-identity.json` containing
  exact public URLs, sizes and hashes. The scoped read-only device transfer
  succeeded and appended `D:\comma_four\sync.log`. Verify the whole hash before executing
  a serialized model. Never decrypt or substitute IQ.Pilot `.iqc` assets.
- Read-only SSH found device at `comma@192.168.12.238` on master `13aedc5225`,
  comma service/UI active, no modeld in process listing, about 8.9 GiB free on
  `/data` (90% full). This is a snapshot, not offroad/thermal authorization.
  Check offroad, temperature, ignition and active jobs immediately before
  isolated benchmarks. No device writes/restarts were performed in that audit.
- Subsequent snapshot: `IsOffroad=1`, battery **Discharging**, USB/DC online 0.
  The owner then explicitly confirmed comma is in their room connected to the
  wall outlet over USB-C and instructed proceeding. Record this discrepancy;
  **USB online status is not a blocker under this owner instruction.** Proceed
  with bounded isolated inference while monitoring battery and relevant hardware
  temperatures; stop on actual low battery, overheating or vehicle engagement.
  Do not modify power, safety, service or vehicle-control state.
- GPU is QCOM **Adreno630v2**. This compiled model has no verified desktop
  CUDA/ORT execution path. Device tinygrad checkout is `f6fc4e3`, while the
  catalog's matching `e837e367` Git object exists. Use an isolated runtime if
  necessary; never change the active shipping tinygrad checkout for a benchmark.
- Current APIs: `ModelState(1344, 760, chestnut=False)`, `_init_combined()` and
  `run(bufs, transforms, inputs)` in `openpilot/sunnypilot/modeld_v2/modeld.py`.
  A `COMBINED_MODEL_PKL` override alone incorrectly loses RDF's 20 Hz identity,
  generation and smoothing. Pass the verified catalog bundle explicitly.
  Catalog smoothing overrides are lateral 0.1 / longitudinal 0.3; these identify
  the source bundle, not manually chosen optimized controller settings.
- Inputs: `img` and `big_img` each `[1,12,128,256]`, desire `[1,33,8]`,
  recurrent features `[1,32,512]`, traffic convention/action timing each `[1,2]`.
  Keep true chronological history; image skip four, five-frame image queue,
  128-feature and 132-desire history. Exclude at least 6.6 s of genuine warmup
  from scoring. Delay changes alter model `action_t` and need corresponding
  inference. Do not substitute repeated frames or future measured motion.
- Archive benchmark scripts are stale: `bench_model.py` uses unsupported
  `usbgpu`/`prepare_only`, and `validate_active_rl.py` assumes absent contracts.
  Adapt to the installed source APIs before running anything.

## Required next work

1. Capture timing correction is implemented and the full reports regenerated.
   Extracted `model_t` is modelV2 **publication** monotonic time; a 50 ms
   publication-age gate does not bound camera/model processing latency.
   `controller_extract.model_capture_times()` now matches exact
   `model_frame_id` to `raw_encode_frame_id` / `raw_encode_t`. No publication
   fallback or ambiguous frame-ID matching is allowed. Metrics interpolate only
   admitted measured label brackets; stop distance starts at the same capture.
   The stopping law subtracts already-travelled capture-to-decision distance.
   Do not certify prior publication-origin endpoint metrics as time-aligned.
2. Source-valid onset gating in stops is implemented: planner-negative onset requires
   valid/fresh planner messages; commanded-negative onset requires valid/fresh
   carOutput. Missing/invalid onsets have explicit coverage, not fake timings.
3. Finish exact RDF v2 runtime/dataflow verification and isolated inference.
   Preserve recurrent/history state, synchronized narrow/wide camera inputs,
   calibration, frame IDs and causal vehicle inputs. Retain continuous outputs
   and latency distributions. Recorded old model outputs are not fresh RDF v2.
4. Implement reproducible automatic search over a stated finite/bounded space,
   with immutable train/validation/test **whole chronological route** splits.
   Parameters, search seed, objective, bounds, supported operating ranges and
   all failed/unsupported candidates must be retained. Optimize validation;
   read the held-out test only after selection. A universal global BEST cannot
   be proved from a finite offline search; report the strongest supported
   candidate within the evaluated space.
5. Treat physical bounds/feasibility and regressions as constraints before
   comfort. Cover acceleration, deceleration onset and sustained stopping,
   speed/lateral tracking, jerk, saturation and driver interactions separately.
   Keep STEER_MAX literal **409**, native 3/7 rates and safety/driver envelopes.
   Do not trade an acceleration improvement for worse stopping or steering.
6. Generate real footage comparisons including stop events, with BEV and
   trajectory overlays, actual commands/response, and clear intervention and
   uncertainty labels. Camera projection is locally flat-ground geometry.
   Frozen recorded images/targets cannot prove closed-loop changed-camera
   behavior or a road-safe outcome. Keep production disabled pending gates.

## Existing implementation and checks

Reuse `controller_extract.py`, `controller_metrics.py`, `controller_stops.py`,
`controller_dynamics.py`, `controller_tune.py`, `controller_video.py` and their
regression tests. Do not build a second extraction stack.

New modules: `rdf_frames.py` builds exact real dual-camera packets with production
NV12 padding and canonical recorded calibration warps. `rdf_bench.py` runs the
verified RDFMV2 QCOM artifact in an isolated monitored process. It records all
outputs, excludes 132 genuine history frames, and separately records conservative
serial output availability. Cold initialization, production kernel warmup and
fresh camera-history warmup are distinct costs; never discard them silently.
`controller_model_overlay.py` validates exact model/frame identity and compensates
paths by measured past motion at the displayed camera timestamp.

`rdf-packets/e9/packet.json` records 240 real paired frames (5043-5282) from
`F:\routes\000000e9--7a28169241--4`, starting about 12 s into the segment.
Maximum pair skew is 18.54 microseconds; each padded stream is 582,942,720 bytes.
All 240 recorded SP turn messages were invalid: neutral desire is admitted only
because fresh valid blinkers are both OFF and recorded lane-change state is OFF,
which proves the canonical controller's neutral branch. Preserve that proof and
invalid-SP flags; do not generalize it to turning or unknown states.

RDF runtime functional probes have produced finite plans/actions and changing
recurrent state on Adreno630v2. Those short synthetic probes are not footage or
sustained inference-rate acceptance. Exact replay receipts live under `rdf-v2/`.

The **actual e9 replay completed**, exit 0: all 240 consecutive real dual-camera
pairs processed, all finite; 132 genuine history frames excluded, 108 scored and
108 within the declared 200 ms serial availability deadline. Scored full compute
median 38.869 ms, p95 42.721 ms, p99 44.854 ms, max 50.145 ms. Production kernel
warmup took 5,034 ms before camera replay and exact temporal-state checks passed;
this cost is reported separately. Battery 65 -> 65 percent, relevant temperature
max 44.6 C, continuously offroad. These 12 seconds do not establish sustained road
20 Hz acceptance. `rdf-e9-12s-transfer.json` preserves the executed adapter and
model/output SHA; a subsequent hardlink-output guard fix did not rerun inference.

Production action timing also includes the RDF catalog's smoothing. The immutable
packet stores base `action_t` .284269/.575; the bench explicitly adds .1/.3 exactly
once, producing effective .384269/.875. The base/effective timing is saved in the
receipt and NPZ. Never omit or double-count smoothing during a delay search.

`videos/rdf-v2-real-replay-e9.mp4` is rendered and inspected: 12 s, 120 distinct
camera frames, all calibrated projections; 53 viewing frames show post-history
fresh RDF, with maximum camera selection age .155 ms. It shows measured future
GT, logged model and fresh RDF with actual torque/acceleration/speed/curvature.
Driver actions remain visible and flagged in this descriptive video. Separate
driver-free quantitative evaluation admits only 15 of 108 one-second windows,
and zero two/three-second windows because of driver interventions. Do not turn
those exclusions into a controller-performance claim.

The second STOP replay **completed**, exit 0, from `0000000c--6c5e8ce437--21`,
start32s/duration24s, 480 fresh pairs, covering an uninterrupted comma-controlled
stop. Its historical source gitlink is unknown and remains separate from the
current replay checkout. All 348 post-history frames are deadline-admissible:
full compute median37.70ms/p9542.18ms/max49.97ms. Preloop warmup changed image queues;
the canonical queue/cache reset restored all 14 temporal arrays before real input,
and the full warmup/verification cost6.35s is recorded. Battery65->65%, max45.3C.
`rdf-v2/rdf-stop-0c-24s.{json,npz,log}` and transfer/executed-source receipts retain
these facts. No production changes or actuation occurred.

`videos/rdf-v2-real-stop-0c.mp4` has 240 calibrated frames and measured GT; 173 show
post-history fresh RDF. Approach, near-stop and a six-frame stopping sequence were
inspected. The BEV now uses explicitly labelled 6-70m forward scales so the last
metres of a stop remain visible. Quantitative capture-aligned comparison on this
single historical route admits 287 overlapping three-second frame windows:
fresh RDF mean endpoint error .480m versus logged model1.106m. These are correlated
prediction labels on one clip, not independent trials or a tuned-controller result.
At standstill raw RDF acceleration may be slightly positive while `shouldStop` is
true. A native controller replay must preserve that stop flag and the LongControl
state machine; acceleration sign alone must never infer a restart.

## New data acquisition for stronger plant identification

`device-recent-inventory.json` found397 device segments/42 routes, of which395
segments/41 routes are missing from both desktop raw archives. Missing rlogs total
3.906GiB;394 of those segments also have both cameras on the device. Latest device
route is `000001d8--f62ae0bcb9` (35 segments). The desktop archive reaches route10a,
so earlier rough references to ee as its latest route were incomplete.

The archive worker is authorized to pull **only** those395 missing rlogs first,
using one persistent documented rsync transport, source/target hashes, partial
staging, atomic no-overwrite completion, PID/progress receipts and sync.log.
Destination is authoritative `D:\comma_four\routes\<flat-segment>\rlog.zst`.
Do not fetch all26GiB of cameras indiscriminately; select footage after analysis.
Check existing worker/process/receipts before any restart to avoid duplicate pulls.

The existing extractor accepts logs without local cameras and explicitly records
missing videos in its manifest. Once the pull is verified, extract only the new
root into `dataset-recent`, with the same schema snapshot and six workers. Do not
rerun or overwrite the original1426-record dataset. A successful log extraction
does not mean footage/RDF readiness. Check recent whole-route IDs against existing
immutable held-out routes before training, and keep actual provenance cohorts.

`controller_search.py` searches the finite grid 0, .025, .05, .1, .2, .4, with
separate acceleration/braking gains, target tracking selection, identical supported
windows and jerk/slew/saturation/regression constraints. Historical cohort
`1af6dc6837be` supports a **partial longitudinal research candidate**: gains .4/.4,
validation balanced target MAE .0448447 -> .0318879 (93/120 common windows), frozen
test .0455741 -> .0337994 (104/120). Both gains are at the upper search boundary;
no convergence or global-best claim is supported. All 216 complete-controller grid
combinations reject because lateral identification is untrusted. Current-model/409
cohorts reject their plant gates. See `search-control/`; production remains false.
Stopping below 3 m/s remains outside this plant search and is not tuned by it.

`enrich_route_metadata()` recovers missing static CP only from the same route's
own recorded configuration with matching commit/dirty state. Keep inheritance
source refs. `cohort_key()` preserves actual active bundle/controller/CP/source
identity while excluding unselected model caches and catalog refresh times.
Unknown pins and dirty builds remain uncertain. `hold_calibration()` explicitly
holds the same valid calibration event for at most 500 ms; all dynamic
telemetry still expires at 200 ms.

Experiments already evaluated separate cohorts:

- `experiment/`, cohort `456c58c31933`, 140 segments / 14 routes, recorded 409.
  Lateral insufficient; longitudinal untrusted due missing validation support.
- `experiment-current-model/`, cohort `6795b0728c7c`, 78 segments / eight routes,
  recorded 409. Lateral insufficient; longitudinal untrusted. These are logged
  historical model outputs, **not** a verified RDF v2 rerun.
- `experiment-control/`, cohort `1af6dc6837be`, 751 segments / 36 routes.
  Lateral diagnostic only: tau optimum at grid bound 0.8 s and validation
  coverage 22%. Longitudinal has a limited supported offline candidate.
  Do not combine this older cohort into an unrelated current-build tune.
- `controller_dynamics` stores state-conditioned median transitions; it never
  invents unsupported gains. Neither its sparse support nor the steady tables
  currently establish a general low-speed turn plant.
- `controller_stops.stopping_request()` is a bounded kinematic sensitivity law,
  not a hand-selected production tune. Latencies are training-event medians;
  margin settings are unoptimized. Most cohorts remain underidentified.

Checks already performed on the then-current code state:

- 82 combined extractor/tuner/video/metrics/dynamics tests, Ruff: exit 0.
  Stops/metrics later had 36 owned tests, Ruff and py_compile: exit 0.
  New timing/validity changes require their affected checks.
- Port guards: **115 passed**, including exact opendbc-pin consistency.
- Torque-chain mutation harness: **17 chain + five enforcement cases caught**.
- Real CarInterface/CarController/CANPacker wire test: one passed,
  17 deselected; 400 cycles at seven speeds, wire torque 409 and normalized 1.0.
- No live road test, panda reflash, controller deployment or runtime acceptance
  has been performed. Do not rerun unchanged gates for reassurance.

Keep one owner per check. Inspect current processes/leases before heavy GPU,
Unreal/Blender, device inference or concurrent data jobs; other projects are
active on this host. The owner is asleep: execute routine choices autonomously,
finish independent work before surfacing a concrete blocker, and preserve data.

## October 13 recent-data checkpoint

The 395-log pull completed: all 4,193,979,510 bytes verified against source hashes,
zero failures. `device-recent-log-transfer/receipt.json` is authoritative; do not
restart that completed transfer. `dataset-recent` now contains 390 decoded logs.
Five named failures remain: four have no carState and one has truncated zstd.
The initial 74 newer-schema failures were investigated using exact committed
schemas from retained `/data/iqpilot`; 71 recovered, three had no state. Original
manifests, failures, metadata and decoded NPZs are preserved by recovery receipts.

IQ.Pilot's selector can relabel held model plans with a current frameId. Exact
EncodeIdx ID matching therefore cannot establish those plans' capture time.
`model_capture_times(data, metadata)` rejects all IQ paths until actual source
exposure is established; frameAge0/nonBIG alone cannot override this gate. New
extraction retains frameAge/BIG/trust fields. Real consumers pass metadata.
`recent-recovery/capture-protection-receipt.json` records additive annotations to
71 metadata files, original snapshots and unchanged NPZs/splits. IQ yaw/roll and
frame-level active-model provenance remain unavailable. Its small source/config
cohorts are kept separate, including recorded incompatible vehicle fingerprints.

Recent Sunnypilot cohort `f9c0b765b31b` has 319 segments / 27 routes, dirty source
`e748a4dda814dd0dd171fe76f2feab2399bd429d` and unknown opendbc identity. The original
16/5/6 chronological route reservation is frozen: 187 training, 28 validation and
104 test segments. `experiment-recent/f9c0b765b31b` identified long delay0.1s and
tau1.2s from training only, interior to the expanded training grid. The 63
training candidates retain support/rejection traces; 131/180 common windows
support this estimate. Validation admits only13/18 windows (MAE0.0885m/s²), below
the20-window threshold. Lateral steady and transient support fail. No axis is
admitted for gain optimization, and an audit hook records **zero opens of all104
held-out NPZs**. This is diagnostic identification, not a new tune or RDF result.

The training search now excludes individually unsupported dynamic candidates
before freezing common comparison windows. Delay/tau endpoints remain explicit
boundary optima. Joint gain search propagates simulated longitudinal speed into
lateral demand and response, constrains speed/path regressions and retains all216
finite combinations. Reference and simulated speed share per-step zero clamping.
Even an admitted moving candidate cannot become a complete controller without
native low-speed stopping evidence. Previously inspected historical test results
remain closed; no old gain selection was retuned against them.

`controller_native_replay.py` executes actual current-master LongControl and native
Hyundai acceleration bounds on a synthetic100Hz causal grid, without actuation.
The completed old0c stop diagnostic admits1736ticks:1030PID and706stopping. Every
stopping tick has a positive raw RDF target but negative native brake command;
shouldStop is preserved. All admitted ticks retain uncertain initialization,
because no observed disengagement resets the recorded controller state. It holds
recorded vehicle motion fixed and proves command semantics, not physical outcome.
The optional native video overlay binds raw log/RDF/native hashes and exposes state,
shouldStop, commands and initialization uncertainty. Original videos are retained.
The real overlay also retains brief PID/stopping switches around segment49s;
positive commands during those PID intervals are a separate discontinuity to
evaluate, rather than assuming all positive raw model targets imply release.

`controller_clip_selection.py` selects32s camera-contiguous windows from training
routes only, excluding overrides and requiring all post-history axis telemetry
valid/active. `recent-clips/selection-strict.json` selected acceleration b7--26
and turning b8--8 on distinct routes. A training-only stopping census found65
episodes, five initially long-active and only one uninterrupted completion.
That b1--25 episode crosses the camera-segment boundary for genuine history;
no32s single-segment uninterrupted stop was invented. Cameras are fetched only
for selected clips through a sole owned integrity-checked transfer.

New checks have separate owners/receipts: tune26, search20, clip selection8,
extractor/metrics/stops81, renderer32 tests passed; native replay13 passed on the
actual POSIX device runtime (host had nine passes/four platform skips). Ruff and
diff checks are recorded per owned change. These are checkpoint checks, never a
road test, deployment, safety reflash or a complete controller-performance gate.

New recent-camera preparation exposed Windows raw-frame pipe stalls. The packet
decoder now uses headless FFmpeg with explicit thread limits and a120s bound,
temporary compact files, file-backed error capture and exact decoded byte counts
before native padding. It fails on decoding errors and retains failed attempts.
Sixteen packet tests pass, including large stderr, actual exit preservation,
short/extra frame rejection and timeout. Clip intervals are cut by exact frozen
first/last frame IDs; a nominal32s duration can otherwise include641 frames due
to the real camera clock, so preparation uses the measured midpoint end boundary.

Compressed-only preparation subsequently completed both new640-pair packets:
acceleration77.682s, turning119.462s. One bounded compact RAM buffer per camera is
native-padded directly into zstd. Metadata explicitly separates stored compressed
files from computed raw hashes/bytes; device expansion must verify the latter.
There are no host raw NV12 files in these packets. The first acceleration camera
reuses the exact successful manual compact decode with command/exit/source/index/
pixel-hash receipts; it is not claimed as a fresh decode. Failures retain clearly
named partial files and receipts. Thirty-seven owned packet tests and Ruff passed;
see `recent-clips/compressed-builder-validation.json` and v5 per-build receipts.
The device owner serializes staging, expansion verification, monitored inference
and new comparison videos. These packets are preparation proof, not model results.
