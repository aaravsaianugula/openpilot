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

Both new RDF jobs subsequently completed, with exact 640 consecutive camera pairs
each and 132 genuine history frames excluded per job. All 508 scored outputs in
each clip met conservative serial availability. Scored compute median/p95/max was
38.214/42.501/68.939ms for acceleration and 38.043/41.789/72.508ms for turning.
Startup outliers remain recorded separately; these short runs do not establish
sustained road acceptance. Both stayed offroad with battery65%; peak monitored
temperatures were45.9C and47.6C. No services, params or vehicle state were changed.

`videos/rdf-v2-real-recent-acceleration.mp4` and
`videos/rdf-v2-real-recent-turning.mp4` each contain320 distinct calibrated frames,
including253 fresh-RDF-visible frames, with real measured motion and hardware
traces. Per-video livecheck receipts record actual commands/exits/source hashes,
inspected stills and six-frame sequences. These are fixed-image prediction
comparisons, not optimized controller outcomes. There is no accepted current/RDF
full controller tune. The newer extraction has390 decoded logs across40 decoded
routes;41 raw routes were transferred. The dirty f9 source's empty recorded GitDiff
does not establish its actual opendbc pin; the source-provenance audit preserves
that missing evidence rather than substituting the base commit's gitlink.

`controller_native_lateral.py` now runs the actual current-master torqueV0 and
jerk-aware extension against the recent turning RDF output, using full raw CS,
vehicleParameters, CP/SP, learner and delay payloads. Typed recorded Params are
read-only; recorded latActive is checked, not recomputed by the engagement system.
The core's published delay plus global0.0 remains separate from extension lag and
RDF catalog smoothing. A publication-as-receive proxy for learner frequency is
explicit; historical socket timing and dirty-controller parity remain unknown.

The actual bounded POSIX replay exited0:3200 synthetic100Hz ticks,2535 admitted,
665 genuine-history/unavailable ticks. All admitted rows retain cold-state
uncertainty and fixed recorded hardware feedback. Raw requests ranged
[-0.2232,0.3733]; native driver/rate limiting produced[-71,146] CAN-equivalent
counts. This clip does not reach the409 ceiling. The native409/3-up/7-down and
driver/angle-request contracts are fixed; full CAN packing is not performed here.
Seventeen native POSIX tests passed, including cold requests above1 before native
limiting and inherited reset-state persistence. Two failed API-wiring attempts,
executed adapters, inputs and logs are preserved. Independent review also fixed
ambiguous frame identity, actual NN constructor asset binding and preparation/
runtime adapter identity. `rdf-v2/recent-turning-native-lateral-transfer.json`
records exact commands, exits, sources, hashes and limitations. No physical
controller improvement, actuation or deployment follows from this diagnostic.

The selected b1--24/25 stop cameras were subsequently found absent on the device
and at exact checked desktop archive paths; their rlogs remain preserved. No
camera transfer, packet build or new stop inference was fabricated. The initial
2GiB-only device reserve missed the active deleter's10%/5GiB thresholds. Exact
camera deletion timing/causality is unknown. Only four newly generated, freshly
hash-verified recent replay buffers were removed:6,218,055,680bytes, restoring
15.717GB free. Compressed packets/results, older buffers, raw logs, code, Params
and services were preserved; exact path/inode/hash receipts record the cleanup.

`device_storage_guard.py` must pass BEFORE every device copy/expansion phase,
counting all simultaneous compressed/raw/temp/output allocations. It reads the
actual hash-pinned native deleter policy without importing it and requires
`max(native MIN_BYTES, ceil(total*MIN_PERCENT/100)) + 2GiB` free AFTER allocation.
It is admission arithmetic, not a reservation. Launchers must block on nonzero
exit and recheck before each serialized phase. Nineteen tests and Ruff passed;
the actual POSIX zero-allocation check exited0, while requesting the earlier
6.218GB allocation exited2 without allocating anything. The frozen RDF adapter
is unchanged. See `recent-clips/storage-guard-livecheck.json`.

`rdf_route_frames.py` implements exact two-source compressed packets with plural
log/camera proof and per-frame source/local-index mapping. Thirty-six tests and
Ruff passed; actual read-only two-log preflight confirmed276 causal inputs and
identical FULL CP/SP snapshots, including fields omitted from old sidecars. Its
real-camera path was unexercised at that checkpoint because b1 footage was unavailable.
Single-source consumers must explicitly bind both sources or reject the packet.

The new `videos/rdf-v2-real-recent-turning-native.mp4` overlays actual native
lateral requests and native driver/rate-limited CAN-equivalent counts against
recorded hardware. It has320 calibrated frames,253 fresh RDF paths and252
admitted native samples; first-availability phase differences are not backfilled.
Every native sample remains initialization-uncertain. The helper binds source/
RDF/output/adapter hashes and recomputes native driver/rate/angle enforcement;
clipped curvature is displayed, not independently replayed. Twenty-one helper
tests and32 affected renderer tests passed; the actual render exited0, with a
peak-curve still and six-frame sequence inspected. No model/controller inference
was repeated. Exact receipts sit beside the new video; originals are preserved.

`controller_route_seams.py` now creates a separate analysis view for two proven
TRAIN segment boundaries. It restores only missing-asof prefix entries from
actual valid, causal raw events within the original freshness limits. Observed
invalid/stale events, original time grids, camera mappings and source arrays are
preserved. Full raw CP/SP, source/schema and frozen training membership are bound;
unknown model capture origins remain unknown. Twenty-five scoped tests and Ruff
passed. The actual derive and two-route stopping amendment exited0; a failed
evaluation caused by omitted static route-context inheritance is preserved, then
fixed using the original census's canonical enrichment without rerunning derive.

The b1 route changes12 to11 episode rows and3 to2 initially active predictions;
its sole uninterrupted completion remains one. The b7 route changes11 to10 rows,
retaining one initially active prediction and zero completed controlled stops.
Manual braking and disengagement across its seam now remain visible censors.
This is a two-route amendment, not a global recount, plant fit or stopline result.
See `recent-seam-code-validation.json`, `recent-seam-repair-v1/manifest.json` and
`recent-seam-stops-amendment-v1.json`.

The frozen replacement stopping diagnostic spans b7--25/26 with320 exact pairs,
132 genuine history and188 subsequent inputs. The two25 cameras were subsequently
downloaded and all four source files verified; only74,943,401bytes were newly
transferred. Segment26's selected indices0..153 do not overlap the
earlier replay. Only37 scored exposures precede the exact raw brake event; output
availability must also precede that censor. Subsequent footage is takeover-only
display, excluded from controlled scoring. Preparation and inference receipts,
not this selection, establish whether that diagnostic actually ran.

The first real two-source packet build exited0 in55.319s, with320 exact IDs,
continuous history across the50.000682ms segment boundary and no raw host files.
Both compressed camera streams were separately decompressed in bounded RAM and
verified against their777,256,960byte native hashes. Full source/static contracts,
calibration, action timing and input files stayed unchanged. See
`recent-clips/build-stopping-intervention-v1-receipt.json`; packet preparation
does not establish inference completion or controlled stopping performance.

`controller_longitudinal_comparison.py` pins the exact historical selected.4/.4
candidate, search trace and identified plant. Its predeclared mixed-direction
validation window was chosen by target-direction support, not candidate error.
It generates deterministic paired baseline/candidate traces without refitting,
searching or opening test data. Both simulated paths explicitly use recorded
future yaw for illustration; this is not a lateral response prediction.
Nineteen scoped tests and Ruff passed. The actual30-frame
`videos/historical-selected-longitudinal.mp4` render exited0, with calibrated
camera overlay, conditional BEV and measured/simulated command, acceleration and
speed traces. A still and seven-frame sequence were inspected. This3s window's
target-acceleration MAE is0.071452 baseline versus0.051931 selected, descriptive
only. Exact phase receipts preserve commands, exits, stdout/stderr and failures.
Earlier simulation videos used different gains and do not represent this result.

The B7 stopping/intervention RDF job subsequently completed with exit 0 and all
seven transferred artifacts hash-verified. Across five completed jobs there are
2,320 real synchronized pairs, 660 genuine history frames and 1,660 subsequent
timing-admissible outputs. B7 has 188 such outputs, but only 36 were ready before
the exact raw manual-brake censor at 68914.104516183. Thirty-seven exposures
precede braking; the last corresponding output was already too late. There are
zero complete three-second controlled GT windows and no completed stop in this
clip. Later predictions are takeover display only. Compute median/p95/p99/max
was 39.460/45.574/53.658/58.112 ms; battery 65%, maximum temperature 44.3C and offroad
state stayed within the bounded test limits. Startup and temporal-state reset
proof remain separate in the frozen bench receipts.

The first two-source video loader failed before decode because it treated the
frozen producer's boolean history mask as a scalar; the corrected loader passed
against both actual logs, four cameras and all causal inputs. A subsequent
file-backed render completed both exact camera decodes, then exceeded its600s
outer bound during composition. Its final partial contains 150 complete frames,
not the stale 11-frame reading observed earlier. Both decoded buffers, the 720MB
partial, executed source and actual timeout/exit receipts are preserved. A bounded
single-frame reproduction passed and did not reproduce a blocking stack, so the
failure's cause remains unproven. Explicit bounded RAM recovery is being checked;
no successful B7 video or native stopping outcome follows from that attempt.

`native-parameter-contract.json` records the actual current native source contract
and matches four normalized source hashes to existing POSIX replay receipts.
Historical inverse-response gains are research parameters, not native Ki values.
Native optimization against a trusted plant and low-speed stopping evidence is
still unfinished; neither the successful RDF inference nor fixed-feedback native
diagnostics establishes an accepted full tune.

The explicit RAM recovery subsequently passed: 160 distinct encoded frames,
159 calibrated projections and 92 fresh visible RDF paths. Seventeen displayed
frames have pre-brake output availability; 75 are post-censor takeover displays.
Both camera buffers were read once with exact executed select/source/index/size
proof and fresh hashes. The 1,258,291,200 byte arrays were admitted below a 2GiB cap;
composition took 7.0133s and encoding 3.4140s, with all 768,000,000 pipe bytes confirmed.
The encoder deadline covers the writer and child; real pipe-backpressure tests
exercise timeout and cleanup. Old RGBs and the 720MB partial remain preserved.
No camera decode, model inference or native controller replay was repeated.

`videos/rdf-v2-real-stopping-intervention-16s-ram-v3.mp4` has SHA256
`56313748b8015ee45bad978f70b47caa163b4bb053fab64d6c123180897362ad`.
The actual render and encoder exited 0. Pre/post stills and an eight-frame seam
sequence were inspected, including the explicitly missing telemetry/calibration
frame. The artifact distinguishes availability from outcomes and labels driver
takeover; it computes no controlled improvement score. Exact sources, commands,
exits, preserved files and test receipts are in the adjacent `livecheck.json`.

The new `controller_native_route.py` subsequently prepared both actual raw logs
and ran the unchanged current-master LongControl on the real two-source B7 tape.
Nineteen actual POSIX tests passed; the actual replay exited 0 and all ten pulled
files were hash-verified. Of 1,600 synthetic 100Hz ticks, 936 were admitted:
178 pre-brake active PID commands and 758 post-censor off-state diagnostics.
Every admitted tick remains initialization-uncertain. No admitted RDF target has
`shouldStop=true` in this clip; it contains no native stopping-state completion.
The positive-target/true-stop behavior is covered by a separate actual-native
regression fixture and is not attributed to this footage.

The recorded CP has KiV `[0.0]`, and the verified native constructor fixes Kp at
zero. Under this conditional replay the approach command therefore follows
acceleration feedforward without integral acceleration-error correction. Changing
`stopAccel` alone cannot change this approach while `shouldStop` remains false.
This narrows the tuning problem; it does not establish an optimal replacement
gain or prove the historical dirty controller matched current master.

Exact sources, commands, exits, process groups, monitoring and results are in
`rdf-v2/stopping-intervention-native-long-v1/transfer-receipt.json`. Both owned
process groups exited, with offroad state, battery65% and maximum temperature38.1C
recorded across seven monitor snapshots. Device allocation guards passed before
the small source/input upload and execution. Params, services, safety, vehicle
state and the existing native adapter were preserved; no model inference repeated.

`controller_native_search.py` now executes the actual native longitudinal class
against simulated speed/acceleration feedback and empirical supported response
cells. The 100Hz substep preserves the identified20Hz backward-Euler pole; this
is an explicit resampling assumption, not proof of high-frequency car dynamics.
It separates future action-time feedforward from capture/horizon-aligned physical
tracking targets, binds consumed RDF action times and smoothing, and refuses the
latest explicitly inadmissible output without falling back to an older target.

The finite Ki/stopAccel selector preserves actual baseline settings and records
all failed or unsupported trials. It compares the same supported validation
windows, applying direction, comfort, saturation and stopping regressions. Stop
feasibility requires a causal stationary model prediction and a hash-bound margin
derived from at least20 uninterrupted controlled TRAIN outcomes across two routes.
One physical log cannot be relabelled as two independent routes. Observed stop
residuals do not become stopline ground truth. Native stopping may begin near
standstill; a continuous prior moving approach is required rather than forcing an
early native state. Current data has neither trusted native plants nor that margin.

Forty-three host tests passed with one actual-runtime skip. The unchanged core's
full actual POSIX suite then passed44 cases, including real capnp CP/SP and
LongControl stopping semantics. A separately added real-native moving PID/Ki
regression passed alone, with44 prior cases deselected: changing Ki changes actual
native commands, while replacing future recorded motion leaves commands, integral
state and simulated motion unchanged. These are synthetic algorithm/API fixtures,
not car-performance results. Independent source review fixed the cross-route log
alias defect and closed on source SHA
`b24eea007afad36e818211f30a1c7fe563b2fa5f9ad1b9acd1fd24d63ee68ac5`.
Exact manifests, commands, exits, guards, process groups and health sit in
`recent-clips/native-search-api-v1-result` and `native-search-pid-api-v2-result`.
No drive arrays, fits, held-out data or model inference were used for these checks.

The new plural native overlay binds both logs, four cameras, packet/RDF, prepared
native tape, result/transfer receipts, executed adapters and eight native sources.
It verifies the actual100Hz raw inputs, admission, target/stop/censor masks and
acceleration bounds before drawing. Shared single-source consumers are unchanged.
Thirteen affected validator/layout tests and Ruff passed. The actual bounded RAM
render exited0, with160 frames,159 calibrated views,92 fresh RDF paths and93 native
samples. Its native caption and takeover banner occupy separate space. An actual
post-brake still and four-frame seam sequence were inspected, including missing
display telemetry. No decode, model inference or native replay was repeated.
`videos/rdf-v2-real-stopping-intervention-native-v1.mp4` has SHA256
`f4144992c15ae9796cfb824cd10a80d5743ffc83d1f9a5379cc3a01af80167ca`.
All commands remain cold and conditional on fixed recorded feedback. No accepted
native long/lat tune, physical improvement or completed stop follows.

### Left/right and combined acceptance — October 13 follow-up

The owner explicitly requires both turn directions and simultaneous longitudinal/
lateral tuning. The existing combined selector already propagates simulated long
speed into lateral demand and response. Its previous pooled lateral tracking score
could hide a direction regression. `controller_turns.py` now supplies frozen FRD
reference direction, causal entry/steady/exit labels, balanced direction objectives
and separate direction/phase support, error, jerk and saturation gates. Root
integration rejects old pooled-only rows and a left-turn regression even when the
combined longitudinal/lateral mean improves. Sixteen helper tests and twenty-three
integration/search tests passed under their separate owners; Ruff/diff checks passed.

The actual new diagnostic opens only the two previously selected TRAIN arrays.
`recent-clips/turn-selected-coverage-v2.json` records 508 scored analysis-grid rows
per clip, no invalid/override rows after the canonical held-calibration helper,
335 left-request rows in acceleration and 142 in turning, and no qualifying right
requests in either. These are recorded demand/motion coverage, not candidate
performance or frame-capture GT alignment. The first raw-calibration view and its
driver remain preserved as v1; v2 uses the same view as the frozen clip selector.
All reserved test arrays remain unopened. A separate new right-turn selection is
limited to the frozen TRAIN candidate catalog, with no tuning-error ranking.

Native lateral simulation needs actual rack-angle dynamics in addition to measured
yaw response. Existing 20Hz acceleration-only plants cannot supply that feedback.
Hyundai's recorded SAS_Speed is unsigned; it must not classify turn direction or be
invented as a signed rack derivative. No 50ms response may be applied every10ms.
The new native runtime adapter is under implementation/review; actual API proof and
a supported multirate joint plant remain required. No full native tune is accepted.

### Native lateral runtime proof and shorter right-turn acquisition

The new runtime adapter now passed actual POSIX execution against all44 independent
host-bound native source/asset hashes and the unchanged master/opendbc pins. Twenty-
one fixture cases passed in the first actual run. Its real-native case initially
correctly rejected a synthetic CP lacking RAISED_LIMITS (native384 instead of409).
After that fixture correction it reached frame300 and exposed an exact decimal
assertion against capnp Float32 friction. Both named failures, commands and outputs
remain preserved. Only the corrected actual case was rerun, passing1 with21
deselected; the other cases' AST nodes remain unchanged. Real v0/jerk proof covers
A/B/A provider isolation, both turn signs, simulated angle feedback, frame300
effective overrides and native reset versus cold restart. These are synthetic
API checks, not measured vehicle or tuned performance.

Independent review also corrected a test that had supplied raw requested torque
as delivered feedback. It now uses applied CAN counts/409 and verifies next-tick
steering-limited feedback. Runtime source was unchanged throughout the fixture
corrections. All owned process groups exited, with offroad battery65% and execution
monitor temperatures at most38.7C; Params, services, device source and safety were preserved.
Exact evidence is under `recent-clips/native-lateral-runtime-api-v{1,2,3}-*`.

The full recent TRAIN32s census opened187 arrays and retained1049 window outcomes:
no eligible right-turn window, mostly because manual intervention cut the horizon.
Acquisition now accepts explicit durations with the unchanged132 history frames.
Thirty-four selector tests and Ruff passed. The new16/24s census opened only the
same187 TRAIN sources, retaining2997 new anchors and reusing the32s result without
rerunning it. Fourteen eligible right-turn windows span four routes. The first
24s selection, b8--9 frame11204..11683, contains273 right-request rows with100%
measured-yaw sign agreement and92/110/71 entry/steady/exit rows. Camera availability
is unverified; b1--24 remains known absent from the prior storage incident. No
model inference, fit, controller search or test-array access occurred in this census.

### Raw rack/yaw motion proof and right-camera availability

The selected b8--9 narrow/wide pair is present on the device, with74,774,822 bytes
and full source hashes matching the frozen TRAIN selection. Its full CP/SP records
precede the first history exposure. Storage rejected the minimum new replay peak.
After concrete root review, only the two newly generated, compressed-reconstructible
B7 stopping-intervention NV12 buffers were removed:1,554,513,920 bytes, with fresh
literal path/inode/link/hash and before/after evidence. Old e9/stop buffers, all
camera footage, packets, compressed streams and results remain preserved. Exact
future packet bytes and per-phase native-deleter guards are still required.

Independent review found two raw-motion producer defects before its first decode:
sendcan bus0 had been confused with returned echo source128, and physical aliases
could bypass a logical TRAIN path. Six first-red regressions reproduced them. The
fixed source passed19 tests and Ruff, and the reviewer closed both findings without
repeating checks. Original source/test snapshots and failures remain preserved.

One bounded new raw-motion run then decoded six selected TRAIN logs in34.528 seconds,
exit0, with no fit or old/test array opens. All twelve output files are hash-bound.
The first launcher preflight had rejected the later IQ-protection split-file hash,
before any schema/raw decode. The successful launcher uses the preserved original
frozen split and verifies the current selected-cohort partitions are identical.
Both launcher states remain recorded; no dataset or split was rewritten.

The actual tapes retain roughly100Hz carState/rack data and20Hz calibrated LLK yaw.
Every selected log has finite roll but nonfinite roll uncertainty, so the strict
LLK quality mask admits zero rows; required `CustomTorqueParams` is also absent.
No uncertainty, policy value, signed vendor rate or physical actuation was invented.
The new multirate joint response and native coupled algorithms remain under work;
no current full long/lat candidate or physical improvement is accepted.

The new pure joint multirate core passed19 owned tests and Ruff. It advances one
nominal50ms joint rack/yaw/unsigned-rate median after five completed native input
nodes, holding feedback between updates. It retains actual48..52ms sample deltas,
strict source/static/policy/TRAIN admission, invalid/request-false interpositions,
two-route/100-sample cell support and no extrapolation. ZOH/Markov response,
physical delivery and native100Hz feedback are explicitly unvalidated, so its
selection and production flags stay false.

The first actual driver rejected periodic duplicate static snapshots before any
fit. Inspection proved the duplicates byte-identical. The corrected new v2 driver
binds each route's own earliest matching full CP/SP prefix, keeps the first failure,
and runs only the six already produced TRAIN tapes. Full semantic CP/SP is equal
across these routes, but exact raw typed policies differ, so they remain two groups.
The diagnostic retains3401 and3386 excluded windows respectively, zero accepted
windows/cells and no candidate. Missing roll uncertainty, policy and historical
source identity were not filled or bypassed. No old or reserved NPZ was opened.

Root also fixed a concrete full-CP proof failure: firmware Cap'n Proto Data bytes
could not be JSON-hashed. Two new cases failed first, then three focused cases
passed with45 prior cases deselected. An actual recorded CP/SP with17 firmware Data
fields hashes without stripping anything and changes identity when one Data value
changes. AST comparison proves the other native search functions and all25 prior
test functions unchanged. Independent review accepted this schema-valid CP/SP path;
the Data marker is not claimed to be a generic type-injective serialization scheme.
No native controller/API behavior was changed or rerun by this hash correction.

### Moving native joint search: host proof, actual runtime pending

The new joint runner now caches the existing native LongControl trials and supplies
their pre-transition simulated speed/acceleration to the actual lateral adapter.
It keeps simulated rack/yaw/rate feedback between empirical50ms transitions and
passes five unshifted CAN-command/speed nodes to each joint update. The same raw
RDF current-time geometry and fixed reference path/turn populations apply across
candidates. Independent acceleration/braking and left/right entry/steady/exit,
jerk, saturation, speed/path and stopping gates retain every failed common window.
No complete candidate or production path is enabled.

Review found and closed three admission defects: an unfinished terminal macro could
be called complete; native horizon mismatch was checked after LongControl ran; and
the inference-time lateral delay lacked an independent source witness. Five new
cases reproduced the defects before correction. Complete five-tick horizons,
native horizon equality and exact frame/capture/delay/packet/source witnesses now
fail before either controller factory. Inference delay is distinct from the later
native core delay and extension cache. Root also caught an unused obsolete direct
Kp assignment in the POSIX fixture; the pinned schema and AST check verified its
removal without changing controller behavior.

The final owned host suite passed37 with one POSIX skip; Ruff passed. Root and
independent changed-line review found no remaining scoped source blocker. Original
source/test snapshots, red tests and final commands/hashes are in
`recent-clips/native-joint-source-review-42baf`. The actual POSIX case has not run.
The comma's saved hostname/address and strict known-host SSH checks time out;
targeted neighbor/hostname discovery found no reachable alternate. No new device
allocation, packet expansion, right-turn inference or joint CPU job was launched.

The right-turn host packet is complete:480exact pairs/132history/348subsequent,
283.868seconds, builder/wrapper exit0. Its633,514,128 compressed bytes plus
2,331,770,880 expanded bytes and268,435,456 temporary/output headroom require
3,233,720,464 additional device bytes. This is a plan, not current storage admission.
Identity/source/offroad/health/job checks and phase guards remain mandatory when
connectivity returns. The two preceding wrapper failures happened before decode
and remain preserved; the successful packet must not be rebuilt.

The moving runner cannot finish standstill/hold windows. The next separate regime
must preserve the same continuous native handle, permit legitimate inactive
zero-CAN/request-false behavior and query a separately admitted empirical transition/
hold plant. Raw-plan availability and undefined zero-speed curvature are different
contracts; no curvature, rack state, contact response or model-stop ground truth
may be invented. Current source, roll uncertainty, policy and low-speed evidence
remain insufficient. No road test, deployment, actuation or flash occurred.

### Separate low-speed and hold response: source checks

The new empirical stop/hold model preserves exact five-node active/request patterns
and allows measured inactive angle/yaw/rate changes instead of freezing them.
Source/static/typed-policy timing, zero driver input, raw validity, independent
output/echo clocks, support and no-extrapolation gates remain explicit. It keeps
contact, nonreversing and100Hz feedback validation false, with no selectable tune.
The final35-case host suite and Ruff passed. Independent review reproduced a real
prefix bug: a phased car-state predecessor consumed at start-minus20ms could sit
outside the15ms quality span. Every consumed as-of row is now checked; three first-red
failures and nine affected passing cases are retained with original source snapshots.

No data fit was repeated. The new native approach/stop/hold runner remains under
implementation/review, and device connectivity still blocks actual native evidence.
The Elantra overlay guard command passed all115 checks without modifying controllers.

Root also traced the missing calibrated roll uncertainty to the committed location
publisher: both recorded-base e748 and current-base13a blobs explicitly initialize
that std vector with NaNs. Calibrated angular velocity receives computed covariance.
The blobs are identical and source-bound in validation-llk-roll-uncertainty-source-v1.
This explains a publisher limitation, not the dirty runtime or a valid uncertainty
substitute. Existing selection/source/uncertainty gates have not been relaxed.

### Native approach, stopping and hold: host implementation

The new continuous native stop/hold runner passed39 host cases and Ruff. It reuses
the same lateral handle, source/delay/learner admission and cached native LongControl
feedback, with separate moving and inactive/mixed response models. The shared finite
search evaluates moving and stopping windows on fixed common IDs. Model stop-target
distance, hold motion and unknown full-horizon geometry remain separate contracts;
no model endpoint becomes stopline ground truth and no restart regime is invented.

First-red regressions closed an unaligned moving-macro admission defect, a cached
Long failure summary crash, uncensored simulated release after zero and unsupported
terminal macros extending hold evidence. Supported steps are now distinct from
used command rows, and hold/turn metrics consume successful complete macros only.
Actual native stop/hold execution remains unperformed; all contact, initialization,
100Hz feedback and source gaps keep selection absent and production disabled.

Independent review also found absent recorded hardware-standstill eligibility.
The corrected runner requires a source/hash-bound raw uint64 carState publication
stream and validates latest as-of clocks,15ms freshness/gaps, every consumed invalid
interposition and recorded standstill/speed compatibility before either controller.
Three first-red cases reached the cache before correction; the final suite covers
missing, malformed, stale, wrong-source, clock, gap and invalid-state boundaries.
Measured witness values never enter simulated feedback, and invalid future rows
cannot change the checked window. Original source/tests/receipts remain snapshotted.
One final integration correction matches the frozen producers' integer-times1e-9
clock conversion exactly. Division by1e9 can differ by one ULP for the same original
nanosecond timestamp; the earlier fixture had hidden this by recanonicalizing clocks.
Two first-red producer-clock/as-of cases passed after correction, with no tolerance
widening or edits to existing producers. Final39-case checks supersede earlier
19/37-case source snapshots, which remain preserved.

One strict saved-identity SSH recheck after65 minutes again timed out (exit255,
5.053s); the remote never ran. Its exclusive v2 connectivity receipt preserves the
command, fingerprint and prior completed-job state. No allocation, upload, right-turn
inference or native API job occurred. The prepared packet remains ready, and the
owner's earlier address question remains pending without repeated prompts.

### Local native joint API proof

The previously pending moving and continuous approach/stop/hold cases now each
passed once against actual native LongControl and V0 classes on isolated x86 WSL.
The exact 13a/b45/msgq tree, 44 native files/assets, two built libraries and 21 research
snapshot modules verify before and after execution. The moving case reports 1 pass
in 4.78 seconds; the continuous stop case reports 1 pass in 4.43 seconds. Both JUnit files contain
one non-skipped successful case, and all owned process groups exited.

The stop case checks 200 ticks/40 macros on one actual lateral handle, matching
simulated longitudinal feedback, native state transitions, zero inactive command
and request behavior, the 409 ceiling, rate bounds and synthetic hold motion. Its raw plans,
source/support labels, uncertainty and plant cells are explicit fixtures. Selection
and production remain disabled; no physical stopping or tuning benefit follows.

The artifact-only environment installed 17 pinned wheels, then exactly one proven
missing locked zstandard wheel. The original post-install scan timeout, first
native include-path error and pytest capture/import failures are preserved. Header
generation, dependencies and successful builds were reused rather than repeated.
Final proof is `native-posix-api-v1/phase3-api-v3.windows-postproof.json`; native
source, recorded data, device Params/services and production controllers are unchanged.
Fresh right-turn inference still requires connectivity to the comma. No road test,
deployment, vehicle actuation or panda flash occurred.

### Supported-macro scoring and longitudinal payload observations

The moving scorer now excludes failed/partial macros and failed windows from
turn evidence on the fixed population. A cached native error no longer crashes
the grid summary. Final Joint 41 host checks, two affected Stop summary cases and
Ruff passed. One affected actual moving native case passed; its native execution
function and test AST match final source. The final summary-only filter has its
own regression. The unchanged actual Stop case was not repeated.

One reviewed, exclusive derivation processed the six existing TRAIN Motion tapes
without re-extracting logs or fitting a model. Requested acceleration, independent
post-CarController reported acceleration and actual canValid retain original
publication clocks and invalid rows. Counts are 35,451/35,437/35,439; usable-row
counts are 35,409/35,407/35,411 respectively. These are stream quality counts,
not plant admission. Original hashes are unchanged, and an explicit NPZ open
allowlist excluded all reserved arrays. Twenty-five real-capnp fixtures and Ruff
passed before the actual 4.676s worker exit 0. The caller checks helper hashes
before imports, authenticates the recorded schema and writes additive outputs
only. Reviewed failure/timeout bounds and exact receipts remain in the tuning tree.

The next algorithm boundary is a genuine five-state two-way empirical response,
not another unchanged lateral or longitudinal fit. Native LongControl and lateral
control must share simulated speed/acceleration/rack/yaw/rate and advance together
after successful complete macros. Candidate reported acceleration requires exact
native request-to-report processing; recorded output cannot become candidate
feedback. Existing source/policy/roll/low-speed/ZOH gates remain unchanged.
Right-turn inference remains pending saved-device connectivity. No accepted full
tune, new inference, deployment, road test or actuation occurred in this checkpoint.

### Five-state response and native acceleration-report boundary

The pure two-way model now advances speed/acceleration/rack/yaw/vendor-rate
together from21 features: initial state, physical LLK roll and five requested
accels, independent published accel reports and CAN counts. Thirty-seven host
checks/Ruff passed, including source/split-before-array-access, invalid consumed
prefixes, malformed model and steering-dependent simulated speed/acceleration.
No actual fit or held-out access occurred. Original source ineligibility cannot
be repaired by additive metadata. Kinematic consistency, joint median coherence,
100Hz ZOH and low-speed response still block selection.

A separate pinned Hyundai native software-output adapter preserves prior-report
publication, actual apply phase and skipped applies. Its first actual POSIX case
failed at cold enum representation; the native constructor uses ordinal0 while
the initial adapter expected a name. That run, cleanup and source/Params proof
are retained. A narrow schema-bound correction passed53 host cases with one
opt-in native skip, rejecting invalid enum types/ordinals and retaining native
state/phase fences. The corrected actual API case passed once in5.03s with
synthetic inputs on isolated pinned x86 POSIX. This is API evidence only.

The shared runner remains unfinished. Actual card source consumes the prior CC
before CO/CS publication, then applies that snapshot. Native lateral calculation
and steering hardware apply must be separated so skipped applies and later CC
arrivals cannot advance the rack command or limiter early. Fixed physical targets,
left/right phase populations and successful-macro coverage remain mandatory.
No new real fit, comma inference, accepted tune or physical improvement is claimed.

The separate lateral compute/apply API now preserves native control history
without speculative hardware updates. Skipped applies hold transmitted counts
and angle-counter state, and later queued CC arrivals cannot substitute for an
earlier consumed snapshot. The legacy wrapper remains compatible. Twenty-three
host cases passed; one new actual native case passed once in4.19s, with exact
source/assets/libraries/dependencies and isolated Params proof. Ruff's initial
C408 test-helper construction was preserved and corrected as a literal-only
equivalent; unchanged host cases were not repeated. The source/API checkpoint
does not complete the shared-state runner, source-bound pre-roll, packet-calibration
binding or actual50ms macro-boundary validation, which remain in progress.

### Shared native three-adapter integration

The separate shared-state runner now keeps simulated speed, acceleration, rack,
yaw and unsigned vendor rate together. Native LongControl, native lateral control
and Hyundai request-to-report processing consume causal candidate publications;
recorded future feedback supplies no simulated state. Command calculation and
hardware apply remain separate, and the empirical50ms response commits at the
actual macro boundary. Candidate-independent physical targets, left/right phase
populations, successful support and all failed finite-grid rows remain explicit.

Thirty host checks/Ruff and independent review passed. One new actual native
three-adapter composition case passed once in21.10s on isolated pinned x86 POSIX,
with a non-skipped JUnit case, exit0 and owned process group gone. Native source,
dependencies, libraries,26 research snapshots and the Params trap were verified.
The A/B/A case checks two-way feedback and future-unused-GT invariance using
arbitrary predeclared synthetic plant cells; it is not empirical tuning evidence.
Preparation/proof-only path and symlink-stat failures are preserved without another
native run. The full tuning goal remains incomplete: real source/policy, roll
uncertainty, latency history, ZOH and low-speed/contact support still forbid a
current full candidate. No deployment, road test, actuation or panda flash occurred.

### Right-turn replay and shared low-speed component

The owner explicitly approved removing exactly10 verified replay-input copies.
One bounded cleanup removed5,442,267,182 bytes, preserving original footage,
models/results and all14 protected files/three source hashes. Native storage
guards then passed each new right-packet phase. Transfer, two streaming expansions,
one exact RDF inference and result pull all exited0. No old inference was repeated.

The right clip adds480 real narrow/wide pairs with132 genuine history/348 later
outputs; all348 meet the replay deadline. Median/p95 compute is38.7/45.8ms with
startup/warmup separate. One new video render exited0 in27.985s:240 distinct
calibrated frames/173 fresh RDF overlays. Actual peak and eight-frame sequence
inspection confirms right-turn camera/BEV/trajectory/hardware traces and explicit
fixed-image/unvalidated labels. No continuous playback or physical tuning gain is
claimed. Six verified clips now total2,800 pairs/2,008 timely outputs. Packet and
results remain retained; every future allocation needs a fresh storage guard.

A new pure five-state low-speed/hold predictor passed35 synthetic host cases,
Ruff and bounded independent review. Exact queued compute/CC-consume/CS/apply/echo
clocks, initial standstill and next categorical standstill remain separate from
speed. Ambiguous categories and unsupported active/request patterns are refused.
Original unknown source eligibility cannot be promoted. First-red unimplemented
API, duplicate-declaration and malformed-metadata failures are retained.
No real fit or native hook ran. Next integration must extend the shared native
event recurrence through approach/stop/hold without the old LongCache, preserving
shouldStop, continuous handles, causal flags, censoring and fixed geometry prefixes.
Full candidate selection and production remain disabled.

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
