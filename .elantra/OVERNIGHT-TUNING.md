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
- `archive-analysis.json`: enriched, 28 provenance cohorts, 24,653 admitted
  three-second windows. Numeric model-to-GT metrics remain provisional until
  exposure-origin alignment is corrected (see below).
- `stopping-analysis.json`: 415 initial model-stop episodes; 36 initially
  comma-long-active / 379 inactive; 312 observed stops, 53 timeouts, 50 motion
  block endings. Only five uninterrupted comma-controlled completed stops.
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
- The old flat archive's v2 manifests are one-byte placeholders, not weights.
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
  Existing isolated-bench guard requires external power. **Do not start sustained
  comma inference on battery.** Continue desktop implementation/data work and
  recheck power at follow-ups; the physical power condition is a real blocker.
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

1. Fix timing interpretation before optimizing against trajectory error.
   Extracted `model_t` is modelV2 **publication** monotonic time; a 50 ms
   publication-age gate does not bound camera/model processing latency.
   `controller_extract.model_capture_times()` now matches exact
   `model_frame_id` to `raw_encode_frame_id` / `raw_encode_t`. No publication
   fallback or ambiguous frame-ID matching is allowed. Anchor measured GT at
   camera exposure time, or explicitly report capture age and any remaining
   grid-origin error. Do not certify prior endpoint metrics as time-aligned.
2. Complete source-valid onset gating in stops: planner-negative onset requires
   valid/fresh planner messages; commanded-negative onset requires valid/fresh
   carOutput. Missing/invalid onsets need explicit coverage, not fake timings.
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
