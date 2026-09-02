# Elantra 2024-25 port, on a branch that keeps itself current

This branch is **sunnypilot master with the Elantra 2024-25 port on top**, rebuilt weekly.

Install on the device with the custom software URL:

```
aaravsaianugula/master
```

Supported: Hyundai Elantra 2024-25, Elantra Hybrid 2024-25, i30 Hybrid 2024. Requires SCC,
LFA and LKA, a comma 3X or comma 4, and (almost certainly) the Hyundai K harness.

## Why this branch exists

The community port branch is maintained by merging sunnypilot master into it by hand and
bumping a submodule. That works until it doesn't: the branch drifts, sometimes by months.

So this one is **derived, never merged**. Every sync throws the branch away and rebuilds it:
hard-reset to a chosen upstream commit, replay a small overlay, publish. There is no merge
history to rot, and no conflict to resolve, because there is no merge.

## What the port actually is

Almost nothing lives in this repo. The entire Elantra port is 10 files in **opendbc**:

| File | What it does |
|---|---|
| `car/hyundai/values.py` | `HYUNDAI_ELANTRA_2024` (`CAMERA_SCC \| CHECKSUM_CRC8 \| RAISED_LIMITS`) and `HYUNDAI_ELANTRA_HEV_2024` (no `RAISED_LIMITS`), harness Hyundai K; both flag words; the 409 ceiling |
| `car/hyundai/interface.py` | Bridges `HyundaiFlags.RAISED_LIMITS` into the `safetyParam` bit — the only path by which panda learns the ceiling is raised |
| `car/hyundai/fingerprints.py` | ECU firmware fingerprints |
| `car/hyundai/hyundaican.py` | Both platforms added to the LKAS11 LDWS mode list |
| `dbc/generator/hyundai/hyundai_can.dbc` | `LFAHDA_MFC` (0x485) 4 → 8 bytes |
| `safety/modes/hyundai.h` | Panda TX allow-list for 0x485 (4 → 8 bytes) and `HYUNDAI_STEERING_LIMITS_RAISED = HYUNDAI_LIMITS(512, 3, 7)` — **512, not 409**: panda is deliberately permissive and the 103-count gap is unenforced |
| `safety/modes/hyundai_common.h` | `HYUNDAI_PARAM_RAISED_LIMITS = 1024` and the `hyundai_raised_limits` flag panda selects on |
| `car/torque_data/substitute.toml` | Borrowed torque parameters |
| `sunnypilot/car/car_list.json` | The three entries in sunnypilot's car list |
| `car/tests/routes.py` | CI test routes |

**The steering torque ceiling is a flat 409 counts at every speed** for `HYUNDAI_ELANTRA_2024`
only, against the 384-count HKG default. It is not a speed schedule; there is no lookup table
and no interpolation. `STEER_MAX` is a *gain*, not just a ceiling — every command is multiplied
by it — so 409 raises every command by 6.51 %, at every speed.

**409 is the MDPS's measured acceptance limit, not a preference.** It accepts 409 and trips
`CF_Mdps_ToiFlt` at 410. Two builds went above it — a schedule peaking at 500 and one at 450 —
and between them put 158 frames over 409 on the wire and took 19 fault onsets while steering, at
commanded counts 410 through 433. Flat 409 has ~1.59M engaged frames with none. It is the *value*
and not the schedule: `np.interp` clamps below its first breakpoint, so every one of those onsets
fired under a stationary ceiling. **Do not raise it.** Full evidence in
`ROAD-TEST-cn7-lateral.md`; the scanner is `eps_census.py`.

**panda does not police this.** It enforces a flat 512, so a build that commands 450 sails
through and the EPS is what stops the car. The guards therefore assert the **literal** 409 —
`opendbc <= panda` is vacuous here, since 450 satisfies it and still breaks steering.

**The `hyundai_can.dbc` / `hyundai.h` pair is the safety-critical part.** The dbc says how
many bytes openpilot packs into 0x485; the safety header says how many panda will let out.
If they ever disagree the car either refuses to steer or panda's allow-list is wider than the
message it guards. `guards.py` asserts them together, and the sync aborts if they diverge.

This is also why the branch is built from source rather than from a prebuilt sunnypilot
release: prebuilt branches ship a panda binary compiled against stock opendbc, which would
reject the 8-byte frame.

## The overlay

Everything this branch adds on top of upstream, and nothing more:

*Files that are entirely ours — restored wholesale each sync, so they never conflict:*
- `.elantra/` (this directory: sync, guards, verify, the drive scanners, tests, manifest — see
  [What is in this directory](#what-is-in-this-directory))
- `.github/workflows/elantra-sync.yaml`
- `openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_updates.py` — the panel
- `openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_manifest.py` — its pure logic,
  importable without a UI so `.elantra/test_port_panel.py` tests the shipping code

*Upstream files we touch — ~26 lines, deliberately tiny, since this is the only conflict
surface in the design:*
- `.gitmodules` — opendbc URL → our fork
- `opendbc_repo` — the gitlink
- `openpilot/common/params_keys.h` — three params
- `openpilot/system/updated/updated.py` — publishes the build manifest (read-only; no new
  logic in the update or finalize path, because that path is what bricks devices)
- `openpilot/selfdrive/ui/sunnypilot/mici/layouts/settings.py` — registers the panel

The overlay diff is **derived from the previous build** rather than stored as a static patch,
so editing any of those files by hand and committing is all it takes — the next sync picks
the change up automatically.

## The weekly sync

`.github/workflows/elantra-sync.yaml`, Mondays 09:00 UTC, plus **Run workflow** for an
off-cycle sync.

1. Walk sunnypilot master newest-first and take the first commit whose **own CI went green**.
   sunnypilot reports through check-runs, so that is what gets queried — the combined status
   endpoint returns `pending` forever and would make the gate a no-op.
2. Rebuild `aaravsaianugula/opendbc:elantra` = opendbc master + the port delta, recomputed
   fresh from `sunnypilot/opendbc:elantra-2024-port` on every run so upstream fixes to the
   port arrive automatically.
3. Reset master to the chosen commit, replay the overlay, pin the gitlink, write the manifest.
4. Run `guards.py`. Run opendbc's own Hyundai tests.
5. Move `master-previous` to the outgoing build, then force-push `master`.
6. Run `verify_published.py` against what actually landed on GitHub.

**Nothing is published unless every gate passes.** On any failure the run aborts, opens an
issue, and leaves `master` exactly where it was — the car keeps running the last good build.

## From the car

Settings → **elantra port** (comma 4):

- which sunnypilot commit this build came from, and whether its CI passed
- how long ago the branch was synced
- **verified builds only** — hides Install for a staged build whose upstream CI is not green
- **check for update** / **install update**
- **roll back to previous build** — repoints the stock updater at `master-previous`

The device can't *trigger* a sync, only pull what the weekly job published. "last synced"
is shown so that's visible rather than mysterious.

## Adding your own car's firmware

The most likely first failure is the car not fingerprinting: the port lists two camera
firmwares and yours may differ. Read the versions off the device (Settings → Device), then:

```bash
cd /path/to/opendbc-checkout
git checkout elantra
# edit opendbc/car/hyundai/fingerprints.py, add your FW under CAR.HYUNDAI_ELANTRA_2024
git diff > /path/to/openpilot/.elantra/local-extras.patch
```

Commit that file to `master`. Every sync applies it on top of the recomputed port delta, so
your firmware survives every rebuild.

## Credentials

Two SSH deploy keys, each write-scoped to exactly one repository -- narrower than a personal
access token, which would carry account-wide `repo` scope. Both private keys live only as
Actions secrets; no copy was kept on disk.

| Secret | Deploy key on | Why |
|---|---|---|
| `OPENDBC_DEPLOY_KEY` | `aaravsaianugula/opendbc` | `GITHUB_TOKEN` is scoped to this repo and cannot reach another one |
| `OPENPILOT_DEPLOY_KEY` | `aaravsaianugula/openpilot` | `GITHUB_TOKEN` can *never* push a commit touching `.github/workflows`, and mirroring upstream changes workflow files constantly. There is no permissions flag for this |

To rotate either: generate a new ed25519 keypair, add the public half as a write-enabled
deploy key on that repo, `gh secret set <NAME>` the private half, delete the old deploy key.

## Working on this locally

`master` is force-pushed on every sync, so a local branch goes stale the moment the job
runs. Before editing anything:

```bash
git fetch fork master && git checkout -B master fork/master
```

Then edit, commit, push. The next sync derives its overlay from whatever is on `master`, so
committed changes to the overlay files carry forward automatically -- there is no separate
patch to keep in step.

Note the repo's LFS lives on GitLab and we have no write access to it. Push with
`--no-verify` to skip the pre-push hook; the overlay adds no LFS-tracked files, so there is
genuinely nothing for it to upload.

## Running it by hand

```bash
python .elantra/sync.py --dry-run            # build and validate, publish nothing
python .elantra/sync.py                      # build, validate, publish
python .elantra/sync.py --ref <sha>          # pin an upstream commit
python .elantra/guards.py --opendbc <path>   # just the structural checks
python .elantra/verify_published.py          # check what is live on GitHub
```

## What is in this directory

Nothing here ships to the car. `.elantra/` is the overlay's own tooling: the guards that hold the
port's invariants, the scanners that read recorded drives, and the tests for both. Three things
decide where a tool can run, and they trip people up:

- **The C safety suite needs a compiler**, and this laptop has none. Device only.
- **`opendbc/car/tests/*` collapse on Windows** — `get_interface_attr` splits `os.walk` output on
  `/`, which never matches a backslash path, so it discovers zero brands. Upstream bug, not a
  regression. Device only.
- **Anything reading recorded drives needs `/data/media/0/realdata`,** so the scanners run on the
  device even though they are pure Python.

Everything else runs on the laptop with:

```bash
export PYTHONPATH="D:/Coding/opendbc-elantra-wt-lateral;D:/Coding/sunnypilot-elantra-wt-lateral"
```

### Guards and sync — the invariants

| Tool | What it does | Runs on |
|---|---|---|
| `guards.py` | 97 structural guards over the port: the flag chain, the literal 409, the dbc/panda 0x485 pair, the opendbc pin. Text/AST level, no opendbc import. `--opendbc <path>` required; add `--repo <path>` for the superproject guards, or they silently skip | laptop |
| `sync.py` | Rebuilds the branch from sunnypilot master with the port on top. The branch is **rebuilt, never merged** | laptop |
| `verify_published.py` | Verifies the branch actually on GitHub, not the one built locally | laptop |
| `replay_safety.py` | Replays recorded drives through the **compiled** safety and proves panda accepts every frame | device |

### Reading the drives

| Tool | Answers | Runs on |
|---|---|---|
| `lateral_report.py` | Per-route, per-speed-band authority report: pinned %, demand vs delivered, every MDPS12 fault bit, provenance. Two independent estimators that must agree | device |
| `lateral_watch.sh` | Runs the above from a systemd timer after every drive, offroad only. **A crontab does not work — `/var` is tmpfs and is wiped on boot** | device |
| `eps_census.py` | *What ceiling did each route actually run, and did the EPS fault?* Recovers the compiled `STEER_MAX` per route and carries every fault channel | device |
| `demand_decomp.py` | *Where does the low-speed demand go?* Walks planner → controller → counts → delivered, frame by frame | device |
| `ceiling_replay.py` | *Would a higher ceiling reach the car, or does the slew rate eat it first?* Replays the recorded torque under a grid of (ceiling, rate) | device |
| `torque_projection.py` | Prices a ceiling change as a one-step counterfactual against real drives. **Refuses any candidate above 409** | device |
| `render_headroom_bar.py` | Renders the headroom arc through a scripted sweep, offscreen, to PNGs | laptop |

Two things about the scanners are load-bearing, and both were learned the hard way:

- **MDPS12 `0x251` must be pinned to `src 0`.** The same address also arrives on `src 1` carrying
  something else entirely — 599 of its frames per segment have bit 14 set. Accepting any
  `src < 128` invented 599 phantom EPS faults per segment against a true count of zero.
- **A route's build label proves nothing.** Route `000000bd` was called the schedule build and
  peaked at 337 counts, never reaching its ceiling. Identify a build from the drive's own
  **pinned opendbc gitlink** (`git ls-tree <openpilot-commit> opendbc_repo`), and confirm it
  against `eps_census.py`'s recovered ceiling — which works because `STEER_MAX` is a gain, so
  `torqueOutputCan / actuators.torque` returns it even on a gentle drive.

### Tests

Run these as **scripts, not under pytest** — they are `main()`-style harnesses that exit non-zero
on failure, and `pytest` collects nothing from them and reports "no tests ran" while exiting 5.
Only `test_torque_projection.py` is a real pytest module.

| Test | Covers |
|---|---|
| `test_lateral_report.py` | Decoding, the `src 0` bus filter, provenance tagging, and the scan loop's fault accounting driven through a fake `LogReader` |
| `test_scanner_decoders.py` | That all four scanners decode the same bytes the same way. They each carry their own copy of the CAN layouts because they deploy separately, so this is the only thing stopping them drifting apart |
| `test_guard_torque_chain.py` | That every single-link break in the torque chain is caught. `--opendbc <path>` required |
| `test_guard_opendbc_pin.py` | That the pin guard fails on every divergence it is meant to catch |
| `test_steer_headroom.py` | The headroom indicator's decision logic, 72 cases |
| `test_torque_projection.py` | The projection tool, including its refusal to price anything above 409 (**pytest**) |
| `test_port_panel.py` | The port panel's decision logic |

```bash
# the whole laptop-side gate, exit codes intact -- never pipe these through `tail`,
# which returns ITS exit code and turns a red gate green
python .elantra/guards.py --opendbc D:/Coding/opendbc-elantra-wt-lateral --repo .
python .elantra/test_guard_torque_chain.py --opendbc D:/Coding/opendbc-elantra-wt-lateral
python .elantra/test_lateral_report.py
python .elantra/test_scanner_decoders.py
python .elantra/test_steer_headroom.py
python .elantra/test_guard_opendbc_pin.py
python -m pytest .elantra/test_torque_projection.py -q
python -m ruff check .elantra/
```


## Known limits

- **comma's stock big model needs a Chestnut prebuilt branch, which this isn't.**
  `CHESTNUT_BRANCHES` in `common/version.py` maps only `staging`/`dev`/`release-*`, so master
  gets no nag alert and no `big_driving_supercombo.onnx`. eGPU big models come from
  sunnypilot's runtime model catalog instead — the RDF/POP/CD210 bundles the port's users
  actually run. That path is branch-independent and works here.
- **The 0x485 widening applies to every Hyundai, not just the Elantra.** That is why the port
  is not upstream. It is inherent to the port, not something this branch introduces.
- **Green CI is not a road test.** sunnypilot master is a development branch. Weekly plus a
  CI gate is a much better filter than the community branch has, but `master-previous` exists
  because it is still a filter and not a guarantee.
