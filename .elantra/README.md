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

Almost nothing lives in this repo. The entire Elantra port is 16 files in **opendbc**:

| File | What it does |
|---|---|
| `car/hyundai/values.py` | `HYUNDAI_ELANTRA_2024` (`CHECKSUM_CRC8 \| CAMERA_SCC \| RAISED_LIMITS \| LFAHDA_MFC_8`) and `HYUNDAI_ELANTRA_HEV_2024` (same minus `RAISED_LIMITS`, plus `HYBRID`), harness Hyundai K; both flag words; the 409 ceiling |
| `car/hyundai/interface.py` | Bridges `RAISED_LIMITS` and `LFAHDA_MFC_8` into `safetyParam` bits — the only path by which panda, which cannot see `carFingerprint`, learns either |
| `car/hyundai/fingerprints.py` | ECU firmware fingerprints |
| `car/hyundai/hyundaican.py` | Both platforms added to the LKAS11 LDWS mode list |
| `dbc/generator/hyundai/_hyundai_can_common.dbc` | The shared Hyundai CAN body, extracted verbatim so the CN7 can import it instead of forking all 1,698 lines |
| `dbc/generator/hyundai/hyundai_can.dbc` | Now an import of the above plus `LFAHDA_MFC` at its **unchanged 4 bytes** — every other Hyundai CAN platform is untouched |
| `dbc/generator/hyundai/hyundai_can_cn7.dbc` | The same import plus `LFAHDA_MFC` at **8 bytes**, which is what the CN7 bus carries |
| `safety/modes/hyundai.h` | The 0x485 TX allow-list carries **both** lengths (it matches on exact length, so one entry cannot admit both), `hyundai_tx_hook` narrows it back per platform, and `HYUNDAI_STEERING_LIMITS_RAISED = HYUNDAI_LIMITS(512, 3, 7)` — **512, not 409**: panda is deliberately permissive and the 103-count gap is unenforced |
| `safety/modes/hyundai_common.h` | `HYUNDAI_PARAM_RAISED_LIMITS = 1024`, `HYUNDAI_PARAM_LFAHDA_MFC_8 = 2048`, and the two flags panda selects on |
| `car/torque_data/substitute.toml` | Borrowed torque parameters |
| `sunnypilot/car/car_list.json` | The three entries in sunnypilot's car list (CI byte-compares this against the generator) |
| `car/tests/routes.py` | CI test routes |
| `car/hyundai/tests/test_hyundai.py` | The ceiling reaching the wire, the flag chain, the 0x485 split |
| `safety/tests/test_hyundai.py` | `TestHyundaiSafetyRaisedLimits`, and the fix for four classes pytest silently skipped |
| `safety/tests/common.py` | The standstill probe in m/s, and 0x485 out of the cross-mode TX probe |
| `docs/CARS.md` | Generated; the three new rows |

**The steering torque ceiling is a flat 409 counts at every speed** for `HYUNDAI_ELANTRA_2024`
only, against the 384-count HKG default. It is not a speed schedule; there is no lookup table
and no interpolation. `STEER_MAX` is a *gain*, not just a ceiling — every command is multiplied
by it — so 409 raises every command by 6.51 %, at every speed.

**409 is the MDPS's measured acceptance limit, not a preference.** It accepts 409 and trips
`CF_Mdps_ToiFlt` at 410: 19 fault onsets at commanded counts 410 through 433, against ~1.59M
engaged frames at 409 or below with none at all. Every onset fired below 8.6 m/s under a
ceiling that was constant at those speeds, so the boundary is the value and not any mechanism.
**Do not raise it, and do not test a raise on the road** — the experiment has already run.
Evidence and tables in `ROAD-TEST-cn7-lateral.md`; the scanner is `eps_census.py`.

**Raising it further would not fix what the car is actually short of.** With a flat 409
commanded, the median that reaches the EPS at 3–7 m/s is 236 counts. `STEER_DELTA_UP = 3` at
100 Hz takes 1.36 s to ramp 0 → 409, and panda's `max_rt_delta = 112` per 250 ms caps any
increase at `STEER_DELTA_UP = 4` (1.03 s) before panda starts rejecting frames outright. The
ramp and the lateral tune are the binding constraints down there, not the ceiling.

**panda does not police this.** It enforces a flat 512, so a build that commands 450 sails
through and the EPS is what stops the car. The guards therefore assert the **literal** 409 —
`opendbc <= panda` is vacuous here, since 450 satisfies it and still breaks steering.

**The `hyundai_can_cn7.dbc` / `hyundai.h` pair is the safety-critical part.** The dbc says how
many bytes openpilot packs into 0x485; the safety header says how many panda will let out.
If they ever disagree the car either refuses to steer or panda's allow-list is wider than the
message it guards. `guards.py` asserts three things together — the CN7 dbc is 8, the shared dbc
is still 4, and panda carries both with `hyundai_tx_hook` tying length to the flag — and the
sync aborts if they diverge.

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
- `openpilot/selfdrive/ui/sunnypilot/mici/onroad/steer_headroom.py` — the headroom arc's
  decision logic, pure: no pyray, no cereal, an injected clock
- `openpilot/selfdrive/ui/sunnypilot/mici/onroad/steer_headroom_bar.py` — the widget itself,
  a subclass of upstream's `TorqueBar`

*Upstream files we touch — 31 added lines and 1 removed across five files, deliberately tiny,
since this is the only conflict surface in the design:*
- `.gitmodules` — opendbc URL → our fork
- `opendbc_repo` — the gitlink
- `openpilot/common/params_keys.h` — three params
- `openpilot/system/updated/updated.py` — publishes the build manifest (read-only; no new
  logic in the update or finalize path, because that path is what bricks devices)
- `openpilot/selfdrive/ui/sunnypilot/mici/layouts/settings.py` — registers the panel
- `openpilot/selfdrive/ui/sunnypilot/mici/onroad/hud_renderer.py` — four lines, swapping in
  `SteerHeadroomBar`, which delegates straight back to upstream on any car without the flag

The overlay diff is **derived from the previous build** rather than stored as a static patch,
so editing any of those files by hand and committing is all it takes — the next sync picks
the change up automatically.

## The weekly sync

`.github/workflows/elantra-sync.yaml`, Mondays 09:00 UTC, plus **Run workflow** for an
off-cycle sync.

1. Walk sunnypilot master newest-first and take the first commit whose **own CI went green**.
   sunnypilot reports through check-runs, so that is what gets queried — the combined status
   endpoint returns `pending` forever and would make the gate a no-op.
2. Rebuild `aaravsaianugula/opendbc:master` = upstream opendbc master + our own port delta,
   recomputed on every run from the previously published `aaravsaianugula/opendbc:master`, so
   upstream fixes arrive automatically and a fix committed there by hand is picked up next run.
   Four checks abort before anything is built: our master having no delta at all, an empty
   diff, a path outside `OPENDBC_DELTA_PATHS`, or a missing `REQUIRED_IN_DELTA` token.
3. Reset master to the chosen commit, replay the overlay, pin the gitlink, write the manifest.
4. Run `guards.py`. Run opendbc's own Hyundai tests, including its Elantra ones.
5. Push opendbc (`master-previous` first, then `master`), then the superproject the same way.
   opendbc lands first, or the gitlink does not resolve and the device cannot clone it.
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
export PYTHONPATH="D:/Coding/opendbc-elantra-wt-cn7;D:/Coding/sunnypilot-elantra-wt-cn7"
```

### Guards and sync — the invariants

| Tool | What it does | Runs on |
|---|---|---|
| `guards.py` | Structural guards over the port: the flag chain, the literal 409, the dbc/panda 0x485 pair, the opendbc pin. The run prints the count (97 with `--repo`, 67 without) — read it there rather than from this table, which will rot. Text/AST level, no opendbc import. `--opendbc <path>` required; add `--repo <path>` for the superproject guards, or they silently skip | laptop |
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

Two invariants the scanners depend on:

- **MDPS12 `0x251` must be pinned to `src 0`.** The same address also arrives on `src 1`
  carrying something else entirely, and 599 of its frames per segment have bit 14 set.
  Accepting any `src < 128` invents 599 phantom EPS faults per segment.
- **Identify a build by its pinned opendbc gitlink, never by a route label.** A label records
  what someone meant to drive; the gitlink records what the binary was
  (`git ls-tree <openpilot-commit> opendbc_repo`). Confirm it against `eps_census.py`'s
  recovered ceiling, which works because `STEER_MAX` is a gain, so
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
python .elantra/guards.py --opendbc D:/Coding/opendbc-elantra-wt-cn7 --repo .
python .elantra/test_guard_torque_chain.py --opendbc D:/Coding/opendbc-elantra-wt-cn7
python .elantra/test_lateral_report.py
python .elantra/test_scanner_decoders.py
python .elantra/test_steer_headroom.py
python .elantra/test_guard_opendbc_pin.py
python -m pytest .elantra/test_torque_projection.py -q
python -m ruff check .elantra/
```


## Rules that hold

Each of these cost something to learn. They are stated as rules because that is all that is
worth carrying forward.

1. **Pin CAN scans to `src 0`.** See above.
2. **Identify a build by its opendbc gitlink**, never by a route label or by `safetyParam`.
3. **Assert the literal 409.** A relational check against panda's 512 is vacuous: 450 passes
   it and stops the car steering.
4. **Keep `HYUNDAI_STEERING_LIMITS_RAISED` declared at the start of a line.** Both parsers of
   it are line-anchored. An unanchored read takes the first match in the file, so a stale
   number in the comment above would be read instead — and it fails *open*.
5. **The device's `/data/elantra-lateral/lateral_report.py` is a copy and does not follow the
   repo.** Freshness-check it before trusting a report.
6. **`CF_Mdps_FailStat` is not the fault signature.** It fires on clean routes at high steering
   angle with driver input. Use ToiFlt-while-steering and `steerTempUnavailable`.
7. **A `ToiActive` dropout is not automatically a fault.** openpilot itself cuts
   `CF_Lkas_ActToi` for 2 frames in every 89 above 85°, and the MDPS answers by clearing it.
8. **Never pipe a gate through `tail`**, which returns its own exit code. And never hard-code
   a count the tool prints.
9. **Before rebuilding or force-pushing a branch, tag anything the docs cite.** The rollback
   table in `ROAD-TEST-cn7-lateral.md` names opendbc commits by sha; once a branch is rewritten
   those are reachable only from a tag. `archive/pre-rebuild-elantra-lateral` exists for this
   reason on both repos.

## Known limits

- **comma's stock big model needs a Chestnut prebuilt branch, which this isn't.**
  `CHESTNUT_BRANCHES` in `common/version.py` maps only `staging`/`dev`/`release-*`, so master
  gets no nag alert and no `big_driving_supercombo.onnx`. eGPU big models come from
  sunnypilot's runtime model catalog instead — the RDF/POP/CD210 bundles the port's users
  actually run. That path is branch-independent and works here.
- **The raised ceiling is this car, measured once.** 409 is where one CN7's MDPS stops
  accepting, on one firmware. carrotpilot ships 409 for all HKG but has no 2024-25 CN7
  platform at all — its CN7 is the pre-facelift `HYUNDAI_ELANTRA_2021` — so the fleet evidence
  behind the number is the same generation, not the same car.
- **Green CI is not a road test.** sunnypilot master is a development branch. Weekly plus a
  CI gate is a much better filter than the community branch has, but `master-previous` exists
  because it is still a filter and not a guarantee.
