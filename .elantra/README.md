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

Almost nothing lives in this repo. The entire Elantra port is 8 files in **opendbc**:

| File | What it does |
|---|---|
| `car/hyundai/values.py` | `HYUNDAI_ELANTRA_2024`, `HYUNDAI_ELANTRA_HEV_2024` — `CAMERA_SCC \| CHECKSUM_CRC8`, harness Hyundai K |
| `car/hyundai/fingerprints.py` | ECU firmware fingerprints |
| `car/hyundai/hyundaican.py` | Both platforms added to the LKAS11 LDWS mode list |
| `dbc/generator/hyundai/hyundai_can.dbc` | `LFAHDA_MFC` (0x485) 4 → 8 bytes |
| `safety/modes/hyundai.h` | Panda TX allow-list for 0x485, 4 → 8 bytes |
| `car/torque_data/substitute.toml` | Borrowed torque parameters |
| `sunnypilot/car/car_list.json` | The three entries in sunnypilot's car list |
| `car/tests/routes.py` | CI test routes |

**The last two rows of the dbc/safety pair are the safety-critical part.** The dbc says how
many bytes openpilot packs into 0x485; the safety header says how many panda will let out.
If they ever disagree the car either refuses to steer or panda's allow-list is wider than the
message it guards. `guards.py` asserts them together, and the sync aborts if they diverge.

This is also why the branch is built from source rather than from a prebuilt sunnypilot
release: prebuilt branches ship a panda binary compiled against stock opendbc, which would
reject the 8-byte frame.

## The overlay

Everything this branch adds on top of upstream, and nothing more:

*Files that are entirely ours — restored wholesale each sync, so they never conflict:*
- `.elantra/` (this directory: sync, guards, verify, tests, manifest)
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

## Custom code must survive the weekly sync

The sync rebuilds `master` by hard-resetting to an upstream commit and putting back only what
`.elantra/overlay.py` names. **A file that is not in the registry is not "left alone" — it is
deleted**, with no error, because from the rebuild's point of view it never existed. This is
the price of a derived branch, and it is the one thing about the design that surprises people.

The registry has three lists:

| List | For | Restored how | Conflicts? |
|---|---|---|---|
| `OVERLAY_ADDED` | files that are wholly ours | copied wholesale from the previous build | never |
| `OVERLAY_MODIFIED` | upstream files we edit | three-way replay of a diff derived from the previous build | yes — this is the entire conflict surface |
| `OVERLAY_GITLINKS` | `opendbc_repo`, `tinygrad_repo` | re-pinned with `git update-index --cacheinfo` | no, but see below |

**The procedure.** Add the file → add its path to the right list *in the same commit* → run
`python .elantra/test_overlay_registration.py` → push. Anything under `.elantra/` is already
covered by the directory entry and needs no separate registration.

**The two ways to get it wrong**, and what each looks like:

- **Unregistered.** The file is gone next Monday. Nothing fails, nothing warns; you find out
  when something imports it on the car and the settings screen dies.
- **Registered but absent.** `restore_paths()` refuses and the sync aborts **entirely** — no
  build for anyone until it is fixed. Louder, and therefore the safer of the two.

Both are caught by `test_overlay_registration.py`, which CI runs on every push to `master`
(`.github/workflows/elantra-guard.yaml`) rather than six days later.

Note `CLAUDE.md` is a special case: upstream's `.gitignore` ignores it, so it was committed
with `git add -f`.

### tinygrad and the eGPU

sunnypilot pins a snapshot of its own tinygrad fork. We replay a five-file NV-USB delta from
`aaravsaianugula/tinygrad:nv-usb3` onto whatever it now pins, and re-pin the gitlink — the
same shape as the opendbc rebuild, so rebases of tinygrad PR #17369 flow through
automatically.

**This delta will stop applying.** It is an unmerged PR against a fast-moving tree, not a
stable API. When that happens the sync aborts with `master` untouched, opens an issue, and
prints the refresh recipe. The car keeps its last good build, eGPU and all.

The branch invariant matters: `nv-usb3` must be **exactly one non-merge commit on top of a
commit that exists in `sunnypilot/tinygrad`**. sunnypilot's tinygrad and upstream tinygrad are
separate lineages, and a merge-base computed across them would drag weeks of unrelated
tinygrad churn into the "delta" — a failure that *succeeds*, which is worse than a conflict.
`build_tinygrad()` checks this explicitly and refuses rather than trusting the diff to be small.

See `.elantra/EGPU.md` for what is proven and what is not.

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

## Known limits

- **comma's stock big model needs a Chestnut prebuilt branch, which this isn't.**
  `CHESTNUT_BRANCHES` in `common/version.py` maps only `staging`/`dev`/`release-*`, so master
  gets no nag alert and no `big_driving_supercombo.onnx`. eGPU big models come from
  sunnypilot's runtime model catalog instead — the RDF/POP/CD210 bundles the port's users
  actually run. That path is branch-independent and works here.
- **The 0x485 widening applies to every Hyundai, not just the Elantra.** That is why the port
  is not upstream. It is inherent to the port, not something this branch introduces.
- **The eGPU patch is unmerged upstream.** Expect to refresh `nv-usb3` every few weeks. Until
  you do, the branch stays on its last good build rather than publishing one without eGPU
  support.
- **NVIDIA eGPU support is unproven on hardware.** The vendor abstraction, guards and tests
  are real and green; whether a 3080 Ti actually runs a model over the chestnut's USB3 bridge
  has not been tested by anyone on a comma four. See `.elantra/EGPU.md`.
- **Green CI is not a road test.** sunnypilot master is a development branch. Weekly plus a
  CI gate is a much better filter than the community branch has, but `master-previous` exists
  because it is still a filter and not a guarantee.
