# CLAUDE.md

This is a **derived** fork of sunnypilot. `master` is thrown away and rebuilt from upstream
every Monday 09:00 UTC by `.elantra/sync.py`. It is force-pushed, never merged.

## The standing rule

> **Every file this branch adds to, or changes in, upstream must be listed in
> `.elantra/overlay.py`. Anything not listed is deleted by the next sync — silently.**

Not a convention. The rebuild resets the tree to an upstream commit and restores only what
the registry names, so an unregistered file is not "left alone" — from the rebuild's point of
view it never existed. There is no error, no conflict, no warning. It is simply gone.

`.elantra/test_overlay_registration.py` enforces this, and CI runs it on every push to
`master` (`.github/workflows/elantra-guard.yaml`).

## Registering new code

1. Write the code.
2. Add its path to `.elantra/overlay.py`, **in the same commit**:
   - a file that is entirely ours (new module, asset, workflow) → `OVERLAY_ADDED`
   - an upstream file we edit → `OVERLAY_MODIFIED`, **and** a hook string in `OVERLAY_HOOKS`
     that proves our edit is still there. Use an identifier, never a formatted source line —
     upstream reflows code constantly and a whitespace-exact match fails for the wrong reason.
   - a submodule pin → `OVERLAY_GITLINKS`, and teach `sync.py` how to build it
3. `python .elantra/test_overlay_registration.py` — must pass before you push.
4. `git push --no-verify fork master` — LFS lives on GitLab and we have no write access, so
   `--no-verify` is required to skip the pre-push hook.

**Exception:** anything under `.elantra/` is already covered by the `.elantra` directory
entry. That is the one place where "you must register it" does not apply.

**This file is a special case.** Upstream sunnypilot's `.gitignore` ignores `CLAUDE.md`
(line 100), so it was committed with `git add -f` and a plain `git add -A` will not pick it
up if it is ever deleted and recreated. It stays tracked once tracked, and the weekly sync
restores it by explicit path, so this only matters if someone removes it. If
`test_overlay_registration.py` ever reports `CLAUDE.md` as a registered path that does not
exist, this is why.

Both halves matter. A file you forget to register vanishes next Monday. A path you register
that does not exist makes `restore_paths()` raise and blocks the sync **entirely** — nobody
gets a build until it is fixed. The second is louder, and therefore less dangerous.

## Rules of thumb

- **Prefer a new file to editing an upstream one.** `OVERLAY_ADDED` never conflicts;
  `OVERLAY_MODIFIED` is the entire conflict surface of this design. If a feature can be a new
  module plus a one-line import hook, make it so. The eGPU support is the worked example:
  seven new files, five changed lines in one upstream file.
- **Never edit `opendbc_repo/` or `tinygrad_repo/` in place.** Those are gitlinks. Changes go
  in the fork branches (`aaravsaianugula/opendbc:elantra`,
  `aaravsaianugula/tinygrad:nv-usb3`), or in `.elantra/local-extras.patch` for opendbc.
- **`master` is force-pushed weekly.** Always `git fetch fork master && git checkout -B master
  fork/master` before editing. A local branch older than Monday is already wrong.
- **Never `git push` without `--no-verify`.**

## Before you claim a change survives the sync

`.elantra/sync.py --dry-run` resolves `refs/remotes/fork/master` — **not your HEAD.** A dry
run against an unpushed commit rebuilds a branch that never had your file, and passes. That is
the most confidently wrong signal in this repo.

Do not point `--repo` at a checkout you care about either: the sync runs
`git checkout --force -B sync-work <target>` in it.

Use the rehearsal in `.elantra/README.md`, which clones to a throwaway whose `fork` remote is
your local repo.

## Layout

| Path | What |
|---|---|
| `.elantra/overlay.py` | the registry — single source of truth |
| `.elantra/sync.py` | the weekly rebuild |
| `.elantra/guards.py` | structural checks; nothing publishes if one fails |
| `.elantra/verify_published.py` | post-push check against GitHub |
| `.elantra/test_*.py` | plain scripts, not pytest. Run them directly. |
| `.elantra/EGPU.md` | the NVIDIA eGPU work, and what is still unproven |

## House style for `.elantra/`

Plain executable scripts, stdlib only, `main() -> int`, a `check()`/`case()` helper that
prints `ok`/`FAIL` per line. No pytest, no fixtures, no third-party imports — these have to
run on a bare Python 3 on the device and in CI.

Ruff runs over `.elantra/` with `ISC` enabled and `allow-multiline = false`, so build long
strings with explicit `+`, never implicit adjacent literals.
