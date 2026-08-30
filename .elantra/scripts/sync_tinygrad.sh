#!/bin/bash
# Push the repo's tinygrad into the WSL working copy that actually runs.
#
# Two trees exist and only one of them executes. The repo copy under D:/ is what git tracks and
# what gets reviewed; /root/src/tinygrad_repo is what every bench imports, because WSL2 reaches
# Windows drives over 9p where importing tinygrad measures 0.66 s against 0.10 s native. Editing
# the wrong one produces a measurement that does not correspond to any committed code, which has
# already happened on this project.
#
# One direction only, repo -> WSL. Nothing should ever be authored in /root.
set -eu
SRC=${SRC:-/mnt/d/Coding/sunnypilot-elantra/tinygrad_repo/tinygrad}
DST=${DST:-/root/src/tinygrad_repo/tinygrad}

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' "$SRC/" "$DST/"

# Stale bytecode from a previous sync outlives --delete when a .py is only edited, not removed.
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Prove it: a silent partial sync is the failure mode this guards against.
if diff -rq --exclude='__pycache__' "$SRC" "$DST" >/dev/null; then
  echo "tinygrad synced: $SRC -> $DST"
else
  echo "SYNC INCOMPLETE -- trees still differ:" >&2
  diff -rq --exclude='__pycache__' "$SRC" "$DST" >&2
  exit 1
fi
