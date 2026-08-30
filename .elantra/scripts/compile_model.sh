#!/bin/bash
# Compile the big driving model for the eGPU, with the kernel audit and the full DEBUG=2 log kept.
#
# Replaces the ad-hoc /root/*_wsl.sh scripts. Two things it does that they did not:
#
#   1. KERNEL_AUDIT. beam_search_22 is keyed on an opaque ast.key hash and the compile cache on IR
#      text, so neither says which kernel a row belongs to. Without the join you can see that a
#      kernel shipped with a one-thread workgroup and you cannot find the row that decided it.
#   2. The DEBUG=2 output is kept. The per-dispatch '*** AMD' list is the only per-kernel timing
#      this project has, and it has been thrown away twice.
#
# BEAM_DEV_TIMEOUT=0: tinygrad treats ANY device-side timeout as a hang -- it sets error_state and
# calls recover(), which on this Navi 23 dies with KeyError: regBIF_BX_PF0_RSMU_INDEX and takes the
# run with it. A deadline therefore converts merely-slow candidates into fatal ones. early_stop
# still prunes on the host side.
#
# AM_POWER_LIMIT=100 is a deliberately conservative cap and is not binding: metered at it the dock
# peaks at 85.8 W on the 12 V rail with no converter faults, and the model draws 30-37 W. The car's
# own limit is 180 W (2024 Elantra owner's manual, 20 A socket fuse).
#
#   usage: compile_model.sh <tag>
#   env:   any BEAM_*/AMD_*/PCONTIG override is honoured; they are echoed into the log so an
#          arm can never be confused with another after the fact.
set -eu

TAG="${1:?usage: compile_model.sh <tag>   e.g. compile_model.sh audit0}"
OUT="/root/models/big_driving_gfx1032_${TAG}.pkl"
LOG="/root/runs/${TAG}.compile.log"
AUDIT="/root/runs/${TAG}.audit.jsonl"
mkdir -p /root/runs
rm -f "$AUDIT"

export PYTHONPATH=/root/src/tinygrad_repo:/root/src:/root/deps/opendbc_repo
export XDG_CACHE_HOME=/root/tgcache

# A schedule experiment must not overwrite the working cache. BEAM_UPCAST_MAX and
# BEAM_LOCAL_MAX are NOT part of the beam_search key, so a wider search silently replaces
# the rows that produced the current best build, and if the wider schedules turn out
# slower there is nothing left to fall back to. Point CACHEDB at a copy instead -- the
# compiled-ELF rows come along with it, so compiles stay fast.
if [ -n "${CACHEDB:-}" ]; then
  export CACHEDB
  if [ ! -f "$CACHEDB" ]; then
    mkdir -p "$(dirname "$CACHEDB")"
    echo "seeding $CACHEDB from /root/tgcache/tinygrad/cache.db"
    /root/tgvenv/bin/python - "$CACHEDB" <<'PY'
import sqlite3, sys
# .backup rather than cp: a torn copy of a WAL database is worse than no copy.
src = sqlite3.connect("file:/root/tgcache/tinygrad/cache.db?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[1])
src.backup(dst)
print("  seeded, beam rows:", dst.execute("select count(*) from beam_search_22").fetchone()[0])
dst.close(); src.close()
PY
  fi
fi
export DEV=USB+AMD:LLVM WARP_DEV=CPU FLOAT16=1 JIT_BATCH_SIZE=0 GMMU=0 TC_OPT=2
export DEBUG=2
export BEAM=${BEAM:-2} IGNORE_BEAM_CACHE=${IGNORE_BEAM_CACHE:-0} BEAM_MAX_TASKS_PER_CHILD=200
export BEAM_DEV_TIMEOUT=${BEAM_DEV_TIMEOUT:-0}
export BEAM_LOCAL_MAX=${BEAM_LOCAL_MAX:-256} BEAM_UPCAST_MAX=${BEAM_UPCAST_MAX:-64}
export HCQDEV_WAIT_TIMEOUT_MS=${HCQDEV_WAIT_TIMEOUT_MS:-20000}
export AM_POWER_LIMIT=${AM_POWER_LIMIT:-100}
export KERNEL_AUDIT="$AUDIT"

# A killed process does not drop its libusb claim instantly, and slot_cycle opens the bridge with
# set_configuration, which fails "Resource busy" against a stale claim.
for attempt in $(seq 1 10); do
  if /root/tgvenv/bin/python -u /root/slot_cycle.py; then break; fi
  echo "slot cycle attempt $attempt failed (device still claimed), retrying"
  sleep 6
done
sleep 5

{
  echo "=== compile_model.sh tag=$TAG  $(date -Is) ==="
  # The beam cache key contains none of these, so an arm that changes one silently reuses the
  # previous arm's schedule. Recording them is the only way to tell two runs apart afterwards.
  for v in BEAM BEAM_PADTO BEAM_LOCAL_MAX BEAM_UPCAST_MAX BEAM_DEV_TIMEOUT IGNORE_BEAM_CACHE CACHEDB \
           AMD_FMA_MIX AMD_DOT2 AMD_ACC_SEED AMD_WGP_MODE AMD_ELIDE_FLUSH LLVM_ZEXT_INDEX PCONTIG FLOAT16 JIT_BATCH_SIZE GMMU TC_OPT NOOPT; do
    echo "env  $v=${!v-<unset>}"
  done
  echo "out  $OUT"
} | tee "$LOG"

set +e
/root/tgvenv/bin/python -u /root/src/openpilot/selfdrive/modeld/compile_modeld.py \
  --model-size 512x256 \
  --camera-resolutions 1928x1208 1344x760 \
  --onnx /root/models/big_driving_supercombo.onnx \
  --output "$OUT" \
  --frame-skip 4 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "compile exit=$rc   log=$LOG   audit=$AUDIT   ($(wc -l < "$AUDIT" 2>/dev/null || echo 0) audited kernels)"
exit "$rc"
