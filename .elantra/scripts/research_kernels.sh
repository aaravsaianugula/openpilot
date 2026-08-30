#!/bin/bash
# Re-search a chosen set of kernels under new BEAM settings, without disturbing the working cache.
#
# The blunt way to change a BEAM knob is IGNORE_BEAM_CACHE=1, which re-searches all 93 kernels: one
# to two hours, and every one of them is another chance for the device to hang mid-search. It is
# also indiscriminate -- kernels that are already fine get re-rolled, and a slower result for one of
# them can quietly cancel out the win you were testing for.
#
# This is the targeted form. Seed a copy of the cache, delete the beam rows for exactly the kernels
# under test (found by ast_key, which only exists as a joinable value because KERNEL_AUDIT records
# it), then compile normally with the cache ON. Every other kernel is a cache hit and keeps its
# known-good schedule; the ones named here are the only thing that moves.
#
# The knobs worth re-searching for are the ones that are NOT in the beam key -- BEAM_UPCAST_MAX,
# BEAM_LOCAL_MAX, BEAM_PADTO -- which is exactly why they cannot be A/B'd by just setting them.
#
#   usage: research_kernels.sh <tag> <audit.jsonl> <kernel-name> [kernel-name ...]
#   env:   BEAM_UPCAST_MAX / BEAM_LOCAL_MAX / BEAM_PADTO etc. are passed through to the compile.
set -eu

TAG="${1:?usage: research_kernels.sh <tag> <audit.jsonl> <kernel> [kernel...]}"; shift
AUDIT="${1:?need the audit jsonl that names these kernels}"; shift
[ "$#" -gt 0 ] || { echo "name at least one kernel to re-search" >&2; exit 2; }

export CACHEDB="/root/tgcache_${TAG}/cache.db"
mkdir -p "$(dirname "$CACHEDB")"

if [ ! -f "$CACHEDB" ]; then
  echo "seeding $CACHEDB"
  /root/tgvenv/bin/python - "$CACHEDB" <<'PY'
import sqlite3, sys
# .backup, not cp: the source is a WAL database and a torn copy is worse than no copy.
src = sqlite3.connect("file:/root/tgcache/tinygrad/cache.db?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[1]); src.backup(dst)
print("  beam rows carried over:", dst.execute("select count(*) from beam_search_22").fetchone()[0])
dst.close(); src.close()
PY
fi

/root/tgvenv/bin/python - "$CACHEDB" "$AUDIT" "$@" <<'PY'
import json, sqlite3, sys
db, audit, names = sys.argv[1], sys.argv[2], set(sys.argv[3:])
keys = {}
for line in open(audit):
    r = json.loads(line)
    if r["name"] in names and r.get("beam") and r["beam"].get("device") == "AMD":
        keys[r["name"]] = r["beam"]["ast_key"]
missing = names - set(keys)
if missing:
    print("  NOT FOUND in the audit (will not be re-searched):", ", ".join(sorted(missing)))
con = sqlite3.connect(db)
gone = 0
for n, k in sorted(keys.items()):
    cur = con.execute("delete from beam_search_22 where lower(hex(ast)) = ?", (k.lower(),))
    print("  cleared %-42s %s (%d row)" % (n, k[:16], cur.rowcount))
    gone += cur.rowcount
con.commit()
left = con.execute("select count(*) from beam_search_22").fetchone()[0]
print("  %d rows cleared, %d kept -- only the cleared ones will re-search" % (gone, left))
con.close()
if gone == 0:
    sys.exit("nothing was cleared; the compile would just hit the cache and measure nothing")
PY

echo
exec /mnt/d/Coding/sunnypilot-elantra/.elantra/scripts/compile_model.sh "$TAG"
