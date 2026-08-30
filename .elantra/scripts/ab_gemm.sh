#!/bin/bash
# A/B a codegen change on the kernel shape that carries most of the model, on the real card.
#
# This is the fast loop. A full model compile is ~15 minutes; this is about a minute per run and
# exercises the GEMM shape .elantra/gemm_bench.py isolates (128 x 1536 x 6144) -- ~22 of the model's
# 93 AMD kernels do exactly that many multiply-accumulates.
#
# Why a codegen A/B is safe against the beam cache, when a *schedule* A/B is not: the beam key is
# {ast, amt, allow_test_size, device, suffix}, and a renderer change does not touch the AST, so both
# arms get an identical schedule and the only difference is the code generated from it. A knob that
# changes the schedule (BEAM_PADTO, PCONTIG, the BEAM caps) needs IGNORE_BEAM_CACHE=1 in the
# *environment* instead -- search.py binds it as a default argument at import, so Context() cannot.
#
# DEBUG=2 is mandatory and is not verbosity: it is what makes tinygrad time each dispatch on the
# device and populate GlobalCounters.time_sum_s. Without it gemm_bench prints 0.000 ms/GEMM and
# 0 GFLOPS while the kernels still run and the accuracy gate still passes -- a run that looks
# successful and measured nothing. That cost one whole A/B here.
#
# Three repeats minimum, and the range is reported rather than the mean: this rig has produced 1980
# and 1570 GFLOPS for the same cached schedule on two boots.
#
#   usage: ab_gemm.sh <reps> "ARM1_ENV=..." "ARM2_ENV=..." ...
#   e.g.   ab_gemm.sh 3 "AMD_DOT2=0" "AMD_DOT2=1"
set -u
REPS="${1:?usage: ab_gemm.sh <reps> \"ENV=v ...\" ...}"; shift
mkdir -p /root/runs/ab

for arm in "$@"; do
  tag=$(echo "$arm" | tr ' =' '__')
  for r in $(seq 1 "$REPS"); do
    log="/root/runs/ab/${tag}.r${r}.log"
    # A killed process does not drop its libusb claim instantly, and slot_cycle opens the bridge
    # with set_configuration, which fails "Resource busy" against a stale claim.
    for a in 1 2 3 4 5; do
      /root/tgvenv/bin/python -u /root/slot_cycle.py >/dev/null 2>&1 && break
      sleep 6
    done
    sleep 3
    env PYTHONPATH=/root/src/tinygrad_repo:/root/src \
        XDG_CACHE_HOME=/root/tgcache DEV=USB+AMD:LLVM FLOAT16=1 GMMU=0 DEBUG=2 \
        AM_POWER_LIMIT=100 BEAM=2 BEAM_DEV_TIMEOUT=0 \
        BEAM_LOCAL_MAX=256 BEAM_UPCAST_MAX=64 HCQDEV_WAIT_TIMEOUT_MS=20000 \
        $arm \
        /root/tgvenv/bin/python -u /root/src/.elantra/gemm_bench.py > "$log" 2>&1
    echo "  ran [$arm r$r] exit=$?"
  done
done

# Parsed in python, not sed: tinygrad writes ANSI colour codes into the middle of these lines and
# every shell attempt to strip them here has been more fragile than the measurement it reports.
/root/tgvenv/bin/python - "$@" <<'PY'
import glob, re, statistics, sys
ansi = re.compile(chr(27) + r"\[[0-9;]*m")
row = re.compile(r"(stage-3 fc\d)\s+([\d.]+) ms/GEMM\s+(\d+) GFLOPS\s+([\d.]+)% fp32 peak\s+([\d.]+)% packed")
print()
print("=== summary: ms/GEMM and GFLOPS per run ===")
for arm in sys.argv[1:]:
    tag = arm.replace(" ", "_").replace("=", "_")
    per = {}
    for f in sorted(glob.glob("/root/runs/ab/%s.r*.log" % tag)):
        for m in row.finditer(ansi.sub("", open(f, errors="replace").read())):
            per.setdefault(m.group(1), []).append((float(m.group(2)), int(m.group(3)), float(m.group(5))))
    print(arm)
    if not per:
        print("   no measurements parsed -- check DEBUG=2 and that the run reached the device")
    for shape, vals in per.items():
        g = [v[1] for v in vals]
        print("   %-12s GFLOPS %s   mean %.0f  spread %.0f%%   packed-peak %.1f%%"
              % (shape, " ".join(str(x) for x in g), statistics.mean(g),
                 100 * (max(g) - min(g)) / max(statistics.mean(g), 1), vals[0][2]))
PY
