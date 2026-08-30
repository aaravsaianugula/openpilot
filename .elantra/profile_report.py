#!/usr/bin/env python3
"""Rank the model's kernels by measured time, annotated with the schedule that produced them.

Joins two things that have never been joined on this project:

  * a DEBUG=2 compile log -- one '*** AMD' line per dispatch during the eager pass, which is the
    only per-kernel timing that exists, and
  * a KERNEL_AUDIT jsonl -- the kernel-name <-> beam-key <-> local_size map that neither cache
    table records.

Two traps this handles, both of which have already produced wrong conclusions here:

  1. tinygrad writes ANSI colour codes *inside* kernel names, so `grep r_8_16_1536_7_7 profile.log`
     returns nothing while the kernel is right there. Everything is stripped before matching.
  2. The eager per-dispatch times sum to several times the graphed frame, because each eager
     dispatch pays a USB submit-and-wait that the captured graph does not. The *ranking* is
     meaningful; the absolute microseconds are not the frame. Both are printed, with the ratio, so
     the difference cannot be mistaken for a result.

tinygrad's own GFLOPS column is NOT trustworthy on a BEAM-tuned build -- `estimates.ops` is
counted before the upcast expansion, so a kernel with 4x upcast reports a quarter of its real
arithmetic. Use .elantra/model_flops.py for the model's true FLOP count.

  usage: profile_report.py <compile.log> [audit.jsonl]
"""
import json
import re
import sys
from collections import defaultdict

ANSI = re.compile(chr(27) + r"\[[0-9;]*m")
LINE = re.compile(
    r"\*\*\* (\w+)\s+(\d+)\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+ GB\s+tm\s+([\d.]+)(us|ms)"
    r"(?:/\s*[\d.]+ms\s+\(\s*([\d.]+)\s+GFLOPS\s+(\d+)\|(\d+)\s+GB/s)?")


def parse_log(path):
  dispatches, batched = [], []
  for raw in open(path, errors="replace"):
    if "***" not in raw:
      continue
    line = ANSI.sub("", raw).rstrip()
    m = LINE.search(line)
    if not m:
      continue
    dev, idx, name, tm, unit, gflops, membw, ldsbw = m.groups()
    tm = float(tm) * (1000.0 if unit == "ms" else 1.0)
    name = name.strip()
    rec = (dev, int(idx), name, tm, float(gflops or 0), float(membw or 0), float(ldsbw or 0))
    (batched if name.startswith("batched") else dispatches).append(rec)
  return dispatches, batched


def main():
  if len(sys.argv) < 2:
    print(__doc__)
    return 2
  log = sys.argv[1]
  audit_path = sys.argv[2] if len(sys.argv) > 2 else None

  dispatches, batched = parse_log(log)
  sched = {}
  if audit_path:
    for l in open(audit_path):
      r = json.loads(l)
      sched[r["name"]] = r

  kernels = [d for d in dispatches if d[0] == "AMD" and not d[2].startswith("copy")]
  copies = [d for d in dispatches if d[0] == "AMD" and d[2].startswith("copy")]
  if not kernels:
    print(f"no AMD kernel dispatches found in {log}")
    return 1
  total = sum(k[3] for k in kernels)

  print(f"log: {log}")
  print(f"  {len(kernels)} kernel dispatches, {len(copies)} copies, eager total {total/1000:.1f} ms")
  for b in batched:
    print(f"  graphed: {b[2]}  tm {b[3]/1000:.2f} ms   (tinygrad says {b[4]:.0f} GFLOPS)")
  # Compare against this device's own graph, not the CPU warp graph that also appears here.
  if amd_graphs := [b[3] for b in batched if b[0] == "AMD"]:
    g = min(amd_graphs)
    print(f"  eager/graphed ratio {total/g:.1f}x -- eager per-dispatch time is NOT frame time")

  agg = defaultdict(lambda: [0.0, 0, 0.0, 0.0])
  for _, _, name, tm, gf, mb, ld in kernels:
    agg[name][0] += tm
    agg[name][1] += 1
    agg[name][2], agg[name][3] = gf, ld

  # B/FLOP is bytes of load/store issued per FLOP, i.e. tinygrad's `lds` estimate over its
  # `ops` estimate. Use `lds`, NOT `mem`: mem counts each distinct byte once, so it reports the
  # footprint (9 GB/s here) rather than the traffic (1275 GB/s), and reading the wrong one turns
  # an issue-bound kernel into a fictitious bandwidth-bound one. Being a ratio over the same
  # interval, it is immune to the eager pass's submit overhead. Lower is better; a bigger
  # register tile is what lowers it.
  print(f"\n{'':4}{'share':>7} {'cum':>7} {'total':>9} {'n':>4} {'mean':>10} {'thr':>5} {'B/FLOP':>7}  kernel / schedule")
  cum = 0.0
  for i, (name, (t, n, gf, mb)) in enumerate(sorted(agg.items(), key=lambda x: -x[1][0])):
    pct = 100 * t / total
    cum += pct
    if i >= 25 and pct < 0.5:
      continue
    s = sched.get(name)
    thr = "?" if s is None else s["threads"]
    opts = "" if s is None else ",".join(o.split("OptOps.")[1].rstrip(")").replace(", axis=", "@").replace(", arg=", "x")
                                         for o in s["beam"]["opts"]) if s and s["beam"] else "(no beam)"
    ai = (ld / gf) if gf else 0.0
    flag = "" if ai < 0.5 else " <-ldst heavy"
    print(f"{i+1:>3} {pct:6.2f}% {cum:6.2f}% {t/1000:8.2f}ms {n:4d} {t/n:9.1f}us {thr:>5} {ai:7.2f}  {name[:38]}{flag}")
    if opts:
      print(f"{'':52}   {opts[:96]}")

  # Time-weighted workgroup histogram: the question is never "how many kernels are small", it is
  # "how much of the frame is spent in them".
  print(f"\n  time by workgroup size (threads per workgroup):")
  hist = defaultdict(lambda: [0.0, 0])
  unknown = 0.0
  for name, (t, n, _gf, _mb) in agg.items():
    s = sched.get(name)
    if s is None:
      unknown += t
      continue
    hist[s["threads"]][0] += t
    hist[s["threads"]][1] += 1
  for thr in sorted(hist):
    t, k = hist[thr]
    print(f"    {thr:>5} threads  {k:>3} kernels  {t/1000:8.2f}ms  {100*t/total:5.2f}%")
  memt = sum(t for _, (t, _, gf, ld) in agg.items() if gf and ld / gf >= 0.5)
  print()
  print(f"  time in kernels issuing >=0.5 bytes of load/store per FLOP "
        f"(memory instructions crowd out MACs): {100*memt/total:.1f}%")
  sub = sum(t for thr, (t, _) in hist.items() if thr < 32)
  print(f"    sub-wavefront (<32): {sub/1000:.2f}ms = {100*sub/total:.2f}% of kernel time")
  if unknown:
    print(f"    unmatched by the audit: {unknown/1000:.2f}ms = {100*unknown/total:.2f}%")
  return 0


if __name__ == "__main__":
  sys.exit(main())
