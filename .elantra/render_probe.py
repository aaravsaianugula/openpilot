#!/usr/bin/env python3
"""Render a kernel for gfx1032 and report what the multiply-accumulate compiled to -- no GPU.

Codegen changes can be evaluated without the eGPU attached. BEAM's own compile workers run under
ALLOW_DEVICE_USAGE=0 and call to_program(ast, renderer) with the device never opened, so the same
route works here: schedule the op on CPU, retarget the AST's params at AMD, render with
AMDLLVMRenderer, and compile the result with llc for gfx1032.

That turns a codegen experiment from "boot the dock, run BEAM, wait" into a couple of seconds, and
it answers the question the throughput number cannot: did the rewrite actually fire, and did the
backend select the instruction it was supposed to enable?

  usage: render_probe.py [M K N]
  env:   AMD_FMA_MIX=0/1 and any other renderer knob; ARCH=gfx1032; SHOW=ir|isa|both to dump text.
"""
import os
import re
import subprocess
import sys
from collections import Counter

os.environ.setdefault("ALLOW_DEVICE_USAGE", "0")

ARCH = os.environ.get("ARCH", "gfx1032")
SHOW = os.environ.get("SHOW", "")

IR_PATTERNS = (
  ("fmul half  (old MAC)", r"fmul[^\n]* half "),
  ("fmul float (fma_mix)", r"fmul[^\n]* float "),
  ("fpext half->float", r"fpext half"),
  ("fadd float", r"fadd[^\n]* float "),
  ("fadd half", r"fadd[^\n]* half "),
  ("load half (scalar)", r"load half,"),
  ("load <N x half>", r"load <\d+ x half>"),
  ("llvm.amdgcn.fdot2", r"llvm\.amdgcn\.fdot2"),
  ("vector ALU (<N x ...>)", r"= f(mul|add|sub)[^\n]*<\d+ x "),
)

ISA_PATTERNS = (
  ("v_dot2c_f32_f16", r"v_dot2c?_f32_f16"),
  ("v_fma_mix_f32", r"v_fma_mix"),
  ("v_pk_* packed", r"v_pk_"),
  ("v_mul_f16", r"v_mul_f16"),
  ("v_cvt_f32_f16", r"v_cvt_f32_f16"),
  ("v_add_f32", r"v_add_f32"),
  ("v_fmac/fma_f32", r"v_(fmac|fma|mac|mad)_f32"),
  ("v_fmac/fma_f16", r"v_(fmac|fma|fmaak|mac|mad)_f16"),
  ("global_load_ushort", r"global_load_ushort"),
  ("global_load_dwordxN", r"global_load_dwordx"),
  ("scratch spill", r"buffer_(load|store)_dword"),
)


def count(text, patterns):
  out = Counter()
  for label, pat in patterns:
    n = len(re.findall(pat, text))
    if n:
      out[label] = n
  return out


def tool(stem):
  for cand in (stem + "-18", stem + "-19", stem + "-20", stem):
    if subprocess.run(["which", cand], capture_output=True).returncode == 0:
      return cand
  return None


def llc(ir, arch, feats):
  """Compile the rendered IR the way tinygrad does: the default<O2> module pipeline, then codegen.

  Running llc alone is not equivalent and quietly produces worse code. LLVMCompiler.compile_to_obj
  calls LLVMRunPasses(module, 'default<O2>', target_machine) before emitting, and the AMDGPU
  target registers passes into that pipeline -- InferAddressSpaces among them. Skip it and the
  kernel's generic pointers are never proved global, so every access comes out as flat_load with a
  full 64-bit address computed in VGPRs instead of global_load with a scalar base and an immediate
  offset. That is a large difference in instruction count and it is not one the renderer caused.
  """
  o, l = tool("opt"), tool("llc")
  if o is None or l is None:
    return "no opt/llc on PATH"
  common = ["-mtriple=amdgcn-amd-amdhsa", "-mcpu=" + arch] + (["-mattr=" + feats] if feats else [])
  p1 = subprocess.run([o, "-passes=default<O2>"] + common + ["-S", "-o", "-"],
                      input=ir, capture_output=True, text=True)
  if p1.returncode != 0:
    return "OPT FAILED: " + p1.stderr[:400]
  p2 = subprocess.run([l] + common + ["-O3", "-o", "-"], input=p1.stdout, capture_output=True, text=True)
  return p2.stdout if p2.returncode == 0 else "LLC FAILED: " + p2.stderr[:400]


def main():
  m, k, n = (int(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (128, 1536, 6144)))

  from dataclasses import replace
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import Ops

  ren = AMDLLVMRenderer(Target(device="AMD", arch=ARCH))
  fma_mix = os.environ.get("AMD_FMA_MIX", "1")
  print(f"arch {ARCH}   AMD_FMA_MIX={fma_mix}   shape M={m} K={k} N={n}")

  a = Tensor.empty(m, k, dtype=dtypes.half, device="CPU")
  b = Tensor.empty(k, n, dtype=dtypes.half, device="CPU")
  linear = a.matmul(b).schedule_linear()
  asts, seen = [], set()
  for u in linear.toposort():
    if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK and u.src[0].key not in seen:
      seen.add(u.src[0].key)
      asts.append(u.src[0])
  if not asts:
    print("no CALL(SINK) in the linear graph")
    return 1

  for i, ast in enumerate(asts):
    # Retarget the buffers at AMD, exactly as codegen/opt/search.py does before compiling a
    # candidate on a worker that has no device open.
    ast = ast.substitute({p: p.replace(arg=replace(p.arg, device="AMD"))
                          for p in ast.toposort() if p.op is Ops.PARAM})
    prg = to_program(ast, ren)
    src = next((u.arg for u in prg.toposort()
                if isinstance(getattr(u, "arg", None), str) and "define" in u.arg), None)
    if src is None:
      print("could not locate rendered source on the program uop")
      return 1

    name = re.search(r"void @([A-Za-z0-9_]+)\(", src)
    print(f"\nkernel {i}: {name.group(1) if name else '?'}")
    print("  LLVM IR:")
    for label, c in count(src, IR_PATTERNS).items():
      print(f"    {label:<24} {c:6d}")

    isa = llc(src, ARCH, "" if os.environ.get("AMD_WGP_MODE", "0") != "0" else "+cumode")
    if isa.startswith(("LLC FAILED", "OPT FAILED", "no opt")):
      print(f"  {isa}")
      continue
    print("  gfx1032 ISA:")
    for label, c in count(isa, ISA_PATTERNS).items():
      print(f"    {label:<24} {c:6d}")

    # The families above only cover the arithmetic and the loads. What is left -- address
    # arithmetic, exec-mask branching, waitcnt stalls -- is the other half of the kernel, and it
    # is what decides whether making the MAC cheaper actually makes the kernel faster.
    mnem = Counter()
    for line in isa.splitlines():
      mm = re.match(r"\s+([a-z][a-z0-9_]+)", line)
      if mm and mm.group(1) != "s_code_end":
        mnem[mm.group(1)] += 1
    total = sum(mnem.values())
    valu = sum(v for kk, v in mnem.items() if kk.startswith("v_"))
    salu = sum(v for kk, v in mnem.items() if kk.startswith("s_"))
    mem = sum(v for kk, v in mnem.items() if kk.startswith(("global_", "ds_", "buffer_", "scratch_", "flat_")))
    print(f"    {'TOTAL instructions':<24} {total:6d}   (VALU {valu}, SALU/ctrl {salu}, memory {mem})")
    print("    top mnemonics: " + ", ".join(f"{nm}={ct}" for nm, ct in mnem.most_common(10)))

    if SHOW in ("ir", "both"):
      print("\n--- IR ---\n" + src)
    if SHOW in ("isa", "both"):
      print("\n--- ISA ---\n" + isa)
  return 0


if __name__ == "__main__":
  sys.exit(main())
