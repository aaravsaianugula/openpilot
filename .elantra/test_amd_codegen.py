#!/usr/bin/env python3
"""
Tests for the gfx10 fp16 multiply-accumulate lowering in AMDLLVMRenderer.

What this protects: tinygrad renders an fp16 MAC as `fmul half` -> `fpext half to float` ->
`fadd float`, because sum_acc_dtype widens the accumulator to fp32 while the multiply stays in
fp16. The convert between them blocks the backend from contracting anything, so the MAC costs
three VALU ops where the hardware can do it in one. Widening both operands *before* the multiply
lets LLVM select a single `v_fma_mix_f32` (fp16 sources through op_sel, fp32 accumulate).

Measured on the RX 6600 XT: 1.233x and 1.218x on the two stage-3 GEMM shapes of the big driving
model, with the numerical error going *down* -- the old form rounds every product to fp16 before
adding it, and this keeps the product in fp32.

Two layers of test, because either can regress alone:
  - the UOp rewrite fires, fires commuted, does not loop, and leaves other architectures alone;
  - the rewrite actually reaches the machine code, i.e. the emitted ISA really does contain
    v_fma_mix_f32 and no longer contains v_mul_f16.

The second layer shells out to opt/llc and is skipped, loudly, if they are not installed. No GPU
is needed for any of this: BEAM's own compile workers render and compile with the device closed,
and this takes the same route.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCH = os.environ.get("ARCH", "gfx1032")

failures: list[str] = []
passes: list[str] = []
skips: list[str] = []


def case(name: str, got, want) -> None:
    if got == want:
        passes.append(name)
    else:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def check(name: str, condition: bool, detail: str = "") -> None:
    case(name + ((": " + detail) if detail and not condition else ""), bool(condition), True)


def build(tinygrad_path: Path):
    sys.path.insert(0, str(tinygrad_path))
    os.environ.setdefault("ALLOW_DEVICE_USAGE", "0")
    from tinygrad.dtype import dtypes
    from tinygrad.helpers import Target
    from tinygrad.renderer.llvmir import AMDLLVMRenderer
    from tinygrad.uop.ops import Ops, UOp
    return dtypes, Target, AMDLLVMRenderer, Ops, UOp


def mac_graph(UOp, Ops, dtypes, flip=False):
    """acc + (a*b).cast(float) with half a, b -- the shape every reduce in the model has."""
    leaf = lambda dt, n: UOp(Ops.NOOP, dt, (), n)  # noqa: E731
    a, b, acc = leaf(dtypes.half, "a"), leaf(dtypes.half, "b"), leaf(dtypes.float, "acc")
    prod = (a * b).cast(dtypes.float)
    return (prod + acc) if flip else (acc + prod)


def renderer(AMDLLVMRenderer, Target, arch):
    return AMDLLVMRenderer(Target(device="AMD", arch=arch))


# ------------------------------------------------------------------ the rewrite

def test_the_rewrite_fires(mods):
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    r = renderer(AMDLLVMRenderer, Target, ARCH)
    out = r.extra_matcher.rewrite(mac_graph(UOp, Ops, dtypes))
    check("the fp16 MAC is rewritten on gfx10", out is not None)
    if out is None:
        return
    muls = [s for s in out.src if s.op is Ops.MUL]
    check("the multiply survives", len(muls) == 1)
    if not muls:
        return
    case("the multiply is now float", muls[0].dtype, dtypes.float)
    check("both operands are casts from half",
          all(s.op is Ops.CAST and s.src[0].dtype == dtypes.half for s in muls[0].src),
          f"got {[(s.op.name, str(s.dtype)) for s in muls[0].src]}")


def test_it_fires_with_the_operands_the_other_way_round(mods):
    """ADD is commutative and tinygrad canonicalises either way, so both orders must match."""
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    r = renderer(AMDLLVMRenderer, Target, ARCH)
    check("commuted MAC is rewritten too",
          r.extra_matcher.rewrite(mac_graph(UOp, Ops, dtypes, flip=True)) is not None)


def test_it_does_not_rewrite_its_own_output(mods):
    """The rewrite must terminate: its result must not match the pattern again."""
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    r = renderer(AMDLLVMRenderer, Target, ARCH)
    out = r.extra_matcher.rewrite(mac_graph(UOp, Ops, dtypes))
    check("rewriting the result again is a no-op", out is None or r.extra_matcher.rewrite(out) is None)


def test_other_architectures_are_untouched(mods):
    """Contained to gfx10. gfx11/gfx12 have WMMA and reach these reduces through the TC path."""
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    for arch in ("gfx1100", "gfx1200", "gfx942"):
        r = renderer(AMDLLVMRenderer, Target, arch)
        check(f"{arch} is not rewritten",
              r.extra_matcher.rewrite(mac_graph(UOp, Ops, dtypes)) is None)


def test_the_kill_switch_works(mods):
    """AMD_FMA_MIX=0 has to restore the old lowering exactly, for bisecting on the car."""
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    import tinygrad.helpers as helpers
    prev = os.environ.get("AMD_FMA_MIX")
    os.environ["AMD_FMA_MIX"] = "0"
    # getenv memoises, so the cache has to be dropped for the new value to be seen.
    helpers.getenv.cache_clear()
    try:
        r = renderer(AMDLLVMRenderer, Target, ARCH)
        check("AMD_FMA_MIX=0 disables the rewrite",
              r.extra_matcher.rewrite(mac_graph(UOp, Ops, dtypes)) is None)
    finally:
        if prev is None:
            os.environ.pop("AMD_FMA_MIX", None)
        else:
            os.environ["AMD_FMA_MIX"] = prev
        helpers.getenv.cache_clear()


# ------------------------------------------------- does it reach the machine code

def tool(stem):
    for cand in (stem + "-18", stem + "-19", stem + "-20", stem):
        if subprocess.run(["which", cand], capture_output=True).returncode == 0:
            return cand
    return None


def render_gemm_isa(tinygrad_path: Path, arch: str, m=128, k=1536, n=6144):
    """Render the model's dominant GEMM shape for `arch` and compile it, exactly as tinygrad does.

    `opt -passes=default<O2>` before `llc` is not optional: LLVMCompiler.compile_to_obj runs that
    module pipeline before emitting, and the AMDGPU target registers InferAddressSpaces into it.
    Skipping it leaves the kernel's generic pointers unproven, so every access comes out as
    flat_load and the instruction mix is not the one that runs.
    """
    from dataclasses import replace

    from tinygrad import Tensor, dtypes
    from tinygrad.codegen import to_program
    from tinygrad.helpers import Target
    from tinygrad.renderer.llvmir import AMDLLVMRenderer
    from tinygrad.uop.ops import Ops

    ren = AMDLLVMRenderer(Target(device="AMD", arch=arch))
    a = Tensor.empty(m, k, dtype=dtypes.half, device="CPU")
    b = Tensor.empty(k, n, dtype=dtypes.half, device="CPU")
    linear = a.matmul(b).schedule_linear()
    ast = next(u.src[0] for u in linear.toposort()
               if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK)
    ast = ast.substitute({p: p.replace(arg=replace(p.arg, device="AMD"))
                          for p in ast.toposort() if p.op is Ops.PARAM})
    prg = to_program(ast, ren)
    src = next(u.arg for u in prg.toposort()
               if isinstance(getattr(u, "arg", None), str) and "define" in u.arg)

    o, l = tool("opt"), tool("llc")
    if o is None or l is None:
        return src, None
    common = ["-mtriple=amdgcn-amd-amdhsa", "-mcpu=" + arch, "-mattr=+cumode"]
    p1 = subprocess.run([o, "-passes=default<O2>"] + common + ["-S", "-o", "-"],
                        input=src, capture_output=True, text=True)
    if p1.returncode != 0:
        return src, None
    p2 = subprocess.run([l] + common + ["-O3", "-o", "-"], input=p1.stdout, capture_output=True, text=True)
    return src, (p2.stdout if p2.returncode == 0 else None)


def dot2_on() -> bool:
    # Must track AMDLLVMRenderer's own default, which is OFF: dot2 emits correctly but its gain is
    # inside this rig's noise and unmeasured on the card. If the renderer default flips, flip this.
    return os.environ.get("AMD_DOT2", "0") != "0"


def test_the_rewrite_reaches_the_emitted_ir(mods, tinygrad_path):
    """A rewrite that fires but never reaches the IR would be invisible to every test above."""
    src, _ = render_gemm_isa(tinygrad_path, ARCH)
    # True in both lowerings: the fp16 multiply is what costs the extra instructions, and neither
    # v_fma_mix_f32 nor v_dot2c_f32_f16 leaves one behind.
    check("emitted IR has no fmul on half operands", " half " not in _fmul_lines(src),
          f"{_fmul_lines(src)[:120]!r}")
    if dot2_on():
        check("emitted IR calls llvm.amdgcn.fdot2", "llvm.amdgcn.fdot2" in src)
    else:
        check("emitted IR multiplies in float", "float" in _fmul_lines(src))


def _fmul_lines(src: str) -> str:
    return " | ".join(ln.strip() for ln in src.splitlines() if " = fmul" in ln)[:400]


def test_the_backend_selects_a_one_instruction_mac(mods, tinygrad_path):
    """The whole point: the multiply-accumulate must become one instruction, not three.

    Which one depends on the mode -- v_dot2c_f32_f16 when AMD_DOT2 is on (two MACs per
    instruction), v_fma_mix_f32 otherwise (one). Either way v_mul_f16 has to be gone, because that
    is the instruction that only exists in the three-op form.
    """
    _, isa = render_gemm_isa(tinygrad_path, ARCH)
    if isa is None:
        skips.append("ISA check skipped: no opt/llc on PATH (install llvm-18 to run it)")
        return
    want = "v_dot2c_f32_f16" if dot2_on() else "v_fma_mix_f32"
    check(f"ISA contains {want}", want in isa)
    check("ISA no longer contains v_mul_f16", "v_mul_f16" not in isa)
    check("ISA no longer converts every product with v_cvt_f32_f16 per MAC",
          isa.count("v_cvt_f32_f16") < 24, f"{isa.count('v_cvt_f32_f16')} converts")


def test_dot2_pairs_every_mac(mods, tinygrad_path):
    """dot2 must pair the whole reduce chain, not one MAC in four.

    The first version of this rule only matched ADD(ADD(acc, MUL), MUL), which pairs one MAC per
    four-deep chain because tinygrad folds the first MAC of each accumulator into a bare multiply.
    The fix was a second pattern for ADD(MUL, MUL). This pins that: with dot2 on there should be no
    v_fma_mix_f32 left over on this shape.
    """
    if not dot2_on():
        skips.append("dot2 pairing check skipped: AMD_DOT2=0")
        return
    _, isa = render_gemm_isa(tinygrad_path, ARCH)
    if isa is None:
        return
    check("no unpaired v_fma_mix_f32 remains", "v_fma_mix_f32" not in isa,
          f"{isa.count('v_fma_mix_f32')} unpaired MACs")


def test_dot2_packing_costs_nothing(mods, tinygrad_path):
    """dot2's operands must be the registers the loads already produced, not shuffled copies.

    a0*b0 + a1*b1 is invariant under swapping the two products and under swapping either product's
    own factors, so there are four equivalent spellings and nothing makes tinygrad's incoming order
    the useful one. Measured on this shape, half the pairs arrived reversed -- and reversing a pair
    turns a <4 x half> register that could be fed to v_dot2c as-is into a v_mov plus a
    v_alignbit_b32. That packing tax was larger than the saving, which is the entire reason dot2
    measured as a wash and was left switched off.

    v_alignbit_b32 is the specific tell: it only appears here to reassemble a swapped pair.
    """
    if not dot2_on():
        skips.append("dot2 packing check skipped: AMD_DOT2=0")
        return
    _, isa = render_gemm_isa(tinygrad_path, ARCH)
    if isa is None:
        skips.append("dot2 packing check skipped: no opt/llc on PATH")
        return
    check("no v_alignbit_b32 -- no pair is assembled backwards", "v_alignbit_b32" not in isa,
          f"{isa.count('v_alignbit_b32')} pairs still need reassembling")


def test_pair_ordering_cannot_make_things_worse(mods, tinygrad_path):
    """Operands that are not lane reads must come back in the order they went in.

    The reordering is a scoring search over four equivalent spellings. If none of them makes a pair
    free -- which is every case where the halves did not come from one vector load -- it has to fall
    through to the original order rather than picking an arbitrary winner, so the rule can only ever
    remove instructions and never add a shuffle where there was none.
    """
    dtypes, Target, AMDLLVMRenderer, Ops, UOp = mods
    sys.path.insert(0, str(tinygrad_path))
    from tinygrad.renderer.llvmir import _order_dot2_pairs, _pair_is_free

    leaf = lambda n: UOp(Ops.NOOP, dtypes.half, (), n)  # noqa: E731
    a0, b0, a1, b1 = leaf("a0"), leaf("b0"), leaf("a1"), leaf("b1")
    got = _order_dot2_pairs((a0, b0), (a1, b1))
    case("non-lane operands keep their original order", got, ((a0, b0), (a1, b1)))
    check("a plain leaf is never mistaken for a free pair", not _pair_is_free(a0, a1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()

    if not (args.tinygrad / "tinygrad" / "renderer" / "llvmir.py").is_file():
        print(f"no tinygrad renderer at {args.tinygrad}; nothing to test")
        return 2

    mods = build(args.tinygrad)
    print(f"tinygrad: {args.tinygrad}   arch: {ARCH}")

    test_the_rewrite_fires(mods)
    test_it_fires_with_the_operands_the_other_way_round(mods)
    test_it_does_not_rewrite_its_own_output(mods)
    test_other_architectures_are_untouched(mods)
    test_the_kill_switch_works(mods)
    test_the_rewrite_reaches_the_emitted_ir(mods, args.tinygrad)
    test_the_backend_selects_a_one_instruction_mac(mods, args.tinygrad)
    test_dot2_pairs_every_mac(mods, args.tinygrad)
    test_dot2_packing_costs_nothing(mods, args.tinygrad)
    test_pair_ordering_cannot_make_things_worse(mods, args.tinygrad)

    print("\n" + "-" * 60)
    for s in skips:
        print("  SKIP " + s)
    if failures:
        print(f"FAILED: {len(failures)} case(s) failed, {len(passes)} passed\n")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"PASSED: all {len(passes)} cases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
