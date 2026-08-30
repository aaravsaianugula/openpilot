#!/usr/bin/env python3
"""Count the big driving model's arithmetic straight from the ONNX. No GPU, no tinygrad.

This exists to settle a 4.1x disagreement. tinygrad's own `batched 497` line reports 48.7
GFLOP/frame on the BEAM-tuned build and 202 GFLOP/frame on the un-tuned one -- the same model, the
same counter, two schedules. Every performance conclusion on this project divides by one of those
two numbers, and they lead to opposite answers about whether 20 Hz is reachable: at 48.7 GFLOP a
25 ms frame is 20% of this card's fp32-rate peak, and at 202 GFLOP it is 85%, which no kernel will
reach.

tinygrad's `estimates.ops` counts ALU ops in the *lowered* kernel, so it moves with the schedule
and cannot arbitrate. The ONNX graph is schedule-independent, so it can.

Multiply-accumulates are counted, then doubled, which is the convention both figures above use.
Elementwise work is reported separately rather than folded in: it is real arithmetic but it is
bandwidth-bound, and burying it in a single number is how a "% of peak" stops meaning anything.

  usage: model_flops.py [model.onnx]
"""
import sys
from collections import defaultdict

import onnx
from onnx import shape_inference

DEFAULT = "/root/models/big_driving_supercombo.onnx"


def prod(xs):
  n = 1
  for x in xs:
    n *= x
  return n


def shapes_of(model):
  """Every tensor's static shape, from inputs, initializers and inferred value_info."""
  out = {}
  for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
    dims = []
    ok = True
    for d in vi.type.tensor_type.shape.dim:
      if d.HasField("dim_value") and d.dim_value > 0:
        dims.append(d.dim_value)
      else:
        ok = False
        break
    if ok and dims:
      out[vi.name] = dims
  for init in model.graph.initializer:
    out[init.name] = list(init.dims)
  return out


def macs_for(node, shapes):
  """MACs for one node, or 0 for ops that are not multiply-accumulate work.

  Returns (macs, elementwise_ops). Anything whose shapes could not be resolved returns None so the
  caller can report it rather than silently contributing zero -- an undercount that looks like a
  result is the failure mode this whole script exists to avoid.
  """
  op = node.op_type
  outs = [shapes.get(o) for o in node.output]
  ins = [shapes.get(i) for i in node.input]

  if op == "Conv":
    if outs[0] is None or ins[1] is None:
      return None
    # out = (N, Cout, *spatial); weight = (Cout, Cin/groups, *kernel)
    w = ins[1]
    return (prod(outs[0]) * prod(w[1:]), 0)

  if op in ("MatMul", "Gemm"):
    if outs[0] is None or ins[1] is None:
      return None
    b = ins[1]
    # contraction length is the second-to-last weight dim for MatMul/Gemm(transB=0)
    k = b[-2] if len(b) >= 2 else b[0]
    if op == "Gemm":
      transB = next((a.i for a in node.attribute if a.name == "transB"), 0)
      k = b[-1] if transB else b[-2]
    return (prod(outs[0]) * k, 0)

  if op in ("ConvTranspose",):
    if outs[0] is None or ins[1] is None:
      return None
    w = ins[1]
    return (prod(outs[0]) * prod(w[1:]), 0)

  if op in ("Einsum",):
    return None  # not present in this model; refuse to guess if it appears

  # Everything else is elementwise / shape / reduction. Count output elements as a proxy for the
  # bandwidth-bound work, and keep it out of the MAC total.
  if outs[0] is None:
    return (0, 0)
  return (0, prod(outs[0]))


def main():
  path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
  print(f"model: {path}")
  print("running shape inference (large model, this takes a minute)...", flush=True)
  inferred = path + ".inferred"
  shape_inference.infer_shapes_path(path, inferred)
  model = onnx.load(inferred, load_external_data=False)
  shapes = shapes_of(model)
  print(f"  {len(model.graph.node)} nodes, {len(shapes)} tensors with static shapes")

  macs = defaultdict(int)
  elem = defaultdict(int)
  counts = defaultdict(int)
  unresolved = defaultdict(int)
  for node in model.graph.node:
    counts[node.op_type] += 1
    r = macs_for(node, shapes)
    if r is None:
      unresolved[node.op_type] += 1
      continue
    m, e = r
    macs[node.op_type] += m
    elem[node.op_type] += e

  total_macs = sum(macs.values())
  total_elem = sum(elem.values())
  print("\n  MAC-bearing ops:")
  for op in sorted(macs, key=lambda k: -macs[k]):
    if macs[op]:
      print(f"    {op:<16} {counts[op]:>5} nodes  {macs[op] / 1e9:10.3f} GMAC"
            f"  {100.0 * macs[op] / total_macs:5.1f}% of MACs")
  print(f"\n  total  {total_macs / 1e9:.3f} GMAC = {2 * total_macs / 1e9:.3f} GFLOP")
  print(f"  elementwise output elements: {total_elem / 1e9:.3f} G (bandwidth-bound, not in the total)")
  if unresolved:
    print(f"\n  *** UNRESOLVED (shapes missing, NOT counted): {dict(unresolved)}")
    print("  *** the total above is a lower bound")
  return 0


if __name__ == "__main__":
  sys.exit(main())
