#!/usr/bin/env python3
"""
Tests for eliding the per-dispatch CS_PARTIAL_FLUSH on the AMD compute queue.

What this protects: tinygrad's AMDComputeQueue.exec ended every single dispatch with a
CS_PARTIAL_FLUSH, so a captured graph of 497 dispatches drained and refilled the GPU 497 times a
frame. No two kernels ever overlapped and every kernel paid its own wave-drain tail at near-zero
occupancy. The flush is not what makes a dispatch safe -- the command processor does not retire
DISPATCH_DIRECT until every workgroup has been handed to the SPI, and a launched wave carries its
own latched PGM/RSRC/USER_DATA/TMPRING state, so a following dispatch's register writes cannot
reach back into it. It is only needed when a *later* command has to observe this dispatch's
writes, which is what ROCm/PAL do: partial flush at barriers, nowhere else.

The rule, in tinygrad/runtime/graph/hcq.py:flush_schedule -- a drain waits for every wave
outstanding on the device, so one drain covers every dispatch encoded before it. Track the newest
dispatch guaranteed retired and pay for a drain only when a consumer would otherwise start while a
producer is still in flight, placing it on that consumer's immediate predecessor.

Two layers, because either can regress alone:
  - the decision itself, replayed against a model of the hardware that fails the moment a consumer
    is allowed to start while a producer of its own may still be in flight;
  - the wiring, checked on the real source: that the packet is behind the flag, that the flag
    defaults to keeping today's behaviour, and that the graph only ever passes it to queue types
    that opted in.

No GPU and no AMD device needed. ops_amd.py refuses to import off Linux, so the wiring layer reads
it with ast instead of importing it -- which also means these cases run identically on the dev box
and on the car.
"""
import argparse
import ast
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

failures: list[str] = []
passes: list[str] = []


def case(name: str, got, want) -> None:
    if got == want:
        passes.append(name)
    else:
        failures.append(f"{name}: got {got!r}, want {want!r}")


def check(name: str, condition: bool, detail: str = "") -> None:
    case(name + ((": " + detail) if detail and not condition else ""), bool(condition), True)


def build(tinygrad_path: Path):
    sys.path.insert(0, str(tinygrad_path))
    import tinygrad.runtime.graph.hcq as g
    return g


# ---------------------------------------------------------------------------
# a model of the queue, strict enough to catch an unsafe decision
# ---------------------------------------------------------------------------

def replay(dispatches: list[int], deps: dict[int, list[int]], flush: dict[int, bool]) -> list[str]:
    """
    Walk the queue in ring order the way the command processor does and report every place the
    schedule lets a dispatch read memory a producer of its own may still be writing.

    `done` is the newest dispatch the queue can prove has retired: nothing until the first drain,
    and after a drain after dispatch d, everything up to and including d, because a partial flush
    waits for all outstanding waves rather than just one dispatch's.
    """
    unsafe, done = [], -1
    for j in dispatches:
        for k in deps.get(j, []):
            if k > done:
                unsafe.append(f"dispatch {j} reads {k}, which may still be in flight (drained through {done})")
        if flush.get(j):
            done = j
    return unsafe


def linear_chain(n: int) -> tuple[list[int], dict[int, list[int]]]:
    return list(range(n)), {j: [j - 1] for j in range(1, n)}


def independent(n: int) -> tuple[list[int], dict[int, list[int]]]:
    return list(range(n)), {}


def random_dag(n: int, density: float, rng: random.Random) -> tuple[list[int], dict[int, list[int]]]:
    deps = {}
    for j in range(n):
        producers = [k for k in range(j) if rng.random() < density]
        if producers:
            deps[j] = producers
    return list(range(n)), deps


# a plausible slice of the driving model: a trunk, a second branch that runs alongside it, a
# concat that pulls all of it together, and a residual add that reaches back past the concat.
MODEL_SHAPE = (
    list(range(12)),
    {1: [0], 2: [1], 4: [3], 5: [2], 8: [4, 5, 6, 7], 9: [8], 10: [9, 2], 11: [10]},
)


# ---------------------------------------------------------------------------
# layer 1: the decision
# ---------------------------------------------------------------------------

def test_a_dependent_dispatch_always_gets_its_flush(g):
    """Every consumer in a straight chain has to see its producer drained."""
    for n in (2, 3, 8, 64, 497):
        dispatches, deps = linear_chain(n)
        flush = g.flush_schedule(dispatches, deps)
        check(f"chain of {n} replays safely", replay(dispatches, deps, flush) == [],
              detail=str(replay(dispatches, deps, flush))[:200])
        # in a chain every dispatch but the last has a consumer, so every one but the last must drain
        check(f"chain of {n} drains all but the last", sum(flush.values()) == n - 1, detail=f"{sum(flush.values())} drains")
        check(f"chain of {n} never drains the last dispatch", flush[dispatches[-1]] is False)


def test_a_run_of_independent_dispatches_costs_at_most_one_flush(g):
    for n in (2, 16, 497):
        dispatches, deps = independent(n)
        flush = g.flush_schedule(dispatches, deps)
        check(f"{n} independent dispatches drain at most once", sum(flush.values()) <= 1, detail=f"{sum(flush.values())} drains")
        check(f"{n} independent dispatches need no drain at all", sum(flush.values()) == 0)
        check(f"{n} independent dispatches replay safely", replay(dispatches, deps, flush) == [])

    # the interesting version: a long independent run that something at the end consumes. One drain
    # has to cover the whole run, because a partial flush waits for every outstanding wave.
    for n in (4, 16, 128):
        dispatches = list(range(n + 1))
        deps = {n: list(range(n))}
        flush = g.flush_schedule(dispatches, deps)
        check(f"{n} independent + 1 consumer drains exactly once", sum(flush.values()) == 1, detail=f"{sum(flush.values())} drains")
        check(f"{n} independent + 1 consumer drains on the predecessor", flush[n - 1] is True)
        check(f"{n} independent + 1 consumer replays safely", replay(dispatches, deps, flush) == [])


def test_one_drain_covers_every_earlier_producer(g):
    """
    0 and 1 both feed 9. Draining once, late, covers both -- and the drain that 4 forces for 3 also
    covers a later read of 2, which is the whole point of the rule.
    """
    dispatches = list(range(10))
    deps = {9: [0, 1]}
    flush = g.flush_schedule(dispatches, deps)
    check("two old producers, one consumer -> one drain", sum(flush.values()) == 1, detail=f"{sum(flush.values())} drains")
    check("two old producers, one consumer replays safely", replay(dispatches, deps, flush) == [])

    dispatches, deps = list(range(6)), {4: [3], 5: [2]}
    flush = g.flush_schedule(dispatches, deps)
    check("a later read of an older producer rides the earlier drain", sum(flush.values()) == 1, detail=f"{sum(flush.values())} drains")
    check("that schedule replays safely", replay(dispatches, deps, flush) == [])


def test_the_model_shaped_graph(g):
    dispatches, deps = MODEL_SHAPE
    flush = g.flush_schedule(dispatches, deps)
    check("model-shaped graph replays safely", replay(dispatches, deps, flush) == [],
          detail=str(replay(dispatches, deps, flush))[:200])
    check("model-shaped graph saves drains", sum(flush.values()) < len(dispatches),
          detail=f"{sum(flush.values())} of {len(dispatches)}")


def test_random_graphs_are_never_unsafe(g):
    rng = random.Random(20260828)
    worst, total_saved, n_graphs = None, 0, 0
    for _ in range(400):
        n = rng.randint(2, 60)
        dispatches, deps = random_dag(n, rng.choice([0.02, 0.08, 0.25, 0.6]), rng)
        flush = g.flush_schedule(dispatches, deps)
        if (unsafe := replay(dispatches, deps, flush)):
            worst = (deps, unsafe)
            break
        # never worse than what it replaces, and the last dispatch is always free
        if sum(flush.values()) > n:
            worst = (deps, [f"emitted {sum(flush.values())} drains for {n} dispatches"])
            break
        if flush[dispatches[-1]]:
            worst = (deps, ["drained after the final dispatch, which nothing can consume"])
            break
        total_saved += n - sum(flush.values())
        n_graphs += 1
    check("400 random graphs all replay safely", worst is None, detail=str(worst)[:300])
    check("random graphs actually save drains", total_saved > 0, detail=f"{total_saved} drains saved over {n_graphs} graphs")


def test_the_baseline_is_what_it_replaces(g):
    """Flushing after every dispatch is trivially safe; that is the behaviour the flag has to keep."""
    dispatches, deps = MODEL_SHAPE
    baseline = dict.fromkeys(dispatches, True)
    check("draining after every dispatch replays safely", replay(dispatches, deps, baseline) == [])
    check("draining after every dispatch costs one drain each", sum(baseline.values()) == len(dispatches))


class FakeQueue:
    """Stands in for a HWQueue. flush_elision_plan only ever reads the opt-in flag off one."""

    def __init__(self, name: str, opted_in: bool = True):
        self.name, self.supports_flush_elision = name, opted_in

    def __repr__(self):
        return self.name


def test_only_queues_that_opted_in_get_a_plan(g):
    opted_in, opted_out = FakeQueue("amd"), FakeQueue("aql", opted_in=False)
    items = {opted_in: [(j, True) for j in range(4)], opted_out: [(j, True) for j in range(4, 8)]}
    deps = {1: [0], 2: [1], 3: [2], 5: [4], 6: [5], 7: [6]}

    plan = g.flush_elision_plan(items, deps, set())
    check("the opted-in queue is planned", sorted(plan) == [0, 1, 2, 3], detail=str(sorted(plan)))
    check("the opted-out queue is left entirely alone", all(j not in plan for j in range(4, 8)))
    check("the planned queue still replays safely", replay([0, 1, 2, 3], deps, plan) == [])


def test_a_queue_carrying_a_non_dispatch_is_left_alone(g):
    """A drain can only be attached to an exec, so a queue with an RDMA ring on it has no safe spot."""
    q = FakeQueue("amd")
    plan = g.flush_elision_plan({q: [(0, True), (1, False), (2, True)]}, {2: [0]}, set())
    check("a queue with a non-dispatch item gets no plan", plan == {}, detail=str(plan))

    plan = g.flush_elision_plan({q: [(0, True), (1, True), (2, True)]}, {2: [0]}, set())
    check("the same queue with only dispatches does get a plan", sorted(plan) == [0, 1, 2], detail=str(plan))


def test_an_rdma_peer_queue_is_left_alone(g):
    q = FakeQueue("amd")
    all_dispatches = {q: [(j, True) for j in range(4)]}
    check("a compute queue RDMA writes into gets no plan", g.flush_elision_plan(all_dispatches, {1: [0]}, {q}) == {})
    check("the same queue with no RDMA does get a plan", g.flush_elision_plan(all_dispatches, {1: [0]}, set()) != {})


def test_two_eligible_queues_are_planned_independently(g):
    a, b = FakeQueue("amd:0"), FakeQueue("amd:1")
    # b's dispatches are interleaved with a's in ji order, which is how the graph really numbers them
    items = {a: [(0, True), (2, True), (4, True)], b: [(1, True), (3, True), (5, True)]}
    deps = {2: [0], 5: [3]}  # each dependency is within its own queue

    plan = g.flush_elision_plan(items, deps, set())
    check("both queues are planned", sorted(plan) == [0, 1, 2, 3, 4, 5], detail=str(sorted(plan)))
    check("queue a drains only where it must", [plan[j] for j in (0, 2, 4)] == [True, False, False],
          detail=str([plan[j] for j in (0, 2, 4)]))
    check("queue b drains only where it must", [plan[j] for j in (1, 3, 5)] == [False, True, False],
          detail=str([plan[j] for j in (1, 3, 5)]))
    check("queue a replays safely", replay([0, 2, 4], deps, plan) == [])
    check("queue b replays safely", replay([1, 3, 5], deps, plan) == [])


def test_an_empty_plan_means_todays_encoding(g):
    check("no queues at all -> empty plan", g.flush_elision_plan({}, {}, set()) == {})
    check("one dispatch on its own needs no drain", g.flush_elision_plan({FakeQueue('amd'): [(0, True)]}, {}, set()) == {0: False})


# ---------------------------------------------------------------------------
# layer 2: the wiring, read off the real source
# ---------------------------------------------------------------------------

def func_in(tree: ast.AST, name: str, cls: str | None = None) -> ast.FunctionDef | None:
    scope: ast.AST | None = tree
    if cls is not None:
        scope = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if scope is None:
            return None
    return next((n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name), None)


def class_attr(tree: ast.AST, cls: str, attr: str):
    node = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls), None)
    if node is None:
        return "<no such class>"
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == attr:
            return ast.literal_eval(stmt.value) if stmt.value is not None else None
    return "<unset>"


def test_the_env_var_defaults_to_todays_behaviour(g):
    tree = ast.parse((REPO / "tinygrad_repo" / "tinygrad" / "runtime" / "graph" / "hcq.py").read_text(encoding="utf-8"))

    # read the shipped default off the source, not off g.AMD_ELIDE_FLUSH, so this still passes while the flag is being A/B'd
    getenvs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getenv"
               and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "AMD_ELIDE_FLUSH"]
    check("AMD_ELIDE_FLUSH is read from the environment exactly once", len(getenvs) == 1)
    if getenvs:
        check("AMD_ELIDE_FLUSH ships defaulting to off", len(getenvs[0].args) == 2 and ast.literal_eval(getenvs[0].args[1]) == 0,
              detail=ast.dump(getenvs[0]))

    init = func_in(tree, "__init__", cls="HCQGraph")
    check("HCQGraph.__init__ found", init is not None)
    if init is None:
        return

    # every write to ji_flush other than the empty initialiser must sit under `if AMD_ELIDE_FLUSH:`
    gated = [n for n in ast.walk(init)
             if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "AMD_ELIDE_FLUSH"]
    writes_under_gate = sum(1 for blk in gated for n in ast.walk(blk) if isinstance(n, ast.Attribute) and n.attr == "ji_flush")
    all_writes = [n for n in ast.walk(init) if isinstance(n, ast.Attribute) and n.attr == "ji_flush"]
    check("the elision block is gated on AMD_ELIDE_FLUSH", len(gated) == 1, detail=f"{len(gated)} gates")
    # the ungated references are the `self.ji_flush: dict = {}` initialiser and the read in the encode loop
    check("ji_flush is only populated under the gate", writes_under_gate > 0 and len(all_writes) - writes_under_gate == 2,
          detail=f"{writes_under_gate} gated, {len(all_writes)} total")

    # with the gate off ji_flush stays empty, so exec must have a path that passes no flush at all
    execs = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "exec"]
    plain = [n for n in execs if not any(k.arg == "flush" for k in n.keywords)]
    check("the encode loop keeps a call that passes no flush", len(plain) == 1, detail=f"{len(plain)} of {len(execs)} exec calls")
    check("the encode loop has exactly one call that does pass flush", len(execs) - len(plain) == 1)

    # and the queue has to have opted in before the graph is even willing to compute a decision
    plan = func_in(tree, "flush_elision_plan")
    guard = [n for n in ast.walk(plan) if isinstance(n, ast.Attribute) and n.attr == "supports_flush_elision"] if plan else []
    check("the graph checks supports_flush_elision before deciding", len(guard) == 1)


def test_the_packet_is_behind_the_flag():
    src = (REPO / "tinygrad_repo" / "tinygrad" / "runtime" / "ops_amd.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = func_in(tree, "exec", cls="AMDComputeQueue")
    check("AMDComputeQueue.exec found", fn is not None)
    if fn is None:
        return

    args = [a.arg for a in fn.args.args]
    check("exec takes a flush argument", "flush" in args, detail=str(args))
    default = dict(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults)).get("flush")
    check("flush defaults to True so non-graph callers are untouched",
          default is not None and ast.literal_eval(default) is True, detail=ast.dump(default) if default else "no default")

    def is_partial_flush(node) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "CS_PARTIAL_FLUSH"

    emits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "pkt3"
             and any(is_partial_flush(sub) for sub in ast.walk(n))]
    check("exec emits CS_PARTIAL_FLUSH exactly once", len(emits) == 1, detail=f"{len(emits)} emits")

    guarded = [n for n in ast.walk(fn) if isinstance(n, ast.If)
               and any(isinstance(t, ast.Name) and t.id == "flush" for t in ast.walk(n.test))
               and any(e in ast.walk(n) for e in emits)]
    check("the CS_PARTIAL_FLUSH packet sits under an if that tests flush", len(guarded) == 1, detail=f"{len(guarded)} guards")

    # the acquire_mem invalidate is deliberately left unconditional: it is what makes a dependent
    # dispatch see its producer's writes once the drain has pushed them to GL2.
    acq = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "acquire_mem"]
    check("acquire_mem is still called once per dispatch", len(acq) == 1, detail=f"{len(acq)} calls")
    conditional = {id(n) for i in ast.walk(fn) if isinstance(i, ast.If) for n in ast.walk(i)}
    check("acquire_mem is not conditional", all(id(n) not in conditional for n in acq))

    # the AQL sibling has to stay signature-compatible with the parent it overrides
    aql = func_in(tree, "exec", cls="AMDComputeAQLQueue")
    check("AMDComputeAQLQueue.exec found", aql is not None)
    if aql is not None:
        check("AQL exec stays signature-compatible", [a.arg for a in aql.args.args] == args,
              detail=str([a.arg for a in aql.args.args]))
        aql_emits = [n for n in ast.walk(aql) if isinstance(n, ast.Call) and any(is_partial_flush(s) for s in ast.walk(n))]
        check("AQL never emitted a partial flush to begin with", len(aql_emits) == 0)

    check("AMDComputeQueue opts in", class_attr(tree, "AMDComputeQueue", "supports_flush_elision") is True)
    check("AMDComputeAQLQueue opts out", class_attr(tree, "AMDComputeAQLQueue", "supports_flush_elision") is False)

    base = ast.parse((REPO / "tinygrad_repo" / "tinygrad" / "runtime" / "support" / "hcq.py").read_text(encoding="utf-8"))
    check("HWQueue defaults to opted out", class_attr(base, "HWQueue", "supports_flush_elision") is False)


def test_other_backends_keep_their_signature():
    """Nothing may have grown a flush parameter it does not implement."""
    for name in ("ops_nv.py", "ops_qcom.py", "ops_cpu.py"):
        tree = ast.parse((REPO / "tinygrad_repo" / "tinygrad" / "runtime" / name).read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "exec"]:
            check(f"{name} exec left alone", "flush" not in [a.arg for a in fn.args.args])
        cls = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        opted = [c for c in cls if class_attr(tree, c, "supports_flush_elision") is True]
        check(f"{name} does not opt in", opted == [], detail=str(opted))


# ---------------------------------------------------------------------------

def print_trace(g) -> None:
    dispatches, deps = MODEL_SHAPE
    flush = g.flush_schedule(dispatches, deps)
    done = -1
    print("\n  a plausible 12-dispatch slice, one queue, in ring order")
    print("  " + "-" * 74)
    print(f"  {'disp':>4}  {'reads':<14} {'newest producer':>15} {'drained thru':>13}  {'drain after?':>12}")
    for j in dispatches:
        producers = deps.get(j, [])
        newest = max(producers) if producers else None
        safe = newest is None or newest <= done
        print(f"  {j:>4}  {str(producers or '-'):<14} {str(newest if newest is not None else '-'):>15} "
              f"{done:>13}  {('DRAIN' if flush[j] else '.'):>12}   {'' if safe else '<-- UNSAFE'}")
        if flush[j]:
            done = j
    print("  " + "-" * 74)
    print(f"  {sum(flush.values())} drains instead of {len(dispatches)}. "
          f"Dispatch 2 rides the drain that 4 forced for 3; 4, 5 and 6 overlap freely.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tinygrad", type=Path, default=REPO / "tinygrad_repo")
    args = ap.parse_args()

    if not (args.tinygrad / "tinygrad" / "runtime" / "graph" / "hcq.py").is_file():
        print(f"no tinygrad hcq graph at {args.tinygrad}; nothing to test")
        return 2

    g = build(args.tinygrad)
    print(f"tinygrad: {args.tinygrad}   AMD_ELIDE_FLUSH={g.AMD_ELIDE_FLUSH}")

    test_a_dependent_dispatch_always_gets_its_flush(g)
    test_a_run_of_independent_dispatches_costs_at_most_one_flush(g)
    test_one_drain_covers_every_earlier_producer(g)
    test_the_model_shaped_graph(g)
    test_only_queues_that_opted_in_get_a_plan(g)
    test_a_queue_carrying_a_non_dispatch_is_left_alone(g)
    test_an_rdma_peer_queue_is_left_alone(g)
    test_two_eligible_queues_are_planned_independently(g)
    test_an_empty_plan_means_todays_encoding(g)
    test_random_graphs_are_never_unsafe(g)
    test_the_baseline_is_what_it_replaces(g)
    test_the_env_var_defaults_to_todays_behaviour(g)
    test_the_packet_is_behind_the_flag()
    test_other_backends_keep_their_signature()

    print_trace(g)

    print("\n" + "-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s) failed, {len(passes)} passed\n")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"PASSED: all {len(passes)} cases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
