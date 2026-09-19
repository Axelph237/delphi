"""
Benchmark: numpy / bitmask path vs pure-Python fallback for key CST functions,
plus end-to-end grow_cst stability across varying input sizes.

The optimized path is the default (used by seed_grow / the public API).
The Python fallback is exercised by calling lower-level functions without a NumpyContext.
"""

from __future__ import annotations
import time
import random

from compiler.cst import (
    Clause, Batch,
    build_occurence_list, sort_clauses, grow_block, merge_adjacent,
    grow_cst,
)
from compiler.numpy_context import NumpyContext, build_numpy_context
from compiler.hrse import HRSENode


# ── helpers ───────────────────────────────────────────────────────────────────

def make_cnf(n_vars: int, n_clauses: int, width: int, seed: int = 42) -> list[Clause]:
    rng = random.Random(seed)
    clauses = []
    for _ in range(n_clauses):
        vs = rng.sample(range(1, n_vars + 1), min(width, n_vars))
        clauses.append(Clause(frozenset(vs), 0))
    return clauses


def timeit(fn, n_reps: int = 5) -> float:
    """Return minimum wall time over n_reps calls (seconds)."""
    best = float("inf")
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fmt(s: float) -> str:
    if s >= 1:
        return f"{s:.3f}s"
    if s >= 1e-3:
        return f"{s*1e3:.2f}ms"
    return f"{s*1e6:.1f}µs"


def ratio(py: float, np_: float) -> str:
    r = py / np_ if np_ > 0 else float("inf")
    return f"{r:.1f}×"


# ── sort_clauses ──────────────────────────────────────────────────────────────

def bench_sort_clauses(n_vars: int, n_clauses: int, width: int):
    clauses = make_cnf(n_vars, n_clauses, width)
    omap = build_occurence_list(clauses)
    ctx = build_numpy_context(clauses)  # pre-built; passed to fast path

    def py_sort():
        c = clauses[:]
        c.sort(key=lambda cl: (
            sum(len(omap.get(v, set())) - 1 for v in cl.normed_variables),
            len(cl.normed_variables)
        ))

    def np_sort():
        c = clauses[:]
        sort_clauses(c, omap, ctx)  # uses pre-built padded matrix from ctx

    t_py = timeit(py_sort)
    t_np = timeit(np_sort)
    print(f"  sort_clauses  n={n_clauses:>5}, vars={n_vars:>4}, w={width:>2}  |  "
          f"Python {fmt(t_py):>8}  NumPy {fmt(t_np):>8}  speedup {ratio(t_py, t_np):>6}")


# ── grow_block ────────────────────────────────────────────────────────────────

def bench_grow_block(n_vars: int, n_clauses: int, width: int, budget: int = 50):
    clauses = make_cnf(n_vars, n_clauses, width)
    omap = build_occurence_list(clauses)
    ctx = build_numpy_context(clauses)

    def py_run():
        c = clauses[:]
        grow_block(budget, c, omap, ctx=None)

    def np_run():
        c = clauses[:]
        grow_block(budget, c, omap, ctx=ctx)

    t_py = timeit(py_run)
    t_np = timeit(np_run)
    print(f"  grow_block    n={n_clauses:>5}, vars={n_vars:>4}, w={width:>2}  |  "
          f"Python {fmt(t_py):>8}  Bitmsk {fmt(t_np):>8}  speedup {ratio(t_py, t_np):>6}")


# ── merge_adjacent ────────────────────────────────────────────────────────────

def bench_merge_adjacent(n_vars: int, n_clauses: int, width: int, budget: int = 200):
    clauses = make_cnf(n_vars, n_clauses, width)
    omap = build_occurence_list(clauses)
    ctx = build_numpy_context(clauses)

    partition: list[Batch] = []
    remaining = clauses[:]
    while remaining:
        b = grow_block(budget, remaining, omap, ctx=ctx)
        partition.append(b)

    def run():
        merge_adjacent(list(partition), budget)

    t = timeit(run)
    print(f"  merge_adjacent n={n_clauses:>4}, vars={n_vars:>4}, w={width:>2}  |  "
          f"(Python-only)  {fmt(t):>8}")


# ── main ──────────────────────────────────────────────────────────────────────

def bench_grow_cst(m: int, k: int, n_vars: int, width: int, n_reps: int = 3):
    """End-to-end grow_cst benchmark: build HRSE tree, generate clauses, time grow_cst."""
    root = HRSENode.new(m, k)
    assert root is not None, f"HRSENode.new({m}, {k}) returned None"

    rng = random.Random(42)
    clauses = [
        Clause(frozenset(rng.sample(range(1, n_vars + 1), min(width, n_vars))), 0)
        for _ in range(m)
    ]

    t = timeit(lambda: grow_cst(root, clauses[:]), n_reps=n_reps)
    print(f"  grow_cst  m={m:>4}, k={k:>3}, vars={n_vars:>4}, w={width:>2}  |  {fmt(t):>8} per call")


if __name__ == "__main__":
    print("=" * 74)
    print("sort_clauses  (fast path uses pre-built padded matrix from build_numpy_context)")
    print("=" * 74)
    bench_sort_clauses(n_vars=100,  n_clauses=500,   width=5)
    bench_sort_clauses(n_vars=200,  n_clauses=2000,  width=10)
    bench_sort_clauses(n_vars=500,  n_clauses=5000,  width=15)
    bench_sort_clauses(n_vars=1000, n_clauses=10000, width=20)
    bench_sort_clauses(n_vars=1000, n_clauses=10000, width=50)

    print()
    print("=" * 74)
    print("grow_block  (fast path uses Python int bitmasks for intersection counting)")
    print("=" * 74)
    bench_grow_block(n_vars=100,  n_clauses=500,  width=5,   budget=50)
    bench_grow_block(n_vars=200,  n_clauses=2000, width=10,  budget=100)
    bench_grow_block(n_vars=500,  n_clauses=5000, width=20,  budget=200)
    bench_grow_block(n_vars=500,  n_clauses=5000, width=50,  budget=200)
    bench_grow_block(n_vars=1000, n_clauses=5000, width=100, budget=500)

    print()
    print("=" * 74)
    print("merge_adjacent  (Python dict path; bitmask doesn't help counts)")
    print("=" * 74)
    bench_merge_adjacent(n_vars=100, n_clauses=500,  width=5,  budget=50)
    bench_merge_adjacent(n_vars=200, n_clauses=2000, width=10, budget=100)
    bench_merge_adjacent(n_vars=500, n_clauses=3000, width=20, budget=200)

    print()
    print("=" * 74)
    print("grow_cst  end-to-end  (stability across varying input sizes)")
    print("=" * 74)
    # k=4 → max 3 clauses, k=5 → 6, k=6 → 12, k=7 → 24, k=8 → 48, k=9 → 96
    bench_grow_cst(m=3,  k=4,  n_vars=20,  width=3)
    bench_grow_cst(m=6,  k=5,  n_vars=30,  width=3)
    bench_grow_cst(m=10, k=6,  n_vars=50,  width=5)
    bench_grow_cst(m=20, k=7,  n_vars=100, width=5)
    bench_grow_cst(m=40, k=8,  n_vars=200, width=8)
    bench_grow_cst(m=80, k=9,  n_vars=400, width=10)
    bench_grow_cst(m=80, k=9,  n_vars=400, width=20)
    bench_grow_cst(m=80, k=9,  n_vars=400, width=50)
