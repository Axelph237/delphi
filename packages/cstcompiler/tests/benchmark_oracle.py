"""End-to-end CST oracle depth compared with the LTC paper's Table III.

Run from packages/cstcompiler:

    uv run python -m tests.benchmark_oracle                    # n = 40, 80; 5 instances each
    uv run python -m tests.benchmark_oracle --paper            # all four size classes, 20 instances
    uv run python -m tests.benchmark_oracle --classes 400 --instances 3

Depth follows the paper's cost model [LTC §VII-A]: Clifford+T depth, with every
multi-controlled X charged the depth of its Khattar–Gidney decomposition. The paper
does not say which Khattar–Gidney variant it measured, so both 2-ancilla variants are
reported. "2-dirty" borrows two qubits in any state, which every MCX here has available.
"2-clean" assumes two |0⟩ ancillae are free at every MCX, which is optimistic.
Borrowed ancilla wires are not charged, matching the paper's per-gate substitution, so
both columns assume each MCX borrows qubits that are otherwise idle at that moment.

--decompose also runs Qiskit's own Khattar–Gidney 2-dirty decomposition on the first
instance of each row and reports its true Clifford+T depth. Qiskit's synthesis pass lends
every MCX the same borrowed qubits, so gates that the model runs in parallel serialize
there; the column shows how far a naive ancilla assignment sits from the model.

The instances are fresh random 4-CNF formulas in the paper's configuration, not the
paper's own DIMACS files, so agreement is statistical rather than per-instance.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from functools import cache

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlledGate
from qiskit.synthesis import synth_mcx_2_clean_kg24, synth_mcx_2_dirty_kg24
from qiskit.transpiler.passes import HLSConfig

from cstcompiler.clause_pack import cst_to_oracle
from cstcompiler.cst import grow_cst
from cstcompiler.backbone import HRSENode
from cstcompiler.numpy_context import build_numpy_context
from tests.utils import random_kcnf

CLIFFORD_T = ["h", "s", "sdg", "t", "tdg", "x", "cx"]
MCX_SYNTHESIS = {"2-dirty": synth_mcx_2_dirty_kg24, "2-clean": synth_mcx_2_clean_kg24}
CLAUSE_WIDTH = 4
CLAUSE_DENSITY = 9.931

# [LTC Table III] n → [(a_q, SOTA depth, CST depth)], each a mean over 20 instances
TABLE_III = {
    40: [(80, 56_793, 16_263), (440, 32_093, 2_363), (793, 28_217, 1_506)],
    80: [(80, 87_033, 28_101), (880, 34_353, 2_524), (1587, 29_681, 1_662)],
    400: [(100, 762_265, 115_194), (3200, 41_481, 6_112), (7943, 32_337, 1_922)],
    800: [(200, 624_185, 114_340), (6400, 44_185, 6_500), (15887, 33_621, 2_026)],
}


@cache
def mcx_depth(controls: int, variant: str) -> int:
    """Clifford+T depth of an X gate with `controls` controls under the given Khattar–Gidney variant."""
    if controls <= 1:
        return 1
    circuit = MCX_SYNTHESIS[variant](controls)
    return transpile(circuit, basis_gates=CLIFFORD_T, optimization_level=0).depth()


def clifford_t_depth(circuit: QuantumCircuit, variant: str) -> int:
    """Critical-path depth of an X/CX/MCX circuit with each gate charged its Clifford+T depth."""
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    ready = [0] * circuit.num_qubits
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [index[q] for q in instruction.qubits]
        controls = operation.num_ctrl_qubits if isinstance(operation, ControlledGate) else 0
        finish = max(ready[q] for q in qubits) + mcx_depth(controls, variant)
        for q in qubits:
            ready[q] = finish
    return max(ready, default=0)


def decomposed_depth(circuit: QuantumCircuit) -> int:
    """True Clifford+T depth after Qiskit's Khattar–Gidney 2-dirty synthesis of every MCX."""
    hls_config = HLSConfig(mcx=["2_dirty_kg24"])
    return transpile(circuit, basis_gates=CLIFFORD_T, optimization_level=0, hls_config=hls_config).depth()


def oracle_circuit(n: int, budget: int, seed: int) -> QuantumCircuit:
    """Synthesize the CST oracle for one random 4-CNF instance in the paper's configuration."""
    m = int(CLAUSE_DENSITY * n)
    clauses = random_kcnf(n, m, CLAUSE_WIDTH, random.Random(f"ltc-table-iii-{n}-{seed}"))
    ctx = build_numpy_context(clauses)
    hrse_root = HRSENode.new(m, budget)
    assert ctx is not None and hrse_root is not None
    root = grow_cst(hrse_root, list(clauses))
    assert root is not None
    circuit, _, _ = cst_to_oracle(ctx.n_vars, root, ctx)
    return circuit


def _mean_sd(values: list[int]) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):>9,.0f} ± {sd:<7,.0f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=int, nargs="+", default=[40, 80], choices=sorted(TABLE_III))
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--paper", action="store_true", help="all four size classes with 20 instances, as in the paper")
    parser.add_argument("--decompose", action="store_true", help="also report Qiskit's real 2-dirty decomposition depth")
    args = parser.parse_args()
    classes, instances = (sorted(TABLE_III), 20) if args.paper else (args.classes, args.instances)

    header = (
        f"{'n':>4} {'m':>5} {'a_q':>6} | {'ours, 2-dirty':^19} {'ours, 2-clean':^19} | "
        f"{'paper CST':>9} {'paper SOTA':>10} | {'ours/paper':>10} | {'norm. ours':>11} {'norm. paper':>11}"
        + (f" | {'Qiskit HLS':>10}" if args.decompose else "")
    )
    print(f"CST oracle Clifford+T depth, random {CLAUSE_WIDTH}-CNF, {instances} instances per row")
    print(header)
    print("-" * len(header))

    for n in classes:
        m = int(CLAUSE_DENSITY * n)
        for budget, paper_sota, paper_cst in TABLE_III[n]:
            start = time.perf_counter()
            circuits = [oracle_circuit(n, budget, seed) for seed in range(instances)]
            dirty = [clifford_t_depth(c, "2-dirty") for c in circuits]
            clean = [clifford_t_depth(c, "2-clean") for c in circuits]
            decomposed = f" | {decomposed_depth(circuits[0]):>10,}" if args.decompose else ""
            ratio = f"{statistics.mean(clean) / paper_cst:.2f}–{statistics.mean(dirty) / paper_cst:.2f}"
            normalized = f"{statistics.mean(clean) / paper_sota:.3f}–{statistics.mean(dirty) / paper_sota:.3f}"
            print(
                f"{n:>4} {m:>5} {budget:>6} | {_mean_sd(dirty)} {_mean_sd(clean)} | "
                f"{paper_cst:>9,} {paper_sota:>10,} | {ratio:>10} | {normalized:>11} {paper_cst / paper_sota:>11.3f}"
                f"{decomposed}   ({time.perf_counter() - start:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
