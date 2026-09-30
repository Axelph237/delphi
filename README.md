# Delphi

Delphi is a quantum compiler for **High-Level Oracle Synthesis** — a synthesis pipeline that takes a high-level specification of a search problem and produces an optimized quantum oracle circuit.

The intended workflow is:

1. A user describes search-space constraints in natural language.
2. An LLM assistant refines these into a formal set of clauses (a SAT-style CNF formula).
3. The compiler maps those clauses onto an efficient quantum oracle using a hierarchy of ancilla-qubit modules called an **HRSE tree**.

## Why it matters

Quantum search algorithms (Grover's, QAOA variants) require an oracle that marks solutions to a search problem. Building that oracle naively wastes ancilla qubits and gate depth. Delphi automates the structural optimization — choosing how to group and schedule clauses so that qubits can be reused — making otherwise impractical oracle sizes feasible.

## Architecture

```
natural language
       │  LLM
       ▼
   CNF clauses
       │  compiler
       ├── HRSE tree synthesis   (hrse.py)
       ├── CST construction      (cst.py)
       └── oracle mapping        (clause_pack.py, Qiskit)
```

### HRSE Tree (`packages/cstcompiler/src/cstcompiler/hrse.py`)

A **Hierarchical Recursive Synthesis-Evaluation (HRSE) tree** is the structural blueprint for the oracle. Each node represents a module that uses a fixed number of ancilla qubits (`size`). Child nodes are strictly smaller, enabling qubit reuse across levels.

Trees are synthesized by the **ASDT algorithm** (`asdt`), which produces an optimal tree for `m` clauses within a budget of `k` ancilla qubits. The maximum clause capacity for a given `k` is `ceil(3 × 2^(k−4))`.

```python
from cstcompiler.hrse import HRSENode

root = HRSENode.new(m=10, k=6)  # tree for 10 clauses, 6 ancilla qubits
```

### CST (`packages/cstcompiler/src/cstcompiler/cst.py`)

A **Clustered Synthesis Tree (CST)** mirrors the HRSE tree and assigns a *partition* of clause batches to each node. Batches are built greedily by the **SeedGrow heuristic**:

- **`grow_cst(root, clauses)`** — the top-level entrypoint. Traverses the HRSE tree top-down and assigns clauses to nodes so that higher nodes (more qubits) handle the most-conflicted clauses first.
- **`seed_grow(node, remaining, omap)`** — fills one CST node by repeatedly calling `grow_block`.
- **`grow_block(budget, remaining, omap)`** — greedily builds one batch up to a variable-count budget, choosing clauses by conflict degree.
- **`merge_adjacent(partition, budget)`** — post-processes a partition by merging consecutive under-budget batches.
- **`sort_clauses(clauses, omap)`** — sorts clauses by ascending conflict degree so seeds are chosen optimally.

#### Performance

The inner loops use two complementary techniques to avoid Python-speed bottlenecks:

| Operation | Technique | Speedup |
|---|---|---|
| `sort_clauses` | Pre-built padded index matrix + `np.lexsort` | 8–45× |
| `grow_block` conflict counting | Python integer bitmasks + `.bit_count()` | 5–20× |

A `NumpyContext` (built once per `grow_cst` call) holds all pre-computed structures and is threaded through every inner call so no work is repeated per node.

## Pipeline status

| Step | Status |
|---|---|
| User-LLM interaction for defining clauses | planned |
| Conversion of high-level language to CNF | **complete** ✅ |
| HRSE tree synthesis (ASDT algorithm) | **complete** ✅ |
| CST construction (SeedGrow heuristic) | **complete** ✅ |
| Mapping CST to optimized oracle circuit | **complete** ✅ |

## Getting started

The repository holds two packages under `packages/`, each with its own git repository:

- `cstcompiler` holds the compiler (HRSE trees, CST construction, oracle mapping).
- `delphi-interface` (imported as `delphi_interface`) holds the LLM interface and depends on `cstcompiler`.

The repository root is a [uv](https://docs.astral.sh/uv/) workspace containing both packages.

```bash
# Create .venv with both packages installed in editable mode.
# Run this from the repository root. Inside a package, uv sync removes the other package.
uv sync

# Run the test suites
(cd packages/cstcompiler && uv run pytest)
(cd packages/delphi-interface && uv run pytest)

# Run performance benchmarks (from packages/cstcompiler)
uv run python -m tests.benchmark_cst

# Compare oracle Clifford+T depth with the LTC paper's Table III (from packages/cstcompiler)
uv run python -m tests.benchmark_oracle
```

```python
from cstcompiler.hrse import HRSENode
from cstcompiler.cst import Clause, grow_cst

root = HRSENode.new(m=10, k=6)
clauses = [Clause(frozenset(vars)) for vars in [...]]  # your CNF clauses

cst = grow_cst(root, clauses)
```

## References

- [Modeling and Resource Optimization for Quantum Oracles (2026)](https://arxiv.org/html/2605.21380v1)
- [From Leaves to Clusters: Depth-Efficient SAT-Oracle Synthesis Based on the HRSE Model (2026)](https://arxiv.org/pdf/2607.11401)
