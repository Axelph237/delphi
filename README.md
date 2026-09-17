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
       └── oracle mapping        (planned)
```

### HRSE Tree (`compiler/hrse.py`)

A **Hierarchical Reed-Solomon Encoding (HRSE) tree** is the structural blueprint for the oracle. Each node represents a module that uses a fixed number of ancilla qubits (`size`). Child nodes are strictly smaller, enabling qubit reuse across levels.

Trees are synthesized by the **ASDT algorithm** (`asdt`), which produces an optimal tree for `m` clauses within a budget of `k` ancilla qubits. The maximum clause capacity for a given `k` is `ceil(3 × 2^(k−4))`.

```python
from compiler.hrse import HRSENode

root = HRSENode.new(m=10, k=6)  # tree for 10 clauses, 6 ancilla qubits
```

### CST (`compiler/cst.py`)

A **Compute Structure Tree (CST)** mirrors the HRSE tree and assigns a *partition* of clause batches to each node. Batches are built greedily by the **SeedGrow heuristic**:

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
| Conversion of high-level language to CNF | planned |
| HRSE tree synthesis (ASDT algorithm) | **complete** |
| CST construction (SeedGrow heuristic) | **complete** |
| Mapping CST to optimized oracle circuit | planned |

## Getting started

```bash
# Install dependencies (Python 3.12+, numpy)
pip install numpy pytest

# Run the test suite
pytest tests/

# Run performance benchmarks
python -m tests.benchmark_cst
```

```python
from compiler.hrse import HRSENode
from compiler.cst import Clause, grow_cst

root = HRSENode.new(m=10, k=6)
clauses = [Clause(frozenset(vars)) for vars in [...]]  # your CNF clauses

cst = grow_cst(root, clauses)
```

## References

- [Ancilla-Efficient Quantum Oracle Synthesis via Hierarchical Reed-Solomon Encoding (2025)](https://arxiv.org/html/2605.21380v1)
- [Supplementary paper on CST construction (2025)](https://arxiv.org/pdf/2607.11401)
