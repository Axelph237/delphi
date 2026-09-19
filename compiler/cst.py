from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
from typing_extensions import Set

from .bucket_queue import BucketQueue
from .numpy_context import NumpyContext, build_numpy_context
from compiler.hrse import HRSENode


# ---------- Type Aliases ----------

type variable = int
type omap = dict[variable, set[Clause]]


# ---------- Data Types ----------

@dataclass(frozen=True)
class Clause:
    normed_variables: tuple[variable, ...]
    variable_polarity_mask: int

    def __init__(self, variables: Set[variable], polarity_mask: int):
        object.__setattr__(self, 'normed_variables', tuple(sorted(variables)))
        object.__setattr__(self, 'variable_polarity_mask', polarity_mask)


@dataclass
class Batch:
    r"""A set of clauses evaluated in parallel within one time step.

    Greedy construction: once a clause is added it is never removed.
    Redundancy R(β) = Σ_z max{k_z(β) − 1, 0} counts ancilla needed for
    shared variables.
    """
    _variables: dict[variable, int]
    _clauses: set[Clause]
    redundancy: int

    def __init__(self, clauses: Set[Clause] = frozenset()):
        self._clauses = set()
        self._variables = dict()
        self.redundancy = 0
        for c in clauses:
            self.add_clause(c)

    def add_clause(self, clause: Clause):
        if clause in self._clauses:
            return
        self._clauses.add(clause)
        delta = 0
        for v in clause.normed_variables:
            existing = self._variables.get(v, 0)
            if existing:
                delta += 1
            self._variables[v] = existing + 1
        self.redundancy += delta

    @staticmethod
    def _merge(a: Batch, b: Batch) -> Batch:
        """Merge two Batches in O(|vars_a| + |vars_b|) without re-adding clauses."""
        new_vars = dict(a._variables)
        for v, c in b._variables.items():
            new_vars[v] = new_vars.get(v, 0) + c
        new_batch = object.__new__(Batch)
        new_batch._variables = new_vars
        new_batch._clauses = a._clauses | b._clauses
        new_batch.redundancy = sum(c - 1 for c in new_vars.values() if c > 1)
        return new_batch

    @property
    def variables(self) -> frozenset[variable]:
        return frozenset(self._variables)

    @property
    def variables_to_count(self):
        return self._variables

    @property
    def clauses(self) -> frozenset[Clause]:
        return frozenset(self._clauses)


class CSTNode(HRSENode):
    partition: list[Batch]
    subsumed_variables: set[variable]
    max_clause_width: int

    def __init__(self, hrse_node: HRSENode, parent: CSTNode | None = None):
        super().__init__(hrse_node.size, hrse_node.depth, parent)
        self.out_deg = hrse_node.out_deg
        self.children = list(hrse_node.children)
        self.complexity = hrse_node.complexity
        self.covered_leaves = hrse_node.covered_leaves
        self.partition = []
        self.subsumed_variables = set()
        self.max_clause_width = 0

    def add_batch(self, batch: Batch):
        """Prepend a batch; update subsumed variables and max clause width."""
        self.partition.insert(0, batch)
        self.subsumed_variables |= batch.variables
        if batch._clauses:
            self.max_clause_width = max(
                self.max_clause_width,
                max(len(c.normed_variables) for c in batch._clauses),
            )

    def set_partition(self, partition: list[Batch]):
        """Replace the partition and recompute derived attributes."""
        self.partition = partition
        self.subsumed_variables = set()
        self.max_clause_width = 0
        for batch in partition:
            self.subsumed_variables |= batch.variables
            if batch._clauses:
                self.max_clause_width = max(
                    self.max_clause_width,
                    max(len(c.normed_variables) for c in batch._clauses),
                )


# ---------- Occurrence Map & Clause Sorting ----------

def build_occurence_list(clauses: list[Clause]) -> omap:
    r"""Build the occurrence map v ↦ {C_i, C_j, ...}"""
    vars_to_clauses: omap = dict()
    for c in clauses:
        for v in c.normed_variables:
            vars_to_clauses.setdefault(v, set()).add(c)
    return vars_to_clauses


def freq(z: variable, omap: omap) -> int:
    """Number of clauses in `omap` that contain variable z."""
    clauses = omap.get(z)
    return len(clauses) if clauses is not None else 0


def conflict_deg(C: Clause, omap: omap) -> int:
    r"""Conflict degree d_i = Σ_{z∈Ĉ_i}(freq(z) − 1)."""
    return sum((freq(v, omap) - 1 for v in C.normed_variables), 0)


def redundancy_impact(C: Clause, var_set: Set[variable]) -> int:
    r"""Number of variables in C already present in `var_set` (the δ_i overlap count)."""
    return sum(1 for v in C.normed_variables if v in var_set)


def sort_clauses(clauses: list[Clause], omap: omap, ctx: NumpyContext | None = None) -> Iterator[Clause]:
    r"""Sort clauses ascending by (conflict_degree, clause_length, insertion_order).

    When `ctx` is provided (from `build_numpy_context`), uses a vectorized NumPy
    gather over a pre-built padded index matrix. Without `ctx`, falls back to a
    pure-Python on-the-fly build.
    """
    if not clauses:
        return iter(clauses)

    if ctx is not None:
        freq_sums = ctx.padded_freq[ctx.pad_idx].sum(axis=1)
        conflict_degrees = freq_sums - ctx.clause_lengths
        order = np.lexsort((ctx.clause_lengths, conflict_degrees))
        clauses[:] = [clauses[int(i)] for i in order]
        return iter(clauses)

    # Fallback: build NumPy structures on the fly (used when no pre-built ctx is available).
    all_vars = sorted({v for c in clauses for v in c.normed_variables})
    var_to_idx: dict[variable, int] = {v: i for i, v in enumerate(all_vars)}
    n_vars = len(all_vars)
    n = len(clauses)

    freq_arr = np.array([len(omap.get(v, set())) for v in all_vars], dtype=np.int64)
    max_w = max(len(c.normed_variables) for c in clauses)
    pad_idx = np.full((n, max_w), n_vars, dtype=np.intp)
    lengths = np.empty(n, dtype=np.int64)
    for i, c in enumerate(clauses):
        idxs = [var_to_idx[v] for v in c.normed_variables]
        pad_idx[i, : len(idxs)] = idxs
        lengths[i] = len(idxs)

    padded_freq = np.empty(n_vars + 1, dtype=np.int64)
    padded_freq[:n_vars] = freq_arr
    padded_freq[n_vars] = 0

    freq_sums = padded_freq[pad_idx].sum(axis=1)
    conflict_degrees = freq_sums - lengths
    order = np.lexsort((lengths, conflict_degrees))
    clauses[:] = [clauses[int(i)] for i in order]
    return iter(clauses)


# ---------- SeedGrow Algorithm ----------

def is_feasible(batch: Batch, partition: list[Batch], ancilla_budget: int) -> bool:
    r"""True when Σ_{h≤j}|β_h| + R(β_j) ≤ a_q (paper eq. 10)."""
    occupied_ancilla = sum(len(b._clauses) for b in partition)
    return occupied_ancilla + batch.redundancy <= ancilla_budget


def grow_block(
    budget: int,
    unassigned_clauses: list[Clause],
    omap: omap,
    ctx: NumpyContext | None = None,
) -> Batch:
    r"""Grow a single batch β greedily from the front of `unassigned_clauses`.

    Seeds from `unassigned_clauses[0]`, then absorbs any clause whose redundancy
    impact fits within `budget`. Removes all added clauses from `unassigned_clauses`
    in-place and returns the batch.
    """
    seed_clause = unassigned_clauses[0]
    batch = Batch({seed_clause})
    removed = {seed_clause}

    max_width = max(len(c.normed_variables) for c in unassigned_clauses)
    batch_vars = batch._variables

    # Integer bitmask: bit i set ↔ variable i is in the batch.
    # Enables O(1) overlap counting via .bit_count() instead of per-variable loops.
    batch_mask: int = ctx.clause_masks[seed_clause] if ctx is not None else 0
    next_clause_mask: int = 0

    in_queue: set[Clause] = set()
    conflict_buckets = BucketQueue[Clause](max_key=max_width)
    for c in unassigned_clauses:
        if c is seed_clause:
            continue
        if ctx is not None:
            impact = (batch_mask & ctx.clause_masks[c]).bit_count()
        else:
            impact = sum(1 for v in c.normed_variables if v in batch_vars)
        conflict_buckets.add(impact, c)
        in_queue.add(c)

    next_clause, next_impact = conflict_buckets.get_min()
    while next_clause is not None and next_impact is not None and next_impact <= budget - batch.redundancy:
        conflict_buckets.remove(next_impact, next_clause)
        in_queue.discard(next_clause)
        removed.add(next_clause)

        next_vars = next_clause.normed_variables
        if ctx is not None:
            next_clause_mask = ctx.clause_masks[next_clause]

        for v in next_vars:
            for c in omap[v]:
                if c not in in_queue:
                    continue
                if ctx is not None:
                    c_mask = ctx.clause_masks[c]
                    old_key = (batch_mask & c_mask).bit_count()
                    delta = (c_mask & next_clause_mask & ~batch_mask).bit_count()
                else:
                    old_key = sum(1 for v2 in c.normed_variables if v2 in batch_vars)
                    delta = sum(1 for v2 in c.normed_variables if v2 not in batch_vars and v2 in next_vars)
                if delta:
                    conflict_buckets.update_key(old_key + delta, old_key, c)

        batch.add_clause(next_clause)
        if ctx is not None:
            batch_mask |= next_clause_mask

        next_clause, next_impact = conflict_buckets.get_min()

    unassigned_clauses[:] = [c for c in unassigned_clauses if c not in removed]
    return batch


def merge_adjacent(partition: list[Batch], budget: int) -> list[Batch]:
    """Merge consecutive batches whenever the merged batch remains feasible.

    Iterates left-to-right; for each head batch, absorbs as many following batches
    as possible while `occupied_ancilla + merged_redundancy ≤ budget`.
    """
    if not partition:
        return []

    prefix = [0] * (len(partition) + 1)
    for i, b in enumerate(partition):
        prefix[i + 1] = prefix[i] + len(b._clauses)

    new_partition: list[Batch] = []
    for i, head_batch in enumerate(partition):
        occupied = prefix[i]

        if occupied + head_batch.redundancy > budget:
            new_partition.append(head_batch)
            continue

        acc_vars = dict(head_batch._variables)
        acc_clauses = set(head_batch._clauses)
        acc_redundancy = head_batch.redundancy
        merged = False

        for j in range(i + 1, len(partition)):
            b = partition[j]
            delta = 0
            for v, c in b._variables.items():
                existing = acc_vars.get(v, 0)
                delta += c if existing else (c - 1 if c > 1 else 0)
            tent_redundancy = acc_redundancy + delta
            if occupied + tent_redundancy > budget:
                break
            for v, c in b._variables.items():
                acc_vars[v] = acc_vars.get(v, 0) + c
            acc_clauses |= b._clauses
            acc_redundancy = tent_redundancy
            merged = True

        if not merged:
            new_partition.append(head_batch)
        else:
            new_batch = object.__new__(Batch)
            new_batch._variables = acc_vars
            new_batch._clauses = acc_clauses
            new_batch.redundancy = acc_redundancy
            new_partition.append(new_batch)

    return new_partition


def seed_grow(
    node: HRSENode,
    leaf_clauses: list[Clause],
    omap: omap,
    ctx: NumpyContext | None = None,
) -> CSTNode | None:
    r"""Cluster `leaf_clauses` into a feasible ordered partition Π (Algorithm 1).

    `leaf_clauses` are the clauses assigned to node's direct HRSE leaf children —
    one per leaf, pre-sorted by conflict degree. Budget starts at b ← a_q − m and
    grows by |β| after each batch (b ← b + |β|), ensuring all clauses are assigned.

    Returns a CSTNode with the partition set, or None if the result is infeasible.
    """
    if node.size == 0 or node.size < len(leaf_clauses):
        return None

    m = len(leaf_clauses)
    node_ctx = ctx if ctx is not None else build_numpy_context(leaf_clauses)
    ancilla = node.size
    remaining = list(leaf_clauses)
    partition: list[Batch] = []
    budget = ancilla - m       # b ← a_q − m

    while remaining:
        new_batch = grow_block(budget, remaining, omap, node_ctx)
        partition.insert(0, new_batch)          # prepend → final list is already in order
        budget += len(new_batch._clauses)       # b ← b + |β|

    merged = merge_adjacent(partition, ancilla)
    if not merged or not is_feasible(merged[-1], merged, ancilla):
        return None

    cst_node = CSTNode(node)
    cst_node.set_partition(merged)
    return cst_node


# ---------- Tree Construction ----------

def grow_cst(root: HRSENode, clauses: list[Clause]) -> CSTNode | None:
    r"""Build a CST for the HRSE tree rooted at `root`.

    Each HRSE leaf receives exactly one clause; each internal node clusters its
    direct leaf children via SeedGrow. Steps:

    1. Sort clauses globally by conflict degree.
    2. Assign one clause per leaf in pre-order DFS.
    3. Build CSTNodes top-down: leaves → singleton partitions;
       internal nodes → SeedGrow on direct leaf children, then recurse.
    """
    if not clauses:
        return None

    var_occurences = build_occurence_list(clauses)
    ctx = build_numpy_context(clauses)
    sort_clauses(clauses, var_occurences, ctx)

    leaf_clause_map = _assign_clauses_to_leaves(root, clauses)
    return _build_cst_subtree(root, None, leaf_clause_map, var_occurences, ctx)


def _assign_clauses_to_leaves(root: HRSENode, sorted_clauses: list[Clause]) -> dict[int, Clause]:
    """Map each HRSE leaf to one clause via pre-order DFS, returning id(leaf) → Clause.

    Excess leaves (when |leaves| > |clauses|) receive no entry. Using id(node) as
    the key is safe for the duration of the synchronous grow_cst call.
    """
    clause_iter = iter(sorted_clauses)
    result: dict[int, Clause] = {}

    def dfs(node: HRSENode) -> None:
        if node.is_leaf():
            try:
                result[id(node)] = next(clause_iter)
            except StopIteration:
                pass
        for child in node.children:
            dfs(child)

    dfs(root)
    return result


def _build_cst_subtree(
    hrse_node: HRSENode,
    parent_cst: CSTNode | None,
    leaf_clause_map: dict[int, Clause],
    omap: omap,
    ctx: NumpyContext | None,
) -> CSTNode | None:
    """Pre-order DFS transformer: convert HRSENodes into CSTNodes.

    Leaf HRSENodes → singleton CSTNodes (one Batch containing one Clause).
    Internal HRSENodes → SeedGrow on their direct leaf children, then recurse
    into non-leaf children. Leaf children are subsumed into the partition and
    are not kept in .children.
    """
    if hrse_node.is_leaf():
        clause = leaf_clause_map.get(id(hrse_node))
        if clause is None:
            return None
        cst_node = CSTNode(hrse_node, parent_cst)
        cst_node.set_partition([Batch({clause})])
        return cst_node

    leaf_clauses: list[Clause] = [
        clause
        for child in hrse_node.children
        if child.is_leaf()
        if (clause := leaf_clause_map.get(id(child))) is not None
    ]

    cst_node = seed_grow(hrse_node, leaf_clauses, omap, ctx) if leaf_clauses else None
    if cst_node is None:
        cst_node = CSTNode(hrse_node)
    cst_node.parent = parent_cst

    child_cst_nodes: list[CSTNode] = []
    for child in hrse_node.children:
        if not child.is_leaf():
            child_cst = _build_cst_subtree(child, cst_node, leaf_clause_map, omap, ctx)
            if child_cst is not None:  # pragma: no branch
                child_cst_nodes.append(child_cst)
    cst_node.children = child_cst_nodes  # type: ignore[assignment]

    return cst_node
