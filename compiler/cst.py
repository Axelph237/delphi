from __future__ import annotations

from .bucket_queue import BucketQueue

from typing_extensions import Set, Iterable
from collections.abc import Iterator
from compiler.hrse import HRSENode
from dataclasses import dataclass, field
from itertools import count


# ---------- CST Objects ----------

type variable = int

type omap = dict[variable, set[Clause]]   # An occurence mapping

type Partition = Iterable[Batch]

@dataclass(frozen=True)
class Clause:
    variables: frozenset[variable]


@dataclass
class Batch:
    r"""
    Batchs are greedily built, and thus should never have a clause
    retroactively removed.
    """
    _variables: dict[variable, int]
    _clauses: set[Clause]

    redundancy: int   # $\boldsymbol{R(\beta) = \sum_z{max\{k_z{\beta} - 1, 0\}}}$

    def __init__(self, clauses: Set[Clause] = set()):
        self._clauses = set()
        self._variables = dict()
        self.redundancy = 0

        for c in clauses:
            self.add_clause(c)

    def add_clause(self, clause: Clause):
        if clause in self._clauses:
            return

        self._clauses.add(clause)

        # Single pass: count redundancy impact and update variable counts simultaneously
        delta = 0
        for v in clause.variables:
            existing = self._variables.get(v, 0)
            if existing:
                delta += 1
            self._variables[v] = existing + 1
        self.redundancy += delta

    @staticmethod
    def _merge(a: Batch, b: Batch) -> Batch:
        """Merge two Batch objects in O(|vars_a| + |vars_b|) without re-adding clauses."""
        new_vars = dict(a._variables)
        for v, c in b._variables.items():
            new_vars[v] = new_vars.get(v, 0) + c
        new_batch = object.__new__(Batch)
        new_batch._variables = new_vars
        new_batch._clauses = a._clauses | b._clauses
        # max(c-1, 0) == c-1 for c>=2 only; counts are always >= 1 so skip max()
        new_batch.redundancy = sum(c - 1 for c in new_vars.values() if c > 1)
        return new_batch

    @property
    def variables(self) -> frozenset[variable]:
        return frozenset(self._variables)

    @property
    def clauses(self) -> frozenset[Clause]:
        return frozenset(self._clauses)


@dataclass
class CSTNode:
    shadows: HRSENode   # The node in the HRSE tree that this CSTNode represents
    parent: CSTNode | None

    # CST node specific members
    max_clause_width: int   # $\boldsymbol{k = \text{max}_i|\widehat{C_i}|}$ is the max normalized clause width
    subsumed_variables: set[int]
    partition: Partition   # List of batches in evaluation order

    def __init__(self, shadows: HRSENode, parent: CSTNode | None = None):
        self.shadows = shadows
        self.parent = parent

        self.subsumed_variables = set()
        self.max_clause_width = 0
        self.partition = []

    def set_partition(self, partition: Partition):
        self.partition = partition
        self.subsumed_variables = set()
        for b in partition:
            self.subsumed_variables |= b._variables.keys()

    def add_batch(self, batch: Batch):
        """Adds a batch to the CST node, updating the set of subsumed variables"""
        self.partition.insert(0, batch)
        self.subsumed_variables |= batch._variables.keys()
        if batch._clauses:
            self.max_clause_width = max(
                self.max_clause_width,
                max(len(c.variables) for c in batch._clauses)
            )


# ---------- SeedGrow Heuristic ----------

r"""
In the original paper, clauses were expected to be predistributed across HRSE leaves. However,
neither paper gives a proper ordering for this. Instead, here, we use SeedGrow on a global
clause set $\boldsymbol{R = \{C_1, ..., C_n\}}$ where $\boldsymbol{n}$ is the size of the CNF.
"""

def build_occurence_list(clauses: list[Clause]) -> omap:
    r"""Creates an occurence mapping of $\boldsymbol{v \mapsto [C_i, C_j, ...]}$"""
    vars_to_clauses: omap = dict()

    for c in clauses:
        for v in c.variables:
            vars_to_clauses.setdefault(v, set()).add(c)

    return vars_to_clauses


def freq(z: variable, omap: omap) -> int:
    r"""Returns the frequency of a variable $\boldsymbol{z \in \text{omap}}$"""
    clauses = omap.get(z)
    return len(clauses) if clauses is not None else 0


def conflict_deg(C: Clause, omap: omap) -> int:
    r"""
    Returns the conflict degree $\boldsymbol{d_i}$ of a clause $\boldsymbol{C_i}$, given by

    $\quad\boldsymbol{d_i=\sum_{z\in\widehat{C_i}}(v_z - 1)}$

    """
    return sum((freq(v, omap) - 1 for v in C.variables), 0)


def redundancy_impact(C: Clause, var_set: Set[variable]) -> int:
    r"""
    Returns the redundancy impact of a clause $\boldsymbol{C_i}$ given a set of variables $\boldsymbol{U}$.
    This is an adjustment of the factor $\boldsymbol{\delta_i}$ minimzation to a static difference.

    $\quad\boldsymbol{\delta_i = |\widehat{C_i} \cap U|}$

    """
    return sum(1 for v in C.variables if v in var_set)


def is_feasible(batch: Batch, partition: Partition, ancilla_budget: int) -> bool:
    r"""
    A given batch is feasible in a partition if the sum of all outbit ancilla used by previous
    clauses, and required ancilla for redundant variables in this batch, are less than the
    total ancilla budget.

    $\quad\boldsymbol{\sum_{h \leq j}|\beta_{h}| + R(\beta_j) \leq a_q}$

    """
    occupied_ancilla = sum(len(b._clauses) for b in partition)
    return occupied_ancilla + batch.redundancy < ancilla_budget


def sort_clauses(clauses: list[Clause], omap: omap) -> Iterator[Clause]:
    r"""
    Sorts clauses lowest conflict. Conflict is resolved by $\boldsymbol{( d_i, |\widehat{C_i}, i|)}$
    Or, in other words, is broken  by the following priorities:
    1. Conflict degree $\boldsymbol{d_i}$
    2. Clause length $\boldsymbol{|\widehat{C_i}|}$
    3. Insertion order $\boldsymbol{i}$
    """
    clauses.sort(key=lambda c: (
        conflict_deg(c, omap),
        len(c.variables)))

    return iter(clauses)


def grow_cst(root: HRSENode, clauses: list[Clause]):
    r""" Let:
        $\boldsymbol{R}$ be the set of clauses $\boldsymbol{[C_1, ..., C_m]}$
        $\boldsymbol{k}$ be the max clause width $\boldsymbol{\text{arg max}_{i\in R}|\widehat{C_i}|}$
    """

    # 1. Pre-heuristic setup
    #   Initialize variable occurence
    var_occurences = build_occurence_list(clauses)
    #   Sort clauses by the conflict degrees $\boldsymbol{d_i}$
    sort_clauses(clauses, var_occurences)


# ---------- Paper Defined Methods ----------


def seed_grow(node: HRSENode, remaining_clauses: list[Clause], omap: omap) -> CSTNode | None:
    r"""
    Greedily builds a partition $\boldsymbol{\Pi}$ at a compute node $\boldsymbol{v}$ by iteratively adding the clause
    of lowest redundancy impact $\boldsymbol{\delta_i = |\widehat{C_i} \cap U|}$ without exceeding the current allowance
    $\boldsymbol{a_q - i}$
    """
    if node.size == 0:
        return None

    num_leaves = sum((1 for n in node.children if n.is_leaf()))   # $\boldsymbol{m}$
    if node.size < num_leaves:
        return None

    partition: Partition = []
    budget = node.size - num_leaves

    while budget > 0 and remaining_clauses:
        new_batch = grow_block(budget, remaining_clauses, omap)
        partition.insert(0, new_batch)   # prepend, removing need to reverse order
        budget += len(new_batch._clauses)   # $\boldsymbol{b \leftarrow b + |\beta|}$

    merged_partition = merge_adjacent(partition, budget)
    if not merged_partition:
        return None
    if is_feasible(merged_partition[-1], merged_partition, budget):   # One final feasibility check
        cst_node = CSTNode(node)
        cst_node.set_partition(merged_partition)
        return cst_node
    else:
        return None


def merge_adjacent(partition: list, budget: int) -> list:
    """Merges adjacent batches whenever possible in a given partition"""
    if not partition:
        return []

    # Precompute prefix sums of batch clause counts to avoid recomputing per iteration
    prefix = [0] * (len(partition) + 1)
    for i, b in enumerate(partition):
        prefix[i + 1] = prefix[i] + len(b._clauses)

    new_partition = []
    for i, head_batch in enumerate(partition):
        occupied = prefix[i]  # ancilla occupied by all batches before position i

        # Check if head_batch itself is infeasible at position i (mirrors original j=i iteration)
        if occupied + head_batch.redundancy >= budget:
            new_partition.append(head_batch)
            continue

        # Accumulate merges in-place: one dict/set copy per i instead of one per accepted merge.
        # This reduces total dict-copy work from O(n²·k) to O(n·k).
        acc_vars = dict(head_batch._variables)
        acc_clauses = set(head_batch._clauses)
        acc_redundancy = head_batch.redundancy
        merged = False

        for j in range(i + 1, len(partition)):
            b = partition[j]
            # Compute tentative redundancy: merge b._variables into acc_vars incrementally
            delta = 0
            for v, c in b._variables.items():
                existing = acc_vars.get(v, 0)
                delta += c if existing else (c - 1 if c > 1 else 0)
            tent_redundancy = acc_redundancy + delta
            if occupied + tent_redundancy >= budget:
                break
            # Accept: update accumulator in-place
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



def grow_block(budget: int, unassigned_clauses: list[Clause], omap: omap) -> Batch:
    r""" Grows a single batch $\boldsymbol{\beta}$ in a CST node """

    # 1. Batch setup. Get seed and initialize batch variables
    seed_clause = unassigned_clauses[0]  # peek without popping; removed via batch filter below
    batch = Batch({seed_clause})   # $\boldsymbol{\beta \leftarrow {s}};\ U \leftarrow \widehat{C_s};\ r \leftarrow 0$
    removed = {seed_clause}

    # Bound BucketQueue by max clause width (O(k) buckets instead of O(n))
    max_width = max(len(c.variables) for c in unassigned_clauses)

    # 2. Batch growth setup: batch_vars is the live dict (mutations from add_clause are visible)
    batch_vars = batch._variables

    #   Create initial conflict buckets of remaining clauses
    in_queue: set[Clause] = set()
    conflict_buckets = BucketQueue[Clause](max_key=max_width)
    for c in unassigned_clauses:
        if c is seed_clause:
            continue
        impact = 0
        for v in c.variables:
            if v in batch_vars:
                impact += 1
        conflict_buckets.add(impact, c)
        in_queue.add(c)
    next_clause, next_impact = conflict_buckets.get_min()
    while next_clause is not None and next_impact is not None and next_impact <= budget - batch.redundancy:
        # 2.1 Remove clause from conflict buckets
        conflict_buckets.remove(next_impact, next_clause)
        in_queue.discard(next_clause)
        removed.add(next_clause)

        # 2.2 Update conflict degrees incrementally (avoid frozenset for new_U)
        next_vars = next_clause.variables

        for v in next_vars:
            for c in omap[v]:
                if c not in in_queue:
                    continue
                # Single scan: compute old_key and delta (new vars from next_vars) simultaneously
                old_key = 0
                delta = 0
                for v2 in c.variables:
                    if v2 in batch_vars:
                        old_key += 1
                    elif v2 in next_vars:
                        delta += 1
                if delta:
                    conflict_buckets.update_key(old_key + delta, old_key, c)

        # 2.3 Add clause to batch
        batch.add_clause(next_clause)

        # 2.4 Get next set of clauses
        next_clause, next_impact = conflict_buckets.get_min()

    # Remove all assigned clauses from the shared list in a single O(n) pass
    unassigned_clauses[:] = [c for c in unassigned_clauses if c not in removed]
    return batch
