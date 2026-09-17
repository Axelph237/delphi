from __future__ import annotations

from typing_extensions import Set, Iterable
from .bucket_queue import BucketQueue
from _typeshed import SupportsNext
from compiler.hrse import HRSENode
from dataclasses import dataclass, field
from itertools import count, islice


# ---------- CST Objects ----------

_clause_id = count()
def next_clause_id() -> int:
    return next(_clause_id)

type variable = int

type omap = dict[variable, set[Clause]]   # An occurence mapping

type Partition = Iterable[Batch]

@dataclass(frozen=True)
class Clause:
    variables: frozenset[variable]
    id: int = field(default_factory=next_clause_id)


@dataclass
class Batch:
    """
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
        self._clauses.add(clause)

        self.redundancy += redundancy_impact(clause, self.variables)

        for v in clause.variables:
            count = self._variables.get(v, 0) + 1
            self._variables[v] = count

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
        self.partition = []

    def set_partition(self, partition: Partition):
        self.partition = partition
        for b in partition:
            self.subsumed_variables |= b.variables

    def add_batch(self, batch: Batch):
        """Adds a batch to the CST node, updating the set of subsumed variables"""
        # TODO: Make order the batches correctly
        self.partition.insert(0, batch)
        self.subsumed_variables |= batch.variables


# ---------- SeedGrow Heuristic ----------

"""
In the original paper, clauses were expected to be predistributed across HRSE leaves. However,
neither paper gives a proper ordering for this. Instead, here, we use SeedGrow on a global
clause set $\boldsymbol{R = \{C_1, ..., C_n\}}$ where $\boldsymbol{n}$ is the size of the CNF.
"""

def build_occurence_list(clauses: list[Clause]) -> omap:
    """Creates an occurence mapping of $\boldsymbol{v \mapsto [C_i, C_j, ...]}$"""
    vars_to_clauses: omap = dict()
    
    for c in clauses:
        for v in c.variables:
            vars_to_clauses.setdefault(v, set()).add(c)

    return vars_to_clauses


def freq(z: variable, omap: omap) -> int:
    """Returns the frequency of a variable $\boldsymbol{z \in \text{omap}}$"""
    clauses = omap.get(z)
    return len(clauses) if clauses is not None else 0


def conflict_deg(C: Clause, omap: omap) -> int:
    """
    Returns the conflict degree $\boldsymbol{d_i}$ of a clause $\boldsymbol{C_i}$, given by

    $\quad\boldsymbol{d_i=\sum_{z\in\widehat{C_i}}(v_z - 1)}$

    """
    return sum((freq(v, omap) - 1 for v in C.variables), 0)


def redundancy_impact(C: Clause, var_set: Set[variable]) -> int:
    """
    Returns the redundancy impact of a clause $\boldsymbol{C_i}$ given a set of variables $\boldsymbol{U}$.
    This is an adjustment of the factor $\boldsymbol{\delta_i}$ minimzation to a static difference.

    $\quad\boldsymbol{\gamma_i = |\widehat{C_i} \setminus U|}$

    """
    return len(C.variables - var_set)


def is_feasible(batch: Batch, partition: Partition, ancilla_budget: int) -> bool:
    """
    A given batch is feasible in a partition if the sum of all outbit ancilla used by previous
    clauses, and required ancilla for redundant variables in this batch, are less than the 
    total ancilla budget.

    $\quad\boldsymbol{\sum_{h \leq j}|\beta_{h}| + R(\beta_j) \leq a_q}$

    """
    occupied_ancilla = sum((len(b.clauses) for i, b in enumerate(partition)), 0) 
    return occupied_ancilla + batch.redundancy < ancilla_budget


def sort_clauses(clauses: list[Clause], omap: omap) -> SupportsNext[Clause]:
    """
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
    """ Let:
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
    """
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

    while budget > 0:
        new_batch = grow_block(budget, remaining_clauses, omap)
        partition.insert(0, new_batch)   # prepend, removing need to reverse order
        budget += len(new_batch.clauses)   # $\boldsymbol{b \leftarrow b + |\beta|}$
    
    merged_partition = merge_adjacent(partition, budget)
    if is_feasible(merged_partition[-1], merged_partition, budget):   # One final feasibility check
        cst_node = CSTNode(node)
        cst_node.set_partition(merged_partition)
        return cst_node
    else:
        return None
    

def merge_adjacent(partition: Partition, budget: int):
    """ Merges adjacent batches whenever possible in a given partition """

    new_partition = []
    for i, head_batch in enumerate(partition):

        merged_batch = head_batch
        for j in range(i, len(partition)):
            adjacent_batch = partition[j]
            # Propose a new batch of the two adjoining batches
            proposed_batch = Batch(merged_batch.clauses | adjacent_batch.clauses)

            if not is_feasible(proposed_batch, islice(partition, 0, i), budget):
                break
            merged_batch = proposed_batch
        new_partition.append(merged_batch)
    return new_partition
        


def grow_block(budget: int, unassigned_clauses: list[Clause], omap: omap):
    """ Grows a single batch $\boldsymbol{\beta}$ in a CST node """

    # 1. Batch setup. Get seed and initialize batch variables
    seed_clause = unassigned_clauses.pop(0)
    batch = Batch({seed_clause})   # $\boldsymbol{\beta \leftarrow {s}};\ U \leftarrow \widehat{C_s};\ r \leftarrow 0$

    #   Create initial conflict buckets of remaining clauses
    conflict_buckets = BucketQueue[Clause](max_key=len(omap.keys()))
    for c in unassigned_clauses:
        conflict_buckets.add(redundancy_impact(c, batch.variables), c)
    
    # 2. Batch growth. Iteratively add clauses to the batch
    valid_clause = lambda impact: impact <= budget - batch.redundancy

    next_clause, next_impact = conflict_buckets.get_min()
    while next_clause is not None and next_impact is not None and valid_clause(next_impact):
        # 2.1 Remove clause from conflict buckets
        conflict_buckets.remove(next_impact, next_clause)

        # 2.2 Update conflict degrees
        new_U = batch.variables | next_clause.variables   # It's the new you!

        for v in next_clause.variables:
            for c in omap[v]:
                if c is next_clause:
                    continue
                new_key = redundancy_impact(c, new_U)
                old_key = redundancy_impact(c, batch.variables)
                conflict_buckets.update_key(new_key, old_key, c)

        # 2.3 Add clause to batch
        batch.add_clause(next_clause)

        # 2.4 Get next set of clauses
        next_clause, next_impact = conflict_buckets.get_min()

    return batch
    
    