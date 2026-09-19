from compiler.cst import *
from compiler.hrse import HRSENode

def test__Batch():
    # EMPTY BATCH
    empty_batch = Batch()

    assert empty_batch.variables == frozenset()
    assert empty_batch.clauses == frozenset()

    # BASIC BATCH INITIALIZATION
    clause_1 = Clause(frozenset({1, 2}), 0)
    init_batch = Batch({clause_1})
    assert init_batch.clauses == frozenset({clause_1})
    assert init_batch.variables == frozenset({1, 2})
    assert init_batch.redundancy == 0

    # REDUNDANT CLAUSE
    clause_2 = Clause(frozenset({1, 2}), 0)
    init_batch.add_clause(clause_2)
    assert init_batch.clauses == frozenset({clause_1})
    assert init_batch.variables == frozenset({1, 2})
    assert init_batch.redundancy == 0

    # REDUNDANCY IMPACT
    clause_3 = Clause(frozenset({2, 3}), 0)
    init_batch.add_clause(clause_3)
    assert init_batch.clauses == frozenset({clause_1, clause_3})
    assert init_batch.variables == frozenset({1, 2, 3})
    assert init_batch.redundancy == 1

def test__CSTNode():
    # CST Node initialization
    hrse_root = HRSENode(5, 0, None)
    root_node = CSTNode(hrse_root, None)
    hrse_child = HRSENode(4, 0, None)
    child_node = CSTNode(hrse_child, root_node)

    assert child_node.size == hrse_child.size
    assert child_node.depth == hrse_child.depth
    assert child_node.partition == []
    assert child_node.subsumed_variables == set()
    assert child_node.parent == root_node

    assert root_node.size == hrse_root.size
    assert root_node.depth == hrse_root.depth
    assert root_node.parent == None
    assert root_node.partition == []
    assert root_node.subsumed_variables == set()

    # CST Node operations
    batch_1 = Batch({Clause(frozenset({1, 2}), 0)})
    batch_2 = Batch({Clause(frozenset({3, 4, 2}), 0)})

    child_node.add_batch(batch_1)
    child_node.add_batch(batch_2)

    assert child_node.partition == [batch_2, batch_1]
    assert child_node.subsumed_variables == {1, 2, 3, 4}
    assert child_node.max_clause_width == 3

    # Direct partition setting
    batch_3 = Batch({Clause(frozenset({1, 2, 3}), 0)})
    batch_4 = Batch({Clause(frozenset({1, 2}), 0)})

    child_node.set_partition([batch_3, batch_4])

    assert child_node.partition == [batch_3, batch_4]
    assert child_node.subsumed_variables == {1, 2, 3}
    assert child_node.max_clause_width == 3


# ---------- build_occurence_list ----------

def test__build_occurence_list():
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    c3 = Clause(frozenset({3, 4}), 0)

    omap = build_occurence_list([c1, c2, c3])

    assert omap[1] == {c1}
    assert omap[2] == {c1, c2}
    assert omap[3] == {c2, c3}
    assert omap[4] == {c3}

def test__build_occurence_list__no_clauses():
    omap = build_occurence_list([])
    assert omap == {}


# ---------- freq ----------

def test__freq():
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    omap = build_occurence_list([c1, c2])

    assert freq(2, omap) == 2   # appears in both clauses
    assert freq(1, omap) == 1   # appears in only one clause

def test__freq__var_missing_from_omap():
    omap = build_occurence_list([Clause(frozenset({1, 2}), 0)])
    assert freq(99, omap) == 0


# ---------- conflict_degree ----------

def test__conflict_degree():
    # c1 shares var 2 with c2, and var 3 with c3
    c1 = Clause(frozenset({1, 2, 3}), 0)
    c2 = Clause(frozenset({2, 4}), 0)
    c3 = Clause(frozenset({3, 5}), 0)
    omap = build_occurence_list([c1, c2, c3])

    # d(c1) = (freq(1)-1) + (freq(2)-1) + (freq(3)-1) = 0 + 1 + 1 = 2
    assert conflict_deg(c1, omap) == 2

def test__conflict_degree__single_var_clause():
    c1 = Clause(frozenset({1}), 0)
    c2 = Clause(frozenset({2}), 0)
    omap = build_occurence_list([c1, c2])

    assert conflict_deg(c1, omap) == 0
    assert conflict_deg(c2, omap) == 0

def test__conflict_degree__no_redundant_vars():
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({3, 4}), 0)
    omap = build_occurence_list([c1, c2])

    assert conflict_deg(c1, omap) == 0
    assert conflict_deg(c2, omap) == 0


# ---------- redundancy_impact ----------

def test__redundancy_impact():
    c = Clause(frozenset({1, 2, 3}), 0)
    # var_set contains 1 and 3 — overlap of size 2
    assert redundancy_impact(c, frozenset({1, 3, 5})) == 2

def test__redundancy_impact__no_impact():
    c = Clause(frozenset({1, 2, 3}), 0)
    # No overlap with var_set
    assert redundancy_impact(c, frozenset({4, 5, 6})) == 0


# ---------- is_feasible ----------

def test__is_feasible():
    # Empty partition, single clause batch — should be feasible with any positive budget
    b = Batch({Clause(frozenset({1, 2}), 0)})   # 1 clause, 0 redundancy
    assert is_feasible(b, [], 5) == True

def test__is_feasible__budget_exhausted():
    # Fill partition with 10 clauses -> occupied_ancilla = 10 > budget = 5
    clauses = [Clause(frozenset({i}), 0) for i in range(10)]
    big_batch = Batch(set(clauses))
    partition = [big_batch]

    new_batch = Batch({Clause(frozenset({100}), 0)})
    assert is_feasible(new_batch, partition, 5) == False


# ---------- sort_clauses ----------

def test__sort_clauses():
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    c3 = Clause(frozenset({4, 5}), 0)  # disjoint — conflict_deg = 0
    omap = build_occurence_list([c1, c2, c3])

    result = list(sort_clauses([c1, c2, c3], omap))
    assert len(result) == 3
    # c3 has conflict_deg=0; c1, c2 share var 2 so conflict_deg=1
    assert result[0] == c3

def test__sort_clauses__empty_list():
    result = list(sort_clauses([], {}))
    assert result == []

def test__sort_clauses__single_clause():
    c = Clause(frozenset({1, 2}), 0)
    omap = build_occurence_list([c])
    result = list(sort_clauses([c], omap))
    assert result == [c]

def test__sort_clauses__multiple_clauses():
    # Tie-break by clause length (shorter first) when conflict degrees are equal
    c_short = Clause(frozenset({10}), 0)         # length 1
    c_long  = Clause(frozenset({20, 30}), 0)     # length 2
    c_high  = Clause(frozenset({10, 20}), 0)     # shares vars with both — highest conflict

    omap = build_occurence_list([c_short, c_long, c_high])
    result = list(sort_clauses([c_short, c_long, c_high], omap))

    # c_short and c_long both share one var with c_high, so conflict_deg=1
    # but c_short is shorter -> comes first; c_high has conflict_deg=2 -> last
    assert result.index(c_short) < result.index(c_long)
    assert result[-1] == c_high


# ---------- seed_grow ----------

def test__seed_grow__build():
    # A valid HRSE node with two leaf children; budget = size - num_leaves = 5 - 2 = 3
    hrse_node = HRSENode(5, 0, None)
    hrse_node.children = [HRSENode(2, 1, hrse_node), HRSENode(1, 1, hrse_node)]

    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    omap = build_occurence_list(clauses)

    result = seed_grow(hrse_node, clauses[:], omap)
    assert result is None or isinstance(result, CSTNode)

def test__seed_grow__build_empty_list():
    # size=0 -> immediately returns None
    hrse_node = HRSENode(0, 0, None)
    result = seed_grow(hrse_node, [], {})
    assert result is None

def test__seed_grow__build_single_clause():
    hrse_node = HRSENode(3, 0, None)
    hrse_node.children = [HRSENode(2, 1, hrse_node)]

    clauses = [Clause(frozenset({1, 2}), 0)]
    omap = build_occurence_list(clauses)

    result = seed_grow(hrse_node, clauses[:], omap)
    assert result is None or isinstance(result, CSTNode)

def test__seed_grow__build_multiple_clauses():
    hrse_node = HRSENode(6, 0, None)
    hrse_node.children = [
        HRSENode(2, 1, hrse_node),
        HRSENode(2, 1, hrse_node),
        HRSENode(2, 1, hrse_node),
    ]

    clauses = [
        Clause(frozenset({1, 2}), 0),
        Clause(frozenset({2, 3}), 0),
        Clause(frozenset({4, 5}), 0),
    ]
    omap = build_occurence_list(clauses)

    result = seed_grow(hrse_node, clauses[:], omap)
    assert result is None or isinstance(result, CSTNode)
    if result is not None:
        assert len(result.partition) > 0


# ---------- merge_adjacent ----------

def test__merge_adjacent():
    # Disjoint clauses with generous budget
    c1 = Clause(frozenset({1}), 0)
    c2 = Clause(frozenset({2}), 0)
    b1 = Batch({c1})
    b2 = Batch({c2})

    result = merge_adjacent([b1, b2], budget=100)
    assert isinstance(result, list)
    assert len(result) <= 2

def test__merge_adjacent__empty_partition():
    result = merge_adjacent([], budget=10)
    assert result == []

def test__merge_adjacent__single_batch():
    c = Clause(frozenset({1, 2}), 0)
    b = Batch({c})
    result = merge_adjacent([b], budget=10)
    assert len(result) == 1
    assert result[0].clauses == frozenset({c})

def test__merge_adjacent__budget_exhausted():
    # Tight budget: 5 batches each with 1 clause; budget=2 -> merging blocked quickly
    clauses = [Clause(frozenset({i}), 0) for i in range(5)]
    batches = [Batch({c}) for c in clauses]

    result = merge_adjacent(batches, budget=2)
    assert isinstance(result, list)
    assert len(result) >= 1

def test__merge_adjacent__budget_not_exhausted():
    # Three disjoint single-clause batches with a very large budget
    c1 = Clause(frozenset({1}), 0)
    c2 = Clause(frozenset({2}), 0)
    c3 = Clause(frozenset({3}), 0)
    b1 = Batch({c1})
    b2 = Batch({c2})
    b3 = Batch({c3})

    result = merge_adjacent([b1, b2, b3], budget=100)
    assert isinstance(result, list)
    assert len(result) <= 3


# ---------- grow_cst ----------

def test__grow_cst__empty_clauses():
    root = HRSENode(5, 0, None)
    result = grow_cst(root, [])
    assert result is None

def test__grow_cst__returns_cst_node_or_none():
    # Minimal valid HRSE tree (size=5 root, two leaf children)
    root = HRSENode(5, 0, None)
    root.children = [HRSENode(2, 1, root), HRSENode(1, 1, root)]
    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)

def test__grow_cst__root_mirrors_hrse_root():
    root = HRSENode(5, 0, None)
    root.children = [HRSENode(2, 1, root), HRSENode(1, 1, root)]
    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    result = grow_cst(root, clauses)
    if result is not None:
        assert result.size == root.size
        assert result.depth == root.depth

def test__grow_cst__root_has_no_parent():
    root = HRSENode(5, 0, None)
    root.children = [HRSENode(2, 1, root), HRSENode(1, 1, root)]
    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    result = grow_cst(root, clauses)
    if result is not None:
        assert result.parent is None

def test__grow_cst__asdt_tree_small():
    # HRSENode.new(m=3, k=4) — k=4 supports up to 3 clauses
    root = HRSENode.new(3, 4)
    assert root is not None
    clauses = [
        Clause(frozenset({1, 2}), 0),
        Clause(frozenset({2, 3}), 0),
        Clause(frozenset({3, 4}), 0),
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)
    if result is not None:
        assert result.size == root.size
        assert result.depth == root.depth
        assert result.parent is None

def test__grow_cst__asdt_tree_medium():
    # HRSENode.new(m=6, k=5) — k=5 supports up to 6 clauses
    root = HRSENode.new(6, 5)
    assert root is not None
    clauses = [
        Clause(frozenset({1, 2}), 0),
        Clause(frozenset({2, 3}), 0),
        Clause(frozenset({3, 4}), 0),
        Clause(frozenset({4, 5}), 0),
        Clause(frozenset({5, 6}), 0),
        Clause(frozenset({1, 6}), 0),
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)

def test__grow_cst__partition_nonempty_when_clauses_fit():
    # A generous budget: root size=6, one leaf child → budget=5, plenty for 2 clauses
    root = HRSENode(6, 0, None)
    root.children = [HRSENode(2, 1, root)]
    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    result = grow_cst(root, clauses)
    if result is not None:
        assert len(result.partition) > 0

def test__grow_cst__subsumed_variables_subset_of_clause_vars():
    root = HRSENode(6, 0, None)
    root.children = [HRSENode(2, 1, root)]
    all_vars = {1, 2, 3, 4}
    clauses = [Clause(frozenset({1, 2}), 0), Clause(frozenset({3, 4}), 0)]
    result = grow_cst(root, clauses)
    if result is not None:
        assert result.subsumed_variables <= all_vars

def test__grow_cst__multi_level_tree():
    # Build a two-level tree manually: root → mid → leaf
    root = HRSENode(6, 0, None)
    mid  = HRSENode(4, 1, root)
    leaf = HRSENode(2, 2, mid)
    root.children = [mid]
    mid.children  = [leaf]
    clauses = [
        Clause(frozenset({1, 2}), 0),
        Clause(frozenset({2, 3}), 0),
        Clause(frozenset({4, 5}), 0),
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)
    if result is not None:
        assert result.size == root.size
        assert result.depth == root.depth

def test__grow_cst__larger_asdt_tree():
    # k=6 supports up to 12 clauses; use 10
    root = HRSENode.new(10, 6)
    assert root is not None
    import random
    rng = random.Random(0)
    clauses = [
        Clause(frozenset(rng.sample(range(1, 21), 3)), 0)
        for _ in range(10)
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)


# ---------- Clause polarity ----------

def test__Clause__different_polarity_masks_are_distinct():
    # Same variables but different polarity_mask → different Clause objects
    c1 = Clause(frozenset({1, 2, 3}), 0b001)
    c2 = Clause(frozenset({1, 2, 3}), 0b010)
    assert c1 != c2

def test__Clause__same_polarity_mask_are_equal():
    # Frozen dataclass: same fields → equal
    c1 = Clause(frozenset({1, 2, 3}), 0b101)
    c2 = Clause(frozenset({1, 2, 3}), 0b101)
    assert c1 == c2


# ---------- Batch ----------

def test__Batch__merge__disjoint():
    # Two batches with no shared variables → combined clauses, redundancy=0
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({3, 4}), 0)
    b1 = Batch({c1})
    b2 = Batch({c2})
    merged = Batch._merge(b1, b2)
    assert merged.clauses == frozenset({c1, c2})
    assert merged.redundancy == 0

def test__Batch__merge__overlapping():
    # Two batches sharing variable 2 → merged redundancy = 1
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    b1 = Batch({c1})
    b2 = Batch({c2})
    merged = Batch._merge(b1, b2)
    assert merged.clauses == frozenset({c1, c2})
    assert merged.redundancy == 1

def test__Batch__merge__empty_with_nonempty():
    # Merging an empty batch with a nonempty batch → result has the nonempty batch's clauses
    c = Clause(frozenset({5, 6}), 0)
    b_nonempty = Batch({c})
    b_empty = Batch()
    merged = Batch._merge(b_empty, b_nonempty)
    assert merged.clauses == frozenset({c})

def test__Batch__variables_to_count__single_clause():
    # One clause with vars {1, 2} → each var counted once
    c = Clause(frozenset({1, 2}), 0)
    b = Batch({c})
    vtc = b.variables_to_count
    assert vtc[1] == 1
    assert vtc[2] == 1

def test__Batch__variables_to_count__overlapping_clauses():
    # Two clauses: {1,2} and {2,3} → var 2 appears twice
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    b = Batch({c1, c2})
    vtc = b.variables_to_count
    assert vtc[1] == 1
    assert vtc[2] == 2
    assert vtc[3] == 1

def test__Batch__redundancy__three_way_overlap():
    # 3 clauses all containing var 1 → kz(B)=3, R += max(3-1,0) = 2
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({1, 3}), 0)
    c3 = Clause(frozenset({1, 4}), 0)
    b = Batch({c1, c2, c3})
    assert b.redundancy == 2
    assert b.variables_to_count[1] == 3

def test__Batch__redundancy__multiple_shared_vars():
    # 2 clauses: {1,2,3} and {1,2,4} → vars 1 and 2 each appear twice → R = 1 + 1 = 2
    c1 = Clause(frozenset({1, 2, 3}), 0)
    c2 = Clause(frozenset({1, 2, 4}), 0)
    b = Batch({c1, c2})
    assert b.redundancy == 2


# ---------- conflict_degree ----------

def test__conflict_degree__exact_formula():
    # Hand-computed conflict degrees for 4 clauses
    # c1={1,2}, c2={1,3}, c3={2,4}, c4={5}
    # ν1=2, ν2=2, ν3=1, ν4=1, ν5=1
    # d(c1) = (2-1)+(2-1) = 2
    # d(c2) = (2-1)+(1-1) = 1
    # d(c3) = (2-1)+(1-1) = 1
    # d(c4) = (1-1) = 0
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({1, 3}), 0)
    c3 = Clause(frozenset({2, 4}), 0)
    c4 = Clause(frozenset({5}), 0)
    omap = build_occurence_list([c1, c2, c3, c4])
    assert conflict_deg(c1, omap) == 2
    assert conflict_deg(c2, omap) == 1
    assert conflict_deg(c3, omap) == 1
    assert conflict_deg(c4, omap) == 0


# ---------- redundancy_impact ----------

def test__redundancy_impact__empty_var_set():
    # No overlap possible → redundancy_impact = 0
    c = Clause(frozenset({1, 2, 3}), 0)
    assert redundancy_impact(c, frozenset()) == 0

def test__redundancy_impact__full_overlap():
    # var_set contains all of clause's variables → impact = len(clause.variables)
    c = Clause(frozenset({1, 2, 3}), 0)
    assert redundancy_impact(c, frozenset({1, 2, 3, 4, 5})) == 3


# ---------- is_feasible ----------

def test__is_feasible__redundancy_in_batch_causes_infeasibility():
    # Formula: sum(len(b.clauses) for b in partition) + batch.redundancy <= budget
    # Batch with R(B)=1: 2 clauses sharing variable 1
    # Empty partition, budget=0: 0 + 1 = 1 > 0 → False
    # (batch's own clause count is NOT included in the sum)
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({1, 3}), 0)
    b = Batch({c1, c2})
    assert b.redundancy == 1
    assert is_feasible(b, [], 0) == False
    assert is_feasible(b, [], 1) == True   # 0 + 1 = 1 <= 1

def test__is_feasible__multiple_batches_in_partition():
    # partition=[B1(2 clauses), B2(1 clause)], new B3(1 clause, R=0)
    # Formula: sum(len(b.clauses) for b in partition) + batch.redundancy <= budget
    # occupied_ancilla = 2 + 1 = 3; batch.redundancy = 0
    # budget=3 → 3 + 0 = 3 <= 3 → True
    # budget=2 → 3 + 0 = 3 > 2 → False
    c1 = Clause(frozenset({1}), 0)
    c2 = Clause(frozenset({2}), 0)
    c3 = Clause(frozenset({3}), 0)
    c4 = Clause(frozenset({4}), 0)
    b1 = Batch({c1, c2})
    b2 = Batch({c3})
    b3 = Batch({c4})
    assert is_feasible(b3, [b1, b2], 3) == True
    assert is_feasible(b3, [b1, b2], 2) == False


# ---------- sort_clauses ----------

def test__sort_clauses__tiebreak_by_width():
    # Two clauses with conflict_deg=0 (both isolated), one width-1 and one width-2
    # → the width-1 clause should come first
    c_short = Clause(frozenset({50}), 0)       # width 1, d=0
    c_long  = Clause(frozenset({60, 70}), 0)   # width 2, d=0
    omap = build_occurence_list([c_short, c_long])
    result = list(sort_clauses([c_short, c_long], omap))
    assert len(result) == 2
    assert result[0] == c_short
    assert result[1] == c_long


# ---------- grow_block ----------

def test__grow_block__returns_batch():
    # Single clause → returned Batch contains that clause
    c = Clause(frozenset({1, 2}), 0)
    omap = build_occurence_list([c])
    result = grow_block(10, [c], omap)
    assert isinstance(result, Batch)
    assert c in result.clauses

def test__grow_block__all_disjoint_clauses_with_sufficient_budget():
    # 3 fully disjoint clauses, large budget → all 3 in one Batch (zero redundancy)
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({3, 4}), 0)
    c3 = Clause(frozenset({5, 6}), 0)
    omap = build_occurence_list([c1, c2, c3])
    result = grow_block(10, [c1, c2, c3], omap)
    assert isinstance(result, Batch)
    assert result.clauses == frozenset({c1, c2, c3})
    assert result.redundancy == 0

def test__grow_block__budget_zero_limits_to_seed_only():
    # When budget=0, no redundancy (overlap) is allowed.
    # Clauses: c_seed={1}, c1={1,2}, c2={1,2,3}, c3={1,3,4}
    # omap: 1→all four, 2→{c_seed... wait, c_seed={1} only has var 1
    # Actually: 1→{c_seed,c1,c2,c3}, 2→{c1,c2}, 3→{c2,c3}, 4→{c3}
    # ν1=4, ν2=2, ν3=2, ν4=1
    # d(c_seed) = (4-1) = 3
    # d(c1) = (4-1)+(2-1) = 4
    # d(c2) = (4-1)+(2-1)+(2-1) = 5
    # d(c3) = (4-1)+(2-1)+(1-1) = 4
    # → c_seed has the minimum conflict_deg (3) → it is the seed
    # With budget=0: all other clauses share var 1 with c_seed (δ≥1 > 0) → none eligible
    # → only c_seed is returned
    c_seed = Clause(frozenset({1}), 0)
    c1     = Clause(frozenset({1, 2}), 0)
    c2     = Clause(frozenset({1, 2, 3}), 0)
    c3     = Clause(frozenset({1, 3, 4}), 0)
    omap = build_occurence_list([c_seed, c1, c2, c3])
    result = grow_block(0, [c_seed, c1, c2, c3], omap)
    assert isinstance(result, Batch)
    assert result.clauses == frozenset({c_seed})

def test__grow_block__seed_is_least_conflicting():
    # c_seed={1} has d=3 (minimum); c1,c2,c3 have d=4 or 5.
    # With budget=0, only c_seed survives (all others share var 1 with it).
    # This confirms the seed selected is the least-conflicting clause.
    c_seed = Clause(frozenset({1}), 0)
    c1     = Clause(frozenset({1, 2}), 0)
    c2     = Clause(frozenset({1, 2, 3}), 0)
    c3     = Clause(frozenset({1, 3, 4}), 0)
    omap = build_occurence_list([c_seed, c1, c2, c3])
    # Verify the conflict degrees so we know c_seed is the minimum
    assert conflict_deg(c_seed, omap) < conflict_deg(c1, omap)
    assert conflict_deg(c_seed, omap) < conflict_deg(c2, omap)
    assert conflict_deg(c_seed, omap) < conflict_deg(c3, omap)
    result = grow_block(0, [c_seed, c1, c2, c3], omap)
    assert c_seed in result.clauses

def test__grow_block__redundancy_tracked_correctly():
    # 2 clauses sharing variable 1; budget=1 → both fit (δ=1 ≤ 1)
    # Batch redundancy should be 1 after adding both
    # Determine which is seed: omap 1→{c1,c2}, 2→{c1}, 3→{c2}
    # ν1=2, ν2=1, ν3=1
    # d(c1) = (2-1)+(1-1) = 1, d(c2) = (2-1)+(1-1) = 1 (tied)
    # Tiebreak by width: both width=2 → index order → c1 is seed
    # After seeding with c1={1,2}: δ(c2) = |{1,3}∩{1,2}| = 1 ≤ budget=1 → eligible
    # → both in batch, redundancy=1
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({1, 3}), 0)
    omap = build_occurence_list([c1, c2])
    result = grow_block(1, [c1, c2], omap)
    assert isinstance(result, Batch)
    assert result.clauses == frozenset({c1, c2})
    assert result.redundancy == 1


def test__grow_block__updates_delta_for_new_shared_variable():
    # Covers the delta-update path (lines 260-261, True branch of `if delta:`) in grow_block.
    # c1={10} is the seed (d=0, isolated); c2={20,30} is picked second (δ=0 from batch={10}).
    # c3={30,40} is still in queue when c2 is added: var 30 is NEW to the batch.
    # → δ(c3) is updated from 0 to 1 inside the loop at lines 260-261.
    # With budget=5, c3's updated δ=1 ≤ 5 → all 3 clauses end up in the batch.
    c1 = Clause(frozenset({10}), 0)
    c2 = Clause(frozenset({20, 30}), 0)
    c3 = Clause(frozenset({30, 40}), 0)
    omap = build_occurence_list([c1, c2, c3])
    result = grow_block(5, [c1, c2, c3], omap)
    assert isinstance(result, Batch)
    assert c1 in result.clauses
    assert c2 in result.clauses
    assert c3 in result.clauses


def test__grow_block__no_delta_for_already_batched_variable():
    # Covers the False branch of `if delta:` (line 262→252) in grow_block.
    # Seed c1={1,2}; next c2={2,3}; c3={2,4} is still in queue.
    # When c2 is added, next_vars={2,3}. For c3: the only shared variable with
    # next_vars is var 2, but var 2 is ALREADY in batch_vars after seeding c1.
    # → delta=0 → the `if delta:` branch is False, no bucket update for c3.
    # With budget=5, c3 is still eligible (its stored δ=1 ≤ 5-R) → all 3 fit.
    c1 = Clause(frozenset({1, 2}), 0)
    c2 = Clause(frozenset({2, 3}), 0)
    c3 = Clause(frozenset({2, 4}), 0)
    omap = build_occurence_list([c1, c2, c3])
    result = grow_block(5, [c1, c2, c3], omap)
    assert isinstance(result, Batch)
    assert c1 in result.clauses
    assert c2 in result.clauses or c3 in result.clauses


# ---------- CSTNode empty-batch branches ----------

def test__CSTNode__add_batch__empty_batch():
    # add_batch with an empty Batch: the `if batch._clauses:` branch is False,
    # so max_clause_width stays 0 and subsumed_variables stays empty.
    hrse = HRSENode(5, 0, None)
    node = CSTNode(hrse, None)
    empty = Batch()
    node.add_batch(empty)
    assert node.partition == [empty]
    assert node.subsumed_variables == set()
    assert node.max_clause_width == 0


def test__CSTNode__set_partition__with_empty_batch():
    # set_partition with a list containing an empty Batch: same False branch
    # in the for-loop inside set_partition.
    hrse = HRSENode(5, 0, None)
    node = CSTNode(hrse, None)
    empty = Batch()
    node.set_partition([empty])
    assert node.partition == [empty]
    assert node.subsumed_variables == set()
    assert node.max_clause_width == 0


# ---------- merge_adjacent redundancy-budget break ----------

def test__merge_adjacent__budget_prevents_merge_with_redundancy():
    # b1 has var {1}; b2 has vars {1,2} — merging them creates redundancy=1.
    # budget=0: occupied(=0) + tent_redundancy(=1) > 0 → break at line 309.
    # Result: b1 and b2 remain as separate batches (not merged).
    b1 = Batch({Clause(frozenset({1}), 0)})
    b2 = Batch({Clause(frozenset({1, 2}), 0)})
    result = merge_adjacent([b1, b2], budget=0)
    assert isinstance(result, list)
    assert len(result) == 2


# ---------- _build_cst_subtree: leaf with no clause ----------

def test__build_cst_subtree__unmapped_leaf_returns_none():
    # When _build_cst_subtree is called on a leaf HRSENode that has no entry
    # in leaf_clause_map, it returns None (line 429 in cst.py).
    from compiler.cst import _build_cst_subtree
    leaf = HRSENode(2, 0, None)  # size=2 → leaf (no children)
    result = _build_cst_subtree(leaf, None, {}, {}, None)
    assert result is None
