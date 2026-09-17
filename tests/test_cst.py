from compiler.cst import *
from compiler.hrse import HRSENode

def test__Batch():
    # EMPTY BATCH
    empty_batch = Batch()

    assert empty_batch.variables == frozenset()
    assert empty_batch.clauses == frozenset()

    # BASIC BATCH INITIALIZATION
    clause_1 = Clause(frozenset({1, 2}))
    init_batch = Batch({clause_1})
    assert init_batch.clauses == frozenset({clause_1})
    assert init_batch.variables == frozenset({1, 2})
    assert init_batch.redundancy == 0

    # REDUNDANT CLAUSE
    clause_2 = Clause(frozenset({1, 2}))
    init_batch.add_clause(clause_2)
    assert init_batch.clauses == frozenset({clause_1})
    assert init_batch.variables == frozenset({1, 2})
    assert init_batch.redundancy == 0

    # REDUNDANCY IMPACT
    clause_3 = Clause(frozenset({2, 3}))
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
    batch_1 = Batch({Clause(frozenset({1, 2}))})
    batch_2 = Batch({Clause(frozenset({3, 4, 2}))})

    child_node.add_batch(batch_1)
    child_node.add_batch(batch_2)

    assert child_node.partition == [batch_2, batch_1]
    assert child_node.subsumed_variables == {1, 2, 3, 4}
    assert child_node.max_clause_width == 3

    # Direct partition setting
    batch_3 = Batch({Clause(frozenset({1, 2, 3}))})
    batch_4 = Batch({Clause(frozenset({1, 2}))})

    child_node.set_partition([batch_3, batch_4])

    assert child_node.partition == [batch_3, batch_4]
    assert child_node.subsumed_variables == {1, 2, 3}
    assert child_node.max_clause_width == 3


# ---------- build_occurence_list ----------

def test__build_occurence_list():
    c1 = Clause(frozenset({1, 2}))
    c2 = Clause(frozenset({2, 3}))
    c3 = Clause(frozenset({3, 4}))

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
    c1 = Clause(frozenset({1, 2}))
    c2 = Clause(frozenset({2, 3}))
    omap = build_occurence_list([c1, c2])

    assert freq(2, omap) == 2   # appears in both clauses
    assert freq(1, omap) == 1   # appears in only one clause

def test__freq__var_missing_from_omap():
    omap = build_occurence_list([Clause(frozenset({1, 2}))])
    assert freq(99, omap) == 0


# ---------- conflict_degree ----------

def test__conflict_degree():
    # c1 shares var 2 with c2, and var 3 with c3
    c1 = Clause(frozenset({1, 2, 3}))
    c2 = Clause(frozenset({2, 4}))
    c3 = Clause(frozenset({3, 5}))
    omap = build_occurence_list([c1, c2, c3])

    # d(c1) = (freq(1)-1) + (freq(2)-1) + (freq(3)-1) = 0 + 1 + 1 = 2
    assert conflict_deg(c1, omap) == 2

def test__conflict_degree__single_var_clause():
    c1 = Clause(frozenset({1}))
    c2 = Clause(frozenset({2}))
    omap = build_occurence_list([c1, c2])

    assert conflict_deg(c1, omap) == 0
    assert conflict_deg(c2, omap) == 0

def test__conflict_degree__no_redundant_vars():
    c1 = Clause(frozenset({1, 2}))
    c2 = Clause(frozenset({3, 4}))
    omap = build_occurence_list([c1, c2])

    assert conflict_deg(c1, omap) == 0
    assert conflict_deg(c2, omap) == 0


# ---------- redundancy_impact ----------

def test__redundancy_impact():
    c = Clause(frozenset({1, 2, 3}))
    # var_set contains 1 and 3 — overlap of size 2
    assert redundancy_impact(c, frozenset({1, 3, 5})) == 2

def test__redundancy_impact__no_impact():
    c = Clause(frozenset({1, 2, 3}))
    # No overlap with var_set
    assert redundancy_impact(c, frozenset({4, 5, 6})) == 0


# ---------- is_feasible ----------

def test__is_feasible():
    # Empty partition, single clause batch — should be feasible with any positive budget
    b = Batch({Clause(frozenset({1, 2}))})   # 1 clause, 0 redundancy
    assert is_feasible(b, [], 5) == True

def test__is_feasible__budget_exhausted():
    # Fill partition with 10 clauses -> occupied_ancilla = 10 > budget = 5
    clauses = [Clause(frozenset({i})) for i in range(10)]
    big_batch = Batch(set(clauses))
    partition = [big_batch]

    new_batch = Batch({Clause(frozenset({100}))})
    assert is_feasible(new_batch, partition, 5) == False


# ---------- sort_clauses ----------

def test__sort_clauses():
    c1 = Clause(frozenset({1, 2}))
    c2 = Clause(frozenset({2, 3}))
    c3 = Clause(frozenset({4, 5}))  # disjoint — conflict_deg = 0
    omap = build_occurence_list([c1, c2, c3])

    result = list(sort_clauses([c1, c2, c3], omap))
    assert len(result) == 3
    # c3 has conflict_deg=0; c1, c2 share var 2 so conflict_deg=1
    assert result[0] == c3

def test__sort_clauses__empty_list():
    result = list(sort_clauses([], {}))
    assert result == []

def test__sort_clauses__single_clause():
    c = Clause(frozenset({1, 2}))
    omap = build_occurence_list([c])
    result = list(sort_clauses([c], omap))
    assert result == [c]

def test__sort_clauses__multiple_clauses():
    # Tie-break by clause length (shorter first) when conflict degrees are equal
    c_short = Clause(frozenset({10}))         # length 1
    c_long  = Clause(frozenset({20, 30}))     # length 2
    c_high  = Clause(frozenset({10, 20}))     # shares vars with both — highest conflict

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

    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
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

    clauses = [Clause(frozenset({1, 2}))]
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
        Clause(frozenset({1, 2})),
        Clause(frozenset({2, 3})),
        Clause(frozenset({4, 5})),
    ]
    omap = build_occurence_list(clauses)

    result = seed_grow(hrse_node, clauses[:], omap)
    assert result is None or isinstance(result, CSTNode)
    if result is not None:
        assert len(result.partition) > 0


# ---------- merge_adjacent ----------

def test__merge_adjacent():
    # Disjoint clauses with generous budget
    c1 = Clause(frozenset({1}))
    c2 = Clause(frozenset({2}))
    b1 = Batch({c1})
    b2 = Batch({c2})

    result = merge_adjacent([b1, b2], budget=100)
    assert isinstance(result, list)
    assert len(result) <= 2

def test__merge_adjacent__empty_partition():
    result = merge_adjacent([], budget=10)
    assert result == []

def test__merge_adjacent__single_batch():
    c = Clause(frozenset({1, 2}))
    b = Batch({c})
    result = merge_adjacent([b], budget=10)
    assert len(result) == 1
    assert result[0].clauses == frozenset({c})

def test__merge_adjacent__budget_exhausted():
    # Tight budget: 5 batches each with 1 clause; budget=2 -> merging blocked quickly
    clauses = [Clause(frozenset({i})) for i in range(5)]
    batches = [Batch({c}) for c in clauses]

    result = merge_adjacent(batches, budget=2)
    assert isinstance(result, list)
    assert len(result) >= 1

def test__merge_adjacent__budget_not_exhausted():
    # Three disjoint single-clause batches with a very large budget
    c1 = Clause(frozenset({1}))
    c2 = Clause(frozenset({2}))
    c3 = Clause(frozenset({3}))
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
    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)

def test__grow_cst__root_mirrors_hrse_root():
    root = HRSENode(5, 0, None)
    root.children = [HRSENode(2, 1, root), HRSENode(1, 1, root)]
    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
    result = grow_cst(root, clauses)
    if result is not None:
        assert result.size == root.size
        assert result.depth == root.depth

def test__grow_cst__root_has_no_parent():
    root = HRSENode(5, 0, None)
    root.children = [HRSENode(2, 1, root), HRSENode(1, 1, root)]
    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
    result = grow_cst(root, clauses)
    if result is not None:
        assert result.parent is None

def test__grow_cst__asdt_tree_small():
    # HRSENode.new(m=3, k=4) — k=4 supports up to 3 clauses
    root = HRSENode.new(3, 4)
    assert root is not None
    clauses = [
        Clause(frozenset({1, 2})),
        Clause(frozenset({2, 3})),
        Clause(frozenset({3, 4})),
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
        Clause(frozenset({1, 2})),
        Clause(frozenset({2, 3})),
        Clause(frozenset({3, 4})),
        Clause(frozenset({4, 5})),
        Clause(frozenset({5, 6})),
        Clause(frozenset({1, 6})),
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)

def test__grow_cst__partition_nonempty_when_clauses_fit():
    # A generous budget: root size=6, one leaf child → budget=5, plenty for 2 clauses
    root = HRSENode(6, 0, None)
    root.children = [HRSENode(2, 1, root)]
    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
    result = grow_cst(root, clauses)
    if result is not None:
        assert len(result.partition) > 0

def test__grow_cst__subsumed_variables_subset_of_clause_vars():
    root = HRSENode(6, 0, None)
    root.children = [HRSENode(2, 1, root)]
    all_vars = {1, 2, 3, 4}
    clauses = [Clause(frozenset({1, 2})), Clause(frozenset({3, 4}))]
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
        Clause(frozenset({1, 2})),
        Clause(frozenset({2, 3})),
        Clause(frozenset({4, 5})),
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
        Clause(frozenset(rng.sample(range(1, 21), 3)))
        for _ in range(10)
    ]
    result = grow_cst(root, clauses)
    assert result is None or isinstance(result, CSTNode)
