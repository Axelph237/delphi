import pytest
from delphi.compiler.hrse import HRSENode, asdt

make_root = lambda s: HRSENode(size=s, depth=0, parent=None)


# ---------- HRSENode Tests ----------

def validate_tree_structure(node: HRSENode, parent: HRSENode | None = None):
    # Verify no loops
    if node is parent:
        raise RecursionError("No loops allowed in the tree")
    
    # (1) Verify node monotonicity
    lemma_mono_nonneg = node.size >= 0
    lemma_mono_leqparent = node.size <= parent.size if parent is not None else True  # [MDRO Eq. 2]
    if not (lemma_mono_nonneg and lemma_mono_leqparent):
        raise ValueError(f"Node size monotonicity violated: parent {parent.size if parent else "N/A"}, child {node.size}")

    # (3) Verify leaf node
    lemma_leaf_size = (node.size <= 2) if node.is_leaf() else True  # [MDRO Eq. 4]
    lemma_leaf_no_children = (node.children != []) if node.is_leaf() else True
    if not (lemma_leaf_size and lemma_leaf_no_children):
        raise ValueError(f"Leaf node constraint violated: a node of size {node.size} must be a leaf node.")

    # (2) Verify distinct children
    if node.children:
        sizes = set()
        for child in node.children:
            if child.size in sizes:  # [MDRO Eq. 3]
                raise ValueError(f"Children size distinction violated: Duplicate child size {child.size}")
            sizes.add(child.size)

    


# [MDRO §III.A]
def test_node_creation():
    node = make_root(3)

    d = vars(node).copy()
    d.pop('id', None)
    assert d == {
        'size': 3,
        'depth': 0,
        'complexity': None,
        'covered_leaves': None,
        'out_deg': 0,
        'parent': None,
        'children': [],
    }

def test_node_child_add():
    root = make_root(3)

    child = HRSENode(2, 1, root)
    root.add_child(child)

    assert root.out_deg == 1
    assert root.children == [child]
    
# [MDRO Eq. 4]
def test_node_saturation():
    root = make_root(3)
    # A node of size 3 must be split into 2 children
    root.add_child(HRSENode(2, 1, root))
    root.add_child(HRSENode(1, 1, root))

    assert root.is_saturated()

def test_saturation_violation():
    root_1 = make_root(2)

    with pytest.raises(ValueError):
        root_1.add_child(HRSENode(1, 1, root_1))

    root_2 = make_root(1)
    
    with pytest.raises(ValueError):
        root_2.add_child(HRSENode(1, 1, root_2))

# [MDRO Eq. 2]
def test_node_monotonicity_violation():
    root = make_root(3)
    child = HRSENode(4, 1, root)

    with pytest.raises(ValueError):
        root.add_child(child)

# [MDRO Eq. 3]
def test_node_sibling_size_violation():
    root = make_root(3)
    child_1 = HRSENode(2, 1, root)
    root.add_child(child_1)

    child_2 = HRSENode(2, 1, root)
    with pytest.raises(ValueError):
        root.add_child(child_2)


# ---------- create_hrse_tree() Tests ----------

def compare_trees(n1: HRSENode, n2: HRSENode):
    """Helper function for the deep comparison of HRSE trees"""
    if n1 is None and n2 is None:
        return True

    def attributes(n: HRSENode):
        """Returns a copy"""
        d = vars(n).copy()
        d.pop('children')
        d.pop('parent')
        d.pop('id', None)
        return d

    n1_attr = attributes(n1)
    n2_attr = attributes(n2)

    if n1_attr != n2_attr:
        return False

    for c1, c2 in zip(n1.children, n2.children):
        if not compare_trees(c1, c2):
            return False

    return True


# [MDRO Alg. 1]
def test_empty_construction():
    assert asdt(0, 3) is None


# [MDRO Alg. 1]
def test_tree_construction():
    # "c1_d1" -> child 1 @ depth 1

    # Optimal tree of 6 clauses from 5 auxilary qubits
    m = 6
    k = 5

    # Root
    root = make_root(k)
    
    # Depth: 1
    c1_d1 = HRSENode(k - 1, 1, root)
    c2_d1 = HRSENode(k - 2, 1, root)
    c3_d1 = HRSENode(k - 3, 1, root)
    c4_d1 = HRSENode(k - 4, 1, root)

    root.add_child(c1_d1)
    root.add_child(c2_d1)
    root.add_child(c3_d1)
    root.add_child(c4_d1)

    # Depth: 2
    c1_d2 = HRSENode(c1_d1.size - 1, 2, c1_d1)
    c2_d2 = HRSENode(c1_d1.size - 2, 2, c1_d1)
    c3_d2 = HRSENode(c1_d1.size - 3, 2, c1_d1)

    c1_d1.add_child(c1_d2)
    c1_d1.add_child(c2_d2)
    c1_d1.add_child(c3_d2)

    # Expected tree
    generated_root = asdt(m, k)    

    assert generated_root != None
    assert compare_trees(root, generated_root)


# [MDRO §IV.A]
def test_incomplete_tree():
    with pytest.raises(ValueError):
        asdt(6, 4)

def test_large_tree():
    m = 100
    k = 10

    root = asdt(m, k)

    assert root != None
    validate_tree_structure(root)
    