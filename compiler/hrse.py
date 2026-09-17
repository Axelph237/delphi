from math import ceil
from bisect import insort


MIN_NODE_SIZE = 2

class HRSENode:
    """
    A single computational unit in an HRSE Tree.

    Attributes:
    - size: The number of auxilary qubits in this node.
    - depth: The recursion level of this node.
    - complexity: The number of basic gates in the quantum circuit for this node.
    - covered_leaves: The total number of underlying constraint functions implemented by this node.
    - out_deg: The number of submodules contained within this node.
    """

    size: int  # s
    depth: int  # d
    complexity: int | None  # c
    covered_leaves: int | None  # l
    out_deg: int  # k

    parent: HRSENode | None
    children: list[HRSENode]


    def __init__(self, size, depth, parent):
        if size < 0:
            raise ValueError("HRSE Node must have a positive size.")
        
        if depth < 0:
            raise ValueError("HRSE Node must have a non-negative depth.")

        self.size = size
        self.depth = depth
        self.complexity = None
        self.covered_leaves = None
        self.out_deg = 0

        self.children = []
        self.parent = parent

    @staticmethod
    def new(num_clauses: int, budget: int):
        return asdt(num_clauses, budget)


    def add_child(self, child: HRSENode):
        """
        Adds child node to this node, updating observers and attributes

        Returns: the candidacy of this node after adding the child
        """

        # Check saturation
        if self.is_saturated():
            raise ValueError(f"Node of size {self.size}, depth {self.depth} is saturated, cannot add child")
      
        # Validate monotonicity
        if child.size >= self.size:
            raise ValueError("Child size must be less than parent size")

        # Validate distinct sibling sizes
        if child.size in [n.size for n in self.children]:
            raise ValueError("Child size must be distinct from siblings")

        # Prevent repeated children
        if child in self.children:
            return self.is_candidate()

        # Add node to self and update
        self.children.append(child)
        self.out_deg = len(self.children)

        return self.is_candidate()

    def is_saturated(self):
        return self.out_deg == self.size - 1 or self.size <= MIN_NODE_SIZE

    def is_candidate(self):
        return not self.is_saturated()

    def is_leaf(self):
        return self.out_deg == 0


def max_covered_leaves(k: int):
    """
    Returns the maximum number of constraint functions that can be implemented by an HRSE tree with k auxiliary qubits.

    k: number of auxilary qubits

    **Proof:**

    c(k) is the max clauses covered by an HRSE tree constructed by ASDT with k auxilary qubits.
    In other words, c(k) = sum(c(j) for j = 2 ... k - 1)

    1. Base cases
        a. c(k) = 1 for k = 1, 2
        b. c(3) = 2
        c. c(4) = c(3) + c(2) = 2 + 1 = 3

    2. Consider some k >= 5
        c(k) = sum(c(j) for j = 2 ... k - 1)
        and
        c(k - 1) = sum(c(j) for j = 2 ... k - 2)
        then
        c(k) - c(k - 1) = c(k - 1) -> c(k) = 2 * c(k - 1)

    3. Geometry
        Since c(k) = 2 * c(k - 1) for k >= 5,
        c(k) is a geometric sequence of ratio 2, with base 3, anchored at 4
        c(k) = 3 * 2 ^ (k - 4) for k >= 4

    4. General formula, for k > 0:
        c(k) = ceil( 3 * 2 ^ (k - 4) )
        as c(k) is an integer for all k >= 4, ceil( 3 * 1 / 2) = 2, and ceil( 3 * 1 / n) = 1 for n >= 3.

    **Conclusion:**
        The maximum number of leaves covered by an HRSE tree constructed by ASDT with k auxilary qubits is
        c(k) = ceil(3 * 2 ^ (k - 4)) for k > 0
    """
    return ceil(3 * (2 ** (k - 4)))


def asdt(m: int, k: int):
    """
    Creates an optimal HRSE tree $T_k(m)$ using the ASDT algorithm

    m: number of clauses
    k: number of auxilary qubits

    Returns the root node of the HRSE tree, which is optimal in terms of size and depth
    """

    # Primary sort (strategy 1: min-depth), then secondary sort (strategy 2: max-size)

    if m == 0:
        return None

    if m > max_covered_leaves(k):
        raise ValueError(f"Cannot implement more than {max_covered_leaves(k)} clauses with {k} auxiliary qubits")

    asdt_compare = lambda n: (n.depth, -n.size)

    nodes: list[HRSENode] = []
    edges: list[tuple[HRSENode, HRSENode]] = []

    def add_to_tree(node: HRSENode, parent: HRSENode):
        parent.add_child(node)
        nodes.append(node)
        edges.append((parent, node))

    candidate_nodes: list[HRSENode] = []

    # Initialize Root
    root = HRSENode(k, 0, None)
    nodes.append(root)
    if root.is_candidate():
        candidate_nodes.append(root)

    # Interatively add nodes to root
    leaf_count = 1
    while leaf_count < m:
        # print(f"---------------- leaf_count = {leaf_count}")
        # print_tree(root)
        if len(candidate_nodes) == 0:
            raise RuntimeError("Incomplete HRSE tree cannot be expanded.")

        # 1. Get next candidate node
        parent = candidate_nodes[0]

        next_size = parent.size - (parent.out_deg + 1)  # satisfy node monotonicity

        # 2. Add new leaf/leaves to tree
        # If it is a leaf node, it must split into two nodes
        if parent.is_leaf():
            # Create an additional node and decrement the remaining space
            split_n = HRSENode(next_size, parent.depth + 1, parent)
            add_to_tree(split_n, parent)

            if split_n.is_candidate():
                insort(candidate_nodes, split_n, key=asdt_compare)
            
            next_size -= 1  # satisfy node distinct sibling sizes

        # Create a new node
        n = HRSENode(next_size, parent.depth + 1, parent)
        add_to_tree(n, parent)
        if n.is_candidate():
            insort(candidate_nodes, n, key=asdt_compare)

        if parent.is_saturated():
            candidate_nodes.pop(0)

        leaf_count += 1

    # print(f"---------------- leaf_count = {leaf_count}")
    # print_tree(root)

    return root