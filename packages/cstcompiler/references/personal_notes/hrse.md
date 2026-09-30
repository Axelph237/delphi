# Hierarchical Recursive Synthesis and Evaluation (HRSE) model via
https://arxiv.org/html/2605.21380v1


HRSE Model: $T = (V, E, A)$

- $V$: nodes representing computation units
- $E$: edges representing containment relationships between units
- $A$: attributes of each unit (5-dimension tuple).

Where
$$
A(V_i) = ( s(v_i), d(v_i), k(v_i), c(v_i), l(v_i))
$$

**Def 1:** A $U^{(d)}_i$ is the $i$-th compute unit at depth $d$.

**1. Structural Feature Mapping**

(1) Node Correspondence: An oracle module $U^{(d)}_i$ corresponds to a $v_i \in V$

(2) Edge Correspondence: Hierarchical module inclusion $U^{(d+1)}_j \subset U^{(d)}_i$ is represented by $(v_i,v_j) \in E$. 
    This denotes $v_j$ is the child of $v_i$ and $U^{(d+1)}_j$ is a sub-oracle of $U^{(d)}_i$

**2. Attribute Feautre Mapping**

The attributes of a node in $V$ are one-to-one with its corresponding oracle module:

(1) Node Size: The **scale** $s(v_i)$ of $v_i$ is the number of aux. qubits used by $U_i$.

(2) Node Depth: The **depth** $d(v_i)$ of $v_i$ is the recursion level of $U_i$.

(3) Node Complexity: The **complexity** $c(v_i)$ of $v_i$ is the number of basic gates 
   in the quantum circuit for $U_i$

(4) Covered Leaf Count: the **covered leaf count** $l(v_i)$ of $v_i$ equals the total number of underlying
   constraint functions implemented by $U_i$

(5) Node Out-degree: the **out-degree** $k(v_i)$ of $v_i$ is the number of submodules contained
   within module $U_i$

**3. Validity Constraints**

A tree $T$ must satisfy the following constraints to ensure an HRSE tree can be mapped to a
valid quantum circuit with the appropriate aux. qubit allocations.

(1) Monotonicity of Node Size: For an edge $(v_i, v_j) \in E$
$$
0 \leq s(v_j) \leq s(v_i)
$$

(2) Node Size Disinction among Siblings: The sizes of all children of some node must be distinct.

(3) Leaf Node Constraint: A node $v_i$ of $s(v_i) \leq 2$ must be a leaf node.