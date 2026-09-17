# Clustered Synthesis Tree (CST)
https://arxiv.org/pdf/2607.11401

## Clause Grouping
Given an internal compute node $C$, the `clause_group` function takes, as input, the set of clause leaves attached directly to $C$ and produces an ordered cluster partition $\Pi(v)$, the sequence of clustered clause evaluations available under the ancilla budget.

For a given clause $C_i$, let $\widehat{C}_i$ be the set of variables in $C_i$, regardless of polarity. For a cluster $\beta$, let $k_z(\beta)$ is the number of clauses in $\beta$ that contains $z$. Thus, the redundancy of a cluster is:
$$
R(\beta) = \sum_z{max\{k_z(\beta) - 1, 0\}}
$$
Clusters are evaluated sequentially. Let $K$ be the cluster count of an internal compute code, equivalent to the number of sequential evaluation stages at that node. The problem is to construct $\Pi$ s.t. it minimizes $K$.

### Grouping Complexity

Suppose an internal compute node $v$. The variable set of clauses directly attached to $v$ is $\widehat{C}_1, ..., \widehat{C}_j$. At the root, $j$ is the total number of clauses. Additionally $a_q(v)$ is the local ancilla budget of the node, with $a_q$ equalling the total ancilla budget at the root. The question lies in if a set of clauses can be grouped into at most $L$ clusters given a budget $a_q$. The search is then:
$$
\text{f} : (\widehat{C}_1, ..., \widehat{C}_j), L, a_q \mapsto bool
$$
The constraints are the following: given our input, can we create an ordered partition of $C_1, ..., C_j$ such that:
$$
\text{G.1} \qquad \Pi = \{\beta_1, ..., \beta_K\}, \quad K \leq L
$$
and
$$
\text{G.2} \qquad \sum_{h=1}^n |\beta_h|+R(\beta_n)) \leq a_q, \quad \forall n = 1, ..., K
$$
This problem is in NP, and its optimization is classified as NP-hard.

### SeedGrow
SeedGrow is a heuristic for finding an approximate solution to the Grouping problem above. The SeedGrow heuristic runs at each computation node to find the best clustering of its leaves. A cluster is called a batch $\beta$ for SeedGrow. The ordered partition of a computation node still must follow the redundancy constraint ($\text{G.2}$).

Because later clause evaluation clusters have a smaller ancilla budget, SeedGrow constructs a partition in reverse evaluation order. This is because the last batch $\beta_j$ has an $a_q - j$ redundancy allowance. The next batch's ($\beta_{j-1}$) redundancy allownace then increases by the size of the previous batch $|\beta_j|$.

At each cluster, SeedGrow starts from a low-conflict seed clause. Let $v_z$ equal the number of clauses at the current node containing variable $z$. For any clause $C_i$, its conflict degree is
$$
d_i = \sum_{z \in \widehat{C_i}}(v_z - 1)
$$
In other words, how strongly $C_i$ shares variables with other clauses at this node. SeedGrow starts with the clause smallest $d_i$. SeedGrow then repeatedly adds clauses with the lowest redundancy impact. 
$$
\delta_i = |\widehat{C_i} \cap U|
$$
without exceeding the current allowance, where $U$ is the set of variables already contained at the compute node.

After all batches have been computed, their order is reversed back to proper evaluation order. Then, `MergeAdjacent` greedily merges consecutive batches while still satisfying the budget constraint ($\text{G.2}$).

The runtime is polynomial to the number of clauses. At the current node, let $k = \text{max}_i|\widehat{C_i}|$ be the max normalized clause width. Normalizing clause variable sets and computing all conflict degrees takes $O(jk+jlogj)$:

- $O(jlogj)$ comes from sorting the conflict degrees.
- $O(jk)$ comes from at most $j$ calls of `GrowBlock` with bucketed candidate queues and incremental updates to of $\delta_i$.

Thus, with a linear merge time of $j$, SeedGrow has a total complexity of $O(j^2k)$ and uses $O(jk)$ space to store all normalized clause sets.