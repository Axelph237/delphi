# Adaptive Space-Depth Trade-off (ASDT) algorithm via.
https://arxiv.org/html/2605.21380v1

## Definition

HRSE model tree: $T_k(m)$

Incrementally expands nodes using one of two strategies:

Strategy 1 (minimum-depth): selects the node with the smallest depth

Strategy 2 (maximum-size): selects the node with the largest size, minimizing the
   total number of non-leaf nodes contributing to cost.

Ties across strategy 1 are broken with strategy 2. The use of these strategies produces an
optimal HRSE tree $T_k(m)$.

A post-order traversal of the final tree computes the complexity $c(v_i)$ and number of 
   leaf nodes $l(v_i)$ of each node $v_i$

## Construction

### Terminology

**Saturated Node**: a node that cannot accept more children

$k(v) = s(v) - 1$ or $s(v) \leq 2$

**Unsaturated Node**: a node that has children, but can still accept more

$0 \lt k(v) \lt s(v) - 1$

**Candidate Node**: any node that can accept new child nodes

$s(v) \gt 2$ and $k(v) \lt s(v) - 1$
