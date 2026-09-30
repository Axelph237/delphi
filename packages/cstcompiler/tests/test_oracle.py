from __future__ import annotations

import random

import pytest

from cstcompiler.clause_pack import cst_to_oracle
from cstcompiler.synthesis import grow_cst
from cstcompiler.backbone import HRSENode, max_covered_leaves
from cstcompiler.numpy_context import build_numpy_context

from tests.utils import cnf_truth_mask, input_masks, random_kcnf, simulate


def _budgets(m: int) -> list[int]:
    """Smallest ASDT-feasible budget, a midpoint, and the paper's maximum 2m − 1."""
    smallest = next(k for k in range(1, 2 * m) if max_covered_leaves(k) >= m)
    return sorted({smallest, (smallest + 2 * m - 1) // 2, 2 * m - 1})


# [LTC §VII] 180 small instances, all 2^n inputs, both target initializations
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("m", [3, 5, 8, 12])
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test__cst_to_oracle__computes_f_and_restores_every_qubit(n, k, m, seed):
    clauses = random_kcnf(n, m, k, random.Random(f"{n}-{k}-{m}-{seed}"))
    ctx = build_numpy_context(clauses)
    assert ctx is not None

    masks = input_masks(ctx.n_vars)
    all_ones = 2 ** (2 ** ctx.n_vars) - 1
    f = cnf_truth_mask(clauses, {int(ctx.idx_to_var[i]): masks[i] for i in range(ctx.n_vars)}, all_ones)

    for budget in _budgets(m):
        hrse_root = HRSENode.new(m, budget)
        assert hrse_root is not None
        grow_result = grow_cst(hrse_root, list(clauses))
        assert grow_result is not None
        root, _ = grow_result
        assert root is not None
        circuit, x_register, [target] = cst_to_oracle(root, ctx)
        assert circuit.num_qubits == ctx.n_vars + budget

        for c in (0, all_ones):
            wires = masks + [0] * budget
            wires[target] = c
            result = simulate(circuit, wires, all_ones)

            expected = masks + [0] * budget
            expected[target] = c ^ f
            assert result[:ctx.n_vars] == masks, f"input register changed at budget {budget}"
            assert result[target] == c ^ f, f"target is not c ⊕ f(x) at budget {budget}"
            assert result == expected, f"ancilla not restored to |0⟩ at budget {budget}"
