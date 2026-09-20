from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .cst import Clause, variable


class NumpyContext(NamedTuple):
    """Pre-computed structures enabling vectorized operations across a clause set.

    clause_masks uses Python arbitrary-precision integers as bitmasks (one bit per variable
    at its remapped 0-based index). Bitwise AND + int.bit_count() replaces per-variable
    Python loops for set-intersection counting, which is ~10–100x faster than dict lookups
    for typical clause widths.

    pad_idx / padded_freq / clause_lengths enable a single matrix gather+sum in sort_clauses
    that computes all n conflict degrees simultaneously, avoiding n sequential Python key calls.
    """
    var_to_idx: dict[variable, int]        # original variable ID → 0-based index
    idx_to_var: np.ndarray                 # idx_to_var[i] == original variable ID
    n_vars: int                            # total unique variable count
    freq_arr: np.ndarray                   # freq_arr[i] = clause count for variable i
    clause_arrs: dict[Clause, np.ndarray]  # clause → index array
    clause_masks: dict[Clause, int]        # clause → Python int bitmask (for grow_block)
    # Sort-matrix fields (indexed by position in the clause list passed to build_numpy_context)
    pad_idx: np.ndarray                    # shape (n_clauses, max_width); padding sentinel = n_vars
    padded_freq: np.ndarray                # freq_arr extended by one 0 for the sentinel
    clause_lengths: np.ndarray             # shape (n_clauses,); dtype int64


def build_numpy_context(clauses: list[Clause]) -> NumpyContext | None:
    r"""
    Build numpy structures, bitmasks, and the sort padded-matrix for a clause set.
    Returns None when the clause list is empty.
    """
    if not clauses:
        return None

    all_vars = sorted({v for c in clauses for v in c.normed_variables})
    var_to_idx: dict[int, int] = {v: i for i, v in enumerate(all_vars)}
    n = len(all_vars)
    idx_to_var = np.array(all_vars, dtype=np.int64)
    freq_arr = np.zeros(n, dtype=np.int64)
    clause_arrs: dict = {}
    clause_masks: dict = {}

    max_w = max(len(c.normed_variables) for c in clauses)
    m = len(clauses)
    pad_idx = np.full((m, max_w), n, dtype=np.intp)   # sentinel = n (points to freq 0)
    clause_lengths = np.empty(m, dtype=np.int64)

    for i, c in enumerate(clauses):
        arr = np.array([var_to_idx[v] for v in c.normed_variables], dtype=np.intp)
        clause_arrs[c] = arr
        freq_arr[arr] += 1
        mask = 0
        for v in c.normed_variables:
            mask |= 1 << var_to_idx[v]
        clause_masks[c] = mask
        pad_idx[i, : len(arr)] = arr
        clause_lengths[i] = len(arr)

    padded_freq = np.empty(n + 1, dtype=np.int64)
    padded_freq[:n] = freq_arr
    padded_freq[n] = 0   # sentinel: padding columns contribute 0 to freq sums

    return NumpyContext(
        var_to_idx, idx_to_var, n, freq_arr,
        clause_arrs, clause_masks,
        pad_idx, padded_freq, clause_lengths,
    )
