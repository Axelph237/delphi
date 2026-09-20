from __future__ import annotations

from collections.abc import Callable
import pytest
from unittest.mock import patch
from pytket import Circuit
from pytket.circuit import CircBox

from delphi.compiler.clause_pack import (
    AncillaScheduler,
    clause_oracle,
    clause_pack,
    node_to_oracle,
    cst_to_oracle,
)
from delphi.compiler.cst import Clause, Batch, CSTNode
from delphi.compiler.hrse import HRSENode
from delphi.compiler.numpy_context import build_numpy_context


# ---------- Helpers ----------

def _safe[T](fn: Callable[[], T | None]) -> T:
    result = fn()
    if result is None:
        raise RuntimeError
    return result

def _clause(vars_: set[int], mask: int = 0) -> Clause:
    return Clause(frozenset(vars_), mask)


def _cst_leaf(size: int, clause: Clause, parent=None) -> CSTNode:
    """Create a leaf CSTNode with a single-clause batch."""
    hrse = HRSENode(size, 0, parent)
    node = CSTNode(hrse, parent)
    node.set_partition([Batch({clause})])
    return node


def _scheduler(size: int, start: int) -> AncillaScheduler:
    return AncillaScheduler(size, start=start)


# ---------- AncillaScheduler — __init__ ----------

# [LTC §VI]
class TestAncillaSchedulerInit:
    def test__AncillaScheduler__list_input(self):
        s = AncillaScheduler([1, 3, 5])
        assert s.free_ancilla == {1, 3, 5}
        assert s.allocated_ancilla == set()

    def test__AncillaScheduler__empty_list(self):
        s = AncillaScheduler([])
        assert s.free_ancilla == set()
        assert s.allocated_ancilla == set()

    def test__AncillaScheduler__int_input_zero_start(self):
        s = AncillaScheduler(3)
        assert s.free_ancilla == {0, 1, 2}
        assert s.allocated_ancilla == set()

    def test__AncillaScheduler__int_input_nonzero_start(self):
        s = AncillaScheduler(3, start=10)
        assert s.free_ancilla == {10, 11, 12}
        assert s.allocated_ancilla == set()

    def test__AncillaScheduler__zero_int(self):
        s = AncillaScheduler(0)
        assert s.free_ancilla == set()


# ---------- AncillaScheduler — available_ancilla property ----------

# [LTC §VI]
class TestAvailableAncilla:
    def test__available_ancilla__initial_count(self):
        s = AncillaScheduler(5)
        assert s.available_ancilla == 5

    def test__available_ancilla__decreases_on_allocate(self):
        s = AncillaScheduler(5)
        s.allocate(3)
        assert s.available_ancilla == 2

    def test__available_ancilla__restores_on_free(self):
        s = AncillaScheduler(5)
        alloc = _safe(lambda: s.allocate(3))
        s.free(alloc)
        assert s.available_ancilla == 5


# ---------- AncillaScheduler — allocate ----------

# [LTC §VI]
class TestAllocate:
    def test__allocate__default_one(self):
        s = AncillaScheduler([7])
        result = s.allocate()
        assert result == [7]
        assert s.free_ancilla == set()
        assert s.allocated_ancilla == {7}

    def test__allocate__zero_count(self):
        s = AncillaScheduler(3)
        result = s.allocate(0)
        assert result == []
        assert s.available_ancilla == 3

    def test__allocate__multiple(self):
        s = AncillaScheduler(5)
        result = _safe(lambda: s.allocate(3))
        assert len(result) == 3
        assert len(s.free_ancilla) == 2
        assert len(s.allocated_ancilla) == 3

    def test__allocate__exactly_all(self):
        s = AncillaScheduler(3)
        result = s.allocate(3)
        assert result is not None
        assert len(result) == 3
        assert s.free_ancilla == set()

    def test__allocate__insufficient_returns_none(self):
        s = AncillaScheduler(2)
        result = s.allocate(3)
        assert result is None
        assert s.available_ancilla == 2  # unchanged

    def test__allocate__empty_pool_returns_none(self):
        s = AncillaScheduler(0)
        assert s.allocate() is None

    def test__allocate__one_over_boundary_returns_none(self):
        s = AncillaScheduler(3)
        assert s.allocate(4) is None


# ---------- AncillaScheduler — safe_allocate ----------

# [LTC §VI]
class TestSafeAllocate:
    def test__safe_allocate__success(self):
        s = AncillaScheduler(3)
        result = s.safe_allocate(2)
        assert len(result) == 2

    def test__safe_allocate__raises_when_empty(self):
        s = AncillaScheduler(0)
        with pytest.raises(ValueError, match="No ancilla left to allocate"):
            s.safe_allocate()

    def test__safe_allocate__raises_when_insufficient(self):
        s = AncillaScheduler(1)
        with pytest.raises(ValueError, match="No ancilla left to allocate"):
            s.safe_allocate(2)

    def test__safe_allocate__default_count_one(self):
        s = AncillaScheduler([42])
        result = s.safe_allocate()
        assert result == [42]


# ---------- AncillaScheduler — free ----------

# [LTC §VI]
class TestFree:
    def test__free__empty_list_noop(self):
        s = AncillaScheduler(3)
        s.free([])  # must not raise
        assert s.available_ancilla == 3

    def test__free__allocated(self):
        s = AncillaScheduler([5, 6, 7])
        allocated = _safe(lambda: s.allocate(2))
        s.free(allocated)
        assert s.allocated_ancilla == set()
        assert len(s.free_ancilla) == 3

    def test__free__unallocated_raises(self):
        s = AncillaScheduler([5, 6])
        with pytest.raises(ValueError, match="Cannot free unallocated ancilla 5"):
            s.free([5])

    def test__free__double_free_raises(self):
        s = AncillaScheduler([5])
        allocated = _safe(lambda: s.allocate())
        s.free(allocated)
        with pytest.raises(ValueError, match="Cannot free unallocated ancilla"):
            s.free(allocated)

    def test__free__freed_qubit_is_reallocatable(self):
        s = AncillaScheduler([5])
        allocated = _safe(lambda: s.allocate())
        s.free(allocated)
        result = _safe(lambda: s.allocate())
        assert result == [5]


# ---------- clause_oracle ----------

# [LTC §V.C]
class TestClauseOracle:
    def test__clause_oracle__returns_circbox(self):
        c = _clause({1})
        assert isinstance(clause_oracle(c), CircBox)

    def test__clause_oracle__single_var_positive_polarity(self):
        c = _clause({1}, mask=0)
        cb = clause_oracle(c)
        assert cb.n_qubits == 2  # 1 var + 1 output

    def test__clause_oracle__single_var_negative_polarity(self):
        c = _clause({1}, mask=1)  # bit 0 set → x1 negated
        cb = clause_oracle(c)
        assert cb.n_qubits == 2

    def test__clause_oracle__two_vars_mixed_polarity(self):
        c = _clause({1, 2}, mask=0b01)  # ¬x1, x2
        cb = clause_oracle(c)
        assert cb.n_qubits == 3

    def test__clause_oracle__three_vars_all_positive(self):
        c = _clause({1, 3, 5}, mask=0)
        cb = clause_oracle(c)
        assert cb.n_qubits == 4  # 3 vars + 1 output

    def test__clause_oracle__all_negative_polarity(self):
        c = _clause({1, 2, 3}, mask=0b111)
        cb = clause_oracle(c)
        assert cb.n_qubits == 4

    def test__clause_oracle__empty_clause(self):
        # normed_variables = (), circuit has 1 qubit, CnX on 1 qubit = X
        c = _clause(set(), mask=0)
        cb = clause_oracle(c)
        assert cb.n_qubits == 1


# ---------- clause_pack — isolated tests with contiguous register indices ----------

# [LTC Eq. 26]
class TestClausePackIsolated:
    """Use contiguous x/w/y register indices to avoid Bug 4."""

    # [LTC Eq. 33]
    def test__clause_pack__single_clause_single_var_no_redundancy(self):
        c = _clause({1}, mask=0)
        batch = Batch({c})
        assert batch.redundancy == 0

        ctx = _safe(lambda: build_numpy_context([c]))
        # x=[0], w=[], y=[1]
        circuit = clause_pack([0], [], [1], batch, ctx)
        assert circuit is not None

    def test__clause_pack__single_clause_two_vars_no_redundancy(self):
        c = _clause({1, 2}, mask=0)
        batch = Batch({c})
        ctx = _safe(lambda: build_numpy_context([c]))
        circuit = clause_pack([0, 1], [], [2], batch, ctx)
        assert circuit is not None

    # [LTC Eq. 25]
    def test__clause_pack__two_clauses_shared_variable_with_redundancy(self):
        c1 = _clause({1, 2}, mask=0)
        c2 = _clause({1, 3}, mask=0)
        batch = Batch({c1, c2})
        assert batch.redundancy == 1

        ctx = _safe(lambda: build_numpy_context([c1, c2]))
        # 3 variables → x=[0,1,2]; 1 w qubit; 2 y qubits
        circuit = clause_pack([0, 1, 2], [3], [4, 5], batch, ctx)
        assert circuit is not None

    # [LTC Eq. 33]
    def test__clause_pack__insufficient_ancilla_raises_runtime_error(self):
        c = _clause({1}, mask=0)
        batch = Batch({c})
        ctx = _safe(lambda: build_numpy_context([c]))
        # Need 1 y qubit but provide 0
        with pytest.raises(RuntimeError, match="Not enough ancilla"):
            clause_pack([0], [], [], batch, ctx)

    def test__clause_pack__exactly_at_capacity_succeeds(self):
        c = _clause({1}, mask=0)
        batch = Batch({c})
        ctx = _safe(lambda: build_numpy_context([c]))
        # redundancy=0, 1 clause → need exactly 1 qubit total (y)
        circuit = clause_pack([0], [], [1], batch, ctx)
        assert circuit is not None

    def test__clause_pack__empty_batch_returns_empty_circuit(self):
        batch = Batch()
        # Use a dummy clause to build ctx (batch iteration is empty)
        dummy = _clause({1}, mask=0)
        ctx = _safe(lambda: build_numpy_context([dummy]))
        circuit = clause_pack([0], [], [], batch, ctx)
        assert circuit is not None

    def test__clause_pack__negative_polarity_clause(self):
        c = _clause({1, 2}, mask=0b01)  # ¬x1 ∨ x2
        batch = Batch({c})
        ctx = _safe(lambda: build_numpy_context([c]))
        circuit = clause_pack([0, 1], [], [2], batch, ctx)
        assert circuit is not None

    # [LTC Eq. 25]
    def test__clause_pack__three_clauses_all_sharing_one_variable(self):
        # All three clauses share variable 1 → redundancy = 2
        c1 = _clause({1, 2}, mask=0)
        c2 = _clause({1, 3}, mask=0)
        c3 = _clause({1, 4}, mask=0)
        batch = Batch({c1, c2, c3})
        assert batch.redundancy == 2

        ctx = _safe(lambda: build_numpy_context([c1, c2, c3]))
        # 4 variables → x=[0,1,2,3]; 2 w qubits; 3 y qubits
        circuit = clause_pack([0, 1, 2, 3], [4, 5], [6, 7, 8], batch, ctx)
        assert circuit is not None

    # [LTC §V.C]
    def test__clause_pack__w_register_not_fully_freed_raises(self):
        """Artificially cause the w_scheduler leak check to fire.

        We patch w_scheduler.free to be a no-op so allocated_ancilla stays non-empty.
        """
        c1 = _clause({1, 2}, mask=0)
        c2 = _clause({1, 3}, mask=0)  # redundancy=1
        batch = Batch({c1, c2})
        ctx = _safe(lambda: build_numpy_context([c1, c2]))

        with patch.object(AncillaScheduler, 'free', lambda self, a: None):
            with pytest.raises(RuntimeError, match="ClausePack failed to free"):
                clause_pack([0, 1, 2], [3], [4, 5], batch, ctx)


# ---------- node_to_oracle ----------

# [LTC Alg. 3]
class TestNodeToOracle:
    def _simple_ctx(self, *clauses):
        return _safe(lambda: build_numpy_context(list(clauses)))

    def test__node_to_oracle__leaf_node_single_clause(self):
        c = _clause({1}, mask=0)
        ctx = self._simple_ctx(c)
        node = _cst_leaf(size=3, clause=c)
        x_reg = [0]
        sched = AncillaScheduler(node.size, start=len(x_reg))

        circuit, out_reg = node_to_oracle(x_reg, sched, node, ctx)
        assert circuit is not None
        assert len(out_reg) == 1

    def test__node_to_oracle__node_output_register_exhausted_raises(self):
        c = _clause({1}, mask=0)
        ctx = self._simple_ctx(c)
        node = _cst_leaf(size=3, clause=c)
        # Empty scheduler — node_output_register cannot be allocated
        sched = AncillaScheduler(0, start=1)

        with pytest.raises(ValueError, match="has no allocable ancilla"):
            node_to_oracle([0], sched, node, ctx)

    def test__node_to_oracle__batch_allocation_failure_raises(self):
        c = _clause({1}, mask=0)
        ctx = self._simple_ctx(c)
        # size=1: only 1 ancilla → used for node_output_register; nothing left for y_register
        node = _cst_leaf(size=1, clause=c)
        sched = AncillaScheduler(1, start=1)

        with pytest.raises(ValueError, match="is too small"):
            node_to_oracle([0], sched, node, ctx)

    # [LTC §III.C]
    def test__node_to_oracle__leaf_hrse_child_is_skipped(self):
        c = _clause({1}, mask=0)
        ctx = self._simple_ctx(c)
        node = _cst_leaf(size=4, clause=c)

        # Add a leaf HRSENode child (is_leaf() = True) — must be skipped
        leaf_hrse = HRSENode(2, 1, None)
        node.children = [leaf_hrse]
        node.out_deg = 1

        sched = AncillaScheduler(node.size, start=1)
        circuit, out_reg = node_to_oracle([0], sched, node, ctx)
        assert circuit is not None

    def test__node_to_oracle__orphan_hrse_non_leaf_child_raises_type_error(self):
        c = _clause({1}, mask=0)
        ctx = self._simple_ctx(c)
        node = _cst_leaf(size=5, clause=c)

        # Non-leaf HRSENode (not CSTNode) as child → TypeError
        hrse_parent = HRSENode(3, 1, None)
        hrse_leaf = HRSENode(2, 2, hrse_parent)
        hrse_parent.children = [hrse_leaf]
        hrse_parent.out_deg = 1

        node.children = [hrse_parent]
        node.out_deg = 1

        sched = AncillaScheduler(node.size, start=1)
        with pytest.raises(TypeError, match="orphan HRSE node"):
            node_to_oracle([0], sched, node, ctx)

    def test__node_to_oracle__cst_child_recurses(self):
        """Parent CSTNode with a non-leaf CSTNode child — exercises Phase 2 recursion."""
        c_parent = _clause({1}, mask=0)
        c_child = _clause({2}, mask=0)
        ctx = _safe(lambda: build_numpy_context([c_parent, c_child]))

        # Build child CSTNode (non-leaf: has a leaf HRSENode child)
        hrse_c = HRSENode(4, 1, None)
        child_cst = CSTNode(hrse_c, None)
        child_cst.set_partition([Batch({c_child})])
        child_cst.children = [HRSENode(2, 2, None)]  # leaf → skipped in Phase 2
        child_cst.out_deg = 1

        # Build parent CSTNode
        hrse_p = HRSENode(12, 0, None)
        parent_cst = CSTNode(hrse_p, None)
        parent_cst.set_partition([Batch({c_parent})])
        parent_cst.children = [child_cst]
        parent_cst.out_deg = 1

        x_reg = [0, 1]
        sched = AncillaScheduler(parent_cst.size, start=len(x_reg))

        circuit, out_reg = node_to_oracle(x_reg, sched, parent_cst, ctx)
        assert circuit is not None
        assert len(out_reg) == 1

    def test__node_to_oracle__empty_partition_and_no_children(self):
        """No partition, no children → CnX on just node_output_register (= X gate)."""
        hrse = HRSENode(3, 0, None)
        node = CSTNode(hrse, None)
        node.set_partition([])

        # Need a ctx — use a dummy clause just to have var_to_idx
        dummy = _clause({1}, mask=0)
        ctx = _safe(lambda: build_numpy_context([dummy]))

        sched = AncillaScheduler(3, start=1)
        circuit, out_reg = node_to_oracle([0], sched, node, ctx)
        assert circuit is not None
        assert len(out_reg) == 1


# ---------- cst_to_oracle ----------

# [LTC Alg. 3]
class TestCstToOracle:
    def test__cst_to_oracle__root_size_zero_raises_value_error(self):
        hrse = HRSENode(0, 0, None)
        node = CSTNode(hrse, None)
        node.set_partition([])
        ctx = _safe(lambda: build_numpy_context([_clause({1})]))
        with pytest.raises(ValueError, match="no allocable ancilla registers"):
            cst_to_oracle(1, node, ctx)

    def test__cst_to_oracle__simple_single_variable_formula(self):
        c = _clause({1}, mask=0)
        ctx = _safe(lambda: build_numpy_context([c]))

        hrse = HRSENode(3, 0, None)
        root = CSTNode(hrse, None)
        root.set_partition([Batch({c})])

        cst_circuit, x_reg, out_reg = cst_to_oracle(1, root, ctx)
        assert cst_circuit is not None
        assert x_reg == [0]
        assert len(out_reg) == 1

    def test__cst_to_oracle__two_variable_single_clause(self):
        c = _clause({1, 2}, mask=0)
        ctx = _safe(lambda: build_numpy_context([c]))

        hrse = HRSENode(4, 0, None)
        root = CSTNode(hrse, None)
        root.set_partition([Batch({c})])

        cst_circuit, x_reg, out_reg = cst_to_oracle(2, root, ctx)
        assert x_reg == [0, 1]
        assert len(out_reg) == 1

    # [LTC §VI]
    def test__cst_to_oracle__ancilla_leak_raises_runtime_error(self, monkeypatch):
        """If node_to_oracle leaks allocations the RuntimeError check fires."""
        c = _clause({1}, mask=0)
        ctx = _safe(lambda: build_numpy_context([c]))

        hrse = HRSENode(5, 0, None)
        root = CSTNode(hrse, None)
        root.set_partition([])

        def _leaking_node_to_oracle(x_register, scheduler, node, _ctx):
            # Intentionally allocate extra without freeing
            _extra = scheduler.allocate(2)
            circ = Circuit(len(x_register) + node.size)
            out = scheduler.allocate(1)
            return circ, out

        monkeypatch.setattr('compiler.clause_pack.node_to_oracle', _leaking_node_to_oracle)
        with pytest.raises(RuntimeError, match="Failed to fully free"):
            cst_to_oracle(1, root, ctx)

    def test__cst_to_oracle__negative_polarity_clause(self):
        c = _clause({1, 2}, mask=0b01)  # ¬x1 ∨ x2
        ctx = _safe(lambda: build_numpy_context([c]))

        hrse = HRSENode(4, 0, None)
        root = CSTNode(hrse, None)
        root.set_partition([Batch({c})])

        cst_circuit, x_reg, out_reg = cst_to_oracle(2, root, ctx)
        assert cst_circuit is not None
