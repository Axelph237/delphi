from __future__ import annotations

import io
import itertools

import numpy as np
import pytest
from cnfc import Formula, Not, Or
from qiskit.quantum_info import Statevector

from cstcompiler.clause_pack import cst_to_oracle
from cstcompiler.synthesis import Clause, grow_cst
from cstcompiler.backbone import HRSENode
from cstcompiler.numpy_context import build_numpy_context
from delphi_interface.cnfc import cnfc_clauses
from delphi_interface.dimacs_cnf import clause_to_named_literals, parse_dimacs

# Its only solution is (1, 1, 0); flipping every sign gives (0, 0, 1), so a polarity bug changes the answer
DIMACS = """\
c var 1 : x
c var 2 : y
p cnf 3 3
1 -2 0
2 3 0
-3 0
"""
LITERALS: list[tuple[int, ...]] = [(1, -2), (2, 3), (-3,)]


def _satisfies(literals: list[tuple[int, ...]], assignment: dict[int, int]) -> bool:
    return all(any(assignment[abs(lit)] == (lit > 0) for lit in clause) for clause in literals)


def _marked_assignments(clauses: list[Clause]) -> set[tuple[int, ...]]:
    """Simulate the compiled CST oracle on every input and return the assignments it marks."""
    ctx = build_numpy_context(clauses)
    hrse_root = HRSENode.new(len(clauses), 4)
    assert ctx is not None and hrse_root is not None
    root = grow_cst(hrse_root, list(clauses))
    assert root is not None
    circuit, _, [target] = cst_to_oracle(ctx.n_vars, root, ctx)

    marked = set()
    for bits in itertools.product([0, 1], repeat=ctx.n_vars):
        index = sum(bit << i for i, bit in enumerate(bits))
        state = Statevector.from_int(index, 2 ** circuit.num_qubits).evolve(circuit)
        result = int(np.argmax(np.abs(state.data)))
        assert (result & ~(1 << target)) == index, "oracle changed the input or left an ancilla dirty"
        if (result >> target) & 1:
            marked.add(tuple(bits[i] for i in np.argsort(ctx.idx_to_var)))
    return marked


EXPECTED = {
    bits for bits in itertools.product([0, 1], repeat=3)
    if _satisfies(LITERALS, {v: bits[v - 1] for v in (1, 2, 3)})
}
assert EXPECTED == {(1, 1, 0)}


def test__parse_dimacs__oracle_marks_exactly_the_satisfying_assignments():
    clauses = parse_dimacs(io.StringIO(DIMACS)).clauses
    assert _marked_assignments(clauses) == EXPECTED


def test__cnfc_clauses__oracle_marks_exactly_the_satisfying_assignments():
    formula = Formula()
    x, y, z = formula.AddVars("x y z")
    formula.Add(Or(x, Not(y)))
    formula.Add(Or(y, z))
    formula.Add(Not(z))
    assert _marked_assignments(list(cnfc_clauses(formula))) == EXPECTED


def test__parse_dimacs__literals_round_trip():
    clauses = parse_dimacs(io.StringIO(DIMACS)).clauses
    assert [c.literals for c in clauses] == LITERALS


def test__clause_to_named_literals__marks_negated_variables():
    result = parse_dimacs(io.StringIO(DIMACS))
    assert [clause_to_named_literals(c, result.variable_names) for c in result.clauses] == [
        ["x", "~y"], ["y", "3"], ["~3"],
    ]


def test__parse_dimacs__empty_clause():
    clauses = parse_dimacs(io.StringIO("p cnf 1 1\n0\n")).clauses
    assert clauses == [Clause(set(), 0)]


def test__parse_dimacs__clause_with_variable_and_its_negation_raises():
    with pytest.raises(ValueError, match="both 1 and -1"):
        parse_dimacs(io.StringIO("p cnf 2 1\n1 -1 2 0\n"))
