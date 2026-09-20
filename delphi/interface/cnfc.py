from collections.abc import Generator
from delphi.compiler.cst import Clause

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cnfc import Formula

def cnfc_clauses(formula: Formula) -> Generator[Clause]:
    """Yields all clauses of the formula as compiler.Clause objects"""
    for clause in formula.buffer.AllClauses():
        var_set = set()
        polarity_mask = 0
        for v in clause:
            var_set.add(abs(v))
            polarity_mask |= (1 << abs(v))
        yield Clause(var_set, polarity_mask)