from collections.abc import Generator
from cstcompiler.synthesis import Clause

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cnfc import Formula

def cnfc_clauses(formula: Formula) -> Generator[Clause]:
    """Yields all clauses of the formula as cstcompiler.Clause objects"""
    for clause in formula.buffer.AllClauses():
        yield Clause.from_literals(clause)