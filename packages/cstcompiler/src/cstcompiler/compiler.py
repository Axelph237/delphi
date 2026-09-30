from __future__ import annotations

from io import StringIO
from typing_extensions import Optional
from typing import TextIO
from qiskit import QuantumCircuit
from cstcompiler.clause_pack import cst_to_oracle
from attr import dataclass, field
from collections.abc import Iterable, Generator
from cstcompiler.numpy_context import NumpyContext
from cstcompiler.synthesis import CSTNode, Clause, grow_cst
from cstcompiler.backbone import asdt
import re


@dataclass
class VariableMapping:
    variable_ids: dict[str, int] = field(factory=dict)
    variable_names: dict[int, str] = field(factory=dict)
    next_var_id: int = field(default=1)  # literals are signed, so id 0 could never be negated

    def add(self, var: str):
        var_id = self.next_var_id
        self.next_var_id += 1
        self.variable_ids[var] = var_id
        self.variable_names[var_id] = var

    def remove(self, key: str | int):
        var: str
        var_id: int
        if isinstance(key, str):
            var = key
            var_id = self.variable_ids[key]
        else:
            var_id = key
            var = self.variable_names[key]
        
        del self.variable_names[var_id]
        del self.variable_ids[var]


@dataclass(kw_only=True)
class CST:
    variable_mapping: VariableMapping | None
    root: CSTNode
    ctx: NumpyContext

    def to_oracle(self) -> tuple[QuantumCircuit, list[int], list[int]]:
        return cst_to_oracle(
            root=self.root,
            ctx=self.ctx
        )






    @staticmethod
    def from_dimacs(dimacs: str, ancilla_budget: int) -> CST:
        var_map = VariableMapping()

        return CST.from_Clauses(
            clauses=clauses_from_dimacs(dimacs, var_map),
            ancilla_budget=ancilla_budget,
            var_map=var_map
        )

    @staticmethod
    def from_named_literals(clauses: Iterable[Iterable[str]], ancilla_budget: int) -> CST:
        """Takes an iterable of string iterables. Named literals starting with ~/- are considered negated."""

        var_map = VariableMapping()
        clause_lst: list[Clause] = []

        def numeric_lits(named_lits: Iterable[str]) -> Generator[int]:
            for nl in named_lits:
                negated = nl.startswith(("~", "-"))
                name = nl[1:] if negated else nl
                if name not in var_map.variable_ids:
                    var_map.add(name)
                var_id = var_map.variable_ids[name]
                yield -var_id if negated else var_id

        for var_set in clauses:
            clause = Clause.from_literals(numeric_lits(var_set))
            clause_lst.append(clause)

        return CST.from_Clauses(clause_lst, ancilla_budget, var_map)


    @staticmethod
    def from_Clauses(clauses: list[Clause], ancilla_budget: int, var_map = None) -> CST:
        hrse_root = asdt(len(clauses), ancilla_budget)
        if hrse_root is None:
            raise RuntimeError("Cannot implement CST: bad given clauses and busget.")

        grow_result = grow_cst(hrse_root, clauses)
        if grow_result is None:
            raise RuntimeError("Cannot implement CST: empty clause list")

        cst_root, numpy_ctx = grow_result
        if cst_root is None or numpy_ctx is None:
            raise RuntimeError("Cannot implement CST: returned cst is None")

        return CST(
            root=cst_root,
            variable_mapping=var_map,
            ctx=numpy_ctx
        )


_COMMENT_VAR_RE = re.compile(r'^c var (\d+) : (.+)$')
_HEADER_RE = re.compile(r'^p cnf (\d+) (\d+)$')

def clauses_from_dimacs(source: str | TextIO, var_map: Optional[VariableMapping] = None):
    clauses = []
    header_seen = False

    if var_map is None:
        var_map = VariableMapping()

    if isinstance(source, str):
        source = StringIO(source)

    for raw_line in source:
        line = raw_line.strip()

        if not line:
            continue

        # Comment lines: may carry "c var N : name" entries
        if line.startswith('c'):
            match = _COMMENT_VAR_RE.match(line)
            if match:
                vid, name = match.groups()
                var_map.variable_ids[name] = int(vid)
                var_map.variable_names[int(vid)] = name
            continue

        # Header line: "p cnf <num_vars> <num_clauses>"
        if line.startswith('p'):
            match = _HEADER_RE.match(line)
            if not match:
                raise ValueError(f'Malformed header line: {line!r}')
            num_vars, num_clauses = int(match.group(1)), int(match.group(2))
            header_seen = True
            continue

        # Clause lines: space-separated signed integers terminated by 0
        if not header_seen:
            raise ValueError(f'Encountered clause line before header: {line!r}')

        literals = [int(x) for x in line.split()]
        if literals[-1] != 0:
            raise ValueError(f'Clause line not terminated by 0: {line!r}')
        literals = literals[:-1]  # drop the terminating 0

        # An empty clause is unsatisfiable by definition and becomes an empty Clause
        clauses.append(Clause.from_literals(literals))
    return clauses