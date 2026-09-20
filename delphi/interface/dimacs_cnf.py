from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Set
from delphi.compiler.cst import Clause

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

variable = int  # DIMACS variable IDs are positive integers


@dataclass
class DimacsParseResult:
    # Maps variable ID -> name from "c var N : name" comment lines.
    # Variables with no comment entry are absent from this dict.
    variable_names: dict[variable, str]
    clauses: list[Clause]
    num_vars: int   # as declared in the "p cnf" header
    num_clauses: int  # as declared in the "p cnf" header


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_COMMENT_VAR_RE = re.compile(r'^c var (\d+) : (.+)$')
_HEADER_RE = re.compile(r'^p cnf (\d+) (\d+)$')


def parse_dimacs(f) -> DimacsParseResult:
    """Parse a DIMACS CNF file from a file-like object.

    Returns a DimacsParseResult containing:
      - variable_names: mapping of variable ID -> name (from cnfc comment lines)
      - clauses: list of Clause objects
      - num_vars / num_clauses: values declared in the 'p cnf' header
    """
    variable_names: dict[variable, str] = {}
    clauses: list[Clause] = []
    num_vars = 0
    num_clauses = 0
    header_seen = False

    for raw_line in f:
        line = raw_line.strip()

        if not line:
            continue

        # Comment lines: may carry "c var N : name" entries
        if line.startswith('c'):
            match = _COMMENT_VAR_RE.match(line)
            if match:
                vid, name = match.groups()
                variable_names[int(vid)] = name
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

        if not literals:
            # Empty clause — unsatisfiable by definition; represent as empty Clause
            clauses.append(Clause(set(), 0))
            continue

        clauses.append(_clause_from_literals(literals))

    return DimacsParseResult(
        variable_names=variable_names,
        clauses=clauses,
        num_vars=num_vars,
        num_clauses=num_clauses,
    )


def _clause_from_literals(literals: list[int]) -> Clause:
    """Convert a list of signed DIMACS literals into a Clause.

    The polarity mask is built over the *sorted* variable IDs:
      bit i = 1  ->  normed_variables[i] appears positive in the clause
      bit i = 0  ->  normed_variables[i] appears negated in the clause
    """
    variables: set[variable] = {abs(lit) for lit in literals}
    polarity: dict[variable, bool] = {abs(lit): lit > 0 for lit in literals}

    sorted_vars = sorted(variables)
    mask = 0
    for i, var in enumerate(sorted_vars):
        if polarity[var]:
            mask |= (1 << i)

    return Clause(variables, mask)


# ---------------------------------------------------------------------------
# Convenience: parse from a file path
# ---------------------------------------------------------------------------

def parse_dimacs_file(path: str) -> DimacsParseResult:
    with open(path) as f:
        return parse_dimacs(f)


# ---------------------------------------------------------------------------
# Optional helpers on the result
# ---------------------------------------------------------------------------

def clause_to_named_literals(
    clause: Clause,
    variable_names: dict[variable, str],
) -> list[str]:
    """Render a Clause as a list of human-readable literal strings.

    Variables without a name entry fall back to their integer ID.
    Negated variables are prefixed with '~'.

    Example: ['x', '~y', '3']
    """
    result = []
    for i, var in enumerate(clause.normed_variables):
        positive = bool((clause.variable_polarity_mask >> i) & 1)
        name = variable_names.get(var, str(var))
        result.append(name if positive else f'~{name}')
    return result


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python dimacs_parser.py <path_to_cnf_file>')
        sys.exit(1)

    result = parse_dimacs_file(sys.argv[1])

    print(f'Declared: {result.num_vars} variables, {result.num_clauses} clauses')
    print(f'Named variables: {result.variable_names}')
    print()
    for i, clause in enumerate(result.clauses):
        named = clause_to_named_literals(clause, result.variable_names)
        print(f'Clause {i}: {clause}')
        print(f'       -> {named}')