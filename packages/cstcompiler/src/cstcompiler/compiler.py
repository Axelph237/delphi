from collections.abc import Iterable
from __future__ import annotations

from cstcompiler.synthesis import CSTNode


class CST:
    root: CSTNode

    def __init__(self, root: CSTNode):
        self.root = root

    def to_oracle():
        raise NotImplementedError

    @staticmethod
    def from_dimacs(dimacs: str) -> CST:
        raise NotImplementedError

    @staticmethod
    def from_clauses(clauses: Iterable[Iterable[int]]) -> CST:
        raise NotImplementedError