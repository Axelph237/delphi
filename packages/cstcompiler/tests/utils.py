from __future__ import annotations

import random

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate

from cstcompiler.synthesis import Clause


def print_tree(root, details: str = "", recurse_symbol="children", indent=0):
    prefix = "  " * indent

    print(f"{prefix}{type(root).__name__}: {root!r}")
    print(f"{prefix}{details}")

    for child in getattr(root, recurse_symbol, []):
        print_tree(child, details, recurse_symbol, indent + 1)


def random_kcnf(n: int, m: int, k: int, rng: random.Random) -> list[Clause]:
    """Uniform random k-CNF over variables 1..n: each clause has k distinct variables, each negated with probability 1/2."""
    clauses = []
    for _ in range(m):
        variables = sorted(rng.sample(range(1, n + 1), k))
        mask = sum(1 << i for i in range(k) if rng.random() < 0.5)
        clauses.append(Clause(set(variables), mask))
    return clauses


def input_masks(n: int) -> list[int]:
    """Bit-sliced inputs: bit a of mask i is bit i of assignment a, over all 2^n assignments."""
    return [sum(1 << a for a in range(2 ** n) if (a >> i) & 1) for i in range(n)]


def cnf_truth_mask(clauses: list[Clause], var_masks: dict[int, int], all_ones: int) -> int:
    """Bit-sliced f(x) = ∧ C_i(x), where a set polarity bit marks a negated literal."""
    f = all_ones
    for clause in clauses:
        value = 0
        for i, v in enumerate(clause.normed_variables):
            negated = (clause.variable_polarity_mask >> i) & 1
            value |= var_masks[v] ^ (all_ones if negated else 0)
        f &= value
    return f


def simulate(circuit: QuantumCircuit, wires: list[int], all_ones: int) -> list[int]:
    """Run a circuit of X and multi-controlled X gates on bit-sliced classical wires."""
    wires = list(wires)
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [index[q] for q in instruction.qubits]
        if operation.name == "x":
            wires[qubits[0]] ^= all_ones
        elif isinstance(operation, ControlledGate) and operation.base_gate is not None and operation.base_gate.name == "x":
            if operation.ctrl_state != 2 ** operation.num_ctrl_qubits - 1:
                raise ValueError(f"Unsupported control state on {operation.name}")
            active = all_ones
            for control in qubits[:-1]:
                active &= wires[control]
            wires[qubits[-1]] ^= active
        else:
            raise ValueError(f"Non-classical gate {operation.name} in oracle circuit")
    return wires
