from __future__ import annotations

from qiskit import QuantumCircuit

from .cst import CSTNode, Clause, Batch
from .numpy_context import NumpyContext


class AncillaScheduler:
    free_ancilla: set[int]
    allocated_ancilla: set[int]

    def __init__(self, ancilla: list[int] | int, start: int = 0):
        self.free_ancilla = set(ancilla if isinstance(ancilla, list) else range(start, ancilla + start))
        self.allocated_ancilla = set()

    @property
    def available_ancilla(self):
        return len(self.free_ancilla)

    @property
    def width(self) -> int:
        """Number of circuit qubits needed to address every ancilla this scheduler manages."""
        managed = self.free_ancilla | self.allocated_ancilla
        return max(managed) + 1 if managed else 0

    def allocate(self, count: int = 1):
        if len(self.free_ancilla) - count < 0:
            return None
        else:
            allocate = {self.free_ancilla.pop() for _ in range(count)}
            self.allocated_ancilla |= allocate
            return list(allocate)

    def safe_allocate(self, count: int = 1):
        allocated = self.allocate(count)
        if allocated is None:
            raise ValueError(f"No ancilla left to allocate.")
        return allocated

    def free(self, ancilla: list[int]):
        for a in ancilla:
            if a not in self.allocated_ancilla:
                raise ValueError(f"Cannot free unallocated ancilla {a}")
            self.allocated_ancilla.remove(a)
            self.free_ancilla.add(a)


def _mcx(circuit: QuantumCircuit, controls: list[int], target: int):
    """Toggle `target` by the conjunction of `controls`; the empty conjunction is true."""
    if controls:
        circuit.mcx(controls, target)
    else:
        circuit.x(target)


# [LTC Alg. 3]
def cst_to_oracle(x_register_size: int, root: CSTNode, ctx: NumpyContext) -> tuple[QuantumCircuit, list[int], list[int]]:
    r"""
    Converts a feasible CST $\boldsymbol{\mathcal{T}}$ into a unitary $\boldsymbol{U_\mathcal{T}}$ that implements the oracle for the corresponding SAT problem.

    The root's target $t_{v_0}$ is the SAT-oracle target, so the circuit maps
    $|x\rangle|c\rangle|0\rangle \mapsto |x\rangle|c \oplus f(x)\rangle|0\rangle$.
    """

    scheduler = AncillaScheduler(root.size, start=x_register_size)
    x_register = list(range(0, x_register_size))
    cst_oracle, cst_output_reg = node_to_oracle(x_register, scheduler, root, ctx)

    if len(scheduler.allocated_ancilla) != 1:
        raise RuntimeError(f"Failed to fully free all ancilla used in CST. {len(scheduler.allocated_ancilla)} ancilla still in use.")

    return cst_oracle, x_register, cst_output_reg


# [LTC §V.C]
def clause_oracle(clause: Clause) -> QuantumCircuit:
    r"""De Morgan block $|x\rangle|y\rangle \mapsto |x\rangle|y \oplus C(x)\rangle$ on the clause's variable wires, then its output wire."""
    width = len(clause.normed_variables)
    clause_circuit = QuantumCircuit(width + 1)

    # Set mask bits mark negated literals, which already control on |1⟩ = ¬literal
    positive = [i for i in range(width) if not (clause.variable_polarity_mask >> i) & 1]

    if positive:
        clause_circuit.x(positive)
    _mcx(clause_circuit, list(range(width)), width)
    if positive:
        clause_circuit.x(positive)
    clause_circuit.x(width)

    return clause_circuit


# [LTC Eq. 27, Eq. 29]
def fan_out(source: int, targets: list[int]) -> list[tuple[int, int]]:
    r"""CNOT fan-out tree $|x\rangle|0\rangle^{\otimes k-1} \mapsto |x\rangle^{\otimes k}$ of depth $\lceil \log_2 k \rceil$, as (control, target) pairs."""
    wires = [source]
    pending = list(targets)
    gates: list[tuple[int, int]] = []
    while pending:
        layer = list(zip(wires, pending))
        pending = pending[len(layer):]
        wires += [target for _, target in layer]
        gates += layer
    return gates


# [LTC §V, Eq. 26]
def clause_pack(
        x_register: list[int],
        w_register: list[int],
        y_register: list[int],
        batch: Batch,
        ctx: NumpyContext
    ) -> QuantumCircuit:
        r"""$\mathrm{ClausePack}_B = U_{rep}^\dagger U_{eval} U_{rep}$: $|x\rangle|0\rangle_W|y\rangle_Y \mapsto |x\rangle|0\rangle_W|y \oplus F_B(x)\rangle_Y$ [LTC Eq. 24]"""
        # 1 ancilla for every redundant variable, and 1 ancilla for every clause output
        if batch.redundancy + len(batch.clauses) > len(w_register) + len(y_register):
            raise RuntimeError("Not enough ancilla qubits available for clause packing.")

        all_qubits = x_register + w_register + y_register
        clause_pack_circuit = QuantumCircuit(max(all_qubits) + 1 if all_qubits else 0)
        w_scheduler = AncillaScheduler(w_register)
        y_scheduler = AncillaScheduler(y_register)

        # Stage 1. Variable Replication $\boldsymbol{U_{\text{rep}}}$ [LTC §V.B, Eq. 27]
        # variable_copies: the wires holding a copy of each variable, one per clause that reads it
        variable_copies: dict[int, list[int]] = {}
        replication: list[tuple[int, int]] = []
        for v, occurrences in batch.variables_to_count.items():
            source = ctx.var_to_idx[v]
            copies = w_scheduler.safe_allocate(occurrences - 1) if occurrences > 1 else []
            replication += fan_out(source, copies)
            variable_copies[v] = [source] + copies

        for control, target in replication:
            clause_pack_circuit.cx(control, target)

        # Stage 2. Parallel Clause Evaluation $\boldsymbol{U_{\text{eval}}}$ [LTC Eq. 30]
        for clause in batch.clauses:
            variable_wires = [variable_copies[v].pop() for v in clause.normed_variables]
            [output_wire] = y_scheduler.safe_allocate()
            clause_pack_circuit.compose(clause_oracle(clause), qubits=variable_wires + [output_wire], inplace=True)

        # Stage 3. Reverse Replication $\boldsymbol{U_{\text{rep}}^\dagger}$ [LTC §V.C]
        for control, target in reversed(replication):
            clause_pack_circuit.cx(control, target)
        w_scheduler.free([target for _, target in replication])

        if len(w_scheduler.allocated_ancilla) != 0:
            raise RuntimeError(f"ClausePack failed to free all redundancy allocated ancilla in W register.")

        return clause_pack_circuit


# [LTC §VI Alg. 3]
def node_to_oracle(x_register: list[int], scheduler: AncillaScheduler, node: CSTNode, ctx: NumpyContext) -> tuple[QuantumCircuit, list[int]]:

    node_circuit = QuantumCircuit(max(max(x_register, default=-1) + 1, scheduler.width))

    node_output_register = scheduler.allocate()
    if node_output_register is None:
        raise ValueError(f"Node {node} has no allocable ancilla")

    # evaluated: each unit's circuit $\boldsymbol{C[w]}$ and exposed outputs $\boldsymbol{O_w}$, in evaluation order
    evaluated: list[tuple[QuantumCircuit, list[int]]] = []

    # 1. Child Evaluation [LTC Alg. 3, lines 11-15]
    # Internal children run first; SeedGrow reserved room for their held outputs
    for child in node.children:
        if child.is_leaf():
            continue
        if not isinstance(child, CSTNode):
            raise TypeError(f"Found orphan HRSE node in CST during circuit mapping.")
        child_oracle, output_register = node_to_oracle(x_register, scheduler, child, ctx)
        evaluated.append((child_oracle, output_register))
        node_circuit.compose(child_oracle, inplace=True)

    # 2. Partition (ClausePack) Evaluation [LTC Alg. 3, lines 4-6]
    for batch in node.partition:
        w_register = scheduler.allocate(batch.redundancy)
        y_register = scheduler.allocate(len(batch.clauses))
        if w_register is None or y_register is None:
            raise ValueError(f"Node {node} is too small to evaluate batch {batch}. Not enough ancilla.")
        cp = clause_pack(x_register, w_register, y_register, batch, ctx)
        scheduler.free(w_register)  # ClausePack restores W, so later clusters reuse it [LTC §V.B]
        evaluated.append((cp, y_register))
        node_circuit.compose(cp, inplace=True)

    # 3. Multiplex outputs [LTC Alg. 3, line 17]
    eval_output_registers = [q for _, out_regs in evaluated for q in out_regs]  # $\boldsymbol{Q_u}$
    _mcx(node_circuit, eval_output_registers, node_output_register[0])

    # 4. Uncompute in reverse evaluation order [LTC Alg. 3, lines 18-20]
    for oracle, out_regs in reversed(evaluated):
        node_circuit.compose(oracle.inverse(), inplace=True)
        scheduler.free(out_regs)

    return node_circuit, node_output_register
