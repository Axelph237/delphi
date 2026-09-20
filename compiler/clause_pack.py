from __future__ import annotations

from compiler.hrse import HRSENode
from compiler.cst import CSTNode, Clause, Batch
from compiler.numpy_context import NumpyContext
from pytket import Circuit
from pytket.circuit import CircBox, OpType


# [LTC §VI]
class AncillaScheduler:
    free_ancilla: set[int]
    allocated_ancilla: set[int]

    def __init__(self, ancilla: list[int] | int, start: int = 0):
        self.free_ancilla = set(ancilla if isinstance(ancilla, list) else range(start, ancilla + start))
        self.allocated_ancilla = set()
    
    @property
    def available_ancilla(self):
        return len(self.free_ancilla)

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


# [LTC Alg. 3]
def cst_to_oracle(x_register_size: int, root: CSTNode, ctx: NumpyContext) -> tuple[Circuit, list[int], list[int]]:
    r"""
    Converts a feasible CST $\boldsymbol{\mathcal{T}}$ into a unitary $\boldsymbol{U_\mathcal{T}}$ that implements the oracle for the corresponding SAT problem.
    """

    scheduler = AncillaScheduler(root.size, start=x_register_size)
    cst_output_reg = scheduler.allocate()
    if cst_output_reg is None:
        raise ValueError(f"CST root has no allocable ancilla registers.")

    cst_oracle = Circuit(x_register_size + root.size)

    x_register = list(range(0, x_register_size))
    root_oracle, root_output_registers = node_to_oracle(x_register, scheduler, root, ctx)

    cst_oracle.append(root_oracle)
    cst_oracle.add_gate(OpType.CnX, root_output_registers + cst_output_reg)
    cst_oracle.append(root_oracle.dagger())

    scheduler.free(root_output_registers)

    if len(scheduler.allocated_ancilla) != 1:
        raise RuntimeError(f"Failed to fully free all ancilla used in CST. {len(scheduler.allocated_ancilla)} ancilla still in use.")

    return cst_oracle, x_register, cst_output_reg


# [LTC §V.C]
def clause_oracle(clause: Clause) -> CircBox:
    clause_circuit = Circuit(len(clause.normed_variables) + 1)

    for i, v in enumerate(clause.normed_variables):
        if (clause.variable_polarity_mask >> i) & 1:
            clause_circuit.X(i)   # Negative polarity variables
        clause_circuit.X(i)   # Toffoli prep

    clause_circuit.add_gate(OpType.CnX, clause_circuit.qubits)

    return CircBox(clause_circuit)


# [LTC §V, Eq. 26]
def clause_pack(
        x_register: list[int],
        w_register: list[int],
        y_register: list[int],
        batch: Batch,
        ctx: NumpyContext
    ):
        # 1 ancilla for every redundant variable, and 1 ancilla for every clause output
        if batch.redundancy + len(batch.clauses) > len(w_register) + len(y_register):
            raise RuntimeError("Not enough ancilla qubits available for clause packing.")

        # var_allocated_on: maps each variable to the list of qubits it is allocated on,
        # where the first element is the wire, and the second is the wire allocating that bit,
        # and the third is a boolean indicating whether the wire is claimed or not.
        variable_allocations: dict[int, list[tuple[int, int]]] = {}

        all_qubits = x_register + w_register + y_register
        clause_pack_circuit = Circuit(max(all_qubits) + 1 if all_qubits else 0)
        w_scheduler = AncillaScheduler(w_register)
        y_scheduler = AncillaScheduler(y_register)

        for clause in batch.clauses:
            # Stage 1. Variable Replication $\boldsymbol{U_{\text{rep}}^\dagger}$ [LTC §V.B, Eq. 27]
            variable_wires = []
            for v in clause.normed_variables:
                if v not in variable_allocations:
                    on_wire = ctx.var_to_idx[v]
                    variable_allocations[v] = [(-1, on_wire)]
                else:
                    [on_wire] = w_scheduler.safe_allocate()
                    from_wire = ctx.var_to_idx[v]
                    variable_allocations[v].append((from_wire, on_wire))
                    clause_pack_circuit.add_gate(OpType.CX, [from_wire, on_wire])

                variable_wires.append(on_wire)

            # Stage 2. Parallel Clause Evaluation $\boldsymbol{U_{\text{eval}}}$ [LTC Eq. 30]
            clause_circuit = clause_oracle(clause)
            [output_wire] = y_scheduler.safe_allocate()
            clause_pack_circuit.add_circbox(clause_circuit, variable_wires + [output_wire])

        # Stage 3. Reverse Replication $\boldsymbol{U_{\text{rep}}}$ [LTC §V.C]
        for var, allocations in variable_allocations.items():
            if len(allocations) <= 1:
                continue
            # Uncompute here
            for from_wire, to_wire in reversed(allocations[1:]):
                clause_pack_circuit.add_gate(OpType.CX, [from_wire, to_wire])
                w_scheduler.free([to_wire])

        if len(w_scheduler.allocated_ancilla) != 0:
            raise RuntimeError(f"ClausePack failed to free all redundancy allocated ancilla in W register.")
        
        return clause_pack_circuit


# [LTC §VI Alg. 3]
def node_to_oracle(x_register: list[int], scheduler: AncillaScheduler, node: CSTNode, ctx: NumpyContext) -> tuple[Circuit, list[int]]:

    node_circuit = Circuit(len(x_register) + node.size)

    node_output_register = scheduler.allocate()
    if node_output_register is None:
        raise ValueError(f"Node {node} has no allocable ancilla")

    # 1. Partition (ClausePack) Evaluation [LTC Alg. 3, lines 4-6]
    cp_oracles: list[Circuit] = []
    cp_oracles_out_registers: list[list[int]] = []
    for batch in node.partition:
        w_register = scheduler.allocate(batch.redundancy)
        y_register = scheduler.allocate(len(batch.clauses))
        if w_register is None or y_register is None:
            raise ValueError(f"Node {node} is too small to evaluate batch {batch}. Not enough ancilla.")
        cp = clause_pack(x_register, w_register, y_register, batch, ctx)
        cp_oracles_out_registers.append(y_register)

        # Add to circuit
        cp_oracles.append(cp)
        node_circuit.append(cp)

    # 2. Child Evaluation [LTC Alg. 3, lines 11-15]
    child_oracles = []
    child_oracle_out_registers = []
    for child in node.children:
        if child.is_leaf():
            continue
        if not isinstance(child, CSTNode):
            raise TypeError(f"Found orphan HRSE node in CST during circuit mapping.")
        child_oracle, output_register = node_to_oracle(x_register, scheduler, child, ctx)
        child_oracles.append(child_oracle)
        child_oracle_out_registers.append(output_register)
        node_circuit.append(child_oracle)

    # 3. Multiplex outputs [LTC Alg. 3, line 17]
    eval_output_registers = sum((out_regs for out_regs in
        cp_oracles_out_registers + child_oracle_out_registers
    ), []) # $\boldsymbol{|Y_o\rangle}$

    node_circuit.add_gate(OpType.CnX, eval_output_registers + node_output_register)

    # 4. Free Child Evaluations [LTC Alg. 3, lines 18-20]
    for co, co_out_regs in zip(child_oracles, child_oracle_out_registers):
        node_circuit.append(co.dagger())
        scheduler.free(co_out_regs)

    # 5. Free Partition (ClausePack) Evaluations
    for cp, cp_out_regs in zip(cp_oracles, cp_oracles_out_registers):
        node_circuit.append(cp.dagger())
        scheduler.free(cp_out_regs)

    return node_circuit, node_output_register

