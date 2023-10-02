from __future__ import annotations

from functools import reduce, singledispatch

from openfermion import QubitOperator
from openfermion.utils import count_qubits

from qadence import operations
from qadence.blocks import AbstractBlock, AddBlock, CompositeBlock, PrimitiveBlock, ScaleBlock
from qadence.blocks.utils import add, kron
from qadence.parameters import evaluate


@singledispatch
def to_openfermion(block: AbstractBlock) -> QubitOperator:
    raise ValueError(f"Unable to convert type {type(block)} to QubitOperator.")


@to_openfermion.register
def _(block: PrimitiveBlock) -> QubitOperator:
    pauli, qubit = block.name, block.qubit_support[0]
    return QubitOperator(f"{pauli}{qubit}")


@to_openfermion.register
def _(block: operations.I) -> QubitOperator:
    return QubitOperator("")


@to_openfermion.register
def _(block: CompositeBlock) -> QubitOperator:
    return reduce(lambda x, y: x * y, [to_openfermion(b) for b in block.blocks])


@to_openfermion.register
def _(block: AddBlock) -> QubitOperator:
    return reduce(lambda x, y: x + y, [to_openfermion(b) for b in block.blocks])


@to_openfermion.register
def _(block: ScaleBlock) -> QubitOperator:
    op = to_openfermion(block.block)
    return op * evaluate(block.parameters.parameter)


@to_openfermion.register
def _(block: AbstractBlock) -> QubitOperator:
    return to_openfermion(block)


def from_openfermion(op: QubitOperator) -> AbstractBlock:
    n_qubits = count_qubits(op)

    def _convert_gate(gate: tuple[int, str]) -> PrimitiveBlock:
        (i, pauli) = gate
        return getattr(operations, pauli)(i)  # type: ignore [no-any-return]

    @singledispatch
    def _convert(op: QubitOperator) -> AbstractBlock:
        if isinstance(op, QubitOperator):
            return _convert(op.terms)
        else:
            raise ValueError(f"Can only conver QubitOperators. Found {type(op)}.")

    @_convert.register
    def _(op: dict) -> AbstractBlock:
        bs = [_convert(term) * coef for term, coef in op.items()]
        return add(*bs) if len(bs) > 1 else bs[0]  # type: ignore [no-any-return]

    @_convert.register
    def _(op: tuple) -> AbstractBlock:
        if len(op) == 0:
            return operations.I(n_qubits - 1)
        bs = [_convert_gate(gate) for gate in op]
        return kron(*bs) if len(bs) > 1 else bs[0]  # type: ignore [no-any-return]

    return _convert(op)
