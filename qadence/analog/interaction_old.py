from __future__ import annotations

from copy import deepcopy
from functools import singledispatch
from itertools import product
from math import dist as euclidean_distance
from typing import Any, Callable, Union, overload

import torch
from sympy import cos, sin

from qadence.analog.utils import C6_DICT
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    AnalogBlock,
    AnalogChain,
    AnalogKron,
    ConstantAnalogRotation,
    Interaction,
    WaitBlock,
)
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock, ScaleBlock
from qadence.blocks.utils import _construct, add, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.operations import HamEvo, I, N, X, Y, wait
from qadence.qubit_support import QubitSupport
from qadence.register import Register
from qadence.transpile.transpile import blockfn_to_circfn


def _qubitposition(register: Register, i: int) -> tuple[int, int]:
    (x, y) = list(register.coords.values())[i]
    return (x, y)


def ising_interaction(
    register: Register, pairs: list[tuple[int, int]], rydberg_level: int = 60
) -> AbstractBlock:
    """
    Computes the Rydberg Ising interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_6 / R**6) * kron(N_i, N_j)

    Args:
        register: the register of qubits.
        pairs: a list of all pairs of interacting qubits.
        rydberg_level: determines the value of C_6
    """
    c6 = C6_DICT[rydberg_level]

    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c6 / rij**6) * kron(N(i), N(j))

    return add(term(i, j) for (i, j) in pairs)


def xy_interaction(
    register: Register, pairs: list[tuple[int, int]], c3: float = 3700.0
) -> AbstractBlock:
    """
    Computes the Rydberg XY interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_3 / R**3) * (kron(X_i, X_j) + kron(Y_i, Y_j))

    Args:
        register: the register of qubits.
        pairs: a list of all pairs of interacting qubits.
        c3: the coefficient value of C_3 in units of [rad . µm^3 / µs]
    """

    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c3 / rij**3) * (kron(X(i), X(j)) + kron(Y(i), Y(j)))

    return add(term(i, j) for (i, j) in pairs)


INTERACTIONS = {Interaction.NN: ising_interaction, Interaction.XY: xy_interaction}


def rot_generator(block: ConstantAnalogRotation) -> AbstractBlock:
    omega = block.parameters.omega
    delta = block.parameters.delta
    phase = block.parameters.phase
    support = block.qubit_support

    x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
    z_terms = delta * add(N(i) for i in support)
    return x_terms - z_terms  # type: ignore[no-any-return]


@overload
def add_interaction(circuit: QuantumCircuit, **kwargs: Any) -> QuantumCircuit:
    ...


@overload
def add_interaction(block: AbstractBlock, **kwargs: Any) -> AbstractBlock:
    ...


@overload
def add_interaction(register: Register, block: AbstractBlock, **kwargs: Any) -> AbstractBlock:
    ...


@singledispatch
def add_interaction(
    x: Register | QuantumCircuit | AbstractBlock,
    *args: Any,
    interaction: Interaction | Callable = Interaction.NN,
    spacing: float = 1.0,
) -> QuantumCircuit | AbstractBlock:
    """Turns blocks or circuits into (a chain of) `HamEvo` blocks.

    This includes a chosen interaction term.

    This is a `@singledipatch`ed function which can be called in three ways:

    * With a `QuantumCircuit` which contains all necessary information: `add_interaction(circuit)`
    * With a `Register` and an `AbstractBlock`: `add_interaction(reg, block)`
    * With an `AbstractBlock` only: `add_interaction(block)`

    See the section about [analog blocks](/digital_analog_qc/analog-basics.md) for
    detailed information about how which types of blocks are translated.

    Arguments:
        x: Circuit or block to be emulated. See the examples on which argument
            combinations are accepted.
        interaction: Type of interaction that is added. Can also be a function that accepts a
            register and a list of edges that define which qubits interact (see the examples).
        spacing: All qubit coordinates are multiplied by `spacing`.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence import QuantumCircuit, AnalogRX, add_interaction

    c = QuantumCircuit(2, AnalogRX(2.0))
    e = add_interaction(c)
    print(str(e.block.generator)) # markdown-exec: hide
    ```
    You can also use `add_interaction` directly on a block, but you have to provide either
    the `Register` or define a non-global qubit support.
    ```python exec="on" source="material-block" result="json"
    from qadence import AnalogRX, Register, add_interaction

    b = AnalogRX(2.0)
    r = Register(1)
    e = add_interaction(r, b)
    print(e.generator) # markdown-exec: hide

    # or provide only the block with local qubit support
    # in this case the register is created via `Register(b.n_qubits)`
    e = add_interaction(AnalogRX(2.0, qubit_support=(0,)))
    print(e.generator)
    ```
    You can specify a custom `interaction` function which has to accept a `Register` and a list
    of `edges: list[tuple[int, int]]`:
    ```python exec="on" source="material-block" result="json"
    from qadence import AnalogRX, Register, add_interaction
    from qadence.analog.utils import ising_interaction

    def int_fn(r: Register, pairs: list[tuple[int, int]]) -> AbstractBlock:
        # do either something completely custom
        # ...
        # or e.g. change the default kwargs to `ising_interaction`
        return ising_interaction(r, pairs, rydberg_level=70)

    b = AnalogRX(2.0)
    r = Register(1)
    e = add_interaction(r, b, interaction=int_fn)
    ```
    """
    raise ValueError(f"`add_interaction` is not implemented for {type(x)}")


@add_interaction.register  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit, **kwargs: Any) -> QuantumCircuit:
    reg = circuit.register
    return blockfn_to_circfn(lambda b: add_interaction(reg, b, **kwargs))(circuit)


@add_interaction.register  # type: ignore[attr-defined]
def _(block: AbstractBlock, **kwargs: Any) -> AbstractBlock:
    return add_interaction(Register(block.n_qubits), block, **kwargs)


@add_interaction.register  # type: ignore[attr-defined]
def _(
    register: Register,
    block: AbstractBlock,
    interaction: Union[Interaction, Callable] = Interaction.NN,
    spacing: float = 1.0,
) -> AbstractBlock:
    try:
        fn = interaction if callable(interaction) else INTERACTIONS[Interaction(interaction)]
    except KeyError:
        raise KeyError(
            "Function `add_interaction` only supports NN and XY, or a custom callable function."
        )
    reg = register._scale_positions(spacing)
    return _add_interaction(block, reg, fn)  # type: ignore[arg-type]


@singledispatch
def _add_interaction(b: AbstractBlock, r: Register, interaction: Callable) -> AbstractBlock:
    raise NotImplementedError(f"Cannot emulate {type(b)}")


@_add_interaction.register
def _(b: CompositeBlock, r: Register, i: Callable) -> AbstractBlock:
    return _construct(type(b), tuple(map(lambda b: _add_interaction(b, r, i), b.blocks)))


@_add_interaction.register
def _(block: ScaleBlock, register: Register, interaction: Callable) -> AbstractBlock:
    if isinstance(block.block, AnalogBlock):
        raise NotImplementedError("Scaling emulated analog blocks is not implemented.")
    return block


@_add_interaction.register
def _(block: PrimitiveBlock, register: Register, interaction: Callable) -> AbstractBlock:
    return block


@_add_interaction.register
def _(block: WaitBlock, register: Register, interaction: Callable) -> AbstractBlock:
    duration = block.parameters.duration

    support = tuple(range(register.n_qubits))
    assert support == block.qubit_support if not block.qubit_support.is_global else True
    pairs = list(filter(lambda x: x[0] < x[1], product(support, support)))

    return HamEvo(interaction(register, pairs), duration / 1000) if len(pairs) else I(0)


@_add_interaction.register
def _(block: ConstantAnalogRotation, register: Register, interaction: Callable) -> AbstractBlock:
    # convert "global" to indexed qubit suppport so that we can re-use `kron` dispatched function
    b = deepcopy(block)
    b.qubit_support = QubitSupport(*range(register.n_qubits))
    return _add_interaction(kron(b), register, interaction)


@_add_interaction.register
def _(block: AnalogKron, register: Register, interaction: Callable) -> AbstractBlock:
    from qadence import block_to_tensor

    w_block = wait(duration=block.duration, qubit_support=block.qubit_support)
    i_terms = add_interaction(register, w_block, interaction=interaction)

    generator = add(rot_generator(b) for b in block.blocks if isinstance(b, ConstantAnalogRotation))
    generator = generator if i_terms == I(0) else generator + i_terms.generator  # type: ignore[attr-defined]  # noqa: E501

    norm = torch.norm(block_to_tensor(generator)).item()
    return HamEvo(generator / norm, norm * block.duration / 1000)


@_add_interaction.register
def _(block: AnalogChain, register: Register, interaction: Callable) -> AbstractBlock:
    return chain(add_interaction(register, b, interaction=interaction) for b in block.blocks)
