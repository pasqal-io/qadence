from __future__ import annotations

from copy import deepcopy
from functools import singledispatch
from itertools import product
from math import dist as euclidean_distance
from typing import Any, Callable, Union, overload

import torch
from sympy import cos, sin

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
from qadence.blocks.utils import _construct
from qadence.circuit import QuantumCircuit
from qadence.operations import HamEvo, I, N, X, Y, add, chain, kron, wait
from qadence.qubit_support import QubitSupport
from qadence.register import Register
from qadence.transpile.transpile import blockfn_to_circfn

C6_DICT = {
    50: 96120.72,
    51: 122241.6,
    52: 154693.02,
    53: 194740.36,
    54: 243973.91,
    55: 304495.01,
    56: 378305.98,
    57: 468027.05,
    58: 576714.85,
    59: 707911.38,
    60: 865723.02,
    61: 1054903.11,
    62: 1281042.11,
    63: 1550531.15,
    64: 1870621.31,
    65: 2249728.57,
    66: 2697498.69,
    67: 3224987.51,
    68: 3844734.37,
    69: 4571053.32,
    70: 5420158.53,
    71: 6410399.4,
    72: 7562637.31,
    73: 8900342.14,
    74: 10449989.62,
    75: 12241414.53,
    76: 14308028.03,
    77: 16687329.94,
    78: 19421333.62,
    79: 22557029.94,
    80: 26146720.74,
    81: 30248886.65,
    82: 34928448.69,
    83: 40257623.67,
    84: 46316557.88,
    85: 53194043.52,
    86: 60988354.64,
    87: 69808179.15,
    88: 79773468.88,
    89: 91016513.07,
    90: 103677784.57,
    91: 117933293.96,
    92: 133943541.9,
    93: 151907135.94,
    94: 172036137.34,
    95: 194562889.89,
    96: 219741590.56,
    97: 247850178.91,
    98: 279192193.77,
    99: 314098829.39,
    100: 352931119.11,
}


def _qubitposition(register: Register, i: int) -> tuple[int, int]:
    (x, y) = list(register.coords.values())[i]
    return (x, y)


def ising_interaction(
    register: Register, pairs: list[tuple[int, int]], rydberg_level: int = 60
) -> AbstractBlock:
    c6 = C6_DICT[rydberg_level]

    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c6 / rij**6) * kron(N(i), N(j))

    return add(term(i, j) for (i, j) in pairs)


def xy_interaction(
    register: Register, pairs: list[tuple[int, int]], c3: float = 3700.0
) -> AbstractBlock:
    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c3 / rij**3) * (kron(X(i), X(j)) + kron(Y(i), Y(j)))

    return add(term(i, j) for (i, j) in pairs)


INTERACTIONS = {Interaction.NN: ising_interaction, Interaction.XY: xy_interaction}


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
    """Turns blocks or circuits into (a chain of) `HamEvo` blocks including a
    chosen interaction term.

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
    from qadence.transpile.emulate import ising_interaction

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


def rot_generator(block: ConstantAnalogRotation) -> AbstractBlock:
    omega = block.parameters.omega
    delta = block.parameters.delta
    phase = block.parameters.phase
    support = block.qubit_support

    x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
    z_terms = delta * add(N(i) for i in support)
    return x_terms - z_terms  # type: ignore[no-any-return]


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
