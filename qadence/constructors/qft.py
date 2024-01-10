from __future__ import annotations

from typing import Any

import torch

from qadence.blocks import AbstractBlock, add, chain, kron, tag
from qadence.constructors import hamiltonian_factory
from qadence.operations import CPHASE, SWAP, H, HamEvo, I, Z
from qadence.types import PI, Interaction, Strategy

from .daqc import daqc_transform


def qft(
    n_qubits: int,
    support: tuple[int, ...] = None,
    inverse: bool = False,
    reverse_in: bool = False,
    swaps_out: bool = False,
    strategy: Strategy = Strategy.DIGITAL,
    gen_build: AbstractBlock | None = None,
) -> AbstractBlock:
    """
    The Quantum Fourier Transform.

    Depending on the application, user should be careful with qubit ordering
    in the input and output. This can be controlled with reverse_in and swaps_out
    arguments.

    Args:
        n_qubits: number of qubits in the QFT
        support: qubit support to use
        inverse: True performs the inverse QFT
        reverse_in: Reverses the input qubits to account for endianness
        swaps_out: Performs swaps on the output qubits to match the "textbook" QFT.
        strategy: Strategy.Digital or Strategy.sDAQC
        gen_build: building block Ising Hamiltonian for the DAQC transform.
            Defaults to constant all-to-all Ising.

    Examples:
        ```python exec="on" source="material-block" result="json"
        from qadence import qft

        n_qubits = 3

        qft_circuit = qft(n_qubits, strategy = "sDAQC")
        ```
    """

    if support is None:
        support = tuple(range(n_qubits))

    assert len(support) <= n_qubits, "Wrong qubit support supplied"

    if reverse_in:
        support = support[::-1]

    qft_layer_dict = {
        Strategy.DIGITAL: _qft_layer_digital,
        Strategy.SDAQC: _qft_layer_sDAQC,
        Strategy.BDAQC: _qft_layer_bDAQC,
        Strategy.ANALOG: _qft_layer_analog,
    }

    try:
        layer_func = qft_layer_dict[strategy]
    except KeyError:
        raise KeyError(f"Strategy {strategy} not recognized.")

    qft_layers = reversed(range(n_qubits)) if inverse else range(n_qubits)

    qft_circ = chain(
        layer_func(
            n_qubits=n_qubits, support=support, layer=layer, inverse=inverse, gen_build=gen_build
        )  # type: ignore
        for layer in qft_layers
    )

    if swaps_out:
        swap_ops = [SWAP(support[i], support[n_qubits - i - 1]) for i in range(n_qubits // 2)]
        qft_circ = chain(*swap_ops, qft_circ) if inverse else chain(qft_circ, *swap_ops)

    return tag(qft_circ, tag="iQFT") if inverse else tag(qft_circ, tag="QFT")


########################
# STANDARD DIGITAL QFT #
########################


def _qft_layer_digital(
    n_qubits: int,
    support: tuple[int, ...],
    layer: int,
    inverse: bool,
    gen_build: AbstractBlock | None = None,
) -> AbstractBlock:
    """
    Apply the Hadamard gate followed by CPHASE gates.

    This corresponds to one layer of the QFT.
    """
    qubit_range_layer = (
        reversed(range(layer + 1, n_qubits)) if inverse else range(layer + 1, n_qubits)
    )
    rots = []
    for j in qubit_range_layer:
        angle = torch.tensor(
            ((-1) ** inverse) * 2 * PI / (2 ** (j - layer + 1)), dtype=torch.cdouble
        )
        rots.append(CPHASE(support[j], support[layer], angle))  # type: ignore
    if inverse:
        return chain(*rots, H(support[layer]))  # type: ignore
    return chain(H(support[layer]), *rots)  # type: ignore


########################################
# DIGITAL-ANALOG QFT (with sDAQC)      #
# [1] https://arxiv.org/abs/1906.07635 #
########################################


def _theta(k: int) -> float:
    """Equation (16) from [1]."""
    return float(PI / (2 ** (k + 1)))


def _alpha(c: int, m: int, k: int) -> float:
    """Equation (16) from [1]."""
    if c == m:
        return float(PI / (2 ** (k - m + 2)))
    else:
        return 0.0


def _sqg_gen(n_qubits: int, support: tuple[int, ...], m: int, inverse: bool) -> list[AbstractBlock]:
    """Equation (13) from [1].

    Creates the generator corresponding to single-qubit rotations coming
    out of the CPHASE decomposition. The paper also includes the generator
    for the Hadamard of each layer here, but we left it explicit at
    the start of each layer.
    """
    k_sqg_list = reversed(range(2, n_qubits - m + 2)) if inverse else range(2, n_qubits - m + 2)

    sqg_gen_list = []
    for k in k_sqg_list:
        sqg_gen = (
            kron(I(support[j]) for j in range(n_qubits)) - Z(support[k + m - 2]) - Z(support[m - 1])
        )
        sqg_gen_list.append(_theta(k) * sqg_gen)

    return sqg_gen_list


def _tqg_gen(n_qubits: int, support: tuple[int, ...], m: int, inverse: bool) -> list[AbstractBlock]:
    """Equation (14) from [1].

    Creates the generator corresponding to the two-qubit ZZ
    interactions coming out of the CPHASE decomposition.
    """
    k_tqg_list = reversed(range(2, n_qubits + 1)) if inverse else range(2, n_qubits + 1)

    tqg_gen_list = []
    for k in k_tqg_list:
        for c in range(1, k):
            tqg_gen = kron(Z(support[c - 1]), Z(support[k - 1]))
            tqg_gen_list.append(_alpha(c, m, k) * tqg_gen)

    return tqg_gen_list


def _qft_layer_sDAQC(
    n_qubits: int,
    support: tuple[int, ...],
    layer: int,
    inverse: bool,
    gen_build: AbstractBlock | None,
) -> AbstractBlock:
    """
    QFT Layer using the sDAQC technique.

    Following the paper:

    -- [1] https://arxiv.org/abs/1906.07635

    4 - qubit edge case is not implemented.

    Note: the paper follows an index convention of running from 1 to N. A few functions
    here also use that convention to be consistent with the paper. However, for qadence
    related things the indices are converted to [0, N-1].
    """

    # TODO: Properly check and include support for changing qubit support
    allowed_support = tuple(range(n_qubits))
    if support != allowed_support and support != allowed_support[::-1]:
        raise NotImplementedError("Changing support for DigitalAnalog QFT not yet supported.")

    if gen_build is None:
        gen_build = hamiltonian_factory(n_qubits, interaction=Interaction.NN)

    m = layer + 1  # Paper index convention

    # Generator for the single-qubit rotations contributing to the CPHASE gate
    sqg_gen_list = _sqg_gen(n_qubits=n_qubits, support=support, m=m, inverse=inverse)

    # Ising model representing the CPHASE gates two-qubit interactions
    tqg_gen_list = _tqg_gen(n_qubits=n_qubits, support=support, m=m, inverse=inverse)

    if len(sqg_gen_list) > 0:
        # Single-qubit rotations (leaving the Hadamard explicit)
        sq_gate = chain(H(support[m - 1]), HamEvo(add(*sqg_gen_list), -1.0))

        # Two-qubit interaction in the CPHASE converted with sDAQC
        gen_cphases = add(*tqg_gen_list)
        transformed_daqc_circuit = daqc_transform(
            n_qubits=gen_build.n_qubits,
            gen_target=gen_cphases,
            t_f=-1.0,
            gen_build=gen_build,
        )

        layer_circ = chain(
            sq_gate,
            transformed_daqc_circuit,
        )
        if inverse:
            return layer_circ.dagger()
        return layer_circ  # type: ignore
    else:
        return chain(H(support[m - 1]))  # type: ignore


########################################
# DIGITAL-ANALOG QFT (with bDAQC)      #
# [1] https://arxiv.org/abs/1906.07635 #
########################################


def _qft_layer_bDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


############
## ANALOG ##
############


def _qft_layer_analog(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
