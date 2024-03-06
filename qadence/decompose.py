from __future__ import annotations

import itertools
from enum import Enum
from logging import getLogger
from typing import Any, List, Tuple, Union

import sympy

from qadence.blocks import AbstractBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.parameters import Parameter, evaluate

# from qadence.types import TNumber, TParameter
from qadence.types import PI
from qadence.types import LTSOrder as Order

logger = getLogger(__name__)

# flatten a doubly-nested list
flatten = lambda a: list(itertools.chain(*a))  # noqa: E731


def change_to_z_basis(block: AbstractBlock, sign: int) -> list[AbstractBlock]:
    """A simple function to do basis transformation of blocks of X and Y.

    This needs to be generalized beyond 2 terms
    """

    # import here due to circular import issue
    from qadence.operations import RX, H, X, Y

    qubit = block.qubit_support[0]

    if isinstance(block, X):
        return [H(target=qubit)]

    elif isinstance(block, Y):
        return [RX(parameter=sign * PI / 2.0, target=qubit)]

    return []


def time_discretisation(
    parameter: Parameter, max_steps: int = 10
) -> Tuple[Union[float, complex, Any], int]:
    """Checks and returns a numerically stable.

    time step that is used in the product formula
    """
    # the approximation gives better results for t -> 0
    # the delta t needs to be numerically stable
    # ! constant time steps
    # ! the more steps, the more computationally expensive circuit
    # https://arxiv.org/pdf/1403.3469.pdf

    # check the time and log warning on duration if needed
    time = evaluate(parameter)

    if (time / max_steps) > 1e-3:  # type: ignore
        logger.warning(
            """Please consider running the H evolution for
             a shorter time to get a better approximation."""
        )

    t_delta = parameter / max_steps  # ! check numerical stability
    return t_delta, max_steps


def lie_trotter_suzuki(
    block: AbstractBlock | List, parameter: Parameter, order: Enum = Order.BASIC
) -> list[AbstractBlock]:
    # get the block and transform it to a list of blocks
    # do the correct decomposition
    # return a list of blocks

    if not isinstance(block, list):
        block_list = unroll_block_with_scaling(block)
    else:  # recursive for 4th order
        block_list = block

    if order == Order.BASIC:  # Lie-Trotter 1st order
        return decompose_pauli_exp(block_list, parameter)

    else:  # Suzuki-Trotter 2nd and 4th order
        # ! handle time properly, break up into small time steps
        # get a useful numerically stable time step

        t_delta, t_steps = time_discretisation(parameter)

        if order == Order.ST2:  # Suzuki-Trotter 2nd order
            outer = decompose_pauli_exp(block_list[:-1], Parameter(t_delta / 2.0))
            inner = decompose_pauli_exp([block_list[-1]], Parameter(t_delta))
            return (outer + inner + list(reversed(outer))) * t_steps

        else:  # Suzuki-Trotter 4th order
            p2 = (4 - 4 ** (1 / 3)) ** -1  # minimises the 'ideal' error in the recursive formula
            outer = lie_trotter_suzuki(block_list, Parameter((t_delta * p2)), order=Order.ST2)
            inner = lie_trotter_suzuki(
                block_list, Parameter((1 - 4 * p2) * t_delta), order=Order.ST2
            )
            return (2 * outer + inner + outer * 2) * t_steps


def decompose_pauli_exp(block_list: list, parameter: Parameter | sympy.Expr) -> list[AbstractBlock]:
    """A simple function to do decompositions of Pauli exponential operators into digital gates."""

    # import here due to circular import issue
    from qadence.operations import CNOT, RZ

    blocks = []

    for bl, scale in block_list:
        # extract Pauli operations and raise an error in case
        # a non-Pauli operation is found since it cannot be
        # decomposed
        n_blocks = len(get_pauli_blocks(bl))

        # ensure that we keep the parameter as trainable
        fact = 2.0 * parameter * scale

        blist: list[AbstractBlock] = bl if n_blocks >= 2 else [bl]  # type: ignore[assignment]
        indx = [b.qubit_support[0] for b in blist]
        ztarget = max(indx)

        cnot_sequence = [CNOT(i, i + 1) for i in range(min(indx), max(indx))]
        basis_fwd = [change_to_z_basis(blist[i], 1) for i in range(len(blist))]
        rot = [RZ(parameter=Parameter(fact), target=ztarget)]
        basis_bkd = [change_to_z_basis(blist[i], -1) for i in range(len(blist) - 1, -1, -1)]

        # NOTE
        # perform the following operations in sequence to perform the decomposition of a
        # polynomial Pauli term, for more details, see: https://arxiv.org/abs/1001.3855
        # - change to Z basis for all the needed qubit operators
        # - apply a CNOT ladder on the full qubit support where operators are acting
        # - apply a RZ rotation on the last qubit
        # - apply the reverse CNOT ladder
        # - go back to the original basis
        blocks.extend(
            flatten(basis_fwd)
            + cnot_sequence
            + rot
            + list(reversed(cnot_sequence))
            + flatten(basis_bkd)
        )

    return blocks
