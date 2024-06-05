from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from torch import Tensor

from qadence.backends.jax_utils import jarr_to_tensor
from qadence.execution import run
from qadence.states import (
    equivalent_state,
    ghz_block,
    ghz_state,
    is_normalized,
    one_block,
    one_state,
    product_block,
    product_state,
    rand_bitstring,
    uniform_block,
    uniform_state,
    zero_block,
    zero_state,
)


@pytest.mark.parametrize(
    "n_qubits",
    [2, 4, 6],
)
@pytest.mark.parametrize(
    "state_generators",
    [
        (one_state, one_block),
        (zero_state, zero_block),
        (uniform_state, uniform_block),
        (ghz_state, ghz_block),
    ],
)
def test_base_states(n_qubits: int, state_generators: tuple[Callable, Callable]) -> None:
    state_func, block_func = state_generators
    state_direct = state_func(n_qubits)
    block = block_func(n_qubits)
    state_block = run(block)
    assert is_normalized(state_direct)
    assert is_normalized(state_block)
    assert equivalent_state(state_direct, state_block)


@pytest.mark.parametrize(
    "n_qubits, backend",
    [(2, "pyqtorch"), (4, "horqrux"), (6, "braket")],
)
def test_product_state(n_qubits: int, backend: str) -> None:
    bitstring = rand_bitstring(n_qubits)
    state_direct = product_state(bitstring, backend=backend)
    block = product_block(bitstring)
    state_block = run(block, backend=backend)

    if not isinstance(state_direct, Tensor):
        state_direct = jarr_to_tensor(np.asarray([state_direct]))
    if not isinstance(state_block, Tensor):
        state_block = jarr_to_tensor(np.asarray([state_block]))

    assert is_normalized(state_direct)
    assert is_normalized(state_block)
    assert equivalent_state(state_direct, state_block)
