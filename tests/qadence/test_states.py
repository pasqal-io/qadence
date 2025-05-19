from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from torch import Tensor

from qadence.backends.jax_utils import jarr_to_tensor
from qadence.backends.utils import pyqify
from qadence.execution import run
from qadence.states import (
    DensityMatrix,
    density_mat,
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
    partial_trace,
    von_neumann_entropy,
    purity,
    fidelity,
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
    [
        (2, "pyqtorch"),
        (4, "horqrux"),
    ],
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


def test_density_mat() -> None:
    state_direct = product_state("00")
    state_dm = density_mat(state_direct)
    assert len(state_dm.shape) == 3
    assert isinstance(state_dm, DensityMatrix)
    assert state_dm.shape[0] == 1
    assert state_dm.shape[1] == state_dm.shape[2] == 4

    state_dm2 = density_mat(state_dm)
    assert isinstance(state_dm2, DensityMatrix)
    assert torch.allclose(state_dm2, state_dm)

    with pytest.raises(ValueError):
        pyqify(state_dm2.unsqueeze(0))

    with pytest.raises(ValueError):
        pyqify(state_dm2.view((1, 2, 8)))

    with pytest.raises(ValueError):
        pyqify(state_dm2.view((2, 4, 2)))


def test_density_mat_utilsfn() -> None:
    zero_zero_state = product_state("00")
    dm_00 = density_mat(zero_zero_state)
    dm_0 = density_mat(product_state("0"))
    assert torch.allclose(partial_trace(dm_00, keep_indices=[0]), dm_0)
    assert torch.allclose(von_neumann_entropy(dm_00), torch.zeros(1, dtype=torch.double))

    bell_state = DensityMatrix(torch.zeros(1, 4, 4))
    bell_state[0, 0, 0] = 0.5
    bell_state[0, 0, 3] = 0.5
    bell_state[0, 3, 0] = 0.5
    bell_state[0, 3, 3] = 0.5

    reduced_bell = partial_trace(bell_state, keep_indices=[0])
    mixed_state = torch.eye(2, dtype=reduced_bell.dtype).unsqueeze(0) / 2.0
    assert torch.allclose(reduced_bell, mixed_state)
    assert torch.allclose(von_neumann_entropy(bell_state), torch.zeros(1, dtype=torch.double))

    for n in [1, 2, 3, 4]:
        assert torch.allclose(purity(dm_00, n), torch.ones(1, dtype=torch.double))
        assert torch.allclose(purity(bell_state, n), torch.ones(1, dtype=torch.double))
        assert torch.allclose(
            purity(mixed_state, n), (2 ** (1 - n)) * torch.ones(1, dtype=torch.double)
        )

    assert torch.allclose(fidelity(dm_00, dm_00), torch.ones(1, dtype=torch.double))
    assert torch.allclose(fidelity(bell_state, bell_state), torch.ones(1, dtype=torch.double))
