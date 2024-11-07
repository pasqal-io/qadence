from __future__ import annotations

from functools import cached_property
from logging import getLogger

import numpy as np
import torch
from torch.linalg import eigvals

from qadence.blocks import PrimitiveBlock
from qadence.noise import NoiseHandler

logger = getLogger(__name__)


class MatrixBlock(PrimitiveBlock):
    """
    Generates a MatrixBlock from a given matrix.

    Arguments:
        matrix (torch.Tensor | np.ndarray): The matrix from which to create the MatrixBlock.
        qubit_support (tuple[int]): The qubit_support of the block.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch

    from qadence.circuit import QuantumCircuit
    from qadence.types import BackendName, DiffMode
    from qadence.blocks.matrix import MatrixBlock
    from qadence.model import QuantumModel
    from qadence.operations import X, Z
    from qadence.states import random_state

    n_qubits = 1
    XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
    state = random_state(n_qubits)
    matblock = MatrixBlock(XMAT, (0,))

    qm_mat = QuantumModel(
        circuit=QuantumCircuit(n_qubits, matblock),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    qm = QuantumModel(
        circuit=QuantumCircuit(n_qubits, X(0)),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    wf_mat = qm_mat.run({}, state)
    exp_mat = qm_mat.expectation({})
    wf = qm.run({}, state)
    exp = qm.expectation({})

    assert torch.all(torch.isclose(wf_mat, wf)) and torch.isclose(exp, exp_mat)
    ```
    """

    name = "MatrixBlock"
    matrix: torch.Tensor

    def __init__(
        self,
        matrix: torch.Tensor | np.ndarray,
        qubit_support: tuple[int, ...],
        noise: NoiseHandler | None = None,
        check_unitary: bool = True,
        check_hermitian: bool = False,
    ) -> None:
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
        if matrix.ndim == 3 and matrix.size(0) == 1:
            matrix = matrix.squeeze(0)
        if not matrix.ndim == 2:
            raise TypeError("Please provide a 2D matrix.")
        if not self.is_square(matrix):
            raise TypeError("Please provide a square matrix.")
        if check_hermitian:
            if not self.is_hermitian(matrix):
                logger.warning("Provided matrix is not hermitian.")
        if check_unitary:
            if not self.is_unitary(matrix):
                logger.warning("Provided matrix is not unitary.")
        self.matrix = matrix.clone()
        super().__init__(qubit_support, noise)

    @cached_property
    def eigenvalues_generator(self) -> torch.Tensor:
        return torch.log(self.eigenvalues) * 1j

    @property
    def eigenvalues(self) -> torch.Tensor:
        ev = eigvals(self.matrix)
        _, indices = torch.sort(ev.real)
        return ev[indices]

    @property
    def n_qubits(self) -> int:
        return np.log2(self.matrix.size()[0])  # type:ignore[no-any-return]

    @staticmethod
    def is_square(m: torch.Tensor) -> bool:
        return m.shape[0] == m.shape[1]  # type:ignore[no-any-return]

    @staticmethod
    def is_hermitian(m: torch.Tensor) -> bool:
        return MatrixBlock.is_square(m) and torch.allclose(
            m.t().conj(), m
        )  # type:ignore[no-any-return]

    @staticmethod
    def is_unitary(m: torch.Tensor) -> bool:
        if not MatrixBlock.is_square(m):
            return False
        prod = torch.mm(m, m.t().conj())
        i = torch.eye(m.shape[0], dtype=torch.complex128)
        return torch.allclose(prod, i)  # type:ignore[no-any-return]

    def expand_to(self, n_qubits: int = 1) -> torch.Tensor:
        from qadence.blocks.block_to_tensor import _fill_identities

        if n_qubits > 1:
            return _fill_identities(
                self.matrix, self.qubit_support, tuple([i for i in range(n_qubits)])
            )
        return self.matrix
