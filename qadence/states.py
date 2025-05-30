from __future__ import annotations

import random
from functools import singledispatch
from typing import List

import torch
from numpy.typing import ArrayLike
from pyqtorch.utils import DensityMatrix
from torch import Tensor, concat
from torch.distributions import Categorical, Distribution

from qadence.blocks import ChainBlock, KronBlock, PrimitiveBlock, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.execution import run
from qadence.logger import get_script_logger
from qadence.operations import CNOT, RX, RY, RZ, H, I, X
from qadence.types import PI, BackendName, Endianness, StateGeneratorType
from qadence.utils import basis_to_int

# Modules to be automatically added to the qadence namespace
__all__ = [
    "uniform_state",
    "zero_state",
    "one_state",
    "product_state",
    "rand_product_state",
    "ghz_state",
    "random_state",
    "uniform_block",
    "one_block",
    "zero_block",
    "product_block",
    "rand_product_block",
    "ghz_block",
    "pmf",
    "normalize",
    "is_normalized",
    "rand_bitstring",
    "equivalent_state",
    "DensityMatrix",
    "density_mat",
    "overlap",
    "partial_trace",
    "von_neumann_entropy",
    "purity",
    "fidelity",
]

ATOL_64 = 1e-14  # 64 bit precision
NORMALIZATION_ATOL = ATOL_64
DTYPE = torch.cdouble

parametric_single_qubit_gates: List = [RX, RY, RZ]

logger = get_script_logger(__name__)
# PRIVATE


def _rand_haar_fast(n_qubits: int) -> Tensor:
    # inspired by https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/random.html#random_statevector
    N = 2**n_qubits
    x = -torch.log(torch.rand(N))
    sumx = torch.sum(x)
    phases = torch.rand(N) * 2.0 * PI
    return (torch.sqrt(x / sumx) * torch.exp(1j * phases)).reshape(1, N)


def _rand_haar_slow(n_qubits: int) -> Tensor:
    """
    Detailed in https://arxiv.org/pdf/math-ph/0609050.pdf.

    Textbook implementation, but very expensive. For 12 qubits it takes several seconds.
    For 1 qubit it seems to produce the same distribution as the measure above.
    """
    N = 2**n_qubits
    A = torch.zeros(N, N, dtype=DTYPE).normal_(0, 1)
    B = torch.zeros(N, N, dtype=DTYPE).normal_(0, 1)
    Z = A + 1.0j * B
    Q, R = torch.linalg.qr(Z)
    Lambda = torch.diag(torch.diag(R) / torch.diag(R).abs())
    haar_unitary = torch.matmul(Q, Lambda)
    return torch.matmul(haar_unitary, zero_state(n_qubits).squeeze(0)).unsqueeze(0)


def _from_op(op: type[PrimitiveBlock], n_qubits: int) -> KronBlock:
    return kron(op(i) for i in range(n_qubits))  # type: ignore[arg-type]


def _block_from_bitstring(bitstring: str) -> KronBlock:
    n_qubits = len(bitstring)
    gates = []
    for i, b in zip(range(n_qubits), bitstring):
        gates.append(X(i)) if b == "1" else gates.append(I(i))  # type: ignore[arg-type]
    return kron(*gates)


def _state_from_bitstring(
    bitstring: str, batch_size: int, endianness: Endianness = Endianness.BIG
) -> Tensor:
    n_qubits = len(bitstring)
    wf_batch = torch.zeros(batch_size, 2**n_qubits, dtype=DTYPE)
    k = basis_to_int(basis=bitstring, endianness=endianness)
    wf_batch[:, k] = torch.tensor(1.0 + 0j, dtype=DTYPE)
    return wf_batch


def _abstract_random_state(
    n_qubits: int, batch_size: int = 1
) -> QuantumCircuit | list[QuantumCircuit]:
    qc_list = []
    for i in range(batch_size):
        gates_list = []
        for i in range(n_qubits):
            gate = parametric_single_qubit_gates[
                random.randrange(len(parametric_single_qubit_gates))
            ]
            angle = random.uniform(-2, 2)
            gates_list.append(gate(i, angle))
        qc_list.append(QuantumCircuit(n_qubits, chain(*gates_list)))
    return qc_list[0] if batch_size == 1 else qc_list


# STATES


def uniform_state(n_qubits: int, batch_size: int = 1) -> Tensor:
    """
    Generates the uniform state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.
        batch_size (int): The batch size.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import uniform_state

    state = uniform_state(n_qubits=2)
    print(state)
    ```
    """
    norm = 1 / torch.sqrt(torch.tensor(2**n_qubits))
    return norm * torch.ones(batch_size, 2**n_qubits, dtype=DTYPE)


def zero_state(n_qubits: int, batch_size: int = 1) -> Tensor:
    """
    Generates the zero state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits for which the zero state is to be generated.
        batch_size (int): The batch size for the zero state.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import zero_state

    state = zero_state(n_qubits=2)
    print(state)
    ```
    """
    bitstring = "0" * n_qubits
    return _state_from_bitstring(bitstring, batch_size)


def one_state(n_qubits: int, batch_size: int = 1) -> Tensor:
    """
    Generates the one state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.
        batch_size (int): The batch size.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import one_state

    state = one_state(n_qubits=2)
    print(state)
    ```
    """
    bitstring = "1" * n_qubits
    return _state_from_bitstring(bitstring, batch_size)


@singledispatch
def product_state(
    bitstring: str,
    batch_size: int = 1,
    endianness: Endianness = Endianness.BIG,
    backend: BackendName = BackendName.PYQTORCH,
) -> ArrayLike:
    """
    Creates a product state from a bitstring.

    Arguments:
        bitstring (str): A bitstring.
        batch_size (int) : Batch size.
        backend (BackendName): The backend to use. Default is "pyqtorch".

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import product_state

    print(product_state("1100", backend="pyqtorch"))
    print(product_state("1100", backend="horqrux"))
    ```
    """
    if batch_size:
        logger.debug(
            "The input `batch_size` is going to be deprecated. "
            "For now, default batch_size is set to 1."
        )
    return run(product_block(bitstring), backend=backend, endianness=endianness)


@product_state.register
def _(bitstrings: list) -> Tensor:  # type: ignore
    return concat(tuple(product_state(b) for b in bitstrings), dim=0)


def rand_product_state(n_qubits: int, batch_size: int = 1) -> Tensor:
    """
    Creates a random product state.

    Arguments:
        n_qubits (int): The number of qubits.
        batch_size (int): How many bitstrings to use.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import rand_product_state

    print(rand_product_state(n_qubits=2, batch_size=2))
    ```
    """
    wf_batch = torch.zeros(batch_size, 2**n_qubits, dtype=DTYPE)
    rand_pos = torch.randint(0, 2**n_qubits, (batch_size,))
    wf_batch[torch.arange(batch_size), rand_pos] = torch.tensor(1.0 + 0j, dtype=DTYPE)
    return wf_batch


def ghz_state(n_qubits: int, batch_size: int = 1) -> Tensor:
    """
    Creates a GHZ state.

    Arguments:
        n_qubits (int): The number of qubits.
        batch_size (int): How many bitstrings to use.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import ghz_state

    print(ghz_state(n_qubits=2, batch_size=2))
    ```
    """
    norm = 1 / torch.sqrt(torch.tensor(2))
    return norm * (zero_state(n_qubits, batch_size) + one_state(n_qubits, batch_size))


def random_state(
    n_qubits: int,
    batch_size: int = 1,
    backend: str = BackendName.PYQTORCH,
    type: StateGeneratorType = StateGeneratorType.HAAR_MEASURE_FAST,
) -> Tensor:
    """
    Generates a random state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.
        backend (str): The backend to use.
        batch_size (int): The batch size.
        type : StateGeneratorType.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import random_state, StateGeneratorType
    from qadence.states import random_state, is_normalized, pmf
    from qadence.types import BackendName
    from torch.distributions import Distribution

    ### We have the following options:
    print([g.value for g in StateGeneratorType])

    n_qubits = 2
    # The default is StateGeneratorType.HAAR_MEASURE_FAST
    state = random_state(n_qubits=n_qubits)
    print(state)

    ### Lets initialize a state using random rotations, i.e., StateGeneratorType.RANDOM_ROTATIONS.
    random = random_state(n_qubits=n_qubits, type=StateGeneratorType.RANDOM_ROTATIONS)
    print(random)
    ```
    """

    if type == StateGeneratorType.HAAR_MEASURE_FAST:
        state = concat(tuple(_rand_haar_fast(n_qubits) for _ in range(batch_size)), dim=0)
    elif type == StateGeneratorType.HAAR_MEASURE_SLOW:
        state = concat(tuple(_rand_haar_slow(n_qubits) for _ in range(batch_size)), dim=0)
    elif type == StateGeneratorType.RANDOM_ROTATIONS:
        state = run(_abstract_random_state(n_qubits, batch_size))  # type: ignore
    assert all(list(map(is_normalized, state)))
    return state


# DENSITY MATRIX


def density_mat(state: Tensor) -> DensityMatrix:
    """
    Computes the density matrix from a pure state vector.

    Arguments:
        state: The pure state vector :math:`|\\psi\\rangle`.

    Returns:
        Tensor: The density matrix :math:`\\rho = |\psi \\rangle \\langle\\psi|`.
    """
    if isinstance(state, DensityMatrix):
        return state
    return DensityMatrix(torch.einsum("bi,bj->bij", (state, state.conj())))


# BLOCKS


def uniform_block(n_qubits: int) -> KronBlock:
    """
    Generates the abstract uniform state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.

    Returns:
        A KronBlock representing the uniform state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import uniform_block

    block = uniform_block(n_qubits=2)
    print(block)
    ```
    """
    return _from_op(H, n_qubits=n_qubits)


def one_block(n_qubits: int) -> KronBlock:
    """
    Generates the abstract one state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.

    Returns:
        A KronBlock representing the one state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import one_block

    block = one_block(n_qubits=2)
    print(block)
    ```
    """
    return _from_op(X, n_qubits=n_qubits)


def zero_block(n_qubits: int) -> KronBlock:
    """
    Generates the abstract zero state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.

    Returns:
        A KronBlock representing the zero state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import zero_block

    block = zero_block(n_qubits=2)
    print(block)
    ```
    """
    return _from_op(I, n_qubits=n_qubits)


def product_block(bitstring: str) -> KronBlock:
    """
    Creates an abstract product state from a bitstring.

    Arguments:
        bitstring (str): A bitstring.

    Returns:
        A KronBlock representing the product state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import product_block

    print(product_block("1100"))
    ```
    """
    return _block_from_bitstring(bitstring)


def rand_product_block(n_qubits: int) -> KronBlock:
    """
    Creates a block representing a random abstract product state.

    Arguments:
        n_qubits (int): The number of qubits.

    Returns:
        A KronBlock representing the product state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import rand_product_block

    print(rand_product_block(n_qubits=2))
    ```
    """
    return product_block(rand_bitstring(n_qubits))


def ghz_block(n_qubits: int) -> ChainBlock:
    """
    Generates the abstract ghz state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits.

    Returns:
        A ChainBlock representing the GHZ state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import ghz_block

    block = ghz_block(n_qubits=2)
    print(block)
    ```
    """
    cnots = chain(CNOT(i - 1, i) for i in range(1, n_qubits))
    return chain(H(0), cnots)


# UTILITIES


def pmf(wf: Tensor) -> Distribution:
    """
    Converts a wave function into a torch Distribution.

    Arguments:
        wf (torch.Tensor): The wave function as a torch tensor.

    Returns:
        A torch.distributions.Distribution.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import uniform_state, pmf

    print(pmf(uniform_state(2)).probs)
    ```
    """
    return Categorical(torch.abs(torch.pow(wf, 2)))


def normalize(wf: Tensor) -> Tensor:
    """
    Normalizes a wavefunction or batch of wave functions.

    Arguments:
        wf (torch.Tensor): Normalized wavefunctions.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import uniform_state, normalize

    print(normalize(uniform_state(2, 2)))
    ```
    """
    if wf.dim() == 1:
        return wf / torch.sqrt((wf.abs() ** 2).sum())
    else:
        return wf / torch.sqrt((wf.abs() ** 2).sum(1)).unsqueeze(1)


def is_normalized(wf: Tensor, atol: float = NORMALIZATION_ATOL) -> bool:
    """
    Checks if a wave function is normalized.

    Arguments:
        wf (torch.Tensor): The wave function as a torch tensor.
        atol (float) : The tolerance.

    Returns:
        A bool.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import uniform_state, is_normalized

    print(is_normalized(uniform_state(2)))
    ```
    """
    if wf.dim() == 1:
        wf = wf.unsqueeze(0)
    sum_probs: Tensor = (wf.abs() ** 2).sum(dim=1)
    ones = torch.ones_like(sum_probs)
    return torch.allclose(sum_probs, ones, rtol=0.0, atol=atol)  # type: ignore[no-any-return]


def rand_bitstring(N: int) -> str:
    """
    Creates a random bistring.

    Arguments:
        N (int): The length of the bitstring.

    Returns:
        A string.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import rand_bitstring

    print(rand_bitstring(N=8))
    ```
    """
    return "".join(str(random.randint(0, 1)) for _ in range(N))


def overlap(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact overlap between two statevectors.

    Arguments:
        s0 (torch.Tensor): A statevector or batch of statevectors.
        s1 (torch.Tensor): A statevector or batch of statevectors.

    Returns:
        A torch.Tensor with the result.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.states import rand_bitstring

    print(rand_bitstring(N=8))
    ```
    """
    from qadence.overlap import overlap_exact

    return overlap_exact(s0, s1)


def equivalent_state(
    s0: torch.Tensor, s1: torch.Tensor, rtol: float = 0.0, atol: float = NORMALIZATION_ATOL
) -> bool:
    fidelity = overlap(s0, s1)
    expected = torch.ones_like(fidelity)
    return torch.allclose(fidelity, expected, rtol=rtol, atol=atol)  # type: ignore[no-any-return]


# DensityMatrix utility functions


def partial_trace(rho: DensityMatrix, keep_indices: list[int]) -> DensityMatrix:
    """
    Compute the partial trace of a density matrix for a system of several qubits with batch size.

    This function also permutes qubits according to the order specified in keep_indices.

    Args:
        rho (DensityMatrix) : Density matrix of shape [batch_size, 2**n_qubits, 2**n_qubits].
        keep_indices (list[int]): Index of the qubit subsystems to keep.

    Returns:
        DensityMatrix: Reduced density matrix after the partial trace,
        of shape [batch_size, 2**n_keep, 2**n_keep].
    """
    from pyqtorch.utils import dm_partial_trace

    return dm_partial_trace(rho.permute((1, 2, 0)), keep_indices).permute((2, 0, 1))


def von_neumann_entropy(rho: DensityMatrix, eps: float = 1e-12) -> torch.Tensor:
    """Calculate the von Neumann entropy of a quantum density matrix.

    The von Neumann entropy is defined as S(ρ) = -Tr(ρ log₂ ρ) = -∑ᵢ λᵢ log₂ λᵢ,
    where λᵢ are the eigenvalues of ρ.

    Args:
        rho: Density matrix of shape [batch_size, dim, dim]
        eps: Small value to avoid log(0) for zero eigenvalues

    Returns:
        Von Neumann entropy for each density matrix in the batch, shape [batch_size]
    """

    # Compute eigenvalues for each density matrix in the batch
    # For a Hermitian density matrix, eigenvalues should be real and non-negative
    eigenvalues = torch.linalg.eigvalsh(rho)

    # Normalize eigenvalues to ensure they sum to 1 (trace preservation)
    # This step might be redundant but helps with numerical stability
    eigenvalues = eigenvalues / torch.sum(eigenvalues, dim=1, keepdim=True)

    # Filter out very small eigenvalues to avoid numerical issues
    valid_eigenvalues = eigenvalues.clone()
    valid_eigenvalues[valid_eigenvalues < eps] = eps

    # Compute the entropy: -∑ᵢ λᵢ log₂ λᵢ
    # Using natural logarithm and converting to base 2
    log_base_conversion = torch.log(torch.tensor(2.0, device=rho.device))
    entropy = -torch.sum(
        valid_eigenvalues * torch.log(valid_eigenvalues) / log_base_conversion, dim=1
    )

    return entropy


def purity(rho: DensityMatrix, order: int = 2) -> Tensor:
    """Compute the n-th purity of a density matrix.

    Args:
        rho (DensityMatrix): Density matrix.
        order (int, optional): Exponent n.

    Returns:
        Tensor: Tr[rho ** n]
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(rho)

    # Compute the sum of eigenvalues raised to power n
    return torch.sum(eigenvalues**order, dim=1)


def fidelity(rho: DensityMatrix, sigma: DensityMatrix) -> Tensor:
    """Calculate the fidelity between two quantum states represented by density matrices.

    The fidelity is defined as F(ρ,σ) = Tr[√(√ρ σ √ρ)], or equivalently,
    F(ρ,σ) = ||√ρ·√σ||₁ where ||·||₁ is the trace norm.

    Args:
        rho: First density matrix of shape [batch_size, dim, dim]
        sigma: Second density matrix of shape [batch_size, dim, dim]

    Returns:
        Fidelity between each pair of density matrices in the batch, shape [batch_size]
    """

    # Compute square root of rho
    rho_eigvals, rho_eigvecs = torch.linalg.eigh(rho)

    # Ensure non-negative eigenvalues
    rho_eigvals = torch.clamp(rho_eigvals, min=0)

    # Compute square root using eigendecomposition
    sqrt_eigvals = torch.sqrt(rho_eigvals)

    # Compute √ρ for each batch element
    sqrt_rho = torch.zeros_like(rho)
    for i in range(rho.shape[0]):
        sqrt_rho[i] = torch.mm(
            rho_eigvecs[i],
            torch.mm(
                torch.diag(sqrt_eigvals[i]).to(dtype=rho_eigvecs.dtype), rho_eigvecs[i].t().conj()
            ),
        )

    # Compute √ρ σ √ρ for each batch element
    inner_product = torch.zeros_like(rho)
    for i in range(rho.shape[0]):
        inner_product[i] = torch.mm(sqrt_rho[i], torch.mm(sigma[i], sqrt_rho[i]))

    # Compute eigenvalues of inner product
    inner_eigvals = torch.linalg.eigvalsh(inner_product)

    # Ensure non-negative eigenvalues
    inner_eigvals = torch.clamp(inner_eigvals, min=0)

    # Compute the fidelity as the sum of the square roots of eigenvalues
    fidelity_values = torch.sum(torch.sqrt(inner_eigvals), dim=1)

    return fidelity_values
