from __future__ import annotations

from uuid import UUID

import torch

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    ControlBlock,
    KronBlock,
    ParametricBlock,
    ParametricControlBlock,
    PrimitiveBlock,
    ScaleBlock,
)
from qadence.blocks.primitive import ProjectorBlock
from qadence.blocks.utils import chain, kron, uuid_to_expression
from qadence.parameters import evaluate, stringify

# from qadence.states import product_state
from qadence.types import PI, Endianness, TensorType, TNumber

J = torch.tensor(1j)

ZEROMAT = torch.zeros((2, 2), dtype=torch.cdouble).unsqueeze(0)
IMAT = torch.eye(2, dtype=torch.cdouble).unsqueeze(0)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble).unsqueeze(0)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble).unsqueeze(0)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble).unsqueeze(0)
NMAT = torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).unsqueeze(0)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=torch.cdouble).unsqueeze(0)
SDAGMAT = torch.tensor([[1, 0], [0, -1j]], dtype=torch.cdouble).unsqueeze(0)
TMAT = torch.tensor([[1, 0], [0, torch.exp(J * PI / 4)]], dtype=torch.cdouble).unsqueeze(0)
TDAGMAT = torch.tensor([[1, 0], [0, torch.exp(J * PI / 4)]], dtype=torch.cdouble).unsqueeze(0)
HMAT = (
    1
    / torch.sqrt(torch.tensor(2))
    * torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble).unsqueeze(0)
)


OPERATIONS_DICT = {
    "Zero": ZEROMAT,
    "I": IMAT,
    "X": XMAT,
    "Y": YMAT,
    "Z": ZMAT,
    "S": SMAT,
    "SDagger": SDAGMAT,
    "T": TMAT,
    "TDagger": TDAGMAT,
    "H": HMAT,
    "N": NMAT,
}


def _fill_identities(
    block_mat: torch.Tensor,
    qubit_support: tuple,
    full_qubit_support: tuple | list,
    diag_only: bool = False,
    endianness: Endianness = Endianness.BIG,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Returns a Kronecker product of a block matrix with identities.

    The block matrix can defined on a subset of qubits and the full matrix is
    filled with identities acting on the unused qubits.

    Args:
        block_mat (torch.Tensor): matrix of an arbitrary gate
        qubit_support (tuple): qubit support of `block_mat`
        full_qubit_support (tuple): full qubit support of the circuit
        diag_only (bool): Use diagonals only

    Returns:
        torch.Tensor: augmented matrix with dimensions (2**nqubits, 2**nqubits)
        or a tensor (2**n_qubits) if diag_only
    """
    qubit_support = tuple(sorted(qubit_support))
    block_mat = block_mat.to(device)
    mat = IMAT.to(device) if qubit_support[0] != full_qubit_support[0] else block_mat
    if diag_only:
        mat = torch.diag(mat.squeeze(0))
    for i in full_qubit_support[1:]:
        if i == qubit_support[0]:
            other = torch.diag(block_mat.squeeze(0)) if diag_only else block_mat
            if endianness == Endianness.LITTLE:
                mat = torch.kron(other, mat)
            else:
                mat = torch.kron(mat.contiguous(), other.contiguous())
        elif i not in qubit_support:
            other = torch.diag(IMAT.squeeze(0).to(device)) if diag_only else IMAT.to(device)
            if endianness == Endianness.LITTLE:
                mat = torch.kron(other.contiguous(), mat.contiguous())
            else:
                mat = torch.kron(mat.contiguous(), other.contiguous())

    return mat


def _rot_matrices(theta: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    """
    Args:

        theta(torch.Tensor): input parameter
        generator(torch.Tensor): the tensor of the generator
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    batch_size = theta.size(0)

    cos_t = torch.cos(theta / 2).unsqueeze(1).unsqueeze(2)
    cos_t = cos_t.repeat((1, 2, 2))
    sin_t = torch.sin(theta / 2).unsqueeze(1).unsqueeze(2)
    sin_t = sin_t.repeat((1, 2, 2))

    batch_imat = IMAT.repeat(batch_size, 1, 1)
    batch_generator = generator.repeat(batch_size, 1, 1)

    return cos_t * batch_imat - 1j * sin_t * batch_generator


def _u_matrix(theta: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Args:

        theta(tuple[torch.Tensor]): tuple of torch Tensor with 3 elements
            per each parameter of the arbitrary rotation
    Returns:
        torch.Tensor: matrix corresponding to the U gate after applying theta
    """
    z_phi = _rot_matrices(theta[0], OPERATIONS_DICT["Z"])
    y_theta = _rot_matrices(theta[1], OPERATIONS_DICT["Y"])
    z_omega = _rot_matrices(theta[2], OPERATIONS_DICT["Z"])

    res = torch.matmul(y_theta, z_phi)
    res = torch.matmul(z_omega, res)
    return res


def _phase_matrix(theta: torch.Tensor | TNumber) -> torch.Tensor:
    """
    Args:

        theta(torch.Tensor): input parameter
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    exp_t = torch.exp(1j * theta).unsqueeze(1).unsqueeze(2)
    exp_t = exp_t.repeat((1, 2, 2))
    return 0.5 * (IMAT + ZMAT) + exp_t * 0.5 * (IMAT - ZMAT)


def _parametric_matrix(gate: ParametricBlock, values: dict[str, torch.Tensor]) -> torch.Tensor:
    from qadence.operations import PHASE, RX, RY, RZ, U

    theta = _gate_parameters(gate, values)
    if isinstance(gate, (RX, RY, RZ)):
        pmat = _rot_matrices(
            theta[0], OPERATIONS_DICT[gate.generator.name]  # type:ignore[union-attr]
        )
    elif isinstance(gate, U):
        pmat = _u_matrix(theta)
    elif isinstance(gate, PHASE):
        pmat = _phase_matrix(theta[0])
    return pmat


def _controlled_block_with_params(
    block: ParametricControlBlock | ControlBlock,
) -> tuple[AbstractBlock, dict[str, torch.Tensor]]:
    """Redefines parameterized/non-parameterized controlled block.

    Args:
        block (ParametricControlBlock): original controlled rotation block

    Returns:
        AbstractBlock: redefined controlled rotation block
        dict with new parameters which are added
    """
    from qadence.operations import I
    from qadence.utils import P1

    # redefine controlled rotation block in a way suitable for matrix evaluation
    control = block.qubit_support[:-1]
    target = block.qubit_support[-1]
    p1 = kron(P1(qubit) for qubit in control)
    p0 = I(control[0]) - p1
    c_block = kron(p0, I(target)) + kron(p1, block.blocks[0])

    uuid_expr = uuid_to_expression(c_block)
    newparams = {
        stringify(expr): evaluate(expr, {}, as_torch=True)
        for uuid, expr in uuid_expr.items()
        if expr.is_number
    }

    return c_block, newparams


def _swap_block(block: AbstractBlock) -> AbstractBlock:
    """Redefines SWAP block.

    Args:
        block (AbstractBlock): original SWAP block

    Returns:
        AbstractBlock: redefined SWAP block
    """
    from qadence.operations import CNOT

    # redefine controlled rotation block in a way suitable for matrix evaluation
    control = block.qubit_support[0]
    target = block.qubit_support[1]
    swap_block = chain(CNOT(control, target), CNOT(target, control), CNOT(control, target))

    return swap_block


def _cswap_block(block: AbstractBlock) -> torch.Tensor:
    from qadence.operations import Toffoli

    control = block.qubit_support[0]
    target1 = block.qubit_support[1]
    target2 = block.qubit_support[2]

    cswap_block = chain(
        Toffoli((control, target2), target1),
        Toffoli((control, target1), target2),
        Toffoli((control, target2), target1),
    )

    return cswap_block


def _extract_param_names_or_uuids(b: AbstractBlock, uuids: bool = False) -> tuple[str, ...]:
    if isinstance(b, ParametricBlock):
        return (
            tuple(b.parameters.uuids())
            if uuids
            else tuple(map(stringify, b.parameters.expressions()))
        )
    else:
        return ()


def is_valid_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def _gate_parameters(b: AbstractBlock, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    uuids = is_valid_uuid(list(values.keys())[0])
    ks = _extract_param_names_or_uuids(b, uuids=uuids)
    return tuple(values[k] for k in ks)


def block_to_diagonal(
    block: AbstractBlock,
    qubit_support: tuple | list | None = None,
    use_full_support: bool = True,
    endianness: Endianness = Endianness.BIG,
    device: torch.device = None,
) -> torch.Tensor:
    if block.is_parametric:
        raise TypeError("Sparse observables cant be parametric.")
    if not block._is_diag_pauli:
        raise TypeError("Sparse observables can only be used on paulis which are diagonal.")
    if qubit_support is None:
        if use_full_support:
            qubit_support = tuple(range(0, block.n_qubits))
        else:
            qubit_support = block.qubit_support
    nqubits = len(qubit_support)  # type: ignore [arg-type]
    if isinstance(block, (ChainBlock, KronBlock)):
        v = torch.ones(2**nqubits, dtype=torch.cdouble)
        for b in block.blocks:
            v *= block_to_diagonal(b, qubit_support)
    if isinstance(block, AddBlock):
        t = torch.zeros(2**nqubits, dtype=torch.cdouble)
        for b in block.blocks:
            t += block_to_diagonal(b, qubit_support)
        v = t
    elif isinstance(block, ScaleBlock):
        _s = evaluate(block.scale, {}, as_torch=True)  # type: ignore[attr-defined]
        _s = _s.detach()  # type: ignore[union-attr]
        v = _s * block_to_diagonal(block.block, qubit_support)

    elif isinstance(block, PrimitiveBlock):
        v = _fill_identities(
            OPERATIONS_DICT[block.name],
            block.qubit_support,
            qubit_support,  # type: ignore [arg-type]
            diag_only=True,
            endianness=endianness,
        )
    return v


# version that will accept user params
def block_to_tensor(
    block: AbstractBlock,
    values: dict[str, TNumber | torch.Tensor] = {},
    qubit_support: tuple | None = None,
    use_full_support: bool = True,
    tensor_type: TensorType = TensorType.DENSE,
    endianness: Endianness = Endianness.BIG,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert a block into a torch tensor.

    Arguments:
        block (AbstractBlock): The block to convert.
        values (dict): A optional dict with values for parameters.
        qubit_support (tuple): The qubit_support of the block.
        use_full_support (bool): True infers the total number of qubits.
        tensor_type (TensorType): the target tensor type.

    Returns:
        A torch.Tensor.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence import hea, hamiltonian_factory, Z, block_to_tensor

    block = hea(2,2)
    print(block_to_tensor(block))

    # In case you have a diagonal observable, you can use
    obs = hamiltonian_factory(2, detuning = Z)
    print(block_to_tensor(obs, tensor_type="SparseDiagonal"))
    ```
    """

    # FIXME: default use_full_support to False. In general, it would
    # be more efficient to do that, and make sure that computations such
    # as observables only do the matmul of the size of the qubit support.

    if tensor_type == TensorType.DENSE:
        from qadence.blocks import embedding

        (ps, embed) = embedding(block)
        return _block_to_tensor_embedded(
            block,
            embed(ps, values),
            qubit_support,
            use_full_support,
            endianness=endianness,
            device=device,
        )

    elif tensor_type == TensorType.SPARSEDIAGONAL:
        t = block_to_diagonal(block, endianness=endianness)
        indices, values, size = torch.nonzero(t), t[t != 0], len(t)
        indices = torch.stack((indices.flatten(), indices.flatten()))
        return torch.sparse_coo_tensor(indices, values, (size, size))


# version that accepts embedded params
def _block_to_tensor_embedded(
    block: AbstractBlock,
    values: dict[str, TNumber | torch.Tensor] = {},
    qubit_support: tuple | None = None,
    use_full_support: bool = True,
    endianness: Endianness = Endianness.BIG,
    device: torch.device = None,
) -> torch.Tensor:
    from qadence.blocks import MatrixBlock
    from qadence.operations import CSWAP, SWAP, HamEvo

    # get number of qubits
    if qubit_support is None:
        if use_full_support:
            qubit_support = tuple(range(0, block.n_qubits))
        else:
            qubit_support = block.qubit_support
    nqubits = len(qubit_support)

    if isinstance(block, (ChainBlock, KronBlock)):
        # create identity matrix of appropriate dimensions
        mat = IMAT.clone().to(device)
        for i in range(nqubits - 1):
            mat = torch.kron(mat, IMAT.to(device))

        # perform matrix multiplications
        for b in block.blocks:
            other = _block_to_tensor_embedded(
                b, values, qubit_support, endianness=endianness, device=device
            )
            mat = torch.matmul(other, mat)

    elif isinstance(block, AddBlock):
        # create zero matrix of appropriate dimensions
        mat = ZEROMAT.clone().to(device)
        for _ in range(nqubits - 1):
            mat = torch.kron(mat, ZEROMAT.to(device))

        # perform matrix summation
        for b in block.blocks:
            mat = mat + _block_to_tensor_embedded(
                b, values, qubit_support, endianness=endianness, device=device
            )

    elif isinstance(block, HamEvo):
        if block.qubit_support:
            if isinstance(block.generator, AbstractBlock):
                # get matrix representation of generator
                gen_mat = _block_to_tensor_embedded(
                    block.generator, values, qubit_support, endianness=endianness, device=device
                )

                # calculate evolution matrix
                (p,) = _gate_parameters(block, values)
                prefac = -J.to(device) * p
                mat = torch.linalg.matrix_exp(prefac * gen_mat)
            elif isinstance(block.generator, torch.Tensor):
                gen_mat = block.generator.to(device)

                # calculate evolution matrix
                (p, _) = _gate_parameters(block, values)
                prefac = -J.to(device) * p
                mat = torch.linalg.matrix_exp(prefac * gen_mat)

                # add missing identities on unused qubits
                mat = _fill_identities(
                    mat, block.qubit_support, qubit_support, endianness=endianness, device=device
                )
            else:
                raise TypeError(
                    f"Generator of type {type(block.generator)} not supported in HamEvo."
                )
        else:
            raise ValueError("qubit_support is not defined for HamEvo block.")

        mat = mat.unsqueeze(0) if len(mat.size()) == 2 else mat

    elif isinstance(block, CSWAP):
        cswap_block = _cswap_block(block)
        mat = _block_to_tensor_embedded(
            cswap_block, values, qubit_support, endianness=endianness, device=device
        )

    elif isinstance(block, (ControlBlock, ParametricControlBlock)):
        c_block, newparams = _controlled_block_with_params(block)
        newparams.update(values)
        mat = _block_to_tensor_embedded(
            c_block, newparams, qubit_support, endianness=endianness, device=device
        )

    elif isinstance(block, ScaleBlock):
        (scale,) = _gate_parameters(block, values)
        mat = scale * _block_to_tensor_embedded(
            block.block, values, qubit_support, endianness=endianness, device=device
        )

    elif isinstance(block, ParametricBlock):
        block_mat = _parametric_matrix(block, values).to(device)

        # add missing identities on unused qubits
        mat = _fill_identities(
            block_mat, block.qubit_support, qubit_support, endianness=endianness, device=device
        )

    elif isinstance(block, MatrixBlock):
        mat = block.matrix.unsqueeze(0)
        # FIXME: properly handle identity filling in matrix blocks
        # mat = _fill_identities(
        #    block.matrix.unsqueeze(0),
        #    block.qubit_support,
        #    qubit_support,
        #    endianness=endianness,
        # )

    elif isinstance(block, SWAP):
        swap_block = _swap_block(block)
        mat = _block_to_tensor_embedded(
            swap_block, values, qubit_support, endianness=endianness, device=device
        )

    elif block.name in OPERATIONS_DICT.keys():
        block_mat = OPERATIONS_DICT[block.name]

        # add missing identities on unused qubits
        mat = _fill_identities(
            block_mat, block.qubit_support, qubit_support, endianness=endianness, device=device
        )

    elif isinstance(block, ProjectorBlock):
        from qadence.states import product_state

        bra = product_state(block.bra)
        ket = product_state(block.ket)

        block_mat = torch.kron(ket, bra.T)
        block_mat = block_mat.unsqueeze(0) if len(block_mat.size()) == 2 else block_mat

        mat = _fill_identities(
            block_mat.to(device),
            block.qubit_support,
            qubit_support,
            endianness=endianness,
            device=device,
        )

    else:
        raise TypeError(f"Conversion for block type {type(block)} not supported.")

    return mat
