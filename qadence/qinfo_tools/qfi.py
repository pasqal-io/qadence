from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad

from qadence import DiffMode, Overlap, OverlapMethod
from qadence.backend import BackendName
from qadence.blocks import parameters, primitive_blocks
from qadence.circuit import QuantumCircuit


def symsqrt(a: Tensor, cond: int | float | None = None, return_rank: bool = False) -> Tensor:
    """Computes the symmetric square root of a positive definite matrix.

    Taken from https://github.com/pytorch/pytorch/issues/25481
    """

    s, u = torch.linalg.eigh(a)
    cond_dict = {
        torch.float32: 1e3 * 1.1920929e-07,
        torch.float64: 1e6 * 2.220446049250313e-16,
    }

    if cond in [None, -1]:
        cond = cond_dict[a.dtype]

    above_cutoff = abs(s) > cond * torch.max(abs(s))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B


def _overlap_with_inputs(
    circuit: QuantumCircuit,
    vparams: tuple | None = None,
    backend: BackendName = BackendName.PYQTORCH,
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,
) -> Overlap:
    """Gives an OverlapModel consisting of the overlap of the circuit with itself.

    The vparamaters
    of the underlying QuantumModel are the vparameters of the bra state, so it's differentiable.

    Args:
        circuit (QuantumCircuit): _description_
        vparams (tuple): _description_
        backend (BackendName, optional): _description_. Defaults to BackendName.pyq.
        overlap_method (OverlapMethod, optional): _description_. Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): _description_. Defaults to DiffMode.ad.

    Returns:
        Overlap(): _description_
    """
    if vparams is not None:
        blocks = primitive_blocks(circuit.block)
        iter_index = iter(range(len(blocks)))
        for block in blocks:
            params = parameters(block)
            for p in params:
                if p.trainable:
                    p.value = float(vparams[next(iter_index)])

    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    return ovrlp_model


def get_quantum_fisher(
    circuit: QuantumCircuit,
    var_values: tuple | list | Tensor | None = None,
    fm_dict: dict[str, Tensor] = {},
    backend: BackendName = BackendName.PYQTORCH,
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,
) -> Tensor:
    """Function to calculate the exact Quantum Fisher Information (QFI) matrix.

    of the quantum state generated.

    by the quantum circuit. The QFI matrix is calculated as the Hessian of the
    fidelity of the quantum state

    Args:
        circuit (QuantumCircuit): _description_
        backend (BackendName, optional): _description_. Defaults to BackendName.pyq.
        overlap_method (OverlapMethod, optional): _description_. Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): _description_. Defaults to DiffMode.ad.

    Returns:
        torch.Tensor: QFI matrix
    """

    if fm_dict == {}:
        blocks = primitive_blocks(circuit.block)
        for block in blocks:
            params = parameters(block)
            for p in params:
                if not p.trainable:
                    fm_dict[p.name] = torch.tensor([p.value])

    # Get Overlap() model
    ovrlp_model = _overlap_with_inputs(
        circuit,
        var_values,  # type: ignore [arg-type]
        backend=backend,
        overlap_method=overlap_method,
        diff_mode=diff_mode,
    )
    ovrlp = ovrlp_model(bra_param_values=fm_dict, ket_param_values=fm_dict)

    # Retrieve variational parameters of the overlap model
    # Importantly, the vparams of the overlap model are the vparams of the bra tensor,
    # Which means if we differentiate wrt vparams we are differentiating only wrt the
    # parameters in the bra and not in the ket
    vparams = {k: v for k, v in ovrlp_model._params.items() if v.requires_grad}

    # Jacobian of the overlap
    ovrlp_grad = grad(
        ovrlp,
        list(vparams.values()),
        torch.ones_like(ovrlp),
        create_graph=True,
        allow_unused=True,
    )

    # Hessian of the overlap = QFI matrix
    n_params = ovrlp_model.num_vparams
    fid_hess = torch.empty((n_params, n_params))
    for i in range(n_params):
        ovrlp_grad2 = grad(
            ovrlp_grad[i],
            list(vparams.values()),
            torch.ones_like(ovrlp_grad[i]),
            create_graph=True,
            allow_unused=True,
        )
        for j in range(n_params):
            fid_hess[i, j] = ovrlp_grad2[j]

    return -2 * fid_hess


def get_quantum_fisher_spsa(
    circuit: QuantumCircuit,
    k: int,
    var_values: tuple | list | Tensor | None = None,
    fm_dict: dict[str, Tensor] = {},
    previous_qfi_estimator: Tensor = None,
    epsilon: float = 10e-4,
    beta: float = 10e-3,
    backend: BackendName = BackendName.PYQTORCH,
    overlap_method: OverlapMethod = OverlapMethod.EXACT,
    diff_mode: DiffMode = DiffMode.AD,
) -> Tensor:
    """Function to calculate the Quantum Fisher Information (QFI) matrix.

    of the quantum state generated.

    by the quantum circuit. The QFI matrix is calculated as the Hessian of
    the fidelity of the quantum state

    Args:
        circuit (QuantumCircuit): _description_
        backend (BackendName, optional): _description_. Defaults to BackendName.pyq.
        overlap_method (OverlapMethod, optional): _description_. Defaults to OverlapMethod.EXACT.
        diff_mode (DiffMode, optional): _description_. Defaults to DiffMode.ad.

    Returns:
        torch.Tensor: QFI matrix
    """
    num_vparams = len(var_values) if var_values is not None else 0

    # Random directions
    Delta_1 = torch.Tensor(np.random.choice([-1, 1], size=(num_vparams, 1)))
    Delta_2 = torch.Tensor(np.random.choice([-1, 1], size=(num_vparams, 1)))

    ovrlp_model = Overlap(
        circuit,
        circuit,
        backend=backend,
        diff_mode=diff_mode,
        method=overlap_method,
    )

    if fm_dict == {}:
        blocks = primitive_blocks(circuit.block)
        for block in blocks:
            params = parameters(block)
            for p in params:
                if not p.trainable:
                    fm_dict[p.name] = torch.Tensor([p.value])

    # Calculate the shifted parameters
    vparams_tensors = torch.Tensor(var_values).reshape((num_vparams, 1))
    vparams = {k: v for (k, v) in ovrlp_model._params.items() if v.requires_grad}
    vparams_plus1_plus2 = dict(zip(vparams.keys(), vparams_tensors + epsilon * (Delta_1 + Delta_2)))
    vparams_plus1 = dict(zip(vparams.keys(), vparams_tensors + epsilon * (Delta_1)))
    vparams_minus1_plus2 = dict(
        zip(vparams.keys(), vparams_tensors + epsilon * (-Delta_1 + Delta_2))
    )
    vparams_minus1 = dict(zip(vparams.keys(), vparams_tensors + epsilon * (-Delta_1)))

    # Overlaps with the shifted parameters
    ovrlp_shifter_plus1_plus2 = ovrlp_model(
        bra_param_values=fm_dict | vparams,
        ket_param_values=fm_dict | vparams_plus1_plus2,
    )
    ovrlp_shifter_plus1 = ovrlp_model(
        bra_param_values=fm_dict | vparams, ket_param_values=fm_dict | vparams_plus1
    )
    ovrlp_shifter_minus1_plus2 = ovrlp_model(
        bra_param_values=fm_dict | vparams,
        ket_param_values=fm_dict | vparams_minus1_plus2,
    )
    ovrlp_shifter_minus1 = ovrlp_model(
        bra_param_values=fm_dict | vparams, ket_param_values=fm_dict | vparams_minus1
    )

    delta_F = (
        ovrlp_shifter_plus1_plus2
        - ovrlp_shifter_plus1
        - ovrlp_shifter_minus1_plus2
        + ovrlp_shifter_minus1
    )

    fid_hess = (
        (1 / 4)
        * (delta_F / (epsilon**2))
        * (
            torch.matmul(Delta_1, Delta_2.transpose(0, 1))
            + torch.matmul(Delta_2, Delta_1.transpose(0, 1))
        )
    )
    qfi_mat = -2 * fid_hess

    # Calculate the new estimator from the old estimator of qfi_mat
    if k == 0:
        qfi_mat_estimator = qfi_mat
    else:
        qfi_mat_estimator = (1 / (k + 1)) * (k * previous_qfi_estimator + qfi_mat)  # type: ignore

    # Get the positive-semidefinite version of the matrix for the update rule in QNG
    qfi_mat_positive_sd = symsqrt(torch.matmul(qfi_mat_estimator, qfi_mat_estimator))
    qfi_mat_positive_sd = qfi_mat_positive_sd + beta * torch.eye(num_vparams)

    return qfi_mat_estimator, qfi_mat_positive_sd
