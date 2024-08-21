from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor

from qadence.types import PI
from qadence.utils import _round_complex


def general_psr(spectrum: Tensor, n_eqs: int | None = None, shift_prefac: float = 0.5) -> Callable:
    """Define whether single_gap_psr or multi_gap_psr is used.

    Args:
        spectrum (Tensor): Spectrum of the operation we apply PSR onto.
        n_eqs (int | None, optional): Number of equations. Defaults to None.
            If provided, we keep the n_eqs higher spectral gaps.
        shift_prefac (float, optional): Shift prefactor. Defaults to 0.5.

    Returns:
        Callable: single_gap_psr or multi_gap_psr function for
            concerned operation.
    """
    diffs = _round_complex(spectrum - spectrum.reshape(-1, 1))
    sorted_unique_spectral_gaps = torch.unique(torch.abs(torch.tril(diffs)))

    # We have to filter out zeros
    sorted_unique_spectral_gaps = sorted_unique_spectral_gaps[sorted_unique_spectral_gaps > 0]
    n_eqs = (
        len(sorted_unique_spectral_gaps)
        if n_eqs is None
        else min(n_eqs, len(sorted_unique_spectral_gaps))
    )
    sorted_unique_spectral_gaps = torch.tensor(list(sorted_unique_spectral_gaps)[:n_eqs])

    if n_eqs == 1:
        return single_gap_psr
    else:
        return partial(
            multi_gap_psr,
            spectral_gaps=sorted_unique_spectral_gaps,
            shift_prefac=shift_prefac,
        )


def single_gap_psr(
    expectation_fn: Callable[[dict[str, Tensor]], Tensor],
    param_dict: dict[str, Tensor],
    param_name: str,
    spectral_gap: Tensor = torch.tensor([2], dtype=torch.get_default_dtype()),
    shift: Tensor = torch.tensor([PI / 2], dtype=torch.get_default_dtype()),
) -> Tensor:
    """Implements single qubit PSR rule.

    Args:
        expectation_fn (Callable[[dict[str, Tensor]], Tensor]): backend-dependent function
        to calculate expectation value

        param_dict (dict[str, Tensor]): dict storing parameters of parameterized blocks
        param_name (str): name of parameter with respect to that differentiation is performed

    Returns:
        Tensor: tensor containing derivative values
    """
    device = torch.device("cpu")
    try:
        device = [v.device for v in param_dict.values()][0]
    except Exception:
        pass
    spectral_gap = spectral_gap.to(device=device)
    shift = shift.to(device=device)
    # + pi/2 shift
    shifted_params = param_dict.copy()
    shifted_params[param_name] = shifted_params[param_name] + shift
    f_plus = expectation_fn(shifted_params)

    # - pi/2 shift
    shifted_params = param_dict.copy()
    shifted_params[param_name] = shifted_params[param_name] - shift
    f_min = expectation_fn(shifted_params)

    return spectral_gap * (f_plus - f_min) / (4 * torch.sin(spectral_gap * shift / 2))


def multi_gap_psr(
    expectation_fn: Callable[[dict[str, Tensor]], Tensor],
    param_dict: dict[str, Tensor],
    param_name: str,
    spectral_gaps: Tensor,
    shift_prefac: float = 0.5,
) -> Tensor:
    """Implements multi-gap multi-qubit GPSR rule.

    Args:
        expectation_fn (Callable[[dict[str, Tensor]], Tensor]): backend-dependent function
        to calculate expectation value

        param_dict (dict[str, Tensor]): dict storing parameters values of parameterized blocks
        param_name (str): name of parameter with respect to that differentiation is performed
        spectral_gaps (Tensor): tensor containing spectral gap values
        shift_prefac (float): prefactor governing the magnitude of parameter shift values -
        select smaller value if spectral gaps are large

    Returns:
        Tensor: tensor containing derivative values
    """
    n_eqs = len(spectral_gaps)
    batch_size = max(t.size(0) for t in param_dict.values())

    # get shift values
    shifts = shift_prefac * torch.linspace(PI / 2 - PI / 5, PI / 2 + PI / 5, n_eqs)
    device = torch.device("cpu")
    try:
        device = [v.device for v in param_dict.values()][0]
    except Exception:
        pass
    spectral_gaps = spectral_gaps.to(device=device)
    shifts = shifts.to(device=device)
    # calculate F vector and M matrix
    # (see: https://arxiv.org/pdf/2108.01218.pdf on p. 4 for definitions)
    F = []
    M = torch.empty((n_eqs, n_eqs)).to(device=device)
    n_obs = 1
    for i in range(n_eqs):
        # + shift
        shifted_params = param_dict.copy()
        shifted_params[param_name] = shifted_params[param_name] + shifts[i]
        f_plus = expectation_fn(shifted_params)

        # - shift
        shifted_params = param_dict.copy()
        shifted_params[param_name] = shifted_params[param_name] - shifts[i]
        f_minus = expectation_fn(shifted_params)

        F.append((f_plus - f_minus))

        # calculate M matrix
        for j in range(n_eqs):
            M[i, j] = 4 * torch.sin(shifts[i] * spectral_gaps[j] / 2)

    # get number of observables from expectation value tensor
    if f_plus.numel() > 1:
        batch_size = F[0].shape[0]
        n_obs = F[0].shape[1]

    # reshape F vector
    F = torch.stack(F).reshape(n_eqs, -1)

    # calculate R vector
    R = torch.linalg.solve(M, F)

    # calculate df/dx
    dfdx = torch.sum(spectral_gaps[:, None] * R, dim=0).reshape(batch_size, n_obs)

    return dfdx
