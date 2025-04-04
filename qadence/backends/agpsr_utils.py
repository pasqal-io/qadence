from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor


def variance(shifts: Tensor, spectral_gaps: Tensor) -> Tensor:
    """Calculate the exact variance of deirivative estimation using aGPSR.

    Args:
        shifts (Tensor): shifts to apply for each spectral gap
        spectral_gaps (Tensor): tensor containing spectral gap values

    Returns:
        Tensor: variance tensor
    """

    # calculate inverse of M (see: https://arxiv.org/pdf/2108.01218.pdf on p. 4 for definitions)
    M = 4 * torch.sin(torch.outer(torch.as_tensor(shifts), spectral_gaps) / 2)
    try:
        # calculate the variance of derivative estimation by solving a linear equation system
        a = torch.linalg.solve(M, spectral_gaps.reshape(-1, 1))
        var = 2 * torch.matmul(a.T, a)
    except RuntimeError:  # matrix M is singulkar
        # fallback method of variance calculation using inverse matrix
        M_inv = torch.linalg.pinv(M)
        a = torch.matmul(spectral_gaps.reshape(1, -1), M_inv)
        var = 2 * torch.matmul(a, a.T)

    return var


@lru_cache
def calculate_optimal_shifts(
    n_eqs: int,
    spectral_gaps: Tensor,
    lb: float,
    ub: float,
) -> Tensor:
    """Calculates optimal shift values for GPSR algorithm.

    Args:
        n_eqs (int): number of equations in linear equation system for derivative estimation
        spectral_gaps (Tensor): tensor containing spectral gap values
        lb (float): lower bound of optimal shift value search interval
        ub (float): upper bound of optimal shift value search interval

    Returns:
        Tensor: optimal shift values
    """
    if not (lb and ub):
        raise ValueError("Both lower and upper bounds of optimization interval must be given.")

    constraints = []

    # specify solution bound constraints
    for i in range(n_eqs):

        def fn_lb(x, i=i):  # type: ignore [no-untyped-def]
            return x[i] - lb

        def fn_ub(x, i=i):  # type: ignore [no-untyped-def]
            return ub - x[i]

        constraints.append({"type": "ineq", "fun": fn_lb})
        constraints.append({"type": "ineq", "fun": fn_ub})

    # specify constraints for solutions to be unique
    for i in range(n_eqs - 1, 0, -1):
        for j in range(i):

            def fn(x, i=i, j=j):  # type: ignore [no-untyped-def]
                return np.abs(x[i] - x[j]) - 0.02

            constraints.append({"type": "ineq", "fun": fn})

    init_guess = torch.linspace(lb, ub, n_eqs)

    def minimize_variance(
        var_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        res = minimize(
            fun=var_fn,
            x0=init_guess,
            args=(spectral_gaps,),
            method="COBYLA",
            constraints=constraints,
        )
        return torch.as_tensor(res.x)

    return minimize_variance(variance)
