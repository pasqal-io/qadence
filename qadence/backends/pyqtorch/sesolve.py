from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def krylov_exp(
    t: float,
    state: Tensor,
    ham: Tensor,
) -> Tensor:
    max_krylov = 80
    exp_tolerance = 1e-10
    norm_tolerance = 1e-10

    def exponentiate() -> tuple[torch.Tensor, bool]:
        # approximate next iteration by modifying T, and unmodifying
        T[i - 1, i] = 0
        T[i + 1, i] = 1
        exp = torch.linalg.matrix_exp(-1j * t * T[: i + 2, : i + 2].clone())
        T[i - 1, i] = T[i, i - 1]
        T[i + 1, i] = 0

        e1 = abs(exp[i, 0])
        e2 = abs(exp[i + 1, 0]) * n
        if e1 > 10 * e2:
            error = e2
        elif e2 > e1:
            error = e1
        else:
            error = (e1 * e2) / (e1 - e2)

        converged = error < exp_tolerance
        return exp[:, 0], converged

    lanczos_vectors = [state]
    T = torch.zeros(max_krylov + 1, max_krylov + 1, dtype=state.dtype)

    # step 0 of the loop
    v = torch.matmul(ham, state)
    a = torch.matmul(v.conj().T, state)
    n = torch.linalg.vector_norm(v)
    T[0, 0] = a
    v = v - a * state

    for i in range(1, max_krylov):
        # this block should not be executed in step 0
        b = torch.linalg.vector_norm(v)
        if b < norm_tolerance:
            exp = torch.linalg.matrix_exp(-1j * t * T[:i, :i])
            weights = exp[:, 0]
            converged = True
            break

        T[i, i - 1] = b
        T[i - 1, i] = b
        state = v / b
        lanczos_vectors.append(state)
        weights, converged = exponentiate()
        if converged:
            break

        v = torch.matmul(ham, state)
        a = torch.matmul(v.conj().T, state)
        n = torch.linalg.vector_norm(v)
        T[i, i] = a
        v = v - a * lanczos_vectors[i] - b * lanczos_vectors[i - 1]

    if not converged:
        raise RecursionError(
            "exponentiation algorithm did not converge to precision in allotted number of steps."
        )

    result = lanczos_vectors[0] * weights[0]
    for i in range(1, len(lanczos_vectors)):
        result += lanczos_vectors[i] * weights[i]

    return result


def sesolve_krylov(H: Tensor | Callable, psi0: Tensor, tsave: list | Tensor) -> Tensor:
    t_tot = 0.0
    psi = [psi0]
    psi_init = psi0
    for i, t in enumerate(tsave):
        if isinstance(H, Callable):  # type: ignore [arg-type]
            if i < len(tsave) - 1:
                ham = H(t)
        else:
            ham = H
        dt = t - t_tot

        if dt > 0.0:
            psi_t = krylov_exp(dt, psi_init, ham)
            psi.append(psi_t)
            t_tot += dt
            psi_init = psi_t
        else:
            continue

    return psi
