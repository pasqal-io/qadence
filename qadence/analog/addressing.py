from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import sympy
import torch
from numpy import pi

from qadence.parameters import Parameter, evaluate
from qadence.types import StrEnum

GLOBAL_MAX_AMPLITUDE = 300
GLOBAL_MAX_DETUNING = 2 * pi * 2000
LOCAL_MAX_AMPLITUDE = 3
LOCAL_MAX_DETUNING = 2 * pi * 20


class WeightConstraint(StrEnum):
    """Supported types of constraints for addressing weights."""

    NORMALIZE = "normalize"
    """Normalize weights so that they sum up to 1."""

    RESTRICT = "restrict"
    """Restrict weight values to interval [0, 1]."""


def sigmoid(x: torch.Tensor, a: float, b: float) -> sympy.Expr:
    return 1.0 / (1.0 + sympy.exp(-a * (x + b)))


@dataclass
class AddressingPattern:
    # number of qubits
    n_qubits: int

    # list of weights for fixed amplitude pattern that cannot be changed during the execution
    weights_amp: dict[int, float | torch.Tensor | Parameter]

    # list of weights for fixed detuning pattern that cannot be changed during the execution
    weights_det: dict[int, float | torch.Tensor | Parameter]

    # amplitude can also be chosen as a variational parameter if needed
    amp: float | torch.Tensor | Parameter = LOCAL_MAX_AMPLITUDE

    # detuning can also be chosen as a variational parameter if needed
    det: float | torch.Tensor | Parameter = LOCAL_MAX_DETUNING

    def _validate_weights(
        self,
        weights: dict[int, float | torch.Tensor | Parameter],
    ) -> None:
        for v in weights.values():
            if not isinstance(v, Parameter):
                if not (v >= 0.0 and v <= 1.0):
                    raise ValueError("Addressing pattern weights must sum fall in range [0.0, 1.0]")

    def _constrain_weights(
        self,
        weights: dict[int, float | torch.Tensor | Parameter],
    ) -> dict:
        # augment weight dict if needed
        weights = {
            i: Parameter(0.0)
            if i not in weights
            else (Parameter(weights[i]) if not isinstance(weights[i], Parameter) else weights[i])
            for i in range(self.n_qubits)
        }

        # restrict weights to [0, 1] range
        weights = {
            k: abs(v * (sigmoid(v, 20, 1.0) - sigmoid(v, 20.0, -1.0))) for k, v in weights.items()
        }

        return weights

    def _constrain_max_vals(self) -> None:
        # enforce constraints:
        # 0 <= amp <= GLOBAL_MAX_AMPLITUDE
        # 0 <= abs(det) <= GLOBAL_MAX_DETUNING
        self.amp = abs(
            self.amp
            * (
                sympy.Heaviside(self.amp + GLOBAL_MAX_AMPLITUDE)
                - sympy.Heaviside(self.amp - GLOBAL_MAX_AMPLITUDE)
            )
        )
        self.det = -abs(
            self.det
            * (
                sympy.Heaviside(self.det + GLOBAL_MAX_DETUNING)
                - sympy.Heaviside(self.det - GLOBAL_MAX_DETUNING)
            )
        )

    def _create_local_constraint(self, val: sympy.Expr, weights: dict, max_val: float) -> dict:
        # enforce local constraints:
        # amp * w_amp_i < LOCAL_MAX_AMPLITUDE or
        # abs(det) * w_det_i < LOCAL_MAX_DETUNING
        local_constr = {k: val * v for k, v in weights.items()}
        local_constr = {
            k: sympy.Heaviside(v) - sympy.Heaviside(v - max_val) for k, v in local_constr.items()
        }

        return local_constr

    def _create_global_constraint(
        self, val: sympy.Expr, weights: dict, max_val: float
    ) -> sympy.Expr:
        # enforce global constraints:
        # amp * sum(w_amp_0, w_amp_1, ...) < GLOBAL_MAX_AMPLITUDE or
        # abs(det) * sum(w_det_0, w_det_1, ...) < GLOBAL_MAX_DETUNING
        weighted_vals_global = val * sum([v for v in weights.values()])
        weighted_vals_global = sympy.Heaviside(weighted_vals_global) - sympy.Heaviside(
            weighted_vals_global - max_val
        )

        return weighted_vals_global

    def __post_init__(self) -> None:
        # validate amplitude/detuning weights
        self._validate_weights(self.weights_amp)
        self._validate_weights(self.weights_det)

        # validate maximum global amplitude/detuning values
        if not isinstance(self.amp, Parameter):
            if self.amp > GLOBAL_MAX_AMPLITUDE:
                warn("Maximum absolute value of amplitude is exceeded")
        if not isinstance(self.det, Parameter):
            if abs(self.det) > GLOBAL_MAX_DETUNING:
                warn("Maximum absolute value of detuning is exceeded")

        # constrain amplitude/detuning parameterized weights to [0.0, 1.0] interval
        self.weights_amp = self._constrain_weights(self.weights_amp)
        self.weights_det = self._constrain_weights(self.weights_det)

        # constrain max global amplitude and detuning to strict interval
        self._constrain_max_vals()

        # create additional local and global constraints for amplitude/detuning masks
        self.local_constr_amp = self._create_local_constraint(
            self.amp, self.weights_amp, LOCAL_MAX_AMPLITUDE
        )
        self.local_constr_det = self._create_local_constraint(
            -self.det, self.weights_det, LOCAL_MAX_DETUNING
        )
        self.global_constr_amp = self._create_global_constraint(
            self.amp, self.weights_amp, GLOBAL_MAX_AMPLITUDE
        )
        self.global_constr_det = self._create_global_constraint(
            -self.det, self.weights_det, GLOBAL_MAX_DETUNING
        )

        # validate number of qubits in mask
        if max(list(self.weights_amp.keys())) >= self.n_qubits:
            raise ValueError("Amplitude weight specified for non-existing qubit")
        if max(list(self.weights_det.keys())) >= self.n_qubits:
            raise ValueError("Detuning weight specified for non-existing qubit")

    def evaluate(self, weights: dict, values: dict) -> dict:
        # evaluate weight expressions with actual values
        return {k: evaluate(v, values, as_torch=True).flatten() for k, v in weights.items()}  # type: ignore [union-attr]
