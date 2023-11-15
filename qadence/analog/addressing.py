from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import sympy
import torch
from numpy import pi

from qadence.parameters import Parameter, evaluate
from qadence.types import StrEnum

DEFAULT_MAX_AMPLITUDE = 2 * pi * 3
DEFAULT_MAX_DETUNING = 2 * pi * 20


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

    # maximum amplitude can also be chosen as a variational parameter if needed
    max_amp: float | torch.Tensor | Parameter = DEFAULT_MAX_AMPLITUDE

    # maximum detuning can also be chosen as a variational parameter if needed
    max_det: float | torch.Tensor | Parameter = DEFAULT_MAX_DETUNING

    # weight constraint
    weight_constraint: WeightConstraint = WeightConstraint.NORMALIZE

    def _normalize_weights(self) -> None:
        self.weights_amp = {
            k: Parameter(v) if not isinstance(v, Parameter) else abs(v)
            for k, v in self.weights_amp.items()
        }
        sum_weights_amp = sum(list(self.weights_amp.values()))
        self.weights_amp = {k: v / sum_weights_amp for k, v in self.weights_amp.items()}

        self.weights_det = {
            k: Parameter(v) if not isinstance(v, Parameter) else abs(v)
            for k, v in self.weights_det.items()
        }
        sum_weights_det = sum(list(self.weights_det.values()))
        self.weights_det = {k: v / sum_weights_det for k, v in self.weights_det.items()}

    def _restrict_weights(self) -> None:
        self.weights_amp = {
            k: v * (sigmoid(v, 20, 0.0) - sigmoid(v, 20.0, -1.0))
            for k, v in self.weights_amp.items()
        }
        self.weights_det = {
            k: v * (sigmoid(v, 20.0, 0.0) - sigmoid(v, 20.0, -1.0))
            for k, v in self.weights_det.items()
        }

    def _restrict_max_vals(self) -> None:
        self.max_amp = self.max_amp * (
            sympy.Heaviside(self.max_amp) - sympy.Heaviside(self.max_amp - DEFAULT_MAX_AMPLITUDE)
        )
        self.max_det = self.max_det * (
            sympy.Heaviside(self.max_det) - sympy.Heaviside(self.max_det - DEFAULT_MAX_DETUNING)
        )

    def __post_init__(self) -> None:
        # validate weights
        if all([not isinstance(v, Parameter) for v in self.weights_amp.values()]):
            if not torch.isclose(
                torch.tensor(list(self.weights_amp.values())).sum(),
                torch.tensor(1.0),
                atol=1e-3,
            ):
                raise ValueError("Amplitude addressing pattern weights must sum to 1.0")
        if all([not isinstance(v, Parameter) for v in self.weights_det.values()]):
            if not torch.isclose(
                torch.tensor(list(self.weights_det.values())).sum(),
                torch.tensor(1.0),
                atol=1e-3,
            ):
                raise ValueError("Detuning addressing pattern weights must sum to 1.0")

        # validate detuning value
        if not isinstance(self.max_amp, Parameter):
            if self.max_amp > DEFAULT_MAX_AMPLITUDE:
                warn("Maximum absolute value of amplitude is exceeded")
        if not isinstance(self.max_det, Parameter):
            if self.max_det > DEFAULT_MAX_DETUNING:
                warn("Maximum absolute value of detuning is exceeded")

        # augment weight dicts if needed
        self.weights_amp = {
            i: Parameter(0.0) if i not in self.weights_amp else self.weights_amp[i]
            for i in range(self.n_qubits)
        }
        self.weights_det = {
            i: Parameter(0.0) if i not in self.weights_det else self.weights_det[i]
            for i in range(self.n_qubits)
        }

        # apply weight constraint
        if self.weight_constraint == WeightConstraint.NORMALIZE:
            self._normalize_weights()
        elif self.weight_constraint == WeightConstraint.RESTRICT:
            self._restrict_weights()
        else:
            raise ValueError("Weight constraint type not found.")

        # restrict max amplitude and detuning to strict interval
        self._restrict_max_vals()

        # validate number of qubits in mask
        if max(list(self.weights_amp.keys())) >= self.n_qubits:
            raise ValueError("Amplitude weight specified for non-existing qubit")
        if max(list(self.weights_det.keys())) >= self.n_qubits:
            raise ValueError("Detuning weight specified for non-existing qubit")

    def evaluate(self, weights: dict, values: dict) -> dict:
        # evaluate weight expressions with actual values
        return {k: evaluate(v, values, as_torch=True).flatten() for k, v in weights.items()}  # type: ignore [union-attr]
