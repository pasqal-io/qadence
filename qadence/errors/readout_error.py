

# uniform incorporation of noise (make sure counting is correct as discussed with Roland)
# add suggestions on shot noise (gaussian and poisson -
# what is the best way to incorporate these in comparison to uniform)?
# add to backend
# use measurements architecture
# check circ, qiskit, tket, pennylane noise models
# add proper citations where needed
# remove from quantum_model as this is not general enough!
# tests!!


from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from itertools import chain

import numpy as np
import torch
from torch.distributions import normal, poisson, uniform

from qadence.logger import get_logger

logger = get_logger(__name__)


class WhiteNoise(Enum):
    """White noise distributions"""

    UNIFORM = staticmethod(uniform.Uniform(low=0.0, high=1.0))
    "uniform white noise"

    GAUSSIAN = staticmethod(normal.Normal(loc=0.0, scale=1.0))
    "gaussian white noise"

    POISSON = staticmethod(poisson.Poisson(rate=0.1))
    "poisson white noise"


class ErrorModel(ABC):  # the abstract parent class of our error models
    # should take into account multiple possible future error types/models

    name = "ErrorModel"

    @abstractmethod
    def bit_flip(self, qbit: int) -> int:
        pass

    @abstractmethod
    def phase_flip(self, qbit: int) -> int:
        pass


class ReadoutError(ErrorModel):
    """ """

    name = "ReadoutError"

    def __init__(
        self, n_qubits: int, err_probabilities: torch.Tensor | None = None, fidelity: float = 0.1
    ) -> None:
        self.n_qubits = n_qubits
        self.fidelity = fidelity

        if err_probabilities is not None:
            self.err_probabilities = err_probabilities
            assert (
                self.err_probabilities.shape[0] == self.err_probabilities.shape[1]
            ), "The error probabilities matrix needs to be square."
            assert self.err_probabilities.shape == (2**self.n_qubits, 2**self.n_qubits), (
                "The error probabilities matrix needs to be " "2 ^ n_qubits x 2 ^ n_qubits."
            )
        else:  # assume uniform readout error
            self.err_probabilities = np.ones([2**self.n_qubits, 2**self.n_qubits]) * fidelity

    def bitstring_to_array(self, bitstring: str) -> np.array:
        return np.array([int(i) for i in bitstring])

    def array_to_bitstring(self, bitstring: np.array) -> str:
        return "".join(([str(i) for i in bitstring]))

    def bit_flip(self, qbit: int) -> int:
        return 1 if qbit == 0 else 0

    def phase_flip(self, qbit: int) -> int:
        return NotImplemented

    def bs_corruption(self, bitstring: str, shots: int) -> list:
        # the noise_matrix should be available to the user if they want to do error correction
        self.noise_matrix = self.noise_distribution.sample(  # type: ignore[attr-defined]
            [shots, self.n_qubits]
        )

        # simplest approach - en event occurs if its probability is higher than expected
        # by random chance
        err_idx = torch.nonzero((self.noise_matrix < self.fidelity), as_tuple=True)[1]
        all_bitstrings = [bitstring] * (shots - len(err_idx))  # add the majority correct bitstrings

        def func_distort(idx: int) -> str:
            bitstring_copy = self.bitstring_to_array(bitstring)
            bitstring_copy[idx] = self.bit_flip(bitstring_copy[idx])
            return self.array_to_bitstring(bitstring_copy)

        all_bitstrings.extend([func_distort(idx) for idx in err_idx])
        return all_bitstrings

    def __call__(
        self,
        bitstring: str | dict,
        shots: int = 1000,
        seed: int | None = None,
        noise_distribution: Enum = WhiteNoise.UNIFORM,
    ) -> list[Counter]:
        self.noise_distribution = noise_distribution

        # for ensuring reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(bitstring, dict):
            return [
                Counter(
                    list(
                        chain(
                            *[
                                self.bs_corruption(bitstring, shots)
                                for bitstring, shots in bitstring.items()
                            ]
                        )
                    )
                )
            ]
        elif isinstance(bitstring, str):
            return [Counter(self.bs_corruption(bitstring, shots))]
        else:
            raise NotImplementedError

        # noise distributions - uniform white noise, gaussian white noise, poisson white noise
        # proper noise model - additive and multiplicative noise models


############## there will be some time for these to properly exist...


class GateError(ErrorModel):
    name = "GateError"

    def __init__(self) -> None:
        pass


class DecoherenceError(ErrorModel):
    name = "DecoherenceError"

    def __init__(self) -> None:
        pass
