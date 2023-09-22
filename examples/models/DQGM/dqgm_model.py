from __future__ import annotations

from typing import List, Tuple, Union

import torch

from qadence import QuantumCircuit, QuantumModel, chain, exp_fourier_feature_map, qft, tag
from qadence.backend import BackendName
from qadence.backends.pytorch_wrapper import DiffMode
from qadence.blocks import AbstractBlock
from qadence.transpile import set_trainable


class DQGM(QuantumModel):
    def __init__(
        self,
        ansatz: AbstractBlock,
        n_features: int = 1,
        n_qubits_per_feature: Union[int, List[int]] = 1,
        backend: str = BackendName.PYQTORCH,
        diff_mode: str = DiffMode.AD,
        feature_range: Tuple[float] = (0.0, 1.0),
    ):
        """
        This class is a specific type of quantum model known as "DQGM",
        and is inspired from the paper "Protocols for Trainable
        and Differentiable Quantum Generative Modelling"

        Args:
            ansatz: variational unitary to use
            n_features: number of features in the model
            n_qubits_per_feature: number of qubits to use per feature
            backend: which numerical backend to use for circuit execution
            diff_mode: type of differentiation to use for the quantum circuits
            feature_range: range of feature, tuple of 2 floats
        """
        if n_features == 1:
            fname = "x"
            fm = exp_fourier_feature_map(
                n_qubits=n_qubits_per_feature, param=fname, feature_range=feature_range
            )
            self.feat_params = [fname]
        else:
            # TODO here we can vertically stack (kron) multiple fm's in the future
            raise NotImplementedError("More than 1 feature not yet supported!")
        self.feature_range = feature_range
        # we can have either a larger FM or larger ansatz, but for the overall circuit take the max
        self.n_qubits_total = max(n_qubits_per_feature * n_features, ansatz.n_qubits)
        # construct the circuit and attach
        self._ansatz = ansatz
        self.circuit = QuantumCircuit(self.n_qubits_total, fm, ansatz)
        # init some caching attrs
        self._sampling_model = None
        self._indices = None
        self._backend_name = backend
        super().__init__(circuit=self.circuit, backend=backend, diff_mode=diff_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = {p: x[:, n] for n, p in enumerate(self.feat_params)}
        ket = self.run(vals)
        # The QNN part of DQGM assumes a cost function |<0|psi>|^2,
        # so the first element of the wavefunction abs squared
        return torch.abs(ket[:, 0]) ** 2

    def init_sampling_model(self, n_qubits: int = None):
        """
        :param n_qubits: num of qubits to use for sampling. Can be greater than in training stage.
        """
        ansatz_dagger = self._ansatz.dagger()

        n = ansatz_dagger.n_qubits if n_qubits is None else n_qubits
        set_trainable(ansatz_dagger, False)

        tag(ansatz_dagger, "ansatz transpose")

        circuit = QuantumCircuit(
            n, chain(ansatz_dagger, qft(n, reverse_in=True, inverse=True, swaps_out=True))
        )

        self._sampling_model = QuantumModel(
            circuit, backend=self._backend_name, diff_mode=self.backend.diff_mode
        )

    def probabilities(self) -> torch.Tensor:
        """
        This method computes the probabilities of sampling bitstrings at the output
        of the generative model part of DQGM

        :return: tensor with probabilities of each bitstring
        """
        if self._sampling_model is None:
            raise UserWarning("Please init first the sampling model with .init_sampling_model()")
        ket = self._sampling_model.run(self.vparams)
        return torch.abs(ket) ** 2

    def sample(self, n_shots: int = 1000) -> torch.Tensor:
        if self._sampling_model is None:
            raise UserWarning("Please init first the sampling model with .init_sampling_model()")
        return self._sampling_model.sample(self.vparams, n_shots=n_shots)[0]
