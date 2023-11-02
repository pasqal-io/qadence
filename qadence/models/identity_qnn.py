from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
from torch import Tensor

from qadence.backend import BackendConfiguration, BackendName
from qadence.backends.pytorch_wrapper import DiffMode
from qadence.blocks import AbstractBlock, chain, kron, tag
from qadence.blocks.utils import parameters
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.models.quantum_model import QuantumModel
from qadence.operations import CNOT, RX, RY


def identity_block(
    n_qubits: int,
    layer: int = 0,
    ops: list[type[AbstractBlock]] = [RX, RY],
    periodic: bool = False,
) -> AbstractBlock:
    """
    Identity block for barren plateau mitigation.
    The initial configuration of this block is equal to an identity unitary
    but can be trained in the same fashion as other ansatzes reaching same level
    of expressivity.
    """

    left_rotations = [
        kron(
            gate(n, "alpha" + f"_{layer}{n + n_qubits*i}")  # type: ignore [arg-type]
            for n in range(n_qubits)
        )
        for i, gate in enumerate(ops)
    ]

    if not periodic:
        left_cnots = [chain(CNOT(n, n + 1) for n in range(n_qubits - 1))]
    else:
        left_cnots = [chain(CNOT(n, (n + 1) % n_qubits) for n in range(n_qubits))]

    centre_rotations = [kron(RX(n, "theta" + f"_{layer}{n}") for n in range(n_qubits))]

    right_cnots = reversed(*left_cnots)

    right_rotations = [
        kron(
            gate(
                n,  # type: ignore [arg-type]
                "beta" + f"_{layer}{n + n_qubits*(len(ops)-i-1)}",  # type: ignore [arg-type]
            )
            for n in range(n_qubits)
        )
        for i, gate in enumerate(reversed(ops))
    ]

    krons = [
        *left_rotations,
        *left_cnots,
        *centre_rotations,
        *right_cnots,
        *right_rotations,
    ]

    return tag(chain(*krons), tag=f"BPMA-{layer}")


class IdentityQNN(QuantumModel):
    def __init__(
        self,
        feature_map: AbstractBlock,
        depth: int = 1,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
        transform: Callable[[Tensor], Tensor] = None,
        backend: BackendName = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        protocol: Measurements | None = None,
        configuration: BackendConfiguration | dict | None = None,
    ):
        self.circ = self._create_circuit(feature_map, depth)

        super().__init__(
            circuit=self.circ,
            observable=observable,
            backend=backend,
            diff_mode=diff_mode,
            protocol=protocol,
            configuration=configuration,
        )

        self._initialize_circuit(depth)
        self.transform = transform if transform else lambda x: x

    def _create_circuit(self, feature_map: AbstractBlock, depth: int) -> QuantumCircuit:
        n_qubits = feature_map.n_qubits
        identity_ansatz = []
        for i in range(depth):
            identity_ansatz.append(identity_block(n_qubits, layer=i))

        block = chain(feature_map, *identity_ansatz)
        return QuantumCircuit(n_qubits, block)

    def _initialize_circuit(self, depth: int) -> None:
        def set_block_parameters(
            block: AbstractBlock,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            params = parameters(block)
            params_dict = {
                k: list(v) for k, v in itertools.groupby(params, key=lambda x: str(x).split("_")[0])
            }
            alpha = np.random.uniform(0, np.pi, size=len(params_dict["alpha"]))
            theta = np.zeros(len(params_dict["theta"]))
            beta = -alpha
            return alpha, beta, theta

        alphas = []
        betas = []
        thetas = []
        for d in range(depth):
            block = self.circ.get_blocks_by_tag(f"BPMA-{d}")[0]
            alpha, beta, theta = set_block_parameters(block)
            alphas += list(alpha)
            betas += list(beta)
            thetas += list(theta)

        initialization_values = alphas + betas + thetas
        self.reset_vparams(initialization_values)

    def forward(
        self,
        values: dict[str, Tensor] | Tensor,
        state: Tensor | None = None,
        protocol: Measurements | None = None,
    ) -> Tensor:
        if not isinstance(values, dict):
            values = self._format_to_dict(values)
        if protocol is None:
            protocol = self._protocol

        return self.transform(self.expectation(values=values, state=state, protocol=protocol))

    def _format_to_dict(self, values: Tensor) -> dict[str, Tensor]:
        if len(values.size()) == 1:
            values = values.reshape(-1, 1)

        assert len(values.size()) == 2, "The shape of the input tensor is wrong!"
        assert (
            values.size()[1] == self.in_features
        ), "The number of dimensions in the tensor is wrong"

        names = [p.name for p in self.inputs]
        res = {}
        for i, name in enumerate(names):
            res[name] = values[:, i]
        return res
