from __future__ import annotations

from typing import Any, Callable

import networkx as nx
import numpy as np
import qutip
import sympy
import torch.nn as nn
from openfermion import QubitOperator
from pytest import fixture  # type: ignore
from sympy import Expr
from torch import Tensor, tensor

from qadence import QNN, QuantumModel
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import chain, kron, tag, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization
from qadence.operations import CNOT, RX, RY, X, Y, Z
from qadence.parameters import Parameter, TimeParameter
from qadence.register import Register
from qadence.types import PI, BackendName, DiffMode

BASIC_NQUBITS = 4
FM_NQUBITS = 2


@fixture
def batched_noisy_pulser_sim() -> Tensor:
    return tensor([[0.4160, 0.4356, 0.4548, 0.4737]])


@fixture
def noisy_pulser_sim() -> Tensor:
    return tensor([[0.3597]])


@fixture
def noiseless_pulser_sim() -> Tensor:
    return tensor([[0.3961]])


@fixture
def BasicFeatureMap() -> AbstractBlock:
    return feature_map(BASIC_NQUBITS)


@fixture
def BasicAnsatz() -> AbstractBlock:
    return hea(BASIC_NQUBITS, BASIC_NQUBITS)


@fixture
def BasicQuantumCircuit(BasicAnsatz: AbstractBlock) -> QuantumCircuit:
    return QuantumCircuit(BASIC_NQUBITS, BasicAnsatz)


@fixture
def BasicFMQuantumCircuit() -> QuantumCircuit:
    return QuantumCircuit(FM_NQUBITS, feature_map(FM_NQUBITS), hea(FM_NQUBITS, FM_NQUBITS * 4))


@fixture
def BasicObservable() -> AbstractBlock:
    return total_magnetization(BASIC_NQUBITS)


@fixture
def BasicRegister() -> Register:
    n_qubits = 4
    graph = nx.Graph()
    graph.add_nodes_from({i: (i, 0) for i in range(n_qubits)})
    graph.add_edge(0, 1)
    return Register(graph)


@fixture
def BasicExpression() -> Expr:
    return Parameter("x") + Parameter("y", trainable=False) * 2.0212


class BasicNetwork(nn.Module):
    def __init__(self, n_neurons: int = 5) -> None:
        super().__init__()
        network = [
            nn.Linear(1, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 1),
        ]
        self.network = nn.Sequential(*network)
        self.n_neurons = n_neurons

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class BasicNetworkNoInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = nn.Parameter(tensor([1.0]))
        self.scale = nn.Parameter(tensor([1.0]))

    def forward(self) -> Tensor:
        res = self.scale * (self.x - 2.0) ** 2
        return res


@fixture
def parametric_circuit() -> QuantumCircuit:
    nqubits = 4
    x = Parameter("x", trainable=False)

    block1 = RY(0, 3 * x)
    block2 = RX(1, "theta1")
    block3 = RX(2, "theta2")
    block4 = RX(3, "theta3")
    block5 = RY(0, PI)
    block6 = RX(1, PI)
    block7 = CNOT(2, 3)

    comp_block = chain(
        *[
            kron(*[X(0), X(1), Z(2), Z(3)]),
            kron(*[block1, block2, block3, block4]),
            kron(*[block5, block6, block7]),
        ]
    )

    return QuantumCircuit(nqubits, comp_block)


@fixture
def duplicate_expression_circuit() -> QuantumCircuit:
    nqubits = BASIC_NQUBITS
    x = Parameter("x", trainable=False)

    fm = chain(RY(i, 3 * x) for i in range(nqubits))
    expr = Parameter("theta_0") * Parameter("theta_1") + Parameter("theta_2")
    rotblock = chain(RX(i, expr) for i in range(nqubits))

    comp_block = chain(
        *[
            chain(*[X(0), X(1), Z(2), Z(3)]),
            chain(*[fm, rotblock]),
        ]
    )

    return QuantumCircuit(nqubits, comp_block)


@fixture
def cost_operator() -> QubitOperator:
    nqubits = BASIC_NQUBITS
    operator = QubitOperator()

    for qubit in range(nqubits):
        operator += QubitOperator(f"Z{qubit}", coefficient=1.0)

    return operator


@fixture
def Basic() -> nn.Module:
    return BasicNetwork()


@fixture
def BasicNoInput() -> nn.Module:
    return BasicNetworkNoInput()


@fixture
def simple_circuit() -> QuantumCircuit:
    kron_block = kron(X(0), X(1))
    return QuantumCircuit(BASIC_NQUBITS, kron_block)


@fixture
def observable() -> AbstractBlock:
    return kron(X(0), Z(2)) + 1.5 * kron(Y(1), Z(2))


@fixture
def pauli_decomposition(observable: AbstractBlock) -> list:
    return list(unroll_block_with_scaling(observable))


@fixture
def expected_rotated_circuit() -> list[QuantumCircuit]:
    layer = X(0) ^ X(1)
    final_layer1 = chain(layer, RY(0, -PI / 2.0))
    final_layer2 = chain(layer, RX(1, PI / 2.0))
    return [QuantumCircuit(2, final_layer1), QuantumCircuit(2, final_layer2)]


@fixture
def BasicQuantumModel(
    BasicQuantumCircuit: QuantumCircuit, BasicObservable: AbstractBlock
) -> QuantumModel:
    return QuantumModel(
        BasicQuantumCircuit, BasicObservable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )


@fixture
def BasicQNN(BasicFMQuantumCircuit: QuantumCircuit, BasicObservable: AbstractBlock) -> QNN:
    return QNN(
        BasicFMQuantumCircuit,
        total_magnetization(FM_NQUBITS),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )


@fixture
def BasicAdjointQNN(BasicFMQuantumCircuit: QuantumCircuit, BasicObservable: AbstractBlock) -> QNN:
    return QNN(
        BasicFMQuantumCircuit,
        total_magnetization(FM_NQUBITS),
        inputs=["phi"],
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.ADJOINT,
    )


@fixture
def SmallCircuit() -> QuantumCircuit:
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(FM_NQUBITS)])
    tag(fm, "feature_map")

    ansatz = hea(n_qubits=FM_NQUBITS, depth=1)
    tag(ansatz, "ansatz")
    return QuantumCircuit(FM_NQUBITS, fm, ansatz)


@fixture
def omega() -> float:
    return 20.0


@fixture
def feature_param_x() -> float:
    return 2.0


@fixture
def feature_param_y() -> float:
    return 3.5


@fixture
def time_param() -> str:
    return "t"


@fixture
def qadence_generator(omega: float, time_param: str, feature_param_y: float) -> AbstractBlock:
    t = TimeParameter(time_param)
    x = Parameter("x", trainable=False)
    generator_t = omega * (sympy.sin(feature_param_y * t) * X(0) + x * (t**2) * Y(1))
    return generator_t  # type: ignore [no-any-return]


@fixture
def qutip_generator(omega: float, feature_param_x: float, feature_param_y: float) -> Callable:
    def generator_t(t: float, args: Any) -> qutip.Qobj:
        return qutip.Qobj(
            omega
            * (
                np.sin(feature_param_y * t) * qutip.tensor(qutip.sigmax(), qutip.qeye(2))
                + feature_param_x * (t**2) * qutip.tensor(qutip.qeye(2), qutip.sigmay())
            ).full()
        )

    return generator_t
