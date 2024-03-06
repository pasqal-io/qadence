from __future__ import annotations

import os

import numpy as np
import sympy
from braket.circuits import Noise
from braket.devices import LocalSimulator

from qadence import (
    CNOT,
    RX,
    RZ,
    Parameter,
    QuantumCircuit,
    backend_factory,
    chain,
    total_magnetization,
)
from qadence.logger import get_script_logger
from qadence.types import BackendName, DiffMode

logger = get_script_logger("braket_digital")

# def circuit(n_qubits):
#     # make feature map with input parameters
#     fm = chain(RX(0, 3 * x), RZ(1, z), CNOT(0, 1))
#     fm = set_trainable(fm, value=False)

#     # make trainable ansatz
#     ansatz = []
#     for i, q in enumerate(range(n_qubits)):
#         ansatz.append(
#             chain(
#                 RX(q, f"theta_0{i}"),
#                 RZ(q, f"theta_1{i}"),
#                 RX(q, f"theta_2{i}"),
#             )
#         )
#     ansatz = kron(ansatz[0], ansatz[1])
#     ansatz *= CNOT(0, 1)

#     block = chain(fm, ansatz)
#     circ = QuantumCircuit(n_qubits=n_qubits, blocks=block)
#     return circ


def circuit(n_qubits):
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=False)

    fm = chain(RX(0, 3 * x), RZ(1, sympy.exp(y)), RX(0, np.pi / 2), RZ(1, "theta"))
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)
    return circ


if __name__ == "__main__":
    import torch

    torch.manual_seed(10)

    n_qubits = 2
    logger.info(f"Running example {os.path.basename(__file__)} with n_qubits = {n_qubits}")
    circ = circuit(n_qubits)

    observable = total_magnetization(n_qubits=n_qubits)
    braket_backend = backend_factory(backend=BackendName.BRAKET, diff_mode=DiffMode.GPSR)

    batch_size = 1
    values = {
        "x": torch.rand(batch_size, requires_grad=True),
        "y": torch.rand(batch_size, requires_grad=True),
    }

    # you can unpack the conversion result or just use conv.circuit, etc.
    conv = braket_backend.convert(circ, observable)
    (braket_circuit, braket_observable, embed, params) = conv

    wf = braket_backend.run(braket_circuit, embed(params, values))
    expval = braket_backend.expectation(braket_circuit, braket_observable, embed(params, values))
    dexpval_braket = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), retain_graph=True
    )[0]

    pyq_backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    conv = pyq_backend.convert(circ, observable)

    wf = pyq_backend.run(conv.circuit, conv.embedding_fn(conv.params, values))
    expval = pyq_backend.expectation(
        conv.circuit, conv.observable, conv.embedding_fn(conv.params, values)
    )
    dexpval_pyq = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), retain_graph=True
    )[0]

    assert torch.allclose(dexpval_braket, dexpval_pyq, atol=1e-4, rtol=1e-4)

    # sample
    samples = braket_backend.sample(braket_circuit, embed(params, values), n_shots=1000)
    print(f"Samples: {samples}")

    ## use the backend with the low-level interface

    # retrieve parameters
    params = embed(params, values)

    # use the native representation directly
    native = braket_circuit.native

    # define a noise channel
    noise = Noise.Depolarizing(probability=0.1)

    # add noise to every gate in the circuit
    native.apply_gate_noise(noise)

    # use density matrix simulator for noise simulations
    device = LocalSimulator("braket_dm")
    native = braket_backend.assign_parameters(braket_circuit, params)
    result = device.run(native, shots=1000).result().measurement_counts
    print("With noise")
    print(result)
    print("Noisy circuit")

    # obtain the braket diagram
    print(native.diagram())
