from __future__ import annotations

import os

import numpy as np
import torch

from qadence import (
    RX,
    RY,
    BackendName,
    FeatureParameter,
    H,
    Overlap,
    OverlapMethod,
    QuantumCircuit,
    QuantumModel,
    VariationalParameter,
    chain,
    kron,
    tag,
)
from qadence.logger import get_script_logger

logger = get_script_logger("Overlap")

n_qubits = 1
logger.info(f"Running example {os.path.basename(__file__)} with n_qubits = {n_qubits}")

# prepare circuit for bras
param_bra = FeatureParameter("phi")
block_bra = kron(*[RX(qubit, param_bra) for qubit in range(n_qubits)])
fm_bra = tag(block_bra, tag="feature-map-bra")
circuit_bra = QuantumCircuit(n_qubits, fm_bra)

# prepare circuit for kets
param_ket = FeatureParameter("psi")
block_ket = kron(*[RX(qubit, param_ket) for qubit in range(n_qubits)])
fm_ket = tag(block_ket, tag="feature-map-ket")
circuit_ket = QuantumCircuit(n_qubits, fm_ket)

# values for circuits
values_bra = {"phi": torch.Tensor([np.pi, np.pi / 4, np.pi / 3])}
values_ket = {"psi": torch.Tensor([np.pi, np.pi / 2, np.pi / 5])}

backend_name = BackendName.PYQTORCH

# calculate overlap with exact method
ovrlp = Overlap(circuit_bra, circuit_ket, backend=backend_name, method=OverlapMethod.EXACT)
ovrlp_exact = ovrlp(values_bra, values_ket)
print("Exact overlap:\n", ovrlp_exact)

# calculate overlap with shots
ovrlp = Overlap(circuit_bra, circuit_ket, backend=backend_name, method=OverlapMethod.JENSEN_SHANNON)
ovrlp_js = ovrlp(values_bra, values_ket, n_shots=10000)
print("Jensen-Shannon overlap:\n", ovrlp_js)


class LearnHadamard(QuantumModel):
    def __init__(
        self,
        train_circuit: QuantumCircuit,
        target_circuit: QuantumCircuit,
        backend: BackendName = BackendName.PYQTORCH,
    ):
        super().__init__(circuit=train_circuit, backend=backend)

        self.overlap_fn = Overlap(
            train_circuit, target_circuit, backend=backend, method=OverlapMethod.EXACT
        )

    def forward(self):
        return self.overlap_fn()


phi = VariationalParameter("phi")
theta = VariationalParameter("theta")

train_circuit = QuantumCircuit(1, chain(RX(0, phi), RY(0, theta)))
target_circuit = QuantumCircuit(1, H(0))

model = LearnHadamard(train_circuit, target_circuit)


# Applies the Hadamard on the 0 state
print("BEFORE TRAINING:")
print(model.overlap_fn.ket_model.run({}).detach())
print(model.overlap_fn.run({}).detach())
print()

optimizer = torch.optim.Adam(model.parameters(), lr=0.25)
loss_criterion = torch.nn.MSELoss()
n_epochs = 1000
loss_save = []

for i in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_criterion(torch.tensor([[1.0]]), model())
    loss.backward()
    optimizer.step()
    loss_save.append(loss.item())


# Applies the Hadamard on the 0 state
print("AFTER TRAINING:")
print(model.overlap_fn.ket_model.run({}).detach())
print(model.overlap_fn.run({}).detach())
