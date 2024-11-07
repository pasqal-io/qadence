In `qadence`, the `QuantumModel` is the central class point for executing
`QuantumCircuit`s.  The idea of a `QuantumModel` is to decouple the backend
execution from the management of circuit parameters and desired quantum
computation output.

In the following, we create a custom `QuantumModel` instance which introduces
some additional optimizable parameters:
*  an adjustable scaling factor in front of the observable to measured
*  adjustable scale and shift factors to be applied to the model output before returning the result

This can be easily done using PyTorch flexible model definition, and it will
automatically work with the rest of `qadence` infrastructure.


```python exec="on" source="material-block" session="custom-model"
import torch
from qadence import QuantumModel, QuantumCircuit


class CustomQuantumModel(QuantumModel):

    def __init__(self, circuit: QuantumCircuit, observable, backend="pyqtorch", diff_mode="ad"):
        super().__init__(circuit, observable=observable, backend=backend, diff_mode=diff_mode)

        self.n_qubits = circuit.n_qubits

        # define some additional parameters which will scale and shift (variationally) the
        # output of the QuantumModel
        # you can use all torch machinery for building those
        self.scale_out = torch.nn.Parameter(torch.ones(1))
        self.shift_out = torch.nn.Parameter(torch.ones(1))

    # override the forward pass of the model
    # the forward pass is the output of your QuantumModel and in this case
    # it's the (scaled) expectation value of the total magnetization with
    # a variable coefficient in front
    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:

        # scale the observable
        res = self.expectation(values)

        # scale and shift the result before returning
        return self.shift_out + res * self.scale_out
```

The custom model can be used like any other `QuantumModel`:
```python exec="on" source="material-block" result="json" session="custom-model"
from qadence import Parameter, RX, CNOT, QuantumCircuit
from qadence import chain, kron, hamiltonian_factory, Z
from sympy import acos

def quantum_circuit(n_qubits):

    x = Parameter("x", trainable=False)
    fm = kron(RX(i, acos(x) * (i+1)) for i in range(n_qubits))

    ansatz = kron(RX(i, f"theta{i}") for i in range(n_qubits))
    ansatz = chain(ansatz, CNOT(0, n_qubits-1))

    block = chain(fm, ansatz)
    block.tag = "circuit"
    return QuantumCircuit(n_qubits, block)

n_qubits = 4
batch_size = 10
circuit = quantum_circuit(n_qubits)
observable = hamiltonian_factory(n_qubits, detuning=Z)  # Total magnetization

model = CustomQuantumModel(circuit, observable, backend="pyqtorch")

values = {"x": torch.rand(batch_size)}
res = model(values)
print("Model output: ", res)
assert len(res) == batch_size
```


## Quantum model with wavefunction overlaps

`QuantumModel`'s can also use different quantum operations in their forward
pass, such as wavefunction overlaps described [here](../../content/overlap.md). Beware that the resulting overlap tensor
has to be differentiable to apply gradient-based optimization. This is only applicable to the `"EXACT"` overlap method.

Here we show how to use overlap calculation when fitting a parameterized quantum circuit to act as a standard Hadamard gate.

```python exec="on" source="material-block" result="json" session="custom-model"
from qadence import RY, RX, H, Overlap

# create a quantum model which acts as an Hadamard gate after training
class LearnHadamard(QuantumModel):
    def __init__(
        self,
        train_circuit: QuantumCircuit,
        target_circuit: QuantumCircuit,
        backend="pyqtorch",
    ):
        super().__init__(circuit=train_circuit, backend=backend)
        self.overlap_fn = Overlap(train_circuit, target_circuit, backend=backend, method="exact", diff_mode='ad')

    def forward(self):
        return self.overlap_fn()

    # compute the wavefunction of the associated train circuit
    def wavefunction(self):
        return model.overlap_fn.run({})


train_circuit = QuantumCircuit(1, chain(RX(0, "phi"), RY(0, "theta")))
target_circuit = QuantumCircuit(1, H(0))

model = LearnHadamard(train_circuit, target_circuit)

# get the overlap between model and target circuit wavefunctions
print(model())
```

This model can then be trained with the standard Qadence helper functions.

```python exec="on" source="material-block" result="json" session="custom-model"
from qadence import run
from qadence.ml_tools import Trainer, TrainConfig

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

def loss_fn(model: LearnHadamard, _unused) -> tuple[torch.Tensor, dict]:
    loss = criterion(torch.tensor([[1.0]]), model())
    return loss, {}

config = TrainConfig(max_iter=2500)
trainer = Trainer(
    model, optimizer, config, loss_fn
)
model, optimizer = trainer.fit()

wf_target = run(target_circuit)
assert torch.allclose(wf_target, model.wavefunction(), atol=1e-2)
```
