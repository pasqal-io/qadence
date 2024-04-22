Qadence offers convenience functions for serializing and deserializing any
quantum program. This is useful for storing quantum programs and
sending them for execution over the network via an API.

!!! note
    Qadence currently uses a custom JSON serialization as interchange format. Support for QASM
    format for digital quantum programs is currently under consideration.

* `serialize/deserialize`: serialize and deserialize a Qadence object into a dictionary
* `save/load`: save and load a Qadence object to a file with one of the supported
  formats. Currently, these are `.json` and the PyTorch-compatible `.pt` format.

Let's start with serialization into a dictionary.

```python exec="on" source="material-block" session="seralize_2"
import torch
from qadence import QuantumCircuit, QuantumModel, DiffMode
from qadence import chain, hamiltonian_factory, feature_map, hea, Z
from qadence.serialization import serialize, deserialize

n_qubits = 4

my_block = chain(feature_map(n_qubits, param="x"), hea(n_qubits, depth=2))
obs = hamiltonian_factory(n_qubits, detuning=Z)

# Use the block defined above to create a quantum circuit
# serialize/deserialize it
qc = QuantumCircuit(n_qubits, my_block)
qc_dict = serialize(qc)
qc_deserialized = deserialize(qc_dict)
assert qc == qc_deserialized

# Let's wrap it in a QuantumModel
# and serialize it
qm = QuantumModel(qc, obs, diff_mode=DiffMode.AD)
qm_dict = serialize(qm)
qm_deserialized = deserialize(qm_dict)

# Check if the loaded QuantumModel returns the same expectation
values = {"x": torch.rand(10)}
assert torch.allclose(qm.expectation(values=values), qm_deserialized.expectation(values=values))
```


Finally, we can save the quantum circuit and the model with the two supported formats.

```python exec="on" source="material-block" session="seralize_2"
from qadence.serialization import serialize, deserialize, save, load, SerializationFormat

qc_fname = "circuit"
save(qc, folder=".", file_name=qc_fname, format=SerializationFormat.PT)
loaded_qc = load(f"{qc_fname}.pt")
assert qc == loaded_qc

qm_fname = "model"
save(qm, folder=".", file_name=qm_fname, format=SerializationFormat.JSON)
model = load(f"{qm_fname}.json")
assert isinstance(model, QuantumModel)
import os  # markdown-exec: hide

os.remove(f"{qc_fname}.pt")  # markdown-exec: hide
os.remove(f"{qm_fname}.json")  # markdown-exec: hide
```
