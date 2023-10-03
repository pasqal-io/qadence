!!! warning "Serialization"
    either on a separate page or move to API? (I would prefer the latter I believe)

```python exec="on" session="seralize"
from rich import print
from qadence.draw import html_string
display = lambda x: print(html_string(x))
```
Here you will learn about some convenience tools offered by Qadence for
constructing quantum programs, state preparation, and serialization of `qadence` objects.


## Serialize and deserialize quantum programs

Qadence offers some convenience functions for serializing and deserializing any
quantum program. This can be very useful for storing quantum programs and
sending them over the network via an API.

!!! note
    Qadence currently uses a custom JSON serialization format. Support for QASM
    format for digital quantum programs will come soon!

Qadence serialization offers two sets of serialization functions which work with
all the main components of Qadence:
* `serialize/deserialize`: serialize and deserialize a Qadence object into a dictionary
* `save/load`: save and load a Qadence object to a file with one of the supported
  formats. This is built on top of the `serialize`/`deserialize` routines.
  Currently, these are `.json` and the PyTorch-compatible `.pt` format.

Let's start with serialization into a dictionary.

```python exec="on" source="material-block" session="seralize_2"
import torch
from qadence import QuantumCircuit, QuantumModel
from qadence import chain, total_magnetization, feature_map, hea
from qadence.serialization import serialize, deserialize
from qadence.serialization import serialize, deserialize

n_qubits = 4

my_block = chain(feature_map(n_qubits, param="x"), hea(n_qubits, depth=2))
obs = total_magnetization(n_qubits)

# use the block defined above to create a quantum circuit
# serialize/deserialize it
qc = QuantumCircuit(n_qubits, my_block)
qc_dict = serialize(qc)
qc_deserialized = deserialize(qc_dict)
assert qc == qc_deserialized

# you can also let's wrap it in a QuantumModel
# and also serialize it
qm = QuantumModel(qc, obs, diff_mode='ad')
qm_dict = serialize(qm)
qm_deserialized = deserialize(qm_dict)

# check if the loaded QuantumModel returns the same expectation
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
import os # markdown-exec: hide
os.remove(f"{qc_fname}.pt") # markdown-exec: hide
os.remove(f"{qm_fname}.json") # markdown-exec: hide
```
