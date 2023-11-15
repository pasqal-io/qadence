In Qadence, quantum programs can be executed by specifying the layout of a register of resources as a lattice.
Built-in [`Register`][qadence.register.Register] types can be used or constructed for arbitrary topologies.
Common register topologies are available and illustrated in the plot below.

```python exec="on" html="1"
import numpy as np
import matplotlib.pyplot as plt
from qadence.register import LatticeTopology, Register

argss = [
    (("line", 4), (-1,4), (-2,2)),
    (("square", 3),  (-2,2), (-2,2)),
    (("circle", 8), (-1.5,1.5), (-1.5,1.5)),
    (("rectangular_lattice", 2, 3), (-1,3), (-1.5,2.0)),
    (("triangular_lattice", 2, 3), (-2,3), (-2,3)),
    (("honeycomb_lattice", 2, 3), (-1,7), (-1,7)),
    (("all_to_all", 7), (-1.3,1.3), (-1.3,1.3)),
]
# make sure that we are plotting all different constructors
assert len(argss) == len(LatticeTopology)-1

s = np.sqrt(len(argss))
width, height = int(np.floor(s)), int(np.ceil(s))
while width * height < len(argss):
    height += 1

fig, axs = plt.subplots(width, height, figsize=(width*5.5, height*2.6))
fig.suptitle("Predefined register topolgies")
axs = axs.flatten()
for i, (args, xl, yl) in enumerate(argss):
    reg = Register.lattice(*args)
    plt.sca(axs[i])
    reg.draw(show=False)
    axs[i].set_title(f"{args[0]}")
    axs[i].set(aspect="equal")
    axs[i].set_xlim(*xl)
    axs[i].set_ylim(*yl)
# make rest of plots invisible
for i in range(len(argss), len(axs)):
    ax = axs[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

## Building and drawing registers

Built-in topologies are directly accessible in the `Register`:

```python exec="on" source="material-block" html="1" session="register"
from docs import docsutils # markdown-exec: hide
import matplotlib.pyplot as plt # markdown-exec: hide
from qadence import Register

reg = Register.all_to_all(n_qubits = 2)
reg_line = Register.line(n_qubits = 2)
reg_circle = Register.circle(n_qubits = 2)
reg_squre = Register.square(qubits_side = 2)
reg_rect = Register.rectangular_lattice(qubits_row = 2, qubits_col = 2)
reg_triang = Register.triangular_lattice(n_cells_row = 2, n_cells_col = 2)
reg_honey = Register.honeycomb_lattice(n_cells_row = 2, n_cells_col = 2)
```

Qubit coordinates are saved as node properties in the underlying NetworkX graph, but can
be accessed directly with the `coords` property.

```python exec="on" source="material-block" result="json" session="register"
reg = Register.square(2)
print(reg.coords)
```

Register coordinates can be re-scaled with the `rescale_coords` method, or a `scale` can be
passed directly as an optional argument at register creation.

```python exec="on" source="material-block" result="json" session="register"
scaled_reg_1 = reg.rescale_coords(scale = 2.0)
scaled_reg_2 = Register.square(2, scale = 2.0)
print(scaled_reg_1.coords)
print(scaled_reg_2.coords)
```

The distance between qubits can also be directly accessed with the `distances` and `all_distances`
properties.

```python exec="on" source="material-block" result="json" session="register"
print("Distance between qubits connected by an edge:")  # markdown-exec: hide
print(reg.distances)
print("Distance between all qubit pairs in the graph")  # markdown-exec: hide
print(reg.all_distances)
```

By calling the `Register` directly, either the number of nodes or a specific graph can be given as input.
If passing a custom graph directly, the node positions will not be defined automatically, and should be
previously saved in the `"pos"` node property. If not, `reg.coords` will return empty tuples and all
distances will be 0.

```python exec="on" source="material-block" result="json" session="register"
import networkx as nx

# Same as Register.all_to_all(n_qubits = 2):
reg = Register(2)

# Register from a custom graph:
graph = nx.complete_graph(3)

# Set node positions, in this case a simple line:
for i, node in enumerate(graph.nodes):
    graph.nodes[node]["pos"] = (1.0 * i, 0.0)

reg = Register(graph)

print(reg.distances)
```


Alternatively, arbitrarily shaped registers can also be constructed by providing the node coordinates.
In this case, there will be no edges automatically created in the connectivity graph.

```python exec="on" source="material-block" html="1"
import numpy as np
from qadence import Register

reg = Register.from_coordinates(
    [(x, np.sin(x)) for x in np.linspace(0, 2*np.pi, 10)]
)

import matplotlib.pyplot as plt # markdown-exec: hide
plt.clf() # markdown-exec: hide
fig = plt.gcf() # markdown-exec: hide
fig.set_size_inches(2*np.pi, 2.5) # markdown-exec: hide
plt.tight_layout() # markdown-exec: hide
reg.draw(show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

!!! warning "Units for qubit coordinates"
    In general, Qadence makes no assumption about the units given to qubit coordinates.
    However, if used in the context of a Hamiltonian coefficient, care should be taken by the user to guarantee the
    quantity $H.t$ is **dimensionless** for exponentiation in the PyQTorch backend, where it is assumed that $\hbar = 1$.
	For registers passed to the [Pulser](https://github.com/pasqal-io/Pulser) backend, coordinates are in $\mu \textrm{m}$.


## Connectivity graphs

Register topology is often assumed in digital simulations to be an all-to-all qubit connectivity.
When running on real devices that enable the [digital-analog](../digital_analog_qc/index.md) computing paradigm,
qubit interactions must be specified either by specifying distances between qubits,
or by defining edges in the register connectivity graph.

It is possible to access the abstract graph nodes and edges to work with if needed as in the [perfect state
transfer](../index.md#analog-emulation-of-a-perfect-state-transfer) example.

```python exec="on" source="material-block" result="json" session="reg-usage"
from qadence import Register

reg = Register.rectangular_lattice(2,3)
print(f"{reg.nodes = }") # markdown-exec: hide
print(f"{reg.edges = }") # markdown-exec: hide
```

Just like the property `all_distances`, there is also an `all_edges` property for convencience:

```python exec="on" source="material-block" result="json" session="reg-usage"
print(reg.all_edges)
```

More details about the usage of Registers in the digital-analog paradigm can be found in the [digital-analog basics](../digital_analog_qc/analog-basics.md) section.
