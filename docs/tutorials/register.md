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
    reg.draw()
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

```python exec="on" source="material-block" html="1"
from qadence import Register

reg = Register.honeycomb_lattice(2, 3)
import matplotlib.pyplot as plt # markdown-exec: hide
plt.clf() # markdown-exec: hide
reg.draw()
from docs import docsutils # markdown-exec: hide
fig = plt.gcf() # markdown-exec: hide
fig.set_size_inches(3, 3) # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

Arbitrarily shaped registers can be constructed by providing coordinates.

!!! note "Registers defined from coordinates"
	`Register` constructed via the `from_coordinates` method do not define edges in the connectivity graph.

```python exec="on" source="material-block" html="1"
import numpy as np
from qadence import Register

reg = Register.from_coordinates(
    [(x, np.sin(x)) for x in np.linspace(0, 2*np.pi, 10)]
)

import matplotlib.pyplot as plt # markdown-exec: hide
plt.clf() # markdown-exec: hide
reg.draw()
fig = plt.gcf() # markdown-exec: hide
fig.set_size_inches(4, 2) # markdown-exec: hide
plt.tight_layout() # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

!!! warning "Units for qubit coordinates"
    Qubits coordinates in Qadence are *dimensionless* but converted to the required unit when executed on a backend.
	For instance, [Pulser](https://github.com/pasqal-io/Pulser) uses $\mu \textrm{m}$.

## Connectivity graphs

Register topology is often asssumed in simulations to be an all-to-all qubit connectivity.
When running on real devices that enable the [digital-analog](/digital_analog_qc/index.md) computing paradigm,
qubit interaction must be specified either by specifying distances between qubits,
or by defining edges in the register connectivity graph.

It is possible to access the abstract graph nodes and edges to work with if needed as in the [perfect state
transfer](/#perfect-state-transfer) example.

```python exec="on" source="material-block" result="json" session="reg-usage"
from qadence import Register

reg = Register.rectangular_lattice(2,3)
print(f"{reg.nodes = }") # markdown-exec: hide
print(f"{reg.edges = }") # markdown-exec: hide
```

It is possible to customize qubit interaction through the [`add_interaction`][qadence.transpile.emulate.add_interaction] method.
In that case, `Register.coords` are accessible from the concrete graph:


```python exec="on" source="material-block" result="json" session="reg-usage"
print(f"{reg.coords = }") # markdown-exec: hide
```

More details about their usage in the digital-analog paradigm can be found in the [digital-analog basics](/digital_analog_qc/analog-basics) section.
