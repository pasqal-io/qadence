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

To construct programs that work with interacting qubit systems the
[`Register`][qadence.register.Register] lets you construct arbitrary topologies of qubit registers.

Qadence provides a few commonly used register lattices, such as `"line"` or `"rectangular_lattice"`.
The available topologies are shown in the plot above.

## Building registers

As an example, lets construct a honeycomb lattice and draw it:
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

You can also construct arbitrarily shaped registers by manually providing coordinates.
Note that there are no edges defined in `Register`s that are constructed via `from_coordinates`.

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

!!! warning "Qubit coordinate units"
    The coordinates of qubits in `qadence` are *dimensionless*, e.g. for the Pulser backend they are
    converted to $\mu m$.

## Usage

In the digital computing paradigm, register topology is often disregarded in
simulations and an all-to-all qubit connectivity is assumed. This is of course not the case when
running on real devices.  In the [digital-analog](/digital_analog_qc/index.md) computing paradigm,
we have to specify how qubits interact either by taking into account the distances between qubits,
or by manually defining edges in the register graph.

### Abstract graphs

We can ignore the register coordinates and only deal with the edges that are present in the
`Register.edges`. For instance, this is the case in the [perfect state
transfer](/#perfect-state-transfer) example.

```python exec="on" source="material-block" result="json" session="reg-usage"
from qadence import Register

reg = Register.rectangular_lattice(2,3)
print(f"{reg.nodes=}")
print(f"{reg.edges=}")
```

### Graphs with coordinates

If interactions are based on the distance of the individual qubits in the register then instead of
the edges, we deal with `Register.coords` like in
[`add_interaction`][qadence.transpile.emulate.add_interaction].

```python exec="on" source="material-block" result="json" session="reg-usage"
print(f"{reg.coords=}")
```

You might have already seen the [simplest example](/#digital-analog-emulation) that makes
use of register coordinates. See the [digital-analog section](/digital_analog_qc/analog-basics)
for more details.
