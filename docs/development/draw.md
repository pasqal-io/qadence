# `qadence.draw` example plots

Mostly for quick, manual checking of correct plotting output.

```python exec="on" source="material-block" html="1"
from qadence import X, Y, kron
from qadence.draw import display

b = kron(X(0), Y(1))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(b)) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import X, Y, chain
from qadence.draw import display

b = chain(X(0), Y(0))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(b)) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import X, Y, chain
from qadence.draw import display

b = chain(X(0), Y(1))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(b)) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import X, Y, add
from qadence.draw import display

b = add(X(0), Y(1), X(2))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(b)) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import CNOT, RX, HamEvo, X, Y, Z, chain, kron

rx = kron(RX(3,0.5), RX(2, "x"))
rx.tag = "rx"
gen = chain(Z(i) for i in range(4))

# `chain` puts things in sequence
block = chain(
	kron(X(0), Y(1), rx),
	CNOT(2,3),
	HamEvo(gen, 10)
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import feature_map, hea, chain, ReuploadScaling

block = chain(feature_map(4, reupload_scaling=ReuploadScaling.TOWER), hea(4,2))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```


## Developer documentation

This section contains examples in pure graphviz that can be used to understand roughly what is done
in the actual drawing backend.

```python exec="on" source="material-block" result="json" session="draw-dev"
import graphviz

font_name = "Sans-Serif"
font_size = "8"

graph_attr = {
    "rankdir": "LR",  # LR = left to right, TB = top to bottom
    "nodesep": "0.1",  # In inches, tells distance between nodes without edges
    "compound": "true",  # Needed to draw properly edges in hamevo when content is hidden
    "splines": "false",  # Needed to draw control gates vertical lines one over the other
}  # These are the default values for graphs

node_attr = {
    "shape": "box",  # 'box' for normal nodes, 'point' for control gates or 'plaintext' for starting nodes (the qubit label).
    "style": "rounded",  # Unfortunately we can't specify the radius of the rounded, at least for this version
    "fontname": font_name,
    "fontsize": font_size,
    "width": "0.1",  # In inches, it doesn't get tinier than the label font.
    "height": "0.1"  # In inches, it doesn't get tinier than the label font.
}  # These are the defaults values that can be overridden at node declaration.

default_cluster_attr = {
    "fontname": font_name,
    "fontsize": font_size,
    "labelloc": "b",  # location of cluster label. b as bottom, t as top
    "style": "rounded"
} # These are the defaults values that can be overridden at sub graph declaration

hamevo_cluster_attr = {
    "label": "HamEvo(t=10)"
}
hamevo_cluster_attr.update(default_cluster_attr)

h = graphviz.Graph(graph_attr=graph_attr, node_attr=node_attr)
h.node("Hello World!")
h
```

```python exec="on" source="material-block" result="json" session="draw-dev"
# Define graph
h = graphviz.Graph(node_attr=node_attr, graph_attr=graph_attr)

# Add start and end nodes
for i in range(4):
    h.node(f's{i}', shape="plaintext", label=f'{i}', group=f"{i}")
    h.node(f'e{i}', style='invis', group=f"{i}")

# Add nodes
h.node('X', group="0")
h.node('Y', group="1")

# Add hamevo and its nodes
hamevo = graphviz.Graph(name='cluster_hamevo', graph_attr=hamevo_cluster_attr)
for i in range(4):
    hamevo.node(f'z{i}', shape="box", style="invis", label=f'{i}', group=f"{i}")
h.subgraph(hamevo)

# Add rx gates cluster and its nodes
cluster_attr = {"label": "RX gates"}
cluster_attr.update(default_cluster_attr)
cluster = graphviz.Graph(name="cluster_0", graph_attr=cluster_attr)
cluster.node('RX(x)', group="2")
cluster.node('RX(0.5)', group="3")
h.subgraph(cluster)

h.node('cnot0', label='', shape='point', width='0.1', group='0')
h.node('cnot1', label='X', group='1')
h.node('cnot2', label='', shape='point', width='0.1', group='2')
h.node('cnot3', label='', shape='point', width='0.1', group='3')

# Add edges
h.edge('s0', 'X')
h.edge('X', 'cnot0')
h.edge('cnot0', 'z0', lhead='cluster_hamevo')
h.edge('z0', 'e0', ltail='cluster_hamevo')
h.edge('s1', 'Y')
h.edge('Y', 'cnot1')
h.edge('cnot1', 'z1', lhead='cluster_hamevo')
h.edge('z1', 'e1', ltail='cluster_hamevo')
h.edge('s2', 'RX(x)')
h.edge('RX(x)', 'cnot2')
h.edge('cnot2', 'z2', lhead='cluster_hamevo')
h.edge('z2', 'e2', ltail='cluster_hamevo')
h.edge('s3', 'RX(0.5)')
h.edge('RX(0.5)', 'cnot3')
h.edge('cnot3', 'z3', lhead='cluster_hamevo')
h.edge('z3', 'e3', ltail='cluster_hamevo')
h.edge('cnot1', 'cnot0', constraint='false')  # constraint: false is needed to draw vertical edges
h.edge('cnot1', 'cnot2', constraint='false')  # constraint: false is needed to draw vertical edges
h.edge('cnot1', 'cnot3', constraint='false')  # constraint: false is needed to draw vertical edges
h
```

### Example of cluster of clusters
```python exec="on" source="material-block" result="json" session="draw-dev"
# Define graph
h = graphviz.Graph(node_attr=node_attr, graph_attr=graph_attr)

# Define start and end nodes
for i in range(4):
    h.node(f's{i}', shape="plaintext", label=f'{i}', group=f"{i}")
    h.node(f'e{i}', style='invis', group=f"{i}")

# Define outer cluster
cluster_attr = {"label": "Outer cluster"}
cluster_attr.update(default_cluster_attr)
outer_cluster = graphviz.Graph(name="cluster_outer", graph_attr=cluster_attr)

# Define inner cluster 1 and its nodes
cluster_attr = {"label": "Inner cluster 1"}
cluster_attr.update(default_cluster_attr)
inner1_cluster = graphviz.Graph(name="cluster_inner1", graph_attr=cluster_attr)
inner1_cluster.node("a0", group="0")
inner1_cluster.node("a1", group="1")
outer_cluster.subgraph(inner1_cluster)

# Define inner cluster 2 and its nodes
cluster_attr = {"label": "Inner cluster 2"}
cluster_attr.update(default_cluster_attr)
inner2_cluster = graphviz.Graph(name="cluster_inner2", graph_attr=cluster_attr)
inner2_cluster.node("a2", group="2")
inner2_cluster.node("a3", group="3")
outer_cluster.subgraph(inner2_cluster)

# This has to be done here, after inner clusters definitions
h.subgraph(outer_cluster)

# Define more nodes
for i in range(4):
    h.node(f"b{i}", group=f"{i}")

for i in range(4):
    h.edge(f's{i}', f'a{i}')
    h.edge(f'a{i}', f'b{i}')
    h.edge(f'b{i}', f'e{i}')

h
```
