from __future__ import annotations

import io
from typing import Any

from graphviz import Graph

from .themes import Dark, Light
from .utils import make_diagram
from .vizbackend import Cluster, QuantumCircuitDiagram


def display(
    x: Any,
    qcd: QuantumCircuitDiagram | Cluster | None = None,
    layout: str = "LR",
    theme: str = "light",
    fill: bool = True,
    **kwargs: Any,
) -> Graph:
    """Display a block, circuit, or quantum model.

    The `kwargs` are forwarded to
    the underlying `nx.Graph`, so you can e.g. specify the size of the resulting plot via
    `size="2,2"` (see examples)

    Arguments:
        x: `AbstractBlock`, `QuantumCircuit`, or `QuantumModel`.
        qcd: Circuit diagram to plot the block into.
        layout: Can be either "LR" (left-right), or "TB" (top-bottom).
        theme: Available themes are: ["light", "dark", "black", "white"].
        fill: Whether to fill the passed `x` with identities.
        kwargs: Passed on to `nx.Graph`

    Examples:
    ```python exec="on" source="material-block" html="1"
    from qadence import X, Y, kron
    from qadence.draw import display

    b = kron(X(0), Y(1))
    def display(*args, **kwargs): return args # markdown-exec: hide
    display(b, size="1,1", theme="dark")
    ```
    """
    return make_diagram(x, **kwargs).show()


def savefig(x: Any, filename: str, *args: Any, **kwargs: Any) -> None:
    """Save a block, circuit, or quantum model to file. Accepts the same args/kwargs as `display`.

    Arguments:
        x: `AbstractBlock`, `QuantumCircuit`, or `QuantumModel`.
        filename: Should end in svg/png.
        args: Same as in `display`.
        kwargs: Same as in `display`.

    Examples:
    ```python exec="on" source="material-block" html="1"
    from qadence import X, Y, kron
    from qadence.draw import display

    b = kron(X(0), Y(1))
    def savefig(*args, **kwargs): return args # markdown-exec: hide
    savefig(b, "test.svg", size="1,1", theme="dark")
    ```
    """
    make_diagram(x, *args, **kwargs).savefig(filename)


def html_string(x: Any, *args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()

    qcd = make_diagram(x, *args, **kwargs)
    qcd._build()

    buffer.write(qcd.graph.pipe(format="svg").decode("utf-8"))
    buffer.seek(0)
    return buffer.read()
