from __future__ import annotations

import io
from typing import Any

from graphviz import Graph

from .themes import Dark, Light
from .utils import make_diagram
from .vizbackend import QuantumCircuitDiagram


def display(x: Any, *args: Any, **kwargs: Any) -> Graph:
    return make_diagram(x, *args, **kwargs).show()


def savefig(x: Any, filename: str, *args: Any, **kwargs: Any) -> None:
    make_diagram(x, *args, **kwargs).savefig(filename)


def html_string(x: Any, *args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()

    qcd = make_diagram(x, *args, **kwargs)
    qcd._build()

    buffer.write(qcd.graph.pipe(format="svg").decode("utf-8"))
    buffer.seek(0)
    return buffer.read()
