from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deepdiff import DeepDiff
from networkx.classes.reportviews import EdgeView, NodeView

from qadence.types import LatticeTopology

# Modules to be automatically added to the qadence namespace
__all__ = ["Register"]


def _scale_node_positions(graph: nx.Graph, scale: float) -> None:
    scaled_nodes = {}
    for k, node in graph.nodes.items():
        (x, y) = node["pos"]
        scaled_nodes[k] = {"pos": (x * scale, y * scale)}
    nx.set_node_attributes(graph, scaled_nodes)


class Register:
    def __init__(self, support: nx.Graph | int):
        """A 2D register of qubits which includes their coordinates (needed for e.g. analog
        computing). The coordinates are ignored in backends that don't need them. The easiest
        way to construct a register is via its classmethods like `Register.triangular_lattice`.

        Arguments:
            support: A graph or number of qubits. Nodes can include a `"pos"` attribute
                such that e.g.: `graph.nodes = {0: {"pos": (2,3)}, 1: {"pos": (0,0)}, ...}` which
                will be used in backends that need qubit coordinates.
                See the classmethods for simple construction of some predefined lattices if you
                don't want to build a graph manually.
                If you pass an integer the resulting register is the same as
                `Register.all_to_all(n_qubits)`.

        Examples:
        ```python exec="on" source="material-block" result="json"
        from qadence import Register

        reg = Register.honeycomb_lattice(2,3)
        reg.draw()
        ```
        """
        self.graph = support if isinstance(support, nx.Graph) else alltoall_graph(support)

    @property
    def n_qubits(self) -> int:
        return len(self.graph)

    @classmethod
    def from_coordinates(
        cls, coords: list[tuple], lattice: LatticeTopology | str = LatticeTopology.ARBITRARY
    ) -> Register:
        graph = nx.Graph()
        for i, pos in enumerate(coords):
            graph.add_node(i, pos=pos)
        return cls(graph)

    @classmethod
    def line(cls, n_qubits: int) -> Register:
        return cls(line_graph(n_qubits))

    @classmethod
    def circle(cls, n_qubits: int, scale: float = 1.0) -> Register:
        graph = nx.grid_2d_graph(n_qubits, 1, periodic=True)
        graph = nx.relabel_nodes(graph, {(i, 0): i for i in range(n_qubits)})
        coords = nx.circular_layout(graph)
        values = {i: {"pos": pos} for i, pos in coords.items()}
        nx.set_node_attributes(graph, values)
        _scale_node_positions(graph, scale)
        return cls(graph)

    @classmethod
    def square(cls, qubits_side: int, scale: float = 1.0) -> Register:
        n_points = 4 * (qubits_side - 1)

        def gen_points() -> np.ndarray:
            rotate_left = np.array([[0.0, -1.0], [1.0, 0.0]])
            increment = np.array([0.0, 1.0])

            points = [np.array([0.0, 0.0])]
            counter = 1
            while len(points) < n_points:
                points.append(points[-1] + increment)

                counter = (counter + 1) % qubits_side
                if counter == 0:
                    increment = rotate_left.dot(increment)
                    counter = 1
            points = np.array(points)  # type: ignore[assignment]
            points -= np.mean(points, axis=0)

            return points  # type: ignore[return-value]

        graph = nx.grid_2d_graph(n_points, 1, periodic=True)
        graph = nx.relabel_nodes(graph, {(i, 0): i for i in range(n_points)})
        values = {i: {"pos": point} for i, point in zip(graph.nodes, gen_points())}
        nx.set_node_attributes(graph, values)
        _scale_node_positions(graph, scale)
        return cls(graph)

    @classmethod
    def all_to_all(cls, n_qubits: int) -> Register:
        return cls(alltoall_graph(n_qubits))

    @classmethod
    def rectangular_lattice(
        cls, qubits_row: int, qubits_col: int, side_length: float = 1.0
    ) -> Register:
        graph = nx.grid_2d_graph(qubits_col, qubits_row)
        values = {i: {"pos": node} for (i, node) in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
        nx.set_node_attributes(graph, values)
        _scale_node_positions(graph, side_length)
        return cls(graph)

    @classmethod
    def triangular_lattice(
        cls, n_cells_row: int, n_cells_col: int, side_length: float = 1.0
    ) -> Register:
        return cls(triangular_lattice_graph(n_cells_row, n_cells_col, side_length))

    @classmethod
    def honeycomb_lattice(cls, n_cells_row: int, n_cells_col: int, scale: float = 1.0) -> Register:
        graph = nx.hexagonal_lattice_graph(n_cells_row, n_cells_col)
        graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
        _scale_node_positions(graph, scale)
        return cls(graph)

    @classmethod
    def lattice(cls, topology: LatticeTopology | str, *args: Any, **kwargs: Any) -> Register:
        return getattr(cls, topology)(*args, **kwargs)  # type: ignore[no-any-return]

    def draw(self, show: bool = True) -> None:
        coords = {i: n["pos"] for i, n in self.graph.nodes.items()}
        nx.draw(self.graph, with_labels=True, pos=coords)
        if show:
            plt.gcf().show()

    def __getitem__(self, item: int) -> Any:
        return self.graph.nodes[item]

    @property
    def support(self) -> set:
        return set(self.graph.nodes)

    @property
    def coords(self) -> dict:
        return {i: tuple(node.get("pos", ())) for i, node in self.graph.nodes.items()}

    @property
    def edges(self) -> EdgeView:
        return self.graph.edges

    @property
    def nodes(self) -> NodeView:
        return self.graph.nodes

    def _scale_positions(self, scale: float) -> Register:
        g = deepcopy(self.graph)
        _scale_node_positions(g, scale)
        return Register(g)

    def _to_dict(self) -> dict:
        return {"graph": nx.node_link_data(self.graph)}

    @classmethod
    def _from_dict(cls, d: dict) -> Register:
        return cls(nx.node_link_graph(d["graph"]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return (
            DeepDiff(self.coords, other.coords, ignore_order=False) == {}
            and nx.is_isomorphic(self.graph, other.graph)
            and self.n_qubits == other.n_qubits
        )


def line_graph(n_qubits: int, spacing: float = 1.0) -> nx.Graph:
    """Create graph representing linear lattice.

    Args:
        n_qubits (int): number of nodes in the graph

    Returns:
        graph instance
    """
    graph = nx.Graph()
    for i in range(n_qubits):
        graph.add_node(i, pos=(i * spacing, 0.0))
    for i, j in zip(range(n_qubits - 1), range(1, n_qubits)):
        graph.add_edge(i, j)
    return graph


def triangular_lattice_graph(
    n_cells_row: int, n_cells_col: int, side_length: float = 1.0
) -> nx.Graph:
    graph = nx.triangular_lattice_graph(n_cells_row, n_cells_col)
    graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
    _scale_node_positions(graph, side_length)
    return graph


def alltoall_graph(n_qubits: int) -> nx.Graph:
    if n_qubits == 2:
        return line_graph(2)
    elif n_qubits == 3:
        return triangular_lattice_graph(1, 1)

    graph = nx.complete_graph(n_qubits)
    # set seed to make sure the produced graphs are reproducible
    coords = nx.spring_layout(graph, seed=0)
    for i, pos in coords.items():
        graph.nodes[i]["pos"] = tuple(pos)
    return graph
