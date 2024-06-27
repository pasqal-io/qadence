from __future__ import annotations

from copy import deepcopy
from itertools import product
from math import dist
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deepdiff import DeepDiff
from networkx.classes.reportviews import EdgeView, NodeView

from qadence.analog import IdealDevice, RydbergDevice
from qadence.types import LatticeTopology

# Modules to be automatically added to the qadence namespace
__all__ = ["Register"]


DEFAULT_DEVICE = IdealDevice()


def _scale_node_positions(graph: nx.Graph, min_distance: float, spacing: float) -> None:
    scaled_nodes = {}
    scale_factor = spacing / min_distance
    for k, node in graph.nodes.items():
        (x, y) = node["pos"]
        scaled_nodes[k] = {"pos": (x * scale_factor, y * scale_factor)}
    nx.set_node_attributes(graph, scaled_nodes)


class Register:
    def __init__(
        self,
        support: nx.Graph | int,
        spacing: float | None = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ):
        """
        A register of qubits including 2D coordinates.

        Instantiating the Register class directly is only recommended for building custom registers.
        For most uses where a predefined lattice is desired it is recommended to use the various
        class methods available, e.g. `Register.triangular_lattice`.

        Arguments:
            support: A NetworkX graph or number of qubits. Nodes can include a `"pos"` attribute
                such that e.g.: `graph.nodes = {0: {"pos": (2,3)}, 1: {"pos": (0,0)}, ...}` which
                will be used in backends that need qubit coordinates. Passing a number of qubits
                calls `Register.all_to_all(n_qubits)`.
            spacing: Value set as the distance between the two closest qubits. The spacing
                argument is also available for all the class method constructors.

        Examples:
        ```python exec="on" source="material-block"
        from qadence import Register

        reg_all = Register.all_to_all(n_qubits = 4)
        reg_line = Register.line(n_qubits = 4)
        reg_circle = Register.circle(n_qubits = 4)
        reg_squre = Register.square(qubits_side = 2)
        reg_rect = Register.rectangular_lattice(qubits_row = 2, qubits_col = 2)
        reg_triang = Register.triangular_lattice(n_cells_row = 2, n_cells_col = 2)
        reg_honey = Register.honeycomb_lattice(n_cells_row = 2, n_cells_col = 2)
        ```
        """
        if device_specs is not None and not isinstance(device_specs, RydbergDevice):
            raise ValueError("Device specs are not valid. Please pass a `RydbergDevice` instance.")

        self.device_specs = device_specs

        self.graph = support if isinstance(support, nx.Graph) else alltoall_graph(support)

        if spacing is not None and self.min_distance != 0.0:
            _scale_node_positions(self.graph, self.min_distance, spacing)

    @property
    def n_qubits(self) -> int:
        """Total number of qubits in the register."""
        return len(self.graph)

    @classmethod
    def from_coordinates(
        cls,
        coords: list[tuple],
        lattice: LatticeTopology | str = LatticeTopology.ARBITRARY,
        spacing: float | None = None,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a register from a list of qubit coordinates.

        Each node is added to the underlying
        graph with the respective coordinates, but the edges are left empty.

        Arguments:
            coords: List of qubit coordinate tuples.
        """
        graph = nx.Graph()
        for i, pos in enumerate(coords):
            graph.add_node(i, pos=pos)
        return cls(graph, spacing, device_specs)

    @classmethod
    def line(
        cls,
        n_qubits: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a line register.

        Arguments:
            n_qubits: Total number of qubits.
        """
        return cls(line_graph(n_qubits), spacing, device_specs)

    @classmethod
    def circle(
        cls,
        n_qubits: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a circle register.

        Arguments:
            n_qubits: Total number of qubits.
        """
        graph = nx.grid_2d_graph(n_qubits, 1, periodic=True)
        graph = nx.relabel_nodes(graph, {(i, 0): i for i in range(n_qubits)})
        coords = nx.circular_layout(graph)
        values = {i: {"pos": pos} for i, pos in coords.items()}
        nx.set_node_attributes(graph, values)
        return cls(graph, spacing, device_specs)

    @classmethod
    def square(
        cls,
        qubits_side: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a square register.

        Arguments:
            qubits_side: Number of qubits on one side of the square.
        """
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
        return cls(graph, spacing, device_specs)

    @classmethod
    def all_to_all(
        cls,
        n_qubits: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a register with an all-to-all connectivity graph.

        The graph is projected
        onto a 2D space and the qubit coordinates are set using a spring layout algorithm.

        Arguments:
            n_qubits: Total number of qubits.
        """
        return cls(alltoall_graph(n_qubits), spacing, device_specs)

    @classmethod
    def rectangular_lattice(
        cls,
        qubits_row: int,
        qubits_col: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        graph = nx.grid_2d_graph(qubits_col, qubits_row)
        values = {i: {"pos": node} for (i, node) in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
        nx.set_node_attributes(graph, values)
        """
        Build a rectangular lattice register.

        Arguments:
            qubits_row: Number of qubits in each row.
            qubits_col: Number of qubits in each column.
        """
        return cls(graph, spacing, device_specs)

    @classmethod
    def triangular_lattice(
        cls,
        n_cells_row: int,
        n_cells_col: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a triangular lattice register.

        Each cell is a triangle made up of three qubits.

        Arguments:
            n_cells_row: Number of cells in each row.
            n_cells_col: Number of cells in each column.
        """
        return cls(triangular_lattice_graph(n_cells_row, n_cells_col), spacing, device_specs)

    @classmethod
    def honeycomb_lattice(
        cls,
        n_cells_row: int,
        n_cells_col: int,
        spacing: float = 1.0,
        device_specs: RydbergDevice = DEFAULT_DEVICE,
    ) -> Register:
        """
        Build a honeycomb lattice register.

        Each cell is an hexagon made up of six qubits.

        Arguments:
            n_cells_row: Number of cells in each row.
            n_cells_col: Number of cells in each column.
        """
        graph = nx.hexagonal_lattice_graph(n_cells_row, n_cells_col)
        graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
        return cls(graph, spacing, device_specs)

    @classmethod
    def lattice(cls, topology: LatticeTopology | str, *args: Any, **kwargs: Any) -> Register:
        return getattr(cls, topology)(*args, **kwargs)  # type: ignore[no-any-return]

    def draw(self, show: bool = True) -> None:
        """Draw the underlying NetworkX graph representing the register."""
        coords = {i: n["pos"] for i, n in self.graph.nodes.items()}
        nx.draw(self.graph, with_labels=True, pos=coords)
        if show:
            plt.gcf().show()

    def __getitem__(self, item: int) -> Any:
        return self.graph.nodes[item]

    @property
    def nodes(self) -> NodeView:
        """Return the NodeView of the underlying NetworkX graph."""
        return self.graph.nodes

    @property
    def edges(self) -> EdgeView:
        """Return the EdgeView of the underlying NetworkX graph."""
        return self.graph.edges

    @property
    def support(self) -> set:
        """Return the set of qubits in the register."""
        return set(self.nodes)

    @property
    def coords(self) -> dict:
        """Return the dictionary of qubit coordinates."""
        return {i: tuple(node.get("pos", ())) for i, node in self.nodes.items()}

    @property
    def all_node_pairs(self) -> EdgeView:
        """Return a list of all possible qubit pairs in the register."""
        return list(filter(lambda x: x[0] < x[1], product(self.support, self.support)))

    @property
    def distances(self) -> dict:
        """Return a dictionary of distances for all qubit pairs in the register."""
        coords = self.coords
        return {edge: dist(coords[edge[0]], coords[edge[1]]) for edge in self.all_node_pairs}

    @property
    def edge_distances(self) -> dict:
        """
        Return a dictionary of distances for the qubit pairs that are.

        connected by an edge in the underlying NetworkX graph.
        """
        coords = self.coords
        return {edge: dist(coords[edge[0]], coords[edge[1]]) for edge in self.edges}

    @property
    def min_distance(self) -> float:
        """Return the minimum distance between two qubts in the register."""
        distances = self.distances
        value: float = min(self.distances.values()) if len(distances) > 0 else 0.0
        return value

    def rescale_coords(self, scaling: float) -> Register:
        """
        Rescale the coordinates of all qubits in the register.

        Arguments:
            scaling: Scaling value.
        """
        g = deepcopy(self.graph)
        _scale_node_positions(g, min_distance=1.0, spacing=scaling)
        return Register(g, spacing=None, device_specs=self.device_specs)

    def _to_dict(self) -> dict:
        return {
            "graph": nx.node_link_data(self.graph),
            "device_specs": self.device_specs._to_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> Register:
        device_dict = d.get("device_specs", None)
        if device_dict is None:
            device_dict = DEFAULT_DEVICE._to_dict()

        return cls(
            support=nx.node_link_graph(d["graph"]),
            device_specs=RydbergDevice._from_dict(device_dict),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return (
            DeepDiff(self.coords, other.coords, ignore_order=False) == {}
            and nx.is_isomorphic(self.graph, other.graph)
            and self.n_qubits == other.n_qubits
        )


def line_graph(n_qubits: int) -> nx.Graph:
    graph = nx.Graph()
    for i in range(n_qubits):
        graph.add_node(i, pos=(i, 0.0))
    for i, j in zip(range(n_qubits - 1), range(1, n_qubits)):
        graph.add_edge(i, j)
    return graph


def triangular_lattice_graph(n_cells_row: int, n_cells_col: int) -> nx.Graph:
    graph = nx.triangular_lattice_graph(n_cells_row, n_cells_col)
    graph = nx.relabel_nodes(graph, {(i, j): k for k, (i, j) in enumerate(graph.nodes)})
    return graph


def alltoall_graph(n_qubits: int) -> nx.Graph:
    if n_qubits == 2:
        return line_graph(2)
    if n_qubits == 3:
        return triangular_lattice_graph(1, 1)

    graph = nx.complete_graph(n_qubits)
    # set seed to make sure the produced graphs are reproducible
    coords = nx.spring_layout(graph, seed=0)
    for i, pos in coords.items():
        graph.nodes[i]["pos"] = tuple(pos)
    return graph
