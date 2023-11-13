from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from graphviz import Graph

from .themes import BaseTheme, Black, Dark, Light, White

THEMES = {"light": Light, "dark": Dark, "black": Black, "white": White}
LAYOUTS = ["LR", "TB"]


class QuantumCircuitDiagram:
    """This class plots a quantum circuit using Graphviz."""

    __valid_layouts = LAYOUTS
    __themes = THEMES

    __default_graph_attr = {
        "compound": "true",  # This helps to draw edges when clusters do not have to show
        "splines": "false",  # This helps edges perpendicular to wires to occur on the same axis
    }

    def __init__(
        self,
        layout: str = "LR",
        nb_wires: int = 1,
        theme: BaseTheme | str = "light",
        measurement: bool = False,
        **kwargs: Any,
    ):
        self._layout = self._validate_layout(layout)
        self.nb_wires = self._validate_nb_wires(nb_wires)
        self.theme = self._validate_theme(theme)
        self.graph = Graph()
        self.graph.attr(**kwargs)
        # each wire is represented by a list of nodes
        self._wires: List[List[Node]] = [[] for _ in range(self.nb_wires)]
        self._clusters: List[Cluster] = []
        self._measurement: bool = measurement
        self._start_nodes: List[Node] = []
        self._end_nodes: List[Node] = []

    @staticmethod
    def _validate_nb_wires(nb_wires: int) -> int:
        if nb_wires < 1:
            raise ValueError(
                f"Invalid nb_wires {nb_wires}. Only positive integers greater "
                f"than zero are supported."
            )
        return nb_wires

    def _get_graph_attr(self) -> Dict:
        graph_attr = {
            "rankdir": self._layout,
        }
        graph_attr.update(self.__default_graph_attr)
        graph_attr.update(self.theme.get_graph_attr())
        return graph_attr

    def _validate_layout(self, layout: str) -> str:
        layout = layout.upper()
        if layout not in self.__valid_layouts:
            raise ValueError(f"Invalid layout {layout}. Supported: {self.__valid_layouts}")
        return layout

    def _validate_theme(self, theme: BaseTheme | str) -> BaseTheme:
        if isinstance(theme, BaseTheme):
            return theme

        theme = theme.lower()
        if theme not in self.__themes:
            raise ValueError(f"Invalid theme {theme}. Supported: {list(self.__themes.keys())}")
        return self.__themes[theme]()

    def _validate_wire(self, wire: int) -> int:
        if wire >= self.nb_wires:
            raise ValueError(
                f"Invalid wire {wire}. This circuit has {self.nb_wires} wires numbered from 0 "
                f"to {self.nb_wires - 1}"
            )
        return wire

    def _validate_wires_couple(self, wire1: int, wire2: int) -> Tuple[int, int]:
        """Helper to ensure we are not creating a swap gate or a control gate on the same wire."""
        wire1 = self._validate_wire(wire1)
        wire2 = self._validate_wire(wire2)
        if wire1 == wire2:
            raise ValueError("Invalid wires couple: they can't be the same.")
        return wire1, wire2

    def _create_start_node(self, wire: int) -> Node:
        node = self.create_node(
            wire,
            append=False,
            shape="plaintext",
            label=f"{wire}",
            color=self.theme.color,
            fontcolor=self.theme.color,
            fontname=self.theme.fontname,
            fontsize=str(float(self.theme.fontsize) * 3 / 4),
        )
        self._start_nodes.append(node)
        node.generate()
        return node

    def _create_end_node(self, wire: int) -> Node:
        kwargs = {"label": "", "shape": "plaintext"}
        if self._measurement:
            kwargs.update({"image": self.theme.load_measurement_icon(), "fixedsize": "true"})
        node = self.create_node(wire, self, append=False, **kwargs)
        self._end_nodes.append(node)
        node.generate()
        return node

    def _create_swap_departure_node(
        self, wire: int, parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None
    ) -> Node:
        pw = self.theme.get_edge_attr()["penwidth"]
        return self.create_node(
            wire,
            self if parent is None else parent,
            shape="octagon",
            width="0.03",
            height="0.03",
            style="filled",
            color=self.theme.color,
            fillcolor=self.theme.color,
            penwidth="1",
            label="",
        )

    def _create_swap_arrival_node(
        self,
        wire: int,
        swap_departure_node: Node = None,
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
    ) -> Node:
        pw = self.theme.get_edge_attr()["penwidth"]
        return self.create_node(
            wire,
            self if parent is None else parent,
            swap_departure_node=swap_departure_node,
            shape="octagon",
            width="0.03",
            height="0.03",
            style="filled",
            color=self.theme.color,
            fillcolor=self.theme.color,
            penwidth="1",
            label="",
        )

    def _create_control_gate_departure_node(
        self,
        wire: int,
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
        **kwargs: Any,
    ) -> Node:
        return self.create_node(wire=wire, parent=self if parent is None else parent, **kwargs)

    def _create_control_gate_arrival_node(
        self,
        wire: int,
        control_gate_departure_node: Node,
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
    ) -> None:
        self.create_node(
            wire=wire,
            parent=self if parent is None else parent,
            control_gate_departure_node=control_gate_departure_node,
            label="",
            shape="point",
            width="0.2",
            color=self.theme.color,
        )

    def _connect(
        self,
        node1: Node,
        node2: Node,
        **kwargs: str,
    ) -> None:
        """It creates an edge between node1 and node2."""
        edge_attr = self.theme.get_edge_attr()
        edge_attr.update(kwargs)
        if (
            not getattr(node1.parent, "show", True) and node1.parent == node2.parent
        ):  # if we are connecting 2 nodes inside a cluster with show: false
            edge_attr.update({"style": "invis"})
        else:
            if not getattr(node1.parent, "show", True):
                edge_attr.update({"ltail": node1.parent.get_name()})
            if not getattr(node2.parent, "show", True):
                edge_attr.update({"lhead": node2.parent.get_name()})
        self.graph.edge(node1.id, node2.id, **edge_attr)

    def create_control_gate(
        self,
        from_wire: int,
        to_wires: List[int],
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
        **kwargs: Any,
    ) -> None:
        """
        A control gate consists of a node in a wire, and a list of nodes on other different wires.

        All wires are connected with vertical edges between them.
        """
        for to_wire in to_wires:
            from_wire, to_wire = self._validate_wires_couple(from_wire, to_wire)
        departure_node = self._create_control_gate_departure_node(
            from_wire, parent=parent, **kwargs
        )
        for to_wire in to_wires:
            self._create_control_gate_arrival_node(to_wire, departure_node, parent=parent)

    def create_swap_gate(
        self,
        wire_1: int,
        wire_2: int,
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
    ) -> None:
        """
        A swap gate.

        It consists in 4 invisible nodes, 2 invisible edges on the same wire, and 2 dotted
        edges on the destination wires, empty edges on the others
        """
        wire_1, wire_2 = self._validate_wires_couple(wire_1, wire_2)
        departure_node_1 = self._create_swap_departure_node(wire_1, parent=parent)
        departure_node_2 = self._create_swap_departure_node(wire_2, parent=parent)
        self._create_swap_arrival_node(wire_2, swap_departure_node=departure_node_1, parent=parent)
        self._create_swap_arrival_node(wire_1, swap_departure_node=departure_node_2, parent=parent)

    def create_node(
        self,
        wire: int,
        parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None,
        append: bool = True,
        **kwargs: Union[str, Node, None],
    ) -> Node:
        """Create a node on the specified wire for the Quantum Circuit Diagram or for a cluster."""
        self._validate_wire(wire)
        node = Node(wire, self if parent is None else parent, **kwargs)
        if append:
            self._wires[wire].append(node)
        return node

    def create_identity_node(
        self, wire: int, parent: Optional[Union[QuantumCircuitDiagram, Cluster]] = None
    ) -> "Node":
        pw = self.theme.get_edge_attr()["penwidth"]
        return self.create_node(
            wire,
            self if parent is None else parent,
            shape="square",
            label="",
            width="0",
            height="0",
            style="filled",
            color=self.theme.color,
            penwidth=str(float(pw) * 0.5),
        )

    def create_cluster(
        self,
        label: str,
        parent: Optional[Cluster] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Cluster:
        """
        A cluster is a tagged sub diagram of the Quantum Circuit Diagram.

        It's implemented using
        Graphviz's sub graphs.
        """
        cluster = Cluster(
            self if parent is None else parent,
            label,
            show=show,
            **kwargs,
        )
        self._clusters.append(cluster)
        return cluster

    def _build(self) -> None:
        self.graph.graph_attr.update(self._get_graph_attr())
        # keep track of control gates to connect them after generation
        control_gates: list[list[Node]] = []
        # keep track of swap gates to connect them after generation
        swap_gates: list[list[Node]] = []
        for i, wire in enumerate(self._wires):
            start_node = self._create_start_node(i)
            prev_node = start_node
            for j, node in enumerate(wire):
                node.generate()
                if getattr(node, "swap_departure_node", None):
                    self._connect(prev_node, node, style="invis")
                else:
                    self._connect(prev_node, node)
                prev_node = node
                if getattr(node, "control_gate_departure_node", None):
                    control_gates.append([node.control_gate_departure_node, node])  # type:ignore
                if getattr(node, "swap_departure_node", None):
                    swap_gates.append([node.swap_departure_node, node])  # type:ignore
            end_node = self._create_end_node(i)
            self._connect(wire[-1], end_node)

        for cluster in reversed(self._clusters):
            cluster.generate()

        # Connect control and swap gates
        for gate in control_gates:
            self._connect(gate[0], gate[1], constraint="false")
        for gate in swap_gates:
            self._connect(gate[0], gate[1])

        # Connect all starting and ending nodes with invisible edges, this keeps wires in order
        def _linelink(nodes: list) -> None:
            for i in range(1, len(nodes)):
                self._connect(nodes[i - 1], nodes[i], constraint="false", style="invis")

        _linelink(self._start_nodes)
        _linelink(self._end_nodes)

    @staticmethod
    def _runtime() -> str:
        try:
            ipy_str = str(type(get_ipython()))  # type: ignore[name-defined] # noqa
            if "zmqshell" in ipy_str:
                return "jupyter"
            elif "terminal" in ipy_str:
                return "ipython"
            else:
                raise
        except Exception as e:
            return "terminal"

    def show(self) -> Graph:
        self._build()
        if not self._runtime() == "jupyter":
            self.graph.view()
        return self.graph

    def savefig(self, filename: str) -> None:
        fn = Path(filename)
        self._build()
        self.graph.format = fn.suffix[1:]
        self.graph.render(str(fn.parent / fn.stem), view=False)
        # self.graph.save(filename)


class Node:
    def __init__(
        self,
        wire: int,
        parent: Any,
        **kwargs: Any,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.wire = wire
        self.parent = parent
        self.control_gate_departure_node = kwargs.pop("control_gate_departure_node", None)
        self.swap_departure_node = kwargs.pop("swap_departure_node", None)
        self.attrs = {"group": str(wire)}
        self.attrs.update(kwargs)

    def generate(self) -> None:
        self.parent.graph.node(self.id, **self.attrs)


class Cluster:
    def __init__(
        self,
        parent: Any,
        label: str,
        show: bool = True,
        **kwargs: str,
    ) -> None:
        self._id: str = uuid.uuid4().hex
        self.parent: Union[QuantumCircuitDiagram, Cluster] = parent
        self._label: str = label
        graph_attr: Dict[str, str] = {"label": self._label}
        graph_attr.update(kwargs)
        self.graph: Graph = Graph(name=self.get_name(), graph_attr=graph_attr)
        self.show: bool = show

    def __repr__(self) -> str:
        return f"{self._id} - {self._label}"

    def _get_qcd(self) -> QuantumCircuitDiagram:
        parent = self.parent
        while isinstance(parent, Cluster):
            parent = parent.parent
        return parent

    @property
    def theme(self) -> BaseTheme:
        return self._get_qcd().theme

    @property
    def nb_wires(self) -> int:
        return self._get_qcd().nb_wires

    def get_name(self) -> str:
        return f"cluster_{self._id}"

    def generate(self) -> None:
        self.parent.graph.subgraph(self.graph)

    def create_node(self, wire: int, **kwargs: str) -> Node:
        if not self.show:
            kwargs.update({"style": "invis"})
        return self._get_qcd().create_node(wire, parent=self, append=True, **kwargs)

    def create_identity_node(self, wire: int) -> None:
        self._get_qcd().create_identity_node(wire, self)

    def create_cluster(self, label: str, show: bool = True, **kwargs: Any) -> Cluster:
        return self._get_qcd().create_cluster(label, parent=self, show=show, **kwargs)

    def create_control_gate(
        self, from_wire: int, to_wires: List[int], label: str, **kwargs: Any
    ) -> None:
        self._get_qcd().create_control_gate(from_wire, to_wires, label=label, parent=self, **kwargs)

    def create_swap_gate(self, wire_1: int, wire_2: int) -> None:
        self._get_qcd().create_swap_gate(wire_1, wire_2, self)
