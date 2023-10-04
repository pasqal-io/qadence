from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


class BaseTheme:
    name = ""
    background_color = ""
    color = ""
    fontname = "JetBrains Mono"
    fontsize = "20"
    primitive_node: dict[str, str] = {}
    fixed_parametric_node: dict[str, str] = {}
    feature_parametric_node: dict[str, str] = {}
    variational_parametric_node: dict[str, str] = {}
    hamevo_cluster: dict[str, str] = {}
    add_cluster: dict[str, str] = {}
    scale_cluster: dict[str, str] = {}

    @classmethod
    def get_graph_attr(self) -> Dict[str, str]:
        return {
            "bgcolor": self.background_color,
            "nodesep": "0.15",  # This defines the distance between wires
        }

    @classmethod
    def get_node_attr(self) -> Dict[str, str]:
        return {
            "color": self.color,
            "fontcolor": self.color,
            "fontname": self.fontname,
            "fontsize": self.fontsize,
            "width": "0.8",
            "height": "0.8",
            "style": "filled,rounded",
            "shape": "box",
            "penwidth": "2.7",
            "margin": "0.3,0.1",
        }

    @classmethod
    def get_edge_attr(self) -> Dict[str, str]:
        return {"color": self.color, "penwidth": "2.7"}

    @classmethod
    def get_cluster_attr(self) -> Dict[str, str]:
        return {
            "color": self.color,
            "fontcolor": self.color,
            "fontname": self.fontname,
            "fontsize": str(int(int(self.fontsize) * 2 / 3)),
            "labelloc": "t",
            "style": "rounded",
            "penwidth": "2.0",
        }

    @classmethod
    def get_primitive_node_attr(self) -> Dict[str, str]:
        attrs = self.get_node_attr()
        attrs.update(self.primitive_node)
        return attrs

    @classmethod
    def get_fixed_parametric_node_attr(self) -> Dict[str, str]:
        attrs = self.get_node_attr()
        attrs.update(self.fixed_parametric_node)
        return attrs

    @classmethod
    def get_feature_parametric_node_attr(self) -> Dict[str, str]:
        attrs = self.get_node_attr()
        attrs.update(self.feature_parametric_node)
        return attrs

    @classmethod
    def get_variational_parametric_node_attr(self) -> Dict[str, str]:
        attrs = self.get_node_attr()
        attrs.update(self.variational_parametric_node)
        return attrs

    @classmethod
    def get_add_cluster_attr(self) -> Dict[str, str]:
        attrs = self.get_cluster_attr()
        attrs.update(self.add_cluster)
        attrs["bgcolor"] = attrs["fillcolor"]
        return attrs

    @classmethod
    def get_scale_cluster_attr(self) -> Dict[str, str]:
        attrs = self.get_cluster_attr()
        attrs.update(self.scale_cluster)
        attrs["bgcolor"] = attrs["fillcolor"]
        return attrs

    @classmethod
    def get_hamevo_cluster_attr(self) -> Dict[str, str]:
        attrs = self.get_cluster_attr()
        attrs.update(self.hamevo_cluster)
        attrs["bgcolor"] = attrs["fillcolor"]
        return attrs

    @classmethod
    def load_measurement_icon(self) -> str:
        basedir = Path(os.path.abspath(os.path.dirname(__file__)))
        return os.path.join(basedir, "assets", self.name, "measurement.svg")


class Dark(BaseTheme):
    name = "dark"
    background_color = "black"
    color = "white"
    primitive_node = {"fillcolor": "#d03a2f", "color": "#f0c1be"}
    variational_parametric_node = {"fillcolor": "#3182bd", "color": "#afd3e9"}
    fixed_parametric_node = {"fillcolor": "#e6550d", "color": "#fed4b0"}
    feature_parametric_node = {"fillcolor": "#2d954d", "color": "#b2e8c2"}
    hamevo_cluster = {"fillcolor": "black", "color": "grey"}
    add_cluster = {"fillcolor": "black", "color": "grey"}
    scale_cluster = {"fillcolor": "black", "color": "grey"}


class Light(BaseTheme):
    name = "light"
    background_color = "white"
    color = "black"
    primitive_node = {"color": "#d03a2f", "fillcolor": "#f0c1be"}
    variational_parametric_node = {"color": "#3182bd", "fillcolor": "#afd3e9"}
    fixed_parametric_node = {"color": "#e6550d", "fillcolor": "#fed4b0"}
    feature_parametric_node = {"color": "#2d954d", "fillcolor": "#b2e8c2"}
    hamevo_cluster = {"color": "black", "fillcolor": "lightgrey"}
    add_cluster = {"color": "black", "fillcolor": "lightgrey"}
    scale_cluster = {"color": "black", "fillcolor": "lightgrey"}


class Black(BaseTheme):
    name = "black"
    background_color = "black"
    color = "white"
    primitive_node = {"color": "white", "fillcolor": "black"}
    variational_parametric_node = {"color": "white", "fillcolor": "black"}
    fixed_parametric_node = {"color": "white", "fillcolor": "black"}
    feature_parametric_node = {"color": "white", "fillcolor": "black"}
    hamevo_cluster = {"color": "white", "fillcolor": "black"}
    add_cluster = {"color": "white", "fillcolor": "black"}
    scale_cluster = {"color": "white", "fillcolor": "black"}


class White(BaseTheme):
    name = "white"
    background_color = "white"
    color = "black"
    primitive_node = {"color": "black", "fillcolor": "white"}
    variational_parametric_node = {"color": "black", "fillcolor": "white"}
    fixed_parametric_node = {"color": "black", "fillcolor": "white"}
    feature_parametric_node = {"color": "black", "fillcolor": "white"}
    hamevo_cluster = {"color": "black", "fillcolor": "white"}
    add_cluster = {"color": "black", "fillcolor": "white"}
    scale_cluster = {"color": "black", "fillcolor": "white"}
