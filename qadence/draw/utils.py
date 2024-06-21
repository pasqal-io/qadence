from __future__ import annotations

import re
import shutil
import warnings
from copy import deepcopy
from functools import singledispatch
from tempfile import NamedTemporaryFile
from typing import Any

import sympy

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    CompositeBlock,
    ControlBlock,
    ParametricBlock,
    ParametricControlBlock,
    PrimitiveBlock,
    ScaleBlock,
    chain,
)
from qadence.blocks.analog import ConstantAnalogRotation, InteractionBlock
from qadence.circuit import QuantumCircuit
from qadence.model import QuantumModel
from qadence.operations import RX, RY, RZ, SWAP, HamEvo, I
from qadence.transpile.block import fill_identities
from qadence.utils import format_parameter

from .vizbackend import Cluster, QuantumCircuitDiagram

# FIXME: move this to a config/theme?
USE_LATEX_LABELS = False
USE_SUBSCRIPTS = True


def fixed(expr: sympy.Basic) -> bool:
    return expr.is_number  # type: ignore


def trainable(expr: sympy.Basic) -> bool:
    for s in expr.free_symbols:
        if hasattr(s, "trainable") and s.trainable:
            return True
    return False


def _get_latex_label(block: AbstractBlock, color: str = "red", fontsize: int = 30) -> str:
    from latex2svg import default_params, latex2svg
    from lxml import etree

    # FIXME: use this and e.g.
    # qcd.create_node(
    #     wire, label="", image=_get_label(block, qcd.theme.color), block_type="primitive")
    name = sympy.Function(type(block).__name__)
    if isinstance(block, (RX, RY, RZ)):
        p = block.parameters.parameter
        expr = name(p)
    else:
        expr = name

    lx = sympy.latex(expr, mode="inline")

    latex_params = deepcopy(default_params)
    preamble = r"""
    \usepackage[utf8x]{inputenc}
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{amstext}
    \usepackage{xcolor}
    \everymath{\color{black}}
    \everydisplay{\color{black}}
    \def\m@th{\normalcolor\mathsurround\z@}

    """
    preamble = preamble.replace("black", color)
    preamble += r"\usepackage[bitstream-charter]{mathdesign}"
    latex_params["preamble"] = preamble

    svg = latex2svg(lx, params=latex_params)
    svgstr = svg["svg"]
    root = etree.fromstring(svgstr)
    # (x, y_min, width, height) = map(lambda x: fontsize*float(x), root.get("viewBox").split(" "))
    # root.set("viewBox", f"{x_min} {y_min} {width} {height}")
    root.set("width", str(round(svg["width"] * fontsize)))
    root.set("height", str(round(svg["height"] * fontsize)))
    svgstr = etree.tostring(root, encoding="utf-8").decode("utf-8")

    fi = NamedTemporaryFile(delete=False, mode="w", suffix=".svg")
    fi.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    fi.write(svgstr)
    return fi.name


def _is_number(s: str) -> bool:
    # match numbers with an optional minus sign and optional decimal part.
    pattern = re.compile(r"^-?\d+(\.\d+)?$")
    return bool(pattern.match(s))


def _subscript(x: str) -> str:
    offset = ord("₁") - ord("1")
    return chr(ord(x) + offset) if _is_number(x) else x


def _index_to_subscripts(x: str) -> str:
    return re.sub(r"_\d+", lambda match: "".join(map(_subscript, match.group()[1:])), x)


def _expr_string(expr: sympy.Basic) -> str:
    return _index_to_subscripts(format_parameter(expr))


def _get_label(block: AbstractBlock) -> str:
    name = sympy.Function(type(block).__name__)
    if isinstance(block, ParametricBlock):
        p = block.parameters.parameter
        expr = name(p)
    else:
        expr = name
    return _expr_string(expr) if USE_SUBSCRIPTS else format_parameter(expr)


@singledispatch
def make_diagram(
    x: Any,
    qcd: QuantumCircuitDiagram | Cluster | None = None,
    layout: str = "LR",
    theme: str = "light",
    fill: bool = True,
    **kwargs: Any,
) -> QuantumCircuitDiagram:
    raise ValueError(f"Cannot construct circuit diagram for type: {type(x)}")


@make_diagram.register
def _(circuit: QuantumCircuit, *args: Any, **kwargs: Any) -> QuantumCircuitDiagram:
    return make_diagram(circuit.block, *args, nb_wires=circuit.n_qubits, **kwargs)


@make_diagram.register
def _(model: QuantumModel, *args: Any, **kwargs: Any) -> QuantumCircuitDiagram:
    if model.out_features is not None:
        if model.out_features > 1:
            raise ValueError("Cannot visualize QuantumModel with more than one observable.")

        obs = deepcopy(model._observable[0].original)  # type: ignore [index]
        obs.tag = "Obs."

        block: AbstractBlock = chain(model._circuit.original.block, obs)
    else:
        block = model._circuit.original.block
    return make_diagram(block, *args, **kwargs)


@make_diagram.register
def _(
    block: AbstractBlock,
    qcd: QuantumCircuitDiagram = None,
    layout: str = "LR",
    theme: str = "light",
    fill: bool = True,
    **kwargs: Any,
) -> QuantumCircuitDiagram:
    if fill:
        from qadence.transpile import transpile

        block = transpile(
            lambda b: fill_identities(b, 0, b.n_qubits - 1),
            # FIXME: enabling flatten can sometimes prevent wires from bending
            # but flatten currently gets rid of some tags... fix that and comment in:
            # flatten
        )(block)

    if qcd is None:
        nb_wires = kwargs.pop("nb_wires") if "nb_wires" in kwargs else block.n_qubits
        qcd = QuantumCircuitDiagram(nb_wires=nb_wires, layout=layout, theme=theme, **kwargs)

    if isinstance(block, I):
        wire = block.qubit_support[0]
        qcd.create_identity_node(wire)

    elif isinstance(block, SWAP):
        qcd.create_swap_gate(*block.qubit_support)  # type: ignore

    elif isinstance(block, (ControlBlock, ParametricControlBlock)):
        from_wire = block.qubit_support[-1]
        to_wires = block.qubit_support[:-1]
        (b,) = block.blocks
        if isinstance(b, ParametricBlock):
            if block.parameters.parameter.is_number:  # type: ignore[union-attr]
                attrs = qcd.theme.get_fixed_parametric_node_attr()
            elif trainable(block.parameters.parameter):  # type: ignore[union-attr]
                attrs = qcd.theme.get_variational_parametric_node_attr()
            else:
                attrs = qcd.theme.get_feature_parametric_node_attr()
        else:
            attrs = qcd.theme.get_primitive_node_attr()

        qcd.create_control_gate(
            from_wire, to_wires, label=_get_label(b), **attrs  # type: ignore[arg-type]
        )

    elif isinstance(block, HamEvo):
        labels = [block.name, f"t = {_expr_string(block.parameters.parameter)}"]
        start, stop = min(block.qubit_support), block.n_qubits
        _make_cluster(qcd, labels, start, stop, qcd.theme.get_hamevo_cluster_attr())

    elif isinstance(block, AddBlock):
        labels = ["AddBlock"]
        start, stop = min(block.qubit_support), block.n_qubits
        _make_cluster(qcd, labels, start, stop, qcd.theme.get_add_cluster_attr())

    elif isinstance(block, InteractionBlock):
        labels = ["Interaction", f"t = {_expr_string(block.parameters.duration)}"]
        is_global = block.qubit_support.is_global
        start = 0 if is_global else min(block.qubit_support)
        stop = qcd.nb_wires if is_global else block.n_qubits
        _make_cluster(qcd, labels, start, stop, qcd.theme.get_add_cluster_attr())

    elif isinstance(block, ConstantAnalogRotation):
        labels = [
            "AnalogRot",
            f"α = {_expr_string(block.parameters.alpha)}",
            f"t = {_expr_string(block.parameters.duration)}",
            f"Ω = {_expr_string(block.parameters.omega)}",
            f"δ = {_expr_string(block.parameters.delta)}",
            f"φ = {_expr_string(block.parameters.phase)}",
        ]
        is_global = block.qubit_support.is_global
        start = 0 if is_global else min(block.qubit_support)
        stop = qcd.nb_wires if is_global else block.n_qubits
        _make_cluster(qcd, labels, start, stop, qcd.theme.get_add_cluster_attr())

    elif isinstance(block, ScaleBlock):
        s = f"[* {_expr_string(block.scale)}]"
        label = s if block.tag is None else f"{block.tag}: {s}"
        cluster = qcd.create_cluster(label, **qcd.theme.get_scale_cluster_attr())  # type: ignore
        make_diagram(block.block, cluster)

    elif isinstance(block, ParametricBlock):
        wire = block.qubit_support[0]

        if block.parameters.parameter.is_number:
            attrs = qcd.theme.get_fixed_parametric_node_attr()
        elif trainable(block.parameters.parameter):
            attrs = qcd.theme.get_variational_parametric_node_attr()
        else:
            attrs = qcd.theme.get_feature_parametric_node_attr()

        if USE_LATEX_LABELS and shutil.which("latex"):
            qcd.create_node(
                wire,
                label="",
                image=_get_latex_label(block, attrs["color"]),
                **attrs,  # type: ignore[arg-type]
            )
        else:
            if USE_LATEX_LABELS:
                warnings.warn(
                    "To get prettier circuit drawings, consider installing LaTeX.", UserWarning
                )
            qcd.create_node(wire, label=_get_label(block), **attrs)  # type: ignore[arg-type]

    elif isinstance(block, PrimitiveBlock):
        wire = block.qubit_support[0]
        attrs = qcd.theme.get_primitive_node_attr()
        if USE_LATEX_LABELS and shutil.which("latex"):
            qcd.create_node(
                wire,
                label="",
                image=_get_latex_label(block, attrs["color"]),
                **attrs,  # type: ignore[arg-type]
            )
        else:
            if USE_LATEX_LABELS:
                warnings.warn(
                    "To get prettier circuit drawings, consider installing LaTeX.", UserWarning
                )
            qcd.create_node(wire, label=_get_label(block), **attrs)  # type: ignore[arg-type]

    elif isinstance(block, CompositeBlock):
        for inner_block in block:
            if inner_block.tag is not None:
                cluster = qcd.create_cluster(
                    inner_block.tag, **qcd.theme.get_cluster_attr()  # type: ignore
                )
                make_diagram(inner_block, cluster, fill=False)
            else:
                make_diagram(inner_block, qcd, fill=False)

    else:
        raise ValueError(f"Don't know how to draw block of type {type(block)}.")

    return qcd


def _make_cluster(
    qcd: QuantumCircuitDiagram, labels: list[str], start: int, stop: int, attrs: dict
) -> None:
    """Construct a cluster with the list of labels centered vertically (in terms of wires).

    If there are fewer wires than labels, plot all lables in one line, assuming that the first
    element in `labels` is the block type.
    """
    N = stop - start

    # draw labels line by line
    if N > len(labels):
        cluster = qcd.create_cluster("", show=False, **attrs)
        before = (N - len(labels)) // 2
        after = N - len(labels) - before
        lines = ["" for _ in range(before)] + labels + ["" for _ in range(after)]
        for i, label in zip(range(start, stop), lines):
            _attrs = deepcopy(qcd.theme.get_node_attr())
            _attrs["shape"] = "none"
            _attrs["style"] = "rounded"
            cluster.show = True
            cluster.create_node(i, label=label, **_attrs)
            cluster.show = False

    # draw all labels in one line if there are too few wires
    else:
        cluster = qcd.create_cluster("", show=False, **attrs)
        label = f"{labels[0]}({', '.join(s.replace(' ', '') for s in labels[1:])})"
        for i in range(start, stop):
            if i == ((stop - start) // 2 + start):
                _attrs = deepcopy(qcd.theme.get_node_attr())
                _attrs["shape"] = "none"
                _attrs["style"] = "rounded"
                cluster.show = True
                cluster.create_node(i, label=label, **_attrs)
                cluster.show = False
            else:
                cluster.create_node(i, label="I", **qcd.theme.get_node_attr())
