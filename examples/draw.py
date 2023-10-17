from __future__ import annotations

import os
from sympy import cos, sin

from qadence import *
from qadence.draw import display, savefig
from qadence.draw.themes import BaseTheme


class CustomTheme(BaseTheme):
    background_color = "white"
    color = "black"
    fontname = "Comic Sans MS"
    fontsize = "30"
    primitive_node = {"fillcolor": "green", "color": "black"}
    variational_parametric_node = {"fillcolor": "blue", "color": "black"}
    fixed_parametric_node = {"fillcolor": "red", "color": "black"}
    feature_parametric_node = {"fillcolor": "yellow", "color": "black"}
    hamevo_cluster = {"fillcolor": "pink", "color": "black"}
    add_cluster = {"fillcolor": "white", "color": "black"}
    scale_cluster = {"fillcolor": "white", "color": "black"}


x = Parameter("x")
y = Parameter("y", trainable=False)

constants = kron(tag(kron(X(0), Y(1), H(2)), "a"), tag(kron(Z(5), Z(6)), "z"))
constants.tag = "const"

fixed = kron(RX(0, 0.511111), RY(1, 0.8), RZ(2, 0.9), CRZ(3, 4, 2.2), PHASE(6, 1.1))
fixed.tag = "fixed"

feat = kron(RX(0, y), RY(1, sin(y)), RZ(2, cos(y)), CRZ(3, 4, y**2), PHASE(6, y))
feat.tag = "feat"

vari = kron(RX(0, x**2), RY(1, sin(x)), CZ(3, 2), MCRY([4, 5], 6, "x"))
vari.tag = "vari"

hamevo = HamEvo(kron(*map(Z, range(constants.n_qubits))), 10)

b = chain(
    feature_map(constants.n_qubits, reupload_scaling="Tower"),
    hea(constants.n_qubits, 1),
    constants,
    fixed,
    hamevo,
    feat,
    HamEvo(kron(*map(Z, range(constants.n_qubits))), 10),
    AnalogRX("x"),
    AnalogRX("x", qubit_support=(3, 4, 5)),
    wait("x"),
    vari,
    add(*map(X, range(constants.n_qubits))),
    2.1 * kron(*map(X, range(constants.n_qubits))),
    SWAP(0, 1),
    kron(SWAP(0, 1), SWAP(3, 4)),
)
# b = chain(feature_map(4, fm_type="tower"), hea(4,1, strategy=Strategy.SDAQC))
# d = make_diagram(b)
# d.show()

circuit = QuantumCircuit(b.n_qubits, b)
# you can use the custom theme like this
# display(circuit, theme=CustomTheme())


if os.environ.get("CI") == "true":
    savefig(circuit, "test.svg")
else:
    display(circuit, theme="light")

# FIXME: this is not working yet because total_magnetization blocks completely mess up the
# graph layout for some reason :(
# o = total_magnetization(b.n_qubits)
# m = QuantumModel(c, o)
# d = make_diagram(m)
# d.show()
