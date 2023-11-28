from __future__ import annotations

from typing import Union

from sympy import cos, sin

from qadence.analog.addressing import AddressingPattern
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    ConstantAnalogRotation,
)
from qadence.blocks.utils import add
from qadence.operations import I, N, X, Y, Z
from qadence.register import Register

# Ising coupling coefficient depending on the Rydberg level
# Include a normalization to the Planck constant hbar
# In units of [rad . µm^6 / µs]

C6_DICT = {
    50: 96120.72,
    51: 122241.6,
    52: 154693.02,
    53: 194740.36,
    54: 243973.91,
    55: 304495.01,
    56: 378305.98,
    57: 468027.05,
    58: 576714.85,
    59: 707911.38,
    60: 865723.02,
    61: 1054903.11,
    62: 1281042.11,
    63: 1550531.15,
    64: 1870621.31,
    65: 2249728.57,
    66: 2697498.69,
    67: 3224987.51,
    68: 3844734.37,
    69: 4571053.32,
    70: 5420158.53,
    71: 6410399.4,
    72: 7562637.31,
    73: 8900342.14,
    74: 10449989.62,
    75: 12241414.53,
    76: 14308028.03,
    77: 16687329.94,
    78: 19421333.62,
    79: 22557029.94,
    80: 26146720.74,
    81: 30248886.65,
    82: 34928448.69,
    83: 40257623.67,
    84: 46316557.88,
    85: 53194043.52,
    86: 60988354.64,
    87: 69808179.15,
    88: 79773468.88,
    89: 91016513.07,
    90: 103677784.57,
    91: 117933293.96,
    92: 133943541.9,
    93: 151907135.94,
    94: 172036137.34,
    95: 194562889.89,
    96: 219741590.56,
    97: 247850178.91,
    98: 279192193.77,
    99: 314098829.39,
    100: 352931119.11,
}


def rot_generator(block: ConstantAnalogRotation) -> AbstractBlock:
    omega = block.parameters.omega
    delta = block.parameters.delta
    phase = block.parameters.phase
    support = block.qubit_support

    x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
    z_terms = delta * add(N(i) for i in support)
    return x_terms - z_terms  # type: ignore[no-any-return]


def add_pattern(register: Register, pattern: Union[AddressingPattern, None]) -> AbstractBlock:
    support = tuple(range(register.n_qubits))
    if pattern is not None:
        amp = pattern.amp
        det = pattern.det
        weights_amp = pattern.weights_amp
        weights_det = pattern.weights_det
        local_constr_amp = pattern.local_constr_amp
        local_constr_det = pattern.local_constr_det
        global_constr_amp = pattern.global_constr_amp
        global_constr_det = pattern.global_constr_det
    else:
        amp = 0.0
        det = 0.0
        weights_amp = {i: 0.0 for i in support}
        weights_det = {i: 0.0 for i in support}
        local_constr_amp = {i: 0.0 for i in support}
        local_constr_det = {i: 0.0 for i in support}
        global_constr_amp = 0.0
        global_constr_det = 0.0

    p_amp_terms = (
        (1 / 2)  # type: ignore [operator]
        * amp
        * global_constr_amp
        * add(X(i) * weights_amp[i] * local_constr_amp[i] for i in support)  # type: ignore [operator]
    )
    p_det_terms = (
        -det  # type: ignore [operator]
        * global_constr_det
        * add(0.5 * (I(i) - Z(i)) * weights_det[i] * local_constr_det[i] for i in support)  # type: ignore [operator]
    )
    return p_amp_terms + p_det_terms  # type: ignore[no-any-return]
