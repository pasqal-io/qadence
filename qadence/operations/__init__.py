from __future__ import annotations

from .analog import (
    AnalogEntanglement,
    AnalogInteraction,
    AnalogRot,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    AnalogSWAP,
    ConstantAnalogRotation,
    entangle,
)
from .control_ops import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CZ,
    MCPHASE,
    MCRX,
    MCRY,
    MCRZ,
    MCZ,
    Toffoli,
)
from .ham_evo import HamEvo
from .parametric import PHASE, RX, RY, RZ, U
from .primitive import SWAP, H, I, N, Projector, S, SDagger, T, TDagger, X, Y, Z, Zero

# gate sets
# FIXME: this could be inferred by the number of qubits if we had
# a class property for each operation. The number of qubits can default
# to None for operations which do not have it by default
# this would allow to greatly simplify the tests
pauli_gateset: list = [I, X, Y, Z]
# FIXME: add Tdagger when implemented
single_qubit_gateset = [X, Y, Z, H, I, RX, RY, RZ, U, S, SDagger, T, PHASE]
two_qubit_gateset = [CNOT, SWAP, CZ, CRX, CRY, CRZ, CPHASE]
three_qubit_gateset = [CSWAP]
multi_qubit_gateset = [Toffoli, MCRX, MCRY, MCRZ, MCPHASE, MCZ]
analog_gateset = [
    HamEvo,
    ConstantAnalogRotation,
    AnalogEntanglement,
    AnalogSWAP,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    AnalogInteraction,
    entangle,
]
non_unitary_gateset = [Zero, N, Projector]


# Modules to be automatically added to the qadence namespace
__all__ = [
    "X",
    "Y",
    "Z",
    "N",
    "H",
    "I",
    "Zero",
    "RX",
    "RY",
    "RZ",
    "U",
    "CNOT",
    "CZ",
    "MCZ",
    "HamEvo",
    "CRX",
    "MCRX",
    "CRY",
    "MCRY",
    "CRZ",
    "MCRZ",
    "T",
    "TDagger",
    "S",
    "SDagger",
    "SWAP",
    "PHASE",
    "CPHASE",
    "CSWAP",
    "MCPHASE",
    "Toffoli",
    "entangle",
    "AnalogEntanglement",
    "AnalogInteraction",
    "AnalogRot",
    "AnalogRX",
    "AnalogRY",
    "AnalogRZ",
    "AnalogSWAP",
    "Projector",
]
