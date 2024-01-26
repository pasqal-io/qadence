from __future__ import annotations

from .qfi import get_quantum_fisher, get_quantum_fisher_spsa
from .qng import QuantumNaturalGradient

# Modules to be automatically added to the qadence namespace
__all__ = [
    "QuantumNaturalGradient",
    "get_quantum_fisher",
    "get_quantum_fisher_spsa",
]
