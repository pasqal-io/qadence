from __future__ import annotations

from .identity_qnn import IdentityQNN
from .qnn import QNN
from .quantum_model import QuantumModel

# Modules to be automatically added to the qadence namespace
__all__ = ["QNN", "QuantumModel", "IdentityQNN"]
