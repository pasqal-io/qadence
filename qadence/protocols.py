from __future__ import annotations

import importlib

from qadence.logger import get_logger

logger = get_logger(__name__)

try:
    module = importlib.import_module("qadence_protocols.protocols")
    available_protocols = getattr(module, "available_protocols")
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'qadence_protocols' module is not present."
        " Please install the 'qadence-protocol' package."
    )
