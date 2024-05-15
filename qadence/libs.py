from __future__ import annotations

import importlib

from qadence.logger import get_logger

logger = get_logger(__name__)

try:
    module = importlib.import_module("qadence_libs.protocols")
    available_libs = getattr(module, "available_libs")
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'qadence_libs' module is not present." " Please install the 'qadence-libs' package."
    )
