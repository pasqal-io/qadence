from __future__ import annotations

import importlib

from qadence.logger import get_logger

logger = get_logger(__name__)

try:
    module = importlib.import_module("qermod")
    AbstractNoise = getattr(module, "AbstractNoise")
    NoiseCategory = getattr(module, "NoiseCategory")
    available_protocols = getattr(module, "protocols")
    PrimitiveNoise = getattr(available_protocols, "PrimitiveNoise")
    serialize_noise = getattr(module, "serialize")
    deserialize_noise = getattr(module, "deserialize")
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'qermod' module is not present." " Please install the 'qermod' package."
    )
