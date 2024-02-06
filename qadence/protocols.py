from __future__ import annotations

import importlib
from string import Template

from qadence.backend import Backend
from qadence.blocks.abstract import TAbstractBlock
from qadence.logger import get_logger
from qadence.types import BackendName, DiffMode, Engine

backends_namespace = Template("qadence.backends.$name")

logger = get_logger(__name__)

try:
    module = importlib.import_module("qadence_protocols.mitig")
    available_protocols = getattr(module, "available_protocols")
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'qadence_protocols' module is not present. Please install the 'qadence-protocol' package.")
