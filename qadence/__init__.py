from __future__ import annotations

import logging
import logging.config
import os
from importlib import import_module
from pathlib import Path

import yaml
from torch import cdouble, set_default_dtype
from torch import float64 as torchfloat64

DEFAULT_FLOAT_DTYPE = torchfloat64
DEFAULT_COMPLEX_DTYPE = cdouble
set_default_dtype(DEFAULT_FLOAT_DTYPE)

logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_CONFIG_PATH = os.environ.get("QADENCE_LOG_CONFIG", f"{Path(__file__).parent}/log_config.yaml")
LOG_BASE_LEVEL = os.environ.get("QADENCE_LOG_LEVEL", "").upper()

with open(LOG_CONFIG_PATH, "r") as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

logger: logging.Logger = logging.getLogger(__name__)
LOG_LEVEL = logging_levels.get(LOG_BASE_LEVEL, logging.INFO)  # type: ignore[arg-type]
logger.setLevel(LOG_LEVEL)
[
    h.setLevel(LOG_LEVEL)  # type: ignore[func-returns-value]
    for h in logger.handlers
    if h.get_name() == "console"
]
logger.debug(f"Qadence logger successfully setup with log level {LOG_LEVEL}")

from .analog import *
from .backend import *
from .backends import *
from .blocks import *
from .circuit import *
from .constructors import *
from .engines import *
from .exceptions import *
from .execution import *
from .measurements import *
from .ml_tools import *
from .model import *
from .noise import *
from .operations import *
from .overlap import *
from .parameters import *
from .register import *
from .serialization import *
from .states import *
from .transpile import *
from .types import *
from .utils import *

"""Fetch the functions defined in the __all__ of each sub-module.

Import to the qadence name space. Make sure each added submodule has the respective definition:

    - `__all__ = ["function0", "function1", ...]`

Furthermore, add the submodule to the list below to automatically build
the __all__ of the qadence namespace. Make sure to keep alphabetical ordering.
"""

list_of_submodules = [
    ".analog",
    ".backends",
    ".blocks",
    ".circuit",
    ".constructors",
    ".noise",
    ".exceptions",
    ".execution",
    ".measurements",
    ".ml_tools",
    ".model",
    ".operations",
    ".overlap",
    ".parameters",
    ".register",
    ".serialization",
    ".states",
    ".transpile",
    ".types",
    ".utils",
]

__all__ = []
for submodule in list_of_submodules:
    __all_submodule__ = getattr(import_module(submodule, package="qadence"), "__all__")
    __all__ += __all_submodule__
