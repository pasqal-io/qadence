from __future__ import annotations

from importlib import import_module

from torch import cdouble, set_default_dtype
from torch import float64 as torchfloat64

from .analog import *
from .backend import *
from .backends import *
from .blocks import *
from .circuit import *
from .constructors import *
from .exceptions import *
from .execution import *
from .measurements import *
from .ml_tools import *
from .models import *
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

DEFAULT_FLOAT_DTYPE = torchfloat64
DEFAULT_COMPLEX_DTYPE = cdouble
set_default_dtype(DEFAULT_FLOAT_DTYPE)

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
    ".models",
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
