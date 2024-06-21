from __future__ import annotations

import importlib
from logging import getLogger
from string import Template
from typing import TypeVar

from qadence.backend import Backend, BackendConfiguration
from qadence.blocks.abstract import TAbstractBlock
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.logger import get_logger
from qadence.types import BackendName, DiffMode, Engine

backends_namespace = Template("qadence.backends.$name")
BackendClsType = TypeVar("BackendClsType", bound=Backend)
EngineClsType = TypeVar("EngineClsType", bound=DifferentiableBackend)

logger = getLogger(__name__)


def import_config(backend_name: str | BackendName) -> BackendConfiguration:
    module_path = f"qadence.backends.{backend_name}.config"
    cfg: BackendConfiguration
    try:
        module = importlib.import_module(module_path)
        cfg = getattr(module, "Configuration")
    except (ModuleNotFoundError, ImportError) as e:
        raise Exception(f"Failed to import backend config of {backend_name} due to {e}.")
    return cfg


def import_backend(backend_name: str | BackendName) -> Backend:
    module_path = f"qadence.backends.{backend_name}.backend"
    backend: Backend
    try:
        module = importlib.import_module(module_path)
        backend = getattr(module, "Backend")
    except (ModuleNotFoundError, ImportError) as e:
        raise Exception(f"Failed to import backend {backend_name} due to {e}.")
    return backend


def import_engine(engine_name: str | Engine) -> DifferentiableBackend:
    module_path = f"qadence.engines.{engine_name}.differentiable_backend"
    engine: DifferentiableBackend
    try:
        module = importlib.import_module(module_path)
        engine = getattr(module, "DifferentiableBackend")
    except (ModuleNotFoundError, ImportError) as e:
        raise Exception(f"Failed to import backend {engine_name} due to {e}.")
    return engine


def _available_engines() -> dict[Engine, DifferentiableBackend]:
    """Returns a dictionary of currently installed, native qadence engines."""
    res: dict[Engine, DifferentiableBackend] = {}
    for engine in Engine.list():
        try:
            res[engine] = import_engine(engine)
        except (ModuleNotFoundError, ImportError):
            pass
    logger.debug(f"Found engines: {res.keys()}")
    return res


def _available_backends() -> dict[BackendName, Backend]:
    """Returns a dictionary of currently installed, native qadence backends."""
    res: dict[BackendName, Backend] = {}
    for backend in BackendName.list():
        try:
            res[backend] = import_backend(backend)
        except (ModuleNotFoundError, ImportError):
            pass
    logger.debug(f"Found backends: {res.keys()}")
    return res


def _supported_gates(name: BackendName | str) -> list[TAbstractBlock]:
    """Returns a list of supported gates for the queried backend 'name'."""
    from qadence import operations

    name = str(BackendName(name).name.lower())

    try:
        backend_namespace = backends_namespace.substitute(name=name)
        module = importlib.import_module(backend_namespace)
    except KeyError:
        pass
    _supported_gates = getattr(module, "supported_gates", None)
    assert (
        _supported_gates is not None
    ), f"{name} backend should define a 'supported_gates' variable"
    return [getattr(operations, gate) for gate in _supported_gates]


def _gpsr_fns() -> dict:
    """Fallback function for native Qadence GPSR functions if extensions is not present."""
    # avoid circular import
    from qadence.backends.gpsr import general_psr

    return {DiffMode.GPSR: general_psr}


def _validate_diff_mode(backend: Backend, diff_mode: DiffMode) -> None:
    """Fallback function for native Qadence diff_mode if extensions is not present."""
    if not backend.supports_ad and diff_mode == DiffMode.AD:
        raise TypeError(f"Backend {backend.name} does not support diff_mode {DiffMode.AD}.")
    elif not backend.supports_adjoint and diff_mode == DiffMode.ADJOINT:
        raise TypeError(f"Backend {backend.name} does not support diff_mode {DiffMode.ADJOINT}.")


def _set_backend_config(backend: Backend, diff_mode: DiffMode) -> None:
    """Fallback function for native Qadence backends if extensions is not present.

    Args:
        backend (Backend): A backend for execution.
        diff_mode (DiffMode): A differentiation mode.
    """

    _validate_diff_mode(backend, diff_mode)

    # (1) When using PSR with any backend or (2) we use the backends Pulser or Braket,
    # we have to use gate-level parameters

    # We can use expression-level parameters for AD.
    if backend.name == BackendName.PYQTORCH:
        backend.config.use_single_qubit_composition = True
        backend.config._use_gate_params = diff_mode != "ad"
    else:
        backend.config._use_gate_params = True


# if proprietary qadence_plus is available import the
# right function since more backends are supported
try:
    module = importlib.import_module("qadence_extensions.extensions")
    available_backends = getattr(module, "available_backends")
    available_engines = getattr(module, "available_engines")
    supported_gates = getattr(module, "supported_gates")
    get_gpsr_fns = getattr(module, "gpsr_fns")
    set_backend_config = getattr(module, "set_backend_config")
except ModuleNotFoundError:
    available_backends = _available_backends
    available_engines = _available_engines
    supported_gates = _supported_gates
    get_gpsr_fns = _gpsr_fns
    set_backend_config = _set_backend_config
