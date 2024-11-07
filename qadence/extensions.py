from __future__ import annotations

import importlib
from logging import getLogger
from typing import TypeVar

from qadence.backend import Backend, BackendConfiguration
from qadence.blocks.abstract import TAbstractBlock
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.types import BackendName, DiffMode, Engine

BackendClsType = TypeVar("BackendClsType", bound=Backend)
EngineClsType = TypeVar("EngineClsType", bound=DifferentiableBackend)

logger = getLogger(__name__)


class ConfigNotFoundError(ModuleNotFoundError):
    ...


class BackendNotFoundError(ModuleNotFoundError):
    ...


class EngineNotFoundError(ModuleNotFoundError):
    ...


class SupportedGatesNotFoundError(ModuleNotFoundError):
    ...


def import_config(backend_name: str | BackendName) -> BackendConfiguration:
    module_path = f"qadence.backends.{backend_name}.config"
    cfg: BackendConfiguration
    try:
        module = importlib.import_module(module_path)
        cfg = getattr(module, "Configuration")
    except (ModuleNotFoundError, ImportError) as e:
        msg = f"Failed to import backend config for '{backend_name}' due to: '{e.msg}'."
        raise ConfigNotFoundError(msg)
    return cfg


def import_backend(backend_name: str | BackendName) -> Backend:
    module_path = f"qadence.backends.{backend_name}.backend"
    backend: Backend
    try:
        module = importlib.import_module(module_path)
    except (ModuleNotFoundError, ImportError) as e:
        # If backend is not in Qadence, search in extensions.
        module_path = f"qadence_extensions.backends.{backend_name}.backend"
        try:
            module = importlib.import_module(module_path)
        except (ModuleNotFoundError, ImportError) as e:
            msg = f"Failed to import backend '{backend_name}' due to: '{e.msg}'."
            raise BackendNotFoundError(msg)
    backend = getattr(module, "Backend")
    return backend


def _available_backends() -> dict[BackendName, Backend]:
    """Return a dictionary of currently installed, native qadence backends."""
    res: dict[BackendName, Backend] = dict()
    for backend in BackendName.list():
        try:
            res[backend] = import_backend(backend)
        except BackendNotFoundError as e:
            raise e
    logger.debug(f"Found backends: {res.keys()}")
    return res


def import_engine(engine_name: str | Engine) -> DifferentiableBackend:
    module_path = f"qadence.engines.{engine_name}.differentiable_backend"
    engine: DifferentiableBackend
    try:
        module = importlib.import_module(module_path)
        engine = getattr(module, "DifferentiableBackend")
    except (ModuleNotFoundError, ImportError) as e:
        msg = f"Failed to import engine '{engine_name}' due to: '{e.msg}'."
        raise EngineNotFoundError(msg)
    return engine


def _available_engines() -> dict[Engine, DifferentiableBackend]:
    """Return a dictionary of currently installed, native qadence engines."""
    res: dict[Engine, DifferentiableBackend] = dict()
    for engine in Engine.list():
        try:
            res[engine] = import_engine(engine)
        except EngineNotFoundError as e:
            raise e
    logger.debug(f"Found engines: {res.keys()}")
    return res


def _supported_gates(backend_name: str) -> list[TAbstractBlock]:
    """Return a list of supported gates for the queried backend 'name'."""
    from qadence import operations

    backend_name = BackendName(backend_name)  # Validate backend name.
    module_path = f"qadence.backends.{backend_name}"

    try:
        module = importlib.import_module(module_path)
    except (ModuleNotFoundError, ImportError) as e:
        msg = f"Failed to import supported gates for '{backend_name}' due to: '{e.msg}'."
        raise SupportedGatesNotFoundError(msg)
    _supported_gates = getattr(module, "supported_gates")
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

    # (1) When using PSR with any backend or (2) we use the backends Pulser,
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
