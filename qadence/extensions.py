from __future__ import annotations

import importlib
from string import Template

from qadence.backend import Backend
from qadence.blocks.abstract import TAbstractBlock
from qadence.logger import get_logger
from qadence.types import BackendName, DiffMode

backends_namespace = Template("qadence.backends.$name")

logger = get_logger(__name__)


def _available_backends() -> dict:
    """Fallback function for native Qadence available backends if extensions is not present."""
    res = {}
    for backend in BackendName.list():
        module_path = f"qadence.backends.{backend}.backend"
        try:
            module = importlib.import_module(module_path)
            BackendCls = getattr(module, "Backend")
            res[backend] = BackendCls
        except (ImportError, ModuleNotFoundError):
            pass
    return res


def _supported_gates(name: BackendName | str) -> list[TAbstractBlock]:
    """Fallback function for native Qadence backend supported gates if extensions is not present."""
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


def _validate_backend_config(backend: Backend) -> None:
    if backend.config.use_gradient_checkpointing:
        msg = "use_gradient_checkpointing is deprecated."
        import warnings

        warnings.warn(msg, UserWarning)
        logger.warn(msg)


def _set_backend_config(backend: Backend, diff_mode: DiffMode) -> None:
    """Fallback function for native Qadence backends if extensions is not present.

    Args:
        backend (Backend): A backend for execution.
        diff_mode (DiffMode): A differentiation mode.
    """

    _validate_diff_mode(backend, diff_mode)
    _validate_backend_config(backend)

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
    supported_gates = getattr(module, "supported_gates")
    get_gpsr_fns = getattr(module, "gpsr_fns")
    set_backend_config = getattr(module, "set_backend_config")
except ModuleNotFoundError:
    available_backends = _available_backends
    supported_gates = _supported_gates
    get_gpsr_fns = _gpsr_fns
    set_backend_config = _set_backend_config
