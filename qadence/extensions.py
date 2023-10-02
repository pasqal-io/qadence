from __future__ import annotations

import importlib
from string import Template
from typing import TypeVar

from qadence.backend import Backend
from qadence.blocks import (
    AbstractBlock,
)
from qadence.types import BackendName, DiffMode

TAbstractBlock = TypeVar("TAbstractBlock", bound=AbstractBlock)

backends_namespace = Template("qadence.backends.$name")


def _available_backends() -> dict:
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
    # avoid circular import
    from qadence.backends.gpsr import general_psr

    return {DiffMode.GPSR: general_psr}


def _validate_diff_mode(backend: Backend, diff_mode: DiffMode) -> None:
    if not backend.supports_ad and diff_mode == DiffMode.AD:
        raise TypeError(f"Backend {backend.name} does not support diff_mode {DiffMode.AD}.")


def _set_backend_config(backend: Backend, diff_mode: DiffMode) -> None:
    """_summary_

    Args:
        backend (Backend): _description_
        diff_mode (DiffMode): _description_
    """

    _validate_diff_mode(backend, diff_mode)

    if not backend.supports_ad or diff_mode != DiffMode.AD:
        backend.config._use_gate_params = True

    # (1) When using PSR with any backend or (2)  we use the backends Pulser or Braket,
    # we have to use gate-level parameters

    else:
        assert diff_mode == DiffMode.AD
        backend.config._use_gate_params = False
        # We can use expression-level parameters for AD.
        if backend.name == BackendName.PYQTORCH:
            backend.config.use_single_qubit_composition = True

        # For pyqtorch, we enable some specific transpilation passes.


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
