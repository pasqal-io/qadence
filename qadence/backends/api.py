from __future__ import annotations

from qadence.backend import Backend, BackendConfiguration
from qadence.backends.pytorch_wrapper import DifferentiableBackend
from qadence.extensions import available_backends, set_backend_config
from qadence.types import BackendName, DiffMode

__all__ = ["backend_factory", "config_factory"]


def backend_factory(
    backend: BackendName | str,
    diff_mode: DiffMode | str | None = None,
    configuration: BackendConfiguration | dict | None = None,
) -> Backend | DifferentiableBackend:
    backend_inst: Backend | DifferentiableBackend
    backend_name = BackendName(backend)
    backends = available_backends()

    try:
        BackendCls = backends[backend_name]
    except (KeyError, ValueError):
        raise NotImplementedError(f"The requested backend '{backend_name}' is not implemented.")

    default_config = BackendCls.default_configuration()
    if configuration is None:
        configuration = default_config
    elif isinstance(configuration, dict):
        configuration = config_factory(backend_name, configuration)
    else:
        # NOTE: types have to match exactly, hence we use `type`
        if not isinstance(configuration, type(BackendCls.default_configuration())):
            raise ValueError(
                f"Given config class '{type(configuration)}' does not match the backend",
                f" class: '{BackendCls}'. Expected: '{type(BackendCls.default_configuration())}.'",
            )

    # Create the backend
    backend_inst = BackendCls(
        config=configuration
        if configuration is not None
        else BackendCls.default_configuration()  # type: ignore[attr-defined]
    )

    # Set backend configurations which depend on the differentiation mode
    set_backend_config(backend_inst, diff_mode)

    if diff_mode is not None:
        backend_inst = DifferentiableBackend(backend_inst, DiffMode(diff_mode))
    return backend_inst


def config_factory(name: BackendName | str, config: dict) -> BackendConfiguration:
    backends = available_backends()

    try:
        BackendCls = backends[BackendName(name)]
    except KeyError:
        raise NotImplementedError(f"The requested backend '{name}' is not implemented!")

    BackendConfigCls = type(BackendCls.default_configuration())
    return BackendConfigCls(**config)  # type: ignore[no-any-return]
