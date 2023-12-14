from __future__ import annotations

from qadence.backend import Backend, BackendConfiguration
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.extensions import available_backends, available_engines, set_backend_config
from qadence.types import BackendName, DiffMode, Engine

__all__ = ["backend_factory", "config_factory"]


def backend_factory(
    backend: BackendName | str,
    diff_mode: DiffMode | str | None = None,
    configuration: BackendConfiguration | dict | None = None,
) -> Backend | DifferentiableBackend:
    backend_inst: Backend | DifferentiableBackend
    backends = available_backends()
    try:
        backend_name = BackendName(backend)
    except ValueError:
        raise NotImplementedError(f"The requested backend '{backend}' is not implemented.")
    try:
        BackendCls = backends[backend_name]
    except Exception as e:
        raise ImportError(
            f"The requested backend '{backend_name}' is either not installed\
              or could not be imported due to {e}."
        )

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
    # Wrap the quantum Backend in a DifferentiableBackend if a diff_mode is passed.
    if diff_mode is not None:
        try:
            engine_name = Engine(backend_inst.engine)
        except ValueError:
            raise NotImplementedError(
                f"The requested engine '{backend_inst.engine}' is not implemented."
            )
        try:
            diff_backend_cls = available_engines()[engine_name]
            backend_inst = diff_backend_cls(backend=backend_inst, diff_mode=DiffMode(diff_mode))  # type: ignore[arg-type]
        except Exception as e:
            raise ImportError(
                f"The requested engine '{engine_name}' is either not installed\
                or could not be imported due to {e}."
            )
    return backend_inst


def config_factory(name: BackendName | str, config: dict) -> BackendConfiguration:
    backends = available_backends()

    try:
        BackendCls = backends[BackendName(name)]
    except KeyError:
        raise NotImplementedError(f"The requested backend '{name}' is not implemented!")

    BackendConfigCls = type(BackendCls.default_configuration())
    return BackendConfigCls(**config)  # type: ignore[no-any-return]
