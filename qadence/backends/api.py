from __future__ import annotations

from qadence.backend import Backend, BackendConfiguration
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.extensions import (
    BackendNotFoundError,
    ConfigNotFoundError,
    EngineNotFoundError,
    import_backend,
    import_config,
    import_engine,
    set_backend_config,
)
from qadence.logger import get_logger
from qadence.types import BackendName, DiffMode

__all__ = ["backend_factory", "config_factory"]
logger = get_logger(__name__)


def backend_factory(
    backend: BackendName | str,
    diff_mode: DiffMode | str | None = None,
    configuration: BackendConfiguration | dict | None = None,
) -> Backend | DifferentiableBackend:
    backend_inst: Backend | DifferentiableBackend
    try:
        BackendCls = import_backend(backend)
        default_config = BackendCls.default_configuration()
        if configuration is None:
            configuration = default_config
        elif isinstance(configuration, dict):
            configuration = config_factory(backend, configuration)
        else:
            # NOTE: types have to match exactly, hence we use `type`
            if not isinstance(configuration, type(BackendCls.default_configuration())):
                expected_cfg = BackendCls.default_configuration()
                raise ValueError(
                    f"Given config class '{type(configuration)}' does not match the backend",
                    f" class: '{BackendCls}'. Expected: '{type(expected_cfg)}.'",
                )

        # Instantiate the backend
        backend_inst = BackendCls(  # type: ignore[operator]
            config=configuration
            if configuration is not None
            else BackendCls.default_configuration()
        )
        set_backend_config(backend_inst, diff_mode)
        # Wrap the quantum Backend in a DifferentiableBackend if a diff_mode is passed.
        if diff_mode is not None:
            diff_backend_cls = import_engine(backend_inst.engine)
            backend_inst = diff_backend_cls(backend=backend_inst, diff_mode=DiffMode(diff_mode))  # type: ignore[operator]
        return backend_inst
    except (BackendNotFoundError, EngineNotFoundError, ConfigNotFoundError) as e:
        logger.error(e.msg)
        raise e


def config_factory(backend_name: BackendName | str, config: dict) -> BackendConfiguration:
    cfg: BackendConfiguration
    try:
        BackendConfigCls = import_config(backend_name)
        cfg = BackendConfigCls(**config)  # type: ignore[operator]
    except ConfigNotFoundError as e:
        logger.error(e.msg)
        raise e
    return cfg
