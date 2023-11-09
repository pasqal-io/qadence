from __future__ import annotations

from pydantic import BaseModel, validator

from qadence.types import BackendName, DiffMode


class ExperimentComposer(BaseModel):
    BACKEND: BackendName
    DIFF_MODE: DiffMode

    @validator("BACKEND")
    @classmethod
    def validate_backend(cls, value: BackendName) -> BackendName:
        return BackendName(value)

    @validator("DIFF_MODE")
    @classmethod
    def validate_diffmode(cls, value: DiffMode, values: dict) -> DiffMode:
        validated_diffmode = DiffMode(value)
        if (
            validated_diffmode == DiffMode.AD
            and (backend := values["BACKEND"]) != BackendName.PYQTORCH
        ):
            raise ValueError(
                f"Backend {backend} does not support diff_mode {validated_diffmode}."
                f"Please choose {DiffMode.GPSR} instead."
            )
        return validated_diffmode
