from pydantic import BaseModel, validator

from qadence.types import BackendName, DiffMode, StrEnum



class ExperimentComposer(BaseModel):
    BACKEND: BackendName
    DIFFMODE: DiffMode

    @validator("BACKEND")
    @classmethod
    def validate_backend(cls, value: StrEnum) -> StrEnum:
        return BackendName(value)

    @validator("DIFFMODE")
    @classmethod
    def validate_diffmode(cls, value: StrEnum, values: dict) -> StrEnum:
        validated_diffmode = DiffMode(value)
        if validated_diffmode == DiffMode.AD:
            if (backend := values['BACKEND']) in [BackendName.BRAKET, BackendName.PULSER]:
                raise ValueError(f"Backend {backend} does not support diff_mode {validated_diffmode}.")
        return validated_diffmode
