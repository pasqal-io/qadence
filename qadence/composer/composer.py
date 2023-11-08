from pydantic import BaseModel, validator

from qadence.types import BackendName, DiffMode



class ExperimentComposer(BaseModel):
    BACKEND: BackendName
    DIFFMODE: DiffMode

    @validator("BACKEND")
    @classmethod
    def validate_backend(cls, value: BackendName) -> BackendName:
        return BackendName(value)

    @validator("DIFFMODE")
    @classmethod
    def validate_diffmode(cls, value: DiffMode, values: dict) -> DiffMode:
        validated_diffmode = DiffMode(value)
        if validated_diffmode == DiffMode.AD and (backend := values['BACKEND']) != BackendName.PYQTORCH:
            raise ValueError(f"Backend {backend} does not support diff_mode {validated_diffmode}.")
        return validated_diffmode
