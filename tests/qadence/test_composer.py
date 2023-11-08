import pytest

from qadence.composer.composer import ExperimentComposer
from qadence.types import BackendName, DiffMode


@pytest.mark.parametrize(
    "backend",
    BackendName.list(),
)
def test_composer_validation_for_all_backends_and_gpsr(backend: BackendName, diff_mode: DiffMode = DiffMode.GPSR):
    assert ExperimentComposer(BACKEND=backend, DIFFMODE=diff_mode)

@pytest.mark.parametrize(
    "backend",
    [BackendName.BRAKET, BackendName.PULSER],
)
def test_composer_raise_error_for_backend_not_supporting_ad(backend: BackendName, diff_mode: DiffMode = DiffMode.AD):
    with pytest.raises(ValueError):
        ExperimentComposer(BACKEND=backend, DIFFMODE=diff_mode)
