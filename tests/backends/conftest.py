from __future__ import annotations
from typing import Any

import pytest
from pytest import fixture  # type: ignore

from qadence import Z, X

list_obs = [[Z(0)], [Z(0), X(1), "xobs" * Z(0)]]


@pytest.fixture(params=list_obs)
def adjoint_observables(
    request: pytest.Fixture,
) -> Any:
    return request.param