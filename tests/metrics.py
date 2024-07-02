from __future__ import annotations

from qadence.types import BackendName

ATOL_64 = 1e-14  # 64 bit precision
ATOL_32 = 1e-07  # 32 bit precision
ATOL_E6 = 1e-06  # some tests do not pass ATOL_32; to fix

RTOL_LOW_ACCEPTANCE = 1.0e-2
RTOL_MIDDLE_ACCEPTANCE = 5.0e-2
RTOL_HIGH_ACCEPTANCE = 1.0e-1

LOW_ACCEPTANCE = 2.0e-2
MIDDLE_ACCEPTANCE = 6.0e-2
HIGH_ACCEPTANCE = 0.5

JS_ACCEPTANCE = 7.5e-2
PSR_ACCEPTANCE = 1e-5
GPSR_ACCEPTANCE = 1e-1
ADJOINT_ACCEPTANCE = ATOL_E6
PULSER_GPSR_ACCEPTANCE = 6.0e-2
ATOL_DICT = {
    BackendName.PYQTORCH: ATOL_32,
    BackendName.HORQRUX: ATOL_32,
    BackendName.PULSER: 1e-02,
    BackendName.BRAKET: 1e-02,
}
MAX_COUNT_DIFF = 20
SMALL_SPACING = 7.0
LARGE_SPACING = 30.0
DIGITAL_DECOMP_ACCEPTANCE_HIGH = 1e-2
DIGITAL_DECOMP_ACCEPTANCE_LOW = 1e-3
JAX_CONVERSION_ATOL = ATOL_E6
MEASUREMENT_ATOL = 2.0e-1
TIME_DEPENDENT_GENERATOR_ATOL = 1e-3
