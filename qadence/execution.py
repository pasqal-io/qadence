from __future__ import annotations

from collections import Counter
from functools import singledispatch
from typing import Any, Union

from torch import Tensor, concat, no_grad

from qadence.backend import BackendConfiguration
from qadence.backends.api import backend_factory
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.qubit_support import QubitSupport
from qadence.register import Register
from qadence.types import BackendName, DiffMode, Endianness

# Modules to be automatically added to the qadence namespace
__all__ = ["run", "sample", "expectation"]


def _n_qubits_block(block: AbstractBlock) -> int:
    if isinstance(block.qubit_support, QubitSupport) and block.qubit_support.is_global:
        raise ValueError(
            "Unable to determine the number of qubits for a block with global qubit support.\
             Please supply the number of qubits explicitly: QuantumCircuit(n_qubits, block)"
        )
    else:
        return block.n_qubits


@singledispatch
def run(
    x: Union[QuantumCircuit, AbstractBlock, Register, int],
    *args: Any,
    values: Union[dict, None] = None,
    state: Tensor = None,
    backend: BackendName = BackendName.PYQTORCH,
    endianness: Endianness = Endianness.BIG,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> Tensor:
    """Convenience wrapper for the `QuantumModel.run` method.

     This is a
    `functools.singledispatch`ed function so it can be called with a number of different arguments.
    See the examples of the [`expectation`][qadence.execution.expectation] function. This function
    works exactly the same.

    Arguments:
        x: Circuit, block, or (register+block) to run.
        values: User-facing parameter dict.
        state: Initial state.
        backend: Name of the backend to run on.
        endianness: The target device endianness.
        configuration: The backend configuration.

    Returns:
        A wavefunction
    """
    raise ValueError(f"Cannot run {type(x)}")


@run.register
def _(
    circuit: QuantumCircuit,
    values: Union[dict, None] = None,
    state: Tensor = None,
    backend: BackendName = BackendName.PYQTORCH,
    endianness: Endianness = Endianness.BIG,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> Tensor:
    diff_mode = None
    if backend == BackendName.PYQTORCH:
        diff_mode = DiffMode.AD
    bknd = backend_factory(backend, diff_mode=diff_mode, configuration=configuration)
    conv = bknd.convert(circuit)
    with no_grad():
        return bknd.run(
            circuit=conv.circuit,
            param_values=conv.embedding_fn(conv.params, values or dict()),
            state=state,
            endianness=endianness,
        )


@run.register
def _(register: Register, block: AbstractBlock, **kwargs: Any) -> Tensor:
    return run(QuantumCircuit(register, block), **kwargs)


@run.register
def _(n_qubits: int, block: AbstractBlock, **kwargs: Any) -> Tensor:
    return run(Register(n_qubits), block, **kwargs)


@run.register
def _(block: AbstractBlock, **kwargs: Any) -> Tensor:
    n_qubits = _n_qubits_block(block)
    return run(Register(n_qubits), block, **kwargs)


@run.register
def _(circs: list, **kwargs: Any) -> Tensor:  # type: ignore[misc]
    results = ()
    for c in circs:
        results += run(c, **kwargs)  # type:ignore[assignment]
    return concat(results, dim=0)


@singledispatch
def sample(
    x: Union[QuantumCircuit, AbstractBlock, Register, int],
    *args: Any,
    values: Union[dict, None] = None,
    state: Union[Tensor, None] = None,
    n_shots: int = 100,
    backend: BackendName = BackendName.PYQTORCH,
    endianness: Endianness = Endianness.BIG,
    noise: Union[NoiseHandler, None] = None,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> list[Counter]:
    """Convenience wrapper for the `QuantumModel.sample` method.

    Arguments:
        x: Circuit, block, or (register+block) to run.
        values: User-facing parameter dict.
        state: Initial state.
        n_shots: Number of shots per element in the batch.
        backend: Name of the backend to run on.
        endianness: The target device endianness.
        noise: The noise model to use if any.
        configuration: The backend configuration.

    Returns:
        A list of Counter instances with the sample results
    """
    raise ValueError(f"Cannot sample from {type(x)}")


@sample.register
def _(
    circuit: QuantumCircuit,
    values: Union[dict, None] = None,
    state: Union[Tensor, None] = None,
    n_shots: int = 100,
    backend: BackendName = BackendName.PYQTORCH,
    noise: Union[NoiseHandler, None] = None,
    endianness: Endianness = Endianness.BIG,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> list[Counter]:
    diff_mode = None
    if backend == BackendName.PYQTORCH:
        diff_mode = DiffMode.AD
    bknd = backend_factory(backend, diff_mode=diff_mode, configuration=configuration)
    conv = bknd.convert(circuit)
    return bknd.sample(
        circuit=conv.circuit,
        param_values=conv.embedding_fn(conv.params, values or dict()),
        n_shots=n_shots,
        state=state,
        noise=noise,
        endianness=endianness,
    )


@sample.register
def _(register: Register, block: AbstractBlock, **kwargs: Any) -> Tensor:
    return sample(QuantumCircuit(register, block), **kwargs)


@sample.register
def _(n_qubits: int, block: AbstractBlock, **kwargs: Any) -> Tensor:
    return sample(Register(n_qubits), block, **kwargs)


@sample.register
def _(block: AbstractBlock, **kwargs: Any) -> Tensor:
    n_qubits = _n_qubits_block(block)
    return sample(Register(n_qubits), block, **kwargs)


@singledispatch
def expectation(
    x: Union[QuantumCircuit, AbstractBlock, Register, int],
    observable: Union[list[AbstractBlock], AbstractBlock],
    values: Union[dict, None] = None,
    state: Tensor = None,
    backend: BackendName = BackendName.PYQTORCH,
    diff_mode: Union[DiffMode, str, None] = None,
    noise: Union[NoiseHandler, None] = None,
    endianness: Endianness = Endianness.BIG,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> Tensor:
    """Convenience wrapper for the `QuantumModel.expectation` method.

    Arguments:
        x: Circuit, block, or (register+block) to run.
        observable: Observable(s) w.r.t. which the expectation is computed.
        values: User-facing parameter dict.
        state: Initial state.
        backend: Name of the backend to run on.
        diff_mode: Which differentiation mode to use.
        endianness: The target device endianness.
        configuration: The backend configuration.

    Returns:
        A wavefunction

    ```python exec="on" source="material-block"
    from qadence import RX, Z, Register, QuantumCircuit, expectation

    reg = Register(1)
    block = RX(0, 0.5)
    observable = Z(0)
    circ = QuantumCircuit(reg, block)

    # You can compute the expectation for a
    # QuantumCircuit with a given observable.
    expectation(circ, observable)

    # You can also use only a block.
    # In this case the register is constructed automatically to
    # Register.line(block.n_qubits)
    expectation(block, observable)

    # Or a register and block
    expectation(reg, block, observable)
    ```
    """

    raise ValueError(f"Cannot execute {type(x)}")


@expectation.register
def _(
    circuit: QuantumCircuit,
    observable: Union[list[AbstractBlock], AbstractBlock],
    values: Union[dict, None] = None,
    state: Tensor = None,
    backend: BackendName = BackendName.PYQTORCH,
    diff_mode: Union[DiffMode, str, None] = None,
    measurement: Measurements = None,
    noise: Union[NoiseHandler, None] = None,
    mitigation: Mitigations = None,
    endianness: Endianness = Endianness.BIG,
    configuration: Union[BackendConfiguration, dict, None] = None,
) -> Tensor:
    observable = observable if isinstance(observable, list) else [observable]
    if backend == BackendName.PYQTORCH:
        diff_mode = DiffMode.AD
    bknd = backend_factory(backend, diff_mode=diff_mode, configuration=configuration)
    conv = bknd.convert(circuit, observable)

    def _expectation() -> Tensor:
        return bknd.expectation(
            circuit=conv.circuit,
            observable=conv.observable,  # type: ignore[arg-type]
            param_values=conv.embedding_fn(conv.params, values or dict()),
            state=state,
            measurement=measurement,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

    # Do not compute gradients if no diff_mode is provided.
    if diff_mode is None:
        with no_grad():
            return _expectation()
    else:
        return _expectation()


@expectation.register
def _(
    register: Register,
    block: AbstractBlock,
    observable: Union[list[AbstractBlock], AbstractBlock],
    **kwargs: Any,
) -> Tensor:
    return expectation(QuantumCircuit(register, block), observable, **kwargs)


@expectation.register
def _(
    n_qubits: int,
    block: AbstractBlock,
    observable: Union[list[AbstractBlock], AbstractBlock],
    **kwargs: Any,
) -> Tensor:
    reg = Register(n_qubits)
    return expectation(QuantumCircuit(reg, block), observable, **kwargs)


@expectation.register
def _(
    block: AbstractBlock, observable: Union[list[AbstractBlock], AbstractBlock], **kwargs: Any
) -> Tensor:
    n_qubits = _n_qubits_block(block)
    return expectation(QuantumCircuit(Register(n_qubits), block), observable, **kwargs)
