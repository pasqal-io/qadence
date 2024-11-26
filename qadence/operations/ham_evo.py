from __future__ import annotations

from copy import deepcopy
from functools import cached_property
from logging import getLogger
from typing import Any, Union

import numpy as np
import sympy
import torch
from torch import Tensor

from qadence.blocks import AbstractBlock, TimeEvolutionBlock
from qadence.blocks.block_to_tensor import block_to_tensor
from qadence.blocks.utils import (
    add,  # noqa
    block_is_commuting_hamiltonian,
    block_is_qubit_hamiltonian,
    chain,
    expressions,
)
from qadence.decompose import lie_trotter_suzuki
from qadence.parameters import (
    Parameter,
    ParamMap,
    evaluate,
    extract_original_param_entry,
)
from qadence.types import LTSOrder, OpName, TGenerator, TParameter

logger = getLogger(__name__)


class HamEvo(TimeEvolutionBlock):
    """
    The Hamiltonian evolution operator U(t).

    For time-independent Hamiltonians the solution is exact:

        U(t) = exp(-iGt)

    where G represents an Hermitian generator, or Hamiltonian and t represents the
    time parameter. For time-dependent Hamiltonians, the solution is obtained by
    numerical integration of the Schrodinger equation.

    Arguments:
        generator: Hamiltonian generator, either symbolic as an AbstractBlock,
            or as a torch.Tensor or numpy.ndarray.
        parameter: The time parameter for evolution operator. For the time-independent
            case, it represents the actual value for which the evolution will be
            evaluated. For the time-dependent case, it should be an instance of
            TimeParameter to signal the solver the variable that will be integrated over.
        qubit_support: The qubits on which the evolution will be performed on. Only
            required for generators that are not a composition of blocks.
        duration: (optional) duration of the evolution in case of time-dependent
            generator. By default, a FeatureParameter with tag "duration" will
            be initialized, and the value will then be required in the values dict.
        noise_operators: (optional) the list of jump operators to use when using
            a shrodinger solver, allowing to perform noisy simulations.

    Examples:

    ```python exec="on" source="material-block" result="json"
    from qadence import X, HamEvo, PI, add, run
    from qadence import FeatureParameter, TimeParameter
    import torch

    n_qubits = 3

    # Hamiltonian as a block composition
    hamiltonian = add(X(i) for i in range(n_qubits))
    hevo = HamEvo(hamiltonian, parameter=torch.rand(2))
    state = run(hevo)

    # Hamiltonian as a random matrix
    hamiltonian = torch.rand(2, 2, dtype=torch.complex128)
    hevo = HamEvo(hamiltonian, parameter=torch.rand(2), qubit_support=(0,))
    state = run(hevo)

    # Time-dependent Hamiltonian
    t = TimeParameter("t")
    hamiltonian = t * add(X(i) for i in range(n_qubits))
    hevo = HamEvo(hamiltonian, parameter=t)
    state = run(hevo, values = {"duration": torch.tensor(1.0)})

    # Adding noise operators
    noise_ops = [X(0)]
    hevo = HamEvo(hamiltonian, parameter=t, noise_operators=noise_ops)
    ```
    """

    name = OpName.HAMEVO
    draw_generator: bool = False

    def __init__(
        self,
        generator: Union[TGenerator, AbstractBlock],
        parameter: TParameter,
        qubit_support: tuple[int, ...] = None,
        duration: TParameter | None = None,
        noise_operators: list[AbstractBlock] = list(),
    ):
        params = {}
        if qubit_support is None and not isinstance(generator, AbstractBlock):
            raise ValueError("You have to supply a qubit support for non-block generators.")
        super().__init__(qubit_support if qubit_support else generator.qubit_support)
        if isinstance(generator, AbstractBlock):
            qubit_support = generator.qubit_support
            if generator.is_parametric:
                params = {str(e): e for e in expressions(generator)}
            if generator.is_time_dependent:
                if isinstance(duration, str):
                    duration = Parameter(duration, trainable=False)
                elif duration is None:
                    duration = Parameter("duration", trainable=False)
            if not generator.is_time_dependent and duration is not None:
                raise TypeError(
                    "Duration argument is only supported for time-dependent generators."
                )
        elif isinstance(generator, torch.Tensor):
            if duration is not None:
                raise TypeError(
                    "Duration argument is only supported for time-dependent generators."
                )
            msg = "Please provide a square generator."
            if len(generator.shape) == 2:
                assert generator.shape[0] == generator.shape[1], msg
            elif len(generator.shape) == 3:
                assert generator.shape[1] == generator.shape[2], msg
                assert generator.shape[0] == 1, "Qadence doesnt support batched generators."
            else:
                raise TypeError(
                    "Only 2D or 3D generators are supported.\
                                In case of a 3D generator, the batch dim\
                                is expected to be at dim 0."
                )
            params = {str(generator.__hash__()): generator}
        elif isinstance(generator, (sympy.Basic, sympy.Array)):
            if duration is not None:
                raise TypeError(
                    "Duration argument is only supported for time-dependent generators."
                )
            params = {str(generator): generator}
        else:
            raise TypeError(
                f"Generator of type {type(generator)} not supported.\
                            If you're using a numpy.ndarray, please cast it to a torch tensor."
            )
        if duration is not None:
            params = {"duration": Parameter(duration), **params}
        params = {"parameter": Parameter(parameter), **params}
        self.parameters = ParamMap(**params)
        self.time_param = parameter
        self.generator = generator
        self.duration = duration

        if len(noise_operators) > 0:
            if not all(
                [
                    len(set(op.qubit_support + self.qubit_support) - set(self.qubit_support)) == 0
                    for op in noise_operators
                ]
            ):
                raise ValueError(
                    "Noise operators should be defined"
                    " over the same or a subset of the qubit support"
                )
            if True in [op.is_parametric for op in noise_operators]:
                raise ValueError("Parametric operators are not supported")
        self.noise_operators = noise_operators

    @classmethod
    def num_parameters(cls) -> int:
        return 2

    @cached_property
    def eigenvalues_generator(
        self, max_num_evals: int | None = None, max_num_gaps: int | None = None
    ) -> Tensor:
        from qadence.utils import eigenvalues

        if isinstance(self.generator, AbstractBlock):
            generator_tensor = block_to_tensor(self.generator)
        elif isinstance(self.generator, Tensor):
            generator_tensor = self.generator
        return eigenvalues(generator_tensor, max_num_evals, max_num_gaps)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.exp(
            -1j * evaluate(self.parameters.parameter, as_torch=True) * self.eigenvalues_generator
        )

    @property
    def n_qubits(self) -> int:
        if isinstance(self.generator, Tensor):
            n_qubits = int(np.log2(self.generator.shape[-1]))
        else:
            n_qubits = self.generator.n_qubits  # type: ignore [union-attr]

        return n_qubits

    def dagger(self) -> Any:
        p = list(self.parameters.expressions())[0]
        return HamEvo(deepcopy(self.generator), -extract_original_param_entry(p))

    def digital_decomposition(self, approximation: LTSOrder = LTSOrder.ST4) -> AbstractBlock:
        """Decompose the Hamiltonian evolution into digital gates.

        Args:
            approximation (str, optional): Choose the type of decomposition. Defaults to "st4".
                Available types are:
                * 'basic' = apply first-order Trotter formula and decompose each term of
                    the exponential into digital gates. It is exact only if applied to an
                    operator whose terms are mutually commuting.
                * 'st2' = Trotter-Suzuki 2nd order formula for approximating non-commuting
                    Hamiltonians.
                * 'st4' = Trotter-Suzuki 4th order formula for approximating non-commuting
                    Hamiltonians.

        Returns:
            AbstractBlock: a block with the digital decomposition
        """

        # psi(t) = exp(-i * H * t * psi0)
        # psi(t) = exp(-i * lambda * t * psi0)
        # H = sum(Paulin) + sum(Pauli1*Pauli2)
        logger.info("Quantum simulation of the time-independent Schrödinger equation.")

        blocks = []

        # how to change the type/dict to enum effectively

        # when there is a term including non-commuting matrices use st2 or st4

        # 1) should check that the given generator respects the constraints
        # single-qubit gates

        assert isinstance(
            self.generator, AbstractBlock
        ), "Only a generator represented as a block can be decomposed"

        if block_is_qubit_hamiltonian(self.generator):
            try:
                block_is_commuting_hamiltonian(self.generator)
                approximation = LTSOrder.BASIC  # use the simpler approach if the H is commuting
            except TypeError:
                logger.warning(
                    """Non-commuting terms in the Pauli operator.
                    The Suzuki-Trotter approximation is applied."""
                )

            blocks.extend(
                lie_trotter_suzuki(
                    block=self.generator,
                    parameter=self.parameters.parameter,
                    order=LTSOrder[approximation],
                )
            )

            # 2) return an AbstractBlock instance with the set of gates
            # resulting from the decomposition

            return chain(*blocks)
        else:
            raise NotImplementedError(
                "The current digital decomposition can be applied only to Pauli Hamiltonians."
            )

    def __matmul__(self, other: AbstractBlock) -> AbstractBlock:
        return super().__matmul__(other)
