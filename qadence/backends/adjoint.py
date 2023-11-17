from __future__ import annotations

from typing import Any

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit as PyQCircuit
from pyqtorch.parametric import Parametric as PyQParametric
from pyqtorch.primitive import Primitive as PyQPrimitive
from pyqtorch.utils import overlap, param_dict
from torch import Tensor, no_grad, tensor
from torch.autograd import Function
from torch.nn import Module

from qadence.backends.pyqtorch.convert_ops import PyQHamiltonianEvolution, ScalePyQOperation
from qadence.blocks.abstract import AbstractBlock


class AdjointExpectation(Function):
    """
    The adjoint differentiation method (https://arxiv.org/pdf/2009.02823.pdf).

    is able to perform a backward pass in O(P) time and maintaining
    atmost 3 states where P is the number of parameters in a variational circuit.

    Pseudo-code of the algorithm:

    c: a variational circuit
    c.gates = gate0,gate1,..gateN, where N denotes the last gate in c
    o: a observable
    state: an initial state

    1. Forward pass.
    for gate in c.gates: # We apply gate0, gate1 to gateN.
        state = gate(state)

    projected_state = o(state)  # Apply the observable to the state.
    expval = overlap(state, projected_state)  # Compute the expected value.

    2. Backward pass:
    grads = []
    for gate in reversed(c.gates): # Iterate through c.gates in "reverse", so gateN, gateN-1 etc.
        state = dagger(gate)(state)  # 'Undo' the gate by applying its dagger.
        if gate is Parametric:
            mu = jacobian(gate)(state) # Compute the jacobian of the gate w.r.t its parameter.
            grads.append(2 * overlap(mu, projected_state) # Compute the gradient.
        projected_state = dagger(gate)(projected_state)  # 'Undo' the gate from the projected_state.

    Current Limitations:

    (1) The adjoint method is only available in the pyqtorch backend.
    (2) Parametric observables are not supported.
    (3) Multiple observables are not supported.
    (4) Higher order derivatives are not natively supported.
    (5) Only expectation values can be differentiated, not wave functions.
    """

    @staticmethod
    @no_grad()
    def forward(
        ctx: Any,
        circuit: PyQCircuit,
        observable: PyQCircuit,
        state: Tensor,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        for param in param_values:
            param = param.detach()
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.param_names = param_names
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return overlap(ctx.out_state, ctx.projected_state)

    @staticmethod
    @no_grad()
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        param_values = ctx.saved_tensors
        values = param_dict(ctx.param_names, param_values)

        def _apply_adjoint(ctx: Any, op: Module) -> list:
            grads: list = []
            if isinstance(op, PyQHamiltonianEvolution):
                generator = op.block.generator
                time_param = values[op.param_names[0]]
                ctx.out_state = apply_operator(ctx.out_state, op.dagger(values), op.qubit_support)
                # A HamEvo can have a parametrized (1) time evolution and/or (2) generator.
                if (
                    isinstance(generator, AbstractBlock)
                    and generator.is_parametric
                    and values[op.param_names[1]].requires_grad
                ):
                    # If the generator contains a trainable parameter, we compute its gradient.
                    mu = apply_operator(
                        ctx.out_state, op.jacobian_generator(values), op.qubit_support
                    )
                    grads.append(2 * overlap(ctx.projected_state, mu))
                if time_param.requires_grad:
                    # If the time evolution is trainable, we compute its gradient.
                    mu = apply_operator(ctx.out_state, op.jacobian_time(values), op.qubit_support)
                    grads.append(2 * overlap(ctx.projected_state, mu))
                ctx.projected_state = apply_operator(
                    ctx.projected_state, op.dagger(values), op.qubit_support
                )
            elif isinstance(op, ScalePyQOperation):
                ctx.out_state = apply_operator(ctx.out_state, op.dagger(values), op.qubit_support)
                scaled_pyq_op = op.operations[0]
                if (
                    isinstance(scaled_pyq_op, PyQParametric)
                    and values[scaled_pyq_op.param_name].requires_grad
                ):
                    mu = apply_operator(
                        ctx.out_state,
                        scaled_pyq_op.jacobian(values),
                        scaled_pyq_op.qubit_support,
                    )
                    grads.append(2 * overlap(ctx.projected_state, mu))

                if values[op.param_name].requires_grad:
                    grads.append(2 * -values[op.param_name])
                ctx.projected_state = apply_operator(
                    ctx.projected_state, op.dagger(values), op.qubit_support
                )
            elif isinstance(op, PyQCircuit):
                grads = [g for sub_op in op.reverse() for g in _apply_adjoint(ctx, sub_op)]
            elif isinstance(op, PyQPrimitive):
                ctx.out_state = apply_operator(ctx.out_state, op.dagger(values), op.qubit_support)
                if isinstance(op, PyQParametric) and values[op.param_name].requires_grad:
                    mu = apply_operator(
                        ctx.out_state,
                        op.jacobian(values),
                        op.qubit_support,
                    )
                    grads.append(2 * overlap(ctx.projected_state, mu))
                ctx.projected_state = apply_operator(
                    ctx.projected_state, op.dagger(values), op.qubit_support
                )
            else:
                raise TypeError(
                    f"AdjointExpectation does not support a backward pass for type {type(op)}."
                )

            return grads

        grads = list(
            reversed(
                [grad_out * g for op in ctx.circuit.reverse() for g in _apply_adjoint(ctx, op)]
            )
        )
        num_grads = len(grads)
        num_params = len(ctx.saved_tensors)
        diff = num_params - num_grads
        grads = grads + [tensor([0]) for _ in range(diff)]
        # Set observable grads to 0
        ctx.save_for_backward(*grads)
        return (None, None, None, None, *grads)
