from __future__ import annotations

from typing import Any

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit as PyQCircuit
from pyqtorch.parametric import Parametric as PyQParametric
from pyqtorch.primitive import Primitive as PyQPrimitive
from pyqtorch.utils import overlap, param_dict
from torch import Tensor, no_grad, tensor
from torch.autograd import Function

from qadence.backends.pyqtorch.convert_ops import PyQHamiltonianEvolution, ScalePyQOperation
from qadence.blocks.abstract import AbstractBlock


class AdjointExpectation(Function):
    @no_grad()
    @staticmethod
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

    @no_grad()
    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        def _apply_adjoint(ctx: Any, circuit: PyQCircuit, grad_out: Tensor = tensor([1])) -> Any:
            param_values = ctx.saved_tensors
            values = param_dict(ctx.param_names, param_values)
            grads: list = []
            for op in circuit.reverse():
                if isinstance(op, (PyQHamiltonianEvolution)):
                    generator = op.block.generator
                    time_param = values[op.param_names[0]]

                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
                    if (
                        isinstance(generator, AbstractBlock)
                        and generator.is_parametric
                        and values[op.param_names[1]].requires_grad
                    ):
                        mu = apply_operator(
                            ctx.out_state, op.jacobian_generator(values), op.qubit_support
                        )
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads
                    elif time_param.requires_grad:
                        mu = apply_operator(
                            ctx.out_state, op.jacobian_time(values), op.qubit_support
                        )
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
                elif isinstance(op, ScalePyQOperation):
                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
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
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads

                    if values[op.param_name].requires_grad:
                        grads += [grad_out * 2 * -values[op.param_name]]
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
                elif isinstance(op, PyQCircuit):
                    grads += [grad_out * g for g in _apply_adjoint(ctx, op)]
                elif isinstance(op, (PyQPrimitive)):
                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
                    if isinstance(op, (PyQParametric)) and values[op.param_name].requires_grad:
                        mu = apply_operator(ctx.out_state, op.jacobian(values), op.qubit_support)
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
                else:
                    raise TypeError(
                        f"AdjointExpectation does not support a backward pass for type {type(op)}."
                    )
            return grads

        grads = _apply_adjoint(ctx, ctx.circuit, grad_out=grad_out)
        num_grads = len(grads)
        num_params = len(ctx.saved_tensors)
        diff = num_params - num_grads
        grads = grads + [tensor([0]) for _ in range(diff)]
        # Set observable grads to 0
        return (None, None, None, None, *grads)
