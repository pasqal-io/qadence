from __future__ import annotations

from typing import Any

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit as PyQCircuit
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import overlap, param_dict
from torch import Tensor
from torch.autograd import Function

from qadence.backends.pyqtorch.convert_ops import ScalePyQOperation


class AdjointExpectation(Function):
    @staticmethod
    def forward(
        ctx: Any,
        circuit: PyQCircuit,
        observable: PyQCircuit,
        state: Tensor,
        param_names: list[str],
        *param_values: Tensor,
    ) -> Tensor:
        ctx.circuit = circuit
        ctx.observable = observable
        ctx.param_names = param_names
        values = param_dict(param_names, param_values)
        ctx.out_state = circuit.run(state, values)
        ctx.projected_state = observable.run(ctx.out_state, values)
        ctx.save_for_backward(*param_values)
        return overlap(ctx.out_state, ctx.projected_state)

    # TODO filter observable params so the length of param_values is the same as grads
    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        def _circuit_backward(ctx: Any, circuit: PyQCircuit = None) -> Any:
            if circuit is None:
                circuit = ctx.circuit
            param_values = ctx.saved_tensors
            values = param_dict(ctx.param_names, param_values)
            grads: list = []
            for op in circuit.reverse():
                if isinstance(op, ScalePyQOperation):
                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
                    if values[op.param_name].requires_grad:
                        grads += [grad_out * 2 * -values[op.param_name]]
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
                elif isinstance(op, PyQCircuit):
                    grads += [grad_out * g for g in _circuit_backward(ctx, op)]
                elif isinstance(op, (Primitive, Parametric)):
                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
                    if isinstance(op, Parametric) and values[op.param_name].requires_grad:
                        mu = apply_operator(ctx.out_state, op.jacobian(values), op.qubit_support)
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
                else:
                    raise TypeError(
                        f"AdjointExpectation does not support a backward passe for type {type(op)}."
                    )
            return grads

        grads = _circuit_backward(ctx)

        return (None, None, None, None, *grads)
