from __future__ import annotations

from typing import Any

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit as PyQCircuit
from pyqtorch.parametric import Parametric
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
                    return [
                        param_values[op.param_name]
                    ]  # the gradient of a scaleblock param ist just itself
                if isinstance(op, PyQCircuit):
                    grads += _circuit_backward(ctx, op)
                else:
                    ctx.out_state = apply_operator(
                        ctx.out_state, op.dagger(values), op.qubit_support
                    )
                    if isinstance(op, Parametric):
                        mu = apply_operator(ctx.out_state, op.jacobian(values), op.qubit_support)
                        grads = [grad_out * 2 * overlap(ctx.projected_state, mu)] + grads
                    ctx.projected_state = apply_operator(
                        ctx.projected_state, op.dagger(values), op.qubit_support
                    )
            return grads

        return (None, None, None, None, *_circuit_backward(ctx))
