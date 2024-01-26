import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn

from qadence import QuantumCircuit
from qadence.qinfo_tools.qfi import (
    get_quantum_fisher,
    get_quantum_fisher_spsa,
)


class QuantumNaturalGradient(Optimizer):
    """Implements the Quantum Natural Gradient Algorithm"""

    def __init__(
        self,
        params,
        circuit: QuantumCircuit = required,
        lr: float = required,
        iteration_number=0,
        approximation="exact",
        epsilon=10e-2,
        beta=10e-3,
    ):
        # if not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        self.iteration_number = iteration_number
        self.prev_qfi_estimator = 0

        defaults = dict(
            circuit=circuit,
            lr=lr,
            approximation=approximation,
            epsilon=epsilon,
            beta=beta,
        )
        super(QuantumNaturalGradient, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["approximation"] == "exact":
                data_vec = torch.Tensor([v for v in group["params"] if (v.requires_grad)])
                grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])
                metric_tensor = (1 / 4) * get_quantum_fisher(group["circuit"]) + group[
                    "beta"
                ] * torch.eye(len(data_vec))
                metric_tensor_inv = torch.inverse(metric_tensor)
                transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                it = iter(range(len(data_vec)))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # with torch.no_grad():
                    p.data.add_(transf_grad[next(it)], alpha=-group["lr"])

            elif group["approximation"] == "spsa":
                data_vec = torch.Tensor([v for v in group["params"] if (v.requires_grad)])
                grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])
                qfi_estimator, qfi_mat_positive_sd = get_quantum_fisher_spsa(
                    circuit=group["circuit"],
                    k=self.iteration_number,
                    var_values=group["params"],
                    previous_qfi_estimator=self.prev_qfi_estimator,
                    epsilon=group["epsilon"],
                    beta=group["beta"],
                )
                metric_tensor = (1 / 4) * qfi_mat_positive_sd
                metric_tensor_inv = torch.inverse(metric_tensor)
                transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

                it = iter(range(len(data_vec)))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # with torch.no_grad():
                    p.data.add_(transf_grad[next(it)], alpha=-group["lr"])

                self.iteration_number += 1
                self.prev_qfi_estimator = qfi_estimator

            else:
                raise NotImplementedError(
                    "Non valid approximation, please choose 'exact' or 'spsa'."
                )

        return loss


# class QuantumNaturalGradient(Optimizer):
#     """Implements the Quantum Natural Gradient Algorithm"""

#     def __init__(
#         self,
#         params,
#         circuit: QuantumCircuit = required,
#         lr: float = required,
#         approximation="exact",
#         epsilon=10e-3,
#         beta=10e-2,
#     ):
#         if not 0.0 <= lr:
#             raise ValueError(f"Invalid learning rate: {lr}")

#         defaults = dict(
#             lr=lr,
#             circuit=circuit,
#             approximation=approximation,
#             epsilon=epsilon,
#             beta=beta,
#             use_sgd=False,
#         )
#         super(QuantumNaturalGradient, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super().__setstate__(state)

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             if group["approximation"] == "exact":
#                 data_vec = torch.Tensor([v for v in group["params"] if (v.requires_grad)])
#                 grad_vec = torch.Tensor([v.grad.data for v in group["params"] if v.requires_grad])
#                 metric_tensor = (1 / 4) * get_quantum_fisher(group["circuit"]) + group[
#                     "epsilon"
#                 ] * torch.eye(len(data_vec))
#                 metric_tensor_inv = torch.adjoint(metric_tensor)

#                 transf_grad = torch.matmul(metric_tensor_inv, grad_vec)

#                 it = iter(range(len(data_vec)))
#                 for p in group["params"]:
#                     if p.grad is None:
#                         continue
#                     # with torch.no_grad():
#                     p.data.add_(transf_grad[next(it)], alpha=-group["lr"])

#         return loss
