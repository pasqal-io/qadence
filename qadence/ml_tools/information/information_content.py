from __future__ import annotations

import functools
from logging import getLogger
from math import log, sqrt
from statistics import NormalDist
from typing import Any, Callable

import torch
from torch import nn
from torch.func import functional_call  # type: ignore

logger = getLogger("ml_tools")


class InformationContent:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        xs: Any,
        epsilons: torch.Tensor,
        variation_multiple: int = 20,
    ) -> None:
        """Information Landscape class.

        This class handles the study of loss landscape from information theoretic
        perspective and provides methods to get bounds on the norm of the
        gradient from the Information Content of the loss landscape.

        Args:
            model: The quantum or classical model to analyze.
            loss_fn: Loss function that takes model output and calculates loss
            xs: Input data to evaluate the model on
            epsilons: The thresholds to use for discretization of the finite derivatives
            variation_multiple: The number of sets of variational parameters to generate per each
                variational parameter. The number of variational parameters required for the
                statistical analysis scales linearly with the amount of them present in the
                model. This is that linear factor.

        Notes:
            This class provides flexibility in terms of what the model, the loss function,
            and the xs are. The only requirement is that the loss_fn takes the model and xs as
            arguments and returns the loss, and another dictionary of other metrics.

            Thus, assumed structure:
                loss_fn(model, xs) -> (loss, metrics, ...)

            Example: A Classifier
                ```python
                model = nn.Linear(10, 1)

                def loss_fn(
                    model: nn.Module,
                    xs: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]:
                    criterion = nn.MSELoss()
                    inputs, labels = xs
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    metrics = {"loss": loss.item()}
                    return loss, metrics

                xs = (torch.randn(10, 10), torch.randn(10, 1))

                info_landscape = InfoLandscape(model, loss_fn, xs)
                ```
                In this example, the model is a linear classifier, and the `xs` include both the
                inputs and the target labels. The logic for calculation of the loss from this lies
                entirely within the `loss_fn` function. This can then further be used to obtain the
                bounds on the average norm of the gradient of the loss function.

            Example: A Physics Informed Neural Network
                ```python
                class PhysicsInformedNN(nn.Module):
                    // <Initialization Logic>

                    def forward(self, xs: dict[str, torch.Tensor]):
                        return {
                            "pde_residual": pde_residual(xs["pde"]),
                            "boundary_condition": bc_term(xs["bc"]),
                        }

                def loss_fn(
                    model: PhysicsInformedNN,
                    xs: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]:
                    pde_residual, bc_term = model(xs)
                    loss = torch.mean(torch.sum(pde_residual**2, dim=1), dim=0)
                        + torch.mean(torch.sum(bc_term**2, dim=1), dim=0)

                    return loss, {"pde_residual": pde_residual, "bc_term": bc_term}

                xs = {
                    "pde": torch.linspace(0, 1, 10),
                    "bc": torch.tensor([0.0]),
                }

                info_landscape = InfoLandscape(model, loss_fn, xs)
                ```

                In this example, the model is a Physics Informed Neural Network, and the `xs`
                are the inputs to the different residual components of the model. The logic
                for calculation of the residuals lies within the PhysicsInformedNN class, and
                the loss function is defined to calculate the loss that is to be optimized
                from these residuals. This can then further be used to obtain the
                bounds on the average norm of the gradient of the loss function.

            The first value that the `loss_fn` returns is the loss value that is being optimized.
            The function is also expected to return other value(s), often the metrics that are
            used to calculate the loss. These values are ignored for the purpose of this class.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.xs = xs
        self.epsilons = epsilons
        self.device = next(model.parameters()).device

        self.param_shapes = {}
        self.total_params = 0

        for name, param in model.named_parameters():
            self.param_shapes[name] = param.shape
            self.total_params += param.numel()
        self.n_variations = variation_multiple * self.total_params
        self.all_variations = torch.empty(
            (self.n_variations, self.total_params), device=self.device
        ).uniform_(0, 2 * torch.pi)

    def reshape_param_variations(self) -> dict[str, torch.Tensor]:
        """Reshape variations of the model's variational parameters.

        Returns:
            Dictionary of parameter tensors, each with shape [n_variations, *param_shape]
        """
        param_variations = {}
        start_idx = 0

        for name, shape in self.param_shapes.items():
            param_size = torch.prod(torch.tensor(shape)).item()
            param_variations[name] = self.all_variations[
                :, start_idx : start_idx + param_size
            ].view(self.n_variations, *shape)
            start_idx += param_size

        return param_variations

    def batched_loss(self) -> torch.Tensor:
        """Calculate loss for all parameter variations in a batched manner.

        Returns: Tensor of loss values for each parameter variation
        """
        param_variations = self.reshape_param_variations()
        losses = torch.zeros(self.n_variations, device=self.device)

        for i in range(self.n_variations):
            params = {name: param[i] for name, param in param_variations.items()}
            current_model = lambda x: functional_call(self.model, params, (x,))
            losses[i] = self.loss_fn(current_model, self.xs)[0]

        return losses

    def randomized_finite_der(self) -> torch.Tensor:
        """
        Calculate normalized finite difference of loss on doing random walk in the parameter space.

        This serves as a proxy for the derivative of the loss with respect to parameters.

        Returns:
            Tensor containing normalized finite differences (approximate directional derivatives)
            between consecutive points in the random walk. Shape: [n_variations - 1]
        """
        losses = self.batched_loss()

        return (losses[1:] - losses[:-1]) / (
            torch.norm(self.all_variations[1:] - self.all_variations[:-1], dim=1) + 1e-8
        )

    def discretize_derivatives(self) -> torch.Tensor:
        """
        Convert finite derivatives into discrete values.

        Returns:
            Tensor containing discretized derivatives with shape [n_epsilons, n_variations-2]
            Each row contains {-1, 0, 1} values for that epsilon
        """
        derivatives = self.randomized_finite_der()

        derivatives = derivatives.unsqueeze(0)
        epsilons = self.epsilons.unsqueeze(1)

        discretized = torch.zeros((len(epsilons), len(derivatives[0])), device=self.device)
        discretized[derivatives > epsilons] = 1
        discretized[derivatives < -epsilons] = -1

        return discretized

    def calculate_transition_probabilities_batch(self) -> torch.Tensor:
        """
        Calculate transition probabilities for multiple epsilon values.

        Returns:
            Tensor of shape [n_epsilons, 6] containing probabilities for each transition type
            Columns order: [+1to0, +1to-1, 0to+1, 0to-1, -1to0, -1to+1]
        """
        discretized = self.discretize_derivatives()

        current = discretized[:, :-1]
        next_val = discretized[:, 1:]

        transitions = torch.stack(
            [
                ((current == 1) & (next_val == 0)).sum(dim=1),
                ((current == 1) & (next_val == -1)).sum(dim=1),
                ((current == 0) & (next_val == 1)).sum(dim=1),
                ((current == 0) & (next_val == -1)).sum(dim=1),
                ((current == -1) & (next_val == 0)).sum(dim=1),
                ((current == -1) & (next_val == 1)).sum(dim=1),
            ],
            dim=1,
        ).float()

        total_transitions = current.size(1)
        probabilities = transitions / total_transitions

        return probabilities

    @functools.cached_property
    def calculate_IC(self) -> torch.Tensor:
        """
        Calculate Information Content for multiple epsilon values.

        Returns: Tensor of IC values for each epsilon [n_epsilons]
        """
        probs = self.calculate_transition_probabilities_batch()

        mask = probs > 1e-4

        ic_terms = torch.where(mask, -probs * torch.log(probs), torch.zeros_like(probs))
        ic_values = ic_terms.sum(dim=1) / torch.log(torch.tensor(6.0))

        return ic_values

    def max_IC(self) -> tuple[float, float]:
        """
        Get the maximum Information Content and its corresponding epsilon.

        Returns: Tuple of (maximum IC value, optimal epsilon)
        """
        max_ic, max_idx = torch.max(self.calculate_IC, dim=0)
        max_epsilon = self.epsilons[max_idx]
        return max_ic.item(), max_epsilon.item()

    def sensitivity_IC(self, eta: float) -> float:
        """
        Find the minimum value of epsilon such that the information content is less than eta.

        Args:
            eta: Threshold value, the sensitivity IC.

        Returns: The epsilon value that gives IC that is less than the sensitivity IC.
        """
        ic_values = self.calculate_IC
        mask = ic_values < eta
        epsilons = self.epsilons[mask]
        return float(epsilons.min().item())

    @staticmethod
    @functools.lru_cache
    def q_value(H_value: float) -> float:
        """
        Compute the q value.

        q is the solution to the equation:
        H(x) = 4h(x) + 2h(1/2 - 2x)

        It is the value of the probability of 4 of the 6 transitions such that
        the IC is the same as the IC of our system.

        This quantity is useful in calculating the bounds on the norms of the gradients.

        Args:
            H_value (float): The information content.

        Returns:
            float: The q value
        """

        x = torch.linspace(0.001, 0.16667, 10000)

        H = -4 * x * torch.log(x) / torch.log(torch.tensor(6)) - 2 * (0.5 - 2 * x) * torch.log(
            0.5 - 2 * x
        ) / torch.log(torch.tensor(6))
        err = torch.abs(H - H_value)
        idx = torch.argmin(err)
        return float(x[idx].item())

    def get_grad_norm_bounds_max_IC(self) -> tuple[float, float]:
        """
        Compute the bounds on the average norm of the gradient.

        Returns:
            tuple[Tensor, Tensor]: The lower and upper bounds.
        """
        max_IC, epsilon_m = self.max_IC()
        lower_bound = (
            epsilon_m
            * sqrt(self.total_params)
            / (NormalDist().inv_cdf(1 - 2 * self.q_value(max_IC)))
        )
        upper_bound = (
            epsilon_m
            * sqrt(self.total_params)
            / (NormalDist().inv_cdf(0.5 * (1 + 2 * self.q_value(max_IC))))
        )

        if max_IC < log(2, 6):
            logger.warning(
                "Warning: The maximum IC is less than the required value. The bounds may be"
                + " inaccurate."
            )

        return lower_bound, upper_bound

    def get_grad_norm_bounds_sensitivity_IC(self, eta: float) -> float:
        """
        Compute the bounds on the average norm of the gradient.

        Args:
            eta (float): The sensitivity IC.

        Returns:
            Tensor: The lower bound.
        """
        epsilon_sensitivity = self.sensitivity_IC(eta)
        upper_bound = (
            epsilon_sensitivity * sqrt(self.total_params) / (NormalDist().inv_cdf(1 - 3 * eta / 2))
        )
        return upper_bound
