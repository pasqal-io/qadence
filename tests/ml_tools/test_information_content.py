from __future__ import annotations

from math import isclose

import pytest
import torch
import torch.nn as nn

from qadence.ml_tools.information.information_content import InformationContent


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_loss_fn(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(x)
    return torch.mean(output**2), output


@pytest.fixture
def setup_landscape() -> InformationContent:
    model = SimpleModel()
    xs = torch.randn(10, 2)  # 10 samples, 2 features each
    epsilons = torch.logspace(-4, 4, 10)
    landscape = InformationContent(
        model=model, loss_fn=test_loss_fn, xs=xs, epsilons=epsilons, variation_multiple=5
    )
    return landscape


def test_initialization(setup_ic: InformationContent) -> None:
    info_content = setup_ic

    assert callable(info_content.loss_fn)
    assert isinstance(info_content.epsilons, torch.Tensor)

    assert len(info_content.param_shapes) > 0
    assert info_content.total_params > 0
    assert info_content.n_variations == info_content.total_params * 5


def test_reshape_param_variations(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    param_variations = info_content.reshape_param_variations()

    assert set(param_variations.keys()) == set(info_content.param_shapes.keys())

    for name, tensor in param_variations.items():
        expected_shape = (info_content.n_variations, *info_content.param_shapes[name])
        assert tensor.shape == expected_shape


def test_batched_loss(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    losses = info_content.batched_loss()

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (info_content.n_variations,)
    assert not torch.isnan(losses).any()
    assert not torch.isinf(losses).any()


def test_randomized_finite_der(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    derivatives = info_content.randomized_finite_der()

    assert isinstance(derivatives, torch.Tensor)
    assert derivatives.shape == (info_content.n_variations - 1,)
    assert not torch.isnan(derivatives).any()
    assert not torch.isinf(derivatives).any()


def test_discretize_derivatives(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    discretized = info_content.discretize_derivatives()

    assert discretized.shape == (len(info_content.epsilons), info_content.n_variations - 1)

    unique_values = torch.unique(discretized)
    assert all(val in [-1.0, 0.0, 1.0] for val in unique_values.tolist())


def test_calculate_transition_probabilities_batch(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    probs = info_content.calculate_transition_probabilities_batch()

    assert probs.shape == (len(info_content.epsilons), 6)

    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.all(torch.sum(probs, dim=1) <= torch.ones(len(info_content.epsilons)))


def test_calculate_IC(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    ic_values = info_content.calculate_IC

    assert ic_values.shape == (len(info_content.epsilons),)
    assert torch.all(ic_values >= 0)
    assert torch.all(ic_values <= 1)


def test_max_IC(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    max_ic, optimal_epsilon = info_content.max_IC()

    assert isinstance(max_ic, float)
    assert isinstance(optimal_epsilon, float)
    assert 0 <= max_ic <= 1
    assert optimal_epsilon > 0


def test_sensitivity_IC(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    eta = 0.5
    epsilon = info_content.sensitivity_IC(eta)

    assert isinstance(epsilon, float)
    assert epsilon > 0


def test_q_value(setup_ic: InformationContent) -> None:
    info_content = setup_ic
    H_value = 1.0
    q = info_content.q_value(H_value)

    assert isinstance(q, float)
    assert isclose(q, 1 / 6, abs_tol=1e-5)


def test_grad_norm_bounds(setup_ic: InformationContent) -> None:
    info_content = setup_ic

    lower, upper = info_content.get_grad_norm_bounds_max_IC()
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= upper

    eta = 2e-2
    upper_bound = info_content.get_grad_norm_bounds_sensitivity_IC(eta)
    assert isinstance(upper_bound, float)
    assert upper_bound > 0
