# library imports
from __future__ import annotations

import timeit
from itertools import product

import numpy as np

# Torch-related imports
import torch
from torch import Tensor, linspace, nn, ones_like, optim, rand, sin, tensor
from torch.autograd import grad

from qadence import QNN, AbstractBlock, DiffMode, get_logger
from qadence.blocks.utils import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization

logger = get_logger(__name__)
DIFF_MODE = DiffMode.AD
DTYPE = torch.complex128
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PLOT = False
LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_POINTS = 150
N_EPOCHS = 1000


def setup_circ_obs(n_qubits: int, depth: int) -> tuple[QuantumCircuit, AbstractBlock]:
    # define a simple DQC model
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    # parallel Fourier feature map
    split = n_qubits // len(VARIABLES)
    fm = kron(
        *[
            feature_map(n_qubits=split, support=support, param=param)
            for param, support in zip(
                VARIABLES,
                [
                    list(list(range(n_qubits))[i : i + split])
                    for i in range(n_qubits)
                    if i % split == 0
                ],
            )
        ]
    )
    # choosing a cost function
    obs = total_magnetization(n_qubits=n_qubits)
    # building the circuit and the quantum model
    circ = QuantumCircuit(n_qubits, chain(fm, ansatz))
    return circ, obs


single_domain_torch = linspace(0, 1, steps=N_POINTS)
domain_torch = tensor(list(product(single_domain_torch, single_domain_torch)))


def calc_derivative(outputs, inputs) -> Tensor:
    """
    Returns the derivative of a function output.

    with respect to its inputs
    """
    if not inputs.requires_grad:
        inputs.requires_grad = True
    return grad(
        inputs=inputs,
        outputs=outputs,
        grad_outputs=ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


class DomainSampling(nn.Module):
    """
    Collocation points sampling from domains uses uniform random sampling.

    Problem-specific MSE loss function for solving the 2D Laplace equation.
    """

    def __init__(
        self,
        net: nn.Module | QNN,
        n_inputs: int = len(VARIABLES),
        n_colpoints: int = N_POINTS,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.net = net
        self.n_colpoints = n_colpoints
        self.n_inputs = n_inputs
        self.device = device
        self.dtype = dtype

    def sample(self) -> Tensor:
        return rand(size=(self.n_colpoints, self.n_inputs), device=self.device, dtype=self.dtype)

    def left_boundary(self) -> Tensor:  # u(0,y)=0
        sample = self.sample()
        sample[:, 0] = 0.0
        return self.net(sample).pow(2).mean()

    def right_boundary(self) -> Tensor:  # u(L,y)=0
        sample = self.sample()
        sample[:, 0] = 1.0
        return self.net(sample).pow(2).mean()

    def top_boundary(self) -> Tensor:  # u(x,H)=0
        sample = self.sample()
        sample[:, 1] = 1.0
        return self.net(sample).pow(2).mean()

    def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
        sample = self.sample()
        sample[:, 1] = 0.0
        return (self.net(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()

    def interior(self) -> Tensor:  # uxx+uyy=0
        sample = self.sample().requires_grad_()
        first_both = calc_derivative(self.net(sample), sample)
        second_both = calc_derivative(first_both, sample)
        return (second_both[:, 0] + second_both[:, 1]).pow(2).mean()


def torch_solve() -> None:
    circ, obs = setup_circ_obs(N_QUBITS, DEPTH)
    model = QNN(
        circuit=circ, observable=obs, backend="pyqtorch", diff_mode="ad", inputs=VARIABLES
    ).to(device=DEVICE, dtype=DTYPE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sol = DomainSampling(
        net=model,
        n_inputs=len(VARIABLES),
        n_colpoints=N_POINTS,
        device=DEVICE,
        dtype=torch.float64 if DTYPE == torch.cdouble else torch.float32,
    )
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss = (
            sol.left_boundary()
            + sol.right_boundary()
            + sol.top_boundary()
            + sol.bottom_boundary()
            + sol.interior()
        )
        loss.backward()
        opt.step()


if __name__ == "__main__":
    res = {"n_qubits": N_QUBITS, "n_epochs": N_EPOCHS, "device": DEVICE}
    for dtype in ["torch.cfloat", "torch.cdouble"]:
        pp_run_times = timeit.repeat(
            "torch_solve()", f"DTYPE={dtype}", number=1, repeat=1, globals=globals()
        )
        pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
        res[dtype] = f"mean_runtime: {pp_mean}, std_runtime: {pp_std}"
