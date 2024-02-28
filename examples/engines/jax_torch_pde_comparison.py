# library imports
from __future__ import annotations

import timeit
from functools import reduce
from itertools import product
from operator import add

# Jax-related imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# Torch-related imports
import torch
from jax import Array, jit, value_and_grad, vmap
from numpy.typing import ArrayLike
from torch import Tensor, exp, linspace, nn, ones_like, optim, rand, sin, tensor
from torch.autograd import grad

from qadence import QNN, AbstractBlock, BackendName, DiffMode, backend_factory, get_logger
from qadence.blocks.utils import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization

logger = get_logger(__name__)
DIFF_MODE = DiffMode.AD
TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
JAX_DEVICE = jax.devices()[0].device_kind
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

single_domain_jax = jnp.linspace(0, 1, num=N_POINTS)
domain_jax = jnp.array(list(product(single_domain_jax, single_domain_jax)))


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
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.net = net
        self.n_colpoints = n_colpoints
        self.n_inputs = n_inputs
        self.device = device

    def sample(self) -> Tensor:
        return rand(size=(self.n_colpoints, self.n_inputs), device=self.device)

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


def torch_solve(circ: QuantumCircuit, obs: AbstractBlock, backend: BackendName) -> np.ndarray:
    model = QNN(
        circuit=circ, observable=obs, backend=backend, diff_mode=DIFF_MODE, inputs=VARIABLES
    ).to(TORCH_DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sol = DomainSampling(
        net=model, n_inputs=len(VARIABLES), n_colpoints=N_POINTS, device=TORCH_DEVICE
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
    return model(domain_torch.to(TORCH_DEVICE)).reshape(N_POINTS, N_POINTS).detach().cpu().numpy()


def jax_solve(circ, obs) -> Array:
    bknd = backend_factory(backend=BackendName.HORQRUX, diff_mode=DIFF_MODE)
    conv_circ, conv_obs, embedding_fn, params = bknd.convert(circ, obs)

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    @jit
    def exp_fn(params: dict[str, Array], inputs: dict[str, Array]) -> ArrayLike:
        return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))

    def loss_fn(params: dict[str, Array], x: Array, y: Array) -> Array:
        def pde_loss(x: float, y: float) -> Array:
            l_b, r_b, t_b, b_b = list(
                map(
                    lambda d: exp_fn(params, d),
                    [
                        {"x": jnp.zeros(1), "y": y},  # u(0,y)=0
                        {"x": jnp.ones(1), "y": y},  # u(L,y)=0
                        {"x": x, "y": jnp.ones(1)},  # u(x,H)=0
                        {"x": x, "y": jnp.zeros(1)},  # u(x,0)=f(x)
                    ],
                )
            )
            b_b -= jnp.sin(jnp.pi * x)
            hessian = jax.jacfwd(jax.grad(lambda d: exp_fn(params, d)))({"x": x, "y": y})
            interior = hessian["x"]["x"] + hessian["y"]["y"]  # uxx+uyy=0
            return reduce(add, list(map(lambda t: jnp.power(t, 2), [l_b, r_b, t_b, b_b, interior])))

        return jnp.mean(vmap(pde_loss, in_axes=(0, 0))(x, y))

    def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # collocation points sampling and training
    def sample_points(n_in: int, n_p: int) -> ArrayLike:
        return np.random.uniform(0, 1.0, (n_in, n_p))

    @jit
    def train_step(i: int, inputs: tuple) -> tuple:
        params, opt_state = inputs
        x, y = sample_points(2, N_POINTS)
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        params, opt_state = optimize_step(params, opt_state, grads)
        return params, opt_state

    params, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (params, opt_state))
    return vmap(lambda domain: exp_fn(params, {"x": domain[0], "y": domain[1]}), in_axes=(0,))(
        domain_jax
    ).reshape(N_POINTS, N_POINTS)


if __name__ == "__main__":
    res = {
        "n_qubits": N_QUBITS,
        "n_epochs": N_EPOCHS,
        "jax_device": JAX_DEVICE,
        "torch_device": TORCH_DEVICE,
    }
    for engine, fn in zip(
        ["horqrux", "pyqtorch", "emu_c"],
        [
            "jax_solve(circ,obs).block_until_ready()",
            "torch_solve(circ,obs, 'pyqtorch')",
            "torch_solve(circ,obs, 'emu_c')",
        ],
    ):
        setup = "circ,obs = setup_circ_obs(n_qubits=N_QUBITS, depth=DEPTH)"
        pp_run_times = timeit.repeat(fn, setup, number=1, repeat=5, globals=globals())
        pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
        res[engine] = f"mean_runtime: {pp_mean}, std_runtime: {pp_std}"

    # {'n_qubits': 4, 'n_epochs': 1000, 'jax_device': 'NVIDIA A100-SXM4-40GB',
    # 'torch_device': device(type='cuda'),

    # 'horqrux': 'mean_runtime: 51.33197688870132, std_runtime: 0.0',
    # 'pyqtorch': 'mean_runtime: 251.40893555991352, std_runtime: 0.0'}
    # 'emu-c- no truncation': 'mean_runtime: 167.01573771610856, std_runtime: 0.0'}
    # 'emu-c- cotengra truncation': 'mean_runtime: 260.01573771610856, std_runtime: 0.0'}
    print(res)
    logger.info(res)
    import pickle

    with open("results.pkl", "wb") as f:
        pickle.dump(res, f)

    if PLOT:
        backend = "pyqtorch"
        circ, obs = setup_circ_obs(N_QUBITS, DEPTH)
        dqc_sol_jax = jax_solve(circ, obs).block_until_ready()
        dqc_sol_torch = torch_solve(circ, obs, backend)
        analytic_sol = (
            (exp(-np.pi * domain_torch[:, 0]) * sin(np.pi * domain_torch[:, 1]))
            .reshape(N_POINTS, N_POINTS)
            .T
        ).numpy()
        fig, ax = plt.subplots(1, 3, figsize=(7, 7))
        ax[0].imshow(analytic_sol, cmap="turbo")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        ax[0].set_title("Analytical solution u(x,y)")
        ax[1].imshow(dqc_sol_torch, cmap="turbo")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_title("Torch DQC")
        ax[2].imshow(dqc_sol_jax, cmap="turbo")
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("y")
        ax[2].set_title("JAX DQC")
        plt.show()


# {'n_qubits': 4, 'n_epochs': 1000, 'jax_device': 'cpu', 'torch_device': device(type='cuda'), 'horqrux': 'mean_runtime: 139.53169030491262, std_runtime: 2.0769610744187563', 'pyqtorch': 'mean_runtime: 237.94209703579546, std_runtime: 1.1005077375594445', 'emu_c': 'mean_runtime: 242.04887232519687, std_runtime: 1.0783225551830007'}


# {'n_qubits': 4, 'n_epochs': 1000, 'jax_device': 'cpu', 'torch_device': device(type='cpu'), 'horqrux': 'mean_runtime: 69.70531777479918, std_runtime: 0.49274159367518355', 'pyqtorch': 'mean_runtime: 72.95039291661233, std_runtime: 1.2634803707358608', 'emu_c': 'mean_runtime: 58.82601000840077, std_runtime: 1.219633167820975'}
