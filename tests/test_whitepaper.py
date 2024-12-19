#######################################################################
# ---------------------------- IMPORTANT ---------------------------- #
# The tests in this file replicate the code snippets used in the      #
# Qadence whitepaper. If an MR breaks these tests, they SHOULD NOT    #
# be changed. Instead, make sure appropriate backwards compatibility  #
# is provided. If you MUST change the tests, first check with the     #
# Q.Libs team about updating the arXiv paper.                         #
#######################################################################

from __future__ import annotations


def test_code_sample_1() -> None:
    from qadence import CPHASE, PI, H, chain, kron

    def cphases(qs: tuple, l: int):  # type: ignore
        return chain(CPHASE(qs[j], qs[l], PI / 2 ** (j - l)) for j in range(l + 1, len(qs)))

    def QFT(qs: tuple):  # type: ignore
        return chain(H(qs[l]) * cphases(qs, l) for l in range(len(qs)))

    # Easily compose a QFT and its inverse
    qft_block = kron(QFT((0, 1, 2)), QFT((3, 4, 5)).dagger())


def test_code_sample_2() -> None:
    import sympy

    from qadence import PI, RX, FeatureParameter, Parameter, VariationalParameter, X, Y

    # Defining a Feature and Variational parameter
    theta, phi = VariationalParameter("theta"), FeatureParameter("phi")

    # Alternative syntax
    theta, phi = Parameter("theta", trainable=True), Parameter("phi", trainable=False)

    # Arbitrarily compose parameters with sympy
    expr = sympy.acos((theta + phi)) * PI

    gate = RX(0, expr)  # Use as unitary gate arguments
    h_op = expr * (X(0) + Y(0))  # Or as scaling coefficients for Hermitian operators


def test_code_sample_3_4() -> None:
    from qadence import PI, RX, QuantumCircuit, Register, kron

    reg = Register.line(3)  # A simple line register of 3 qubits

    # Use other topologies, and set the spacing between qubits
    reg = Register.from_coordinates([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
    reg = Register.circle(n_qubits=10, spacing=10)
    reg = Register.triangular_lattice(n_cells_row=2, n_cells_col=2)

    n_qubits = 4

    block = kron(RX(i, i * PI / 2) for i in range(n_qubits))

    circuit = QuantumCircuit(reg, block)
    circuit = QuantumCircuit(n_qubits, block)


def test_code_sample_5() -> None:
    import torch

    from qadence import (
        RX,
        DiffMode,
        FeatureParameter,
        QuantumCircuit,
        QuantumModel,
        Z,
        add,
        hea,
        kron,
        product_state,
    )

    # Circuit setup
    n_qubits = 4

    # Parameter initialization
    param = FeatureParameter("phi")

    # Feature map through manual block composition
    feature_map = kron(RX(i, i * param) for i in range(n_qubits))

    # Variational ansatz through a contructor function
    ansatz = hea(n_qubits, depth=1)

    circ = QuantumCircuit(n_qubits, feature_map * ansatz)

    # Observable to measure
    obs = add(Z(i) for i in range(n_qubits))

    # Initialize the QuantumModel, setting the differentiation mode
    model = QuantumModel(circuit=circ, observable=obs, diff_mode=DiffMode.GPSR)

    # Input values for "phi", requires_grad=True allows differentiation w.r.t. "phi"
    values = {"phi": torch.rand(1, requires_grad=True)}

    # Optionally, we can use a custom initial state, here defined as |1000>.
    state = product_state("1000")

    # Model execution
    out_state = model.run(values=values, state=state)
    samples = model.sample(values=values, state=state)
    expectation = model.expectation(values=values, state=state)

    # Standard usage of torch.autograd API to compute the gradient w.r.t to "phi"
    dexp_dphi = torch.autograd.grad(expectation, values["phi"], torch.ones_like(expectation))


def test_code_sample_6() -> None:
    from qadence import AnalogInteraction, AnalogRot, AnalogRX, AnalogRY, AnalogRZ

    # Global rotations automatically translated to the Hamiltonian parameters
    rx, ry, rz = AnalogRX(angle="th1"), AnalogRY(angle="th2"), AnalogRZ(angle="th3")

    # Evolve the interaction term
    analog_int = AnalogInteraction(duration="t")

    # Fully control all the parameters
    da_rot = AnalogRot(omega="om", phase="ph", delta="d", duration="t")


def test_code_sample_7() -> None:
    from qadence import Interaction, Register, X, hamiltonian_factory

    reg = Register.triangular_lattice(n_cells_row=2, n_cells_col=2, spacing=2.0)

    # Create the interaction strength term with 1/r decay
    strength_list = [1.0 / reg.distances[p] for p in reg.all_node_pairs]

    # Initialize NN Hamiltonian
    nn_ham = hamiltonian_factory(
        reg,  # Register with the Hamiltonian topology
        interaction=Interaction.NN,  # Type of interaction to use
        interaction_strength=strength_list,  # List of all interaction strengths
        detuning=X,  # Pauli operator for the detuning
        detuning_strength="d",  # Parameterize the detuning strength
        use_all_node_pairs=True,  # Use all pairs instead of graph edges
    )


def test_appendix_code_sample_1() -> None:
    import sympy
    import torch

    from qadence import (
        RX,
        BackendName,
        DiffMode,
        FeatureParameter,
        QuantumCircuit,
        kron,
        total_magnetization,
    )
    from qadence.backends import backend_factory

    def differentiate(diff_mode, circuit, observable, values):  # type: ignore
        # Instantiate a differentiable backend with the given differentiation mode
        backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=diff_mode)

        # Convert instructions to a native representation on the chosen backend object
        converted = backend.convert(circuit, observable)
        embedded_params = converted.embedding_fn(converted.params, values)

        # Compute and differentiate the expectation w.r.t. the "x" parameter
        # using the standard torch.autograd engine
        expval = backend.expectation(
            converted.circuit, converted.observable, param_values=embedded_params
        )
        return torch.autograd.grad(expval, values["x"], torch.ones_like(expval))[0]

    n_qubits = 4
    x = FeatureParameter("x")
    block = kron(RX(i, (i + 1) * sympy.acos(x)) for i in range(n_qubits))
    circuit = QuantumCircuit(n_qubits, block)
    observable = total_magnetization(n_qubits)

    values = {"x": torch.rand(10, requires_grad=True)}
    diff_ad = differentiate(DiffMode.AD, circuit, observable, values)
    diff_gpsr = differentiate(DiffMode.GPSR, circuit, observable, values)
    diff_adjoint = differentiate(DiffMode.ADJOINT, circuit, observable, values)

    # Check that derivatives are matching
    check_eq = lambda x, y: torch.all(torch.isclose(x, y)).item()
    assert check_eq(diff_ad, diff_gpsr) and check_eq(diff_gpsr, diff_adjoint)


def test_appendix_code_sample_2_3() -> None:
    from qadence import Interaction, N, Register, Strategy, daqc_transform, hamiltonian_factory, qft

    reg = Register.triangular_lattice(n_cells_row=2, n_cells_col=2, spacing=2.0)

    # Create the interaction strength term with 1/r decay
    strength_list = [1.0 / reg.distances[p] for p in reg.all_node_pairs]

    # Initialize NN Hamiltonian
    nn_ham = hamiltonian_factory(
        reg,  # Register with the Hamiltonian topology
        interaction=Interaction.NN,  # Type of interaction to use
        interaction_strength=strength_list,  # List of all interaction strengths
        use_all_node_pairs=True,  # Use all pairs instead of graph edges
    )

    # Target Hamiltonian to evolve for a specific time t_f
    h_target = N(0) @ N(1) + N(1) @ N(2) + N(2) @ N(0)
    t_f = 5.0

    transformed_ising = daqc_transform(
        n_qubits=reg.n_qubits,
        gen_target=h_target,
        t_f=t_f,
        gen_build=nn_ham,
        strategy=Strategy.SDAQC,
    )

    # Changing the strategy overrides the default Strategy.DIGITAL
    qft_daqc = qft(n_qubits=3, strategy=Strategy.SDAQC, gen_build=nn_ham)


def test_appendix_code_sample_4() -> None:
    import torch
    from numpy.random import uniform
    from torch.autograd import grad

    from qadence import QNN, BasisSet, QuantumCircuit, chain, feature_map, hea, total_magnetization

    torch.manual_seed(404)

    n_qubits = 4
    depth = 3
    lr = 0.01
    n_points = 20

    # Building the DQC model
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    fm = feature_map(n_qubits=n_qubits, param="x", fm_type=BasisSet.CHEBYSHEV)
    obs = total_magnetization(n_qubits=n_qubits)
    circuit = QuantumCircuit(n_qubits, chain(fm, ansatz))
    model = QNN(circuit=circuit, observable=obs, inputs=["x"])

    # using Adam as an optimizer of choice
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Define a problem-specific MSE loss function for the ODE df/dx=4x^3+x^2-2x-1/2
    def loss_fn(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        dfdx = grad(inputs=inputs, outputs=outputs.sum(), create_graph=True)[0]
        ode_loss = dfdx - (4 * inputs**3 + inputs**2 - 2 * inputs - 0.5)
        boundary_loss = model(torch.zeros_like(inputs)) - torch.ones_like(inputs)
        return ode_loss.pow(2).mean() + boundary_loss.pow(2).mean()

    for epoch in range(100):
        opt.zero_grad()
        cp = torch.tensor(
            uniform(low=-0.99, high=0.99, size=(n_points, 1)), requires_grad=True
        ).float()
        loss = loss_fn(inputs=cp, outputs=model(cp))
        loss.backward()
        opt.step()

    # Compare the trained model to the ground truth
    sample_points = torch.linspace(-1.0, 1.0, steps=100).reshape(-1, 1)

    analytic_sol = (
        sample_points**4
        + (1 / 3) * sample_points**3
        - sample_points**2
        - (1 / 2) * sample_points
        + 1
    )

    # DQC solution
    dqc_sol = model(sample_points).detach().numpy()

    x_data = sample_points.detach().numpy()


def test_appendix_code_sample_5_to_9() -> None:
    from itertools import product

    import torch

    from qadence import QNN, I, Parameter, QuantumCircuit, Z, add, chain, feature_map, hea, kron

    def calc_derivative(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute model derivative using torch.autograd."""
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad

    x_min = 0.0
    x_max = 1.0

    class LossSampling(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.n_inputs = 2  # (x, y)

        def left_boundary(self, n_points: int) -> torch.Tensor:
            """U(0, y) = sin(pi*x)."""
            samples = x_min + (x_max - x_min) * torch.rand((n_points, self.n_inputs))
            samples[:, 0] = 0.0
            return (self.model(samples) - torch.sin(torch.pi * samples[:, 1])).pow(2).mean()

        def right_boundary(self, n_points: int) -> torch.Tensor:
            """U(X, y) = 0."""
            samples = x_min + (x_max - x_min) * torch.rand((n_points, self.n_inputs))
            samples[:, 0] = x_max
            return self.model(samples).pow(2).mean()

        def top_boundary(self, n_points: int) -> torch.Tensor:
            """U(x, 0) = 0."""
            samples = x_min + (x_max - x_min) * torch.rand((n_points, self.n_inputs))
            samples[:, 1] = 0.0
            return self.model(samples).pow(2).mean()

        def bottom_boundary(self, n_points: int) -> torch.Tensor:
            """U(x, Y) = 0."""
            samples = x_min + (x_max - x_min) * torch.rand((n_points, self.n_inputs))
            samples[:, 1] = x_max
            return self.model(samples).pow(2).mean()

        def interior(self, n_points: int) -> torch.Tensor:
            samples = x_min + (x_max - x_min) * torch.rand(
                (n_points, self.n_inputs), requires_grad=True
            )
            first_deriv = calc_derivative(self.model(samples), samples)
            second_deriv = calc_derivative(first_deriv, samples)
            return (second_deriv[:, 0] + second_deriv[:, 1]).pow(2).mean()

    n_qubits = 4
    depth = 3

    ansatz = hea(n_qubits=n_qubits, depth=depth)

    # Parallel Fourier feature map
    fm_x = feature_map(
        n_qubits=2,
        support=(0, 1),
        param="x",
    )
    fm_y = feature_map(
        n_qubits=2,
        support=(2, 3),
        param="y",
    )

    fm = kron(fm_x, fm_y)

    # Scaled and shifted total magnetization
    ident_shift = Parameter("a") * kron(I(i) for i in range(n_qubits))
    total_mag = add(Z(i) for i in range(n_qubits))
    observable = total_mag

    circuit = QuantumCircuit(n_qubits, chain(ansatz, fm, ansatz))

    model = QNN(circuit=circuit, observable=observable, inputs=["x", "y"])

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = LossSampling(model=model)

    n_epochs = 10
    n_points = 100

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss_total = (
            loss.left_boundary(n_points)
            + loss.right_boundary(n_points)
            + loss.top_boundary(n_points)
            + loss.bottom_boundary(n_points)
            + loss.interior(n_points)
        )
        loss_total.backward()
        opt.step()

    n_steps = 100

    domain_1d = torch.linspace(0, 1.0, steps=n_steps)
    domain = torch.tensor(list(product(domain_1d, domain_1d)))

    # analytical solution
    def exact_sol(domain: torch.Tensor):  # type: ignore
        exp_x = torch.exp(-torch.pi * domain[:, 0])
        sin_y = torch.sin(torch.pi * domain[:, 1])
        return exp_x * sin_y

    # DQC solution
    dqc_sol = model(domain).reshape(n_steps, n_steps).T.detach()
    exact = exact_sol(domain).reshape(n_steps, n_steps).T


def test_appendix_code_sample_10_to_end() -> None:
    import nevergrad as ng
    import numpy as np
    import torch
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform

    from qadence import (
        AnalogRX,
        AnalogRZ,
        QuantumCircuit,
        QuantumModel,
        Register,
        RydbergDevice,
        chain,
    )
    from qadence.ml_tools import TrainConfig, Trainer, num_parameters

    def qubo_register_coords(Q: np.ndarray, device: RydbergDevice) -> list:
        """Compute coordinates for register."""

        def evaluate_mapping(new_coords, *args):  # type: ignore
            """Cost function to minimize. Ideally, the pairwise.

            distances are conserved
            """
            Q, shape = args
            new_coords = np.reshape(new_coords, shape)
            interaction_coeff = device.coeff_ising
            new_Q = squareform(interaction_coeff / pdist(new_coords) ** 6)
            return np.linalg.norm(new_Q - Q)

        shape = (len(Q), 2)
        np.random.seed(0)
        x0 = np.random.random(shape).flatten()
        res = minimize(
            evaluate_mapping,
            x0,
            args=(Q, shape),
            method="Nelder-Mead",
            tol=1e-6,
            options={"maxiter": 200000, "maxfev": None},
        )
        return [(x, y) for (x, y) in np.reshape(res.x, (len(Q), 2))]

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # QUBO problem weights (real-value symmetric matrix)
    Q = np.array(
        [
            [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ]
    )

    # Loss function to guide the optimization routine
    def loss(model: QuantumModel, *args) -> tuple[torch.Tensor, dict]:  # type: ignore
        to_arr_fn = lambda bitstring: np.array(list(bitstring), dtype=int)
        cost_fn = lambda arr: arr.T @ Q @ arr  # type: ignore
        samples = model.sample({}, n_shots=1000)[0]
        cost_fn = sum(samples[key] * cost_fn(to_arr_fn(key)) for key in samples)
        return torch.tensor(cost_fn / sum(samples.values())), {}  # type: ignore

    # Device specification and atomic register
    device = RydbergDevice(rydberg_level=70)

    reg = Register.from_coordinates(qubo_register_coords(Q, device), device_specs=device)

    # Analog variational quantum circuit
    layers = 2
    block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(layers)])
    circuit = QuantumCircuit(reg, block)

    model = QuantumModel(circuit)
    initial_counts = model.sample({}, n_shots=100)[0]

    Trainer.set_use_grad(False)

    config = TrainConfig(max_iter=10)

    optimizer = ng.optimizers.NGOpt(budget=config.max_iter, parametrization=num_parameters(model))

    trainer = Trainer(model, optimizer, config, loss)

    trainer.fit()

    optimal_counts = model.sample({}, n_shots=100)[0]
