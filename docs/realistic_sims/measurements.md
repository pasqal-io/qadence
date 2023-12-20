Sample-based measurement protocols are fundamental tools for the prediction and estimation of a quantum state as the result of NISQ programs executions. Their resource efficient implementation is a current and active research field. Qadence offers two main measurement protocols: _quantum state tomography_ and _classical shadows_.

## Quantum state tomography

The fundamental task of quantum state tomography is to learn an approximate classical description of an output quantum state described by a density matrix $\rho$, from repeated measurements of copies on a chosen basis. To do so, $\rho$ is expanded in a basis of observables (the tomography step) and for a given observable $\hat{\mathcal{O}}$, the expectation value is calculated with $\langle \hat{\mathcal{O}} \rangle=\textrm{Tr}(\hat{\mathcal{O}}\rho)$. A number of measurement repetitions in a suitable basis is then required to estimate $\langle \hat{\mathcal{O}} \rangle$.

The main drawback is the scaling in measurements for the retrieval of the classical expression for a $n$-qubit quantum state as $2^n \times 2^n$, together with a large amount of classical post-processing.

For an observable expressed as a Pauli string $\hat{\mathcal{P}}$, the expectation value for a state $|\psi \rangle$ can be derived as:

$$
\langle \hat{\mathcal{P}} \rangle=\langle \psi | \hat{\mathcal{P}} |\psi \rangle=\langle \psi | \hat{\mathcal{R}}^\dagger \hat{\mathcal{D}} \hat{\mathcal{R}} |\psi \rangle
$$

The operator $\hat{\mathcal{R}}$ diagonalizes $\hat{\mathcal{P}}$ and rotates the state into an eigenstate in the computational basis. Therefore, $\hat{\mathcal{R}}|\psi \rangle=\sum\limits_{z}a_z|z\rangle$ and the expectation value can finally be expressed as:


$$
\langle \hat{\mathcal{P}} \rangle=\sum_{z,z'}\langle z |\bar{a}_z\hat{\mathcal{D}}a_{z'}|z'\rangle = \sum_{z}|a_z|^2(-1)^{\phi_z(\hat{\mathcal{P}})}
$$


In Qadence, running a tomographical experiment is made simple by defining a `Measurements` object that captures all options for execution:

```python exec="on" source="material-block" session="measurements" result="json"
from torch import tensor
from qadence import hamiltonian_factory, BackendName, DiffMode
from qadence import Parameter, chain, kron, RX, RY, Z, QuantumCircuit
from qadence.measurements import Measurements

# Define parameters for a circuit.
theta1 = Parameter("theta1", trainable=False)
theta2 = Parameter("theta2", trainable=False)
theta3 = Parameter("theta3", trainable=False)
theta4 = Parameter("theta4", trainable=False)

blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3), RY(1, theta4)),
)

values = {
    "theta1": tensor([0.5]),
    "theta2": tensor([1.5]),
    "theta3": tensor([2.0]),
    "theta4": tensor([2.5]),
}

# Create a circuit and an observable.
circuit = QuantumCircuit(2, blocks)
observable = hamiltonian_factory(2, detuning=Z)

# Create a model.
model = QuantumModel(
    circuit=circuit,
    observable=observable,
    backend=BackendName.PYQTORCH,
    diff_mode=DiffMode.GPSR,
)

# Define a measurement protocol by passing the shot budget as an option.
tomo_options = {"n_shots": 100000}
tomo_measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options=tomo_options)

# Get the exact expectation value.
exact_values = model.expectation(
	values=values,
)

# Run the tomography experiment.
estimated_values_tomo = model.expectation(
    values=values,
    measurement=tomo_measurement,
)

print(f"Exact expectation value = {exact_values}") # markdown-exec: hide
print(f"Estimated expectation value tomo = {estimated_values_tomo}") # markdown-exec: hide
```


## Classical shadows

Recently, a much less resource demanding protocol based on _classical shadows_ has been proposed[^1]. It combines ideas from shadow tomography[^2] and randomized measurement protocols capable of learning a classical shadow of an unknown quantum state $\rho$. It relies on deliberately discarding the full classical characterization of the quantum state, and instead focuses on accurately predicting a restricted set of properties that provide efficient protocols for the study of the system.

A random measurement consists of applying random unitary rotations before a fixed measurement on each copy of a state. Appropriately averaging over these measurements produces an efficient estimator for the expectation value of an observable. This protocol therefore creates a robust classical representation of the quantum state or classical shadow. The captured measurement information is then reuseable for multiple purposes, _i.e._ any observable expected value and available for noise mitigation postprocessing.

A classical shadow is therefore an unbiased estimator of a quantum state $\rho$. Such an estimator is obtained with the following procedure[^1]: first, apply a random unitary gate $U$ to rotate the state: $\rho \rightarrow U \rho U^\dagger$ and then perform a basis measurement to obtain a $n$-bit measurement $|\hat{b}\rangle \in \{0, 1\}^n$. Both unitary gates $U$ and the measurement outcomes $|\hat{b}\rangle$ are stored on a classical computer for postprocessing v $U^\dagger |\hat{b}\rangle\langle \hat{b}|U$, a classical snapshot of the state $\rho$. The whole procedure can be seen as a quantum channel $\mathcal{M}$ that maps the initial unknown quantum state $\rho$ to the average result of the measurement protocol:

$$
\mathbb{E}[U^\dagger |\hat{b}\rangle\langle \hat{b}|U] = \mathcal{M}(\rho) \Rightarrow \rho = \mathbb{E}[\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)]
$$

It is worth noting that the single classical snapshot $\hat{\rho}=\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)$ equals $\rho$ in expectation: $\mathbb{E}[\hat{\rho}]=\rho$ despite $\mathcal{M}^{-1}$ not being a completely positive map. Repeating this procedure $N$ times results in an array of $N$ independent, classical snapshots of $\rho$ called the classical shadow:

$$
S(\rho, N) = \{ \hat{\rho}_1=\mathcal{M}^{-1}(U_1^\dagger |\hat{b}_1\rangle\langle \hat{b}_1|U_1),\cdots,\hat{\rho}_N=\mathcal{M}^{-1}(U_N^\dagger |\hat{b}_N\rangle\langle \hat{b}_N|U_N)\}
$$

Along the same lines as the example before, estimating the expectation value using classical shadows in Qadence only requires to pass the right set of parameters to the `Measurements` object:


```python exec="on" source="material-block" session="measurements" result="json"

# Classical shadows are defined up to some accuracy and confidence.
shadow_options = {"accuracy": 0.1, "confidence": 0.1}  # Shadow size N=54400.
shadow_measurement = Measurements(protocol=Measurements.SHADOW, options=shadow_options)

# Run the experiment with classical shadows.
estimated_values_shadow = model.expectation(
    values=values,
    measurement=shadow_measurement,
)

print(f"Estimated expectation value shadow = {estimated_values_shadow}") # markdown-exec: hide
```


## References

[^1]: [Hsin-Yuan Huang, Richard Kueng and John Preskill, Predicting Many Properties of a Quantum System from Very Few Measurements (2020)](https://arxiv.org/abs/2002.08953)

[^2]: S. Aaronson. Shadow tomography of quantum states. In _Proceedings of the 50th Annual A ACM SIGACT Symposium on Theory of Computing_, STOC 2018, pages 325–338, New York, NY, USA, 2018. ACM
