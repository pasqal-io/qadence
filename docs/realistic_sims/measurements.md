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


## Classical shadows

Recently, a much less resource demanding protocol based on _classical shadows_[^1] has been proposed. It combines ideas from shadow tomography[^2] and randomized measurement protocols capable of learning a classical shadow of an unknown quantum state $\rho$. It relies on deliberately discarding the full classical characterization of the quantum state, and instead focuses on accurately predicting a restricted set of properties that provide efficient protocols for the study of the system, _i.e._ the expectation values of observables.

A random measurement consists of applying random unitary rotations before a fixed measurement on each copy of a state. Appropriately averaging over these measurements produces an efficient estimator for the expectation value of an observable. This protocol therefore creates a robust classical representation of the quantum state or classical shadow. The captured measurement information is then reuseable for multiple purposes, _i.e._ any observable expected value and available for noise mitigation postprocessing.

A classical shadow is therefore an unbiased estimator of a quantum state $\rho$. Such an estimator is obtained with the following procedure[^1]: first, apply a random unitary gate $U$ to rotate the state: $\rho \rightarrow U \rho U^\dagger$ and then perform a basis measurement to obtain a $n$-bit measurement $|\hat{b}\rangle \in \{0, 1\}^n$. Both unitary gates $U$ and the measurement outcomes $|\hat{b}\rangle$ are stored on a classical computer for postprocessing under the form $U^\dagger |\hat{b}\rangle\langle \hat{b}|U$, a classical snapshot of the state $\rho$. The whole procedure can be seen as a quantum channel $\mathcal{M}$ that maps the initial unknown quantum state $\rho$ to the average result of the measurement protocol:

$$
\mathbb{E}[U^\dagger |\hat{b}\rangle\langle \hat{b}|U] = \mathcal{M}(\rho) \Rightarrow \rho = \mathbb{E}[\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)]
$$

It is worth noting that the single classical snapshot $\hat{\rho}=\mathcal{M}^{-1}(U^\dagger |\hat{b}\rangle\langle \hat{b}|U)$ equals $\rho$ in expectation: $\mathbb{E}[\hat{\rho}]=\rho$ despite $\mathcal{M}^{-1}$ not being a completely positive map.



## References

[^1]: [Hsin-Yuan Huang, Richard Kueng and John Preskill, Predicting Many Properties of a Quantum System from Very Few Measurements (2020)](https://arxiv.org/abs/2002.08953)

[^2]: S. Aaronson. Shadow tomography of quantum states. In _Proceedings of the 50th Annual A ACM SIGACT Symposium on Theory of Computing_, STOC 2018, pages 325–338, New York, NY, USA, 2018. ACM
