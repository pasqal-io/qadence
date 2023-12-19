Sample-based measurement protocols are fundamental tools for the prediction and estimation of a quantum state as the result of NISQ programs executions. Their resource efficient implementation is a current and active research field. Qadence offers two main measurement protocols: _quantum state tomography_ and _classical shadows_.

## Quantum state tomography

The fundamental task of quantum state tomography is to learn an approximate description of an output quantum state described by a density matrix $\rho$, from repeated measurements of copies on a chosen basis. To do so, $\rho$ is expanded in a basis of observables (the tomography step). For a given observable $\mathcal{\hat{O}}$, the expectation value is calculated with $\langle \mathcal{\hat{O}} \rangle=\textrm{Tr}(\mathcal{\hat{O}}\rho)$. A number of measurement repetitions in a suitable basis is then required to estimate $\langle \mathcal{\hat{O}} \rangle$.
