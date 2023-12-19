Sample-based measurement protocols are fundamental tools for the prediction and estimation of a quantum state as the result of NISQ programs executions. Their resource efficient implementation is a current and active research field. Qadence offers two main measurement protocols: _quantum state tomography_ and _classical shadows_.

## Quantum state tomography

The fundamental task of quantum state tomography is to learn an approximate description of an output quantum state described by a density matrix $\rho$, from repeated measurements of copies on a chosen basis. To do so, $\rho$ is expanded in a basis of observables (the tomography step). For a given observable $\hat{\mathcal{O}}$, the expectation value is calculated with $\langle \hat{\mathcal{O}} \rangle=\textrm{Tr}(\hat{\mathcal{O}}\rho)$. A number of measurement repetitions in a suitable basis is then required to estimate $\langle \hat{\mathcal{O}} \rangle$.

The fundamental drawback with quantum state tomography is the measurements scaling for the retrieval of the classical expression for a $n$-qubit quantum state as $2^n \times 2^n$, together with a large amount of classical post-processing. 
