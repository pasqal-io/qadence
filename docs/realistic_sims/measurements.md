Sample-based measurement protocols are fundamental tools for the prediction and estimation of a quantum state as the result of NISQ programs executions. Their resource efficient implementation is a current and active research field. Qadence offers two main measurement protocols: _quantum state tomography_ and _classical shadows_.

## Quantum state tomography

The fundamental task of quantum state tomography is to learn an approximate description of an output quantum state described by a density matrix $\rho$, by repeatedly measuring a copy on a chosen basis. The expectation value of a given observable $\mathcal{\hat{O}}$, is calculated with $\langle \mathcal{\hat{O}} \rangle$
