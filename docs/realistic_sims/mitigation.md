Beyond running noisy simulations, Qadence offers a number of noise mitigation techniques to achieve better accuracy of simulation outputs. Currently, mitigation addresses readout errors and depolarizing and dephasing noise for analog blocks.

## Readout error mitigation

The complete implementation of the mitigation technique is to measure $T$ and classically apply $T^{−1}$ to measured probability distributions. However there are several limitations of this approach:

- The complete implementation requires $2^n$ characterization experiments (probability measurements), which is not scalable. The classical processing of the calibration data is also inefficient.
- The matrix $T$ may become singular for large $n$, preventing direct inversion.
- The inverse $T^{−1}$ might not be a stochastic matrix, meaning that it can produce negative corrected probabilities.
- The correction is not rigorously justified, so we cannot be sure that we are only removing SPAM errors and not otherwise corrupting an estimated probability distribution.

Qadence relies on the assumption of _uncorrelated_ readout errors:

$$
T=T_1\otimes T_2\otimes \dots \otimes T_n
$$

for which the inversion is straightforward:

$$
T^{-1}=T_1^{-1}\otimes T_2^{-1}\otimes \dots \otimes T_n^{-1}
$$

However, even for a reduced $n$ the third limitation holds. This can be avoided by reformulating into a minimization problem[^1]:

$$
\lVert Tp_{\textrm{corr}}-p_{\textrm{raw}}\rVert_{2}^{2}
$$

subjected to physicality constraints $0 \leq p_{corr}(x) \leq 1$ and $\lVert p_{corr} \rVert = 1$. At this point, two methods are implemented to solve this problem. The first one relies on solving using standard optimization tools, the second on Maximum-Likelihood Estimation[^2]. In Qadence, this can be user defined using the mitigation protocol:

```python exec="on" source="material-block" session="mitigation" result="json"
from qadence import QuantumModel, QuantumCircuit, kron, H, Z
from qadence import hamiltonian_factory
from qadence.noise import Noise
from qadence.mitigations import Mitigations
from qadence.types import ReadOutOptimization

# Simple circuit and observable construction.
block = kron(H(0), Z(1))
circuit = QuantumCircuit(2, block)
observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

# Construct a quantum model.
model = QuantumModel(circuit=circuit, observable=observable)

# Define a noise model to use:
noise = Noise(protocol=Noise.READOUT)
# Define the mitigation method solving the minimization problem:
options={"optimization_type": ReadOutOptimization.CONSTRAINED}  # ReadOutOptimization.MLE for the alternative method.
mitigation = Mitigations(protocol=Mitigations.READOUT, options=options)

# Run noiseless, noisy and mitigated simulations.
n_shots = 100
noiseless_samples = model.sample(n_shots=n_shots)
noisy_samples = model.sample(noise=noise, n_shots=n_shots)
mitigated_samples = model.sample(
    noise=noise, mitigation=mitigation, n_shots=n_shots
)

print(f"noiseless {noiseless_samples}")
print(f"noisy {noisy_samples}")
print(f"mitigated {mitigated_samples}")
```

## [WIP] Zero-noise extrapolation for analog blocks

Zero-noise extrapolation (ZNE) is an error mitigation technique in which an expectation value is computed at different noise levels and, as a second step, the ideal expectation value is inferred by extrapolating the measured results to the zero-noise limit. In digital computing, this is typically implemented by "folding" the circuit and its dagger to artificially increase the noise through sequences of identities[^3]. In the analog ZNE variation, analog blocks are time stretched to again artificially increase noise[^3].


## References

[^1]: [Michael R. Geller and Mingyu Sun, Efficient correction of multiqubit measurement errors, (2020)](https://arxiv.org/abs/2001.09980)

[^2]: [Smolin _et al._, Maximum Likelihood, Minimum Effort, (2011)](https://arxiv.org/abs/1106.5458)

[^3]: [Mitiq: What's the theory behind ZNE?](https://mitiq.readthedocs.io/en/stable/guide/zne-5-theory.html)
