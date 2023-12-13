Beyond running noisy simulations, Qadence offers a number of noise mitigation techniques to achieve better accuracy of simulation outputs. Currently, mitigation addresses readout errors and depolarizing and dephasing noise for analog blocks.

## Readout error mitigation

The complete implementation of the mitigation technique is to measure $T$ and classically apply $T^{−1}$ to measured probability distributions. However there are several limitations of this approach:

- The complete implementation requires $2^n$ characterization experiments (probability measurements), which is not scalable. The classical processing of the calibration data is also inefficient.
- The matrix $T$ may become singular for large $n$, preventing direct inversion.
- The inverse $T^{−1}$ might not be a stochastic matrix, meaning that it can produce negative corrected probabilities.
- The correction is not rigorously justified, so we cannot be sure that we are only removing SPAM errors and not otherwise corrupting an estimated probability distribution.



