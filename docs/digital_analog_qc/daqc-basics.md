# Digital-Analog Quantum Computation

_**Digital-analog quantum computation**_ (DAQC) is a universal quantum computing
paradigm [^1]. The main ingredients of a DAQC program are:

- Fast single-qubit operations (digital).
- Multi-partite entangling operations acting on all qubits (analog).

Analog operations are typically assumed to follow device-specific interacting qubit Hamiltonians, such as the Ising Hamiltonian [^2]. The most common realization of the DAQC paradigm is on neutral atoms quantum computing platforms.

## Digital-Analog Emulation

Qadence simplifies the execution of DAQC programs on neutral-atom devices
by providing a simplified interface for adding interaction and interfacing
with pulse-level programming in `pulser`[^3].


## DAQC Transform

Furthermore, essential to digital-analog computation is the ability to represent an arbitrary Hamiltonian
with the evolution of a fixed and device-amenable Hamiltonian. Such a transform was described in the
DAQC[^2] paper for ZZ interactions, which is natively implemented in Qadence.

## References

[^1]: [Dodd et al., Universal quantum computation and simulation using any entangling Hamiltonian and local unitaries, PRA 65, 040301 (2002).](https://arxiv.org/abs/quant-ph/0106064)

[^2]: [Parra-Rodriguez et al., Digital-Analog Quantum Computation, PRA 101, 022305 (2020).](https://arxiv.org/abs/1812.03637)

[^3]: [Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays](https://pulser.readthedocs.io/en/stable/)
