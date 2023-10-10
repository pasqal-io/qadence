# Digital-Analog Quantum Computation

_**Digital-analog quantum computation**_ (DAQC) is a universal quantum computing
paradigm[^1], based on two primary computations:

- Fast single-qubit operations (digital).
- Multi-partite entangling operations acting on all qubits (analog).

The DAQC paradigm is typically implemented on quantum computing hardware based on neutral-atoms where both these computations are realizable.

## Digital-Analog Emulation

Qadence simplifies the execution of DAQC programs on either emulated or real neutral-atom devices
by providing a simplified interface for customizing interactions and interfacing
with pulse-level programming in `Pulser`[^3].

## Digital-Analog Transformation

Furthermore, the essence of digital-analog computation is the ability to represent any analog operation, _i.e._ any arbitrary Hamiltonian, using an
auxiliary device-amenable Hamiltonian, such as the ubiquitous Ising model[^2]. This is at the core of the DAQC implementation in Qadence.

## References

[^1]: [Dodd _et al._, Universal quantum computation and simulation using any entangling Hamiltonian and local unitaries, PRA 65, 040301 (2002).](https://arxiv.org/abs/quant-ph/0106064)

[^2]: [Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays](https://pulser.readthedocs.io/en/stable/)

[^3]: [Parra-Rodriguez _et al._, Digital-Analog Quantum Computation, PRA 101, 022305 (2020).](https://arxiv.org/abs/1812.03637)
