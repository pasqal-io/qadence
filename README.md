# Qadence

*Qadence* is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable interaction defined on _**arbitrary qubit register layouts**_.

[![pre-commit](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml)
[![tests](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml)

## Feature highlights

* A [block-based system](tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and extensible manner. Heavily inspired by
  [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](digital_analog_qc/analog-basics.md) to work with _**interacting qubit systems**_
  using [arbitrary qubit registers](tutorials/register.md).

* Intuitive, [expression-based system](tutorials/parameters.md) built on top of `sympy` to construct
  _**parametric quantum programs**_.

* [Higher-order generalized parameter shift](link to psr tutorial) rules for _**differentiating
  arbitrary quantum operations**_ on real hardware.

* Out-of-the-box automatic differentiability of quantum programs using [https://pytorch.org](https://pytorch.org)

* `QuantumModel`s to make `QuantumCircuit`s differentiable and runnable on a variety of different
  backends like state vector simulators, tensor network emulators and real devices.

Documentation can be found here: [https://pasqal-qadence.readthedocs-hosted.com/en/latest](https://pasqal-qadence.readthedocs-hosted.com/en/latest).


## Citation

If you use Qadence for a publication, we kindly ask you to cite our work using the bibtex citation:

```
@misc{qadence2023pasqal,
  url = {https://github.com/pasqal-io/qadence},
  title = {Qadence: {A} {D}igital-analog quantum programming interface.},
  year = {2023}
}
```


## Installation guide

Qadence can be install with `pip` as follows:

```bash
pip install qadence[pulser,visualization]
```

The default backend for qadence is [`pyqtorch`](https://github.com/pasqal-io/pyqtorch) (a
differentiable state vector simulator).  You can install one or all of the following additional
backends and the circuit visualization library using the following extras:

* `braket`: install the Amazon Braket quantum backend
* `emu-c`: install the Pasqal circuit tensor network emulator EMU-C
* `pulser`: install the [Pulser](https://github.com/pasqal-io/Pulser) backend. Pulser is a framework
  for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: install the library necessary to visualize quantum circuits.

!!! warning
    In order to correctly install the "visualization" extra, you need to have `graphviz` installed
    in your system. This depends on the operating system you are using:

    ```bash
    # on Ubuntu
    sudo apt install graphviz

    # on MacOS
    brew install graphviz

    # via conda
    conda install python-graphviz
    ```
---
