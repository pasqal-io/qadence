# Qadence

**Qadence** is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable qubit interaction defined on _**arbitrary register topologies**_ realizable on neutral atom devices.

[![pre-commit](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml)
[![tests](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml)
[![Build documentation](https://github.com/pasqal-io/qadence/actions/workflows/build_docs.yml/badge.svg)](https://pasqal-io.github.io/qadence/latest)

Documentation can be found [here](https://pasqal-io.github.io/qadence/latest).

## Feature highlights

* A [block-based system](tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and scalable manner, inspired by the Julia quantum SDK
  [Yao.jl](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](digital_analog_qc/analog-basics.md) to work with _**interacting neutral-atom qubit systems**_
  using [arbitrary registers topologies](tutorials/register.md).

* An intuitive [expression-based system](tutorials/parameters.md) developed on top of the symbolic library [Sympy](https://www.sympy.org/en/index.html) to construct _**parametric quantum programs**_ easily.

* [High-order generalized parameter shift rules](link to psr tutorial) for _**differentiating parametrized quantum operations**_.

* Out-of-the-box _**automatic differentiability**_ of quantum programs with [PyTorch](https://pytorch.org/) integration.

* _**Efficient execution**_ on a variety of different purpose backends: from state vector simulators to tensor network emulators and real devices.

## Installation guide

Qadence can be installed from PyPI with `pip` as follows:

```bash
pip install qadence
```

The default backend for Qadence is [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator for digital-analog simulation. It is possible to install additional backends and the circuit visualization library using the following extras:

* `braket`: the [Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) backend.
* `pulser`: the [Pulser](https://github.com/pasqal-io/Pulser) backend for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: to display diagrammatically quantum circuits.

by running:

```bash
pip install qadence[braket, pulser, visualization]
```

!!! warning
    In order to correctly install the `visualization` extra, the `graphviz` package needs to be installed
    in your system:

    ```bash
    # on Ubuntu
    sudo apt install graphviz

    # on MacOS
    brew install graphviz

    # via conda
    conda install python-graphviz
    ```

## Citation

If you use Qadence for a publication, we kindly ask you to cite our work using the following BibTex entry:

```
@misc{qadence2023pasqal,
  url = {https://github.com/pasqal-io/qadence},
  title = {Qadence: {A} {D}igital-analog quantum programming interface.},
  year = {2023}
}
```
