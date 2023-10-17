# Qadence

**Qadence** is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable qubit interaction defined on _**arbitrary register topologies**_ realizable on neutral atom devices.

[![Linting](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml)
[![Tests](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml)
[![Documentation](https://github.com/pasqal-io/qadence/actions/workflows/build_docs.yml/badge.svg)](https://pasqal-io.github.io/qadence/latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pypi](https://badge.fury.io/py/qadence.svg)](https://pypi.org/project/qadence/)

## Feature highlights

* A [block-based system](docs/tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and scalable manner, inspired by the Julia quantum SDK
  [Yao.jl](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](docs/digital_analog_qc/analog-basics.md) to work with _**interacting neutral-atom qubit systems**_
  using [arbitrary registers topologies](docs/tutorials/register.md).

* An intuitive [expression-based system](docs/tutorials/parameters.md) developed on top of the symbolic library [Sympy](https://www.sympy.org/en/index.html) to construct _**parametric quantum programs**_ easily.

* [High-order generalized parameter shift rules](docs/advanced_tutorials/differentiability.md) for _**differentiating parametrized quantum operations**_.

* Out-of-the-box _**automatic differentiability**_ of quantum programs with [PyTorch](https://pytorch.org/) integration.

* _**Efficient execution**_ on a variety of different purpose backends: from state vector simulators to tensor network emulators and real devices.

## Installation guide

Qadence is available on [PyPI](https://pypi.org/project/qadence/) and can be installed using `pip` as follows:

```bash
pip install qadence
```

The default, pre-installed backend for Qadence is [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator for digital-analog simulation. It is possible to install additional backends and the circuit visualization library using the following extras:

* `braket`: the [Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) backend.
* `pulser`: the [Pulser](https://github.com/pasqal-io/Pulser) backend for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: to display diagrammatically quantum circuits.

To install other backends or the visualization tool, please use:

```bash
pip install "qadence[braket,pulser,visualization]"
```

**IMPORTANT**
Before installing `qadence` with the `visualization` extra, make sure to install the `graphviz` package
on your system:

```bash
# on Ubuntu
sudo apt install graphviz

# on MacOS
brew install graphviz

# via conda
conda install python-graphviz
```

## Contributing

Before making a contribution, please review our [code of conduct](docs/CODE_OF_CONDUCT.md).
To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/qadence/issues).
To contribute to `qadence`, please refer to [contributing guidelines](docs/CONTRIBUTING.md).

### Install from source

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `qadence` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

**WARNING**
`hatch` will not combine nicely with other environment managers such as Conda. If you still want to use Conda,
install it from source using `pip`:

```bash
# within the Conda environment
python -m pip install -e .
```

## Citation

If you use Qadence for a publication, we kindly ask you to cite our work using the following BibTex entry:

```latex
@misc{qadence2023pasqal,
  url = {https://github.com/pasqal-io/qadence},
  title = {Qadence: {A} {D}igital-analog quantum programming interface.},
  year = {2023}
}
```

## License
Qadence is a free and open source software package, released under the Apache License, Version 2.0.
