## Installation guide

Qadence can be installed from PyPI with `pip` as follows:

```bash
pip install qadence
```

The default backend for Qadence is [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator for digital-analog simulation. It is possible to install additional backends and the circuit visualization library using the following extras:

* `braket`: the [Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) backend.
* `pulser`: the [Pulser](https://github.com/pasqal-io/Pulser) backend for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: to display diagrammatically quantum circuits.

To just get qadence with the `pyqtorch` backend, simply run:

```bash
pip install qadence
```

To install other backends or the visualization tool, please use:

```bash
pip install "qadence[braket,pulser,visualization]"
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

## Install from source

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `qadence` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

!!! warning
    `hatch` will not combine nicely with other environment managers such Conda. If you want to use Conda,
    install it from source using `pip`:

    ```bash
    # within the Conda environment
    python -m pip install -e .
    ```

## Citation

If you use Qadence for a publication, we kindly ask you to cite our work using the following BibTex entry:

```latex
@article{qadence2024pasqal,
  title = {Qadence: a differentiable interface for digital-analog programs.},
  author={Dominik Seitz and Niklas Heim and João P. Moutinho and Roland Guichard and Vytautas Abramavicius and Aleksander Wennersteen and Gert-Jan Both and Anton Quelle and Caroline de Groot and Gergana V. Velikova and Vincent E. Elfving and Mario Dagrada},
  journal={arXiv:2401.09915},
  url = {https://github.com/pasqal-io/qadence},
  year = {2024}
}
```
