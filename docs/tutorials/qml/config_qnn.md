In `qadence`, the [`QNN`][qadence.ml_tools.models.QNN] is a variational quantum model that can potentially take multi-dimensional input.

The `QNN` class needs a circuit and a list of observables; the number of feature parameters in the input circuit determines the number of input features (i.e. the dimensionality of the classical data given as input) whereas the number of observables determines the number of outputs of the quantum neural network.

The circuit has two parts, the feature map and the ansatz. The feature map is responsible for encoding the input data into the quantum state, while the ansatz is responsible for the variational part of the model. In addition, a third part of the QNN is the observables, which is (a list of) operators that are measured at the end of the circuit.

In [QML Constructors](../../content/qml_constructors.md) we have seen how to construct the feature map and the ansatz. In this tutorial, we will see how to do the same using configs.

One convenient way to construct these three parts of the model is to use the config classes, namely,
[`ObservableConfig`][qadence.constructors.hamiltonians.ObservableConfig], [`FeatureMapConfig`][qadence.ml_tools.config.FeatureMapConfig], [`AnsatzConfig`][qadence.ml_tools.config.AnsatzConfig]. These classes allow you to specify the type of circuit and the parameters of the circuit in a structured way.

## Defining the Observable

The model output is the expectation value of the defined observable(s). We use the `ObservableConfig` class to specify the observable.

It can be used to create Hamiltonians with 2-qubit interactions and single-qubit detunings. Any Hamiltonian supported by [`hamiltonian_factory`][qadence.constructors.hamiltonians.hamiltonian_factory] can be specified as an observable.
For example, suppose we want to measure the Z operator:

```python exec="on" source="material-block" session="config" html="1"
from qadence import create_observable, ObservableConfig, Z

observable_config = ObservableConfig(
    detuning=Z,
    interaction = None,
    scale = 2.0,
    shift=-1.0,
)

observable = create_observable(register=4, config=observable_config)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(observable)) # markdown-exec: hide
```

We have specified the observable Hamiltonian to be one with $Z$-detuning. The result is linearly scaled by 2.0 and shifted by -1.0. The shift or the scale can optionally also be a [VariationalParameter][qadence.parameters.VariationalParameter]

It is also possible to import some common Hamiltonians, such as `total_magnetization_config`, `zz_hamiltonian_config` and, `ising_hamiltonian_config`.

For example, the total magnetization configuration:

```python exec="on" source="material-block" session="config" html="1"
from qadence import create_observable
from qadence.constructors import total_magnetization_config

observable_total_magnetization  = create_observable(register=4, config=total_magnetization_config())
from qadence.draw import html_string # markdown-exec: hide
```

Alternatively, you can define the observable as a list of observables, in which case the QNN will output a list of values.

### Scaling and Shifting the QNN Output
For any observable, by appropriately choosing the scale $\alpha$ and shift $\beta$, you can constrain the QNN output within a desired range. This is particularly useful for normalizing measurements or ensuring that values remain within a meaningful interval for optimization.
To accomplish this, you need to determine the maximum and minimum values that the QNN output can take. For an observable, these extreme values are the two extreme eigenvalues $\lambda_{max}$ and $\lambda_{min}$ of the concerned Hamiltonian. Using these values, you can set the scale $\alpha$ and shift $\beta$ so that the QNN output is mapped to a specific range $[a,b]$:

$$\alpha = \frac{b-a}{\lambda_{max}-\lambda_{min}}$$

$$\beta = \frac{a\lambda_{max}-b\lambda_{min}}{\lambda_{max}-\lambda_{min}}$$

This transformation ensures that:

$$ a \leq \alpha \lambda + \beta \leq b,\quad\forall \lambda \in [\lambda_{min},\lambda_{max}] $$

For full details on the `ObservableConfig` class, see the [API documentation][qadence.constructors.hamiltonians.ObservableConfig].

## Defining the Feature Map

Let us say we want to build a 4-qubit QNN that takes two inputs, namely, the $x$ and the $y$ coordinates of a point in the plane. We can use the `FeatureMapConfig` class to specify the feature map.

```python exec="on" source="material-block" session="config" html="1"
from qadence import BasisSet, chain, create_fm_blocks, FeatureMapConfig, ReuploadScaling

fm_config = FeatureMapConfig(
    num_features=2,
    inputs = ["x", "y"],
    basis_set=BasisSet.CHEBYSHEV,
    reupload_scaling=ReuploadScaling.TOWER,
    feature_range={
        "x": (-1.0, 1.0),
        "y": (0.0, 1.0),
    },
)

fm_blocks = create_fm_blocks(register=4, config=fm_config)
feature_map = chain(*fm_blocks)
print(html_string(feature_map)) # markdown-exec: hide
```

We have specified that the feature map should take two features, and have named the [`FeatureParameter`][qadence.parameters.FeatureParameter] "x" and "y" respectively. Both these parameters are encoded using the Chebyshev basis set, and the reupload scaling is set to `ReuploadScaling.TOWER`. One can optionally add the basis and the reupload scaling for each parameter separately.

The `feature_range` parameter is a dictionary that specifies the range of values that each feature comes from. This is useful for scaling the input data to the range that the encoding function can handle. In default case, this range is mapped to the target range of the Chebyshev basis set which is $[-1, 1]$. One can also specify the target range for each feature separately.

For full details on the `FeatureMapConfig` class, see the [API documentation][qadence.ml_tools.config.FeatureMapConfig].

## Defining the Ansatz

The next part of the QNN is the ansatz. We use `AnsatzConfig` class to specify the type of ansatz.

Let us say, we want to follow this feature map with 2 layers of hardware efficient ansatz.

```python exec="on" source="material-block" session="config" html="1"
from qadence import AnsatzConfig, AnsatzType, create_ansatz, Strategy

ansatz_config = AnsatzConfig(
    depth=2,
    ansatz_type=AnsatzType.HEA,
    ansatz_strategy=Strategy.DIGITAL,
)

ansatz = create_ansatz(register=4, config=ansatz_config)

print(html_string(ansatz)) # markdown-exec: hide
```

We have specified that the ansatz should have a depth of 2, and the ansatz type is "hea" (Hardware Efficient Ansatz). The ansatz strategy is set to "digital", which means digital gates are being used. One could alternatively use "analog" or "rydberg" as the ansatz strategy.

For full details on the `AnsatzConfig` class, see the [API documentation][qadence.ml_tools.config.AnsatzConfig].

## Defining the QNN from the Configs

To build the QNN, we can now use the `QNN` class as a `QuantumModel` subtype. In addition to the feature map, ansatz and the observable configs, we can also specify options such as the `backend`, `diff_mode`, etc. For full details on the `QNN` class, see the [API documentation][qadence.ml_tools.models.QNN] or the documentation on the config constructor [here][qadence.ml_tools.models.QNN.from_configs].

```python exec="on" source="material-block" session="config" html="1"
from qadence import BackendName, DiffMode, QNN

qnn = QNN.from_configs(
    register=4,
    obs_config=observable_config,
    fm_config=fm_config,
    ansatz_config=ansatz_config,
    backend=BackendName.PYQTORCH,
    diff_mode=DiffMode.AD,
)

print(html_string(qnn)) # markdown-exec: hide
```
