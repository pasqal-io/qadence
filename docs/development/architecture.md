Qadence as a software library mixes functional and object-oriented programming. We do that by maintaining core objects and operating on them with functions.

Furthermore, Qadence strives at keeping the lower level abstraction layers for automatic differentiation and quantum computation
fully stateless while only the frontend layer which is the main user-facing interface is stateful.

!!! note "**Code design philosopy**"
    Functional, stateless core with object-oriented, stateful user interface.

## Abstraction layers

In Qadence there are 4 main objects spread across 3 different levels of abstraction:

* **Frontend layer**: The user facing layer and encompasses two objects:
    * [`QuantumCircuit`][qadence.circuit.QuantumCircuit]: A class representing an abstract quantum
      circuit not tight not any particular framework. Parameters are represented symbolically using
      `sympy` expressions.
    * [`QuantumModel`][qadence.QuantumModel]: The models are higher-level abstraction
      providing an interface for executing different kinds of common quantum computing models such
      quantum neural networks (QNNs), quantum kernels etc.

* **Differentiation layer**: Intermediate layer has the purpose of integrating quantum
  computation with a given automatic differentiation engine. It is meant to be purely stateless and
  contains one object:
    * [`DifferentiableBackend`][qadence.engines.torch.DifferentiableBackend]:
      An abstract class whose concrete implementation wraps a quantum backend and make it
      automatically differentiable using different engines (e.g. PyTorch or Jax).
      Note, that today only PyTorch is supported but there is plan to add also a Jax
      differentiable backend which will require some changes in the base class implementation.

* **Quantum layer**: The lower-level layer which directly interfaces with quantum emulators
  and processing units. It is meant to be purely stateless and it contains one base object which is
  specialized for each supported backend:
    * [`Backend`][qadence.backend.Backend]: An abstract class whose concrete implementation
      enables the execution of quantum circuit with a variety of quantum backends (normally non
      automatically differentiable by default) such as PyQTorch, Pulser or Braket.


## Main components

### `QuantumCircuit`

We consider `QuantumCircuit` to be an abstract object, i.e. it is not tied to any backend. However, it blocks are even more abstract. This is because we consider `QuantumCircuit`s "real", whereas the blocks are largely considered just syntax.

Unitary `QuantumCircuits` (this encompasses digital, or gate-based, circuits as well as analog circuits) are constructed by [`PrimitiveBlocks`] using a syntax that allows you to execute them in sequence, dubbed `ChainBlock` in the code, or in parallel (i.e. at the same time) where applicable, dubbed `KronBlock` in the code.
Notice that this differs from other packages by providing more control of the layout of the circuit than conventional packages like Qiskit, and from Yao where the blocks are the primary type.

### `QuantumModel`

`QuantumModel`s are meant to be the main entry point for quantum computations in `qadence`. In general, they take one or more
quantum circuit as input and they wrap all the necessary boiler plate code to make the circuit executable and differentiable
on the chosen backend.

Models are meant to be specific for a certain kind of quantum problem or algorithm and you can easily create new ones starting
from the base class `QuantumModel`, as explained in the [custom model tutorial](../advanced_tutorials/custom-models.md). Currently, Qadence offers
a `QNN` model class which provides convenient methods to work with quantum neural networks with multi-dimensional inputs
and outputs.

### `DifferentiableBackend`

The differentiable backend is a thin wrapper which takes as input a `QuantumCircuit` instance and a chosen quantum backend and make the circuit execution routines (expectation value, overalap, etc.) differentiable. Qadence offers both a PyTorch and Jax differentiation engine.

### Quantum `Backend`

For execution the primary object is the `Backend`. Backends maintain the same user-facing interface, and internally connects to other libraries to execute circuits. Those other libraries can execute the code on QPUs and local or cloud-based emulators. The `Backends` use PyTorch tensors to represent data and leverages PyTorchs autograd to help compute derivatives of circuits.

## Symbolic parameters

To illustrate how parameters work in Qadence, let's consider the following simple block composed of just two rotations:

```python exec="on" source="material-block" session="architecture"
import sympy
from qadence import Parameter, RX

param = Parameter("phi", trainable=False)
block = RX(0, param) * RX(1, sympy.acos(param))
```

The rotation angles assigned to `RX` (and to any Qadence quantum operation) are defined as arbitrary expressions of `Parameter`'s. `Parameter` is a subclass of `sympy.Symbol`, thus fully interoperable with it.

To assign values of the parameter `phi` in a quantum model, one should use a dictionary containing the a key with parameter name and the corresponding values values:

```python exec="on" source="material-block" session="architecture"
import torch
from qadence import run

values = {"phi": torch.rand(10)}
wf = run(block, values=values)
```

This is the only interface for parameter assignment exposed to the user. Under the hood, parameters applied to every quantum operation are identified in different ways:

* By default, with a stringified version of the `sympy` expression supplied to the quantum operation. Notice that multiple operations can have the same expression.

* In certain case, e.g. for constructing parameter shift rules, one must access a *unique* identifier of the parameter for each quantum operation. Therefore, Qadence also creates unique identifiers for each parametrized operation (see the [`ParamMap`][qadence.parameters.ParamMap] class).

By default, when one constructs a new backend, the parameter identifiers are the `sympy` expressions
which are used when converting an abstract block into a native circuit for the chosen backend.
However, one can use the unique identifiers as parameter names by setting the private flag
`_use_gate_params` to `True` in the backend configuration
[`BackendConfiguration`][qadence.backend.BackendConfiguration].
This is automatically set when PSR differentiation is selected (see next section for more details).

You can see the logic for choosing the parameter identifier in [`get_param_name`][qadence.backend.BackendConfiguration.get_param_name].

## Differentiation with parameter shift rules (PSR)

In Qadence, parameter shift rules are applied by implementing a custom `torch.autograd.Function` class for PyTorch and the `custom_vjp` in the Jax Engine, respectively.

A custom PyTorch `Function` looks like this:

```python
import torch
from torch.autograd import Function

class CustomFunction(Function):

    # forward pass implementation giving the output of the module
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, params: torch.Tensor):
        ctx.save_for_backward(inputs, params)
        ...

    # backward pass implementation giving the derivative of the module
    # with respect to the parameters. This must return the whole vector-jacobian
    # product to integrate within the autograd engine
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, params = ctx.saved_tensors
        ...
```

The class `PSRExpectation` under `qadence.engines.torch.differentiable_expectation` implements parameter shift rules for all parameters using
a custom function as the one above. There are a few implementation details to keep in mind if you want
to modify the PSR code:

* **PyTorch `Function` only works with tensor arguments**. Parameters in Qadence are passed around as
  dictionaries with parameter names as keys and current parameter values (tensors)
  as values. This works for both variational and feature parameters. However, the `Function` class
  only work with PyTorch tensors as input, not dictionaries. Therefore, the forward pass of
  `PSRExpectation` accepts one argument `param_keys` with the
  parameter keys and a variadic positional argument `param_values` with the parameter values one by
  one. The dictionary is reconstructed within the `forward()` pass body.

* **Higher-order derivatives with PSR**. Higher-order PSR derivatives can be tricky. Parameter shift
  rules calls, under the hood, the `QuantumBackend` expectation value routine that usually yield a
  non-differentiable output. Therefore, a second call to the backward pass would not work. However,
  Qadence employs a very simple trick to make higher-order derivatives work: instead of using
  directly the expectation value of the quantum backend, the PSR backward pass uses the PSR forward
  pass itself as expectation value function (see the code below). In this way, multiple calls to the
  backward pass are allowed since the `expectation_fn` routine is always differentiable by
  definition. Notice that this implementation is simple but suboptimal since, in some corner cases,
  higher-order derivates might include some repeated terms that, with this implementation, are
  always recomputed.

```python
# expectation value used in the PSR backward pass
def expectation_fn(params: dict[str, Tensor]) -> Tensor:
    return PSRExpectation.apply(
        ctx.expectation_fn,
        ctx.param_psrs,
        params.keys(),
        *params.values(),
    )
```

* **Operation parameters must be uniquely identified for PSR to work**. Parameter shift rules work at the level of individual quantum operations. This means that, given a parameter `x`, one needs to sum the contributions from shifting the parameter values of **all** the operation where the parameter `x` appears. When constructing the PSR rules, one must access a unique parameter identifier for each operation even if the corresponding user-facing parameter is the same. Therefore, when PSR differentiation is selected, the flag `_use_gate_params` is automatically set to `True` in the backend configuration [`BackendConfiguration`][qadence.backend.BackendConfiguration] (see previous section).

* **PSR must not be applied to observable parameters**. In Qadence, Pauli observables can also be parametrized. However, the tunable parameters of observables are purely classical and should not be included in the differentiation with PSRs. However, the quantum expectation value depends on them, thus they still need to enter into the PSR evaluation. To solve this issue, the code sets the `requires_grad` attribute of all observable parameters to `False` when constructing the PSRs for the circuit as in the snippet below:

```python
for obs in observable:
    for param_id, _ in uuid_to_eigen(obs).items():
        param_to_psr[param_id] = lambda x: torch.tensor([0.0], requires_grad=False)
```
