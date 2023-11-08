from __future__ import annotations

from typing import Any, Counter, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter as TorchParam

from qadence.backend import ConvertedObservable
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.ml_tools import promote_to_tensor
from qadence.models import QNN, QuantumModel
from qadence.noise import Noise
from qadence.utils import Endianness

logger = get_logger(__name__)


def _set_fixed_operation(
    dim: int,
    x: float | np.ndarray | Tensor | None = None,
    operation_name: str = "scale",
) -> Tensor:
    dim = dim if dim > 0 else 1
    if x is None:
        if operation_name == "shift":
            x = torch.zeros(dim)
        elif operation_name == "scale":
            x = torch.ones(dim)
        else:
            NotImplementedError
    res = promote_to_tensor(x, requires_grad=False).squeeze(0)
    assert (
        res.numel() == dim
    ), f"Number of {operation_name} values is {res.numel()}\
    and does not match number of dimensions = {dim}."
    return res


class TransformedModule(torch.nn.Module):
    """
    This class accepts a torch.nn.Module or a QuantumModel/QNN.

    Wraps it with either non-trainble or trainable scaling and shifting parameters
    for both input and output. When given a torch.nn.Module,
    in_features and out_features need to be passed.

    Args:
        model: The original model to transform.
        in_features: The number of input dimensions of the model.
        out_features: The number of output dimensions of the model.
        input_scaling: The rescaling factor for the model input. Defaults to None.
        input_shifting: The translation factor for the model input. Defaults to None.
        output_scaling: The rescaling factor for the model output. Defaults to None.
        output_shifting: The translation factor for the model output. Defaults to None.

    Example:
    ```
    import torch
    from torch.nn import Parameter as TorchParam
    from qadence.models import QNN, TransformedModule
    from qadence.circuit import QuantumCircuit
    from qadence.blocks import chain
    from qadence.constructors import hamiltonian_factory, hea
    from qadence import Parameter, QuantumCircuit, Z

    n_qubits = 2
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    ansatz = hea(n_qubits=n_qubits, depth=3)
    observable = hamiltonian_factory(n_qubits, detuning = Z)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)

    model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)
    assert not torch.isnan(pred)

    transformed_model = TransformedModule(
        model=model,
        in_features=None,
        out_features=None,
        input_scaling=TorchParam(torch.tensor(1.0)),
        input_shifting=0.0,
        output_scaling=1.0,
        output_shifting=TorchParam(torch.tensor(0.0))
    )
    pred_transformed = transformed_model(input_values)
    ```
    """

    def __init__(
        self,
        model: torch.nn.Module | QuantumModel | QNN,
        in_features: int | None = None,
        out_features: int | None = None,
        input_scaling: TorchParam | float | int | torch.Tensor | None = None,
        input_shifting: TorchParam | float | int | torch.Tensor | None = None,
        output_scaling: TorchParam | float | int | torch.Tensor | None = None,
        output_shifting: TorchParam | float | int | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        if in_features is None and out_features is None:
            assert isinstance(model, (QuantumModel, QNN))
            self.in_features = model.in_features
            self.out_features = model.out_features if model.out_features else 1
        else:
            self.in_features = in_features  # type: ignore[assignment]
            self.out_features = out_features  # type: ignore[assignment]
        if not isinstance(input_scaling, torch.Tensor):
            self.register_buffer(
                "_input_scaling",
                _set_fixed_operation(self.in_features, input_scaling, "scale"),
            )
        else:
            self._input_scaling = input_scaling
        if not isinstance(input_shifting, torch.Tensor):
            self.register_buffer(
                "_input_shifting",
                _set_fixed_operation(self.in_features, input_shifting, "shift"),
            )
        else:
            self._input_shifting = input_shifting
        if not isinstance(output_scaling, torch.Tensor):
            self.register_buffer(
                "_output_scaling",
                _set_fixed_operation(self.out_features, output_scaling, "scale"),
            )
        else:
            self._output_scaling = output_scaling
        if not isinstance(output_shifting, torch.Tensor):
            self.register_buffer(
                "_output_shifting",
                _set_fixed_operation(self.out_features, output_shifting, "shift"),
            )
        else:
            self._output_shifting = output_shifting

    def _format_to_dict(self, values: Tensor) -> dict[str, Tensor]:
        """Format an input tensor into the format required by the forward pass.

        The tensor is assumed to have dimensions: n_batches x in_features where in_features
        corresponds to the number of input features of the QNN
        """

        if len(values.size()) == 1:
            values = values.reshape(-1, 1)
        if len(values.size()) != 2 or values.shape[1] != len(self.model.inputs):
            raise ValueError(
                f"Model expects in_features={self.model.in_features} but got {values.size()[1]}."
            )
        names = [p.name for p in self.model.inputs]
        res = {}
        for i, name in enumerate(names):
            res[name] = values[:, i]
        return res

    def _transform_x(self, x: dict[str, torch.Tensor] | Tensor) -> dict[str, Tensor] | Tensor:
        """
        X can either be a torch Tensor in when using torch.nn.Module, or a standard values dict.

        Scales and shifts the tensors in the values dict, containing Featureparameters.
        Transformation of inputs can be used to speed up training and avoid potential issues
        with numerical stability that can arise due to differing feature scales.
        If none are provided, it uses 0. for shifting and 1. for scaling (hence, identity).

        Arguments:
            values: A torch Tensor or a dict containing values for Featureparameters.

        Returns:
            A Tensor or dict containing transformed (scaled and/or shifted) Featureparameters.
        """

        if isinstance(self.model, (QuantumModel, QNN)):
            if not isinstance(x, dict):
                x = self._format_to_dict(x)
            return {
                key: self._input_scaling * (val + self._input_shifting) for key, val in x.items()
            }

        else:
            assert isinstance(self.model, torch.nn.Module) and isinstance(x, Tensor)
            return self._input_scaling * (x + self._input_shifting)

    def forward(self, x: dict[str, Tensor] | Tensor, *args: Any, **kwargs: Any) -> Tensor:
        y = self.model(self._transform_x(x), *args, **kwargs)
        return self._output_scaling * y + self._output_shifting

    def run(
        self,
        values: dict[str, torch.Tensor],
        state: torch.Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        return self.model.run(values=self._transform_x(values), state=state, endianness=endianness)

    def sample(
        self,
        values: dict[str, torch.Tensor],
        n_shots: int = 1000,
        state: torch.Tensor | None = None,
        noise: Noise | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        return self.model.sample(  # type: ignore[no-any-return]
            values=self._transform_x(values),
            n_shots=n_shots,
            state=state,
            endianness=endianness,
            noise=noise,
        )

    def expectation(
        self,
        values: dict[str, torch.Tensor],
        observable: List[ConvertedObservable] | ConvertedObservable | None = None,
        state: torch.Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        """
        Computes standard expectation.

        However, scales and shifts the output tensor of the underlying model.
        If none are provided, it uses 0. for shifting and 1. for scaling.
        Transformation of ouputs can be used if the magnitude
        of the targets exceeds the domain (-1,1).
        """
        exp = self.model.expectation(
            values=self._transform_x(values),
            observable=observable if observable is not None else self.model._observable,
            state=state,
            measurement=measurement,
            noise=noise,
            endianness=endianness,
        )
        return self._output_scaling * exp + self._output_shifting

    def _to_dict(self, save_params: bool = True) -> dict:
        from qadence.serialization import serialize

        def store_fn(x: torch.Tensor) -> list[float]:
            res: list[float]
            if x.requires_grad:
                res = x.detach().numpy().tolist()
            else:
                res = x.numpy().tolist()
            return res  # type: ignore[no-any-return]

        _d = serialize(self.model, save_params=save_params)

        return {
            self.__class__.__name__: _d,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "_input_scaling": store_fn(self._input_scaling),
            "_output_scaling": store_fn(self._output_scaling),
            "_input_shifting": store_fn(self._input_shifting),
            "_output_shifting": store_fn(self._output_shifting),
        }

    @classmethod
    def _from_dict(cls, d: dict, as_torch: bool = False) -> TransformedModule:
        from qadence.serialization import deserialize

        _m: QuantumModel | QNN = deserialize(d[cls.__name__], as_torch)  # type: ignore[assignment]
        return cls(
            _m,
            in_features=d["in_features"],
            out_features=d["out_features"],
            input_scaling=torch.tensor(d["_input_scaling"]),
            output_scaling=torch.tensor(d["_output_scaling"]),
            input_shifting=torch.tensor(d["_input_shifting"]),
            output_shifting=torch.tensor(d["_output_shifting"]),
        )
