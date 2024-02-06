from __future__ import annotations

import os
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor, nn

from qadence.backend import (
    Backend,
    BackendConfiguration,
    BackendName,
    ConvertedCircuit,
    ConvertedObservable,
)
from qadence.backends.api import backend_factory, config_factory
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import chain, unique_parameters
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.parameters import Parameter
from qadence.types import DiffMode, Endianness

logger = get_logger(__name__)


class QuantumModel(nn.Module):
    """The central class of qadence that executes `QuantumCircuit`s and make them differentiable.

    This class should be used as base class for any new quantum model supported in the qadence
    framework for information on the implementation of custom models see
    [here](/advanced_tutorials/custom-models.md).
    """

    backend: Backend | DifferentiableBackend
    embedding_fn: Callable
    _params: nn.ParameterDict
    _circuit: ConvertedCircuit
    _observable: list[ConvertedObservable] | None

    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
        backend: BackendName | str = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        configuration: BackendConfiguration | dict | None = None,
    ):
        """Initialize a generic QuantumModel instance.

        Arguments:
            circuit: The circuit that is executed.
            observable: Optional observable(s) that are used only in the `expectation` method. You
                can also provide observables on the fly to the expectation call directly.
            backend: A backend for circuit execution.
            diff_mode: A differentiability mode. Parameter shift based modes work on all backends.
                AD based modes only on PyTorch based backends.
            measurement: Optional measurement protocol. If None, use
                exact expectation value with a statevector simulator.
            configuration: Configuration for the backend.
            noise: A noise model to use.

        Raises:
            ValueError: if the `diff_mode` argument is set to None
        """
        super().__init__()

        if not isinstance(circuit, QuantumCircuit):
            TypeError(
                f"The circuit should be of type '<class QuantumCircuit>'. Got {type(circuit)}."
            )

        if diff_mode is None:
            raise ValueError("`diff_mode` cannot be `None` in a `QuantumModel`.")

        self.backend = backend_factory(
            backend=backend, diff_mode=diff_mode, configuration=configuration
        )

        if isinstance(observable, list) or observable is None:
            observable = observable
        else:
            observable = [observable]

        def _is_feature_param(p: Parameter) -> bool:
            return not p.trainable and not p.is_number

        if observable is None:
            self.inputs = list(filter(_is_feature_param, circuit.unique_parameters))
        else:
            uparams = unique_parameters(chain(circuit.block, *observable))
            self.inputs = list(filter(_is_feature_param, uparams))

        conv = self.backend.convert(circuit, observable)
        self.embedding_fn = conv.embedding_fn
        self._circuit = conv.circuit
        self._observable = conv.observable
        self._backend_name = backend
        self._diff_mode = diff_mode
        self._measurement = measurement
        self._noise = noise
        self._mitigation = mitigation
        self._params = nn.ParameterDict(
            {
                str(key): nn.Parameter(val, requires_grad=val.requires_grad)
                for key, val in conv.params.items()
            }
        )

    @property
    def vparams(self) -> OrderedDict:
        return OrderedDict({k: v.data for k, v in self._params.items() if v.requires_grad})

    @property
    def vals_vparams(self) -> Tensor:
        """Dictionary with parameters which are actually updated during optimization."""
        vals = torch.tensor([v for v in self._params.values() if v.requires_grad])
        vals.requires_grad = False
        return vals.flatten()

    @property
    def in_features(self) -> int:
        """Number of inputs."""
        return len(self.inputs)

    @property
    def out_features(self) -> int | None:
        """Number of outputs."""
        return 0 if self._observable is None else len(self._observable)

    @property
    def num_vparams(self) -> int:
        """The number of variational parameters."""
        return len(self.vals_vparams)

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        return self.backend.circuit(circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> Any:
        return self.backend.observable(observable, n_qubits)

    def reset_vparams(self, values: Sequence) -> None:
        """Reset all the variational parameters with a given list of values."""
        current_vparams = OrderedDict({k: v for k, v in self._params.items() if v.requires_grad})

        assert (
            len(values) == self.num_vparams
        ), "Pass an iterable with the values of all variational parameters"
        for i, k in enumerate(current_vparams.keys()):
            current_vparams[k].data = torch.tensor([values[i]])

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.run(*args, **kwargs)

    def run(
        self,
        values: dict[str, Tensor] = None,
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        if values is None:
            values = {}
        params = self.embedding_fn(self._params, values)
        return self.backend.run(self._circuit, params, state=state, endianness=endianness)

    def sample(
        self,
        values: dict[str, torch.Tensor] = {},
        n_shots: int = 1000,
        state: torch.Tensor | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        params = self.embedding_fn(self._params, values)
        if noise is None:
            noise = self._noise
        else:
            self._noise = noise
        if mitigation is None:
            mitigation = self._mitigation
        return self.backend.sample(
            self._circuit,
            params,
            n_shots=n_shots,
            state=state,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

    def expectation(
        self,
        values: dict[str, Tensor] = {},
        observable: list[ConvertedObservable] | ConvertedObservable | None = None,
        state: Optional[Tensor] = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        """Compute expectation using the given backend.

        Returns:
            A torch.Tensor of shape n_batches x n_obs
        """
        if observable is None:
            if self._observable is None:
                raise ValueError(
                    "Provide an AbstractBlock as the observable to compute expectation."
                    "Either pass a 'native_observable' directly to 'QuantumModel.expectation'"
                    "or pass a (non-native) '<class AbstractBlock>' to the 'QuantumModel.__init__'."
                )
            observable = self._observable

        params = self.embedding_fn(self._params, values)
        if measurement is None:
            measurement = self._measurement
        if noise is None:
            noise = self._noise
        else:
            self._noise = noise
        if mitigation is None:
            mitigation = self._mitigation
        return self.backend.expectation(
            circuit=self._circuit,
            observable=observable,
            param_values=params,
            state=state,
            measurement=measurement,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

    def overlap(self) -> Tensor:
        raise NotImplementedError("The overlap method is not implemented for this model.")

    def _to_dict(self, save_params: bool = False) -> dict[str, Any]:
        d = dict()
        try:
            if isinstance(self._observable, list):
                abs_obs = [obs.abstract._to_dict() for obs in self._observable]
            else:
                abs_obs = [dict()]

            d = {
                "circuit": self._circuit.abstract._to_dict(),
                "observable": abs_obs,
                "backend": self._backend_name,
                "diff_mode": self._diff_mode,
                "measurement": self._measurement._to_dict()
                if self._measurement is not None
                else {},
                "noise": self._noise._to_dict() if self._noise is not None else {},
                "backend_configuration": asdict(self.backend.backend.config),  # type: ignore
            }
            param_dict_conv = {}
            if save_params:
                param_dict_conv = {name: param.data for name, param in self._params.items()}
            d = {self.__class__.__name__: d, "param_dict": param_dict_conv}
            logger.debug(f"{self.__class__.__name__} serialized to {d}.")
        except Exception as e:
            logger.warning(f"Unable to serialize {self.__class__.__name__} due to {e}.")
        return d

    @classmethod
    def _from_dict(cls, d: dict, as_torch: bool = False) -> QuantumModel:
        from qadence.serialization import deserialize

        qm: QuantumModel
        try:
            qm_dict = d[cls.__name__]
            qm = cls(
                circuit=QuantumCircuit._from_dict(qm_dict["circuit"]),
                observable=(
                    None
                    if not isinstance(qm_dict["observable"], list)
                    else [deserialize(q_obs) for q_obs in qm_dict["observable"]]  # type: ignore[misc]
                ),
                backend=qm_dict["backend"],
                diff_mode=qm_dict["diff_mode"],
                measurement=Measurements._from_dict(qm_dict["measurement"]),
                noise=Noise._from_dict(qm_dict["noise"]),
                configuration=config_factory(qm_dict["backend"], qm_dict["backend_configuration"]),
            )

            if as_torch:
                conv_pd = torch.nn.ParameterDict()
                param_dict = d["param_dict"]
                for n, param in param_dict.items():
                    conv_pd[n] = torch.nn.Parameter(param)
                qm._params = conv_pd
            logger.debug(f"Initialized {cls.__name__} from {d}.")

        except Exception as e:
            logger.warning(f"Unable to deserialize object {d} to {cls.__name__} due to {e}.")

        return qm

    def save(
        self, folder: str | Path, file_name: str = "quantum_model.pt", save_params: bool = True
    ) -> None:
        if not os.path.isdir(folder):
            raise FileNotFoundError
        try:
            torch.save(self._to_dict(save_params), folder / Path(file_name))
        except Exception as e:
            print(f"Unable to write QuantumModel to disk due to {e}")

    @classmethod
    def load(
        cls, file_path: str | Path, as_torch: bool = False, map_location: str | torch.device = "cpu"
    ) -> QuantumModel:
        qm_pt = {}
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if os.path.isdir(file_path):
            from qadence.ml_tools.saveload import get_latest_checkpoint_name

            file_path = file_path / get_latest_checkpoint_name(file_path, "model")

        try:
            qm_pt = torch.load(file_path, map_location=map_location)
        except Exception as e:
            print(f"Unable to load QuantumModel due to {e}")
        return cls._from_dict(qm_pt, as_torch)

    def assign_parameters(self, values: dict[str, Tensor]) -> Any:
        """Return the final, assigned circuit that is used in e.g. `backend.run`."""
        params = self.embedding_fn(self._params, values)
        return self.backend.assign_parameters(self._circuit, params)

    def to(self, device: torch.device) -> QuantumModel:
        try:
            if isinstance(self._circuit.native, torch.nn.Module):
                # Backends which are not torch-based cannot be moved to 'device'
                self._params = self._params.to(device)
                self._circuit.native = self._circuit.native.to(device)
                if self._observable is not None:
                    if isinstance(self._observable, ConvertedObservable):
                        self._observable.native = self._observable.native.to(device)
                    elif isinstance(self._observable, list):
                        for obs in self._observable:
                            obs.native = obs.native.to(device)
                logger.debug(f"Moved {self} to device {device}.")
        except Exception as e:
            logger.warning(f"Unable to move {self} to device {device} due to {e}.")
        return self
