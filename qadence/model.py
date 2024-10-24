from __future__ import annotations

import os
from collections import Counter, OrderedDict
from dataclasses import asdict
from logging import getLogger
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
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.parameters import Parameter
from qadence.types import DiffMode, Endianness

logger = getLogger(__name__)


class QuantumModel(nn.Module):
    """The central class of qadence that executes `QuantumCircuit`s and make them differentiable.

    This class should be used as base class for any new quantum model supported in the qadence
    framework for information on the implementation of custom models see
    [here](../tutorials/advanced_tutorials/custom-models.md).

    Example:
    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import QuantumModel, QuantumCircuit, RX, RY, Z, PI, chain, kron
    from qadence import FeatureParameter, VariationalParameter

    theta = VariationalParameter("theta")
    phi = FeatureParameter("phi")

    block = chain(
        kron(RX(0, theta), RY(1, theta)),
        kron(RX(0, phi), RY(1, phi)),
    )

    circuit = QuantumCircuit(2, block)

    observable = Z(0) + Z(1)

    model = QuantumModel(circuit, observable)
    values = {"phi": torch.tensor([PI, PI/2]), "theta": torch.tensor([PI, PI/2])}

    wf = model.run(values)
    xs = model.sample(values, n_shots=100)
    ex = model.expectation(values)
    print(wf)
    print(xs)
    print(ex)
    ```
    ```
    """

    backend: Backend | DifferentiableBackend
    embedding_fn: Callable
    _params: nn.ParameterDict
    _circuit: ConvertedCircuit
    _observable: list[ConvertedObservable] | None
    logger.debug("Initialised")

    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
        backend: BackendName | str = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
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
        """Variational parameters."""
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
        """Get backend-converted circuit.

        Args:
            circuit: QuantumCircuit instance.

        Returns:
            Backend circuit.
        """
        return self.backend.circuit(circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> Any:
        """Get backend observable.

        Args:
            observable: Observable block.
            n_qubits: Number of qubits

        Returns:
            Backend observable.
        """
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
        """Calls run method with arguments.

        Returns:
            Tensor: A torch.Tensor representing output.
        """
        return self.run(*args, **kwargs)

    def run(
        self,
        values: dict[str, Tensor] = None,
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        r"""Run model.

        Given an input state $| \psi_0 \rangle$,
        a set of variational parameters $\vec{\theta}$
        and the unitary representation of the model $U(\vec{\theta})$
        we return $U(\vec{\theta}) | \psi_0 \rangle$.

        Arguments:
            values: Values dict which contains values for the parameters.
            state: Optional input state to apply model on.
            endianness: Storage convention for binary information.

        Returns:
            A torch.Tensor representing output.
        """
        if values is None:
            values = {}

        params = self.embedding_fn(self._params, values)

        return self.backend.run(self._circuit, params, state=state, endianness=endianness)

    def sample(
        self,
        values: dict[str, torch.Tensor] = {},
        n_shots: int = 1000,
        state: torch.Tensor | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        """Obtain samples from model.

        Arguments:
            values: Values dict which contains values for the parameters.
            n_shots: Observable part of the expectation.
            state: Optional input state to apply model on.
            noise: A noise model to use.
            mitigation: A mitigation protocol to use.
            endianness: Storage convention for binary information.

        Returns:
            A list of Counter instances with the sample results.
        """
        params = self.embedding_fn(self._params, values)
        if noise is None:
            noise = self._noise
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
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        r"""Compute expectation using the given backend.



        Given an input state $|\psi_0 \rangle$,
        a set of variational parameters $\vec{\theta}$
        and the unitary representation of the model $U(\vec{\theta})$
        we return $\langle \psi_0 | U(\vec{\theta}) | \psi_0 \rangle$.

        Arguments:
            values: Values dict which contains values for the parameters.
            observable: Observable part of the expectation.
            state: Optional input state.
            measurement: Optional measurement protocol. If None, use
                exact expectation value with a statevector simulator.
            noise: A noise model to use.
            mitigation: A mitigation protocol to use.
            endianness: Storage convention for binary information.

        Raises:
            ValueError: when no observable is set.

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
        """Overlap of model.

        Raises:
            NotImplementedError: The overlap method is not implemented for this model.
        """
        raise NotImplementedError("The overlap method is not implemented for this model.")

    def _to_dict(self, save_params: bool = True) -> dict[str, Any]:
        """Convert QuantumModel to a dictionary for serialization.

        Arguments:
            save_params: Save parameters. Defaults to True.

        Returns:
            The dictionary
        """
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
                "measurement": (
                    self._measurement._to_dict() if self._measurement is not None else dict()
                ),
                "noise": self._noise._to_dict() if self._noise is not None else dict(),
                "backend_configuration": asdict(self.backend.backend.config),  # type: ignore
            }
            param_dict_conv = {}
            if save_params:
                param_dict_conv = {name: param for name, param in self._params.items()}
            d = {self.__class__.__name__: d, "param_dict": param_dict_conv}
            logger.debug(f"{self.__class__.__name__} serialized to {d}.")
        except Exception as e:
            logger.warning(f"Unable to serialize {self.__class__.__name__} due to {e}.")
        return d

    @classmethod
    def _from_dict(cls, d: dict, as_torch: bool = False) -> QuantumModel:
        """Initialize instance of QuantumModel from dictionary.

        Args:
            d: Dictionary.
            as_torch: Load parameters as torch tensors. Defaults to False.

        Returns:
            QuantumModel instance
        """
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
                noise=NoiseHandler._from_dict(qm_dict["noise"]),
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

    def load_params_from_dict(self, d: dict, strict: bool = True) -> None:
        """Copy parameters from dictionary into this QuantumModel.

        Unlike :meth:`~qadence.QuantumModel.from_dict`, this method does not create a new
        QuantumModel instance, but rather loads the parameters into the same QuantumModel.
        The behaviour of this method is similar to :meth:`~torch.nn.Module.load_state_dict`.

        The dictionary is assumed to have the format as saved via
        :meth:`~qadence.QuantumModel.to_dict`

        Args:
            d (dict): The dictionary
            strict (bool, optional):
                Whether to strictly enforce that the parameter keys in the dictionary and
                in the model match exactly. Default: ``True``.
        """
        param_dict = d["param_dict"]
        missing_keys = set(self._params.keys()) - set(param_dict.keys())
        unexpected_keys = set(param_dict.keys()) - set(self._params.keys())

        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.append(f"Unexpected key(s) in dictionary: {unexpected_keys}")
            if len(missing_keys) > 0:
                error_msgs.append(f"Missing key(s) in dictionary: {missing_keys}")
            if len(error_msgs) > 0:
                errors_string = "\n\t".join(error_msgs)
                raise RuntimeError(
                    f"Error(s) loading the parameter dictionary due to: \n\t{errors_string}\n"
                    "This error was thrown because the `strict` argument is set `True`."
                    "If you don't need the parameter keys of the dictionary to exactly match "
                    "the model parameters, set `strict=False`."
                )

        for n, param in param_dict.items():
            try:
                with torch.no_grad():
                    self._params[n].copy_(
                        torch.nn.Parameter(param, requires_grad=param.requires_grad)
                    )
            except Exception as e:
                logger.warning(f"Unable to load parameter {n} from dictionary due to {e}.")

    def save(
        self, folder: str | Path, file_name: str = "quantum_model.pt", save_params: bool = True
    ) -> None:
        """Save model.

        Arguments:
            folder: Folder where model is saved.
            file_name: File name for saving model. Defaults to "quantum_model.pt".
            save_params: Save parameters if True. Defaults to True.

        Raises:
            FileNotFoundError: If folder is not a directory.
        """
        if not os.path.isdir(folder):
            raise FileNotFoundError
        try:
            torch.save(self._to_dict(save_params), folder / Path(file_name))
        except Exception as e:
            logger.error(f"Unable to write QuantumModel to disk due to {e}")

    @classmethod
    def load(
        cls, file_path: str | Path, as_torch: bool = False, map_location: str | torch.device = "cpu"
    ) -> QuantumModel:
        """Load QuantumModel.

        Arguments:
            file_path: File path to load model from.
            as_torch: Load parameters as torch tensor. Defaults to False.
            map_location (str | torch.device, optional): Location for loading. Defaults to "cpu".

        Returns:
            QuantumModel from file_path.
        """
        qm_pt = {}
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if os.path.isdir(file_path):
            from qadence.ml_tools.saveload import get_latest_checkpoint_name

            file_path = file_path / get_latest_checkpoint_name(file_path, "model")

        try:
            qm_pt = torch.load(file_path, map_location=map_location)
        except Exception as e:
            logger.error(f"Unable to load QuantumModel due to {e}")
        return cls._from_dict(qm_pt, as_torch)

    def assign_parameters(self, values: dict[str, Tensor]) -> Any:
        """Return the final, assigned circuit that is used in e.g. `backend.run`.

        Arguments:
            values: Values dict which contains values for the parameters.

        Returns:
            Final, assigned circuit that is used in e.g. `backend.run`
        """
        params = self.embedding_fn(self._params, values)
        return self.backend.assign_parameters(self._circuit, params)

    def to(self, *args: Any, **kwargs: Any) -> QuantumModel:
        """Conversion method for device or types.

        Returns:
            QuantumModel with conversions.
        """
        from pyqtorch import QuantumCircuit as PyQCircuit

        try:
            if isinstance(self._circuit.native, PyQCircuit):
                self._circuit.native = self._circuit.native.to(*args, **kwargs)
                if self._observable is not None:
                    if isinstance(self._observable, ConvertedObservable):
                        self._observable.native = self._observable.native.to(*args, **kwargs)
                    elif isinstance(self._observable, list):
                        for obs in self._observable:
                            obs.native = obs.native.to(*args, **kwargs)
                self._params = self._params.to(
                    device=self._circuit.native.device,
                    dtype=(
                        torch.float64
                        if self._circuit.native.dtype == torch.cdouble
                        else torch.float32
                    ),
                )
                logger.debug(f"Moved {self} to {args}, {kwargs}.")
            else:
                logger.debug("QuantumModel.to only supports pyqtorch.QuantumCircuits.")
        except Exception as e:
            logger.warning(f"Unable to move {self} to {args}, {kwargs} due to {e}.")
        return self

    @property
    def device(self) -> torch.device:
        """Get device.

        Returns:
            torch.device
        """
        return (
            self._circuit.native.device
            if self.backend.backend.name == "pyqtorch"  # type: ignore[union-attr]
            else torch.device("cpu")
        )


# Modules to be automatically added to the qadence namespace
__all__ = ["QuantumModel"]  # type: ignore
