from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor

from qadence import BackendName
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.api import backend_factory
from qadence.backends.pulser.backend import Backend
from qadence.blocks import block_to_tensor
from qadence.blocks.utils import expression_to_uuids
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.parameters import Parameter
from qadence.utils import Endianness


def zne_pulse(stretches: dict, zne_datasets: list, zne_value: Tensor) -> list:
    # Rearrange the dataset by selecting each element in the batches.
    poly_fits = []
    # stretch_param = list(stretches.values())[0]
    for datasets in zne_datasets:  # Loop over batches of observables.
        for dataset in datasets:
            # Polynomial fit function.
            poly_fits.append(np.poly1d(np.polyfit(stretches, dataset, 4)))

    return list(
        map(lambda p: p(zne_value.item()), poly_fits)
    )  # Return the zero-noise fitted value.


def zne_noise(noise: Noise, zne_datasets: list) -> list:
    # Rearrange the dataset by selecting each element in the batches.
    noise_probas = noise.options.get("noise_probas")
    poly_fits = []
    for datasets in zne_datasets:  # Loop over batches of observables.
        for i in range(len(datasets[0])):  # Loop over batch length.
            rearranged_dataset = [s[i] for s in datasets]
            # Polynomial fit function.
            poly_fits.append(np.poly1d(np.polyfit(noise_probas, rearranged_dataset, 4)))

    return list(map(lambda p: p(0.0), poly_fits))  # Return the zero-noise fitted value.


def analog_zne(
    backend_name: BackendName,
    circuit: ConvertedCircuit,
    observable: list[ConvertedObservable] | ConvertedObservable,
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    noise: Noise | None = None,
    mitigation: Mitigations | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    assert noise
    assert mitigation
    noise_model = mitigation.options.get("noise_model", None)
    if noise_model is None:
        KeyError(f"A noise model should be choosen from {Noise.list()}. Got {noise_model}.")
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    backend = cast(Backend, backend)
    stretches = mitigation.options.get("stretches", None)
    # Signals to use the stretches as parameters for the ZNE data.
    # They should be embedded before use.
    # expression_to_uuids
    if stretches is not None:
        # Retrieve the parameter name and values to stretch over.
        param_name, stretch_vals = list(stretches.items())[0]
        # Convert to Parameter type.
        param = Parameter(param_name)
        # Get the map from parameters to uuids.
        expr_to_uuids = expression_to_uuids(circuit.abstract.block)
        # Retrieve the parameter uuid.
        param_uuid = expr_to_uuids[param][0]
        # Retrieve the parameter value.
        param_value = param_values[param_uuid]
        # Store the initial <uuid, value> pair to change back later.
        init_val = {param_uuid: param_value}
        # Use stretched values in-place of the param value.
        param_values[param_uuid] = param_value * stretch_vals
    zne_datasets = []
    # Get noisy density matrices.
    noisy_density_matrices = backend.run_dm(
        circuit, param_values=param_values, state=state, noise=noise, endianness=endianness
    )
    # Convert observables to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observables = [np.flip(block_to_tensor(obs.original).numpy()) for obs in observable]
    # Create ZNE datasets by looping over batches.
    for observable in converted_observables:
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_datasets.append(
            [
                [dm.expect(observable)[0][-1] for dm in density_matrices]
                for density_matrices in noisy_density_matrices
            ]
        )
    # Zero-noise extrapolate.
    if stretches is not None:
        extrapolated_exp_values = zne_pulse(
            stretches=param_values[param_uuid],
            zne_datasets=zne_datasets,
            zne_value=list(init_val.values())[0],
        )
        # Set the initial values back.
        param_uuid, param_value = list(init_val.items())[0]
        param_values[param_uuid] = param_value
    else:
        extrapolated_exp_values = zne_noise(noise=noise, zne_datasets=zne_datasets)
    return torch.tensor(extrapolated_exp_values)


def mitigate(
    backend_name: BackendName,
    circuit: ConvertedCircuit,
    observable: list[ConvertedObservable] | ConvertedObservable,
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    noise: Noise | None = None,
    mitigation: Mitigations | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    mitigated_exp = analog_zne(
        backend_name=backend_name,
        circuit=circuit,
        observable=observable,
        param_values=param_values,
        state=state,
        measurement=measurement,
        noise=noise,
        mitigation=mitigation,
        endianness=endianness,
    )
    return mitigated_exp
