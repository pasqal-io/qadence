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
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.utils import Endianness


def zne(noise: Noise, zne_datasets: list) -> list:
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
    extrapolated_exp_values = zne(noise=noise, zne_datasets=zne_datasets)
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
