from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor

from qadence import BackendName
from qadence.backends.api import backend_factory
from qadence.backends.pulser.backend import Backend
from qadence.blocks import block_to_tensor
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import ConstantAnalogRotation, WaitBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.operations import AnalogRot
from qadence.transpile import apply_fn_to_blocks
from qadence.utils import Endianness


def zne_pulse(
    stretches: Tensor, zne_datasets: list[list], n_observables: int, n_params: int
) -> Tensor:
    poly_fits = []
    for o in range(n_observables):
        batched_observables: list = []
        for p in range(1 if n_params == 0 else n_params):
            rearranged_dataset = [s[o][p] for s in zne_datasets]
            # Polynomial fit function.
            poly_fit = np.poly1d(np.polyfit(stretches, rearranged_dataset, len(stretches) - 1))
            # Return the zero-noise extrapolated value.
            batched_observables.append(poly_fit(0.0))
        poly_fits.append(batched_observables)

    return torch.tensor(poly_fits)


def zne_noise(noise_probas: Tensor, zne_datasets: list[list]) -> Tensor:
    from matplotlib import pyplot as plt

    # Rearrange the dataset by selecting each element in the batches.
    poly_fits = []
    for datasets in zne_datasets:  # Loop over batches of observables.
        batched_fits = []
        for i in range(len(datasets[0])):  # Loop over batch length.
            rearranged_dataset = [s[i] for s in datasets]
            plt.plot(noise_probas, rearranged_dataset, "o")
            # Polynomial fit function.
            poly_fit = np.poly1d(
                np.polyfit(noise_probas, rearranged_dataset, len(noise_probas) - 1)
            )
            # Return the zero-noise extrapolated value.
            plt.plot(noise_probas, poly_fit(noise_probas), "-")
            plt.plot(0.0, poly_fit(0.0), "*")
            plt.show()
            batched_fits.append(poly_fit(0.0))
        poly_fits.append(batched_fits)

    return torch.tensor(poly_fits)


def pulse_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: Noise,
    stretches: dict[str, Tensor],
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    # They should be embedded before use.
    # expression_to_uuids
    # Retrieve the parameter name and values to stretch over.
    breakpoint()
    param_name, stretch_vals = list(stretches.items())[0]
    # Convert to Parameter type.
    param = Parameter(param_name)
    # Get the map from parameters to uuids.
    expr_to_uuids = expression_to_uuids(circuit.block)
    # Retrieve the parameter uuid.
    param_uuid = expr_to_uuids[param][0]
    # Retrieve the parameter value.
    param_value = param_values[param_uuid]
    # Store the initial <uuid, value> pair to change back later.
    init_val = {param_uuid: param_value}
    # Use stretched values in-place of the param value.
    param_values[param_uuid] = param_value * stretch_vals
    # Run a batch experiment and get noisy density matrices.
    breakpoint()

    conv_circuit = backend.circuit(circuit)
    noisy_density_matrices = backend.run_dm(
        conv_circuit, param_values=param_values, state=state, noise=noise, endianness=endianness
    )
    # Convert observables to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observables = [
        np.flip(block_to_tensor(observable).numpy()) for observable in observables
    ]
    # Create ZNE datasets by looping over batches.
    zne_datasets = []
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
    extrapolated_exp_values = zne_pulse(
        stretches=param_values[param_uuid],
        zne_datasets=zne_datasets,
        zne_value=list(init_val.values())[0].item(),
    )
    # Set the initial values back.
    param_uuid, param_value = list(init_val.items())[0]
    param_values[param_uuid] = param_value
    return extrapolated_exp_values


def analog_zne(
    backend_name: BackendName,
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    noise: Noise | None = None,
    mitigation: Mitigations | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    assert noise
    assert mitigation
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    backend = cast(Backend, backend)
    noise_model = mitigation.options.get("noise_model", None)
    if noise_model is None:
        KeyError(f"A noise model should be choosen from {Noise.list()}. Got {noise_model}.")
    stretches = mitigation.options.get("stretches", None)
    if stretches is not None:
        extrapolated_exp_values = pulse_experiment(
            backend=backend,
            circuit=circuit,
            observables=observables,
            param_values=param_values,
            noise=noise,
            stretches=stretches,
            endianness=endianness,
            state=state,
        )
    else:
        zne_datasets: list = []
        noise_probas = noise.options.get("noise_probas")
        extrapolated_exp_values = zne_noise(noise_probas=noise_probas, zne_datasets=zne_datasets)
    return extrapolated_exp_values


def mitigate(
    backend_name: BackendName,
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
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
        observables=observables,
        param_values=param_values,
        state=state,
        measurement=measurement,
        noise=noise,
        mitigation=mitigation,
        endianness=endianness,
    )
    return mitigated_exp
