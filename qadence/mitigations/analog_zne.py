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
from qadence.blocks.analog import ConstantAnalogRotation, InteractionBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.operations import AnalogRot
from qadence.transpile import apply_fn_to_blocks
from qadence.utils import Endianness


def zne(noise_levels: Tensor, zne_datasets: list[list]) -> Tensor:
    poly_fits = []
    for dataset in zne_datasets:  # Looping over batched observables.
        batched_observable: list = []
        n_params = len(dataset[0])
        for p in range(n_params):  # Looping over the batched parameters.
            rearranged_dataset = [s[p] for s in dataset]
            # Polynomial fit function.
            poly_fit = np.poly1d(
                np.polyfit(noise_levels, rearranged_dataset, len(noise_levels) - 1)
            )
            # Return the zero-noise extrapolated value.
            batched_observable.append(poly_fit(0.0))
        poly_fits.append(batched_observable)

    return torch.tensor(poly_fits)


def pulse_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: NoiseHandler,
    stretches: Tensor,
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    def mutate_params(block: AbstractBlock, stretch: float) -> AbstractBlock:
        """Closure to retrieve and stretch analog parameters."""
        # Check for stretchable analog block.
        if isinstance(block, (ConstantAnalogRotation, InteractionBlock)):
            stretched_duration = block.parameters.duration * stretch
            stretched_omega = block.parameters.omega / stretch
            stretched_delta = block.parameters.delta / stretch
            # The Hamiltonian scaling has no effect on the phase parameter
            phase = block.parameters.phase
            qubit_support = block.qubit_support
            return AnalogRot(
                duration=stretched_duration,
                omega=stretched_omega,
                delta=stretched_delta,
                phase=phase,
                qubit_support=qubit_support,
            )
        return block

    zne_datasets = []
    noisy_density_matrices: list = []
    for stretch in stretches:
        # FIXME: Iterating through the circuit for every stretch
        # and rebuilding the block leaves is inefficient.
        # Best to retrieve the parameters once
        # and rebuild the blocks.
        stre = stretch.item()
        block = apply_fn_to_blocks(circuit.block, mutate_params, stre)
        stretched_register = circuit.register.rescale_coords(stre)
        stretched_circuit = QuantumCircuit(stretched_register, block)
        conv_circuit = backend.circuit(stretched_circuit)
        noisy_density_matrices.append(
            # Contain a single experiment result for the stretch.
            backend.run(
                conv_circuit,
                param_values=param_values,
                state=state,
                noise=noise,
                endianness=endianness,
            )[0]
        )
    # Convert observable to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observable = [np.flip(block_to_tensor(obs).numpy()) for obs in observable]
    # Create ZNE datasets by looping over batches.
    for observable in converted_observable:
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_datasets.append(
            [
                [dm.expect(observable)[0][-1] for dm in density_matrices]
                for density_matrices in noisy_density_matrices
            ]
        )
    # Zero-noise extrapolate.
    extrapolated_exp_values = zne(
        noise_levels=stretches,
        zne_datasets=zne_datasets,
    )
    return extrapolated_exp_values


def noise_level_experiment(
    backend: Backend,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor],
    noise: NoiseHandler,
    endianness: Endianness,
    state: Tensor | None = None,
) -> Tensor:
    protocol, options = noise.protocol[-1], noise.options[-1]
    noise_probs = options.get("noise_probs")
    zne_datasets: list = []
    # Get noisy density matrices.
    conv_circuit = backend.circuit(circuit)
    noisy_density_matrices = backend.run(
        conv_circuit, param_values=param_values, state=state, noise=noise, endianness=endianness
    )
    # Convert observable to Numpy types compatible with QuTip simulations.
    # Matrices are flipped to match QuTip conventions.
    converted_observable = [np.flip(block_to_tensor(obs).numpy()) for obs in observable]
    # Create ZNE datasets by looping over batches.
    for observable in converted_observable:
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_datasets.append(
            [
                [dm.expect(observable)[0][-1] for dm in density_matrices]
                for density_matrices in noisy_density_matrices
            ]
        )
    # Zero-noise extrapolate.
    extrapolated_exp_values = zne(noise_levels=noise_probs, zne_datasets=zne_datasets)
    return extrapolated_exp_values


def analog_zne(
    backend_name: BackendName,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    noise: NoiseHandler | None = None,
    mitigation: Mitigations | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    assert noise
    assert mitigation
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    backend = cast(Backend, backend)
    noise_model = mitigation.options.get("noise_model", None)
    if noise_model is None:
        raise KeyError("A noise model should be specified.")
    stretches = mitigation.options.get("stretches", None)
    if stretches is not None:
        extrapolated_exp_values = pulse_experiment(
            backend=backend,
            circuit=circuit,
            observable=observable,
            param_values=param_values,
            noise=noise,
            stretches=stretches,
            endianness=endianness,
            state=state,
        )
    else:
        extrapolated_exp_values = noise_level_experiment(
            backend=backend,
            circuit=circuit,
            observable=observable,
            param_values=param_values,
            noise=noise,
            endianness=endianness,
            state=state,
        )
    return extrapolated_exp_values


def mitigate(
    backend_name: BackendName,
    circuit: QuantumCircuit,
    observable: list[AbstractBlock],
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    noise: NoiseHandler | None = None,
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
