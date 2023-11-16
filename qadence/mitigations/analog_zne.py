from __future__ import annotations

from typing import cast

import numpy as np
import torch
from pulser_simulation import SimConfig
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


def zne(noise_probas: list, zne_dataset: list) -> float:
    # Rearrange the dataset by selecting each element in the batches.
    # TODO: Correct for arbitrary batches.
    rearranged_dataset = [s[0][0] for s in zne_dataset]
    # Polynomial fit.
    p = np.poly1d(np.polyfit(noise_probas, rearranged_dataset, 4))
    return float(p(0.0))  # Return the zero-noise fitted value.


def analog_zne(
    backend_name: BackendName,
    circuit: ConvertedCircuit,
    observable: list[ConvertedObservable] | ConvertedObservable,
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
    mitigation: Mitigations | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    assert mitigation
    noise_model = mitigation.options.get("noise_model", None)
    if noise_model is None:
        KeyError(f"A noise model should be choosen from {Noise.list()}. Got {noise_model}.")
    noise_probas = mitigation.options.get("noise_probas", None)
    if noise_probas is None:
        KeyError(f"A range of noise probabilies should be passed. Got {noise_probas}.")
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    backend = cast(Backend, backend)  # Cast the Pulser backend.
    backend_config = backend.config
    # Construct the ZNE dataset.
    zne_dataset = []
    for noise_proba in noise_probas:
        # Setting the backend config to account for the noise.
        if noise_model == Noise.DEPOLARIZING:
            backend_config.sim_config = SimConfig(noise=noise_model, depolarizing_prob=noise_proba)
        elif noise_model == Noise.DEPHASING:
            backend_config.sim_config = SimConfig(noise=noise_model, dephasing_prob=noise_proba)
        # Get density matrices in the noisy case.
        density_matrices = backend.run_dm(
            circuit, param_values=param_values, state=state, endianness=endianness
        )
        # Convert observables to Numpy types compatible with QuTip simulations.
        # Matrices are flipped to match QuTip conventions.
        observables_np = [np.flip(block_to_tensor(obs.original).numpy()) for obs in observable]
        # Get expectation values at the end of the time serie [0,t]
        # at intervals of the sampling rate.
        zne_dataset.append(
            [[dm.expect(obs)[0][-1] for obs in observables_np] for dm in density_matrices]
        )
    # Zero-noise extrapolate.
    exp_val = zne(noise_probas=noise_probas, zne_dataset=zne_dataset)
    return torch.tensor([exp_val])


def mitigate(
    backend_name: BackendName,
    circuit: ConvertedCircuit,
    observable: list[ConvertedObservable] | ConvertedObservable,
    param_values: dict[str, Tensor] = {},
    state: Tensor | None = None,
    measurement: Measurements | None = None,
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
        mitigation=mitigation,
        endianness=endianness,
    )
    return mitigated_exp
