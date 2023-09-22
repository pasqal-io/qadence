from __future__ import annotations

import numpy as np
from pulser.parametrized.decorators import parametrize
from pulser.waveforms import ConstantWaveform

# determined by hardware team as a safe resolution
MAX_AMPLITUDE_SCALING = 0.1
EPS = 1e-9


class SquareWaveform(ConstantWaveform):
    def __init__(self, duration: int, value: float):
        super().__init__(duration, value)

    @classmethod
    @parametrize
    def from_area(
        cls,
        area: float,
        max_amp: float,
        duration_steps: int = 1,
        min_duration: int = 1,
    ) -> SquareWaveform:
        amp_steps = MAX_AMPLITUDE_SCALING * max_amp

        duration = max(
            duration_steps * np.round(area / (duration_steps * max_amp) * 1e3),
            min_duration,
        )
        amplitude = min(
            amp_steps * np.ceil(area / (amp_steps * duration) * 1e3),
            max_amp,
        )
        delta = np.abs(1e-3 * duration * amplitude - area)

        new_duration = duration + duration_steps
        new_amplitude = max(
            amp_steps * np.ceil(area / (amp_steps * new_duration) * 1e3),
            max_amp,
        )
        new_delta = np.abs(1e-3 * new_duration * new_amplitude - area)

        while new_delta < delta:
            duration = new_duration
            amplitude = new_amplitude
            delta = new_delta

            new_duration = duration + duration_steps
            new_amplitude = max(
                amp_steps * np.ceil(area / (amp_steps * new_duration) * 1e3),
                max_amp,
            )
            new_delta = np.abs(1e-3 * new_duration * new_amplitude - area)

        return cls(duration, amplitude)

    @classmethod
    @parametrize
    def from_duration(
        cls,
        duration: int,
        max_amp: float,
        duration_steps: int = 1,
        min_duration: int = 1,
    ) -> SquareWaveform:
        amp_steps = MAX_AMPLITUDE_SCALING * max_amp

        duration = max(
            duration_steps * np.round(duration / duration_steps),
            min_duration,
        )
        amplitude = min(
            amp_steps * np.ceil(max_amp / (amp_steps + EPS) * 1e3),
            max_amp,
        )

        return cls(duration, amplitude)
