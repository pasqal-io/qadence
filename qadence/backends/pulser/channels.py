from __future__ import annotations

from dataclasses import dataclass

from pulser.channels.channels import Rydberg

GLOBAL_CHANNEL = "Global"
LOCAL_CHANNEL = "Local"


@dataclass(frozen=True)
class CustomRydberg(Rydberg):
    name: str = "Rydberg"

    duration_steps: int = 1  # ns
    amplitude_steps: float = 0.01  # rad/µs
