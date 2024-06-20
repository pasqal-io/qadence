from __future__ import annotations

from logging import getLogger

from qadence.blocks.primitive import NoisyPrimitiveBlock
#TODO: ADD NoisyPrimitiveBlock to __init__ qadence.block

from qadence.types import OpName

logger = getLogger(__name__)

class BitFlip(NoisyPrimitiveBlock):
    """The Bitflip noise gate."""

    name = OpName.BITFLIP

    def __init__(self, target: int, noise_probability: float | tuple[float, ...]):
        super().__init__((target,), noise_probability)
    
    @property
    def generator(self) -> None:
        raise ValueError("Property `generator` not available for non-unitary operator.")

    @property
    def eigenvalues_generator(self) -> None:
        raise ValueError("Property `eigenvalues_generator` not available for non-unitary operator.")
