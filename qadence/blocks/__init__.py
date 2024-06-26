# flake8: noqa
import warnings
from typing import Any

from qadence.register import Register
from qadence.blocks.analog import AnalogBlock, Interaction

from .abstract import AbstractBlock
from .primitive import (
    ParametricBlock,
    PrimitiveBlock,
    TimeEvolutionBlock,
    ScaleBlock,
    ParametricControlBlock,
    ControlBlock,
    ProjectorBlock,
)
from .composite import AddBlock, ChainBlock, CompositeBlock, KronBlock, PutBlock
from .matrix import MatrixBlock
from .manipulate import from_openfermion, to_openfermion
from .utils import (
    add,
    chain,
    kron,
    tag,
    put,
    block_is_commuting_hamiltonian,
    block_is_qubit_hamiltonian,
    parameters,
    primitive_blocks,
    get_pauli_blocks,
    has_duplicate_vparams,
)
from .block_to_tensor import block_to_tensor
from .embedding import embedding

# Modules to be automatically added to the qadence namespace
__all__ = ["add", "chain", "kron", "tag", "block_to_tensor"]
