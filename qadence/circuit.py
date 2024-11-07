from __future__ import annotations

from dataclasses import dataclass
from itertools import chain as flatten
from logging import getLogger
from pathlib import Path
from typing import Iterable

from sympy import Basic

from qadence.blocks import AbstractBlock, AnalogBlock, CompositeBlock, chain
from qadence.blocks.utils import parameters, primitive_blocks, unique_parameters
from qadence.parameters import Parameter
from qadence.register import Register

# Modules to be automatically added to the qadence namespace
__all__ = ["QuantumCircuit"]

logger = getLogger(__name__)


@dataclass(eq=False)  # Avoid unhashability errors due to mutable attributes.
class QuantumCircuit:
    """Am abstract QuantumCircuit instance.

    It needs to be passed to a quantum backend for execution.
    """

    block: AbstractBlock
    register: Register

    def __init__(self, support: int | Register, *blocks: AbstractBlock):
        """
        Arguments:

            support: `Register` or number of qubits. If an integer is provided, a register is
                constructed with `Register.all_to_all(x)`
            *blocks: (Possibly multiple) blocks to construct the circuit from.
        """
        self.block = chain(*blocks) if len(blocks) != 1 else blocks[0]
        self.register = Register(support) if isinstance(support, int) else support

        global_block = isinstance(self.block, AnalogBlock) and self.block.qubit_support.is_global
        if not global_block and len(self.block) and self.block.n_qubits > self.register.n_qubits:
            raise ValueError(
                f"Register with {self.register.n_qubits} qubits is too small for the "
                f"given block with {self.block.n_qubits} qubits"
            )

    @property
    def n_qubits(self) -> int:
        return self.register.n_qubits

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumCircuit):
            raise TypeError(f"Cannot compare {type(self)} to {type(other)}.")
        if self.block != other.block:  # type: ignore[call-overload]
            return False
        if self.register != other.register:
            return False
        return True

    def __hash__(self) -> int:
        return hash(self._to_json())

    def __iter__(self) -> Iterable:
        if isinstance(self.block, CompositeBlock):
            yield from self.block
        else:
            yield self.block

    def __contains__(self, other: object) -> bool:
        if isinstance(other, AbstractBlock):
            if isinstance(self.block, CompositeBlock):
                return other in self.block
            else:
                return other == self.block
        elif isinstance(other, Parameter):
            return other in self.unique_parameters
        else:
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")

    @property
    def unique_parameters(self) -> list[Parameter]:
        """Return the unique parameters in the circuit.

        These parameters are the actual user-facing parameters which
        can be assigned by the user. Multiple gates can contain the
        same unique parameter

        Returns:
            list[Parameter]: List of unique parameters in the circuit
        """
        return unique_parameters(self.block)

    @property
    def num_unique_parameters(self) -> int:
        return len(self.unique_parameters) if self.unique_parameters else 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters())

    def parameters(self) -> list[Parameter | Basic] | list[tuple[Parameter | Basic, ...]]:
        """Extract all parameters for primitive blocks in the circuit.

        Notice that this function returns all the unique Parameters used
        in the quantum circuit. These can correspond to constants too.

        Returns:
            List[tuple[Parameter]]: A list of tuples containing the Parameter
            instance of each of the primitive blocks in the circuit or, if the `flatten`
            flag is set to True, a flattened list of all circuit parameters
        """
        return parameters(self.block)

    def dagger(self) -> QuantumCircuit:
        """Reverse the QuantumCircuit by calling dagger on the block."""
        return QuantumCircuit(self.n_qubits, self.block.dagger())

    def get_blocks_by_tag(self, tag: str) -> list[AbstractBlock]:
        """Extract one or more blocks using the human-readable tag.

        This function recursively explores all composite blocks to find
        all the occurrences of a certain tag in the blocks.

        Args:
            tag (str): the tag to look for

        Returns:
            list[AbstractBlock]: The block(s) corresponding to the given tag
        """

        def _get_block(block: AbstractBlock) -> list[AbstractBlock]:
            blocks = []
            if block.tag == tag:
                blocks += [block]
            if isinstance(block, CompositeBlock):
                blocks += flatten(*[_get_block(b) for b in block.blocks])
            return blocks

        return _get_block(self.block)

    def is_empty(self) -> bool:
        return len(primitive_blocks(self.block)) == 0

    def serialize(self) -> str:
        raise NotImplementedError

    @staticmethod
    def deserialize(json: str) -> QuantumCircuit:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.block.__repr__()

    def _to_dict(self) -> dict:
        return {
            "block": self.block._to_dict(),
            "register": self.register._to_dict(),
        }

    def _to_json(self, path: Path | str | None = None) -> str:
        import json

        qc_dumped = json.dumps(self._to_dict())
        if path is not None:
            path = Path(path)
            try:
                with open(path, "w") as file:
                    file.write(qc_dumped)
            except Exception as e:
                logger.error(f"Unable to write QuantumCircuit to disk due to {e}")

        return qc_dumped

    @classmethod
    def _from_dict(cls, d: dict) -> QuantumCircuit:
        from qadence import blocks as qadenceblocks
        from qadence import operations

        RootBlock = (
            getattr(operations, d["block"]["type"])
            if hasattr(operations, d["block"]["type"])
            else getattr(qadenceblocks, d["block"]["type"])
        )

        return QuantumCircuit(
            Register._from_dict(d["register"]),
            RootBlock._from_dict(d["block"]),
        )

    @classmethod
    def _from_json(cls, path: str | Path) -> QuantumCircuit:
        import json

        loaded_dict: dict = dict()
        if isinstance(path, str):
            path = Path(path)
        try:
            with open(path, "r") as file:
                loaded_dict = json.load(file)

        except Exception as e:
            logger.error(f"Unable to load QuantumCircuit due to {e}")

        return QuantumCircuit._from_dict(loaded_dict)
