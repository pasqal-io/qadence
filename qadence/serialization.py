from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, get_args
from typing import Union as TypingUnion

import torch
from arpeggio import NoMatch
from arpeggio.cleanpeg import ParserPEG
from sympy import *
from sympy import core, srepr

from qadence import QNN, QuantumCircuit, QuantumModel, operations, parameters
from qadence import blocks as qadenceblocks
from qadence.blocks import AbstractBlock
from qadence.blocks.utils import tag
from qadence.parameters import Parameter
from qadence.register import Register
from qadence.types import SerializationFormat

# Modules to be automatically added to the qadence namespace
__all__ = ["deserialize", "load", "save", "serialize"]


logger = getLogger(__name__)


def file_extension(file: Path | str) -> str:
    FORMAT = ""
    if isinstance(file, str):
        _, extension = os.path.splitext(file)
        FORMAT = extension[1:].upper()
    elif isinstance(file, os.PathLike):
        _, extension = os.path.splitext(str(file))
        FORMAT = extension[1:].upper()
    return FORMAT


SUPPORTED_OBJECTS = [
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    QNN,
    Register,
    core.Basic,
    torch.nn.Module,
]
SUPPORTED_TYPES = TypingUnion[
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    Register,
    core.Basic,
    torch.nn.Module,
]

ALL_BLOCK_NAMES = [
    n for n in dir(qadenceblocks) if not (n.startswith("__") and n.endswith("__"))
] + [n for n in dir(operations) if not (n.startswith("__") and n.endswith("__"))]
SYMPY_EXPRS = [n for n in dir(core) if not (n.startswith("__") and n.endswith("__"))]
QADENCE_PARAMETERS = [n for n in dir(parameters) if not (n.startswith("__") and n.endswith("__"))]


THIS_PATH = Path(__file__).parent
GRAMMAR_FILE = THIS_PATH / "serial_expr_grammar.peg"


@lru_cache
def _parser_fn() -> ParserPEG:
    with open(GRAMMAR_FILE, "r") as f:
        grammar = f.read()
    return ParserPEG(grammar, "Program")


_parsing_serialize_expr = _parser_fn()


def parse_expr_fn(code: str) -> bool:
    """
    A parsing expressions function that checks whether a given code is valid on.

    the parsing grammar. The grammar is defined to be compatible with `sympy`
    expressions, such as `Float('-0.33261030434342942', precision=53)`, while
    avoiding code injection such as `2*3` or `__import__('os').system('ls -la')`.

    Args:
        code (str): code to be parsed and checked.

    Returns:
        Boolean indicating whether the code matches the defined grammar or not.
    """

    parser = _parsing_serialize_expr
    try:
        parser.parse(code)
    except NoMatch:
        return False
    else:
        return True


@dataclass
class SerializationModel:
    """
    A serialization model class to serialize data from `QuantumModel`s,.

    `torch.nn.Module` and similar structures. The data included in the
    serialization logic includes: the `AbstractBlock` and its children
    classes, `QuantumCircuit`, `Register`, and `sympy` expressions
    (including `Parameter` class from `qadence.parameters`).

    A children class must define the `value` attribute type and how to
    handle it, since it is the main property for the class to be used
    by the serialization process. For instance:

    ```python
    @dataclass
    class QuantumCircuitSerialization(SerializationModel):
        value: QuantumCircuit = dataclass_field(init=False)

        def __post_init__(self) -> None:
            self.value = (
                QuantumCircuit._from_dict(self.d)
                if isinstance(self.d, dict)
                else self.d
            )
    ```
    """

    d: dict = dataclass_field(default_factory=dict)
    value: Any = dataclass_field(init=False)


@dataclass
class BlockTypeSerialization(SerializationModel):
    value: AbstractBlock = dataclass_field(init=False)

    def __post_init__(self) -> None:
        block = (
            getattr(operations, self.d["type"])
            if hasattr(operations, self.d["type"])
            else getattr(qadenceblocks, self.d["type"])
        )._from_dict(self.d)
        if self.d["tag"] is not None:
            block = tag(block, self.d["tag"])
        self.value = block


@dataclass
class QuantumCircuitSerialization(SerializationModel):
    value: QuantumCircuit = dataclass_field(init=False)

    def __post_init__(self) -> None:
        self.value = QuantumCircuit._from_dict(self.d) if isinstance(self.d, dict) else self.d


@dataclass
class RegisterSerialization(SerializationModel):
    value: Register = dataclass_field(init=False)

    def __post_init__(self) -> None:
        self.value = Register._from_dict(self.d)


@dataclass
class ModelSerialization(SerializationModel):
    as_torch: bool = False
    value: torch.nn.Module = dataclass_field(init=False)

    def __post_init__(self) -> None:
        module_name = list(self.d.keys())[0]
        obj = globals().get(module_name, None)
        if obj is None:
            obj = self._resolve_module(module_name)
        if hasattr(obj, "_from_dict"):
            self.value = obj._from_dict(self.d, self.as_torch)
        elif hasattr(obj, "load_state_dict"):
            self.value = obj.load_state_dict(self.d[module_name])
        else:
            msg = (
                f"Unable to deserialize object '{module_name}'. "
                f"Supported types are {SUPPORTED_OBJECTS}."
            )
            logger.error(TypeError(msg))
            raise TypeError(msg)

    @staticmethod
    def _resolve_module(module: str) -> Any:
        for loaded_module in sys.modules.keys():
            if "qadence" in loaded_module:
                obj = getattr(sys.modules[loaded_module], module, None)
                if obj:
                    return obj
        raise ValueError(f"Couldn't resolve module '{module}'.")


@dataclass
class ExpressionSerialization(SerializationModel):
    value: str | core.Expr | float = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if parse_expr_fn(self.d["expression"]):
            expr = eval(self.d["expression"])
            if hasattr(expr, "free_symbols"):
                for s in expr.free_symbols:
                    s.value = float(self.d["symbols"][s.name]["value"])
            self.value = expr
        else:
            raise ValueError(f"Invalid expression: {self.d['expression']}")


def save_pt(d: dict, file_path: str | Path) -> None:
    torch.save(d, file_path)


def save_json(d: dict, file_path: str | Path) -> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(d))


def load_pt(file_path: str | Path, map_location: str) -> Any:
    return torch.load(file_path, map_location=map_location)


def load_json(file_path: str | Path, map_location: str) -> Any:
    with open(file_path, "r") as file:
        return json.load(file)


FORMAT_DICT = {
    SerializationFormat.PT: (".pt", save_pt, load_pt, True),
    SerializationFormat.JSON: (".json", save_json, load_json, False),
}


def serialize(obj: SUPPORTED_TYPES, save_params: bool = False) -> dict:
    """
    Supported Types:

    AbstractBlock | QuantumCircuit | QuantumModel | torch.nn.Module | Register | Module
    Serializes a qadence object to a dictionary.

    Arguments:
        obj (AbstractBlock | QuantumCircuit | QuantumModel | Register | torch.nn.Module):
    Returns:
        A dict.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import serialize, deserialize, hea, hamiltonian_factory, Z
    from qadence import QuantumCircuit, QuantumModel

    n_qubits = 2
    myblock = hea(n_qubits=n_qubits, depth=1)
    block_dict = serialize(myblock)
    print(block_dict)

    ## Lets use myblock in a QuantumCircuit and serialize it.

    qc = QuantumCircuit(n_qubits, myblock)
    qc_dict = serialize(qc)
    qc_deserialized = deserialize(qc_dict)
    assert qc == qc_deserialized

    ## Finally, let's wrap it in a QuantumModel
    obs = hamiltonian_factory(n_qubits, detuning = Z)
    qm = QuantumModel(qc, obs, backend='pyqtorch', diff_mode='ad')

    qm_dict = serialize(qm)
    qm_deserialized = deserialize(qm_dict)
    # Lets check if the loaded QuantumModel returns the same expectation
    assert torch.isclose(qm.expectation({}), qm_deserialized.expectation({}))
    ```
    """
    if not isinstance(obj, get_args(SUPPORTED_TYPES)):
        logger.error(TypeError(f"Serialization of object type {type(obj)} not supported."))

    d: dict = dict()
    try:
        if isinstance(obj, core.Expr):
            symb_dict = dict()
            expr_dict = {"name": str(obj), "expression": srepr(obj)}
            symbs: set[Parameter | core.Basic] = obj.free_symbols
            if symbs:
                symb_dict = {"symbols": {str(s): s._to_dict() for s in symbs}}
            d = {**expr_dict, **symb_dict}
        else:
            if hasattr(obj, "_to_dict"):
                model_to_dict: Callable = obj._to_dict
                d = (
                    model_to_dict(save_params)
                    if isinstance(obj, torch.nn.Module)
                    else model_to_dict()
                )
            elif hasattr(obj, "state_dict"):
                d = {type(obj).__name__: obj.state_dict()}
            else:
                raise ValueError(f"Cannot serialize object {obj}.")
    except Exception as e:
        logger.error(f"Serialization of object {obj} failed due to {e}")
    return d


def deserialize(d: dict, as_torch: bool = False) -> SUPPORTED_TYPES:
    """
    Supported Types:

    AbstractBlock | QuantumCircuit | QuantumModel | Register | torch.nn.Module
    Deserializes a dict to one of the supported types.

    Arguments:
        d (dict): A dict containing a serialized object.
        as_torch (bool): Whether to transform to torch for the deserialized object.
    Returns:
        AbstractBlock, QuantumCircuit, QuantumModel, Register, torch.nn.Module.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import serialize, deserialize, hea, hamiltonian_factory, Z
    from qadence import QuantumCircuit, QuantumModel

    n_qubits = 2
    myblock = hea(n_qubits=n_qubits, depth=1)
    block_dict = serialize(myblock)
    print(block_dict)

    ## Lets use myblock in a QuantumCircuit and serialize it.

    qc = QuantumCircuit(n_qubits, myblock)
    qc_dict = serialize(qc)
    qc_deserialized = deserialize(qc_dict)
    assert qc == qc_deserialized

    ## Finally, let's wrap it in a QuantumModel
    obs = hamiltonian_factory(n_qubits, detuning = Z)
    qm = QuantumModel(qc, obs, backend='pyqtorch', diff_mode='ad')

    qm_dict = serialize(qm)
    qm_deserialized = deserialize(qm_dict)
    # Lets check if the loaded QuantumModel returns the same expectation
    assert torch.isclose(qm.expectation({}), qm_deserialized.expectation({}))
    ```
    """
    obj: SerializationModel
    if d.get("expression"):
        obj = ExpressionSerialization(d)
    elif d.get("block") and d.get("register"):
        obj = QuantumCircuitSerialization(d)
    elif d.get("graph"):
        obj = RegisterSerialization(d)
    elif d.get("type"):
        obj = BlockTypeSerialization(d)
    else:
        obj = ModelSerialization(d, as_torch=as_torch)
    return obj.value


def save(
    obj: SUPPORTED_TYPES,
    folder: str | Path,
    file_name: str = "",
    format: SerializationFormat = SerializationFormat.JSON,
) -> None:
    """
    Same as serialize/deserialize but for storing/loading files.

    Supported types:
    AbstractBlock | QuantumCircuit | QuantumModel | Register | torch.nn.Module
    Saves a qadence object to a json/.pt.

    Arguments:
        obj (AbstractBlock | QuantumCircuit | QuantumModel | Register):
                Either AbstractBlock, QuantumCircuit, QuantumModel, Register.
        file_name (str): The name of the file.
        format (str): The type of file to save.
    Returns:
        None.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from pathlib import Path
    import os

    from qadence import save, load, hea, hamiltonian_factory, Z
    from qadence import QuantumCircuit, QuantumModel

    n_qubits = 2
    myblock = hea(n_qubits=n_qubits, depth=1)
    qc = QuantumCircuit(n_qubits, myblock)
    # Lets store the circuit in a json file
    save(qc, '.', 'circ')
    loaded_qc = load(Path('circ.json'))
    qc == loaded_qc
    os.remove('circ.json')
    ## Let's wrap it in a QuantumModel and store that
    obs = hamiltonian_factory(n_qubits, detuning = Z)
    qm = QuantumModel(qc, obs, backend='pyqtorch', diff_mode='ad')
    save(qm, folder= '.',file_name= 'quantum_model')
    qm_loaded = load('quantum_model.json')
    os.remove('quantum_model.json')
    ```
    """
    if not isinstance(obj, get_args(SUPPORTED_TYPES)):
        logger.error(f"Serialization of object type {type(obj)} not supported.")
    folder = Path(folder)
    if not folder.is_dir():
        logger.error(NotADirectoryError)
    if file_name == "":
        file_name = type(obj).__name__
    try:
        suffix, save_fn, _, save_params = FORMAT_DICT[format]
        d = serialize(obj, save_params)
        file_path = folder / Path(file_name + suffix)
        save_fn(d, file_path)
        logger.debug(f"Successfully saved {obj} from to {folder}.")
    except Exception as e:
        logger.error(f"Unable to write {type(obj)} to disk due to {e}")


def load(file_path: str | Path, map_location: str = "cpu") -> SUPPORTED_TYPES:
    """
    Same as serialize/deserialize but for storing/loading files.

    Supported types: AbstractBlock | QuantumCircuit | QuantumModel | Register
    Loads a .json or .pt file to one of the supported types.

    Arguments:
        file_path (str): The name of the file.
        map_location (str): In case of a .pt file, on which device to load the object (cpu,cuda).
    Returns:
        A object of type AbstractBlock, QuantumCircuit, QuantumModel, Register.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from pathlib import Path
    import os

    from qadence import save, load, hea, hamiltonian_factory, Z
    from qadence import QuantumCircuit, QuantumModel

    n_qubits = 2
    myblock = hea(n_qubits=n_qubits, depth=1)
    qc = QuantumCircuit(n_qubits, myblock)
    # Lets store the circuit in a json file
    save(qc, '.', 'circ')
    loaded_qc = load(Path('circ.json'))
    qc == loaded_qc
    os.remove('circ.json')
    ## Let's wrap it in a QuantumModel and store that
    obs = hamiltonian_factory(n_qubits, detuning = Z)
    qm = QuantumModel(qc, obs, backend='pyqtorch', diff_mode='ad')
    save(qm, folder= '.',file_name= 'quantum_model')
    qm_loaded = load('quantum_model.json')
    os.remove('quantum_model.json')
    ```
    """
    d = {}
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        raise FileNotFoundError
    FORMAT = file_extension(file_path)
    _, _, load_fn, _ = FORMAT_DICT[FORMAT]  # type: ignore[index]
    try:
        d = load_fn(file_path, map_location)
        logger.debug(f"Successfully loaded {d} from {file_path}.")
    except Exception as e:
        logger.error(f"Unable to load Object from {file_path} due to {e}")
    return deserialize(d)
