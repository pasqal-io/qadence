from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, get_args
from typing import Union as TypingUnion

import torch
from sympy import *
from sympy import Basic, Expr, srepr

from qadence import QuantumCircuit, operations
from qadence import blocks as qadenceblocks
from qadence.blocks import AbstractBlock
from qadence.blocks.utils import tag
from qadence.logger import get_logger
from qadence.ml_tools.models import TransformedModule
from qadence.models import QNN, QuantumModel
from qadence.parameters import Parameter
from qadence.register import Register
from qadence.types import SerializationFormat

# Modules to be automatically added to the qadence namespace
__all__ = ["deserialize", "load", "save", "serialize"]


logger = get_logger(__name__)


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
    TransformedModule,
    Register,
    Basic,
    torch.nn.Module,
]
SUPPORTED_TYPES = TypingUnion[
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    QNN,
    TransformedModule,
    Register,
    Basic,
    torch.nn.Module,
]


ALL_BLOCK_NAMES = [
    n for n in dir(qadenceblocks) if not (n.startswith("__") and n.endswith("__"))
] + [n for n in dir(operations) if not (n.startswith("__") and n.endswith("__"))]


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

    AbstractBlock | QuantumCircuit | QuantumModel | TransformedModule | Register | Module
    Serializes a qadence object to a dictionary.

    Arguments:
        obj (AbstractBlock | QuantumCircuit | QuantumModel | Register | Module):
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
    d: dict = {}
    try:
        if isinstance(obj, Expr):
            symb_dict = {}
            expr_dict = {"name": str(obj), "expression": srepr(obj)}
            symbs: set[Parameter | Basic] = obj.free_symbols
            if symbs:
                symb_dict = {"symbols": {str(s): s._to_dict() for s in symbs}}
            d = {**expr_dict, **symb_dict}
        elif isinstance(obj, (QuantumModel, QNN, TransformedModule)):
            d = obj._to_dict(save_params)
        elif isinstance(obj, torch.nn.Module):
            d = {type(obj).__name__: obj.state_dict()}
        else:
            d = obj._to_dict()
    except Exception as e:
        logger.error(f"Serialization of object {obj} failed due to {e}")
    return d


def deserialize(d: dict, as_torch: bool = False) -> SUPPORTED_TYPES:
    """
    Supported Types:

    AbstractBlock | QuantumCircuit | QuantumModel | TransformedModule | Register | Module
    Deserializes a dict to one of the supported types.

    Arguments:
        d (dict): A dict containing a serialized object.
    Returns:
        AbstractBlock, QuantumCircuit, QuantumModel, TransformedModule, Register, Module.

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
    obj: Any
    if d.get("expression"):
        expr = eval(d["expression"])
        if hasattr(expr, "free_symbols"):
            for symb in expr.free_symbols:
                symb.value = float(d["symbols"][symb.name]["value"])
        obj = expr
    elif d.get("QuantumModel"):
        obj = QuantumModel._from_dict(d, as_torch)
    elif d.get("QNN"):
        obj = QNN._from_dict(d, as_torch)
    elif d.get("TransformedModule"):
        obj = TransformedModule._from_dict(d, as_torch)
    elif d.get("block") and d.get("register"):
        obj = QuantumCircuit._from_dict(d)
    elif d.get("graph"):
        obj = Register._from_dict(d)
    elif d.get("type"):
        if d["type"] in ALL_BLOCK_NAMES:
            block: AbstractBlock = (
                getattr(operations, d["type"])._from_dict(d)
                if hasattr(operations, d["type"])
                else getattr(qadenceblocks, d["type"])._from_dict(d)
            )
            if d["tag"] is not None:
                block = tag(block, d["tag"])
            obj = block
    else:
        import warnings

        msg = warnings.warn(
            "In order to load a custom torch.nn.Module, make sure its imported in the namespace."
        )
        try:
            module_name = list(d.keys())[0]
            obj = getattr(globals(), module_name)
            obj.load_state_dict(d[module_name])
        except Exception as e:
            logger.error(
                TypeError(
                    f"{msg}. Unable to deserialize object due to {e}.\
                    Supported objects are: {SUPPORTED_OBJECTS}"
                )
            )
    return obj


def save(
    obj: SUPPORTED_TYPES,
    folder: str | Path,
    file_name: str = "",
    format: SerializationFormat = SerializationFormat.JSON,
) -> None:
    """
    Same as serialize/deserialize but for storing/loading files.

    Supported types:
    AbstractBlock | QuantumCircuit | QuantumModel | TransformedModule | Register | torch.nn.Module
    Saves a qadence object to a json/.pt.

    Arguments:
        obj (AbstractBlock | QuantumCircuit | QuantumModel | Register):
                Either AbstractBlock, QuantumCircuit, QuantumModel, TransformedModule, Register.
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

    Supported types: AbstractBlock | QuantumCircuit | QuantumModel | TransformedModule | Register
    Loads a .json or .pt file to one of the supported types.

    Arguments:
        file_path (str): The name of the file.
        map_location (str): In case of a .pt file, on which device to load the object (cpu,cuda).
    Returns:
        A object of type AbstractBlock, QuantumCircuit, QuantumModel, TransformedModule, Register.

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
