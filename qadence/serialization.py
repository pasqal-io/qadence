from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    get_args,
    Optional,
    Union as TypingUnion
)
from abc import ABC
from importlib import import_module
from dataclasses import dataclass, field, InitVar

import ast
import torch
from sympy import core, srepr

from qadence import QuantumCircuit, operations, parameters
from qadence import blocks as qadenceblocks
from qadence.blocks import AbstractBlock
from qadence.blocks.utils import tag
from qadence.logger import get_logger
from qadence.models import QuantumModel  #, QNN
# from qadence.ml_tools.models import TransformedModule
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
    # QNN,
    # TransformedModule,
    Register,
    core.Basic,
    torch.nn.Module,
]
SUPPORTED_TYPES = TypingUnion[
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    # QNN,
    # TransformedModule,
    Register,
    core.Basic,
    torch.nn.Module,
]

EXPR_TYPES = TypingUnion[core.Expr, float, int, str, tuple, dict]


ALL_BLOCK_NAMES = [
    n for n in dir(qadenceblocks) if not (n.startswith("__") and n.endswith("__"))
] + [n for n in dir(operations) if not (n.startswith("__") and n.endswith("__"))]
SYMPY_EXPRS = [n for n in dir(core) if not (n.startswith("__") and n.endswith("__"))]
QADENCE_PARAMETERS = [n for n in dir(parameters) if not (n.startswith("__") and n.endswith("__"))]


def parse_expr(expr: str) -> list:
    """
    Parser function for the string deserialization.

    Arguments:
        expr (str): The expression to be parsed.

    Returns:
        Nested lists with strings.
    """
    s_expr: str = ""
    parsed_expr: list = []
    # parenthesis counter
    p_count: int = 0
    for k in expr.replace(" ", ""):
        if k == "(":
            if s_expr:
                caller = [s_expr.replace(",", "").replace(" ", "")]
                if p_count % 2 == 0:
                    parsed_expr.extend(caller)
                else:
                    parsed_expr.append(caller)
                s_expr = ""
            p_count += 1
        elif k == ")":
            p_count -= 1
            if s_expr:
                args: list = []
                for q in s_expr.split(","):
                    u = q.split("=")
                    if len(u) == 1:
                        args.extend(u)
                    else:
                        args.append(tuple(u))
                if p_count % 2 != 0:
                    parsed_expr[-1].append(args)
                else:
                    parsed_expr.append(args)
                s_expr = ""
        else:
            s_expr += k
    return parsed_expr


def eval_expr(expr: list) -> EXPR_TYPES:
    """
    Evaluate the parsed expression from the `parse_expr` function. This procedure
    avoids the use of Python's `eval` to execute arbitrary string. Meant for use
    on `deserialize` function to acquire the expressions back, from either a
    dictionary or a json file.

    Arguments:
        expr (list): The parsed expression from `parse_expr`.

    Returns:
        The evaluation result.
    """
    if isinstance(expr, list):
        vals: list = []
        args: dict = dict()
        caller: Optional[Callable] = None
        for k in expr:
            p = eval_expr(k)
            if isinstance(p, type):
                caller = p
            elif isinstance(p, dict):
                args.update(p)
            elif isinstance(p, tuple):
                vals.extend(p[0])
                args.update(p[1])
            else:
                vals.append(p)

        if callable(caller):
            return caller(*vals, **args)
        return vals, args
    elif isinstance(expr, tuple):
        return {expr[0]: expr[1]}
    elif isinstance(expr, str):
        if expr in QADENCE_PARAMETERS:
            return getattr(parameters, expr)
        if expr in SYMPY_EXPRS:
            return getattr(core, expr)
        try:
            return ast.literal_eval(expr)
        except Exception as e:
            return expr


@dataclass
class SerialModel(ABC):
    value: Any = field(init=False)


@dataclass
class BlockTypeSerial(SerialModel):
    d: InitVar[Any]
    value: AbstractBlock = field(init=False)

    def __post_init__(self, d: Any) -> None:
        block = (
            getattr(operations, d["type"])
            if hasattr(operations, d["type"])
            else getattr(qadenceblocks, d["type"])
        )._from_dict(d)
        if d["tag"] is not None:
            block = tag(block, d["tag"])
        self.value = block


@dataclass
class QuantumCircuitSerial(SerialModel):
    d: InitVar[dict]
    value: QuantumCircuit = field(init=False)

    def __post_init__(self, d: dict) -> None:
        self.value = (
            QuantumCircuit._from_dict(d)
            if isinstance(d, dict)
            else d
        )


@dataclass
class GraphSerial(SerialModel):
    d: InitVar[dict]
    value: Register = field(init=False)

    def __post_init__(self, d: dict) -> None:
        self.value = Register._from_dict(d)


@dataclass
class ModelSerial(SerialModel):
    d: InitVar[dict]
    as_torch: bool = False
    value: torch.nn.Module = field(init=False)

    def __post_init__(self, d: dict) -> None:
        module_name = list(d.keys())[0]
        obj = globals().get(module_name, None)
        if obj is None:
            obj = self._resolve_module(module_name)
        if hasattr(obj, "_from_dict"):
            self.value = obj._from_dict(d, self.as_torch)
        elif hasattr(obj, "load_state_dict"):
            self.value = obj.load_state_dict(d[module_name])
        else:
            raise ValueError(f"Module '{module_name}' not found.")

    @staticmethod
    def _resolve_module(module: str) -> Any:
        for loaded_module in sys.modules.keys():
            if "qadence" in loaded_module:
                obj = getattr(sys.modules[loaded_module], module, None)
                if obj:
                    return obj
        raise ValueError(f"Couldn't resolve module '{module}'.")


@dataclass
class ExpressionSerial(SerialModel):
    d: InitVar[dict]
    value: str | core.Expr | float = field(init=False)

    def __post_init__(self, d: dict) -> None:
        parsed_expr = parse_expr(d["expression"])
        expr = eval_expr(parsed_expr)
        if hasattr(expr, "free_symbols"):
            # TODO: check whether it is really necessary
            for s in expr.free_symbols:
                s.value = float(d["symbols"][s.name]["value"])
        self.value = expr


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
    # if not isinstance(obj, get_args(SUPPORTED_TYPES)):
    #     logger.error(TypeError(f"Serialization of object type {type(obj)} not supported."))
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
                fn: Callable = obj._to_dict
                d = (
                    fn(save_params)
                    if isinstance(obj, torch.nn.Module)
                    else fn()
                )
            else:
                d = {type(obj).__name__: obj.state_dict()}
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
        AbstractBlock, QuantumCircuit, QuantumModel, TransformedModule, Register, torch.nn.Module.

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
    obj: SerialModel
    if d.get("expression"):
        obj = ExpressionSerial(d)
    elif d.get("block") and d.get("register"):
        obj = QuantumCircuitSerial(d)
    elif d.get("graph"):
        obj = GraphSerial(d)
    elif d.get("type"):
        obj = BlockTypeSerial(d)
    else:
        obj = ModelSerial(d, as_torch=as_torch)
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
