from __future__ import annotations

import math
import re
from collections import Counter
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy
from numpy.typing import ArrayLike
from torch import Tensor, stack, vmap
from torch import complex as make_complex
from torch.linalg import eigvals

from rich.tree import Tree

from qadence.blocks import AbstractBlock
from qadence.types import Endianness, ResultType, TNumber, ParamDictType

if TYPE_CHECKING:
    from qadence.operations import Projector


# Modules to be automatically added to the qadence namespace
__all__ = []  # type: ignore


logger = getLogger(__name__)


def merge_separate_params(param_dict: ParamDictType) -> ParamDictType:
    """Merge circuit and observables parameters."""
    merged_dict: ParamDictType = dict()
    for p in param_dict.values():
        merged_dict |= p
    return merged_dict


def check_param_dict_values(param_dict: ParamDictType) -> bool:
    """Check if `param_dict` contains array values.

    Args:
        param_dict (ParamDictType): Dictionary of parameters.

    Returns:
        bool: True if values are arrays, False if values are dict.
    """
    if all(isinstance(p, dict) for p in param_dict.values()):
        return False
    return True


def basis_to_int(basis: str, endianness: Endianness = Endianness.BIG) -> int:
    """
    Converts a computational basis state to an int.

    - `endianness = "Big"` reads the most significant bit in qubit 0 (leftmost).
    - `endianness = "Little"` reads the least significant bit in qubit 0 (leftmost).

    Arguments:
        basis (str): A computational basis state.
        endianness (Endianness): The Endianness when reading the basis state.

    Returns:
        The corresponding integer.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.utils import basis_to_int, Endianness

    k = basis_to_int(basis="10", endianness=Endianness.BIG)
    print(k)
    ```
    """
    if endianness == Endianness.BIG:
        return int(basis, 2)
    else:
        return int(basis[::-1], 2)


def int_to_basis(
    k: int, n_qubits: int | None = None, endianness: Endianness = Endianness.BIG
) -> str:
    """
    Converts an integer to its corresponding basis state.

    - `endianness = "Big"` stores the most significant bit in qubit 0 (leftmost).
    - `endianness = "Little"` stores the least significant bit in qubit 0 (leftmost).

    Arguments:
        k (int): The int to convert.
        n_qubits (int): The total number of qubits in the basis state.
        endianness (Endianness): The Endianness of the resulting basis state.

    Returns:
        A computational basis state.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.utils import int_to_basis, Endianness

    bs = int_to_basis(k=1, n_qubits=2, endianness=Endianness.BIG)
    print(bs)
    ```
    """
    if n_qubits is None:
        n_qubits = int(math.log(k + 0.6) / math.log(2)) + 1
    assert k <= 2**n_qubits - 1, "k can not be larger than 2**n_qubits-1."
    basis = format(k, "0{}b".format(n_qubits))
    if endianness == Endianness.BIG:
        return basis
    else:
        return basis[::-1]


def nqubits_to_basis(
    n_qubits: int,
    result_type: ResultType = ResultType.STRING,
    endianness: Endianness = Endianness.BIG,
) -> list[str] | Tensor | np.array:
    """
    Creates all basis states for a given number of qubits, endianness and format.

    Arguments:
        n_qubits: The total number of qubits.
        result_type: The data type of the resulting states.
        endianness: The Endianness of the resulting states.

    Returns:
        The full computational basis for n_qubits.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence.utils import nqubits_to_basis, Endianness, ResultType
    basis_type = ResultType.Torch
    bs = nqubits_to_basis(n_qubits=2, result_type= basis_type, endianness=Endianness.BIG)
    print(bs)
    ```
    """
    basis_strings = [int_to_basis(k, n_qubits, endianness) for k in range(0, 2**n_qubits)]
    if result_type == ResultType.STRING:
        return basis_strings
    else:
        basis_list = [list(map(int, tuple(basis))) for basis in basis_strings]
        if result_type == ResultType.TORCH:
            return stack([Tensor(basis) for basis in basis_list])
        elif result_type == ResultType.NUMPY:
            return np.stack([np.array(basis) for basis in basis_list])


def samples_to_integers(samples: Counter, endianness: Endianness = Endianness.BIG) -> Counter:
    """
    Converts a Counter of basis state samples to integer values.

    Args:
        samples (Counter({bits: counts})): basis state sample counter.
        endianness (Endianness): endianness to use for conversion.

    Returns:
        Counter({ints: counts}): samples converted
    """

    return Counter({basis_to_int(k, endianness): v for k, v in samples.items()})


def format_number(x: float | complex, num_digits: int = 3) -> str:
    if isinstance(x, int):
        return f"{x}"
    elif isinstance(x, float):
        return f"{x:.{num_digits}f}"
    elif isinstance(x, complex):
        re = "" if np.isclose(x.real, 0) else f"{x.real:.{num_digits}f}"
        im = "" if np.isclose(x.imag, 0) else f"{x.imag:.{num_digits}f}"
        if len(re) > 0 and len(im) > 0:
            return f"{re}+{im}j"
        elif len(re) > 0 and len(im) == 0:
            return re
        elif len(re) == 0 and len(im) > 0:
            return f"{im}j"
        else:
            return "0"
    else:
        raise ValueError(f"Unknown number type: {type(x)}")


def format_parameter(p: sympy.Basic, num_digits: int = 3) -> str:
    """Format numerical values within a sympy expression."""

    def round_expr(expr: sympy.Basic, num_digits: int) -> sympy.Basic:
        return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})

    expr = round_expr(p, num_digits)
    conv_str = str(expr)
    if expr.is_real and len(conv_str) > num_digits + 2:
        conv_str = conv_str[0 : num_digits + 2]
    return conv_str


def print_sympy_expr(expr: sympy.Expr, num_digits: int = 3) -> str:
    """
    Converts numerical values in a sympy expression.

    The result is a numerical expression with fewer digits for better readability.
    """
    from qadence.parameters import sympy_to_numeric

    round_dict = {sympy_to_numeric(n): round(n, num_digits) for n in expr.atoms(sympy.Number)}
    return str(expr.xreplace(round_dict))


def isclose(
    x: TNumber | Any, y: TNumber | Any, rel_tol: float = 1e-5, abs_tol: float = 1e-07
) -> bool:
    if isinstance(x, complex) or isinstance(y, complex):
        return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)  # type: ignore

    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)  # type: ignore


def eigenvalues(
    x: Tensor, max_num_evals: int | None = None, max_num_gaps: int | None = None
) -> Tensor:
    # FIXME: Currently doing full decomposition everytime because it is generally faster
    # than converting to numpy and using the scipy routines to select first k-eigenvalues.
    # Furthermore, we should exploit the matrices being Hermitian.

    # get all eigenvalues of generator
    eigenspectrum = eigvals(x).squeeze(0)

    if max_num_evals and not max_num_gaps:
        # get specified number of eigenvalues of generator
        eigenvals = eigenspectrum[:max_num_evals]
    elif max_num_gaps and not max_num_evals:
        # get eigenvalues of generator corresponding to specified number of spectral gaps
        k = int(np.ceil(0.5 * (1 + np.sqrt(1 + 8 * max_num_gaps))))
        eigenvals = eigenspectrum[:k]
    else:
        eigenvals = eigenspectrum

    return eigenvals


def _round_complex(t: Tensor, decimals: int = 4) -> Tensor:
    def _round(_t: Tensor) -> Tensor:
        r = _t.real.round(decimals=decimals)
        i = _t.imag.round(decimals=decimals)
        return make_complex(r, i)

    fn = vmap(_round)
    return fn(t)


def is_qadence_shape(state: ArrayLike, n_qubits: int) -> bool:
    if len(state.shape) < 2:
        raise ValueError(
            f"Provided state is required to have atleast two dimensions. Got shape {state.shape}"
        )
    return state.shape[1] == 2**n_qubits  # type: ignore[no-any-return]


def validate_values_and_state(
    state: ArrayLike | None, n_qubits: int, param_values: dict[str, Tensor] = None
) -> None:
    if state is not None:
        from qadence.backends.utils import infer_batchsize

        if isinstance(state, Tensor):
            if state is not None:
                batch_size_state = (
                    state.shape[0] if is_qadence_shape(state, n_qubits=n_qubits) else state.size(-1)
                )
                batch_size_values = infer_batchsize(param_values)
                if batch_size_state != batch_size_values and (
                    batch_size_values > 1 and batch_size_state > 1
                ):
                    raise ValueError(
                        "Batching of parameter values and states is only valid for the cases:\
                                    (1) batch_size_values == batch_size_state\
                                    (2) batch_size_values == 1 and batch_size_state > 1\
                                    (3) batch_size_values > 1 and batch_size_state == 1."
                    )
        else:
            if not is_qadence_shape(state, n_qubits) or state.shape[0] > 1:
                raise ValueError("Jax only supports unbatched states.")


def one_qubit_projector(state: str, target: int) -> Projector:
    """Returns the projector for a single qubit system.

    Args:
        state (str): The state of the projector.
        target (int): The target qubit.

    Returns:
        Projector: The projector operator.
    """
    from qadence.operations import Projector

    assert state in ["0", "1"], "State must be either '0' or '1'."
    return Projector(ket=state, bra=state, qubit_support=target)


def one_qubit_projector_matrix(state: str) -> Tensor:
    """Returns the projector for a single qubit system.

    Args:
        state (str): The state of the projector.

    Returns:
        Tensor: The projector operator.
    """
    return one_qubit_projector(state, 0).tensor().squeeze()


P0 = partial(one_qubit_projector, "0")
P1 = partial(one_qubit_projector, "1")


def block_to_mathematical_expression(block: Tree | AbstractBlock) -> str:
    """Convert a block to a readable mathematical expression.

        Useful for printing Observables as a mathematical expression.

    Args:
        block (AbstractBlock): Tree instance.

    Returns:
        str: A mathematical expression.
    """
    block_tree: Tree = block.__rich_tree__() if isinstance(block, AbstractBlock) else block
    block_title = block_tree.label if isinstance(block_tree.label, str) else ""
    if "AddBlock" in block_title:
        block_title = " + ".join(
            [block_to_mathematical_expression(block_child) for block_child in block_tree.children]
        )
    if "KronBlock" in block_title:
        block_title = " ⊗ ".join(
            [block_to_mathematical_expression(block_child) for block_child in block_tree.children]
        )
    if "mul" in block_title:
        block_title = re.findall("\[mul:\s*(.+?)\]", block_title)[0]

        try:
            if "." in block_title:
                coeff = float(block_title)
            else:
                coeff = int(block_title)
            if coeff == 0:
                block_title = ""
            elif coeff == 1:
                block_title = block_to_mathematical_expression(block_tree.children[0])
            else:
                block_title += " * " + block_to_mathematical_expression(block_tree.children[0])

        except ValueError:  # In case block_title is a non-numeric str (e.g. parameter name)
            block_title += " * " + block_to_mathematical_expression(block_tree.children[0])

    first_part = block_title[:3]
    if first_part in [" + ", " ⊗ ", " * "]:
        block_title = block_title[3:]

    # if too many trees, add parentheses.
    nb_children = len(block_tree.children)
    if nb_children > 1:
        block_title = "(" + block_title + ")"
    return block_title
