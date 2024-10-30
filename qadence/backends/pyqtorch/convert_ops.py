from __future__ import annotations

import re
from functools import partial, reduce
from itertools import chain as flatten
from typing import Any, Callable, Sequence

import pyqtorch as pyq
import sympy
import torch
from pyqtorch.embed import ConcretizedCallable
from torch import (
    Tensor,
    cdouble,
    complex64,
    float64,
    tensor,
)
from torch.nn import Module

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    MatrixBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
    TimeEvolutionBlock,
)
from qadence.blocks.primitive import ProjectorBlock
from qadence.noise import NoiseHandler
from qadence.operations import (
    U,
    multi_qubit_gateset,
    non_unitary_gateset,
    single_qubit_gateset,
    three_qubit_gateset,
    two_qubit_gateset,
)
from qadence.types import NoiseProtocol, OpName

from .config import Configuration

SYMPY_TO_PYQ_MAPPING = {
    sympy.Pow: "pow",
    sympy.cos: "cos",
    sympy.Add: "add",
    sympy.Mul: "mul",
    sympy.sin: "sin",
    sympy.log: "log",
    sympy.tan: "tan",
    sympy.tanh: "tanh",
    sympy.Heaviside: "hs",
    sympy.Abs: "abs",
    sympy.exp: "exp",
    sympy.acos: "acos",
    sympy.asin: "asin",
    sympy.atan: "atan",
}


# Tdagger is not supported currently
supported_gates = list(set(OpName.list()) - set([OpName.TDAGGER]))
"""The set of supported gates.

Tdagger is currently not supported.
"""


def is_single_qubit_chain(block: AbstractBlock) -> bool:
    return (
        isinstance(block, (ChainBlock))
        and block.n_supports == 1
        and all([isinstance(b, (ParametricBlock, PrimitiveBlock)) for b in block])
        and not any([isinstance(b, (ScaleBlock, U)) for b in block])
    )


def extract_parameter(block: ScaleBlock | ParametricBlock, config: Configuration) -> str | Tensor:
    """Extract the parameter as string or its tensor value.

    Args:
        block (ScaleBlock | ParametricBlock): Block to extract parameter from.
        config (Configuration): Configuration instance.

    Returns:
        str | Tensor: Parameter value or symbol.
    """
    if not block.is_parametric:
        tensor_val = tensor([block.parameters.parameter], dtype=complex64)
        return (
            tensor([block.parameters.parameter], dtype=float64)
            if torch.all(tensor_val.imag == 0)
            else tensor_val
        )

    return config.get_param_name(block)[0]


def replace_underscore_floats(s: str) -> str:
    """Replace underscores with periods for all floats in given string.

    Needed for correct parsing of string by sympy parser.

    Args:
        s (str): string expression

    Returns:
        str: transformed string expression
    """

    # Regular expression to match floats written with underscores instead of dots
    float_with_underscore_pattern = r"""
        (?<!\w)            # Negative lookbehind to ensure not part of a word
        -?                 # Optional negative sign
        \d+                # One or more digits (before underscore)
        _                  # The underscore acting as decimal separator
        \d+                # One or more digits (after underscore)
        ([eE][-+]?\d+)?    # Optional exponent part for scientific notation
        (?!\w)             # Negative lookahead to ensure not part of a word
    """

    # Function to replace the underscore with a dot
    def underscore_to_dot(match: re.Match) -> Any:
        return match.group(0).replace("_", ".")

    # Compile the regular expression
    pattern = re.compile(float_with_underscore_pattern, re.VERBOSE)

    return pattern.sub(underscore_to_dot, s)


def sympy_to_pyq(expr: sympy.Expr) -> ConcretizedCallable | Tensor:
    """Convert sympy expression to pyqtorch ConcretizedCallable object.

    Args:
        expr (sympy.Expr): sympy expression

    Returns:
        ConcretizedCallable: expression encoded as ConcretizedCallable
    """

    # base case - independent argument
    if len(expr.args) == 0:
        try:
            res = torch.as_tensor(float(expr))
        except Exception as e:
            res = str(expr)

            if "/" in res:  # Found a rational
                res = torch.as_tensor(float(sympy.Rational(res).evalf()))
        return res

    # Recursively iterate through current function arguments
    all_results = []
    for arg in expr.args:
        res = sympy_to_pyq(arg)
        all_results.append(res)

    # deal with multi-argument (>2) sympy functions: converting to nested
    # ConcretizedCallable objects
    if len(all_results) > 2:

        def fn(x: str | ConcretizedCallable, y: str | ConcretizedCallable) -> Callable:
            return partial(ConcretizedCallable, call_name=SYMPY_TO_PYQ_MAPPING[expr.func])(  # type: ignore [no-any-return]
                abstract_args=[x, y]
            )

        concretized_callable = reduce(fn, all_results)
    else:
        concretized_callable = ConcretizedCallable(SYMPY_TO_PYQ_MAPPING[expr.func], all_results)
    return concretized_callable


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = None
) -> Sequence[Module | Tensor | str | sympy.Expr]:
    if isinstance(block, (Tensor, str, sympy.Expr)):  # case for hamevo generators
        if isinstance(block, Tensor):
            block = block.permute(1, 2, 0)  # put batch size in the back
        return [block]
    qubit_support = block.qubit_support
    if n_qubits is None:
        n_qubits = max(qubit_support) + 1

    if config is None:
        config = Configuration()

    noise: NoiseHandler | None = None
    if hasattr(block, "noise") and block.noise:
        noise = convert_digital_noise(block.noise)

    if isinstance(block, ScaleBlock):
        scaled_ops = convert_block(block.block, n_qubits, config)
        scale = extract_parameter(block, config=config)

        # replace underscore by dot when underscore is between two numbers in string
        if isinstance(scale, str):
            scale = replace_underscore_floats(scale)

        if isinstance(scale, str) and not config._use_gate_params:
            param = sympy_to_pyq(sympy.parse_expr(scale))
        else:
            param = scale

        return [pyq.Scale(pyq.Sequence(scaled_ops), param)]

    elif isinstance(block, TimeEvolutionBlock):
        if getattr(block.generator, "is_time_dependent", False):
            config._use_gate_params = False
            generator = convert_block(block.generator, config=config)[0]  # type: ignore [arg-type]
        elif isinstance(block.generator, sympy.Basic):
            generator = config.get_param_name(block)[1]

        elif isinstance(block.generator, Tensor):
            m = block.generator.to(dtype=cdouble)
            generator = convert_block(
                MatrixBlock(
                    m,
                    qubit_support=qubit_support,
                    check_unitary=False,
                    check_hermitian=True,
                )
            )[0]
        else:
            generator = convert_block(block.generator, n_qubits, config)[0]  # type: ignore[arg-type]
        time_param = config.get_param_name(block)[0]

        return [
            pyq.HamiltonianEvolution(
                qubit_support=qubit_support,
                generator=generator,
                time=time_param,
                cache_length=0,
            )
        ]

    elif isinstance(block, MatrixBlock):
        return [pyq.primitives.Primitive(block.matrix, block.qubit_support, noise=noise)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if isinstance(block, AddBlock):
            return [pyq.Add(ops)]  # add
        elif is_single_qubit_chain(block) and config.use_single_qubit_composition:
            return [pyq.Merge(ops)]  # for chains of single qubit ops on the same qubit
        else:
            return [pyq.Sequence(ops)]  # for kron and chain
    elif isinstance(block, tuple(non_unitary_gateset)):
        if isinstance(block, ProjectorBlock):
            projector = getattr(pyq, block.name)
            if block.name == OpName.N:
                return [projector(target=qubit_support, noise=noise)]
            else:
                return [
                    projector(
                        qubit_support=qubit_support,
                        ket=block.ket,
                        bra=block.bra,
                        noise=noise,
                    )
                ]
        else:
            return [getattr(pyq, block.name)(qubit_support[0])]
    elif isinstance(block, tuple(single_qubit_gateset)):
        pyq_cls = getattr(pyq, block.name)
        if isinstance(block, ParametricBlock):
            if isinstance(block, U):
                op = pyq_cls(
                    qubit_support[0],
                    *config.get_param_name(block),
                    noise=noise,
                )
            else:
                param = extract_parameter(block, config)
                op = pyq_cls(qubit_support[0], param, noise=noise)
        else:
            op = pyq_cls(qubit_support[0], noise=noise)  # type: ignore [attr-defined]
        return [op]
    elif isinstance(block, tuple(two_qubit_gateset)):
        pyq_cls = getattr(pyq, block.name)
        if isinstance(block, ParametricBlock):
            op = pyq_cls(
                qubit_support[0],
                qubit_support[1],
                extract_parameter(block, config),
                noise=noise,
            )
        else:
            op = pyq_cls(
                qubit_support[0], qubit_support[1], noise=noise  # type: ignore [attr-defined]
            )
        return [op]
    elif isinstance(block, tuple(three_qubit_gateset) + tuple(multi_qubit_gateset)):
        block_name = block.name[1:] if block.name.startswith("M") else block.name
        pyq_cls = getattr(pyq, block_name)
        if isinstance(block, ParametricBlock):
            op = pyq_cls(
                qubit_support[:-1],
                qubit_support[-1],
                extract_parameter(block, config),
                noise=noise,
            )
        else:
            if "CSWAP" in block_name:
                op = pyq_cls(
                    qubit_support[:-2], qubit_support[-2:], noise=noise  # type: ignore [attr-defined]
                )
            else:
                op = pyq_cls(
                    qubit_support[:-1], qubit_support[-1], noise=noise  # type: ignore [attr-defined]
                )
        return [op]
    else:
        raise NotImplementedError(
            f"Non supported operation of type {type(block)}. "
            "In case you are trying to run an `AnalogBlock`, make sure you "
            "specify the `device_specs` in your `Register` first."
        )


def convert_digital_noise(noise: NoiseHandler) -> pyq.noise.NoiseProtocol:
    digital_part = noise.filter(NoiseProtocol.DIGITAL)
    if digital_part is None:
        return None
    return pyq.noise.NoiseProtocol(
        [
            pyq.noise.NoiseProtocol(proto, option.get("error_probability"))
            for proto, option in zip(digital_part.protocol, digital_part.options)
        ]
    )
