from __future__ import annotations

from typing import Callable, Iterable, List

import jax.numpy as jnp
import numpy as np
import sympy
import sympy2jax as s2j
import sympytorch  # type: ignore [import]
import torch

from qadence.blocks import (
    AbstractBlock,
)
from qadence.blocks.utils import (
    expressions,
    parameters,
    uuid_to_expression,
)
from qadence.parameters import evaluate, make_differentiable, stringify
from qadence.types import Engine, ParamDictType, ReturnType, TNumber


def concretize_parameter(value: TNumber, requires_grad: bool, engine: Engine) -> ReturnType:
    if engine == Engine.JAX:
        return jnp.array([value], dtype=jnp.float64)
    else:
        return torch.tensor([value], requires_grad=requires_grad)


def unique(x: Iterable) -> List:
    return list(set(x))


def embedding(
    block: AbstractBlock, to_gate_params: bool = False, engine: Engine = Engine.TORCH
) -> tuple[ParamDictType, Callable[[ParamDictType, ParamDictType], ParamDictType],]:
    """Construct embedding function which maps user-facing parameters to either *expression-level*.

    parameters or *gate-level* parameters. The constructed embedding function has the signature:

         embedding_fn(params: ParamDictType, inputs: ParamDictType) -> ParamDictType:

    which means that it maps the *variational* parameter dict `params` and the *feature* parameter
    dict `inputs` to one new parameter dict `embedded_dict` which holds all parameters that are
    needed to execute a circuit on a given backend. There are two different *modes* for this
    mapping:

    - *Expression-level* parameters: For AD-based optimization. For every unique expression we end
      up with one entry in the embedded dict:
      `len(embedded_dict) == len(unique_parameter_expressions)`.
    - *Gate-level* parameters: For PSR-based optimization or real devices. One parameter for each
      gate parameter, regardless if they are based on the same expression. `len(embedded_dict) ==
      len(parametric_gates)`. This is needed because PSR requires to shift the angles of **every**
      gate where the same parameter appears.

    Arguments:
        block: parametrized block into which we want to embed parameters.
        to_gate_params: A boolean flag whether to generate gate-level parameters or
            expression-level parameters.

    Returns:
        A tuple with variational parameter dict and the embedding function.
    """

    unique_expressions = unique(expressions(block))
    unique_symbols = [p for p in unique(parameters(block)) if not isinstance(p, sympy.Array)]
    unique_const_matrices = [e for e in unique_expressions if isinstance(e, sympy.Array)]
    unique_expressions = [e for e in unique_expressions if not isinstance(e, sympy.Array)]

    # NOTE
    # there are 3 kinds of parameters in qadence
    # - non-trainable which are considered as inputs for classical data
    # - trainable which are the variational parameters to be optimized
    # - fixed: which are non-trainable parameters with fixed value (e.g. pi/2)
    #
    # both non-trainable and trainable parameters can have the same element applied
    # to different operations in the quantum circuit, e.g. assigning the same parameter
    # to multiple gates.
    non_numeric_symbols = [p for p in unique_symbols if not p.is_number]
    trainable_symbols = [p for p in non_numeric_symbols if p.trainable]
    constant_expressions = [expr for expr in unique_expressions if expr.is_number]
    # we dont need to care about constant symbols if they are contained in an symbolic expression
    # we only care about gate params which are ONLY a constant

    embeddings: dict[sympy.Expr, sympytorch.SymPyModule | s2j.SymbolicModule] = {
        expr: make_differentiable(expr=expr, engine=engine)
        for expr in unique_expressions
        if not expr.is_number
    }

    uuid_to_expr = uuid_to_expression(block)

    def embedding_fn(params: ParamDictType, inputs: ParamDictType) -> ParamDictType:
        embedded_params: dict[sympy.Expr, ReturnType] = {}
        for expr, fn in embeddings.items():
            angle: ReturnType
            values = {}
            for symbol in expr.free_symbols:
                if symbol.name in inputs:
                    value = inputs[symbol.name]
                elif symbol.name in params:
                    value = params[symbol.name]
                else:
                    msg_trainable = "Trainable" if symbol.trainable else "Non-trainable"
                    raise KeyError(
                        f"{msg_trainable} parameter '{symbol.name}' not found in the "
                        f"inputs list: {list(inputs.keys())} nor the "
                        f"params list: {list(params.keys())}."
                    )
                values[symbol.name] = value
            angle = fn(**values)
            # do not reshape parameters which are multi-dimensional
            # tensors, such as for example generator matrices
            if not len(angle.squeeze().shape) > 1:
                angle = angle.reshape(-1)
            embedded_params[expr] = angle

        for e in constant_expressions + unique_const_matrices:
            embedded_params[e] = params[stringify(e)]

        if to_gate_params:
            gate_lvl_params: ParamDictType = {}
            for uuid, e in uuid_to_expr.items():
                gate_lvl_params[uuid] = embedded_params[e]
            return gate_lvl_params
        else:
            return {stringify(k): v for k, v in embedded_params.items()}

    params: ParamDictType
    params = {p.name: concretize_parameter(p.value, True, engine) for p in trainable_symbols}
    params.update(
        {
            stringify(expr): concretize_parameter(
                evaluate(expr), requires_grad=False, engine=engine
            )
            for expr in constant_expressions
        }
    )
    fn = jnp.array if engine == Engine.JAX else torch.tensor
    params.update(
        {
            stringify(expr): fn(np.array(expr.tolist(), dtype=np.cdouble))
            for expr in unique_const_matrices
        }
    )
    return params, embedding_fn
