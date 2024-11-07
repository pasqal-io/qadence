from __future__ import annotations

from logging import getLogger
from typing import Any, ItemsView, KeysView, ValuesView, get_args
from uuid import uuid4

import jsonschema
import numpy as np
import sympy
from sympy import *
from sympy import Array, Basic, Expr, Symbol, sympify
from sympy.physics.quantum.dagger import Dagger
from sympytorch import SymPyModule as torchSympyModule
from torch import Tensor, heaviside, no_grad, rand, tensor

from qadence.types import DifferentiableExpression, Engine, TNumber

# Modules to be automatically added to the qadence namespace
__all__ = ["FeatureParameter", "Parameter", "VariationalParameter", "ParamMap", "TimeParameter"]

logger = getLogger(__name__)

dagger_expression = Dagger

ParameterJSONSchema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "trainable": {"type": "string"},
        "value": {"type": "string"},
    },
    "oneOf": [
        {
            "allOf": [
                {"required": ["name"]},
                {"not": {"required": ["trainable"]}},
                {"not": {"required": ["value"]}},
            ]
        },
        {"allOf": [{"required": ["name", "trainable"]}, {"not": {"required": ["value"]}}]},
        {"required": ["name", "trainable", "value"]},
    ],
}


class Parameter(Symbol):
    """
    A wrapper on top of `sympy.Symbol`.

    Includes two additional keywords: `trainable` and `value`.
    This class is to define both feature parameter and variational parameters.
    """

    trainable: bool
    """Trainable parameters are *variational* parameters.

    Non-trainable parameters are *feature*
    parameters.
    """
    value: TNumber
    """(Initial) value of the parameter."""

    is_time: bool

    def __new__(
        cls, name: str | TNumber | Tensor | Basic | Parameter, **assumptions: Any
    ) -> Parameter | Basic | Expr | Array:
        """
        Arguments:

            name: When given a string only, the class
                constructs a trainable Parameter with a a randomly initialized value.
            **assumptions: are passed on to the parent class `sympy.Symbol`. Two new assumption
                kwargs are supported by this constructor: `trainable: bool`, and `value: TNumber`.

        Example:
        ```python exec="on" source="material-block" result="json"
        from qadence.parameters import Parameter, VariationalParameter

        theta = Parameter("theta")
        print(f"{theta}: trainable={theta.trainable} value={theta.value}")
        assert not theta.is_number

        # you can specify both trainable/value in the constructor
        theta = Parameter("theta", trainable=True, value=2.0)
        print(f"{theta}: trainable={theta.trainable} value={theta.value}")

        # VariationalParameter/FeatureParameter are constructing
        # trainable/untrainable Parameters
        theta = VariationalParameter("theta", value=2.0)
        assert theta == Parameter("theta", trainable=True, value=2.0)

        # When provided with a numeric type, Parameter constructs a sympy numeric type":
        constant_zero = Parameter(0)
        assert constant_zero.is_number

        # When passed a Parameter or a sympy expression, it just returns it.
        expr = Parameter("x") * Parameter("y")
        print(f"{expr=} : {expr.free_symbols}")
        ```
        """
        p: Parameter
        if isinstance(name, get_args(TNumber)):
            return sympify(name)
        elif isinstance(name, Tensor):
            if name.numel() == 1:
                return sympify(name)
            else:
                return Array(name.detach().numpy())
        elif isinstance(name, Parameter):
            p = super().__new__(cls, name.name, **assumptions)
            p.name = name.name
            p.trainable = name.trainable
            p.value = name.value
            p.is_time = name.is_time
            return p
        elif isinstance(name, (Basic, Expr)):
            if name.is_number:
                return sympify(evaluate(name))
            return name
        elif isinstance(name, str):
            p = super().__new__(cls, name, **assumptions)
            p.trainable = assumptions.get("trainable", True)
            p.value = assumptions.get("value", None)
            p.is_time = assumptions.get("is_time", False)
            if p.value is None:
                p.value = rand(1).item()
            return p
        else:
            raise TypeError(f"Parameter does not support type {type(name)}")

    def __eq__(self, other: object) -> bool:
        from qadence.utils import isclose

        if isinstance(other, str):
            return self.name == other  # type: ignore[no-any-return]

        elif isinstance(other, Parameter):
            return (
                self.name == other.name
                and self.trainable == other.trainable
                and isclose(self.value, other.value)
            )
        elif isinstance(other, Expr):
            return self in other.free_symbols
        elif isinstance(other, Symbol):
            return self.name == other.name  # type: ignore[no-any-return]

        return False

    def __hash__(self) -> Any:
        return super().__hash__()

    def _to_dict(self) -> dict:
        d = {"name": self.name, "trainable": str(self.trainable), "value": str(self.value)}
        try:
            jsonschema.validate(d, ParameterJSONSchema)
            return d
        except jsonschema.exceptions.ValidationError as e:
            logger.exception(f"Parameter dict {d} doesnt comply to {ParameterJSONSchema} with {e}.")
            return {}

    @classmethod
    def _from_dict(cls, d: dict) -> Parameter | None:
        try:
            jsonschema.validate(d, ParameterJSONSchema)
            trainable = True if d["trainable"] == "True" else False
            return cls(name=d["name"], trainable=trainable, value=float(d["value"]))
        except jsonschema.exceptions.ValidationError as e:
            logger.exception(f"Parameter dict {d} doesnt comply to {ParameterJSONSchema} with {e}.")
            return None


def FeatureParameter(name: str, **kwargs: Any) -> Parameter:
    """Shorthand for `Parameter(..., trainable=False)`."""
    return Parameter(name, trainable=False, **kwargs)


def VariationalParameter(name: str, **kwargs: Any) -> Parameter:
    """Shorthand for `Parameter(..., trainable=True)`."""
    return Parameter(name, trainable=True, **kwargs)


def TimeParameter(name: str) -> Parameter:
    """Shorthand for `Parameter(..., trainable=False, is_time=True)`."""
    return Parameter(name, trainable=False, is_time=True)


def extract_original_param_entry(
    param: Expr,
) -> TNumber | Tensor | Expr:
    """
    Given an Expression, what was the original "param" given by the user? It is either.

    going to be a numeric value, or a sympy Expression (in case a string was given,
    it was converted via Parameter("string").
    """
    return param if not param.is_number else evaluate(param)


def heaviside_func(x: Tensor, _: Any) -> Tensor:
    with no_grad():
        res = heaviside(x, tensor(0.5))
    return res


def torchify(expr: Expr) -> torchSympyModule:
    extra_funcs = {sympy.core.numbers.ImaginaryUnit: 1.0j, sympy.Heaviside: heaviside_func}
    return torchSympyModule(expressions=[sympy.N(expr)], extra_funcs=extra_funcs)


def make_differentiable(expr: Expr, engine: Engine = Engine.TORCH) -> DifferentiableExpression:
    diff_expr: DifferentiableExpression
    if engine == Engine.JAX:
        from qadence.backends.jax_utils import jaxify

        diff_expr = jaxify(expr)
    else:
        diff_expr = torchify(expr)
    return diff_expr


def sympy_to_numeric(expr: Basic) -> TNumber:
    if expr.as_real_imag()[1] != 0:
        return complex(expr)
    else:
        return float(expr)


def evaluate(expr: Expr, values: dict | None = None, as_torch: bool = False) -> TNumber | Tensor:
    """
    Arguments:

        expr: An expression consisting of Parameters.
        values: values dict which contains values for the Parameters,
            if empty, Parameter.value will be used.
        as_torch: Whether to retrieve a torch-differentiable expression result.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence.parameters import Parameter, evaluate

    expr = Parameter("x") * Parameter("y")

    # Unless specified, Parameter initialized random values
    # Lets evaluate this expression and see what the result is
    res = evaluate(expr)
    print(res)

    # We can also evaluate the expr using a custom dict
    d = {"x": 1, "y":2}
    res = evaluate(expr, d)
    print(res)

    # Lastly, if we want a differentiable result, lets put the as_torch flag
    res = evaluate(expr, d, as_torch=True)
    print(res)
    ```
    """
    res: Basic
    res_value: TNumber | Tensor
    query: dict[Parameter, TNumber | Tensor] = dict()
    values = values or dict()
    if isinstance(expr, Array):
        return Tensor(expr.tolist())
    else:
        if not expr.is_number:
            for s in expr.free_symbols:
                if s.name in values.keys():
                    query[s] = values[s.name]
                elif hasattr(s, "value"):
                    query[s] = s.value
                else:
                    raise ValueError(f"No value provided for symbol {s.name}")
        if as_torch:
            res_value = make_differentiable(expr)(**{s.name: tensor(v) for s, v in query.items()})
        else:
            res = expr.subs(query)
            res_value = sympy_to_numeric(res)
        return res_value


def stringify(expr: Basic) -> str:
    name: str = ""
    if isinstance(expr, Array):
        return str(np.array(expr.tolist())).replace(".", "_")
    else:
        if expr.is_number:
            expr_hash = hash(sympy_to_numeric(expr))
            name = "fix_" + str(expr_hash)
        else:
            name = str(expr).replace(".", "_")
        return name


class ParamMap:
    """Connects UUIDs of parameters to their expressions and names.

    This class is not user-facing
    and only needed for more complex block definitions. It provides convenient access to
    expressions/UUIDs/names needed in different backends.

    Arguments:
        kwargs: Parameters.

    Example:
    ```python exec="on" source="material-block" result="json"
    import sympy
    from qadence.parameters import ParamMap

    (x,y) = sympy.symbols("x y")
    ps = ParamMap(omega=2.0, duration=x+y)

    print(f"{ps.names() = }")
    print(f"{ps.expressions() = }")
    print(f"{ps.uuids() = }")
    ```
    """

    def __init__(self, **kwargs: str | TNumber | Tensor | Basic | Parameter):
        self._name_dict: dict[str, tuple[str, Basic]] = {}
        self._uuid_dict: dict[str, str] = {}
        for name, v in kwargs.items():
            param = v if isinstance(v, sympy.Basic) else Parameter(v)
            uuid = str(uuid4())
            self._name_dict[name] = (uuid, param)
            self._uuid_dict[uuid] = param

    def __getattr__(self, name: str) -> Basic:
        _name_dict = self.__getattribute__("_name_dict")
        if name in _name_dict:
            (_, param) = _name_dict[name]
            return param
        else:
            return self.__getattribute__(name)

    def uuid(self, name: str) -> str:
        (_uuid, _) = self._name_dict[name]
        return _uuid

    def param_str(self, name: str) -> str:
        return stringify(self.param(name))

    def uuid_param(self, name: str) -> tuple[str, Basic]:
        return self._name_dict[name]

    def names(self) -> KeysView:
        return self._name_dict.keys()

    def uuids(self) -> KeysView:
        return self._uuid_dict.keys()

    def expressions(self) -> ValuesView:
        return self._uuid_dict.values()

    def items(self) -> ItemsView:
        return self._uuid_dict.items()

    def __repr__(self) -> str:
        s = repr(self._name_dict)
        s = s.replace("{", "(")
        s = s.replace("}", ")")
        return "ParamMap" + s

    def _to_dict(self) -> dict:
        from qadence.serialization import serialize

        d = {name: (uuid, serialize(expr)) for (name, (uuid, expr)) in self._name_dict.items()}
        return {"_name_dict": d}

    @classmethod
    def _from_dict(cls, d: dict) -> ParamMap:
        from qadence.serialization import deserialize

        res = ParamMap()
        for name, (uuid, v) in d["_name_dict"].items():
            param: Parameter = deserialize(v)  # type: ignore[assignment]
            res._name_dict[name] = (uuid, param)
            res._uuid_dict[uuid] = param
        return res
