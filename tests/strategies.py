from __future__ import annotations

import random
import string
from functools import reduce
from typing import Any, Callable, Set

import hypothesis.strategies as st
from hypothesis.strategies._internal import SearchStrategy
from sympy import Basic, Expr, acos, asin, atan, cos, sin, tan
from torch import Tensor

from qadence.blocks import (
    AbstractBlock,
    ParametricBlock,
    add,
    chain,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.extensions import supported_gates
from qadence.ml_tools.utils import rand_featureparameters
from qadence.operations import (
    analog_gateset,
    multi_qubit_gateset,
    non_unitary_gateset,
    pauli_gateset,
    single_qubit_gateset,
    three_qubit_gateset,
    two_qubit_gateset,
)
from qadence.parameters import FeatureParameter, Parameter, VariationalParameter
from qadence.types import PI, BackendName, ParameterType, TNumber

PARAM_NAME_LENGTH = 1
MIN_SYMBOLS = 1
MAX_SYMBOLS = 3
FEAT_PARAM_MIN = -1.0
FEAT_PARAM_MAX = 1.0

VAR_PARAM_MIN = -2 * PI
VAR_PARAM_MAX = 2 * PI

TRIG_FNS = [cos, sin, tan, acos, asin, atan]

PARAM_RANGES = {
    "Feature": (FEAT_PARAM_MIN, FEAT_PARAM_MAX),
    "Variational": (VAR_PARAM_MIN, VAR_PARAM_MAX),
    "Fixed": (VAR_PARAM_MIN, VAR_PARAM_MAX),
}

OPS_DICT = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}

supported_gates_map: dict = {k: supported_gates(k) for k in BackendName.list()}
supported_gates_list: list[Set] = [
    set(supported_gates(name)) for name in BackendName.list() if name != BackendName.PULSER
]

full_gateset = list(
    reduce(lambda fs, s: fs.union(s), supported_gates_list)  # type: ignore[attr-defined]
)
minimal_gateset = list(
    reduce(lambda fs, s: fs.intersection(s), supported_gates_list)  # type: ignore[attr-defined]
)
digital_gateset = list(set(full_gateset) - set(analog_gateset) - set(non_unitary_gateset))

MIN_N_QUBITS = 1
MAX_N_QUBITS = 4
MIN_CIRCUIT_DEPTH = 1
MAX_CIRCUIT_DEPTH = 4
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 4

N_QUBITS_STRATEGY: SearchStrategy[int] = st.integers(min_value=MIN_N_QUBITS, max_value=MAX_N_QUBITS)
CIRCUIT_DEPTH_STRATEGY: SearchStrategy[int] = st.integers(
    min_value=MIN_CIRCUIT_DEPTH, max_value=MAX_CIRCUIT_DEPTH
)
BATCH_SIZE_STRATEGY: SearchStrategy[int] = st.integers(
    min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE
)


def get_param(
    draw: Callable[[SearchStrategy[Any]], Any],
    param_type: ParameterType,
    name_len: int,
    value: TNumber,
) -> Basic:
    def rand_name(length: int) -> str:
        letters = string.ascii_letters
        result_str = "".join(random.choice(letters) for i in range(length))
        return result_str

    p: Basic
    if param_type == ParameterType.FEATURE:
        p = FeatureParameter(rand_name(name_len), value=value)
        with_trig: SearchStrategy[bool] = st.booleans()
        if draw(with_trig):
            p = draw(st.sampled_from(TRIG_FNS))(p)
    elif param_type == ParameterType.VARIATIONAL:
        p = VariationalParameter(rand_name(name_len), value=value)
    else:
        p = Parameter(value)
    return p


# A strategy to generate random parameters.
def rand_parameter(draw: Callable[[SearchStrategy[Any]], Any]) -> Basic:
    param_type = draw(st.sampled_from([p for p in ParameterType]))
    min_v, max_v = PARAM_RANGES[param_type]
    value = draw(st.floats(min_value=min_v, max_value=max_v))
    name_len = draw(st.integers(min_value=1, max_value=PARAM_NAME_LENGTH))
    return get_param(draw, param_type=param_type, name_len=name_len, value=value)


# A strategy to generate random expressions.
def rand_expression(draw: Callable[[SearchStrategy[Any]], Any]) -> Expr:
    n_symbols: SearchStrategy[int] = st.integers(min_value=MIN_SYMBOLS, max_value=MAX_SYMBOLS)
    N = draw(n_symbols)
    expr = rand_parameter(draw)
    if N > 1:
        for _ in range(N - 1):
            other = rand_parameter(draw)
            op = draw(st.sampled_from([op for op in OPS_DICT.keys()]))
            expr = OPS_DICT[op](expr, other)
    return expr


# A strategy to generate random blocks.
def rand_digital_blocks(gate_list: list[AbstractBlock]) -> Callable:
    @st.composite
    def blocks(
        # ops_pool: list[AbstractBlock] TO BE ADDED
        draw: Callable[[SearchStrategy[Any]], Any],
        n_qubits: SearchStrategy[int] = st.integers(min_value=1, max_value=4),
        depth: SearchStrategy[int] = st.integers(min_value=1, max_value=8),
    ) -> AbstractBlock:
        total_qubits = draw(n_qubits)
        gates_list: list = []
        qubit_indices = {0}

        pool_1q = [gate for gate in single_qubit_gateset if gate in gate_list]
        pool_1q_fixed = [gate for gate in pool_1q if not issubclass(gate, ParametricBlock)]
        pool_1q_param = list(set(pool_1q) - set(pool_1q_fixed))
        pool_2q = [gate for gate in two_qubit_gateset if gate in gate_list]
        pool_2q_fixed = [
            gate for gate in two_qubit_gateset if not issubclass(gate, ParametricBlock)
        ]
        pool_2q_param = list(set(pool_2q) - set(pool_2q_fixed))
        pool_3q = [gate for gate in three_qubit_gateset if gate in gate_list]
        pool_nq = [gate for gate in multi_qubit_gateset if gate in gate_list]
        pool_nq_fixed = [
            gate for gate in multi_qubit_gateset if not issubclass(gate, ParametricBlock)
        ]
        pool_nq_param = list(set(pool_nq) - set(pool_nq_fixed))

        for _ in range(draw(depth)):
            if total_qubits == 1:
                gate = draw(st.sampled_from(pool_1q))
            elif total_qubits >= 2:
                gate = draw(st.sampled_from(gate_list))

            qubit = draw(st.integers(min_value=0, max_value=total_qubits - 1))
            qubit_indices = qubit_indices.union({qubit})

            if gate in pool_1q:
                if gate in pool_1q_fixed:
                    gates_list.append(gate(qubit))
                elif gate in pool_1q_param:
                    angles = [rand_expression(draw) for _ in range(gate.num_parameters())]
                    gates_list.append(gate(qubit, *angles))

            elif gate in pool_2q:
                target = draw(
                    st.integers(min_value=0, max_value=total_qubits - 1).filter(
                        lambda x: x != qubit
                    )
                )
                qubit_indices = qubit_indices.union({target})
                if gate in pool_2q_fixed:
                    gates_list.append(gate(qubit, target))
                elif gate in pool_2q_param:
                    gates_list.append(gate(qubit, target, rand_expression(draw)))

            elif gate in pool_3q:
                target1 = draw(
                    st.integers(min_value=0, max_value=total_qubits - 1).filter(
                        lambda x: x != qubit
                    )
                )
                target2 = draw(
                    st.integers(min_value=0, max_value=total_qubits - 1).filter(
                        lambda x: x != qubit and x != target1
                    )
                )
                gates_list.append(gate(qubit, target1, target2))

            elif gate in pool_nq:
                target1 = draw(
                    st.integers(min_value=0, max_value=total_qubits - 1).filter(
                        lambda x: x != qubit
                    )
                )
                target2 = draw(
                    st.integers(min_value=0, max_value=total_qubits - 1).filter(
                        lambda x: x != qubit and x != target1
                    )
                )
                if gate in pool_nq_fixed:
                    gates_list.append(gate((qubit, target1), target2))
                elif gate in pool_nq_param:
                    gates_list.append(gate((qubit, target1), target2, rand_expression(draw)))
        return chain(*gates_list)

    return blocks  # type: ignore[no-any-return]


@st.composite
def digital_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
) -> QuantumCircuit:
    block = draw(rand_digital_blocks(digital_gateset)(n_qubits, depth))
    total_qubits = max(block.qubit_support) + 1
    return QuantumCircuit(total_qubits, block)


@st.composite
def restricted_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
) -> QuantumCircuit:
    block = draw(rand_digital_blocks(minimal_gateset)(n_qubits, depth))
    total_qubits = max(block.qubit_support) + 1
    return QuantumCircuit(total_qubits, block)


# A strategy to generate both a circuit and a batch of values for each FeatureParameter.
@st.composite
def batched_digital_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
    batch_size: SearchStrategy[int] = BATCH_SIZE_STRATEGY,
) -> tuple[QuantumCircuit, dict[str, Tensor]]:
    circuit = draw(digital_circuits(n_qubits, depth))
    b_size = draw(batch_size)
    inputs = rand_featureparameters(circuit, b_size)
    return circuit, inputs


@st.composite
def restricted_batched_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
    batch_size: SearchStrategy[int] = BATCH_SIZE_STRATEGY,
) -> tuple[QuantumCircuit, dict[str, Tensor]]:
    circuit = draw(restricted_circuits(n_qubits, depth))
    b_size = draw(batch_size)
    inputs = rand_featureparameters(circuit, b_size)
    return circuit, inputs


# A strategy to generate random observables under the form
# of an add block of numerically scaled kron blocks.
@st.composite
def observables(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
) -> AbstractBlock:
    total_qubits = draw(n_qubits)
    add_layer = []
    qubit_indices = {0}
    for _ in range(draw(depth)):
        kron_layer = []
        for qubit in range(draw(st.integers(min_value=1, max_value=total_qubits))):
            gate = draw(st.sampled_from(pauli_gateset))
            kron_layer.append(gate(qubit))
        scale = draw(st.floats(min_value=-10.0, max_value=10.0))
        kron_block = scale * kron(*kron_layer)
        add_layer.append(kron_block)
    scale_add: float = draw(st.floats(min_value=-10.0, max_value=10.0))
    add_block = scale_add * add(*add_layer)
    return add_block
