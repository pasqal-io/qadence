from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
import sympy

from qadence.blocks import (
    AddBlock,
    ChainBlock,
    KronBlock,
    ParametricBlock,
    ScaleBlock,
    add,
    block_is_qubit_hamiltonian,
    chain,
    has_duplicate_vparams,
    kron,
    put,
    tag,
)
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import (
    expressions,
    get_blocks_by_expression,
    get_pauli_blocks,
    parameters,
    primitive_blocks,
)
from qadence.constructors import (
    hea,
    ising_hamiltonian,
    single_z,
    total_magnetization,
    zz_hamiltonian,
)
from qadence.operations import CNOT, CRX, CRY, RX, RY, H, I, X, Y, Z, Zero
from qadence.parameters import Parameter, evaluate
from qadence.transpile import invert_endianness, reassign, set_trainable
from qadence.types import TNumber


def test_1qubit_blocks() -> None:
    for B in [X, Y, Z]:
        b1 = B(1)  # type: ignore [abstract]
        assert b1.n_qubits == 2  # type: ignore [attr-defined]
        assert b1.qubit_support == (1,)
        assert block_is_qubit_hamiltonian(b1)

        b2 = B(0)  # type: ignore [abstract]
        assert b2.n_qubits == 1  # type: ignore [attr-defined]
        assert b2.qubit_support == (0,)
        assert block_is_qubit_hamiltonian(b2)


def test_block_is_qubit_ham_constructors() -> None:
    n_qubits = 4

    assert block_is_qubit_hamiltonian(single_z(0))
    assert block_is_qubit_hamiltonian(total_magnetization(n_qubits))
    assert block_is_qubit_hamiltonian(zz_hamiltonian(n_qubits))
    assert block_is_qubit_hamiltonian(ising_hamiltonian(n_qubits))


def test_chain_block_only() -> None:
    block_tag = str(uuid4())
    block = chain(X(0), Z(0), Z(4), Y(4))
    tag(block, block_tag)

    assert block.qubit_support == (0, 4)
    assert block.tag == block_tag

    pbs = primitive_blocks(block)
    assert len(pbs) == 4


def test_kron_block_only() -> None:
    block_tag = str(uuid4())
    block = kron(X(0), Y(1), CNOT(2, 3))
    tag(block, block_tag)

    assert block.qubit_support == tuple(range(4))
    assert block.tag == block_tag

    pbs = primitive_blocks(block)
    assert len(pbs) == 3

    with pytest.raises(AssertionError, match="Make sure blocks act on distinct qubits!"):
        block = kron(X(0), Y(1), CNOT(1, 2))

    with pytest.raises(AssertionError, match="Make sure blocks act on distinct qubits!"):
        block = kron(X(0), Y(0), CNOT(1, 2))


@pytest.mark.parametrize(
    "block",
    [
        # FIXME: Defining it in this way will yield a nested AddBlock
        # this will break the tests and it is not exactly what we would like
        # a possible solution would be to add a "simplify()" method for
        # making sure no nested AddBlock are present
        # X(0) + X(1) * 2.0 + Y(1) * 3.0 + Z(3),
        add(X(0), X(1) * 2.0, Y(1) * 3.0, Z(3))
    ],
)
def test_add_block_only(block: AddBlock) -> None:
    block_tag = str(uuid4())
    tag(block, block_tag)

    assert block.qubit_support == (0, 1, 3)
    assert block.tag == block_tag

    b2 = block.blocks[1]
    assert evaluate(b2.parameters.parameter) == 2.0  # type: ignore [attr-defined]
    b3 = block.blocks[2]
    assert evaluate(b3.parameters.parameter) == 3.0  # type: ignore [attr-defined]


def test_composition() -> None:
    block_tag = str(uuid4())
    block = chain(
        chain(X(0), X(1)),
        chain(X(2), X(3)),
        kron(CNOT(0, 1), Y(3), Y(4)),
        chain(X(5), Y(5)),
        add(Z(1), Z(2), Z(3) * 3.0),
    )
    tag(block, block_tag)

    assert block.qubit_support == tuple(range(6))
    assert block.tag == block_tag

    pbs = primitive_blocks(block)
    assert len(pbs) == 12

    # test composition of references to blocks
    block1 = X(0)
    block2 = Y(1)
    block3 = RX(2, Parameter("theta"))
    block4 = CNOT(3, 4)

    assert isinstance(block1 + block2, AddBlock)
    assert isinstance(block1 * block2 * block3, ChainBlock)
    assert isinstance(block1 @ block2 @ block4, KronBlock)

    comp_block = chain(
        (block1 @ block2 @ block3),
        (block3 * block4),
        block1 + block2,
    )
    assert isinstance(comp_block[1], ChainBlock)
    assert isinstance(comp_block[2][0], X)  # type: ignore [index]

    all_blocks = primitive_blocks(comp_block)
    assert len(all_blocks) == 7


def test_precedence() -> None:
    assert X(0) + Z(0) * Z(1) == add(X(0), chain(Z(0), Z(1)))
    assert X(0) + Z(0) @ Z(1) == add(X(0), kron(Z(0), Z(1)))
    assert X(0) * Z(0) @ Z(1) == kron(chain(X(0), Z(0)), Z(1))

    assert 2 * Z(0) @ Z(1) == kron(2 * Z(0), Z(1))
    assert 2 * Z(0) * Z(1) == chain(2 * Z(0), Z(1))

    assert 2 * (Z(0) @ Z(1)) == 2 * kron(Z(0), Z(1))
    assert 2 * (Z(0) * Z(1)) == 2 * chain(Z(0), Z(1))


def test_reassign() -> None:
    b = X(0)
    c = reassign(b, {0: 2})
    assert c.qubit_support == (2,)

    b = chain(X(1), CNOT(2, 4))  # type: ignore [assignment]
    c = reassign(b, {1: 0, 2: 1, 4: 3})
    assert c.qubit_support == (0, 1, 3)
    assert c.blocks[0].qubit_support == (0,)  # type: ignore
    assert c.blocks[1].qubit_support == (1, 3)  # type: ignore

    b = kron(  # type: ignore [assignment]
        chain(RX(3, 2.0), CNOT(3, 5)), CNOT(6, 7), kron(X(9), Z(8))
    )
    c = reassign(b, {i: i - 3 for i in b.qubit_support})
    assert c.qubit_support == (0, 2, 3, 4, 5, 6)
    assert c.blocks[0].qubit_support == (0, 2)  # type: ignore
    assert c.blocks[0].blocks[1].qubit_support == (0, 2)  # type: ignore
    assert c.blocks[1].qubit_support == (3, 4)  # type: ignore


def test_put_block() -> None:
    b = chain(X(2), X(3), kron(X(3), X(4)))

    b = put(b, 3, 5)  # type: ignore
    assert b.qubit_support == (3, 4, 5)
    assert b.blocks[0].qubit_support == (0, 1, 2)
    assert b.n_qubits == 3

    # with pytest.raises(AssertionError, match="You are trying to put a block with 3"):
    #     b = chain(X(2), X(3), kron(X(3), X(4)))  # type: ignore
    #     b = put(b, 3, 4)  # type: ignore [assignment]


# TODO: Update to the new interface
def test_repr() -> None:
    assert X(1).__repr__() == "X(1)"
    # assert RX(2, Parameter("theta")).__repr__() == "RX(2) [params: (theta(trainable=True),)]"
    assert CNOT(3, 4).__repr__() == "CNOT(3,4)"


@pytest.mark.xfail
def test_ascii() -> None:
    # FIXME: test ascii printing
    from rich.console import Console

    console = Console()

    circ = chain(kron(X(0), X(2)), chain(X(3), X(2)))
    circ.tag = "chain chain"
    # print(circ)

    console.size = (40, 10)  # type: ignore
    # print(circ.__layout__().tree)

    assert False


def test_set_trainable() -> None:
    block1 = X(0)
    block2 = Y(1)
    theta = Parameter("theta")
    scale = Parameter("scale")
    block3 = RX(2, theta)
    block4 = CNOT(3, 4)

    comp_block = chain(
        (block1 @ block2 @ block3),
        scale * (block3 * block4),
        block1 + block2,
        CRY(2, 3, "rot_theta"),
    )

    params = parameters(comp_block)
    assert len(params) == 3
    for p in params:
        assert p.trainable

    non_trainable_b = set_trainable(comp_block, value=False)  # type: ignore [assignment]
    assert isinstance(non_trainable_b, ChainBlock)
    assert len(non_trainable_b) == 4

    for p in params:
        assert not p.trainable


@pytest.mark.parametrize(
    "parameter",
    [3 * sympy.acos(Parameter("x", trainable=False)), Parameter("x", trainable=True), 1.0, "x"],
)
def test_parameterised_gates_syntax(parameter: Parameter | TNumber | sympy.Expr | str) -> None:
    rx = RX(0, parameter)

    for p in rx.parameters.expressions():
        if p.is_number:
            assert evaluate(p) == parameter
        else:
            assert isinstance(p, (sympy.Expr, Parameter))  # type: ignore

        if isinstance(parameter, str):
            assert p.name == parameter
            assert p.value > 0  # type: ignore [operator]
            assert p.trainable

        if isinstance(parameter, Parameter):
            assert p == parameter  # type: ignore

        if isinstance(parameter, sympy.Expr):
            assert p.free_symbols == parameter.free_symbols
            assert [s.trainable for s in p.free_symbols] == [
                s.trainable for s in parameter.free_symbols
            ]


def test_tag_blocks() -> None:
    block1 = X(0)
    block2 = Y(1)
    block3 = RX(2, Parameter("theta"))
    block4 = CNOT(3, 4)

    comp_block = chain(
        tag(block1 @ block2 @ block3, "Feature Map"),
        tag(block3 * block4, "Variational Ansatz"),
        block1 + block2,
    )
    tags = [block.tag for block in comp_block.blocks]
    assert "Feature Map" in tags


def test_reverse() -> None:
    block1 = X(0)
    block2 = Y(1)
    block3 = RX(2, Parameter("theta"))
    block4 = CNOT(3, 4)

    inv_block1 = X(4)
    inv_block2 = Y(3)
    inv_block3 = RX(2, Parameter("theta"))
    inv_block4 = CNOT(1, 0)

    comp_block = chain(block1, block2, block3, block4)
    inverted = invert_endianness(comp_block, 5, False)
    inv_block = chain(inv_block1, inv_block2, inv_block3, inv_block4)
    for b1, b2 in zip(inverted.blocks, inv_block.blocks):  # type: ignore [attr-defined]
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support


@pytest.mark.parametrize(
    "in_place",
    [False, pytest.param(True, marks=pytest.mark.xfail(reason="Treacherous syntax needs fixing"))],
)
def test_duplicate_manipulation(in_place: bool) -> None:
    # if done "in place" (without deepcopying each object)
    # it fails due two block1 appearing twice and the inversion
    # will happen twice
    block1 = X(0)
    block2 = Y(1)
    block4 = CNOT(3, 4)
    inv_block1 = X(4)
    inv_block2 = Y(3)
    inv_block4 = CNOT(1, 0)

    comp_block = chain(block1, block2, block1, block4)
    inverted = invert_endianness(comp_block, n_qubits=5, in_place=in_place)

    inv_block = chain(inv_block1, inv_block2, inv_block1, inv_block4)
    for b1, b2 in zip(inverted.blocks, inv_block.blocks):  # type: ignore [attr-defined]
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support


def test_control_gate_manipulation() -> None:
    cry = CRY(0, 1, "theta")
    inv_block = invert_endianness(cry)
    cry_double_inverted = invert_endianness(inv_block)
    assert cry_double_inverted.qubit_support == cry.qubit_support


def test_reassign_cnotchain() -> None:
    myqubitmap = {1: 0, 0: 1, 3: 2, 2: 3}
    orig_cnotchain = chain(CNOT(0, 1), CNOT(2, 3))
    target_cnotchain = chain(CNOT(1, 0), CNOT(3, 2))
    new_cnotchain = reassign(orig_cnotchain, myqubitmap)
    for b1, b2 in zip(target_cnotchain.blocks, new_cnotchain.blocks):  # type: ignore
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support


def test_reassign_parametrized_controlgate_chain() -> None:
    myqubitmap = {1: 0, 0: 1, 3: 2, 2: 3}
    orig_chain = chain(CRY(0, 1, "theta_0"), CRX(2, 3, "theta_1"))
    target_chain = chain(CRY(1, 0, "theta_0"), CRX(3, 2, "theta_1"))
    new_chain = reassign(orig_chain, myqubitmap)
    for b1, b2 in zip(target_chain.blocks, new_chain.blocks):  # type: ignore
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support
        assert b1.parameters.parameter.name == b2.parameters.parameter.name  # type: ignore [attr-defined] # noqa: E501
        assert b1.blocks[0].qubit_support[0] == b2.blocks[0].qubit_support[0]  # type: ignore [attr-defined] # noqa: E501


def test_reassign_identity() -> None:
    identitymap = {0: 0, 1: 1, 2: 2, 3: 3}
    orig_cnotchain = chain(CNOT(0, 1), CNOT(2, 3))
    new_cnotchain = reassign(orig_cnotchain, identitymap)
    for b1, b2 in zip(orig_cnotchain.blocks, new_cnotchain.blocks):  # type: ignore
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support


def test_lookup_block_by_param() -> None:
    x = Parameter("x", trainable=False)
    block1 = RY(0, 3 * x)
    block2 = RX(1, "theta1")
    block3 = RX(2, "theta2")
    block4 = RX(3, "theta3")
    block5 = RY(0, np.pi)
    block6 = RX(1, np.pi)
    block7 = CNOT(2, 3)

    comp_block = chain(
        *[
            kron(*[X(0), X(1), Z(2), Z(3)]),
            kron(*[block1, block2, block3, block4]),
            kron(*[block5, block6, block7]),
        ]
    )

    exprs = expressions(comp_block)
    assert exprs
    for expr in exprs:
        bs = get_blocks_by_expression(comp_block, expr)
        for b in bs:
            assert isinstance(b, ParametricBlock)


def test_addition_multiplication() -> None:
    b = X(0) * X(1)
    assert isinstance(b, ChainBlock)

    b = X(0) * 2.0
    assert evaluate(b.parameters.parameter) == 2.0  # type: ignore [attr-defined]

    b = 2.0 * X(0)
    assert evaluate(b.parameters.parameter) == 2.0  # type: ignore [attr-defined]

    b = 2.0 * (2.0 * X(0))
    assert isinstance(b.block, X)  # type: ignore[attr-defined]
    assert evaluate(b.parameters.parameter) == 4.0  # type: ignore [attr-defined]

    phi = Parameter("phi")
    b = 2 * (phi * X(0))
    assert b.parameters.parameter == 2 * phi  # type: ignore[attr-defined]

    b = X(0) + X(1)
    assert isinstance(b, AddBlock)

    b = X(0) - 2.3 * X(1)
    assert isinstance(b, AddBlock)
    assert evaluate(b.blocks[1].parameters.parameter) == -2.3  # type: ignore[attr-defined]

    b = -X(1)
    assert isinstance(b, ScaleBlock)
    assert evaluate(b.parameters.parameter) == -1.0

    b = +X(1)
    assert isinstance(b, X)

    b = (I(0) - Z(0)) / 2
    assert isinstance(b, ScaleBlock)
    assert evaluate(b.parameters.parameter) == 0.5

    with pytest.raises(TypeError, match="Can only add a block to another block."):
        X(0) + 1.0  # type: ignore [operator]

    with pytest.raises(TypeError, match="Cannot divide block by another block."):
        X(0) / X(1)

    b = X(0) @ X(1)
    assert isinstance(b, KronBlock)

    block = X(0) ^ 3
    assert isinstance(block, KronBlock)
    assert all(isinstance(b, X) for b in block.blocks)

    block = RX(2, "phi") ^ 2
    assert isinstance(block, KronBlock)
    assert all(isinstance(b, RX) for b in block.blocks)
    assert block.qubit_support == (2, 3)


def test_inplace_operations() -> None:
    a = Zero()
    a += add(X(0), Y(1), Z(2))  # type: ignore[misc]
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 3

    a = add(X(0), Y(1))
    a += Z(2)
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 3

    a = X(0)
    a += add(Y(1), Z(2))
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 3

    a = X(0)
    a += Y(1)
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 2

    a = Zero()
    a -= add(X(0), Y(1), Z(2))
    assert isinstance(a, ScaleBlock)
    assert evaluate(a.parameters.parameter) == -1.0

    a = add(X(0), Y(1))
    a -= Z(2)
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 3

    a = X(0)
    a -= add(Y(1), Z(2))
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 3

    a = X(0)
    a -= Y(1)
    assert isinstance(a, AddBlock)
    assert len(a.blocks) == 2

    with pytest.raises(AssertionError, match="Make sure blocks act on distinct qubits!"):
        a = I(0)
        a @= X(0)

    a = kron(X(0), Y(1))
    a @= Z(2)
    assert isinstance(a, KronBlock)
    assert len(a.blocks) == 3

    a = X(0)
    a @= kron(X(1), X(2))
    assert isinstance(a, KronBlock)
    assert len(a.blocks) == 3

    a = I(0)
    a *= X(0)
    assert isinstance(a, X)

    a = sum(X(j) for j in range(3))
    assert isinstance(a, AddBlock)

    a = X(0)
    a /= 4
    assert isinstance(a, ScaleBlock)
    assert evaluate(a.parameters.parameter) == 1 / 4

    a = Zero()
    a **= 3
    assert isinstance(a, Zero)

    a = X(0)
    a **= 3
    assert isinstance(a, ChainBlock)
    assert len(a.blocks) == 3
    assert all(isinstance(block, X) for block in a.blocks)

    a = 2 * X(0)
    a **= 3
    assert isinstance(a, ScaleBlock)
    assert evaluate(a.parameters.parameter) == 8
    assert isinstance(a.block, ChainBlock)
    assert len(a.block.blocks) == 3


def test_duplicate_parameters() -> None:
    n_qubits = 4
    depth = 2

    hea1 = hea(n_qubits=n_qubits, depth=depth)
    hea2 = hea(n_qubits=n_qubits, depth=depth)

    block1 = chain(hea1, hea2)
    assert has_duplicate_vparams(block1)

    hea1 = hea(n_qubits=n_qubits, depth=depth, param_prefix="0")
    hea2 = hea(n_qubits=n_qubits, depth=depth, param_prefix="1")

    block2 = chain(hea1, hea2)
    assert not has_duplicate_vparams(block2)


def test_pauli_blocks() -> None:
    b1 = 0.1 * kron(X(0), X(1)) + 0.2 * kron(Z(0), Z(1)) + 0.3 * kron(Y(2), Y(3))
    b2 = chain(Z(0) * Z(1), CNOT(0, 1)) + CNOT(2, 3)

    paulis = get_pauli_blocks(b1)
    primitives = primitive_blocks(b1)
    assert len(paulis) == len(primitives)

    paulis = get_pauli_blocks(b2)
    primitives = primitive_blocks(b2)
    assert len(paulis) != len(primitives)


def test_block_from_dict_primitive() -> None:
    # Primitive
    myx = X(0)
    block_dict = myx._to_dict()
    myx_copy = X._from_dict(block_dict)
    assert myx == myx_copy


def test_block_from_dict_parametric() -> None:
    # Parametric
    myrx = RX(0, "theta")
    block_dict = myrx._to_dict()
    myrx_copy = RX._from_dict(block_dict)
    assert myrx == myrx_copy


def test_block_from_dict_chain() -> None:
    # Composite
    from qadence.blocks import ChainBlock

    mychain = chain(RX(0, "theta"), RY(1, "epsilon"))
    block_dict = mychain._to_dict()
    mychain_copy = ChainBlock._from_dict(block_dict)
    assert mychain == mychain_copy


@pytest.mark.parametrize("n_qubits", [2, 4, 6, 8])
def test_block_from_dict_hea_qubits(n_qubits: int) -> None:
    # hea
    from qadence.blocks import ChainBlock

    depth = 2
    myhea = hea(n_qubits, depth)
    block_dict = myhea._to_dict()
    myhea_copy = ChainBlock._from_dict(block_dict)
    assert myhea == myhea_copy


@pytest.mark.parametrize("depth", [2, 4, 6, 8])
def test_block_from_dict_hea_depth(depth: int) -> None:
    # hea
    from qadence.blocks import ChainBlock

    n_qubits = 4
    myhea = hea(n_qubits, depth)
    block_dict = myhea._to_dict()
    myhea_copy = ChainBlock._from_dict(block_dict)
    assert myhea == myhea_copy


def test_comp_contains_operator() -> None:
    ry = RY(1, "epsilon")
    mychain = chain(RX(0, "theta"), ry)
    assert ry in mychain


def test_eq_kron_order() -> None:
    block0 = kron(Z(0), Z(1))
    block1 = kron(Z(1), Z(0))
    assert block0 == block1


def test_eq_scale_kron() -> None:
    block0 = 0.9 * kron(Z(0), Z(1))
    block1 = 0.9 * kron(Z(1), Z(0))
    assert block0 == block1


def test_eq_scale_add_kron() -> None:
    block0 = kron(Z(0), Z(1)) + kron(X(0), Y(1))
    block1 = kron(Z(1), Z(0)) + kron(Y(1), X(0))
    assert block0 == block1


def test_parametric_scale_eq() -> None:
    from qadence.parameters import VariationalParameter

    p1 = VariationalParameter("p1")
    p2 = VariationalParameter("p2")
    rx0 = RX(1, "a")
    rx1 = RY(2, "b")
    b0 = (
        p1 * kron(Z(0), Z(1), X(2), X(3))
        + p2 * chain(kron(X(0), X(3)), kron(rx0, rx1))
        + 0.5 * X(0)
    )
    b1 = (
        p1 * kron(Z(0), Z(1), X(2), X(3))
        + p2 * chain(kron(X(0), X(3)), kron(rx0, rx1))
        + 0.5 * X(0)
    )

    assert b0 == b1


def test_kron_eq() -> None:
    block1 = kron(X(0), Z(1), Y(2))
    block2 = kron(X(1), Z(2), Y(3))
    block3 = kron(X(0), Z(2), Y(3))

    assert not block1 == block2 and not block1 == block3


def test_kron_chain_eq() -> None:
    assert kron(X(0), X(1)) != chain(X(0), X(1))
    assert kron(Z(0), Z(1)) != chain(Z(0), Z(1))


@pytest.mark.parametrize(
    "block",
    [
        chain(I(n) for n in range(5)),
        kron(I(n) for n in range(5)),
    ],
)
def test_identity_predicate(block: AbstractBlock) -> None:
    assert block.is_identity


def test_composite_containment() -> None:
    kron_block = kron(X(0), Y(1), Z(2))
    assert X(0) in kron_block
    assert Z in kron_block
    add_block = add(X(0), Y(1), Z(2))
    assert X(0) in add_block
    assert Z in add_block
    chain_block = chain(X(0), Y(0), Z(0))
    assert X(0) in chain_block
    assert Z in chain_block
    # Test case for nested blocks.
    nested_block = add(kron(X(0), Y(1)) + kron(Z(0), H(1)))
    assert X(0) in nested_block
    assert Z in nested_block
