from __future__ import annotations

import numpy as np
import pytest

# from qucint.parameters import Parameter
from sympy import I, Matrix, Mul, Symbol, cos, sin, sqrt
from sympy.matrices import matrix2numpy
from sympytorch.sympy_module import torch

from qadence import Parameter as QParameter
from qadence import QuantumCircuit
from qadence.backends.api import backend_factory
from qadence.blocks.sympy_block import (
    RX,
    AddBlock,
    ChainBlock,
    Id,
    KronBlock,
    Parameter,
    X,
    Y,
    Z,
    chain,
    evaluate,
    kron,
)
from qadence.operations import RX as QRX
from qadence.operations import X as QX
from qadence.operations import Y as QY
from qadence.operations import Z as QZ

TO_QUCINT_OPS = {"X": QX, "Y": QY, "Z": QZ, "RX": QRX}


def test_pauli_operators_instantiations_and_basic_parametric_operations() -> None:
    x_gate = X(0)
    theta = Parameter("theta")
    assert isinstance(x_gate, X)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-1, 1)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == ()
    assert x_gate.matrix == Matrix([[0, 1], [1, 0]])
    # Scaling the X-gate.
    x_gate = 2.0 * X(0)
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-2.0, 2.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == ()
    assert x_gate.matrix == Matrix([[0, 2.0], [2.0, 0]])
    x_gate = X(0) * 2.0
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-2.0, 2.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == ()
    assert x_gate.matrix == Matrix([[0, 2.0], [2.0, 0]])
    # Parametrise the X-gate.
    x_gate = theta * X(0)
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-theta, theta)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[0, theta], [theta, 0]])
    x_gate = X(0) * theta
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-theta, theta)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[0, theta], [theta, 0]])
    # Parametrise the X-gate (2).
    x_gate = 2.0 * theta * X(0)
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-2.0 * theta, 2.0 * theta)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    matrix_entry = 2.0 * theta  # Chainblock.
    # Sympy returns matrices with its own custom types.
    assert x_gate.matrix == Matrix([[0, Mul(*matrix_entry.args)], [Mul(*matrix_entry.args), 0]])
    x_gate = X(0) * 2.0 * theta
    assert isinstance(x_gate, ChainBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (-2.0 * theta, 2.0 * theta)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    matrix_entry = 2.0 * theta  # Chainblock.
    # Sympy returns matrices with its own custom types.
    assert x_gate.matrix == Matrix([[0, Mul(*matrix_entry.args)], [Mul(*matrix_entry.args), 0]])
    x_gate = 2.0 + X(0)
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (1.0, 3.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == ()
    assert x_gate.matrix == Matrix([[2.0, 1], [1, 2.0]])
    x_gate = X(0) + 2.0
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (1.0, 3.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == ()
    assert x_gate.matrix == Matrix([[2.0, 1], [1, 2.0]])
    x_gate = theta + X(0)
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (theta + 1, theta - 1)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[theta, 1], [1, theta]])
    x_gate = X(0) + theta
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (theta + 1, theta - 1)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[theta, 1], [1, theta]])
    x_gate = 2.0 * theta + X(0)
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (2.0 * theta + 1.0, 2.0 * theta - 1.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[Mul(*matrix_entry.args), 1], [1, Mul(*matrix_entry.args)]])
    x_gate = X(0) + 2.0 * theta
    assert isinstance(x_gate, AddBlock)
    assert x_gate.qubit_support == (0,)
    assert x_gate.eigenvalues == (2.0 * theta + 1.0, 2.0 * theta - 1.0)
    assert x_gate.n_qubits == 1
    assert x_gate.parameters == (theta,)
    assert x_gate.matrix == Matrix([[Mul(*matrix_entry.args), 1], [1, Mul(*matrix_entry.args)]])
    y_gate = Y(0)
    assert isinstance(y_gate, Y)
    assert y_gate.qubit_support == (0,)
    assert y_gate.eigenvalues == (-1, 1)
    assert y_gate.n_qubits == 1
    assert y_gate.parameters == ()
    assert y_gate.matrix == Matrix([[0, -I], [I, 0]])
    z_gate = Z(0)
    assert isinstance(z_gate, Z)
    assert z_gate.qubit_support == (0,)
    assert z_gate.eigenvalues == (-1, 1)
    assert z_gate.n_qubits == 1
    assert z_gate.parameters == ()
    assert z_gate.matrix == Matrix([[1, 0], [0, -1]])
    id_gate = Id(0)
    assert isinstance(id_gate, Id)
    assert id_gate.qubit_support == (0,)
    assert id_gate.eigenvalues == (1, 1)
    assert id_gate.n_qubits == 1
    assert id_gate.parameters == ()
    assert id_gate.matrix == Matrix([[1, 0], [0, 1]])


def test_rx_instantiation() -> None:
    rx_gate = RX(0, 2.0)
    assert rx_gate.qubit_support == (0,)
    assert rx_gate.n_qubits == 1
    assert rx_gate.parameters == (2.0,)
    x = Symbol("x")
    assert np.allclose(
        matrix2numpy(rx_gate.matrix.evalf(), dtype=np.complex128),
        matrix2numpy(
            Matrix([[cos(2.0 / 2), -I * sin(2.0 / 2)], [-I * sin(2.0 / 2), cos(2.0 / 2)]]).evalf(),
            dtype=np.complex128,
        ),
    )
    assert rx_gate.eigenvalues == (
        -sqrt((cos(2.0 / 2) - 1) * (cos(2.0 / 2) + 1)) + cos(2.0 / 2),
        sqrt((cos(2.0 / 2) - 1) * (cos(2.0 / 2) + 1)) + cos(2.0 / 2),
    )
    assert rx_gate.generator == X(0)
    assert rx_gate.generator.eigenvalues == (-1, 1)
    theta = Parameter("theta")
    rx_gate = RX(0, theta)
    assert rx_gate.qubit_support == (0,)
    assert rx_gate.n_qubits == 1
    assert rx_gate.parameters == (theta,)
    assert rx_gate.matrix == Matrix(
        [[cos(theta / 2), -I * sin(theta / 2)], [-I * sin(theta / 2), cos(theta / 2)]]
    )
    assert rx_gate.eigenvalues == (
        -sqrt((cos(theta / 2) - 1) * (cos(theta / 2) + 1)) + cos(theta / 2),
        sqrt((cos(theta / 2) - 1) * (cos(theta / 2) + 1)) + cos(theta / 2),
    )
    assert rx_gate.generator == X(0)
    assert rx_gate.generator.eigenvalues == (-1, 1)
    rx_gate = RX(0, 2.0 * theta)
    assert rx_gate.qubit_support == (0,)
    assert rx_gate.n_qubits == 1
    assert rx_gate.parameters == (theta,)
    assert rx_gate.matrix == Matrix(
        [[1.0 * cos(theta), -1.0 * I * sin(theta)], [-1.0 * I * sin(theta), 1.0 * cos(theta)]]
    )
    assert rx_gate.eigenvalues == (
        -1.0 * sqrt((cos(theta) - 1) * (cos(theta) + 1)) + 1.0 * cos(theta),
        1.0 * sqrt((cos(theta) - 1) * (cos(theta) + 1)) + 1.0 * cos(theta),
    )
    assert rx_gate.generator == X(0)
    assert rx_gate.generator.eigenvalues == (-1, 1)


def test_matrix_evaluation() -> None:
    # TODO: Add test for batched parameters.
    theta = Parameter("theta")
    values = {"theta": 0.5}
    rx_gate = RX(0, theta)
    mat = evaluate(rx_gate.matrix, values=values)
    # breakpoint()
    exp_mat = torch.tensor(
        [[0.9689 + 0.0000j, 0.0000 - 0.2474j], [0.0000 - 0.2474j, 0.9689 + 0.0000j]],
        dtype=torch.complex128,
    )
    assert torch.allclose(mat, exp_mat, atol=1.0e-5)


def test_eigenvalues_evaluation() -> None:
    # TODO: Add test for batched parameters.
    theta = Parameter("theta")
    values = {"theta": 0.5}
    rx_gate = RX(0, theta)
    eigenvals = evaluate(rx_gate.eigenvalues, values=values)
    # breakpoint()
    exp_eigenvals = torch.tensor([0.9689 - 0.2474j, 0.9689 + 0.2474j], dtype=torch.complex128)
    assert torch.allclose(eigenvals, exp_eigenvals, atol=1.0e-5)


@pytest.mark.skip
def test_parametrisation() -> None:
    w = Parameter("w")
    expr = w * X(0) + w * X(0)
    breakpoint()
    assert expr == 2 * w * X(0)
    breakpoint()
    assert expr.matrix.equals(
        Matrix(
            [[0, 2 * w], [2 * w, 0]],
        )
    )
    assert expr.parameters == (2, w)
    expr = w * (X(0) + X(0))
    assert expr == 2 * w * X(0)
    expr = (w + w) * X(0)
    assert expr == 2 * w * X(0)
    expr = (w * w) * X(0)
    assert expr == w**2 * X(0)
    expr = w * X(0) + w * X(1)
    assert expr == w * (X(0) + X(1))


def test_addblock() -> None:
    x_gate = X(0)
    expr = 2.0 + x_gate
    # breakpoint()
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0,)
    assert expr.eigenvalues == (1.0, 3.0)
    assert expr.n_qubits == 1
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[2.0, 1], [1, 2.0]])
    expr = x_gate + x_gate
    assert expr == 2 * X(0)
    assert isinstance(expr, ChainBlock)
    assert expr.qubit_support == (0,)
    assert expr.eigenvalues == (-2, 2)
    assert expr.n_qubits == 1
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[0, 2], [2, 0]])
    y_gate = Y(1)
    expr = x_gate + y_gate
    assert expr == X(0) + Y(1)
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1)
    assert expr.eigenvalues == (-2, 2, 0)
    assert expr.n_qubits == 2
    assert expr.parameters == ()
    # breakpoint()
    assert expr.matrix == Matrix([[0, -I, 1, 0], [I, 0, 0, 1], [1, 0, 0, -I], [0, 1, I, 0]])
    z_gate = Z(2)
    expr = x_gate + y_gate + z_gate
    # breakpoint()
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1, 2)
    assert expr.eigenvalues == (3, -1, 1, -3)
    assert expr.n_qubits == 3
    assert expr.parameters == ()
    # breakpoint()
    assert expr.matrix == Matrix(
        [
            [1, 0, -I, 0, 1, 0, 0, 0],
            [0, -1, 0, -I, 0, 1, 0, 0],
            [I, 0, 1, 0, 0, 0, 1, 0],
            [0, I, 0, -1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, -I, 0],
            [0, 1, 0, 0, 0, -1, 0, -I],
            [0, 0, 1, 0, I, 0, 1, 0],
            [0, 0, 0, 1, 0, I, 0, -1],
        ]
    )
    theta = Parameter("theta")
    rx_gate = RX(1, theta)
    expr = x_gate + rx_gate
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1)
    # Should check the eigenvalues although
    # sympy returns PieceWise functions.
    # Symbolic solutions do exist.
    # Maybe numerical checks at random points using sympy.equals ?
    assert expr.n_qubits == 2
    assert expr.parameters == (theta,)
    # breakpoint()
    assert expr.matrix == Matrix(
        [
            [cos(theta / 2), -I * sin(theta / 2), 1, 0],
            [-I * sin(theta / 2), cos(theta / 2), 0, 1],
            [1, 0, cos(theta / 2), -I * sin(theta / 2)],
            [0, 1, -I * sin(theta / 2), cos(theta / 2)],
        ]
    )
    expr = 1.5 * x_gate + y_gate
    # expr.matrix
    # breakpoint()
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1)
    assert expr.eigenvalues == (
        -2.50000000000000,
        -0.500000000000000,
        0.500000000000000,
        2.50000000000000,
    )
    assert expr.n_qubits == 2
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[0, -I, 1.5, 0], [I, 0, 0, 1.5], [1.5, 0, 0, -I], [0, 1.5, I, 0]])
    expr = 1.5 * x_gate + 2.0 * y_gate
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1)
    # breakpoint()
    assert expr.eigenvalues == (
        -3.50000000000000,
        -0.500000000000000,
        0.500000000000000,
        3.50000000000000,
    )
    assert expr.n_qubits == 2
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, -2.0 * I, 1.5, 0],
            [2.0 * I, 0, 0, 1.5],
            [1.5, 0, 0, -2.0 * I],
            [0, 1.5, 2.0 * I, 0],
        ]
    )
    expr = kron(x_gate, y_gate) + z_gate
    # breakpoint()
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1, 2)
    assert expr.eigenvalues == (2, 0, -2)
    assert expr.n_qubits == 3
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [1, 0, 0, 0, 0, 0, -I, 0],
            [0, -1, 0, 0, 0, 0, 0, -I],
            [0, 0, 1, 0, I, 0, 0, 0],
            [0, 0, 0, -1, 0, I, 0, 0],
            [0, 0, -I, 0, 1, 0, 0, 0],
            [0, 0, 0, -I, 0, -1, 0, 0],
            [I, 0, 0, 0, 0, 0, 1, 0],
            [0, I, 0, 0, 0, 0, 0, -1],
        ]
    )
    # # expr.n_qubits
    # # expr.n_qubits
    # # breakpoint()
    expr = 1.5 * x_gate + theta * kron(y_gate, z_gate)
    assert isinstance(expr, AddBlock)
    # breakpoint()
    assert expr.qubit_support == (0, 1, 2)
    assert expr.eigenvalues == (
        1.0 * theta + 1.5,
        1.0 * theta - 1.5,
        1.5 - 1.0 * theta,
        -1.0 * theta - 1.5,
    )
    assert expr.n_qubits == 3
    assert expr.parameters == (theta,)
    assert expr.matrix == Matrix(
        [
            [0, 0, -I * theta, 0, 1.5, 0, 0, 0],
            [0, 0, 0, I * theta, 0, 1.5, 0, 0],
            [I * theta, 0, 0, 0, 0, 0, 1.5, 0],
            [0, -I * theta, 0, 0, 0, 0, 0, 1.5],
            [1.5, 0, 0, 0, 0, 0, -I * theta, 0],
            [0, 1.5, 0, 0, 0, 0, 0, I * theta],
            [0, 0, 1.5, 0, I * theta, 0, 0, 0],
            [0, 0, 0, 1.5, 0, -I * theta, 0, 0],
        ]
    )
    expr = kron(x_gate, y_gate) + y_gate * z_gate
    assert isinstance(expr, AddBlock)
    assert expr.qubit_support == (0, 1, 2)
    assert expr.eigenvalues == (-2, 2, 0)
    assert expr.n_qubits == 3
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, -I, 0, 0, 0, -I, 0],
            [0, 0, 0, I, 0, 0, 0, -I],
            [I, 0, 0, 0, I, 0, 0, 0],
            [0, -I, 0, 0, 0, I, 0, 0],
            [0, 0, -I, 0, 0, 0, -I, 0],
            [0, 0, 0, -I, 0, 0, 0, I],
            [I, 0, 0, 0, I, 0, 0, 0],
            [0, I, 0, 0, 0, -I, 0, 0],
        ]
    )
    # breakpoint()


def test_chainblock() -> None:
    x_gate = X(0)
    theta = Parameter("theta")
    expr = x_gate * x_gate
    assert expr == 1
    expr = 1.5 * x_gate
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 1
    assert expr.qubit_support == (0,)
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[0, 1.5], [1.5, 0]])
    assert expr.eigenvalues == (-1.50000000000000, 1.50000000000000)
    expr = theta * x_gate
    # breakpoint()
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 1
    assert expr.qubit_support == (0,)
    assert expr.parameters == (theta,)
    assert expr.matrix == Matrix([[0, theta], [theta, 0]])
    # Doesn't work with Parameter but does with Parameters
    # assert expr.eigenvalues == (-theta, theta)
    # breakpoint()
    # breakpoint()
    y_gate = Y(1)
    expr = x_gate * y_gate
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 2
    assert expr.qubit_support == (0, 1)
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, -I, 0, 0], [I, 0, 0, 0]])
    assert expr.eigenvalues == (-1, 1)
    z_gate = Z(2)
    expr = (x_gate + y_gate) * z_gate
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 3
    assert expr.qubit_support == (0, 1, 2)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, -I, 0, 1, 0, 0, 0],
            [0, 0, 0, I, 0, -1, 0, 0],
            [I, 0, 0, 0, 0, 0, 1, 0],
            [0, -I, 0, 0, 0, 0, 0, -1],
            [1, 0, 0, 0, 0, 0, -I, 0],
            [0, -1, 0, 0, 0, 0, 0, I],
            [0, 0, 1, 0, I, 0, 0, 0],
            [0, 0, 0, -1, 0, -I, 0, 0],
        ]
    )
    assert expr.eigenvalues == (-2, 2, 0)
    expr = (2.0 * Y(0) + Z(1)) * (X(0) + 1.5 * Y(1))
    assert expr.parameters == ()
    expr = kron(x_gate, y_gate) * z_gate
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 3
    assert expr.qubit_support == (0, 1, 2)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, 0, 0, 0, 0, -I, 0],
            [0, 0, 0, 0, 0, 0, 0, I],
            [0, 0, 0, 0, I, 0, 0, 0],
            [0, 0, 0, 0, 0, -I, 0, 0],
            [0, 0, -I, 0, 0, 0, 0, 0],
            [0, 0, 0, I, 0, 0, 0, 0],
            [I, 0, 0, 0, 0, 0, 0, 0],
            [0, -I, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert expr.eigenvalues == (-1, 1)
    expr = 2.0 * X(0) @ Y(1) * (Z(2) + 1.5 * X(1))
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 3
    assert expr.qubit_support == (0, 1, 2)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, 0, 0, -3.0 * I, 0, -2.0 * I, 0],
            [0, 0, 0, 0, 0, -3.0 * I, 0, 2.0 * I],
            [0, 0, 0, 0, 2.0 * I, 0, 3.0 * I, 0],
            [0, 0, 0, 0, 0, -2.0 * I, 0, 3.0 * I],
            [-3.0 * I, 0, -2.0 * I, 0, 0, 0, 0, 0],
            [0, -3.0 * I, 0, 2.0 * I, 0, 0, 0, 0],
            [2.0 * I, 0, 3.0 * I, 0, 0, 0, 0, 0],
            [0, -2.0 * I, 0, 3.0 * I, 0, 0, 0, 0],
        ]
    )
    # assert expr.eigenvalues ==
    # (
    # 4.68908220859157e-128 - 2.23606797749979*I,
    # 4.25779875067892e-128 - 2.23606797749979*I,
    # -3.70039220956824e-127 - 2.23606797749979*I,
    # 3.69347270263832e-127 - 2.23606797749979*I,
    # 1.01777499569063e-64 + 2.23606797749979*I,
    # 6.97868642041278e-64 + 2.23606797749979*I,
    # -1.08031789449903e-63 + 2.23606797749979*I,
    # -3.80257987934937e-64 + 2.23606797749979*I
    # )
    # breakpoint()
    # assert isinstance(expr, ChainBlock)
    # assert expr.n_qubits == 2
    expr = (2.0 * X(0) @ Y(1)) * (
        Z(2) @ (1.5 * X(1))
    )  # Use parenthesis for correctness. TODO: Check why failing if not.
    assert isinstance(expr, ChainBlock)
    assert expr.n_qubits == 3
    assert expr.qubit_support == (0, 1, 2)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, 0, 0, 0, 0, 0, 3.0 * I],
            [0, 0, 0, 0, 0, 0, 3.0 * I, 0],
            [0, 0, 0, 0, 0, 3.0 * I, 0, 0],
            [0, 0, 0, 0, 3.0 * I, 0, 0, 0],
            [0, 0, 0, 3.0 * I, 0, 0, 0, 0],
            [0, 0, 3.0 * I, 0, 0, 0, 0, 0],
            [0, 3.0 * I, 0, 0, 0, 0, 0, 0],
            [3.0 * I, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    eig_val = 3.0 * I
    assert expr.eigenvalues == (-ChainBlock(*eig_val.args), ChainBlock(*eig_val.args))


def test_kronblock() -> None:
    x_gate = X(0)
    y_gate = Y(1)
    expr = X(0) @ Y(1)
    assert isinstance(expr, KronBlock)
    assert expr.n_qubits == 2
    assert expr.qubit_support == (0, 1)
    assert expr.parameters == ()
    assert expr.matrix == Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, -I, 0, 0], [I, 0, 0, 0]])
    assert expr.eigenvalues == (-1, 1)
    expr = 1.5 * X(0) @ Y(1)
    assert isinstance(expr, ChainBlock)  # ChainBlock takes precedence.
    assert expr.n_qubits == 2
    assert expr.qubit_support == (0, 1)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, 0, -1.5 * I],
            [0, 0, 1.5 * I, 0],
            [0, -1.5 * I, 0, 0],
            [1.5 * I, 0, 0, 0],
        ]
    )
    assert expr.eigenvalues == (-1.5, 1.5)
    expr = X(0) @ 2.0 * Y(1)
    # breakpoint()
    assert isinstance(expr, ChainBlock)  # ChainBlock takes precedence.
    assert expr.n_qubits == 2
    assert expr.qubit_support == (0, 1)
    assert expr.parameters == ()
    assert expr.matrix == Matrix(
        [
            [0, 0, 0, -2.0 * I],
            [0, 0, 2.0 * I, 0],
            [0, -2.0 * I, 0, 0],
            [2.0 * I, 0, 0, 0],
        ]
    )
    assert expr.eigenvalues == (-2.0, 2.0)

    with pytest.raises(ValueError):
        expr = (1.5 * X(0) + Y(1)) @ (2.0 * Y(1) * Z(0))
    # breakpoint()


def test_equality() -> None:
    x0 = X(0)
    x0p = X(0)
    x1 = X(1)
    assert x0 == x0
    assert x0p == x0
    assert x0 != x1


def test_idempotency() -> None:
    # Weirdness with Sympy as it should return
    # IdentitiyGate instead of the number 1.
    expr = X(0) ** 2
    assert expr == 1
    expr = X(0) ** 3
    assert expr == X(0)
    expr = X(0) ** 4
    assert expr == 1
    expr = X(0) * X(0)
    assert expr == 1
    expr = X(0) * X(0) * X(0)
    assert expr == X(0)


# @pytest.mark.skip
def test_circuit_conversion() -> None:
    backend = backend_factory(backend="pyq")
    theta = Parameter("theta")
    qtheta = QParameter("qtheta")
    values = {"qtheta": torch.tensor(0.5)}
    blocks = [X(0), Y(0), Z(0), RX(0, theta)]
    circuit = chain(*blocks)

    ops = []
    for block in circuit:
        print(block)
        if isinstance(block, (X, Y, Z)):
            q_op = TO_QUCINT_OPS[block.name]
            ops.append(q_op(block.qubit_support[0]))
        elif isinstance(block, RX):
            q_op = TO_QUCINT_OPS[block.name]
            ops.append(q_op(block.qubit_support[0], qtheta))

    q_circuit = QuantumCircuit(1, *ops)
    circuit_pyq, _, embed, params = backend.convert(q_circuit)
    exp = backend.run(circuit_pyq, embed(params, values))
    assert torch.allclose(
        exp,
        torch.tensor([[0.0000 - 0.9689j, -0.2474 + 0.0000j]], dtype=torch.complex128),
        atol=1.0e-5,
    )
