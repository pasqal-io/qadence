from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qadence.blocks import chain, kron, primitive_blocks, tag
from qadence.circuit import QuantumCircuit
from qadence.constructors import hea
from qadence.draw import savefig
from qadence.operations import CNOT, RX, X, Y
from qadence.parameters import FeatureParameter, Parameter
from qadence.transpile import invert_endianness


def build_circuit(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    param = FeatureParameter("x")
    block = kron(*[RX(qubit, (qubit + 1) * param) for qubit in range(n_qubits)])
    fm = tag(block, tag="FM")

    # this tags it as "HEA"
    ansatz = hea(n_qubits, depth=depth)

    return QuantumCircuit(n_qubits, fm, ansatz)


def test_get_block_by_tag() -> None:
    # standard circuit
    circuit = build_circuit(n_qubits=4)

    ansatz = circuit.get_blocks_by_tag("HEA")
    assert len(ansatz) == 1
    assert ansatz[0].tag == "HEA"
    fm = circuit.get_blocks_by_tag("FM")
    assert len(fm) == 1
    assert fm[0].tag == "FM"

    # multiple blocks
    block = chain(RX(0, 0.5), kron(RX(1, 0.5), RX(2, 0.5)), hea(n_qubits=4), hea(n_qubits=4))
    circuit = QuantumCircuit(4, block)

    ansatz = circuit.get_blocks_by_tag("HEA")
    assert len(ansatz) == 2


def test_invert_endianness() -> None:
    block1 = X(0)
    block2 = Y(1)
    block3 = RX(2, Parameter("theta"))
    block4 = CNOT(3, 4)

    comp_block = chain(
        (block1 @ block2 @ block3),
        (block3 * block4),
        block1 + block2,
    )

    # comp_block = chain(block1, block2, block4)
    orig = chain(*primitive_blocks(comp_block))
    circ = QuantumCircuit(5, comp_block)
    circ2 = invert_endianness(circ)
    new = primitive_blocks(circ2.block)
    orig = invert_endianness(orig, 5, False)  # type: ignore [assignment]
    for b1, b2 in zip(orig, new):  # type: ignore
        assert b1.name == b2.name
        assert b1.qubit_support == b2.qubit_support


def test_circuit_dict() -> None:
    circ = build_circuit(4)
    qc_dict = circ._to_dict()
    qc_copy = QuantumCircuit._from_dict(qc_dict)

    assert circ == qc_copy


def test_circuit_from_dumps() -> None:
    circ = build_circuit(4)
    qc_dumped = circ._to_json()
    loadedqcdict = json.loads(qc_dumped)
    loaded_qc = QuantumCircuit._from_dict(loadedqcdict)

    assert circ == loaded_qc


def test_loaded_circuit_from_json() -> None:
    circ = build_circuit(4)
    from pathlib import Path

    file_name = Path("tmp.json")
    circ._to_json(file_name)
    qc_copy = QuantumCircuit._from_json(file_name)
    os.remove(file_name)

    assert circ == qc_copy


@pytest.mark.parametrize(
    "n_qubits",
    [2, 4],
)
def test_underlying_hea(n_qubits: int) -> None:
    from qadence.blocks import ChainBlock

    param = FeatureParameter("x")
    block = kron(*[RX(qubit, (qubit + 1) * param) for qubit in range(n_qubits)])
    fm = tag(block, tag="FM")

    # this tags it as "HEA"
    ansatz = hea(n_qubits=n_qubits, depth=2)

    mychain = chain(fm, ansatz)
    d = mychain._to_dict()
    mychain1 = ChainBlock._from_dict(d)

    assert mychain == mychain1


def test_circ_operator() -> None:
    x = Parameter("x", trainable=True, value=1.0)
    myrx = RX(0, x)
    qc = QuantumCircuit(1, myrx)
    assert x in qc
    assert myrx in qc
    assert x in myrx


def test_hea_operators() -> None:
    n_qubits = 4
    param = FeatureParameter("x")
    block = kron(*[RX(qubit, (qubit + 1) * param) for qubit in range(n_qubits)])
    fm = tag(block, tag="FM")
    # this tags it as "HEA"
    ansatz = hea(n_qubits=n_qubits, depth=2)
    mychain = chain(fm, ansatz)
    assert param in mychain


@pytest.mark.parametrize("fname", ["circuit.png", "circuit.pdf", "circuit.png"])
@pytest.mark.skip
def test_savefig_circuit(fname: str) -> None:
    circuit = build_circuit(4, depth=2)
    savefig(circuit, fname)
    assert os.path.isfile(fname)
    Path.unlink(Path(fname))
