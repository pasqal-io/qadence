from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from qadence.backend import BackendConfiguration
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import chain, kron, tag
from qadence.circuit import QuantumCircuit
from qadence.divergences import js_divergence
from qadence.measurements import Measurements
from qadence.model import QuantumModel
from qadence.operations import SWAP, H, I, S
from qadence.transpile import reassign
from qadence.types import BackendName, DiffMode, OverlapMethod
from qadence.utils import P0, P1

# Modules to be automatically added to the qadence namespace
__all__ = ["Overlap", "OverlapMethod"]


def _cswap(control: int, target1: int, target2: int) -> AbstractBlock:
    # construct controlled-SWAP block
    cswap_blocks = kron(P0(control), I(target1), I(target2)) + kron(
        P1(control), SWAP(target1, target2)
    )
    cswap = tag(cswap_blocks, f"CSWAP({control}, {target1}, {target2})")

    return cswap


def _controlled_unitary(control: int, unitary_block: AbstractBlock) -> AbstractBlock:
    n_qubits = unitary_block.n_qubits

    # shift qubit support of unitary
    shifted_unitary_block = reassign(unitary_block, {i: control + i + 1 for i in range(n_qubits)})

    # construct controlled-U block
    cu_blocks = kron(P0(control), *[I(control + i + 1) for i in range(n_qubits)]) + kron(
        P1(control), shifted_unitary_block
    )
    cu = tag(cu_blocks, f"c-U({control}, {shifted_unitary_block.qubit_support})")

    return cu


def _is_counter_list(lst: list[Counter]) -> bool:
    return all(map(lambda x: isinstance(x, Counter), lst)) and isinstance(lst, list)


def _select_overlap_method(
    method: OverlapMethod,
    backend: BackendName,
    bra_circuit: QuantumCircuit,
    ket_circuit: QuantumCircuit,
) -> tuple[Callable, QuantumCircuit, QuantumCircuit]:
    if method == OverlapMethod.EXACT:
        fn = overlap_exact

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = fn(bras, kets)
            return overlap

    elif method == OverlapMethod.JENSEN_SHANNON:

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = overlap_jensen_shannon(bras, kets)
            return overlap

    elif method == OverlapMethod.COMPUTE_UNCOMPUTE:
        # create a single circuit from bra and ket circuits
        bra_circuit = QuantumCircuit(
            bra_circuit.n_qubits, bra_circuit.block, ket_circuit.block.dagger()
        )
        ket_circuit = None  # type: ignore[assignment]

        def _overlap_fn(  # type: ignore [misc]
            param_values: dict, bra_calc_fn: Callable, bra_state: Tensor | None, *_: Any
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            overlap = overlap_compute_uncompute(bras)
            return overlap

    elif method == OverlapMethod.SWAP_TEST:
        n_qubits = bra_circuit.n_qubits

        # shift qubit support of bra and ket circuit blocks
        shifted_bra_block = reassign(bra_circuit.block, {i: i + 1 for i in range(n_qubits)})
        shifted_ket_block = reassign(
            ket_circuit.block, {i: i + n_qubits + 1 for i in range(n_qubits)}
        )
        ket_circuit = None  # type: ignore[assignment]

        # construct swap test circuit
        state_blocks = kron(shifted_bra_block, shifted_ket_block)
        cswap_blocks = chain(*[_cswap(0, n + 1, n + 1 + n_qubits) for n in range(n_qubits)])
        swap_test_blocks = chain(H(0), state_blocks, cswap_blocks, H(0))
        bra_circuit = QuantumCircuit(2 * n_qubits + 1, swap_test_blocks)

        def _overlap_fn(  # type: ignore [misc]
            param_values: dict, bra_calc_fn: Callable, bra_state: Tensor | None, *_: Any
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            overlap = overlap_swap_test(bras)
            return overlap

    elif method == OverlapMethod.HADAMARD_TEST:
        n_qubits = bra_circuit.n_qubits

        # construct controlled bra and ket blocks
        c_bra_block = _controlled_unitary(0, bra_circuit.block)
        c_ket_block = _controlled_unitary(0, ket_circuit.block.dagger())

        # construct swap test circuit for Re part
        re_blocks = chain(H(0), c_bra_block, c_ket_block, H(0))
        bra_circuit = QuantumCircuit(n_qubits + 1, re_blocks)

        # construct swap test circuit for Im part
        im_blocks = chain(H(0), c_bra_block, c_ket_block, S(0), H(0))
        ket_circuit = QuantumCircuit(n_qubits + 1, im_blocks)

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = overlap_hadamard_test(bras, kets)
            return overlap

    return _overlap_fn, bra_circuit, ket_circuit


def overlap_exact(bras: Tensor, kets: Tensor) -> Tensor:
    """Calculate overlap using exact quantum mechanical definition.

    Args:
        bras (Tensor): full bra wavefunctions
        kets (Tensor): full ket wavefunctions

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with each ket
    """
    return torch.abs(torch.sum(bras.conj() * kets, dim=1)) ** 2


def fidelity(bras: Tensor, kets: Tensor) -> Tensor:
    return overlap_exact(bras, kets)


def overlap_jensen_shannon(bras: list[Counter], kets: list[Counter]) -> Tensor:
    """Calculate overlap from bitstring counts using Jensen-Shannon divergence method.

    Args:
        bras (list[Counter]): bitstring counts corresponding to bra wavefunctions
        kets (list[Counter]): bitstring counts corresponding to ket wavefunctions

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with each ket
    """
    return 1 - torch.tensor([js_divergence(p, q) for p, q in zip(bras, kets)])


def overlap_compute_uncompute(bras: Tensor | list[Counter]) -> Tensor:
    """Calculate overlap using compute-uncompute method.

    From full wavefunctions or bitstring counts.

    Args:
        bras (Tensor | list[Counter]): full bra wavefunctions or bitstring counts

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with zeros ket
    """
    if isinstance(bras, Tensor):
        # calculate exact overlap of full bra wavefunctions with |0> state
        overlap = torch.abs(bras[:, 0]) ** 2

    elif isinstance(bras, list):
        # estimate overlap as the fraction of shots when "0..00" bitstring was observed
        n_qubits = len(list(bras[0].keys())[0])
        n_shots = sum(list(bras[0].values()))
        overlap = torch.tensor([p["0" * n_qubits] / n_shots for p in bras])

    return overlap


def overlap_swap_test(bras: Tensor | list[Counter]) -> Tensor:
    """Calculate overlap using swap test method.

    From full wavefunctions or bitstring counts.

    Args:
        bras (Tensor | list[Counter]): full bra wavefunctions or bitstring counts

    Returns:
        Tensor: overlap tensor
    """
    if isinstance(bras, Tensor):
        n_qubits = int(np.log2(bras.shape[1]))

        # define measurement operator  |0><0| x I
        proj_op = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        ident_op = torch.diag(torch.tensor([1.0 for _ in range(2 ** (n_qubits - 1))]))
        meas_op = torch.kron(proj_op, ident_op).type(torch.complex128)

        # estimate overlap from ancilla qubit measurement
        prob0 = (bras.conj() * torch.matmul(meas_op, bras.t()).t()).sum(dim=1).real

    elif _is_counter_list(bras):
        # estimate overlap as the fraction of shots when 0 was observed on ancilla qubit
        n_qubits = len(list(bras[0].keys())[0])
        n_shots = sum(list(bras[0].values()))
        prob0 = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras
            ]
        )
    else:
        raise TypeError("Incorrect type passed for bras argument.")

    # construct final overlap tensor
    overlap = 2 * prob0 - 1

    return overlap


def overlap_hadamard_test(
    bras_re: Tensor | list[Counter], bras_im: Tensor | list[Counter]
) -> Tensor:
    """Calculate overlap using Hadamard test method.

    From full wavefunctions or bitstring counts.

    Args:
        bras_re (Tensor | list[Counter]): full bra wavefunctions or bitstring counts
        for estimation of overlap's real part
        bras_im (Tensor | list[Counter]): full bra wavefunctions or bitstring counts
        for estimation of overlap's imaginary part

    Returns:
        Tensor: overlap tensor
    """
    if isinstance(bras_re, Tensor) and isinstance(bras_im, Tensor):
        n_qubits = int(np.log2(bras_re.shape[1]))

        # define measurement operator  |0><0| x I
        proj_op = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        ident_op = torch.diag(torch.tensor([1.0 for _ in range(2 ** (n_qubits - 1))]))
        meas_op = torch.kron(proj_op, ident_op).type(torch.complex128)

        # estimate overlap from ancilla qubit measurement
        prob0_re = (bras_re * torch.matmul(meas_op, bras_re.conj().t()).t()).sum(dim=1).real
        prob0_im = (bras_im * torch.matmul(meas_op, bras_im.conj().t()).t()).sum(dim=1).real

    elif _is_counter_list(bras_re) and _is_counter_list(bras_im):
        # estimate overlap as the fraction of shots when 0 was observed on ancilla qubit
        n_qubits = len(list(bras_re[0].keys())[0])
        n_shots = sum(list(bras_re[0].values()))
        prob0_re = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras_re
            ]
        )
        prob0_im = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras_im
            ]
        )
    else:
        raise TypeError("Incorrect types passed for bras_re and kets_re arguments.")

    # construct final overlap tensor
    overlap = (2 * prob0_re - 1) ** 2 + (2 * prob0_im - 1) ** 2

    return overlap


class Overlap(QuantumModel):
    def __init__(
        self,
        bra_circuit: QuantumCircuit,
        ket_circuit: QuantumCircuit,
        backend: BackendName = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        configuration: BackendConfiguration | dict | None = None,
        method: OverlapMethod = OverlapMethod.EXACT,
    ):
        self.backend_name = backend
        self.method = method

        overlap_fn, bra_circuit, ket_circuit = _select_overlap_method(
            method, backend, bra_circuit, ket_circuit
        )
        self.overlap_fn = overlap_fn

        super().__init__(
            bra_circuit,
            backend=backend,
            diff_mode=diff_mode,
            measurement=measurement,
            configuration=configuration,
        )
        self.bra_feat_param_names = set([inp.name for inp in self.inputs])

        if ket_circuit:
            self.ket_model = QuantumModel(
                ket_circuit,
                backend=backend,
                diff_mode=diff_mode,
                measurement=measurement,
                configuration=configuration,
            )
            self.ket_feat_param_names = set([inp.name for inp in self.ket_model.inputs])
        else:
            self.ket_model = None  # type: ignore [assignment]
            self.ket_feat_param_names = set([])

    def _process_param_values(
        self, bra_param_values: dict[str, Tensor], ket_param_values: dict[str, Tensor]
    ) -> dict:
        # we assume that either batch sizes are equal or 0 in case when no user params
        # are present in bra/ket
        bra_param_values = {
            k: v.reshape(-1) if v.shape == () else v for k, v in bra_param_values.items()
        }
        batch_size_bra = (
            len(list(bra_param_values.values())[0]) if len(bra_param_values) != 0 else 0
        )
        ket_param_values = {
            k: v.reshape(-1) if v.shape == () else v for k, v in ket_param_values.items()
        }
        batch_size_ket = (
            len(list(ket_param_values.values())[0]) if len(ket_param_values) != 0 else 0
        )
        new_bra_param_values = bra_param_values.copy()
        new_ket_param_values = ket_param_values.copy()

        # if len(self.bra_feat_param_names) + len(self.ket_feat_param_names) <= 2:

        if len(self.bra_feat_param_names.union(self.ket_feat_param_names)) == 2:
            # extend bra parameter tensors
            for param_name in new_bra_param_values.keys():
                new_bra_param_values[param_name] = new_bra_param_values[param_name].repeat(
                    batch_size_ket
                )

            # extend ket parameter tensors
            for param_name in new_ket_param_values.keys():
                idxs = torch.cat(
                    [
                        torch.ones(batch_size_bra, dtype=torch.int64) * i
                        for i in range(batch_size_ket)
                    ]
                )
                new_ket_param_values[param_name] = new_ket_param_values[param_name][idxs]

            if self.method in [OverlapMethod.EXACT, OverlapMethod.JENSEN_SHANNON]:
                param_values = {"bra": new_bra_param_values, "ket": new_ket_param_values}
            elif self.method in [
                OverlapMethod.COMPUTE_UNCOMPUTE,
                OverlapMethod.SWAP_TEST,
                OverlapMethod.HADAMARD_TEST,
            ]:
                # merge bra and ket param values to simulate all wavefunctions in one batch
                new_bra_param_values.update(new_ket_param_values)
                param_values = {"bra": new_bra_param_values}
                if self.method == OverlapMethod.HADAMARD_TEST:
                    param_values["ket"] = new_bra_param_values

        elif len(self.bra_feat_param_names.union(self.ket_feat_param_names)) < 2:
            if batch_size_bra == batch_size_ket or batch_size_bra == 0 or batch_size_ket == 0:
                param_values = {"bra": bra_param_values, "ket": ket_param_values}
            else:
                raise ValueError("Batch sizes of both bra and ket parameters must be equal.")

        else:
            raise ValueError("Multiple feature parameters for bra/ket are not currently supported.")

        return param_values

    def forward(  # type: ignore [override]
        self,
        bra_param_values: dict[str, Tensor] = {},
        ket_param_values: dict[str, Tensor] = {},
        bra_state: Tensor | None = None,
        ket_state: Tensor | None = None,
        n_shots: int = 0,
    ) -> Tensor:
        # reformat parameters
        param_values = self._process_param_values(bra_param_values, ket_param_values)

        # determine bra and ket calculation functions
        if n_shots == 0:
            bra_calc_fn = getattr(self, "run")
            ket_calc_fn = getattr(self.ket_model, "run", None)
        else:

            def bra_calc_fn(values: dict, state: Tensor) -> Any:
                return getattr(self, "sample")(values, n_shots, state)

            def ket_calc_fn(values: dict, state: Tensor) -> Any:
                return getattr(self.ket_model, "sample", lambda *_: _)(values, n_shots, state)

        # calculate overlap
        overlap = self.overlap_fn(
            param_values, bra_calc_fn, bra_state, ket_calc_fn, ket_state  # type: ignore [arg-type]
        )

        # reshape output if needed
        if len(self.bra_feat_param_names.union(self.ket_feat_param_names)) < 2:
            overlap = overlap[:, None]
        else:
            batch_size_bra = max(len(list(bra_param_values.values())[0]), 1)
            batch_size_ket = max(len(list(ket_param_values.values())[0]), 1)
            overlap = overlap.reshape((batch_size_ket, batch_size_bra)).t()

        return overlap
