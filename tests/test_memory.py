# import torch
# from qadence import *
# from itertools import product
# from qadence.blocks.block_to_tensor import _fill_identities
# from qadence.states import zero_state
# from memory_profiler import profile
# from pyqtorch.core.utils import _apply_batch_gate
# import pyqtorch.modules as pyq

# # N_QUBITS = 30
# # QUBIT = 0
# # rxmat = block_to_tensor(RX(QUBIT,1.))
# # state = zero_state(N_QUBITS)


# # rxsparsemat = _fill_identities(rxmat,(0,), [i for i in range(N_QUBITS)]).squeeze(0)

# # @profile
# # def sparse_mm():
# #     state = zero_state(N_QUBITS)
# #     mat = rxsparsemat.to_sparse()
# #     breakpoint()
# #     for i in range(1000):
# #         state = torch.sparse.mm(rxsparsemat, state)


# def find_indices_of_bitstrings_with_zero_at_i(N:int, i:int, bit:str ='0'):
#     if i >= N or i < 0:
#         raise ValueError("Index i must be in the range [0, N-1]")

#     indices = []
#     for idx, p in enumerate(product('01', repeat=N)):
#         if p[i] == bit:
#             indices.append(idx)

#     return indices

# def basis_state_indices(n_qubits: int, target: int) -> torch.LongTensor:
#     n_target = 1
#     target = torch.tensor(target)
#     index_base = torch.arange(0, 2**(n_qubits-n_target))
#     t = n_qubits-1-target
#     zero_index = index_base + ((index_base >> t) << t)
#     one_index = index_base + (((index_base >> t) + 1) << t)
#     return torch.cat([zero_index.long(), one_index.long()])


# def apply_kjd_gate(state: torch.Tensor, mat: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
#     indices = basis_state_indices(n_qubits, qubit)
#     _, sorted_indices = torch.sort(indices)
#     return (mat @ state[indices].view(2, 2**n_qubits//2)).flatten()[sorted_indices]


# @profile
# def run_hea(n_iter:int) -> None:

#     N_QUBITS = 28
#     QUBITS = [i for i in range(N_QUBITS)]
#     DEPTH = 1
#     rx = RX(0,1.)
#     rxmat = block_to_tensor(rx)
#     ry = RY(0,1.)
#     rymat = block_to_tensor(ry)
#     rz = RZ(0,1.)
#     rzmat = block_to_tensor(rz)
#     state = zero_state(N_QUBITS).flatten()
#     for i in range(n_iter):
#         for d in range(DEPTH):
#             for qubit in QUBITS:
#                 state = apply_kjd_gate(state, rxmat * rymat * rzmat, qubit, N_QUBITS)

#     assert len(state) == 2 ** N_QUBITS


# @profile
# def run_loop_einsum(n_iter:int) -> None:
#     N_QUBITS = 28
#     QUBITS = [i for i in range(N_QUBITS)]
#     DEPTH = 1
#     rx = RX(0,1.)
#     rxmat = block_to_tensor(rx).reshape(2,2,1)
#     ry = RY(0,1.)
#     rymat = block_to_tensor(ry).reshape(2,2,1)
#     rz = RZ(0,1.)
#     rzmat = block_to_tensor(rz).reshape(2,2,1)

#     rxmat = block_to_tensor(rx).reshape(2,2,1)
#     state = pyq.zero_state(N_QUBITS)

#     for i in range(n_iter):
#         for d in range(DEPTH):
#             for qubit in QUBITS:
#                 state = _apply_batch_gate(state, rxmat * rymat * rzmat, (qubit,), N_QUBITS, 1)

# if __name__ == '__main__':

#     run_loop_einsum(1)
#     # wf= run(QuantumCircuit(N_QUBITS, rx)).flatten()
#     # breakpoint()
#     # if not torch.allclose(wf,state):
#         # print('state different')
