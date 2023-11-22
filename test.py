import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from copy import deepcopy
from matplotlib import cm
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qadence import ReuploadScaling


from qadence.draw import display
from qadence.backend import BackendName
from qadence.transpile import set_trainable
from qadence.backends.pytorch_wrapper import DiffMode
from qadence.constructors.feature_maps import feature_map
from qadence import QNN, RX, RZ, Z, QuantumCircuit, hea, Parameter, add,tag, kron, chain, BasisSet

def quantum_circuit(n_qubits, n_inputs, fm_parameters, depth, fm_sequence):
    # Feature Map
    fm_list = []
    op = RX
    fm_type = BasisSet.FOURIER

    if fm_sequence:

        if fm_parameters != n_inputs:
            raise ValueError("fm_parameters must be equal to n_inputs")
        
        fm_list = [feature_map(n_qubits=n_qubits, param="\u03C6" + str(ii), op=op, fm_type=fm_type, support=tuple(range(n_qubits)),reupload_scaling=ReuploadScaling.EXP) for ii in range(n_inputs)]
        fm = chain(*fm_list)
        tag(fm, "feature_map")

    else:
        if (n_qubits % fm_parameters != 0):
            raise ValueError("fm_parameters should be evenly distributed over n_inputs")
        
        for i in range(n_qubits):
            support_start = i * n_qubits // n_qubits
            support_end = (i + 1) * n_qubits // n_qubits if i < n_qubits - 1 else n_qubits

            if n_inputs > fm_parameters:
                param_i = "\u03C6" + str(i // (n_inputs // fm_parameters))
            else:
                param_i = "\u03C6" + str(i // (n_qubits // n_inputs))

            support = tuple(range(support_start, support_end))
            
            fm_i = feature_map(n_qubits=n_qubits, param=param_i, op=op, fm_type=fm_type, support=support)
            fm_list.append(fm_i)

        fm = kron(*fm_list)
        tag(fm, "feature_map")

    # hardware-efficient ansatz
    ansatz = hea(n_qubits=n_qubits, depth=depth, param_prefix="\u03B8", operations=[RZ,RX,RZ])
    tag(ansatz, "ansatz")
    
    return QuantumCircuit(n_qubits, fm, ansatz)

# Quantum Circuit
n_inputs = 2
fm_parameters = 2
n_qubits = 5
depth = 5
fm_sequence = True

qc_ansatz = quantum_circuit(n_qubits, n_inputs, fm_parameters, depth, fm_sequence)
#display(qc_ansatz)

# Measurements
w = Parameter("w")
observable = add(Z(i)*w for i in range(n_qubits))

# Create the quantum model to use for optimization
model = QNN(qc_ansatz, observable=observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
display(model._circuit.original)
