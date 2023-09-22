from __future__ import annotations

from qadence import QNN, Parameter, QuantumCircuit, feature_map, hea, total_magnetization
from qadence.draw import savefig

n = 5
fm = feature_map(n)
va = hea(n, depth=2)
obs = total_magnetization(n)
circ = QuantumCircuit(n, fm, va)
print(circ)

obs.tag = "TM"
qnn = QNN(circuit=circ, observable=obs * Parameter("w"))
print(qnn)

# in a jupyter notebook you can use qadence.draw.display
savefig(qnn, "test.png", pad=(50, 10))
