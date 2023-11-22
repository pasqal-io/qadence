from torch import pi
from qadence import *
from qadence.draw import display
from qadence.transpile.block import fill_identities


b = chain(CNOT(0,1), CNOT(1,2))
b = chain(kron(CNOT(0,1)), kron(CNOT(1,2)))
b = kron(CNOT(0,1))
b = chain(SWAP(0,1), SWAP(0,3))
print(fill_identities(b, 0, 3))
raise

# b = fill_identities(CPHASE(0,1,pi), start=0, stop=3)
b = fill_identities(b, start=0, stop=1)
print(f"{b = }")
raise
# from IPython import embed; embed()


b = chain(CPHASE(0, 1, pi), CPHASE(0, 2, pi/2), CPHASE(0, 3, pi/4))
display(b)
b = fill_identities(b, 0, 3)
print(f"{b = }")
