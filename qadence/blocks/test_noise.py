from qadence import run
from qadence.operations.noise import BitFlip


bf = BitFlip(target = 0, noise_probability = 0.5)
print(run(bf))