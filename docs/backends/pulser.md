The **Pulser backend** features a basic integration with the pulse-level programming
interface Pulser. This backend offers for now few simple operations
which are translated into a valid, non time-dependent pulse sequence. In particular, one has access to:

* analog rotations: `AnalogRx` and `AnalogRy` blocks
* free evolution blocks (basically no pulse, just interaction): `AnalogWait` block
* a block for creating entangled states: `AnalogEntanglement`
* digital rotation `Rx` and `Ry`

### ::: qadence.backends.pulser.backend

### ::: qadence.backends.pulser.devices

### ::: qadence.backends.pulser.pulses

### ::: qadence.backends.pulser.convert_ops
