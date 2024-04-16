`qadence` offers a block-based system to construct quantum circuits in a flexible manner.

::: qadence.blocks.abstract

## Primitive blocks

::: qadence.blocks.primitive


## Analog blocks

To learn how to use analog blocks and how to mix digital & analog blocks, check out the
[digital-analog section](../digital_analog_qc/analog-basics.md) of the documentation.

Examples on how to use digital-analog blocks can be found in the
*examples folder of the qadence repo:

- Fit a simple sinus: `examples/digital-analog/fit-sin.py`
- Solve a QUBO: `examples/digital-analog/qubo.py`

::: qadence.blocks.analog

## Composite blocks

::: qadence.blocks.utils.chain
    options:
      show_root_heading: true
      show_root_full_path: false

::: qadence.blocks.utils.kron
    options:
      show_root_heading: true
      show_root_full_path: false

::: qadence.blocks.utils.add
    options:
      show_root_heading: true
      show_root_full_path: false

::: qadence.blocks.composite

## Converting blocks to matrices

::: qadence.blocks.block_to_tensor
