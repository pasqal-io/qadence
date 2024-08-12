from __future__ import annotations

import typing
from enum import Enum
from itertools import chain as _flatten
from logging import getLogger
from typing import Generator, List, Type, TypeVar, Union, get_args

from sympy import Array, Basic, Expr
from torch import Tensor

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    KronBlock,
    ParametricBlock,
    PrimitiveBlock,
    PutBlock,
    ScaleBlock,
    TimeEvolutionBlock,
)
from qadence.blocks.analog import (
    AnalogBlock,
    AnalogComposite,
    ConstantAnalogRotation,
    InteractionBlock,
)
from qadence.blocks.analog import chain as analog_chain
from qadence.blocks.analog import kron as analog_kron
from qadence.exceptions import NotPauliBlockError
from qadence.parameters import Parameter

logger = getLogger(__name__)


TPrimitiveBlock = TypeVar("TPrimitiveBlock", bound=PrimitiveBlock)
TCompositeBlock = TypeVar("TCompositeBlock", bound=CompositeBlock)


def _construct(
    Block: Type[TCompositeBlock],
    args: tuple[Union[AbstractBlock, Generator, List[AbstractBlock]], ...],
) -> TCompositeBlock:
    if len(args) == 1 and isinstance(args[0], Generator):
        args = tuple(args[0])
    return Block([b for b in args])  # type: ignore [arg-type]


def chain(*args: Union[AbstractBlock, Generator, List[AbstractBlock]]) -> ChainBlock:
    """Chain blocks sequentially.

    On digital backends this can be interpreted
    loosely as a matrix mutliplication of blocks. In the analog case it chains
    blocks in time.

    Arguments:
        *args: Blocks to chain. Can also be a generator.

    Returns:
        ChainBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, chain

    b = chain(X(0), Y(0))

    # or use a generator
    b = chain(X(i) for i in range(3))
    print(b)
    ```
    """
    # ugly hack to use `AnalogChain` if we are dealing only with analog blocks
    if len(args) and all(
        isinstance(a, AnalogBlock) or isinstance(a, AnalogComposite) for a in args
    ):
        return analog_chain(*args)  # type: ignore[return-value,arg-type]
    return _construct(ChainBlock, args)


def kron(*args: Union[AbstractBlock, Generator]) -> KronBlock:
    """Stack blocks vertically.

    On digital backends this can be intepreted
    loosely as a kronecker product of blocks. In the analog case it executes
    blocks parallel in time.

    Arguments:
        *args: Blocks to kron. Can also be a generator.

    Returns:
        KronBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, kron

    b = kron(X(0), Y(1))

    # or use a generator
    b = kron(X(i) for i in range(3))
    print(b)
    ```
    """
    # ugly hack to use `AnalogKron` if we are dealing only with analog blocks
    if len(args) and all(
        isinstance(a, AnalogBlock) or isinstance(a, AnalogComposite) for a in args
    ):
        return analog_kron(*args)  # type: ignore[return-value,arg-type]
    return _construct(KronBlock, args)


def add(*args: Union[AbstractBlock, Generator]) -> AddBlock:
    """Sums blocks.

    Arguments:
        *args: Blocks to add. Can also be a generator.

    Returns:
        AddBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, add

    b = add(X(0), Y(0))

    # or use a generator
    b = add(X(i) for i in range(3))
    print(b)
    ```
    """
    return _construct(AddBlock, args)


def tag(block: AbstractBlock, tag: str) -> AbstractBlock:
    block.tag = tag
    return block


def put(block: AbstractBlock, min_qubit: int, max_qubit: int) -> PutBlock:
    from qadence.transpile import reassign

    support = tuple(range(min(block.qubit_support), max(block.qubit_support) + 1))
    shifted_block = reassign(block, {i: i - min(support) for i in support})
    return PutBlock(shifted_block, tuple(range(min_qubit, max_qubit + 1)))


def primitive_blocks(block: AbstractBlock) -> List[PrimitiveBlock]:
    """Extract the primitive blocks from a `CompositeBlock`.

    In the case of an `AddBlock`, the `AddBlock` is considered primitive.

    Args:
        blocks: An Iterable of `AbstractBlock`s.
    Returns:
        List[PrimitiveBlock]
    """

    if isinstance(block, ScaleBlock):
        return primitive_blocks(block.block)

    elif isinstance(block, PrimitiveBlock):
        return [block]

    elif isinstance(block, CompositeBlock):
        return list(_flatten(*(primitive_blocks(b) for b in block.blocks)))

    else:
        raise NotImplementedError(f"Non-supported operation of type {type(block)}")


def get_pauli_blocks(block: AbstractBlock, raises: bool = False) -> List[PrimitiveBlock]:
    """Extract Pauli operations from an arbitrary input block.

    Args:
        block (AbstractBlock): The input block to extract Pauli operations from
        raises (bool, optional): Raise an exception if the block contains something
            else than Pauli blocks.

    Returns:
        List[PrimitiveBlock]: List of Pauli operations
    """
    from qadence import operations

    paulis = []
    for b in primitive_blocks(block):
        if isinstance(b, get_args(operations.primitive.TPauliBlock)):
            paulis.append(b)
        else:
            if raises:
                raise NotPauliBlockError(f"{b.name} is not a Pauli operation")
            continue

    return paulis


def parameters(block: AbstractBlock) -> list[Parameter | Basic]:
    """Extract the Parameters of a block."""
    params = []
    exprs = uuid_to_expression(block).values()
    for expr in exprs:
        symbols = list(expr.free_symbols)
        if len(symbols) == 0:
            # assert expr.is_number or isinstance(expr, sympy.Matrix)
            params.append(expr)
        else:
            for s in symbols:
                params.append(s)
    return params


def uuid_to_block(block: AbstractBlock, d: dict[str, Expr] = None) -> dict[str, ParametricBlock]:
    from qadence import operations

    d = {} if d is None else d

    if isinstance(block, ScaleBlock):
        (uuid,) = block.parameters.uuids()
        d[uuid] = block
        uuid_to_block(block.block, d)

    # special analog cases should go away soon
    elif isinstance(
        block, (InteractionBlock, ConstantAnalogRotation, operations.AnalogEntanglement)
    ):
        for uuid in block.parameters.uuids():
            d[uuid] = block

    elif isinstance(block, CompositeBlock) or isinstance(block, AnalogComposite):
        for b in block.blocks:
            d = uuid_to_block(b, d)

    elif isinstance(block, ParametricBlock):
        if isinstance(block, TimeEvolutionBlock) and isinstance(block.generator, AbstractBlock):
            d = uuid_to_block(block.generator, d)
        for uuid in block.parameters.uuids():
            d[uuid] = block

    elif isinstance(block, PrimitiveBlock):
        return d

    else:
        raise NotImplementedError(f"'uuid_to_block' is not implemented for block: {type(block)}")

    return d


def uuid_to_expression(block: AbstractBlock) -> dict[str, Basic]:
    return {k: v.parameters._uuid_dict[k] for k, v in uuid_to_block(block).items()}


def expression_to_uuids(block: AbstractBlock) -> dict[Expr, list[str]]:
    """Creates a mapping between unique expressions and gate-level param_ids."""

    uuid_to_expr = uuid_to_expression(block)
    expr_to_uuid: dict[Expr, list[str]] = {}
    for uuid, expr in uuid_to_expr.items():
        expr_to_uuid.setdefault(expr, []).append(uuid)

    return expr_to_uuid


def uuid_to_eigen(
    block: AbstractBlock, rescale_eigenvals_timeevo: bool = False
) -> dict[str, Tensor]:
    """Creates a mapping between a parametric block's param_id and its' eigenvalues.

    This method is needed for constructing the PSR rules for a given block.

    A PSR shift factor is also added in the mapping for dealing
    with the time evolution case as it requires rescaling.

    Args:
        block (AbstractBlock): Block input
        rescale_eigenvals_timeevo (bool, optional): If True, rescale
        eigenvalues and shift factor
        by 2 times spectral gap
        for the TimeEvolutionBlock case to allow
        differientiating with Hamevo.
        Defaults to False.

    Returns:
        dict[str, Tensor]: Mapping between block's param_id, eigenvalues and
        PSR shift.

    !!! warn
        Will ignore eigenvalues of AnalogBlocks that are not yet computed.
    """

    result = {}
    for uuid, b in uuid_to_block(block).items():
        if b.eigenvalues_generator is not None:
            if b.eigenvalues_generator.numel() > 0:
                # GPSR assumes a factor 0.5 for differentiation
                # so need rescaling
                if isinstance(b, TimeEvolutionBlock) and rescale_eigenvals_timeevo:
                    if b.eigenvalues_generator.numel() > 1:
                        result[uuid] = (
                            b.eigenvalues_generator * 2.0,
                            0.5,
                        )
                    else:
                        result[uuid] = (
                            b.eigenvalues_generator * 2.0,
                            1.0 / (b.eigenvalues_generator.item() * 2.0)
                            if len(b.eigenvalues_generator) == 1
                            else 1.0,
                        )
                else:
                    result[uuid] = (b.eigenvalues_generator, 1.0)

                # leave only angle parameter uuid with eigenvals for ConstantAnalogRotation block
                if isinstance(block, ConstantAnalogRotation):
                    break

    return result


def expressions(block: AbstractBlock) -> list[Basic]:
    """Extract the expressions sitting in the 'parameters' field of a ParametricBlock.

    Each element of 'parameters' is a sympy expression which can be a constant,
    a single parameter or an expression consisting of both symbols and constants.
    """
    return list(set(uuid_to_expression(block).values()))


def block_is_qubit_hamiltonian(block: AbstractBlock) -> bool:
    try:
        _ = get_pauli_blocks(block, raises=True)
        return True
    except NotPauliBlockError:
        return False


def _support_primitive_block(
    block: PrimitiveBlock, support: dict[int, set[str]]
) -> dict[int, set[str]]:
    pauli = block.name.value if isinstance(block.name, Enum) else block.name
    index = block.qubit_support[0]
    if index in support.keys():
        support[index].add(pauli)
    else:
        support[index] = set(pauli)

    return support


def _check_commutation(block: AbstractBlock, support: dict[int, set[str]] | None = None) -> dict:
    # avoid circular import

    if support is None:
        support = {}

    if isinstance(block, AddBlock) or isinstance(block, KronBlock):
        for subblock in block.blocks:
            support = dict(_check_commutation(subblock, support=support))

    elif isinstance(block, ScaleBlock):
        support = dict(_check_commutation(block.block, support=support))

    elif isinstance(block, PrimitiveBlock):
        support = dict(_support_primitive_block(block, support))

    else:
        raise TypeError("Original block was not a Pauli-based QubitHamiltonian!")

    return support


def block_is_commuting_hamiltonian(block: AbstractBlock) -> bool:
    """Check whether a Pauli block is composed by commuting set of operators.

    Args:
        block (AbstractBlock): The Pauli block

    Returns:
        bool: flag which tells whether all the elements in the
            Pauli block are commuting or not
    """
    assert block_is_qubit_hamiltonian(block), "Only working for Pauli blocks"
    support = _check_commutation(block)
    for v in support.values():
        if len(v) > 1:
            return False
    return True


def get_block_by_uuid(block: AbstractBlock, uuid: str) -> ParametricBlock:
    return uuid_to_block(block)[uuid]


def get_blocks_by_expression(
    block: AbstractBlock, expr: Union[Parameter, Expr]
) -> list[AbstractBlock]:
    expr_to_uuids = expression_to_uuids(block)
    uuid_to_blk = uuid_to_block(block)
    return [uuid_to_blk[uuid] for uuid in expr_to_uuids[expr]]


def has_duplicate_vparams(block: AbstractBlock) -> bool:
    """Check if the given block has duplicated variational parameters.

    Args:
        block (AbstractBlock): The block to check

    Returns:
        bool: A boolean indicating whether the block has
            duplicated parameters or not
    """
    params = parameters(block)
    non_number = [p for p in params if not p.is_number]
    trainables = [p for p in non_number if p.trainable]
    return len(set(trainables)) != len(trainables)


@typing.no_type_check
def unroll_block_with_scaling(
    block: AbstractBlock, block_list: list[AbstractBlock] = None
) -> list[tuple[AbstractBlock, Basic]]:
    """Extract a set of terms in the given block with corresponding scales.

    This function takes an input block and extracts a list of operations
    with corresponding scaling factors.

    For example, consider the following block:
        b = 2. * Z(0) * Z(1) + 3. * (kron(X0), X(1)) + kron(Y(0), Y(1)))

    This function will return the following list of tuples:
        res = [
            ([Z(0) * Z(1)], 2.),
            (kron(X(0), X(1)), 3.),
            (kron(Y(0), Y(1)), 3.),
        ]

    Args:
        block (AbstractBlock): the given block
        block_list (list[AbstractBlock], optional): A list of blocks to which append the
            found terms. If None, an empty list is returned.

    Raises:
        TypeError: If the given block does not respect the expected format

    Returns:
        tuple[list[AbstractBlock], float]: A tuple with the list of blocks
            and a scaling factor in front
    """

    # Avoid circular imports.

    def _add_block(
        add_block: AddBlock, blist: list[tuple[AbstractBlock, float]]
    ) -> list[tuple[AbstractBlock, float]]:
        for b in add_block.blocks:
            blist = unroll_block_with_scaling(b, block_list=blist)
        return blist

    if block_list is None:
        block_list = []

    if isinstance(block, ScaleBlock):
        scaled_block = block.block
        scale: Expr = block.scale

        if not isinstance(scaled_block, AddBlock):
            block_list.append((block.block, scale))

        else:
            idx = len(block_list)
            block_list = _add_block(scaled_block, block_list)
            # param_id = block.param_id

            for i in range(idx, len(block_list)):
                (b, mul) = block_list[i]
                fact = scale * mul
                # make sure it gets picked up correctly in the parameters dictionary
                # if hasattr(b, 'param_id'):
                # b.param_id = param_id

                if not mul.is_number:
                    logger.warning(
                        """
Nested add block with multiple variational parameters. This might cause undefined behavior.
Consider rewriting your block as a single AddBlock instance.

For example, if you want to define a parametric observable with multiple variational
parameters, you should make sure that each parametric operation in the
AddBlock is defined separately. To make it more clear, you should write your
block in the following way:

> theta1 = VariationalParameter("theta1")
> theta2 = VariationalParameter("theta2")
>
> generator = theta1 * kron(X(0), X(1)) + theta1 * theta2 * kron(Z(2), Z(3))

and NOT this way:

> theta1 = VariationalParameter("theta1")
> theta2 = VariationalParameter("theta2")
>
> generator = theta1 * (kron(X(0), X(1)) + theta2 * kron(Z(2), Z(3)))
"""
                    )

                block_list[i] = (b, fact)

        return block_list

    elif (
        isinstance(block, KronBlock)
        or isinstance(block, ChainBlock)
        or isinstance(block, PrimitiveBlock)
    ):
        block_list.append((block, Parameter(1.0)))
        return block_list

    elif isinstance(block, AddBlock):
        return _add_block(block, block_list)

    else:
        raise TypeError(
            "Input block has an invalid type! It "
            "should be either a ScaleBlock or one of Add, Chain "
            f"and Kron blocks. Got {type(block)}."
        )


def assert_same_block(b1: AbstractBlock, b2: AbstractBlock) -> None:
    assert type(b1) == type(
        b2
    ), f"Block {b1} is not the same type ({type(b1)}) as Block {b2} ({type(b2)})"
    assert b1.name == b2.name, f"Block {b1} and {b2} don't have the same names!"
    assert (
        b1.qubit_support == b2.qubit_support
    ), f"Block {b1} and {b2} don't have the same qubit support!"
    if isinstance(b1, ParametricBlock) and isinstance(
        b2, ParametricBlock
    ):  # if the block is parametric, we can check some additional things
        assert len(b1.parameters.items()) == len(
            b2.parameters.items()
        ), f"Blocks {b1} and {b2} have differing numbers of parameters."
        for p1, p2 in zip(b1.parameters.expressions(), b2.parameters.expressions()):
            assert p1 == p2


def unique_parameters(block: AbstractBlock) -> list[Parameter]:
    """Return the unique parameters in the block.

    These parameters are the actual user-facing parameters which
    can be assigned by the user. Multiple gates can contain the
    same unique parameter

    Returns:
        list[Parameter]: List of unique parameters in the circuit
    """
    symbols = []
    for p in parameters(block):
        if isinstance(p, Array):
            continue
        elif not p.is_number and p not in symbols:
            symbols.append(p)
    return symbols
