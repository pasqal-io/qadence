from __future__ import annotations

import importlib
from abc import ABC, abstractproperty
from typing import Any, Callable

from qadence.backend import BackendConfiguration
from qadence.blocks import AbstractBlock, CompositeBlock, ParametricBlock, PrimitiveBlock
from qadence.parameters import make_differentiable


def is_leaf(block: AbstractBlock) -> bool:
    return isinstance(block, PrimitiveBlock)


def converter_factory(backend_name: str) -> Callable:
    module_path = f"qadence.backends.{backend_name}.convert_ops"
    convert_block_fn: Callable
    try:
        module = importlib.import_module(module_path)
        convert_block_fn = getattr(module, "convert_block")

    except (ImportError, ModuleNotFoundError):
        pass

    return convert_block_fn


class Node(ABC):
    abstract: AbstractBlock
    native: Any

    @abstractproperty
    def is_leaf(self) -> bool:
        raise NotImplementedError


class BlockNode(Node):
    def __init__(self, block: AbstractBlock, native: Any, parameters: Any = None):
        self.abstract = block
        self.native = native
        self.parameters = parameters
        if self.parameters:
            self.diff_expr = make_differentiable(self.parameters.parameter)

    @property
    def is_leaf(self) -> bool:
        return True

    def is_parametric(self) -> bool:
        return self.parameters is not None


class CompositeNode(Node):
    def __init__(self, children: list[BlockNode]):
        self.children = children

    def native(self) -> Any:
        return tree_map(self, "native", None, [])

    def abstract(self) -> AbstractBlock:
        return tree_map(self, "abstract", None, [])

    def parameters(self) -> list[Any]:
        return tree_map(self, "parameters", None, [])

    @property
    def is_leaf(self) -> bool:
        return False

    def flatten(self) -> list:
        return self

    def unflatten(self, lst: list) -> CompositeNode:
        return self


def tree_factory(block: AbstractBlock, backend: str, config: BackendConfiguration) -> Node:
    convert_fn = converter_factory(backend)

    def build_tree(block: AbstractBlock) -> Node:
        if block is None:
            return

        if isinstance(block, CompositeBlock):
            children = [build_tree(sub_block) for sub_block in block.blocks]
            node = CompositeNode(children)
        else:
            native_block = convert_fn(block, config)
            parameters = None
            if isinstance(block, ParametricBlock):
                parameters = block.parameters

            node = BlockNode(block, native_block, parameters)
        return node

    return build_tree(block)


def tree_map(node: Node, target: Any, fn: Callable = None, data: list = None) -> Any:
    if not data:
        data = []
    if node.is_leaf:
        try:
            data += [getattr(node, target)]
        except Exception:
            pass
    else:
        data += [tree_map(b, target, fn) for b in getattr(node, "children")]
    return data
