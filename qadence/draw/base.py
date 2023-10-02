from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, Tuple, Union

import cairo
from cairo import Context
from rich.console import Console, RenderableType
from rich.tree import Tree

from qadence import QuantumCircuit
from qadence.blocks import AbstractBlock, chain
from qadence.models import QNN
from qadence.models.quantum_model import ConvertedObservable
from qadence.transpile import validate
from qadence.types import FigFormat


class Renderable(ABC):
    @abstractmethod
    def render(self, context: Context) -> None:
        pass

    @abstractmethod
    def measure(self, context: Context) -> tuple[float, float]:
        pass

    @abstractproperty
    def depth(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def _title(self) -> str:
        pass

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._title)
        else:
            tree.add(self._title)
        return tree

    def __repr__(self) -> str:
        console = Console()
        with console.capture() as cap:
            console.print(self.__rich_tree__())
        return cap.get().strip()  # type: ignore [no-any-return]

    def __ascii__(self, console: Console) -> RenderableType:
        raise NotImplementedError


class NestedRenderable(Renderable):
    renderable: Renderable

    @property
    def depth(self) -> int:
        return 1 + self.renderable.depth

    @property
    def _title(self) -> str:
        return f"{type(self).__name__}"

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._title)
        else:
            tree = tree.add(self._title)
        self.renderable.__rich_tree__(tree)
        return tree


class CompositeRenderable(Renderable):
    renderables: list[Renderable]

    @property
    def depth(self) -> int:
        return 1 + max([r.depth for r in self.renderables])

    @property
    def _title(self) -> str:
        return f"{type(self).__name__}"

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._title)
        else:
            tree = tree.add(self._title)
        for rnd in self.renderables:
            tree = tree.add(rnd._title)
            rnd.__rich_tree__(tree)
        return tree


@dataclass
class Padding:
    left: float
    right: float
    top: float
    bottom: float
    gap: float = 5

    def min(self) -> float:
        return min(self.left, self.right, self.top, self.bottom)

    def __iter__(self) -> Iterable:
        return iter((self.left, self.right, self.top, self.bottom))


def _render(
    rnd: Renderable,
    width: int = None,
    height: int = None,
    pad: Tuple[int, int] = (15, 15),
    fig_format: FigFormat = FigFormat.PNG,
) -> cairo.Surface:
    fmt = cairo.FORMAT_ARGB32

    if width is None or height is None:
        with cairo.ImageSurface(fmt, 200, 200) as surface:
            context = Context(surface)
            context.select_font_face("Fira Code", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            (w, h) = rnd.measure(context)
            width, height = int(w), int(h)

    (xoffset, yoffset) = pad
    dims = (width + 2 * xoffset, height + 2 * yoffset)

    if fig_format == FigFormat.SVG:
        surface = cairo.SVGSurface(None, *dims)
    elif fig_format == FigFormat.PDF:
        surface = cairo.PDFSurface(None, *dims)
    else:
        surface = cairo.ImageSurface(fmt, *dims)

    context = Context(surface)
    context.select_font_face("Fira Code", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    context.move_to(xoffset, yoffset)
    rnd.render(context)

    return surface


def render(
    block_or_circ: Union[AbstractBlock, QuantumCircuit, QNN],
    depth: int = None,
    width: int = None,
    height: int = None,
    pad: Tuple[int, int] = (15, 15),
    preprocess: bool = True,
    fig_format: FigFormat = FigFormat.PNG,
) -> cairo.SVGSurface:
    if isinstance(block_or_circ, AbstractBlock):
        block = block_or_circ
    elif isinstance(block_or_circ, QuantumCircuit):
        block = block_or_circ.block
    elif isinstance(block_or_circ, QNN):
        circuit_blocks = [block_or_circ._circuit.abstract.block]
        if isinstance(block_or_circ._observable, list):
            circuit_blocks.extend([block.abstract for block in block_or_circ._observable])
        elif isinstance(block_or_circ._observable, ConvertedObservable):
            circuit_blocks.append(block_or_circ._observable.abstract)
        block = chain(*circuit_blocks)  # type: ignore[union-attr]

    if preprocess:
        block = validate(block)

    if depth is None:
        depth = block.depth

    _, renderable = block.__grid__(depth)
    return _render(renderable, width, height, pad, fig_format=fig_format)


def display(
    block: Union[AbstractBlock, QuantumCircuit, QNN],
    depth: int = None,
    width: int = None,
    height: int = None,
    pad: Tuple[int, int] = (15, 15),
    preprocess: bool = True,
) -> None:
    from io import BytesIO

    from IPython.display import Image
    from IPython.display import display as ipy_display

    surface = render(block, depth, width, height, pad, preprocess)

    with BytesIO() as fileobj:
        surface.write_to_png(fileobj)
        ipy_display(Image(fileobj.getvalue(), width=width))


def html_string(
    block: Union[AbstractBlock, QuantumCircuit, QNN],
    depth: int = None,
    width: int = None,
    height: int = None,
    pad: Tuple[int, int] = (15, 15),
    preprocess: bool = True,
) -> str:
    bytes_buffer = BytesIO()
    surface = render(block, depth, width, height, pad, preprocess)
    surface.write_to_png(bytes_buffer)
    enc = b64encode(bytes_buffer.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{enc}" alt="Pycairo Image">'


def savefig(
    block: Union[AbstractBlock, QuantumCircuit, QNN],
    filename: str,
    depth: int = None,
    width: int = None,
    height: int = None,
    pad: Tuple[int, int] = (15, 15),
    preprocess: bool = True,
    fig_format: FigFormat | str = FigFormat.PNG,
) -> None:
    """Save circuit image as a file of a given format

    Args:
        block (Union[AbstractBlock, QuantumCircuit, QNN]): the block or circuit to save
        filename (str): the filename of the resulting figure
        depth (int, optional): how deep to go in the block hierarchy whe printing the figure
        width (int, optional): width of the figure in points. If None, it is automatically set.
        height (int, optional): height of the figure in points. If None, it is automatically set.
        pad (Tuple[int, int], optional): padding in points to add to the figure borders
        preprocess (bool, optional): Whether or not to move from global to local qubit numbers
            before rendering the block or circuit
        fig_format (FigFormat): the desired format to be chosen between "PNG", "PDF" or "SVG"
    """

    try:
        fig_format = FigFormat(fig_format)
    except ValueError:
        raise ValueError(
            "Figure format not recognized. Select among the"
            "following formats: 'PNG', 'PDF' or 'SVG'"
        )

    # render the figure without any temporary storage file
    surface = render(block, depth, width, height, pad, preprocess, fig_format=fig_format)

    if fig_format == FigFormat.PNG:
        surface.write_to_png(filename)

    else:
        # get width and height
        image_surface = surface.map_to_image(None)
        width = image_surface.get_width()
        height = image_surface.get_height()

        # set the figure as source for another figure to save to a file of the required format
        if fig_format == FigFormat.SVG:
            surface2 = cairo.SVGSurface(filename, width, height)
        elif fig_format == FigFormat.PDF:
            surface2 = cairo.PDFSurface(filename, width, height)

        ctx = cairo.Context(surface2)
        ctx.save()
        ctx.set_source_surface(surface)
        ctx.set_operator(cairo.Operator.SOURCE)
        ctx.paint()
        ctx.restore()
        surface2.flush()

    surface.flush()
