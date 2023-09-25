from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cairo import Context
from rich.tree import Tree

from qadence.draw.base import NestedRenderable, Padding, Renderable
from qadence.draw.box import STANDARD_BLOCK, roundrect
from qadence.draw.composite import Column
from qadence.draw.text import STANDARD_TEXT
from qadence.types import TDrawColor


@dataclass(repr=False)
class GridColumn(NestedRenderable):
    nqubits: int
    grid_renderables: list[tuple[tuple, Renderable]]
    pad: Padding = Padding(0, 0, 0, 0, 0)
    fill_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    border_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    init_qubit: Optional[int] = None

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()

        (column_width, column_height) = self.measure(context)

        # draw bounding box
        roundrect(
            context,
            x,
            y,
            column_width,
            column_height,
            self.pad.min(),
            self.fill_color,
            self.border_color,
        )
        context.stroke()

        # return to original point
        context.move_to(x, y)

        # offset padding to start drawing sub-boxes
        context.rel_move_to(self.pad.left, self.pad.top)

        for support, rnd in self.grid_renderables:
            (w, h) = rnd.measure(context)

            # draw wire lines
            if rnd.depth <= 2:
                draw_wires(context, support, column_width, self.pad, rnd)

            # draw in center
            dx = (column_width - self.pad.left - self.pad.right - w) / 2
            context.rel_move_to(dx, 0)
            rnd.render(context)
            context.rel_move_to(-dx, 0)

            # move down
            dy = h + self.pad.gap
            dy += context.get_line_width()
            context.rel_move_to(0, dy)

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> tuple[float, float]:
        col = Column([x for (_, x) in list(self.grid_renderables)], self.pad)
        return col.measure(context)

    @property
    def depth(self) -> int:
        return 1 + max([r.depth for (_, r) in self.grid_renderables])

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._title)
        else:
            tree = tree.add(self._title)
        for _, rnd in self.grid_renderables:
            rnd.__rich_tree__(tree)
        return tree


def draw_wires(
    context: Context,
    support: tuple,
    column_width: float,
    parent_pad: Padding,
    renderable: Renderable,
    draw_qubit_number: bool = False,
) -> None:
    context.set_source_rgba(0, 0, 0, 1)
    (_x, _y) = context.get_current_point()
    (_, h) = STANDARD_BLOCK.measure(context)
    lw = STANDARD_BLOCK.line_width

    for nbit in support:
        (x, y) = context.get_current_point()

        (left_wire_point) = (x - parent_pad.left - lw, y + h / 2)

        if draw_qubit_number:
            context.move_to(*left_wire_point)
            context.set_font_size(STANDARD_TEXT.fontsize / 2)
            ext = context.text_extents(f"{nbit}")
            context.rel_move_to(0, ext.height * 1.5)
            context.show_text(f"{nbit}")
            context.stroke()

        context.move_to(*left_wire_point)
        context.rel_line_to(column_width + lw * 2, 0)
        context.stroke()
        # FIXME: should not use class attribute for line_width...
        context.move_to(x, y + h + Column.pad.gap + lw)

    context.move_to(_x, _y)
