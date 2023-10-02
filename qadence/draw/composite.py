from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from cairo import Context
from rich.columns import Columns
from rich.console import Console, Group

from qadence.draw.base import CompositeRenderable, Padding, Renderable
from qadence.draw.box import roundrect


@dataclass(repr=False)
class Column(CompositeRenderable):
    renderables: list[Renderable]
    pad: Padding = Padding(8, 8, 0, 0, 0)
    fill_color: Tuple[float, float, float, float] = (0.88, 0.90, 0.95, 1.0)
    border_color: Tuple[float, float, float, float] = (0.88, 0.90, 0.95, 1.0)
    line_width: float = 2.0

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()

        (column_width, column_height) = self.measure(context)

        # draw bounding box
        context.set_line_width(self.line_width)
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

        for rnd in self.renderables:
            (w, h) = rnd.measure(context)

            # draw in center
            dx = (column_width - self.pad.left - self.pad.right - w) / 2
            context.rel_move_to(dx, 0)
            rnd.render(context)
            context.rel_move_to(-dx, 0)

            # move down
            dy = h + self.pad.gap
            # dy = h + rnd.pad.bottom / 2
            dy += context.get_line_width()
            context.rel_move_to(0, dy)

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        # FIXME: where to mutate line_width for correct measurements?

        # initialize column width/height
        column_width = self.pad.left + self.pad.right
        column_height = self.pad.top + self.pad.bottom

        for rnd in self.renderables:
            (w, h) = rnd.measure(context)
            column_height += h + self.pad.gap
            column_height += context.get_line_width()
            column_width = max(column_width, w + self.pad.left + self.pad.right)

        # remove trailing pad
        column_height -= self.pad.gap
        column_height -= context.get_line_width()
        return (column_width, column_height)

    def __ascii__(self, console: Console) -> Group:
        return Group(*[r.__ascii__(console) for r in self.renderables])


@dataclass(repr=False)
class Row(CompositeRenderable):
    renderables: list
    pad: Padding = Padding(8, 8, 0, 0, 0)
    fill_color: Tuple[float, float, float, float] = (0.88, 0.90, 0.95, 1.0)
    border_color: Tuple[float, float, float, float] = (0.88, 0.90, 0.95, 1.0)

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()

        (row_width, row_height) = self.measure(context)

        # draw bounding box
        roundrect(
            context,
            x,
            y,
            row_width,
            row_height,
            self.pad.min(),
            self.fill_color,
            self.border_color,
        )
        context.stroke()

        # return to original point
        context.move_to(x, y)

        # offset padding to start drawing sub-boxes
        context.rel_move_to(self.pad.left, self.pad.top)

        for rnd in self.renderables:
            (w, _) = rnd.measure(context)
            dx = w + self.pad.gap
            rnd.render(context)
            context.rel_move_to(dx, 0)

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        # initialize row width/height
        row_width = self.pad.left + self.pad.right
        row_height = self.pad.top + self.pad.bottom

        for rnd in self.renderables:
            (w, h) = rnd.measure(context)
            row_width += w + self.pad.gap
            row_height = max(row_height, h + self.pad.top + self.pad.bottom)

        # remove trailing pad
        row_width -= self.pad.gap

        return (row_width, row_height)

    def __ascii__(self, console: Console) -> Columns:
        return Columns([r.__ascii__(console) for r in self.renderables])
