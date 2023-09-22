from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from cairo import Context
from rich.console import Console, RenderableType
from rich.panel import Panel

from qadence.draw.base import NestedRenderable, Padding, Renderable
from qadence.draw.config import _MEASUREMENT_CHAR, _TRANSPARENT
from qadence.draw.text import STANDARD_TEXT, Text
from qadence.types import TDrawColor


@dataclass(repr=False)
class Box(NestedRenderable):
    renderable: Renderable
    pad: Padding = Padding(10, 10, 10, 10)
    radius: float = 5.0
    fill_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    border_color: TDrawColor = (0.34, 0.45, 0.67, 1.0)
    line_width: float = 2.0
    outer_pad: Padding = Padding(10, 10, 5, 5)

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()
        (w, h) = self._measure_inner(context)

        # draw rectangle (with outer_pad offset)
        context.set_line_width(self.line_width)
        by = y + self.outer_pad.top
        bx = x + self.outer_pad.left
        roundrect(context, bx, by, w, h, self.radius, self.fill_color, self.border_color)
        context.stroke()

        # draw content
        context.move_to(bx + self.pad.left, by + self.pad.top)
        self.renderable.render(context)

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        return self._measure_outer(context)

    def _measure_inner(self, context: Context) -> Tuple[float, float]:
        context.set_line_width(self.line_width)
        (rw, rh) = self.renderable.measure(context)
        sh = STANDARD_TEXT.measure(context)[1]
        rh = max(rh, sh)
        w = rw + self.pad.left + self.pad.right
        h = rh + self.pad.top + self.pad.bottom
        return (w, h)

    def _measure_outer(self, context: Context) -> Tuple[float, float]:
        (w, h) = self._measure_inner(context)
        w += self.outer_pad.left + self.outer_pad.right
        h += self.outer_pad.top + self.outer_pad.bottom
        return (w, h)

    def __ascii__(self, console: Console) -> RenderableType:
        return Panel(self.renderable.__ascii__(console), expand=False)


def roundrect(
    context: Context,
    x: float,
    y: float,
    width: float,
    height: float,
    r: float,
    fill_color: TDrawColor = None,
    border_color: TDrawColor = None,
) -> None:
    point = context.get_current_point()
    context.move_to(x, y + r)

    # draw rect
    context.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    context.arc(x + width - r, y + r, r, 3 * math.pi / 2, 0)
    context.arc(x + width - r, y + height - r, r, 0, math.pi / 2)
    context.arc(x + r, y + height - r, r, math.pi / 2, math.pi)
    context.close_path()

    # fill it
    if fill_color is not None:
        context.set_source_rgba(*fill_color)
        context.fill_preserve()
    if border_color is not None:
        context.set_source_rgba(*border_color)
        context.stroke()
    context.set_source_rgba(0, 0, 0, 1)

    context.move_to(*point)


@dataclass(repr=False)
class TagBox(NestedRenderable):
    renderable: Renderable
    text: Text
    radius: int = 5
    fill_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    border_color: TDrawColor = (0.34, 0.45, 0.67, 1.0)

    @property
    def _title(self) -> str:
        return f"{type(self).__name__}(text={self.text.text})"

    @property
    def box(self) -> Box:
        return Box(
            self.renderable,
            radius=self.radius,
            fill_color=self.fill_color,
            border_color=self.border_color,
            outer_pad=Padding(0, 0, 0, 0),
        )

    @property
    def tag(self) -> Box:
        pad = Padding(10, 10, 5, 5)
        self.text.fontsize = 10
        return Box(self.text, pad, 0.1, self.fill_color, self.fill_color)

    def render(self, context: Context) -> None:
        (tw, _) = self.tag.measure(context)
        (sh, f) = self.tag_shift(context)
        (bw, _) = self.box.measure(context)

        (x, y) = context.get_current_point()

        # draw content
        dx = max(0, tw - self.box.measure(context)[0]) / 2
        context.move_to(x + dx, y + f * sh)
        self.box.render(context)

        # move to tag position & render
        dx = (bw - tw) / 2
        context.rel_move_to(dx, -f * sh)
        self.tag.render(context)

        # move back to starting point
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        (sh, f) = self.tag_shift(context)
        (bw, bh) = self.box.measure(context)
        (tw, _) = self.tag.measure(context)
        w = max(bw, tw)
        return (w, bh + f * sh)

    def tag_shift(self, context: Context) -> Tuple[float, float]:
        (_, tag_height) = self.tag.measure(context)
        factor = 0.5
        return tag_height, factor

    def __ascii__(self, console: Console) -> Panel:
        return Panel(self.renderable.__ascii__(console), title=self.text.text, expand=False)


@dataclass(repr=False)
class MultiWireBox(NestedRenderable):
    renderable: Renderable
    wires: tuple
    pad: Padding = Padding(10, 10, 10, 10)
    radius: int = 5
    fill_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    border_color: TDrawColor = (0.34, 0.45, 0.67, 1.0)
    line_width: float = 2.0

    @property
    def box(self) -> Renderable:
        from qadence.draw.composite import Column

        return Column(
            [STANDARD_BLOCK for _ in self.wires],
            pad=Padding(0, 0, 0, 0, 0),
            fill_color=(0, 0, 0, 0),
            border_color=(0, 0, 0, 0),
        )

    def render(self, context: Context) -> None:
        (x, y) = context.get_current_point()
        (_, h) = self.measure(context)
        (_, bh) = self.renderable.measure(context)

        # draw wires
        # self.box.render(context)

        # draw actual text box
        dy = (h - bh) / 2 - STANDARD_BLOCK.outer_pad.bottom
        pad = Padding(self.pad.left, self.pad.right, dy, dy)
        box = Box(self.renderable, pad=pad)
        box.render(context)

        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        box = self.box
        _, h = box.measure(context)
        w = self.renderable.measure(context)[0] + self.pad.left + self.pad.right
        w += STANDARD_BLOCK.outer_pad.left + STANDARD_BLOCK.outer_pad.right
        return (w, h)

    @property
    def _title(self) -> str:
        return f"MultiWireBox(n_wires={len(self.wires)})"


@dataclass
class Rect(Renderable):
    size: Tuple[int, int] = (100, 50)
    radius: int = 5
    fill_color: TDrawColor = (0.88, 0.90, 0.95, 1.0)
    border_color: TDrawColor = (0.34, 0.45, 0.67, 1.0)

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()

        # draw rectangle
        roundrect(context, x, y, *self.size, self.radius, self.fill_color, self.border_color)
        context.stroke()

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        return self.size


STANDARD_BLOCK = Box(Text(_MEASUREMENT_CHAR))
IDENTITY_BOX = Box(
    Text(_MEASUREMENT_CHAR, font_color=_TRANSPARENT),
    fill_color=_TRANSPARENT,
    border_color=_TRANSPARENT,
)
