from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

from cairo import Context

from qadence.draw.base import NestedRenderable, Padding, Renderable
from qadence.draw.box import STANDARD_BLOCK, Box
from qadence.draw.composite import Column
from qadence.draw.config import _TRANSPARENT
from qadence.types import TDrawColor


@dataclass
class Control(Renderable):
    radius: int = 5
    fill_color: TDrawColor = (0.87, 0.67, 0.31, 1.0)
    border_color: TDrawColor = (0.84, 0.59, 0.18, 1.0)

    def render(self, context: Context) -> None:
        filled_circle(context, self.radius, self.fill_color, self.border_color)

    def measure(self, context: Context) -> Tuple[float, float]:
        r = self.radius
        return (2 * r, 2 * r)

    @property
    def _title(self) -> str:
        return "Control"

    @property
    def depth(self) -> int:
        raise ValueError("Control renderable has no depth")


def control_circle(
    context: Context,
    r: float,
    fill_color: TDrawColor = (0.98, 0.93, 0.86, 1.0),
    border_color: TDrawColor = (0.84, 0.59, 0.18, 1.0),
) -> None:
    c = r * math.cos(math.pi / 4)
    (x, y) = context.get_current_point()
    (xc, yc) = x + r, y + r

    context.set_source_rgba(*fill_color)
    context.arc(xc, yc, r, 0, 2 * math.pi)
    context.fill()
    context.stroke()
    context.set_source_rgba(*border_color)
    context.arc(xc, yc, r, 0, 2 * math.pi)
    context.stroke()

    context.set_source_rgba(*border_color)
    context.move_to(xc - c, yc - c)
    context.rel_line_to(2 * c, 2 * c)
    context.stroke()
    context.move_to(xc + c, yc - c)
    context.rel_line_to(-2 * c, 2 * c)
    context.stroke()

    context.move_to(x, y)


@dataclass
class Target(Renderable):
    radius: int = 9
    fill_color: TDrawColor = (0.98, 0.93, 0.86, 1.0)
    border_color: TDrawColor = (0.84, 0.59, 0.18, 1.0)

    def render(self, context: Context) -> None:
        control_circle(context, self.radius, self.fill_color, self.border_color)

    def measure(self, context: Context) -> Tuple[float, float]:
        r = self.radius
        return (2 * r, 2 * r)

    @property
    def _title(self) -> str:
        return "Target"

    @property
    def depth(self) -> int:
        raise ValueError("Target renderable has no depth")


def filled_circle(
    context: Context,
    r: float,
    fill_color: TDrawColor,
    border_color: TDrawColor,
) -> None:
    (x, y) = context.get_current_point()
    (xc, yc) = x + r, y + r

    context.set_source_rgba(*fill_color)
    context.arc(xc, yc, r, 0, 2 * math.pi)
    context.fill()
    context.stroke()
    context.arc(xc, yc, r, 0, 2 * math.pi)
    context.set_source_rgba(*border_color)
    context.stroke()

    context.move_to(x, y)


@dataclass
class IconBox(NestedRenderable):
    renderable: Union[Control, Target]
    border_color: TDrawColor = _TRANSPARENT
    fill_color: TDrawColor = _TRANSPARENT
    line_width: float = 2.0

    def render(self, context: Context) -> None:
        (iw, ih) = self.renderable.measure(context)
        (bw, bh) = self.measure(context)
        dx = (bw - iw) / 2 - STANDARD_BLOCK.outer_pad.left
        dy = (bh - ih) / 2 - STANDARD_BLOCK.outer_pad.top

        pad = Padding(dx, dx, dy, dy)
        box = Box(self.renderable, pad, fill_color=self.fill_color, border_color=self.border_color)
        box.render(context)

    def measure(self, context: Context) -> Tuple[float, float]:
        (w, h) = STANDARD_BLOCK.measure(context)
        (rw, rh) = self.renderable.measure(context)
        return (max(w, rw), max(h, rh))


@dataclass
class ControlBox(Renderable):
    nqubits: int
    topicon: IconBox
    bottomicon: IconBox

    def render(self, context: Context) -> None:
        (x, y) = context.get_current_point()
        (bw, bh) = STANDARD_BLOCK.measure(context)
        (_, h) = self.measure(context)
        (cx, cy) = (x + bw / 2, y + bh / 2)
        (tx, ty) = (cx, cy + h - bh)

        # draw line
        context.move_to(cx, cy)
        context.set_source_rgba(*self.topicon.renderable.border_color)
        context.line_to(tx, ty)
        context.stroke()

        # draw icons
        context.move_to(x, y)
        self.topicon.render(context)
        context.rel_move_to(0, h - bh)
        self.bottomicon.render(context)

        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        box = Column(
            [STANDARD_BLOCK for _ in range(self.nqubits)],
            pad=Padding(0, 0, 0, 0),
            fill_color=(0, 0, 0, 0),
            border_color=(0, 0, 0, 0),
        )
        (w, h) = box.measure(context)
        bw = max(self.topicon.measure(context)[0], self.bottomicon.measure(context)[0])
        h -= STANDARD_BLOCK.outer_pad.top * (self.nqubits - 1)
        return (max(w, bw), h)

    @property
    def _title(self) -> str:
        return f"CNOTBox({self.topicon}, {self.bottomicon})"

    @property
    def depth(self) -> int:
        return 1


@dataclass
class SWAPBox(Renderable):
    control: int = 0
    target: int = 1

    def render(self, context: Context) -> None:
        (x, y) = context.get_current_point()
        (bw, bh) = STANDARD_BLOCK.measure(context)
        (_, h) = self.measure(context)
        (cx, cy) = (x, y + bh / 2)
        (tx, ty) = (cx, cy + h - bh)

        # remove wires that will be swapped
        context.set_source_rgba(*STANDARD_BLOCK.fill_color)
        context.set_line_width(STANDARD_BLOCK.line_width * 1.6)
        context.move_to(cx - STANDARD_BLOCK.line_width * 0.3, cy)
        context.line_to(cx + bw, cy)
        context.stroke()
        context.move_to(tx - STANDARD_BLOCK.line_width * 0.3, ty)
        context.line_to(tx + bw, ty)
        context.stroke()

        # draw lines
        context.set_source_rgba(0, 0, 0, 1)
        context.set_dash([STANDARD_BLOCK.line_width * 2])
        context.set_line_width(STANDARD_BLOCK.line_width)
        dy = STANDARD_BLOCK.line_width / 3
        context.move_to(cx, cy)
        context.line_to(tx + bw, ty + dy)
        context.stroke()
        context.move_to(tx, ty + dy)
        context.line_to(tx + bw, cy - dy)
        context.stroke()

        context.set_dash([])
        context.move_to(x, y)

    def measure(self, context: Context) -> Tuple[float, float]:
        n = abs(self.control - self.target)
        box = Column(
            [STANDARD_BLOCK for _ in range(n + 1)],
            pad=Padding(0, 0, 0, 0),
            fill_color=(0, 0, 0, 0),
            border_color=(0, 0, 0, 0),
        )
        (w, h) = box.measure(context)
        h -= STANDARD_BLOCK.outer_pad.top * n
        return (w, h)

    @property
    def _title(self) -> str:
        return f"SWAPBox({self.control}, {self.target})"

    @property
    def depth(self) -> int:
        return 1
