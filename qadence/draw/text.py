from __future__ import annotations

from dataclasses import dataclass

from cairo import Context
from rich.console import Console

from qadence.draw.base import Renderable
from qadence.draw.config import _MEASUREMENT_CHAR
from qadence.types import TDrawColor


@dataclass(repr=False)
class Text(Renderable):
    text: str
    fontsize: int = 15
    font_color: TDrawColor = (0.0, 0.0, 0.0, 1.0)

    def render(self, context: Context) -> None:
        # assume that this is the top left of the rectangle
        (x, y) = context.get_current_point()

        context.set_font_size(self.fontsize)
        ext = context.text_extents(self.text)

        # draw text
        context.move_to(x, y + ext.height)
        context.set_source_rgba(*self.font_color)
        context.show_text(self.text)
        context.stroke()

        # return to original point
        context.move_to(x, y)

    def measure(self, context: Context) -> tuple[float, float]:
        context.set_font_size(self.fontsize)
        ext = context.text_extents(self.text)
        return (max(ext.width, ext.height), ext.height)

    @property
    def _title(self) -> str:
        return self.text

    @property
    def depth(self) -> int:
        return 1

    def __ascii_measure__(self, console: Console) -> tuple[int, int]:
        m = console.measure(self.text)
        return (m.maximum, 1)

    def __ascii__(self, console: Console) -> str:
        return self.text


STANDARD_TEXT = Text(_MEASUREMENT_CHAR)
