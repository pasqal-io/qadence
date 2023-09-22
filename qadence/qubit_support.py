from __future__ import annotations

from typing import Any, Union

from qadence.types import QubitSupportType


def _is_valid_support(t: Any) -> bool:
    return isinstance(t, tuple) and all(i >= 0 for i in t)


class QubitSupport(tuple):
    def __new__(cls, *support: Union[QubitSupportType, str, int, tuple]) -> QubitSupport:
        if len(support) == 1:
            if isinstance(support[0], tuple):
                return QubitSupport(*support[0])
            if support[0] == "global":
                support = (QubitSupportType.GLOBAL,)
                valid = True
            elif support[0] >= 0:  # type: ignore[operator]
                valid = True
            else:
                valid = False
        else:
            valid = _is_valid_support(support)

        if not valid:
            raise ValueError(
                "QubitSupport can be a tuple of ints or 'global'. For example:ℕn"
                "QubitSupport(1,2,3) or QubitSupport('global')\n"
                f"Found: {support}"
            )
        return super(QubitSupport, cls).__new__(cls, support)  # type: ignore[arg-type]

    def __add__(self, other: Any) -> QubitSupport:
        if not isinstance(other, tuple):
            raise ValueError(f"Cannot add type '{type(other)}' to QubitSupport.")
        if self == other:
            return self
        elif self == ("global",):
            return QubitSupport(*range(max(other) + 1)) if len(other) else QubitSupport("global")
        elif other == ("global",):
            return QubitSupport(*range(max(self) + 1)) if len(self) else QubitSupport("global")
        else:
            return QubitSupport(tuple({*self, *other}))

    def __radd__(self, other: Any) -> QubitSupport:
        return self.__add__(other)

    @property
    def is_global(self) -> bool:
        return self == ("global",)

    def is_disjoint(self, other: Any) -> bool:
        oth = QubitSupport(other)
        if self.is_global or oth.is_global:
            return False
        else:
            selfsup = set(self)
            othersup = set(oth)
            return selfsup.isdisjoint(othersup)
