from __future__ import annotations

from copy import deepcopy

from torch import Tensor

from qadence import AbstractBlock, add, block_to_tensor, kron


class TDGenerator:
    def __init__(self, schedule: list[dict]) -> None:
        self._schedule = schedule

    @property
    def duration(self) -> int:
        return self._schedule[-1]["t_end"]  # type: ignore [no-any-return]

    @property
    def schedule(self) -> list[dict]:
        return self._schedule

    @classmethod
    def from_block(cls, duration: int, block: AbstractBlock) -> TDGenerator:
        schedule = [
            {
                "t_start": 0,
                "t_end": duration,
                "generators": [block],
                "fn": None,
                "qubit_support": block.qubit_support,
            }
        ]
        return cls(schedule)

    def __call__(self, t: Tensor) -> Tensor:
        # find appropriate time slot
        for slot in self.schedule:
            if t * 1000 < slot["t_end"]:
                blocks = slot["generators"]
                fn = slot["fn"]
                break

        # construct generator block
        if fn == "add":
            block = add(*blocks)
        elif fn == "kron":
            block = kron(*blocks)  # type: ignore [assignment]
        else:
            block = blocks[0]

        # get matrix representation of the generator block
        mat = block_to_tensor(
            block=block, values={"t": t}, qubit_support=self.schedule[0]["qubit_support"]
        )

        return mat

    def __add__(self, other: TDGenerator) -> TDGenerator:  # type: ignore [return]
        print("add")

    def __matmul__(self, other: TDGenerator) -> TDGenerator:  # type: ignore [return]
        print("kron")

    def __mul__(self, other: TDGenerator) -> TDGenerator:
        # merge schedules when chain operation is called
        new_schedule = deepcopy(self.schedule)
        qubit_support = self.schedule[-1]["qubit_support"]
        t_end = self.schedule[-1]["t_end"]
        for slot in other.schedule:
            new_slot = deepcopy(slot)
            new_slot["t_start"] = t_end
            new_slot["t_end"] = new_slot["t_start"] + slot["t_end"]
            qubit_support = tuple(sorted(set(new_slot["qubit_support"]).union(set(qubit_support))))
            t_end = new_slot["t_end"]
            new_schedule.append(new_slot)

        for slot in new_schedule:
            slot["qubit_support"] = qubit_support

        return TDGenerator(new_schedule)
