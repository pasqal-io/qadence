from __future__ import annotations

from bisect import bisect
from copy import deepcopy

from torch import Tensor

from qadence import AbstractBlock, add, block_to_tensor, kron


class TDGenerator:
    def __init__(self, schedule: dict[int, dict]) -> None:
        self._schedule = schedule

    @property
    def duration(self) -> int:
        return self._schedule[-1]["t_end"]  # type: ignore [no-any-return]

    @property
    def schedule(self) -> dict[int, dict]:
        return self._schedule

    @classmethod
    def from_block(cls, duration: int, block: AbstractBlock) -> TDGenerator:
        schedule = {
            duration: {
                "t_start": 0,
                "t_end": duration,
                "generator": block,
                "qubit_support": block.qubit_support,
            }
        }
        return cls(schedule)

    def _create_new_schedule(self, op: str, other: TDGenerator) -> dict[int, dict]:
        if op in ["add", "kron"]:
            # create new schedule with slot times from both TDGenerator objects
            new_schedule = {}
            new_slot_times = [0] + sorted(
                set(self.schedule.keys()).union(set(other.schedule.keys()))
            )
            self_slot_times = list(self.schedule.keys())
            other_slot_times = list(other.schedule.keys())
            qubit_support = tuple()  # type: ignore [var-annotated]
            for i in range(1, len(new_slot_times)):
                t_start = new_slot_times[i - 1]
                t_end = new_slot_times[i]
                new_slot = {"t_start": t_start, "t_end": t_end}
                new_schedule[t_end] = new_slot

                # construct new block in new slot
                self_idx = bisect(self_slot_times, t_start)
                other_idx = bisect(other_slot_times, t_start)
                blocks_to_add = []
                if self_idx < len(self_slot_times):
                    self_t_slot = self_slot_times[self_idx]
                    blocks_to_add.append(self.schedule[self_t_slot]["generator"])

                if other_idx < len(other_slot_times):
                    other_t_slot = other_slot_times[other_idx]
                    blocks_to_add.append(other.schedule[other_t_slot]["generator"])

                new_slot["generator"] = add(*blocks_to_add) if op == "add" else kron(*blocks_to_add)  # type: ignore [assignment]

                # calculate qubit support of current slot
                qubit_support = tuple(
                    sorted(set(new_slot["generator"].qubit_support).union(set(qubit_support)))  # type: ignore [attr-defined]
                )
                new_slot["qubit_support"] = qubit_support  # type: ignore [assignment]

        elif op == "chain":
            # merge schedules when chain operation is called
            new_schedule = deepcopy(self.schedule)
            t_end = list(self.schedule.keys())[-1]
            qubit_support = self.schedule[t_end]["qubit_support"]
            for slot in other.schedule.values():
                new_slot = deepcopy(slot)
                new_slot["t_start"] = t_end
                new_slot["t_end"] = new_slot["t_start"] + slot["t_end"]
                qubit_support = tuple(
                    sorted(set(new_slot["qubit_support"]).union(set(qubit_support)))  # type: ignore [call-overload]
                )
                t_end = new_slot["t_end"]
                new_schedule[new_slot["t_end"]] = new_slot

        # make sure each new slot has full qubit support
        for slot in new_schedule.values():
            slot["qubit_support"] = qubit_support  # type: ignore [assignment]

        return new_schedule

    def __call__(self, t: Tensor) -> Tensor:
        # find appropriate time slot
        slot_times = list(self.schedule.keys())
        t_slot = slot_times[bisect(slot_times, t * 1000)]
        block = self.schedule[t_slot]["generator"]

        # get matrix representation of the generator block
        mat = block_to_tensor(
            block=block, values={"t": t}, qubit_support=self.schedule[t_slot]["qubit_support"]
        )

        return mat

    def __add__(self, other: TDGenerator) -> TDGenerator:
        new_schedule = self._create_new_schedule("add", other)
        return TDGenerator(new_schedule)

    def __matmul__(self, other: TDGenerator) -> TDGenerator:
        new_schedule = self._create_new_schedule("kron", other)
        return TDGenerator(new_schedule)

    def __mul__(self, other: TDGenerator) -> TDGenerator:
        new_schedule = self._create_new_schedule("chain", other)
        return TDGenerator(new_schedule)
