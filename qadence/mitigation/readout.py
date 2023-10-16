from __future__ import annotations

from torch import rand
from collections import Counter

def corrupt(bitflip_proba: float, counters: list[Counter]) -> None:

    def flip_bit(bitstring: str, n: int) -> str:
        """Flip a bit at position n in bitstring."""
        return "{:02b}".format(int("10", 2) ^ 1 << n)


    # corrupted_bitstr: dict = dict()
    corrupted_counters = []
    for counter in counters:
        print(counter)
        corrupted_counter: Counter = Counter()
        for bitstring, count in counter.items():
            rands = rand(len(bitstring))
            
            for n in range(len(bitstring)):
                corrupted_bitstring = ""
                if rands[n] < bitflip_proba:
                    print(rands[n], bitstring, n, count)
                    corrupted_bitstring = flip_bit(bitstring, n)
                    print(f"corr str {corrupted_bitstring}")
                    # corrupted_bitstr = {flip_bit(bitstring, n): count}
                    # counter[flip_bit(bitstring, n)] = count
                    # counter.pop(bitstring)
            corrupted_counter[corrupted_bitstring] = count
        corrupted_counters.append(corrupted_counter)
    return corrupted_counters
            

def mitigate(bitflip_proba: float, counters: list[Counter]) -> None:
    corrupted_counters = corrupt(sample)
    mitigate = corrupt(corrupted_counters)
    return mitigate
