"""The batch shapes the kernels support, as data (framework).

D2: the scheduler builds batches only out of shapes the kernels accept. A
Constraint carries its source and what breaks when violated; profiles list
theirs, modules may contribute theirs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constraint:
    name: str
    value: object
    source: str
    bites: str          # what goes wrong when a shape violates it


def chunk_for(align: int, token_budget: int, draft_slots: int = 0) -> int:
    """The largest legal prefill chunk inside a token budget, given the alignment.

    GLM's floor((MAX_BATCHED - K) / 2304) * 2304, written once so that a chunk
    is never three side effects deep again -- if a chunk is not what you
    expected, print this.
    """
    usable = token_budget - draft_slots
    if usable < align:
        return 0
    return (usable // align) * align
