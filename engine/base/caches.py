"""What a model caches per sequence and per token, and what a budget buys (framework).

A Cache is one buffer's cost law; total_bytes and max_seq invert a budget
into context x concurrency. Profiles and modules contribute the Caches.
"""
from __future__ import annotations

from dataclasses import dataclass


GIB = 1 << 30


@dataclass(frozen=True)
class Cache:
    name: str
    blocks: int          # how many layers carry one
    per_batch_per_token: float   # bytes per sequence per token of context
    per_batch: float             # bytes per sequence, independent of context
    per_token: float             # bytes per token of context, independent of batch
    note: str


def total_bytes(cs: "list[Cache]", batch: int, seq: int) -> float:
    return sum(c.per_batch_per_token * batch * seq + c.per_batch * batch
               + c.per_token * seq for c in cs)


def max_seq(cs: "list[Cache]", budget_gib: float, batch: int) -> int:
    """Longest context that fits, at this concurrency."""
    per_token = sum(c.per_batch_per_token * batch + c.per_token for c in cs)
    fixed = sum(c.per_batch * batch for c in cs)
    if per_token <= 0:
        return 0
    return int((budget_gib * GIB - fixed) / per_token)
