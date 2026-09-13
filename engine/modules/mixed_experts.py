"""M0 admission for prepared GLM TP4 expert inputs; no serving scheduler.

The actual decode body uses M16 for 1..8 rows and M32 for 9..32.
Only a prefill M128 tail which fits in an existing decode tile is moved.
Counts are work accounting, never a latency or memory-traffic prediction.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertInvocation:
    layer: int
    epoch: int
    slot_generation: int
    source_generation: int
    layout: str = 'glm53-tp4-sf6-v1'

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in
               (self.layer, self.epoch, self.slot_generation, self.source_generation)):
            raise ValueError('invocation identity must use nonnegative integers')
        if self.layer >= 64 or self.layout != 'glm53-tp4-sf6-v1':
            raise ValueError('mixed experts require the fixed GLM TP4 SF6 layout')


def _routes(value, maximum, name):
    rows = tuple(tuple(row) for row in value)
    if not 1 <= len(rows) <= maximum:
        raise ValueError(f'{name} rows must be in 1..{maximum}')
    if any(len(row) != 8 or len(set(row)) != 8 or
           any(type(e) is not int or not 0 <= e < 288 for e in row) for row in rows):
        raise ValueError(f'{name} must retain eight distinct in-range expert routes per row')
    return rows


@dataclass(frozen=True)
class MixedExpertPlan:
    identity: ExpertInvocation
    decode: tuple
    prefill: tuple
    tile_m: int
    # Rows of (local expert, expert row, source kind, original row, route slot).
    # Output route indices equal descriptor indices: decode first, then hot prefill.
    sources: tuple
    experts: tuple
    decode_counts: tuple
    hot_counts: tuple
    cold_routes: tuple
    quota: int

    @property
    def decode_routes(self):
        return len(self.decode) * 8

    @property
    def hot_routes(self):
        return len(self.sources) - self.decode_routes

    def work(self):
        prefill_counts = [0] * 288
        for row in self.prefill:
            for expert in row:
                prefill_counts[expert] += 1
        before = sum((p + 127) // 128 for p in prefill_counts)
        after = sum((p - h + 127) // 128 for p, h in zip(prefill_counts, self.hot_counts))
        decode_before = sum((d + self.tile_m - 1) // self.tile_m for d in self.decode_counts)
        decode_after = sum((d + h + self.tile_m - 1) // self.tile_m
                           for d, h in zip(self.decode_counts, self.hot_counts))
        return dict(decode_tile_m=self.tile_m, decode_tiles=decode_before,
                    mixed_tiles=decode_after, prefill_tiles=before,
                    cold_tiles=after, removed_prefill_tiles=before-after,
                    hot_routes=self.hot_routes, cold_routes=len(self.cold_routes),
                    quota=self.quota, performance_proven=False)


def plan_experts(decode, prefill, *, identity, hot_route_quota=128):
    if not isinstance(identity, ExpertInvocation):
        raise ValueError('an explicit source/layer/generation identity is required')
    if type(hot_route_quota) is not int or not 0 <= hot_route_quota <= 128:
        raise ValueError('hot route quota must be in 0..128')
    decode, prefill = _routes(decode, 32, 'decode'), _routes(prefill, 32768, 'prefill')
    tile = 16 if len(decode) <= 8 else 32
    d, p = [[] for _ in range(288)], [[] for _ in range(288)]
    experts = []
    for row, ids in enumerate(decode):
        for slot, expert in enumerate(ids):
            if not d[expert]:
                experts.append(expert)
            d[expert].append((row, slot))
    for row, ids in enumerate(prefill):
        for slot, expert in enumerate(ids):
            p[expert].append((row, slot))
    # Each admitted tail removes exactly one M128 tile. Prefer the smallest
    # tail, then expert ID: most tiles removed within the finite route quota.
    candidates = []
    for expert in experts:
        spare = ((len(d[expert]) + tile - 1) // tile) * tile - len(d[expert])
        tail = (len(p[expert]) - 1) % 128 + 1 if p[expert] else 0
        if 0 < tail <= spare:
            candidates.append((tail, expert))
    hot = [0] * 288
    remaining = hot_route_quota
    for tail, expert in sorted(candidates):
        if tail <= remaining:
            hot[expert] = tail
            remaining -= tail
    local = {e: i for i, e in enumerate(experts)}
    cursor = [0] * 288
    sources = []
    for row, ids in enumerate(decode):
        for slot, expert in enumerate(ids):
            sources.append((local[expert], cursor[expert], 0, row, slot))
            cursor[expert] += 1
    selected = set()
    for expert in experts:
        for row, slot in p[expert][:hot[expert]]:
            sources.append((local[expert], cursor[expert], 1, row, slot))
            cursor[expert] += 1
            selected.add((row, slot))
    cold = tuple((row, slot) for row in range(len(prefill)) for slot in range(8)
                 if (row, slot) not in selected)
    return MixedExpertPlan(identity, decode, prefill, tile, tuple(sources), tuple(experts),
                           tuple(map(len, d)), tuple(hot), cold, hot_route_quota)
