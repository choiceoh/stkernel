"""Exact cold-route layout for M1b; no device, histogram atomics or scheduler.

One task is one M128 tile and all four intermediate slices. The immutable
windows bound work per launch; they are not a preemption guarantee.
"""
from dataclasses import dataclass

from .mixed_experts import MixedExpertPlan


@dataclass(frozen=True)
class ColdExpertPlan:
    # (expert ID, padded physical row, original prefill row, top-k slot)
    sources: tuple
    counts: tuple
    tile_bases: tuple
    task_expert: tuple
    task_valid_rows: tuple
    windows: tuple
    task_quota: int

    @property
    def physical_rows(self):
        return self.tile_bases[-1] * 128


def plan_cold(mixed, *, task_quota=48):
    if not isinstance(mixed, MixedExpertPlan):
        raise ValueError('cold work requires an owned mixed plan')
    if type(task_quota) is not int or not 1 <= task_quota <= 128:
        raise ValueError('cold task quota must be in 1..128')
    routes = [[] for _ in range(288)]
    for token, slot in mixed.cold_routes:
        routes[mixed.prefill[token][slot]].append((token, slot))
    sources, bases, tasks, valid = [], [0], [], []
    for expert, pairs in enumerate(routes):
        start = bases[-1]
        tiles = (len(pairs) + 127) // 128
        for offset, (token, slot) in enumerate(pairs):
            sources.append((expert, start * 128 + offset, token, slot))
        for tile in range(tiles):
            # Same ABI as publish_uniform_deferred_tasks(gate=4, chunk=4).
            tasks.append(expert | ((start + tile) << 16))
            valid.append(min(128, len(pairs) - tile * 128) | (4 << 20))
        bases.append(start + tiles)
    windows = tuple((start, min(start + task_quota, len(tasks)))
                    for start in range(0, len(tasks), task_quota))
    return ColdExpertPlan(tuple(sources), tuple(map(len, routes)), tuple(bases),
                          tuple(tasks), tuple(valid), windows, task_quota)
