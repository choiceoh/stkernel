"""Plan hot and cold work from one stable prefill index and int32 construction.

All inputs are fresh CPU snapshots. The index is local to this invocation;
only immutable source/task descriptors escape, never a histogram cache.
"""
from .mixed_experts import MixedExpertPlan, plan_experts_packed, validate_packed_routes
from .mixed_completion import ColdExpertPlan
from .route_table import RouteTable


def prepare_routes(decode, prefill, *, identity, hot_route_quota=128, cold_task_quota=None):
    if cold_task_quota is None:
        return plan_experts_packed(decode, prefill, identity=identity, hot_route_quota=hot_route_quota), None
    from .mixed_route_native import prepare_native
    return prepare_native(decode, prefill, identity=identity,
                          hot_route_quota=hot_route_quota, cold_task_quota=cold_task_quota)


def prepare_routes_numpy(decode, prefill, *, identity, hot_route_quota=128, cold_task_quota=None):
    if cold_task_quota is None:
        return plan_experts_packed(decode, prefill, identity=identity, hot_route_quota=hot_route_quota), None
    if type(cold_task_quota) is not int or not 1 <= cold_task_quota <= 128:
        raise ValueError('cold task quota must be in 1..128')
    import numpy as np
    validate_packed_routes(decode, prefill, identity, hot_route_quota)
    decode = tuple(map(tuple, decode.tolist()))
    tile = 16 if len(decode) <= 8 else 32
    flat = prefill.ravel()
    counts = np.bincount(flat, minlength=288).astype(np.int32)
    order = np.argsort(flat.astype(np.uint16), kind='stable').astype(np.int32)
    starts = np.cumsum(counts, dtype=np.int32) - counts
    d, experts = [0] * 288, []
    for row in decode:
        for expert in row:
            if not d[expert]:
                experts.append(expert)
            d[expert] += 1
    candidates = []
    for expert in experts:
        spare = ((d[expert] + tile - 1) // tile) * tile - d[expert]
        tail = (int(counts[expert]) - 1) % 128 + 1 if counts[expert] else 0
        if 0 < tail <= spare:
            candidates.append((tail, expert))
    hot, remaining = [0] * 288, hot_route_quota
    for tail, expert in sorted(candidates):
        if tail <= remaining:
            hot[expert] = tail
            remaining -= tail
    local, cursor, sources = {e: i for i, e in enumerate(experts)}, [0] * 288, []
    for row, ids in enumerate(decode):
        for slot, expert in enumerate(ids):
            sources.append((local[expert], cursor[expert], 0, row, slot))
            cursor[expert] += 1
    keep = np.ones(len(flat), dtype=bool)
    for expert in experts:
        chosen = order[starts[expert]:starts[expert] + hot[expert]]
        for index in chosen.tolist():
            sources.append((local[expert], cursor[expert], 1, index >> 3, index & 7))
            cursor[expert] += 1
        keep[chosen] = False
    cold_order = order[keep[order]]
    original = np.flatnonzero(keep).astype(np.int32)
    cold_routes = np.empty((len(original), 2), dtype=np.int32)
    cold_routes[:, 0], cold_routes[:, 1] = original >> 3, original & 7
    plan = MixedExpertPlan(identity, decode, RouteTable.pack(prefill), tile, tuple(sources),
        tuple(experts), tuple(d), tuple(hot), RouteTable.pack(cold_routes), hot_route_quota)
    # No second sort, per-route advanced lookup, int64 offset plane, or stacked
    # int64 source table. The maximum physical row is < 300k on this geometry.
    cold_counts = counts - np.asarray(hot, dtype=np.int32)
    tiles = (cold_counts + 127) // 128
    bases = np.empty(289, dtype=np.int32)
    bases[0], bases[1:] = 0, np.cumsum(tiles, dtype=np.int32)
    cold_starts = np.cumsum(cold_counts, dtype=np.int32) - cold_counts
    cold_sources = np.empty((len(cold_order), 4), dtype=np.int32)
    cold_sources[:, 0] = np.repeat(np.arange(288, dtype=np.int32), cold_counts)
    cold_sources[:, 1] = (np.arange(len(cold_order), dtype=np.int32)
        + np.repeat(bases[:-1] * 128 - cold_starts, cold_counts))
    cold_sources[:, 2], cold_sources[:, 3] = cold_order >> 3, cold_order & 7
    tasks, valid = [], []
    for expert in range(288):
        for offset in range(int(tiles[expert])):
            tasks.append(expert | ((int(bases[expert]) + offset) << 16))
            valid.append(min(128, int(cold_counts[expert]) - offset * 128) | (4 << 20))
    windows = tuple((start, min(start + cold_task_quota, len(tasks)))
        for start in range(0, len(tasks), cold_task_quota))
    cold = ColdExpertPlan(RouteTable.pack(cold_sources), tuple(cold_counts.tolist()),
        tuple(bases.tolist()), tuple(tasks), tuple(valid), windows, cold_task_quota)
    return plan, cold
