"""One owned pinned host packet and one asynchronous int32 metadata upload."""
from .route_table import RouteTable


def metadata_tables(plan, cold):
    import numpy as np
    counts = [plan.decode_counts[e] + plan.hot_counts[e] for e in plan.experts]
    tables = dict(hot_sources=plan.sources, hot_counts=counts+[0]*(288-len(counts)), hot_experts=plan.experts)
    if cold is not None:
        tokens = [s[3] for s in plan.sources[plan.decode_routes:]]
        rows = sorted(set(tokens))
        local = {token: i for i, token in enumerate(rows)}
        # Invert the immutable expert-major plan once. A token CTA loads the
        # activation once and writes its live routes; -1 excludes moved hot
        # routes without inventing a duplicate expert contribution.
        sources = cold.sources.array() if isinstance(cold.sources, RouteTable) else np.asarray(cold.sources, dtype=np.int32).reshape(-1, 4)
        destinations = np.full((len(plan.prefill), 8), -1, dtype=np.int32)
        destinations[sources[:, 2], sources[:, 3]] = sources[:, 1]
        tables.update(cold_rows=destinations, cold_counts=cold.counts, cold_bases=cold.tile_bases,
            cold_tasks=cold.task_expert, cold_valid=cold.task_valid_rows,
            hot_rows=rows, hot_dest=[local[t] for t in tokens])
    return {k: (v.array() if isinstance(v, RouteTable) else np.asarray(v, dtype=np.int32)).reshape(-1)
            for k, v in tables.items()}


class MixedMetadata:
    def __init__(self, plan, cold, device):
        import torch
        tables = metadata_tables(plan, cold)
        device = torch.device(device)
        # CuTe source tensors require 16-byte alignment even when the previous
        # descriptor has an odd expert/task count. Pad spans, not route rows.
        self._host = torch.empty(sum((v.size + 3) & ~3 for v in tables.values()), dtype=torch.int32,
                                 pin_memory=device.type == 'cuda')
        host, spans, offset = self._host.numpy(), {}, 0
        for name, values in tables.items():
            stop = offset + values.size
            host[offset:stop] = values
            spans[name] = (offset, stop)
            offset = (stop + 3) & ~3
        # Keep the pinned source alive with the invocation, including all
        # failure/cancellation paths. No shared staging buffer can be reused
        # while this owner's readers or consumers are outstanding.
        self._device = self._host.to(device=device, non_blocking=True)
        self._spans = spans

    def __getitem__(self, name):
        start, stop = self._spans[name]
        return self._device[start:stop]
