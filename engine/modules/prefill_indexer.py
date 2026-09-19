"""Partition replicated indexer queries, retaining every cache-writing row.

Only pool IDs cross ranks. Each rank finalizes them against its own page map.
Queries whose complete pool count fits top-k require no score or projection.
"""
from dataclasses import dataclass


def project_query_rows(x, begin, end):
    """Slice real queries, padding only short projection inputs to the FP8 lane."""
    if not 0 <= begin < end <= x.shape[0]:
        raise ValueError('project a nonempty range of real query rows')
    value = x[begin:end]
    if len(value) < 64:
        # DenseLinear switches to its decode W4 lane at <=32 rows. Padding
        # never reaches a cache writer or the subsequent query selection.
        padded = value.new_zeros((64, *value.shape[1:]))
        padded[:len(value)].copy_(value)
        return padded
    return value


def covered_pool_ids(complete, topk_pools, *, out=None):
    """Every visible pool, in a caller-owned destination when provided."""
    import torch
    shape = (complete.numel(), topk_pools)
    if complete.ndim != 1 or topk_pools <= 0:
        raise ValueError('covered selection requires row counts and a positive width')
    if out is None:
        out = torch.empty(shape, dtype=torch.int32, device=complete.device)
    elif (out.shape != shape or out.dtype != torch.int32
          or out.device != complete.device or not out.is_contiguous()):
        raise ValueError('covered selection destination must match contiguous int32 rows')
    ids = torch.arange(topk_pools, device=complete.device, dtype=torch.int32)
    out.copy_(ids.expand(shape))
    return out.masked_fill_(out >= complete[:, None], -1)


def window_pool_ids(complete, topk_pools, sink, recent):
    """A sink and a recent window of the visible pools in a scored selection's place (Windowed-MTP, arXiv 2607.21535:
    the draft attends the first positions and the latest, the target everything it selects): every visible pool while
    they fit `sink + recent`, else the first `sink` and the last `recent`; ascending, -1 after, int32 [rows,
    topk_pools] -- `covered_pool_ids`' form, so the attention reads them as it reads a selection. Device arithmetic
    only: a captured step builds them with no host read."""
    import torch
    if complete.ndim != 1 or topk_pools <= 0 or sink < 0 or recent <= 0 or sink + recent > topk_pools:
        raise ValueError('a window selection is a nonnegative sink and a positive recent count within the width')
    width = sink + recent
    ids = torch.arange(topk_pools, device=complete.device, dtype=torch.int32)[None, :]
    seen = complete.to(torch.int32)[:, None]
    # past the window, the recent pools slide: column j >= sink reads pool j + (seen - width), the last `recent` seen
    slide = torch.where((seen > width) & (ids >= sink), seen - width, torch.zeros((), dtype=torch.int32,
                                                                                device=complete.device))
    return torch.where(ids < torch.clamp(seen, max=width), ids + slide, torch.full((), -1, dtype=torch.int32,
                                                                                  device=complete.device))


@dataclass(frozen=True)
class QueryShard:
    rows: int
    context: int
    rank: int
    world: int
    topk_pools: int
    pool: int

    def __post_init__(self):
        if (any(type(x) is not int for x in (self.rows, self.context, self.rank, self.world,
                                             self.topk_pools, self.pool))
                or self.rows < self.world or self.context < 0 or self.world != 4
                or not 0 <= self.rank < self.world or self.topk_pools <= 0 or self.pool <= 0):
            raise ValueError('prefill query shards require real TP4 rows and positive pool geometry')

    @property
    def capacity(self):
        return (self.rows + self.world - 1) // self.world

    @property
    def begin(self):
        return min(self.rows, self.rank * self.capacity)

    @property
    def end(self):
        return min(self.rows, (self.rank + 1) * self.capacity)

    @property
    def score_begin(self):
        # floor((context + row + 1)/pool) <= topk_pools.
        covered = (self.topk_pools + 1) * self.pool - 1 - self.context
        return min(self.end, max(self.begin, covered))

    @property
    def score_rows(self):
        return self.end - self.score_begin

    @property
    def all_covered(self):
        return (self.context+self.rows)//self.pool <= self.topk_pools

    @property
    def wire_bits(self):
        # 65535 is the invalid marker. Two uint16 IDs travel in an int32
        # NCCL lane; no arithmetic collective interprets the packed word.
        return 16 if self.topk_pools % 2 == 0 and (self.context+self.rows)//self.pool <= 65535 else 32

    def project_input(self, x):
        """Keep small tails on DenseLinear's FP8 prefill lane (>32 rows)."""
        if x.shape[0] != self.rows or not self.score_rows:
            raise ValueError('project only nonempty owned query rows')
        return project_query_rows(x, self.score_begin, self.end)

    def collect(self, scored, complete, comm):
        import torch
        if (comm.world_size != self.world or comm.rank != self.rank
                or complete.shape != (self.rows,)):
            raise ValueError('query result must use its original rank and row ownership')
        if self.all_covered:
            if scored is not None:
                raise ValueError('covered queries must not be scored')
            return covered_pool_ids(complete, self.topk_pools)
        owned = self.end - self.begin
        covered = self.score_begin - self.begin
        local = torch.empty((self.capacity, self.topk_pools), dtype=torch.int32, device=complete.device)
        if covered:
            covered_pool_ids(complete[self.begin:self.score_begin], self.topk_pools, out=local[:covered])
        if self.score_rows:
            if scored is None or scored.shape != (self.score_rows, self.topk_pools):
                raise ValueError('missing scored query rows')
            local[self.score_begin-self.begin:owned].copy_(scored)
        elif scored is not None:
            raise ValueError('covered queries must not be scored')
        if owned < self.capacity:
            local[owned:].fill_(-1)
        if self.wire_bits == 32:
            return comm.all_gather(local, dim=0)[:self.rows]
        packet = local.to(torch.int16).view(torch.int32)
        gathered = comm.all_gather(packet, dim=0).view(torch.int16)[:self.rows]
        result = gathered.to(torch.int32).bitwise_and_(65535)
        return result.masked_fill_(result == 65535, -1)
