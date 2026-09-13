"""Partition replicated indexer queries, retaining every cache-writing row.

Only pool IDs cross ranks. Each rank finalizes them against its own page map.
Queries whose complete pool count fits top-k require no score or projection.
"""
from dataclasses import dataclass


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
        import torch
        value = x[self.score_begin:self.end]
        if len(value) < 64:
            value = torch.cat((value, value.new_zeros((64-len(value), *value.shape[1:]))))
        return value

    def collect(self, scored, complete, comm):
        import torch
        if (comm.world_size != self.world or comm.rank != self.rank
                or complete.shape != (self.rows,)):
            raise ValueError('query result must use its original rank and row ownership')
        if self.all_covered:
            if scored is not None:
                raise ValueError('covered queries must not be scored')
            ids = torch.arange(self.topk_pools, device=complete.device, dtype=torch.int32)
            full = ids.expand(self.rows, -1).contiguous()
            return full.masked_fill_(full >= complete[:,None], -1)
        owned = self.end - self.begin
        ids = torch.arange(self.topk_pools, device=complete.device, dtype=torch.int32)
        local = ids.expand(owned, -1).contiguous()
        local.masked_fill_(local >= complete[self.begin:self.end, None], -1)
        if self.score_rows:
            if scored is None or scored.shape != (self.score_rows, self.topk_pools):
                raise ValueError('missing scored query rows')
            local[self.score_begin-self.begin:].copy_(scored)
        elif scored is not None:
            raise ValueError('covered queries must not be scored')
        if owned < self.capacity:
            local = torch.cat((local, local.new_full((self.capacity-owned, self.topk_pools), -1)))
        if self.wire_bits == 32:
            return comm.all_gather(local, dim=0)[:self.rows]
        packet = local.to(torch.int16).view(torch.int32)
        gathered = comm.all_gather(packet, dim=0).view(torch.int16)[:self.rows]
        result = gathered.to(torch.int32).bitwise_and_(65535)
        return result.masked_fill_(result == 65535, -1)
