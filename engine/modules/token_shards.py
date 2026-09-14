"""Equal transport packets for an arbitrary number of real prefill tokens.

Only communication and row-local projection see padding. Attention and KDA
receive the original rows, so padding never acquires a position or cache slot.
The owner and its two transfer buffers survive; this view belongs to one step.
"""


class TokenShards:
    def __init__(self, owner, rows, rank):
        world = owner.comm.world_size
        if type(rows) is not int or rows < world or not 0 <= rank < world:
            raise ValueError("prefill shards require at least one real token per rank")
        self.owner, self.comm = owner, owner.comm
        self.rows, self.rank, self.world = rows, rank, world
        self.local_rows = (rows + world - 1) // world
        if rows <= (world - 1) * self.local_rows:
            raise ValueError("prefill shards cannot leave a rank without a real token")
        self.padded_rows = self.local_rows * world
        self.project_tiles = owner.project_tiles
        self.fuse_sum = getattr(owner, 'fuse_sum', False)

    @property
    def last_local(self):
        return min(self.local_rows, self.rows - self.rank * self.local_rows) - 1

    def pad(self, x):
        if x.shape[0] != self.rows:
            raise ValueError("prefill collective input must contain exactly the real token rows")
        if self.padded_rows == self.rows:
            return x
        import torch
        return torch.cat((x, x.new_zeros((self.padded_rows - self.rows, *x.shape[1:]))), dim=0)

    def shard(self, x):
        start = self.rank * self.local_rows
        return self.pad(x)[start:start + self.local_rows].contiguous()

    def all_gather(self, x):
        return self.owner.all_gather(x)[:self.rows]

    def all_gather_packets(self, x, *, route=None):
        if x.shape[0] != self.local_rows:
            raise ValueError('FFN packet input must contain exactly one local shard')
        options = {} if route is None else dict(route=route)
        return self.owner.all_gather_packets(x, rows=self.rows, **options)

    def gather_project(self, x, project, *, packet_project=None):
        options = {} if packet_project is None else dict(packet_project=packet_project)
        return self.owner.gather_project(x, project, **options)[:self.rows]

    def reduce_scatter(self, x):
        return self.owner.reduce_scatter(self.pad(x))

    def reduce_scatter_pair(self, x, y):
        if x.shape[0] != self.rows or y.shape[0] != self.rows:
            raise ValueError('prefill sum inputs must contain exactly the real token rows')
        return self.owner.reduce_scatter_pair(x,y,padded_rows=self.padded_rows)

    def gather_result(self, x):
        """Final hidden/auxiliary rows use lossless communication as before."""
        return self.comm.all_gather(x, dim=0)[:self.rows]
