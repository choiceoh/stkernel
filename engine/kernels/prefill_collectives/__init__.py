"""Explicit eager token-shard transport: BF16 below 2048 rows, FP8 v3 above.

Packets carry independently scaled FP8 values and FP32 scales in one byte
exchange. Reduce-scatter sums all source packets in FP32, then rounds once.
Every invocation owns its storage; no process-global buffer or capture state.

The row width and the world come from the bound kernel shape
(engine/base/kernel_shape): the packet kernels are generic in both, so a
profile with another hidden width declares it instead of editing this file.
"""
import torch
import torch.distributed as dist

from engine.kernels.cells import PREFILL_BLOCK as BLOCK     # the packet block, stated once (cells.py)

# Phase2 candidate: include 2K requests in the existing block-scaled FP8
# transport. The wider precision admission still needs full quality proof.
FP8_MIN_ROWS = 2048


class PrefillCollectives:
    def __init__(self, comm, *, project_tiles=False, fuse_sum=False):
        from engine.base.kernel_shape import bound
        cell = bound().comm
        if comm.world_size != cell.world:
            raise ValueError(f"prefill sequence parallelism requires TP{cell.world}")
        self.comm = comm
        self.world, self.hidden = cell.world, cell.hidden
        self.executed = set()
        if type(project_tiles) is not bool:
            raise ValueError("project_tiles must be a boolean")
        self.project_tiles = project_tiles
        self.projector = None
        if type(fuse_sum) is not bool:
            raise ValueError("private fused-sum selection must be a boolean")
        self.fuse_sum = fuse_sum

    def reduce_scatter_pair(self, x, y, *, padded_rows):
        """Private prefill sum/pack fusion with transport-only row padding."""
        self.check(x)
        self.check(y)
        if (x.shape != y.shape or x.device != y.device or type(padded_rows) is not int
                or padded_rows != ((x.shape[0]+self.world-1)//self.world)*self.world):
            raise ValueError('fused prefill sum requires matching inputs and exact transport padding')
        if not self.fuse_sum or padded_rows < FP8_MIN_ROWS:
            summed = x+y
            if padded_rows != x.shape[0]:
                summed = torch.cat((summed, summed.new_zeros((padded_rows-x.shape[0],self.hidden))))
            return self.reduce_scatter(summed)
        from .sum_pack import _pack_sum_rs_payload
        from .kernels import _unpack_sum_payload
        local = padded_rows*self.hidden//self.world
        if local % BLOCK:
            raise ValueError('fused sum packets require whole blocks per destination')
        stride = ((local+4*(local//BLOCK)+127)//128)*128
        payload = torch.empty(stride*self.world,device=x.device,dtype=torch.uint8)
        _pack_sum_rs_payload[(padded_rows*self.hidden//BLOCK,)](
            x,y,payload.view(torch.float8_e4m3fn),payload.view(torch.float32),
            x.numel(),local,stride,BLOCK=BLOCK)
        received = torch.empty_like(payload)
        dist.all_to_all_single(received,payload,group=self.comm.group)
        out = torch.empty((padded_rows//self.world,self.hidden),device=x.device,dtype=x.dtype)
        _unpack_sum_payload[(local//BLOCK,)](
            received.view(torch.float8_e4m3fn),received.view(torch.float32),out,
            local,stride,TP=self.world,BLOCK=BLOCK)
        self.executed.add('fp8_reduce_scatter_sum')
        return out

    def gather_project(self, x, project, *, packet_project=None):
        if not self.project_tiles:
            return project(self.all_gather(x))
        if self.projector is None:
            from .tiles import TiledProjection
            self.projector = TiledProjection(self)
        return self.projector(x, project, packet_project=packet_project)

    def check(self, x):
        if (x.ndim != 2 or x.shape[1] != self.hidden or x.dtype != torch.bfloat16
                or not x.is_cuda or not x.is_contiguous() or x.shape[0] < 32):
            raise ValueError(f"prefill transport requires contiguous CUDA BF16 [rows>=32,{self.hidden}]")
        if (x.shape[0] * x.shape[1]) % BLOCK:
            # the packet kernels walk whole 2048-element blocks; a hidden that does not divide the block
            # needs a row count that completes it (4096 always does, 2560 every fourth row)
            raise ValueError(f"prefill packets are {BLOCK}-element blocks; [{x.shape[0]},{x.shape[1]}] does not tile them")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prefill token shards cannot enter a decode graph")

    @staticmethod
    def pack(x, local_elements):
        from .kernels import _pack_rs_payload
        packet_bytes = ((local_elements + 4*(local_elements//BLOCK) + 127)//128)*128
        peers = x.numel()//local_elements
        payload = torch.empty(packet_bytes*peers, device=x.device, dtype=torch.uint8)
        _pack_rs_payload[(x.numel()//BLOCK,)](
            x, payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
            x.numel(), local_elements, packet_bytes, BLOCK=BLOCK)
        return payload, packet_bytes

    def all_gather(self, x):
        self.check(x)
        world = self.world
        if x.shape[0]*world < FP8_MIN_ROWS:
            return self.comm.all_gather(x, dim=0)
        from .kernels import _unpack_gather
        payload, stride = self.pack(x, x.numel())
        received = torch.empty(payload.numel()*world, device=x.device, dtype=torch.uint8)
        dist.all_gather_into_tensor(received, payload, group=self.comm.group)
        out = torch.empty((x.shape[0]*world, self.hidden), device=x.device, dtype=x.dtype)
        _unpack_gather[(out.numel()//BLOCK,)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), out,
            x.numel(), stride, BLOCK=BLOCK)
        self.executed.add('fp8_all_gather')
        return out

    def all_gather_packets(self, x, *, rows):
        """The ordinary full FP8 exchange with an invocation-owned packet result."""
        from engine.modules.prefill_packets import PacketBatch, PacketGeometry, ffn_packet_rows
        self.check(x)
        geometry = PacketGeometry(rows, x.shape[0], self.hidden, self.world, BLOCK)
        if not ffn_packet_rows(rows):
            raise ValueError('packet FFN requires 8192 < real rows <= 32768')
        payload, stride = self.pack(x, x.numel())
        if stride != geometry.stride:
            raise ValueError('FFN packet stride differs from its declared transport')
        received = torch.empty(geometry.nbytes, device=x.device, dtype=torch.uint8)
        dist.all_gather_into_tensor(received, payload, group=self.comm.group)
        self.executed.update(('fp8_all_gather', 'ffn_packets_v1'))
        return PacketBatch(received, geometry)

    def reduce_scatter(self, x):
        self.check(x)
        world = self.world
        if x.shape[0] % world:
            raise ValueError("prefill reduce-scatter requires equal token shards")
        if x.shape[0] < FP8_MIN_ROWS:
            return self.comm.reduce_scatter_rows(x)
        from .kernels import _unpack_sum_payload
        local = x.numel()//world
        payload, stride = self.pack(x, local)
        received = torch.empty_like(payload)
        dist.all_to_all_single(received, payload, group=self.comm.group)
        out = torch.empty((x.shape[0]//world, self.hidden), device=x.device, dtype=x.dtype)
        _unpack_sum_payload[(local//BLOCK,)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), out,
            local, stride, TP=world, BLOCK=BLOCK)
        self.executed.add('fp8_reduce_scatter')
        return out
