"""Explicit eager token-shard transport: BF16 below 4096 rows, FP8 v3 above.

Packets carry independently scaled FP8 values and FP32 scales in one byte
exchange. Reduce-scatter sums all source packets in FP32, then rounds once.
Every invocation owns its storage; no process-global buffer or capture state.
"""
import torch
import torch.distributed as dist

from .kernels import _pack_rs_payload, _unpack_gather, _unpack_sum_payload

BLOCK = 2048
FP8_MIN_ROWS = 4096


class PrefillCollectives:
    def __init__(self, comm):
        if comm.world_size != 4:
            raise ValueError("prefill sequence parallelism requires TP4")
        self.comm = comm
        self.executed = set()

    @staticmethod
    def check(x):
        if (x.ndim != 2 or x.shape[1] != 4096 or x.dtype != torch.bfloat16
                or not x.is_cuda or not x.is_contiguous() or x.shape[0] < 32):
            raise ValueError("prefill transport requires contiguous CUDA BF16 [rows>=32,4096]")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prefill token shards cannot enter a decode graph")

    @staticmethod
    def pack(x, local_elements):
        packet_bytes = ((local_elements + 4*(local_elements//BLOCK) + 127)//128)*128
        peers = x.numel()//local_elements
        payload = torch.empty(packet_bytes*peers, device=x.device, dtype=torch.uint8)
        _pack_rs_payload[(x.numel()//BLOCK,)](
            x, payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
            x.numel(), local_elements, packet_bytes, BLOCK=BLOCK)
        return payload, packet_bytes

    def all_gather(self, x):
        self.check(x)
        if x.shape[0]*4 < FP8_MIN_ROWS:
            return self.comm.all_gather(x, dim=0)
        payload, stride = self.pack(x, x.numel())
        received = torch.empty(payload.numel()*4, device=x.device, dtype=torch.uint8)
        dist.all_gather_into_tensor(received, payload, group=self.comm.group)
        out = torch.empty((x.shape[0]*4,4096), device=x.device, dtype=x.dtype)
        _unpack_gather[(out.numel()//BLOCK,)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), out,
            x.numel(), stride, BLOCK=BLOCK)
        self.executed.add('fp8_all_gather')
        return out

    def reduce_scatter(self, x):
        self.check(x)
        if x.shape[0] % 4:
            raise ValueError("prefill reduce-scatter requires equal token shards")
        if x.shape[0] < FP8_MIN_ROWS:
            return self.comm.reduce_scatter_rows(x)
        local = x.numel()//4
        payload, stride = self.pack(x, local)
        received = torch.empty_like(payload)
        dist.all_to_all_single(received, payload, group=self.comm.group)
        out = torch.empty((x.shape[0]//4,4096), device=x.device, dtype=x.dtype)
        _unpack_sum_payload[(local//BLOCK,)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), out,
            local, stride, TP=4, BLOCK=BLOCK)
        self.executed.add('fp8_reduce_scatter')
        return out
