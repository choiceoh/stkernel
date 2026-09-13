"""Private route-owned MoE partials; no serving allocator or dispatch override.

The owner exists before graph capture and survives all replays. The larger
output ABI must only be launched through this adapter. Reduction timing is
included in the candidate graph and the normal FP32-to-BF16 copy is retained.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _reduce_routes(Partial, Out, K: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, PARTS)
    values = tl.load(Partial + (row * PARTS + part[:, None]) * K + col[None, :])
    # Each value was BF16-rounded, then route-weighted in FP32 in the producer.
    tl.store(Out + row * K + col, tl.sum(values, axis=0))


class RouteScatter:
    def __init__(self, compiled, rows, width=4096, routes=8, partials=4):
        if rows not in (7, 14, 21, 28) or (width, routes, partials) != (4096, 8, 4):
            raise ValueError('route scatter owns only GLM TP4 decode output')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('route output must be allocated during prewarm')
        self.compiled, self.rows, self.width = compiled, rows, width
        self.parts = routes * partials
        self.scratch = torch.empty((rows * self.parts, width), device='cuda', dtype=torch.float32)

    def __call__(self, *args):
        output = args[21]
        if (output.shape != (self.rows, self.width) or output.dtype != torch.float32
                or not output.is_contiguous() or output.device != self.scratch.device):
            raise ValueError('route scatter requires the existing contiguous FP32 output plane')
        launch = list(args)
        launch[21] = self.scratch
        self.compiled(*launch)
        _reduce_routes[(self.rows, self.width // 128)](
            self.scratch, output, self.width, self.parts, 128, num_warps=4, enable_fp_fusion=False)


@contextmanager
def route_scatter_owner(md, owners):
    original, wrapped = md._get_static_kernel_v2, {}

    def get(*args, **kwargs):
        compiled, mac = original(*args, **kwargs)
        if not kwargs['config'].get('probe_route_scatter'):
            return compiled, mac
        key = (id(compiled), args[2])
        if key not in wrapped:
            owner = RouteScatter(compiled, args[2])
            owners.append(owner)
            wrapped[key] = (owner, mac)
        return wrapped[key]

    with patch.object(md, '_get_static_kernel_v2', get):
        yield
