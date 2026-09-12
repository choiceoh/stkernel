"""Native shared-expert W4 decode: publish the activation once for down.

The two GEMMs use the existing packs and split/reduction plan. The first
GEMM's epilogue rounds gate/up, applies clamped SwiGLU, rounds the result,
and publishes the same FP8 input groups the ordinary down GEMM quantizes.
Native scratch is stream ordered, like DenseLinear's GEMM workspace.
"""
import math

import torch

from . import DenseLinear, extension


class SharedMLP:
    def __init__(self, gate_up, down, limit):
        if (not isinstance(gate_up, DenseLinear) or not isinstance(down, DenseLinear)
                or len(gate_up.packs) != 1 or len(down.packs) != 1
                or gate_up.rows != 2 * down.cols or gate_up.cols != down.rows
                or not 0 < down.cols <= 4096 or down.cols % 128
                or not 0 < gate_up.cols <= 4096 or gate_up.cols % 128
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("shared MLP needs matching single-pack W4 linears and a positive clamp")
        self.gate_up, self.down, self.limit = gate_up, down, float(limit)
        self.executed = False

    def __call__(self, x, rows_ok=None):
        gu, down = self.gate_up, self.down
        if (x.ndim != 2 or not 1 <= x.shape[0] <= 32 or x.shape[1] != gu.cols
                or x.dtype != torch.bfloat16 or not x.is_cuda
                or x.device != gu.packs[0].data.device):
            raise ValueError("shared MLP decode needs 1..32 matching CUDA BF16 rows")
        x = x.contiguous()  # retain the tensor, not only a temporary's raw pointer
        # Observers can be attached after preparation; read them at capture.
        # The optional BF16 activation is written by the epilogue itself, so
        # statistics see the exact value that the down projection quantized.
        if gu.observer is not None:
            gu.observer(x, rows_ok)
        activated = torch.empty((x.shape[0], down.cols), dtype=x.dtype, device=x.device) if down.observer is not None else None
        scratch = torch.empty((x.shape[0], gu.rows), dtype=x.dtype, device=x.device)
        out = torch.empty((x.shape[0], down.rows), dtype=x.dtype, device=x.device)
        a, b = gu.packs[0], down.packs[0]
        extension().run_smlp2(
            [x.data_ptr(), a.data.data_ptr(), a.scale.data_ptr(),
             b.data.data_ptr(), b.scale.data_ptr(), scratch.data_ptr(), out.data_ptr(),
             a.rowscale.data_ptr(), b.rowscale.data_ptr(),
             activated.data_ptr() if activated is not None else 0],
            [1., 1., self.limit, 1., 0.],
            [x.shape[0], gu.cols, gu.rows, down.cols, down.rows,
             a.data.shape[0], a.data.shape[1], b.data.shape[0], b.data.shape[1]])
        if down.observer is not None:
            down.observer(activated, rows_ok)
        gu.executed |= 1
        down.executed |= 1
        self.executed = True
        return out
