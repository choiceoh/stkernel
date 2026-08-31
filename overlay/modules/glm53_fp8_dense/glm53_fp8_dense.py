# SPDX-License-Identifier: Apache-2.0
"""Block-fp8 (W8A8, ue8m0, 128x128) copies of GLM-5.3-Flash's dense projections.

The RedHat nvfp4 checkpoint quantizes the routed experts and leaves every dense
projection bf16: 15.44 GB of the 197.8 GB checkpoint (attn 12.37 + shared
expert 2.16 + first-3 dense MLP 0.91) that a decode step reads in full, every
forward, on every rank -- ~3.86 GB/rank/forward, ~17 ms at the measured
223 GB/s. That read is the `cutlass_80_wmma` block that dominates the step
trace. DSV4 on this same fleet serves its whole model, dense projections
included, in exactly this fp8 block scheme (e4m3, ue8m0, [128,128], dynamic
activation) and holds 9/9 retrieval, 0/16 Korean corruption and pos-1
acceptance 78.5% with it -- so arming this converges GLM's dense path onto the
configuration the faster lane already runs, it does not explore new precision.

Weights are quantized once, after load and before compile/capture, by swapping
each selected Linear's `quant_method`; the bf16 originals stay in place (the
first boot measures speed; freeing the originals is a memory follow-up knob).
Any layer that fails a guard keeps its bf16 path, and any apply() error drops
that layer back permanently. The quantize/GEMM pair is the one fp8_lm_head
already runs under capture on this stack, so the kernel path is proven.

Merged-projection padding: the KDA layers load q/k/v/b/f/g as one
`in_proj_qkvbfg_a` whose row count is not a multiple of the 128 block (nor is
its TP shard), and the merged `gate_up_proj` shards land wherever TP puts
them. The quantized copy zero-pads rows and columns out to the block grid:
column-parallel matrices slice the extra output rows off, row-parallel ones
get exactly-zero contributions from the zero rows. No loader or checkpoint
change; the bf16 weight is untouched.

Armed by VLLM_GLM53_FP8_DENSE=1 (default off). Rollback is the env alone.
"""

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1", "true", "yes", "on")


# Runtime module names, i.e. what the loader merged the checkpoint tensors
# into -- not the checkpoint tensor names. `mlp.gate_up_proj` matches only the
# first-k dense layers (MoE layers keep their projections under
# `mlp.experts`/`mlp.shared_experts`, which these patterns do not match), and
# the low-rank pieces of the KDA merge ride inside `in_proj_qkvbfg_a`, whose
# padded copy covers them.
_INCLUDE = (
    re.compile(
        r"\.self_attn\.(in_proj_qkvbfg_a|fused_qkv_a_proj|q_proj|o_proj"
        r"|out_proj)$"),
    re.compile(r"\.mlp\.shared_experts\.(gate_up_proj|down_proj)$"),
    re.compile(r"\.mlp\.(gate_up_proj|down_proj)$"),
)


def _quantize_fp8_block_padded(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Block-quantize to the deepgemm ue8m0 layout, zero-padded to 128s.

    Chunked over rows so the fp32 staging copy stays under ~1/8 of the weight,
    same as fp8_lm_head. Returns (q, scales, orig_rows, orig_cols)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    with torch.no_grad():
        w = weight.detach().float()
        rows, cols = w.shape
        rpad, cpad = (-rows) % 128, (-cols) % 128
        if rpad:
            w = torch.cat([w, w.new_zeros(rpad, cols)], dim=0)
        if cpad:
            w = torch.cat([w, w.new_zeros(w.shape[0], cpad)], dim=1)
        chunks_q, chunks_s = [], []
        step = max(128, (w.shape[0] // 8) // 128 * 128)
        for r0 in range(0, w.shape[0], step):
            cq, cs = per_block_cast_to_fp8(
                w[r0 : r0 + step], [128, 128], use_ue8m0=True
            )
            chunks_q.append(cq)
            chunks_s.append(cs)
        q, ws = deepgemm_post_process_fp8_weight_block(
            torch.cat(chunks_q, dim=0),
            torch.cat(chunks_s, dim=0),
            (128, 128),
            use_e8m0=True,
        )
        return q, ws, rows, cols


def _fp8_gemm_padded(
    x: torch.Tensor,
    q: torch.Tensor,
    ws: torch.Tensor,
    orig_rows: int,
    orig_cols: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    flat = x.reshape(-1, x.shape[-1])
    cpad = q.shape[1] - orig_cols
    if cpad:
        flat = torch.cat(
            [flat, flat.new_zeros(flat.shape[0], cpad)], dim=1
        )
    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        flat.to(torch.bfloat16), 128
    )
    out = torch.empty(
        flat.shape[0], q.shape[0], dtype=torch.bfloat16, device=flat.device
    )
    fp8_gemm_nt((xq, xs), (q, ws), out)
    if q.shape[0] != orig_rows:
        out = out[:, :orig_rows]
    return out.view(*x.shape[:-1], orig_rows)


class Fp8DenseMethod:
    """Drop-in `quant_method` routing one Linear through the fp8 copy.

    Holds the original method for bias and error fallbacks; an apply() failure
    restores it for good, so the worst case is the bf16 path we started on."""

    def __init__(self, base, q, ws, orig_rows, orig_cols):
        self._base = base
        self._q, self._ws = q, ws
        self._rows, self._cols = orig_rows, orig_cols

    def apply(self, layer, x, bias=None):
        if bias is not None:
            return self._base.apply(layer, x, bias)
        try:
            return _fp8_gemm_padded(x, self._q, self._ws, self._rows,
                                    self._cols)
        except Exception:
            layer.quant_method = self._base
            return self._base.apply(layer, x, bias)


def maybe_build_fp8_dense(model) -> bool:
    """Quantize the selected dense projections of a loaded model in place."""
    if not _flag("VLLM_GLM53_FP8_DENSE"):
        return False
    quantized, skipped, params = [], [], 0
    for name, mod in model.named_modules():
        if not any(p.search(name) for p in _INCLUDE):
            continue
        weight = getattr(mod, "weight", None)
        base = getattr(mod, "quant_method", None)
        if (
            weight is None
            or not isinstance(weight, torch.Tensor)
            or weight.dim() != 2
            or weight.dtype not in (torch.bfloat16, torch.float16)
            or min(weight.shape) < 512
            or base is None
            or type(base).__name__ != "UnquantizedLinearMethod"
        ):
            skipped.append(name)
            continue
        try:
            q, ws, rows, cols = _quantize_fp8_block_padded(weight)
            mod.quant_method = Fp8DenseMethod(base, q, ws, rows, cols)
            quantized.append(name)
            params += weight.numel()
        except Exception as e:
            logger.warning("[fp8-dense] %s stayed bf16: %r", name, e)
            skipped.append(name)
    gb = params * 2 / 1e9
    logger.warning(
        "[fp8-dense] %d linears quantized (%.2f GB bf16 -> fp8 blocks), "
        "%d kept bf16 -- fingerprint for the boot log",
        len(quantized), gb, len(skipped),
    )
    return bool(quantized)
