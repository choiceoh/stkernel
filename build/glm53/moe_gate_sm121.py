# SPDX-License-Identifier: Apache-2.0
"""Fused small-M MoE router gate for sm_121 (GB10).

Not a model-specific optimization. The image's `GateLinear` picks its
accelerated tier from device-family checks that admit only SM90 and SM100
(`is_blackwell` tests family 100), so GB10 lands on Tier-4 for every MoE it
serves: a bf16 `F.linear` followed by a bf16 round trip and an fp32 cast, three
kernels where one will do.

Measured on DeepSeek-V4-Flash (43 layers + 3 mtp, bf16 [256, 4096] gate): the
Tier-4 chain costs 1.71 ms/step and the fused kernel 1.18, worth **C=1 +3.5%**
end to end. Logit error against the same weights drops 6e-2 -> 1.3e-4 because
the bf16 round trip is gone, and the result is bitwise deterministic.

Arms only under VLLM_DSV4_GATE_FUSED with M <= 32, so prefill keeps the stock
Tier-4 path and its numerics unchanged. Warm-up compiles the Triton
specialization buckets before capture; any failure falls back permanently.
"""

import os

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

logger = init_logger(__name__)


# deneb fork: fused small-M MoE router gate (VLLM_DSV4_GATE_FUSED, default off).
# GB10 (sm_121) fails every specialized-tier device gate in GateLinear
# (is_blackwell tests compute-capability family 100), so production routing
# fell to Tier 4: bf16 F.linear (cublas split-K wmma + splitKreduce) plus a
# separate fp32 cast — 3 kernels x 46 gates (43 layers + 3 mtp) = 1.71 ms per
# decode step, the measured grid=[8,2,8] wmma population. The fused tier
# computes the same [M,4096]x[4096,256] GEMM in one triton split-K kernel and
# one deterministic sum-reduce, accumulating fp32 END TO END: Tier 4's bf16
# round-trip of the logits is dropped, so outputs are strictly closer to the
# fp32 reference but bit-DIFFERENT from baseline — hence the kill-switch.
# Prefill and any M > 32 keep the exact Tier-4 path and numerics.
# Armed by either name. The DSV4 one is what production sets; the neutral one
# is what a module that applies to every MoE on this hardware should answer to.
_GATE_FUSED = (
    os.environ.get("VLLM_MOE_GATE_FUSED")
    or os.environ.get("VLLM_DSV4_GATE_FUSED", "0")
).strip().lower() in (
    "1", "true", "yes", "on",
)
_GATE_FUSED_MAX_TOKENS = 32
# Tile config from the 2026-08-10 offline cold-cycle sweeps (46 distinct
# weights cycled through a CUDA graph to defeat L2). SPLIT_K=1 is load-
# bearing: the first in-engine trace showed the generic torch-sum reduce of
# the SK=4 variant stretching 2.2us -> 12.4us inside the decode graph
# (few-CTA latency-bound kernel under contention), eating the win. Single
# kernel, fp32 out direct: BN16/BK512/SK1/stages4/warps4 = 12.04us cold vs
# Tier-4 chain 15.27 (M=6), and no second kernel to stretch.
_GATE_FUSED_SPLIT_K = 1
_GATE_FUSED_BLOCK_N = 16
_GATE_FUSED_BLOCK_K = 512

if _GATE_FUSED:
    import triton
    import triton.language as tl

    @triton.jit
    def _deneb_gate_partial_kernel(
        x_ptr,
        w_ptr,
        part_ptr,
        M,
        stride_xm,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        # program (pid_n, pid_k) accumulates x[M, k-slice] @ w[n-block,
        # k-slice].T in fp32 and stores its partial to part[pid_k]. The final
        # sum over SPLIT_K happens in a fixed-order torch sum (deterministic;
        # no fp32 atomics).
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = rm < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_lo = pid_k * (K // SPLIT_K)
        for k0 in range(0, K // SPLIT_K, BLOCK_K):
            rk = k_lo + k0 + tl.arange(0, BLOCK_K)
            x_tile = tl.load(
                x_ptr + rm[:, None] * stride_xm + rk[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            w_tile = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
            acc = tl.dot(x_tile, tl.trans(w_tile), acc)
        part = part_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :]
        tl.store(part, acc, mask=m_mask[:, None])

    def _deneb_fused_gate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        n_out, k_in = weight.shape
        if _GATE_FUSED_SPLIT_K == 1:
            # single kernel writes fp32 logits directly — no partials, no
            # reduce launch (see the SPLIT_K note above)
            out = torch.empty(
                (num_tokens, n_out), dtype=torch.float32, device=x.device
            )
        else:
            out = torch.empty(
                (_GATE_FUSED_SPLIT_K, num_tokens, n_out),
                dtype=torch.float32,
                device=x.device,
            )
        block_m = 16 if num_tokens <= 16 else 32
        _deneb_gate_partial_kernel[
            (n_out // _GATE_FUSED_BLOCK_N, _GATE_FUSED_SPLIT_K)
        ](
            x,
            weight,
            out,
            num_tokens,
            x.stride(0),
            N=n_out,
            K=k_in,
            BLOCK_M=block_m,
            BLOCK_N=_GATE_FUSED_BLOCK_N,
            BLOCK_K=_GATE_FUSED_BLOCK_K,
            SPLIT_K=_GATE_FUSED_SPLIT_K,
            num_warps=4,
            # stages are smem-bounded: (BM+BN)*BK*2B*(stages-1) must fit the
            # 99KB budget — BM16/s4 = 96KB fits, BM32/s4 = 144KB does not.
            # The sweep ranked BM32 best at s3 anyway (12.21us M=24).
            num_stages=4 if block_m == 16 else 3,
        )
        if _GATE_FUSED_SPLIT_K == 1:
            return out
        return out.sum(dim=0)


class DenebGateLinear(GateLinear):
    """GateLinear with the fused small-M tier of `_deneb_fused_gate` above.

    The fused tier arms only when VLLM_DSV4_GATE_FUSED is on AND the
    init-time prewarm compiled every triton specialization this deployment
    can hit (M buckets {1, %16==0, other} x BLOCK_M {16, 32}); the prewarm
    runs on dummy CUDA tensors before any CUDA-graph capture, so captured
    decode graphs replay an already-loaded kernel. If the prewarm fails, the
    layer permanently falls back to the stock GateLinear tiers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deneb_fused_ok = (
            _GATE_FUSED
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
            and self.weight.dim() == 2
            and self.weight.shape[0] % _GATE_FUSED_BLOCK_N == 0
            and self.weight.shape[1]
            % (_GATE_FUSED_SPLIT_K * _GATE_FUSED_BLOCK_K)
            == 0
            and self.weight.is_contiguous()
            and _deneb_gate_prewarm(tuple(self.weight.shape))
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        if (
            self._deneb_fused_ok
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and 0 < x.shape[0] <= _GATE_FUSED_MAX_TOKENS
            and x.stride(-1) == 1
        ):
            return _deneb_fused_gate(x, self.weight), None
        return super().forward(x)


_DENEB_GATE_PREWARMED: dict[tuple[int, int], bool] = {}


def _deneb_gate_prewarm(shape: tuple[int, int]) -> bool:
    """Compile-and-launch every reachable specialization once, off-capture.

    Triton specializes launches on integer-arg buckets; warming M in
    {1, 6, 16} (BLOCK_M=16) and {17, 32} (BLOCK_M=32) covers all M <= 32.
    Runs on dummy tensors at module-init time (CUDA active, capture not),
    once per weight shape. Any failure disarms the fused tier for good.
    """
    cached = _DENEB_GATE_PREWARMED.get(shape)
    if cached is not None:
        return cached
    ok = False
    try:
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            n_out, k_in = shape
            w = torch.zeros((n_out, k_in), dtype=torch.bfloat16, device="cuda")
            for m in (1, 6, 16, 17, 32):
                x = torch.zeros((m, k_in), dtype=torch.bfloat16, device="cuda")
                _deneb_fused_gate(x, w)
            torch.cuda.synchronize()
            del w, x
            ok = True
            logger.info_once(
                "DSV4 fused small-M router gate armed (triton split-K, fp32 out)."
            )
    except Exception:
        logger.warning(
            "DSV4 fused router gate prewarm failed for shape %s; "
            "falling back to stock GateLinear.",
            shape,
            exc_info=True,
        )
        ok = False
    _DENEB_GATE_PREWARMED[shape] = ok
    return ok
