# SPDX-License-Identifier: Apache-2.0
"""Block-quantized fp8 (W8A16) vocabulary head for a speculative drafter.

The draft head is read once per drafted token, so halving its bytes is worth
more than the quantization costs. Adopted on this fleet for DSV4's drafter as
VLLM_DSPARK_FP8_LM_HEAD; the quantize/GEMM pair below is that implementation,
moved here so a second drafter can use it instead of growing a copy.

Not to be confused with the rowwise `_scaled_mm` draft head, which was measured
on the same fleet and rejected (60.6 against 61.7, acceptance unmoved).

`build_fp8_lm_head` runs after weights load and before capture; it attaches the
quantized pair to the head module, so `Fp8HeadLogitsProcessor._apply_head` finds
them from the argument it is already given and everything else in
`get_top_k_tokens` -- padding mask, top-k reduction, its all-gather, scale, soft
cap -- stays untouched.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor

logger = init_logger(__name__)


def _read_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _quantize_fp8_deepgemm(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a bf16/fp32 weight to the deepgemm fp8 layout (chunked
    over rows so the fp32 staging copy never exceeds ~1/8 of the weight)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    with torch.no_grad():
        w = weight.detach()
        rows = w.shape[0]
        step = max(128, (rows // 8) // 128 * 128)
        chunks_q, chunks_s = [], []
        for r0 in range(0, rows, step):
            cq, cs = per_block_cast_to_fp8(
                w[r0 : r0 + step].float(), [128, 128], use_ue8m0=True
            )
            chunks_q.append(cq)
            chunks_s.append(cs)
        return deepgemm_post_process_fp8_weight_block(
            torch.cat(chunks_q, dim=0),
            torch.cat(chunks_s, dim=0),
            (128, 128),
            use_e8m0=True,
        )


def _fp8_gemm(
    x: torch.Tensor, dg_w: torch.Tensor, dg_ws: torch.Tensor
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        x.to(torch.bfloat16), 128
    )
    out = torch.empty(
        x.shape[0], dg_w.shape[0], dtype=torch.bfloat16, device=x.device
    )
    fp8_gemm_nt((xq, xs), (dg_w, dg_ws), out)
    return out


def build_fp8_lm_head(model) -> bool:
    """Quantize `model.lm_head` in place. Returns whether it took.

    Any failure leaves the head untouched and the caller on the bf16 path: this
    is an optimization, and a drafter that runs slower is better than one that
    does not run.
    """
    if not _read_bool_env("VLLM_SPEC_FP8_LM_HEAD"):
        return False
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    try:
        dg_w, dg_ws = _quantize_fp8_deepgemm(weight)
    except Exception:
        logger.warning_once(
            "VLLM_SPEC_FP8_LM_HEAD=1: draft head quantization failed; "
            "staying on bf16."
        )
        return False
    head._deneb_fp8_w = dg_w
    head._deneb_fp8_ws = dg_ws
    logger.info_once(
        "VLLM_SPEC_FP8_LM_HEAD=1: draft lm_head quantized to fp8 (%s).",
        tuple(dg_w.shape),
    )
    return True


class Fp8HeadLogitsProcessor(LogitsProcessor):
    """LogitsProcessor whose head projection uses the fp8 copy when present."""

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        dg_w = getattr(lm_head, "_deneb_fp8_w", None)
        if (
            dg_w is None
            or embedding_bias is not None
            or (self.head_dtype is not None
                and self.head_dtype != hidden_states.dtype)
        ):
            return super()._apply_head(lm_head, hidden_states, embedding_bias)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = _fp8_gemm(flat, dg_w, lm_head._deneb_fp8_ws)
        return out.view(*hidden_states.shape[:-1], -1)
