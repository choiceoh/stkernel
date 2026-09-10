"""V4.1's grouped output projection, without the 128-granular fp8 kernel.

## Why this file exists

The image's DeepSeek-V4 attention ends with a fused op:

    fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a.weight, scale), z,
               recipe=(1, 1, 128))

`hdr` requires a THREE-dimensional weight -- [groups, rank, width] -- which
vLLM produces because `wo_a.is_bmm = True` makes the fp8 linear method build a
batched weight. That happens on the DeepGEMM block-scaled kernel, which is
selected when the checkpoint's block shape is [128, 128].

V4.1's is [32, 32]:

    layers.0.attn.wo_a.weight  F8_E4M3 [8192, 4096]
    layers.0.attn.wo_a.scale   F8_E8M0 [ 256,  128]   = [8192/32, 4096/32]

so vLLM selects TritonFp8BlockScaledMMKernel instead, which does not build a
batched weight, and the einsum is handed a 2-D tensor:

    b[0] (8192, 4096) dim=2      <- kernel wants dim=3
    RuntimeError: Assertion error (layout.hpp:39): t.dim() == N

Reshaping it to 3-D would silence that assertion and be WRONG: the kernel
reads the weight scales at 128 granularity, and V4.1's describe 32-wide
blocks. The numbers would be finite, plausible, and off by whatever the
neighbouring block's scale happens to be.

## What this does instead

Dequantizes `wo_a` to bf16 once and does the grouped projection in bf16.

That is not a workaround invented here -- it is what DeepSeek's own reference
does, and it says so:

    # wo_a is block-diagonal over groups (each projects only its own heads),
    # hence einsum not Linear. convert.py dequantizes it to bf16; an fp8
    # grouped GEMM would halve the memory.

So the fp8 grouped GEMM is the optimization and bf16 is the baseline, not the
other way round. The cost is +1.3 GiB per rank across 40 layers, against a
71.5 GiB weight budget.

The ACTIVATION still goes through the image's fused inverse-RoPE + fp8 quant
kernel, then is dequantized back. That is deliberate: re-implementing the
inverse RoPE here would be a second copy of a kernel that is already correct,
and the round trip reproduces exactly the values the fp8 path would have fed
its GEMM. It costs precision the fp8 path was going to lose anyway.

A faster lane -- requantizing wo_a from 32x32 to 128x128 blocks so the fused
kernel applies -- is deliberately NOT taken. It would change the numbers the
checkpoint ships against nothing measured.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

QUANT_GROUP = 128          # the activation quant block the fused kernel uses


def dequantize_block_scaled(weight: torch.Tensor, scale: torch.Tensor,
                            block: "tuple[int, int]") -> torch.Tensor:
    """fp8 [O, I] with a [O/bo, I/bi] block scale -> bf16 [O, I].

    Written for any block size rather than 32 specifically: the whole reason
    this file exists is a kernel that assumed one, and repeating that here
    would just move the assumption.
    """
    out_dim, in_dim = weight.shape
    bo, bi = block
    if out_dim % bo or in_dim % bi:
        raise ValueError(
            f"wo_a is {weight.shape} and does not divide into {block} blocks; "
            f"a padded dequantization would read scale rows that describe no "
            f"weights.")
    want = (out_dim // bo, in_dim // bi)
    if tuple(scale.shape) != want:
        raise ValueError(
            f"wo_a scale is {tuple(scale.shape)}, expected {want} for a "
            f"{weight.shape} weight in {block} blocks. A scale of the wrong "
            f"shape is a different block size, and dequantizing with it "
            f"produces finite numbers that are wrong by a per-block factor.")
    w = weight.to(torch.float32).view(out_dim // bo, bo, in_dim // bi, bi)
    s = scale.to(torch.float32).view(out_dim // bo, 1, in_dim // bi, 1)
    return (w * s).view(out_dim, in_dim).to(torch.bfloat16)


def _block_shape(module) -> "tuple[int, int]":
    """The checkpoint's fp8 block shape, from the quant method that loaded it."""
    for holder in (getattr(module, "quant_method", None), module):
        block = getattr(holder, "block_shape", None) or getattr(
            holder, "weight_block_size", None)
        if block:
            return (int(block[0]), int(block[1]))
    raise ValueError(
        "no fp8 block shape on wo_a; without it the scale cannot be applied "
        "and guessing [128, 128] is exactly the bug this replaces.")


def build_wo_a_bf16(attn) -> torch.Tensor:
    """[groups, rank, width] bf16, dequantized from the checkpoint's fp8."""
    wo_a = attn.wo_a
    weight = wo_a.weight
    scale = getattr(wo_a, "weight_scale_inv", None)
    groups = attn.n_local_groups
    rank = attn.o_lora_rank

    if scale is None:
        dense = weight.to(torch.bfloat16)
    else:
        dense = dequantize_block_scaled(weight, scale, _block_shape(wo_a))

    if dense.dim() == 3:
        return dense.contiguous()
    out_dim, width = dense.shape
    if out_dim != groups * rank:
        raise ValueError(
            f"wo_a is {tuple(dense.shape)}; expected its first dim to be "
            f"groups*rank = {groups}*{rank} = {groups * rank}. wo_a is "
            f"block-diagonal over groups, so a first dim that is not that "
            f"product means the group split is not what this rank thinks.")
    return dense.view(groups, rank, width).contiguous()


def dequantize_activation(o_fp8: torch.Tensor,
                          o_scale: torch.Tensor) -> torch.Tensor:
    """[T, G, W] fp8 with [T, G, W/128] fp32 scales -> bf16."""
    tokens, groups, width = o_fp8.shape
    blocks = o_scale.shape[-1]
    if width % blocks:
        raise ValueError(
            f"activation is {width} wide with {blocks} scales, which do not "
            f"divide it")
    per = width // blocks
    return (o_fp8.to(torch.float32).view(tokens, groups, blocks, per)
            * o_scale.to(torch.float32).unsqueeze(-1)
            ).view(tokens, groups, width).to(torch.bfloat16)


def install(attn) -> None:
    """Replace one attention module's `_o_proj` with the bf16 grouped path."""
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    state = {}

    def _o_proj(o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        wo_a = state.get("wo_a")
        if wo_a is None:
            wo_a = state["wo_a"] = build_wo_a_bf16(attn)
            logger.info_once(
                "DeepSeek-V4.1 o-projection: bf16 grouped einsum, %s "
                "(the image's fused fp8 einsum needs 128-granular scales; "
                "this checkpoint ships 32)", tuple(wo_a.shape))
        # tma_aligned_scales=False so the scales come back as fp32 rather than
        # INT32-packed ue8m0: this path multiplies by them directly, and
        # unpacking a layout that exists for the TMA would be work done to
        # undo work.
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o, positions, attn.rotary_emb.cos_sin_cache,
            n_groups=attn.n_local_groups,
            heads_per_group=attn.n_local_heads // attn.n_local_groups,
            nope_dim=attn.nope_head_dim,
            rope_dim=attn.rope_head_dim,
            quant_group_size=QUANT_GROUP,
            tma_aligned_scales=False,
        )
        z = torch.einsum("tgw,grw->tgr", dequantize_activation(o_fp8, o_scale),
                         wo_a)
        return attn.wo_b(z.flatten(1))

    attn._o_proj = _o_proj
