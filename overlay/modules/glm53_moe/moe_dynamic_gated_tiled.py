"""
MoEGatedDynamicKernelTiled -- the stock gated dynamic (prefill) NVFP4 MoE
kernel over TILE-MAJOR expert weights (static v2 cell ``t``, 39차).

The static v5 lane stores each expert's packed weights tile-major (w13 as
[K/512][rows][256 B], w2 as [I_tp/128][rows][64 B]) so its TMA boxes are
contiguous, and hands every kernel the same storage as a 4-D fp4 tensor
(rows, K_in, K/K_in, E). The stock gated kernel builds its TMA descriptors
from the tensor's layout, so it reads that storage correctly once the two K
modes are grouped into one hierarchical K -- its 64 B (128-element) k tiles
divide the 512 / 128-element inner chunks -- which is all this subclass
does before delegating to the stock ``__call__``. Nothing else changes: the
scale-factor storage is the stock layout, the kernel body is the image's.

A subclass rather than an overlay of ``_moe_dynamic/gated.py``: the #368
prefill-reuse lane pins the stock file's SHA-256 and fails closed to stock
when that source changes, so the stock file must stay byte-identical.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from ._moe_dynamic.gated import MoEGatedDynamicKernel


_M64_GATED_SHA256 = "993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445"


@lru_cache(maxsize=1)
def m64_stock_contract_matches():
    # This experiment changes the three M-dependent constructor dimensions.
    # The inherited MMA, staging, route packing and scatter body must be the
    # exact source audited for that change, not a future image implementation.
    try:
        source = Path(__file__).parent / "_moe_dynamic/gated.py"
        return hashlib.sha256(source.read_bytes()).hexdigest() == _M64_GATED_SHA256
    except OSError:
        return False


class MoEGatedDynamicKernelTiled(MoEGatedDynamicKernel):
    """The stock gated kernel; b_w13 / b_down may arrive 4-D (tile-major)."""

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        b_w13: cute.Tensor,        # (rows, K, E) or tile-major (rows, K_in, K/K_in, E)
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,       # (K, I_tp, E) or tile-major (K, K_in, I_tp/K_in, E)
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # (rows, K_in, K_tiles, E) -> (rows, (K_in, K_tiles), E): one
        # hierarchical K the stock tiles divide; a static branch (a plain
        # `if` on the rank trips the DSL's structure check)
        if cutlass.const_expr(len(b_w13.shape) == 4):
            b_w13 = cute.group_modes(b_w13, 1, 3)
        if cutlass.const_expr(len(b_down.shape) == 4):
            b_down = cute.group_modes(b_down, 1, 3)
        return MoEGatedDynamicKernel.__call__(
            self,
            a_input,
            topk_ids,
            topk_weights,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            task_head,
            task_tail,
            task_expert,
            task_valid_rows,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights,
            max_active_clusters,
            stream,
        )


class MoEGatedDynamicKernelM64Tiled(MoEGatedDynamicKernelTiled):
    """Experimental M64 port of the pinned branch-paired gated kernel.

    The stock constructor/factory accepts only M128. Keep it unchanged and
    initialize its shared attributes, then replace all three M-dependent
    constructor dimensions before __call__ derives layouts and pipeline state.
    The 4x2 warp grid has a 64-row M atom, so this uses one M iteration instead
    of two. N128, dual-N64 FC1, physical N128 SFB, K128 and barriers stay fixed.
    Q0 holds 64*128 BF16 elements, enough for the scoped H4096 input row.
    """

    def __init__(self, *, sf_vec_size, mma_tiler_mn, hidden_size,
                 intermediate_size, num_topk, **kwargs):
        if (mma_tiler_mn != (64, 128) or sf_vec_size != 16
                or hidden_size != 4096 or intermediate_size != 512 or num_topk != 8
                or kwargs.get("activation") != "swigluoai_uninterleave"
                or kwargs.get("swiglu_alpha") != 1.0
                or kwargs.get("swiglu_beta") != 0.0
                or kwargs.get("swiglu_limit") != 10.0
                or not m64_stock_contract_matches()):
            raise ValueError("M64 requires the pinned GLM gated kernel contract")
        super().__init__(sf_vec_size=sf_vec_size, mma_tiler_mn=(128, 128), **kwargs)
        self.tile_shape_mnk = (64, 128, 128)
        self.fc1_tile_shape_mnk = (64, 64, 128)
        self.epi_tile = (64, 128)


__all__ = ["MoEGatedDynamicKernelTiled", "MoEGatedDynamicKernelM64Tiled", "m64_stock_contract_matches"]
