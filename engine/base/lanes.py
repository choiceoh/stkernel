"""The engine's default lanes (base): the model-free kernels every profile inherits.

engine/kernels/common holds the kernels whose arguments are the shape. This table is where a profile
takes them from: `served()` binds them once per process, and a profile's own lane table starts from
these and replaces a lane only where its model computes something else. GLM-5.3's target MLP uses
its own clamped activation (engine/kernels/glm_pointwise), so its `swiglu` lane is the profile's; its
drafter's plain gated MLP is this one.

Three common kernels need no entry: engine/base/sampler calls the sampler and the block verification
for every profile, and engine/modules/vocab calls the vocabulary candidates. The native build cache is
a build step, not a lane. What is shared is kernels, not a model's implementation (CHARTER D6): the
pipeline that calls `commit` stays the profile's.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class CommonLanes:
    rmsnorm: object       # (x [..., D], w [D], eps) -> RMS norm over the last dimension, w in the input's dtype
    add_rmsnorm: object   # (a, b, w, eps) -> (a + b, rmsnorm(a + b)) in one launch
    rmsnorm_rope: object  # (x [N, heads, D], w [D], eps, positions [N], theta) -> rope(rmsnorm(x)) in one launch
    rope_table: object    # (device, dim, theta) -> the inverse frequencies, built once; call before capture
    swiglu: object        # (fused [rows, 2 * inter]) -> [rows, inter]: silu(gate) * up, bit-identical to the torch pair
    commit: object        # (picks [n, t], state, accepted=None) -> (count, done, accepted, tokens, ctx_before)


@cache
def served() -> CommonLanes:
    """The common kernels, bound once. Importing them imports Triton. The norms and SwiGLU take their torch
    forms on CPU tensors, as their modules do; `commit` is CUDA only (a profile's CPU path commits in torch)."""
    from engine.kernels.common.decode_commit import advance
    from engine.kernels.common.norm_rope import add_norm, norm, norm_rope, warm
    from engine.kernels.common.swiglu import swiglu
    return CommonLanes(rmsnorm=norm, add_rmsnorm=add_norm, rmsnorm_rope=norm_rope, rope_table=warm,
                       swiglu=swiglu, commit=advance)
