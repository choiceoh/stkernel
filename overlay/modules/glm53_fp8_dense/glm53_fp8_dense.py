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
first boot measures speed; freeing them is a possible follow-up, not a knob
yet). Any layer that fails a guard keeps its bf16 path, and any apply() error
drops that layer back permanently.

The GEMM runs behind ONE registered custom op (`glm53_fp8_dense::gemm`), the
same opacity trick this stack's own `+quant_fp8` ops use: inductor sees an
opaque node instead of tracing into the deepgemm/triton launches (the raw
`per_token_group_quant_fp8_packed_for_deepgemm` / `fp8_gemm_nt` pair is what
fp8_lm_head runs OUTSIDE the compiled region -- calling them bare from decoder
linears, which are inside it, would graph-break at every projection).

Merged-projection padding: the KDA layers load q/k/v/b/f/g as one
`in_proj_qkvbfg_a` whose row count is not a multiple of the 128 block (nor is
its TP shard), and the merged `gate_up_proj` shards land wherever TP puts
them. The quantized copy zero-pads rows and columns out to the block grid:
column-parallel matrices slice the extra output rows off (reshape, not view --
the sliced stride is not view-compatible when M > 1), row-parallel ones get
exactly-zero contributions from the zero rows. Padding is always < 128, so no
all-zero block ever forms and no scale degenerates. No loader or checkpoint
change; the bf16 weight is untouched.

Armed by VLLM_GLM53_FP8_DENSE: 0 off, 1/true W8A8, w4a8 the fp4-weight arm
(activations stay fp8). Rollback is the env alone. VLLM_GLM53_FP8_DENSE_BPROJ
(1/true, default off) extends the pattern set with the low-rank b projections
and the indexer query projection -- a track-record-free extension, so it
opt-ins instead of riding the default (#110).
"""

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# Runtime module names, i.e. what the loader merged the checkpoint tensors
# into -- not the checkpoint tensor names. `mlp.gate_up_proj` matches only the
# first-k dense layers (MoE layers keep their projections under
# `mlp.experts`/`mlp.shared_experts`, which these patterns do not match), and
# the low-rank pieces of the KDA merge ride inside `in_proj_qkvbfg_a`, whose
# padded copy covers them.
# The shared expert's pair runs on the aux stream beside the routed MoE
# (glm5next's MoE runner forks it); the megakernel lane treats those two
# launches as background (its local-quant kernel on as many blocks as it
# has units, VLLM_GLM53_MK_LOCALQ=1) -- hence the name.
_SHARED_EXPERT_RE = re.compile(r"\.mlp\.shared_experts\.(gate_up_proj|down_proj)$")
_INCLUDE = (
    re.compile(
        r"\.self_attn\.(in_proj_qkvbfg_a|fused_qkv_a_proj|q_proj|k_proj"
        r"|v_proj|o_proj|out_proj)$"),
    _SHARED_EXPERT_RE,
    re.compile(r"\.mlp\.(gate_up_proj|down_proj)$"),
)

# The remaining bf16 GEMMs of a decode step (STEP_KERNEL_MAP #108 section 2:
# 145 cutlass_80_wmma launches): the rear halves of the low-rank projections
# the checkpoint leaves bf16. Per-rank shapes from the glm5next sources --
# q_b_proj [4096, 1536], kv_b_proj [4096, 512], indexer.wq_b [4096, 1536]
# (replicated, read in full on every rank). All three clear the
# min(shape) >= 512 guard below; the byte win is ~160 MB/step at C=1
# (~0.9 pct), which is why this arm exists -- but the indexer GEMMs are
# aux-stream contention-stretched rather than bandwidth-bound, so the honest
# expectation is under that. Opt-in via VLLM_GLM53_FP8_DENSE_BPROJ: #110's
# lesson is that an extension with no track record never rides the default.
# Deliberately NOT matched: f_b_proj/g_b_proj ([2048, 128] per rank -- the
# 512 guard would skip them anyway and the byte win is 17 MB/step), and
# wk_weights_proj (the loader upcasts it to bf16 to keep the wk+weights_proj
# fusion; quantizing it would break that contract).
_BPROJ_INCLUDE = (
    re.compile(r"\.self_attn\.(q_b_proj|kv_b_proj)$"),
    re.compile(r"\.indexer\.wq_b$"),
)

_BPROJ_ON = ("1", "true", "yes", "on")

# The DFlash2 drafter (Qwen3-style, glm53_dflash_loader_fp8 calls this pass
# under its own knob) names its projections differently from the target:
# a MERGED qkv_proj (the base patterns list q/k/v_proj and the target's
# fused names, never qkv_proj), the aux-hidden fc ([hidden, 5 x hidden],
# ReplicatedLinear -- K = 20480, read whole on every rank), and two conv
# kernel_projections per layer ([1024, hidden], replicated). With the base
# patterns alone the drafter knob covered o_proj/gate_up/down and left
# 43% of the drafter's bytes bf16 -- the fc alone is 23% (armed 09-03
# trace: fc 792 us, qkv 5 x ~50 us, kernel_projection 10 x ~60 us per
# step). These extend the set for the drafter knob only; the target's
# pattern set is untouched.
_DRAFTER_ENV = "VLLM_DFLASH2_FP8_DENSE"
_DRAFTER_INCLUDE = (
    re.compile(r"\.self_attn\.qkv_proj$"),
    re.compile(r"(^|\.)fc$"),
    re.compile(r"\.(attention_conv|mlp_conv)\.kernel_projection$"),
)

# vLLM keys its torch.compile cache -- and under VLLM_USE_AOT_COMPILE=1 the
# whole AOT artifact -- on the env vars registered in vllm.envs, the vllm
# config and the forward's source, then loads the artifact with guard
# checks disabled. A quant_method swapped in AFTER load is no part of that
# key. So every boot with the drafter knob on served the drafter from the
# artifact of the first boot that ever compiled it (09-03: bf16 F.linear on
# all 30 layer projections), and only the eager fc reached the lane -- the
# bracket measured a 1-of-31 candidate (28차). Registering the knob makes
# each value its own artifact; the getter folds the off-spellings together
# so "0", "", "off" share one.
_DRAFTER_OFF = ("", "0", "false", "no", "off")


def _drafter_knob_value() -> str:
    raw = (os.environ.get(_DRAFTER_ENV) or "0").strip().lower()
    return "0" if raw in _DRAFTER_OFF else raw


def _register_compile_factor(env: str, getter) -> bool:
    """Make `env` part of vLLM's torch.compile / AOT cache key.

    vllm.envs.compile_factors() hashes every entry of
    vllm.envs.environment_variables (minus an ignore list) by calling its
    getter, so an entry added here is hashed like a stock knob. Idempotent;
    False when vllm.envs is not importable (offline probes)."""
    try:
        import vllm.envs as _envs

        table = _envs.environment_variables
    except Exception:
        return False
    if env not in table:
        table[env] = getter
    return True


_register_compile_factor(_DRAFTER_ENV, _drafter_knob_value)


def _spec_k_value() -> str:
    # num_speculative_tokens (the launcher forwards SPEC_K): the drafter's
    # compiled graphs are shaped by it, and a K=5 boot's artifacts killed the
    # next K=7 boot ('expected size 7==5', 29차). Unset = "7" (the default).
    return (os.environ.get("VLLM_GLM53_SPEC_K") or "7").strip()


_register_compile_factor("VLLM_GLM53_SPEC_K", _spec_k_value)


def _include_patterns(env: str = "VLLM_GLM53_FP8_DENSE") -> tuple:
    """Base patterns, plus the drafter set under the drafter knob, plus the
    b-projection arm when its gate is armed."""
    pats = _INCLUDE
    if env == _DRAFTER_ENV:
        pats = pats + _DRAFTER_INCLUDE
    raw = (os.environ.get("VLLM_GLM53_FP8_DENSE_BPROJ") or "").strip().lower()
    if raw in _BPROJ_ON:
        pats = pats + _BPROJ_INCLUDE
    return pats


def _quantize_fp8_block_padded(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Block-quantize to the deepgemm ue8m0 layout, zero-padded to 128s.

    Pads in the weight's own dtype and only floats each row-chunk, so the
    fp32 staging copy never exceeds ~1/8 of the weight (same discipline as
    fp8_lm_head). Returns (q, scales, orig_rows, orig_cols)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    with torch.no_grad():
        w = weight.detach()
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
                w[r0 : r0 + step].float(), [128, 128], use_ue8m0=True
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


def _fp8_dense_gemm(
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
    # view-compatible on the unpadded fast path, copies only for the < 2 pct
    # of linears whose rows needed padding (the KDA merged in_proj).
    return out.reshape(x.shape[:-1] + (orig_rows,))


def _quantize_w4(weight: torch.Tensor, packed_sf: bool):
    """Pack a bf16 [N, K] weight to deep_gemm's e2m1 layout, per-row over K.

    per_token_cast_to_fp4 pads K to the 128 scale granularity internally and
    returns packed [N, K//2] at the ORIGINAL K, so -- unlike the fp8 block
    path -- no row/column padding or output slicing is needed here: every
    linear this module touches has an even K that is already a multiple of
    128 on the activation side.

    The vendored kernel accepts (at least) two scale layouts and its C++
    checks are not readable from here -- packed ue8m0 int32 vs plain float
    -- so the caller probes both and keeps whichever passes the value
    check."""
    from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_e8m0_used

    dg = _import_deep_gemm()
    w = weight.detach()
    packed, sf = dg.per_token_cast_to_fp4(
        w.float(),
        use_ue8m0=is_deep_gemm_e8m0_used(),
        gran_k=128,
        use_packed_ue8m0=packed_sf,
    )
    return packed, sf


# nvfp4 blocks are 16 wide and the scale is e4m3, not the ue8m0 the fp8 path
# uses; the format's max magnitude is 6.0 (e2m1's top grid point).
_NVFP4_BLOCK = 16
_NVFP4_MAX = 6.0
# Lever 7 (28차): the nvfp4 pair as a PREFILL route on the fp8 method. Rows
# above the MK lane's M (MAX_TOK = 32) take mm_fp4 (2.3x the fp8 GEMM at
# prefill shapes, half the weight bytes); decode keeps the W4 lane. Knob-
# gated -- A4 activations are a served-numerics change (quality 9/9, Korean
# 0/16, acceptance profile, prefill ladder) -- and target-only.
_PREFILL_NVFP4_ENV = "VLLM_GLM53_FP8_DENSE_PREFILL_NVFP4"
_NVFP4_PREFILL_MIN_M = 32


def _prefill_nvfp4_enabled(env: str) -> bool:
    if env != "VLLM_GLM53_FP8_DENSE":
        return False
    raw = (os.environ.get(_PREFILL_NVFP4_ENV) or "0").strip().lower()
    return raw in ("1", "true", "on", "yes")
# NOT "auto". auto selects among backends per shape and JIT-compiles what it
# picks. Measured on this part: the first shape costs 73.5 s to compile and
# every shape after it is 0.1-0.3 ms, so ONE pinned backend is one compile for
# all ~180 linears, and the launcher's FLASHINFER_WORKSPACE_BASE=/cache keeps
# even that across boots. "cutlass" was the fastest of the backends that
# accept these shapes (236.6 TFLOP/s at N=4096).
_NVFP4_BACKEND = os.environ.get("VLLM_GLM53_NVFP4_BACKEND", "cutlass")

# Which way the two global scales divide back out is a property of the
# vendored kernel, not of a layer, so it is resolved once and reused. Per
# layer it meant two real kernel launches on every one of ~180 linears,
# inside a build pass that already died once for want of host memory.
_NVFP4_ALPHA: list = [None]


def _nvfp4_global_scale(t: torch.Tensor) -> torch.Tensor:
    """The per-tensor scale that puts this tensor's amax at the format's top.

    e4m3 block scales top out at 448 and the values at 6.0, so the product is
    what the block scale has to reach for the largest block."""
    amax = t.abs().amax().float().clamp_min(1e-12)
    return (448.0 * _NVFP4_MAX / amax).view(1)


def _quantize_nvfp4(weight: torch.Tensor):
    """[N, K] bf16 -> (packed e2m1 [N, K//2], e4m3 block scales, global scale).

    Same format the checkpoint already uses for the MoE experts, on the dense
    projections its recipe left in higher precision. Worth 2.3x on the prefill
    GEMM (236 vs 104 TFLOP/s measured) and half the pack, against 3.7x the
    quantization error -- which is why this scheme is knob-gated and arms only
    on a value check that ran."""
    from flashinfer import nvfp4_quantize

    with torch.no_grad():
        w = weight.detach()
        gs = _nvfp4_global_scale(w)
        packed, sf = nvfp4_quantize(w, gs)
        return packed, sf, gs


def _nvfp4_dense_gemm(
    x: torch.Tensor,
    wq: torch.Tensor,
    wsf: torch.Tensor,
    w_gs: torch.Tensor,
    out_rows: int,
    alpha_scale: float,
) -> torch.Tensor:
    """Both operands in nvfp4 -- that is where the 2x lives.

    Quantizing only the weight leaves the GEMM running at the fp8 issue rate
    (that is the existing w4a8 arm), so the activation is quantized per call,
    dynamically, the way the checkpoint's recipe does it for the experts."""
    from flashinfer import mm_fp4, nvfp4_quantize

    flat = x.reshape(-1, x.shape[-1])
    x_gs = _nvfp4_global_scale(flat)
    xq, xsf = nvfp4_quantize(flat.to(torch.bfloat16), x_gs)
    # Both operands carry their global scale into the values, so the product
    # has to be divided back out once, in the epilogue.
    alpha = (alpha_scale / (x_gs * w_gs)).to(torch.float32)
    out = torch.empty(
        flat.shape[0], out_rows, dtype=torch.bfloat16, device=flat.device
    )
    mm_fp4(xq, wq.T, xsf, wsf.T, alpha, torch.bfloat16, out,
           _NVFP4_BLOCK, False, _NVFP4_BACKEND)
    return out.reshape(x.shape[:-1] + (out_rows,))


def _fp8_fp4_dense_gemm(
    x: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_e8m0_used

    flat = x.reshape(-1, x.shape[-1])
    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        flat.to(torch.bfloat16), 128
    )
    out = torch.empty(
        flat.shape[0], wq.shape[0], dtype=torch.bfloat16, device=flat.device
    )
    _import_deep_gemm().fp8_fp4_gemm_nt(
        (xq, xs),
        (wq, ws),
        out,
        disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
    )
    return out.reshape(x.shape[:-1] + (wq.shape[0],))


# One opaque boundary per GEMM: compile sees a node, capture sees kernel
# launches, and the deepgemm/triton interiors are never traced. Falls back to
# the bare function on re-import (double registration) or an old torch.
try:
    _fp8_dense_gemm_op = torch.library.custom_op(
        "glm53_fp8_dense::gemm", mutates_args=()
    )(_fp8_dense_gemm)

    @_fp8_dense_gemm_op.register_fake
    def _fp8_dense_gemm_fake(
        x, q, ws, orig_rows: int, orig_cols: int
    ) -> torch.Tensor:
        return torch.empty(
            x.shape[:-1] + (orig_rows,), dtype=torch.bfloat16,
            device=x.device,
        )
except Exception:
    _fp8_dense_gemm_op = _fp8_dense_gemm


# Calls of the opaque op, counted in Python: a compiled graph executes the
# op's Python body, a CUDA-graph replay executes nothing here. The drafter
# serving proof (install_drafter_serving_check) reads it per forward.
_OPAQUE_CALLS = 0


def _mk_or_fp8_dense_gemm(
    x: torch.Tensor,
    q: torch.Tensor,
    ws: torch.Tensor,
    orig_rows: int,
    orig_cols: int,
    wq4s: list[torch.Tensor],
    ws4s: list[torch.Tensor],
    gscales: list[float],
) -> torch.Tensor:
    """The megakernel W4 lane when it is armed and the shape is eligible,
    the deepgemm fp8 pair otherwise -- decided at RUN time, inside one
    opaque op.

    Fp8DenseMethod.apply makes the same choice in Python for the target,
    whose forward is eager. The DFlash2 drafter's forward is torch.compiled
    (@support_torch_compile on DFlashQwen3Model), and there the Python
    choice is not an option: dynamo would trace the lane's eligibility test
    (a guard on the token count), the arming import and the extension call
    -- a pybind function it cannot trace. So for a compiled caller the whole
    decision moves behind this boundary, where it runs exactly as it does in
    eager: the lane can DISARM at run time and the fallback is the fp8 pair
    the copy check verified. wq4s/ws4s/gscales carry the MK pack -- one
    entry, or one per K-chunk of a linear wider than the lane's K (the
    drafter's fc) -- and empty lists mean fp8 only.
    """
    global _OPAQUE_CALLS
    _OPAQUE_CALLS += 1
    if wq4s:
        _mk_gemm = _mk_arm = None
        try:  # import only: a boot without the megakernel module is fp8
            from vllm.model_executor.layers.glm53_megakernel import (
                gemm_w4a8 as _mk_gemm,
                maybe_arm as _mk_arm,
            )
        except Exception:
            pass
        if _mk_gemm is not None:
            _mk_arm()
            packs = [(a, b, g) for a, b, g in zip(wq4s, ws4s, gscales)]
            out = _mk_gemm(x, packs[0] if len(packs) == 1 else packs,
                           orig_rows)
            if out is not None:
                return out
    return _fp8_dense_gemm(x, q, ws, orig_rows, orig_cols)


try:
    _mk_or_fp8_dense_gemm_op = torch.library.custom_op(
        "glm53_fp8_dense::gemm_mk_or_fp8", mutates_args=()
    )(_mk_or_fp8_dense_gemm)

    @_mk_or_fp8_dense_gemm_op.register_fake
    def _mk_or_fp8_dense_gemm_fake(
        x, q, ws, orig_rows: int, orig_cols: int, wq4s, ws4s, gscales
    ) -> torch.Tensor:
        return torch.empty(
            x.shape[:-1] + (orig_rows,), dtype=torch.bfloat16,
            device=x.device,
        )
except Exception:
    _mk_or_fp8_dense_gemm_op = _mk_or_fp8_dense_gemm


def _stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def install_drafter_serving_check(model, expected: int, forwards: int = 8) -> None:
    """Prove that the drafter's served graph reaches the opaque op.

    Wraps `model.forward` for its first `forwards` calls and counts opaque
    GEMM calls per forward. A CUDA-graph replay runs no Python and counts
    0, so a forward is judged only when it ran Python: it made calls, or it
    was recorded under stream capture (capture executes the compiled
    graph's ops, so it is definitive and judged at once). A judged forward
    below half of `expected` means the served graph bypasses the swapped
    quant_method -- the stale compile artifact of 28차 -- and is reported
    once, as a WARNING the boot fingerprint carries. The wrapper removes
    itself after the verdict."""
    if expected <= 0 or forwards <= 0:
        return
    orig = model.forward
    state = {"left": forwards, "low": None, "done": False}

    def _report(seen):
        state["done"] = True
        model.__dict__.pop("forward", None)
        if seen is None:
            logger.warning(
                "[fp8-dense] drafter lane: no forward ran Python in the "
                "first %d calls -- SERVING UNKNOWN", forwards)
        elif seen * 2 < expected:
            logger.warning(
                "[fp8-dense] drafter lane NOT SERVING: %d of %d opaque GEMM "
                "calls in a forward -- the served graph bypasses the swapped "
                "quant_method (stale compile artifact?)", seen, expected)
        else:
            logger.warning(
                "[fp8-dense] drafter lane serving: %d of %d opaque GEMM "
                "calls per forward", seen, expected)

    def forward(*args, **kwargs):
        if state["done"]:
            return orig(*args, **kwargs)
        capturing = _stream_capturing()
        before = _OPAQUE_CALLS
        out = orig(*args, **kwargs)
        seen = _OPAQUE_CALLS - before
        state["left"] -= 1
        if seen > 0 or capturing:
            state["low"] = seen if state["low"] is None else min(state["low"], seen)
            if capturing or seen * 2 < expected:
                _report(state["low"])
                return out
        if state["left"] <= 0:
            _report(state["low"])
        return out

    model.forward = forward


try:
    _w4_dense_gemm_op = torch.library.custom_op(
        "glm53_fp8_dense::gemm_w4a8", mutates_args=()
    )(_fp8_fp4_dense_gemm)

    @_w4_dense_gemm_op.register_fake
    def _w4_dense_gemm_fake(x, wq, ws) -> torch.Tensor:
        return torch.empty(
            x.shape[:-1] + (wq.shape[0],), dtype=torch.bfloat16,
            device=x.device,
        )
except Exception:
    _w4_dense_gemm_op = _fp8_fp4_dense_gemm


try:
    _nvfp4_dense_gemm_op = torch.library.custom_op(
        "glm53_fp8_dense::gemm_nvfp4", mutates_args=()
    )(_nvfp4_dense_gemm)

    @_nvfp4_dense_gemm_op.register_fake
    def _nvfp4_dense_gemm_fake(
        x, wq, wsf, w_gs, out_rows: int, alpha_scale: float
    ) -> torch.Tensor:
        return torch.empty(
            x.shape[:-1] + (out_rows,), dtype=torch.bfloat16,
            device=x.device,
        )
except Exception:
    _nvfp4_dense_gemm_op = _nvfp4_dense_gemm


class NvFp4DenseMethod:
    """Both operands in nvfp4: 2.3x the fp8 GEMM, 3.7x its error.

    `_base` is the layer's fp8 METHOD, so a runtime failure drops one notch to
    W8A8 rather than to bf16 -- the same ladder W4A8DenseMethod uses, and for
    the same reason: never let a scheme choice fail a boot.

    The error is the whole story here. The checkpoint's own recipe put fp4 on
    `mlp.experts.*` and left these projections alone, so this scheme goes past
    what its authors were willing to quantize. Offline norms cannot settle
    whether that was necessary or conservative -- the bracket does."""

    def __init__(self, base, wq, wsf, w_gs, out_rows, alpha_scale):
        self._base = base
        self._wq, self._wsf, self._gs = wq, wsf, w_gs
        self._rows = out_rows
        self._alpha = alpha_scale

    def apply(self, layer, x, bias=None):
        if bias is not None:
            return self._base.apply(layer, x, bias)
        if getattr(self, "_bf16_freed", False):
            return _nvfp4_dense_gemm_op(
                x, self._wq, self._wsf, self._gs, self._rows, self._alpha
            )
        try:
            return _nvfp4_dense_gemm_op(
                x, self._wq, self._wsf, self._gs, self._rows, self._alpha
            )
        except Exception:
            layer.quant_method = self._base
            return self._base.apply(layer, x, bias)


class W4A8DenseMethod:
    """Same contract as Fp8DenseMethod, weights one notch lower.

    Activations stay fp8 -- the axis the literature blesses (QServe et al.:
    W4A4 loses 20-25 pct, W4A8 is the compromise) -- and the kernel family
    is the one already carrying the MoE experts (fp8_fp4_gemm_nt, the dense
    form of sm120_fp8_fp4_gemm_1d1d). The calling convention is probed
    eagerly at build time, so a probe failure never reaches this class.

    `_base` is the layer's fp8 METHOD, not bf16: a runtime failure drops one
    notch to W8A8. Memory while armed is triple residency per layer (bf16
    source + fp8 fallback pair + fp4 pair, ~+1 GB/rank over the W8A8 arm)
    -- the price of never being able to fail a boot."""

    def __init__(self, base, wq, ws):
        self._base = base
        self._wq, self._ws = wq, ws

    def apply(self, layer, x, bias=None):
        if bias is not None:
            return self._base.apply(layer, x, bias)
        if getattr(self, "_bf16_freed", False):
            return _w4_dense_gemm_op(x, self._wq, self._ws)
        try:
            return _w4_dense_gemm_op(x, self._wq, self._ws)
        except Exception:
            layer.quant_method = self._base
            return self._base.apply(layer, x, bias)


class Fp8DenseMethod:
    """Drop-in `quant_method` routing one Linear through the fp8 copy.

    Holds the original method for bias and error fallbacks; an apply() failure
    restores it for good, so the worst case is the bf16 path we started on."""

    def __init__(self, base, q, ws, orig_rows, orig_cols):
        self._base = base
        self._q, self._ws = q, ws
        self._rows, self._cols = orig_rows, orig_cols
        # deneb fork (glm53_megakernel): MK_SEG_GEMM pack (W4: e2m1 nibbles
        # + per-16-group exponents, never aliased with the deepgemm pair).
        # None until maybe_build_fp8_dense attaches one; apply() then keeps
        # stock.
        self._mk = None
        # True for a torch.compiled caller (the DFlash2 drafter): apply()
        # then routes through ONE opaque op that makes the MK-or-fp8 choice
        # at run time (_mk_or_fp8_dense_gemm), instead of the Python
        # choice below that dynamo cannot trace.
        self._opaque = False
        self._mk_args = None
        # (wq, wsf, w_gs, rows, alpha): the nvfp4 pair for prefill rows, or
        # None. Built by maybe_build_fp8_dense under _PREFILL_NVFP4_ENV.
        self._nvfp4 = None

    def apply(self, layer, x, bias=None):
        if bias is not None:
            return self._base.apply(layer, x, bias)
        if self._opaque:
            args = self._mk_args
            if args is None:
                mk = self._mk
                if mk is None:
                    args = ([], [], [])
                elif isinstance(mk, list):
                    args = ([p[0] for p in mk], [p[1] for p in mk],
                            [float(p[2]) for p in mk])
                else:
                    args = ([mk[0]], [mk[1]], [float(mk[2])])
                self._mk_args = args
            return _mk_or_fp8_dense_gemm_op(
                x, self._q, self._ws, self._rows, self._cols, *args
            )
        # Lever 7: prefill rows on the nvfp4 pair when one was built. The MK
        # lane below owns M <= 32, so the pair is never read for decode. A
        # failure drops this layer's prefill to the fp8 pair for the rest of
        # the boot -- loudly, and on every rank the same way (the fp8 pair
        # is shape-agnostic, so ranks cannot diverge on it).
        nv = self._nvfp4
        if nv is not None and x.numel() // x.shape[-1] > _NVFP4_PREFILL_MIN_M:
            try:
                return _nvfp4_dense_gemm_op(x, *nv)
            except Exception as e:
                self._nvfp4 = None
                logger.warning(
                    "[fp8-dense] nvfp4 prefill pair failed (M=%d): %r -> fp8 "
                    "pair for the rest of the boot",
                    x.numel() // x.shape[-1], e)
        # deneb fork (glm53_megakernel): one persistent 48-block launch for
        # decode M<=32 (quant fused into the GEMM). Ineligible shapes return
        # None and run the stock pair below. No try/except around an armed
        # launch: the boot self-test is the gate, failures stay loud.
        if self._mk is not None:
            _mk_gemm = _mk_arm = None
            try:  # import only: a boot without the megakernel module is stock
                from vllm.model_executor.layers.glm53_megakernel import (
                    gemm_w4a8 as _mk_gemm,
                    maybe_arm as _mk_arm,
                )
            except Exception:
                pass
            if _mk_gemm is not None:
                _mk_arm()
                _out = _mk_gemm(x, self._mk, self._rows,
                                bg=getattr(self, "_mk_bg", False))
                if _out is not None:
                    return _out
        if getattr(self, "_bf16_freed", False):
            # No net below: the bf16 source was released, so failing loudly
            # here beats reading an empty tensor or diverging from the other
            # ranks. See maybe_free_fp8_dense_bf16.
            return _fp8_dense_gemm_op(
                x, self._q, self._ws, self._rows, self._cols
            )
        try:
            return _fp8_dense_gemm_op(
                x, self._q, self._ws, self._rows, self._cols
            )
        except Exception:
            layer.quant_method = self._base
            return self._base.apply(layer, x, bias)


# A copy whose source moved under it is served for the life of the boot: once
# quant_method is swapped, apply() reads q/ws and the bf16 tensor is never
# touched again. So verify each copy against the tensor it claims to stand for.
# Block-fp8 (ue8m0, 128x128) lands within a few percent of bf16; a stale copy
# is a different matrix and misses by far more, so the two are nowhere near
# each other and the threshold does not need to be tight.
_STALE_RTOL = 0.25


def _copy_matches_source(mod, method, weight, rtol=None, got_fn=None):
    """True/False when the check ran, None when it could not.

    None keeps the layer armed. Refusing to arm because a probe would not
    execute trades a measured speedup for an unmeasured worry; False is the
    only outcome that disarms, and it means the copy really did miss."""
    try:
        x = torch.randn(
            8, weight.shape[1], dtype=weight.dtype, device=weight.device
        )
        ref = method._base.apply(mod, x, None)
        # got_fn MUST be supplied when it matters: calling method.apply()
        # here routes through its OWN error fallback, which swallows the
        # failure, silently swaps layer.quant_method mid-build, returns the
        # fallback's output -- and the copy then "matches" its reference
        # exactly. A direct GEMM callable makes a broken kernel surface as
        # an exception (-> None) or garbage (-> False) instead.
        got = (got_fn(x) if got_fn is not None
               else method.apply(mod, x, None))
        den = ref.float().norm()
        if not torch.isfinite(den) or den == 0:
            return None
        tol = _STALE_RTOL if rtol is None else rtol
        return bool(((got.float() - ref.float()).norm() / den) <= tol)
    except Exception:
        return None



_FREE_BF16_ENV = "VLLM_GLM53_FP8_DENSE_FREE_BF16"


def _free_bf16_enabled() -> bool:
    """Exact opt-in for `maybe_free_fp8_dense_bf16`.

    After `quant_method` is swapped, apply() reads only the fp8 copy (and the
    megakernel's W4 pack), so the bf16 tensor is dead weight -- 2.94 GB per
    rank at TP4, against 1.47 GB of fp8 pack and 0.82 GB of W4 pack. Freeing
    it hands that straight to the KV cache.

    What it costs: the two paths that still reach `self._base.apply` -- a
    bias, and an exception in the fp8 kernel -- would then read an empty
    tensor. No linear this module admits has a bias (checked against the
    checkpoint: the only matching `.bias` entries are in the vision tower,
    which this profile disables), and the error path is the same "a boot
    self-test is the gate" trade the megakernel segments already make. Still
    default OFF and knob-gated, because it is not recoverable at runtime.
    """
    return os.environ.get(_FREE_BF16_ENV, "").strip() == "1"


def maybe_free_fp8_dense_bf16(model, label: str = "") -> int:
    """Release the bf16 sources. Call ONLY when weight loading is finished.

    This cannot live inside `maybe_build_fp8_dense`. That function is written
    to tolerate being called early -- `AutoWeightsLoader` enters a child
    `load_weights` before the checkpoint is walked, and the comment at its
    call site says so -- and a later call rebuilds any premature copy. A
    rebuild repairs a stale copy; it cannot repair a deleted source. Freeing
    from inside the build made the early call destructive and killed a boot:

        parameter.py:221 in load_row_parallel_weight
          shard_size = self.data.shape[self.input_dim]
        IndexError: tuple index out of range

    -- the loader still had rows to place and found a 1-D empty tensor where
    a [N, K] weight had been. So the release is its own pass, driven from
    `GPUModelRunner.load_model` after both the model and the drafter have
    loaded, inside the `DeviceMemoryProfiler` block so the weight figure it
    reports is the one KV sizing then works from.

    Returns the bytes released (0 when the knob is off)."""
    if not _free_bf16_enabled():
        return 0
    freed = 0
    kept_bias = 0
    for mod in model.modules():
        method = getattr(mod, "quant_method", None)
        if not isinstance(method, (Fp8DenseMethod, W4A8DenseMethod, NvFp4DenseMethod)):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or weight.numel() == 0:
            continue
        if getattr(mod, "bias", None) is not None:
            # bias still routes to _base.apply, which reads this tensor
            kept_bias += 1
            continue
        freed += weight.numel() * weight.element_size()
        mod.weight.data = torch.empty(
            0, dtype=weight.dtype, device=weight.device)
        # apply() must stop treating _base as a fallback for this layer: it
        # would read the empty tensor and produce a wrong-shaped result
        # instead of an error. Worse, the fp8 path fails on SHAPE, so it can
        # fail on one rank and not another -- the ranks then take different
        # code paths, the collectives no longer line up, and the step hangs
        # rather than raising. A boot went that way on 2026-09-03: generation
        # fell 44.9 -> 12.8 -> 0 tok/s and the head timed out on
        # "RPC call to sample_tokens", with no traceback anywhere.
        method._bf16_freed = True
    logger.warning(
        "[fp8-dense] %s%s=1: released %.2f GB of bf16 sources (%d linears kept "
        "theirs for a bias); the bias and error fallbacks through the base "
        "method are gone with them",
        (label + ": ") if label else "", _FREE_BF16_ENV, freed / 1e9, kept_bias)
    return freed

def _host_mem_available() -> str:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return "%.1f GiB" % (int(line.split()[1]) / 2**20)
    except Exception:
        pass
    return "?"


def _nvfp4_pair_for(mod, method, weight, rows, name, seen):
    """The nvfp4 pair for a PREFILL route on an fp8 method, or None.

    Same discipline as the nvfp4 scheme: the alpha (dequant) convention is
    settled once on the first layer that can carry it, by whichever sign
    reproduces bf16; then every layer reuses it and a SAMPLE (the first
    four, then one in 16) re-runs the value check. A layer whose check
    fails keeps the fp8 pair for prefill."""
    try:
        wq, wsf, w_gs = _quantize_nvfp4(weight)
    except Exception as e:
        logger.warning("[fp8-dense] nvfp4 prefill pair skipped on %s: %r", name, e)
        return None
    if _NVFP4_ALPHA[0] is None:
        for cand in (1.0, -1.0):
            try:
                if _copy_matches_source(
                        mod, NvFp4DenseMethod(method, wq, wsf, w_gs, rows, cand),
                        weight, rtol=4 * _STALE_RTOL,
                        got_fn=lambda xx: _nvfp4_dense_gemm(
                            xx, wq, wsf, w_gs, rows, cand),
                ) is True:
                    _NVFP4_ALPHA[0] = cand
                    break
            except Exception:
                continue
        logger.warning(
            "[fp8-dense] nvfp4 alpha=%s backend=%s (resolved on %s for the "
            "prefill pair, reused for the rest)",
            _NVFP4_ALPHA[0], _NVFP4_BACKEND, name)
    alpha = _NVFP4_ALPHA[0]
    if alpha is None:
        return None
    if seen < 4 or seen % 16 == 0:
        try:
            ok = _copy_matches_source(
                mod, NvFp4DenseMethod(method, wq, wsf, w_gs, rows, alpha),
                weight, rtol=4 * _STALE_RTOL,
                got_fn=lambda xx: _nvfp4_dense_gemm(
                    xx, wq, wsf, w_gs, rows, alpha),
            ) is not False
        except Exception:
            ok = False
        if not ok:
            logger.warning("[fp8-dense] nvfp4 prefill pair refused on %s (value check)", name)
            return None
    return (wq, wsf, w_gs, rows, alpha)


def _attach_mk_pack(method, weight, cols) -> bool:
    """Attach the megakernel's W4 pack to an fp8 method that will SERVE.

    deneb fork (glm53_megakernel): W4 is e2m1 x per-16 pow2 scale, expanded
    to EXACT e4m3 in-kernel, ~0.56x the fp8 bytes -- on every eligible
    linear, the KDA in_proj included. The fp8 MK arm and its knob are gone,
    so VLLM_GLM53_MK_GEMM=1 means W4 numerics on the decode linears: bracket
    before arming (megakernel README). A build failure only means apply()
    keeps the deepgemm pair, which is the fallback either way.

    Called only once the layer's serving method is settled. It used to run
    before the copy check and before the scheme branch, so a stale layer and
    -- worse -- EVERY layer the nvfp4 arm then took paid for a pack that
    NvFp4DenseMethod.apply never reads: ~1.0 GB/rank of weights built,
    verified and held for nothing, at the peak of the boot that OOM-killed
    srv3 on 2026-09-03.
    """
    try:
        from vllm.model_executor.layers import glm53_megakernel as _mkmod

        # every lane that reads the pack: the standalone GEMM and the two
        # fused-MLP lanes (their hook finds the pack on the linear; an armed
        # segment whose linears carry no pack serves stock in silence)
        wants = (_mkmod.ENABLE_GEMM or getattr(_mkmod, "ENABLE_SMLP", False)
                 or getattr(_mkmod, "ENABLE_SMLP2", False))
        if wants and cols % 128 == 0:
            if cols > _mkmod.MK_GEMM_KMAX:
                # wider than one launch of the lane (the drafter's fc,
                # K = 5 x hidden): a pack per K-chunk, summed by gemm_w4a8
                method._mk = _mkmod.build_mk_weight_w4_kchunks(weight)
            else:
                method._mk = _mkmod.build_mk_weight_w4(weight)
            return method._mk is not None
    except Exception:
        method._mk = None
    return False


def _kda_owns(model, name: str) -> bool:
    """True when MK-KDA will serve this linear inside its own launch.

    The KDA block fuses in_proj/o_proj into the kernel, so for those two the
    layer's quant_method is never called at all -- whichever scheme won it
    is dead weight, and the W4 pack it does read must exist. Ownership is
    decided here, while the bf16 source is still alive: _kda_ensure_packs
    runs on the first eager forward, which is AFTER
    maybe_free_fp8_dense_bf16, so a pack it has to build itself would read
    an emptied tensor.
    """
    try:
        from vllm.model_executor.layers import glm53_megakernel as _mkmod

        if not (_mkmod.ENABLE_KDA or _mkmod.KDA_SHADOW):
            return False
    except Exception:
        return False
    if "." not in name:
        return False
    leaf = name.rsplit(".", 1)[1]
    if leaf not in ("in_proj_qkvbfg_a", "o_proj"):
        return False
    try:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
    except Exception:
        return False
    # o_proj also names the attention block's output projection; the KDA one
    # is the sibling of in_proj_qkvbfg_a.
    return hasattr(parent, "in_proj_qkvbfg_a")


def maybe_build_fp8_dense(model, env: str = "VLLM_GLM53_FP8_DENSE") -> bool:
    """Quantize the selected dense projections of a loaded model in place.

    `env` names the arming knob so the target and the DFlash2 drafter can be
    gated independently: the target changes served output, the drafter changes
    candidate logits -- acceptance is its quality probe.

    Safe to call more than once: a copy an earlier call installed is REBUILT
    from the current weights, not skipped. Call this again once the checkpoint
    is fully loaded and the last call wins -- an early caller cannot leave a
    stale copy behind."""
    raw = (os.environ.get(env) or "0").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    # w4a8: weights one notch lower on the same kernel family
    # (fp8_fp4_gemm_nt, dense form of the MoE expert kernel); activations
    # stay fp8 -- the axis the literature blesses. 1/true keeps W8A8.
    if raw in ("nvfp4", "fp4x4", "w4a4"):
        # both operands in nvfp4 -- the only shape that gets the 2x
        scheme = "nvfp4"
    elif raw in ("w4a8", "w4", "fp4"):
        scheme = "w4a8"
    else:
        scheme = "w8a8"
    # One lane below fp8: the MK W4 pack rides every fp8 copy when MK-GEMM
    # is armed. The fp8-only (no pack) arm that briefly existed for
    # the drafter bracket is gone -- the bracket picked W4 and the operator's
    # rule is that a proven improvement becomes the default and the other
    # side is removed, not kept as a second setting to remember.
    attach_mk = _attach_mk_pack
    quantized, quantized_w4, skipped, stale, params, params_w4 = (
        [], [], [], [], 0, 0)
    mk_packs = 0
    kda_owned = 0
    shapes: dict = {}   # (N, K) -> count, for the boot log: which launches are which
    prefill_nv = _prefill_nvfp4_enabled(env) and scheme == "w8a8"
    nv_prefill = 0
    for name, mod in model.named_modules():
        if not any(p.search(name) for p in _include_patterns(env)):
            continue
        weight = getattr(mod, "weight", None)
        base = getattr(mod, "quant_method", None)
        # Re-arming: unwrap a copy already installed so this call quantizes
        # today's weights and replaces it. The low-precision schemes stack on
        # the fp8 method rather than on bf16 (so a runtime failure drops one
        # notch, not two), so unwrap until the original method reappears --
        # one step would leave an fp8 method here, and the
        # UnquantizedLinearMethod test below would then skip the layer and
        # keep serving the copy this call was meant to replace.
        while isinstance(
            base, (Fp8DenseMethod, W4A8DenseMethod, NvFp4DenseMethod)
        ):
            base = base._base
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
            shapes[tuple(weight.shape)] = shapes.get(tuple(weight.shape), 0) + 1
            q, ws, rows, cols = _quantize_fp8_block_padded(weight)
            method = Fp8DenseMethod(base, q, ws, rows, cols)
            method._mk_bg = bool(_SHARED_EXPERT_RE.search(name))
            # the drafter's forward is torch.compiled: its GEMMs must be
            # one opaque op each (see _mk_or_fp8_dense_gemm)
            method._opaque = env == _DRAFTER_ENV
            if _copy_matches_source(
                mod, method, weight,
                got_fn=lambda xx: _fp8_dense_gemm_op(xx, q, ws, rows, cols),
            ) is False:
                mod.quant_method = base
                stale.append(name)
                continue
            if _kda_owns(model, name):
                # MK-KDA serves this projection inside its kernel, so the
                # dense scheme does not bid for it: an nvfp4/w4a8 copy here
                # would be built, verified and never read. What MK-KDA does
                # read is the W4 pack, and it must be attached now.
                if attach_mk(method, weight, cols):
                    mk_packs += 1
                    kda_owned += 1
                mod.quant_method = method
                quantized.append(name)
                params += weight.numel()
                continue
            if scheme == "nvfp4" and weight.shape[1] % _NVFP4_BLOCK == 0:
                # Same discipline as w4a8 below: an experimental scheme arms
                # only on a value check that actually RAN, against the bf16
                # tensor it claims to stand for. The tolerance is 4x the
                # stale threshold because nvfp4's by-design error on these
                # projections is 1.3e-1 (measured, probes/nvfp4_dense_
                # accuracy.py) while an uncorrelated result lands near
                # sqrt(2) -- the two are still nowhere near each other.
                #
                # alpha carries both global scales back out of the product.
                # Which way round that division goes is a property of the
                # vendored kernel's dequant convention, not something this
                # file can read, so both are tried and whichever reproduces
                # bf16 is kept -- exactly how the w4a8 scale layout is
                # settled.
                # alpha_scale carries a DEQUANT convention, not a quantizer
                # input -- _quantize_nvfp4 never reads it. Quantizing inside
                # the retry loop therefore recomputed the identical triple
                # and, because the right-hand side is built before the names
                # are rebound, held two of them at once. Hoisted: one
                # quantization, one triple, both attempts.
                try:
                    wq, wsf, w_gs = _quantize_nvfp4(weight)
                except Exception:
                    wq = None
                # The convention is settled once, on the first layer that
                # can carry the scheme; after that every layer reuses it and
                # only a SAMPLE re-runs the kernel. A wrong convention is
                # wrong everywhere, and a stale copy is already caught by the
                # fp8 check above, which every layer still gets.
                if wq is not None and _NVFP4_ALPHA[0] is None:
                    for cand in (1.0, -1.0):
                        try:
                            if _copy_matches_source(
                                    mod,
                                    NvFp4DenseMethod(
                                        method, wq, wsf, w_gs, rows, cand),
                                    weight, rtol=4 * _STALE_RTOL,
                                    got_fn=lambda xx: _nvfp4_dense_gemm(
                                        xx, wq, wsf, w_gs, rows, cand),
                            ) is True:
                                _NVFP4_ALPHA[0] = cand
                                break
                        except Exception:
                            continue
                    logger.warning(
                        "[fp8-dense] nvfp4 alpha=%s backend=%s (resolved on "
                        "%s, reused for the rest)",
                        _NVFP4_ALPHA[0], _NVFP4_BACKEND, name)
                armed_nv = False
                if wq is not None and _NVFP4_ALPHA[0] is not None:
                    alpha_scale = _NVFP4_ALPHA[0]
                    nv = NvFp4DenseMethod(
                        method, wq, wsf, w_gs, rows, alpha_scale)
                    seen = len(quantized_w4) + len(quantized)
                    ok = True
                    if seen < 4 or seen % 16 == 0:
                        try:
                            ok = _copy_matches_source(
                                mod, nv, weight, rtol=4 * _STALE_RTOL,
                                got_fn=lambda xx: _nvfp4_dense_gemm(
                                    xx, wq, wsf, w_gs, rows, alpha_scale),
                            ) is not False
                        except Exception:
                            ok = False
                    if ok:
                        # NvFp4DenseMethod.apply goes straight to the nvfp4
                        # kernel: it never reads the MK pack, so this layer
                        # must not build one.
                        mod.quant_method = nv
                        quantized_w4.append(name)
                        params_w4 += weight.numel()
                        armed_nv = True
                if not armed_nv:
                    if attach_mk(method, weight, cols):
                        mk_packs += 1
                    mod.quant_method = method
                    quantized.append(name)
                    params += weight.numel()
                continue
            if scheme == "w4a8" and weight.shape[1] % 2 == 0:
                # Stricter than the fp8 path on purpose: an EXPERIMENTAL
                # scheme arms only on a value check that actually RAN and
                # passed (fp8 arms on "did not fail"). The check runs the
                # real kernel with a random probe batch, so a wrong scale
                # layout produces garbage and refuses to arm; 2x the stale
                # tolerance absorbs e2m1's by-design quantization error
                # (measured 0.02-0.08 rel on row blocks) while uncorrelated
                # garbage lands near sqrt(2).
                for packed_sf in (True, False):
                    try:
                        wq, ws = _quantize_w4(weight, packed_sf=packed_sf)
                        w4_method = W4A8DenseMethod(method, wq, ws)
                        if _copy_matches_source(
                                mod, w4_method, weight,
                                rtol=2 * _STALE_RTOL,
                                got_fn=lambda xx: _fp8_fp4_dense_gemm(
                                    xx, wq, ws),
                        ) is True:
                            mod.quant_method = w4_method
                            quantized_w4.append(name)
                            params_w4 += weight.numel()
                            break
                    except Exception:
                        continue
                else:
                    if attach_mk(method, weight, cols):
                        mk_packs += 1
                    mod.quant_method = method
                    quantized.append(name)
                    params += weight.numel()
                    continue
                continue
            if attach_mk(method, weight, cols):
                mk_packs += 1
            if prefill_nv and weight.shape[1] % _NVFP4_BLOCK == 0:
                pair = _nvfp4_pair_for(
                    mod, method, weight, rows, name,
                    len(quantized_w4) + len(quantized))
                if pair is not None:
                    method._nvfp4 = pair
                    nv_prefill += 1
                if nv_prefill % 20 == 1:
                    # memory watch: the first NVFP4P boot took srv3 from
                    # ~25 GiB free to 5 GiB (earlyoom) inside this pass
                    free, total = torch.cuda.mem_get_info()
                    logger.warning(
                        "[fp8-dense] nvfp4 prefill pair #%d (%s): cuda free "
                        "%.1f/%.1f GiB, host MemAvailable %s",
                        nv_prefill, name, free / 2**30, total / 2**30,
                        _host_mem_available())
            mod.quant_method = method
            quantized.append(name)
            params += weight.numel()
        except Exception as e:
            logger.warning("[fp8-dense] %s stayed bf16: %r", name, e)
            skipped.append(name)
        finally:
            # Return the transient to the driver after EVERY linear. On the
            # unified-memory GB10 a block the caching allocator keeps
            # reserved is host memory the node has lost: the 09-04
            # instrumented boot measured torch reserved +14.9 GiB after the
            # first linear (in_proj, 52 MB bf16) and +4.8 GiB after the
            # second, with allocated flat -- transients of the per-linear
            # steps that the allocator held instead of reusing -- and the
            # sum over the pass took every node from ~50 GiB available to
            # under the 4 GiB kernel watermark ~25 s into weight loading.
            # That was the 09-03 srv3 OOM and both 09-04 wedges; with
            # earlyoom (09-04) it became a deterministic SIGKILL. Freeing the
            # cache per linear bounds the pass at ONE transient at a time.
            torch.cuda.empty_cache()
    logger.warning(
        "[fp8-dense] %s (knob %s=%s): %d linears w4a8 (%.2f GB bf16), "
        "%d linears w8a8 (%.2f GB bf16), %d kept bf16, %d disarmed by the "
        "copy check, %d MK W4 packs (%d held for MK-KDA), %d nvfp4 prefill "
        "pairs%s -- fingerprint for the boot log",
        type(model).__name__, env, scheme,
        len(quantized_w4), params_w4 * 2 / 1e9,
        len(quantized), params * 2 / 1e9, len(skipped), len(stale), mk_packs,
        kda_owned, nv_prefill,
        "; skipped: " + ", ".join(skipped[:8]) if skipped else "",
    )
    if shapes:
        # [N x K] x count: the map from a decode trace's launch classes to
        # linears (28차: the 30-45 us class is [1024 x 4096] and [4096 x 512])
        logger.warning("[fp8-dense] %s shapes: %s", type(model).__name__,
                       ", ".join(f"[{n}x{k}]x{c}" for (n, k), c in sorted(shapes.items(), key=lambda kv: -kv[1])))
    if mk_packs == 0 and quantized_w4:
        # The scheme that wins the layer decides which kernel serves it, and
        # nothing else said so: NvFp4DenseMethod/W4A8DenseMethod.apply never
        # reach the MK path, so an nvfp4 or w4a8 arm turns MK-GEMM off for
        # every layer it takes. Silence here is how "armed" stops meaning
        # "serving" -- the same trap the MK-KDA layout gate was.
        try:
            from vllm.model_executor.layers import glm53_megakernel as _mkmod

            if _mkmod.ENABLE_GEMM:
                logger.warning(
                    "[fp8-dense] %s: VLLM_GLM53_MK_GEMM=1 but the %s arm took "
                    "every eligible linear, so NO layer carries an MK W4 pack "
                    "-- MK-GEMM will not run on the dense projections this "
                    "boot. MK-KDA keeps its own two projections either way, "
                    "and MK-MHC / MK-MLA never read a pack; the exclusion is "
                    "MK-GEMM vs this scheme, per layer.",
                    type(model).__name__, scheme)
        except Exception:
            pass
    if stale:
        logger.warning(
            "[fp8-dense] %s: %d copies did not reproduce their bf16 source "
            "and were reverted: %s -- a copy was taken before the weights "
            "landed",
            type(model).__name__, len(stale), ", ".join(stale[:8]),
        )
    return bool(quantized or quantized_w4)
