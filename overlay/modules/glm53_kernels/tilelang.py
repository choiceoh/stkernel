# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# deneb fork (2026-08-31, glm53_mhc_tilelang): byte-identical to the
# glm53:v13-b12x image's vllm/model_executor/kernels/mhc/tilelang.py
# (preimage sha256 c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce)
# EXCEPT the mhc_fused_post_pre small-M (decode) launch heuristic, which the
# base itself marks "TODO(gnovack): investigate autotuning": tile_n/n_splits
# are env-overridable via VLLM_GLM53_MHC_SMALLM="tile_n,n_splits" (e.g. "6,4").
# Unset or invalid = the stock pair, so an env typo can never change behavior
# silently. Sweep harness: probes/mhc_glm53_bench.py. dsv4 precedent: the same
# TODO heuristic swept to (6,4) at M<8 for +16 pct on the dsv4 lane
# (dsv4_mhc_tilelang R1). Decode is the C=1 verdict channel here.
import os
from functools import cache


def _deneb_persist_tilelang_cache() -> None:
    """Point TileLang's JIT cache at the container's PERSISTENT mount.

    TileLang defaults TILELANG_CACHE_DIR to ~/.tilelang/cache, which is
    inside the container, so every restart re-JITs the MHC pair: the
    2026-09-02 boot spent 23:29:27-32 and 23:30:09-14 compiling
    mhc_pre_big_fuse_with_norm (two distinct JIT keys), ~10 s that a warm
    cache serves from disk. TRITON_CACHE_DIR and VLLM_CACHE_ROOT already
    live on the mount; this follows them.

    tilelang reads the variable on every access (its EnvVar descriptor is a
    live os.environ read), so setting it here -- at the import of the module
    that owns the MHC path -- lands before the first compile even if the
    tilelang package was imported earlier. An explicit setting always wins,
    and no writable mount leaves the default alone.
    """
    if os.environ.get("TILELANG_CACHE_DIR"):
        return
    for cand in ("/cache", "/root/.cache"):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            os.environ["TILELANG_CACHE_DIR"] = os.path.join(cand, "tilelang")
            return


_deneb_persist_tilelang_cache()


_HC_PRENORM_MIN_M = 8


def _deneb_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits):
    """37차: deep_gemm's tf32 prenorm GEMM, with M padded to 8 rows when it is
    smaller.

    The one place a served M below 8 has ever reached this GEMM is a k=5
    speculative boot (6 tokens per request); chain 13's K5 boot spun there
    (rank 0, CPU 200%, 19 min, no new JIT files) and the M=8/16/24 decode
    batches of k=7 and every prefill M (arbitrary, e.g. 27) run through it
    daily. So M < 8 -- and only that -- is served as the proven M=8 shape:
    zero rows appended, GEMM, the live rows copied back. Both outputs are
    row-wise (out = x @ fn^T, sqrsum = |x|^2 per row), so the padding rows are
    inert and never read. Cost at k=5, C=1: one 196 KB zero-copy and two
    ~30 KB copies per layer; at k=7 this branch is never taken.
    """
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    m = x.shape[0]
    if m >= _HC_PRENORM_MIN_M:
        tf32_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits)
        return
    x_pad = torch.zeros((_HC_PRENORM_MIN_M,) + tuple(x.shape[1:]),
                        dtype=x.dtype, device=x.device)
    x_pad[:m].copy_(x)
    mul_pad = torch.empty((out_mul.shape[0], _HC_PRENORM_MIN_M) + tuple(out_mul.shape[2:]),
                          dtype=out_mul.dtype, device=out_mul.device)
    sq_pad = torch.empty((out_sqrsum.shape[0], _HC_PRENORM_MIN_M),
                         dtype=out_sqrsum.dtype, device=out_sqrsum.device)
    tf32_hc_prenorm_gemm(x_pad, fn, mul_pad, sq_pad, n_splits)
    out_mul.copy_(mul_pad[:, :m])
    out_sqrsum.copy_(sq_pad[:, :m])

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# deneb fork: parsed once at import -- a frozen constant is capture-safe (the
# captured decode graph must not branch on an env read). Per-call validity is
# re-checked in _deneb_smallm_pair against the runtime shapes; any doubt there
# falls back to the stock heuristic.
_SMALLM_ENV = "VLLM_GLM53_MHC_SMALLM"


def _deneb_parse_smallm(raw: str):
    """Parse "tile_n,n_splits" -> (int, int), or None unless unambiguous."""
    try:
        a, b = (s.strip() for s in raw.split(","))
        tile_n, n_splits = int(a), int(b)
    except Exception:
        return None
    if tile_n <= 0 or n_splits <= 0:
        return None
    return tile_n, n_splits


_raw_smallm = (os.environ.get(_SMALLM_ENV) or "").strip()
_DENEB_SMALLM = _deneb_parse_smallm(_raw_smallm) if _raw_smallm else None


def _deneb_smallm_pair(num_tokens: int, hidden_size: int, hc_mult: int):
    """(tile_n, n_splits) for this step, or None to run the stock heuristic.

    The kernel's shape contracts, re-checked here because the constants live
    in the caller's frames: tile_n must divide n_out = hc_mult*(hc_mult+2)
    (n_tiles = n_out // tile_n), n_splits must be one the dispatcher's own
    assert admits, and every thread of the default n_thr=256 must own work
    (h_per_split % n_thr == 0, else the serial h-loop silently drops
    elements)."""
    if _DENEB_SMALLM is None:
        return None
    tile_n, n_splits = _DENEB_SMALLM
    if (hc_mult * (hc_mult + 2)) % tile_n:
        return None
    if n_splits not in (1, 2, 4, 8):
        return None
    if hidden_size % n_splits or (hidden_size // n_splits) % 256:
        return None
    return tile_n, n_splits


# deneb fork: prefill big_fuse h_blk override. dsv4 R3 precedent: h_blk=4096
# (single pipelined block) won +5.6% on this kernel family at M=4096 (with
# n_thr 96/160; GLM's stock 96 is in the winning set) while M<=64 is
# stock-optimal -- so the override only applies past 64 tokens. Frozen at
# import (capture-safe); the validator re-checks divisibility per call.
# VLLM_GLM53_MHC_BIGFUSE="h_blk" e.g. "4096". Unset/invalid = stock.
_BIGFUSE_ENV = "VLLM_GLM53_MHC_BIGFUSE"


def _deneb_parse_bigfuse(raw: str):
    """Parse "h_blk[,post_n_thr]" -> (h_blk, post_thr|None), else None.

    dsv4 R3: big_fuse h_blk=4096 with n_thr 96/160 (GLM stock 96 already in
    the winning set, so only h_blk is exposed). dsv4 R2: mhc_post prefill
    (n_thr 512, h_blk 4096) beat its stock (128, 1024) by +3.6% at M=4096 --
    post_thr is that second field."""
    try:
        parts = [int(v.strip()) for v in raw.split(",")]
    except Exception:
        return None
    if len(parts) == 1:
        h_blk, post_thr = parts[0], None
    elif len(parts) == 2:
        h_blk, post_thr = parts
    else:
        return None
    if h_blk not in (1024, 2048, 4096):
        return None
    if post_thr is not None and post_thr not in (128, 256, 512):
        return None
    return h_blk, post_thr


_raw_bigfuse = (os.environ.get(_BIGFUSE_ENV) or "").strip()
_DENEB_BIGFUSE = _deneb_parse_bigfuse(_raw_bigfuse) if _raw_bigfuse else None


# Prefill post-map + prenorm experiment. The stock post result is rounded to
# BF16 before its prenorm GEMM. The fused kernel retains that boundary while
# avoiding the GEMM's reread of the complete HC-expanded residual. It changes
# the dot/reduction schedule and requires GPU differential + serving gates.
_DENEB_PREFILL_POST_PRENORM = (
    os.environ.get("VLLM_GLM53_MHC_PREFILL_POST_PRENORM") == "1"
)


@cache
def _deneb_prefill_post_prenorm_device(device):
    return torch.cuda.get_device_capability(device) == (12, 1)


def _deneb_prefill_post_prenorm(
    comb, residual, post, x, fn, residual_out, gemm_out, sqrsum, n_splits,
):
    """Offer only the dense GLM prefill geometry to the fused experiment."""
    if not _DENEB_PREFILL_POST_PRENORM:
        return False
    if (
        residual.ndim != 3
        or tuple(residual.shape[1:]) != (4, 4096)
        or residual.shape[0] < 128
        or residual.device.type != "cuda"
        or n_splits < 1
        or n_splits > 64
    ):
        return False
    m = residual.shape[0]
    layouts = (
        (residual, (m, 4, 4096), torch.bfloat16),
        (residual_out, (m, 4, 4096), torch.bfloat16),
        (x, (m, 4096), torch.bfloat16),
        (post, (m, 4), torch.float32),
        (comb, (m, 4, 4), torch.float32),
        (fn, (24, 16384), torch.float32),
        (gemm_out, (n_splits, m, 24), torch.float32),
        (sqrsum, (n_splits, m), torch.float32),
    )
    if any(
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != residual.device
        or not tensor.is_contiguous()
        for tensor, shape, dtype in layouts
    ):
        return False
    if not _deneb_prefill_post_prenorm_device(residual.device):
        return False
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _mhc_prefill_post_prenorm_kernel,
    )

    # These outputs are fresh dispatcher allocations. Also reject aliases in
    # direct probe calls: another CTA can still be reading the original row.
    inputs = (residual, x, post, comb, fn)
    input_storages = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    outputs = (residual_out, gemm_out, sqrsum)
    output_storages = {tensor.untyped_storage().data_ptr() for tensor in outputs}
    if len(output_storages) != len(outputs) or input_storages & output_storages:
        return False
    _mhc_prefill_post_prenorm_kernel[((m + 15) // 16, n_splits)](
        comb, residual, post, x, fn, residual_out, gemm_out, sqrsum,
        m, HIDDEN=4096, NSPLITS=n_splits, BM=16, BK=64, BN=32,
        num_warps=4, num_stages=2,
    )
    return True


def _deneb_bigfuse_hblk(num_tokens: int, hidden_size: int):
    """h_blk for the prefill big_fuse kernels, or None to run stock."""
    if _DENEB_BIGFUSE is None or num_tokens <= 64:
        return None
    if hidden_size % _DENEB_BIGFUSE[0]:
        return None
    return _DENEB_BIGFUSE[0]


def _deneb_bigfuse_post(num_tokens: int):
    """(n_thr, h_blk) for the prefill mhc_post kernel, or None for stock."""
    if _DENEB_BIGFUSE is None or num_tokens <= 64:
        return None
    post_thr = _DENEB_BIGFUSE[1]
    if post_thr is None:
        return None
    return post_thr, _DENEB_BIGFUSE[0]


# deneb fork: one-launch decode path. VLLM_GLM53_MHC_ONEPASS=1 routes the
# small-M branch of mhc_fused_post_pre through mhc_onepass_tilelang (in our
# tilelang_kernels.py takeover) -- the FMA kernel and the big-fuse
# (mixes/sinkhorn/norm) kernel folded into one launch per layer, with the
# gemm_out global roundtrip gone. Frozen at import like its sibling knobs
# (the serving process sets env before import; capture then bakes the chosen
# branch). The per-call validator re-checks the kernel's shape contracts.
# Default off: unvalidated on GPU until probes/mhc_glm53_bench.py --onepass
# runs clean.
_ONEPASS_ENV = "VLLM_GLM53_MHC_ONEPASS"
_raw_onepass = (os.environ.get(_ONEPASS_ENV) or "").strip().lower()
_DENEB_ONEPASS = _raw_onepass in ("1", "true", "yes", "on")


def _deneb_onepass_enabled() -> bool:
    return _DENEB_ONEPASS


def _deneb_onepass_ok(hidden_size: int, hc_mult: int) -> bool:
    """Kernel contracts: one tile spans n_out, one warp writes it, and every
    thread owns exact work of the serial h-loop (h % n_thr == 0)."""
    n_out = hc_mult * (hc_mult + 2)
    return n_out <= 32 and hidden_size % 256 == 0


# deneb fork (glm53_megakernel): resolve the MK_SEG_MHC entry point ONCE.
# The hook sits on the decode hot path -- one call per layer per step -- and a
# `from ... import ...` there costs a sys.modules lookup plus two getattrs
# EVERY call, paid even while the segment is disarmed. This caches the
# resolved callable, or None when the module is not mounted: a boot without
# the megakernel is stock and stays stock without retrying the import.
_MK_MODULE = "vllm.model_executor.layers.glm53_megakernel"
_MK_HOOK = None
_MK_HOOK_TRIED = False


def _deneb_mk_hook():
    """The MK_SEG_MHC entry point, resolved at most once per process.

    A permanent answer is cached; a doubtful one is not. Every ImportError
    shape is permanent here -- the module is not mounted, or it is mounted
    without the entry point -- so both cache as None and the lane stays stock
    without paying an import per call. Only a non-import failure (a
    half-initialised package during warmup, a transient read on the bind
    mount) returns stock for THIS call and is retried on the next, because
    caching that would disable the segment for the life of the worker with
    nothing in the log to say so.
    """
    global _MK_HOOK, _MK_HOOK_TRIED
    if not _MK_HOOK_TRIED:
        try:
            from vllm.model_executor.layers.glm53_megakernel import mhc_hook
        except ImportError as e:
            # Both shapes are facts of THIS boot rather than transient: the
            # module is not mounted (ModuleNotFoundError naming it), or it IS
            # mounted without the entry point (a plain ImportError -- an older
            # core beside a newer wiring, which the core/wiring split made a
            # reachable deploy state). Cache both: retrying either would pay
            # an import per decode call for the life of the worker.
            if not isinstance(e, ModuleNotFoundError) or e.name == _MK_MODULE:
                _MK_HOOK, _MK_HOOK_TRIED = None, True
            return None
        except Exception:
            # anything else may be transient (a half-initialised package
            # during warmup): stay stock for THIS call and retry on the next
            return None
        _MK_HOOK, _MK_HOOK_TRIED = mhc_hook, True
    return _MK_HOOK


# 37차: the pre-only entry point (layer 0's standalone pre-mix), resolved the
# same way and cached separately -- a core without it disables THIS hook only.
_MK_PRE_HOOK = None
_MK_PRE_HOOK_TRIED = False


def _deneb_mk_pre_hook():
    global _MK_PRE_HOOK, _MK_PRE_HOOK_TRIED
    if not _MK_PRE_HOOK_TRIED:
        try:
            from vllm.model_executor.layers.glm53_megakernel import mhc_pre_hook
        except ImportError as e:
            if not isinstance(e, ModuleNotFoundError) or e.name == _MK_MODULE:
                _MK_PRE_HOOK, _MK_PRE_HOOK_TRIED = None, True
            return None
        except Exception:
            return None
        _MK_PRE_HOOK, _MK_PRE_HOOK_TRIED = mhc_pre_hook, True
    return _MK_PRE_HOOK


def _torch_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    assert out.shape[0] == 1
    assert sqrsum.shape[0] == 1
    x_float = x.float()
    out[0].copy_(x_float @ fn.t())
    sqrsum[0].copy_(x_float.square().sum(dim=-1))


def _tilelang_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hidden_size: int,
    hc_mult: int,
    tile_n: int = 12,
    n_thr: int = 512,
    n_splits: int = 1,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    assert out.shape[0] == n_splits
    assert sqrsum.shape[0] == n_splits
    assert x.shape[1] == hc_mult * hidden_size
    assert x.shape[1] % n_splits == 0
    assert (x.shape[1] // n_splits) % n_thr == 0
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            n_thr,
            tile_n,
            2,
        )
        return
    if (
        n_splits == 1
        and use_default_config
        and x.shape[0] < 128
        and x.shape[1] % 1024 == 0
    ):
        hc_prenorm_gemm_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            1024,
            4,
            n_splits,
        )
        return
    hc_prenorm_gemm_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        fn.shape[0],
        n_thr,
        tile_n,
        n_splits,
    )


def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # deneb fork (glm53_megakernel), 37차: layer 0's standalone pre-mix in
    # the MHC segment (the fused kernel under identity post coefficients),
    # the same T <= 16 window as the fused wrapper's hook. Every miss returns
    # None and falls through to the stock GEMM + big-fuse pair below, so a
    # disarmed boot is byte-identical to before; an armed launch is not
    # excepted into the stock path (async CUDA failures are uncontainable).
    if num_tokens <= 16 and norm_weight is not None:
        _mk_pre_hook = _deneb_mk_pre_hook()
        if _mk_pre_hook is not None:
            _mk_pre = _mk_pre_hook(
                residual_flat,
                fn.view(hc_mult3, hc_mult, hidden_size),
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_weight,
                norm_eps,
            )
            if _mk_pre is not None:
                _pm, _cm, _li = _mk_pre
                return (
                    _pm.view(*outer_shape, hc_mult, 1),
                    _cm.view(*outer_shape, hc_mult, hc_mult),
                    _li.view(*outer_shape, hidden_size),
                )

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        # these numbers are from deepgemm kernel impl
        block_k = 64
        block_m = 64
        n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))
    else:
        n_splits = 1

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        _deneb_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )

    _bf = _deneb_bigfuse_hblk(num_tokens, hidden_size)
    _bf_kw = {"h_blk": _bf} if _bf else {}
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            **_bf_kw,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            **_bf_kw,
        )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_pre_broadcast_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from ``(T, H)``."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert norm_weight is not None, "broadcast mHC pre currently requires fused RMSNorm"
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    residual_flat = residual
    num_tokens = residual.shape[0]

    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))

    residual_out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    _deneb_hc_prenorm_gemm(
        residual_flat,
        fn_broadcast,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    out = torch.empty_like(residual)
    _post_kw = {}
    # deneb fork: prefill-only retune (dsv4 R2); decode M never passes the gate
    _post = _deneb_bigfuse_post(residual.shape[0])
    if _post is not None:
        _post_kw = {"n_thr": _post[0], "h_blk": _post[1]}
    _mhc_post_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
        **_post_kw,
    )
    return out


def mhc_fused_post_pre_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        # TODO(gnovack): investigate autotuning these heuristics
        # deneb fork: VLLM_GLM53_MHC_SMALLM="tile_n,n_splits" overrides the
        # stock pair for the whole small-M branch when it passes the kernel's
        # shape contracts; anything doubtful falls back to stock.
        _tuned = _deneb_smallm_pair(num_tokens, hidden_size, hc_mult)
        if _tuned is not None:
            tile_n, n_splits = _tuned
        else:
            tile_n = 2 if num_tokens < 8 else 3
            n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4

    # deneb fork (glm53_megakernel): MK_SEG_MHC -- the same small-M fusion in
    # ONE persistent nvcc launch (48 blocks, no TileLang JIT for decode
    # shapes). Arms on the first eligible call, after a self-test that diffs
    # it against the stock pair below; every miss (module not mounted,
    # unarmed, shape, dtype) returns None and falls through, so a disarmed
    # boot is byte-identical to today. Takes precedence over ONEPASS when both
    # are set: it is the same fusion with fewer launches. The arm-then-call
    # contract lives in the core's `mhc_hook`, so this block is the same code
    # dsv4_mhc_tilelang carries -- two image forks, one hook.
    #
    # The window is the WRAPPER's, not the kernel's: this branch is under
    # `use_small_fma` (T <= 16) while the kernel gates at T <= 32, so a step's
    # C x (spec + 1) tokens reach it only at C <= 2. 16 < T <= 32 is the stock
    # post+big_fuse branch, which MK is never offered -- an open door,
    # unmeasured.
    if use_small_fma and norm_weight is not None:
        _mk_hook = _deneb_mk_hook()
        if _mk_hook is not None:
            # an armed shape LAUNCHES here and cannot be excepted into the
            # stock path (async CUDA failures are uncontainable); every
            # eligible miss returns None and falls through
            _mk = _mk_hook(
                x_flat,
                residual_flat,
                post_layer_mix_flat,
                comb_res_mix_flat,
                fn.view(hc_mult3, hc_mult, hidden_size),
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_weight,
                norm_eps,
            )
            if _mk is not None:
                return _mk

    # deneb fork: ONEPASS -- the whole small-M pair in one launch. Placed
    # before the gemm_out allocations because the fused kernel has no
    # gemm_out. Requires the fused norm (the with_norm half of big_fuse is
    # what it inlines); without norm_weight the stock path below runs.
    if (
        use_small_fma
        and _deneb_onepass_enabled()
        and norm_weight is not None
        and _deneb_onepass_ok(hidden_size, hc_mult)
    ):
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            mhc_onepass_tilelang,
        )

        residual_cur = torch.empty_like(residual_flat)
        post_mix_cur = torch.empty(
            num_tokens, hc_mult, dtype=torch.float32, device=residual.device
        )
        comb_mix_cur = torch.empty(
            num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
        )
        layer_input_cur = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        )
        mhc_onepass_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            hc_scale,
            hc_base,
            norm_weight,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
        )
        return (
            residual_cur.view(*outer_shape, hc_mult, hidden_size),
            post_mix_cur.view(*outer_shape, hc_mult, 1),
            comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
            layer_input_cur.view(*outer_shape, hidden_size),
        )
    # ONEPASS is only an optional early return.  The stock small-M path must
    # keep the 4/8-way split selected above when ONEPASS is disabled; running
    # the generic DeepGEMM planner here can produce a split outside this
    # dispatcher's supported set (48 on GB10) for the 256-thread kernel.
    if not use_small_fma:
        if use_deep_gemm:
            # these number are from deepgemm kernel impl
            block_k = 64
            block_m = 64
            n_splits = compute_num_split(
                block_k, hc_hidden_size, cdiv(num_tokens, block_m)
            )
        else:
            n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            n_splits=n_splits,
        )
    elif not _deneb_prefill_post_prenorm(
        comb_res_mix_flat, residual_flat, post_layer_mix_flat, x_flat,
        fn, residual_cur, gemm_out_mul, gemm_out_sqrsum, n_splits,
    ):
        _post_kw = {}
        _post = _deneb_bigfuse_post(num_tokens)
        if _post is not None:
            _post_kw = {"n_thr": _post[0], "h_blk": _post[1]}
        mhc_post_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
            **_post_kw,
        )

        residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
        if use_deep_gemm:
            from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

            _deneb_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
            )

    _bf = _deneb_bigfuse_hblk(num_tokens, hidden_size)
    _bf_kw = {"h_blk": _bf} if _bf else {}
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            **_bf_kw,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            **_bf_kw,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and return the (T, H) bf16 result."""
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if num_tokens == 0:
        return out
    from vllm.model_executor.kernels.mhc.tilelang_kernels import hc_head_fuse_tilelang

    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)

direct_register_custom_op(
    op_name="mhc_fused_post_pre_tilelang",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)

direct_register_custom_op(
    op_name="hc_head_fused_kernel_tilelang",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
