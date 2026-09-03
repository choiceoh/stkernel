# SPDX-License-Identifier: Apache-2.0
"""deneb fork: GLM-5.3-Flash decode megakernel driver (glm53_megakernel.cu).

Three persistent segments for the GB10 decode step, each replacing a
between-collectives run of launches with ONE 48-block launch:

  MK_SEG_MHC   hc fused post+pre (takes over mhc_fused_post_pre_tilelang's
               small-M branch from glm53_mhc_tilelang/tilelang.py)
  MK_SEG_GEMM  W8A8 skinny GEMM, quant fused (Fp8DenseMethod.apply hook in
               glm53_fp8_dense.py)
  MK_SEG_KDA   the whole linear-attention block (kda.py overlay hook)

Arm policy (the osar/w4a8 lessons, applied):
  * every knob defaults OFF; VLLM_GLM53_MEGAKERNEL=1 is the master gate and
    each segment needs its own VLLM_GLM53_MK_<SEG>=1 on top
  * arming requires the device to be exactly cc 12.1 with 48 SMs, and a
    boot SELF-TEST against the stock path per segment (torch.cuda.synchronize
    before the verdict -- a python try/except cannot contain an async CUDA
    launch failure, so nothing arms on an unverified kernel)
  * hooks return None (stock path) for every shape outside the contract;
    an armed hook does NOT try/except around its own launch -- failures are
    loud by design
  * MK_KDA supports a shadow mode: eager steps run BOTH paths and log
    divergence (states included) while stock stays the real output; graph
    steps stay stock. The state-index contract is the open item shadow
    exists to close (README).

The CUDA side keeps no host-mutated device state: the grid barrier is a
never-reset monotonic counter in the workspace held here, so CUDA-graph
replay with baked pointers is exact (the osar done_ctr trick).
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import time

logger = logging.getLogger("vllm.glm53.megakernel")

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "glm53_megakernel.cu")

# Geometry this driver arms for. The kernel constants must match; refusing
# up front is cheaper than a self-test drift hunt.
HC = 4
HIDDEN = 4096
NOUT = HC * (2 + HC)
MAX_TOK = 32
NCHUNK = 16
KDA_H, KDA_D = 16, 128
KDA_QKV = 3 * KDA_H * KDA_D                 # 6144
KDA_INPROJ_N = KDA_QKV + KDA_H + 2 * KDA_D  # 6416
KDA_OUT = KDA_H * KDA_D                     # 2048
KDA_CONV_W = 4                              # short_conv_kernel_size
# Production conv_state width (mamba_utils.kda_state_shape:291):
# conv_kernel_size - 1 + num_spec. The spec headroom is not optional --
# causal_conv1d_update slides its window across the draft-verify tokens and
# reads past a width-(conv_kernel_size - 1) buffer.
KDA_SPEC = 7                                # num_speculative_tokens
KDA_CONV_STATE_W = KDA_CONV_W - 1 + KDA_SPEC  # 10


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


MASTER = _flag("VLLM_GLM53_MEGAKERNEL")
ENABLE_MHC = MASTER and _flag("VLLM_GLM53_MK_MHC")
ENABLE_GEMM = MASTER and _flag("VLLM_GLM53_MK_GEMM")
ENABLE_KDA = MASTER and _flag("VLLM_GLM53_MK_KDA")
KDA_SHADOW = MASTER and _flag("VLLM_GLM53_MK_KDA_SHADOW")
ENABLE_MLA = MASTER and _flag("VLLM_GLM53_MK_MLA")
# MK-GEMM is the W4 arm: e2m1 weights x per-16-group pow2 scale, expanded
# to EXACT e4m3 bytes in-kernel, on EVERY eligible decode linear (the KDA
# in_proj included -- there is no fp8 MK arm to fall back to; the W8 arm
# was removed once W4 beat the stock pair on every shape, so the lane is
# W4 or stock). Arming it changes served numerics: bracket first (README).

# tolerances. The W4 GEMM's by-design (e2m1) error class is 0.02-0.08 rel
# on row blocks; the exact-grid gate below it is 1e-5.
_TOL_GEMM_W4 = 0.15
_TOL_MHC = 1e-3     # fp32 port of the TileLang pair, bf16 rounding only
_TOL_KDA = 2e-2     # fixture (grid-snapped weights): fp8/activation noise only
# The serving shadow diffs the MK arm (W4 packs of the layer's bf16 weights)
# against stock (its fp8 blocks of the same weights): e2m1's by-design error
# (0.02-0.08 rel on row blocks) is inside that diff, so the gate is the
# by-design class. Drift above it is a real fault, not quantization.
_TOL_KDA_SHADOW = 0.15


def _mk_pow2_scale(amax: float) -> float:
    """2^frexp_exp(amax/448): always >= amax/448, so |scaled| <= 448.

    Pure twin of mk_pow2_scale() in the .cu; the build-side weight quant and
    the kernel-side activation quant agree on this rule to the bit.
    """
    if amax <= 0.0 or not math.isfinite(amax):
        return 1.0
    return float(math.ldexp(1.0, math.frexp(amax / 448.0)[1]))


def _mk_pad128(n: int) -> int:
    return -(-n // 128) * 128


def _mk_gemm_eligible(m: int, k: int, n_pad: int) -> bool:
    """MK_SEG_GEMM shape contract (decode M only; prefill stays deepgemm)."""
    return (0 < m <= 32 and k % 128 == 0 and 0 < k <= 4096
            and n_pad % 128 == 0)


def _mk_mhc_eligible(num_tokens: int, hc_mult: int, hidden: int) -> bool:
    return (0 < num_tokens <= 32 and hc_mult == 4 and hidden == 4096
            and hc_mult * hidden == 16384)


# ---------------------------------------------------------------------------
# extension + workspace (built lazily on first ARM, never on import)
# ---------------------------------------------------------------------------
_EXT = None
_WS = None
_ARMED = {"mhc": False, "gemm": False, "kda": False, "mla": False}


def _build_dir(src_md5: str, flags: list) -> str:
    """Where nvcc builds the extension -- on the PERSISTENT cache mount when
    the container has one.

    The compile is 34.4 s (ninja log of the 2026-09-02 boot) and it used to
    land in /root/.mk_build, inside the container, so every restart paid it
    again: 34 s of a 9-minute boot for a .cu that changes on deploys, not on
    restarts. TRITON_CACHE_DIR and VLLM_CACHE_ROOT already point at the host
    mount; this follows them.

    The directory name is a hash of everything the built .so is only valid
    for: the source, the nvcc flags (the probe's knobs change them), and the
    torch / CUDA pair (an image bump changes the ABI). Any of those moving
    picks a fresh directory and an honest recompile -- never a stale .so.
    """
    import torch

    root = os.environ.get("VLLM_GLM53_MK_BUILD_ROOT")
    if not root:
        for cand in ("/cache", "/root/.cache"):
            if os.path.isdir(cand) and os.access(cand, os.W_OK):
                root = os.path.join(cand, "mk_build")
                break
        else:
            root = "/root/.mk_build"  # no mount: container-local, as before
    key = hashlib.md5("|".join(
        [src_md5, *flags, torch.__version__, str(torch.version.cuda)]
    ).encode()).hexdigest()[:12]
    path = os.path.join(root, key)
    os.makedirs(path, exist_ok=True)
    # Siblings are builds of other sources or flag sets, ~2 MB each. Anything
    # untouched for a week is a deploy or a probe sweep that is not coming
    # back; a failure to prune is not a reason to fail the build.
    try:
        cutoff = time.time() - 7 * 86400
        for name in os.listdir(root):
            stale = os.path.join(root, name)
            if stale != path and os.path.isdir(stale) \
                    and os.path.getmtime(stale) < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass
    return path


def _build():
    import torch
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    with open(_SRC, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()[:8]
    with open(_SRC, encoding="utf-8", errors="replace") as f:
        n_kernels = sum(1 for line in f
                        if line.lstrip().startswith("__global__"))
    logger.warning("[megakernel] source md5=%s kernels=%d (%s)",
                   md5, n_kernels, _SRC)
    # -arch=sm_121a is NOT enough here. nvcc accepts it for -cubin and -ptx, but
    # with -c (what cpp_extension uses to build the objects) it emits a plain
    # sm_121 target and ptxas then rejects every ARCHITECTURE-SPECIFIC
    # instruction: the fp4 mma (kind::f8f6f4) and the 2:4 sparse mma
    # (mma.sp::ordered_metadata) both fail with "not supported on .target
    # 'sm_121'". -gencode arch=compute_121a,code=sm_121a keeps the 'a' suffix
    # through -c. Nothing in the kernels needs those two today (e4m3/bf16 mma
    # exist on plain sm_121), so this changes no generated code -- it is what
    # lets the file USE them. Verified 2026-09-03 with a 2x3 flag/mode matrix.
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"] + [
        # Swept by the probe; the .cu carries the shipped defaults.
        f"-DMK_GRID_DEF={os.environ.get('VLLM_GLM53_MK_GRID', '96')}",
        f"-DMK_MHC_GRID_DEF={os.environ.get('VLLM_GLM53_MK_MHC_GRID', '144')}",
        f"-DMK_NBUF_DEF={os.environ.get('VLLM_GLM53_MK_NBUF', '3')}",
    ] + (["-DMK_PHASE_TS=1"]
         if os.environ.get("VLLM_GLM53_MK_PHASE_TS") == "1" else [])
    _EXT = load(
        name="glm53_megakernel",
        sources=[_SRC],
        extra_cuda_cflags=flags,
        build_directory=_build_dir(md5, flags),
        verbose=False,
    )
    return _EXT


def _ensure_workspace(device):
    """One static workspace, strongly held (the b12x lesson: prefill cache
    growth must never invalidate CUDA-graph addresses baked at capture)."""
    global _WS
    import torch

    if _WS is not None:
        return _WS
    z = lambda *s, dt=torch.float32: torch.zeros(  # noqa: E731
        *s, dtype=dt, device=device)
    _WS = {
        "barrier": z(8, dt=torch.int32),
        # mhc needs its OWN counter: the ticket barrier computes
        # (t / grid + 1) * grid, which is only correct if the counter
        # is grid-aligned when the launch starts. Sharing one counter
        # between a 96-block grid and a 48-block grid misaligns it
        # (5 x 48 = 240, and 240 % 96 = 48), and a misaligned mhc
        # launch releases after 48 of its 96 blocks arrive.
        "barrier_mhc": z(8, dt=torch.int32),
        # and MK_SEG_MLA its own: its grid is 96 blocks where the gemm/kda
        # kernels run 48, and the ticket barrier only releases correctly
        # when the counter is aligned to THIS launch's grid.
        "barrier_mla": z(8, dt=torch.int32),
        "yp": z(NCHUNK * MAX_TOK * NOUT),
        "rp": z(NCHUNK * MAX_TOK),
        # [NCHUNK][MAX_TOK]: p3 stores one sumsq per (chunk, token) and
        # p4 reduces them in a fixed order -- see the note in mk_mhc_p3.
        "sq": z(NCHUNK * MAX_TOK),
        "pmix": z(MAX_TOK * HC),
        "ol_stash": z(MAX_TOK * HIDDEN, dt=torch.bfloat16),
        "qkv": z(MAX_TOK * KDA_INPROJ_N, dt=torch.bfloat16),
        "g1": z(MAX_TOK * KDA_OUT, dt=torch.bfloat16),
        "g2": z(MAX_TOK * KDA_OUT, dt=torch.bfloat16),
        "convq": z(MAX_TOK * KDA_QKV, dt=torch.bfloat16),
        "attn": z(MAX_TOK * KDA_OUT, dt=torch.bfloat16),
    }
    return _WS


def _barrier_ptr(ws):
    return ws["barrier"].data_ptr()


# ---------------------------------------------------------------------------
# weight quant for MK_SEG_GEMM / MK_SEG_KDA. Own layout: e4m3 + fp32 pow2
# block scales, rows padded to 128 with zeros. deepgemm's packed ue8m0 layout
# is deliberately NOT reused -- the stock pair stays a byte-identical
# fallback and the two never alias.
# ---------------------------------------------------------------------------
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _mk_w4_scale_exp(amax: float) -> int:
    """Scale exponent s for one 16-group: the smallest 2^s with
    6 * 2^s >= amax, clamped to the kernel LUT's exactness range [-5, 6]
    (exp field 6+s stays in [1, 15); never denormal, never NaN, never the
    sign bit). frexp alone is RIGHT-biased on exact power-of-two
    boundaries (frexp(8) -> 0.5*2^4), so an on-grid group -- amax = 6*2^e,
    e.g. the fixture weights -- would pick e+1 and quantize to a
    renormalized encoding, breaking the on-grid exactness gate. Subtract
    the boundary case back. The price of exponent-field arithmetic: s=6
    caps the representable magnitude at 6*2^6 = 384 < e4m3's 448, so a
    group with amax > 384 SATURATES at 384 (bucketize clips the code);
    the saturation is the documented ceiling, not an overflow."""
    if amax <= 0.0 or not math.isfinite(amax):
        return 0
    e = math.frexp(amax / 6.0)[1]  # frexp: right-biased on exact pow2
    if amax / 6.0 == 2.0 ** (e - 1):  # exact boundary -> one less suffices
        e -= 1
    return max(-5, min(6, e))


def build_mk_weight_w4(weight):
    """(wq4 uint8 [n_pad/128, k/128, 128, 64] nibbles,
        ws4 int8 [n_pad/128, k/128, 128, 8] exponents) -- tile-major.

    Nibble layout: low nibble = even element, high = odd. Round-to-nearest
    on the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} x 2^s. The kernel expands
    each nibble to an exact e4m3 byte, so the mma sees precisely these
    values -- nothing extra is lost downstream of the quantization.
    """
    import torch

    n, k = weight.shape
    if k % 128 != 0:
        raise ValueError(f"K={k} not a multiple of 128")
    n_pad = _mk_pad128(n)
    g = torch.zeros(n_pad, k // 16, 16, dtype=torch.float32,
                    device=weight.device)
    g[:n] = weight.float().view(n, k // 16, 16)
    amax = g.abs().amax(-1)
    s = torch.zeros_like(amax)
    pos = amax > 0
    # vectorized frexp ceil: exponent of the pow2 >= amax/6 (same rule as
    # _mk_w4_scale_exp; the .clamp guard keeps frexp finite)
    ratio = (amax / 6.0).clamp(min=1e-30)
    frac, exp = torch.frexp(ratio)
    # exact pow2 boundary -> frexp over-picks by one (mirror of the scalar
    # _mk_w4_scale_exp rule; keeps on-grid weights bit-identical to their
    # crafted encoding)
    onb = (ratio == torch.exp2(exp - 1.0))
    s[pos] = (exp[pos] - onb[pos].float()).clamp(-5, 6)
    qs = g * torch.exp2(-s).unsqueeze(-1)
    code = torch.bucketize(qs.abs(), torch.tensor(_E2M1_MIDS,
                                                   device=weight.device))
    sign = (qs < 0).to(torch.uint8)
    # Pack PAIRS along k: even k-index -> low nibble, odd -> high. The pair
    # slice must run on the [.., 16] group view. (Found in review: slicing a
    # [.., 8] re-view instead picked elements 0,2,4,6 of each octet AND left
    # numel 4x the target, so reshape raised -- the attach's except then
    # swallowed it into a silently-stock boot, and the self-test exception
    # disarmed every OTHER segment with it. Shape asserts below make any
    # future layout drift fail loudly at build, not silently at serve.)
    q = (code.to(torch.uint8) | (sign << 3)).view(n_pad, k // 16, 16)
    wq4 = (q[..., 0::2] | (q[..., 1::2] << 4)).reshape(n_pad, k // 2)
    ws4 = s.to(torch.int8).contiguous()
    assert wq4.shape == (n_pad, k // 2), \
        f"wq4 {tuple(wq4.shape)} != {(n_pad, k // 2)}"
    assert ws4.shape == (n_pad, k // 16), \
        f"ws4 {tuple(ws4.shape)} != {(n_pad, k // 16)}"
    # Tile-major, like the fp8 pack (#208): one (tile, k-block) record is
    # 128 rows x 64 nibble bytes + 128 x 8 exponents, contiguous, so the
    # kernel's stage_raw4 streams it with cp.async as one 8 KB + 1 KB run
    # instead of touching 128 DRAM pages per tile.
    wq4 = (wq4.view(n_pad // 128, 128, k // 128, 64)
           .permute(0, 2, 1, 3).contiguous())
    ws4 = (ws4.view(n_pad // 128, 128, k // 128, 8)
           .permute(0, 2, 1, 3).contiguous())
    return wq4, ws4


def mk_w4_dequant(wq4, ws4, n_rows):
    """bf16 [n_rows, k] holding exactly the values the kernel's expansion
    feeds the mma for this pack: the pure-torch twin of expand_w4 (nibble
    -> e2m1 grid value x 2^s, sign from nibble bit 3). Weights already on
    the grid round-trip bit-exactly, which is what the exact gate and the
    KDA fixture rely on."""
    import torch

    nt, kt = wq4.shape[0], wq4.shape[1]
    n_pad, k = nt * 128, kt * 128
    q = wq4.permute(0, 2, 1, 3).reshape(n_pad, k // 2)
    nib = torch.stack((q & 0x0F, q >> 4), dim=-1).reshape(n_pad, k)
    grid = torch.tensor(_E2M1_GRID, device=wq4.device)
    mag = grid[(nib & 7).long()]
    sign = torch.where((nib & 8) != 0, -1.0, 1.0)
    s = ws4.permute(0, 2, 1, 3).reshape(n_pad, k // 16).float()
    w = mag * sign * torch.exp2(s).repeat_interleave(16, dim=1)
    return w[:n_rows].to(torch.bfloat16)


def _mk_quant_x_ref(x):
    """fp32 [m, k]: x after the kernel's activation quant (per row, per
    128-k group: pow2 scale 2^frexp_exp(amax/448), e4m3 round-to-nearest,
    rescale). Pure twin of the prologue; with mk_w4_dequant it makes a
    torch fp32 matmul the kernel's exact reference (no fp8 MK arm exists
    to diff against any more)."""
    import torch

    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    amax = g.abs().amax(-1, keepdim=True)
    ratio = (amax / 448.0).clamp(min=1e-30)
    _frac, exp = torch.frexp(ratio)
    scale = torch.where(amax > 0, torch.exp2(exp.float()),
                        torch.ones_like(amax))
    q = (g / scale).to(torch.float8_e4m3fn).float() * scale
    return q.view(m, k)


def _stock_fp8_pair(w):
    """The stock deepgemm-layout pair for the same bf16 weight (self-tests
    and shadow arms need the arm they are diffing against)."""
    from vllm.model_executor.layers.glm53_fp8_dense import (
        _quantize_fp8_block_padded)

    # Keep the pre-padding shape: _fp8_dense_gemm needs it to trim the
    # padded output, and dropping it here is what broke the probe.
    q, ws, rows, cols = _quantize_fp8_block_padded(w)
    return q, ws, rows, cols


# ---------------------------------------------------------------------------
# MK_SEG_GEMM
# ---------------------------------------------------------------------------
def _gemm_call(x, mk_pack, n_rows):
    """mk_pack is (wq4, ws4) from build_mk_weight_w4."""
    import torch

    out = torch.empty(x.shape[0], n_rows, dtype=torch.bfloat16,
                      device=x.device)
    _EXT.run_gemm(x.contiguous(), mk_pack[0], mk_pack[1], out, n_rows)
    return out


def gemm_w4a8(x, mk_pack, n_rows):
    """Fp8DenseMethod.apply hook: None = not armed/eligible (stock runs).

    The lane is W4 or stock: a self-test failure disarms MK-GEMM for
    every linear (stock deepgemm), there is no fp8 MK pack to fall back
    to by construction."""
    if not _ARMED["gemm"] or x.dim() != 2 or mk_pack is None:
        return None
    if mk_pack[0] is None:
        return None
    # the pack is tile-major [n_pad/128, k/128, 128, 64]: the padded n is
    # 128 x its first dim (shape[0] alone is the tile count -- passing it
    # as n_pad failed the n_pad % 128 test on every real shape and the
    # lane silently stayed stock)
    if not _mk_gemm_eligible(x.shape[0], x.shape[1],
                             mk_pack[0].shape[0] * 128):
        return None
    return _gemm_call(x, mk_pack, n_rows)


# ---------------------------------------------------------------------------
# MK_SEG_MHC
# ---------------------------------------------------------------------------
def _mhc_call(x_flat, residual_flat, pm_flat, cm_flat, fn, hc_scale,
              hc_base, norm_weight, num_tokens, rms_eps, pre_eps,
              sinkhorn_eps, post_mult, norm_eps, sinkhorn_repeat):
    import torch

    hc_mult, hidden = residual_flat.shape[1], residual_flat.shape[2]
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(num_tokens, hc_mult, dtype=torch.float32,
                               device=x_flat.device)
    comb_mix_cur = torch.empty(num_tokens, hc_mult * hc_mult,
                               dtype=torch.float32, device=x_flat.device)
    layer_input_cur = torch.empty(num_tokens, hidden, dtype=torch.bfloat16,
                                  device=x_flat.device)
    ws = _ensure_workspace(x_flat.device)
    _EXT.run_mhc(
        [x_flat.data_ptr(), residual_flat.data_ptr(), pm_flat.data_ptr(),
         cm_flat.data_ptr(), fn.data_ptr(), hc_scale.data_ptr(),
         hc_base.data_ptr(), norm_weight.data_ptr(),
         residual_cur.data_ptr(), post_mix_cur.data_ptr(),
         comb_mix_cur.data_ptr(), layer_input_cur.data_ptr(),
         ws["yp"].data_ptr(), ws["rp"].data_ptr(), ws["sq"].data_ptr(),
         ws["pmix"].data_ptr(),
         ws["ol_stash"].data_ptr(), ws["barrier_mhc"].data_ptr()],
        [float(rms_eps), float(pre_eps), float(sinkhorn_eps),
         float(post_mult), float(norm_eps)],
        [num_tokens, int(sinkhorn_repeat)],
    )
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def mhc_fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn,
                       hc_scale, hc_base, rms_eps, hc_pre_eps,
                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                       norm_weight, norm_eps):
    """Same contract as the stock wrapper's small-M branch. None = stock."""
    if not _ARMED["mhc"]:
        return None
    import torch

    if residual.dim() != 3:
        return None
    num_tokens, hc_mult, hidden = residual.shape
    if not _mk_mhc_eligible(num_tokens, hc_mult, hidden):
        return None
    if (x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
            or fn.dtype != torch.float32 or hc_scale.dtype != torch.float32
            or hc_base.dtype != torch.float32):
        return None
    outer = residual.shape[:-2]
    rc, pm, cm, li = _mhc_call(
        x.reshape(-1, hidden), residual.reshape(-1, hc_mult, hidden),
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult, hc_mult),
        fn.reshape(hc_mult * (2 + hc_mult), hc_mult * hidden).contiguous(),
        hc_scale.contiguous(), hc_base.contiguous(),
        norm_weight.to(torch.bfloat16).contiguous(), num_tokens,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps,
        sinkhorn_repeat)
    return (rc.view(*outer, hc_mult, hidden),
            pm.view(*outer, hc_mult, 1),
            cm.view(*outer, hc_mult, hc_mult),
            li.view(*outer, hidden))


# ---------------------------------------------------------------------------
# MK_SEG_KDA
# ---------------------------------------------------------------------------
def _kda_meta(layer):
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    attn_meta = getattr(ctx, "attn_metadata", None)
    if not isinstance(attn_meta, dict):
        return None
    return attn_meta.get(layer.prefix)


def _kda_eligible(meta) -> bool:
    """Pure metadata contract: pure spec-verify decode steps only."""
    if meta is None:
        return False
    if meta.spec_sequence_masks is None or meta.num_spec_decodes <= 0:
        return False
    if meta.num_prefills > 0:
        return False
    if (meta.non_spec_token_indx is not None
            and meta.non_spec_token_indx.numel()):
        return False
    return 0 < meta.num_actual_tokens <= 32


def _kda_layout_ok(layer) -> bool:
    """Static per-boot verdict (cached): state layouts exactly what the
    kernel indexes, conv state dim-first, kv cache attached."""
    import torch

    kv = getattr(layer, "kv_cache", None)
    if not isinstance(kv, tuple) or len(kv) != 2:
        return False
    conv_state, rec_state = kv
    return (layer._merged_conv_weight is not None
            and layer._merged_conv_weight.dtype == torch.float32
            and layer._merged_conv_weight.shape == (KDA_QKV, 4)
            and layer._conv_state_dim_first
            and layer.A_log.dtype == torch.float32
            and layer.dt_bias.dtype == torch.float32
            and conv_state.dtype == torch.float32
            and conv_state.dim() == 3
            and conv_state.shape[1] == KDA_QKV
            # spec-decode allocates the sliding window k-1+num_spec wide;
            # the kernel uses the runtime width as stride over the active
            # [0, 3) window, so any width >= 3 is admissible (a hard (QKV,3)
            # gate never matched production -- review finding).
            # The width must carry the spec headroom, not just the
            # kernel history: causal_conv1d_update slides across the
            # draft-verify tokens and a >= 3 check admitted a buffer
            # the STOCK arm then read out of bounds.
            and conv_state.shape[2] == KDA_CONV_STATE_W
            and conv_state.is_contiguous()
            and rec_state.dtype == torch.float32
            and rec_state.dim() == 4
            and rec_state.shape[1:] == (KDA_H, KDA_D, KDA_D)
            and rec_state.is_contiguous())


def _kda_ensure_packs(layer) -> bool:
    """Per-layer MK packs for in_proj/o_proj, cached on the layer.

    Self-sufficient by design: a shadow boot must not depend on the MK-GEMM
    arm having attached packs through Fp8DenseMethod. Requires the linear to
    run the W8A8 copy (Fp8DenseMethod) so the stock arm of every comparison
    is the SAME quantization axis -- against bf16 stock the self-test's 2e-2
    gate could not tell a broken kernel from quantization noise. Building
    allocates, so it never runs under graph capture.
    """
    import torch

    if getattr(layer, "_mk_packs_ready", False):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False  # first eager warmup call builds; capture never does
    try:
        from vllm.model_executor.layers.glm53_fp8_dense import (
            Fp8DenseMethod)

        in_m = layer.in_proj_qkvbfg_a.quant_method
        o_m = layer.o_proj.quant_method
        if not isinstance(in_m, Fp8DenseMethod) or not isinstance(
                o_m, Fp8DenseMethod):
            return False  # stock in_proj/o_proj is bf16 here -> stay stock
        # The in-kernel in_proj / o_proj GEMMs stream the same W4 packs the
        # linears serve (built here when MK-GEMM is off and none exists).
        def _w4_pack(method, weight):
            p = getattr(method, "_mk", None)
            if p is not None and p[0] is not None:
                return p
            return build_mk_weight_w4(weight)

        layer._mk_in_pack = _w4_pack(in_m, layer.in_proj_qkvbfg_a.weight)
        layer._mk_o_pack = _w4_pack(o_m, layer.o_proj.weight)
        layer._mk_packs_ready = True
        return True
    except Exception:
        logger.exception("[megakernel] kda pack build failed; layer stays "
                         "stock")
        return False


def _kda_device_ok() -> bool:
    """GB10 gate, shared by armed and shadow paths. A pure-shadow boot
    used to build the extension and launch on ANY device with no self-test
    -- the same unverified-kernel posture this module exists to refuse
    (review finding)."""
    import torch

    if torch.cuda.device_count() == 0:
        return False
    major, minor, sms, _ = _build().probe_device()
    return (major, minor) == (12, 1) and sms == 48


def kda_takeover(layer) -> bool:
    """Hot-path gate for the kda.py overlay hook (arms on first call)."""
    if not (ENABLE_KDA or KDA_SHADOW):
        return False
    maybe_arm()
    if not (_ARMED["kda"] or KDA_SHADOW):
        return False
    if _ARMED["kda"]:
        pass  # arm() already cleared the device gate
    elif not _kda_device_ok():
        return False  # shadow-only boot: same GB10 contract, or no launch
    if not _kda_ensure_packs(layer):
        return False
    return _kda_layout_ok(layer) and _kda_eligible(_kda_meta(layer))


def _kda_launch(layer, hidden_states, meta, conv_state, rec_state, out,
                delta_variant=1):
    import torch
    ws = _ensure_workspace(hidden_states.device)
    n_spec = meta.num_spec_decodes
    ow = getattr(layer.o_norm, "weight", None)
    onorm_w = ow if isinstance(ow, torch.Tensor) else torch.ones(
        KDA_D, dtype=torch.bfloat16, device=hidden_states.device)
    _EXT.run_kda(
        [hidden_states.data_ptr(),
         layer._mk_in_pack[0].data_ptr(),
         layer._mk_in_pack[1].data_ptr(),
         layer.f_b_proj.weight.data_ptr(),
         layer.g_b_proj.weight.data_ptr(),
         layer._merged_conv_weight.data_ptr(),
         conv_state.data_ptr(), rec_state.data_ptr(),
         layer.A_log.data_ptr(), layer.dt_bias.data_ptr(),
         meta.spec_query_start_loc[:n_spec + 1].data_ptr(),
         meta.spec_state_indices_tensor.data_ptr(),
         meta.num_accepted_tokens.data_ptr(),
         layer._mk_o_pack[0].data_ptr(),
         layer._mk_o_pack[1].data_ptr(),
         out.data_ptr(),
         ws["qkv"].data_ptr(), ws["g1"].data_ptr(), ws["g2"].data_ptr(),
         ws["convq"].data_ptr(), ws["attn"].data_ptr(),
         _barrier_ptr(ws), onorm_w.data_ptr()],
        [float(layer.kda_lower_bound),
         float(getattr(layer.o_norm, "eps", 1e-5))],
        [int(meta.num_actual_tokens), int(n_spec),
         int(meta.spec_state_indices_tensor.size(-1)),
         int(delta_variant), int(conv_state.shape[-1])],
    )


def kda_block(layer, hidden_states, positions):
    """Whole linear-attention block, one launch + the boundary AR."""
    import torch

    meta = _kda_meta(layer)
    out = torch.empty(meta.num_actual_tokens, layer.hidden_size,
                      dtype=torch.bfloat16, device=hidden_states.device)
    conv_state, rec_state = layer.kv_cache
    _kda_launch(layer, hidden_states.contiguous(), meta, conv_state,
                rec_state, out, delta_variant=_KDA_VARIANT)
    # vLLM's all_reduce is OUT-OF-PLACE: the return value carries the
    # reduced tensor and the input buffer keeps rank-local partial sums.
    # Discarding the return once served partials to every rank (review).
    from vllm.distributed import tensor_model_parallel_all_reduce
    return tensor_model_parallel_all_reduce(out)


class KdaShadowArm:
    """Eager two-arm run: MK into cloned states, stock into the real ones.

    Graph capture never enters here (the overlay checks is_captureing), so
    the clones and the comparison never touch captured memory."""

    def __init__(self, layer, hidden_states):
        import torch

        self.ok = False
        meta = _kda_meta(layer)
        if meta is None:
            return
        conv_state, rec_state = layer.kv_cache
        self.conv_ref = conv_state.clone()
        self.rec_ref = rec_state.clone()
        self.out = torch.empty(meta.num_actual_tokens, layer.hidden_size,
                               dtype=torch.bfloat16,
                               device=hidden_states.device)
        self.conv_mk = conv_state.clone()
        self.rec_mk = rec_state.clone()
        _kda_launch(layer, hidden_states.contiguous(), meta, self.conv_mk,
                    self.rec_mk, self.out)
        # same out-of-place contract: shadow compares against the REDUCED
        # tensor, not this rank's partial (review finding)
        from vllm.distributed import tensor_model_parallel_all_reduce
        self.out = tensor_model_parallel_all_reduce(self.out)
        self.layer, self.ok = layer, True

    _n_calls = 0

    def compare(self, stock_out):
        """Called by the overlay after the stock forward: diffs outputs and
        the states the next step would read. Logs every 64th call and on any
        drift (same cadence discipline as the vocab-mask audit)."""
        conv_state, rec_state = self.layer.kv_cache
        errs = {"out": _rel_err(self.out, stock_out),
                "conv_state": _rel_err(self.conv_mk, conv_state),
                "rec_state": _rel_err(self.rec_mk, rec_state)}
        drift = any(not (v <= _TOL_KDA_SHADOW) or math.isnan(v)
                    for v in errs.values())
        KdaShadowArm._n_calls += 1
        if drift or KdaShadowArm._n_calls % 64 == 1:
            logger.warning("[megakernel] kda shadow #%d rel_errs=%s %s",
                           KdaShadowArm._n_calls,
                           {k: "%.2e" % v for k, v in errs.items()},
                           "DRIFT" if drift else "ok")


# ---------------------------------------------------------------------------
# boot arm + self-tests
# ---------------------------------------------------------------------------
_DRAIN_BUF = None


def _drain_l2():
    import torch
    global _DRAIN_BUF
    if _DRAIN_BUF is None:
        _DRAIN_BUF = torch.zeros(16 << 20, dtype=torch.float32, device="cuda")
    _DRAIN_BUF.sum()
    _DRAIN_BUF.sum()


def _rel_err(a, b) -> float:
    import torch

    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float(d)


def _exact_gate(got, ref32) -> tuple:
    """(rel_err, n_over_ulp) of a bf16 kernel output against an fp32 torch
    reference: the reference is rounded to bf16 first (the kernel rounds
    its fp32 accumulation the same way), and an element counts as OVER
    when it differs from that by more than one bf16 ulp of the reference
    (2^-7 relative, floored at 1e-2 of the row's largest magnitude so
    near-zero elements are judged in absolute terms). A different fp32
    summation order flips a handful of elements by one ulp (rel ~1e-4); a
    layout / expansion bug moves whole rows (rel >= 1e-2, ulps >> 1)."""
    import torch

    refb = ref32.to(torch.bfloat16).float()
    diff = (got.float() - refb).abs()
    floor = refb.abs().amax(dim=-1, keepdim=True) * 1e-2
    ulp = torch.maximum(refb.abs(), floor) * (2.0 ** -7)
    return _rel_err(got, refb), int((diff > ulp).sum().item())


def _selftest_mhc() -> bool:
    """Diff MK_SEG_MHC against the stock TileLang pair at T=8."""
    import torch

    from vllm.model_executor.kernels.mhc import tilelang_kernels as tlk

    torch.manual_seed(0)
    dev = "cuda"
    T = 8
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1
    res = torch.randn(T, HC, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1
    pm = torch.rand(T, HC, dtype=torch.float32, device=dev)
    cm = torch.rand(T, HC, HC, dtype=torch.float32, device=dev)
    fn = torch.randn(NOUT, HC * HIDDEN, dtype=torch.float32,
                     device=dev) * 0.02
    hc_scale = torch.ones(3, dtype=torch.float32, device=dev)
    hc_base = torch.zeros(NOUT, dtype=torch.float32, device=dev)
    nw = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    rms_eps = pre_eps = sink_eps = norm_eps = 1e-6
    post_mult, sinkhorn_repeat = 1.0, 4

    rc, pmc, cmc, li = _mhc_call(
        x, res, pm, cm.reshape(T, HC * HC).contiguous(), fn, hc_scale,
        hc_base, nw, T, rms_eps, pre_eps, sink_eps, post_mult, norm_eps,
        sinkhorn_repeat)

    # stock pair -- the wrapper's small-M branch, verbatim
    n_splits = 4
    yp = torch.empty(n_splits, T, NOUT, dtype=torch.float32, device=dev)
    rp = torch.empty(n_splits, T, dtype=torch.float32, device=dev)
    res_ref = torch.empty_like(res)
    tlk.mhc_fused_tilelang(cm, res, pm, x, fn.view(NOUT, HC, HIDDEN), yp,
                           rp, res_ref, HC, HIDDEN, NOUT, tile_n=2,
                           n_splits=n_splits)
    pm_ref = torch.empty(T, HC, dtype=torch.float32, device=dev)
    cm_ref = torch.empty(T, HC * HC, dtype=torch.float32, device=dev)
    li_ref = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
    tlk.mhc_pre_big_fuse_with_norm_tilelang(
        yp, rp, hc_scale, hc_base, res_ref, pm_ref, cm_ref, li_ref, nw,
        HIDDEN, rms_eps, pre_eps, sink_eps, post_mult, sinkhorn_repeat,
        norm_eps, n_splits=n_splits, hc_mult=HC)
    torch.cuda.synchronize()

    errs = (_rel_err(rc, res_ref), _rel_err(pmc, pm_ref),
            _rel_err(cmc, cm_ref), _rel_err(li, li_ref))
    ok = all(e <= _TOL_MHC for e in errs)
    logger.warning("[megakernel] selftest mhc rel_errs=%s -> %s",
                   ["%.2e" % e for e in errs], "ARM" if ok else "DISARM")
    return ok


def _selftest_gemm() -> bool:
    """Two gates for the W4 lane:

    (a) EXACT expansion: weights built ON the e2m1 x 2^s grid quantize
        losslessly, so the kernel must reproduce a torch fp32 matmul of
        the kernel-quantized activations (_mk_quant_x_ref) against the
        dequantized pack (mk_w4_dequant) to bf16 output rounding: no
        element more than one bf16 ulp off, rel <= 1e-3 (_exact_gate).
        Anything above is a layout / expansion bug, not quantization.
    (b) BY-DESIGN error: random weights through W4 vs the stock W8A8 pair
        on the decode shapes, gated at 0.15 (e2m1's by-design error is
        0.02-0.08 rel on row blocks -- the same tolerance class the
        fp8-dense module uses for its w4 arm).
    """
    import torch

    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm

    torch.manual_seed(0)
    dev = "cuda"
    n, k, m = 1024, 4096, 8
    code = torch.randint(0, 8, (n, k // 16, 16), device=dev)
    sexp = torch.randint(-5, 7, (n, k // 16, 1), device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    w_exact = (grid[code] * torch.exp2(sexp.float())) * torch.where(
        torch.randn_like(code.float()) < 0, -1.0, 1.0)
    w_exact = w_exact.view(n, k).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    pack = build_mk_weight_w4(w_exact)
    got = _gemm_call(x, pack, n)
    ref = _mk_quant_x_ref(x) @ mk_w4_dequant(pack[0], pack[1], n).float().T
    torch.cuda.synchronize()
    e_exact, n_ulp = _exact_gate(got, ref)
    if e_exact > 1e-3 or n_ulp > 0:
        logger.warning("[megakernel] selftest gemm EXACT rel=%.2e over-ulp=%d "
                       "-> DISARM (expansion is not bit-exact)", e_exact, n_ulp)
        return False
    for m, n in ((8, KDA_INPROJ_N), (16, HIDDEN), (32, 1024)):
        w = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.05
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        sq, sws, srows, scols = _stock_fp8_pair(w)
        ref = _fp8_dense_gemm(x, sq, sws, srows, scols)
        got = _gemm_call(x, build_mk_weight_w4(w), n)
        torch.cuda.synchronize()
        e = _rel_err(got, ref)
        if e > _TOL_GEMM_W4:
            logger.warning("[megakernel] selftest gemm m=%d n=%d by-design "
                           "rel=%.2e -> DISARM", m, n, e)
            return False
    logger.warning("[megakernel] selftest gemm exact=%.2e -> ARM", e_exact)
    return True


def hc_scale_ones():
    import torch

    return torch.ones(3, dtype=torch.float32, device="cuda")


def hc_base_zeros():
    import torch

    return torch.zeros(NOUT, dtype=torch.float32, device="cuda")


class _KdaFixture:
    """Synthetic one-request spec-verify step shared by the boot self-test
    and probes/megakernel_glm53_bench.py, so the gate and the probe cannot
    drift apart. acc in [1, 8] rolls the states to different boundaries.

    SLOT is deliberately 1, not 0: the conv/rec state buffers are indexed
    [slots, ...] and a slot-0-only fixture once passed while the kernel had
    the slot stride wrong (found in review 2026-09-01) -- the self-test must
    exercise nonzero-slot addressing to be worth anything.
    """

    SLOT = 1

    def __init__(self, acc: int = 3, seed: int = 0):
        import torch

        torch.manual_seed(seed)
        T = 8
        mkw = lambda *s, dt=torch.bfloat16: (  # noqa: E731
            torch.randn(*s, dtype=dt, device="cuda"))
        self.T, self.acc = T, acc
        self.x = mkw(T, HIDDEN) * 0.1
        # Both arms see weights ON the e2m1 grid: the MK side packs them
        # losslessly (W4 is the only MK arm), the stock side quantizes them
        # to its fp8 blocks -- so the diff below is activation / fp8
        # rounding noise (the 2e-2 class), not e2m1's by-design error.
        self.w_in = mk_w4_dequant(*build_mk_weight_w4(
            mkw(KDA_INPROJ_N, HIDDEN) * 0.02), KDA_INPROJ_N)
        self.w_o = mk_w4_dequant(*build_mk_weight_w4(
            mkw(HIDDEN, KDA_OUT) * 0.02), HIDDEN)
        self.f_b = mkw(KDA_OUT, KDA_D) * 0.1
        self.g_b = mkw(KDA_OUT, KDA_D) * 0.1
        self.conv_w = mkw(KDA_QKV, 4, dt=torch.float32) * 0.2
        self.a_log = mkw(1, 1, KDA_H, 1, dt=torch.float32) * 0.1
        self.dt_bias = torch.zeros(KDA_OUT, dtype=torch.float32,
                                   device="cuda")
        # Width is conv_kernel_size - 1 + num_spec, not conv_kernel_size - 1.
        # mamba_utils.kda_state_shape:291 sizes it that way, and
        # glm5next_kda.get_state_shape says why: causal_conv1d_update with
        # num_accepted_tokens + max_query_len slides a window across the
        # draft-verify tokens and "reads past the allocated width" without
        # the spec headroom. A width-3 fixture made the STOCK arm read and
        # write out of bounds -- which is why its state held values larger
        # than any input (1.2 against a 0.24 max) and why ~64% of channels
        # disagreed even at acc=1. The gate was comparing against garbage.
        self.conv_st = mkw(2, KDA_QKV, KDA_CONV_STATE_W, dt=torch.float32)
        # One recurrent slot PER QUERY POSITION (SLOT .. SLOT + 7), as the
        # engine allocates for spec decode: the stock kernel resumes from
        # slot [acc - 1] and stores the state after token j into slot [j].
        # A single shared slot passed while the kernel resumed from [0] and
        # wrote only at the accepted boundary.
        self.rec_st = (mkw(self.SLOT + 8 + 1, KDA_H, KDA_D, KDA_D,
                           dt=torch.float32) * 0.1)
        self.cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        self.sidx = (self.SLOT + torch.arange(8, dtype=torch.int32,
                                              device="cuda")).view(1, 8)
        self.nacc = torch.tensor([acc], dtype=torch.int32, device="cuda")
        # KDA_D, not KDA_OUT: the CUDA side declares onorm_w as [KDA_D] and
        # indexes it by the within-head dim (a.onorm_w[d]), and the stock
        # FusedRMSNormGated(KDA_D) weight is 128 wide too.
        # Built in __init__, not in _layer_stand_in: stock_run() needs the
        # same values, and creating it lazily on the MK side made stock_run()
        # depend on mk_run() having been called first.
        self.onorm_w = (torch.rand(KDA_D, device="cuda") * 0.4
                        + 0.8).to(torch.bfloat16)
        self._mk_cache = None       # (layer stand-in, meta), quantized once
        self._stock_cache = None    # (q, ws, rows, cols) x {in, o}, once

    def _layer_stand_in(self):
        import torch
        if self._mk_cache is not None:
            return self._mk_cache
        f = self
        in_mk, o_mk = build_mk_weight_w4(f.w_in), build_mk_weight_w4(f.w_o)

        class _P:
            pass

        class _Meta:
            num_spec_decodes = 1
            num_actual_tokens = f.T
            spec_query_start_loc = f.cu
            spec_state_indices_tensor = f.sidx
            num_accepted_tokens = f.nacc

        la = _P()
        la._mk_in_pack = in_mk
        la._mk_o_pack = o_mk
        la._mk_packs_ready = True
        la.f_b_proj = _P()
        la.f_b_proj.weight = f.f_b
        la.g_b_proj = _P()
        la.g_b_proj.weight = f.g_b
        la._merged_conv_weight = f.conv_w
        la.A_log, la.dt_bias = f.a_log, f.dt_bias
        la.kda_lower_bound = -5.0
        # NON-TRIVIAL affine weight: an all-ones o_norm cannot see a missing
        # weight multiply (review finding). The stock arm must use the same
        # values -- FusedRMSNormGated gets them via onorm_w below.
        la.o_norm = _P()
        la.o_norm.eps = 1e-5
        la.o_norm.weight = torch.nn.Parameter(self.onorm_w)
        self._mk_cache = (la, _Meta())
        return self._mk_cache

    def mk_run(self, delta_variant=None, drain=False):
        """MK arm on cloned states -> dict(out, conv_state, rec_state).

        delta_variant sweeps the retrieval/write operand order (see the .cu
        comment): the stock source is not in this repo, so the boot settles
        which variant matches fused_recurrent_kda."""
        # _KDA_VARIANT is defined below this class, so it cannot be a
        # default argument: that binds at class-definition time.
        if delta_variant is None:
            delta_variant = _KDA_VARIANT
        import torch

        la, meta = self._layer_stand_in()
        conv_mk, rec_mk = self.conv_st.clone(), self.rec_st.clone()
        out = torch.empty(self.T, HIDDEN, dtype=torch.bfloat16,
                          device="cuda")
        if drain:
            # diagnosis only: the clones above leave ~11 MB of dirty lines
            # in L2 whose write-back would otherwise run under the kernel's
            # in_proj stream (+20 us on the p0 stamp; the serving chain has
            # no such predecessor). A 64 MB read evicts them first.
            _drain_l2()
        _kda_launch(la, self.x, meta, conv_mk, rec_mk, out,
                    delta_variant=delta_variant)
        torch.cuda.synchronize()
        return {"out": out, "conv_state": conv_mk, "rec_state": rec_mk}

    def pick_variant(self):
        """First delta variant whose output AND states match the stock op
        (gate 2e-2, same as the self-test); None if no variant does."""
        ref = self.stock_run()
        for v in (0, 1, 2):
            got = self.mk_run(delta_variant=v)
            slot = slice(_KdaFixture.SLOT, _KdaFixture.SLOT + 1)
            errs = {k: _rel_err(got[k][slot] if k != "out" else got[k],
                                ref[k][slot] if k != "out" else ref[k])
                    for k in ref}
            if all(e <= _TOL_KDA for e in errs.values()):
                logger.warning("[megakernel] kda delta variant %d matches "
                               "stock (errs=%s)", v,
                               {k: "%.2e" % x for k, x in errs.items()})
                return v
        logger.warning("[megakernel] no delta variant matches stock: %s",
                       {k: "%.2e" % x for k, x in errs.items()})
        return None

    def stock_run(self, debug=False):
        """Stock chain (in_proj gemm, conv update, fused_recurrent_kda,
        gated norm, o_proj gemm) -> dict(out, conv_state, rec_state)."""
        import torch

        from vllm.model_executor.layers.glm53_fp8_dense import (
            _fp8_dense_gemm)
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update)
        from vllm.third_party.flash_linear_attention.ops.kda import (
            FusedRMSNormGated, fused_recurrent_kda)

        T = self.T
        if self._stock_cache is None:
            self._stock_cache = (_stock_fp8_pair(self.w_in)
                                 + _stock_fp8_pair(self.w_o))
        (sq_in, sws_in, rin, cin, sq_o, sws_o, ro, co) = self._stock_cache
        proj = _fp8_dense_gemm(self.x, sq_in, sws_in, rin, cin)
        qkv, beta_raw, f_a, g_a = proj.split(
            [KDA_QKV, KDA_H, KDA_D, KDA_D], dim=-1)
        g1 = f_a @ self.f_b.T
        g2 = g_a @ self.g_b.T
        conv_ref = self.conv_st.clone()
        qkv_c = causal_conv1d_update(
            qkv.contiguous(), conv_ref, self.conv_w, None, "silu",
            conv_state_indices=torch.tensor([self.SLOT], device="cuda"),
            num_accepted_tokens=self.nacc, query_start_loc=self.cu,
            max_query_len=8)
        q = qkv_c[:, :KDA_OUT].view(1, T, KDA_H, KDA_D)
        k = qkv_c[:, KDA_OUT:2 * KDA_OUT].view(1, T, KDA_H, KDA_D)
        v = qkv_c[:, 2 * KDA_OUT:].view(1, T, KDA_H, KDA_D)
        rec_ref = self.rec_st.clone()
        attn, _ = fused_recurrent_kda(
            q=q, k=k, v=v, g=g1.view(1, T, KDA_H, KDA_D),
            beta=beta_raw.view(1, T, KDA_H), initial_state=rec_ref,
            use_qk_l2norm_in_kernel=True, cu_seqlens=self.cu,
            ssm_state_indices=self.sidx, num_accepted_tokens=self.nacc,
            sigmoid_beta=True, a_log=self.a_log, g_bias=self.dt_bias,
            compute_gate=True, lower_bound=-5.0)
        # FusedRMSNormGated is a vLLM CustomOp: its __init__ reads the
        # ambient compilation config, so building one outside a
        # set_current_vllm_config() context asserts. A serving boot always
        # has that context; this stand-alone probe has to supply one.
        from vllm.config import VllmConfig, set_current_vllm_config
        try:
            get_current_vllm_config = None
            from vllm.config import get_current_vllm_config  # noqa: F811
            get_current_vllm_config()
            o_norm = FusedRMSNormGated(KDA_D, eps=1e-5,
                                       activation="sigmoid")
        except (AssertionError, ImportError):
            with set_current_vllm_config(VllmConfig()):
                o_norm = FusedRMSNormGated(KDA_D, eps=1e-5,
                                           activation="sigmoid")
        # Two fixes the stock arm never got to exercise, because it had never
        # run: the module is built on CPU (device mismatch against the cuda
        # activations), and its weight defaults to ones. The MK arm uses
        # self.onorm_w, and _layer_stand_in's own comment says the stock arm
        # must use the SAME values -- "an all-ones o_norm cannot see a missing
        # weight multiply". Comparing ones against onorm_w would have made the
        # whole KDA diff meaningless.
        o_norm = o_norm.to(device="cuda")
        with torch.no_grad():
            o_norm.weight.copy_(self.onorm_w.to(o_norm.weight.dtype))
        core = o_norm(attn.view(T, KDA_H, KDA_D), g2.view(T, KDA_H, KDA_D))
        out = _fp8_dense_gemm(core.reshape(T, KDA_OUT), sq_o, sws_o,
                              ro, co)
        torch.cuda.synchronize()
        res = {"out": out, "conv_state": conv_ref, "rec_state": rec_ref}
        if debug:
            # Split the pipeline for diagnosis: `attn` is the recurrence
            # readout, `core` is it after the gated RMSNorm. Comparing them
            # separately says whether an `out` mismatch comes from phase 3
            # (readout), phase 4 (norm) or phase 5 (o_proj). Kept off the
            # default dict because pick_variant() gates on every key it
            # returns.
            res["attn"] = attn.reshape(T, KDA_OUT)
            res["core"] = core.reshape(T, KDA_OUT)
            # g1 gates the recurrence (phase 3), g2 only the norm (phase 4).
            # rec_state passing while core fails is exactly what a correct
            # g1 with a wrong g2 looks like, so both belong in the split.
            res["g1"] = g1.reshape(T, KDA_OUT)
            res["g2"] = g2.reshape(T, KDA_OUT)
        return res


# Retrieval operand settled from the stock source, not from a sweep.
# fused_recurrent.py, gated delta rule body:
#     b_h  *= exp(b_gk)                      # decay
#     b_v  -= tl.sum(b_h * b_k[None, :], 1)  # error retrieves with k
#     b_h  += (b_v * b_beta)[:, None] * b_k[None, :]
#     b_o   = tl.sum(b_h * b_q[None, :], 1)  # readout with q, post-update
# The .cu picks k for the error only when delta_variant == 1, so 1 is the
# arm that matches. 0 retrieved with q, which is what left `out` wrong at
# rel_err ~4 while rec_state was already near the gate.
_KDA_VARIANT = 1


def _selftest_kda() -> bool:
    """Diff MK_SEG_KDA against the stock chain on synthetic spec metadata;
    outputs AND the rolled states (at the fixture's NONZERO slot) must
    agree. The retrieval/write operand order of fused_recurrent_kda is not
    readable from this repo's source, so the sweep picks the matching delta
    variant at boot (none -> DISARM); serving then uses the settled variant
    (review finding)."""
    global _KDA_VARIANT
    fx = _KdaFixture(acc=3)
    v = fx.pick_variant()
    if v is None:
        return False
    _KDA_VARIANT = v
    return True


def arm() -> None:
    """Boot gate: device check + per-segment self-tests. Idempotent."""
    import torch

    if not MASTER or any(_ARMED.values()):
        return
    if torch.cuda.device_count() == 0:
        logger.warning("[megakernel] no CUDA device; module stays inert")
        return
    ext = _build()
    major, minor, sms, _smem = ext.probe_device()
    if (major, minor) != (12, 1) or sms != 48:
        logger.warning("[megakernel] device cc=%d.%d sms=%d is not GB10 "
                       "(12.1/48); module stays inert", major, minor, sms)
        return
    def _gate(seg, fn):
        # One segment's failure is ITS failure: an exception escaping a
        # self-test used to disarm every other segment with it (the W4
        # packing bug shipped exactly that blast radius until review).
        try:
            return bool(fn())
        except Exception:
            logger.exception("[megakernel] selftest %s raised -> DISARM "
                             "that segment only", seg)
            return False

    if ENABLE_MHC:
        _ARMED["mhc"] = _gate("mhc", _selftest_mhc)
    if ENABLE_GEMM:
        _ARMED["gemm"] = _gate("gemm", _selftest_gemm)
    if ENABLE_MLA:
        _ARMED["mla"] = _gate("mla", _selftest_mla)
    if ENABLE_KDA:
        _ARMED["kda"] = _gate("kda", _selftest_kda)
    logger.warning("[megakernel] armed=%s shadow_kda=%s",
                   dict(_ARMED), KDA_SHADOW)


_armed_once = False


# ---------------------------------------------------------------------------
# MK_SEG_MLA -- sparse MLA decode (see the kernel comment for the cost model)
# ---------------------------------------------------------------------------
MLA_D = 512          # kv_lora_rank
MLA_H = 16           # MLA heads per rank at TP4
MLA_SPLITS_MAX = 64
_MLA_WS = None


MLA_MAX_SPLIT_ROWS = 64   # beyond this the split path's fp32 scratch is silly


def _mla_workspace(device, T: int, splits: int):
    """Split partials + (m, l), grown once and then strongly held: a captured
    graph bakes these addresses, so they must never be reallocated."""
    global _MLA_WS
    import torch

    need = T * splits
    if _MLA_WS is not None and _MLA_WS["cap"] >= need:
        return _MLA_WS
    cap = max(need, MAX_TOK * 2)
    _MLA_WS = {
        "cap": cap,
        "part": torch.zeros(cap * MLA_H * MLA_D, dtype=torch.float32, device=device),
        "pml": torch.zeros(cap * MLA_H * 2, dtype=torch.float32, device=device),
    }
    return _MLA_WS


def mla_splits(T: int) -> int:
    """Slot-axis splits for this row count.

    Measured rule (grid 48, W=2048): the best split is the smallest s with
    T*s a multiple of the resident grid -- every block then gets the same
    number of items and the same slot count per item. T=8 -> 6 (94 us),
    16 -> 3 (162), 24 -> 2 (235), 32 -> 3 (330; 1 and 2 leave a third of the
    blocks walking a second item alone and cost 40%)."""
    if _EXT is None or T <= 0:
        return 1
    if T > MLA_MAX_SPLIT_ROWS:
        # prefill: every row is its own item, the kernel normalises in place
        # and no [T][splits][H][D] fp32 scratch exists (268 MB at T=8192)
        return 1
    grid = int(_EXT.mla_grid())
    forced = os.environ.get("VLLM_GLM53_MK_MLA_SPLITS")   # probe knob, never set in serving
    if forced:
        return max(1, min(MLA_SPLITS_MAX, int(forced)))
    for s in range(1, MLA_SPLITS_MAX + 1):
        if (T * s) % grid == 0:
            return s
    return max(1, min(MLA_SPLITS_MAX, round(grid / T)))


def mla_decode(q_nope, ckv, slots, lens, sm_scale: float, ckv_scale: float,
               out=None):
    """Sparse MLA decode over the indexer's top-k slots.

    q_nope [T, H, D] bf16 (never quantised -- the sparse backend forbids it);
    ckv    e4m3 [num_slots, D] flat latent cache; slots [T, W] int32 global
    slot ids with the valid prefix first; lens [T] int32 valid counts ON THE
    DEVICE (that is what keeps this launch inside the captured graph)."""
    import torch

    T, H, D = q_nope.shape
    assert (H, D) == (MLA_H, MLA_D), f"mla: shape {(H, D)} != {(MLA_H, MLA_D)}"
    assert q_nope.is_contiguous() and slots.is_contiguous()
    assert slots.dtype == torch.int32 and lens.dtype == torch.int32
    ws = _ensure_workspace(q_nope.device)
    splits = mla_splits(T)
    # size the scratch by the splits this shape actually uses, not the cap:
    # MLA_SPLITS_MAX is 64 and sizing by it reserved 134 MB where 25 is used
    mw = (_mla_workspace(q_nope.device, T, splits) if splits > 1
          else {"part": ws["barrier"], "pml": ws["barrier"]})   # unused when splits == 1
    if out is None:
        out = torch.empty_like(q_nope)
    _EXT.run_mla(
        [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(),
         out.data_ptr(), mw["part"].data_ptr(), mw["pml"].data_ptr(),
         ws["barrier_mla"].data_ptr()],
        [float(sm_scale), float(ckv_scale)],
        [int(T), int(slots.shape[1]), int(splits)],
    )
    return out


def mla_decode_ref(q_nope, ckv, slots, lens, sm_scale: float, ckv_scale: float):
    """Pure-torch twin of the kernel, in fp32. Same contract, no pipelining."""
    import torch

    T, H, D = q_nope.shape
    out = torch.zeros(T, H, D, dtype=torch.float32, device=q_nope.device)
    q = q_nope.float()
    for t in range(T):
        n = int(lens[t].item())
        if n <= 0:
            continue
        idx = slots[t, :n].long()
        c = ckv.view(-1, D)[idx].to(torch.float32) * ckv_scale   # [n, D]
        s = (q[t] @ c.T) * sm_scale                              # [H, n]
        p = torch.softmax(s, dim=-1)
        out[t] = p @ c
    return out.to(q_nope.dtype)


def _selftest_mla() -> bool:
    """Diff the kernel against the torch twin on the serving geometry.

    Gates on the ranking-safe band: the twin sums in fp32 in slot order while
    the kernel runs an online softmax over split partials, so the two differ
    in summation order only. bf16 output rounding is 2^-8 relative, which is
    the floor here."""
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    worst = 0.0
    for T, W, ragged in ((8, 2048, False), (16, 2048, True), (32, 512, True), (1, 64, False)):
        num_slots = 4096
        q = torch.randn(T, MLA_H, MLA_D, dtype=torch.bfloat16, device=dev) * 0.3
        cache = (torch.randn(num_slots, MLA_D, device=dev) * 0.5).to(torch.float8_e4m3fn)
        slots = torch.randint(0, num_slots, (T, W), dtype=torch.int32, device=dev)
        if ragged:
            lens = torch.randint(1, W + 1, (T,), dtype=torch.int32, device=dev)
        else:
            lens = torch.full((T,), W, dtype=torch.int32, device=dev)
        sm, ks = MLA_D ** -0.5, 0.7
        got = mla_decode(q, cache.view(torch.uint8), slots, lens, sm, ks)
        ref = mla_decode_ref(q, cache, slots, lens, sm, ks)
        torch.cuda.synchronize()
        worst = max(worst, _rel_err(got.float(), ref.float()))
    if worst > 2e-2:
        logger.warning("[megakernel] selftest mla rel=%.2e -> DISARM", worst)
        return False
    logger.warning("[megakernel] selftest mla rel=%.2e -> ARM", worst)
    return True


def maybe_arm() -> None:
    """Cheap hot-path hook: arms exactly once, on the first eligible call.

    Never inside graph capture: arming compiles the extension and runs the
    self-tests, both of which allocate and sync -- illegal mid-capture. The
    one-shot flag stays unset so the next eager call retries (vLLM always
    warms up eager before capture, so the first hit is normally eager; this
    guard is the belt to that suspenders)."""
    global _armed_once
    if _armed_once:
        return
    try:
        import torch

        if torch.cuda.is_current_stream_capturing():
            return
        _armed_once = True
        arm()
    except Exception:
        logger.exception("[megakernel] arm failed; module inert")
        _armed_once = True
        for k in _ARMED:
            _ARMED[k] = False
