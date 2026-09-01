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


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


MASTER = _flag("VLLM_GLM53_MEGAKERNEL")
ENABLE_MHC = MASTER and _flag("VLLM_GLM53_MK_MHC")
ENABLE_GEMM = MASTER and _flag("VLLM_GLM53_MK_GEMM")
ENABLE_KDA = MASTER and _flag("VLLM_GLM53_MK_KDA")
KDA_SHADOW = MASTER and _flag("VLLM_GLM53_MK_KDA_SHADOW")
# W4 weights (e2m1 x per-16-group pow2 scale, expanded to EXACT e4m3 bytes
# in-kernel): the ledger's last unpulled lever, dense W8A8 -> W4, est.
# -3.7 ms/step. "1" covers every MK-GEMM linear except the KDA in_proj
# (recurrence path -- error accumulates in the state); "all" includes it.
ENABLE_W4 = ENABLE_GEMM and _flag("VLLM_GLM53_MK_W4")
W4_ALL = ENABLE_W4 and (os.environ.get("VLLM_GLM53_MK_W4", "")
                        .strip().lower() == "all")

# tolerances: the W8A8 GEMMs dominate; everything below them is fp32
_TOL_GEMM = 2e-2
_TOL_MHC = 1e-3     # fp32 port of the TileLang pair, bf16 rounding only
_TOL_KDA = 2e-2     # conv/gemm inputs differ the states by W8A8-order noise


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
_ARMED = {"mhc": False, "gemm": False, "kda": False}
_W4_ARMED = False  # set by the W4 self-test; packs exist but stay unused until then


def _build():
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
    os.makedirs("/root/.mk_build", exist_ok=True)
    _EXT = load(
        name="glm53_megakernel",
        sources=[_SRC],
        extra_cuda_cflags=["-O2", "-arch=sm_121a"],
        build_directory="/root/.mk_build",
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
        "yp": z(NCHUNK * MAX_TOK * NOUT),
        "rp": z(NCHUNK * MAX_TOK),
        "sq": z(MAX_TOK),
        "rsq": z(MAX_TOK),
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
def build_mk_weight(weight):
    """(wq uint8 [n_pad, k], ws fp32 [n_pad/128, k/128]) from bf16 [n, k]."""
    import torch

    n, k = weight.shape
    if k % 128 != 0:
        raise ValueError(f"K={k} not a multiple of 128")
    n_pad = _mk_pad128(n)
    wf = weight.float()
    q = torch.zeros(n_pad, k, dtype=torch.uint8, device=weight.device)
    s = torch.ones(n_pad // 128, k // 128, dtype=torch.float32,
                   device=weight.device)
    for n0 in range(0, n_pad, 128):
        rows = wf[n0:min(n0 + 128, n)]
        for k0 in range(0, k, 128):
            blk = rows[:, k0:k0 + 128]
            amax = float(blk.abs().max()) if blk.numel() else 0.0
            scale = _mk_pow2_scale(amax)
            if blk.numel():
                q[n0:n0 + rows.shape[0], k0:k0 + 128] = (
                    blk / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            s[n0 // 128, k0 // 128] = scale
    return q, s


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
    """(wq4 uint8 [n_pad, k/2] nibbles, ws4 int8 [n_pad, k/16] exponents).

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
    return wq4.contiguous(), ws4


def _stock_fp8_pair(w):
    """The stock deepgemm-layout pair for the same bf16 weight (self-tests
    and shadow arms need the arm they are diffing against)."""
    from vllm.model_executor.layers.glm53_fp8_dense import (
        _quantize_fp8_block_padded)

    q, ws, _rows, _cols = _quantize_fp8_block_padded(w)
    return q, ws


# ---------------------------------------------------------------------------
# MK_SEG_GEMM
# ---------------------------------------------------------------------------
def _gemm_call(x, mk_pack, n_rows):
    """mk_pack is (wq8, ws8) or (wq8, ws8, wq4, ws4); the W4 half, when
    present AND armed, replaces the fp8 stream (the pack is not additive:
    a W4 arm stores 0.56x the fp8 bytes, cheaper than the W8 arm itself)."""
    import torch

    out = torch.empty(x.shape[0], n_rows, dtype=torch.bfloat16,
                      device=x.device)
    w4 = mk_pack[2] if len(mk_pack) > 2 else None
    if w4 is not None and _W4_ARMED:
        _EXT.run_gemm_w4(x.contiguous(), w4, mk_pack[3], out, n_rows)
    else:
        _EXT.run_gemm(x.contiguous(), mk_pack[0], mk_pack[1], out, n_rows)
    return out


def gemm_w8a8(x, mk_pack, n_rows):
    """Fp8DenseMethod.apply hook: None = not armed/eligible (stock runs).

    A W4 pack REPLACES the fp8 stream, so a W4 self-test failure disarms
    the whole MK lane for that linear (stock deepgemm), not just the W4
    half -- there is no fp8 pack to fall back to by construction."""
    if not _ARMED["gemm"] or x.dim() != 2:
        return None
    has_w8 = mk_pack[0] is not None
    has_w4 = len(mk_pack) > 2 and mk_pack[2] is not None and _W4_ARMED
    if not (has_w8 or has_w4):
        return None
    ref = mk_pack[0] if has_w8 else mk_pack[2]
    if not _mk_gemm_eligible(x.shape[0], x.shape[1], ref.shape[0]):
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
         ws["rsq"].data_ptr(), ws["pmix"].data_ptr(),
         ws["ol_stash"].data_ptr(), _barrier_ptr(ws)],
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
            and conv_state.shape[2] >= 3 and conv_state.is_contiguous()
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
        # KDA packs are ALWAYS fp8 by policy (the recurrence path does not
        # ride the W4 arm). A W4-only pack on the linear (the VLLM_GLM53_
        # MK_W4=all boot) is a truthy (None, None, ...) tuple, so "or" is
        # not a usable presence test here -- index the fp8 half explicitly.
        def _fp8_pack(method, weight):
            p = getattr(method, "_mk", None)
            if p is not None and p[0] is not None:
                return p
            return build_mk_weight(weight)

        layer._mk_in_pack = _fp8_pack(in_m,
                                      layer.in_proj_qkvbfg_a.weight)
        layer._mk_o_pack = _fp8_pack(o_m, layer.o_proj.weight)
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
                delta_variant=0):
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
        drift = any(not (v <= _TOL_KDA) or math.isnan(v) for v in errs.values())
        KdaShadowArm._n_calls += 1
        if drift or KdaShadowArm._n_calls % 64 == 1:
            logger.warning("[megakernel] kda shadow #%d rel_errs=%s %s",
                           KdaShadowArm._n_calls,
                           {k: "%.2e" % v for k, v in errs.items()},
                           "DRIFT" if drift else "ok")


# ---------------------------------------------------------------------------
# boot arm + self-tests
# ---------------------------------------------------------------------------
def _rel_err(a, b) -> float:
    import torch

    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float(d)


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
    """Diff MK_SEG_GEMM against the stock quant+deepgemm pair."""
    import torch

    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm

    torch.manual_seed(0)
    dev = "cuda"
    for m, n in ((8, KDA_INPROJ_N), (16, HIDDEN), (32, 1024)):
        w = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.05
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        sq, sws = _stock_fp8_pair(w)
        ref = _fp8_dense_gemm(x, sq, sws)
        got = _gemm_call(x, build_mk_weight(w), n)
        torch.cuda.synchronize()
        e = _rel_err(got, ref)
        if e > _TOL_GEMM:
            logger.warning("[megakernel] selftest gemm m=%d n=%d rel=%.2e "
                           "-> DISARM", m, n, e)
            return False
    logger.warning("[megakernel] selftest gemm -> ARM")
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
        self.w_in = mkw(KDA_INPROJ_N, HIDDEN) * 0.02
        self.w_o = mkw(HIDDEN, KDA_OUT) * 0.02
        self.f_b = mkw(KDA_OUT, KDA_D) * 0.1
        self.g_b = mkw(KDA_OUT, KDA_D) * 0.1
        self.conv_w = mkw(KDA_QKV, 4, dt=torch.float32) * 0.2
        self.a_log = mkw(1, 1, KDA_H, 1, dt=torch.float32) * 0.1
        self.dt_bias = torch.zeros(KDA_OUT, dtype=torch.float32,
                                   device="cuda")
        self.conv_st = mkw(2, KDA_QKV, 3, dt=torch.float32)
        self.rec_st = (mkw(2, KDA_H, KDA_D, KDA_D, dt=torch.float32) * 0.1)
        self.cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        self.sidx = torch.full((1, 8), self.SLOT, dtype=torch.int32,
                               device="cuda")
        self.nacc = torch.tensor([acc], dtype=torch.int32, device="cuda")
        self._mk_cache = None       # (layer stand-in, meta), quantized once
        self._stock_cache = None    # (sq_in, sws_in, sq_o, sws_o), once

    def _layer_stand_in(self):
        if self._mk_cache is not None:
            return self._mk_cache
        f = self
        in_mk, o_mk = build_mk_weight(f.w_in), build_mk_weight(f.w_o)

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
        self.onorm_w = (torch.rand(KDA_OUT, device="cuda") * 0.4
                        + 0.8).to(torch.bfloat16)
        la.o_norm = _P()
        la.o_norm.eps = 1e-5
        la.o_norm.weight = torch.nn.Parameter(self.onorm_w)
        self._mk_cache = (la, _Meta())
        return self._mk_cache

    def mk_run(self, delta_variant=0):
        """MK arm on cloned states -> dict(out, conv_state, rec_state).

        delta_variant sweeps the retrieval/write operand order (see the .cu
        comment): the stock source is not in this repo, so the boot settles
        which variant matches fused_recurrent_kda."""
        import torch

        la, meta = self._layer_stand_in()
        conv_mk, rec_mk = self.conv_st.clone(), self.rec_st.clone()
        out = torch.empty(self.T, HIDDEN, dtype=torch.bfloat16,
                          device="cuda")
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

    def stock_run(self):
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
        sq_in, sws_in, sq_o, sws_o = self._stock_cache
        proj = _fp8_dense_gemm(self.x, sq_in, sws_in)
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
        o_norm = FusedRMSNormGated(KDA_D, eps=1e-5, activation="sigmoid")
        core = o_norm(attn.view(T, KDA_H, KDA_D), g2.view(T, KDA_H, KDA_D))
        out = _fp8_dense_gemm(core.reshape(T, KDA_OUT), sq_o, sws_o)
        torch.cuda.synchronize()
        return {"out": out, "conv_state": conv_ref, "rec_state": rec_ref}


_KDA_VARIANT = 0  # settled at arm time by the delta sweep


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


def _selftest_w4() -> bool:
    """Two gates:

    (a) EXACT expansion: weights built ON the e2m1 x 2^s grid quantize
        losslessly, so the W4 kernel must reproduce the W8 kernel to fp32
        accumulation noise (<= 1e-5). This proves the nibble/scale/expansion
        pipeline bit-for-bit; anything above the gate is a layout bug, not
        quantization.
    (b) BY-DESIGN error: random weights through W4 vs the stock W8A8 pair,
        gated at 0.15 (e2m1's by-design error is 0.02-0.08 rel on row
        blocks -- the same tolerance class the fp8-dense module uses for
        its w4 arm).
    """
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    n, k, m = 1024, 4096, 8
    # (a) exact fixture: random codes x random legal exponents
    code = torch.randint(0, 8, (n, k // 16, 16), device=dev)
    sexp = torch.randint(-5, 7, (n, k // 16, 1), device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    w_exact = (grid[code] * torch.exp2(sexp.float())) *                  torch.where(torch.randn_like(code.float()) < 0, -1.0, 1.0)
    w_exact = w_exact.view(n, k).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)

    def _run(pack):
        import torch as _t

        out = _t.empty(m, n, dtype=_t.bfloat16, device=dev)
        if len(pack) == 4 and pack[2] is not None:
            _EXT.run_gemm_w4(x.contiguous(), pack[2], pack[3], out, n)
        else:
            _EXT.run_gemm(x.contiguous(), pack[0], pack[1], out, n)
        return out

    got4 = _run((None, None) + build_mk_weight_w4(w_exact))
    got8 = _run(build_mk_weight(w_exact))
    torch.cuda.synchronize()
    e_exact = _rel_err(got4, got8)
    if e_exact > 1e-5:
        logger.warning("[megakernel] selftest w4 EXACT rel=%.2e -> DISARM "
                       "(expansion is not bit-exact)", e_exact)
        return False
    # (b) by-design error vs the stock pair
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.05
    sq, sws = _stock_fp8_pair(w)
    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm

    ref = _fp8_dense_gemm(x, sq, sws)
    got = _run((None, None) + build_mk_weight_w4(w))
    torch.cuda.synchronize()
    e = _rel_err(got, ref)
    ok = e <= 0.15
    logger.warning("[megakernel] selftest w4 exact=%.2e by-design=%.2e -> %s",
                   e_exact, e, "ARM" if ok else "DISARM")
    return ok


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
        if ENABLE_W4:
            global _W4_ARMED
            _W4_ARMED = _gate("w4", _selftest_w4)
    if ENABLE_KDA:
        _ARMED["kda"] = _gate("kda", _selftest_kda)
    logger.warning("[megakernel] armed=%s w4=%s shadow_kda=%s",
                   dict(_ARMED), _W4_ARMED, KDA_SHADOW)


_armed_once = False


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
