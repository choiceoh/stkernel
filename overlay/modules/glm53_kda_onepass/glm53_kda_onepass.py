# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 KDA decode micro-fusions (bundle 2 of the launch-count campaign).

Two opt-in Triton paths for the stock (non-megakernel) KDA layer, each a
kill-switched knob that defaults to the stock chain:

* ``VLLM_GLM53_KDA_DUAL_GEMM=1`` -- the two low-rank gate GEMMs of a layer,
  ``f_b_proj(f_a)`` and ``g_b_proj(g_a)``, as ONE launch. ``f_a`` and ``g_a``
  are adjacent 128-column slices of the merged ``in_proj_qkvbfg_a`` output,
  so one program reads both slices once and runs two ``[M,128]x[128,N]`` dots.
  34 layers x 2 cutlass wmma launches (4.7-6 us each, grid (8,16,1)) become
  34 launches. Numerics class: fp32-accumulated bf16 GEMM either way; the
  probe reports whether the K=128 accumulation is bit-identical to cuBLAS.

* ``VLLM_GLM53_KDA_ONEPASS=1`` -- for the pure spec-verify decode step (every
  request is a draft-verify block, no prefill rows), the short conv, the
  three q/k/v ``.contiguous()`` copies + the beta copy, the gated delta-rule
  recurrence and the gated RMSNorm (7 launches/layer, 238/step) collapse
  into ONE launch per layer. The kernel is a transcription of the three
  stock Triton kernels (``_causal_conv1d_update_kernel`` IS_SPEC_DECODING
  branch, ``fused_recurrent_gated_delta_rule_fwd_kernel`` KDA/COMPUTE_GATE
  branch, ``layer_norm_gated_fwd_kernel`` RMS/sigmoid branch) with the same
  fp32 op order, the conv output rounded to bf16 exactly where the stock
  chain stores it, so the recurrent state and the pre-norm output are
  bit-identical to stock by construction; only the norm's 128-wide sum of
  squares uses a different reduction tree (one warp per row instead of the
  stock [32,128]/4-warp tile) -- the bf16 reduce-order class, NOT bit-exact.

  Grid = (V/BV, N*H) single-warp programs, the stock recurrent geometry.
  The two cross-program dependencies that the separate launches used to
  provide are last-arriver counters (one atomic per program, no spinning,
  no co-residency requirement): (1) the conv state of a head is rewritten
  only after all V-blocks of that head have read the old history -- every
  program reads the 3 history taps of all 128 q/k channels, so the write
  cannot be split by channel; (2) the gated norm needs the whole 128-wide
  head row, so the last V-block to finish a head normalizes its rows in
  place. Counters live in a module buffer the kernel resets itself.

Applicability is decided per call from the layer's metadata; anything that
is not a pure spec-verify step keeps the stock chain. Prefill is untouched.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

DUAL_GEMM_ENV = "VLLM_GLM53_KDA_DUAL_GEMM"
ONEPASS_ENV = "VLLM_GLM53_KDA_ONEPASS"

# Fleet geometry (glm53-redhat-nvfp4, TP=4): 16 local heads x 128, conv width
# 4, verify block 8 -> conv state_len 10. Everything is read from the tensors
# at call time; these are only the shapes the kernels were tuned for.
_MAX_DUAL_M = 32


def dual_gemm_enabled() -> bool:
    """Exact ``1`` arms; anything else is the stock two-GEMM path."""
    return os.environ.get(DUAL_GEMM_ENV, "").strip() == "1"


def onepass_enabled() -> bool:
    """Exact ``1`` arms; anything else is the stock conv/recurrent/norm chain."""
    return os.environ.get(ONEPASS_ENV, "").strip() == "1"


# ---------------------------------------------------------------------------
# f_b + g_b dual GEMM
# ---------------------------------------------------------------------------


@triton.jit
def _dual_gate_gemm_kernel(
    x_ptr,  # [M, 2*KD] bf16 view: f_a then g_a, row stride stride_x
    stride_x,
    wf_ptr,  # [N, KD] bf16 contiguous
    wg_ptr,  # [N, KD] bf16 contiguous
    of_ptr,  # [M, N] bf16 contiguous
    og_ptr,  # [M, N] bf16 contiguous
    M,
    N: tl.constexpr,
    KD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, KD)
    m_mask = rm < M
    xf = tl.load(
        x_ptr + rm[:, None] * stride_x + rk[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    xg = tl.load(
        x_ptr + rm[:, None] * stride_x + KD + rk[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    wf = tl.load(wf_ptr + rn[:, None] * KD + rk[None, :])
    wg = tl.load(wg_ptr + rn[:, None] * KD + rk[None, :])
    accf = tl.dot(xf, tl.trans(wf))
    accg = tl.dot(xg, tl.trans(wg))
    tl.store(
        of_ptr + rm[:, None] * N + rn[None, :],
        accf.to(of_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )
    tl.store(
        og_ptr + rm[:, None] * N + rn[None, :],
        accg.to(og_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


def dual_gemm_applicable(x_fg: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor) -> bool:
    """The shapes the dual kernel assumes: a 2-D bf16 slice with unit inner
    stride holding f_a|g_a side by side, two contiguous bf16 [N, KD] weights
    with KD a power of two >= 16, N a multiple of 128, and a decode-sized M."""
    if x_fg.dim() != 2 or x_fg.dtype != torch.bfloat16 or x_fg.stride(1) != 1:
        return False
    if w_f.dim() != 2 or w_g.dim() != 2 or w_f.shape != w_g.shape:
        return False
    if w_f.dtype != torch.bfloat16 or w_g.dtype != torch.bfloat16:
        return False
    if not (w_f.is_contiguous() and w_g.is_contiguous()):
        return False
    n_out, kd = w_f.shape
    if x_fg.shape[1] != 2 * kd or kd < 16 or (kd & (kd - 1)) != 0 or kd > 256:
        return False
    if n_out % 128 != 0:
        return False
    return 0 < x_fg.shape[0] <= _MAX_DUAL_M


def dual_gate_gemm(
    x_fg: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_fg[:, :KD] @ w_f.T, x_fg[:, KD:] @ w_g.T)`` in one launch."""
    m = x_fg.shape[0]
    n_out, kd = w_f.shape
    of = torch.empty((m, n_out), dtype=x_fg.dtype, device=x_fg.device)
    og = torch.empty((m, n_out), dtype=x_fg.dtype, device=x_fg.device)
    # 16-row tiles: the probe found the [16 x 128 x 128] tl.dot bit-identical
    # to cuBLAS' 16x16 wmma tile, and a 32-row tile 1 ulp off on a few
    # elements -- so M > 16 takes a second row block instead of a taller one.
    block_m = 16
    _dual_gate_gemm_kernel[(n_out // 128, (m + block_m - 1) // block_m)](
        x_fg,
        x_fg.stride(0),
        w_f,
        w_g,
        of,
        og,
        m,
        N=n_out,
        KD=kd,
        BLOCK_M=block_m,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
    )
    return of, og


# ---------------------------------------------------------------------------
# conv + recurrent + gated norm one-pass (pure spec-verify decode)
# ---------------------------------------------------------------------------


@triton.jit
def _conv_taps(x0, x1, x2, x3, w0, w1, w2, w3):
    # Stock _causal_conv1d_update_kernel, KERNEL_WIDTH == 4, no bias:
    #   acc = 0; for j: acc += x_j * w_j; acc = acc / (1 + exp(-acc))
    # x_j are the bf16 taps (history then the token), w_j the fp32 weights.
    acc = tl.zeros([x0.shape[0]], dtype=tl.float32)
    acc += x0 * w0
    acc += x1 * w1
    acc += x2 * w2
    acc += x3 * w3
    acc = acc / (1 + tl.exp(-acc))
    # The stock kernel stores acc into the bf16 qkv buffer and the recurrent
    # kernel reloads it: round exactly there.
    return acc.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _kda_onepass_spec_kernel(
    x_ptr,  # merged projection rows [T_all, XS] bf16: q | k | v | beta | ...
    stride_x,
    g1_ptr,  # [T_all, H*K] bf16 (f_b output, raw gate logits)
    g2_ptr,  # [T_all, H*V] bf16 (g_b output, norm gate)
    w_ptr,  # conv weight [DIM, W] fp32
    stride_w_dim,
    stride_w_width,
    cs_ptr,  # conv state
    stride_cs_line,
    stride_cs_dim,
    stride_cs_tok,
    h0_ptr,  # recurrent state [lines, H, V, K] fp32
    stride_h0_line,
    cu_ptr,  # [N+1] int32 spec_query_start_loc
    idx_ptr,  # [N, S] int32 spec_state_indices
    stride_idx_seq,
    acc_ptr,  # [N] int32 num_accepted_tokens
    a_log_ptr,  # [H] fp32
    g_bias_ptr,  # [H*K] fp32
    nw_ptr,  # [V] norm weight
    o_ptr,  # [T_all, H*V] bf16 output (normalized in place)
    ctr_ptr,  # [2*NH_TOTAL] int32, all zero between launches
    nh_total,
    scale,
    eps,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    Q_OFF: tl.constexpr,
    K_OFF: tl.constexpr,
    V_OFF: tl.constexpr,
    B_OFF: tl.constexpr,
    SEQLEN: tl.constexpr,  # max_query_len (verify block)
    STATE_LEN: tl.constexpr,  # conv state slots = W-1 + SEQLEN-1
    LOWER_BOUND: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    bos = tl.load(cu_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    n_acc = tl.load(acc_ptr + i_n).to(tl.int64)
    # Stock conv reads/writes the line at slot 0; the stock recurrence resumes
    # from slot [n, acc-1]. Both skip the null block (0); a padded request
    # has every slot at 0, so all NV programs of a (request, head) agree.
    line = tl.load(idx_ptr + i_n * stride_idx_seq).to(tl.int64)
    state_idx = tl.load(idx_ptr + i_n * stride_idx_seq + n_acc - 1).to(tl.int64)
    if (line == 0) | (state_idx <= 0):
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    q_ch = Q_OFF + i_h * K + o_k
    k_ch = K_OFF + i_h * K + o_k
    v_ch = V_OFF + i_h * V + o_v

    # ---- conv history: the (acc-1)-th window of the state line, 3 taps ----
    tok_off = n_acc - 1
    cs_line = cs_ptr + line * stride_cs_line
    cs_hist = cs_line + tok_off * stride_cs_tok
    q0 = tl.load(cs_hist + q_ch * stride_cs_dim)
    q1 = tl.load(cs_hist + q_ch * stride_cs_dim + 1 * stride_cs_tok)
    q2 = tl.load(cs_hist + q_ch * stride_cs_dim + 2 * stride_cs_tok)
    k0 = tl.load(cs_hist + k_ch * stride_cs_dim)
    k1 = tl.load(cs_hist + k_ch * stride_cs_dim + 1 * stride_cs_tok)
    k2 = tl.load(cs_hist + k_ch * stride_cs_dim + 2 * stride_cs_tok)
    v0 = tl.load(cs_hist + v_ch * stride_cs_dim)
    v1 = tl.load(cs_hist + v_ch * stride_cs_dim + 1 * stride_cs_tok)
    v2 = tl.load(cs_hist + v_ch * stride_cs_dim + 2 * stride_cs_tok)
    wq0 = tl.load(w_ptr + q_ch * stride_w_dim + 0 * stride_w_width)
    wq1 = tl.load(w_ptr + q_ch * stride_w_dim + 1 * stride_w_width)
    wq2 = tl.load(w_ptr + q_ch * stride_w_dim + 2 * stride_w_width)
    wq3 = tl.load(w_ptr + q_ch * stride_w_dim + 3 * stride_w_width)
    wk0 = tl.load(w_ptr + k_ch * stride_w_dim + 0 * stride_w_width)
    wk1 = tl.load(w_ptr + k_ch * stride_w_dim + 1 * stride_w_width)
    wk2 = tl.load(w_ptr + k_ch * stride_w_dim + 2 * stride_w_width)
    wk3 = tl.load(w_ptr + k_ch * stride_w_dim + 3 * stride_w_width)
    wv0 = tl.load(w_ptr + v_ch * stride_w_dim + 0 * stride_w_width)
    wv1 = tl.load(w_ptr + v_ch * stride_w_dim + 1 * stride_w_width)
    wv2 = tl.load(w_ptr + v_ch * stride_w_dim + 2 * stride_w_width)
    wv3 = tl.load(w_ptr + v_ch * stride_w_dim + 3 * stride_w_width)

    # ---- last arriver of this head rewrites the head's conv state ----
    # Every program above has read the old history it needs; the one that
    # increments the counter to NV writes the rolled window for all 3*128
    # channels of head i_h (the stock IS_SPEC_DECODING roll: slots
    # [0, VAL) <- old[acc + j], slots [VAL, VAL+T) <- the block's tokens;
    # VAL = STATE_LEN - SEQLEN, the per-request state_len is VAL + T).
    tl.debug_barrier()
    arrived = tl.atomic_add(ctr_ptr + i_nh, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if arrived == NV - 1:
        VAL: tl.constexpr = STATE_LEN - SEQLEN
        o_vv = tl.arange(0, V)
        vq_ch = Q_OFF + i_h * K + o_k
        vk_ch = K_OFF + i_h * K + o_k
        vv_ch = V_OFF + i_h * V + o_vv
        for j in tl.static_range(STATE_LEN):
            if j < VAL:
                src = cs_line + (tok_off + 1 + j) * stride_cs_tok
                nq = tl.load(src + vq_ch * stride_cs_dim)
                nk = tl.load(src + vk_ch * stride_cs_dim)
                nv = tl.load(src + vv_ch * stride_cs_dim)
                dst = cs_line + j * stride_cs_tok
                tl.store(dst + vq_ch * stride_cs_dim, nq)
                tl.store(dst + vk_ch * stride_cs_dim, nk)
                tl.store(dst + vv_ch * stride_cs_dim, nv)
            else:
                if j - VAL < T:
                    row = x_ptr + (bos + (j - VAL)) * stride_x
                    nq = tl.load(row + vq_ch)
                    nk = tl.load(row + vk_ch)
                    nv = tl.load(row + vv_ch)
                    dst = cs_line + j * stride_cs_tok
                    tl.store(dst + vq_ch * stride_cs_dim, nq)
                    tl.store(dst + vk_ch * stride_cs_dim, nk)
                    tl.store(dst + vv_ch * stride_cs_dim, nv)
        tl.atomic_xchg(ctr_ptr + i_nh, 0)

    # ---- recurrence (stock fused_recurrent_gated_delta_rule_fwd_kernel,
    #      IS_KDA + COMPUTE_GATE + SIGMOID_BETA + USE_QK_L2NORM_IN_KERNEL) ----
    b_a_log = tl.exp(tl.load(a_log_ptr + i_h).to(tl.float32))
    p_gk = g1_ptr + (bos * H + i_h) * K + o_k
    p_o = o_ptr + (bos * H + i_h) * V + o_v
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k < K)[None, :]
    p_h0 = (
        h0_ptr + state_idx * stride_h0_line + i_h * V * K
        + o_v[:, None] * K + o_k[None, :]
    )
    b_h = tl.zeros([BV, K], dtype=tl.float32)
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # The token loop keeps the stock recurrent kernel's exact shape: every
    # operand loaded inside the iteration, no loop-carried prefetch. A
    # prefetched (or hoisted) [K] operand changes Triton's layout choice for
    # the loop's vectors and with it the l2norm reduction tree -- the
    # recurrent state came out 1 ulp off the stock chain on ~3% of elements
    # (probe, 2026-09-04). Bit-exactness is worth more than the 3 us/layer.
    for t in range(0, T):
        row = x_ptr + (bos + t) * stride_x
        q3 = tl.load(row + q_ch)
        k3 = tl.load(row + k_ch)
        v3 = tl.load(row + v_ch)
        b_q = _conv_taps(q0, q1, q2, q3, wq0, wq1, wq2, wq3)
        b_k = _conv_taps(k0, k1, k2, k3, wk0, wk1, wk2, wk3)
        b_v = _conv_taps(v0, v1, v2, v3, wv0, wv1, wv2, wv3)
        q0 = q1
        q1 = q2
        q2 = q3
        k0 = k1
        k1 = k2
        k2 = k3
        v0 = v1
        v1 = v2
        v2 = v3

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_gk = tl.load(p_gk).to(tl.float32)
        b_gk += tl.load(g_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
        b_h *= tl.exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = tl.load(row + B_OFF + i_h).to(tl.float32)
        b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        final_idx = tl.load(idx_ptr + i_n * stride_idx_seq + t).to(tl.int64)
        if final_idx > 0:
            p_ht = (
                h0_ptr + final_idx * stride_h0_line + i_h * V * K
                + o_v[:, None] * K + o_k[None, :]
            )
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_gk += H * K
        p_o += H * V

    # ---- last arriver of this head applies the gated RMSNorm in place ----
    tl.debug_barrier()
    done = tl.atomic_add(ctr_ptr + nh_total + i_nh, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if done == NV - 1:
        # One row at a time, 1-D vectors only: a [SEQLEN, V] tile here
        # changed Triton's layout assignment for the whole kernel and put
        # the recurrent state 1 ulp off stock on ~3% of elements (probe,
        # 2026-09-05) -- the same trap as a prefetched loop operand.
        o_vv = tl.arange(0, V)
        b_w = tl.load(nw_ptr + o_vv).to(tl.float32)
        for t in range(0, T):
            p = o_ptr + ((bos + t) * H + i_h) * V + o_vv
            b_x = tl.load(p, cache_modifier=".cg").to(tl.float32)
            b_var = tl.sum(b_x * b_x, axis=0) / V
            b_rstd = 1 / tl.sqrt(b_var + eps)
            b_y = b_x * b_rstd
            b_y = b_y * b_w
            b_g = tl.load(g2_ptr + ((bos + t) * H + i_h) * V + o_vv).to(tl.float32)
            b_y = b_y * tl.sigmoid(b_g)
            tl.store(p, b_y.to(p.dtype.element_ty))
        tl.atomic_xchg(ctr_ptr + nh_total + i_nh, 0)


_COUNTERS: dict[torch.device, torch.Tensor] = {}
# 2 counters x (requests x heads); 4096 covers 256 requests of 16 heads.
_COUNTER_SLOTS = 2 * 4096


def prepare_counters(device: torch.device) -> torch.Tensor:
    """Allocate the arrival counters once, off-capture (the wiring calls this
    when it resolves the knobs, i.e. on the first eager forward). Every launch
    slices the same buffer, so the address a captured graph records stays
    valid and no allocation ever happens under capture."""
    buf = _COUNTERS.get(device)
    if buf is None:
        buf = torch.zeros((_COUNTER_SLOTS,), dtype=torch.int32, device=device)
        _COUNTERS[device] = buf
    return buf


def _counters(device: torch.device, nh_total: int) -> torch.Tensor:
    """Zeroed arrival counters, 2 rows of nh_total. The kernel resets every
    slot it used, so one buffer serves every layer and every replay."""
    if 2 * nh_total > _COUNTER_SLOTS:
        raise ValueError(f"kda one-pass: {nh_total} request-heads exceed the "
                         f"counter buffer ({_COUNTER_SLOTS // 2})")
    return prepare_counters(device)


def onepass_applicable(
    projected: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    spec_query_start_loc,
    spec_state_indices,
    num_accepted_tokens,
    out: torch.Tensor,
    *,
    local_num_heads: int,
    head_dim: int,
    conv_size: int,
    max_query_len: int,
) -> bool:
    """Everything the kernel hard-codes, checked on the real tensors."""
    h, d = local_num_heads, head_dim
    if h <= 0 or d != 128:
        return False
    if conv_size != 4:
        return False
    if conv_bias is not None:
        return False
    proj = h * d
    if projected.dim() != 2 or projected.dtype != torch.bfloat16 or projected.stride(1) != 1:
        return False
    if projected.shape[1] < 3 * proj + h:
        return False
    if g1.dtype != torch.bfloat16 or g2.dtype != torch.bfloat16:
        return False
    if not (g1.is_contiguous() and g2.is_contiguous()):
        return False
    if g1.numel() != projected.shape[0] * proj or g2.numel() != projected.shape[0] * proj:
        return False
    if conv_weight.dim() != 2 or conv_weight.shape != (3 * proj, conv_size):
        return False
    if conv_weight.dtype != torch.float32:
        return False
    if conv_state.dim() != 3 or conv_state.dtype != torch.bfloat16:
        return False
    # (lines, dim, state_len) in either orientation; strides carry the layout
    dims = set(conv_state.shape[1:])
    state_len = conv_size - 1 + (max_query_len - 1)
    if 3 * proj not in dims or state_len not in dims or 3 * proj == state_len:
        return False
    if recurrent_state.dim() != 4 or recurrent_state.dtype != torch.float32:
        return False
    if tuple(recurrent_state.shape[1:]) != (h, d, d):
        return False
    if not recurrent_state[0].is_contiguous():
        return False
    if spec_query_start_loc is None or spec_state_indices is None or num_accepted_tokens is None:
        return False
    if spec_state_indices.dim() != 2 or spec_state_indices.shape[1] < max_query_len:
        return False
    if spec_state_indices.dtype != torch.int32 or spec_query_start_loc.dtype != torch.int32:
        return False
    if num_accepted_tokens.dtype != torch.int32:
        return False
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    if out.numel() < projected.shape[0] * proj:
        return False
    return max_query_len >= 1


def kda_onepass_spec(
    projected: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    num_spec_decodes: int,
    local_num_heads: int,
    head_dim: int,
    max_query_len: int,
    lower_bound: float,
    eps: float,
    block_v: int = 8,
) -> None:
    """One launch: conv update + gated delta-rule recurrence + gated RMSNorm.

    ``projected`` is the merged in_proj output ([tokens, 3*proj + h + ...]);
    ``out`` receives the normalized attention output ([tokens, proj], bf16)
    -- the same buffer the stock chain hands to fused_recurrent_kda(out=)
    and then normalizes in place. States are updated in place exactly like
    the stock kernels (conv line at spec slot 0, recurrent slots per token).
    """
    h, d = local_num_heads, head_dim
    proj = h * d
    n = num_spec_decodes
    if n <= 0:
        return
    state_len = conv_state.shape[1] if conv_state.shape[1] != 3 * proj else conv_state.shape[2]
    if conv_state.shape[1] == 3 * proj:
        stride_cs_dim, stride_cs_tok = conv_state.stride(1), conv_state.stride(2)
    else:
        stride_cs_tok, stride_cs_dim = conv_state.stride(1), conv_state.stride(2)
    nv = d // block_v
    nh_total = n * h
    ctr = _counters(projected.device, nh_total)
    # row 1 of the counters starts at nh_total; both rows are zero on entry
    _kda_onepass_spec_kernel[(nv, nh_total)](
        projected,
        projected.stride(0),
        g1,
        g2,
        conv_weight,
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_state,
        conv_state.stride(0),
        stride_cs_dim,
        stride_cs_tok,
        recurrent_state,
        recurrent_state.stride(0),
        spec_query_start_loc,
        spec_state_indices,
        spec_state_indices.stride(0),
        num_accepted_tokens,
        a_log.reshape(-1),
        g_bias.reshape(-1),
        norm_weight,
        out,
        ctr,
        nh_total,
        d ** -0.5,
        eps,
        H=h,
        K=d,
        V=d,
        BV=block_v,
        NV=nv,
        Q_OFF=0,
        K_OFF=proj,
        V_OFF=2 * proj,
        B_OFF=3 * proj,
        SEQLEN=max_query_len,
        STATE_LEN=state_len,
        LOWER_BOUND=lower_bound,
        num_warps=1,
        num_stages=3,
    )


_ANNOUNCED: set[str] = set()


def announce_once(tag: str, msg: str) -> None:
    """One log line per verdict, not per layer or per step."""
    if tag in _ANNOUNCED:
        return
    _ANNOUNCED.add(tag)
    logger.info("[kda-onepass] %s", msg)
