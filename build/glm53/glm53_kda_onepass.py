# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 KDA decode micro-fusions (bundle 2 of the launch-count campaign).

Two opt-in Triton paths for the stock (non-megakernel) KDA layer, each a
kill-switched knob that defaults to the stock chain. The model overlay
(glm53_mk_kda_wiring/glm5next_kda.py) only imports this module, calls
``resolve()`` once and then ``gate_gemms()`` / ``spec_onepass()``; every
knob, guard, self-test, announce line and counter lives here.

* ``VLLM_GLM53_KDA_DUAL_GEMM=1`` -- the two low-rank gate GEMMs of a layer,
  ``f_b_proj(f_a)`` and ``g_b_proj(g_a)``, as ONE launch. ``f_a`` and ``g_a``
  are adjacent 128-column slices of the merged ``in_proj_qkvbfg_a`` output,
  so one program reads both slices once and runs two ``[16,128]x[128,BN]``
  dots. 34 layers x 2 cutlass wmma launches become 34 launches. The 16-row
  tile matches cuBLAS' 16x16 wmma accumulation bit for bit (probe); decode
  M only (M <= 32), prefill keeps the stock GEMMs.

* ``VLLM_GLM53_KDA_ONEPASS=1`` -- for the pure spec-verify decode step (every
  request is a draft-verify block, no prefill rows), the short conv, the
  three q/k/v ``.contiguous()`` copies + the beta copy, the gated delta-rule
  recurrence and the gated RMSNorm (7 launches/layer) collapse into ONE
  launch per layer. The kernel is a transcription of the three stock Triton
  kernels (``_causal_conv1d_update_kernel`` IS_SPEC_DECODING branch,
  ``fused_recurrent_gated_delta_rule_fwd_kernel`` KDA/COMPUTE_GATE branch,
  ``layer_norm_gated_fwd_kernel`` RMS/sigmoid branch) with the same fp32 op
  order and the conv output rounded to bf16 exactly where the stock chain
  stores it, so conv state, recurrent state and output are bit-identical to
  stock on the probe's fleet-shape cases (the norm's reduction tree is not
  the stock kernel's, so that class stays declared reduce-order).

  Grid = (V/BV, N*H) single-warp programs, the stock recurrent geometry.
  The two cross-program dependencies that the separate launches used to
  provide are last-arriver counters (one ``atom.acq_rel.gpu`` per program,
  no spinning, no co-residency requirement): (1) the conv state of a head
  is rewritten only after all V-blocks of that head have read the old
  history -- every program reads the 3 history taps of all 128 q/k
  channels, so the write cannot be split by channel; (2) the gated norm
  needs the whole 128-wide head row, so the last V-block to finish a head
  normalizes its rows in place. The counters are monotonic: the last
  arriver is the program that brings the count to a multiple of NV, so no
  launch has to reset anything and an int32 wrap is harmless (NV is a
  power of two).

  The kernel keeps the stock kernels' skip semantics separately: the conv
  part runs iff the request's conv line (spec slot 0) is not the null
  block, the recurrence iff its resume slot ``[n, acc-1]`` is valid, and
  when the conv is skipped the recurrence consumes the raw projection rows
  exactly as the stock chain does.

``resolve()`` runs a numerical self-test of the one-pass against the stock
chain once (first eager forward, off-capture) and disarms the knob on any
mismatch -- a Triton or image bump that changes the layout assignment
(and with it a reduction tree) must not serve silently. Applicability is
otherwise decided per call from the layer's metadata; anything that is
not a pure spec-verify step keeps the stock chain. Prefill is untouched.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

DUAL_GEMM_ENV = "VLLM_GLM53_KDA_DUAL_GEMM"
ONEPASS_ENV = "VLLM_GLM53_KDA_ONEPASS"
KNOB_ENVS = (DUAL_GEMM_ENV, ONEPASS_ENV)

# Fleet geometry (glm53-redhat-nvfp4, TP=4): 16 local heads x 128, conv width
# 4, verify block 8 -> conv state_len 10. Everything is read from the tensors
# at call time; these are only the shapes the kernels were tuned for.
_MAX_DUAL_M = 32
# Probe sweep 2026-09-05 (34 layers, graph replay, second of two replays,
# M=8): BLOCK_N 128 = 16 CTAs on 48 SMs, 7.6 us/layer; 64 -> 6.7; 32 -> 6.4
# (64 CTAs). The per-element K=128 accumulation is the same tile-wise, so
# the cuBLAS bit-match for M <= 16 holds for every width.
_DUAL_BLOCK_N = 32
# Arrival counters: 2 rows x (requests x heads); 4096 covers 256 requests of
# 16 heads. A batch beyond that is declined (stock chain), never raised.
_MAX_REQ_HEADS = 4096
_COUNTER_SLOTS = 2 * _MAX_REQ_HEADS


def dual_gemm_enabled() -> bool:
    """Exact ``1`` arms; anything else is the stock two-GEMM path."""
    return os.environ.get(DUAL_GEMM_ENV, "").strip() == "1"


def onepass_enabled() -> bool:
    """Exact ``1`` arms; anything else is the stock conv/recurrent/norm chain."""
    return os.environ.get(ONEPASS_ENV, "").strip() == "1"


# ---------------------------------------------------------------------------
# announce once per verdict (message built only when it will be logged)
# ---------------------------------------------------------------------------

_ANNOUNCED: set[str] = set()


def announce_pending(tag: str) -> bool:
    return tag not in _ANNOUNCED


def announce_once(tag: str, msg: str | Callable[[], str], *, warn: bool = False) -> None:
    """One log line per verdict, not per layer or per step. ``msg`` may be a
    callable so the call site pays no string formatting after the first."""
    if tag in _ANNOUNCED:
        return
    _ANNOUNCED.add(tag)
    text = msg() if callable(msg) else msg
    (logger.warning if warn else logger.info)("[kda-onepass] %s", text)


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
    stride holding f_a|g_a side by side, two contiguous bf16 [N, 128] weights
    (the only KD the probe validated), N a multiple of the N tile, and a
    decode-sized M."""
    if x_fg.dim() != 2 or x_fg.dtype != torch.bfloat16 or x_fg.stride(1) != 1:
        return False
    if w_f.dim() != 2 or w_g.dim() != 2 or w_f.shape != w_g.shape:
        return False
    if w_f.dtype != torch.bfloat16 or w_g.dtype != torch.bfloat16:
        return False
    if not (w_f.is_contiguous() and w_g.is_contiguous()):
        return False
    n_out, kd = w_f.shape
    if kd != 128 or x_fg.shape[1] != 2 * kd:
        return False
    if n_out % _DUAL_BLOCK_N != 0:
        return False
    return 0 < x_fg.shape[0] <= _MAX_DUAL_M


def dual_gate_gemm(
    x_fg: torch.Tensor, w_f: torch.Tensor, w_g: torch.Tensor, block_n: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_fg[:, :KD] @ w_f.T, x_fg[:, KD:] @ w_g.T)`` in one launch."""
    m = x_fg.shape[0]
    n_out, kd = w_f.shape
    bn = _DUAL_BLOCK_N if block_n is None else block_n
    of = torch.empty((m, n_out), dtype=x_fg.dtype, device=x_fg.device)
    og = torch.empty((m, n_out), dtype=x_fg.dtype, device=x_fg.device)
    # 16-row tiles: the probe found the [16 x 128 x 128] tl.dot bit-identical
    # to cuBLAS' 16x16 wmma tile, and a 32-row tile 1 ulp off on a few
    # elements -- so M > 16 takes a second row block instead of a taller one.
    block_m = 16
    _dual_gate_gemm_kernel[(n_out // bn, (m + block_m - 1) // block_m)](
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
        BLOCK_N=bn,
        num_warps=4,
        num_stages=1,
    )
    return of, og


def gate_gemms(
    projected: torch.Tensor, offset: int, head_dim: int,
    w_f: torch.Tensor, w_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The layer's f_b/g_b pair from the merged projection, or None when the
    stock GEMMs must run. Prefill (M > 32) is stock by design and is not
    announced; a decode-sized shape the kernel refuses is announced once."""
    x_fg = projected[:, offset : offset + 2 * head_dim]
    if dual_gemm_applicable(x_fg, w_f, w_g):
        g1, gp = dual_gate_gemm(x_fg, w_f, w_g)
        announce_once(
            "dual", lambda: f"dual gate GEMM serving: x{tuple(x_fg.shape)} "
            f"w{tuple(w_f.shape)} -> one launch per layer")
        return g1, gp
    if x_fg.shape[0] <= _MAX_DUAL_M and announce_pending("dual-stock"):
        announce_once(
            "dual-stock", lambda: "dual gate GEMM: decode shape not admitted "
            f"(x{tuple(x_fg.shape)} {x_fg.dtype} stride {x_fg.stride()}, "
            f"w{tuple(w_f.shape)} {w_f.dtype}) -> stock two GEMMs", warn=True)
    return None


# ---------------------------------------------------------------------------
# conv + recurrent + gated norm one-pass (pure spec-verify decode)
# ---------------------------------------------------------------------------


@triton.jit
def _conv_taps(x0, x1, x2, x3, w0, w1, w2, w3):
    # Stock _causal_conv1d_update_kernel, KERNEL_WIDTH == 4, no bias:
    #   acc = 0; for j: acc += x_j * w_j; acc = acc / (1 + exp(-acc))
    # x_j are the taps in the conv state's dtype (history) and bf16 (the
    # token), w_j the fp32 weights; every product is fp32 like stock.
    acc = tl.zeros([x0.shape[0]], dtype=tl.float32)
    acc += x0 * w0
    acc += x1 * w1
    acc += x2 * w2
    acc += x3 * w3
    acc = acc / (1 + tl.exp(-acc))
    # The stock chain stores acc into the bf16 qkv buffer (or converts the
    # fp32 conv output back to bf16) and the recurrent kernel reloads it:
    # round exactly there.
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
    cs_ptr,  # conv state (lines, dim, state_len) view; bf16 or fp32
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
    ctr_ptr,  # [2*NH_TOTAL] int32 monotonic arrival counters
    nh_total,
    scale,
    eps,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,  # power of two: last arriver = count hits a multiple
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
    # The stock conv runs iff the request's line at spec slot 0 is not the
    # null block; the stock recurrence iff its resume slot [n, acc-1] is
    # valid. Both are per-request, so all NV programs of a (request, head)
    # agree on every branch below (the counters need that).
    line = tl.load(idx_ptr + i_n * stride_idx_seq).to(tl.int64)
    state_idx = tl.load(idx_ptr + i_n * stride_idx_seq + n_acc - 1).to(tl.int64)
    do_conv = line != 0
    do_rec = state_idx > 0
    if (line == 0) & (state_idx <= 0):
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    q_ch = Q_OFF + i_h * K + o_k
    k_ch = K_OFF + i_h * K + o_k
    v_ch = V_OFF + i_h * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V

    # ---- conv history: the (acc-1)-th window of the state line, 3 taps ----
    # (masked off when the conv is skipped: stock never touches the null line)
    tok_off = n_acc - 1
    cs_line = cs_ptr + line * stride_cs_line
    cs_hist = cs_line + tok_off * stride_cs_tok
    hk = mask_k & do_conv
    hv = mask_v & do_conv
    q0 = tl.load(cs_hist + q_ch * stride_cs_dim, mask=hk, other=0.0)
    q1 = tl.load(cs_hist + q_ch * stride_cs_dim + 1 * stride_cs_tok, mask=hk, other=0.0)
    q2 = tl.load(cs_hist + q_ch * stride_cs_dim + 2 * stride_cs_tok, mask=hk, other=0.0)
    k0 = tl.load(cs_hist + k_ch * stride_cs_dim, mask=hk, other=0.0)
    k1 = tl.load(cs_hist + k_ch * stride_cs_dim + 1 * stride_cs_tok, mask=hk, other=0.0)
    k2 = tl.load(cs_hist + k_ch * stride_cs_dim + 2 * stride_cs_tok, mask=hk, other=0.0)
    v0 = tl.load(cs_hist + v_ch * stride_cs_dim, mask=hv, other=0.0)
    v1 = tl.load(cs_hist + v_ch * stride_cs_dim + 1 * stride_cs_tok, mask=hv, other=0.0)
    v2 = tl.load(cs_hist + v_ch * stride_cs_dim + 2 * stride_cs_tok, mask=hv, other=0.0)
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
    # Every program above has read the old history it needs; the one whose
    # arrival brings the head's count to a multiple of NV writes the rolled
    # window for all 3*128 channels of head i_h (the stock IS_SPEC_DECODING
    # roll: slots [0, VAL) <- old[acc + j], slots [VAL, VAL+T) <- the block's
    # tokens; VAL = STATE_LEN - SEQLEN, the per-request state_len is VAL + T).
    if do_conv:
        tl.debug_barrier()
        arrived = tl.atomic_add(ctr_ptr + i_nh, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ((arrived + 1) & (NV - 1)) == 0:
            VAL: tl.constexpr = STATE_LEN - SEQLEN
            o_vv = tl.arange(0, V)
            vv_ch = V_OFF + i_h * V + o_vv
            for j in tl.static_range(STATE_LEN):
                if j < VAL:
                    src = cs_line + (tok_off + 1 + j) * stride_cs_tok
                    nq = tl.load(src + q_ch * stride_cs_dim)
                    nk = tl.load(src + k_ch * stride_cs_dim)
                    nv = tl.load(src + vv_ch * stride_cs_dim)
                    dst = cs_line + j * stride_cs_tok
                    tl.store(dst + q_ch * stride_cs_dim, nq)
                    tl.store(dst + k_ch * stride_cs_dim, nk)
                    tl.store(dst + vv_ch * stride_cs_dim, nv)
                elif j - VAL < T:
                    # token rows enter the state in its dtype (bf16 -> fp32 is
                    # exact, as stock's x.to(conv_state.dtype))
                    row = x_ptr + (bos + (j - VAL)) * stride_x
                    nq = tl.load(row + q_ch).to(cs_ptr.dtype.element_ty)
                    nk = tl.load(row + k_ch).to(cs_ptr.dtype.element_ty)
                    nv = tl.load(row + vv_ch).to(cs_ptr.dtype.element_ty)
                    dst = cs_line + j * stride_cs_tok
                    tl.store(dst + q_ch * stride_cs_dim, nq)
                    tl.store(dst + k_ch * stride_cs_dim, nk)
                    tl.store(dst + vv_ch * stride_cs_dim, nv)

    if do_rec:
        # ---- recurrence (stock fused_recurrent_gated_delta_rule_fwd_kernel,
        #      IS_KDA + COMPUTE_GATE + SIGMOID_BETA + USE_QK_L2NORM_IN_KERNEL) --
        b_a_log = tl.exp(tl.load(a_log_ptr + i_h).to(tl.float32))
        p_gk = g1_ptr + (bos * H + i_h) * K + o_k
        p_o = o_ptr + (bos * H + i_h) * V + o_v
        mask_h = mask_v[:, None] & mask_k[None, :]
        p_h0 = (
            h0_ptr + state_idx * stride_h0_line + i_h * V * K
            + o_v[:, None] * K + o_k[None, :]
        )
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        # The token loop keeps the stock recurrent kernel's exact shape: every
        # operand loaded inside the iteration, no loop-carried prefetch. A
        # prefetched (or hoisted) [K] operand changes Triton's layout choice
        # for the loop's vectors and with it the l2norm reduction tree -- the
        # recurrent state came out 1 ulp off the stock chain on ~3% of
        # elements (probe, 2026-09-04). Bit-exactness is worth more than the
        # 3 us/layer. The self-test in resolve() guards this every boot.
        for t in range(0, T):
            row = x_ptr + (bos + t) * stride_x
            # the token tap in the conv state's dtype (a no-op for bf16; for
            # an fp32 state the exact widening stock does with
            # x.to(conv_state.dtype)) -- so the sliding taps keep one dtype
            q3 = tl.load(row + q_ch).to(cs_ptr.dtype.element_ty)
            k3 = tl.load(row + k_ch).to(cs_ptr.dtype.element_ty)
            v3 = tl.load(row + v_ch).to(cs_ptr.dtype.element_ty)
            # conv skipped (null conv line, stock leaves the rows raw): the
            # recurrence reads the projection as is
            b_q = tl.where(do_conv, _conv_taps(q0, q1, q2, q3, wq0, wq1, wq2, wq3),
                           q3.to(tl.float32))
            b_k = tl.where(do_conv, _conv_taps(k0, k1, k2, k3, wk0, wk1, wk2, wk3),
                           k3.to(tl.float32))
            b_v = tl.where(do_conv, _conv_taps(v0, v1, v2, v3, wv0, wv1, wv2, wv3),
                           v3.to(tl.float32))
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

        # ---- last arriver of this head applies the gated RMSNorm in place --
        tl.debug_barrier()
        done = tl.atomic_add(ctr_ptr + nh_total + i_nh, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ((done + 1) & (NV - 1)) == 0:
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


_COUNTERS: dict[torch.device, torch.Tensor] = {}


def prepare_counters(device: torch.device) -> torch.Tensor:
    """Allocate the arrival counters once, off-capture (``resolve()`` does it
    on the first eager forward). Every launch slices the same buffer, so the
    address a captured graph records stays valid and no allocation ever
    happens under capture. The counters are monotonic: nothing resets them."""
    buf = _COUNTERS.get(device)
    if buf is None:
        buf = torch.zeros((_COUNTER_SLOTS,), dtype=torch.int32, device=device)
        _COUNTERS[device] = buf
    return buf


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
    num_actual_tokens: int,
    num_spec_decodes: int,
    local_num_heads: int,
    head_dim: int,
    conv_size: int,
    max_query_len: int,
    block_v: int = 8,
) -> bool:
    """Everything the kernel hard-codes, checked on the real tensors. The
    rows the kernel touches are [0, num_actual_tokens) of ``out``/``g1``;
    ``projected`` may carry padding rows beyond them."""
    h, d = local_num_heads, head_dim
    if h <= 0 or d != 128:
        return False
    if conv_size != 4:
        return False
    if conv_bias is not None:
        return False
    if num_spec_decodes <= 0 or num_spec_decodes * h > _MAX_REQ_HEADS:
        return False
    nv = d // block_v
    if block_v * nv != d or (nv & (nv - 1)) != 0:
        return False
    proj = h * d
    n = num_actual_tokens
    if n <= 0 or n > projected.shape[0]:
        return False
    if projected.dim() != 2 or projected.dtype != torch.bfloat16 or projected.stride(1) != 1:
        return False
    if projected.shape[1] < 3 * proj + h:
        return False
    if g1.dtype != torch.bfloat16 or g2.dtype != torch.bfloat16:
        return False
    if not (g1.is_contiguous() and g2.is_contiguous()):
        return False
    if g1.numel() < n * proj or g2.numel() < n * proj:
        return False
    if conv_weight.dim() != 2 or conv_weight.shape != (3 * proj, conv_size):
        return False
    if conv_weight.dtype != torch.float32:
        return False
    # (lines, dim, state_len) view -- the layer hands over the dim-first
    # orientation for either cache layout; the strides carry it
    if conv_state.dim() != 3 or conv_state.dtype not in (torch.bfloat16, torch.float32):
        return False
    state_len = conv_size - 1 + (max_query_len - 1)
    if conv_state.shape[1] != 3 * proj or conv_state.shape[2] < state_len:
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
    if spec_state_indices.shape[0] < num_spec_decodes or spec_state_indices.stride(1) != 1:
        return False
    if spec_state_indices.dtype != torch.int32 or spec_query_start_loc.dtype != torch.int32:
        return False
    if not spec_query_start_loc.is_contiguous() or spec_query_start_loc.numel() < num_spec_decodes + 1:
        return False
    if num_accepted_tokens.dtype != torch.int32 or not num_accepted_tokens.is_contiguous():
        return False
    if num_accepted_tokens.numel() < num_spec_decodes:
        return False
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    if out.numel() < n * proj:
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
    ``conv_state`` the (lines, dim, state_len) view (bf16 or fp32); ``out``
    receives the normalized attention output ([tokens, proj], bf16) -- the
    same buffer the stock chain hands to fused_recurrent_kda(out=) and then
    normalizes in place. States are updated in place exactly like the stock
    kernels (conv line at spec slot 0, recurrent slots per token).
    """
    h, d = local_num_heads, head_dim
    proj = h * d
    n = num_spec_decodes
    if n <= 0:
        return
    nv = d // block_v
    nh_total = n * h
    ctr = prepare_counters(projected.device)
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
        conv_state.stride(1),
        conv_state.stride(2),
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
        STATE_LEN=4 - 1 + (max_query_len - 1),
        LOWER_BOUND=lower_bound,
        num_warps=1,
        num_stages=3,
    )


def spec_onepass(
    layer,
    projected: torch.Tensor,
    g1: torch.Tensor,
    g2_flat: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    out: torch.Tensor,
    *,
    num_actual_tokens: int,
    num_spec_decodes: int,
    lower_bound: float,
) -> bool:
    """Run the one-pass for a pure spec-verify step if the tensors admit it.
    Returns True when the launch happened (``out`` is normalized); False
    means the caller runs the stock chain. Verdicts are announced once."""
    mql = spec_state_indices.size(-1)
    if not onepass_applicable(
        projected, g1, g2_flat, conv_weight, layer.q_conv1d.bias, conv_state,
        recurrent_state, spec_query_start_loc, spec_state_indices,
        num_accepted_tokens, out,
        num_actual_tokens=num_actual_tokens, num_spec_decodes=num_spec_decodes,
        local_num_heads=layer.local_num_heads, head_dim=layer.head_dim,
        conv_size=layer.conv_size, max_query_len=mql,
    ):
        announce_once(
            "onepass-stock", lambda: "one-pass KDA: tensors not admitted "
            f"(projected {tuple(projected.shape)} {projected.dtype}, conv_state "
            f"{tuple(conv_state.shape)} {conv_state.dtype}, recurrent "
            f"{tuple(recurrent_state.shape)} {recurrent_state.dtype}, indices "
            f"{tuple(spec_state_indices.shape)}, tokens {num_actual_tokens}/"
            f"{num_spec_decodes} req) -> stock chain", warn=True)
        return False
    kda_onepass_spec(
        projected, g1, g2_flat, conv_weight, conv_state, recurrent_state,
        spec_query_start_loc[: num_spec_decodes + 1], spec_state_indices,
        num_accepted_tokens, layer.A_log, layer.dt_bias, layer.o_norm.weight, out,
        num_spec_decodes=num_spec_decodes, local_num_heads=layer.local_num_heads,
        head_dim=layer.head_dim, max_query_len=mql, lower_bound=float(lower_bound),
        eps=float(layer.o_norm.eps),
    )
    announce_once(
        "onepass", lambda: f"one-pass KDA serving: {num_spec_decodes} requests x "
        f"{mql} verify tokens, heads {layer.local_num_heads} x {layer.head_dim}, "
        f"conv state {conv_state.dtype} -> conv+recurrent+norm in one launch")
    return True


# ---------------------------------------------------------------------------
# stock reference chain + fixture (shared by the self-test and the probe)
# ---------------------------------------------------------------------------


def make_fixture(n_req, lens, accs, layout, seed, *, lines=None, H=16, D=128,
                 W=4, S=8, conv_dtype=torch.bfloat16, device="cuda") -> dict:
    """Random fleet-shape inputs for one spec-verify step. Request n owns
    state lines [1 + n*S, 1 + (n+1)*S) (line 0 is the null block)."""
    g = torch.Generator(device=device).manual_seed(seed)
    proj = H * D
    XS = 3 * proj + H + 2 * D
    T_all = sum(lens)
    lines = lines if lines is not None else 1 + n_req * S + 1
    x = (torch.randn((T_all, XS), generator=g, device=device) * 0.7).to(torch.bfloat16)
    g1 = torch.randn((T_all, proj), generator=g, device=device).to(torch.bfloat16)
    g2 = torch.randn((T_all, proj), generator=g, device=device).to(torch.bfloat16)
    conv_w = torch.randn((3 * proj, W), generator=g, device=device) * 0.4
    state_len = W - 1 + (S - 1)
    if layout == "DS":
        conv_state = (torch.randn((lines, 3 * proj, state_len), generator=g, device=device) * 0.7).to(conv_dtype)
    else:
        conv_state = (torch.randn((lines, state_len, 3 * proj), generator=g, device=device) * 0.7).to(conv_dtype)
    rec = torch.randn((lines, H, D, D), generator=g, device=device) * 0.05
    idx = torch.zeros((n_req, S), dtype=torch.int32, device=device)
    for n in range(n_req):
        idx[n] = torch.arange(1 + n * S, 1 + (n + 1) * S, dtype=torch.int32)
    acc = torch.tensor(accs, dtype=torch.int32, device=device)
    cu = torch.zeros((n_req + 1,), dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=device), 0)
    a_log = torch.randn((1, 1, H, 1), generator=g, device=device) * 0.5
    dt_bias = torch.randn((proj,), generator=g, device=device) * 0.5
    nw = (torch.rand((D,), generator=g, device=device) + 0.5).to(torch.bfloat16)
    return dict(x=x, g1=g1, g2=g2, conv_w=conv_w, conv_state=conv_state, rec=rec,
                idx=idx, acc=acc, cu=cu, a_log=a_log, dt_bias=dt_bias, nw=nw,
                H=H, D=D, S=S, W=W, n_req=n_req, T_all=T_all, proj=proj)


def dim_first(conv_state: torch.Tensor, proj: int) -> torch.Tensor:
    """The (lines, dim, state_len) view of either cache layout -- what the
    layer's _forward hands to the kernels."""
    return conv_state if conv_state.shape[1] == 3 * proj else conv_state.transpose(-1, -2)


def run_stock_chain(fx: dict, conv_state: torch.Tensor, rec: torch.Tensor,
                    out: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
    """The three stock launches the one-pass replaces, on a fixture. The conv
    writes its output INTO the qkv columns (out=x) like the layer's merged
    projection; pass a pre-cloned ``x`` for timing loops, else a clone is
    made here (outside any timing)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.kda import (
        fused_recurrent_kda,
        rms_norm_gated,
    )
    H, D, S, proj = fx["H"], fx["D"], fx["S"], fx["proj"]
    n = fx["n_req"]
    x = fx["x"].clone() if x is None else x
    qkv = x[:, : 3 * proj]
    beta = x[:, 3 * proj: 3 * proj + H].unsqueeze(0)
    g1 = fx["g1"].reshape(1, -1, H, D)
    cs = dim_first(conv_state, proj)
    qkv = causal_conv1d_update(
        qkv, cs, fx["conv_w"], None, activation="silu",
        conv_state_indices=fx["idx"][:, 0][:n], num_accepted_tokens=fx["acc"],
        query_start_loc=fx["cu"], max_query_len=S,
    )
    q, k, v = qkv.split(proj, dim=-1)

    def _r(t):
        return t.reshape(1, -1, H, D)

    fused_recurrent_kda(
        q=_r(q), k=_r(k), v=_r(v), g=g1, beta=beta, initial_state=rec,
        use_qk_l2norm_in_kernel=True, cu_seqlens=fx["cu"][: n + 1],
        ssm_state_indices=fx["idx"], num_accepted_tokens=fx["acc"],
        out=out.unsqueeze(0), sigmoid_beta=True, a_log=fx["a_log"],
        g_bias=fx["dt_bias"], compute_gate=True, lower_bound=-5.0,
    )
    return rms_norm_gated(out, fx["g2"].reshape(-1, H, D), fx["nw"], None, "sigmoid", eps=1e-5)


def run_onepass(fx: dict, conv_state: torch.Tensor, rec: torch.Tensor,
                out: torch.Tensor, block_v: int = 8) -> torch.Tensor:
    """The one-pass launch on the same fixture (raises if not applicable)."""
    H, D, S = fx["H"], fx["D"], fx["S"]
    n = fx["n_req"]
    cs = dim_first(conv_state, fx["proj"])
    if not onepass_applicable(
        fx["x"], fx["g1"], fx["g2"], fx["conv_w"], None, cs, rec, fx["cu"], fx["idx"],
        fx["acc"], out, num_actual_tokens=fx["T_all"], num_spec_decodes=n,
        local_num_heads=H, head_dim=D, conv_size=fx["W"], max_query_len=S,
        block_v=block_v,
    ):
        raise ValueError("one-pass not applicable on this fixture")
    kda_onepass_spec(
        fx["x"], fx["g1"], fx["g2"], fx["conv_w"], cs, rec, fx["cu"][: n + 1], fx["idx"],
        fx["acc"], fx["a_log"], fx["dt_bias"], fx["nw"], out,
        num_spec_decodes=n, local_num_heads=H, head_dim=D, max_query_len=S,
        lower_bound=-5.0, eps=1e-5, block_v=block_v)
    return out


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.bfloat16:
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    return bool(torch.equal(a, b))


def selftest(device: torch.device) -> tuple[bool, str]:
    """Stock chain vs one-pass on fleet-shape fixtures, bit for bit (conv
    state, recurrent state, output), both cache dtypes, acceptance 1/3/8 and
    a varlen block. ~40 MB of scratch, freed on return. Any mismatch (or an
    error) disarms the knob for the boot."""
    cases = [
        ("C=2 acc=[1,3] bf16", 2, [8, 8], [1, 3], "SD", torch.bfloat16),
        ("C=2 acc=[8,3] fp32 DS", 2, [8, 8], [8, 3], "DS", torch.float32),
        ("C=3 varlen acc=[2,8,1]", 3, [8, 3, 1], [2, 8, 1], "SD", torch.bfloat16),
    ]
    try:
        for name, n, lens, accs, layout, cdt in cases:
            fx = make_fixture(n, lens, accs, layout, 4242, conv_dtype=cdt, device=device)
            cs0, rec0 = fx["conv_state"].clone(), fx["rec"].clone()
            cs1, rec1 = fx["conv_state"].clone(), fx["rec"].clone()
            out0 = torch.zeros((fx["T_all"], fx["H"], fx["D"]), dtype=torch.bfloat16, device=device)
            out1 = torch.zeros_like(out0)
            y0 = run_stock_chain(fx, cs0, rec0, out0)
            y1 = run_onepass(fx, cs1, rec1, out1)
            torch.cuda.synchronize(device)
            if not _bits_equal(cs0, cs1):
                return False, f"{name}: conv state differs from stock"
            if not _bits_equal(rec0, rec1):
                return False, f"{name}: recurrent state differs from stock"
            if not _bits_equal(y0, y1):
                d = (y0.float() - y1.float()).abs().max().item()
                return False, f"{name}: output differs from stock (max |d| {d:.3e})"
        return True, f"{len(cases)} cases bit-exact"
    except Exception as e:  # noqa: BLE001 -- a broken self-test disarms, loudly
        return False, f"self-test raised {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# resolve: knobs -> counters -> self-test, once per process
# ---------------------------------------------------------------------------

_STATE: dict = {"resolved": False, "dual": False, "onepass": False, "detail": ""}


def resolve(device: torch.device | None = None) -> dict:
    """Read the knobs once, allocate the arrival counters off-capture and
    run the self-test; returns the shared state dict (``dual``, ``onepass``).
    Idempotent. Call from the first eager forward, never under capture."""
    st = _STATE
    if st["resolved"]:
        return st
    st["resolved"] = True
    st["dual"] = dual_gemm_enabled()
    want_one = onepass_enabled()
    if not (st["dual"] or want_one):
        return st
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if want_one:
        if torch.cuda.is_current_stream_capturing():
            st["detail"] = "resolve() called under CUDA-graph capture; one-pass disarmed"
            logger.warning("[kda-onepass] %s", st["detail"])
            return st
        prepare_counters(device)
        ok, detail = selftest(device)
        st["onepass"] = ok
        st["detail"] = detail
        if ok:
            logger.info("[kda-onepass] self-test PASS (%s) -> one-pass ARMED", detail)
        else:
            logger.warning("[kda-onepass] self-test FAIL (%s) -> one-pass DISARMED, "
                           "stock chain serves", detail)
    logger.info("[kda-onepass] resolve: dual=%s onepass=%s", st["dual"], st["onepass"])
    return st
