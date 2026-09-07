# SPDX-License-Identifier: Apache-2.0
"""deneb fork: GLM-5.3-Flash decode megakernel driver (glm53_megakernel.cu).

Segments for the GB10 decode step, each replacing a between-collectives
run of launches with one launch (MK_SEG_KDA, the persistent linear-attention
block, was sunset in 34차 §8 -- the stock chain with KDA_ONEPASS covers it):

  MK_SEG_MHC   hc fused post+pre (takes over mhc_fused_post_pre_tilelang's
               small-M branch from glm53_mhc_tilelang/tilelang.py)
  MK_SEG_GEMM  W8A8 skinny GEMM, quant fused (Fp8DenseMethod.apply hook in
               glm53_fp8_dense.py)

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
from contextlib import contextmanager

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


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


MASTER = _flag("VLLM_GLM53_MEGAKERNEL")
ENABLE_MHC = MASTER and _flag("VLLM_GLM53_MK_MHC")
# 37차: layer 0's standalone pre-mix through the same kernel (identity post),
# so a decode step touches deep_gemm nowhere. Rides on the MHC segment.
ENABLE_MHC_PRE = ENABLE_MHC and _flag("VLLM_GLM53_MK_MHC_PRE", "1")
ENABLE_GEMM = MASTER and _flag("VLLM_GLM53_MK_GEMM")
ENABLE_MLA = MASTER and _flag("VLLM_GLM53_MK_MLA")
# Experimental exact-selection pair reuse; the old span-scanning UNION lane
# is not used. Default off until output/state and serving brackets close.
ENABLE_MLA_PREFILL_PAIR = ENABLE_MLA and (
    os.environ.get("VLLM_GLM53_MK_MLA_PREFILL_PAIR") == "1"
)
MLA_PREFILL_GROUP = (
    4 if os.environ.get("VLLM_GLM53_MK_MLA_PREFILL_GROUP") == "4" else 2
)
# MK_SEG_SMLP2: the dense MLP (gate_up -> clamped SwiGLU -> down), T <= 32
# -- the shared expert of every MoE layer and the three dense layers -- as
# two PDL-chained v2 (non-persistent) launches: gate_up with the pair-
# activation epilogue, down on the published fp8 groups -- no grid barrier,
# so it shares SMs with the routed MoE kernel the way the standalone v2 lane
# does (30차 §2/§6). Served-numerics unchanged by construction (same packs,
# same rounding points); the bracket is the gate (32차). (The one-launch
# persistent variant, MK_SMLP, was sunset in 34차 §8: never promoted, and
# the barrier held 48 SMs under the routed MoE kernel.)
ENABLE_SMLP2 = MASTER and _flag("VLLM_GLM53_MK_SMLP2")
# The vocab head on the v2 lane (30차 §13): W4 packs of lm_head, one knob per
# endpoint because the two ends of speculative decoding carry different risk
# (fp8_lm_head.Fp8HeadLogitsProcessor): a coarser DRAFT head only moves
# acceptance, the TARGET head's logits are the served output. Both need the
# GEMM segment (its packs and exact gate) and the v2 lane.
ENABLE_HEAD_DRAFT = MASTER and _flag("VLLM_GLM53_MK_HEAD_DRAFT")
ENABLE_HEAD_TARGET = MASTER and _flag("VLLM_GLM53_MK_HEAD_TARGET")
# MK-GEMM is the W4 arm: e2m1 weights x per-16-group pow2 scale, expanded
# to EXACT e4m3 bytes in-kernel, on EVERY eligible decode linear (the KDA
# in_proj included -- there is no fp8 MK arm to fall back to; the W8 arm
# was removed once W4 beat the stock pair on every shape, so the lane is
# W4 or stock). Arming it changes served numerics: bracket first (README).

# tolerances. The W4 GEMM's by-design (e2m1) error class is 0.02-0.08 rel
# on row blocks; the exact-grid gate below it is 1e-5.
_TOL_GEMM_W4 = 0.15
# What both models pass as `sinkhorn_repeat` (`hc_sinkhorn_iters` in
# DeepSeek-V4-Flash's and GLM-5.3-Flash's configs alike). The sinkhorn is a
# RUNTIME loop -- `for it < sinkhorn_repeat - 1` here, `T.serial(...)` in the
# TileLang pair -- and the first row/col pass places its eps differently from
# the loop's, so a gate that runs 3 iterations does not exercise what serving
# runs 19 of. The self-test below and probes/megakernel_glm53_bench.py's
# --sinkhorn default must be THIS number; test_logic pins them together.
SINKHORN_SERVED = 20

# fp32 port of the TileLang pair, bf16 rounding only. CALIBRATED AT
# sinkhorn_repeat=4 (3 loop iterations) and NOT re-derived at SINKHORN_SERVED:
# MK runs the 4x4 sinkhorn in one lane's registers, TileLang in a separate
# kernel with a different reduction order, so normalisation error accumulates
# per iteration and there are now 6x more of them. The self-test logs the
# measured rel_errs next to the count -- if MHC disarms, read them and check
# `--sinkhorn 4` before concluding the kernels diverge: an accumulation
# artefact and a real divergence look the same in a boolean.
_TOL_MHC = 1e-3


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


# The lane's K contract per launch: KBLK_MAX = 32 blocks of 128 in the
# kernel (A quant tiles, the per-row scale stage in smem). A wider linear
# runs as K-chunks -- build_mk_weight_w4_kchunks / gemm_w4a8 below.
MK_GEMM_KMAX = 4096


def _mk_gemm_eligible(m: int, k: int, n_pad: int) -> bool:
    """MK_SEG_GEMM shape contract (decode M only; prefill stays deepgemm)."""
    return (0 < m <= 32 and k % 128 == 0 and 0 < k <= MK_GEMM_KMAX
            and n_pad % 128 == 0)


def _mk_mhc_eligible(num_tokens: int, hc_mult: int, hidden: int) -> bool:
    return (0 < num_tokens <= 32 and hc_mult == 4 and hidden == 4096
            and hc_mult * hidden == 16384)


# ---------------------------------------------------------------------------
# extension + workspace (built lazily on first ARM, never on import)
# ---------------------------------------------------------------------------
_EXT = None
_WS = None
_ARMED = {"mhc": False, "mhc_pre": False, "gemm": False, "mla": False,
          "smlp2": False}


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
        # the v2 (non-persistent) lane's ring depth; 2..4 keep two blocks/SM
        f"-DMK_NBUF2_DEF={os.environ.get('VLLM_GLM53_MK_NBUF2', '3')}",
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


def rebuild(src_path: str) -> dict:
    """32차 item 5 (dev lab): build the extension from another .cu, swap it
    in and re-run the self-tests. The kernels already baked into captured
    graphs stay until a recapture; every eager call sees the new module."""
    import torch
    global _EXT, _armed_once
    from torch.utils.cpp_extension import load

    with open(src_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()[:8]
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"] + [
        f"-DMK_GRID_DEF={os.environ.get('VLLM_GLM53_MK_GRID', '96')}",
        f"-DMK_MHC_GRID_DEF={os.environ.get('VLLM_GLM53_MK_MHC_GRID', '144')}",
        f"-DMK_NBUF2_DEF={os.environ.get('VLLM_GLM53_MK_NBUF2', '3')}",
    ]
    t0 = time.perf_counter()
    ext = load(name=f"glm53_megakernel_{md5}", sources=[src_path],
               extra_cuda_cflags=flags, build_directory=_build_dir(md5, flags),
               verbose=False)
    _EXT = ext
    for k in _ARMED:
        _ARMED[k] = False
    _armed_once = False
    arm()
    logger.warning("[megakernel] dev-lab rebuild md5=%s in %.0f s armed=%s",
                   md5, time.perf_counter() - t0, dict(_ARMED))
    return {"md5": md5, "build_s": round(time.perf_counter() - t0, 1),
            "armed": dict(_ARMED)}


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
        # and MK_SEG_MLA its own: its grid is 96 blocks where the persistent
        # kernels ran 48, and the ticket barrier only releases correctly
        # when the counter is aligned to THIS launch's grid.
        "barrier_mla": z(8, dt=torch.int32),
        "yp": z(NCHUNK * MAX_TOK * NOUT),
        "rp": z(NCHUNK * MAX_TOK),
        # [NCHUNK][MAX_TOK]: p3 stores one sumsq per (chunk, token) and
        # p4 reduces them in a fixed order -- see the note in mk_mhc_p3.
        "sq": z(NCHUNK * MAX_TOK),
        "pmix": z(MAX_TOK * HC),
        "ol_stash": z(MAX_TOK * HIDDEN, dt=torch.bfloat16),
    }
    return _WS


# ---------------------------------------------------------------------------
# weight quant for MK_SEG_GEMM. Own layout: e4m3 + fp32 pow2
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


class MKPack(tuple):
    """The W4 pack: a tuple (wq4, ws4, wgs, rgs, lr_a, lr_b) whose first
    three fields are the 3-tuple every lane read before 33차, so p[0], p[1],
    float(p[2]) keep working; the three new fields are None when the
    corresponding lever is off.

      wq4   uint8 [n_pad/128, k/128, 128, 64]  e2m1 nibbles, tile-major
      ws4   int8  [n_pad/128, k/128, 128, 8]   per-16-group e4m3 scale bytes
      wgs   float  2^-shift of a PER-TENSOR shift (1.0 under the row shift)
      rgs   fp32 [n_pad] or None: 2^-shift_r of the PER-ROW shift (33차
            lever 3), applied to the output column in the kernel epilogue
      lr_a  bf16 [n_pad, r] or None: low-rank error correction, A side
      lr_b  bf16 [r, k] or None: B side; out += (x @ lr_b.T) @ lr_a.T
    """
    __slots__ = ()

    def __new__(cls, wq4, ws4, wgs, rgs=None, lr_a=None, lr_b=None):
        return tuple.__new__(cls, (wq4, ws4, wgs, rgs, lr_a, lr_b))

    wq4 = property(lambda s: s[0])
    ws4 = property(lambda s: s[1])
    wgs = property(lambda s: s[2])
    rgs = property(lambda s: s[3])
    lr_a = property(lambda s: s[4])
    lr_b = property(lambda s: s[5])
    lr_r = property(lambda s: int(s[4].shape[1]) if s[4] is not None else 0)


# Bump when the pack bytes or the tuple layout change: the pack cache keys
# on it, so a stale cache can never serve an old layout to a new kernel.
MK_PACK_VERSION = 4   # 4: the float64 / damping-ladder GPTQ solve (packs of v3 were the fp32 one)
_LR_MAX = 32   # the kernel's LR_MAX: t scratch is [32 rows][LR_MAX] per stream


def _w4_lever(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _w4_search(g, e0, mids, grid):
    """Best (code, d, scale) per 16-group of g [..., 16] fp32 (already
    shifted): the 2-octave x 8-mantissa e4m3-scale search of the RTN packer
    (it reaches the 72-candidate optimum exactly on the real weights), the
    error taken on what the kernel's expanded byte holds. e0 [..., 1] is the
    covering exponent ceil(log2(amax/6)) of the group, already shifted."""
    import torch

    sign = torch.sign(g)
    best = None
    for j in (0.0, 1.0):
        e = (e0 - j).clamp(-5, 5)
        for kk in range(8):
            sc = (1.0 + kk / 8.0) * torch.exp2(e)
            code = torch.bucketize((g / sc).abs(), mids)
            deq = (grid[code] * sign * sc).to(torch.float8_e4m3fn).float()
            err = (deq - g).pow(2).sum(-1, keepdim=True)
            d = (e * 8.0 + kk).to(torch.int8)
            if best is None:
                best = [err, code, d, sc]
            else:
                take = err < best[0]
                best = [torch.where(take, err, best[0]),
                        torch.where(take, code, best[1]),
                        torch.where(take, d, best[2]),
                        torch.where(take, sc, best[3])]
    return best[1], best[2], best[3]


def _w4_quant_cols(w, sc, mids, grid):
    """Quantize the columns w [R, c] with the fixed per-row scale sc [R, 1]:
    (code uint8 [R, c] without the sign bit, deq fp32 [R, c] = the kernel's
    expanded byte)."""
    import torch

    code = torch.bucketize((w / sc).abs(), mids)
    deq = (grid[code] * torch.sign(w) * sc).to(torch.float8_e4m3fn).float()
    return code, deq


def _w4_row_shift(weight, n_pad: int, kg: int, per_row: bool):
    """(need [n_pad, kg] covering exponents, shift [n_pad] int per row,
    clamped fraction). Per-tensor: one median shift on every row (wgs);
    per-row (33차 lever 3): each row centred on its own median, so a row
    living 2^6 from the tensor median no longer clamps its groups at the
    e4m3 exponent floor/ceiling -- the 1.5% of clamped groups on the
    production [6416, 4096] in_proj were exactly those rows."""
    import torch

    n, k = weight.shape
    need = torch.empty(n_pad, kg, dtype=torch.float32, device=weight.device)
    CH = max(128, (((1 << 20) // k) // 128) * 128)
    for r0 in range(0, n, CH):
        r1 = min(r0 + CH, n)
        a = weight[r0:r1].float().view(r1 - r0, kg, 16).abs().amax(-1)
        need[r0:r1] = torch.ceil(torch.log2((a / 6.0).clamp(min=1e-30)))
    need[n:] = 0.0
    live = need[:n]
    if per_row:
        shift = torch.zeros(n_pad, dtype=torch.float32, device=weight.device)
        shift[:n] = -torch.median(live, dim=1).values
    else:
        s = float(-torch.median(live).item()) if n else 0.0
        shift = torch.full((n_pad,), s, dtype=torch.float32,
                           device=weight.device)
    e_sh = live + shift[:n, None]
    clamped = float(((e_sh < -5) | (e_sh > 5)).float().mean()) if n else 0.0
    return need, shift, clamped


def _w4_gptq_codes(weight, shift, need, H, mids, grid, blocksize=128,
                   percdamp=0.01):
    """GPTQ (OBQ error feedback, Frantar et al. 2022) on the e2m1 x e4m3
    grid: columns are quantized in order; each column's rounding error is
    fed forward into the not-yet-quantized columns through the inverse
    Hessian of the layer's INPUT (H = sum x x^T over calibration tokens),
    so the rounding decisions minimise the OUTPUT error x @ (W - Q)^T, not
    the weight error. Group scales are re-derived on the error-updated
    weights when the group starts (groups never cross a block: 16 | 128).
    Same bytes, same kernel: the accuracy is bought at pack time.

    Returns (codes uint8 [n_pad, k] with the sign in bit 3, d int8 [n_pad,
    kg]) in the SHIFTED domain (weights x 2^shift_r), like the RTN path."""
    import torch

    n, k = weight.shape
    n_pad = shift.shape[0]
    kg = k // 16
    dev = weight.device
    W = torch.zeros(n_pad, k, dtype=torch.float32, device=dev)
    W[:n] = weight.float() * torch.exp2(shift[:n, None])
    # The real Hessians (33K tokens, outlier channels 20x, fp32 addmm) are
    # indefinite by rounding at the 1e-7 level: the first GPTQ boot lost 9+
    # linears to "leading minor ... not positive-definite" at 1% damping.
    # Symmetrise, factor in float64, and raise the damping until it holds.
    H = H.to(dev, torch.float64)
    H = 0.5 * (H + H.T)
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    mean_diag = float(torch.mean(torch.diag(H)))
    Hinv = None
    for damp_f in (percdamp, 10 * percdamp, 100 * percdamp, 1.0):
        try:
            Hd = H + torch.eye(k, device=dev, dtype=torch.float64) * (damp_f * mean_diag)
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
            del Hd, L
            if damp_f != percdamp:
                logger.warning("[megakernel] w4 pack GPTQ: Hessian held at damping %.0f%% "
                               "(not at %.0f%%)", 100 * damp_f, 100 * percdamp)
            break
        except Exception:
            Hinv = None
    if Hinv is None:
        raise RuntimeError("Hessian not positive-definite at any damping")
    del H
    codes = torch.zeros(n_pad, k, dtype=torch.uint8, device=dev)
    d_out = torch.zeros(n_pad, kg, dtype=torch.int8, device=dev)
    sc = None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % 16 == 0:
                g16 = W1[:, i:i + 16]
                amax = g16.abs().amax(-1, keepdim=True)
                e0 = torch.ceil(torch.log2((amax / 6.0).clamp(min=1e-30)))
                # a dead / all-zero group: any scale, keep the byte 0
                e0 = torch.where(amax > 0, e0, torch.zeros_like(e0))
                _c, d, sc = _w4_search(g16, e0, mids, grid)
                d_out[:, col // 16] = d.squeeze(-1)
            w = W1[:, i:i + 1]
            code, q = _w4_quant_cols(w, sc, mids, grid)
            sgn = torch.signbit(w).to(torch.uint8) << 3
            codes[:, col] = (code.to(torch.uint8) | sgn).squeeze(-1)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    del W, Hinv
    return codes, d_out


def _w4_rtn_codes(weight, shift, need, mids, grid):
    """Round-to-nearest packer (the 24차..32차 path): per chunk of rows,
    the e4m3-scale search per 16-group on the shifted weights."""
    import torch

    n, k = weight.shape
    n_pad = shift.shape[0]
    kg = k // 16
    dev = weight.device
    q_out = torch.zeros(n_pad, kg, 16, dtype=torch.uint8, device=dev)
    d_out = torch.zeros(n_pad, kg, dtype=torch.int8, device=dev)
    # ~1M elements per row chunk: the search makes a dozen temporaries per
    # candidate per chunk, and the GB10 allocator answered a whole-tensor
    # search by mapping new pages (14.5 GiB reserved on one [6416, 4096]
    # pack, 26차) -- a quarter chunk is a quarter of every temporary.
    CH = max(128, (((1 << 20) // k) // 128) * 128)
    for r0 in range(0, n_pad, CH):
        r1 = min(r0 + CH, n_pad)
        g = torch.zeros(r1 - r0, kg, 16, dtype=torch.float32, device=dev)
        if r0 < n:
            src = weight[r0:min(r1, n)].float()
            g[:src.shape[0]] = src.view(src.shape[0], kg, 16)
        g *= torch.exp2(shift[r0:r1])[:, None, None]
        sign = torch.signbit(g).to(torch.uint8) << 3
        e0 = need[r0:r1].unsqueeze(-1) + shift[r0:r1, None, None]
        code, d, _sc = _w4_search(g, e0, mids, grid)
        q_out[r0:r1] = code.to(torch.uint8) | sign
        d_out[r0:r1] = d.squeeze(-1)
        del g, sign, e0, code, d
    return q_out.view(n_pad, k), d_out


def _w4_lorc(weight, deq_unshifted, H, rank: int):
    """Activation-aware low-rank correction of the quantization error
    (LQER / ZeroQuant-V2 LoRC): E = W - deq; with S = rms of each input
    channel over the calibration tokens (sqrt(diag(H) / ntok)), the SVD of
    E S keeps the directions the layer's real inputs excite, and
    A = U_r S_r, B = V_r^T S^-1 give x @ B^T @ A^T ~ x @ E^T. Without H the
    SVD is plain (white-noise error: r/(sqrt(n)+sqrt(k))^2 of the energy,
    i.e. little). Returns (lr_a bf16 [n_pad, r], lr_b bf16 [r, k],
    captured fraction of the weighted error energy)."""
    import torch

    n, k = weight.shape
    n_pad = deq_unshifted.shape[0]
    E = weight.float() - deq_unshifted[:n]
    if H is not None:
        S = torch.sqrt(torch.diag(H).float().to(E.device).clamp(min=0.0)
                       / max(1.0, float(H.shape[0])))
        S = S.clamp(min=1e-6 * float(S.max()) if float(S.max()) > 0 else 1e-6)
    else:
        S = torch.ones(k, dtype=torch.float32, device=E.device)
    ES = E * S[None, :]
    U, s, V = torch.svd_lowrank(ES, q=min(rank + 8, min(n, k)), niter=4)
    U, s, V = U[:, :rank], s[:rank], V[:, :rank]
    lr_a = torch.zeros(n_pad, rank, dtype=torch.bfloat16, device=E.device)
    lr_a[:n] = (U * s[None, :]).to(torch.bfloat16)
    # ROW-MAJOR [r, k], explicitly: V.T keeps the SVD's transposed strides
    # (a [r, k] tensor laid out column-major) whenever n < k, and the kernel
    # walks lr_b as jj * k + kk -- the first low-rank diagnostic read every
    # n < k shape 2% wrong and every n > k shape right on exactly that
    lr_b = (V.T / S[None, :]).to(torch.bfloat16).contiguous()
    lr_a = lr_a.contiguous()
    tot = float(ES.pow(2).sum())
    cap = float(s.pow(2).sum()) / tot if tot > 0 else 0.0
    return lr_a, lr_b, cap


_CALIB_DIR_DEFAULT = "/cache/mkcalib"
_PACK_CACHE_DEFAULT = "/cache/mkpacks"


def _mk_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


_CALIB_OVERRIDE = None   # probes: (H, ntok) handed straight to the packer


def _calib_hessian_for(name, k=None):
    """The calibration Hessian dumped for this linear on this rank, or None
    (no calibration boot yet, or the knob is off). Read on the pack path
    only; VLLM_GLM53_MK_PACK_GPTQ=0 keeps the packer RTN whatever is on disk.
    `name` is "<ModelClass>/<module name>" (the fp8-dense attach namespaces
    it: the drafter's `model.layers.N.mlp.down_proj` is not the target's --
    on the first calibration boot the drafter read the target's dumps by the
    bare name, 33차 chain 24). A Hessian whose size is not this linear's k is
    ignored, loudly."""
    if not name or _w4_lever("VLLM_GLM53_MK_PACK_GPTQ", "1") != "1":
        return None
    if _CALIB_OVERRIDE is not None:
        return _CALIB_OVERRIDE
    d = os.path.join(_w4_lever("VLLM_GLM53_MK_CALIB_DIR", _CALIB_DIR_DEFAULT),
                     f"rank{_mk_rank()}")
    p = os.path.join(d, name + ".pt")
    if not os.path.exists(p):
        return None
    try:
        import torch
        # A warm pack cache needs only the Hessian's shape/existence to
        # select its GPTQ key. Map the storage so those hits do not first
        # read every k-by-k Hessian into host memory. A cache miss still
        # consumes the same values in _w4_gptq_codes. Older torch releases
        # and legacy (non-ZIP) dumps retain the ordinary load path.
        try:
            blob = torch.load(p, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError) as exc:
            if "mmap" not in str(exc):
                raise
            blob = torch.load(p, map_location="cpu")
        H = blob["H"]
        if k is not None and tuple(H.shape) != (k, k):
            logger.warning("[megakernel] calib: %s is %s, this linear's k is %d "
                           "-> RTN", p, tuple(H.shape), k)
            return None
        return H, int(blob["ntok"])
    except Exception as e:
        logger.warning("[megakernel] calib: %s unreadable (%r) -> RTN", p, e)
        return None


def _weight_md5(weight) -> str:
    import torch

    w = weight.detach().contiguous()
    b = w.view(torch.uint8) if w.dtype != torch.uint8 else w
    # NumPy exposes this contiguous uint8 storage through the buffer
    # protocol. Hash those same bytes directly: tobytes() otherwise makes
    # another full host copy (50 MiB for a KDA in-projection) on every hit.
    return hashlib.md5(b.cpu().numpy()).hexdigest()


def _pack_cache_path(weight, per_row: bool, gptq: bool, rank: int):
    root = _w4_lever("VLLM_GLM53_MK_PACK_CACHE", _PACK_CACHE_DEFAULT)
    if root in ("", "0", "off"):
        return None
    # the bytes AND the shape: two weights of the same seed and element
    # count are byte-identical at different shapes (the bench's [2048 x
    # 4096] and [4096 x 2048] collided on the first run of this cache)
    key = (f"{_weight_md5(weight)}-{weight.shape[0]}x{weight.shape[1]}-"
           f"{str(weight.dtype).split('.')[-1]}-v{MK_PACK_VERSION}-"
           f"{'row' if per_row else 'ten'}-{'gptq' if gptq else 'rtn'}-"
           f"lr{rank}")
    return os.path.join(root, f"rank{_mk_rank()}", key + ".pt")


def build_mk_weight_w4(weight, name=None, per_row=None):
    """The MKPack of a bf16 [n, k] weight -- see MKPack for the fields.

    Nibble layout: low nibble = even element, high = odd. Round-to-nearest
    on the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} x scale.

    The scale is an e4m3 scale -- (1 + k/8) * 2^e, k the 3-bit mantissa --
    stored as the single byte d = (e << 3) + k, the same one byte a pure
    2^e exponent took. The kernel's expansion adds d to a table byte per
    lane, which reproduces e4m3(magnitude * scale) EXACTLY: adding the
    scale's mantissa field to an e4m3 byte is the multiply, because the
    e2m1 magnitudes carry a 1-bit mantissa and the carry into the exponent
    field lands where it should. The only correction is +1 on the three
    codes whose magnitude mantissa is 1.5 (codes 3, 5, 7) when 1 <= k <= 5,
    and that correction does not depend on k -- so it is a second constant
    table, not a select over eight (verified exhaustively over all
    (code, k, e): the byte formula matches e4m3(mag * scale) 504/504).

    Why an e4m3 scale and not the pow2 one this started with: on the real
    checkpoint it is 29-32% less weight quantization error (q/o_proj,
    gate/down_proj, layer 1), for +16 SASS instructions per 4 groups in the
    expansion. A pow2 scale wastes up to 2x of the e2m1 range per group and
    the e2m1 grid is too coarse to absorb that. The scale is picked to
    MINIMISE the group's error, not merely to cover its amax: with the
    cover rule a finer scale grid measured WORSE (it maps the group into
    the sparse top of the e2m1 grid), which is why the two changes only
    pay together.

    33차 (operator: 'accuracy up, speed untouched'), three pack-time levers
    on the same bytes and the same kernel:
      lever 3  VLLM_GLM53_MK_PACK_ROWSHIFT=1 (default): the shift that keeps
               the group exponents inside the byte-add window is per ROW,
               undone on the output column (MKPack.rgs) instead of on the
               activation scales.
      lever 2  VLLM_GLM53_MK_PACK_GPTQ=1 (default): when a calibration boot
               (VLLM_GLM53_MK_CALIB=1) has dumped this linear's input
               Hessian for this rank, the rounding is GPTQ's error feedback
               instead of round-to-nearest.
      lever 4  VLLM_GLM53_MK_PACK_LORC=r (default 0): a rank-r correction of
               the remaining error, activation-aware when the Hessian is
               there; the v2 kernel adds it in the epilogue.
    Packs are cached under VLLM_GLM53_MK_PACK_CACHE (weight md5 + levers +
    MK_PACK_VERSION in the key), so the GPTQ solve is paid once per rank.
    `name` is the linear's module name: the Hessian and the boot log are
    keyed on it; None = RTN, no cache lookup by name (the md5 still keys
    the cache). `per_row` overrides the ROWSHIFT lever: the drafter's
    opaque custom op hands the kernel only the 3-field pack (wq4, ws4,
    wgs), so its packs must carry the shift as wgs -- a per-row pack there
    served every drafter output mis-scaled and the acceptance rate went
    to 0.0% (chain 25, 33차)."""
    import torch

    n, k = weight.shape
    if k % 128 != 0:
        raise ValueError(f"K={k} not a multiple of 128")
    n_pad = _mk_pad128(n)
    kg = k // 16
    per_row = (_w4_lever("VLLM_GLM53_MK_PACK_ROWSHIFT", "1") == "1"
               if per_row is None else bool(per_row))
    calib = _calib_hessian_for(name, k)
    try:
        lr_rank = int(_w4_lever("VLLM_GLM53_MK_PACK_LORC", "0"))
    except ValueError:
        lr_rank = 0
    lr_rank = max(0, min(_LR_MAX, lr_rank))
    if lr_rank % 8:
        lr_rank = (lr_rank // 8) * 8   # the kernel walks r in eights
    cache = None
    try:
        cache = _pack_cache_path(weight, per_row, calib is not None, lr_rank)
    except Exception as e:
        logger.warning("[megakernel] pack cache key failed (%r) -> no cache", e)
    if cache and os.path.exists(cache):
        try:
            blob = torch.load(cache, map_location="cpu")
            if blob.get("version") == MK_PACK_VERSION:
                dev = weight.device
                pk = MKPack(blob["wq4"].to(dev), blob["ws4"].to(dev),
                            float(blob["wgs"]),
                            None if blob["rgs"] is None else blob["rgs"].to(dev),
                            None if blob["lr_a"] is None else blob["lr_a"].to(dev),
                            None if blob["lr_b"] is None else blob["lr_b"].to(dev))
                _PACK_STATS["cached"] += 1
                return pk
        except Exception as e:
            logger.warning("[megakernel] pack cache %s unreadable (%r) -> rebuild",
                           cache, e)
    t0 = time.perf_counter()
    mids = torch.tensor(_E2M1_MIDS, device=weight.device)
    grid = torch.tensor(_E2M1_GRID, device=weight.device)
    need, shift, clamped = _w4_row_shift(weight, n_pad, kg, per_row)
    H = None
    if calib is not None:
        H, ntok = calib
        try:
            q_flat, d_out = _w4_gptq_codes(weight, shift, need, H, mids, grid)
            _PACK_STATS["gptq"] += 1
        except Exception as e:
            # a calibration problem must never cost the linear its pack (the
            # lane would silently serve stock); RTN, and say why
            logger.warning("[megakernel] w4 pack %s %s: GPTQ failed (%r) -> RTN",
                           name, tuple(weight.shape), e)
            _PACK_STATS["gptq_failed"] += 1
            H = None
            torch.cuda.empty_cache()
            q_flat, d_out = _w4_rtn_codes(weight, shift, need, mids, grid)
            _PACK_STATS["rtn"] += 1
    else:
        q_flat, d_out = _w4_rtn_codes(weight, shift, need, mids, grid)
        _PACK_STATS["rtn"] += 1
    if clamped > 0.01:
        # Silent clamping is what made this defect invisible for a campaign:
        # the self-test's weights are O(1) and never reach the floor.
        logger.warning("[megakernel] w4 pack %s: %.1f%% of groups clamp even "
                       "after the %s shift -- the tensor's dynamic range "
                       "exceeds the 11 octaves a byte-add expansion spans",
                       tuple(weight.shape), 100 * clamped,
                       "per-row" if per_row else f"2^{int(shift[0].item())}")
    # Pack PAIRS along k: even k-index -> low nibble, odd -> high. (Found in
    # review: slicing a [.., 8] re-view instead picked elements 0,2,4,6 of
    # each octet AND left numel 4x the target, so reshape raised -- the
    # attach's except then swallowed it into a silently-stock boot, and the
    # self-test exception disarmed every OTHER segment with it. Shape
    # asserts below make any future layout drift fail loudly at build.)
    q_out = q_flat.view(n_pad, kg, 16)
    wq4 = (q_out[..., 0::2] | (q_out[..., 1::2] << 4)).reshape(n_pad, k // 2)
    ws4 = d_out.contiguous()
    assert wq4.shape == (n_pad, k // 2), \
        f"wq4 {tuple(wq4.shape)} != {(n_pad, k // 2)}"
    assert ws4.shape == (n_pad, k // 16), \
        f"ws4 {tuple(ws4.shape)} != {(n_pad, k // 16)}"
    if per_row:
        wgs = 1.0
        rgs = torch.exp2(-shift).contiguous()
    else:
        wgs = float(2.0 ** -float(shift[0].item()))
        rgs = None
    lr_a = lr_b = None
    if lr_rank > 0:
        deq = mk_w4_dequant_rowmajor(wq4, ws4, wgs, rgs)
        lr_a, lr_b, cap = _w4_lorc(weight, deq, H, lr_rank)
        _PACK_STATS["lorc"] += 1
        _PACK_STATS["lorc_cap"] += cap
        del deq
    # Tile-major, like the fp8 pack (#208): one (tile, k-block) record is
    # 128 rows x 64 nibble bytes + 128 x 8 scale bytes, contiguous, so the
    # kernel's stage_raw4 streams it with cp.async as one 8 KB + 1 KB run
    # instead of touching 128 DRAM pages per tile.
    wq4 = (wq4.view(n_pad // 128, 128, k // 128, 64)
           .permute(0, 2, 1, 3).contiguous())
    ws4 = (ws4.view(n_pad // 128, 128, k // 128, 8)
           .permute(0, 2, 1, 3).contiguous())
    pk = MKPack(wq4, ws4, wgs, rgs, lr_a, lr_b)
    _PACK_STATS["build_s"] += time.perf_counter() - t0
    if cache:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = cache + f".tmp{os.getpid()}"
            torch.save({"version": MK_PACK_VERSION, "wq4": wq4.cpu(),
                        "ws4": ws4.cpu(), "wgs": wgs,
                        "rgs": None if rgs is None else rgs.cpu(),
                        "lr_a": None if lr_a is None else lr_a.cpu(),
                        "lr_b": None if lr_b is None else lr_b.cpu(),
                        "name": name, "shape": (n, k)}, tmp)
            os.replace(tmp, cache)
        except Exception as e:
            logger.warning("[megakernel] pack cache write %s failed (%r)", cache, e)
    # Reserved-but-free blocks are host memory this box has lost; give the
    # search's churn back before the next pack. Also matters for the KDA
    # packs built inside the first forward, where a reservation that
    # lingers is subtracted from the KV budget by the memory profiler.
    torch.cuda.empty_cache()
    return pk


_PACK_STATS = {"rtn": 0, "gptq": 0, "gptq_failed": 0, "cached": 0, "lorc": 0,
               "lorc_cap": 0.0, "build_s": 0.0}


def pack_stats_line() -> str:
    """One boot-log line: how the packs of this process were made."""
    s = _PACK_STATS
    cap = (s["lorc_cap"] / s["lorc"]) if s["lorc"] else 0.0
    return (f"packs: rtn={s['rtn']} gptq={s['gptq']} gptq_failed={s['gptq_failed']} cached={s['cached']} "
            f"lorc={s['lorc']} (mean captured error energy {cap:.2f}) "
            f"build {s['build_s']:.0f}s; levers rowshift="
            f"{_w4_lever('VLLM_GLM53_MK_PACK_ROWSHIFT', '1')} gptq="
            f"{_w4_lever('VLLM_GLM53_MK_PACK_GPTQ', '1')} lorc="
            f"{_w4_lever('VLLM_GLM53_MK_PACK_LORC', '0')}")


def mk_w4_dequant_rowmajor(wq4_rm, ws4_rm, wgs=1.0, rgs=None):
    """fp32 [n_pad, k] from the ROW-MAJOR (pre-tile) nibbles [n_pad, k/2]
    and scale bytes [n_pad, k/16]: the kernel's expanded e4m3 byte times
    the shift's undo (wgs, or rgs per row)."""
    import torch

    n_pad, k2 = wq4_rm.shape
    k = k2 * 2
    lo = wq4_rm & 0xF
    hi = wq4_rm >> 4
    q = torch.stack([lo, hi], dim=-1).reshape(n_pad, k)
    grid = torch.tensor(_E2M1_GRID, device=wq4_rm.device)
    mag = grid[(q & 7).long()]
    sign = torch.where((q & 8) != 0, -1.0, 1.0)
    d = ws4_rm.to(torch.int32)
    scale = (1.0 + (d & 7).float() / 8.0) * torch.exp2((d >> 3).float())
    w = mag * sign * scale.repeat_interleave(16, dim=1)
    w = w.to(torch.float8_e4m3fn).float() * wgs
    if rgs is not None:
        w = w * rgs.float()[:, None]
    return w


def mk_w4_dequant(wq4, ws4, n_rows, gscale=1.0, rgs=None):
    """fp32 [n_rows, k] the kernel's expansion reads: nibble -> e2m1 grid
    value x the group's e4m3 scale, rounded into e4m3 the way the kernel's
    table byte is, times the shift's undo (gscale, or rgs per row). Zero
    stays zero; every other code round-trips the grid bit-exactly, which is
    what the exact gate and the by-design gate both need."""
    import torch

    tn, tk, _, _ = wq4.shape
    n_pad, k = tn * 128, tk * 128
    # tile-major [n/128][k/128][128][64] -> row-major [n_pad, k/2]
    wq4_rm = wq4.permute(0, 2, 1, 3).reshape(n_pad, k // 2)
    ws4_rm = ws4.permute(0, 2, 1, 3).reshape(n_pad, k // 16)
    w = mk_w4_dequant_rowmajor(wq4_rm, ws4_rm, gscale, rgs)
    return w[:n_rows]


def mk_pack_dequant(pack, n_rows):
    """fp32 [n_rows, k]: what the lane SERVES for this pack, low-rank
    correction included (W_q + A B)."""
    w = mk_w4_dequant(pack[0], pack[1], n_rows, pack[2],
                      pack[3] if len(pack) > 3 else None)
    if len(pack) > 5 and pack[4] is not None:
        w = w + (pack[4][:n_rows].float() @ pack[5].float())
    return w


def mk_pack_twin(x, pack, n_rows):
    """fp32 [m, n_rows]: what the lane serves for x and this pack -- the
    kernel-quantized activations against the dequantized W4 bytes (row
    scales applied), plus the low-rank correction (33차 lever 4) on the
    UNQUANTIZED x: the reducer blocks read x itself, so the served product
    is x_q W_q^T + x B^T A^T. mk_pack_dequant is the WEIGHT the lane serves
    (W_q + A B), for weight-space error statistics; this is the output."""
    xq = _mk_quant_x_ref(x)
    out = xq @ mk_w4_dequant(pack[0], pack[1], n_rows, pack[2],
                             pack[3] if len(pack) > 3 else None).float().T
    if len(pack) > 5 and pack[4] is not None:
        t = x.float() @ pack[5].float().T                    # [m, r]
        out = out + t @ pack[4][:n_rows].float().T
    return out


def _mk_quant_x_ref(x):
    """fp32 [m, k]: x after the kernel's activation quant (per row, per
    128-k group: the EXACT scale amax/448 (33차 lever 1; it was the pow2
    2^frexp_exp(amax/448) before, which wasted up to one bit of e4m3's
    three), e4m3 round-to-nearest at v * (1/scale), rescale). Pure twin of
    the prologue: the division, the reciprocal (__frcp_rn) and the product
    are the same IEEE fp32 operations in both, so with mk_w4_dequant it
    makes a torch fp32 matmul the kernel's exact reference (no fp8 MK arm
    exists to diff against any more)."""
    import torch

    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    amax = g.abs().amax(-1, keepdim=True)
    # amax * fp32(1/448): the kernel's form, and torch's own for a scalar
    # divisor; the floor is the kernel's fmaxf
    scale = (amax * (1.0 / 448.0)).clamp(min=1e-30)
    rsc = 1.0 / scale
    # the kernel converts with SATFINITE: a product one ulp over 448 (the
    # row's own amax times a rounded reciprocal) saturates instead of NaN
    q = (g * rsc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return q.view(m, k)


# ---------------------------------------------------------------------------
# Calibration: the input Hessians the GPTQ packer needs (33차 lever 2).
# VLLM_GLM53_MK_CALIB=1 makes every W4 launch's Python entry accumulate
# H += x^T x for its pack until VLLM_GLM53_MK_CALIB_TOKENS rows have been
# seen, then each rank dumps <calib dir>/rank<r>/<name>.pt and stops. It is
# the eager entry that counts, so run the calibration traffic BEFORE the
# decode graphs are captured or with enough prefill (prefill never replays).
# ---------------------------------------------------------------------------
_CALIB = {"on": _flag("VLLM_GLM53_MK_CALIB"), "H": {}, "ntok": {},
          "meta": {}, "budget": 0, "seen": 0, "dumped": False}
try:
    _CALIB["budget"] = int(os.environ.get("VLLM_GLM53_MK_CALIB_TOKENS", "32768"))
except ValueError:
    _CALIB["budget"] = 32768
_PACK_META = {}   # wq4 data pointer -> name (set by the fp8-dense attach)


def _pack_key(pack) -> int:
    """The pack's identity for calibration: its nibble tensor's address.
    The drafter's opaque custom op rebuilds the pack tuple on every call
    (lists of tensors in, a fresh tuple out), so id(pack) never matched and
    the drafter's 35 packs got no Hessians on the first calibration boot
    (chain 24); the tensor behind the tuple is the same object."""
    return int(pack[0].data_ptr())


def note_pack_name(pack, name: str) -> None:
    if pack is None or not name:
        return
    if isinstance(pack, list):
        for c, p in enumerate(pack):
            _PACK_META[_pack_key(p)] = f"{name}.k{c}"
    else:
        _PACK_META[_pack_key(pack)] = name


def _calib_observe(x, pack) -> None:
    """Accumulate this launch's input into its pack's Hessian."""
    import torch

    if _CALIB["dumped"] or x.dim() != 2 or pack[0] is None:
        return
    # the decode graph capture runs this entry eagerly with capture active:
    # the finite-row mask below syncs the device (a boolean index / .all()
    # read), which is "operation not permitted when stream is capturing"
    # (CALIB2 boot, 33차 chain 25). Captured steps are replays of dummy
    # rows anyway -- nothing to observe.
    if torch.cuda.is_current_stream_capturing():
        return
    key = _pack_key(pack)
    name = _PACK_META.get(key)
    if name is None:
        return
    x32 = x.detach().float()
    # the batch's padded rows (the token count rounded up to the graph's
    # size) hold whatever memory was there: every k >= 2048 Hessian of the
    # first calibration boot was NaN through them (chain 24) -- keep the
    # finite rows only, and count only those
    finite = torch.isfinite(x32).all(dim=1)
    if not bool(finite.all()):
        x32 = x32[finite]
        if x32.shape[0] == 0:
            return
    H = _CALIB["H"].get(key)
    if H is None:
        H = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32,
                        device=x.device)
        _CALIB["H"][key] = H
        _CALIB["ntok"][key] = 0
        _CALIB["meta"][key] = (name, (int(pack[0].shape[0]) * 128, int(x.shape[1])))
    H.addmm_(x32.T, x32)
    _CALIB["ntok"][key] += int(x32.shape[0])
    # every served linear sees every token, so any pack's count is the
    # budget's progress; the max keeps a late-attached pack from stalling it
    _CALIB["seen"] = max(_CALIB["seen"], _CALIB["ntok"][key])
    if _CALIB["seen"] >= _CALIB["budget"]:
        _calib_dump()


def _calib_dump() -> None:
    import torch

    _CALIB["dumped"] = True
    root = os.path.join(_w4_lever("VLLM_GLM53_MK_CALIB_DIR", _CALIB_DIR_DEFAULT),
                        f"rank{_mk_rank()}")
    os.makedirs(root, exist_ok=True)
    n = 0
    for key, H in _CALIB["H"].items():
        name, shape = _CALIB["meta"][key]
        if not bool(torch.isfinite(H).all()):
            logger.warning("[megakernel] calib: %s Hessian is not finite -> not dumped", name)
            continue
        path = os.path.join(root, name + ".pt")   # "<Model>/<linear>.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"H": H.cpu(), "ntok": _CALIB["ntok"][key], "shape": shape,
                    "name": name}, path)
        n += 1
    logger.warning("[megakernel] calib: dumped %d Hessians to %s (%d tokens "
                   "seen); the next boot's packer uses them (GPTQ)", n, root,
                   _CALIB["seen"])
    _CALIB["H"].clear()
    torch.cuda.empty_cache()


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
def build_mk_weight_w4_kchunks(weight, name=None, per_row=None):
    """[(wq4, ws4, gscale), ...]: one W4 pack per MK_GEMM_KMAX-wide column
    chunk of a [n, k] weight whose k exceeds the lane's per-launch K.

    The DFlash2 drafter's fc is [4096, 5 x 4096]: the aux hidden states of
    five target layers, concatenated. In the armed 09-03 trace it is the
    single largest GEMM of the decode tail -- 168 MB of bf16 read once per
    step, 792 us at 212 GB/s -- and the lane refused it at the eligibility
    test, so it stayed the one dense projection of the drafter that no
    quantized arm could take. Chunking the weight along k instead of
    widening the kernel keeps KBLK_MAX and the smem budget of the 173
    launches/step that already serve at the W4 stream floor untouched; the
    price is len(chunks) launches plus an fp32 sum of the partials
    (gemm_w4a8), and the probe (probes/drafter_fc_check.py) times exactly
    that.
    """
    n, k = weight.shape
    if k % 128 != 0:
        raise ValueError(f"K={k} not a multiple of 128")
    return [build_mk_weight_w4(weight[:, c:c + MK_GEMM_KMAX].contiguous(),
                               name=None if name is None else f"{name}.k{c // MK_GEMM_KMAX}",
                               per_row=per_row)
            for c in range(0, k, MK_GEMM_KMAX)]


_AR_NOTE = None


def _rgs_ptr(pack) -> int:
    """The pack's per-row output scale pointer for a launch (0 = none)."""
    rgs = pack[3] if len(pack) > 3 else None
    return 0 if rgs is None else int(rgs.data_ptr())


def _ar_note(*tensors) -> None:
    """Tell the one-shot AR shim which weights this launch streams.

    The shim (tp_oneshot_ar) attributes the note to the collective that
    preceded this launch and, on the next capture, has that collective's
    kernel warm these bytes into L2 while it waits for the peers
    (VLLM_GLM53_AR_PREFETCH). Resolved once; a lane without the shim
    mounted, or with the knob off, pays one attribute read per launch.
    Only the eager Python of a launch runs this -- graph replay never does.
    """
    global _AR_NOTE
    if _AR_NOTE is None:
        try:
            from vllm.distributed.device_communicators import (
                dsv4_oneshot_shim as _shim)
            _AR_NOTE = _shim.note_consumer
        except Exception:
            _AR_NOTE = False
    if _AR_NOTE:
        _AR_NOTE(tensors)


def _gemm_call(x, mk_pack, n_rows, bg=False):
    """mk_pack is (wq4, ws4, gscale) from build_mk_weight_w4. `bg`: the
    caller marks the launch background -- the shared expert's pair, which
    serving forks onto the aux stream beside the routed MoE -- which the
    v2 lane's LoRC slot keys on (the barrier-free local-quant kernel that
    once keyed on it was sunset in 34차 §8)."""
    import torch

    out = torch.empty(x.shape[0], n_rows, dtype=torch.bfloat16,
                      device=x.device)
    _ar_note(mk_pack[0], mk_pack[1])
    rgs = mk_pack[3] if len(mk_pack) > 3 else None
    lr_a = mk_pack[4] if len(mk_pack) > 4 else None
    lr_b = mk_pack[5] if len(mk_pack) > 5 else None
    if lr_a is not None and not (lr_a.is_contiguous() and lr_b.is_contiguous()
                                 and rgs is None or rgs.is_contiguous()):
        raise ValueError("low-rank factors / row scales must be contiguous "
                         "(the kernel walks them row-major)")
    _EXT.run_gemm(x.contiguous(), mk_pack[0], mk_pack[1], out, n_rows,
                  float(mk_pack[2]), 1 if bg else 0,
                  0 if rgs is None else rgs.data_ptr(),
                  0 if lr_a is None else lr_a.data_ptr(),
                  0 if lr_b is None else lr_b.data_ptr(),
                  0 if lr_a is None else int(lr_a.shape[1]))
    return out


_GEMM_CAPTURED = {"said": False}


def gemm_w4a8(x, mk_pack, n_rows, bg=False):
    """Fp8DenseMethod.apply hook: None = not armed/eligible (stock runs).

    The lane is W4 or stock: a self-test failure disarms MK-GEMM for
    every linear (stock deepgemm), there is no fp8 MK pack to fall back
    to by construction."""
    if not _ARMED["gemm"] or x.dim() != 2 or mk_pack is None:
        return None
    # the served decode is a graph replay: say once, at capture, which lane
    # variant the graph bakes (the .cu reads the same env once: 32차 §13)
    if not _GEMM_CAPTURED["said"]:
        try:
            import torch
            if torch.cuda.is_current_stream_capturing():
                _GEMM_CAPTURED["said"] = True
                # the extension's plan (ksr, units, blocks/SM), not the env
                plan = _EXT.gemm2_plan(int(x.shape[0]), int(n_rows), int(x.shape[1]))
                logger.warning("[megakernel] gemm lane CAPTURED into the decode graph: "
                               "M=%d plan ksr=%d units=%d bps=%d",
                               int(x.shape[0]), int(plan[0]), int(plan[1]), int(plan[2]))
        except Exception:
            _GEMM_CAPTURED["said"] = True
    if _CALIB["on"]:
        if isinstance(mk_pack, list):
            for c, p in enumerate(mk_pack):
                _calib_observe(x[:, c * MK_GEMM_KMAX:(c + 1) * MK_GEMM_KMAX], p)
        else:
            _calib_observe(x, mk_pack)
    if isinstance(mk_pack, list):
        return _gemm_kchunks(x, mk_pack, n_rows, bg)
    if mk_pack[0] is None:
        return None
    # the pack is tile-major [n_pad/128, k/128, 128, 64]: the padded n is
    # 128 x its first dim (shape[0] alone is the tile count -- passing it
    # as n_pad failed the n_pad % 128 test on every real shape and the
    # lane silently stayed stock)
    if not _mk_gemm_eligible(x.shape[0], x.shape[1],
                             mk_pack[0].shape[0] * 128):
        return None
    return _gemm_call(x, mk_pack, n_rows, bg)


def _gemm_kchunks(x, packs, n_rows, bg=False):
    """K-chunked lane (build_mk_weight_w4_kchunks): one launch per
    MK_GEMM_KMAX columns of x, partials summed in fp32, bf16 out. None when
    any chunk fails the per-launch contract (stock runs the whole linear)."""
    import torch

    m, k = x.shape
    if not packs or any(p[0] is None for p in packs):
        return None
    if len(packs) != -(-k // MK_GEMM_KMAX):
        return None
    for c, p in enumerate(packs):
        kc = min(MK_GEMM_KMAX, k - c * MK_GEMM_KMAX)
        if not _mk_gemm_eligible(m, kc, p[0].shape[0] * 128):
            return None
    acc = None
    for c, p in enumerate(packs):
        # _gemm_call takes the slice contiguous: m x 4096 bf16, a copy the
        # size of one tile row -- the weight stream is the cost here
        out = _gemm_call(x[:, c * MK_GEMM_KMAX:(c + 1) * MK_GEMM_KMAX], p,
                         n_rows, bg)
        acc = out.float() if acc is None else acc.add_(out.float())
    return acc.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# The vocabulary head on the v2 lane (30차 §13). fp8_lm_head's
# Fp8HeadLogitsProcessor._apply_head asks head_logits() first; None means
# "not this launch" and its fp8 / bf16 path runs unchanged. The fp8 head is
# 158 MB/rank read at ~190 GB/s (836 us, twice a step: target verify + draft
# candidates), already at the DRAM floor for fp8 bytes; the W4 pack halves
# the bytes. One knob per endpoint: VLLM_GLM53_MK_HEAD_DRAFT / _TARGET.
# ---------------------------------------------------------------------------
_HEAD = {"said": set(), "disarmed": {}}   # endpoint -> why it was disarmed


def _head_disarm(endpoint: str, why: str) -> None:
    if endpoint not in _HEAD["disarmed"]:
        _HEAD["disarmed"][endpoint] = why
        logger.warning("[megakernel] head lane %s DISARMED for this boot: %s "
                       "-> the fp8/bf16 head serves", endpoint, why)


def head_pack(lm_head, endpoint: str):
    """The W4 pack of lm_head.weight ([vocab/TP, hidden] bf16), built once
    per head object and kept on it: the target and the drafter may share one
    head object, so the pack is shared storage while the endpoint gates stay
    independent. None (latched, like the fp8 copy's build) when the build
    failed or the weight is outside the lane's K contract -- never retried
    on the hot path."""
    pack = getattr(lm_head, "_mk_head_pack", None)
    if pack is not None:
        return pack
    if getattr(lm_head, "_mk_head_pack_attempted", False):
        return None
    lm_head._mk_head_pack_attempted = True
    weight = getattr(lm_head, "weight", None)
    try:
        import torch

        if (weight is None or weight.dim() != 2
                or weight.dtype not in (torch.bfloat16, torch.float16)):
            raise ValueError("head weight is not a 2-D bf16/fp16 shard: %s"
                             % (None if weight is None
                                else (tuple(weight.shape), weight.dtype)))
        n, k = weight.shape
        if k % 128 != 0 or k > MK_GEMM_KMAX:
            raise ValueError(f"head K={k} is outside the lane's contract "
                             f"(a multiple of 128, <= {MK_GEMM_KMAX})")
        # `name` keys the calibration Hessian (33차 lever 2) and the boot
        # log; the pack cache keys on the weight's md5 regardless
        pack = build_mk_weight_w4(weight, name="lm_head")
        note_pack_name(pack, "lm_head")
        lm_head._mk_head_pack = pack
        logger.warning("[megakernel] head pack built (%s endpoint first): weight "
                       "%s -> %d tiles (n_pad %d)", endpoint, tuple(weight.shape),
                       int(pack[0].shape[0]), int(pack[0].shape[0]) * 128)
        return pack
    except Exception as e:
        logger.warning("[megakernel] head pack build FAILED (%s, weight %s): %r",
                       endpoint, None if weight is None else tuple(weight.shape), e)
        return None


def _head_first_call_gate(x, pack, n, out) -> float:
    """The lane's output on this x against the pack's dequantized twin on
    two tile groups -- the first eight tiles and the last eight (the padded
    tail, where n_orig masks the stores) -- so the check reads 16 MB of
    dequantized weight, not the head's 634 MB. Raises on a miss; returns the
    worst rel error."""
    tiles = int(pack[0].shape[0])
    rgs = pack[3] if len(pack) > 3 else None
    lr_a = pack[4] if len(pack) > 4 else None
    lr_b = pack[5] if len(pack) > 5 else None
    groups = [(0, min(8, tiles))]
    if tiles > 8:
        groups.append((max(tiles - 8, 8), tiles))
    worst = 0.0
    for t0, t1 in groups:
        c0, c1 = t0 * 128, min(t1 * 128, n)
        rows = c1 - c0
        sub = MKPack(pack[0][t0:t1], pack[1][t0:t1], pack[2],
                     None if rgs is None else rgs[t0 * 128:t1 * 128],
                     None if lr_a is None else lr_a[t0 * 128:t1 * 128], lr_b)
        ref = mk_pack_twin(x, sub, rows)
        e, n_ulp = _exact_gate(out[:, c0:c1], ref)
        if not e <= 1e-3 or n_ulp > 0:
            raise RuntimeError(f"head exact gate on tiles [{t0}, {t1}): "
                               f"rel={e:.2e} over-ulp={n_ulp}")
        worst = max(worst, e)
    return worst


def head_logits(x, lm_head, endpoint: str):
    """bf16 [m, vocab/TP] from the W4 pack on the v2 lane, or None (the
    caller's fp8/bf16 path). Serving needs the GEMM segment armed (its boot
    exact gate passed), the v2 lane (the persistent kernel's static
    distribution was never sized for 303 tiles), a decode-sized batch
    (m <= 32, the lane's contract) and this endpoint's first-call gate: the
    first eligible call -- the eager warm-up, before capture -- is checked
    against the pack's twin; a miss disarms the endpoint for the boot,
    loudly. Then the serving proof line (armed is not serving)."""
    if not _ARMED["gemm"] or x.dim() != 2 or x.shape[0] > 32:
        return None
    if endpoint in _HEAD["disarmed"]:
        return None
    pack = head_pack(lm_head, endpoint)
    if pack is None:
        _head_disarm(endpoint, "no W4 pack for the head")
        return None
    n = int(lm_head.weight.shape[0])
    out = gemm_w4a8(x, pack, n)
    if out is None:
        return None            # this batch is outside the lane's contract
    if endpoint not in _HEAD["said"]:
        _HEAD["said"].add(endpoint)
        import torch

        capturing = torch.cuda.is_current_stream_capturing()
        worst = float("nan")
        if not capturing:
            try:
                worst = _head_first_call_gate(x, pack, n, out)
            except Exception as e:
                _head_disarm(endpoint, f"first-call exact gate: {e!r}")
                return None
        plan = list(_EXT.gemm2_plan(int(x.shape[0]), n, int(x.shape[1])))
        logger.warning("[megakernel] head lane serving: %s endpoint, first eligible "
                       "call m=%d n=%d k=%d plan(ksr/units/bps)=%s, exact "
                       "gate worst rel=%.1e %s", endpoint, int(x.shape[0]), n,
                       int(x.shape[1]), plan, worst,
                       "SKIPPED (capturing)" if capturing else "PASS")
    return out


# ---------------------------------------------------------------------------
# MK_SEG_SMLP2
# ---------------------------------------------------------------------------
SMLP_GU_MAX = 8192  # gate_up scratch width: the dense MLP's 2 x 3072 is the widest per rank

_SMLP2_GU = None


def _smlp2_call(x, gu_pack, d_pack, n_gu, n_int, n_out, limit, alpha=1.0,
                beta=0.0):
    """Two PDL-chained v2 launches: gate_up into a static bf16 scratch
    (its pair epilogue emits the fp8 A groups), then down on those groups
    (a_ready). Same packs, same activation, same rounding points as the
    stock three-launch chain; no grid barrier and no 48-block residency."""
    import torch

    global _SMLP2_GU
    if _SMLP2_GU is None or _SMLP2_GU.device != x.device:
        # one static scratch, strongly held (a captured graph bakes it)
        _SMLP2_GU = torch.empty(MAX_TOK, SMLP_GU_MAX, dtype=torch.bfloat16,
                                device=x.device)
    out = torch.empty(x.shape[0], n_out, dtype=torch.bfloat16, device=x.device)
    _ar_note(gu_pack[0], gu_pack[1])
    _EXT.run_smlp2(
        [x.data_ptr(), gu_pack[0].data_ptr(), gu_pack[1].data_ptr(),
         d_pack[0].data_ptr(), d_pack[1].data_ptr(), _SMLP2_GU.data_ptr(),
         out.data_ptr(), _rgs_ptr(gu_pack), _rgs_ptr(d_pack)],
        [float(gu_pack[2]), float(d_pack[2]), float(limit), float(alpha),
         float(beta)],
        [int(x.shape[0]), int(x.shape[1]), int(n_gu), int(n_int), int(n_out),
         int(gu_pack[0].shape[0]), int(gu_pack[0].shape[1]),
         int(d_pack[0].shape[0]), int(d_pack[0].shape[1])],
    )
    return out


def _smlp_gate(got, ref32):
    """(ok, rel, over_ulp): the exact gate's numbers under a tolerance that
    survives the one place the chain is not bit-reproducible -- the fp32
    sigmoid (expf in the kernel, torch's exp in the twin) differs by an ulp
    or two before the bf16 round, and a flipped bf16 bit there moves one
    fp8 input. A layout / expansion / hand-off bug is 1e-1 and up."""
    e, n_ulp = _exact_gate(got, ref32)
    return (e <= 2e-3 and n_ulp <= max(8, got.numel() // 256)), e, n_ulp


def _smlp_packs(mlp):
    """(gu_pack, d_pack) when both linears carry a single MK W4 pack."""
    packs = []
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        m = getattr(lin, "quant_method", None)
        while m is not None and not hasattr(m, "_mk") and hasattr(m, "_base"):
            m = m._base
        pk = getattr(m, "_mk", None)
        if pk is None or isinstance(pk, list) or pk[0] is None:
            return None
        packs.append(pk)
    return packs[0], packs[1]


_SMLP_SAID = set()
_SMLP_FUSED_CALLS = 0


def _smlp_stock(reason):
    """None for the hook, and the reason once per distinct text: an armed
    segment that the served forward never reaches is the 28차 pattern
    ("armed" read as "serving"); the boot log must be able to tell."""
    if reason not in _SMLP_SAID:
        _SMLP_SAID.add(reason)
        logger.warning("[megakernel] smlp lane stock: %s", reason)
    return None


def smlp_forward(mlp, x):
    """Glm5NextMLP.forward hook: the fused launch, or None (stock runs).

    Eligible when the segment is armed, x is a 2-D bf16 [T <= 32, k] row
    batch, both linears carry single W4 packs (k <= 4096 each) and the
    gate_up width is 2 x the down input. The activation's clamp comes from
    the module (SiluAndMulWithClamp: swiglu_limit, alpha, beta; SiluAndMul
    is limit 0 / alpha 1 / beta 0). No reduction here: the caller keeps
    its own reduce_results contract."""
    import torch

    global _SMLP_FUSED_CALLS
    if not _ARMED["smlp2"]:
        return None
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        return _smlp_stock("x is %s %s, not a 2-D bf16 row batch"
                           % (tuple(x.shape), x.dtype))
    T, k = x.shape
    if T < 1 or T > MAX_TOK:
        return None   # prefill rows: routine, not a reason to say
    if k % 128 != 0 or k > MK_GEMM_KMAX:
        return _smlp_stock("k=%d is not a multiple of 128 <= %d" % (k, MK_GEMM_KMAX))
    packs = _smlp_packs(mlp)
    if packs is None:
        return _smlp_stock("gate_up/down carry no single MK W4 pack")
    gu_pack, d_pack = packs
    n_gu = int(mlp.gate_up_proj.output_size_per_partition)
    n_int = int(mlp.down_proj.input_size_per_partition)
    n_out = int(mlp.down_proj.output_size)
    if (n_gu != 2 * n_int or n_int % 128 != 0 or n_int > MK_GEMM_KMAX
            or n_gu > SMLP_GU_MAX
            or gu_pack[0].shape[1] != k // 128
            or d_pack[0].shape[1] != n_int // 128):
        return _smlp_stock("geometry n_gu=%d n_int=%d k=%d packs %s/%s"
                           % (n_gu, n_int, k, tuple(gu_pack[0].shape),
                              tuple(d_pack[0].shape)))
    act = getattr(mlp, "act_fn", None)
    limit = float(getattr(act, "swiglu_limit", 0.0) or 0.0)
    alpha = float(getattr(act, "alpha", 1.0))
    beta = float(getattr(act, "beta", 0.0))
    _SMLP_FUSED_CALLS += 1
    capturing = False
    try:
        capturing = bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        pass
    if _SMLP_FUSED_CALLS == 1:
        logger.warning("[megakernel] smlp lane serving: first fused call (smlp2) "
                       "T=%d k=%d n_int=%d n_out=%d limit=%.1f capturing=%s",
                       T, k, n_int, n_out, limit, capturing)
    # the served decode is a graph replay: the capture-time call is the
    # proof that replays run the fused block (29차 KDA lesson)
    if capturing and "captured" not in _SMLP_SAID:
        _SMLP_SAID.add("captured")
        logger.warning("[megakernel] smlp lane CAPTURED into the decode graph (smlp2): "
                       "T=%d n_int=%d", T, n_int)
    return _smlp2_call(x.contiguous(), gu_pack, d_pack, n_gu, n_int, n_out,
                       limit, alpha, beta)


def _smlp_ref(x, gu_pack, d_pack, n_gu, n_int, n_out, limit, alpha=1.0,
              beta=0.0):
    """Pure-torch twin of the fused launch: fp32 matmuls of the kernel-
    quantized activations against the dequantized packs, bf16 at the same
    rounding points."""
    import torch

    gu = (_mk_quant_x_ref(x) @ mk_pack_dequant(gu_pack, n_gu).float().T
          ).to(torch.bfloat16)
    g = gu[:, :n_int].float()
    u = gu[:, n_int:].float()
    if limit > 0:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    a = (g * torch.sigmoid(alpha * g) * (u + beta)).to(torch.bfloat16)
    return _mk_quant_x_ref(a) @ mk_pack_dequant(d_pack, n_out).float().T


def _selftest_smlp2() -> bool:
    """MK_SEG_SMLP2's gate: an exact fixture (e2m1-grid packs at the shared
    expert's and the dense MLP's geometry) through the two-launch v2 chain,
    with replay stability."""
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    grid = torch.tensor(_E2M1_GRID, device=dev)

    def exact_weight(n, k):
        code = torch.randint(0, 8, (n, k // 16, 16), device=dev)
        sexp = torch.randint(-12, -2, (n, k // 16, 1), device=dev)
        w = (grid[code] * torch.exp2(sexp.float())) * torch.where(
            torch.randn_like(code.float()) < 0, -1.0, 1.0)
        return w.view(n, k).to(torch.bfloat16)

    limit = 10.0
    worst = 0.0
    for T, n_int, k, n_out in ((8, 512, HIDDEN, HIDDEN), (16, 3072, HIDDEN, HIDDEN),
                               (32, 512, HIDDEN, HIDDEN)):
        n_gu = 2 * n_int
        gu_pack = build_mk_weight_w4(exact_weight(n_gu, k))
        d_pack = build_mk_weight_w4(exact_weight(n_out, n_int))
        x = torch.randn(T, k, dtype=torch.bfloat16, device=dev)
        got = _smlp2_call(x, gu_pack, d_pack, n_gu, n_int, n_out, limit)
        ref = _smlp_ref(x, gu_pack, d_pack, n_gu, n_int, n_out, limit)
        torch.cuda.synchronize()
        ok, e_exact, n_ulp = _smlp_gate(got, ref)
        worst = max(worst, e_exact)
        if not ok:
            logger.warning("[megakernel] selftest smlp2 grid rel=%.2e over-ulp=%d "
                           "(T=%d n_int=%d) -> DISARM", e_exact, n_ulp, T, n_int)
            return False
        again = _smlp2_call(x, gu_pack, d_pack, n_gu, n_int, n_out, limit)
        torch.cuda.synchronize()
        if _rel_err(got, again) > 1e-6:
            logger.warning("[megakernel] selftest smlp2 replay drift -> DISARM")
            return False
    logger.warning("[megakernel] selftest smlp2 exact=%.2e -> ARM", worst)
    return True


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
    _ar_note(fn)
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


_HOOK_SERVED = [0]   # MHC calls mhc_hook actually served (probe receipt)


def mhc_hook(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
             hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
             hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps):
    """The whole call-site contract in ONE entry point: arm, then try.

    Every image's MHC wrapper is a separate full-file fork -- GLM's
    `tilelang.py` and dsv4's `mhc_tilelang.py` cannot share a file. What they
    CAN share is this module, which both profiles mount. Keeping the
    arm-then-call pair here means a wiring is five lines that cannot drift
    from the other lane's five lines, instead of two copies of the same
    twenty (the T <= 16 window correction had to be applied twice before this
    existed).

    Returns None for every miss -- unarmed, ineligible shape, wrong dtype --
    so the caller falls through to its stock path. It does NOT catch: an
    armed launch that fails is an async CUDA failure and a python fallback
    cannot contain it (the w4a8 lesson).

    `_HOOK_SERVED` counts the calls this actually took. It is the receipt a
    probe needs to tell "MK is faster" from "MK was never offered the call"
    (the wrapper only offers T <= 16), and it lives here so that reading it
    does not mean patching a private of this module from outside. The
    increment is on the armed path only, next to a kernel launch.
    """
    maybe_arm()
    out = mhc_fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn,
                             hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value,
                             sinkhorn_repeat, norm_weight, norm_eps)
    if out is not None:
        _HOOK_SERVED[0] += 1
    return out


# ---------------------------------------------------------------------------
# MK_SEG_MHC, pre-only (37차): layer 0's standalone pre-mix
# ---------------------------------------------------------------------------
# The model's first layer has no incoming post state and calls hc_pre alone;
# every other layer's pre rides in hc_fused_post_pre, which MK_SEG_MHC serves
# at T <= 16. That one standalone call was the only place a decode step still
# reached deep_gemm's tf32 prenorm GEMM -- and at k=5 (T=6) it is where chain
# 13's boot spun. The fused kernel's post step is v = pm[j]*x + sum_k cm[k][j]*
# res[k] in fp32, so pm = 0 and cm = I give residual_out == residual bitwise
# and the pre part runs on the untouched residual: the standalone pre, from
# the kernel already armed, with no kernel change. The coefficient buffers
# are three static tensors at T_max (x zeros 256 KB, pm zeros, cm identity
# rows), sliced per call, allocated once at the self-test (never inside a
# graph capture).
_HOOK_SERVED_PRE = [0]
_PRE_LOGGED = [False]
_PRE_BUF = {}
MHC_PRE_TMAX = 32


def _pre_bufs(device):
    """(x zeros [TMAX, HIDDEN] bf16, pm zeros [TMAX, HC] fp32, cm identity
    [TMAX, HC*HC] fp32) for the device, made once."""
    import torch
    key = str(device)
    bufs = _PRE_BUF.get(key)
    if bufs is None:
        x0 = torch.zeros(MHC_PRE_TMAX, HIDDEN, dtype=torch.bfloat16, device=device)
        pm0 = torch.zeros(MHC_PRE_TMAX, HC, dtype=torch.float32, device=device)
        cm_i = torch.eye(HC, dtype=torch.float32, device=device).reshape(1, HC * HC)
        cm_i = cm_i.repeat(MHC_PRE_TMAX, 1).contiguous()
        bufs = _PRE_BUF[key] = (x0, pm0, cm_i)
    return bufs


def mhc_pre_only(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                 hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                 norm_weight, norm_eps):
    """The standalone pre-mix (mhc_pre_tilelang's contract, with norm) from
    the fused kernel under identity post coefficients. None = stock."""
    if not _ARMED["mhc_pre"]:
        return None
    import torch
    if residual.dim() != 3 or norm_weight is None:
        return None
    num_tokens, hc_mult, hidden = residual.shape
    if num_tokens > MHC_PRE_TMAX or not _mk_mhc_eligible(num_tokens, hc_mult, hidden):
        return None
    if (residual.dtype != torch.bfloat16 or fn.dtype != torch.float32
            or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32
            or norm_weight.dtype != torch.bfloat16):
        return None
    x0, pm0, cm_i = _pre_bufs(residual.device)
    _rc, pm, cm, li = _mhc_call(
        x0[:num_tokens], residual.reshape(-1, hc_mult, hidden),
        pm0[:num_tokens], cm_i[:num_tokens], fn, hc_scale, hc_base,
        norm_weight, num_tokens, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, norm_eps, sinkhorn_repeat)
    return pm, cm, li


def mhc_pre_hook(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                 hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                 norm_weight, norm_eps):
    """Arm, then try -- mhc_hook's contract for the standalone pre. Returns
    (post_mix [T, HC], comb_mix [T, HC*HC], layer_input [T, HIDDEN]) or None
    for every miss. Logs once when it first serves a real call: the receipt
    that separates "armed" from "serving" (the 28차 lesson)."""
    maybe_arm()
    out = mhc_pre_only(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                       norm_weight, norm_eps)
    if out is not None:
        _HOOK_SERVED_PRE[0] += 1
        if not _PRE_LOGGED[0]:
            _PRE_LOGGED[0] = True
            logger.warning("[megakernel] mhc-pre hook serving (T=%d): layer 0's "
                           "standalone pre-mix runs in the MHC segment; no "
                           "deep_gemm in the decode step", residual.shape[0])
    return out


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

    d = float((a.float() - b.float()).norm())
    den = float(b.float().norm())
    if not math.isfinite(d) or not math.isfinite(den):
        return math.inf
    error = float(d / den) if den > 0 else float(d)
    # Every caller must fail closed: max(0.0, NaN) is 0.0 and NaN > tol
    # is false. Neither an invalid output nor an invalid oracle can arm a
    # segment, including replay/aggregate gates that use those idioms.
    return error if math.isfinite(error) else math.inf


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
    post_mult, sinkhorn_repeat = 1.0, SINKHORN_SERVED

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
    logger.warning("[megakernel] selftest mhc sinkhorn=%d rel_errs=%s -> %s",
                   sinkhorn_repeat,
                   ["%.2e" % e for e in errs], "ARM" if ok else "DISARM")
    return ok


# The exact e2m1 fixture of the boot gate AND the bench's probe_exact -- one
# builder, so the two cannot drift apart again (the bench once kept an older
# scale range and FAILed on the fixture, not the kernel: 25차).
def _selftest_mhc_pre() -> bool:
    """Diff the pre-only path (fused kernel, identity post) against the stock
    standalone pre at T=8 and, when the boot's spec k is not 7, at T=k+1 --
    the shape a k=5 boot serves and a k=7 production boot never runs (a T it
    never serves must not be able to hang its arm). Two checks: the post step
    under pm=0, cm=I leaves the residual bitwise unchanged (the identity the
    trick rests on), and pm/cm/layer_input match the stock pair."""
    import torch
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tlk
    torch.manual_seed(1)
    dev = "cuda"
    spec_k = (os.environ.get("VLLM_GLM53_SPEC_K") or "7").strip()
    ts = [8]
    if spec_k.isdigit() and int(spec_k) != 7 and 0 < int(spec_k) + 1 <= MHC_PRE_TMAX:
        ts.append(int(spec_k) + 1)
    x0, pm0, cm_i = _pre_bufs(dev)
    fn = torch.randn(NOUT, HC * HIDDEN, dtype=torch.float32, device=dev) * 0.02
    hc_scale = torch.ones(3, dtype=torch.float32, device=dev)
    hc_base = torch.zeros(NOUT, dtype=torch.float32, device=dev)
    nw = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    rms_eps = pre_eps = sink_eps = norm_eps = 1e-6
    post_mult, sinkhorn_repeat = 1.0, SINKHORN_SERVED
    worst, identity = 0.0, True
    for T in ts:
        res = torch.randn(T, HC, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1
        rc, pmc, cmc, li = _mhc_call(
            x0[:T], res, pm0[:T], cm_i[:T], fn, hc_scale, hc_base, nw, T,
            rms_eps, pre_eps, sink_eps, post_mult, norm_eps, sinkhorn_repeat)
        n_splits = 4
        yp = torch.empty(n_splits, T, NOUT, dtype=torch.float32, device=dev)
        rp = torch.empty(n_splits, T, dtype=torch.float32, device=dev)
        res_ref = torch.empty_like(res)
        tlk.mhc_fused_tilelang(cm_i[:T].view(T, HC, HC), res, pm0[:T], x0[:T],
                               fn.view(NOUT, HC, HIDDEN), yp, rp, res_ref, HC,
                               HIDDEN, NOUT, tile_n=2, n_splits=n_splits)
        pm_ref = torch.empty(T, HC, dtype=torch.float32, device=dev)
        cm_ref = torch.empty(T, HC * HC, dtype=torch.float32, device=dev)
        li_ref = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
        tlk.mhc_pre_big_fuse_with_norm_tilelang(
            yp, rp, hc_scale, hc_base, res_ref, pm_ref, cm_ref, li_ref, nw,
            HIDDEN, rms_eps, pre_eps, sink_eps, post_mult, sinkhorn_repeat,
            norm_eps, n_splits=n_splits, hc_mult=HC)
        torch.cuda.synchronize()
        identity = identity and bool(torch.equal(rc, res)) and bool(torch.equal(res_ref, res))
        worst = max(worst, _rel_err(pmc, pm_ref), _rel_err(cmc, cm_ref),
                    _rel_err(li, li_ref))
    ok = identity and worst <= _TOL_MHC
    logger.warning("[megakernel] selftest mhc-pre T=%s identity=%s rel=%.2e -> %s",
                   ts, identity, worst, "ARM" if ok else "DISARM")
    return ok


EXACT_FIXTURE = (1024, 4096, 8)  # n, k, m: 8 tiles -> 32 units, both kernels


def exact_fixture(dev="cuda", shape=None):
    """(x, pack, w_exact, ref): weights ON the e2m1 x 2^s grid at PRODUCTION
    magnitudes (sexp in [-12, -2): the group scale of real dense projections
    is ~2^-7 median; O(1) weights never reached the expansion's floor and hid
    the clamp of 24차), the kernel-quantized activations against the
    dequantized pack as the fp32 reference."""
    import torch

    torch.manual_seed(0)
    n, k, m = EXACT_FIXTURE if shape is None else shape
    code = torch.randint(0, 8, (n, k // 16, 16), device=dev)
    sexp = torch.randint(-12, -2, (n, k // 16, 1), device=dev)
    grid = torch.tensor(_E2M1_GRID, device=dev)
    w_exact = (grid[code] * torch.exp2(sexp.float())) * torch.where(
        torch.randn_like(code.float()) < 0, -1.0, 1.0)
    w_exact = w_exact.view(n, k).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    pack = build_mk_weight_w4(w_exact)
    ref = mk_pack_twin(x, pack, n)
    return x, pack, w_exact, ref


@contextmanager
def _gemm_probe_scope():
    """Restore the exact lane/split overrides, also after a failed gate."""
    state = _EXT.probe_state()
    try:
        yield
    finally:
        _EXT.restore_probe_state(state)


def _selftest_gemm_exact() -> float:
    """Each row class on the exact fixture, plus a forced odd split.

    Return the worst finite error; raise on failure so the normal arm gate
    disarms GEMM. Reuse each weight pack across row classes to keep boot
    work bounded. The split override is restored even on failure.
    """
    import torch

    worst = 0.0
    for shape, rows, ksr in (((1024, HIDDEN, 32), (8, 16, 32), 0),
                              ((2048, HIDDEN, 8), (8,), 3)):
        n, k, _ = shape
        x_all, pack, _w, ref_all = exact_fixture(shape=shape)
        for m in rows:
            x, ref = x_all[:m], ref_all[:m]
            with _gemm_probe_scope():
                _EXT.set_gemm2(ksr)
                plan = list(_EXT.gemm2_plan(m, n, k))
                if ksr and plan[0] != ksr:
                    raise RuntimeError(f"exact gate failed to force ksr={ksr}: {plan}")
                got = _gemm_call(x, pack, n)
                again = _gemm_call(x, pack, n)
            torch.cuda.synchronize()
            e, n_ulp = _exact_gate(got, ref)
            if not e <= 1e-3 or n_ulp > 0:
                raise RuntimeError(
                    f"GEMM exact m={m} n={n} k={k}: rel={e:.2e} over-ulp={n_ulp}")
            worst = max(worst, e)
            if not _rel_err(got, again) <= 1e-6:
                raise RuntimeError(f"GEMM replay drift: m={m} n={n}")
            logger.warning("[megakernel] exact gemm m=%d n=%d: plan=%s PASS",
                           m, n, plan)
    return worst


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

    dev = "cuda"
    e_exact = _selftest_gemm_exact()
    for m, n in ((8, KDA_INPROJ_N), (16, HIDDEN), (32, 1024)):
        w = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.05
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        sq, sws, srows, scols = _stock_fp8_pair(w)
        ref = _fp8_dense_gemm(x, sq, sws, srows, scols)
        got = _gemm_call(x, build_mk_weight_w4(w), n)
        torch.cuda.synchronize()
        e = _rel_err(got, ref)
        if not e <= _TOL_GEMM_W4:
            logger.warning("[megakernel] selftest gemm m=%d n=%d by-design "
                           "rel=%.2e -> DISARM", m, n, e)
            return False
    # the served plans in the fingerprint: (ksr, units, blocks/SM) for the
    # shared expert's gate_up and the in_proj shape. A boot log without
    # this line is a boot whose GEMM plan nobody can name from the log.
    try:
        plan_bg = list(_EXT.gemm2_plan(8, 1024, HIDDEN))
        plan_in = list(_EXT.gemm2_plan(8, KDA_INPROJ_N, HIDDEN))
    except Exception:
        plan_bg = plan_in = None
    logger.warning("[megakernel] selftest gemm exact=%.2e (m=8/16/32 + forced "
                   "split) -> ARM (plans ksr/units/bps: [1024x4096]=%s, "
                   "in_proj=%s)", e_exact, plan_bg, plan_in)
    return True


def hc_scale_ones():
    import torch

    return torch.ones(3, dtype=torch.float32, device="cuda")


def hc_base_zeros():
    import torch

    return torch.zeros(NOUT, dtype=torch.float32, device="cuda")


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
        if ENABLE_MHC_PRE and _ARMED["mhc"]:
            _ARMED["mhc_pre"] = _gate("mhc_pre", _selftest_mhc_pre)
    if ENABLE_GEMM:
        _ARMED["gemm"] = _gate("gemm", _selftest_gemm)
    if ENABLE_MLA:
        _ARMED["mla"] = _gate("mla", _selftest_mla)
    if ENABLE_SMLP2:
        _ARMED["smlp2"] = _gate("smlp2", _selftest_smlp2)
    logger.warning("[megakernel] armed=%s", dict(_ARMED))
    if _flag("VLLM_GLM53_DEV_LAB"):
        try:   # 32차 item 5: the boot-free kernel loop (dev boots only)
            from vllm.model_executor.layers import glm53_dev_lab
            glm53_dev_lab.install()
        except Exception:
            logger.exception("[megakernel] dev lab install failed")


_armed_once = False


# ---------------------------------------------------------------------------
# MK_SEG_MLA -- sparse MLA decode (see the kernel comment for the cost model)
# ---------------------------------------------------------------------------
MLA_D = 512          # kv_lora_rank
MLA_H = 16           # MLA heads per rank at TP4
MLA_SPLITS_MAX = 64
_MLA_WS = None


MLA_MAX_SPLIT_ROWS = 64   # beyond this the split path's fp32 scratch is silly
# The split path's scratch is ONE allocation for the life of the process. A
# captured decode graph bakes its address, and the grow-on-demand this
# started with (cap = max(need, 2*MAX_TOK), re-allocated when a larger call
# came) freed the old tensors under every graph captured before that call.
# v5 routed prefill rows through the same entry point, so the first
# request of a boot -- a 37-token prompt, 48 splits, 1,776 rows -- did
# exactly that, and the decode graphs then wrote their partials into
# whatever the allocator handed out next (28차: two serving deaths, a
# gather index out of bounds and an illegal memory access). So the budget
# is fixed here and mla_splits() never asks for more rows than it holds:
# T * splits <= MLA_WS_ROWS for every T it splits.
MLA_WS_ROWS = 3 * MLA_MAX_SPLIT_ROWS   # 192 rows: 6.3 MB of fp32 partials


def _mla_workspace(device, T: int, splits: int):
    """Split partials + (m, l): allocated once at MLA_WS_ROWS, never grown."""
    global _MLA_WS
    import torch

    if _MLA_WS is None:
        _MLA_WS = {
            "cap": MLA_WS_ROWS,
            "part": torch.zeros(MLA_WS_ROWS * MLA_H * MLA_D, dtype=torch.float32, device=device),
            "pml": torch.zeros(MLA_WS_ROWS * MLA_H * 2, dtype=torch.float32, device=device),
        }
    need = T * splits
    if need > _MLA_WS["cap"]:
        raise RuntimeError(
            f"mla: T={T} x splits={splits} = {need} rows exceed the fixed "
            f"workspace of {MLA_WS_ROWS}; mla_splits() must bound T*splits")
    return _MLA_WS


def mla_splits(T: int) -> int:
    """Slot-axis splits for this row count.

    Measured rule (grid 48, W=2048): the best split is the smallest s with
    T*s a multiple of the resident grid -- every block then gets the same
    number of items and the same slot count per item. T=8 -> 6 (94 us),
    16 -> 3 (162), 24 -> 2 (235), 32 -> 3 (330; 1 and 2 leave a third of the
    blocks walking a second item alone and cost 40%).

    Bounded by the fixed scratch: T*s <= MLA_WS_ROWS. The decode shapes
    above are untouched (48/48/48/96 rows); rows that the rule would have
    split 48 ways (T=37: 1,776 rows for 43 slots a split) take the direct
    path instead, which is what they are -- a prefill chunk."""
    if _EXT is None or T <= 0:
        return 1
    if T > MLA_MAX_SPLIT_ROWS:
        # prefill: every row is its own item, the kernel normalises in place
        # and no [T][splits][H][D] fp32 scratch exists (268 MB at T=8192)
        return 1
    grid = int(_EXT.mla_grid())
    budget = max(1, min(MLA_SPLITS_MAX, MLA_WS_ROWS // T))
    forced = os.environ.get("VLLM_GLM53_MK_MLA_SPLITS")   # probe knob, never set in serving
    if forced:
        return max(1, min(budget, int(forced)))
    for s in range(1, budget + 1):
        if (T * s) % grid == 0:
            return s
    return max(1, min(budget, round(grid / T)))


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
    if (ENABLE_MLA_PREFILL_PAIR and 128 <= T <= 8192
            and 1 <= slots.shape[1] <= 2176
            and q_nope.dtype == torch.bfloat16 and ckv.is_contiguous()
            and ckv.element_size() == 1 and lens.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        return _mla_prefill_pair(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out)
    ws = _ensure_workspace(q_nope.device)
    splits = mla_splits(T)
    assert splits == 1 or T * splits <= MLA_WS_ROWS, (T, splits)
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


def _mla_prefill_pair(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out=None):
    """Two or four adjacent rows reuse FP8 loads with independent membership.

    The schedule holds only slot IDs and one membership bit per row, never gathered
    KV. Its size is bounded by T*W, independent of the physical cache span.
    Per-call storage is deliberately eager-only: it cannot invalidate the
    fixed workspace used by captured decode graphs. Duplicate input slots
    remain duplicate schedule entries, with each row's multiplicity intact.
    Ordering can differ from the indexer, so this is not a bit-exact claim.
    """
    import torch

    T, _, _ = q_nope.shape
    W = slots.shape[1]
    width = MLA_PREFILL_GROUP
    _fn = _mla_prefill_pair
    if not getattr(_fn, "_announced", False):
        # 39차: the bracket's proof that the optional path ran (P1 had none);
        # self-contained so a def exec'd alone in a stub namespace still runs
        _fn._announced = True
        _lg = globals().get("logger")
        if _lg is not None:
            _lg.warning("[megakernel] mla prefill pair engaged (T=%d, W=%d, group=%d)", T, W, width)
    groups = (T + width - 1) // width
    if os.environ.get("VLLM_GLM53_MLA_PAIR_STATS") == "1" and getattr(_fn, "_stats_calls", 0) < 6:
        # 39차 diagnostic (operator: "제안대로 개선"): how much of the sparse
        # selection adjacent rows really share. The forecast assumed 75 %
        # common selection; the pair kernel's work is the UNION of the
        # group's selections, so union/W is the traffic ratio it can reach.
        _fn._stats_calls = getattr(_fn, "_stats_calls", 0) + 1
        try:
            with torch.no_grad():
                valid = torch.arange(W, device=slots.device)[None, :] < lens[:, None]
                s = torch.where(valid, slots.long(), torch.full_like(slots.long(), -1))
                rows = (T // width) * width
                g = s[:rows].view(-1, width, W)                       # [groups, width, W]
                a = g[:, 0]; b = g[:, 1]
                inter = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).any(-1).sum(-1).float()
                la = (a >= 0).sum(-1).float(); lb = (b >= 0).sum(-1).float()
                union = la + lb - inter
                jac = (inter / union.clamp_min(1)).mean().item()
                # the group's union of all `width` rows vs the sum of their lengths
                flat = g.reshape(g.shape[0], -1)
                srt, _ = flat.sort(-1)
                uniq = ((srt[:, 1:] != srt[:, :-1]) & (srt[:, 1:] >= 0)).sum(-1).float() + (srt[:, :1] >= 0).sum(-1).float()
                lsum = (flat >= 0).sum(-1).float()
                ratio = (uniq / lsum.clamp_min(1)).mean().item()
                _lg = globals().get("logger")
                if _lg is not None:
                    _lg.warning("[megakernel] mla pair stats: T=%d W=%d group=%d mean len=%.0f | adjacent-row jaccard=%.3f "
                                "| group union/sum-of-lengths=%.3f (forecast assumed 0.4375 traffic at 75%% common)",
                                T, W, width, lsum.mean().item() / width, jac, ratio)
        except Exception as e:  # noqa: BLE001 -- a diagnostic must never take the lane down
            _lg = globals().get("logger")
            if _lg is not None:
                _lg.warning("[megakernel] mla pair stats failed: %r", e)
    schedule = torch.empty((groups, width * W), dtype=torch.int32, device=q_nope.device)
    membership = torch.empty_like(schedule)
    lengths = torch.empty(groups, dtype=torch.int32, device=q_nope.device)
    if out is None:
        out = torch.empty_like(q_nope)
    launch = (_EXT.run_mla_prefill_group4 if width == 4
              else _EXT.run_mla_prefill_pair)
    launch(
        [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(),
         out.data_ptr(), schedule.data_ptr(), membership.data_ptr(), lengths.data_ptr()],
        [float(sm_scale), float(ckv_scale)], [int(T), int(W)],
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
    # 40 and 100 rows take the direct path (splits == 1: v5's prefill
    # store), which the decode shapes never exercise.
    cases = [(8, 2048, False), (16, 2048, True), (32, 512, True), (1, 64, False),
             (40, 2048, True), (100, 2048, True)]
    if ENABLE_MLA_PREFILL_PAIR:
        # The optional path begins at T=128. Its boot gate must exercise
        # shared selections, repeated slots, low overlap, odd T and empty
        # rows before the existing MLA segment is allowed to arm.
        cases += [(128, 2048, False), (129, 2176, True), (130, 64, True)]
        if MLA_PREFILL_GROUP == 4:
            cases += [(131, 2176, True)]
    for T, W, ragged in cases:
        num_slots = (max(4096, 4 * W)
                     if ENABLE_MLA_PREFILL_PAIR and MLA_PREFILL_GROUP == 4 and T >= 128
                     else 4096)
        q = torch.randn(T, MLA_H, MLA_D, dtype=torch.bfloat16, device=dev) * 0.3
        cache = (torch.randn(num_slots, MLA_D, device=dev) * 0.5).to(torch.float8_e4m3fn)
        slots = torch.randint(0, num_slots, (T, W), dtype=torch.int32, device=dev)
        if ragged:
            lens = torch.randint(1, W + 1, (T,), dtype=torch.int32, device=dev)
        else:
            lens = torch.full((T,), W, dtype=torch.int32, device=dev)
        if ENABLE_MLA_PREFILL_PAIR and T >= 128:
            slots[1].copy_(slots[0])
            lens[1].copy_(lens[0])
            if MLA_PREFILL_GROUP == 4 and not ragged:
                slots[2].copy_(slots[0])
                slots[3].copy_(slots[0])
                lens[2:4].copy_(lens[0].expand(2))
            if ragged:
                lens[2:4].zero_()
            if MLA_PREFILL_GROUP == 4:
                # Four disjoint full lists force weak-reuse fallback; at
                # W2176 their 8704 unique keys also exceed the hash capacity.
                # The following group covers unequal per-row membership and
                # multiplicity on the shared route instead of only identical
                # lists. These fixtures run only with the explicit group knob.
                col = torch.arange(W, dtype=torch.int32, device=dev)
                common = 3 * W // 4
                for row in range(4):
                    slots[4 + row].copy_(col + row * W)
                    slots[8 + row, :common].copy_(col[:common])
                    slots[8 + row, common:].copy_(col[common:] + row * (W - common))
                lens[4:12].fill_(W)
                slots[9, common].copy_(slots[8, 0])
        sm, ks = MLA_D ** -0.5, 0.7
        got = mla_decode(q, cache.view(torch.uint8), slots, lens, sm, ks)
        ref = mla_decode_ref(q, cache, slots, lens, sm, ks)
        torch.cuda.synchronize()
        error = _rel_err(got.float(), ref.float())
        if not error <= 2e-2:
            logger.warning("[megakernel] selftest mla T=%d W=%d rel=%.2e -> DISARM",
                           T, W, error)
            return False
        worst = max(worst, error)
    logger.warning("[megakernel] selftest mla rel=%.2e pair_prefill=%s group=%d -> ARM",
                   worst, ENABLE_MLA_PREFILL_PAIR, MLA_PREFILL_GROUP)
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
