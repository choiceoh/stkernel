"""SM120/SM121 MoE dispatch layer — workspace, compilation, and launch.

Ported from b12x's integration/tp_moe.py. Supports micro (tiny decode),
static (decode), and dynamic (prefill) backends with token-count-based
selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import (
    convert_sf_from_mma_layout,
    convert_sf_to_mma_layout,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .moe_activation import SWIGLUOAI_UNINTERLEAVE, is_gated_activation
from .moe_direct_micro_kernel import (
    MoEDirectMicroKernel,
    build_direct_micro_kernel,
    compile_direct_micro_kernel,
    compiled_direct_micro_accepts_block_dim,
)
from .moe_dynamic_kernel import (
    _MAX_SHARED_INPUT_TOPK,
    _TASK_SLICE_CHUNK,
    MoEDynamicKernel,
)
from .moe_micro_kernel import MoEMicroKernel
from .moe_static_kernel import MoEStaticKernel
from .moe_static_common import STAMP_SLOTS as _STATIC_V2_STAMP_SLOTS
from .moe_static_kernel_v4 import MoEStaticKernelV4
from .moe_sf_pack import (
    SF_PACK_BLOCK,
    SF_STAGE_BYTES,
    pack_sf_inline,
)
from .moe_reform_sf_pack import (
    REFORM_SF_STAGE, prepare_reform_scales,
)
from .moe_static_kernel_v5 import (
    MoEStaticKernelV5,
    TILED_W13_CHUNKS,
    TILED_W13_K_IN,
    TILED_W2_K_IN,
)
from ._moe_dynamic.gated import MoEGatedDynamicKernel
from .moe_dynamic_gated_tiled import MoEGatedDynamicKernelTiled
from .moe_w4a16_fp4_helpers import swizzle_block_scale
from .moe_w4a16_host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    route_pack_numel_capacity,
    unswizzle_block_scale,
    validate_activation,
)
from .moe_w4a16_kernel import run_w4a16_moe
from .moe_w4a16_prepare import (
    W4A16PackedWeights,
    _normalize_source_format,
    prepare_w4a16_packed_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NVFP4_BLOCK_SIZE = 16
_MXFP4_BLOCK_SIZE = 32
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
# Must equal the kernel's task materialization granularity or the task
# queue is mis-sized.
_DYNAMIC_SLICE_CHUNK = _TASK_SLICE_CHUNK
# Probe hook: pin the dynamic tile_m (16/32/64/128) for a shape cell under measurement
# (Qwen3.8-Flash-Next lands at 40 rows per expert -> the table says 32, a tile GLM never
# selects). Measure, then fix the cell in the table below; never an env read.
_DYNAMIC_TILE_M_OVERRIDE: int | None = None
# Probe hooks: pin the micro (decode) kernel's M tile (32/64/128, N 128) and its MAC (max active
# clusters, 1..the SM count) for a decode cell under measurement (probes/engine_qwen38_moe.py:
# Qwen3.8's EP cell, whose zero-weight sentinel sends every 2..8-token launch to micro, where the
# selectors give M64 and the ladder's rungs are capped at GB10's 48 SMs). None keeps the selectors
# (_select_micro_mma_tiler_mn, _select_micro_mac); never an env read.
_MICRO_TILE_M_OVERRIDE: int | None = None
_MICRO_MAC_OVERRIDE: int | None = None
SF_VEC_SIZE = 16


def _admitted_moe():
    """The MoE cell this process is admitted for: the profile's bound kernel shape
    (engine/base/kernel_shape), or the measured GB10 TP4 cell when none is bound. Every
    exact-shape gate below compares against it, so the same dispatcher serves a second
    profile by declaration instead of by editing these functions."""
    from engine.base.kernel_shape import bound
    return bound().moe
# D11 (2026-09-12, ST): nothing in this module reads the environment. Every
# former knob is either the production value baked as a constant or a
# profile-declared knob applied through configure_static_v2() /
# configure_tp_sf6_q0() before any weight view exists (STK_moe_static).
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = True   # FLASHINFER_B12X_MICRO_SHARE_INPUT: never set, default on
# The pinned M128 GLM prefill candidate leaves small dynamic tiles and all
# decode backends on their original implementation. Exact 1 is intentional.
_GLM53_B12X_PREFILL_REUSE = False           # production VLLM_GLM53_B12X_PREFILL_REUSE=0
_GLM53_B12X_PREFILL_FC1_N128 = False        # production VLLM_GLM53_B12X_PREFILL_FC1_N128=0
_GLM53_EP_PREFILL_LOCAL = False             # EP (E=72 local) is not the ST form (TP=4, E=288 local)


_TP_SF6_Q0_ENABLED = False                  # configure_tp_sf6_q0(): STK_moe_static token q0
_TP_SF6_Q0_LAUNCH_LOGGED = False
_GLM53_EP_TILED = False                     # the EP tiled owner (E=72) left the ST package with its knob


def _tp_sf6_q0_eligible(*, enabled, E, m, k, n, num_topk, tile_m,
                         quant_mode, tiled, reform_sf_pack, activation,
                         swiglu_alpha, swiglu_beta, swiglu_limit,
                         share_input_across_experts, cell=None):
    # The recipe's FP32 sum must survive a layer failing lossless SF6
    # compression. Its raw-scale subclass uses the same Q0 and epilogue.
    # Long native chunks retain their original tiled SF6 path. Extending Q0
    # to those chunks passed numerics but did not improve the consumer run.
    # `cell` is the admitted MoE shape (the bound kernel shape's); E is the
    # weights' expert count, i.e. this rank's. The Q0 kernel's own tile (128)
    # and the swigluoai alpha/beta stay its constants.
    cell = _admitted_moe() if cell is None else cell
    return (enabled and type(m) is int and 1 <= m <= 8192
            and (E,k,n,num_topk,tile_m) == (cell.experts_local,cell.hidden,cell.inter_local,cell.topk,128)
            and quant_mode == cell.quant and tiled
            and not share_input_across_experts
            and (activation,swiglu_alpha,swiglu_beta,swiglu_limit)
                == (cell.activation,1.,0.,cell.swiglu_limit))


def _ep_local_prefill_kernel(*, E, m, k, n, num_topk, tile_m, activation,
                             swiglu_alpha, swiglu_beta, swiglu_limit, quant_mode, tiled):
    tiled_ep = (tiled and globals().get("_GLM53_EP_TILED", False)
                and (E, k, n, num_topk) == (72, 4096, 2048, 8))
    if tiled_ep:
        if type(m) is not int or not 1 <= m <= 16384:
            raise ValueError("tiled EP prefill requires 1..16384 tokens")
    elif not _GLM53_EP_PREFILL_LOCAL or not 4096 <= m <= 16384:
        return None
    if ((E, k, n, num_topk) != (72, 4096, 2048, 8)
            or (activation, swiglu_alpha, swiglu_beta, swiglu_limit, quant_mode)
            != ("swigluoai_uninterleave", 1.0, 0.0, 10.0, "nvfp4")):
        return None
    # This call can carry E72 as an invalid-ID sentinel. Once this geometry
    # is selected it MUST NOT silently run a stock kernel on those IDs.
    if _FORCED_BACKEND not in (None, "dynamic"):
        raise ValueError("expert-local prefill cannot use a forced incompatible backend")
    if (tiled and not tiled_ep) or tile_m != 128 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("expert-local prefill requires row-major SM121 M128")
    selected = select_sm120_moe_backend(
        num_tokens=m, num_topk=num_topk, quant_mode=quant_mode,
        num_experts=E, num_local_experts=E, hidden_size=k,
        intermediate_size=n, activation=activation, swiglu_limit=swiglu_limit)
    if selected != "dynamic" and not tiled_ep:
        raise ValueError("expert-local prefill requires dynamic backend selection")
    from .moe_dynamic_ep_local import MoEGatedEPLocalKernel, stock_contract_matches
    if not stock_contract_matches():
        raise RuntimeError("expert-local prefill inherited gated source has drifted")
    return MoEGatedEPLocalKernel


MoEGatedPrefillReuseKernel = None
MoEGatedPrefillN128Kernel = None
_prefill_reuse_announce = [False]
if _GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128:
    # 39차: boot-log anchor for the bracket -- the decline warning below only
    # fires when the lane is refused; this line proves the knob was armed.
    logging.getLogger("flashinfer.b12x").warning(
        "[b12x prefill reuse] armed: reuse=%s fc1_n128=%s",
        _GLM53_B12X_PREFILL_REUSE, _GLM53_B12X_PREFILL_FC1_N128,
    )


def _prefill_reuse_stock_contract_matches(*, fc1_n128: bool = False) -> bool:
    """Load private inherited helpers only after the optional shape gate.

    Upstream may remove a private symbol before the candidate can compare
    its source hash. That must decline this lane, including with the knob
    off, rather than break importing the otherwise unchanged dispatcher.
    An armed knob declines loudly: the operator asked for this lane, so a
    silent stock fallback would read as missing performance, not a guard.
    """
    global MoEGatedPrefillReuseKernel, MoEGatedPrefillN128Kernel
    reason = None
    try:
        from .moe_dynamic_prefill import (
            MoEGatedPrefillReuseKernel as candidate,
            stock_contract_matches,
        )
        if fc1_n128:
            from .moe_dynamic_prefill_n128 import (
                MoEGatedPrefillN128Kernel as wide_candidate,
            )
    except (ImportError, AttributeError) as exc:
        reason = f"deferred import failed: {exc!r}"
    else:
        if not stock_contract_matches():
            reason = "stock gated-source SHA-256 pin drifted (image upgrade?)"
    if reason is not None:
        if _GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128:
            logging.getLogger("flashinfer.b12x").warning(
                "[b12x prefill reuse] declining the opt-in lane, serving the "
                "stock dispatcher instead: %s", reason,
            )
        return False
    MoEGatedPrefillReuseKernel = candidate
    if fc1_n128:
        MoEGatedPrefillN128Kernel = wide_candidate
    return True

# Micro kernel cutover thresholds (routed pairs)
_MICRO_COMPACT_CUTOVER_PAIRS = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 40
# The micro kernel's per-token staging assumes decode-sized batches.
_MICRO_MAX_TOKENS = 8
# Deneb's local-only GLM EP experiment keeps the out-of-range sentinel in a
# full top-k=8 launch, then drops only its exact-zero pairs inside micro before
# row materialization. Exact "1" is intentional: unset, aliases, and typos all
# preserve the stock dispatcher and its cache artifacts.
_B12X_EP_ZERO_WEIGHT_MICRO = False          # production VLLM_B12X_EP_ZERO_WEIGHT_MICRO=0 (experimental EP lane)
_B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS = 72
_B12X_EP_ZERO_WEIGHT_MICRO_TOKENS = 8
_B12X_EP_ZERO_WEIGHT_MICRO_TOPK = 8
_B12X_EP_ZERO_WEIGHT_MICRO_K = 4096
_B12X_EP_ZERO_WEIGHT_MICRO_N = 2048
_B12X_EP_ZERO_WEIGHT_MICRO_SWIGLU_LIMIT = 10.0
# Direct micro takes the smallest decode batches ahead of the MMA micro
# kernel, and only at small intermediate sizes where its CUDA-core dots
# keep up with per-token work. Measured on GB10, pending other GPUs.
_DIRECT_MICRO_CUTOVER_PAIRS = 32
_DIRECT_MICRO_MAX_N = 512
# Test/bench hook: force one backend ("direct_micro", "micro", "static",
# "dynamic"). Deliberately module-level (a monkeypatch target), not an env var.
_FORCED_BACKEND: str | None = None
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = 640
_STATIC_COMPACT_CUTOVER_PAIRS = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: Dict[str, int] = {}


def _b12x_ep_zero_weight_micro_expert_id(
    *,
    enabled: bool,
    state_E: int,
    weight_E: int,
    num_tokens: int,
    k: int,
    n: int,
    num_topk: int,
    activation_precision: str,
    quant_mode: str,
    activation: str,
    swiglu_limit: float | None,
    forced_backend: str | None,
) -> int | None:
    """Return the one sentinel id the opt-in micro variant may discard.

    The Triton pre-pass writes its dense local->weight map at indices below the
    number of unique routed ids. Requiring routed_rows <= state_E proves that a
    sentinel which is numerically equal to E still cannot overflow that map:
    unique_ids <= routed_rows <= len(weight_expert_ids). The kernel receives E
    as a value in the map, never as an index into an E-row weight tensor.
    """
    routed_rows = int(num_tokens) * int(num_topk)
    exact_shape = (
        activation_precision == "fp4"
        and quant_mode == "nvfp4"
        and int(state_E) == _B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS
        and int(weight_E) == _B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS
        and int(num_tokens) == _B12X_EP_ZERO_WEIGHT_MICRO_TOKENS
        and int(num_topk) == _B12X_EP_ZERO_WEIGHT_MICRO_TOPK
        and int(k) == _B12X_EP_ZERO_WEIGHT_MICRO_K
        and int(n) == _B12X_EP_ZERO_WEIGHT_MICRO_N
        and activation == "swigluoai_uninterleave"
        and swiglu_limit == _B12X_EP_ZERO_WEIGHT_MICRO_SWIGLU_LIMIT
        and routed_rows <= int(state_E)
    )
    # The bound expert-parallel cell (configure_ep_zero_weight_micro): the same bound on the map, at the cell's
    # geometry, for every decode-sized launch the skip compiles for (the kernel refuses a single token).
    cell_shape = False
    if _EP_ZERO_WEIGHT_MICRO_CELL:
        cell = _admitted_moe()
        cell_shape = (
            activation_precision == "fp4"
            and quant_mode == cell.quant == "nvfp4"
            and int(state_E) == int(weight_E) == cell.experts_local < cell.experts
            and 2 <= int(num_tokens) <= _MICRO_MAX_TOKENS
            and int(num_topk) == cell.topk
            and int(k) == cell.hidden
            and int(n) == cell.inter_local
            and activation == cell.activation
            and swiglu_limit == cell.swiglu_limit
            and routed_rows <= int(state_E)
        )
    if ((enabled and exact_shape) or cell_shape) and forced_backend is not None:
        raise RuntimeError(
            "the zero-weight EP micro lane cannot run with forced MoE "
            f"backend {forced_backend!r}"
        )
    if not ((enabled and exact_shape) or cell_shape):
        return None
    # The local-only wrapper remaps every remote route to sentinel E at weight
    # zero. The micro kernel verifies both fields before suppressing the row.
    return int(state_E)


_EP_ZERO_WEIGHT_MICRO_CELL = False          # configure_ep_zero_weight_micro(): the bound EP cell's decode skip


def configure_ep_zero_weight_micro(enabled: bool) -> None:
    """An expert-parallel profile (the bound MoE cell holds fewer experts a rank than the model) sends its captured
    decode steps' routes to another rank's experts to sentinel E at weight zero, and the micro kernel drops those pairs
    before they claim a row: no rows, no quantisation, no expert weights read for them. Call before any launch, after
    the kernel shape is bound. `ep_zero_weight_sentinel` tells the caller, per launch shape, whether it may send E."""
    global _EP_ZERO_WEIGHT_MICRO_CELL
    if enabled:
        cell = _admitted_moe()
        if not cell.experts_local < cell.experts or cell.quant != "nvfp4":
            raise ValueError("the zero-weight micro skip serves an expert-parallel NVFP4 cell")
    _EP_ZERO_WEIGHT_MICRO_CELL = bool(enabled)


def ep_zero_weight_sentinel(*, num_tokens: int, num_topk: int, experts: int, hidden_size: int,
                            intermediate_size: int, activation: str, swiglu_limit: float | None) -> "int | None":
    """The sentinel id (E) a launch of this shape may carry for routes this rank does not hold, or None: the launch
    must then name one of its own experts (weight zero). Exactly the static dispatcher's decision for the shape --
    the static backend, and the micro skip admitted for it -- so a caller never hands E to a kernel that indexes
    with it."""
    if select_sm120_moe_backend(num_tokens=num_tokens, num_topk=num_topk, quant_mode="nvfp4", num_experts=experts,
                                num_local_experts=experts, hidden_size=hidden_size,
                                intermediate_size=intermediate_size, activation=activation,
                                swiglu_limit=swiglu_limit) != "static":
        return None
    return _b12x_ep_zero_weight_micro_expert_id(
        enabled=_B12X_EP_ZERO_WEIGHT_MICRO, state_E=experts, weight_E=experts, num_tokens=num_tokens,
        k=hidden_size, n=intermediate_size, num_topk=num_topk, activation_precision="fp4", quant_mode="nvfp4",
        activation=activation, swiglu_limit=swiglu_limit, forced_backend=_FORCED_BACKEND)

# MAC (max active clusters) tuning ladders from b12x decode profiling.
# Each entry is (max_routed_rows, optimal_mac).
_MICRO_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (2, 84),
    (4, 127),
    (8, 107),
    (10, 84),
    (16, 63),
    (20, 84),
)
_STATIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (24, 148),
    (32, 169),
    (40, 132),
    (48, 149),
    (64, 134),
    (80, 175),
    (96, 171),
    (120, 125),
    (128, 130),
    (160, 171),
    (192, 166),
    (256, 141),
    (320, 158),
    (512, 175),
    (640, 188),
)
# Workloads at or below the static cutover (640 routed pairs by default)
# take the static kernel, so only the 1024 entry is normally reachable.
_DYNAMIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (640, 188),
    (1024, 147),
)

# GLM-specific tuning controls.  These are diagnostic inputs, not production
# defaults: unset keeps the shipped selections and kernel cache keys. Parse
# once at import so CUDA graph capture/replay never reads the environment.
# The GLM backend/cutover/MAC-ladder diagnostics were five env knobs
# (ST_GLM53_B12X_FORCE_BACKEND, _STATIC_CUTOVER_PAIRS, _{MICRO,STATIC,DYNAMIC}_
# MAC_LADDER); production glm53.env carried every one of them empty, so the
# defaults below ARE the served configuration. Probes force a backend through
# _FORCED_BACKEND (the module-level hook above), never through the environment.
# Probe hooks: a probe assigns these module attributes directly (parsed values, before
# its first launch), the way _FORCED_BACKEND and _STATIC_V2_OVERRIDE already work.
_GLM53_B12X_FORCE_BACKEND: str | None = None
_GLM53_B12X_STATIC_CUTOVER_PAIRS: int | None = None
_GLM53_B12X_MICRO_MAC_LADDER: Tuple[Tuple[int, int], ...] | None = None
_GLM53_B12X_STATIC_MAC_LADDER: Tuple[Tuple[int, int], ...] | None = None
_GLM53_B12X_DYNAMIC_MAC_LADDER: Tuple[Tuple[int, int], ...] | None = None

# The decode-streaming static kernel: v4 (moe_static_kernel_v4, 38차 `u`, the
# profile default). Admitted for the exact GLM-5.3 TP geometry only; every
# other shape keeps the stock kernel. Value: unset/""/"0" = off; `u` = v4
# (FC1 halves over 512-wide K stages, gate and up in separate stages, FC2 2
# stages unless `g<n>` is given); `v` = v4 + the A ring; plus `f<fc1 stages>`,
# `g<fc2 stages>`, `s` (per-CTA %globaltimer stamps, probe only) and, for
# compatibility, `m32`/`a32`. The v2 (`1`, `m..`, `d`) and v3 (`w`, `e`, `k`)
# lanes were sunset in 34차 §8: those tokens are rejected, not remapped.
# Parsed once at import.
_GLM53_B12X_STATIC_V2_ENV = "STK_moe_static"   # the profile knob this spec comes from (error messages)
_STATIC_V2_DEFAULT = {
    "tile_m": 32, "fc1": 2, "fc2": 2, "a_rows": 32, "stamps": False,
    "wide": True, "skip_sf": False, "skip_a": False, "v4": True, "a_ring": False,
    # 39차: t = tile-major expert weights (moe_static_kernel_v5), h = 64-row
    "tiled": False, "sf_pack": False, "decode_reform": False,
    "reform_sf_pack": False,
    # l<n>: B stages prefetched into L2 n stages ahead by the DMA warp (0 = off); lf<n>: FC2's only
    "l2_prefetch": 0, "l2_prefetch_fc1": True,
    # z: the B stages land as ONE cp.async.bulk each from pre-swizzled tile-major boxes (bulk_b)
    "bulk_b": False,
    "fc2_scale_search": 0,  # ss1/ss2: opt-in static FC2 activation scale search
    "activation_scale_search": 0,  # as1/as2: FC1 + FC2, routed and dense, all backends
}
_STATIC_SUNSET_TOKENS = {
    "1": "the v2 default lane", "d": "the v2 dynamic schedule", "w": "the v3 lane",
    "e": "v3 even waves", "k": "the v3 last-wave split",
}


def _parse_glm53_static_v2(raw: str | None, *, probe: bool = False) -> dict | None:
    """Parse the v2 static-kernel spec; None keeps the stock kernel.

    probe=True admits the timing-only cells `xs` (skip FC1's SFB boxes) and
    `xa` (skip the A + SFA boxes), whose numerics are garbage; the serving
    parse (the env value) rejects them."""
    if raw is None:
        return None
    value = raw.strip()
    if value in ("", "0", "off"):
        return None
    cfg = dict(_STATIC_V2_DEFAULT)
    for token in value.split(","):
        token = token.strip()
        if token in _STATIC_SUNSET_TOKENS:
            raise ValueError(
                f"{_GLM53_B12X_STATIC_V2_ENV}: {token!r} selected {_STATIC_SUNSET_TOKENS[token]}, "
                "sunset in 34차 §8 -- use u (v4) or v (v4 + A ring)"
            )
        if token == "s":
            cfg["stamps"] = True
            continue
        if token == "v":
            # v4 + A ring: A and SFA loaded once per k tile on their own
            # 2-deep ring, shared by the gate and the up stage
            cfg["a_ring"] = True
            continue
        if token == "u":
            # v4 (moe_static_kernel_v4.py), the default configuration
            continue
        if token == "t":
            # v5 (moe_static_kernel_v5.py, 39차): v4 over tile-major expert
            # weights -- the dispatcher re-lays w13/w2 out so every TMA box is
            # one contiguous run
            cfg["tiled"] = True
            continue
        if token == "r":
            cfg["decode_reform"] = True
            continue
        if token == "batch":
            # K=7 with C=2: reuse the M16 expert tile and its packed
            # operand pipeline. The request limit and draft width stay fixed.
            cfg["batch_reform"] = True
            continue
        if token == "sf6":
            cfg["reform_sf_pack"] = True
            continue
        if token in ("ss1", "ss2"):
            cfg["fc2_scale_search"] = int(token[-1])
            continue
        if token in ("as1", "as2"):
            cfg["activation_scale_search"] = int(token[-1])
            continue
        if token == "q":
            # 39차 §4c: the FC1 weight scales arrive 6-bit packed (base + index
            # per 4 KB block) and the MMA warps expand them in the stage buffer.
            # Probe only: the gated prefill kernel reads the same scales and has
            # no expansion, so reject it at boot like xs/xa.
            if not probe:
                raise ValueError(
                    f"{_GLM53_B12X_STATIC_V2_ENV}: q is a probe-only cell "
                    "(packed scales the dynamic kernel cannot read)"
                )
            cfg["sf_pack"] = True
            continue
        if token in ("xs", "xa"):
            if not probe:
                raise ValueError(
                    f"{_GLM53_B12X_STATIC_V2_ENV}: {token} is a probe-only timing cell"
                )
            cfg["skip_sf" if token == "xs" else "skip_a"] = True
            continue
        if len(token) >= 2 and token[0] == "l" and token[1:].isdigit():
            # l<n> (2026-09-16): the static kernel's DMA warp asks L2 for the B stage n stages
            # ahead (cp.async.bulk.prefetch.L2, one request per contiguous 16 KB run, no smem).
            # A hint: the MMA reads the same bytes, so the numerics are the kernel's own.
            cfg["l2_prefetch"] = int(token[1:])
            cfg["l2_prefetch_fc1"] = True
            continue
        if token == "z":
            # z (2026-09-17, the reform tile): the tile-major boxes are stored in the smem stage's own byte
            # order (the canonical Swizzle<3,4,3> over 128 B rows for FC1, <2,4,3> over 64 B rows for FC2,
            # probes/b12x_reform_layout_print.py), and the DMA lane lands each B stage with one 1-D
            # cp.async.bulk instead of a TMA box of 128 / 256 row segments. The bytes the MMA reads are
            # the same; the storage must carry the swizzled kind (the launch checks).
            cfg["bulk_b"] = True
            continue
        if len(token) >= 3 and token[:2] == "lf" and token[2:].isdigit():
            # lf<n>: the same, for the item's FC2 boxes only (FC1's own prefetch measured slower)
            cfg["l2_prefetch"] = int(token[2:])
            cfg["l2_prefetch_fc1"] = False
            continue
        if len(token) < 2 or token[0] not in "mfga" or not token[1:].isdigit():
            raise ValueError(
                f"{_GLM53_B12X_STATIC_V2_ENV} must be 0 or comma-separated "
                f"u|v,f<fc1>,g<fc2>[,l<n>][,m32][,a32][,s][,t][,q][,r][,sf6] cells (got {raw!r})"
            )
        key = {"m": "tile_m", "f": "fc1", "g": "fc2", "a": "a_rows"}[token[0]]
        cfg[key] = int(token[1:])
    if cfg["fc2_scale_search"] and cfg["activation_scale_search"]:
        raise ValueError("choose ss1/ss2 (static FC2) or as1/as2 (all activations), not both")
    if cfg["tile_m"] != 32 or cfg["a_rows"] != 32:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: v4 is tile_m 32, a_rows 32")
    if cfg["fc1"] < 1 or cfg["fc2"] < 1:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: stages must be >= 1")
    if cfg["a_ring"] and cfg["skip_a"]:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: v (A ring) and xa are exclusive")
    # `r` keeps its geometry requirements and its conflicts with the cells that change the scale path
    # (a_ring, sf_pack). The two timing-only skips are not a geometry -- they drop a TMA issue at compile
    # time -- and they are already probe-gated above, so the served parse still cannot reach them.
    if cfg["decode_reform"] and (not cfg["tiled"] or any(
        cfg[key] for key in ("a_ring", "sf_pack")
    ) or cfg["fc1"] != 2 or cfg["fc2"] != 2):
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: r requires t with f2,g2")
    if cfg.get("reform_sf_pack") and not cfg["decode_reform"]:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: sf6 requires t,r")
    if cfg.get("batch_reform") and not (cfg["decode_reform"] and cfg["reform_sf_pack"]):
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: batch requires t,r,sf6")
    if cfg.get("l2_prefetch") and not (cfg["tiled"] and cfg["decode_reform"]):
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: l<n> requires t,r (a tile-major M16 box is one contiguous run)")
    if cfg.get("bulk_b") and not (cfg["tiled"] and cfg["decode_reform"]):
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: z requires t,r (its boxes are the M16 reform stages)")
    return cfg


# The served static-lane spec. No env read: the profile declares STK_moe_static
# (engine/profiles/glm53/boot.declared) and applies it once through
# configure_static_v2() before any weight view or launch exists. None = the
# stock static kernel; "t,r,sf6" = production's 2026-09-09 adoption.
_GLM53_B12X_STATIC_V2: dict | None = None


def _reject_tiled_with_prefill_reuse(cfg: "dict | None") -> None:
    # A tiled cell and the #368 prefill-reuse lane are mutually exclusive (the
    # reuse kernels read row-major storage). Diagnose it when the profile
    # applies the spec -- before any weight view or launch exists -- rather
    # than at import, where the spec is still None. The launch-time checks stay
    # for the _STATIC_V2_OVERRIDE probe hook.
    if (
        cfg
        and cfg.get("tiled")
        and (_GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128)
    ):
        raise ValueError(
            f"{_GLM53_B12X_STATIC_V2_ENV}: a tiled cell (t) cannot serve together "
            "with the prefill reuse lanes -- they read row-major expert weights"
        )
# Probe hook: a config dict overrides the import-time env value; module-level
# (a monkeypatch target), never read from the environment at launch time.
_STATIC_V2_OVERRIDE: dict | None = None


def configure_static_v2(spec: "str | None") -> "dict | None":
    """Apply the profile's static-lane spec (D11 knob STK_moe_static) once per process.

    Refused after any weight view exists: the tiled/row-major choice is keyed into
    every cached view and into the in-place relayout of served weights, so a later
    switch would misread bytes (see _get_weight_views)."""
    global _GLM53_B12X_STATIC_V2
    cfg = _parse_glm53_static_v2(spec)
    if cfg == _GLM53_B12X_STATIC_V2:
        return _GLM53_B12X_STATIC_V2          # the same spec: nothing to apply
    if _WEIGHT_CACHE or _REFORM_SF_CACHE:
        raise RuntimeError("configure_static_v2: weight views already exist in this process; "
                           "the static-lane spec is fixed before the first MoE bind")
    _GLM53_B12X_STATIC_V2 = cfg
    _reject_tiled_with_prefill_reuse(cfg)
    return cfg


def configure_tp_sf6_q0(enabled: bool) -> None:
    """TP prefill Q0 metadata cache over packed SF6 scales (STK_moe_static token q0):
    production's TP recipe (glm53.env: "Restore TP with ... TP_SF6_Q0=1"). Needs t,r,sf6."""
    global _TP_SF6_Q0_ENABLED
    if enabled and not (_GLM53_B12X_STATIC_V2 and _GLM53_B12X_STATIC_V2.get("tiled")
                        and _GLM53_B12X_STATIC_V2.get("reform_sf_pack")):
        raise ValueError("STK_moe_static: q0 needs the t,r,sf6 cells")
    _TP_SF6_Q0_ENABLED = bool(enabled)


_ACTIVATION_SCALE_SEARCH_RADIUS = None      # profile default, fixed before weight views exist


def configure_activation_scale_search(radius: int) -> None:
    """Declare a profile's NVFP4 activation search without selecting GLM's weight layout."""
    global _ACTIVATION_SCALE_SEARCH_RADIUS
    if type(radius) is not int or radius not in (0, 1, 2):
        raise ValueError("activation scale search radius must be 0, 1 or 2")
    if radius == _ACTIVATION_SCALE_SEARCH_RADIUS:
        return
    if _WEIGHT_CACHE or _REFORM_SF_CACHE:
        raise RuntimeError("configure activation scale search before preparing MoE weights")
    _ACTIVATION_SCALE_SEARCH_RADIUS = radius


def _activation_scale_search_for(**geometry) -> int:
    """The bound routed/dense NVFP4 cell only; the old ss recipe stays FC2-only.

    Reuse the admitted FP32-scatter geometry, including the dense MLP cell.
    This does not alter scatter arithmetic, routing or the W4A16 guard.
    """
    cfg = _STATIC_V2_OVERRIDE if _STATIC_V2_OVERRIDE is not None else _GLM53_B12X_STATIC_V2
    radius = int((cfg or {}).get("activation_scale_search", 0))
    if _STATIC_V2_OVERRIDE is None and _ACTIVATION_SCALE_SEARCH_RADIUS is not None:
        radius = _ACTIVATION_SCALE_SEARCH_RADIUS
    if (radius and geometry["quant_mode"] == "nvfp4"
            and _glm_tp_scatter_fp32(**geometry)):
        return radius
    return 0
_STATIC_V2_STAMPS: Dict[Tuple[int, str], "torch.Tensor"] = {}
_STATIC_V2_COUNTERS: Dict[str, "torch.Tensor"] = {}


def _static_v2_config_for(
    *,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_topk: int,
    quant_mode: str,
    activation: str,
    swiglu_limit: float | None,
    activation_precision: str,
) -> dict | None:
    """The v2 config to launch, or None for the stock static kernel."""
    cfg = _STATIC_V2_OVERRIDE if _STATIC_V2_OVERRIDE is not None else _GLM53_B12X_STATIC_V2
    if cfg is None or activation_precision != "fp4":
        return None
    if not _is_admitted_tp_geometry(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_topk=num_topk,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
    ):
        return None
    return cfg


def _static_v2_stamps_tensor(mac: int, device: "torch.device") -> "torch.Tensor":
    key = (int(mac), str(device))
    tensor = _STATIC_V2_STAMPS.get(key)
    if tensor is None:
        tensor = torch.zeros((int(mac), _STATIC_V2_STAMP_SLOTS), dtype=torch.int64, device=device)
        _STATIC_V2_STAMPS[key] = tensor
    return tensor


def _static_v2_counter_tensor(device: "torch.device") -> "torch.Tensor":
    """The dynamic scheduler's claim counter: one int32 the kernel zeroes in
    its phase 0 (launches on one stream are serialized, so one per process
    suffices; a fixed address keeps it CUDA-graph safe)."""
    key = str(device)
    tensor = _STATIC_V2_COUNTERS.get(key)
    if tensor is None:
        tensor = torch.zeros((1,), dtype=torch.int32, device=device)
        _STATIC_V2_COUNTERS[key] = tensor
    return tensor


def _lookup_mac_ladder(
    ladder: Tuple[Tuple[int, int], ...], routed_rows: int
) -> int | None:
    """Look up optimal MAC from a tuning ladder. Returns None if no match."""
    for end_rows, mac in ladder:
        if routed_rows <= end_rows:
            return mac
    return None


def _is_admitted_tp_geometry(
    *,
    num_experts: int | None,
    num_local_experts: int | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_topk: int,
    quant_mode: str,
    activation: str | None,
    swiglu_limit: float | None,
) -> bool:
    """Admit only the MoE geometry the bound kernel shape declares (the deployed
    GLM-5.3 TP-sharded NVFP4 cell: 288 experts, hidden 4096, intermediate 512 per
    rank, top-8 -- when nothing is bound).

    The intermediate size arrives in two spellings: the model's (2048) at the
    wrapper level and the PER-RANK one (512 = 2048 / TP4) that
    ``launch_sm120_static_moe`` derives from the sharded weights and passes
    down. Until 2026-09-05 only the model's was admitted, so every launch-time
    reader of this gate (the forced backend, the cutover, the MAC ladders and
    the static v2 lane) silently kept the stock path -- the first v2 probe
    measured the stock kernel six times over. Both spellings come from the cell.
    """
    cell = _admitted_moe()
    return (
        num_experts == cell.experts
        and num_local_experts == cell.experts_local
        and hidden_size == cell.hidden
        and intermediate_size in (cell.inter_local, cell.inter)
        and num_topk == cell.topk
        and quant_mode == cell.quant
        and activation == cell.activation
        and swiglu_limit == cell.swiglu_limit
    )


def _effective_glm53_forced_backend(
    *,
    num_tokens: int,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_topk: int,
    quant_mode: str,
    activation: str,
    swiglu_limit: float | None,
) -> str | None:
    """Return the monkeypatch hook first, then the exact-shape GLM override."""
    if _FORCED_BACKEND is not None:
        return _FORCED_BACKEND
    if _is_admitted_tp_geometry(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_topk=num_topk,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
    ):
        # Micro is a decode-only kernel.  Keep prefill on automatic dispatch
        # so a diagnostic boot cannot turn an ordinary long call into a
        # forced-micro correctness failure or a giant static-only workspace.
        if _GLM53_B12X_FORCE_BACKEND == "micro" and num_tokens > _MICRO_MAX_TOKENS:
            return None
        return _GLM53_B12X_FORCE_BACKEND
    return None


def _effective_glm53_static_cutover(
    default: int,
    *,
    num_experts: int | None,
    num_local_experts: int | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_topk: int,
    quant_mode: str,
    activation: str | None,
    swiglu_limit: float | None,
) -> int:
    if _is_admitted_tp_geometry(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_topk=num_topk,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
    ):
        cutover = (
            _GLM53_B12X_STATIC_CUTOVER_PAIRS
            if _GLM53_B12X_STATIC_CUTOVER_PAIRS is not None
            else default
        )
        # A decode-only forced-micro run still allocates its static workspace
        # from this boundary. Reserve the full m=8/top-k=8 band even when a
        # simultaneous generic or GLM cutover of zero sends every other call
        # to dynamic.
        if _GLM53_B12X_FORCE_BACKEND == "micro":
            cutover = max(cutover, _MICRO_MAX_TOKENS * num_topk)
        return cutover
    return default


def _effective_glm53_mac_ladder(
    default: Tuple[Tuple[int, int], ...],
    override: Tuple[Tuple[int, int], ...] | None,
    *,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_topk: int,
    quant_mode: str,
    activation: str,
    swiglu_limit: float | None,
) -> Tuple[Tuple[int, int], ...]:
    if override is not None and _is_admitted_tp_geometry(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_topk=num_topk,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
    ):
        return override
    return default


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


# The kernels index the packed activation and scale planes with 32-bit
# offsets, so reject any workspace plane large enough to overflow them.
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1


def _check_memref_limit(name: str, elements: int) -> None:
    if elements > _RUNTIME_MEMREF_LIMIT:
        raise ValueError(
            f"{name} needs {elements} elements, which exceeds the 2^31-1 "
            "runtime memref limit. Reduce the token chunk or expert count "
            "for this launch."
        )


def _normalize_activation_precision(activation_precision: str) -> str:
    """Normalize public activation-precision names to internal modes."""

    normalized = str(activation_precision).lower()
    aliases = {
        "fp4": "fp4",
        "nvfp4": "fp4",
        "w4a4": "fp4",
        "bf16": "bf16",
        "w4a16": "bf16",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "activation_precision must be 'fp4' or 'bf16' "
            f"(got {activation_precision!r})."
        ) from exc


def _normalize_quant_mode(
    quant_mode: str | None = None,
    activation_precision: str | None = None,
) -> str:
    """Normalize public quantization names to the dispatch mode."""
    if quant_mode is None:
        activation_precision = _normalize_activation_precision(
            activation_precision or "fp4"
        )
        return "w4a16" if activation_precision == "bf16" else "nvfp4"

    normalized = str(quant_mode).lower()
    aliases = {
        "fp4": "nvfp4",
        "nvfp4": "nvfp4",
        "w4a4": "nvfp4",
        "mxfp4": "mxfp4",
        "bf16": "w4a16",
        "w4a16": "w4a16",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "quant_mode must be 'nvfp4'/'w4a4', 'mxfp4', or 'w4a16' "
            f"(got {quant_mode!r})."
        ) from exc


def _sf_params_for_quant_mode(quant_mode: str):
    """Return (vector size, CuTe scale dtype) for a W4A4 mode."""
    mode = _normalize_quant_mode(quant_mode)
    if mode == "mxfp4":
        return _MXFP4_BLOCK_SIZE, cutlass.Float8E8M0FNU
    return _NVFP4_BLOCK_SIZE, cutlass.Float8E4M3FN


def _activation_precision_from_quant_mode(quant_mode: str) -> str:
    return "bf16" if _normalize_quant_mode(quant_mode) == "w4a16" else "fp4"


def _normalize_source_format_for_quant_mode(source_format: str, quant_mode: str) -> str:
    normalized = _normalize_source_format(source_format)
    if quant_mode != "w4a16" and normalized == "compressed_tensors":
        raise ValueError(
            "source_format='compressed_tensors' requires quant_mode='w4a16'."
        )
    return normalized


def _is_w4a16(activation_precision: str) -> bool:
    return _normalize_activation_precision(activation_precision) == "bf16"


def _level_tile_n(activation_precision: str = "fp4") -> int:
    if _is_w4a16(activation_precision):
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 tile selector"
        )
    return _LEVEL_TILE_N


def _select_dynamic_tile_m(
    routed_rows: int,
    num_experts: int,
    activation: str = "silu",
) -> int:
    """Pick the dynamic kernel's M-tile from routed rows per expert.

    Small tiles cut per-expert tail padding for sparse routing; 128 amortizes
    best for dense prefill (crossovers measured on gated NVFP4). Workspace
    sizing and the kernel build must both derive the tile from this function,
    or the scratch is mis-sized for what the kernel indexes.
    """
    # A shape cell, expressed as an override. Qwen3.8-Flash-Next lands at
    # 20480/512 = 40 rows per expert, which the table below answers with
    # tile_m 32 -- a tile GLM-5.3 never selects (288 experts at top-8 gives
    # ~57 rows per expert, i.e. 64). Pinning it is how a cell for a new shape
    # starts: measure the tile, then fix it here rather than leave the model
    # on whichever branch the generic table happens to pick.
    if _DYNAMIC_TILE_M_OVERRIDE is not None:
        if _DYNAMIC_TILE_M_OVERRIDE not in (16, 32, 64, 128):
            raise ValueError(f"_DYNAMIC_TILE_M_OVERRIDE must be 16/32/64/128, got {_DYNAMIC_TILE_M_OVERRIDE}")
        return _DYNAMIC_TILE_M_OVERRIDE
    # The profile's measured pin (kernel_shape.MoE.dynamic_tile_m): the cell fixed after it was
    # measured, the way the comment above says a new shape's cell starts. None keeps the table.
    pinned = _admitted_moe().dynamic_tile_m
    if pinned is not None:
        return pinned
    if not is_gated_activation(activation):
        return _LEVEL_TILE_M
    routed_rows = max(1, int(routed_rows))
    num_experts = max(1, int(num_experts))
    if routed_rows < 15 * num_experts:
        return 16
    if routed_rows < 48 * num_experts:
        return 32
    if routed_rows < 96 * num_experts:
        return 64
    return _LEVEL_TILE_M


def _get_static_compact_cutover_pairs(activation_precision: str = "fp4") -> int:
    activation_precision = _normalize_activation_precision(activation_precision)
    cached = _STATIC_COMPACT_CUTOVER_PAIRS_CACHE.get(activation_precision)
    if cached is not None:
        return cached

    # The cutover was four env aliases (FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS
    # and three older names); production never set one, so the default is served.
    cached = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
    _STATIC_COMPACT_CUTOVER_PAIRS_CACHE[activation_precision] = cached
    return cached


def _select_moe_mma_tiler_mn(
    routed_rows: int,
    n: int,
    *,
    resident_clusters: int | None = None,
) -> Tuple[int, int]:
    """Select optimal MoE tile shape based on routed rows and N dimension.

    Uses narrower 64x128 tiles when routed_rows <= 128 and default 128x128
    would leave SMs idle.
    """
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    if routed_rows <= 32 and n <= 256:
        return (64, 128)
    if resident_clusters is not None and resident_clusters < sm_count:
        return coarse_tile
    coarse_tiles = ((routed_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # Single-token decode often lands exactly on the "half the machine"
    # boundary. Keeping the coarse 128x128 tile there leaves the M dimension
    # badly underfilled, so take the narrow 64x128 tile inclusive of equality.
    if routed_rows <= 64 or (
        routed_rows <= 128 and coarse_tiles <= max(1, sm_count // 2)
    ):
        return (64, 128)
    return (128, 128)


def _select_micro_mma_tiler_mn(
    *, state_E: int, weight_E: int, m: int, k: int, n: int, num_topk: int,
    skip_zero_weight_expert_id: int | None, quant_mode: str,
    activation: str, swiglu_alpha: float, swiglu_beta: float,
    swiglu_limit: float | None,
    max_rows: int | None = None,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
) -> Tuple[int, int]:
    if _MICRO_TILE_M_OVERRIDE is not None:
        # M16 is the EP direct-scatter variant's (ep_m16: two MMA warps, a one-row atom); the
        # stock micro atom spans 32 rows, so a pinned tile is 32, 64 or 128.
        if _MICRO_TILE_M_OVERRIDE not in (32, 64, 128):
            raise ValueError(f"_MICRO_TILE_M_OVERRIDE must be 32/64/128, got {_MICRO_TILE_M_OVERRIDE!r}")
        return (_MICRO_TILE_M_OVERRIDE, 128)
    # Only the shared/direct EP candidate uses M16 with two MMA warps.
    # Missing or different execution metadata retains the prior M32 choice;
    # every other geometry keeps the original selector.
    if (
        (state_E, weight_E, m, k, n, num_topk, skip_zero_weight_expert_id)
        == (72, 72, 8, 4096, 2048, 8, 72)
        and (quant_mode, activation, swiglu_alpha, swiglu_beta, swiglu_limit)
        == ("nvfp4", "swigluoai_uninterleave", 1.0, 0.0, 10.0)
    ):
        if (max_rows == 64 and not share_input_across_experts
                and not share_expert_scales and not single_token):
            return (16, 128)
        return (32, 128)
    return _select_moe_mma_tiler_mn(m * num_topk, n)


def _select_micro_mac(
    routed_rows: int,
    n: int,
    base_mac: int,
    ladder: Tuple[Tuple[int, int], ...],
) -> int:
    """The micro kernel's MAC: the tuned ladder's rung for these routed rows, capped by the work
    tiles (routed rows x N slices) and by `base_mac`, the hardware limit (the SM count). A rung
    above the SM count is capped to it. `_MICRO_MAC_OVERRIDE` pins it, never above `base_mac`."""
    if _MICRO_MAC_OVERRIDE is not None:
        if type(_MICRO_MAC_OVERRIDE) is not int or not 1 <= _MICRO_MAC_OVERRIDE <= base_mac:
            raise ValueError(f"_MICRO_MAC_OVERRIDE must be an int in 1..{base_mac}, got {_MICRO_MAC_OVERRIDE!r}")
        return _MICRO_MAC_OVERRIDE
    micro_work_tiles = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
    tuned_mac = _lookup_mac_ladder(ladder, routed_rows)
    return min(tuned_mac or base_mac, micro_work_tiles, base_mac)


def _as_grouped_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Create 6D MMA-compatible scale factor view from swizzled storage."""
    batch = scale_storage.shape[0]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120StaticMoEWorkspace:
    """Scratch buffers for one SM120 static MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation_precision: str
    quant_mode: str

    # Buffers
    row_counts: torch.Tensor  # [state_E] int32
    token_map: torch.Tensor  # [state_E, max_rows] int32
    token_weights: torch.Tensor  # [state_E, max_rows] float32
    packed_input: torch.Tensor  # [state_E, max_rows, k//2] uint8
    packed_input_scale: torch.Tensor  # [state_E, rows_pad_k, cols_pad_k] uint8
    barrier_count: torch.Tensor  # [1] int32
    barrier_epoch: torch.Tensor  # [1] int32
    active_expert_count: torch.Tensor  # [1] int32
    weight_expert_ids: torch.Tensor  # [state_E] int32
    global_to_local_expert: torch.Tensor  # [weight_E] int32
    compact_topk_ids: torch.Tensor  # [state_E] int32, for micro kernel pre-pass

    # Views (set after allocation)
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None

    # Direct micro planes (allocated only when the shape can take that path).
    dm_barrier_count: torch.Tensor | None = None
    dm_barrier_epoch: torch.Tensor | None = None
    dm_intermediate: torch.Tensor | None = None
    dm_input_gs: torch.Tensor | None = None
    dm_down_input_scale: torch.Tensor | None = None
    # Pinned once with the two E72 decode workspaces, before graph capture.
    ep_micro_scatter_fp32: torch.Tensor | None = None
    glm_tp_scatter_fp32: torch.Tensor | None = None


def _direct_micro_candidate(k: int, n: int, num_topk: int, weight_E: int) -> bool:
    """Whether any m in the tiny-decode band can run the direct micro kernel."""
    return any(
        MoEDirectMicroKernel.is_supported(m, k, n, num_topk, weight_E)
        for m in range(1, _MICRO_MAX_TOKENS + 1)
    )


def allocate_sm120_static_workspace(
    *,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
) -> Sm120StaticMoEWorkspace:
    """Allocate workspace buffers for the SM120 static MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_static_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )

    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    # A single dense expert still needs a row axis in its packed-input TMA
    # map. (E=1, rows=1) collapses to a 1D FP4 transfer on SM121. Reserve one
    # scale tile; routing/valid-row counts continue to describe the real M.
    if state_E == weight_E == 1 and quant_mode == "nvfp4":
        max_rows = max(max_rows, 128)
    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)
    _check_memref_limit("static packed_input", state_E * max_rows * (k // 2))
    _check_memref_limit("static packed_input_scale", state_E * rows_pad_k * cols_pad_k)
    packed_input = torch.empty(
        state_E, max_rows, k // 2, dtype=torch.uint8, device=device
    )

    workspace = Sm120StaticMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(state_E, max_rows, dtype=torch.int32, device=device),
        token_weights=torch.zeros(
            state_E, max_rows, dtype=torch.float32, device=device
        ),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            state_E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(state_E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
        compact_topk_ids=torch.empty(
            max(state_E, max_rows), dtype=torch.int32, device=device
        ),
    )

    if _glm_tp_scatter_shape(state_E, weight_E, k, n, num_topk) and quant_mode == "nvfp4":
        workspace.glm_tp_scatter_fp32 = torch.empty(
            (max(1, max_rows // num_topk), k), dtype=torch.float32, device=device)

    # Finalize views
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    if (state_E == weight_E == 72 and k == 4096 and n == 2048
            and quant_mode == "nvfp4"
            and (num_topk, max_rows) in ((1, 8), (8, 64))):
        # One 128 KiB plane per shared fixed workspace, never per layer or
        # per call. Its address must not grow or change after graph capture.
        workspace.ep_micro_scatter_fp32 = torch.empty(
            (8, k), dtype=torch.float32, device=device
        )

    # Direct micro reads weights by global expert id, so its planes are only
    # useful without EP remapping.
    if (
        quant_mode == "nvfp4"
        and state_E == weight_E
        and _direct_micro_candidate(k, n, num_topk, weight_E)
    ):
        dm_rows = min(max_rows, _MICRO_MAX_TOKENS * num_topk)
        # The epoch-based barriers restore their slots after each launch, so
        # the zeroed allocation is the only reset needed (graph-replay safe).
        dm_slots = dm_rows + _MICRO_MAX_TOKENS * 16
        fc2_n_chunks = (n // 2 + 127) // 128
        # The fused kernel binds the intermediate as m * num_topk *
        # fc2_n_chunks * 128 u32 words; size for the largest supported m.
        dm_inter = _MICRO_MAX_TOKENS * num_topk * fc2_n_chunks * 128
        workspace.dm_barrier_count = torch.zeros(
            dm_slots, dtype=torch.int32, device=device
        )
        workspace.dm_barrier_epoch = torch.zeros(
            dm_slots, dtype=torch.int32, device=device
        )
        workspace.dm_intermediate = torch.empty(
            dm_inter, dtype=torch.float32, device=device
        )
        workspace.dm_input_gs = torch.empty(
            weight_E, dtype=torch.float32, device=device
        )
        workspace.dm_down_input_scale = torch.empty(
            weight_E, dtype=torch.float32, device=device
        )
    return workspace


# ---------------------------------------------------------------------------
# Weight views
# ---------------------------------------------------------------------------
@dataclass
class _WeightViews:
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None
    # tile-major expert weights (spec cell t, moe_static_kernel_v5): the
    # views are 4-D over a re-laid-out copy kept alive here; a tiled view must
    # only ever reach a kernel compiled for the tiled layout
    tiled: bool = False
    # the w13 chunk (fp4 per row per k tile) of that layout; every kernel the
    # view reaches is compiled for this chunk (TILED_W13_CHUNKS)
    w13_chunk: int = TILED_W13_K_IN
    # cell q: the FC1 scales 6-bit packed, (E, blocks, stage bytes) u8
    sfb1_packed: torch.Tensor | None = None
    sfb2_packed: torch.Tensor | None = None
    # Packed-only owners never retain a raw scale tensor, including aliases.
    packed_only: bool = False
    reform_scales: object | None = None
    w13_tiled_storage: torch.Tensor | None = None
    w2_tiled_storage: torch.Tensor | None = None
    # cell z: the tiled storage's boxes are pre-swizzled into the reform stages' byte order; only a
    # kernel built with bulk_b may read such a view, and only such a view may reach that kernel
    swizzled: bool = False
    w1_alpha: torch.Tensor | None = None
    w2_alpha: torch.Tensor | None = None
    w1_storage: torch.Tensor | None = None
    w1_scale_storage: torch.Tensor | None = None
    w2_storage: torch.Tensor | None = None
    w2_scale_storage: torch.Tensor | None = None
    _w13_sf_storage: torch.Tensor | None = None
    _down_sf_storage: torch.Tensor | None = None


def _register_cache_eviction(cache: Dict, key: Tuple, *source_tensors) -> None:
    """Evict ``key`` when a source weight tensor is collected, so the cache
    follows the weights' lifetime instead of growing for the whole process.
    """
    for tensor in source_tensors:
        if tensor is not None:
            weakref.finalize(tensor, cache.pop, key, None)


_WEIGHT_CACHE: Dict[Tuple, Tuple] = {}


_SF_PACKED: Dict[Tuple[int, int], torch.Tensor] = {}
_SF_PACK_DUMMY: Dict[str, torch.Tensor] = {}
_REFORM_SF_CACHE: Dict[Tuple, object] = {}


def _prepared_reform_scales(source1, source2, raw1, raw2, *, experts, n, k):
    """One packed owner per scale generation, independent of folded alphas.

    A shared wrapper may rebuild its per-layer alpha views. That must not
    repack identical scales, or retain a new packed copy on every warmup.
    Sources own the generation; cache entries keep packed addresses alive
    across wrapper changes and disappear when those sources are collected.
    """
    key = ("sf6-v1", experts, n, k, tuple(
        (id(t), t.data_ptr(), _sf6_tensor_version(t), tuple(t.shape), str(t.device))
        for t in (source1, source2)))
    cached = _REFORM_SF_CACHE.get(key)
    if cached is not None:
        return cached
    if raw1.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("sf6 scales must be prepared before CUDA graph capture")
    owner = prepare_reform_scales(raw1.view(torch.uint8), raw2.view(torch.uint8),
                                  experts=experts, n=n, k=k)
    _REFORM_SF_CACHE[key] = owner
    _register_cache_eviction(_REFORM_SF_CACHE, key, source1, source2)
    logging.getLogger("flashinfer.b12x").warning(
        "[b12x sf6] %s; packed bytes=%d",
        "prepared FC1+FC2" if owner.enabled else f"raw fallback: {owner.reason}",
        (owner.fc1.numel() + owner.fc2.numel()) if owner.enabled else 0,
    )
    return owner


def consume_packed_scale_storage(views, first, second):
    """Retire arena raw scales once every consumer holds this SF6 owner."""
    from engine.modules.packed_storage import consume
    from .moe_reform_sf_pack import ReformScales
    owner=views.reform_scales
    if not views.packed_only or owner is None or not owner.enabled:
        raise ValueError("raw-scale retirement requires a packed-only SF6 layer")
    for storage,pack in ((first,owner.fc1),(second,owner.fc2)):
        if (not storage.is_contiguous() or storage.device!=pack.device
                or storage.numel()*storage.element_size()<pack.numel()):
            raise ValueError("SF6 pack does not fit its raw-scale arena region")
    a,=consume(first,[owner.fc1])
    b,=consume(second,[owner.fc2])
    replacement=ReformScales(a,b)
    for key,cached in tuple(_REFORM_SF_CACHE.items()):
        if cached is owner:
            _REFORM_SF_CACHE[key]=replacement
    views.reform_scales=replacement
    views.sfb1_packed,views.sfb2_packed=a,b
    first._st_sf6_consumed=second._st_sf6_consumed=True


def _packed_fc1_scales(sf: torch.Tensor, num_experts: int) -> torch.Tensor:
    """(E, blocks per expert, SF_STAGE_BYTES) u8 -- the FC1 weight scales
    6-bit packed per 4 KB block, the unit the kernel stages (39차 §4c). Cached
    on the scale buffer's identity plus generation: packing walks every scale
    byte once, and the caller's buffer is a transient contiguous copy, so a
    pointer-only key can hand back a previous generation's bytes."""
    key = (id(sf), sf.data_ptr(), _sf6_tensor_version(sf), sf.numel(),
           num_experts, str(sf.device))
    got = _SF_PACKED.get(key)
    if got is not None:
        return got
    flat = sf.reshape(-1).view(torch.uint8)
    if flat.numel() % (SF_PACK_BLOCK * num_experts):
        raise ValueError(
            f"packed scales (cell q): {flat.numel()} scale bytes is not "
            f"{num_experts} experts x whole {SF_PACK_BLOCK} B blocks"
        )
    blocks = flat.numel() // SF_PACK_BLOCK
    packed = pack_sf_inline(flat, SF_PACK_BLOCK).reshape(
        num_experts, blocks // num_experts, SF_STAGE_BYTES
    )
    _SF_PACKED[key] = packed
    _register_cache_eviction(_SF_PACKED, key, sf)
    return packed


def _sf_pack_dummy(device: "torch.device") -> torch.Tensor:
    """The 16 B stand-in the kernel takes but never reads off the q lane."""
    key = str(device)
    got = _SF_PACK_DUMMY.get(key)
    if got is None:
        got = torch.zeros((1, 1, 16), dtype=torch.uint8, device=device)
        _SF_PACK_DUMMY[key] = got
    return got


def static_v2_weights_reform_sf_pack(**geometry) -> bool:
    cfg = _static_v2_config_for(**geometry)
    return bool(cfg is not None and cfg.get("reform_sf_pack", False))


def _sf6_tensor_version(tensor) -> int:
    # Inference tensors have no version counter. Their deployment contract is
    # immutable; identity and storage pointers still distinguish load cycles.
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1


def static_v2_weights_sf_pack(**geometry) -> bool:
    """Whether the static lane wants 6-bit packed FC1 scales (cell q)."""
    cfg = _static_v2_config_for(**geometry)
    return bool(cfg is not None and cfg.get("sf_pack", False))


_TILE_MAJOR_ATTR = "_b12x_tile_major"   # False / "plain" / "plain<chunk>" on a weight tensor
# The w13 chunk of the served tile-major relayout, in fp4 elements per row. 256: the M16
# reform's K256 FC1 box (C=1 and C=2 decode) is one chunk, a contiguous 16 KB run, and the
# gated prefill kernels' K128 box half of one; over 512 both read part of every row's chunk.
# The t tile's K512 box reads two chunks and stays exact. Same-build control: a view naming
# TILED_W13_K_IN (measurements/st_c2_moe_chunk_20260915).
_W13_TILE_CHUNK = 256


def _w13_tile_chunk(chunk: "int | None" = None) -> int:
    """The w13 chunk a view or kernel uses: the served one unless named."""
    chunk = _W13_TILE_CHUNK if chunk is None else int(chunk)
    if chunk not in TILED_W13_CHUNKS:
        raise ValueError(f"tiled expert weights: w13 chunk {chunk} is not one of {TILED_W13_CHUNKS}")
    return chunk


def _tile_major_kind(chunk: int) -> str:
    """The marker of a tile-major tensor: "plain" is the original 512 chunk,
    so storage re-laid out before chunks were named keeps its meaning."""
    return "plain" if chunk == TILED_W13_K_IN else f"plain{chunk}"
# sha256 of three 64-byte samples, recorded with _TILE_MAJOR_ATTR so a later
# load cycle that overwrote the bytes can be detected (the marker itself
# survives a plain param.data.copy_ of fresh row-major checkpoint bytes).
_TILE_MAJOR_FP_ATTR = "_b12x_tile_major_fp"


def _tile_major_fingerprint(w: torch.Tensor) -> str:
    """Cheap content probe: first/middle/last 64 bytes of the packed bytes."""
    flat = w.view(-1)
    n = flat.numel()
    sample = torch.cat([flat[:64], flat[n // 2 : n // 2 + 64], flat[-64:]])
    return hashlib.sha256(sample.cpu().numpy().tobytes()).hexdigest()


def _evict_weight_cache_for(*tensors: torch.Tensor) -> None:
    """Drop _WEIGHT_CACHE entries built over any of these tensors' storage."""
    ptrs = {tensor.data_ptr() for tensor in tensors if tensor is not None}
    stale = [
        key
        for key in _WEIGHT_CACHE
        if any(ptr in ptrs for ptr in key[4:])
    ]
    for key in stale:
        del _WEIGHT_CACHE[key]


def invalidate_tile_major_if_reloaded(
    w1_fp4: torch.Tensor, w2_fp4: torch.Tensor
) -> bool:
    """Drop tile-major markers when the bytes were rewritten since the relayout.

    A vLLM re-load cycle copies fresh row-major checkpoint bytes over
    the same Parameters (same pointers) -- a surviving ``_b12x_tile_major``
    marker would then serve those bytes as tile-major. Call at the top of
    every process_weights_after_loading: a matching fingerprint (bytes
    untouched) keeps the markers and the call is a no-op; a mismatch clears
    them so the relayout below re-lays the fresh bytes, and every cached
    weight view over this storage is dropped. Weight writes that bypass
    process_weights_after_loading entirely are outside this guard; the
    serving contract is that every load cycle runs it.
    """
    cleared = False
    for w in (w1_fp4, w2_fp4):
        fp = getattr(w, _TILE_MAJOR_FP_ATTR, None)
        if fp is None:
            continue
        if _tile_major_fingerprint(w) != fp:
            for attr in (_TILE_MAJOR_ATTR, _TILE_MAJOR_FP_ATTR):
                try:
                    delattr(w, attr)
                except AttributeError:
                    pass
            cleared = True
    if cleared:
        _evict_weight_cache_for(w1_fp4, w2_fp4)
    return cleared


def tile_expert_weights_inplace(
    w1_fp4: torch.Tensor, w2_fp4: torch.Tensor, *, w13_chunk: "int | None" = None
) -> None:
    """Re-lay the packed expert weights out tile-major IN PLACE (serving).

    The tensors keep their shapes ([E, rows, K/2] and [E, K, n/2] bytes);
    their bytes become the layouts _tile_expert_weights documents, and the
    tensors are marked (``_b12x_tile_major``) so _get_weight_views(tiled=True)
    views them without a second copy. One transient copy of each tensor
    (the layer's 0.9 + 0.45 GB per rank for GLM-5.3) at weight
    post-processing; a second call is a no-op. The marker carries a content
    fingerprint: invalidate_tile_major_if_reloaded (called by the wrapper at
    every process_weights_after_loading) clears it when a later load cycle
    overwrote the bytes.
    """
    chunk = _w13_tile_chunk(w13_chunk)
    kind = _tile_major_kind(chunk)
    have = getattr(w1_fp4, _TILE_MAJOR_ATTR, False)
    if have:
        if have != kind:
            raise ValueError(f"expert weights are already tile-major ({have}); wanted {kind}")
        return
    w13_t, w2_t = _tile_expert_weights(w1_fp4, w2_fp4, w13_chunk=chunk)
    w1_fp4.view(-1).copy_(w13_t.view(-1))
    w2_fp4.view(-1).copy_(w2_t.view(-1))
    del w13_t, w2_t
    setattr(w1_fp4, _TILE_MAJOR_ATTR, kind)
    setattr(w2_fp4, _TILE_MAJOR_ATTR, kind)
    # Content probe for invalidate_tile_major_if_reloaded: a later load
    # cycle that overwrites the bytes must not inherit the marker.
    setattr(w1_fp4, _TILE_MAJOR_FP_ATTR, _tile_major_fingerprint(w1_fp4))
    setattr(w2_fp4, _TILE_MAJOR_FP_ATTR, _tile_major_fingerprint(w2_fp4))


def _tile_expert_weights(
    w1_fp4: torch.Tensor, w2_fp4: torch.Tensor, *, w13_chunk: "int | None" = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tile-major copies of the packed expert weights (spec cell t).

    w1_fp4 [E, rows, K/2] bytes -> [E, K/c, rows, c/2] for the w13 chunk c
    (512 by origin): for one k tile the rows' c/2 B chunks are adjacent, so
    the kernel's (64 rows x 512 K) TMA box over c = 512 is one contiguous
    16 KB run instead of 64 chunks 2 KB apart. w2_fp4 [E, K, n/2] ->
    [E, n/128, K, 64] likewise for the (128 rows x 128 K) down box (8 KB).
    Bytes only, no arithmetic: the kernel reads exactly the bytes the
    row-major kernel reads, in the same order per tile.
    """
    if w1_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise TypeError("tiled expert weights: packed fp4 bytes (uint8) expected")
    e, rows, kb = w1_fp4.shape
    kin_b = _w13_tile_chunk(w13_chunk) // 2
    if kb % kin_b != 0:
        raise ValueError(f"tiled expert weights: K/2 = {kb} B is not a multiple of {kin_b}")
    w13_t = (
        w1_fp4.reshape(e, rows, kb // kin_b, kin_b).permute(0, 2, 1, 3).contiguous()
    )
    e2, hrows, nb = w2_fp4.shape
    kin2_b = TILED_W2_K_IN // 2
    if nb % kin2_b != 0:
        raise ValueError(f"tiled expert weights: n/2 = {nb} B is not a multiple of {kin2_b}")
    w2_t = (
        w2_fp4.reshape(e2, hrows, nb // kin2_b, kin2_b).permute(0, 2, 1, 3).contiguous()
    )
    return w13_t, w2_t


def _swizzle_tile_boxes(w13_t: torch.Tensor, w2_t: torch.Tensor, *, kind: str = "byte") -> Tuple[torch.Tensor, torch.Tensor]:
    """Cell z: the reform tile's B stages in the smem byte order, so one cp.async.bulk lands a stage.

    w13_t [E, K/256, N, 128 B] (the 256 chunk): a stage is 128 rows x 128 B under Swizzle<3,4,3>; w2_t
    [E, I/128, H, 64 B]: a stage is 256 rows x 64 B under Swizzle<2,4,3>. What the swizzle acts on is the
    question the diagnostics answer -- `kind`:
      byte    the swizzle on BYTE offsets, the TMA hardware's 128B / 64B modes: row r's 16 B chunk c lands at
              c ^ (r % 8) (FC1), c ^ ((r // 2) % 4) (FC2)
      nibble  the swizzle on fp4 ELEMENT offsets, as the DSL's composed layout is written: row r's 8 B unit u
              (16 per FC1 row, 8 per FC2 row) lands at u ^ ((2 r + (u >> 3)) % 8) (FC1), u ^ (r % 4) (FC2)
      plain   no permutation (the TMA writing linear rows)
    Every map is an involution per row, so a second application restores the storage; nothing inside an
    8 B unit moves (probes/b12x_reform_layout_print.py enumerates the layouts)."""
    if w13_t.dtype != torch.uint8 or w2_t.dtype != torch.uint8:
        raise TypeError("swizzled expert weights: tile-major fp4 bytes (uint8) expected")
    if kind not in ("byte", "nibble", "plain"):
        raise ValueError(f"unknown swizzle kind {kind!r}")
    e, kt, rows, kin_b = w13_t.shape
    if kin_b != 128 or rows % 128:
        raise ValueError("cell z needs the 256 w13 chunk (128 B rows) and 128-row FC1 stages")
    e2, kt2, hrows, kin2_b = w2_t.shape
    if kin2_b != 64 or hrows % 256:
        raise ValueError("cell z needs the 64 B w2 rows and 256-row FC2 stages")
    if kind == "plain":
        return w13_t.contiguous().clone(), w2_t.contiguous().clone()
    dev = w13_t.device
    if kind == "byte":
        r = torch.arange(128, device=dev)[:, None]
        c = torch.arange(8, device=dev)[None, :]
        src = c ^ (r % 8)                                            # [128, 8] 16 B chunks
        boxes = w13_t.reshape(e, kt, rows // 128, 128, 8, 16)
        w13_z = boxes.gather(4, src.view(1, 1, 1, 128, 8, 1).expand(e, kt, rows // 128, 128, 8, 16))
        r2 = torch.arange(256, device=dev)[:, None]
        c2 = torch.arange(4, device=dev)[None, :]
        src2 = c2 ^ ((r2 // 2) % 4)                                  # [256, 4]
        boxes2 = w2_t.reshape(e2, kt2, hrows // 256, 256, 4, 16)
        w2_z = boxes2.gather(4, src2.view(1, 1, 1, 256, 4, 1).expand(e2, kt2, hrows // 256, 256, 4, 16))
    else:
        r = torch.arange(128, device=dev)[:, None]
        u = torch.arange(16, device=dev)[None, :]
        src = u ^ ((2 * r + (u >> 3)) % 8)                           # [128, 16] 8 B units
        boxes = w13_t.reshape(e, kt, rows // 128, 128, 16, 8)
        w13_z = boxes.gather(4, src.view(1, 1, 1, 128, 16, 1).expand(e, kt, rows // 128, 128, 16, 8))
        r2 = torch.arange(256, device=dev)[:, None]
        u2 = torch.arange(8, device=dev)[None, :]
        src2 = u2 ^ (r2 % 4)                                         # [256, 8]
        boxes2 = w2_t.reshape(e2, kt2, hrows // 256, 256, 8, 8)
        w2_z = boxes2.gather(4, src2.view(1, 1, 1, 256, 8, 1).expand(e2, kt2, hrows // 256, 256, 8, 8))
    return w13_z.reshape(w13_t.shape).contiguous(), w2_z.reshape(w2_t.shape).contiguous()


def static_v2_weights_layout(**geometry) -> bool:
    """Whether the static lane reads tile-major expert weights (cell t) for
    this geometry -- the wrapper keys its cached weight views on it, so a lane
    switch (probe) rebuilds them and a tiled view never meets a row-major
    kernel."""
    cfg = _static_v2_config_for(**geometry)
    if cfg is None:
        return False
    return bool(cfg.get("tiled", False))


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    tiled: bool = False,
    sf_pack: bool = False,
    reform_sf_pack: bool = False,
    packed_only: bool = False,
    w13_chunk: "int | None" = None,
) -> _WeightViews:
    """Create permuted weight views for the static kernel.

    The kernel expects concatenated w13 data with shape [2*n, k//2, E]
    via a single TMA descriptor.

    tiled=True (spec cell t, moe_static_kernel_v5): the views are 4-D over a
    tile-major COPY of the packed weights -- w13 as [E, K/512, 2n, 256 B]
    and w2 as [E, n/128, K, 64 B] -- so every kernel TMA box is one
    contiguous run of memory. The copy is cached with the scale conversions
    and follows the source weights' lifetime. w13_chunk names the w13 chunk
    (512 above; the served chunk when None); in-place storage must carry it.
    """
    chunk = _w13_tile_chunk(w13_chunk) if tiled else TILED_W13_K_IN
    activation_precision = _normalize_activation_precision(activation_precision)
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    tile_n = _level_tile_n(activation_precision)
    # The kernel splits w13 into gate/up halves by tile index. This only works
    # when the boundary between halves lands on a tile-aligned column.
    if n % tile_n != 0:
        raise ValueError(
            f"intermediate_size ({n}) must be a multiple of {tile_n} "
            f"for the SM120 MoE kernel's gate/up tile split."
        )

    key = (
        activation_precision,
        quant_mode,
        bool(tiled),
        bool(sf_pack),
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
    )
    if reform_sf_pack:
        # In-place weight/scale updates must produce new packed storage; old
        # entries remain owned while the prior captured graph can use them.
        key += ("sf6-v1", tuple((id(t), _sf6_tensor_version(t)) for t in (
            w1_fp4, w1_blockscale, w1_alphas, w2_fp4, w2_blockscale, w2_alphas)))
    if tiled:
        key += ("w13_chunk", chunk)
    if packed_only and not (tiled and reform_sf_pack and not sf_pack):
        raise ValueError("packed-only scales require the tiled SF6 lane")
    # The final model owner keeps these views. Avoid a cache owning either
    # raw aliases or a second generation of the same packed-only layer.
    cached = None if packed_only else _WEIGHT_CACHE.get(key)
    if cached is None:
        if reform_sf_pack and _is_cuda_graph_capturing():
            raise RuntimeError("sf6 weight views must be prepared before CUDA graph capture")
        # Cache the fresh buffers (scale factors + fp32 alphas) -- and the
        # tile-major weight copies when the lane reads them.
        w1_rows = w1_fp4.shape[1]  # 2*n for gated, n for non-gated
        if not tiled:
            # Symmetric with the kind check below: the storage must not be
            # tile-major when the lane reads row-major. The in-place relayout
            # (serving cell t) rewrites the ONLY copy, so a later switch to a
            # row-major lane in this process (the _STATIC_V2_OVERRIDE probe
            # hook) would silently misread the bytes -- refuse it instead.
            if getattr(w1_fp4, _TILE_MAJOR_ATTR, False) or getattr(
                w2_fp4, _TILE_MAJOR_ATTR, False
            ):
                raise ValueError(
                    "row-major static lane over tile-major storage: these "
                    "weights were re-laid in place for a tiled cell (t/z) "
                    "and cannot serve a row-major kernel in this process"
                )
            tiled_storage = (None, None)
        elif getattr(w1_fp4, _TILE_MAJOR_ATTR, False):
            # served in place (tile_expert_weights_inplace): the bytes are
            # tile-major already -- reshape, no copy
            have = getattr(w1_fp4, _TILE_MAJOR_ATTR, False)
            if have != _tile_major_kind(chunk):
                raise ValueError(
                    f"tiled expert weights: storage is {have}, the lane wants {_tile_major_kind(chunk)}"
                )
            e_, rows_, kb_ = w1_fp4.shape
            e2_, hrows_, nb_ = w2_fp4.shape
            if not getattr(w2_fp4, _TILE_MAJOR_ATTR, False):
                raise ValueError("tiled expert weights: w13 is tile-major but w2 is not")
            tiled_storage = (
                w1_fp4.view(e_, kb_ // (chunk // 2), rows_, chunk // 2),
                w2_fp4.view(e2_, nb_ // (TILED_W2_K_IN // 2), hrows_, TILED_W2_K_IN // 2),
            )
        else:
            tiled_storage = _tile_expert_weights(w1_fp4, w2_fp4, w13_chunk=chunk)
        cached = (
            convert_sf_from_mma_layout(
                w1_blockscale,
                m=w1_rows,
                k=k,
                num_groups=w1_fp4.shape[0],
                sf_vec_size=sf_vec_size,
            ).contiguous(),
            convert_sf_from_mma_layout(
                w2_blockscale,
                m=k,
                k=n,
                num_groups=w2_fp4.shape[0],
                sf_vec_size=sf_vec_size,
            ).contiguous(),
            w1_alphas.contiguous().to(torch.float32),
            w2_alphas.contiguous().to(torch.float32),
            tiled_storage,
        )
        if reform_sf_pack:
            owner = _prepared_reform_scales(
                w1_blockscale, w2_blockscale, cached[0], cached[1],
                experts=w1_fp4.shape[0], n=n, k=k,
            )
            cached += (owner,)
        if not packed_only:
            _WEIGHT_CACHE[key] = cached
            _register_cache_eviction(
                _WEIGHT_CACHE, key, w1_fp4, w1_blockscale, w1_alphas,
                w2_fp4, w2_blockscale, w2_alphas,
            )
    w13_sf_contiguous, down_sf_contiguous, w1_alpha, w2_alpha, tiled_storage = cached[:5]
    reform_scales = cached[5] if reform_sf_pack else None
    packed_only = bool(packed_only and reform_scales is not None and reform_scales.enabled)
    if packed_only:
        # Retire this layer's older raw views only. Other live owners keep
        # their storage; unrelated layers' caches are never cleared.
        for old_key in tuple(_WEIGHT_CACHE):
            if old_key[4] == w1_fp4.data_ptr() and old_key[7] == w2_fp4.data_ptr():
                _WEIGHT_CACHE.pop(old_key, None)
        w13_sf_contiguous = down_sf_contiguous = None
    w13_tiled, w2_tiled = tiled_storage
    if tiled:
        # (rows, K_in x2, K_tiles, E) over the tile-major bytes: the x2 dtype's
        # innermost dim doubles to K_in fp4 elements, the other strides
        # (256 B / 64 B, K_tiles x that, the expert) stay byte strides
        w13 = w13_tiled.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0)
        down = w2_tiled.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0)
    else:
        w13 = w1_fp4.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        down = w2_fp4.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    return _WeightViews(
        w13_fp4=w13,
        down_fp4=down,
        tiled=bool(tiled),
        w13_chunk=chunk,
        packed_only=packed_only,
        sfb1_packed=(
            reform_scales.fc1 if reform_scales is not None and reform_scales.enabled
            else _packed_fc1_scales(w13_sf_contiguous, w1_fp4.shape[0]) if sf_pack else None
        ),
        sfb2_packed=reform_scales.fc2 if reform_scales is not None else None,
        reform_scales=reform_scales,
        w13_tiled_storage=w13_tiled,
        w2_tiled_storage=w2_tiled,
        sfb_w13_ptr=None if packed_only else make_ptr(
            sf_dtype,
            w13_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        sfb_down_ptr=None if packed_only else make_ptr(
            sf_dtype,
            down_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        w1_storage=w1_fp4,
        w1_scale_storage=w13_sf_contiguous,
        w2_storage=w2_fp4,
        w2_scale_storage=down_sf_contiguous,
        _w13_sf_storage=w13_sf_contiguous,
        _down_sf_storage=down_sf_contiguous,
    )


def prepare_packed_only_weight_views(
    *, w1_fp4, w1_blockscale, w2_fp4, w2_blockscale, w1_alphas, w2_alphas,
    n, k, num_experts, num_local_experts, num_topk, activation,
    swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=None,
    activation_precision="fp4", quant_mode="nvfp4",
) -> _WeightViews | None:
    """Prepare a final, immutable owner before releasing model raw scales.

    Decline incompatible dispatchers before touching weights. A failed
    lossless pack keeps both original planes. No partial raw release.
    """
    if (activation_precision != "fp4" or quant_mode != "nvfp4"
            or num_experts != num_local_experts
            or _FORCED_BACKEND in ("micro", "direct_micro")
            or _GLM53_B12X_FORCE_BACKEND in ("micro", "direct_micro")
            or _GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128):
        return None
    cfg = _static_v2_config_for(
        num_experts=num_experts, num_local_experts=num_local_experts,
        hidden_size=k, intermediate_size=n, num_topk=num_topk,
        activation=activation, swiglu_limit=swiglu_limit,
        activation_precision=activation_precision, quant_mode=quant_mode,
    )
    if not (cfg and cfg.get("tiled") and cfg.get("reform_sf_pack")):
        return None
    from .moe_dynamic_gated_sf6 import stock_contract_matches
    if not stock_contract_matches():
        logging.getLogger("flashinfer.b12x").warning(
            "[b12x sf6] keeping raw scales: dynamic helper source pin drifted")
        return None
    views = _get_weight_views(
        w1_fp4=w1_fp4, w1_blockscale=w1_blockscale,
        w2_fp4=w2_fp4, w2_blockscale=w2_blockscale,
        w1_alphas=w1_alphas, w2_alphas=w2_alphas, n=n, k=k,
        activation_precision=activation_precision, quant_mode=quant_mode,
        tiled=True, reform_sf_pack=True, packed_only=True,
    )
    return views if views.packed_only else None


def _scale_runtime_addresses(weights, *, direct_sf6: bool) -> tuple[int, int]:
    """Dead legacy pointer slots are backed by packed storage only in SF6.

    Direct kernels never construct raw descriptors from these arguments.
    A dispatcher regression therefore raises before reading freed scales.
    """
    if direct_sf6:
        owner = weights.reform_scales
        if owner is None or not owner.enabled:
            raise RuntimeError("direct SF6 kernel requires both prepared planes")
        return owner.fc1.data_ptr(), owner.fc2.data_ptr()
    if weights.packed_only:
        raise RuntimeError("packed-only SF6 owner cannot launch a raw-scale kernel")
    return weights._w13_sf_storage.data_ptr(), weights._down_sf_storage.data_ptr()


# ---------------------------------------------------------------------------
# Kernel compilation cache
#
# The three kernels below are compiled through the shared on-disk CuTe-DSL
# cache (#3874, #4029; docs/design_docs/cute_dsl_kernel_cache.md), so a fresh
# process JITLinks an exported ``.o`` instead of re-running the MLIR pipeline.
# The in-process dicts stay as the level-1 memoization the design describes.
# A module's key files must be the same for every kernel in it: see
# _cute_dsl_module.
# ---------------------------------------------------------------------------
_CUTE_DSL_MODULE = "st_b12x_moe"


def _kernel_source_files() -> Tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache.

    Every module contributing device code to the three kernels compiled here:
    the kernel bodies, the shared activation and FP4 device helpers, and the
    SM120 layout builders and block-scaled mainloop they are built from.
    """
    from flashinfer.cute_dsl import fp4_common
    from flashinfer.cute_dsl import utils as cute_dsl_utils
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x

    from ._moe_dynamic import gated as moe_dynamic_gated
    from ._moe_dynamic import generic as moe_dynamic_generic
    from . import (
        moe_activation,
        moe_dynamic_kernel,
        moe_micro_kernel,
        moe_static_kernel,
        moe_static_common,
        moe_static_kernel_v4,
        moe_static_kernel_v5,
        moe_dynamic_gated_tiled,
    )

    return (
        __file__,
        os.path.join(os.path.dirname(__file__), "../../runtime/cuda132.lock.json"),
        os.path.join(os.path.dirname(__file__), "fp4_quant.py"),
        os.path.join(os.path.dirname(__file__), "fp4_scale_search.py"),
        os.path.join(os.path.dirname(__file__), "moe_w4a16_fp4_helpers.py"),
        moe_activation.__file__,
        moe_static_kernel.__file__,
        moe_static_common.__file__,
        moe_static_kernel_v4.__file__,
        moe_static_kernel_v5.__file__,
        os.path.join(os.path.dirname(__file__), "moe_reform_sf_pack.py"),
        os.path.join(os.path.dirname(__file__), "moe_sf_pack.py"),
        moe_dynamic_gated_tiled.__file__,
        os.path.join(os.path.dirname(__file__), "moe_dynamic_gated_sf6.py"),
        # Hash the candidate without importing its pinned private helpers.
        os.path.join(os.path.dirname(__file__), "moe_dynamic_prefill.py"),
        os.path.join(os.path.dirname(__file__), "moe_dynamic_prefill_n128.py"),
        moe_micro_kernel.__file__,
        moe_dynamic_kernel.__file__,
        moe_dynamic_gated.__file__,
        moe_dynamic_generic.__file__,
        cute_dsl_utils.__file__,
        fp4_common.__file__,
        dense_blockscaled_gemm_sm120_b12x.__file__,
    )


_MODULE_NAMES: Dict[Tuple[str, ...], str] = {}


def _cute_dsl_module(key_files: Tuple[str, ...]) -> str:
    """The on-disk module of the kernels whose cache key is `key_files`: named by
    the files, in order, and by their contents.

    flashinfer wipes a module directory whose meta.json (the key files' hash)
    differs from the kernel it is building. The dynamic kernels add variant files
    to the static set, so while every family shared one module each build erased
    the others' artifacts: every boot recompiled the prefill and decode kernels
    (about 53 s a rank, 2026-09-15). Naming by the files alone left the same wipe
    between two trees: every node's /cache is shared by production's release and
    whatever tree a window boots beside it, and both built into `st_b12x_moe`.
    Qwen3.8's windows of 2026-09-18 and production's boots between them took
    turns: a window's boot at 17:35 logged "Invalidating stale CuTe-DSL module
    st_b12x_moe_sm121a_cute_dsl" over production's kernels, and production's
    next boot at 17:48 logged it over the window's, recompiled six static
    kernels (the first five back to back, about 9.5 s each) and opened its door
    after 150 s -- as at 17:09, against 105 s at 15:26 and 16:48
    (measurements/qwen38_boot_20260918). With the contents in the name, a tree
    finds its own artifacts whatever ran in between, and a module's meta can
    differ only across a DSL or arch change.

    A key file that cannot be read keeps the name without its contents; flashinfer
    then bypasses the disk cache for that kernel, as it did before.
    """
    key_files = tuple(key_files)
    name = _MODULE_NAMES.get(key_files)
    if name is not None:
        return name
    from flashinfer.jit.cute_dsl_core import _hash_source_files
    key = "\0".join(os.path.basename(path) for path in key_files)
    try:
        key += "\0" + _hash_source_files(key_files)
    except (OSError, TypeError):
        return f"{_CUTE_DSL_MODULE}_{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    name = _MODULE_NAMES[key_files] = f"{_CUTE_DSL_MODULE}_{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    return name


def _disk_kernel_name(prefix: str, cache_key: Tuple) -> str:
    """On-disk specialization name for an in-process kernel cache key.

    The name is the *sole* per-kernel cache key — the module ``meta.json``
    guards only module-wide facts (arch, DSL stack, source hashes) — so it has
    to be injective in every codegen parameter. It is therefore derived from
    the very tuple that keys the in-process cache: a readable shape prefix for
    humans browsing ``cached_ops/``, plus a digest of the exact tuple.

    The digest, rather than a formatted field list, is what makes the mapping
    injective: the keys contain floats and ``None`` (``swiglu_alpha`` /
    ``swiglu_beta`` / ``swiglu_limit``) whose textual forms would collide once
    sanitized into a filename (``1.5`` and ``-1.5`` both sanitize to ``1_5``).
    """
    digest = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _static_kernel_cache_key(
    *,
    activation_precision: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
) -> Tuple:
    """The static kernel's cache key: every parameter affecting its codegen.

    Single source of truth for both cache levels — the in-process dict and,
    through :func:`_disk_kernel_name`, the on-disk artifact name.
    """
    return (
        "static",
        activation_precision,
        quant_mode,
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    )


def _micro_kernel_cache_key(
    *,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    share_input_across_experts: bool,
    share_expert_scales: bool,
    single_token: bool,
    skip_zero_weight_expert_id: int | None,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
    scatter_fp32: bool = False,
    ep_direct_scatter: bool = False,
    shared_fc1_a: bool = False,
    ep_m16: bool = False,
) -> Tuple:
    """The micro kernel's cache key (see :func:`_static_kernel_cache_key`)."""
    key = (
        "micro",
        quant_mode,
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        single_token,
        skip_zero_weight_expert_id,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    )
    if scatter_fp32:
        key += ("glm53_ep_micro_scatter_fp32_v1",)
    if ep_direct_scatter:
        key += ("glm53_ep_micro_direct_scatter_v1",)
    if shared_fc1_a:
        key += ("glm53_ep_micro_shared_fc1_a_v1",)
    if ep_m16:
        key += ("glm53_ep_micro_m16_v1",)
    return key


def _dynamic_kernel_cache_key(
    *,
    activation_precision: str,
    quant_mode: str,
    E: int,
    k: int,
    n: int,
    num_topk: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
    share_input_across_experts: bool,
    prefill_reuse: bool = False,
    prefill_fc1_n128: bool = False,
    ep_local_prefill: bool = False,
    tiled: bool = False,
    reform_sf_pack: bool = False,
    tp_sf6_q0: bool = False,
) -> Tuple:
    """The dynamic kernel's cache key (see :func:`_static_kernel_cache_key`).

    Deliberately free of ``m`` / ``max_rows``: the dynamic kernel takes its
    runtime-shaped operands as pointers, so one artifact serves every batch
    size.
    """
    key = (
        "dynamic",
        activation_precision,
        quant_mode,
        E,
        k,
        n,
        num_topk,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        share_input_across_experts,
        bool(tiled),
    )
    # tiled is part of every key (stock too): the gated subclass and the
    # stock kernel must never share an artifact. Adding it renames stock
    # on-disk artifacts once (a one-time recompile) -- the suffixes below
    # keep the reuse lanes separately keyed on top of it.
    if ep_local_prefill:
        suffix = ("glm53_ep_prefill_local_fp32_v2",)
        if reform_sf_pack:
            suffix += ("glm53_ep_tiled_sf6_v1",)
        return key + suffix
    if prefill_fc1_n128:
        return key + ("glm53_prefill_fc1_n128_v1",)
    if reform_sf_pack:
        suffix = ("sf6_direct_prefill_v1",)
        if tp_sf6_q0:
            suffix += ("glm53_tp_sf6_q0_v1",)
        return key + suffix
    return key + ("glm53_prefill_reuse_v1",) if prefill_reuse else key


_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_static_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
):
    """Compile (or retrieve cached) the SM120 static MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 static compiler"
        )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    # Select tile size based on actual routed rows
    routed_rows = m * num_topk
    mma_tiler_mn = (128, 128)
    if activation_precision == "fp4" and num_topk > 1:
        mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n, resident_clusters=mac)

    scatter_fp32 = _glm_tp_scatter_fp32(
        state_E=state_E, weight_E=weight_E, k=k, n=n, num_topk=num_topk,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    # The native dense/shared expert uses one route per row. Its kernel body
    # already reads the token extent at runtime; bind that extent dynamically
    # so arbitrary prefill tails share the boot-prepared capacity artifact.
    # Routed experts and <=64-row decode retain their existing compilers.
    dynamic_m = (65 <= m <= 16384 and scatter_fp32
                 and (state_E, weight_E, k, n, num_topk) == (1, 1, 4096, 3072, 1))
    cache_key = _static_kernel_cache_key(
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        state_E=state_E,
        weight_E=weight_E,
        m=0 if dynamic_m else m,
        k=k,
        n=n,
        num_topk=num_topk,
        max_rows=max_rows,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    cache_key = (*cache_key, scatter_fp32, "runtime_dense_m_v1") if dynamic_m else (*cache_key, scatter_fp32)
    activation_scale_search = _activation_scale_search_for(
        state_E=state_E, weight_E=weight_E, k=k, n=n, num_topk=num_topk,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    if activation_scale_search:
        cache_key = (*cache_key, "activation_scale_search_v1", activation_scale_search)
    cached = _STATIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    kernel: Any = MoEStaticKernel(
        scatter_fp32=scatter_fp32,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=output_tile_count_n,
        activation_scale_search=activation_scale_search,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
    )

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n  # 2*n for gated, n for non-gated

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    # Build fake tensors for compilation
    token_extent = cute.sym_int32() if dynamic_m else m
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (token_extent, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (token_extent * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (token_extent * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (w1_rows, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32 if scatter_fp32 else a_dtype,
        (token_extent, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _cute_dsl_module(_kernel_source_files()),
        _disk_kernel_name(f"static_m{'dyn' if dynamic_m else m}_k{k}_n{n}_t{num_topk}_r{max_rows}", cache_key),
        lambda: cute.compile(
            kernel,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            active_expert_count_fake,
            weight_expert_ids_fake,
            global_to_local_expert_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _STATIC_KERNEL_CACHE[cache_key] = result
    return result


_STATIC_V2_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _static_v2_cache_key(config: dict, **fields) -> Tuple:
    """Cache key of a v2 static kernel: the stock static key plus its config."""
    cfg = (
        "static_v2",
        int(config["tile_m"]),
        int(config["fc1"]),
        int(config["fc2"]),
        int(config["a_rows"]),
        bool(config["stamps"]),
        bool(config.get("a_ring", False)),
        bool(config.get("skip_sf", False)),
        bool(config.get("skip_a", False)),
        bool(config.get("tiled", False)),
        bool(config.get("sf_pack", False)),
        bool(config.get("decode_reform", False)),
        bool(config.get("reform_sf_pack", False)),
        bool(config.get("sf6_separate", False)),
        bool(config.get("sf6_word_expand", False)),
        bool(config.get("sf6_fc2_word_expand", False)),
        bool(config.get("packed_activation_store", False)),
        bool(config.get("fc1_reuse_a", False)),
        bool(config.get("compact_staging", False)),
        bool(config.get("sf6_registers", False)),
        int(config.get("l2_prefetch", 0)),
        bool(config.get("l2_prefetch_fc1", True)),
        bool(config.get("bulk_b", False)),
        bool(config.get("sync_cleanup", False)),
    )
    if config.get("input_vec16", False):
        cfg += ("input_vec16_v1",)
    if config.get("input_reuse", 0):
        cfg += ("input_reuse_v1", int(config["input_reuse"]))
    if config.get("activation_scale_search", 0):
        cfg += ("activation_scale_search_v1", int(config["activation_scale_search"]))
    if config.get("fc2_scale_search", 0):
        cfg += ("fc2_scale_search_v1", int(config["fc2_scale_search"]))
    # Expanded output and register scatter never alias a served handle.
    if config.get("probe_route_scatter", False):
        cfg += ("probe_route_scatter_v1",)
    if config.get("probe_direct_scatter", False):
        cfg += ("probe_direct_scatter_v1",)
    if config.get("c2_direct_scatter", False):
        cfg += ("c2_direct_scatter_v1",)
    if config.get("c2_scatter_reuse", False):
        cfg += ("c2_scatter_reuse_v1",)
    if config.get("c2_fc2_prefetch", False):
        cfg += ("c2_fc2_prefetch_v1",)
    if config.get("scatter_vec4", False):
        cfg += ("scatter_vec4_v1",)
    if config.get("scatter_packed_load", False):
        cfg += ("scatter_packed_load_v1",)
    return cfg + _static_kernel_cache_key(**fields)


def _static_v2_decode_config(config: dict, m: int) -> dict:
    """Select the declared expert tile before capture, with a stable cache ABI.

    `batch` extends the C1 operand pipeline to the served K7/C2 shape.
    Expert occupancy, including counts beyond M16, remains device input to
    the kernel's existing tile loop. No route-dependent host dispatch.
    """
    if config.get("probe_route_scatter") or config.get("probe_direct_scatter"):
        if not (m in (7, 14, 21, 28) and config.get("tiled")
                and config.get("reform_sf_pack") and (m != 7 or config.get("decode_reform"))
                and not any(config.get(k) for k in ("split", "skip_a", "skip_sf", "even"))):
            raise ValueError("scatter probe requires packed t,r,sf6 at 7/14/21/28 tokens")
    # reform_every_static (probe config only): every static row count takes the
    # M16 reform tile, whose K256 FC1 box is one 256 w13 chunk; the t tile's
    # K512 box spans two of them.
    reform = bool(config.get("decode_reform", False)) and (
        1 <= m <= 8 or (config.get("batch_reform", False) and m == 16)
        or bool(config.get("reform_every_static", False)))
    reuse = int(config.get("input_reuse", 0))
    if reuse not in (0, 1, 2, 3, 4) or (reuse and not (reform and m in (8, 16) and config.get("input_vec16", True))):
        raise ValueError("input reuse requires the eight/sixteen-row vector-input reform cell")
    if reuse in (3, 4) and any(config.get(k) for k in ("even", "split", "probe_route_scatter")):
        raise ValueError("input reuse route preparation requires the ordinary resident scheduler")
    separate = (reform and bool(config.get("reform_sf_pack", False))
                and bool(config.get("sf6_separate", True)))
    word_expand = separate and bool(config.get("sf6_word_expand", True))
    fc2_word_expand = (reform and bool(config.get("reform_sf_pack", False))
                       and bool(config.get("sf6_fc2_word_expand", True)))
    packed_activation_store = reform and bool(config.get("packed_activation_store", True))
    fc1_reuse_a = (reform and bool(config.get("reform_sf_pack", False))
                   and bool(config.get("fc1_reuse_a", True)))
    compact_staging = (fc1_reuse_a and separate and int(config.get("fc1", 2)) % 2 == 0
                       and bool(config.get("compact_staging", True)))
    # A mixed-provenance checkpoint builds a companion lane with reform_sf_pack off beside every
    # sf6 lane, and at m=16 `batch` turns direct scatter on for it too. #1056 bought the boot by
    # refusing that companion the scatter; the kernel now takes it (the scatter reads the epilogue
    # tile, not the FC1 scales), so the companion keeps the cell instead of dropping to staged
    # output. `scatter_reuse` below still needs sf6_registers and stays off for it.
    direct_scatter = bool(reform and m == 16 and config.get("batch_reform")
                          and config.get("c2_direct_scatter", True))
    sf6_registers = compact_staging and bool(config.get("sf6_registers", True))
    scatter_reuse = bool(direct_scatter and sf6_registers and config.get("c2_scatter_reuse", True))
    # Both reform tiles, C1 rows and the C2 batch M16 tile, have one FC1 half
    # and one CTA; their FC2 and scatter reads all follow the FC1 publication.
    sync_cleanup = (reform and bool(config.get("reform_sf_pack", False))
                    and bool(config.get("sync_cleanup", True)))
    # Staged output owns contiguous eight-column spans. The direct-register
    # path owns pairs in different spans and keeps its existing v2 RED.
    scatter_vec4 = (reform and bool(config.get("reform_sf_pack", False))
                    and not direct_scatter and not config.get("probe_direct_scatter", False)
                    and not config.get("probe_route_scatter", False)
                    and bool(config.get("scatter_vec4", True)))
    return dict(config, decode_reform=reform, sf6_separate=separate, sf6_word_expand=word_expand,
                input_vec16=bool(reform and m in (8, 16) and config.get("input_vec16", True)),
                sf6_fc2_word_expand=fc2_word_expand,
                packed_activation_store=packed_activation_store, fc1_reuse_a=fc1_reuse_a,
                compact_staging=compact_staging,
                c2_direct_scatter=direct_scatter,
                c2_scatter_reuse=scatter_reuse,
                c2_fc2_prefetch=bool(scatter_reuse and int(config.get("fc2", 2)) == 2
                                     and config.get("c2_fc2_prefetch", True)),
                sf6_registers=sf6_registers, sync_cleanup=sync_cleanup, scatter_vec4=scatter_vec4,
                scatter_packed_load=bool(scatter_vec4 and config.get("scatter_packed_load", True)))


# Numerically qualified K7 input reuse: C1 caches the quantized token, C2 fans
# it out from registers. Whole-MoE gains are small/cache-dependent; adopted
# with the fixed-K bundle. Explicit input_reuse=0 is the same-build control.
INPUT_REUSE_DEFAULTS = {8: 3, 16: 4}


def _static_v2_input_reuse_config(config: dict, state_E: int, weight_E: int,
                                  m: int, k: int, n: int, num_topk: int,
                                  max_rows: int) -> dict:
    if "input_reuse" in config:
        return config
    # Other model geometries and private schedulers retain their own cell.
    if ((state_E, weight_E, k, n, num_topk) == (288, 288, 4096, 512, 8)
            and m in (8, 16) and max_rows >= m and config.get("decode_reform")
            and config.get("tiled") and config.get("reform_sf_pack")
            and config.get("input_vec16")
            and not any(config.get(key) for key in
                        ("even", "split", "probe_route_scatter", "probe_direct_scatter"))):
        return dict(config, input_reuse=INPUT_REUSE_DEFAULTS[m])
    return config


def _get_static_kernel_v2(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    config: dict,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    w13_chunk: "int | None" = None,
):
    """Compile (or retrieve cached) the decode-streaming static MoE kernel.

    Same fake-tensor contract as :func:`_get_static_kernel` plus the stamps
    tensor ([mac, STAMP_SLOTS] int64) the kernel writes when
    ``config["stamps"]`` is set (and ignores otherwise). w13_chunk is the
    tile-major w13 chunk of the views it will read (the served one if None).
    """
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision != "fp4":
        raise ValueError("static v2 is the NVFP4 lane")
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    if sf_vec_size != 16:
        raise ValueError("static v2 is the NVFP4 (sf_vec_size=16) lane")
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )
    # The explicit batch recipe extends the same tile to K7/C2. All SF6
    # launches read the same packed scales; other shapes keep the t tile.
    config = _static_v2_decode_config(config, m)
    config = _static_v2_input_reuse_config(config, state_E, weight_E, m, k, n,
                                          num_topk, max_rows)
    if config.get("input_reuse", 0):
        cache_bytes = 0 if config["input_reuse"] == 4 else m * (k // 2 + k // 16)
        if config["input_reuse"] in (3, 4):
            cache_bytes += m * num_topk * 8
        spare_bytes = (state_E - m * num_topk) * max_rows * (k // 2)
        if ((state_E, weight_E, k, n, num_topk) != (288, 288, 4096, 512, 8)
                or max_rows < m or cache_bytes > spare_bytes):
            raise ValueError("input reuse cache must fit beyond every reachable compact expert plane")
    reform = config["decode_reform"]
    mma_tiler_mn = (16 if reform else int(config["tile_m"]), 128)
    cache_key = _static_v2_cache_key(
        config,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        state_E=state_E,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        max_rows=max_rows,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    scatter_fp32 = _glm_tp_scatter_fp32(
        state_E=state_E,weight_E=weight_E,k=k,n=n,num_topk=num_topk,
        quant_mode=quant_mode,activation=activation,swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,swiglu_limit=swiglu_limit)
    if (config.get("probe_route_scatter") or config.get("probe_direct_scatter")
            or config.get("c2_direct_scatter")) and not (
            scatter_fp32 and state_E == weight_E == 288 and k == 4096
            and n == 512 and num_topk == 8):
        raise ValueError("scatter probe requires the GLM TP4 FP32 output contract")
    cache_key = (*cache_key,"tp_scatter_fp32_v1",scatter_fp32)
    tiled = bool(config.get("tiled", False))
    chunk = _w13_tile_chunk(w13_chunk) if tiled else TILED_W13_K_IN
    if chunk != TILED_W13_K_IN:
        # The 512 chunk keeps its original key and name (its on-disk handles).
        cache_key = (*cache_key, "w13_chunk", chunk)
    cached = _STATIC_V2_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    l2_prefetch = int(config.get("l2_prefetch", 0))
    bulk_b = bool(config.get("bulk_b", False))
    if bulk_b and (not tiled or not reform or chunk != 256):
        raise ValueError("z needs t,r over the 256 w13 chunk (its boxes are the reform tile's own stages)")
    if l2_prefetch and (not tiled or not reform or chunk != 256):
        # the reform's FC1 box is (128 rows x K256); over the 256 chunk it is one contiguous 16 KB run,
        # over 512 it is half of every row's chunk -- no run to prefetch as one request
        raise ValueError("l<n> needs t,r over the 256 w13 chunk (the FC1 box is then one contiguous run)")
    kernel_cls = MoEStaticKernelV5 if tiled else MoEStaticKernelV4
    kernel: Any = kernel_cls(
        scatter_fp32=scatter_fp32,
        route_scatter=bool(config.get("probe_route_scatter", False)),
        direct_scatter=bool(config.get("probe_direct_scatter") or config.get("c2_direct_scatter")),
        scatter_reuse=bool(config["c2_scatter_reuse"]),
        fc2_prefetch=bool(config["c2_fc2_prefetch"]),
        a_ring=bool(config.get("a_ring", False)),
        sf_pack=bool(config.get("sf_pack", False)),
        decode_reform=reform,
        reform_sf_pack=bool(config.get("reform_sf_pack", False)),
        sf6_separate=bool(config["sf6_separate"]),
        sf6_word_expand=bool(config["sf6_word_expand"]),
        sf6_fc2_word_expand=bool(config["sf6_fc2_word_expand"]),
        packed_activation_store=bool(config["packed_activation_store"]),
        fc1_reuse_a=bool(config["fc1_reuse_a"]),
        compact_staging=bool(config["compact_staging"]),
        sf6_registers=bool(config["sf6_registers"]),
        sync_cleanup=bool(config["sync_cleanup"]),
        scatter_vec4=bool(config["scatter_vec4"]),
        scatter_packed_load=bool(config["scatter_packed_load"]),
        sf_vec_size=sf_vec_size,
        output_tile_count_n=output_tile_count_n,
        fc1_stages=int(config["fc1"]),
        fc2_stages=int(config["fc2"]),
        l2_prefetch=l2_prefetch,
        l2_prefetch_fc1=bool(config.get("l2_prefetch_fc1", True)),
        bulk_b=bulk_b,
        input_vec16=bool(config.get("input_vec16", False)),
        input_reuse=int(config.get("input_reuse", 0)),
        activation_scale_search=int(config.get("activation_scale_search", 0)),
        fc2_scale_search=int(config.get("activation_scale_search", 0) or config.get("fc2_scale_search", 0)),
        stamps=bool(config["stamps"]),
        skip_sf=bool(config.get("skip_sf", False)),
        skip_a=bool(config.get("skip_a", False)),
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
    )

    w1_rows = 2 * n
    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    if tiled:
        # tile-major storage (moe_static_kernel_v5): (rows, K_in, K/K_in, E),
        # K_in the stride-1 mode, then the rows -- one contiguous chunk per
        # (k tile, row), rows adjacent; the runtime view is the same 4-D
        # permutation of the re-laid-out bytes (_get_weight_views(tiled=True))
        if k % chunk != 0 or n % TILED_W2_K_IN != 0:
            raise ValueError(
                f"tiled expert weights need K % {chunk} == 0 and "
                f"I_tp % {TILED_W2_K_IN} == 0 (got K={k}, I_tp={n})"
            )
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (w1_rows, chunk, k // chunk, weight_E),
            stride_order=(1, 0, 2, 3), assumed_align=16,
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (k, TILED_W2_K_IN, n // TILED_W2_K_IN, weight_E),
            stride_order=(1, 0, 2, 3), assumed_align=16,
        )
    else:
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (w1_rows, k, weight_E), stride_order=(1, 0, 2), assumed_align=16
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16
        )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32 if scatter_fp32 else a_dtype,
        (m * num_topk * output_tile_count_n if config.get("probe_route_scatter") else m, k),
        stride_order=(1, 0), assumed_align=16
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16
    )
    stamps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (mac, _STATIC_V2_STAMP_SLOTS), stride_order=(1, 0), assumed_align=8
    )
    next_item_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    # cell q: the packed FC1 scales, (E, blocks per expert, stage bytes) u8; off
    # the lane a 16 B dummy the kernel never reads.
    if config.get("reform_sf_pack"):
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (weight_E, (n * 2 // 128) * (k // 256), REFORM_SF_STAGE),
            stride_order=(2, 1, 0), assumed_align=16,
        )
    elif config.get("sf_pack"):
        _sf_blocks = (n * 2 // 128) * (k // TILED_W13_K_IN)
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (weight_E, _sf_blocks, SF_STAGE_BYTES),
            # row-major, the torch tensor's own order: without this the fake is
            # compact in the OTHER direction (strides (1, E, ...)) and the call
            # fails with "Mismatched sfb1_packed.strides[0]"
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
    else:
        sfb1_packed_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (1, 1, 16), stride_order=(2, 1, 0), assumed_align=16
        )
    sfb2_packed_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (weight_E, (k // 256) * (n // 128), REFORM_SF_STAGE)
        if config.get("reform_sf_pack") else (1, 1, 16),
        stride_order=(2, 1, 0), assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    name = (
        f"static2_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}_tm{config['tile_m']}"
        f"f{config['fc1']}g{config['fc2']}a{config['a_rows']}"
        f"{'s' if config['stamps'] else ''}{'d' if config.get('dynamic') else ''}"
        f"{'w' if config.get('wide') else ''}{'e' if config.get('even') else ''}"
        f"{'k' if config.get('split') else ''}{'u' if config.get('v4') else ''}"
        f"{'v' if config.get('a_ring') else ''}{'t' if config.get('tiled') else ''}"
        f"{'q' if config.get('sf_pack') else ''}"
        f"{'r16n128k256d256' if reform else ''}"
        f"{'sf6v1' if config.get('reform_sf_pack') else ''}"
        f"{'fc1sep' if config.get('sf6_separate') else ''}"
        f"{'word' if config.get('sf6_word_expand') else ''}"
        f"{'fc2word' if config.get('sf6_fc2_word_expand') else ''}"
        f"{'a2u64' if config.get('packed_activation_store') else ''}"
        f"{'a1reuse' if config.get('fc1_reuse_a') else ''}"
        f"{'compact' if config.get('compact_staging') else ''}"
        f"{'sfregs' if config.get('sf6_registers') else ''}"
        f"{'c2scatter' if config.get('c2_direct_scatter') else ''}"
        f"{'reuse' if config.get('c2_scatter_reuse') else ''}"
        f"{'prefetch3' if config.get('c2_fc2_prefetch') else ''}"
        f"{'sync' if config.get('sync_cleanup') else ''}"
        f"{'inputv16' if config.get('input_vec16') else ''}"
        f"{('inputreuse' + str(config['input_reuse'])) if config.get('input_reuse') else ''}"
        f"{'xs' if config.get('skip_sf') else ''}{'xa' if config.get('skip_a') else ''}"
        f"{'' if chunk == TILED_W13_K_IN else f'c{chunk}'}"
    )
    compiled = build_and_load_cute_dsl_kernel(
        _cute_dsl_module(_kernel_source_files()),
        _disk_kernel_name(name, cache_key),
        lambda: cute.compile(
            kernel,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            active_expert_count_fake,
            weight_expert_ids_fake,
            global_to_local_expert_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            stamps_fake,
            next_item_fake,
            sfb1_packed_fake,
            sfb2_packed_fake,
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _STATIC_V2_KERNEL_CACHE[cache_key] = result
    # The serving proof line (22차/28차 lesson: "armed" is not "serving"): the
    # first launch of this shape in a process builds or loads the v2 kernel
    # here, so this line in a worker log means the served wrapper took the
    # v2 lane for that shape. The cached-kernel path is silent otherwise.
    logging.getLogger("flashinfer.b12x").warning(
        "[b12x static v2] lane serving: %s (mac=%d, m=%d, routed=%d, smem=%d B)",
        name, mac, m, m * num_topk, getattr(kernel, "smem_bytes", 0),
    )
    return result


_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _glm_tp_scatter_shape(state_E, weight_E, k, n, num_topk, cell=None):
    """The admitted routed cell (this rank's experts) and the dense/shared MLP served through the E=1 lane."""
    cell = _admitted_moe() if cell is None else cell
    shapes = ((cell.experts_local, cell.hidden, cell.inter_local, cell.topk),
              (1, cell.hidden, cell.dense_inter_local, 1))
    # An EP eager prefill expands local (token, route) pairs into one route per
    # row. It is the same expert cell, including decode-sized compact tails.
    if getattr(cell, "experts", cell.experts_local) > cell.experts_local:
        shapes += ((cell.experts_local, cell.hidden, cell.inter_local, 1),)
    return state_E == weight_E and (weight_E, k, n, num_topk) in shapes


def _bound_ep_prefill_fp32(*, E, k, n, num_topk, quant_mode, activation,
                          swiglu_alpha, swiglu_beta, swiglu_limit, tiled):
    """Row-major bound EP experts, both full routes and compact local pairs."""
    cell = _admitted_moe()
    return (not tiled and getattr(cell, "experts", cell.experts_local) > cell.experts_local
            and E == cell.experts_local and _glm_tp_scatter_fp32(
                state_E=E, weight_E=E, k=k, n=n, num_topk=num_topk,
                quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit, cell=cell))


def _glm_tp_scatter_fp32(*, state_E, weight_E, k, n, num_topk, quant_mode,
                          activation, swiglu_alpha, swiglu_beta, swiglu_limit, cell=None):
    """The fixed TP4 lane sums rounded route partials in FP32 (the admitted cell's)."""
    cell = _admitted_moe() if cell is None else cell
    return (_glm_tp_scatter_shape(state_E, weight_E, k, n, num_topk, cell)
            and quant_mode == cell.quant and activation == cell.activation
            and (swiglu_alpha, swiglu_beta, swiglu_limit) == (1., 0., cell.swiglu_limit))


def _glm_tp_scatter_buffer(workspace, output):
    plane = workspace.glm_tp_scatter_fp32
    if (plane is None or plane.shape[0] < output.shape[0] or plane.shape[1] != output.shape[1]
            or plane.dtype != torch.float32 or plane.device != output.device):
        raise ValueError("GLM TP FP32 scatter workspace was not allocated before launch")
    plane.record_stream(torch.cuda.current_stream(output.device))
    return plane[:output.shape[0]]


def _ep_micro_scatter_fp32(*, state_E, weight_E, m, k, n, num_topk,
                          max_rows, skip_zero_weight_expert_id, quant_mode,
                          activation, swiglu_alpha, swiglu_beta, swiglu_limit):
    """Only the two fixed GLM EP decode calls change their accumulation ABI."""
    return (state_E == weight_E == 72 and (m, k, n) == (8, 4096, 2048)
            and quant_mode == "nvfp4" and activation == "swigluoai_uninterleave"
            and (swiglu_alpha, swiglu_beta, swiglu_limit) == (1., 0., 10.)
            and (num_topk, max_rows, skip_zero_weight_expert_id)
            in ((1, 8, None), (8, 64, 72)))


def _ep_micro_direct_scatter(*, state_E, weight_E, m, k, n, num_topk,
                             max_rows, skip_zero_weight_expert_id, quant_mode,
                             activation, swiglu_alpha, swiglu_beta, swiglu_limit,
                             mma_tiler_mn, share_input_across_experts,
                             share_expert_scales, single_token):
    """Only the padded GLM EP top8 call scatters directly from FC2 registers."""
    return (_ep_micro_scatter_fp32(
                state_E=state_E, weight_E=weight_E, m=m, k=k, n=n,
                num_topk=num_topk, max_rows=max_rows,
                skip_zero_weight_expert_id=skip_zero_weight_expert_id,
                quant_mode=quant_mode, activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit)
            and (num_topk, max_rows, skip_zero_weight_expert_id) == (8, 64, 72)
            and mma_tiler_mn in ((16, 128), (32, 128))
            and not share_input_across_experts and not share_expert_scales
            and not single_token)


def _ep_micro_scatter_buffer(workspace, output):
    """Use the preallocated plane; capture can never replace its storage."""
    current = workspace.ep_micro_scatter_fp32
    if (output.dtype != torch.bfloat16 or tuple(output.shape) != (8, 4096)
            or not output.is_contiguous() or output.device != workspace.device):
        raise ValueError("EP micro FP32 scatter requires contiguous BF16 [8,4096]")
    if (current is None or current.dtype != torch.float32
            or tuple(current.shape) != (8, 4096) or not current.is_contiguous()
            or current.device != output.device):
        raise ValueError("EP micro FP32 scatter workspace was not pinned before launch")
    # The same workspace is serialized across layers, including captured
    # launches. Record side-stream use without allocating or changing its ptr.
    current.record_stream(torch.cuda.current_stream(output.device))
    return current


def _validate_ep_micro_short_output(
    *, workspace, weights, a, topk_ids, topk_weights, physical_output,
    target, num_experts, num_tokens, k, n, top_k, quant_mode,
    activation, swiglu_alpha, swiglu_beta, swiglu_limit, forced_backend,
):
    """Admit a host-only six-row final cast; the kernel still writes eight rows."""
    if (not _B12X_EP_ZERO_WEIGHT_MICRO or forced_backend is not None
            or bool(getattr(weights, "tiled", False)) or top_k != 8
            or not _ep_micro_scatter_fp32(
                state_E=workspace.state_E, weight_E=num_experts,
                m=num_tokens, k=k, n=n, num_topk=top_k,
                max_rows=workspace.max_rows, skip_zero_weight_expert_id=72,
                quant_mode=quant_mode, activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit)
            or tuple(a.shape) != (8, 4096)
            or tuple(topk_ids.shape) != (8, 8)
            or tuple(topk_weights.shape) != (8, 8)):
        raise ValueError("direct T6 output requires the exact padded EP top8 micro lane")
    plane = workspace.ep_micro_scatter_fp32
    if (plane is None or plane.dtype != torch.float32
            or tuple(plane.shape) != (8, 4096) or not plane.is_contiguous()):
        raise ValueError("direct T6 output requires the pinned FP32 M8 plane")
    if (target.dtype != torch.bfloat16 or tuple(target.shape) != (6, 4096)
            or not target.is_contiguous() or target.device.type != "cuda"
            or target.device != workspace.device
            or physical_output.dtype != torch.bfloat16
            or tuple(physical_output.shape) != (8, 4096)
            or not physical_output.is_contiguous()):
        raise ValueError("direct T6 output requires contiguous BF16 [6,4096]")
    start = target.data_ptr()
    end = start + target.numel() * target.element_size()
    # A destination cannot share any prepared input, legacy output or FP32
    # plane. No allocation, data read, stream switch or persistent tensor cache.
    for tensor in (a, topk_ids, topk_weights, physical_output, plane):
        if tensor.device != target.device or not tensor.is_contiguous():
            raise ValueError("direct T6 output device/layout differs")
        other = tensor.data_ptr()
        if start < other + tensor.numel() * tensor.element_size() and other < end:
            raise ValueError("direct T6 output aliases prepared or scatter storage")


def _get_micro_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    skip_zero_weight_expert_id: int | None = None,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    quant_mode: str = "nvfp4",
):
    """Compile (or retrieve cached) the SM120 micro MoE kernel."""
    quant_mode = _normalize_quant_mode(quant_mode)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    mma_tiler_mn = _select_micro_mma_tiler_mn(
        state_E=state_E, weight_E=weight_E, m=m, k=k, n=n,
        num_topk=num_topk, skip_zero_weight_expert_id=skip_zero_weight_expert_id,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit,
        max_rows=max_rows, share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales, single_token=single_token,
    )
    scatter_fp32 = _ep_micro_scatter_fp32(
        state_E=state_E, weight_E=weight_E, m=m, k=k, n=n,
        num_topk=num_topk, max_rows=max_rows,
        skip_zero_weight_expert_id=skip_zero_weight_expert_id,
        quant_mode=quant_mode, activation=activation,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    scatter_fp32 = scatter_fp32 or _glm_tp_scatter_fp32(
        state_E=state_E, weight_E=weight_E, k=k, n=n, num_topk=num_topk,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    ep_direct_scatter = _ep_micro_direct_scatter(
        state_E=state_E, weight_E=weight_E, m=m, k=k, n=n,
        num_topk=num_topk, max_rows=max_rows,
        skip_zero_weight_expert_id=skip_zero_weight_expert_id,
        quant_mode=quant_mode, activation=activation,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit, mma_tiler_mn=mma_tiler_mn,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales, single_token=single_token,
    )
    # Reuse gate/up input loads only in this exact direct-scatter EP lane.
    shared_fc1_a = ep_direct_scatter
    ep_m16 = ep_direct_scatter and mma_tiler_mn == (16, 128)

    cache_key = _micro_kernel_cache_key(
        quant_mode=quant_mode,
        state_E=state_E,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        max_rows=max_rows,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        skip_zero_weight_expert_id=skip_zero_weight_expert_id,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        scatter_fp32=scatter_fp32,
        ep_direct_scatter=ep_direct_scatter,
        shared_fc1_a=shared_fc1_a,
        ep_m16=ep_m16,
    )
    activation_scale_search = _activation_scale_search_for(
        state_E=state_E, weight_E=weight_E, k=k, n=n, num_topk=num_topk,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    if activation_scale_search:
        cache_key = (*cache_key, "activation_scale_search_v1", activation_scale_search)
    cached = _MICRO_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEMicroKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        activation_scale_search=activation_scale_search,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        skip_zero_weight_expert_id=skip_zero_weight_expert_id,
        scatter_fp32=scatter_fp32,
        ep_direct_scatter=ep_direct_scatter,
        shared_fc1_a=shared_fc1_a,
        ep_m16=ep_m16,
    )

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    # Build fake tensors for compilation (identical to static kernel)
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (w1_rows, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32 if scatter_fp32 else a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _cute_dsl_module(_kernel_source_files()),
        _disk_kernel_name(f"micro_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}", cache_key),
        lambda: cute.compile(
            kernel,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            active_expert_count_fake,
            weight_expert_ids_fake,
            global_to_local_expert_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _MICRO_KERNEL_CACHE[cache_key] = result
    return result


# The launch cache skips the per-launch build/configure; the kernel cache
# dedupes compiles across keys that configure to the same artifact
# (m=2..8 differ only in grid_x).
_DIRECT_MICRO_LAUNCH_CACHE: Dict[Tuple, Tuple] = {}
_DIRECT_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}
# The direct micro kernel goes through the same on-disk CuTe-DSL cache as the
# static/dynamic families (2026-09-12): its own module directory, keyed by the
# sources that contribute device code to it and named by their hash, so two
# trees sharing a /cache keep their own (_cute_dsl_module says why).
_DIRECT_MICRO_MODULE = "st_b12x_direct_micro"


def _direct_micro_source_files() -> Tuple[str, ...]:
    from flashinfer.cute_dsl import fp4_common
    from flashinfer.cute_dsl import utils as cute_dsl_utils

    from . import moe_activation, moe_direct_micro_kernel

    return (
        __file__,
        moe_direct_micro_kernel.__file__,
        moe_activation.__file__,
        fp4_common.__file__,
        cute_dsl_utils.__file__,
    )


def _write_json_atomic(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _build_direct_micro_on_disk(kernel, prefix: str, compile_key: Tuple, topk_ids_dtype) -> Tuple[Any, bool]:
    """(compiled, accepts_block_dim) for one direct micro specialization, through
    flashinfer's on-disk CuTe-DSL cache.

    The block-dim verdict (register pressure may cap the launchable CTA below the
    fused body's 512 threads) needs the in-process compiled object -- its kernel
    name and CUDA library handle -- which a ``.o`` reloaded from disk does not
    carry. So the verdict is persisted beside the object as a sidecar when the
    kernel is built, and read back on a cache hit; an object without its sidecar
    is rebuilt rather than served with a guessed verdict.
    """
    from flashinfer.jit.cute_dsl_core import (
        JitSpecCuteDsl,
        _hash_source_files,
        cute_dsl_cache_disabled,
    )

    verdict: Dict[str, bool] = {}

    def compile_fn():
        compiled = compile_direct_micro_kernel(kernel, topk_ids_dtype=topk_ids_dtype, tvm_ffi=True)
        verdict["accepts"] = compiled_direct_micro_accepts_block_dim(compiled, kernel.launch_block_dim)
        return compiled

    if cute_dsl_cache_disabled():
        compiled = compile_fn()
        return compiled, verdict["accepts"]
    try:
        source_sha256 = _hash_source_files(tuple(_direct_micro_source_files()))
    except (OSError, TypeError):
        compiled = compile_fn()
        return compiled, verdict["accepts"]
    spec = JitSpecCuteDsl(f"{_DIRECT_MICRO_MODULE}_{source_sha256[:12]}", _disk_kernel_name(prefix, compile_key),
                          compile_fn, source_sha256)
    sidecar = spec.module_dir / f"{spec.kernel_name}.blockdim.json"
    if spec.object_path.exists() and not sidecar.exists():
        spec.object_path.unlink()
    compiled = spec.build_and_load()
    if "accepts" in verdict:                       # built in this process: persist the verdict beside the .o
        _write_json_atomic(sidecar, {"block_dim": int(kernel.launch_block_dim), "accepts": bool(verdict["accepts"])})
        return compiled, verdict["accepts"]
    try:
        data = json.loads(sidecar.read_text())
        accepts = bool(data["accepts"]) if int(data["block_dim"]) == int(kernel.launch_block_dim) else None
    except (OSError, ValueError, KeyError, TypeError):
        accepts = None
    if accepts is None:                              # unreadable or stale sidecar: rebuild once, never guess
        spec.object_path.unlink(missing_ok=True)
        compiled = spec.build_and_load()
        _write_json_atomic(sidecar, {"block_dim": int(kernel.launch_block_dim), "accepts": bool(verdict["accepts"])})
        return compiled, verdict["accepts"]
    return compiled, accepts


def _get_direct_micro_kernel(
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    fast_math: bool = True,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    device: torch.device | None = None,
):
    """Compile (or retrieve cached) the SM120 direct micro MoE kernel.

    Returns (compiled, grid_x, accepts_block_dim).
    """
    if activation != SWIGLUOAI_UNINTERLEAVE:
        # The kernel constructor only accepts configurable swiglu parameters
        # for swigluoai; other activations use its normalized defaults
        # (accept-and-ignore, matching the MMA kernels).
        swiglu_alpha = None
        swiglu_beta = None
        swiglu_limit = None
    launch_key = (
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        str(_canonical_cuda_device(device)) if device is not None else None,
    )
    cached = _DIRECT_MICRO_LAUNCH_CACHE.get(launch_key)
    if cached is not None:
        return cached
    kernel = build_direct_micro_kernel(
        weight_E,
        m,
        k,
        n,
        num_topk,
        activation=activation,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=m == 1,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        device=device,
    )
    # the kernel's __cache_key__ is a property on the CuTe-DSL kernel classes (moe_w4a16_kernel reads it bare); the
    # first fleet request with M=3 tokens took this path and died calling the tuple (45차 §23)
    cache_key = kernel.__cache_key__
    compile_key = ("direct_micro", cache_key() if callable(cache_key) else cache_key, topk_ids_dtype)
    entry = _DIRECT_MICRO_KERNEL_CACHE.get(compile_key)
    if entry is None:
        # Register pressure can cap the launchable CTA below the fused body's
        # 512 threads; the verdict travels with the on-disk kernel.
        entry = _build_direct_micro_on_disk(
            kernel, f"direct_micro_E{weight_E}_m{m}_k{k}_n{n}_t{num_topk}", compile_key, topk_ids_dtype)
        _DIRECT_MICRO_KERNEL_CACHE[compile_key] = entry
    compiled, accepts = entry
    cached = (compiled, kernel.grid_x, accepts)
    _DIRECT_MICRO_LAUNCH_CACHE[launch_key] = cached
    return cached


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def _expand_to_experts(t: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Broadcast a scalar or [1] tensor to [num_experts], always fp32.

    Both branches must cast: the kernels are compiled against fp32 fake
    tensors for every per-expert scale.
    """
    if t.numel() == 1:
        return t.to(torch.float32).expand(num_experts).contiguous()
    return t.contiguous().to(torch.float32)


def launch_sm120_static_moe(
    *,
    workspace: Sm120StaticMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    _ep_short_output: torch.Tensor | None = None,
    _output_finalize=None,
) -> torch.Tensor:
    """Launch the SM120 static, micro, or direct micro MoE kernel.

    The direct micro kernel takes tiny decode batches (m <= 8, routed_rows
    < 64) when it supports the shape, the MMA micro kernel takes the rest of
    its band (routed_rows <= 20-40), and the static kernel takes the rest.
    The MMA micro path runs a Triton pre-pass to compact routing IDs before
    launching; direct micro routes on global expert ids directly.
    """
    _check_memref_limit("scatter_output", scatter_output.numel())
    activation_precision = _normalize_activation_precision(activation_precision)
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 static launcher"
        )
    forced_backend = _FORCED_BACKEND
    if forced_backend is None and _GLM53_B12X_FORCE_BACKEND is not None:
        forced_backend = _effective_glm53_forced_backend(
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_local_experts=workspace.state_E,
            hidden_size=k,
            intermediate_size=n,
            num_topk=top_k,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )

    if _ep_short_output is not None:
        _validate_ep_micro_short_output(
            workspace=workspace, weights=weights, a=a,
            topk_ids=topk_ids, topk_weights=topk_weights,
            physical_output=scatter_output, target=_ep_short_output,
            num_experts=num_experts, num_tokens=num_tokens, k=k, n=n, top_k=top_k,
            quant_mode=quant_mode, activation=activation,
            swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit, forced_backend=forced_backend,
        )

    if _output_finalize is not None:
        from engine.kernels.moe_output import validate_finalizer
        validate_finalizer(_output_finalize, rows=num_tokens, experts=num_experts,
                           local_experts=workspace.state_E, hidden=k, intermediate=n,
                           topk=top_k, quant_mode=quant_mode, activation=activation,
                           limit=swiglu_limit, alpha=swiglu_alpha, beta=swiglu_beta,
                           tiled=bool(getattr(weights, 'tiled', False)))
        if _ep_short_output is not None or forced_backend not in (None, 'static'):
            raise ValueError('MoE finalizer cannot use an EP output or another forced backend')

    # Flatten routing tensors
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)
    routed_rows = num_tokens * top_k

    # Capture whether input_gs was a single shared scalar BEFORE expansion:
    # the m=1 relu2 shared-input micro optimization only applies when every
    # expert sees the same FC1-input global scale.
    input_gs_is_shared = input_gs.numel() == 1
    down_input_scale_is_shared = down_input_scale.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    # Shared-scale flags let compact W4A4 micro match the ReLU2 single-token
    # specialization.
    share_input_across_experts = (
        activation == "relu2"
        and num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    share_expert_scales = (
        activation == "relu2" and input_gs_is_shared and down_input_scale_is_shared
    )

    # Direct micro takes its band before the MMA micro decision. It reads
    # weights by global expert id, so EP shapes keep the compact path.
    weights_tiled = bool(getattr(weights, "tiled", False))
    if weights_tiled and forced_backend in ("micro", "direct_micro"):
        raise ValueError(
            f"forced {forced_backend} backend reads row-major expert weights; the "
            "static lane serves tile-major weights (STK_moe_static cell t)"
        )
    use_direct_micro = (
        not weights_tiled
        and quant_mode == "nvfp4"
        and workspace.state_E == num_experts
        and workspace.dm_barrier_count is not None
        and workspace.dm_barrier_count.numel() >= routed_rows + num_tokens * 16
        and num_tokens <= _MICRO_MAX_TOKENS
        and routed_rows < _DIRECT_MICRO_CUTOVER_PAIRS
        and n <= _DIRECT_MICRO_MAX_N
        and MoEDirectMicroKernel.is_supported(num_tokens, k, n, top_k, num_experts)
    )
    if forced_backend is not None:
        if forced_backend == "direct_micro":
            if quant_mode != "nvfp4":
                raise ValueError(
                    "forced direct_micro backend only supports quant_mode=nvfp4"
                )
            if workspace.dm_barrier_count is None or not (
                MoEDirectMicroKernel.is_supported(num_tokens, k, n, top_k, num_experts)
            ):
                raise ValueError(
                    "forced direct_micro backend cannot run this shape "
                    f"(m={num_tokens}, k={k}, n={n}, top_k={top_k})"
                )
            if workspace.dm_barrier_count.numel() < routed_rows + num_tokens * 16:
                raise ValueError(
                    "forced direct_micro backend exceeds the workspace barrier "
                    f"capacity ({workspace.dm_barrier_count.numel()} slots < "
                    f"{routed_rows} routed rows + {num_tokens * 16})"
                )
            use_direct_micro = True
        else:
            use_direct_micro = False
    if use_direct_micro:
        compiled, grid_x, block_ok = _get_direct_micro_kernel(
            num_experts,
            num_tokens,
            k,
            n,
            top_k,
            topk_ids_dtype=flat_ids.dtype,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            device=a.device,
        )
        if not block_ok:
            if forced_backend == "direct_micro":
                raise RuntimeError("compiled direct micro MoE kernel cannot launch")
            use_direct_micro = False
    if use_direct_micro:
        # MMA fp4_common quantizers divide by the dequantization scale;
        # direct micro's local helpers multiply by its reciprocal. Normalize
        # into persistent planes (zeros stay zero). Reciprocal-form callers
        # already supply direct micro's multiplier.
        if not input_scales_are_reciprocal:
            workspace.dm_input_gs.copy_(
                torch.where(input_gs != 0, 1.0 / input_gs, input_gs)
            )
            workspace.dm_down_input_scale.copy_(
                torch.where(
                    down_input_scale != 0, 1.0 / down_input_scale, down_input_scale
                )
            )
            launch_gs = workspace.dm_input_gs
            launch_down = workspace.dm_down_input_scale
        else:
            launch_gs = input_gs
            launch_down = down_input_scale
        MoEDirectMicroKernel.launch(
            compiled,
            x=a,
            w1_fp4=weights.w1_storage,
            w1_blockscale=weights.w1_scale_storage,
            w1_alphas=weights.w1_alpha,
            a1_gscale=launch_gs,
            a2_gscale=launch_down,
            inter_fp32=workspace.dm_intermediate,
            w2_fp4=weights.w2_storage,
            w2_blockscale=weights.w2_scale_storage,
            w2_alphas=weights.w2_alpha,
            topk_ids=flat_ids,
            topk_weights=flat_weights,
            out=scatter_output,
            barrier_count=workspace.dm_barrier_count,
            barrier_epoch=workspace.dm_barrier_epoch,
            m=num_tokens,
            grid_x=grid_x,
            tvm_ffi=True,
        )
        return scatter_output

    # Decide micro vs static
    micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS
    if top_k > 1:
        micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK
    skip_zero_weight_expert_id = _b12x_ep_zero_weight_micro_expert_id(
        enabled=_B12X_EP_ZERO_WEIGHT_MICRO,
        state_E=workspace.state_E,
        weight_E=num_experts,
        num_tokens=num_tokens,
        k=k,
        n=n,
        num_topk=top_k,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        forced_backend=forced_backend,
    )
    use_micro = (
        (
            activation_precision == "fp4"
            and num_tokens <= _MICRO_MAX_TOKENS
            and routed_rows <= micro_cutover
        ) or skip_zero_weight_expert_id is not None
    ) and not weights_tiled   # the micro kernels read row-major weights
    if forced_backend is not None:
        if forced_backend == "micro":
            # Forced mode raises on correctness violations, never falls back.
            if num_tokens > _MICRO_MAX_TOKENS:
                raise ValueError(
                    f"forced micro backend supports at most {_MICRO_MAX_TOKENS} "
                    f"tokens (got {num_tokens})"
                )
            if flat_ids.numel() > workspace.compact_topk_ids.numel():
                raise ValueError(
                    "forced micro backend exceeds the workspace compact-id "
                    f"capacity ({workspace.compact_topk_ids.numel()} < "
                    f"{flat_ids.numel()})"
                )
            use_micro = True
        else:
            use_micro = False

    sm_count = get_num_sm(torch.device("cuda"))
    base_mac = min(get_max_active_clusters(1), sm_count)
    static_mac_ladder = _STATIC_MAC_LADDER
    if _GLM53_B12X_STATIC_MAC_LADDER is not None:
        static_mac_ladder = _effective_glm53_mac_ladder(
            static_mac_ladder,
            _GLM53_B12X_STATIC_MAC_LADDER,
            num_experts=num_experts,
            num_local_experts=workspace.state_E,
            hidden_size=k,
            intermediate_size=n,
            num_topk=top_k,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
    tuned_static_mac = _lookup_mac_ladder(static_mac_ladder, routed_rows)
    static_mac = min(tuned_static_mac or base_mac, base_mac)
    if activation_precision == "fp4" and not use_micro and routed_rows < 40:
        static_mac = min(static_mac, 64)
    # set only when the v2 static kernel launches (it takes two extra tensors)
    static_v2_stamps = None
    static_v2_counter = None
    kernel_scatter_output = scatter_output
    glm_tp_fp32 = _glm_tp_scatter_fp32(
        state_E=workspace.state_E, weight_E=num_experts, k=k, n=n, num_topk=top_k,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    if glm_tp_fp32:
        kernel_scatter_output = _glm_tp_scatter_buffer(workspace, scatter_output)

    if use_micro:
        if _ep_micro_scatter_fp32(
            state_E=workspace.state_E, weight_E=num_experts, m=num_tokens,
            k=k, n=n, num_topk=top_k, max_rows=workspace.max_rows,
            skip_zero_weight_expert_id=skip_zero_weight_expert_id,
            quant_mode=quant_mode, activation=activation,
            swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        ):
            kernel_scatter_output = _ep_micro_scatter_buffer(workspace, scatter_output)
        assert flat_ids.numel() <= workspace.compact_topk_ids.numel(), (
            f"compact_topk_ids buffer too small: "
            f"{workspace.compact_topk_ids.numel()} < {flat_ids.numel()}"
        )
        if skip_zero_weight_expert_id is not None:
            # triton_compact writes one local->weight entry per unique routed
            # id. The exact gate proves unique <= routed_rows <= state_E; keep
            # an executable guard here so workspace drift fails before launch.
            if routed_rows > workspace.weight_expert_ids.numel():
                raise RuntimeError(
                    "zero-weight micro compact map is too small: "
                    f"{workspace.weight_expert_ids.numel()} < {routed_rows}"
                )
        # Single-token ReLU2 is non-gated, so the micro kernel can launch on
        # the routed expert ids directly. Gated SiLU still goes through the
        # compact id buffer so the kernel can map compact launch ids back to
        # the physical gate/up weight experts.
        if num_tokens == 1 and activation == "relu2":
            launch_ids = flat_ids
        elif num_tokens == 1:
            compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            compact_ids.copy_(
                torch.arange(
                    flat_ids.numel(),
                    device=flat_ids.device,
                    dtype=torch.int32,
                )
            )
            workspace.weight_expert_ids[: flat_ids.numel()].copy_(
                flat_ids.to(torch.int32)
            )
            workspace.active_expert_count.fill_(flat_ids.numel())
            launch_ids = compact_ids
        else:
            compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            from .triton_compact import compact_topk_ids as _triton_compact_topk_ids

            _triton_compact_topk_ids(
                flat_ids,
                compact_ids,
                workspace.weight_expert_ids,
                workspace.active_expert_count,
            )
            launch_ids = compact_ids
        # Select micro MAC: min of tuned ladder, work tiles, and hardware limit.
        micro_mac_ladder = _MICRO_MAC_LADDER
        if _GLM53_B12X_MICRO_MAC_LADDER is not None:
            micro_mac_ladder = _effective_glm53_mac_ladder(
                micro_mac_ladder,
                _GLM53_B12X_MICRO_MAC_LADDER,
                num_experts=num_experts,
                num_local_experts=workspace.state_E,
                hidden_size=k,
                intermediate_size=n,
                num_topk=top_k,
                quant_mode=quant_mode,
                activation=activation,
                swiglu_limit=swiglu_limit,
            )
        micro_mac = _select_micro_mac(routed_rows, n, base_mac, micro_mac_ladder)
        compiled, mac = _get_micro_kernel(
            workspace.state_E,
            num_experts,
            num_tokens,
            k,
            n,
            top_k,
            workspace.max_rows,
            topk_ids_dtype=launch_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=num_tokens == 1,
            skip_zero_weight_expert_id=skip_zero_weight_expert_id,
            mac_override=micro_mac,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            quant_mode=quant_mode,
        )
    else:
        static_v2_config = _static_v2_config_for(
            num_experts=num_experts,
            num_local_experts=workspace.state_E,
            hidden_size=k,
            intermediate_size=n,
            num_topk=top_k,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
        )
        want_tiled = bool(static_v2_config is not None and static_v2_config.get("tiled"))
        if bool(getattr(weights, "tiled", False)) != want_tiled:
            raise RuntimeError(
                "tiled expert weights and the tiled static lane (spec cell t) must "
                f"agree: views tiled={bool(getattr(weights, 'tiled', False))}, "
                f"lane tiled={want_tiled}"
            )
        if static_v2_config is not None:
            static_v2_config = _static_v2_decode_config(static_v2_config, num_tokens)
            if static_v2_config.get("reform_sf_pack"):
                if weights.reform_scales is None:
                    raise RuntimeError("sf6 layer has no prepared immutable scale owner")
                if not weights.reform_scales.enabled:
                    static_v2_config = dict(static_v2_config, reform_sf_pack=False)
            if bool(static_v2_config.get("bulk_b")) != bool(getattr(weights, "swizzled", False)):
                raise RuntimeError(
                    "cell z and pre-swizzled expert storage must agree: lane bulk_b="
                    f"{bool(static_v2_config.get('bulk_b'))}, views swizzled={bool(getattr(weights, 'swizzled', False))}")
            compiled, mac = _get_static_kernel_v2(
                workspace.state_E,
                num_experts,
                num_tokens,
                k,
                n,
                top_k,
                workspace.max_rows,
                config=static_v2_config,
                topk_ids_dtype=torch.int32,
                input_scales_are_reciprocal=input_scales_are_reciprocal,
                fast_math=fast_math,
                mac_override=static_mac,
                activation=activation,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                activation_precision=activation_precision,
                quant_mode=quant_mode,
                w13_chunk=getattr(weights, "w13_chunk", None),
            )
            if static_v2_config.get("probe_route_scatter") and not getattr(compiled, "owns_route_scatter", False):
                raise RuntimeError("route-scatter probe requires its prewarmed output owner and reduction")
            static_v2_stamps = _static_v2_stamps_tensor(mac, a.device)
            static_v2_counter = _static_v2_counter_tensor(a.device)
        else:
            compiled, mac = _get_static_kernel(
                workspace.state_E,
                num_experts,
                num_tokens,
                k,
                n,
                top_k,
                workspace.max_rows,
                topk_ids_dtype=torch.int32,
                input_scales_are_reciprocal=input_scales_are_reciprocal,
                fast_math=fast_math,
                mac_override=static_mac,
                activation=activation,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                activation_precision=activation_precision,
                quant_mode=quant_mode,
            )
        launch_ids = flat_ids

    # Pointer arguments must be passed as raw ints (data_ptr()) at runtime.
    # No stream argument: the kernels compile against
    # ``make_fake_stream(use_tvm_ffi_env_stream=True)``, so TVM-FFI supplies
    # the caller's current stream and the parameter is absent from the
    # compiled signature.
    sf1_address, sf2_address = _scale_runtime_addresses(
        weights, direct_sf6=bool(static_v2_stamps is not None
                                and static_v2_config.get("reform_sf_pack")))
    runtime_args: Tuple[Any, ...] = (
        a,
        launch_ids,
        flat_weights,
        workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        weights.w13_fp4,
        sf1_address,
        weights.down_fp4,
        sf2_address,
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        kernel_scatter_output,
        workspace.token_map,
        workspace.token_weights,
    )
    if static_v2_stamps is not None:
        runtime_args = runtime_args + (
            static_v2_stamps,
            static_v2_counter,
            weights.sfb1_packed if (static_v2_config.get("sf_pack")
                                   or static_v2_config.get("reform_sf_pack"))
            and weights.sfb1_packed is not None else _sf_pack_dummy(a.device),
            weights.sfb2_packed if static_v2_config.get("reform_sf_pack")
            and weights.sfb2_packed is not None else _sf_pack_dummy(a.device),
        )
    if (_ep_short_output is not None
            and kernel_scatter_output is not workspace.ep_micro_scatter_fp32):
        raise RuntimeError("direct T6 output lost its FP32 kernel target")
    if _output_finalize is not None and (kernel_scatter_output.dtype != torch.float32
            or kernel_scatter_output is scatter_output):
        raise RuntimeError('MoE finalizer lost its separate FP32 scatter owner')
    compiled(*runtime_args)
    if _output_finalize is not None:
        # The callback consumes the borrowed accumulator on this stream before
        # another MoE launch may reuse it. No BF16 output tensor is written.
        return _output_finalize(kernel_scatter_output)
    if _ep_short_output is not None:
        # Keep the physical M8 kernel, zeroing, source address and stream.
        # BF16(FP32[:6]) is exactly the previous BF16(FP32)[:6]; the omitted
        # padded BF16 tensor was only copied, never used in arithmetic.
        _ep_short_output.copy_(kernel_scatter_output[:6])
        return _ep_short_output
    if kernel_scatter_output is not scatter_output:
        # All rounded per-route partials are accumulated before this single
        # conversion. The copy follows the kernel on the caller's stream and
        # is recorded with the same pinned source address during CUDA capture.
        scatter_output.copy_(kernel_scatter_output)

    return scatter_output


# ==========================================================================
# Dynamic backend
# ==========================================================================


def select_sm120_moe_backend(
    *,
    num_tokens: int,
    num_topk: int,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    num_experts: int | None = None,
    num_local_experts: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    activation: str | None = None,
    swiglu_limit: float | None = None,
) -> str:
    """Pick static or dynamic backend based on routed-pair count."""
    mode = _normalize_quant_mode(quant_mode, activation_precision)
    if mode == "w4a16":
        return "w4a16"
    forced_backend = _FORCED_BACKEND
    if (
        forced_backend is None
        and _GLM53_B12X_FORCE_BACKEND is not None
        and None not in (
            num_experts,
            num_local_experts,
            hidden_size,
            intermediate_size,
            activation,
        )
    ):
        forced_backend = _effective_glm53_forced_backend(
            num_tokens=num_tokens,
            num_experts=int(num_experts),
            num_local_experts=int(num_local_experts),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            num_topk=num_topk,
            quant_mode=mode,
            activation=str(activation),
            swiglu_limit=swiglu_limit,
        )
    if forced_backend == "dynamic":
        return "dynamic"
    if forced_backend in ("static", "micro", "direct_micro"):
        # Both micro variants launch through the static workspace path.
        return "static"
    # NVIDIA's first three dense GLM MLPs split FC2 over 24 slices. Their
    # BF16 atomic sum varies between replays; the static family supplies the
    # persistent FP32 sum plane for every batch size. The generic dynamic
    # backend still has BF16 scatter and cannot serve this dense contract.
    if (num_experts == num_local_experts == 1
            and (hidden_size, intermediate_size, num_topk) == (4096, 3072, 1)
            and mode == "nvfp4" and activation == "swigluoai_uninterleave"
            and swiglu_limit == 10.):
        return "static"
    # An expert-local eager prefill -- one route a row over this rank's whole experts (num_topk 1, every expert
    # local) with more rows than a decode step -- runs the dynamic prefill kernel, whose artifact is free of the row
    # count. Below the cutover the static kernel is keyed by rows, capacity and MAC rung: Qwen3.8's first fleet boot
    # (2026-09-18) compiled 75 of them on one rank for a 67-token prompt and served no token. Decode-sized launches
    # (<= _MICRO_MAX_TOKENS rows) keep the static/micro path, whose artifacts are bounded.
    if (num_topk == 1 and num_experts is not None and num_local_experts is not None
            and num_experts == num_local_experts > 1 and num_tokens > _MICRO_MAX_TOKENS):
        return "dynamic"
    routed_rows = num_tokens * num_topk
    cutover = _get_static_compact_cutover_pairs("fp4")
    if _GLM53_B12X_STATIC_CUTOVER_PAIRS is not None:
        cutover = _effective_glm53_static_cutover(
            cutover,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_topk=num_topk,
            quant_mode=mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
    if routed_rows <= cutover:
        return "static"
    return "dynamic"


# ---------------------------------------------------------------------------
# Dynamic workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120DynamicMoEWorkspace:
    """Scratch buffers for one SM120 dynamic MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation_precision: str
    quant_mode: str

    # Core buffers
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor

    # Dynamic-specific
    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    # The M-tile the geometry above was sized for; launches must build the
    # kernel with the same tile.
    tile_m: int = _LEVEL_TILE_M
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    pair_head: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_expert: torch.Tensor
    task_valid_rows: torch.Tensor

    # Views
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None
    # One grow-only accumulator shared by layers using this cached workspace.
    # The kernel zeroes it before routing; it must not be retained per layer.
    ep_scatter_fp32: torch.Tensor | None = None


def _dynamic_task_geometry(
    state_E: int,
    n: int,
    routed_rows: int,
    *,
    tile_m: int = _LEVEL_TILE_M,
    tile_n: int = _LEVEL_TILE_N,
):
    """Compute task queue dimensions from problem geometry.

    Each active expert can introduce at most one additional physical tile
    beyond the base count (due to per-expert tail padding). The task queue
    holds one entry per (m_tile, slice_group) pair — NOT multiplied by E.
    """
    routed_rows = max(1, routed_rows)
    base_m_tiles = _align_up(routed_rows, tile_m) // tile_m
    active_expert_upper_bound = min(state_E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + tile_n - 1) // tile_n)
    slice_groups = max(
        1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


# --- capacity bounds, host-side ------------------------------------------
# `launch_sm120_dynamic_moe` hands the kernel three CAPACITIES -- max_rows,
# physical_tiles_capacity * tile_m, task_capacity -- and the kernel indexes
# with them. A workspace sized for a different shape than this call means the
# kernel walks off its own buffers, and the only symptom is
# cudaErrorIllegalAddress, reported asynchronously at whichever kernel runs
# NEXT. On Qwen3.8-Flash-Next that was the hyper-connection combine, which had
# nothing to do with it.
#
# Host-side arithmetic on integers already in hand: no device sync, no
# allocation. Always on: the DENEB_B12X_BOUNDS=0 escape restored the illegal
# address and was never set anywhere.
_B12X_BOUNDS = True


def _check_dynamic_capacity(workspace, *, routed_rows: int, n: int,
                            where: str) -> None:
    if not _B12X_BOUNDS:
        return
    want_tiles, _, want_tasks = _dynamic_task_geometry(
        workspace.state_E, n, routed_rows,
        tile_m=workspace.tile_m,
        tile_n=_level_tile_n(workspace.activation_precision),
    )
    want_rows = want_tiles * workspace.tile_m
    short = []
    if routed_rows > workspace.routed_rows_capacity:
        short.append(f"routed rows {routed_rows} > capacity "
                     f"{workspace.routed_rows_capacity}")
    if want_rows > workspace.max_rows:
        short.append(f"padded rows {want_rows} > max_rows {workspace.max_rows}")
    if want_tiles > workspace.physical_tiles_capacity:
        short.append(f"tiles {want_tiles} > physical_tiles_capacity "
                     f"{workspace.physical_tiles_capacity}")
    if want_tasks > workspace.task_capacity:
        short.append(f"tasks {want_tasks} > task_capacity "
                     f"{workspace.task_capacity}")
    _b12x_bounds_note(workspace, routed_rows, n, want_rows, want_tiles,
                      want_tasks)
    if short:
        raise ValueError(
            f"b12x dynamic MoE workspace too small [{where}]: "
            + "; ".join(short)
            + f". state_E={workspace.state_E} n={n} "
              f"tile_m={workspace.tile_m} routed_rows={routed_rows}.")


_B12X_NOTED = set()


def _b12x_bounds_note(ws, routed_rows, n, want_rows, want_tiles, want_tasks):
    """One line per distinct shape, so a passing check is still evidence."""
    key = (ws.state_E, n, ws.tile_m, routed_rows)
    if key in _B12X_NOTED or len(_B12X_NOTED) > 24:
        return
    _B12X_NOTED.add(key)
    # stderr, not a logger: this file lives in flashinfer's namespace and the
    # host process's logging config does not necessarily adopt it. A check
    # whose passing leaves no trace is a check nobody can cite.
    import sys as _sys

    _sys.stderr.write(
        f"[b12x-bounds] state_E={ws.state_E} n={n} tile_m={ws.tile_m} "
        f"routed_rows={routed_rows} -> rows {want_rows}/{ws.max_rows} "
        f"tiles {want_tiles}/{ws.physical_tiles_capacity} "
        f"tasks {want_tasks}/{ws.task_capacity}\n")
    _sys.stderr.flush()


def allocate_sm120_dynamic_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    tile_m: int | None = None,
    _prefill_tile64: bool = False,
) -> Sm120DynamicMoEWorkspace:
    """Allocate workspace buffers for the SM120 dynamic MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_dynamic_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    tile_m = _select_dynamic_tile_m(routed_rows, state_E, activation) if tile_m is None else tile_m
    if tile_m not in (16, 32, 64, 128):
        raise ValueError("unsupported dynamic workspace tile M")
    physical_tiles, _, max_tasks = _dynamic_task_geometry(
        state_E,
        n,
        routed_rows,
        tile_m=tile_m,
        tile_n=_level_tile_n(activation_precision),
    )
    rows_padded = physical_tiles * tile_m
    # The kernel addresses activation scales in 128-row SF atoms regardless of
    # tile_m, so the scale plane must cover the last partial atom.
    scale_rows = _align_up(rows_padded, 128)
    if type(_prefill_tile64) is not bool:
        raise TypeError('private prefill tile64 allocation override must be bool')
    if _prefill_tile64:
        if ((state_E, weight_E, k, n, num_topk, tile_m) != (288, 288, 4096, 512, 8, 64)
                or quant_mode != 'nvfp4' or activation != 'swigluoai_uninterleave'):
            raise ValueError('private M64 scale allocation requires exact native geometry')
        # One complete physical SFA atom per logical M64 tile. Different
        # experts never share an atom; the lower 64 rows hold their scales.
        scale_rows = physical_tiles * 128
    cols_pad_k = _align_up(k // sf_vec_size, 4)
    _check_memref_limit("dynamic packed_input", rows_padded * (k // 2))
    _check_memref_limit("dynamic packed_input_scale", scale_rows * cols_pad_k)
    packed_input = torch.empty(1, rows_padded, k // 2, dtype=torch.uint8, device=device)

    workspace = Sm120DynamicMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=rows_padded,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=physical_tiles,
        task_capacity=max_tasks,
        tile_m=tile_m,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(rows_padded, dtype=torch.float32, device=device),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            scale_rows, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
    )

    # Finalize views
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# ---------------------------------------------------------------------------
# Dynamic kernel compilation
# ---------------------------------------------------------------------------


class _DynamicMoELaunch:
    """Thin JIT wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(
        self,
        kernel,
        k,
        num_topk,
        activation_precision: str = "fp4",
        sf_vec_size: int = _NVFP4_BLOCK_SIZE,
        reform_sf_pack: bool = False,
        prefill_tile64: bool = False,
        prefill_packets: bool = False,
    ):
        activation_precision = _normalize_activation_precision(activation_precision)
        if activation_precision == "bf16":
            raise ValueError(
                "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher wrapper"
            )
        self._kernel = kernel
        self._k = k
        self._packed_storage_cols = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = _align_up(k // sf_vec_size, 4)
        self._reform_sf_pack = bool(reform_sf_pack)
        self._prefill_tile64 = bool(prefill_tile64)
        self._prefill_packets = bool(prefill_packets)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_expert_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        sfb1_packed: cute.Tensor,
        sfb2_packed: cute.Tensor,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        packet_stride: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        input_stride = self._k
        if cutlass.const_expr(self._prefill_packets):
            # The packet-only Q0 reader treats this layout field as the byte
            # distance between source-rank packets, never as BF16 row storage.
            # Carry it at runtime so v1/v2 and ragged chunks share one kernel.
            input_stride = packet_stride
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(input_stride, 1)))
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout(
                (rows_padded * self._packed_storage_cols,), stride=(1,)
            ),
        )
        # Activation scales live in 128-row SF atoms; the plane is allocated
        # through the last partial atom even when rows_padded is not aligned.
        scale_rows = ((rows_padded + 127) // 128) * 128
        if cutlass.const_expr(self._prefill_tile64):
            # Each compact 64-row tile owns a complete physical SFA atom.
            # Its logical view must cover the allocation, including zeroing.
            scale_rows = rows_padded * 2
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((scale_rows * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        packed_args = ()
        if cutlass.const_expr(self._reform_sf_pack):
            packed_args = (sfb1_packed, sfb2_packed)
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
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
            token_weights_t,
            *packed_args,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _long_prefill_sf6_word_unpack(*, m, E, k, n, num_topk, tile_m,
                                 quant_mode, tiled, reform_sf_pack,
                                 activation, swiglu_alpha, swiglu_beta,
                                 swiglu_limit, ep_local, tp_sf6_q0,
                                 share_input_across_experts):
    return (type(m) is int and 8192 < m <= 32768
            and (E, k, n, num_topk, tile_m) == (288, 4096, 512, 8, 128)
            and quant_mode == 'nvfp4' and tiled and reform_sf_pack
            and activation == 'swigluoai_uninterleave'
            and (swiglu_alpha, swiglu_beta, swiglu_limit) == (1.0, 0.0, 10.0)
            and not ep_local and not tp_sf6_q0 and not share_input_across_experts)


def _short_prefill_q0_word_unpack(*, m, tp_sf6_q0, reform_sf_pack, ep_local):
    # The existing Q0 selector has already checked exact GLM geometry and math.
    # Keep small decode and the raw-scale subclass on their established source.
    return (type(m) is int and 64 < m <= 8192 and tp_sf6_q0
            and reform_sf_pack and not ep_local)


def _prefill_scale_expansion_eligible(*, m, E, k, n, num_topk, tile_m,
                                     quant_mode, tiled, activation,
                                     swiglu_alpha, swiglu_beta, swiglu_limit,
                                     share_input_across_experts):
    return (type(m) is int and 64 < m <= 32768
            and (E, k, n, num_topk, tile_m) == (288, 4096, 512, 8, 128)
            and quant_mode == 'nvfp4' and tiled and not share_input_across_experts
            and (activation, swiglu_alpha, swiglu_beta, swiglu_limit)
                == ('swigluoai_uninterleave', 1., 0., 10.))


def _prefill_m64_eligible(*, m, E, k, n, num_topk, tile_m, quant_mode,
                        tiled, reform_sf_pack, activation, swiglu_alpha,
                        swiglu_beta, swiglu_limit, share_input_across_experts):
    return (type(m) is int and 64 < m <= 8192 and tile_m == 64 and reform_sf_pack
            and _prefill_scale_expansion_eligible(
                m=m, E=E, k=k, n=n, num_topk=num_topk, tile_m=128,
                quant_mode=quant_mode, tiled=tiled, activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=share_input_across_experts))


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    share_input_across_experts: bool = False,
    tile_m: int = _LEVEL_TILE_M,
    quant_mode: str = "nvfp4",
    tiled: bool = False,
    reform_sf_pack: bool = False,
    _tp_sf6_q0_override: bool | None = None,
    _prefill_scale_expansion: bool = False,
    _prefill_tile64: bool = False,
    _prefill_n128: bool = False,
    _prefill_q0_batch8: bool = False,
    _prefill_packets: bool = False,
    w13_chunk: "int | None" = None,
):
    """Compile (or retrieve cached) the SM120 dynamic MoE kernel.

    tiled=True: the expert weights are tile-major (static v2 cell t,
    moe_static_kernel_v5) and arrive as 4-D tensors; the gated kernel's
    subclass MoEGatedDynamicKernelTiled groups them (the stock file stays
    untouched: #368 pins its hash). w13_chunk is their w13 chunk (served if None).
    """
    chunk = _w13_tile_chunk(w13_chunk) if tiled else TILED_W13_K_IN
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic compiler"
        )
    # Both dynamic implementations reserve 32 route slots per token for the
    # shared-input fast path. Larger top-k values remain correct by using the
    # generic per-route producer instead.
    share_input_across_experts = bool(
        share_input_across_experts
        and activation_precision == "fp4"
        and num_topk <= _MAX_SHARED_INPUT_TOPK
    )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    if reform_sf_pack and not (tiled and quant_mode == "nvfp4"
                              and is_gated_activation(activation)):
        raise ValueError("dynamic SF6 requires gated tiled NVFP4 weights")
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    base_mac = min(get_max_active_clusters(1), sm_count)
    dynamic_mac_ladder = _DYNAMIC_MAC_LADDER
    if _GLM53_B12X_DYNAMIC_MAC_LADDER is not None:
        dynamic_mac_ladder = _effective_glm53_mac_ladder(
            dynamic_mac_ladder,
            _GLM53_B12X_DYNAMIC_MAC_LADDER,
            num_experts=E,
            num_local_experts=E,
            hidden_size=k,
            intermediate_size=n,
            num_topk=num_topk,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
    tuned_mac = _lookup_mac_ladder(dynamic_mac_ladder, m * num_topk)
    mac = min(tuned_mac or base_mac, base_mac)
    # tile_m comes from the workspace's shared selection so the kernel's task
    # and scale indexing matches the allocated scratch geometry.
    mma_tiler_mn = (tile_m, _level_tile_n(activation_precision))
    ep_local_cls = _ep_local_prefill_kernel(
        E=E, m=m, k=k, n=n, num_topk=num_topk, tile_m=tile_m, activation=activation,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit,
        quant_mode=quant_mode, tiled=tiled)
    if ep_local_cls is not None:
        share_input_across_experts = False  # per-expert scales, local route count
    prefill_reuse = (
        (_GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128)
        and m >= 3456
        and E == 288
        and k == 4096
        and n == 512
        and num_topk == 8
        and activation_precision == "fp4"
        and quant_mode == "nvfp4"
        and mma_tiler_mn == (128, 128)
        and activation == "swigluoai_uninterleave"
        and swiglu_alpha == 1.0
        and swiglu_beta == 0.0
        and swiglu_limit == 10.0
        and torch.cuda.get_device_capability() == (12, 1)
        and _prefill_reuse_stock_contract_matches(
            fc1_n128=_GLM53_B12X_PREFILL_FC1_N128,
        )
    )
    prefill_fc1_n128 = prefill_reuse and _GLM53_B12X_PREFILL_FC1_N128
    _announce = globals().get("_prefill_reuse_announce")
    if ((_GLM53_B12X_PREFILL_REUSE or _GLM53_B12X_PREFILL_FC1_N128) and m >= 3456
            and _announce is not None and not _announce[0]):
        # 39차: one line per boot with the values the gate saw, so a bracket
        # can tell "engaged" from "silently ineligible" (P1 read 0 % with
        # the knob armed and no log at all)
        _announce[0] = True
        logging.getLogger("flashinfer.b12x").warning(
            "[b12x prefill reuse] %s on the first large prefill: m=%d E=%d k=%d n=%d topk=%d "
            "act_precision=%s quant=%s tiler=%s activation=%s swiglu=(%s,%s,%s) fc1_n128=%s",
            "ENGAGED" if prefill_reuse else "NOT taken", m, E, k, n, num_topk,
            activation_precision, quant_mode, mma_tiler_mn, activation,
            swiglu_alpha, swiglu_beta, swiglu_limit, prefill_fc1_n128)

    # A private exact selector lets startup compare actual packed SF6 weights
    # through two cache-isolated handles without mutating process-wide flags.
    if _tp_sf6_q0_override is not None and type(_tp_sf6_q0_override) is not bool:
        raise TypeError("TP SF6 Q0 override must be bool or None")
    tp_sf6_q0_enabled = (_TP_SF6_Q0_ENABLED if _tp_sf6_q0_override is None
                        else _tp_sf6_q0_override)
    if type(_prefill_tile64) is not bool:
        raise TypeError('private prefill tile64 override must be bool')
    if _prefill_tile64 and (not tp_sf6_q0_enabled or _prefill_scale_expansion
            or not _prefill_m64_eligible(
                m=m, E=E, k=k, n=n, num_topk=num_topk, tile_m=tile_m,
                quant_mode=quant_mode, tiled=tiled, reform_sf_pack=reform_sf_pack,
                activation=activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=share_input_across_experts)):
        raise ValueError('private M64 requires exact short packed TP Q0 prefill')
    tp_sf6_q0 = _tp_sf6_q0_eligible(
        enabled=tp_sf6_q0_enabled,E=E,m=m,k=k,n=n,num_topk=num_topk,
        tile_m=128 if _prefill_tile64 else tile_m,
        quant_mode=quant_mode,tiled=tiled,reform_sf_pack=reform_sf_pack,
        activation=activation,swiglu_alpha=swiglu_alpha,swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,share_input_across_experts=share_input_across_experts)
    if _tp_sf6_q0_override is True and not tp_sf6_q0:
        raise ValueError("explicit TP SF6 Q0 selection is outside exact eligibility")
    if tp_sf6_q0:
        from .moe_dynamic_gated_sf6_q0 import MoEGatedDynamicKernelSF6Q0, stock_contract_matches
        from .moe_dynamic_gated_raw_q0 import MoEGatedDynamicKernelRawQ0
        if torch.cuda.get_device_capability() != (12,1) or not stock_contract_matches():
            raise RuntimeError("TP SF6 Q0 requires pinned SM121 source")

    if type(_prefill_q0_batch8) is not bool:
        raise TypeError('private Q0 batch8 override must be bool')
    if _prefill_q0_batch8 and (not tp_sf6_q0 or _prefill_tile64 or not 64 < m <= 8192):
        raise ValueError('private Q0 batch8 requires exact M128 short-prefill FP32 scatter')

    if type(_prefill_scale_expansion) is not bool:
        raise TypeError("prefill scale expansion override must be bool")
    if type(_prefill_n128) is not bool or (_prefill_n128 and not _prefill_scale_expansion):
        raise ValueError('private N128 requires the exact expanded-scale prefill contract')
    if _prefill_scale_expansion and (reform_sf_pack or ep_local_cls is not None
            or (m <= 8192 and not tp_sf6_q0)
            or not _prefill_scale_expansion_eligible(
                m=m, E=E, k=k, n=n, num_topk=num_topk, tile_m=tile_m,
                quant_mode=quant_mode, tiled=tiled, activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=share_input_across_experts)):
        raise ValueError("expanded scales require the exact eager GLM prefill arithmetic")

    cache_key = _dynamic_kernel_cache_key(
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        E=E,
        k=k,
        n=n,
        num_topk=num_topk,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        tiled=tiled,
        prefill_reuse=prefill_reuse,
        prefill_fc1_n128=prefill_fc1_n128,
        ep_local_prefill=ep_local_cls is not None,
        reform_sf_pack=reform_sf_pack,
        tp_sf6_q0=tp_sf6_q0,
    )
    cache_key = (*cache_key, "tp_prefill_scatter_fp32_v1", tp_sf6_q0)
    bound_ep_fp32 = _bound_ep_prefill_fp32(
        E=E, k=k, n=n, num_topk=num_topk, quant_mode=quant_mode, activation=activation,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit, tiled=tiled)
    if bound_ep_fp32:
        cache_key = (*cache_key, "bound_ep_prefill_fp32_v1")
    prefill_word_unpack = _long_prefill_sf6_word_unpack(
        m=m, E=E, k=k, n=n, num_topk=num_topk, tile_m=tile_m,
        quant_mode=quant_mode, tiled=tiled, reform_sf_pack=reform_sf_pack,
        activation=activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit, ep_local=ep_local_cls is not None,
        tp_sf6_q0=tp_sf6_q0, share_input_across_experts=share_input_across_experts)
    if type(_prefill_packets) is not bool:
        raise TypeError('private FFN packet selector must be bool')
    if _prefill_packets:
        if (not prefill_word_unpack or prefill_reuse or _prefill_scale_expansion
                or _prefill_tile64 or _prefill_n128 or _prefill_q0_batch8):
            raise ValueError('FFN packets require the ordinary long-prefill SF6 M128 body')
        cache_key = (*cache_key, 'long_prefill_fp8_packets_v2_stride')
    if prefill_word_unpack:
        cache_key = (*cache_key, 'long_prefill_sf6_route_words_fp32_v2')
    short_word_unpack = _short_prefill_q0_word_unpack(
        m=m, tp_sf6_q0=tp_sf6_q0, reform_sf_pack=reform_sf_pack,
        ep_local=ep_local_cls is not None)
    if short_word_unpack:
        cache_key = (*cache_key, 'short_prefill_q0_words_v1')
    if _prefill_scale_expansion:
        cache_key = (*cache_key, 'temporary_prefill_raw_scales_v1')
    if _prefill_n128:
        cache_key = (*cache_key, 'private_prefill_n128_tiled_v1')
    if _prefill_q0_batch8:
        cache_key = (*cache_key, 'private_prefill_q0_batch8_v1')
    if _prefill_tile64:
        cache_key = (*cache_key, 'private_prefill_m64_fp32_v1')
    if chunk != TILED_W13_K_IN:
        cache_key = (*cache_key, 'w13_chunk', chunk)
    activation_scale_search = _activation_scale_search_for(
        state_E=E, weight_E=E, k=k, n=n, num_topk=num_topk,
        quant_mode=quant_mode, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit)
    if activation_scale_search:
        cache_key = (*cache_key, "activation_scale_search_v1", activation_scale_search)
    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    scratch_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel: Any = None if _prefill_tile64 else MoEDynamicKernel(
        scatter_fp32=bound_ep_fp32,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        activation_scale_search=activation_scale_search,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        hidden_size=k,
        intermediate_size=n,
        num_topk=num_topk,
    )
    if tiled and ep_local_cls is None:
        # the tiled layout is read by the gated kernel's subclass only; the
        # #368 prefill-reuse lane subclasses the stock kernel and would read
        # the 4-D tensors as row-major -- the two cannot combine yet
        if prefill_reuse:
            raise ValueError(
                "tiled expert weights (static v2 cell t) and the prefill-reuse lane "
                "cannot combine yet: turn one of them off"
            )
        if not (_prefill_tile64 or _prefill_n128) and not isinstance(kernel, MoEGatedDynamicKernel):
            raise ValueError(
                "tiled expert weights (static v2 cell t) need the gated dynamic "
                f"kernel for prefill; the dispatcher selected {type(kernel).__name__}"
            )
        tiled_cls = MoEGatedDynamicKernelRawQ0 if tp_sf6_q0 else MoEGatedDynamicKernelTiled
        if _prefill_scale_expansion and not tp_sf6_q0:
            from .moe_dynamic_prefill_raw_route import MoEGatedDynamicKernelPrefillRawRoute
            tiled_cls = MoEGatedDynamicKernelPrefillRawRoute
        if _prefill_n128:
            from .moe_dynamic_prefill_n128_tiled import MoEGatedPrefillN128TiledQ0, MoEGatedPrefillN128TiledLong
            tiled_cls = MoEGatedPrefillN128TiledQ0 if tp_sf6_q0 else MoEGatedPrefillN128TiledLong
        tiled_kwargs = {}
        if reform_sf_pack:
            from .moe_dynamic_gated_sf6 import MoEGatedDynamicKernelSF6
            tiled_cls = MoEGatedDynamicKernelSF6Q0 if tp_sf6_q0 else MoEGatedDynamicKernelSF6
            if prefill_word_unpack:
                from .moe_dynamic_gated_sf6_prefill import MoEGatedDynamicKernelSF6Prefill
                tiled_cls = MoEGatedDynamicKernelSF6Prefill
                if _prefill_packets:
                    from .moe_dynamic_prefill_packets import MoEGatedDynamicKernelSF6Packets
                    tiled_cls = MoEGatedDynamicKernelSF6Packets
            elif short_word_unpack:
                from .moe_dynamic_gated_sf6_q0_words import MoEGatedDynamicKernelSF6Q0Words
                tiled_cls = MoEGatedDynamicKernelSF6Q0Words
            if _prefill_tile64:
                from .moe_dynamic_prefill_m64 import MoEGatedDynamicKernelPrefillM64
                tiled_cls = MoEGatedDynamicKernelPrefillM64
            tiled_kwargs = dict(reform_sf_pack=True)
        if _prefill_q0_batch8:
            from .moe_prefill_q0_batch8 import PrefillQ0Batch8Packed, PrefillQ0Batch8Raw, PrefillQ0Batch8N128
            tiled_cls = (PrefillQ0Batch8N128 if _prefill_n128 else
                         PrefillQ0Batch8Packed if reform_sf_pack else PrefillQ0Batch8Raw)
        kernel = tiled_cls(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation_scale_search=activation_scale_search,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            share_input_across_experts=share_input_across_experts,
            **tiled_kwargs,
        )
    if prefill_reuse:
        candidate_cls = (
            MoEGatedPrefillN128Kernel
            if prefill_fc1_n128
            else MoEGatedPrefillReuseKernel
        )
        kernel = candidate_cls(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation_scale_search=activation_scale_search,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            share_input_across_experts=share_input_across_experts,
        )
    if ep_local_cls is not None:
        ep_kwargs = {}
        if reform_sf_pack:
            from .moe_dynamic_ep_local import MoEGatedEPLocalKernelSF6
            ep_local_cls = MoEGatedEPLocalKernelSF6
            ep_kwargs = dict(reform_sf_pack=True)
        kernel = ep_local_cls(
            sf_vec_size=sf_vec_size, mma_tiler_mn=mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            activation_scale_search=activation_scale_search,
            fast_math=fast_math, activation=activation, swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit,
            share_input_across_experts=False, **ep_kwargs)
    launch = _DynamicMoELaunch(
        kernel,
        k=k,
        num_topk=num_topk,
        activation_precision=activation_precision,
        sf_vec_size=sf_vec_size,
        reform_sf_pack=reform_sf_pack,
        prefill_tile64=_prefill_tile64,
        prefill_packets=_prefill_packets,
    )

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # Runtime-shaped tensors passed as pointers
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    packed_a_fake = make_ptr(
        scratch_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    if tiled:
        # tile-major weights (moe_static_kernel_v5): the same 4-D shapes the
        # static v5 compile uses; the tiled gated subclass groups the K modes
        if k % chunk != 0 or n % TILED_W2_K_IN != 0:
            raise ValueError(
                f"tiled expert weights need K % {chunk} == 0 and "
                f"I_tp % {TILED_W2_K_IN} == 0 (got K={k}, I_tp={n})"
            )
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype,
            (w1_rows, chunk, k // chunk, E),
            stride_order=(1, 0, 2, 3),
            assumed_align=16,
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype,
            (k, TILED_W2_K_IN, n // TILED_W2_K_IN, E),
            stride_order=(1, 0, 2, 3),
            assumed_align=16,
        )
    else:
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype,
            (w1_rows, k, E),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
        b_down_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype,
            (k, n, E),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E + 1,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    scatter_dtype = cutlass.Float32 if ep_local_cls is not None or tp_sf6_q0 or prefill_word_unpack or bound_ep_fp32 else a_dtype
    scatter_fake = make_ptr(scatter_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    packed1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (E, (2*n // 128) * (k // 256), REFORM_SF_STAGE)
        if reform_sf_pack else (1, 1, 16), stride_order=(2, 1, 0), assumed_align=16)
    packed2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (E, (k // 256) * (n // 128), REFORM_SF_STAGE)
        if reform_sf_pack else (1, 1, 16), stride_order=(2, 1, 0), assumed_align=16)
    key_files = _kernel_source_files() + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ('moe_dynamic_prefill_packets.py', 'moe_w4a16_fp4_helpers.py'))
        if _prefill_packets else ()) + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ('moe_prefill_q0_batch8.py', '_prefill_q0_batch8.py'))
        if _prefill_q0_batch8 else ()) + (
        (os.path.join(os.path.dirname(__file__), 'moe_dynamic_prefill_n128_tiled.py'),)
        if _prefill_n128 else ()) + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ('moe_dynamic_prefill_m64.py', '_prefill_m64_bodies.py'))
        if _prefill_tile64 else ()) + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ("moe_dynamic_prefill_raw_route.py", "moe_dynamic_gated_sf6_prefill.py",
               "moe_dynamic_gated_raw_q0.py", "moe_dynamic_gated_sf6_q0.py"))
        if _prefill_scale_expansion else ()) + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ("moe_dynamic_gated_sf6_words.py", "moe_dynamic_gated_sf6_prefill.py"))
        if prefill_word_unpack else ()) + (
        tuple(os.path.join(os.path.dirname(__file__), name) for name in
              ("moe_dynamic_gated_sf6_words.py", "moe_dynamic_gated_sf6_q0_words.py"))
        if short_word_unpack else ()) + (
        (os.path.join(os.path.dirname(__file__), "moe_dynamic_gated_sf6_q0.py"),)
        if tp_sf6_q0 else ()) + (
        (os.path.join(os.path.dirname(__file__), "moe_dynamic_gated_raw_q0.py"),)
        if tp_sf6_q0 and not reform_sf_pack else ()) + (
        (os.path.join(os.path.dirname(__file__), "moe_dynamic_ep_local.py"),)
        if ep_local_cls is not None or tp_sf6_q0 else ())
    compiled = build_and_load_cute_dsl_kernel(
        _cute_dsl_module(key_files),
        _disk_kernel_name(f"dynamic_e{E}_k{k}_n{n}_t{num_topk}{'_tiled' if tiled else ''}"
                          f"{'' if chunk == TILED_W13_K_IN else f'_c{chunk}'}", cache_key),
        lambda: cute.compile(
            launch,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            pair_head_fake,
            task_head_fake,
            task_tail_fake,
            task_expert_fake,
            task_valid_rows_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            expert_write_rows_fake,
            expert_tile_base_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            packed1_fake,
            packed2_fake,
            1,
            1,
            1,
            1,  # runtime Int32 placeholders
            128,  # runtime packet byte stride; unused by ordinary BF16 input
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=key_files,
    )

    if prefill_reuse:
        logging.getLogger("flashinfer.b12x").warning(
            "[b12x prefill reuse] compiled exact GLM M128 lane: "
            "Q0 8 rows, parallel E288 scan/top8 reserve, FC2 A/SFA retained; "
            "FC1_N128=%s; m=%d k=%d n=%d experts=%d mac=%d",
            prefill_fc1_n128, m, k, n, E, mac,
        )
    result = (compiled, mac)
    _DYNAMIC_KERNEL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Dynamic launch
# ---------------------------------------------------------------------------
def _ep_local_scatter_buffer(workspace, output, num_tokens, k, *, tp=False, long_prefill=False, bound_ep=False):
    """Get this shared workspace's FP32 sum while preserving the BF16 ABI.

    Grow-only: the buffer is sized by the largest call so far. The legacy `tp`
    path keeps its 16384-row ceiling; the long-prefill SF6 lane may grow to its
    32768-row predicate, a maximum production boots pre-pay in the memory
    gate's 32,256-token far prefill pass, before the door opens -- the first
    long request allocates nothing here.
    """
    ceiling = 32768 if long_prefill else 16384
    width = 4096
    if bound_ep:
        width = _admitted_moe().hidden
        # Compact rows can exceed source tokens by top-k. Bound the byte
        # extent rather than borrowing GLM's source-token ceiling.
        ceiling = (2**31 - 1) // (k * 4)
    if (output.dtype != torch.bfloat16 or tuple(output.shape) != (num_tokens, k)
            or not output.is_contiguous() or output.device != workspace.device
            or k != width
            or not (1 if tp or bound_ep or getattr(workspace, "ep_tiled", False) else 4096) <= num_tokens <= ceiling):
        raise ValueError(f"expert-local FP32 scatter requires contiguous CUDA BF16 [T,{width}]")
    current = workspace.ep_scatter_fp32
    if current is not None and (current.dtype != torch.float32 or current.device != output.device
            or current.ndim != 2 or current.shape[1] != k or not current.is_contiguous()):
        raise ValueError("expert-local FP32 scatter workspace has incompatible storage")
    if current is None or current.shape[0] < num_tokens:
        if getattr(workspace, "ep_tiled", False):
            raise RuntimeError("tiled EP scatter must be allocated before inference")
        current = torch.empty((num_tokens, k), dtype=torch.float32, device=output.device)
        workspace.ep_scatter_fp32 = current
    # Launch arguments use data_ptr(), so the custom kernel cannot tell the
    # allocator about a later nondefault stream before this buffer grows.
    current.record_stream(torch.cuda.current_stream(output.device))
    return current[:num_tokens]


def _prefill_m64_workspace(original, num_tokens):
    """A separate eager workspace; M128 and captured decode owners stay intact."""
    key = ('private_prefill_m64_workspace_v1', original.state_E, original.weight_E,
           original.k, original.n, original.num_topk, str(original.device))
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None and cached.routed_rows_capacity >= num_tokens * original.num_topk:
        return cached
    capacity = 1 << (num_tokens - 1).bit_length()
    cached = allocate_sm120_dynamic_workspace(
        state_E=original.state_E, weight_E=original.weight_E,
        routed_rows=capacity * original.num_topk, k=original.k, n=original.n,
        num_topk=original.num_topk, device=original.device,
        activation='swigluoai_uninterleave', quant_mode='nvfp4',
        tile_m=64, _prefill_tile64=True)
    _WORKSPACE_CACHE[key] = cached
    return cached


def launch_sm120_dynamic_moe(
    *,
    workspace: Sm120DynamicMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    _tp_sf6_q0_override: bool | None = None,
    _prefill_scale_expansion: bool | None = None,
    _prefill_tile64: bool | None = None,
    _prefill_n128: bool = False,
    _prefill_q0_batch8: bool = False,
    _packet_input=None,
) -> torch.Tensor:
    """Launch the SM120 dynamic MoE kernel."""
    global _TP_SF6_Q0_LAUNCH_LOGGED
    if _packet_input is not None:
        from engine.modules.prefill_packets import PacketBatch, ffn_packet_rows
        if (a is not None or not isinstance(_packet_input, PacketBatch)
                or not ffn_packet_rows(num_tokens)
                or (_packet_input.geometry.rows, _packet_input.geometry.hidden) != (num_tokens, k)
                or (num_experts, k, n, top_k, workspace.tile_m) != (288, 4096, 512, 8, 128)
                or input_gs.numel() != num_experts
                or _prefill_scale_expansion or _prefill_tile64 or _prefill_n128 or _prefill_q0_batch8):
            raise ValueError('FFN packet launch requires its explicit real-row/weight/workspace contract')
        device = _packet_input.received.device
        if (scatter_output.device != device or workspace.device != device
                or tuple(scatter_output.shape) != (num_tokens, k)
                or scatter_output.dtype != torch.bfloat16 or not scatter_output.is_contiguous()
                or tuple(topk_ids.shape) != (num_tokens, top_k) or topk_weights.shape != topk_ids.shape
                or topk_ids.device != device or topk_weights.device != device):
            raise ValueError('FFN packet routes and output must use exactly the real rows on one device')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('FFN packet MoE is eager-only')
    else:
        if a is None:
            raise ValueError('dynamic MoE requires a BF16 input or an explicit packet owner')
        device = a.device
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher"
        )
    _check_memref_limit("scatter_output", scatter_output.numel())
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)
    input_gs_is_shared = input_gs.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    direct_sf6 = bool(weights.reform_scales is not None and weights.reform_scales.enabled)
    if direct_sf6:
        from .moe_dynamic_gated_sf6 import stock_contract_matches
        direct_sf6 = bool(stock_contract_matches())
    if _prefill_n128:
        if _prefill_tile64 is True or _prefill_scale_expansion is False:
            raise ValueError('private N128 cannot combine with M64 or disabled scale expansion')
        _prefill_tile64, _prefill_scale_expansion = False, True
    if type(_prefill_q0_batch8) is not bool:
        raise TypeError('private Q0 batch8 override must be bool')
    if _prefill_q0_batch8:
        if _prefill_tile64 is True or not 64 < num_tokens <= 8192 or workspace.tile_m != 128:
            raise ValueError('private Q0 batch8 requires a short M128 workspace')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('private Q0 batch8 is eager-only pending GPU qualification')
        _prefill_tile64 = False
        if _prefill_scale_expansion is None:
            _prefill_scale_expansion = False
    if _prefill_tile64 is None:
        # Unqualified experiments must never become the serving default.
        _prefill_tile64 = False
    if type(_prefill_tile64) is not bool:
        raise TypeError('private prefill tile64 override must be bool')
    if _prefill_tile64:
        required_scale_bytes = workspace.physical_tiles_capacity * 128 * (k // 16)
        if workspace.scale_flat.numel() < required_scale_bytes:
            raise ValueError('private M64 workspace lacks a physical SFA atom per logical tile')
        if not _prefill_m64_eligible(
                m=num_tokens, E=num_experts, k=k, n=n, num_topk=top_k,
                tile_m=workspace.tile_m, quant_mode=quant_mode,
                tiled=bool(getattr(weights, 'tiled', False)), reform_sf_pack=direct_sf6,
                activation=activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=input_gs_is_shared):
            raise ValueError('private M64 launch requires its exact M64 workspace and packed weights')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('private M64 prefill is eager-only pending GPU qualification')
    if _prefill_scale_expansion is None:
        _prefill_scale_expansion = False
    if type(_prefill_scale_expansion) is not bool:
        raise TypeError("prefill scale expansion override must be bool")
    expanded_scales = None
    if _prefill_scale_expansion:
        if not direct_sf6 or not _prefill_scale_expansion_eligible(
                m=num_tokens, E=num_experts, k=k, n=n, num_topk=top_k,
                tile_m=workspace.tile_m, quant_mode=quant_mode,
                tiled=bool(getattr(weights, "tiled", False)), activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=input_gs_is_shared):
            raise ValueError("temporary expansion requires exact packed GLM prefill weights")
        from .moe_sf6_prefill_scales import expand_scales
        # Keep both owners through submission. They are produced and consumed
        # on the same execution stream; no aliases are installed on weights.
        expanded_scales = expand_scales(weights.reform_scales, experts=num_experts,
                                       hidden=k, intermediate=n)
        sf1_address, sf2_address = (v.data_ptr() for v in expanded_scales)
        direct_sf6 = False
    else:
        sf1_address, sf2_address = _scale_runtime_addresses(weights, direct_sf6=direct_sf6)
    ep_local = _ep_local_prefill_kernel(
        E=num_experts, m=num_tokens, k=k, n=n, num_topk=top_k,
        tile_m=workspace.tile_m, activation=activation, swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit, quant_mode=quant_mode,
        tiled=bool(getattr(weights, "tiled", False))) is not None
    tp_scatter_fp32 = _tp_sf6_q0_eligible(
        enabled=_TP_SF6_Q0_ENABLED if _tp_sf6_q0_override is None else _tp_sf6_q0_override,
        E=num_experts, m=num_tokens, k=k, n=n, num_topk=top_k,
        tile_m=128 if _prefill_tile64 else workspace.tile_m, quant_mode=quant_mode,
        tiled=bool(getattr(weights, "tiled", False)), reform_sf_pack=direct_sf6,
        activation=activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit, share_input_across_experts=input_gs_is_shared)
    long_prefill_fp32 = _long_prefill_sf6_word_unpack(
        m=num_tokens, E=num_experts, k=k, n=n, num_topk=top_k, tile_m=workspace.tile_m,
        quant_mode=quant_mode, tiled=bool(getattr(weights, 'tiled', False)), reform_sf_pack=direct_sf6,
        activation=activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit, ep_local=ep_local, tp_sf6_q0=tp_scatter_fp32,
        share_input_across_experts=input_gs_is_shared)
    tp_scatter_fp32 = tp_scatter_fp32 or long_prefill_fp32
    bound_ep_fp32 = _bound_ep_prefill_fp32(
        E=num_experts, k=k, n=n, num_topk=top_k, quant_mode=quant_mode, activation=activation,
        swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta, swiglu_limit=swiglu_limit,
        tiled=bool(getattr(weights, "tiled", False)))
    accumulator = (_ep_local_scatter_buffer(workspace, scatter_output, num_tokens, k,
                                            tp=tp_scatter_fp32, long_prefill=long_prefill_fp32, bound_ep=bound_ep_fp32)
                   if ep_local or tp_scatter_fp32 or bound_ep_fp32 else scatter_output)
    compiled, mac = _get_dynamic_kernel(
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        workspace.max_rows,
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
        share_input_across_experts=input_gs_is_shared,
        tile_m=workspace.tile_m,
        quant_mode=quant_mode,
        tiled=bool(getattr(weights, "tiled", False)),
        reform_sf_pack=direct_sf6,
        _tp_sf6_q0_override=_tp_sf6_q0_override,
        _prefill_scale_expansion=_prefill_scale_expansion,
        _prefill_tile64=_prefill_tile64,
        _prefill_n128=_prefill_n128,
        _prefill_q0_batch8=_prefill_q0_batch8,
        _prefill_packets=_packet_input is not None,
        w13_chunk=getattr(weights, "w13_chunk", None),
    )

    # Dynamic kernel: runtime-shaped args are DataPointer (pass data_ptr()),
    # fixed-shape args are Tensor (pass torch tensor directly).  No stream
    # argument -- see the note in launch_sm120_static_moe.
    runtime_args: Tuple[Any, ...] = (
        a.data_ptr() if _packet_input is None else _packet_input.received.data_ptr(),
        flat_ids.data_ptr(),
        flat_weights.data_ptr(),
        workspace.packed_a_view.data_ptr(),
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat.data_ptr(),
        workspace.scale_flat.data_ptr(),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_expert.data_ptr(),
        workspace.task_valid_rows.data_ptr(),
        weights.w13_fp4,
        sf1_address,
        weights.down_fp4,
        sf2_address,
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        accumulator.data_ptr(),
        workspace.token_map.data_ptr(),
        workspace.token_weights.data_ptr(),
        weights.sfb1_packed if direct_sf6 else _sf_pack_dummy(device),
        weights.sfb2_packed if direct_sf6 else _sf_pack_dummy(device),
        num_tokens,
        workspace.max_rows,
        workspace.physical_tiles_capacity * workspace.tile_m,
        workspace.task_capacity,
        _packet_input.geometry.stride if _packet_input is not None else 0,
    )
    _check_dynamic_capacity(workspace, routed_rows=num_tokens * top_k, n=n,
                            where="launch_sm120_dynamic_moe")
    compiled(*runtime_args)
    # Canary overrides and graph capture are not a production launch witness.
    # Keep this after the actual call and emit once without adding a GPU sync.
    if (_TP_SF6_Q0_ENABLED and _tp_sf6_q0_override is None and not _TP_SF6_Q0_LAUNCH_LOGGED
            and _tp_sf6_q0_eligible(
                enabled=_TP_SF6_Q0_ENABLED, E=num_experts, m=num_tokens,
                k=k, n=n, num_topk=top_k, tile_m=workspace.tile_m,
                quant_mode=quant_mode, tiled=bool(getattr(weights, "tiled", False)),
                reform_sf_pack=direct_sf6, activation=activation,
                swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit, share_input_across_experts=input_gs_is_shared)
            and not torch.cuda.is_current_stream_capturing()):
        cell = _admitted_moe()
        print(f"[tp-sf6-q0] LAUNCHED E{cell.experts}/H{cell.hidden}/I{cell.inter_local}/top{cell.topk} T={num_tokens}", flush=True)
        _TP_SF6_Q0_LAUNCH_LOGGED = True
    if ep_local or tp_scatter_fp32 or bound_ep_fp32:
        # CuTe and copy_ use the current PyTorch stream; completion of all
        # atomic updates precedes this single FP32 -> BF16 conversion.
        scatter_output.copy_(accumulator)
    return scatter_output


# ==========================================================================
# W4A16 route-packing implementation
# ==========================================================================
@dataclass(kw_only=True)
class Sm120W4A16MoEWorkspace:
    """Scratch buffers for the SM120 W4A16 MoE path."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation: str
    activation_precision: str
    quant_mode: str
    routed_rows_capacity: int
    route_num_experts: int

    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_map: torch.Tensor | None = None


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _canonical_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _w4a16_workspace_geometry(
    *,
    routed_rows: int,
    route_num_experts: int,
    num_topk: int,
    k: int,
    n: int,
    is_gated: bool,
    device: torch.device,
) -> tuple[int, int, int, int, int]:
    route_slots = 1
    route_blocks = 1
    fc1_c_tmp_elements = 1
    fc2_c_tmp_elements = 1
    fc1_cols = (2 if is_gated else 1) * int(n)
    sms = get_num_sm(device)
    # Size the route buffers for the power-of-2 capacity so route packing keeps
    # a single triton specialization across token counts.
    routed_rows_capacity = route_pack_numel_capacity(
        int(routed_rows), topk=int(num_topk)
    )
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        slots = max_packed_route_slots(
            routed_rows_capacity,
            int(block_size),
            int(route_num_experts),
        )
        blocks = (slots + int(block_size) - 1) // int(block_size)
        route_slots = max(route_slots, slots)
        route_blocks = max(route_blocks, blocks)
        fc1_c_tmp_elements = max(
            fc1_c_tmp_elements,
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )
        fc2_c_tmp_elements = max(
            fc2_c_tmp_elements,
            packed_gemm_scratch_elements(
                size_n=int(k),
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )
    return (
        route_slots,
        route_blocks,
        fc1_c_tmp_elements,
        fc2_c_tmp_elements,
        fc1_cols,
    )


def _make_w4a16_expert_map(
    *,
    state_E: int,
    weight_E: int,
    device: torch.device,
) -> torch.Tensor | None:
    if int(state_E) == int(weight_E):
        return None
    if int(state_E) > int(weight_E):
        raise ValueError("num_local_experts cannot exceed num_experts")
    expert_map = torch.empty((int(weight_E),), dtype=torch.int32, device=device)
    expert_map.fill_(-1)
    expert_map[: int(state_E)].copy_(
        torch.arange(int(state_E), dtype=torch.int32, device=device)
    )
    return expert_map


def _allocate_sm120_w4a16_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation: str = "silu",
) -> Sm120W4A16MoEWorkspace:
    is_gated = validate_activation(activation)
    routed_rows = max(1, int(routed_rows))
    route_num_experts = int(weight_E) if int(state_E) != int(weight_E) else int(state_E)
    (
        route_slots,
        route_blocks,
        fc1_c_tmp_elements,
        fc2_c_tmp_elements,
        fc1_cols,
    ) = _w4a16_workspace_geometry(
        routed_rows=routed_rows,
        route_num_experts=route_num_experts,
        num_topk=num_topk,
        k=k,
        n=n,
        is_gated=is_gated,
        device=device,
    )
    return Sm120W4A16MoEWorkspace(
        state_E=int(state_E),
        weight_E=int(weight_E),
        max_rows=routed_rows,
        k=int(k),
        n=int(n),
        num_topk=int(num_topk),
        device=device,
        activation=activation,
        activation_precision="bf16",
        quant_mode="w4a16",
        routed_rows_capacity=routed_rows,
        route_num_experts=route_num_experts,
        intermediate_cache13=torch.empty(
            (routed_rows * max(fc1_cols, int(k)),),
            dtype=torch.bfloat16,
            device=device,
        ),
        intermediate_cache2=torch.empty(
            (routed_rows, int(n)),
            dtype=torch.bfloat16,
            device=device,
        ),
        fc1_c_tmp=torch.empty(
            (fc1_c_tmp_elements,),
            dtype=torch.float32,
            device=device,
        ),
        fc2_c_tmp=torch.empty(
            (fc2_c_tmp_elements,),
            dtype=torch.float32,
            device=device,
        ),
        packed_route_indices=torch.empty(
            (route_slots,),
            dtype=torch.int32,
            device=device,
        ),
        block_expert_ids=torch.empty(
            (route_blocks,),
            dtype=torch.int32,
            device=device,
        ),
        packed_route_count=torch.empty((1,), dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            (route_num_experts + 1,),
            dtype=torch.int32,
            device=device,
        ),
        expert_map=_make_w4a16_expert_map(
            state_E=state_E,
            weight_E=weight_E,
            device=device,
        ),
    )


_W4A16_WEIGHT_CACHE: Dict[Tuple, W4A16PackedWeights] = {}


def _get_w4a16_packed_weights(
    *,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    activation: str,
    params_dtype: torch.dtype,
    source_format: str = "modelopt",
) -> W4A16PackedWeights:
    key = (
        activation,
        params_dtype,
        source_format,
        tuple(w1_weight.shape),
        tuple(w1_weight_sf.shape),
        tuple(w1_alpha.shape),
        tuple(w2_weight.shape),
        tuple(w2_weight_sf.shape),
        tuple(w2_alpha.shape),
        w1_weight.data_ptr(),
        w1_weight_sf.data_ptr(),
        w1_alpha.data_ptr(),
        w2_weight.data_ptr(),
        w2_weight_sf.data_ptr(),
        w2_alpha.data_ptr(),
    )
    cached = _W4A16_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "W4A16 packed weights are not initialized for CUDA graph capture; "
            "run once before capture so the prepared weights are cached."
        )
    prepared = prepare_w4a16_packed_weights(
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        activation=activation,
        params_dtype=params_dtype,
        source_format=source_format,
    )
    _W4A16_WEIGHT_CACHE[key] = prepared
    _register_cache_eviction(
        _W4A16_WEIGHT_CACHE,
        key,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
    )
    return prepared


def _validate_w4a16_workspace(
    workspace: Sm120W4A16MoEWorkspace,
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation: str,
) -> None:
    validate_activation(activation)
    if workspace.state_E != int(state_E) or workspace.weight_E != int(weight_E):
        raise ValueError("pre-allocated W4A16 workspace expert geometry mismatch")
    if workspace.k != int(k) or workspace.n != int(n):
        raise ValueError("pre-allocated W4A16 workspace hidden geometry mismatch")
    if workspace.num_topk != int(num_topk):
        raise ValueError("pre-allocated W4A16 workspace top-k mismatch")
    if getattr(workspace, "activation", None) != activation:
        raise ValueError("pre-allocated W4A16 workspace activation mismatch")
    if _canonical_cuda_device(workspace.device) != _canonical_cuda_device(device):
        raise ValueError(
            f"pre-allocated W4A16 workspace is on {workspace.device}, expected {device}"
        )
    if workspace.routed_rows_capacity < max(1, int(routed_rows)):
        raise ValueError(
            "pre-allocated W4A16 workspace is too small for the requested "
            f"routed rows ({workspace.routed_rows_capacity} < {routed_rows})"
        )


def _launch_sm120_w4a16_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    source_format: str = "modelopt",
    _workspace=None,
    _prepared_weights=None,
) -> torch.Tensor:
    prepared = (
        _prepared_weights
        if isinstance(_prepared_weights, W4A16PackedWeights)
        else _get_w4a16_packed_weights(
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            activation=activation,
            params_dtype=a.dtype,
            source_format=source_format,
        )
    )
    if int(prepared.num_experts) != int(num_local_experts):
        raise ValueError("num_local_experts must match w1_weight.shape[0] for W4A16.")
    num_tokens = int(topk_ids.size(0))
    routed_rows = num_tokens * int(top_k)
    k = int(a.size(1))
    n = int(prepared.intermediate_size)

    if _workspace is None:
        workspace = _get_cached_workspace(
            backend="w4a16",
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
            quant_mode="w4a16",
            activation=activation,
        )
    else:
        workspace = _workspace
    if not isinstance(workspace, Sm120W4A16MoEWorkspace):
        raise TypeError("expected a W4A16 workspace for quant_mode='w4a16'")
    _validate_w4a16_workspace(
        workspace,
        state_E=num_local_experts,
        weight_E=num_experts,
        routed_rows=routed_rows,
        k=k,
        n=n,
        num_topk=top_k,
        device=a.device,
        activation=activation,
    )

    return run_w4a16_moe(
        a,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=workspace.intermediate_cache13,
        intermediate_cache2=workspace.intermediate_cache2,
        output=scatter_output,
        fc1_c_tmp=workspace.fc1_c_tmp,
        fc2_c_tmp=workspace.fc2_c_tmp,
        packed_route_indices=workspace.packed_route_indices,
        block_expert_ids=workspace.block_expert_ids,
        packed_route_count=workspace.packed_route_count,
        expert_offsets=workspace.expert_offsets,
        expert_map=workspace.expert_map,
        fast_math=fast_math,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


# ==========================================================================
# Workspace cache (for functional API path)
# ==========================================================================

_Sm120Workspace = Union[
    Sm120StaticMoEWorkspace,
    Sm120DynamicMoEWorkspace,
    Sm120W4A16MoEWorkspace,
]

# Stores the workspace with the largest capacity seen per key and never
# shrinks within a process. clear_sm120_moe_caches() releases everything.
_WORKSPACE_CACHE: Dict[Tuple, _Sm120Workspace] = {}


def cached_workspace_owners() -> tuple:
    """Snapshot the allocation owners used by a warmed/captured workload.

    Functional calls grow this cache, replacing smaller workspaces. A CUDA
    graph caller must retain this snapshot after each capture until its graphs
    are reset: CUDA holds raw pointers, including the grid barrier counters.
    The eager cache can then grow without invalidating an earlier graph.
    """
    return tuple(_WORKSPACE_CACHE.values())


def clear_sm120_moe_caches() -> None:
    """Release every module-level SM12x MoE cache.

    References held by callers are unaffected.
    """
    _WORKSPACE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _SF_PACKED.clear()
    _REFORM_SF_CACHE.clear()
    _W4A16_WEIGHT_CACHE.clear()
    _PADDED_WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    _MICRO_KERNEL_CACHE.clear()
    _DIRECT_MICRO_LAUNCH_CACHE.clear()
    _DIRECT_MICRO_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()


def _dynamic_workspace_tile_m(*, routed_rows, state_E, weight_E, k, n,
                              num_topk, quant_mode, activation, swiglu_limit):
    cfg = _static_v2_config_for(
        num_experts=weight_E, num_local_experts=state_E, hidden_size=k,
        intermediate_size=n, num_topk=num_topk, quant_mode=quant_mode,
        activation=activation, swiglu_limit=swiglu_limit, activation_precision="fp4")
    if cfg and cfg.get("tiled") and cfg.get("reform_sf_pack"):
        # The inherited gated pipeline is M128 only. Smaller requests use
        # its valid-row tail handling over the same direct packed storage.
        return 128
    return _select_dynamic_tile_m(routed_rows, state_E, activation)


def allocate_sm120_moe_workspace(
    *,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    max_rows: int | None = None,
    routed_rows: int | None = None,
    quant_mode: str | None = None,
    activation_precision: str | None = None,
    backend: str | None = None,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> _Sm120Workspace:
    """Allocate the right SM120 MoE workspace from a quantization mode."""
    mode = _normalize_quant_mode(quant_mode, activation_precision)
    capacity_rows = routed_rows if routed_rows is not None else max_rows
    if capacity_rows is None:
        raise ValueError("routed_rows or max_rows is required")
    capacity_rows = max(1, int(capacity_rows))
    device = torch.device(device)

    if mode == "w4a16":
        if backend not in (None, "w4a16"):
            raise ValueError("quant_mode='w4a16' does not use static/dynamic backend")
        return _allocate_sm120_w4a16_workspace(
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation=activation,
        )

    activation_precision = "fp4"
    if backend is None:
        backend = select_sm120_moe_backend(
            num_tokens=max(
                1, (capacity_rows + max(1, int(num_topk)) - 1) // max(1, int(num_topk))
            ),
            num_topk=int(num_topk),
            activation_precision=activation_precision,
            quant_mode=mode,
            num_experts=weight_E,
            num_local_experts=state_E,
            hidden_size=k,
            intermediate_size=n,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
    if backend == "dynamic":
        return allocate_sm120_dynamic_workspace(
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation_precision=activation_precision,
            activation=activation,
            quant_mode=mode,
            tile_m=_dynamic_workspace_tile_m(
                routed_rows=capacity_rows, state_E=state_E, weight_E=weight_E,
                k=k, n=n, num_topk=num_topk, quant_mode=mode,
                activation=activation, swiglu_limit=swiglu_limit),
        )
    if backend == "static":
        return allocate_sm120_static_workspace(
            state_E=state_E,
            weight_E=weight_E,
            max_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation_precision=activation_precision,
            quant_mode=mode,
        )
    raise ValueError(f"unsupported SM120 MoE backend {backend!r}")


def _get_cached_workspace(
    *,
    backend: str,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> _Sm120Workspace:
    """Get or allocate a cached workspace for the given problem shape.

    Reuses the cached workspace if it has enough capacity for the requested
    routed_rows. For static workspaces, max_rows is the direct capacity.
    For dynamic workspaces, routed_rows_capacity is used because the dynamic
    geometry (physical tiles, task queue slots) depends on the original
    routed_rows, not just max_rows.
    """
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)
    # Key dynamic workspaces on the tile band of this call's routed_rows; a
    # larger cached workspace must not pin small calls to its 128 tile.
    tile_m = (
        _dynamic_workspace_tile_m(
            routed_rows=max(1, routed_rows), state_E=state_E, weight_E=weight_E,
            k=k, n=n, num_topk=num_topk, quant_mode=quant_mode,
            activation=activation, swiglu_limit=swiglu_limit)
        if backend == "dynamic" and quant_mode != "w4a16"
        else None
    )
    cache_key = (
        state_E,
        weight_E,
        k,
        n,
        num_topk,
        str(device),
        backend,
        quant_mode,
        activation,
        tile_m,
    )
    cached = _WORKSPACE_CACHE.get(cache_key)

    if cached is not None:
        if isinstance(cached, Sm120DynamicMoEWorkspace):
            if cached.routed_rows_capacity >= max(1, routed_rows):
                assert tile_m is None or cached.tile_m == tile_m
                return cached
        elif isinstance(cached, Sm120W4A16MoEWorkspace):
            if cached.routed_rows_capacity >= max(1, routed_rows):
                return cached
        else:
            if cached.max_rows >= max(1, routed_rows):
                return cached

    if quant_mode == "w4a16" and _is_cuda_graph_capturing():
        raise RuntimeError(
            "W4A16 workspace is not initialized for CUDA graph capture; "
            "provide a preallocated workspace from "
            "allocate_sm120_moe_workspace(..., quant_mode='w4a16') or warm the "
            "functional path before capture."
        )
    workspace = allocate_sm120_moe_workspace(
        state_E=state_E,
        weight_E=weight_E,
        routed_rows=routed_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        quant_mode=quant_mode,
        activation_precision=activation_precision,
        backend=backend,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )

    _WORKSPACE_CACHE[cache_key] = workspace
    return workspace


# ==========================================================================
# Unified dispatch
# ==========================================================================
_PADDED_WEIGHT_CACHE: Dict[Tuple, Tuple] = {}


def _pad_intermediate_to_tile(
    w1_weight,
    w1_weight_sf,
    w2_weight,
    w2_weight_sf,
    fc2_input_scale,
    n,
    tile,
    h,
    num_experts,
    is_gated,
    quant_mode="nvfp4",
):
    """Zero-pad W4A4 weights + scale factors so the intermediate size is a
    multiple of ``tile`` (gate/up tile-split requirement); padded channels are
    zero, so the result is numerically identical.
    """
    quant_mode = _normalize_quant_mode(quant_mode)
    sf_vec_size, _ = _sf_params_for_quant_mode(quant_mode)
    n_pad = ((n + tile - 1) // tile) * tile
    if n_pad == n:
        return w1_weight, w1_weight_sf, w2_weight, w2_weight_sf, fc2_input_scale, n
    E = int(num_experts)
    fc2_input_scale_src = fc2_input_scale
    key = (
        n,
        tile,
        h,
        E,
        bool(is_gated),
        quant_mode,
        w1_weight.data_ptr(),
        w1_weight_sf.data_ptr(),
        w2_weight.data_ptr(),
        w2_weight_sf.data_ptr(),
        fc2_input_scale_src.data_ptr() if fc2_input_scale_src is not None else 0,
    )
    cached = _PADDED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    def mma_to_logical(sf, m, k):
        sw = convert_sf_from_mma_layout(
            sf,
            m=m,
            k=k,
            num_groups=E,
            sf_vec_size=sf_vec_size,
        )
        m_pad = ((m + 127) // 128) * 128
        sw = sw.reshape(E, m_pad, -1)
        cb = (k + sf_vec_size - 1) // sf_vec_size
        # MXFP4 logical scales remain raw UE8M0 bytes.
        if quant_mode == "mxfp4":
            cols_padded = ((cb + 3) // 4) * 4
            return torch.stack(
                [
                    sw[e]
                    .reshape(m_pad // 128, cols_padded // 4, 32, 4, 4)
                    .permute(0, 3, 2, 1, 4)
                    .contiguous()
                    .reshape(m_pad, cols_padded)[:m, :cb]
                    for e in range(E)
                ],
                0,
            )
        # NVFP4 logical scales are decoded float32 magnitudes.
        return torch.stack(
            [unswizzle_block_scale(sw[e], rows=m, cols_blocks=cb) for e in range(E)], 0
        )

    def logical_to_mma(log, m, k):
        sw = torch.stack([swizzle_block_scale(log[e]) for e in range(E)], 0)
        scale_dtype = torch.uint8 if quant_mode == "mxfp4" else torch.float8_e4m3fn
        sw2d = sw.reshape(E * sw.shape[1], sw.shape[2]).to(scale_dtype)
        return convert_sf_to_mma_layout(
            sw2d,
            m=m,
            k=k,
            num_groups=E,
            sf_vec_size=sf_vec_size,
        )

    def pad_dim(t, dim, old, new):
        if new == old:
            return t
        shp = list(t.shape)
        shp[dim] = new - old
        return torch.cat([t, t.new_zeros(shp)], dim=dim)

    if is_gated:
        # w1 packs [up(0:n), gate(n:2n)] rows; pad each half so the split stays
        # tile-aligned, then re-concat.
        up, gate = w1_weight[:, :n, :], w1_weight[:, n : 2 * n, :]
        w1p = torch.cat([pad_dim(up, 1, n, n_pad), pad_dim(gate, 1, n, n_pad)], dim=1)
        log1 = mma_to_logical(w1_weight_sf, m=2 * n, k=h)
        up_sf, gate_sf = log1[:, :n, :], log1[:, n : 2 * n, :]
        log1p = torch.cat(
            [pad_dim(up_sf, 1, n, n_pad), pad_dim(gate_sf, 1, n, n_pad)], dim=1
        )
        w1_sf_p = logical_to_mma(log1p, m=2 * n_pad, k=h)
    else:
        w1p = pad_dim(w1_weight, 1, n, n_pad)
        log1 = mma_to_logical(w1_weight_sf, m=n, k=h)
        w1_sf_p = logical_to_mma(pad_dim(log1, 1, n, n_pad), m=n_pad, k=h)

    # w2 reduces over the intermediate dim: pad its packed columns + SF columns.
    w2p = pad_dim(w2_weight, 2, n // 2, n_pad // 2)
    log2 = mma_to_logical(w2_weight_sf, m=h, k=n)
    cb_n = (n + sf_vec_size - 1) // sf_vec_size
    cb_np = (n_pad + sf_vec_size - 1) // sf_vec_size
    w2_sf_p = logical_to_mma(pad_dim(log2, 2, cb_n, cb_np), m=h, k=n_pad)

    if fc2_input_scale_src is not None and fc2_input_scale_src.numel() == n:
        fc2_input_scale = pad_dim(fc2_input_scale_src, 0, n, n_pad)
    result = (w1p, w1_sf_p, w2p, w2_sf_p, fc2_input_scale, n_pad)
    _PADDED_WEIGHT_CACHE[key] = result
    _register_cache_eviction(
        _PADDED_WEIGHT_CACHE,
        key,
        w1_weight,
        w1_weight_sf,
        w2_weight,
        w2_weight_sf,
        fc2_input_scale_src,
    )
    return result


def launch_sm120_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor] = None,
    input_global_scale: Optional[torch.Tensor] = None,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    source_format: str = "modelopt",
    _workspace=None,
    _weight_views=None,
    _prepared_weights=None,
    _ep_short_output: torch.Tensor | None = None,
    _output_finalize=None,
    _prefill_tile64: bool | None = None,
) -> torch.Tensor:
    """Unified SM120 MoE dispatch — selects static or dynamic by token count.

    input_global_scale overrides w1_alpha as the FC1 input-quant scale and
    is folded into the multiplier internally.  With _weight_views supplied,
    w1_alpha must already contain the fold.

    Optional _workspace and _weight_views can be pre-allocated and reused
    across calls to avoid per-call allocation overhead (wrapper path).
    When not provided (functional API path), a module-level workspace cache
    is used to avoid re-allocating on every call.
    """
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)

    if _ep_short_output is not None and (
        not isinstance(_workspace, Sm120StaticMoEWorkspace)
        or quant_mode != "nvfp4" or source_format != "modelopt"
    ):
        raise ValueError("direct T6 output requires an explicit NVFP4 static workspace")

    num_tokens = topk_ids.size(0)
    k = a.size(1)  # hidden_size
    is_gated = is_gated_activation(activation)
    # w1_weight.size(1) is 2*n for gated or n for non-gated
    intermediate_size = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
    n = intermediate_size
    if quant_mode == "mxfp4" and k % 128 != 0:
        raise ValueError(f"MXFP4 b12x hidden_size ({k}) must be a multiple of 128.")

    if _output_finalize is not None:
        from engine.kernels.moe_output import validate_finalizer
        validate_finalizer(_output_finalize, rows=num_tokens, experts=num_experts,
                           local_experts=num_local_experts, hidden=k, intermediate=n,
                           topk=top_k, quant_mode=quant_mode, activation=activation,
                           limit=swiglu_limit, alpha=swiglu_alpha, beta=swiglu_beta,
                           tiled=bool(getattr(_weight_views, 'tiled', False)))

    # W4A4 kernels need a tile-aligned gate/up split.
    if quant_mode != "w4a16" and n % _LEVEL_TILE_N != 0 and _weight_views is None:
        (
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            fc2_input_scale,
            n,
        ) = _pad_intermediate_to_tile(
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            fc2_input_scale,
            n,
            _LEVEL_TILE_N,
            k,
            w1_weight.size(0),
            is_gated,
            quant_mode,
        )

    routed_rows = num_tokens * top_k

    if quant_mode == "w4a16":
        return _launch_sm120_w4a16_moe(
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            scatter_output=scatter_output,
            fast_math=fast_math,
            activation=activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            source_format=source_format,
            _workspace=_workspace,
            _prepared_weights=_prepared_weights,
        )

    if fc2_input_scale is None:
        if quant_mode == "nvfp4":
            raise ValueError("fc2_input_scale is required when quant_mode='nvfp4'.")
        # MXFP4 has no tensor-wide FC2 input scale. Reuse an existing
        # per-expert tensor because the shared kernel signature still carries
        # the argument; the MXFP4 quantizer ignores it.
        down_input_scale = w2_alpha
    else:
        down_input_scale = fc2_input_scale
    if quant_mode == "nvfp4" and input_global_scale is not None:
        input_gs = input_global_scale
        if _weight_views is None:
            # Alpha must carry input_gs back or the output magnitude is wrong.
            # The wrapper folds before building _weight_views; don't fold twice.
            w1_alpha = (
                w1_alpha.to(torch.float32) * input_global_scale.to(torch.float32)
            ).contiguous()
    else:
        input_gs = w1_alpha

    weights_tiled = static_v2_weights_layout(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=k,
        intermediate_size=n,
        num_topk=top_k,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
    )
    weights_reform_sf_pack = static_v2_weights_reform_sf_pack(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=k,
        intermediate_size=n,
        num_topk=top_k,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
    )
    weights_sf_pack = static_v2_weights_sf_pack(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=k,
        intermediate_size=n,
        num_topk=top_k,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
    )
    weights = (
        _weight_views
        if _weight_views is not None
        else _get_weight_views(
            w1_fp4=w1_weight,
            w1_blockscale=w1_weight_sf,
            w2_fp4=w2_weight,
            w2_blockscale=w2_weight_sf,
            w1_alphas=w1_alpha,
            w2_alphas=w2_alpha,
            n=n,
            k=k,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            tiled=weights_tiled,
            sf_pack=weights_sf_pack,
            reform_sf_pack=weights_reform_sf_pack,
        )
    )

    # Resolve workspace and backend selection.
    # When a pre-allocated workspace is provided (CUDA graph wrapper path),
    # infer the backend from the workspace type so they stay in sync —
    # the caller already committed to a backend at allocation time.
    if _workspace is not None:
        workspace = _workspace
        workspace_activation_precision = getattr(
            workspace, "activation_precision", activation_precision
        )
        if workspace_activation_precision != activation_precision:
            raise ValueError(
                "pre-allocated workspace activation_precision does not match "
                f"requested activation_precision={activation_precision!r}."
            )
        workspace_quant_mode = getattr(workspace, "quant_mode", quant_mode)
        if workspace_quant_mode != quant_mode:
            raise ValueError(
                "pre-allocated workspace quant_mode does not match "
                f"requested quant_mode={quant_mode!r}."
            )
        if isinstance(workspace, Sm120DynamicMoEWorkspace):
            if num_local_experts != num_experts:
                raise ValueError(
                    "pre-allocated dynamic SM120 MoE workspace requires "
                    "num_local_experts == num_experts because dynamic expert "
                    "buffers are indexed by global topk ids."
                )
            # A pre-allocated dynamic workspace keeps its stored tile_m even
            # for smaller calls; its geometry was sized for that tile.
            backend = "dynamic"
        else:
            backend = "static"
    else:
        backend = select_sm120_moe_backend(
            num_tokens=num_tokens,
            num_topk=top_k,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=k,
            intermediate_size=n,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        # The dynamic kernel indexes row_counts/expert_write_rows directly with
        # topk_ids but those buffers are sized with num_local_experts. Unless
        # num_local_experts == num_experts, fall back to the static backend which
        # has global-to-local expert remapping.
        if backend == "dynamic" and num_local_experts != num_experts:
            backend = "static"
        workspace = _get_cached_workspace(
            backend=backend,
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )

    if bool(getattr(weights, "tiled", False)) and backend not in ("static", "dynamic"):
        # the tiled layout is read by the v5 static kernel and the overlaid
        # gated dynamic kernel; every other lane reads row-major weights
        raise NotImplementedError(
            "tiled expert weights (STK_moe_static cell t) reached the "
            f"{backend} backend, which reads the row-major layout"
        )
    if _output_finalize is not None and backend != 'static':
        raise ValueError('MoE finalizer requires the static FP32 scatter backend')
    if _prefill_tile64 is not None:
        # The private M64 prefill lane, reachable from the served route so a gate can
        # measure what would ship instead of a probe-shaped approximation of it. Explicit
        # only: the parameter defaults to None, nothing in the engine passes it, and the
        # launcher below still refuses capture and re-checks exact eligibility. Making it
        # a DEFAULT needs the GPU verdict first (probes/engine_moe_prefill_m64.py).
        if type(_prefill_tile64) is not bool:
            raise TypeError("private prefill tile64 override must be bool or None")
        if _prefill_tile64 and backend != "dynamic":
            raise ValueError(f"private M64 prefill is a dynamic-backend lane, not {backend}")
    if backend == "dynamic":
        if _prefill_tile64:
            # Its own eager workspace, derived from the one this call resolved: the M128
            # owner and every captured decode owner keep their storage.
            workspace = _prefill_m64_workspace(workspace, num_tokens)
        return launch_sm120_dynamic_moe(
            _prefill_tile64=_prefill_tile64,
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
        )
    else:
        return launch_sm120_static_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            _ep_short_output=_ep_short_output,
            _output_finalize=_output_finalize,
        )
