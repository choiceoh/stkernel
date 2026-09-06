"""SM120/SM121 MoE dispatch layer — workspace, compilation, and launch.

Ported from b12x's integration/tp_moe.py. Supports micro (tiny decode),
static (decode), and dynamic (prefill) backends with token-count-based
selection.
"""

from __future__ import annotations

import hashlib
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
from .moe_static_kernel_v5 import (
    MoEStaticKernelV5,
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
SF_VEC_SIZE = 16
_FORCE_MOE_W4A16_ENV = "FLASHINFER_B12X_FORCE_MOE_W4A16"
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)
# The pinned M128 GLM prefill candidate leaves small dynamic tiles and all
# decode backends on their original implementation. Exact 1 is intentional.
_GLM53_B12X_PREFILL_REUSE = (
    os.environ.get("VLLM_GLM53_B12X_PREFILL_REUSE") == "1"
)
_GLM53_B12X_PREFILL_FC1_N128 = (
    os.environ.get("VLLM_GLM53_B12X_PREFILL_FC1_N128") == "1"
)
MoEGatedPrefillReuseKernel = None
MoEGatedPrefillN128Kernel = None


def _prefill_reuse_stock_contract_matches(*, fc1_n128: bool = False) -> bool:
    """Load private inherited helpers only after the optional shape gate.

    Upstream may remove a private symbol before the candidate can compare
    its source hash. That must decline this lane, including with the knob
    off, rather than break importing the otherwise unchanged dispatcher.
    """
    global MoEGatedPrefillReuseKernel, MoEGatedPrefillN128Kernel
    try:
        from .moe_dynamic_prefill import (
            MoEGatedPrefillReuseKernel as candidate,
            stock_contract_matches,
        )
        if fc1_n128:
            from .moe_dynamic_prefill_n128 import (
                MoEGatedPrefillN128Kernel as wide_candidate,
            )
    except (ImportError, AttributeError):
        return False
    if not stock_contract_matches():
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
_B12X_EP_ZERO_WEIGHT_MICRO_ENV = "VLLM_B12X_EP_ZERO_WEIGHT_MICRO"
_B12X_EP_ZERO_WEIGHT_MICRO = (
    os.environ.get(_B12X_EP_ZERO_WEIGHT_MICRO_ENV) == "1"
)
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
    if enabled and exact_shape and forced_backend is not None:
        raise RuntimeError(
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1 cannot run with forced MoE "
            f"backend {forced_backend!r}"
        )
    if not enabled or not exact_shape:
        return None
    # The local-only wrapper remaps every remote route to sentinel E at weight
    # zero. The micro kernel verifies both fields before suppressing the row.
    return int(state_E)

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
_GLM53_B12X_FORCE_BACKEND_ENV = "VLLM_GLM53_B12X_FORCE_BACKEND"
_GLM53_B12X_STATIC_CUTOVER_ENV = "VLLM_GLM53_B12X_STATIC_CUTOVER_PAIRS"
_GLM53_B12X_MICRO_MAC_LADDER_ENV = "VLLM_GLM53_B12X_MICRO_MAC_LADDER"
_GLM53_B12X_STATIC_MAC_LADDER_ENV = "VLLM_GLM53_B12X_STATIC_MAC_LADDER"
_GLM53_B12X_DYNAMIC_MAC_LADDER_ENV = "VLLM_GLM53_B12X_DYNAMIC_MAC_LADDER"


def _parse_glm53_forced_backend(raw: str | None) -> str | None:
    """Parse the optional GLM backend diagnostic without permissive aliases."""
    if raw is None or not raw.strip() or raw.strip() == "auto":
        return None
    value = raw.strip()
    if value not in ("micro", "static", "dynamic"):
        raise ValueError(
            f"{_GLM53_B12X_FORCE_BACKEND_ENV} must be auto, micro, static, "
            f"or dynamic (got {raw!r})"
        )
    return value


def _parse_glm53_static_cutover(raw: str | None) -> int | None:
    """Parse an optional routed-pair cutover; zero deliberately forces dynamic."""
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{_GLM53_B12X_STATIC_CUTOVER_ENV} must be a non-negative integer "
            f"(got {raw!r})"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{_GLM53_B12X_STATIC_CUTOVER_ENV} must be non-negative "
            f"(got {raw!r})"
        )
    return value


def _parse_glm53_mac_ladder(
    raw: str | None,
    env_name: str,
) -> Tuple[Tuple[int, int], ...] | None:
    """Parse ``max_rows:mac`` cells with strictly increasing row bounds."""
    if raw is None or not raw.strip():
        return None
    cells = []
    prior_rows = 0
    for cell in raw.split(","):
        fields = cell.split(":")
        if len(fields) != 2:
            raise ValueError(
                f"{env_name} must be comma-separated max_rows:mac cells "
                f"(got {raw!r})"
            )
        try:
            max_rows, mac = (int(field.strip()) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"{env_name} cells must contain integers (got {cell!r})"
            ) from exc
        if max_rows <= prior_rows or mac <= 0:
            raise ValueError(
                f"{env_name} row bounds must increase strictly and MAC must be "
                f"positive (got {raw!r})"
            )
        cells.append((max_rows, mac))
        prior_rows = max_rows
    return tuple(cells)


_GLM53_B12X_FORCE_BACKEND = _parse_glm53_forced_backend(
    os.environ.get(_GLM53_B12X_FORCE_BACKEND_ENV)
)
_GLM53_B12X_STATIC_CUTOVER_PAIRS = _parse_glm53_static_cutover(
    os.environ.get(_GLM53_B12X_STATIC_CUTOVER_ENV)
)
_GLM53_B12X_MICRO_MAC_LADDER = _parse_glm53_mac_ladder(
    os.environ.get(_GLM53_B12X_MICRO_MAC_LADDER_ENV),
    _GLM53_B12X_MICRO_MAC_LADDER_ENV,
)
_GLM53_B12X_STATIC_MAC_LADDER = _parse_glm53_mac_ladder(
    os.environ.get(_GLM53_B12X_STATIC_MAC_LADDER_ENV),
    _GLM53_B12X_STATIC_MAC_LADDER_ENV,
)
_GLM53_B12X_DYNAMIC_MAC_LADDER = _parse_glm53_mac_ladder(
    os.environ.get(_GLM53_B12X_DYNAMIC_MAC_LADDER_ENV),
    _GLM53_B12X_DYNAMIC_MAC_LADDER_ENV,
)

# The decode-streaming static kernel: v4 (moe_static_kernel_v4, 38차 `u`, the
# profile default). Admitted for the exact GLM-5.3 TP geometry only; every
# other shape keeps the stock kernel. Value: unset/""/"0" = off; `u` = v4
# (FC1 halves over 512-wide K stages, gate and up in separate stages, FC2 2
# stages unless `g<n>` is given); `v` = v4 + the A ring; plus `f<fc1 stages>`,
# `g<fc2 stages>`, `s` (per-CTA %globaltimer stamps, probe only) and, for
# compatibility, `m32`/`a32`. The v2 (`1`, `m..`, `d`) and v3 (`w`, `e`, `k`)
# lanes were sunset in 34차 §8: those tokens are rejected, not remapped.
# Parsed once at import.
_GLM53_B12X_STATIC_V2_ENV = "VLLM_GLM53_B12X_STATIC_V2"
_STATIC_V2_DEFAULT = {
    "tile_m": 32, "fc1": 2, "fc2": 2, "a_rows": 32, "stamps": False,
    "wide": True, "skip_sf": False, "skip_a": False, "v4": True, "a_ring": False,
    # 39차: t = tile-major expert weights (moe_static_kernel_v5), h = 64-row
    # FC1 SFB boxes, z = t pre-swizzled + one cp.async.bulk per B stage
    "tiled": False, "sf_half": False, "bulk_b": False,
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
        if token == "z":
            # v6 (39차): t's storage pre-swizzled into the smem order + one
            # 1-D cp.async.bulk per B stage (probe only until the gated
            # prefill kernel reads that order)
            cfg["tiled"] = True
            cfg["bulk_b"] = True
            continue
        if token == "h":
            # 39차: FC1 SFB boxes of 64 rows (the half a stage reads) instead
            # of the 128-row block
            cfg["sf_half"] = True
            continue
        if token in ("xs", "xa"):
            if not probe:
                raise ValueError(
                    f"{_GLM53_B12X_STATIC_V2_ENV}: {token} is a probe-only timing cell"
                )
            cfg["skip_sf" if token == "xs" else "skip_a"] = True
            continue
        if len(token) < 2 or token[0] not in "mfga" or not token[1:].isdigit():
            raise ValueError(
                f"{_GLM53_B12X_STATIC_V2_ENV} must be 0 or comma-separated "
                f"u|v,f<fc1>,g<fc2>[,m32][,a32][,s][,t][,z][,h] cells (got {raw!r})"
            )
        key = {"m": "tile_m", "f": "fc1", "g": "fc2", "a": "a_rows"}[token[0]]
        cfg[key] = int(token[1:])
    if cfg["tile_m"] != 32 or cfg["a_rows"] != 32:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: v4 is tile_m 32, a_rows 32")
    if cfg["fc1"] < 1 or cfg["fc2"] < 1:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: stages must be >= 1")
    if cfg["a_ring"] and cfg["skip_a"]:
        raise ValueError(f"{_GLM53_B12X_STATIC_V2_ENV}: v (A ring) and xa are exclusive")
    return cfg


_GLM53_B12X_STATIC_V2 = _parse_glm53_static_v2(os.environ.get(_GLM53_B12X_STATIC_V2_ENV))
# Probe hook: a config dict overrides the import-time env value; module-level
# (a monkeypatch target), never read from the environment at launch time.
_STATIC_V2_OVERRIDE: dict | None = None
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
    if not _is_glm53_b12x_tp_geometry(
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


def _is_glm53_b12x_tp_geometry(
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
    """Admit only the deployed GLM-5.3 TP-sharded NVFP4 MoE geometry.

    The intermediate size arrives in two spellings: the model's 2048 at the
    wrapper level and the PER-RANK 512 (2048 / TP4) that
    ``launch_sm120_static_moe`` derives from the sharded weights and passes
    down. Until 2026-09-05 only 2048 was admitted, so every launch-time
    reader of this gate (the forced backend, the cutover, the MAC ladders and
    the static v2 lane) silently kept the stock path -- the first v2 probe
    measured the stock kernel six times over.
    """
    return (
        num_experts == 288
        and num_local_experts == 288
        and hidden_size == 4096
        and intermediate_size in (512, 2048)
        and num_topk == 8
        and quant_mode == "nvfp4"
        and activation == "swigluoai_uninterleave"
        and swiglu_limit == 10.0
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
    if _is_glm53_b12x_tp_geometry(
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
    if _is_glm53_b12x_tp_geometry(
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
    if override is not None and _is_glm53_b12x_tp_geometry(
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


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _normalize_activation_precision(activation_precision: str) -> str:
    """Normalize public activation-precision names to internal modes."""
    if os.environ.get(_FORCE_MOE_W4A16_ENV, "0") == "1":
        return "bf16"

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
    if os.environ.get(_FORCE_MOE_W4A16_ENV, "0") == "1":
        return "w4a16"
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

    cutover_names: tuple[str, ...] = (
        "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
        "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
    )
    cutover = _first_env(*cutover_names)
    if cutover is None:
        cached = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
    else:
        cached = max(0, int(cutover))
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
    swizzled: bool = False   # cell z: the tiled bytes are in the smem order (bulk copies)
    w13_tiled_storage: torch.Tensor | None = None
    w2_tiled_storage: torch.Tensor | None = None
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


_TILE_MAJOR_ATTR = "_b12x_tile_major"   # False / "plain" / "swz" on a weight tensor
_SWZ_PERM: Dict[Tuple[str, str], torch.Tensor] = {}


def _swizzle_perms(device: "torch.device") -> Tuple[torch.Tensor, torch.Tensor]:
    """Byte permutations that turn a plain tile-major B tile into the static
    kernel's smem byte order (probes/b12x_static_layout_print.py --dump):

    FC1 stage (64 rows x 512 K, 16 KB): smem element offset lin = r*256 +
    k%256 + (k//256)*16384, phys = lin ^ (((lin>>7)&7)<<4); in bytes the two
    K halves are 8 KB blocks of 64 x 128 B rows and 8 B units are XORed with
    ((lin_b>>6)&7). FC2 stage (128 rows x 128 K, 8 KB): lin = r*128 + k,
    phys = lin ^ (((lin>>7)&3)<<4) -- 64 B rows, 8 B units XORed with (r&3).
    Both XORs touch only bits below the ones they read, so each is its own
    inverse. Returned as gather indices: dest[d] = src[perm[d]]."""
    key = ("perm", str(device))
    cached = _SWZ_PERM.get(key)
    if cached is not None:
        return cached
    d = torch.arange(16384, dtype=torch.int64)
    lin = d ^ (((d >> 6) & 7) << 3)
    khalf, rem = lin >> 13, lin & 8191
    r, kb = rem >> 7, rem & 127
    perm16k = r * 256 + khalf * 128 + kb          # plain tile: [64 rows][256 B]
    d2 = torch.arange(8192, dtype=torch.int64)
    perm8k = d2 ^ (((d2 >> 6) & 3) << 3)          # plain tile: [128 rows][64 B]
    cached = (perm16k.to(device), perm8k.to(device))
    _SWZ_PERM[key] = cached
    return cached


def tile_expert_weights_inplace(
    w1_fp4: torch.Tensor, w2_fp4: torch.Tensor, *, swizzled: bool = False
) -> None:
    """Re-lay the packed expert weights out tile-major IN PLACE (serving).

    The tensors keep their shapes ([E, rows, K/2] and [E, K, n/2] bytes);
    their bytes become the layouts _tile_expert_weights documents, and the
    tensors are marked (``_b12x_tile_major``) so _get_weight_views(tiled=True)
    views them without a second copy. One transient copy of each tensor
    (the layer's 0.9 + 0.45 GB per rank for GLM-5.3) at weight
    post-processing; a second call is a no-op.
    """
    kind = "swz" if swizzled else "plain"
    have = getattr(w1_fp4, _TILE_MAJOR_ATTR, False)
    if have:
        if have != kind:
            raise ValueError(f"expert weights are already tile-major ({have}); wanted {kind}")
        return
    w13_t, w2_t = _tile_expert_weights(w1_fp4, w2_fp4, swizzled=swizzled)
    w1_fp4.view(-1).copy_(w13_t.view(-1))
    w2_fp4.view(-1).copy_(w2_t.view(-1))
    del w13_t, w2_t
    setattr(w1_fp4, _TILE_MAJOR_ATTR, kind)
    setattr(w2_fp4, _TILE_MAJOR_ATTR, kind)


def _tile_expert_weights(
    w1_fp4: torch.Tensor, w2_fp4: torch.Tensor, *, swizzled: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tile-major copies of the packed expert weights (spec cell t).

    w1_fp4 [E, rows, K/2] bytes -> [E, K/512, rows, 256]: for one k tile the
    rows' 256 B chunks are adjacent, so the kernel's (64 rows x 512 K) TMA box
    is one contiguous 16 KB run instead of 64 chunks 2 KB apart. w2_fp4
    [E, K, n/2] -> [E, n/128, K, 64] likewise for the (128 rows x 128 K) down
    box (8 KB). Bytes only, no arithmetic: the kernel reads exactly the bytes
    the row-major kernel reads, in the same order per tile.
    """
    if w1_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise TypeError("tiled expert weights: packed fp4 bytes (uint8) expected")
    e, rows, kb = w1_fp4.shape
    kin_b = TILED_W13_K_IN // 2
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
    if swizzled:
        # cell z: each (64 rows x 256 B) / (128 rows x 64 B) unit into the
        # kernel's smem byte order, so a stage is one linear bulk copy
        if rows % 64 != 0 or hrows % 128 != 0:
            raise ValueError("swizzled tiles need rows % 64 == 0 and H % 128 == 0")
        perm16k, perm8k = _swizzle_perms(w1_fp4.device)
        w13_t = w13_t.view(e, kb // kin_b, rows // 64, 64 * kin_b)[..., perm16k].contiguous()
        w2_t = w2_t.view(e2, nb // kin2_b, hrows // 128, 128 * kin2_b)[..., perm8k].contiguous()
        w13_t = w13_t.view(e, kb // kin_b, rows, kin_b)
        w2_t = w2_t.view(e2, nb // kin2_b, hrows, kin2_b)
    return w13_t, w2_t


def static_v2_weights_tiled(
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
) -> bool:
    """Whether the static lane this process would take for the geometry reads
    tile-major weights -- the wrapper keys its cached weight views on it, so
    a lane switch (probe) rebuilds them and a tiled view never meets a
    row-major kernel."""
    cfg = _static_v2_config_for(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_topk=num_topk,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
    )
    return bool(cfg is not None and cfg.get("tiled", False))


def static_v2_weights_layout(**geometry) -> Tuple[bool, bool]:
    """(tiled, swizzled) the static lane reads for the geometry -- see
    static_v2_weights_tiled; swizzled (cell z) is the bulk-copy byte order."""
    cfg = _static_v2_config_for(**geometry)
    if cfg is None:
        return False, False
    return bool(cfg.get("tiled", False)), bool(cfg.get("bulk_b", False))


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
    swizzled: bool = False,
) -> _WeightViews:
    """Create permuted weight views for the static kernel.

    The kernel expects concatenated w13 data with shape [2*n, k//2, E]
    via a single TMA descriptor.

    tiled=True (spec cell t, moe_static_kernel_v5): the views are 4-D over a
    tile-major COPY of the packed weights -- w13 as [E, K/512, 2n, 256 B]
    and w2 as [E, n/128, K, 64 B] -- so every kernel TMA box is one
    contiguous run of memory. The copy is cached with the scale conversions
    and follows the source weights' lifetime.
    """
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
        bool(swizzled),
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        # Cache the fresh buffers (scale factors + fp32 alphas) -- and the
        # tile-major weight copies when the lane reads them.
        w1_rows = w1_fp4.shape[1]  # 2*n for gated, n for non-gated
        if not tiled:
            tiled_storage = (None, None)
        elif getattr(w1_fp4, _TILE_MAJOR_ATTR, False):
            # served in place (tile_expert_weights_inplace): the bytes are
            # tile-major already -- reshape, no copy
            have = getattr(w1_fp4, _TILE_MAJOR_ATTR, False)
            if have != ("swz" if swizzled else "plain"):
                raise ValueError(
                    f"tiled expert weights: storage is {have}, the lane wants "
                    f"{'swz' if swizzled else 'plain'}"
                )
            e_, rows_, kb_ = w1_fp4.shape
            e2_, hrows_, nb_ = w2_fp4.shape
            if not getattr(w2_fp4, _TILE_MAJOR_ATTR, False):
                raise ValueError("tiled expert weights: w13 is tile-major but w2 is not")
            tiled_storage = (
                w1_fp4.view(e_, kb_ // (TILED_W13_K_IN // 2), rows_, TILED_W13_K_IN // 2),
                w2_fp4.view(e2_, nb_ // (TILED_W2_K_IN // 2), hrows_, TILED_W2_K_IN // 2),
            )
        else:
            tiled_storage = _tile_expert_weights(w1_fp4, w2_fp4, swizzled=swizzled)
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
        _WEIGHT_CACHE[key] = cached
        _register_cache_eviction(
            _WEIGHT_CACHE,
            key,
            w1_fp4,
            w1_blockscale,
            w1_alphas,
            w2_fp4,
            w2_blockscale,
            w2_alphas,
        )
    w13_sf_contiguous, down_sf_contiguous, w1_alpha, w2_alpha, tiled_storage = cached
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
        swizzled=bool(swizzled),
        w13_tiled_storage=w13_tiled,
        w2_tiled_storage=w2_tiled,
        sfb_w13_ptr=make_ptr(
            sf_dtype,
            w13_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        sfb_down_ptr=make_ptr(
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


# ---------------------------------------------------------------------------
# Kernel compilation cache
#
# The three kernels below are compiled through the shared on-disk CuTe-DSL
# cache (#3874, #4029; docs/design_docs/cute_dsl_kernel_cache.md), so a fresh
# process JITLinks an exported ``.o`` instead of re-running the MLIR pipeline.
# The in-process dicts stay as the level-1 memoization the design describes.
# ---------------------------------------------------------------------------
_CUTE_DSL_MODULE = "b12x_moe"


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
        moe_activation.__file__,
        moe_static_kernel.__file__,
        moe_static_common.__file__,
        moe_static_kernel_v4.__file__,
        moe_static_kernel_v5.__file__,
        moe_dynamic_gated_tiled.__file__,
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
) -> Tuple:
    """The micro kernel's cache key (see :func:`_static_kernel_cache_key`)."""
    return (
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
    tiled: bool = False,
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
    # Preserve the stock key exactly; the opt-in kernel must never reuse a
    # stock artifact (or poison a later stock call in the same process).
    if prefill_fc1_n128:
        return key + ("glm53_prefill_fc1_n128_v1",)
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

    cache_key = _static_kernel_cache_key(
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
    cached = _STATIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    kernel: Any = MoEStaticKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=output_tile_count_n,
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
        a_dtype,
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
        _CUTE_DSL_MODULE,
        _disk_kernel_name(f"static_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}", cache_key),
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
        bool(config.get("sf_half", False)),
        bool(config.get("bulk_b", False)),
    )
    return cfg + _static_kernel_cache_key(**fields)


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
):
    """Compile (or retrieve cached) the decode-streaming static MoE kernel.

    Same fake-tensor contract as :func:`_get_static_kernel` plus the stamps
    tensor ([mac, STAMP_SLOTS] int64) the kernel writes when
    ``config["stamps"]`` is set (and ignores otherwise).
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
    mma_tiler_mn = (int(config["tile_m"]), 128)
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
    cached = _STATIC_V2_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    tiled = bool(config.get("tiled", False))
    kernel_cls = MoEStaticKernelV5 if tiled else MoEStaticKernelV4
    kernel: Any = kernel_cls(
        a_ring=bool(config.get("a_ring", False)),
        sf_half=bool(config.get("sf_half", False)),
        bulk_b=bool(config.get("bulk_b", False)),
        sf_vec_size=sf_vec_size,
        output_tile_count_n=output_tile_count_n,
        fc1_stages=int(config["fc1"]),
        fc2_stages=int(config["fc2"]),
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
        if k % TILED_W13_K_IN != 0 or n % TILED_W2_K_IN != 0:
            raise ValueError(
                f"tiled expert weights need K % {TILED_W13_K_IN} == 0 and "
                f"I_tp % {TILED_W2_K_IN} == 0 (got K={k}, I_tp={n})"
            )
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype, (w1_rows, TILED_W13_K_IN, k // TILED_W13_K_IN, weight_E),
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
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16
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
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    name = (
        f"static2_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}_tm{config['tile_m']}"
        f"f{config['fc1']}g{config['fc2']}a{config['a_rows']}"
        f"{'s' if config['stamps'] else ''}{'d' if config.get('dynamic') else ''}"
        f"{'w' if config.get('wide') else ''}{'e' if config.get('even') else ''}"
        f"{'k' if config.get('split') else ''}{'u' if config.get('v4') else ''}"
        f"{'v' if config.get('a_ring') else ''}{'t' if config.get('tiled') else ''}"
        f"{'h' if config.get('sf_half') else ''}{'z' if config.get('bulk_b') else ''}"
        f"{'xs' if config.get('skip_sf') else ''}{'xa' if config.get('skip_a') else ''}"
    )
    compiled = build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
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

    # Micro always selects tile from routed rows (not just for multi-topk)
    routed_rows = m * num_topk
    mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)

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
    )
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
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        skip_zero_weight_expert_id=skip_zero_weight_expert_id,
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
        a_dtype,
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
        _CUTE_DSL_MODULE,
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
    compile_key = ("direct_micro", kernel.__cache_key__, topk_ids_dtype)
    entry = _DIRECT_MICRO_KERNEL_CACHE.get(compile_key)
    if entry is None:
        compiled = compile_direct_micro_kernel(kernel, topk_ids_dtype=topk_ids_dtype)
        # Register pressure can cap the launchable CTA below the fused body's
        # 512 threads; probe once per compiled kernel.
        accepts = compiled_direct_micro_accepts_block_dim(
            compiled, kernel.launch_block_dim
        )
        entry = (compiled, accepts)
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
            "static lane serves tile-major weights (VLLM_GLM53_B12X_STATIC_V2 cell t)"
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
        # The kernel takes multiplier-form scales only; invert reciprocal
        # inputs into the persistent workspace planes (zeros stay zero,
        # matching the MMA kernels).
        if input_scales_are_reciprocal:
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

    if use_micro:
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
        micro_work_tiles = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
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
        tuned_mac = _lookup_mac_ladder(micro_mac_ladder, routed_rows)
        micro_mac = min(tuned_mac or base_mac, micro_work_tiles, base_mac)
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
        want_swz = bool(static_v2_config is not None and static_v2_config.get("bulk_b"))
        if (bool(getattr(weights, "tiled", False)) != want_tiled
                or bool(getattr(weights, "swizzled", False)) != want_swz):
            raise RuntimeError(
                "tiled expert weights and the tiled static lane (spec cells t / z) must "
                f"agree: views tiled={bool(getattr(weights, 'tiled', False))} "
                f"swizzled={bool(getattr(weights, 'swizzled', False))}, "
                f"lane tiled={want_tiled} swizzled={want_swz}"
            )
        if static_v2_config is not None:
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
            )
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
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
    )
    if static_v2_stamps is not None:
        runtime_args = runtime_args + (static_v2_stamps, static_v2_counter)
    compiled(*runtime_args)

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
    tile_m = _select_dynamic_tile_m(routed_rows, state_E, activation)
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
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
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
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


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
):
    """Compile (or retrieve cached) the SM120 dynamic MoE kernel.

    tiled=True: the expert weights are tile-major (static v2 cell t,
    moe_static_kernel_v5) and arrive as 4-D tensors; the gated kernel's
    subclass MoEGatedDynamicKernelTiled groups them (the stock file stays
    untouched: #368 pins its hash).
    """
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
    )
    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    scratch_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel: Any = MoEDynamicKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
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
    if tiled:
        # the tiled layout is read by the gated kernel's subclass only; the
        # #368 prefill-reuse lane subclasses the stock kernel and would read
        # the 4-D tensors as row-major -- the two cannot combine yet
        if prefill_reuse:
            raise ValueError(
                "tiled expert weights (static v2 cell t) and the prefill-reuse lane "
                "cannot combine yet: turn one of them off"
            )
        if not isinstance(kernel, MoEGatedDynamicKernel):
            raise ValueError(
                "tiled expert weights (static v2 cell t) need the gated dynamic "
                f"kernel for prefill; the dispatcher selected {type(kernel).__name__}"
            )
        kernel = MoEGatedDynamicKernelTiled(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            share_input_across_experts=share_input_across_experts,
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
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            share_input_across_experts=share_input_across_experts,
        )
    launch = _DynamicMoELaunch(
        kernel,
        k=k,
        num_topk=num_topk,
        activation_precision=activation_precision,
        sf_vec_size=sf_vec_size,
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
        if k % TILED_W13_K_IN != 0 or n % TILED_W2_K_IN != 0:
            raise ValueError(
                f"tiled expert weights need K % {TILED_W13_K_IN} == 0 and "
                f"I_tp % {TILED_W2_K_IN} == 0 (got K={k}, I_tp={n})"
            )
        b_w13_fake = cute.runtime.make_fake_compact_tensor(
            weight_dtype,
            (w1_rows, TILED_W13_K_IN, k // TILED_W13_K_IN, E),
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
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        _disk_kernel_name(f"dynamic_e{E}_k{k}_n{n}_t{num_topk}{'_tiled' if tiled else ''}", cache_key),
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
            1,
            1,
            1,
            1,  # runtime Int32 placeholders
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
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
def launch_sm120_dynamic_moe(
    *,
    workspace: Sm120DynamicMoEWorkspace,
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
) -> torch.Tensor:
    """Launch the SM120 dynamic MoE kernel."""
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
    )

    # Dynamic kernel: runtime-shaped args are DataPointer (pass data_ptr()),
    # fixed-shape args are Tensor (pass torch tensor directly).  No stream
    # argument -- see the note in launch_sm120_static_moe.
    runtime_args: Tuple[Any, ...] = (
        a.data_ptr(),
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
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output.data_ptr(),
        workspace.token_map.data_ptr(),
        workspace.token_weights.data_ptr(),
        num_tokens,
        workspace.max_rows,
        workspace.physical_tiles_capacity * workspace.tile_m,
        workspace.task_capacity,
    )
    compiled(*runtime_args)

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


def clear_sm120_moe_caches() -> None:
    """Release every module-level SM12x MoE cache.

    References held by callers are unaffected.
    """
    _WORKSPACE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _W4A16_WEIGHT_CACHE.clear()
    _PADDED_WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    _MICRO_KERNEL_CACHE.clear()
    _DIRECT_MICRO_LAUNCH_CACHE.clear()
    _DIRECT_MICRO_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()


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
        _select_dynamic_tile_m(max(1, routed_rows), state_E, activation)
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

    num_tokens = topk_ids.size(0)
    k = a.size(1)  # hidden_size
    is_gated = is_gated_activation(activation)
    # w1_weight.size(1) is 2*n for gated or n for non-gated
    intermediate_size = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
    n = intermediate_size
    if quant_mode == "mxfp4" and k % 128 != 0:
        raise ValueError(f"MXFP4 b12x hidden_size ({k}) must be a multiple of 128.")

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

    weights_tiled, weights_swizzled = static_v2_weights_layout(
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
            swizzled=weights_swizzled,
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

    if bool(getattr(weights, "swizzled", False)) and backend != "static":
        raise NotImplementedError(
            "pre-swizzled expert weights (VLLM_GLM53_B12X_STATIC_V2 cell z) reached the "
            f"{backend} backend: only the static kernel's bulk copies read that order"
        )
    if bool(getattr(weights, "tiled", False)) and backend not in ("static", "dynamic"):
        # the tiled layout is read by the v5 static kernel and the overlaid
        # gated dynamic kernel; every other lane reads row-major weights
        raise NotImplementedError(
            "tiled expert weights (VLLM_GLM53_B12X_STATIC_V2 cell t) reached the "
            f"{backend} backend, which reads the row-major layout"
        )
    if backend == "dynamic":
        return launch_sm120_dynamic_moe(
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
        )
