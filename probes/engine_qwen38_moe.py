"""Qwen3.8's routed experts at its expert-parallel cell on one GB10: the served b12x NVFP4 lane held to its oracle, then
the micro kernel's tiles and MAC rungs for decode and the dynamic kernel's tile_m for prefill (carry campaign C4).

engine/kernels/cells.py admits the MoE lane at GLM-5.3's TP4 cell only; Qwen3.8 at TEP=4 (512 routed experts, 128 whole
experts a rank, intermediate 640 kept whole, top-10, silu) serves it by declaration. The operator's rule
(engine/QWEN38_CARRY.md, Q4) admits an MoE cell by its kernel record: the served lane within 2% of its oracle, and a tile
sweep. This is that record, on synthetic experts at the per-rank cell: the single-GPU lane has no Qwen3.8 checkpoint.

What the dispatcher does with this cell (engine/kernels/b12x/moe_dispatch.py, read 2026-09-17):
  decode   a captured step (lanes.served()'s moe, compact=False) keeps every route and remaps another rank's to sentinel 128
           at weight 0 (ep_zero_weight_sentinel). The bound cell's zero-weight skip then sends EVERY 2..8-token launch to
           the micro kernel, above the 40-pair micro cutover too (20..80 routed rows <= 128 local experts). Its tile is
           _select_moe_mma_tiler_mn's M64 at all four; its MAC, min(ladder rung, work tiles, SMs), is 48 at all four:
           _MICRO_MAC_LADDER has a rung only up to 20 rows (84), and every rung sits above GB10's 48 SMs. num_experts is
           the rank's 128, so _is_admitted_tp_geometry (512) is false: neither configure_static_v2 nor the _GLM53_B12X_*
           ladder overrides reach this cell. Direct micro is out (I 640 > 512).
  prefill  an eager step (compact=True) dispatches only this rank's pairs, one route a row: ~N*10/4 of them, the static
           kernel at <= 640 pairs (compiled per pair count; N=128 lands there) and the dynamic kernel above -- its generic
           implementation, the gated one wants I <= 512 -- at tile _select_dynamic_tile_m(pairs, 128): N*10/512 rows an
           expert, tile 32 below 48, 64 below 96, 128 above.

Arms, each through a probe hook that keeps the selectors when None:
  decode   N 2..32 (rows 1..8 x K+1 tokens at K=1 and K=3; micro to 8 tokens, the static kernel above -- checks only,
           no arms) x micro tile_m 32/64/128 (moe_dispatch._MICRO_TILE_M_OVERRIDE; M16 is
           the GLM EP direct-scatter variant's) x MAC 16/24/32/48 (_MICRO_MAC_OVERRIDE, never above the SM count): each a
           captured graph of lanes.served()'s moe. The dispatcher's own arm is the one its selectors took.
  prefill  N 1024/4096/8192 x dynamic tile_m 16/32/64/128 (_DYNAMIC_TILE_M_OVERRIDE): the b12x call the eager lane makes
           on this rank's pairs, over three layers' experts.

Experts and routes. A layer is 128 synthetic experts written the way the preshard writes Qwen3.8's
(engine/profiles/qwen38/specs.nvfp4_from_bf16 over Gaussian BF16 weights: [2*640, 2560] packs with rows [up; gate] and
[2560, 640] packs, group-16 E4M3 scales tile-interleaved, FP32 weight_scale_2 multipliers), with per-expert weight spreads
and input scales so a misread expert or scale shows. Routes come from the served router (lanes.route_softmax_topk) over
i.i.d. logits: a uniform draw of 10 of 512 experts a token, three quarters of them on other ranks.

Gates, before any timing, raising on failure:
  oracle    the served output within 2% (max |a-b| / max |b|) of engine/modules/moe.expert_gemm's dataflow over this rank's
            routes -- dequantised NVFP4 weights, activations quantised per 16 under the ModelOpt input scales -- with the
            kernels' reciprocal quantiser (probes/engine_moe_real_check.hardware_quant) and their rounding: FC1 in FP32,
            the SwiGLU output rounded to BF16 before FC2 quantises it, a route's output rounded to BF16, weighted, rounded,
            summed in FP32, rounded once. With exact division in the quantiser's place it is lanes.reference()'s moe on
            the same pairs, byte for byte; that form is reported beside it and not gated: at rounding thresholds it picks
            other FP4 activation bytes, 5-10% on GLM-5.3's real weights where the reciprocal form held 0.66%
            (measurements/st_engine_completion_20260911).
  zero      every route on another rank (all sentinels) and every weight zero give exact zeros.
  replay    a captured decode graph equals the eager call on the capture's inputs and on new ones, and eager repeats
            agree: every element within one adjacent BF16 value or 0.1% of the largest (the kernels' FP32 sums have no
            fixed order; probes/engine_modelopt_check.repeat_stable, held per element). Byte equality is reported.
  launched  the kernel each call took, read back from the dispatcher's own cache key: micro with the sentinel skip at
            decode, the static/dynamic kernel the backend selector names at prefill, and a pinned arm's tile and MAC.
A pinned arm is timed only when exact: within the replay bound of the dispatcher's arm, within 2% of the oracle, zero
with every route foreign. An arm that fails to compile is reported and skipped.

Timings: medians and minima in alternating arm order. `cold` follows a 64 MiB write elsewhere on weights the previous
launch did not read (decode: a new route pattern touches other experts; prefill: the next layer's experts), `warm` repeats
the launch at once. The lane runs beside production, so read medians, not samples. `*_best` rows name the fastest exact
arm per shape by cold median; `cell` gathers them into what cells.py would record (decode: micro tile and MAC per N;
prefill: the one tile kernel_shape.MoE.dynamic_tile_m can pin).

Not measured here: whether pack_stage_bytes (engine/kernels/b12x/moe_reform_sf_pack.py) packs this cell's expert scale
stages into SF6. That depends on the real scale bytes, and synthetic scales say nothing about them; the real scales live
with the checkpoint on srv2, a separate CPU pass.

    bash bench/fleet.sh run --gpu qwen38-moe 50 'Qwen3.8 MoE EP cell: oracle 2%, micro tile/MAC, prefill tile_m' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_moe
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Qwen3.8-Flash-Next's text config, the fields engine/profiles/qwen38/shapes.kernel_shape reads: the wizard's fixture
# (tests/test_engine_kernel_shape.QWEN38_TEXT_CONFIG, off the checkpoint on srv2), carried here because the single-GPU lane
# has no checkpoint; tests/test_probe_qwen38_moe.py holds the two equal.
TEXT_CONFIG = {
    "hidden_size": 2560, "hc_count": 4, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
    "indexer_compress_ratio": 4, "indexer_budget": 2048, "num_experts": 512, "moe_intermediate_size": 640,
    "num_experts_per_tok": 10, "hidden_act": "silu", "shared_expert_intermediate_size": 640,
}
RANK = 0                                   # rank 0 holds experts [0, 128)
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)     # the served row ladder (operator 2026-09-18: eight rows); a row is K + 1 tokens
SPEC_KS = (1, 3)                            # the checkpoint's K=1 and the served chain K=3 (fleet --spec-k 3)
PREFILL_CHECKS = (16, 64, 128, 1024, 4096)     # 16 and 64: the short prompts whose pairs the dynamic kernel now takes
PREFILL_TIMINGS = (1024, 4096, 8192)
MICRO_TILES = (32, 64, 128)
MICRO_MACS = (16, 24, 32, 48)
DYNAMIC_TILES = (16, 32, 64, 128)
HOOKS = ("_MICRO_TILE_M_OVERRIDE", "_MICRO_MAC_OVERRIDE", "_DYNAMIC_TILE_M_OVERRIDE")
STATIC_CUTOVER_PAIRS = 640                 # moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
ORACLE_RELATIVE = 0.02
REPEAT_RELATIVE = 0.001
LAYERS = 3
DECODE_PATTERNS = 16
DECODE_ITERATIONS = 8
PREFILL_ITERATIONS = 6
TRASH_MIB = 64
X_STD = 0.5                                # the hidden rows the experts read
WEIGHT_STD = 0.02                          # each expert's is this times 2**U(-1, 1)
ROUTER_STD = 2.0                           # router logits
CALIBRATION_TOKENS = 256                   # the rows each expert's FC2 input scale is calibrated on
E4M3_FP4 = 448.0 * 6.0                     # ModelOpt: a global scale is amax / (E4M3 max * FP4 max)
SEED = 20260917


@dataclass(frozen=True)
class Cell:
    """The routed-expert cell a rank serves: the model's experts, this rank's, the widths, top-k, and where the rank's
    expert range starts."""
    experts: int
    local: int
    hidden: int
    inter: int
    topk: int
    first_expert: int
    spec_k: int


def kernel_shape():
    """Qwen3.8's per-rank kernel shape at TP=4 (engine/profiles/qwen38/shapes.kernel_shape)."""
    from engine.profiles.qwen38 import shapes
    return shapes.kernel_shape(TEXT_CONFIG)


def cell_of(shape, rank: int = RANK) -> Cell:
    m = shape.moe
    return Cell(m.experts, m.experts_local, m.hidden, m.inter_local, m.topk, rank * m.experts_local, shape.spec_k)


def decode_tokens(c: Cell) -> tuple:
    """Every captured decode launch's tokens over the row ladder at K=1 and K=3: the micro shapes (<= 8) and the static
    ones above the micro cap (10..32) that an eight-row boot captures -- 12 (three rows at K=3) killed the four-row K=3
    boot on 2026-09-18 in its first launch. `ST_PROBE_DECODE_TOKENS=16,24` names the shapes instead: a static shape
    that faults takes the CUDA context with it, so the shapes after it are judged one process each."""
    named = os.environ.get("ST_PROBE_DECODE_TOKENS")
    if named:
        return tuple(sorted({int(m) for m in named.split(",")}))
    return tuple(sorted({rows * (k + 1) for rows in DECODE_ROWS for k in SPEC_KS}))


def decode_order(tokens, cap: int) -> list:
    """The checks' order: the micro shapes largest first, as a boot captures them, then the static shapes smallest
    first -- the shape that killed the four-row boot is judged before the larger ones, and a dead CUDA context after it
    still leaves its verdict."""
    return sorted((m for m in tokens if m <= cap), reverse=True) + sorted(m for m in tokens if m > cap)


# ---- routes ---------------------------------------------------------------------------------------------------------

def routes(tokens: int, c: Cell, generator, device):
    """The served router (lanes.route_softmax_topk: softmax in fp32 over every expert, top-k, renormalised) over i.i.d.
    logits -> (ids int32 [N, k] global, weights f32 [N, k]). The top-k of i.i.d. scores is a uniform draw of k distinct
    experts: a route lands on this rank with probability local / experts, an expert gets a token's route with probability
    k / experts."""
    from engine.profiles.qwen38.lanes import route_softmax_topk
    logits = torch.randn(tokens, c.experts, generator=generator, device=device) * ROUTER_STD
    return route_softmax_topk(logits, c.topk)


def local_mask(ids, c: Cell):
    """(routes on this rank [N, k] bool, ids shifted to the rank's range [N, k] int64)."""
    shifted = ids.long() - c.first_expert
    return (shifted >= 0) & (shifted < c.local), shifted


def foreign_routes(ids, c: Cell):
    """Routes of the same shape, every one on another rank's experts and distinct within a row: all sentinels."""
    others = torch.cat([torch.arange(0, c.first_expert), torch.arange(c.first_expert + c.local, c.experts)]).to(ids.device)
    tokens, k = ids.shape
    return others[torch.arange(tokens * k, device=ids.device).view(tokens, k) % others.numel()].to(torch.int32)


def route_stats(ids, c: Cell) -> dict:
    mine, shifted = local_mask(ids, c)
    counts = torch.bincount(shifted[mine], minlength=c.local).float()
    return dict(tokens=int(ids.shape[0]), routes=int(ids.numel()), local_pairs=int(mine.sum()),
                sentinel_fraction=round(1.0 - float(mine.float().mean()), 4), active_local_experts=int((counts > 0).sum()),
                rows_per_expert_mean=round(float(counts.mean()), 3), rows_per_expert_max=int(counts.max()))


def pattern(tokens: int, c: Cell, generator, device, *, min_local: int = 1):
    """(x bf16 [N, H], ids int32 [N, k], weights f32 [N, k]) with at least `min_local` routes on this rank."""
    while True:
        ids, weights = routes(tokens, c, generator, device)
        if int(local_mask(ids, c)[0].sum()) >= min_local:
            break
    x = (torch.randn(tokens, c.hidden, generator=generator, device=device) * X_STD).to(torch.bfloat16)
    return x, ids, weights


def pairs_of(x, ids, weights, c: Cell):
    """What an eager step dispatches (lanes.served()'s moe, compact=True): this rank's (token, route) pairs, one route a
    row, as local expert ids -> (token [P], x [P, H], ids int32 [P, 1], weights f32 [P, 1])."""
    from engine.profiles.qwen38.lanes import local_routes
    local, w = local_routes(ids, weights, c.first_expert, c.local)
    shifted = ids.to(torch.int32) - c.first_expert
    token, route = ((shifted >= 0) & (shifted < c.local)).nonzero(as_tuple=True)
    return token, x.index_select(0, token), local[token, route][:, None].contiguous(), w[token, route][:, None].contiguous()


# ---- experts --------------------------------------------------------------------------------------------------------

@dataclass
class Experts:
    """One layer's local experts in the rank-file layout (engine/profiles/qwen38/specs._routed_specs)."""
    w13: torch.Tensor          # [E, 2I, H/2] u8, rows [up; gate]
    w13_sf: torch.Tensor       # [E, 2I * H/16] e4m3, tile-interleaved (engine/modules/nvfp4_sf)
    w2: torch.Tensor           # [E, H, I/2] u8
    w2_sf: torch.Tensor        # [E, H * I/16] e4m3
    scales: object             # engine/modules/modelopt_scales.ModelOptScales

    def dense(self, e: int):
        """Local expert e's (up [I, H], gate [I, H], down [H, I]) FP32 weights, dequantised as expert_gemm does."""
        from engine.modules.moe import dequant_nvfp4
        from engine.modules.nvfp4_sf import unswizzle_sf
        two_i, hidden = self.w13.shape[1], self.w2.shape[1]
        inter = two_i // 2
        s13 = unswizzle_sf(self.w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
        s2 = unswizzle_sf(self.w2_sf[e].view(torch.uint8), hidden, inter // 16).view(torch.float8_e4m3fn)
        g13, g2 = self.scales.weight13[e], self.scales.weight2[e]
        return (dequant_nvfp4(self.w13[e, :inter], s13[:inter], g13), dequant_nvfp4(self.w13[e, inter:], s13[inter:], g13),
                dequant_nvfp4(self.w2[e], s2, g2))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13, self.w13_sf, self.w2, self.w2_sf))


def build_experts(c: Cell, generator, device) -> Experts:
    """`c.local` synthetic experts packed as the preshard packs Qwen3.8's: Gaussian BF16 weights through
    specs.nvfp4_from_bf16 (per-16 E4M3 scales under a global multiplier), scales tile-interleaved. Each expert's weights
    get their own spread and its FC1 input scale its own factor; the FC2 input scale is calibrated on what the expert's
    SwiGLU emits, as ModelOpt calibrates it."""
    from engine.modules.modelopt_scales import ModelOptScales
    from engine.modules.nvfp4_sf import swizzle_sf_batch
    from engine.profiles.qwen38.specs import nvfp4_from_bf16
    E, H, I = c.local, c.hidden, c.inter
    w13 = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=device)
    s13 = torch.empty(E, 2 * I, H // 16, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device=device)
    s2 = torch.empty(E, H, I // 16, dtype=torch.float8_e4m3fn, device=device)
    weight13 = torch.empty(E, dtype=torch.float32, device=device)
    weight2 = torch.empty(E, dtype=torch.float32, device=device)
    spread = 2 ** (torch.rand(E, 2, generator=generator, device=device) * 2 - 1)
    for e in range(E):
        first = torch.randn(2 * I, H, generator=generator, device=device) * (WEIGHT_STD * spread[e, 0])
        w13[e], s13[e], weight13[e] = nvfp4_from_bf16(first.to(torch.bfloat16))
        second = torch.randn(H, I, generator=generator, device=device) * (WEIGHT_STD * spread[e, 1])
        w2[e], s2[e], weight2[e] = nvfp4_from_bf16(second.to(torch.bfloat16))
    experts = Experts(w13, swizzle_sf_batch(s13.view(torch.uint8)).view(torch.float8_e4m3fn).contiguous(), w2,
                      swizzle_sf_batch(s2.view(torch.uint8)).view(torch.float8_e4m3fn).contiguous(),
                      SimpleNamespace(weight13=weight13, weight2=weight2))
    input13 = (8 * X_STD / E4M3_FP4) * 2 ** (torch.rand(E, generator=generator, device=device) * 0.5)
    calibration = torch.randn(CALIBRATION_TOKENS, H, generator=generator, device=device) * X_STD
    amax = torch.empty(E, dtype=torch.float32, device=device)
    for e in range(E):
        up, gate, _ = experts.dense(e)
        amax[e] = (torch.nn.functional.silu(calibration @ gate.T) * (calibration @ up.T)).abs().amax()
    input2 = amax * (1.25 / E4M3_FP4) * 2 ** (torch.rand(E, generator=generator, device=device) * 0.5)
    experts.scales = ModelOptScales.bind(weight13, input13.contiguous(), weight2, input2.contiguous(), experts=E,
                                         device=w13.device)
    return experts


# ---- oracles --------------------------------------------------------------------------------------------------------

def oracle(x, ids, weights, experts: Experts, c: Cell, quant):
    """This rank's routed partial of global routes `ids` [N, k]: engine/modules/moe.expert_gemm's dataflow over the local
    routes only -- dequantised NVFP4 weights, activations quantised per 16 by `quant` under the ModelOpt input scales
    (moe.quant_nvfp4_act divides exactly; the kernels' reciprocal sequence is engine_moe_real_check.hardware_quant) --
    rounded where the micro, static and dynamic kernels round: FC1 in FP32, silu(gate) * up rounded to BF16 before it is
    quantised for FC2 (their sC store), FC2 rounded to BF16, weighted and rounded again (scatter_add_bf16x2_to_f32),
    summed per token in FP32, route slot by slot, and rounded once. With moe.quant_nvfp4_act it is lanes.reference()'s
    moe on the same pairs, byte for byte (tests/test_probe_qwen38_moe.py)."""
    from engine.modules.moe import dequant_nvfp4_act
    silu = torch.nn.functional.silu
    mine, shifted = local_mask(ids, c)
    token, slot = mine.nonzero(as_tuple=True)
    expert = shifted[token, slot]
    gain = weights[token, slot].float()
    s = experts.scales
    contribution = torch.empty(token.numel(), x.shape[1], dtype=torch.float32, device=x.device)

    def gemm(values, weight, input_scale):
        packed, sf = quant(values, input_scale)
        return dequant_nvfp4_act(packed, sf, input_scale) @ weight.T

    for e in expert.unique().tolist():
        pick = (expert == e).nonzero(as_tuple=True)[0]
        up_w, gate_w, down_w = experts.dense(e)
        xe = x[token[pick]].float()
        up, gate = gemm(xe, up_w, s.input13[e]), gemm(xe, gate_w, s.input13[e])
        y = gemm((silu(gate) * up).bfloat16().float(), down_w, s.input2[e])
        contribution[pick] = (y.bfloat16().float() * gain[pick, None]).bfloat16().float()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for j in range(ids.shape[1]):
        route = (slot == j).nonzero(as_tuple=True)[0]
        out.index_add_(0, token[route], contribution[route])
    return out.bfloat16()


def reference_partial(x, ids, weights, experts: Experts, c: Cell, reference):
    """lanes.reference()'s moe (engine/modules/moe.expert_gemm, division rounding) on this rank's pairs, one route a row --
    what the eager step dispatches -- summed per token in FP32 and rounded once."""
    mine, _ = local_mask(ids, c)
    token, slot = mine.nonzero(as_tuple=True)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    if token.numel():
        pairs = reference.moe(x.index_select(0, token), ids[token, slot][:, None], weights[token, slot][:, None],
                              experts.w13, experts.w13_sf, experts.w2, experts.w2_sf, scales=experts.scales,
                              first_expert=c.first_expert)
        out.index_add_(0, token, pairs.float())
    return out.to(x.dtype)


def bucketed(quant):
    """`quant` over rows padded to a power of two, the padding cut away. The Triton quantiser takes its element count as
    a constexpr, so every new row count would compile; a zero row quantises alone and never touches another."""
    def run(values, scale):
        rows = values.shape[0]
        width = 1 << max(0, (rows - 1).bit_length())
        if width != rows:
            values = torch.cat([values, values.new_zeros(width - rows, values.shape[1])])
        packed, sf = quant(values, scale)
        return packed[:rows], sf[:rows]
    return run


def dispatch(x, ids, weights, experts: Experts, prepared, output=None):
    """The b12x call lanes.served()'s moe makes (its `dispatch`), on the weight views its moe_prepare built."""
    from engine.kernels.b12x import b12x_fused_moe
    views, sf13, sf2 = prepared
    s = experts.scales
    E = experts.w13.shape[0]
    output = torch.empty_like(x, memory_format=torch.contiguous_format) if output is None else output
    return b12x_fused_moe(x=x.contiguous(), output=output, w1_weight=experts.w13, w1_weight_sf=sf13, w2_weight=experts.w2,
                          w2_weight_sf=sf2, token_selected_experts=ids.contiguous(),
                          token_final_scales=weights.contiguous(), num_experts=E, num_local_experts=E, top_k=ids.shape[1],
                          w1_alpha=s.alpha13, w2_alpha=s.alpha2, fc2_input_scale=s.input2,
                          input_global_scale=s.input13, activation="silu", swiglu_alpha=1.0, swiglu_beta=0.0,
                          swiglu_limit=None, activation_precision="fp4", quant_mode="nvfp4", _weight_views=views)


# ---- comparisons and choices ----------------------------------------------------------------------------------------

def relative(a, b) -> float:
    """max |a - b| / max |b| over the whole tensor: the lanes' 2% metric (probes/engine_kernel_check.py)."""
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def _bf16_order(t):
    """BF16 values as integers in value order (+0 and -0 both 0): adjacent values differ by one."""
    bits = t.bfloat16().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


def bf16_ulps(a, b) -> int:
    """The largest distance between two tensors in adjacent BF16 values (probes/engine_modelopt_check.py)."""
    return int((_bf16_order(a) - _bf16_order(b)).abs().max())


def stable(a, b) -> bool:
    """Two runs of one launch agree, element by element: within one adjacent BF16 value (the kernels' FP32 sums have no
    fixed order, so a sum on a halfway case rounds either way) or within 0.1% of b's largest magnitude. This is
    probes/engine_modelopt_check.repeat_stable held per element: a one-value step at the largest element is 0.78% of it
    (BF16 keeps 7 mantissa bits) and a sign flip at a cancelling sum is thousands of values apart, and the two can meet
    in one tensor."""
    a, b = a.float(), b.float()
    near = (a - b).abs() <= REPEAT_RELATIVE * b.abs().max()
    adjacent = (_bf16_order(a) - _bf16_order(b)).abs() <= 1
    return bool((near | adjacent).all())


def repeat_diagnosis(x, ids, weights, experts, prepared, c, repeats: int = 4) -> dict:
    """Where an eager step's repeats part: the b12x call on this rank's pairs `repeats` times on identical inputs (the
    first is the reference), and the FP32 index_add sum `repeats` times over one call's pair outputs. For each, how many
    rows or tokens differ and by how much -- the call, the sum, or neither (then the parting was outside both)."""
    token, xp, idp, wp = pairs_of(x, ids, weights, c)
    calls = [dispatch(xp, idp, wp, experts, prepared).clone() for _ in range(repeats)]
    first = calls[0]

    def parted(tensors):
        rows = set()
        worst_ulps, worst_abs = 0, 0.0
        for t in tensors[1:]:
            differs = (t.view(torch.int16) != tensors[0].view(torch.int16)).view(t.shape[0], -1).any(dim=1)
            rows.update(differs.nonzero().flatten().tolist())
            worst_ulps = max(worst_ulps, bf16_ulps(t, tensors[0]))
            worst_abs = max(worst_abs, float((t.float() - tensors[0].float()).abs().max()))
        return dict(rows=len(rows), of=tensors[0].shape[0], first_rows=sorted(rows)[:8], ulps=worst_ulps,
                    max_abs=worst_abs)

    def summed():
        total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        total.index_add_(0, token, first.float())
        return total.bfloat16()
    sums = [summed() for _ in range(repeats)]
    duplicated = int((torch.bincount(token, minlength=x.shape[0]) > 1).sum())
    return dict(pairs=int(token.numel()), repeats=repeats, call=parted(calls), index_add=parted(sums),
                tokens_with_several_routes=duplicated)


def samples_summary(cold, warm) -> dict:
    return dict(cold_us=round(statistics.median(cold), 1), warm_us=round(statistics.median(warm), 1),
                cold_min_us=round(min(cold), 1), warm_min_us=round(min(warm), 1), samples=len(cold))


def fastest(rows, key: str = "cold_us"):
    """The fastest exact row by `key`, or None."""
    exact = [row for row in rows if row.get("exact")]
    return min(exact, key=lambda row: row[key]) if exact else None


def overall_tile(rows, key: str = "cold_us"):
    """One dynamic tile for every prefill shape (kernel_shape.MoE.dynamic_tile_m pins one): among the tiles timed exact at
    every shape, the one whose time over its shape's fastest exact tile has the smallest mean. None when no tile was."""
    by_tokens = {}
    for row in rows:
        if row.get("exact"):
            by_tokens.setdefault(row["tokens"], {})[row["tile_m"]] = row[key]
    if not by_tokens:
        return None
    common = set.intersection(*(set(times) for times in by_tokens.values()))
    if not common:
        return None
    score = {tile: statistics.mean(times[tile] / min(times.values()) for times in by_tokens.values()) for tile in common}
    return min(sorted(score), key=lambda tile: score[tile])


# ---- dispatcher instruments -----------------------------------------------------------------------------------------

@contextmanager
def pinned(module, **values):
    """Set module attributes (the dispatcher's probe hooks) for the block; restore what was there however it ends."""
    missing = [name for name in values if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{getattr(module, '__name__', module)} has no attribute {missing}")
    saved = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield module
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


# Where a launch record's fields sit in the dispatcher's kernel cache keys (moe_dispatch._micro_kernel_cache_key,
# _static_kernel_cache_key, _dynamic_kernel_cache_key); tests/test_probe_qwen38_moe.py builds keys with those functions.
KEY_FIELDS = {
    "micro": dict(m=4, max_rows=8, mac=9, tile=10, skip=17),
    "static": dict(m=5, max_rows=9, mac=10, tile=11),
    "dynamic": dict(mac=7, tile=8),
}
GETTERS = {"_get_micro_kernel": ("micro", "_MICRO_KERNEL_CACHE"), "_get_static_kernel": ("static", "_STATIC_KERNEL_CACHE"),
           "_get_static_kernel_v2": ("static_v2", "_STATIC_V2_KERNEL_CACHE"),
           "_get_dynamic_kernel": ("dynamic", "_DYNAMIC_KERNEL_CACHE"),
           "_get_direct_micro_kernel": ("direct_micro", "_DIRECT_MICRO_LAUNCH_CACHE")}


def kernel_fields(family: str, key) -> dict:
    record = dict(kernel=family)
    if key is None or family not in KEY_FIELDS:
        return record
    for name, index in KEY_FIELDS[family].items():
        value = key[index]
        record[name] = list(value) if isinstance(value, tuple) else value
    return record


class Launches:
    """Which compiled kernel each b12x launch took: the dispatcher's kernel getters wrapped for the block, the variant
    read back from the getter's own cache key -- the record that a pin took effect, and of what a check covered."""

    def __init__(self, md):
        self.md, self.log, self.saved = md, [], {}

    def __enter__(self):
        for name, (family, cache) in GETTERS.items():
            original = getattr(self.md, name)
            self.saved[name] = original
            setattr(self.md, name, self._wrap(original, family, cache))
        return self

    def __exit__(self, *exc):
        for name, original in self.saved.items():
            setattr(self.md, name, original)
        self.saved.clear()
        return False

    def _wrap(self, original, family, cache):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            key = next((k for k, v in getattr(self.md, cache).items() if v is result), None)
            self.log.append(kernel_fields(family, key))
            return result
        return wrapped

    def last(self) -> dict:
        return self.log[-1] if self.log else {}


def load(inputs, source) -> None:
    for dst, src in zip(inputs, source):
        dst.copy_(src)


# ---- the GPU run ----------------------------------------------------------------------------------------------------

class _Probe:
    def __init__(self, report, md, lanes, c: Cell, quant, device="cuda"):
        self.report, self.md, self.lanes, self.c, self.quant, self.device = report, md, lanes, c, quant, device
        self.checks = []
        self.unstable = []          # eager prefill checks whose repeats parted, with their diagnosis
        self.owners = []
        self.start = self.end = None

    def setup(self, shape):
        c, md = self.c, self.md
        self.generator = torch.Generator(device=self.device).manual_seed(SEED)
        sms = int(md.get_num_sm(torch.device("cuda")))
        self.base_mac = int(min(md.get_max_active_clusters(1), sms))
        self.report("device", name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                    sms=sms, base_mac=self.base_mac, shape=shape.describe(), cell=asdict(c),
                    decode_tokens=list(decode_tokens(c)), micro_tiles=list(MICRO_TILES), micro_macs=list(MICRO_MACS),
                    dynamic_tiles=list(DYNAMIC_TILES))
        began = time.perf_counter()
        self.layers = [build_experts(c, self.generator, self.device) for _ in range(LAYERS)]
        torch.cuda.synchronize()
        s = self.layers[0].scales
        self.report("experts", layers=LAYERS, seconds=round(time.perf_counter() - began, 1),
                    mib_per_layer=round(self.layers[0].nbytes() / 2**20, 1),
                    weight13=[float(s.weight13.min()), float(s.weight13.max())],
                    input13=[float(s.input13.min()), float(s.input13.max())],
                    weight2=[float(s.weight2.min()), float(s.weight2.max())],
                    input2=[float(s.input2.min()), float(s.input2.max())])
        torch.cuda.empty_cache()
        self.lane = self.lanes.served()
        self.reference = self.lanes.reference()
        self.prepared = [self.lane.moe_prepare(L.w13, L.w13_sf, L.w2, L.w2_sf, c.topk, scales=L.scales)
                         for L in self.layers]
        self.trash = torch.empty(TRASH_MIB * 2**20 // 4, device=self.device)
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def served(self, x, ids, weights, *, compact, layer=0):
        L = self.layers[layer]
        return self.lane.moe(x, ids, weights, L.w13, L.w13_sf, L.w2, L.w2_sf, scales=L.scales,
                             first_expert=self.c.first_expert, compact=compact)

    def timed(self, fn) -> float:
        self.start.record()
        fn()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1e3

    def fail(self, event, row, failures):
        if failures:
            raise AssertionError(f"{event}: {failures} ({json.dumps(row, default=str)[:2000]})")

    # -- decode ---------------------------------------------------------------------------------------------------------
    def decode_check(self, m: int) -> dict:
        """The dispatcher's own arm at m tokens -- the micro kernel with the zero-weight skip up to the micro cap, the
        static kernel above it (foreign routes at local expert 0, weight 0) -- captured, and held to the oracle, zeros,
        replay and repeats."""
        from probes.engine_decode_fusions import _capture
        c, md, experts, gen = self.c, self.md, self.layers[0], self.generator
        family = "micro" if m <= md._MICRO_MAX_TOKENS else "static"
        patterns = [pattern(m, c, gen, self.device, min_local=2)] + [pattern(m, c, gen, self.device)
                                                                for _ in range(DECODE_PATTERNS - 1)]
        x0, ids0, w0 = patterns[0]
        foreign = (x0, foreign_routes(ids0, c), w0)
        inputs = [t.clone() for t in patterns[0]]
        sentinel = md.ep_zero_weight_sentinel(num_tokens=m, num_topk=c.topk, experts=c.local, hidden_size=c.hidden,
                                              intermediate_size=c.inter, activation="silu", swiglu_limit=None)
        with Launches(md) as launches:
            graph, out = _capture(lambda: self.served(*inputs, compact=False))
            launched = launches.last()
            self.owners.append(self.lane.graph_resources())
            eager = self.served(x0, ids0, w0, compact=False).clone()
            again = self.served(x0, ids0, w0, compact=False).clone()
            eager1 = self.served(*patterns[1], compact=False).clone()
            zero = int(torch.count_nonzero(self.served(x0, ids0, torch.zeros_like(w0), compact=False)))
            foreign_eager = int(torch.count_nonzero(self.served(*foreign, compact=False)))
        graph.replay()
        replay = out.clone()
        load(inputs, patterns[1])
        graph.replay()
        replay1 = out.clone()
        load(inputs, foreign)
        graph.replay()
        foreign_replay = int(torch.count_nonzero(out))
        load(inputs, patterns[0])
        recip = oracle(x0, ids0, w0, experts, c, self.quant)
        division = reference_partial(x0, ids0, w0, experts, c, self.reference)
        if family == "micro":
            selector_tile = list(md._select_micro_mma_tiler_mn(
                state_E=c.local, weight_E=c.local, m=m, k=c.hidden, n=c.inter, num_topk=c.topk,
                skip_zero_weight_expert_id=sentinel, quant_mode="nvfp4", activation="silu", swiglu_alpha=1.0,
                swiglu_beta=0.0, swiglu_limit=None, max_rows=launched.get("max_rows")))
            selector_mac = md._select_micro_mac(m * c.topk, c.inter, self.base_mac, md._MICRO_MAC_LADDER)
        else:
            selector_tile = selector_mac = None
        row = dict(family=family, **route_stats(ids0, c), sentinel=sentinel,      # route_stats carries `tokens`
                   pool_sentinel_fraction=round(statistics.mean(route_stats(p[1], c)["sentinel_fraction"]
                                                                for p in patterns), 4),
                   launched=launched, selector_tile_m=selector_tile, selector_mac=selector_mac,
                   finite=bool(torch.isfinite(eager.float()).all()), output_absmax=float(eager.float().abs().max()),
                   oracle_relative=relative(eager, recip), reference_relative=relative(eager, division),
                   repeat_relative=relative(again, eager), repeat_ulps=bf16_ulps(again, eager),
                   repeat_stable=stable(again, eager),
                   replay_relative=relative(replay, eager), replay_ulps=bf16_ulps(replay, eager),
                   replay_stable=stable(replay, eager), replay_bytes_equal=bool(torch.equal(replay, eager)),
                   new_inputs_replay_relative=relative(replay1, eager1), new_inputs_replay_ulps=bf16_ulps(replay1, eager1),
                   new_inputs_replay_stable=stable(replay1, eager1),
                   zero_weights_nonzero=zero, all_foreign_nonzero=[foreign_eager, foreign_replay])
        failures = []
        if family == "micro":
            if sentinel != c.local:
                failures.append(f"the zero-weight sentinel is {sentinel}, not {c.local}: the EP micro skip is not engaged")
            if launched.get("kernel") != "micro" or launched.get("skip") != c.local:
                failures.append("the captured launch did not take the micro kernel with the sentinel skip")
        else:
            if sentinel is not None:
                failures.append(f"the zero-weight sentinel {sentinel} is engaged above the micro cap")
            if launched.get("kernel") != "static":
                failures.append(f"the captured launch took {launched.get('kernel')!r}, not the static kernel")
        if not row["finite"] or row["output_absmax"] == 0.0:
            failures.append("the output is not finite and nonzero")
        if row["oracle_relative"] > ORACLE_RELATIVE:
            failures.append(f"oracle {row['oracle_relative']:.4f} > {ORACLE_RELATIVE}")
        if not row["repeat_stable"]:
            failures.append("eager repeats differ")
        if not row["replay_stable"]:
            failures.append("the captured replay differs from eager")
        if not row["new_inputs_replay_stable"]:
            failures.append("the replay on new inputs differs from eager")
        if zero or foreign_eager or foreign_replay:
            failures.append("zero weights or all-foreign routes left nonzero output")
        row["passed"] = not failures
        self.report("decode_check", **row, failures=failures)
        self.checks.append(("decode", m, row["oracle_relative"], row["reference_relative"]))
        self.fail("decode_check", row, failures)
        return dict(tokens=m, family=family, patterns=patterns, foreign=foreign, inputs=inputs, oracle=recip, base=replay,
                    default=(launched.get("tile", [None])[0], launched.get("mac")), graph=graph, out=out,
                    launched=launched)

    def decode_sweep(self, d: dict) -> dict:
        """Every micro tile x MAC arm at d's shape captured and judged exact, then the exact ones timed."""
        from probes.engine_decode_fusions import _capture
        c, md, m = self.c, self.md, d["tokens"]
        arms = {d["default"]: dict(graph=d["graph"], out=d["out"], default=True, cold=[], warm=[])}
        for tile in MICRO_TILES:
            for mac in MICRO_MACS:
                if (tile, mac) in arms:
                    continue
                arm_row = dict(tokens=m, tile_m=tile, mac=mac)
                if mac > self.base_mac:
                    self.report("decode_arm", **arm_row, status="skipped",
                                reason=f"MAC above the {self.base_mac} clusters the device launches")
                    continue
                load(d["inputs"], d["patterns"][0])
                try:
                    with pinned(md, _MICRO_TILE_M_OVERRIDE=tile, _MICRO_MAC_OVERRIDE=mac), Launches(md) as launches:
                        graph, out = _capture(lambda: self.served(*d["inputs"], compact=False))
                        launched = launches.last()
                except Exception as exc:      # a variant that does not compile is a finding, not the end of the sweep
                    self.report("decode_arm", **arm_row, status="failed", error=repr(exc)[:600])
                    continue
                self.owners.append(self.lane.graph_resources())
                if (launched.get("kernel"), launched.get("tile"), launched.get("mac")) != ("micro", [tile, 128], mac):
                    raise AssertionError(f"decode_arm {arm_row}: the pinned launch took {launched}")
                graph.replay()
                got = out.clone()
                load(d["inputs"], d["foreign"])
                graph.replay()
                foreign_nonzero = int(torch.count_nonzero(out))
                load(d["inputs"], d["patterns"][0])
                arm_row.update(launched=launched, finite=bool(torch.isfinite(got.float()).all()),
                               oracle_relative=relative(got, d["oracle"]), default_relative=relative(got, d["base"]),
                               default_ulps=bf16_ulps(got, d["base"]), default_stable=stable(got, d["base"]),
                               default_bytes_equal=bool(torch.equal(got, d["base"])), all_foreign_nonzero=foreign_nonzero)
                exact = (arm_row["finite"] and arm_row["oracle_relative"] <= ORACLE_RELATIVE and foreign_nonzero == 0
                         and arm_row["default_stable"])
                self.report("decode_arm", **arm_row, status="exact" if exact else "inexact")
                if exact:
                    arms[(tile, mac)] = dict(graph=graph, out=out, default=False, cold=[], warm=[])
                else:
                    graph.reset()
        order = list(arms)
        for iteration in range(DECODE_ITERATIONS):
            for key in (order if iteration % 2 == 0 else order[::-1]):
                arm = arms[key]
                for p in d["patterns"]:
                    load(d["inputs"], p)
                    self.trash.zero_()
                    torch.cuda.synchronize()
                    arm["cold"].append(self.timed(arm["graph"].replay))
                    arm["warm"].append(self.timed(arm["graph"].replay))
        rows = []
        for (tile, mac), arm in arms.items():
            rows.append(self.report("decode_timing", tokens=m, tile_m=tile, mac=mac, default=arm["default"], exact=True,
                                    **samples_summary(arm["cold"], arm["warm"])))
            arm["graph"].reset()
        best = fastest(rows)
        default = next(row for row in rows if row["default"])
        return self.report("decode_best", tokens=m, tile_m=best["tile_m"], mac=best["mac"], cold_us=best["cold_us"],
                           warm_us=best["warm_us"], default_tile_m=default["tile_m"], default_mac=default["mac"],
                           default_cold_us=default["cold_us"], default_warm_us=default["warm_us"],
                           cold_speedup=round(default["cold_us"] / best["cold_us"], 3), arms_timed=len(rows))

    # -- prefill --------------------------------------------------------------------------------------------------------
    def prefill_check(self, tokens: int) -> None:
        """The eager lane at `tokens` with the dispatcher's own kernel: oracle, zeros, repeats."""
        c, md, experts = self.c, self.md, self.layers[0]
        x, ids, w = pattern(tokens, c, self.generator, self.device)
        stats = route_stats(ids, c)
        backend = md.select_sm120_moe_backend(num_tokens=stats["local_pairs"], num_topk=1, quant_mode="nvfp4",
                                              num_experts=c.local, num_local_experts=c.local, hidden_size=c.hidden,
                                              intermediate_size=c.inter, activation="silu", swiglu_limit=None)
        with Launches(md) as launches:
            out = self.served(x, ids, w, compact=True).clone()
            launched = launches.last()
            again = self.served(x, ids, w, compact=True).clone()
            zero = int(torch.count_nonzero(self.served(x, ids, torch.zeros_like(w), compact=True)))
            foreign = int(torch.count_nonzero(self.served(x, foreign_routes(ids, c), w, compact=True)))
        recip = oracle(x, ids, w, experts, c, self.quant)
        division = reference_partial(x, ids, w, experts, c, self.reference)
        diagnosis = repeat_diagnosis(x, ids, w, experts, self.prepared[0], c) if not stable(again, out) else None
        row = dict(**stats, backend=backend, launched=launched,
                   selector_tile_m=(md._select_dynamic_tile_m(stats["local_pairs"], c.local, "silu")
                                    if backend == "dynamic" else None),
                   finite=bool(torch.isfinite(out.float()).all()), output_absmax=float(out.float().abs().max()),
                   oracle_relative=relative(out, recip), reference_relative=relative(out, division),
                   repeat_relative=relative(again, out), repeat_ulps=bf16_ulps(again, out),
                   repeat_stable=stable(again, out), repeat_diagnosis=diagnosis, zero_weights_nonzero=zero,
                   all_foreign_nonzero=foreign)
        failures = []
        if launched.get("kernel") != backend:
            failures.append(f"the launch took {launched.get('kernel')}, the backend selector names {backend}")
        if not row["finite"] or row["output_absmax"] == 0.0:
            failures.append("the output is not finite and nonzero")
        if row["oracle_relative"] > ORACLE_RELATIVE:
            failures.append(f"oracle {row['oracle_relative']:.4f} > {ORACLE_RELATIVE}")
        if zero or foreign:
            failures.append("zero weights or all-foreign routes left nonzero output")
        # a repeat that differs is a finding the record carries (`unstable`), not a stop: the oracle held and the sweeps
        # after it time the same served kernel (measurements/qwen38_lane_20260917: the first run stopped here)
        row["passed"] = not failures
        row["unstable"] = not row["repeat_stable"]
        self.report("prefill_check", **row, failures=failures)
        self.checks.append(("prefill", tokens, row["oracle_relative"], row["reference_relative"]))
        if row["unstable"]:
            self.unstable.append(dict(tokens=tokens, backend=backend, repeat_ulps=row["repeat_ulps"],
                                      repeat_relative=row["repeat_relative"], diagnosis=diagnosis))
        self.fail("prefill_check", row, failures)

    def prefill_sweep(self, tokens: int) -> dict:
        """Every dynamic tile at `tokens`, judged exact on layer 0, then the exact ones timed over the layers."""
        c, md = self.c, self.md
        patterns = [pattern(tokens, c, self.generator, self.device) for _ in range(LAYERS)]
        pairs = [pairs_of(*p, c) for p in patterns]
        outputs = [torch.empty_like(p[1]) for p in pairs]
        widest = sorted(range(LAYERS), key=lambda L: -pairs[L][1].shape[0])   # its workspace serves the others
        recip = oracle(*patterns[0], self.layers[0], c, self.quant)

        def call(L):
            _, xp, idp, wp = pairs[L]
            return dispatch(xp, idp, wp, self.layers[L], self.prepared[L], outputs[L])

        def summed(L):
            total = torch.zeros(patterns[L][0].shape, dtype=torch.float32, device=self.device)
            total.index_add_(0, pairs[L][0], outputs[L].float())
            return total.bfloat16()

        with Launches(md) as launches:
            for L in widest:
                call(L)
            launched = launches.last()
        call(0)
        base = outputs[0].clone()
        stats = route_stats(patterns[0][1], c)
        if launched.get("kernel") != "dynamic":
            raise AssertionError(f"prefill_sweep at {tokens}: the eager dispatch took {launched}, not the dynamic kernel")
        default = launched["tile"][0]
        self.report("prefill_default", **stats, launched=launched,
                    selector_tile_m=md._select_dynamic_tile_m(stats["local_pairs"], c.local, "silu"),
                    oracle_relative=relative(summed(0), recip))
        arms = {}
        for tile in DYNAMIC_TILES:
            arm_row = dict(tokens=tokens, tile_m=tile, default=tile == default)
            try:
                with pinned(md, _DYNAMIC_TILE_M_OVERRIDE=tile), Launches(md) as launches:
                    for L in widest:
                        call(L)
                    launched = launches.last()
                    call(0)
                    got = outputs[0].clone()
                    call(0)
                    again = outputs[0].clone()
            except Exception as exc:
                self.report("prefill_arm", **arm_row, status="failed", error=repr(exc)[:600])
                continue
            if (launched.get("kernel"), launched.get("tile")) != ("dynamic", [tile, 128]):
                raise AssertionError(f"prefill_arm {arm_row}: the pinned launch took {launched}")
            arm_row.update(launched=launched, finite=bool(torch.isfinite(got.float()).all()),
                           oracle_relative=relative(summed(0), recip), default_relative=relative(got, base),
                           default_ulps=bf16_ulps(got, base), default_stable=stable(got, base),
                           default_bytes_equal=bool(torch.equal(got, base)), repeat_relative=relative(again, got),
                           repeat_ulps=bf16_ulps(again, got), repeat_stable=stable(again, got))
            exact = (arm_row["finite"] and arm_row["oracle_relative"] <= ORACLE_RELATIVE
                     and arm_row["default_stable"] and arm_row["repeat_stable"])
            if tile == default and not exact:
                raise AssertionError(f"prefill_arm {arm_row}: the dispatcher's own tile is not exact")
            self.report("prefill_arm", **arm_row, status="exact" if exact else "inexact")
            if exact:
                arms[tile] = dict(cold=[], warm=[])
        order = list(arms)
        for iteration in range(PREFILL_ITERATIONS):
            for tile in (order if iteration % 2 == 0 else order[::-1]):
                arm = arms[tile]
                with pinned(md, _DYNAMIC_TILE_M_OVERRIDE=tile):
                    for L in range(LAYERS):
                        self.trash.zero_()
                        torch.cuda.synchronize()
                        arm["cold"].append(self.timed(lambda: call(L)))
                        arm["warm"].append(self.timed(lambda: call(L)))
        rows = [self.report("prefill_timing", tokens=tokens, tile_m=tile, default=tile == default, exact=True,
                            local_pairs=[int(p[1].shape[0]) for p in pairs], **samples_summary(arm["cold"], arm["warm"]))
                for tile, arm in arms.items()]
        best = fastest(rows)
        base_row = next(row for row in rows if row["default"])
        self.report("prefill_best", tokens=tokens, tile_m=best["tile_m"], cold_us=best["cold_us"], warm_us=best["warm_us"],
                    default_tile_m=default, default_cold_us=base_row["cold_us"], default_warm_us=base_row["warm_us"],
                    cold_speedup=round(base_row["cold_us"] / best["cold_us"], 3), tiles_timed=len(rows))
        return dict(rows=rows, best=best, default=default)

    def summary(self, decode, prefill) -> None:
        rows = [row for result in prefill for row in result["rows"]]
        self.report("cell", experts=self.c.experts, experts_local=self.c.local, hidden=self.c.hidden,
                    inter_local=self.c.inter, topk=self.c.topk, activation="silu", quant="nvfp4",
                    oracle_max_relative=max(check[2] for check in self.checks),
                    reference_max_relative=max(check[3] for check in self.checks),
                    decode=[dict(tokens=b["tokens"], tile_m=b["tile_m"], mac=b["mac"], cold_us=b["cold_us"],
                                 default_tile_m=b["default_tile_m"], default_mac=b["default_mac"],
                                 cold_speedup=b["cold_speedup"]) for b in sorted(decode, key=lambda b: b["tokens"])],
                    prefill=[dict(tokens=r["best"]["tokens"], tile_m=r["best"]["tile_m"], cold_us=r["best"]["cold_us"],
                                  default_tile_m=r["default"]) for r in prefill],
                    dynamic_tile_m=overall_tile(rows), prefill_unstable=self.unstable,
                    not_measured="pack_stage_bytes: SF6 packing depends on the real scale bytes (srv2), not synthetic ones")


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))
        return row

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    from engine.base import kernel_shape as ks
    shape = ks.bind(kernel_shape())
    c = cell_of(shape)
    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.qwen38 import lanes
    from probes.engine_qwen38_moe_precision import Quant
    probe = _Probe(report, md, lanes, c, Quant(lanes.MOE_ACTIVATION_SCALE_SEARCH))
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False          # the oracles' GEMMs in exact FP32
    try:
        with torch.inference_mode(), pinned(md, _EP_ZERO_WEIGHT_MICRO_CELL=md._EP_ZERO_WEIGHT_MICRO_CELL,
                                            **{name: None for name in HOOKS}):
            probe.setup(shape)
            # correctness first -- every served default -- then the prefill tiles, then the micro variants
            decode = [probe.decode_check(m) for m in decode_order(decode_tokens(c), md._MICRO_MAX_TOKENS)]
            for tokens in () if os.environ.get("ST_PROBE_DECODE_TOKENS") else PREFILL_CHECKS:
                probe.prefill_check(tokens)              # a named decode run is about those shapes alone
            if os.environ.get("ST_PROBE_CHECKS_ONLY") == "1":
                # the correctness gates alone (a production window, not a lane ticket): no sweeps, no timings
                report("checks_only", decode=[d["tokens"] if isinstance(d, dict) else None for d in decode],
                       prefill=list(PREFILL_CHECKS), unstable=probe.unstable)
            else:
                prefill = [probe.prefill_sweep(tokens) for tokens in PREFILL_TIMINGS]
                # the tile x MAC arms are the micro kernel's; a static shape has its one served kernel
                decode_best = [probe.decode_sweep(d) for d in decode if d["family"] == "micro"]
                probe.summary(decode_best, prefill)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return events


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
