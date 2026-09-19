"""FlashInfer's FP4 GEMM and MoE kernels in the seed image, judged on one GB10 (sm_121a) against exact references
(probe, single-GPU lane).

    fp4_gemm  U5: the dense NVFP4 lane a checkpoint quantised through its attention projections and shared experts
              needs. The weight is ModelOpt's NVFP4 (e2m1 two per byte, even element low; an e4m3 scale per 16 along
              K; a fp32 weight_scale_2 multiplier), its scales handed over in FlashInfer's 128x4 interleave
              (block_scale_interleave). W4A4: flashinfer.mm_fp4 on each backend it names -- b12x (the SM120 dense
              kernel, gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py), cute-dsl (gemm_mm_fp4_cute_dsl.py's harness
              over the SM100 kernel), cutlass, cudnn, auto -- the activation quantised by flashinfer.nvfp4_quantize.
              W4A16: prepare_bf16_fp4_weights + mm_bf16_fp4 (gemm/gemm_bf16_fp4*.py; cute-dsl, cudnn). Every arm is
              held to the fp32 product of exactly the bytes it consumed (the weight, and for W4A4 the activation,
              dequantised back through the interleaved scales) and to torch.mm in bf16 on the unquantised weight, and
              timed beside that torch.mm and the engine's FP8 lane (FP8Linear, deep_gemm): (N, K) in GEMM_SHAPES,
              M in GEMM_ROWS
    fp4_moe   U3: MXFP4 experts (e2m1, one E8M0 scale per 32 along K), swiglu over [gate | up], MOE_CASES at
              MOE_TOKENS. CuteDslMxfp8Mxfp4MoEWrapper and cute_dsl_fused_moe_mxfp8_mxfp4
              (fused_moe/cute_dsl/fused_moe_mxfp8_mxfp4.py: MXFP8 activations from flashinfer.mxfp8_quantize x MXFP4
              weights -- the W4A8 the item asks for; gate/up rows interleaved in 64-row groups; declared SM100/103
              only) and B12xMoEWrapper / b12x_fused_moe with quant_mode="mxfp4" (b12x_moe.py, declared SM120/121;
              bf16 in, it quantises activations to MXFP4 itself -- W4A4-MX, not W4A8; w1 rows [up | gate]). Held to
              engine/modules/moe.apply_experts in fp32 on the experts dequantised back through the scales the kernels
              read, with the activation the kernel consumed (MXFP8) or bf16 and this probe's MXFP4 snap of it (b12x).
              Both kernels round the swiglu output again (MXFP8 / MXFP4) and the reference does not; a reference with
              the gate/up halves swapped says whether a convention is off

    python3 probes/engine_kernel_check.py --lanes sm121_fp4_gemm --output /cache/sm121-fp4-gemm.json
    python3 probes/engine_kernel_check.py --lanes sm121_fp4_moe --output /cache/sm121-fp4-moe.json

Each report carries the device capability, where every entry point is defined in the installed FlashInfer with the
capabilities it declares ("entries"), per-arm counts with the first error ("arms": what refused 12.1 and how), the
layout helpers' CPU self-check (`python3 probes/engine_sm121_fp4.py selfcheck` runs it alone) and GPU-side checks of
them against FlashInfer's own interleave, quantisers and dequantisers ("layout_checks"). Numbers, not a verdict: a
kernel that wins here is bound by a pull request that says so, and a speed claim on the served engine is the fleet's
(D17). An arm that fails records its error and the others still run; one that leaves the CUDA context unusable ends
the run with what it has ("aborted").
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.engine_sm121_candidates import _device, _error, _time, _write  # noqa: E402 -- stdlib-only at import

GEMM_SHAPES = {                                   # (N, K) of the weight
    "4096x4096": (4096, 4096),
    "12288x4096": (12288, 4096),
    "4096x12288": (4096, 12288),
    "2560x1536": (2560, 1536),
}
GEMM_ROWS = (1, 8, 16, 128, 1024, 8192)
W4A4_BACKENDS = ("b12x", "cutlass", "cudnn", "cute-dsl", "auto")      # mm_fp4's at f0922749
W4A16_BACKENDS = ("cute-dsl", "cudnn")                                 # mm_bf16_fp4's
MOE_CASES = {                                     # local experts, top-k, hidden, intermediate
    "E32 top8 h4096 i1536": (32, 8, 4096, 1536),
    "E8 top2 h4096 i2048": (8, 2, 4096, 2048),
}
MOE_TOKENS = (1, 16, 256, 4096)
MEMORY_CAP_GIB = 8
WEIGHT_STD = 0.02

FP4_MAX, FP8_MAX = 6.0, 448.0
NVFP4_GROUP, MX_GROUP = 16, 32
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)   # by 4-bit code

MIXED_ARGS = ("x", "x_sf", "token_selected_experts", "token_final_scales", "w1_weight", "w1_weight_sf", "w1_alpha",
              "w2_weight", "w2_weight_sf", "w2_alpha")
B12X_ARGS = ("x", "w1_weight", "w1_weight_sf", "w2_weight", "w2_weight_sf", "token_selected_experts",
             "token_final_scales", "w1_alpha", "w2_alpha", "fc2_input_scale")


# -- the layouts (torch only, any device) ---------------------------------------------------------------------------
def _pad(n: int, to: int) -> int:
    return (n + to - 1) // to * to


def e2m1_encode(x):
    """float -> e2m1 code (bit 3 the sign): nearest, ties to the even code, magnitudes past 6 saturating."""
    import torch
    grid = torch.tensor(E2M1[:8], device=x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    mag = x.float().abs().clamp(max=FP4_MAX)
    low, high = torch.bucketize(mag, mids), torch.bucketize(mag, mids, right=True)
    code = torch.where(low % 2 == 0, low, high)                      # the two differ only on a tie
    return (code | ((x < 0).long() << 3)).to(torch.uint8)


def pack_e2m1(codes):
    """[..., K] codes -> [..., K/2] bytes, the even element in the low nibble (ModelOpt, OCP MX, FlashInfer)."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(packed):
    """[..., K/2] bytes (uint8 or float4_e2m1fn_x2) -> [..., K] fp32."""
    import torch
    b = packed.view(torch.uint8)
    codes = torch.stack((b & 0xF, b >> 4), dim=-1).flatten(-2)
    return torch.tensor(E2M1, device=b.device)[codes.long()]


def quant_nvfp4(w):
    """ModelOpt's NVFP4 of a [N, K] weight -> (packed [N, K/2] uint8, weight_scale [N, K/16] e4m3, weight_scale_2
    fp32 []): weight_scale_2 = amax / (6 * 448), a 16-block's scale its amax / 6 / weight_scale_2 in e4m3."""
    import torch
    tiny = torch.finfo(torch.float32).tiny
    blocks = w.float().unflatten(-1, (w.shape[-1] // NVFP4_GROUP, NVFP4_GROUP))
    scale_2 = (w.float().abs().max() / (FP4_MAX * FP8_MAX)).clamp_min(tiny)
    scale = (blocks.abs().amax(-1) / FP4_MAX / scale_2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    step = (scale.float() * scale_2).clamp_min(tiny).unsqueeze(-1)
    return pack_e2m1(e2m1_encode((blocks / step).flatten(-2))), scale, scale_2


def dequant_nvfp4(packed, scale, scale_2):
    """[..., K] fp32 = e2m1 * scale[..., k // 16] * scale_2; `scale` linear (e4m3 or its bytes), at least K/16 wide."""
    import torch
    vals = unpack_e2m1(packed)
    s = scale.view(torch.float8_e4m3fn) if scale.dtype == torch.uint8 else scale
    s = s.float()[..., : vals.shape[-1] // NVFP4_GROUP].repeat_interleave(NVFP4_GROUP, dim=-1)
    return vals * s * torch.as_tensor(scale_2, dtype=torch.float32, device=vals.device)


def quant_mxfp4(w):
    """OCP MXFP4 of [..., K] -> (packed [..., K/2] uint8, E8M0 scale bytes [..., K/32]): a 32-block's scale is the
    smallest power of two that brings its amax to 6 or under, so nothing saturates (an all-zero block: 2^-127)."""
    import torch
    blocks = w.float().unflatten(-1, (w.shape[-1] // MX_GROUP, MX_GROUP))
    exp = torch.ceil(torch.log2(blocks.abs().amax(-1) / FP4_MAX)).clamp(-127, 127)
    codes = e2m1_encode((blocks / torch.exp2(exp).unsqueeze(-1)).flatten(-2))
    return pack_e2m1(codes), (exp + 127).to(torch.uint8)


def dequant_mxfp4(packed, scale):
    """[..., K] fp32 = e2m1 * 2^(scale byte - 127), one E8M0 scale per 32 along K (linear)."""
    import torch
    vals = unpack_e2m1(packed)
    s = torch.exp2(scale.view(torch.uint8).float() - 127)[..., : vals.shape[-1] // MX_GROUP]
    return vals * s.repeat_interleave(MX_GROUP, dim=-1)


def dequant_mxfp8(q, scale):
    """[..., K] fp32 = e4m3 * 2^(scale byte - 127), one E8M0 scale per 32 along K (linear)."""
    import torch
    s = torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(MX_GROUP, dim=-1)
    return q.float() * s[..., : q.shape[-1]]


def swizzle_128x4(scale):
    """[R, C] scale bytes -> FlashInfer's 128x4 interleave, flat, zero padded to ceil128(R) * ceil4(C): 512-byte tiles
    of 128 rows x 4 columns, (r, c) at ((r // 128) * ceil4(C) / 4 + c // 4) * 512 + (r % 32) * 16
    + ((r % 128) // 32) * 4 + c % 4 (gemm/gemm_bf16_fp4.py _unswizzle_sf_128x4; cute_dsl/utils.py
    convert_sf_to_mma_layout)."""
    import torch
    b = scale.view(torch.uint8)
    r, c = b.shape
    padded = torch.zeros(_pad(r, 128), _pad(c, 4), dtype=torch.uint8, device=b.device)
    padded[:r, :c] = b
    return padded.view(-1, 4, 32, padded.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def unswizzle_128x4(swizzled, rows, cols):
    """The inverse: [rows, cols] scale bytes out of a 128x4-interleaved buffer (any shape, read flat)."""
    import torch
    rp, cp = _pad(rows, 128), _pad(cols, 4)
    flat = swizzled.contiguous().view(torch.uint8).reshape(-1)[: rp * cp]
    return flat.view(rp // 128, cp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rp, cp)[:rows, :cols]


def mma_storage(view):
    """A convert_sf_to_mma_layout view -- (32, 4, m_tiles, 4, k_tiles, E) with strides (16, 4, k_tiles * 512, 1, 512,
    m_tiles * k_tiles * 512), what fused_moe_mxfp8_mxfp4._mxfp8_weight_scale_shape_and_strides demands -- back to its
    storage: [E, flat 128x4-interleaved bytes]."""
    return view.permute(5, 2, 4, 0, 1, 3).reshape(view.shape[5], -1)


def mma_to_linear(view, rows, cols):
    """[E, rows, cols] scale bytes out of a convert_sf_to_mma_layout view: what the grouped kernels read, linear."""
    import torch
    storage = mma_storage(view)
    return torch.stack([unswizzle_128x4(storage[e], rows, cols) for e in range(storage.shape[0])])


def interleave_up_gate(w, group=64):
    """[E, 2I, ...] rows [up | gate] -> 64-row groups alternating up, gate, up, ... (the CuTe-DSL gated GEMM1's B;
    tests/moe/test_cute_dsl_mxfp8_mxfp4_fused_moe.py _interleave_linear_and_gate)."""
    e, rows = w.shape[:2]
    return w.reshape(e, 2, rows // (2 * group), group, *w.shape[2:]).transpose(1, 2).reshape(w.shape)


def deinterleave_up_gate(w, group=64):
    e, rows = w.shape[:2]
    return w.reshape(e, rows // (2 * group), 2, group, *w.shape[2:]).transpose(1, 2).reshape(w.shape)


def selfcheck() -> dict:
    """The helpers above on CPU tensors of known values; an AssertionError names what is wrong."""
    import torch
    gen = torch.Generator().manual_seed(0)
    checked = {}

    assert unpack_e2m1(torch.tensor([0x21, 0xF8, 0x7F], dtype=torch.uint8)).tolist() == [0.5, 1.0, -0.0, -6.0,
                                                                                          -6.0, 6.0]
    assert torch.equal(e2m1_encode(torch.tensor(E2M1)), torch.tensor([*range(8), 0, *range(9, 16)], dtype=torch.uint8))
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.3, -100.0])
    assert e2m1_encode(ties).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 9, 15], e2m1_encode(ties).tolist()
    every = torch.arange(256, dtype=torch.uint8)
    low, high = every & 0x0F, every >> 4
    positive_zero = torch.where(low == 8, 0, low) | (torch.where(high == 8, 0, high) << 4)
    assert torch.equal(pack_e2m1(e2m1_encode(unpack_e2m1(every))), positive_zero)
    try:
        from engine.modules.quant import _unpack_fp4
        assert torch.equal(unpack_e2m1(every), _unpack_fp4(every))
        checked["e2m1 vs engine.modules.quant._unpack_fp4"] = "all 256 bytes"
    except ImportError as exc:
        checked["e2m1 vs engine.modules.quant._unpack_fp4"] = f"skipped: {exc}"
    checked["e2m1"] = "code table, low nibble first, nearest-even ties, saturation, byte round trip (-0 -> +0)"

    menu = torch.tensor([448.0, 1.0, 0.5, 2.5, 2.0 ** -9, 14.0, 208.0, 0.1015625])       # e4m3-exact; 2^-9 subnormal
    rows, k = 200, 64
    codes = torch.randint(0, 16, (rows, k), generator=gen, dtype=torch.uint8)
    codes[codes == 8] = 0                                            # -0 encodes as +0
    codes[:, ::NVFP4_GROUP] = 7                                      # every block holds a 6: its amax is 6 x its scale
    scale = menu[torch.randint(0, len(menu), (rows, k // NVFP4_GROUP), generator=gen)]
    scale[0, 0] = 448.0                                              # the global amax 6 x 448: weight_scale_2 == 1
    w = unpack_e2m1(pack_e2m1(codes)) * scale.repeat_interleave(NVFP4_GROUP, dim=1)
    packed, got_scale, scale_2 = quant_nvfp4(w)
    assert scale_2.item() == 1.0 and torch.equal(got_scale.float(), scale) and torch.equal(packed, pack_e2m1(codes))
    assert torch.equal(dequant_nvfp4(packed, got_scale, scale_2), w)
    assert torch.equal(dequant_nvfp4(packed, got_scale.view(torch.uint8), 0.5), w * 0.5)
    swz = swizzle_128x4(got_scale)
    assert swz.numel() == 256 * 4
    assert torch.equal(dequant_nvfp4(packed, unswizzle_128x4(swz, rows, k // NVFP4_GROUP), scale_2), w)
    checked["nvfp4"] = f"{rows}x{k} of known codes and e4m3 scales: quantise -> same bytes, dequant exact, via 128x4"

    odd = torch.randint(0, 256, (200, 7), generator=gen, dtype=torch.uint8)
    flat = swizzle_128x4(odd)
    assert flat.numel() == 256 * 8 and torch.equal(unswizzle_128x4(flat, 200, 7), odd)
    for r, c in ((0, 0), (31, 3), (32, 0), (97, 2), (133, 5), (199, 6)):
        assert flat[((r // 128) * 2 + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4] == odd[r, c]
    whole = unswizzle_128x4(flat, 256, 8)
    assert whole[200:].count_nonzero() == 0 and whole[:, 7].count_nonzero() == 0
    checked["128x4"] = "200x7 bytes: round trip, six hand-computed offsets, zero padding"

    codes = torch.randint(0, 16, (96, 128), generator=gen, dtype=torch.uint8)
    codes[codes == 8] = 0
    codes[:, ::MX_GROUP] = 15                                        # every block holds a -6
    exp = torch.randint(-20, 21, (96, 128 // MX_GROUP), generator=gen)
    w = unpack_e2m1(pack_e2m1(codes)) * torch.exp2(exp.float()).repeat_interleave(MX_GROUP, dim=1)
    packed, sb = quant_mxfp4(w)
    assert torch.equal(packed, pack_e2m1(codes)) and torch.equal(sb.long(), exp + 127)
    assert torch.equal(dequant_mxfp4(packed, sb), w) and torch.equal(dequant_mxfp4(packed, sb).bfloat16().float(), w)
    zp, zs = quant_mxfp4(torch.zeros(2, 64))
    assert zs.eq(0).all() and zp.eq(0).all() and dequant_mxfp4(zp, zs).eq(0).all()
    q = torch.linspace(-4, 4, 64).to(torch.float8_e4m3fn).reshape(1, 64)
    got = dequant_mxfp8(q, torch.tensor([[127, 130]], dtype=torch.uint8))
    assert torch.equal(got[0, :32], q.float()[0, :32]) and torch.equal(got[0, 32:], q.float()[0, 32:] * 8)
    checked["mxfp4 / mxfp8"] = "96x128 of known codes and E8M0 exponents: same bytes, dequant exact (and in bf16)"

    e, rows, cols, mt, kt = 3, 200, 10, 2, 3                         # 10 scale columns = K 320 at 32
    lin = torch.randint(0, 256, (e, rows, cols), generator=gen, dtype=torch.uint8)
    storage = torch.stack([swizzle_128x4(lin[x]) for x in range(e)])
    view = storage.view(e, mt, kt, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)      # convert_sf_to_mma_layout's view
    assert tuple(view.shape) == (32, 4, mt, 4, kt, e) and view.stride() == (16, 4, kt * 512, 1, 512, mt * kt * 512)
    assert torch.equal(mma_storage(view), storage) and torch.equal(mma_to_linear(view, rows, cols), lin)
    checked["mma view"] = "3 experts x 200x10: the validator's shape and strides, back to linear"

    w1 = torch.arange(512 * 3).reshape(1, 512, 3)                    # rows [up 0..255 | gate 256..511]
    il = interleave_up_gate(w1)
    assert torch.equal(il[0, :64], w1[0, :64]) and torch.equal(il[0, 64:128], w1[0, 256:320])
    assert torch.equal(il[0, 128:192], w1[0, 64:128]) and torch.equal(deinterleave_up_gate(il), w1)
    checked["gate/up interleave"] = "64-row groups up, gate, up, ...; inverse"
    return checked


# -- reporting --------------------------------------------------------------------------------------------------------
class _Lost(Exception):
    """The CUDA context did not survive a failure; the run stops with what it has."""


def _fail(report, key, exc):
    import torch
    report["unavailable"][key] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        torch.cuda.synchronize()
    except Exception as lost:                                        # noqa: BLE001
        raise _Lost(f"after {key}: {type(lost).__name__}: {lost}"[:300]) from exc


def _record(report, arm, key, exc) -> str:
    status = report["arms"].setdefault(arm, {"ran": 0, "failed": 0})
    status["failed"] += 1
    msg = f"{type(exc).__name__}: {exc}"[:300]
    status.setdefault("first_error", msg)
    _fail(report, key, exc)
    return msg


def _attempt(report, row, arm, where, make, strikes):
    """row[arm] = make(), or its failure under report["unavailable"][f"{arm} {where}"]. An arm that fails the same way
    twice running (`strikes`: arm -> (error, count)) is not tried again for the rest of the shape or case."""
    if strikes.get(arm, ("", 0))[1] >= 2:
        return
    try:
        row[arm] = make()
    except Exception as exc:                                         # noqa: BLE001 -- the failure is the answer
        last, count = strikes.get(arm, ("", 0))
        msg = f"{type(exc).__name__}: {exc}"[:300]
        strikes[arm] = (msg, count + 1 if msg == last else 1)
        _record(report, arm, f"{arm} {where}", exc)
        return
    report["arms"].setdefault(arm, {"ran": 0, "failed": 0})["ran"] += 1
    strikes.pop(arm, None)


def _judge(call, refs: dict, flops: int) -> dict:
    """The first call (compilation included), its finiteness and error against each reference, then the timing."""
    import torch
    began = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    row = {"first_call_s": round(time.perf_counter() - began, 2), "finite": bool(torch.isfinite(out).all())}
    row.update({f"vs_{name}": _error(out, ref) for name, ref in refs.items() if ref is not None})
    timed = _time(call)
    timed["TFLOPS"] = round(flops / timed["median_us"] / 1e6, 2)
    return {**row, **timed}


def _missing(fn, *names) -> list:
    """The argument names `fn` does not take (none when it takes **kwargs: it cannot be told from here)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return [name for name in names if name not in params]


def _guard(report, name, fn, *names):
    """Record where `fn` is defined and the capabilities it declares; fn when it takes every argument name this probe
    passes it, else None with the gap recorded as unsupported (no argument is guessed)."""
    if fn is None:
        return None
    entry = {"at": None, "declared_ccs": None, "missing_args": _missing(fn, *names)}
    try:
        target = inspect.unwrap(fn)
        entry["at"] = f"{(inspect.getsourcefile(target) or '').rsplit('/flashinfer/', 1)[-1]}:" \
                      f"{inspect.getsourcelines(target)[1]}"
    except (OSError, TypeError):
        pass
    ccs = getattr(fn, "_supported_ccs", None)
    entry["declared_ccs"] = sorted(ccs) if ccs else None
    report["entries"][name] = entry
    if entry["missing_args"]:
        report["unavailable"][name] = f"unsupported: {name} takes no {entry['missing_args']}"
        return None
    return fn


def _flashinfer(report, names) -> dict:
    """The named top-level FlashInfer exports that import here; each that does not is recorded."""
    try:
        import flashinfer
    except Exception as exc:                                         # noqa: BLE001 -- the failure is the answer
        report["unavailable"]["flashinfer"] = f"{type(exc).__name__}: {exc}"[:300]
        return {}
    report["flashinfer"] = getattr(flashinfer, "__version__", None)
    found = {}
    for name in names:
        try:
            found[name] = getattr(flashinfer, name)
        except Exception as exc:                                     # noqa: BLE001
            report["unavailable"][f"flashinfer.{name}"] = f"{type(exc).__name__}: {exc}"[:300]
    return found


def _selfcheck_status() -> str:
    import traceback
    try:
        selfcheck()
        return "ok"
    except Exception as exc:                                         # noqa: BLE001
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        return f"{type(exc).__name__} at line {line}: {exc}"[:300]


def _equal_fraction(a, b) -> float:
    import torch
    return round(float((a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).float().mean()), 4)


# -- U5: dense FP4 GEMM ------------------------------------------------------------------------------------------------
def _gemm_shape(report, fi, fp8_linear, label, n, k) -> dict:
    import torch
    checks = report["layout_checks"]
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") * WEIGHT_STD).to(torch.bfloat16)
    packed, scale, scale_2 = quant_nvfp4(w)                          # the checkpoint, ModelOpt's four tensors less one
    shape = {"n": n, "k": k, "weight_scale_2": scale_2.item(), "rows": {}}
    w_sf = swizzle_128x4(scale).view(_pad(n, 128), _pad(k // NVFP4_GROUP, 4))
    if "block_scale_interleave" in fi:
        try:
            theirs = fi["block_scale_interleave"](scale.view(torch.uint8)).view(torch.uint8).reshape(w_sf.shape)
            checks[f"{label} weight: block_scale_interleave == swizzle_128x4"] = torch.equal(theirs, w_sf)
            w_sf = theirs
        except Exception as exc:                                     # noqa: BLE001
            _fail(report, f"block_scale_interleave {label}", exc)
    w_exact = dequant_nvfp4(packed, unswizzle_128x4(w_sf, n, k // NVFP4_GROUP), scale_2)     # what the kernels read
    checks[f"{label} weight: read back through the interleave == the checkpoint"] = torch.equal(
        w_exact, dequant_nvfp4(packed, scale, scale_2))
    if "e2m1_and_ufp8sf_scale_to_float" in fi:
        try:
            r = min(n, 128)
            theirs = fi["e2m1_and_ufp8sf_scale_to_float"](
                packed[:r].cpu(), scale[:r].view(torch.uint8).cpu().reshape(-1), scale_2.reshape(1).cpu(),
                sf_vec_size=NVFP4_GROUP, ufp8_type=1, is_sf_swizzled_layout=False)
            checks[f"{label} dequant_nvfp4 vs e2m1_and_ufp8sf_scale_to_float, rows 0..{r}"] = _error(
                dequant_nvfp4(packed[:r], scale[:r], scale_2).cpu(), theirs.float().reshape(r, k))
        except Exception as exc:                                     # noqa: BLE001
            _fail(report, f"e2m1_and_ufp8sf_scale_to_float {label}", exc)
    if "nvfp4_quantize" in fi:
        try:
            q, s = fi["nvfp4_quantize"](w, (1.0 / scale_2).reshape(1), sfLayout=fi["SfLayout"].layout_linear,
                                        do_shuffle=False)
            checks[f"{label} quant_nvfp4 vs nvfp4_quantize: equal bytes (packed, scales)"] = [
                _equal_fraction(q, packed), _equal_fraction(s.view(torch.uint8).reshape(n, -1), scale)]
        except Exception as exc:                                     # noqa: BLE001
            _fail(report, f"nvfp4_quantize weight {label}", exc)

    w4a16 = {}
    if "prepare_bf16_fp4_weights" in fi and "mm_bf16_fp4" in fi:
        for backend in W4A16_BACKENDS:
            try:
                w4a16[backend] = fi["prepare_bf16_fp4_weights"](packed, w_sf, scale_2.reshape(1).float(),
                                                                backend=backend)
            except Exception as exc:                                 # noqa: BLE001
                _record(report, f"w4a16 {backend}", f"w4a16 {backend} prepare {label}", exc)
    try:
        fp8 = fp8_linear(w)
    except Exception as exc:                                         # noqa: BLE001
        _record(report, "fp8 deep_gemm (FP8Linear)", f"fp8 deep_gemm {label}", exc)
        fp8 = None

    strikes = {}
    for m in GEMM_ROWS:
        where = f"{label} M{m}"
        torch.manual_seed(m)
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        flops = 2 * m * n * k
        bf16 = torch.mm(x, w.T)
        row = {}
        _attempt(report, row, "torch.mm bf16", where, partial(_judge, partial(torch.mm, x, w.T), {}, flops), strikes)
        if fp8 is not None:
            _attempt(report, row, "fp8 deep_gemm (FP8Linear)", where,
                     partial(_judge, partial(fp8, x), {"bf16_mm": bf16}, flops), strikes)
        exact16 = x.float() @ w_exact.T
        for backend, (b, b_sf, alpha) in w4a16.items():
            call = partial(fi["mm_bf16_fp4"], x, b, b_sf, alpha, backend=backend, out_dtype=torch.bfloat16)
            _attempt(report, row, f"w4a16 {backend}", where,
                     partial(_judge, call, {"exact": exact16, "bf16_mm": bf16}, flops), strikes)
        del exact16
        if "nvfp4_quantize" in fi and "SfLayout" in fi:
            x_gsf = (FP4_MAX * FP8_MAX / x.float().abs().max()).reshape(1)
            quantize = partial(fi["nvfp4_quantize"], x, x_gsf, sfLayout=fi["SfLayout"].layout_128x4, do_shuffle=False)
            try:
                x_fp4, x_sf = quantize()
                row["nvfp4_quantize (activation)"] = _time(quantize)
            except Exception as exc:                                 # noqa: BLE001
                _fail(report, f"nvfp4_quantize {where}", exc)
                x_fp4 = None
            if x_fp4 is not None:
                x_lin = unswizzle_128x4(x_sf, m, k // NVFP4_GROUP)
                try:
                    q_lin, s_lin = fi["nvfp4_quantize"](x, x_gsf, sfLayout=fi["SfLayout"].layout_linear,
                                                        do_shuffle=False)
                    checks.setdefault("activation: nvfp4_quantize 128x4 read back == its layout_linear", {})[where] = (
                        torch.equal(q_lin.view(torch.uint8), x_fp4.view(torch.uint8))
                        and torch.equal(s_lin.view(torch.uint8).reshape(m, -1), x_lin))
                except Exception as exc:                             # noqa: BLE001
                    _fail(report, f"nvfp4_quantize layout_linear {where}", exc)
            if x_fp4 is not None and "mm_fp4" in fi:
                exact4 = dequant_nvfp4(x_fp4, x_lin, 1.0 / x_gsf) @ w_exact.T
                alpha = (scale_2 / x_gsf).float().reshape(1)
                for backend in W4A4_BACKENDS:
                    call = partial(fi["mm_fp4"], x_fp4, packed.T, x_sf, w_sf.T, alpha, torch.bfloat16, None,
                                   block_size=NVFP4_GROUP, use_8x4_sf_layout=False, backend=backend, use_nvfp4=True)
                    _attempt(report, row, f"w4a4 {backend}", where,
                             partial(_judge, call, {"exact": exact4, "bf16_mm": bf16}, flops), strikes)
                auto = getattr(fi["mm_fp4"], "suitable_auto_backends", None)
                if "w4a4 auto" in row and isinstance(auto, (list, tuple)):
                    row["w4a4 auto"]["candidates"] = list(auto)
                del exact4
        shape["rows"][m] = row
        print(json.dumps({where: row}), flush=True)
        del x, bf16
    gave_up = sorted(arm for arm, (_, count) in strikes.items() if count >= 2)
    if gave_up:
        shape["gave_up_after_two_identical_failures"] = gave_up
    return shape


def run_gemm(output=None) -> dict:
    import torch
    from engine.kernels.dense import FP8Linear
    report = {"lane": "sm121_fp4_gemm", "unavailable": {}, "arms": {}, "entries": {}, "layout_checks": {},
              "shapes": {}}
    _device(report, MEMORY_CAP_GIB)
    cc = report["device"]["capability"][0] * 10 + report["device"]["capability"][1]
    report["helpers_selfcheck"] = _selfcheck_status()
    torch.set_float32_matmul_precision("highest")                   # the references in true fp32
    fi = _flashinfer(report, ("SfLayout", "block_scale_interleave", "e2m1_and_ufp8sf_scale_to_float", "mm_fp4",
                              "nvfp4_quantize", "mm_bf16_fp4", "prepare_bf16_fp4_weights"))
    guards = {
        "mm_fp4": ("a", "b", "a_descale", "b_descale", "alpha", "out_dtype", "out", "block_size", "use_8x4_sf_layout",
                   "backend", "use_nvfp4"),
        "nvfp4_quantize": ("a", "a_global_sf", "sfLayout", "do_shuffle"),
        "prepare_bf16_fp4_weights": ("b", "b_descale", "alpha", "backend"),
        "mm_bf16_fp4": ("a", "b", "b_descale", "alpha", "backend", "out_dtype"),
        "e2m1_and_ufp8sf_scale_to_float": ("sf_vec_size", "ufp8_type", "is_sf_swizzled_layout"),
        "block_scale_interleave": (),
    }
    for name, args in guards.items():
        if _guard(report, name, fi.get(name), *args) is None:
            fi.pop(name, None)
    if "mm_fp4" in fi:
        supported = {}
        for backend in W4A4_BACKENDS[:-1]:
            try:
                supported[backend] = bool(fi["mm_fp4"].is_backend_supported(backend, cc))
            except Exception as exc:                                 # noqa: BLE001
                supported[backend] = f"{type(exc).__name__}: {exc}"[:120]
        report["entries"]["mm_fp4"][f"is_backend_supported(backend, {cc})"] = supported
    try:
        with torch.inference_mode():
            for label, (n, k) in GEMM_SHAPES.items():
                report["shapes"][label] = _gemm_shape(report, fi, FP8Linear, label, n, k)
                torch.cuda.empty_cache()
    except _Lost as lost:
        report["aborted"] = str(lost)
    print(_write(output, report), flush=True)
    return report


# -- U3: MXFP4 MoE -----------------------------------------------------------------------------------------------------
def _moe_checkpoint(e, h, i) -> dict:
    """Per expert gate, up [I, H] and down [H, I] in MXFP4 from bf16 masters N(0, WEIGHT_STD^2), seeded, one expert at
    a time: {name: (packed [E, rows, cols/2] uint8, E8M0 bytes [E, rows, cols/32])}."""
    import torch
    torch.manual_seed(0)
    out = {}
    for name, rows, cols in (("gate", i, h), ("up", i, h), ("down", h, i)):
        packed = torch.empty(e, rows, cols // 2, dtype=torch.uint8, device="cuda")
        scale = torch.empty(e, rows, cols // MX_GROUP, dtype=torch.uint8, device="cuda")
        for x in range(e):
            packed[x], scale[x] = quant_mxfp4((torch.randn(rows, cols, device="cuda") * WEIGHT_STD).to(torch.bfloat16))
        out[name] = packed, scale
    return out


def _mma_scales(report, fi, name, scale, k):
    """[E, R, K/32] E8M0 bytes -> the grouped kernels' MMA view: FlashInfer's 128x4 interleave per expert (checked
    against swizzle_128x4), then convert_sf_to_mma_layout(..., sf_vec_size=32). None when FlashInfer cannot."""
    import torch
    if "convert_sf_to_mma_layout" not in fi:
        return None
    e, r, _ = scale.shape
    storage = torch.stack([swizzle_128x4(scale[x]) for x in range(e)])
    if "block_scale_interleave" in fi:
        try:
            theirs = fi["block_scale_interleave"](scale.contiguous()).view(torch.uint8).reshape(e, -1)
            report["layout_checks"][f"{name}: block_scale_interleave == swizzle_128x4"] = torch.equal(theirs, storage)
            storage = theirs
        except Exception as exc:                                     # noqa: BLE001
            _fail(report, f"block_scale_interleave {name}", exc)
    try:
        return fi["convert_sf_to_mma_layout"](storage.reshape(e * _pad(r, 128), -1), m=r, k=k, num_groups=e,
                                              sf_vec_size=MX_GROUP)
    except Exception as exc:                                         # noqa: BLE001
        _fail(report, f"convert_sf_to_mma_layout {name}", exc)
        return None


def _moe_reference(apply_experts, gate_up, down, ids, weights, x, swapped=False):
    """engine/modules/moe.apply_experts in fp32 on the dequantised experts (gate_up rows [gate | up], the engine's)."""
    import torch

    def expert(e):
        w1 = gate_up[e].float()
        return (torch.cat(w1.chunk(2)[::-1]) if swapped else w1), down[e].float()
    return apply_experts(x.float(), ids, weights.float(), gate_up.shape[0], expert, "silu")


def _mxfp4_cross_check(report, fi):
    """This probe's MXFP4 reading against FlashInfer's own quantiser and dequantiser, on one bf16 tensor."""
    import torch
    if "mxfp4_quantize" not in fi:
        return
    checks = report["layout_checks"]
    torch.manual_seed(1)
    probe = (torch.randn(128, 4096, device="cuda") * WEIGHT_STD).to(torch.bfloat16)
    mine_p, mine_s = quant_mxfp4(probe)
    try:
        linear = fi["SfLayout"].layout_linear
        q, s = fi["mxfp4_quantize"](probe, sfLayout=linear)
        s = s.view(torch.uint8).reshape(128, -1)
        checks["mxfp4: quant_mxfp4 vs mxfp4_quantize, equal bytes (packed, scales)"] = [
            _equal_fraction(q, mine_p), _equal_fraction(s, mine_s)]
        if "mxfp4_dequantize" in fi:
            theirs = fi["mxfp4_dequantize"](q, s, sfLayout=linear)
            checks["mxfp4: dequant_mxfp4 vs mxfp4_dequantize, of mxfp4_quantize's bytes"] = _error(
                dequant_mxfp4(q, s).cpu(), theirs.float().cpu().reshape(128, -1))
        _, s128 = fi["mxfp4_quantize"](probe, sfLayout=fi["SfLayout"].layout_128x4)
        checks["mxfp4: mxfp4_quantize 128x4 read back == its layout_linear"] = torch.equal(
            unswizzle_128x4(s128, 128, 4096 // MX_GROUP), s)
    except Exception as exc:                                         # noqa: BLE001
        _fail(report, "mxfp4 cross-check", exc)


def _moe_case(report, fi, label, e, topk, h, i) -> dict:
    import torch
    from engine.modules.moe import apply_experts, route_softmax_topk
    checks = report["layout_checks"]
    case = {"experts": e, "top_k": topk, "hidden": h, "intermediate": i, "tokens": {}}
    ckpt = _moe_checkpoint(e, h, i)
    (gate_p, gate_s), (up_p, up_s), (down_p, down_s) = ckpt["gate"], ckpt["up"], ckpt["down"]
    # b12x: w1 rows [up | gate] (moe_dispatch._pad_intermediate_to_tile, tests/moe/utils.py); the CuTe-DSL pipeline:
    # those rows interleaved in 64-row groups; w2 the same bytes for both
    b_w1, b_w1_s = torch.cat([up_p, gate_p], 1).contiguous(), torch.cat([up_s, gate_s], 1).contiguous()
    del ckpt, gate_p, gate_s, up_p, up_s
    c_w1, c_w1_s = interleave_up_gate(b_w1), interleave_up_gate(b_w1_s)
    b_w1_sf = _mma_scales(report, fi, f"{label} b12x w1", b_w1_s, h)
    c_w1_sf = _mma_scales(report, fi, f"{label} cute-dsl w1", c_w1_s, h)
    w2_sf = _mma_scales(report, fi, f"{label} w2", down_s, i)

    # the reference experts: dequantised back from what b12x reads, rearranged to the engine's [gate | up]
    w1_read = mma_to_linear(b_w1_sf, 2 * i, h // MX_GROUP) if b_w1_sf is not None else b_w1_s
    w2_read = mma_to_linear(w2_sf, h, i // MX_GROUP) if w2_sf is not None else down_s
    checks[f"{label} b12x w1 scales read back == the checkpoint's"] = torch.equal(w1_read, b_w1_s)
    checks[f"{label} w2 scales read back == the checkpoint's"] = torch.equal(w2_read, down_s)
    if c_w1_sf is not None:
        checks[f"{label} cute-dsl w1 (bytes, scales read back) de-interleaved == b12x's"] = (
            torch.equal(deinterleave_up_gate(c_w1), b_w1)
            and torch.equal(deinterleave_up_gate(mma_to_linear(c_w1_sf, 2 * i, h // MX_GROUP)), b_w1_s))
    gate_up = torch.empty(e, 2 * i, h, dtype=torch.bfloat16, device="cuda")
    down = torch.empty(e, h, i, dtype=torch.bfloat16, device="cuda")
    for x in range(e):
        up_d, gate_d = dequant_mxfp4(b_w1[x], w1_read[x]).split(i)
        gate_up[x] = torch.cat([gate_d, up_d]).to(torch.bfloat16)   # exact: MXFP4 values are bf16 values
        down[x] = dequant_mxfp4(down_p[x], w2_read[x]).to(torch.bfloat16)
        if x == 0:
            checks[f"{label} expert 0 dequantised exactly in bf16"] = torch.equal(gate_up[0].float(),
                                                                                  torch.cat([gate_d, up_d]))
    del w1_read, w2_read

    arms = {}
    have_scales = b_w1_sf is not None and w2_sf is not None
    cls = fi.get("CuteDslMxfp8Mxfp4MoEWrapper")
    if cls is not None and c_w1_sf is not None and w2_sf is not None and "mxfp8_quantize" in fi:
        try:
            arms["cute-dsl mxfp8 x mxfp4 wrapper"] = cls(num_experts=e, top_k=topk, hidden_size=h,
                                                         intermediate_size=i).run
        except Exception as exc:                                     # noqa: BLE001
            _record(report, "cute-dsl mxfp8 x mxfp4 wrapper", f"CuteDslMxfp8Mxfp4MoEWrapper() {label}", exc)
    if "cute_dsl_fused_moe_mxfp8_mxfp4" in fi and c_w1_sf is not None and w2_sf is not None and "mxfp8_quantize" in fi:
        arms["cute-dsl mxfp8 x mxfp4 functional"] = partial(fi["cute_dsl_fused_moe_mxfp8_mxfp4"], num_experts=e,
                                                            top_k=topk)
    cls = fi.get("B12xMoEWrapper")
    if cls is not None and have_scales:
        try:
            arms["b12x mxfp4 wrapper"] = cls(num_experts=e, top_k=topk, hidden_size=h, intermediate_size=i,
                                             quant_mode="mxfp4", activation="silu").run
        except Exception as exc:                                     # noqa: BLE001
            _record(report, "b12x mxfp4 wrapper", f"B12xMoEWrapper() {label}", exc)
    if "b12x_fused_moe" in fi and have_scales:
        arms["b12x mxfp4 functional"] = partial(fi["b12x_fused_moe"], num_experts=e, top_k=topk, quant_mode="mxfp4",
                                                activation="silu")

    w1_alpha = torch.ones(e, dtype=torch.float32, device="cuda")
    w2_alpha = torch.ones(e, dtype=torch.float32, device="cuda")
    fc2_scale = torch.ones(1, dtype=torch.float32, device="cuda")   # ignored for mxfp4; the tests pass it
    strikes = {}
    for t in MOE_TOKENS:
        where = f"{label} T{t}"
        torch.manual_seed(t)
        x = torch.randn(t, h, device="cuda").to(torch.bfloat16)
        ids, weights = route_softmax_topk(torch.randn(t, e, device="cuda"), topk, normalize=True)
        ids32, w32 = ids.to(torch.int32).contiguous(), weights.float().contiguous()
        flops = 6 * t * topk * h * i
        ref = partial(_moe_reference, apply_experts, gate_up, down, ids, w32)
        refs = {"x_bf16": ref(x), "halves_swapped": ref(x, swapped=True)}
        row = {"routed_rows": t * topk}
        if "select_sm120_moe_backend" in fi:
            try:
                row["b12x_backend"] = fi["select_sm120_moe_backend"](num_tokens=t, num_topk=topk, quant_mode="mxfp4")
            except Exception as exc:                                 # noqa: BLE001
                _fail(report, f"select_sm120_moe_backend {where}", exc)
        mixed = {}
        if any(arm.startswith("cute-dsl") for arm in arms):
            try:
                xq, xsf = fi["mxfp8_quantize"](x, is_sf_swizzled_layout=False)
                xsf = xsf.view(torch.uint8).reshape(t, h // MX_GROUP)
                row["mxfp8_quantize (activation)"] = _time(partial(fi["mxfp8_quantize"], x,
                                                                   is_sf_swizzled_layout=False))
                mixed = dict(x=xq, x_sf=xsf, token_selected_experts=ids32, token_final_scales=w32, w1_weight=c_w1,
                             w1_weight_sf=c_w1_sf, w1_alpha=w1_alpha, w2_weight=down_p, w2_weight_sf=w2_sf,
                             w2_alpha=w2_alpha)
                mixed_refs = {"x_mxfp8": ref(dequant_mxfp8(xq, xsf)), **refs}
            except Exception as exc:                                 # noqa: BLE001
                _fail(report, f"mxfp8_quantize {where}", exc)
        b12x = dict(x=x, w1_weight=b_w1, w1_weight_sf=b_w1_sf, w2_weight=down_p, w2_weight_sf=w2_sf,
                    token_selected_experts=ids32, token_final_scales=w32, w1_alpha=w1_alpha, w2_alpha=w2_alpha,
                    fc2_input_scale=fc2_scale)
        b12x_refs = None
        for arm, fn in arms.items():
            if arm.startswith("cute-dsl"):
                if not mixed:
                    continue
                make = partial(_judge, partial(fn, **mixed), mixed_refs, flops)
            else:
                if b12x_refs is None:
                    b12x_refs = {**refs, "x_mxfp4_snap": ref(dequant_mxfp4(*quant_mxfp4(x)))}
                make = partial(_judge, partial(fn, **b12x), b12x_refs, flops)
            _attempt(report, row, arm, where, make, strikes)
        case["tokens"][t] = row
        print(json.dumps({where: row}), flush=True)
        del x, refs, b12x_refs, mixed
    gave_up = sorted(arm for arm, (_, count) in strikes.items() if count >= 2)
    if gave_up:
        case["gave_up_after_two_identical_failures"] = gave_up
    return case


def run_moe(output=None) -> dict:
    import torch
    report = {"lane": "sm121_fp4_moe", "unavailable": {}, "arms": {}, "entries": {}, "layout_checks": {},
              "cases": {}}
    _device(report, MEMORY_CAP_GIB)
    report["helpers_selfcheck"] = _selfcheck_status()
    torch.set_float32_matmul_precision("highest")                   # the references in true fp32
    fi = _flashinfer(report, ("SfLayout", "block_scale_interleave", "mxfp4_quantize", "mxfp4_dequantize",
                              "mxfp8_quantize", "CuteDslMxfp8Mxfp4MoEWrapper", "cute_dsl_fused_moe_mxfp8_mxfp4",
                              "B12xMoEWrapper", "b12x_fused_moe"))
    for name, module in (("convert_sf_to_mma_layout", "flashinfer.cute_dsl.utils"),
                         ("select_sm120_moe_backend", "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch")):
        try:
            fi[name] = getattr(__import__(module, fromlist=[name]), name)
        except Exception as exc:                                     # noqa: BLE001
            report["unavailable"][f"{module}.{name}"] = f"{type(exc).__name__}: {exc}"[:300]
    guards = {
        "mxfp4_quantize": ("a", "sfLayout"),
        "mxfp4_dequantize": ("a_fp4", "a_sf", "sfLayout"),
        "mxfp8_quantize": ("input", "is_sf_swizzled_layout"),
        "cute_dsl_fused_moe_mxfp8_mxfp4": (*MIXED_ARGS, "num_experts", "top_k"),
        "b12x_fused_moe": (*B12X_ARGS, "num_experts", "top_k", "quant_mode", "activation"),
        "convert_sf_to_mma_layout": ("sf", "m", "k", "num_groups", "sf_vec_size"),
        "select_sm120_moe_backend": ("num_tokens", "num_topk", "quant_mode"),
        "block_scale_interleave": (),
    }
    for name, args in guards.items():
        if _guard(report, name, fi.get(name), *args) is None:
            fi.pop(name, None)
    for name, init_args, run_args in (
            ("CuteDslMxfp8Mxfp4MoEWrapper", ("num_experts", "top_k", "hidden_size", "intermediate_size"), MIXED_ARGS),
            ("B12xMoEWrapper", ("num_experts", "top_k", "hidden_size", "intermediate_size", "quant_mode", "activation"),
             B12X_ARGS)):
        cls = fi.get(name)
        if cls is not None and not (_guard(report, f"{name}.__init__", cls.__init__, *init_args)
                                    and _guard(report, f"{name}.run", cls.run, *run_args)):
            fi.pop(name)
    try:
        with torch.inference_mode():
            _mxfp4_cross_check(report, fi)
            for label, (e, topk, h, i) in MOE_CASES.items():
                report["cases"][label] = _moe_case(report, fi, label, e, topk, h, i)
                torch.cuda.empty_cache()
    except _Lost as lost:
        report["aborted"] = str(lost)
    print(_write(output, report), flush=True)
    return report


if __name__ == "__main__":
    if sys.argv[1] == "selfcheck":
        print(json.dumps(selfcheck(), indent=1))
    else:
        {"gemm": run_gemm, "moe": run_moe}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else None)
