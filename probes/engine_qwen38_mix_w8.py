"""Qwen C=1 mixer hypothesis: FP8 weights with the two existing BF16-row folds.

The previous hc_fp8 arm lost the folds and quantized activations. This prototype
instead widens each block-scaled FP8 weight tile to BF16 inside the two kernels.
Activations, gates, dot tiles, split order and intermediate roundings follow
gated_residual.mix_rows. Weight quantization IS an arithmetic change: report its
drift separately from kernel error against the dequantized weight recipe.

Real rank-file mixers, rotated beyond L2; alternating CUDA graphs on one GB10.
This probe changes no serving path. End-to-end quality/TP4 speed remain unproven.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl

from engine.kernels import gated_residual as hcr
from engine.kernels.common import skinny_gemv as sg


@triton.jit
def _dot(X, W, WS, sx, sw, ss, rows, cols, M, N, k0, k1,
         BN: tl.constexpr, BK: tl.constexpr, FP32_DOT: tl.constexpr):
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k in range(k0, k1, BK):
        ks = k + tl.arange(0, BK)
        km = ks < k1
        x = tl.load(X + rows[:, None] * sx + ks[None, :],
                    mask=(rows[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sw + ks[None, :],
                    mask=(cols[:, None] < N) & km[None, :], other=0.0).to(tl.float32)
        scale = tl.load(WS + (cols[:, None] // 128) * ss + ks[None, :] // 128,
                        mask=(cols[:, None] < N) & km[None, :], other=0.0)
        w = (w * scale).to(tl.bfloat16)
        if FP32_DOT:
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    return acc


@triton.jit
def _down(X, W, WS, MIX, INJ, PART, LOCKS, M, N, K, sx, sw, ss, sm, si, HC_F,
          R: tl.constexpr, HC: tl.constexpr, INJECT: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, FP32_DOT: tl.constexpr):
    pn, pk = tl.program_id(0), tl.program_id(1)
    rows, cols = tl.arange(0, 16), pn * BN + tl.arange(0, BN)
    k0, k1 = sg.split_span(K, pk, SPLIT, BK)
    acc = _dot(X, W, WS, sx, sw, ss, rows, cols, M, N, k0, k1, BN, BK, FP32_DOT)
    if SPLIT > 1:
        total, last = sg.split_sum(acc, PART, LOCKS, pn, pk, rows, cols, M, N, SPLIT)
        if last:
            hcr._gate_store(total, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HC, INJECT)
    else:
        hcr._gate_store(acc, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HC, INJECT)


@triton.jit
def _up(G, W, WS, NORMED, OUT, M, sg_, sw, ss, sn, so, HC_F,
        HID: tl.constexpr, R: tl.constexpr, HC: tl.constexpr,
        BD: tl.constexpr, BK: tl.constexpr, FP32_DOT: tl.constexpr):
    rows = tl.arange(0, 16)
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    live = (rows[:, None] < M) & (d[None, :] < HID)
    acc = tl.zeros((16, BD), dtype=tl.float32)
    for s in tl.static_range(HC):
        u = _dot(G, W, WS, sg_, sw, ss, rows, s * HID + d, M, (s + 1) * HID,
                 0, R, BD, BK, FP32_DOT)
        g = tl.sigmoid(u.to(OUT.dtype.element_ty).to(tl.float32)).to(OUT.dtype.element_ty).to(tl.float32)
        n = tl.load(NORMED + rows[:, None] * sn + (s * HID + d)[None, :], mask=live, other=0.0).to(tl.float32)
        acc += (g * n).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + rows[:, None] * so + d[None, :], (acc / HC_F).to(OUT.dtype.element_ty), mask=live)


def mix(x, down, up, hc, rank, *, inject=True):
    """Packed weights (e4m3 matrix, block-128 scales); all row values remain BF16."""
    rows, width = x.shape
    if not 1 <= rows <= 16 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("requires 1..16 contiguous BF16 rows")
    n = rank + (hc if inject else 0)
    bn, bk, split, warps, stages = sg.CONFIGS[(n, width)]
    w, ws = down
    gates = torch.empty(rows, rank, device=x.device, dtype=x.dtype)
    inj = torch.empty(rows, hc, device=x.device, dtype=x.dtype) if inject else gates
    partial = torch.empty(split, rows, n, device=x.device, dtype=torch.float32)
    locks = sg.prepare(x.device)
    _down[(triton.cdiv(n, bn), split)](
        x, w, ws, gates, inj, partial, locks, rows, n, width, x.stride(0), w.stride(0), ws.stride(0),
        gates.stride(0), inj.stride(0), float(hc), R=rank, HC=hc, INJECT=inject,
        BN=bn, BK=bk, SPLIT=split, FP32_DOT=not x.is_cuda, num_warps=warps, num_stages=stages)
    out = torch.empty(rows, width // hc, device=x.device, dtype=x.dtype)
    w, ws = up
    bd, bk, warps, stages = hcr.UP_TILE
    _up[(triton.cdiv(width // hc, bd),)](
        gates, w, ws, x, out, rows, gates.stride(0), w.stride(0), ws.stride(0), x.stride(0), out.stride(0),
        float(hc), HID=width // hc, R=rank, HC=hc, BD=bd, BK=bk, FP32_DOT=not x.is_cuda,
        num_warps=warps, num_stages=stages)
    return out, inj if inject else None


def pack(w):
    from deep_gemm import per_block_cast_to_fp8
    n, k = w.shape
    padded = torch.nn.functional.pad(w, (0, -k % 128, 0, -n % 128))
    q, s = per_block_cast_to_fp8(padded.float(), use_ue8m0=True)
    q, s = q.contiguous(), s.contiguous()
    dequant = (q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()[:n, :k].contiguous()
    return (q, s), dequant


def error(a, b):
    return max(hcr.drift(got, ref)[0] for got, ref in zip(a, b) if got is not None)


def equal(a, b):
    return all(torch.equal(got, ref) for got, ref in zip(a, b) if got is not None)


def run(output=None, ranks=None):
    from engine.profiles.qwen38 import facts
    from engine.profiles.qwen38.fleet import rank_loader
    assert torch.cuda.get_device_capability() == (12, 1), "GB10 timing only"
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    rank_id = max(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
    F = facts.load(ranks)
    sg.prepare("cuda")
    sites = [(f"L{layer}.hc.{side}.", "down_inject") for layer in range(8) for side in ("attn", "mlp")]
    sites.append(("close.", "down"))
    names = [prefix + suffix for prefix, down in sites for suffix in (down, "up")]
    reader = rank_loader(ranks / f"rank{rank_id}of4.safetensors", expected_layout=F.weight_layout)
    weights = reader.load(names)
    pairs = []
    for prefix, down_name in sites:
        down, up = weights[prefix + down_name], weights[prefix + "up"]
        dq, df = pack(down)
        uq, uf = pack(up)
        pairs.append((down, up, dq, uq, df, uf, down_name == "down_inject"))
    torch.manual_seed(7)
    report = dict(device=torch.cuda.get_device_name(), rank=rank_id, sites=[p for p, _ in sites],
                  scope="single GPU mixer component; random activations, real weights; no quality verdict",
                  source_bytes=sum(d.numel() * 2 + u.numel() * 2 for d, u, *_ in pairs),
                  packed_bytes=sum(q.numel() + s.numel() * 4 for p in pairs for q, s in p[2:4]), rows={})
    trash = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    for rows in (1, 4, 8, 16):
        x = torch.randn(rows, F.hc * F.hidden, device="cuda", dtype=torch.bfloat16)
        checks = []
        # Two inputs at every distinct mixer (including the closing mixer's no-injection form).
        for seed in range(2):
            x.normal_()
            for down, up, dq, uq, df, uf, inject in pairs:
                got = mix(x, dq, uq, F.hc, F.hc_rank, inject=inject)
                recipe = hcr.mix_rows(x, df, uf, F.hc, inject=inject)
                original = hcr.mix_rows(x, down, up, F.hc, inject=inject)
                finite = all(bool(torch.isfinite(t).all()) for t in got if t is not None)
                recipe_error = error(got, recipe)
                checks.append(dict(recipe_error=recipe_error, recipe_equal=equal(got, recipe),
                                   quantization_error=error(got, original), finite=finite))
                if not finite or not recipe_error <= 2 ** -7:
                    raise AssertionError(f"mixer kernel failed recipe: {checks[-1]}")
        keep, graphs, expected = [], {}, {}
        for arm in ("bf16", "w8a16"):
            def call():
                outs = []
                for down, up, dq, uq, _, _, inject in pairs:
                    outs.append(hcr.mix_rows(x, down, up, F.hc, inject=inject) if arm == "bf16" else
                                mix(x, dq, uq, F.hc, F.hc_rank, inject=inject))
                return outs
            call()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs = call()
            g.replay()
            expected[arm] = [tuple(t.clone() if t is not None else None for t in pair) for pair in outs]
            keep.append(outs)
            graphs[arm] = g
        samples = {arm: [] for arm in graphs}
        for repeat in range(10):
            for arm in (tuple(graphs) if repeat % 2 == 0 else tuple(reversed(graphs))):
                trash.fill_(repeat)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(begin.elapsed_time(end) * 1000 / len(pairs))
        for (arm, g), outputs in zip(graphs.items(), keep):
            g.replay()
            if not all(equal(a, b) for a, b in zip(outputs, expected[arm])) or int(sg.prepare("cuda").abs().sum()):
                raise AssertionError(f"replay or arrival words changed: {arm}")
        row = dict(checks=checks, samples_us=samples,
                   median_us={a: statistics.median(s) for a, s in samples.items()})
        row["speedup"] = row["median_us"]["bf16"] / row["median_us"]["w8a16"]
        report["rows"][rows] = row
        print(json.dumps(dict(rows=rows, median_us=row["median_us"], speedup=row["speedup"],
                              max_recipe_error=max(c["recipe_error"] for c in checks),
                              max_quantization_error=max(c["quantization_error"] for c in checks))), flush=True)
        for g in graphs.values():
            g.reset()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report
