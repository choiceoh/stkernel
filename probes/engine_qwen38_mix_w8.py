"""Qwen C=1 mixer hypothesis: FP8 weights with the two existing BF16-row folds.

The previous hc_fp8 arm lost the folds and quantized activations. This prototype
instead widens each FP8 weight tile to BF16 inside the two kernels and scales
the dot's FP32 partial once per 128-wide block. Activations, gates and output
roundings follow gated_residual.mix_rows; accumulation order differs. Weight
quantization IS an arithmetic change: report its
drift separately from kernel error against the dequantized weight recipe.

Real rank-file mixers, rotated beyond L2; alternating CUDA graphs on one GB10.
This probe changes no serving path. End-to-end quality/TP4 speed remain unproven.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch

from engine.kernels import gated_residual as hcr
from engine.kernels.common import skinny_gemv as sg


from engine.kernels.gated_residual_w8 import mix


def pack(w):
    from engine.kernels.gated_residual_w8 import pack as pack_weight
    q, s = pack_weight(w)
    n, k = w.shape
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
