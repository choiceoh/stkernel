"""Qwen3.8's PLE injection on one GB10, component by component (probe, single-GPU lane `qwen38_ple_conv`).

The prefill census solves the PLE's share of a chunk from four small nets; beside a serving production its differences
are too noisy to judge a few ms by (measurements/qwen38_prefill_mixer_20260919: the census baseline 960 -> 1,632 ms).
This times the one piece engine/kernels/ngram_gate.conv_add replaced -- the dilated causal conv, its silu and the add
of the gated rows, engine/modules/causal_conv's torch form and an add -- against the one launch, at a prefill chunk's
rows and a decode step's, interleaved over many rounds; the minimum is the judge. Held first: the launch within two BF16
steps of the torch form (the conv's fp32 sum is the kernel's own order).

    python3 probes/engine_kernel_check.py --lanes qwen38_ple_conv --output /cache/qwen38-ple-conv.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HC, HIDDEN, K, DILATION = 4, 2560, 4, 3          # Qwen3.8's streams, the PLE conv's taps and its dilation (ngram_size)
ROWS = (16, 512, 4096)
CALLS = 8                                        # launches a graph
ROUNDS = 21


def run(output=None) -> dict:
    import torch
    from engine.kernels import ngram_gate
    from engine.modules.causal_conv import causal_conv1d
    torch.manual_seed(0)
    c = HC * HIDDEN
    weight = (torch.randn(c, K) * 0.3).to("cuda")                   # fp32, as served
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "rows": {}}
    for t in ROWS:
        normed = torch.randn(t, c, device="cuda", dtype=torch.bfloat16)
        gated = torch.randn(t, c, device="cuda", dtype=torch.bfloat16)
        held = torch.randn(c, (K - 1) * DILATION, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(gated)

        def torch_form():
            local, _ = causal_conv1d(normed, weight, None, held, "silu", dilation=DILATION)
            out.copy_(gated + local)
            return out

        def launch():
            return ngram_gate.conv_add(normed, gated, weight, held, DILATION, out=out)

        want = torch_form().clone()
        got = launch().clone()
        err = float((got.float() - want.float()).abs().max() / want.float().abs().max())
        differ = float((got != want).float().mean())
        graphs = {}
        for name, fn in (("torch conv + add", torch_form), ("conv_add", launch)):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    fn()
            graphs[name] = g
        times = {name: [] for name in graphs}
        for r in range(ROUNDS):
            for name, g in (graphs.items() if r % 2 == 0 else list(graphs.items())[::-1]):
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        report["rows"][t] = {"us_a_call": row, "max_err": round(err, 6), "differ": round(differ, 6)}
        print(json.dumps({f"ple conv rows {t}": {k: v["min"] for k, v in row.items()}, "max_err": err,
                          "differ": differ}), flush=True)
        del graphs, normed, gated, held, out
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
