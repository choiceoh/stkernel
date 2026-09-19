"""Qwen3.8's router projection on one GB10 (probe, single-GPU lane `qwen38_router`).

engine/kernels/router_fp32 (#1286) projects the router in IEEE FP32 at every row count -- the BF16 rows widened to FP32
and multiplied by the FP32 router with TF32 off, so prefill and decode score alike. A decode step's handful of rows is
cheap either way; a prefill chunk's 4,096 rows are 10.7 GFLOP a layer on the SIMT FP32 units, 48 layers and the MTP
head's a chunk. This times that projection against the BF16 one it replaced (the router and the shared gate as one
[513, 2560] weight, cuBLAS's) and the tensor-core one (router_logits_mma over the BF16 gates) at a decode step's rows and
a prefill chunk's, interleaved over many rounds; and holds the MMA router to the IEEE one and to itself across row
counts (a row's bits in a decode launch and a prefill launch).

    python3 probes/engine_kernel_check.py --lanes qwen38_router --output /cache/qwen38-router.json
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

HIDDEN, EXPERTS = 2560, 512
ROWS = (16, 512, 4096)
CALLS = 8
ROUNDS = 21


def run(output=None) -> dict:
    import torch
    from engine.kernels import router_fp32
    torch.manual_seed(0)
    gates = torch.randn(EXPERTS + 1, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    router = gates[:EXPERTS].float().contiguous()
    router_fp32.build()
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "rows": {}}
    for m in ROWS:
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        bf16_router = gates[:EXPERTS]
        arms = {"bf16 mm [513] (before #1286)": lambda: torch.mm(x, gates.t()),
                "bf16 mm [512] fp32 out": lambda: torch.mm(x, bf16_router.t(), out_dtype=torch.float32),
                "fp32 router (router_fp32)": lambda: router_fp32.router_logits(x, router),
                "mma router (router_logits_mma, bf16 gates)": lambda: router_fp32.router_logits_mma(x, bf16_router),
                "widen only (x.float())": lambda: x.float()}
        ieee = router_fp32.router_logits(x, router)
        mma = router_fp32.router_logits_mma(x, bf16_router)
        exact = (x.double() @ router.double().t())
        checks = {"mma_vs_ieee_max": float((mma - ieee).abs().max()),
                  "mma_vs_fp64_max": float((mma.double() - exact).abs().max()),
                  "ieee_vs_fp64_max": float((ieee.double() - exact).abs().max()),
                  "mma_rows_alike": all(bool(torch.equal(router_fp32.router_logits_mma(x[:r].contiguous(), bf16_router),
                                                         mma[:r])) for r in (1, 4, 16) if r <= m),
                  "ieee_rows_alike": all(bool(torch.equal(router_fp32.router_logits(x[:r].contiguous(), router),
                                                          ieee[:r])) for r in (1, 4, 16) if r <= m),
                  "mma_top8_is_ieee_top8": bool(torch.equal(mma.topk(8).indices.sort(1).values,
                                                            ieee.topk(8).indices.sort(1).values))}
        graphs = {}
        for name, fn in arms.items():
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            keep = []
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    keep.append(fn())
            graphs[name] = (g, keep)
        times = {name: [] for name in graphs}
        for r in range(ROUNDS):
            for name, (g, _) in (graphs.items() if r % 2 == 0 else list(graphs.items())[::-1]):
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        report["rows"][m] = {"us_a_call": row, "checks": checks}
        print(json.dumps({f"router rows {m}": {k: v["min"] for k, v in row.items()}, "checks": checks}), flush=True)
        del graphs
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
