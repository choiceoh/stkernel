"""GPU-side cost of a dense projection's TP4 packets on one GB10: the W4 GEMM then the packet kernel's copy into its TX
slot, against the GEMM writing the reserved slot itself (GLM-5.3's direct producer, #826), same oracle build (probe,
single-GPU lane; carry X2's gate, the operator's decision of 2026-09-19).

GLM serves the direct producer by default with no speed claim (measurements/st_gb10_direct_io_20260913). Qwen3.8 would
take it at a new width (2560) only if it wins here by 1 us a sum or more. Both arms end in the oracle's fold, so what
differs is the packet grid's copy, fence and tickets on one side and a one-thread reservation, the GEMM's system-fenced
slot store and a one-thread publication on the other. The CPU lands the peers ahead of every chain (no RDMA, no skew,
engine_oneshot_consumer_timing's method); B/A/A/B, warm and after a 128 MiB eviction, at GLM's direct shapes.

    python3 probes/engine_kernel_check.py --lanes direct_producer_timing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N = 4096
COLS = (2048, 3072, 4096)          # GLM's direct writers: the KDA output, the MLP down projection, the MLA output
ROWS = (8, 16)                     # C=1 and C=2 at K=7
CHAIN, REPLAYS = 8, 32


def run(output=None) -> list:
    import torch
    from engine.kernels.dense import DenseLinear
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    ext.prepare(host, device, 1, 0, 0)
    cold = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(826)
    rows_out = []
    for cols in COLS:
        layer = DenseLinear((torch.randn(N, cols, device="cuda", generator=generator) * 0.02).bfloat16(), prefill=False)
        for rows in ROWS:
            if layer.slot_writer(rows) is None:
                raise RuntimeError(f"no direct writer at {rows} rows of {N}x{cols}")
            x = torch.randn(rows, cols, device="cuda", generator=generator).bfloat16()
            metadata = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
            peers = torch.randn(3, rows * N, device="cuda", generator=generator).bfloat16().view(torch.uint8).cpu()
            outs = {name: torch.empty(rows, N, device="cuda", dtype=torch.bfloat16) for name in ("gemm", "direct")}

            def gemm_then_packets(out=outs["gemm"]):
                ext.consume(ext.oneshot_packets(layer(x)), out)

            def direct(out=outs["direct"]):
                slot = ext.reserve_packets(metadata)
                layer._write_slot(x, slot)
                ext.consume(ext.publish_packets(metadata, slot), out)

            arms = {"gemm + packets": gemm_then_packets, "direct": direct}
            for cache in ("warm", "evicted"):
                graphs = {}
                try:
                    for name, step in arms.items():
                        start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))

                        def chain(step=step, start=start, end=end):
                            if cache == "evicted":
                                cold.fill_(19)
                            start.record()
                            for _ in range(CHAIN):
                                step()
                            end.record()

                        torch.cuda.synchronize()
                        ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                        chain()
                        torch.cuda.synchronize()
                        ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            chain()
                        graphs[name] = graph, start, end
                    samples = {name: [] for name in arms}
                    order = list(arms)
                    for block in range(4):
                        for name in (order if block % 2 == 0 else order[::-1]):
                            graph, start, end = graphs[name]
                            for _ in range(REPLAYS // 4):
                                torch.cuda.synchronize()
                                ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                                graph.replay()
                                end.synchronize()
                                samples[name].append(start.elapsed_time(end) * 1000.0 / CHAIN)
                    torch.cuda.synchronize()
                    row = {"shape": [N, cols], "rows": rows, "cache": cache, "chain": CHAIN,
                           "same_bytes": bool(torch.equal(outs["gemm"], outs["direct"])),
                           "us_a_sum": {name: {"median": round(median(v), 2), "min": round(min(v), 2)}
                                        for name, v in samples.items()}}
                    row["direct_saves_us_median"] = round(row["us_a_sum"]["gemm + packets"]["median"]
                                                          - row["us_a_sum"]["direct"]["median"], 2)
                    print(json.dumps({"direct_producer_timing": row}), flush=True)
                    rows_out.append(row)
                finally:
                    torch.cuda.synchronize()
                    for graph, *_ in graphs.values():
                        graph.reset()
    if ext.tickets(host) != ext.published(host) * 48:
        raise RuntimeError("publication tickets drifted during timing")
    if not all(row["same_bytes"] for row in rows_out):
        raise RuntimeError("the direct writer's packets differ from the GEMM's")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(rows_out, indent=1) + "\n")
    return rows_out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
