"""engine/kernels/common/skinny_gemv against cuBLAS at the decode step's own shapes, on one GB10 (probe, single-GPU lane).

The decode step census (probes/engine_qwen38_step.py, 2026-09-19) put 11.2 ms of a C=1 step's 20.2 in BF16 GEMMs of 2 to
8 rows: the hyper-connection mixers' two a site (cuBLAS picks sm80 WMMA kernels, about 201 GB/s -- 6.2 ms), the router
(cuBLAS gemv, about 105 GB/s -- 1.2 ms). Every one of them is a weight read once for a handful of rows, so its floor is
its bytes over the memory's bandwidth (273 GB/s). This times the kernel's configurations (BLOCK_N, BLOCK_K, SPLIT, warps,
stages) against torch.mm at 1, 4, 8 and 16 rows -- a draft step at C=1, a K=3 verify at C=1, 2 and 4 -- both inside CUDA
graphs, interleaved A/B so production's own steps on the same GPU land on every arm, with the weights rotated over more
copies than the L2 holds (a step reads a mixer's weights once and 1,600 other launches between). The verdict a shape
reads is the configuration fastest summed over the rows, and its speedup at each row count.

    python3 probes/engine_kernel_check.py --lanes qwen38_gemv --output /cache/qwen38-gemv.json
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

ROWS = (1, 4, 8, 16)
COPIES_BYTES = 64 << 20        # weights rotated over at least this many bytes: more than GB10's L2
CALLS = 16                     # calls a graph (one per copy in turn)
ROUNDS = 9

# (outputs, K) of W [outputs, K] -> the configurations tried: (BLOCK_N, BLOCK_K, SPLIT, warps, stages)
SHAPES = {
    "router": ((513, 2560), ((32, 256, 1, 4, 3), (16, 256, 1, 4, 3), (16, 512, 1, 4, 2), (32, 256, 2, 4, 3),
                             (16, 256, 2, 4, 3), (16, 256, 5, 4, 2), (32, 256, 5, 4, 2), (16, 256, 10, 4, 1),
                             (32, 128, 4, 4, 3), (64, 256, 5, 4, 2), (16, 256, 5, 2, 2), (32, 256, 5, 8, 2))),
    "hc down": ((324, 10240), ((16, 256, 8, 4, 3), (16, 256, 10, 4, 3), (16, 256, 20, 4, 2), (16, 256, 40, 4, 1),
                               (32, 256, 10, 4, 3), (32, 256, 20, 4, 2), (16, 512, 10, 4, 2), (32, 128, 20, 4, 3),
                               (64, 256, 20, 4, 2), (16, 256, 10, 2, 3), (32, 256, 10, 8, 3), (16, 256, 4, 4, 3))),
    "hc up": ((10240, 320), ((64, 128, 1, 4, 3), (32, 64, 1, 4, 3), (64, 64, 1, 4, 3), (128, 64, 1, 4, 3),
                             (16, 512, 1, 4, 1), (32, 512, 1, 4, 1), (64, 512, 1, 4, 1), (128, 512, 1, 8, 1),
                             (32, 64, 1, 2, 3), (16, 64, 1, 2, 3), (64, 64, 1, 8, 3), (32, 128, 1, 4, 2))),
}


def run(output=None) -> dict:
    import torch
    from engine.kernels.common import skinny_gemv
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "shapes": {}}
    for label, ((n, k), configs) in SHAPES.items():
        nbytes = n * k * 2
        copies = max(CALLS, -(-COPIES_BYTES // nbytes))
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        shape = {"weight_MB": round(nbytes / 1e6, 2), "rows": {}}
        for m in ROWS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ weights[0].float().t())
            keep = []                                     # a graph writes its outputs' addresses: they live with it

            def graph_of(fn):
                outs = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(CALLS)]
                keep.append(outs)
                for i in range(CALLS):
                    fn(weights[i % copies], outs[i])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(CALLS):
                        fn(weights[(i * 7) % copies], outs[i])
                return g

            arms = {"cublas": graph_of(lambda w, o: torch.mm(x, w.t(), out=o))}
            errors = {}
            for config in configs:
                got = skinny_gemv.gemv(x, weights[0], config).float()
                again = skinny_gemv.gemv(x, weights[0], config).float()
                errors[str(config)] = (round(float((got - ref).abs().max() / ref.abs().max()), 6),
                                       bool(torch.equal(got, again)))
                arms[str(config)] = graph_of(lambda w, o, config=config: skinny_gemv.gemv(x, w, config, out=o))
            times = {name: [] for name in arms}
            for _ in range(ROUNDS):
                for name, g in arms.items():              # interleaved: contention lands on every arm alike
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - began) / CALLS * 1e6)
            del arms, keep
            row = {}
            for name, samples in times.items():
                us = statistics.median(samples)
                row[name] = {"us": round(us, 2), "GBps": round(nbytes / us / 1e3, 1)}
                if name in errors:
                    row[name].update(rel_err=errors[name][0], repeatable=errors[name][1])
            shape["rows"][m] = row
            best = min((c for c in row if c != "cublas"), key=lambda c: row[c]["us"])
            print(json.dumps({f"{label} rows {m}": {"cublas": row["cublas"], "best": best, **row[best],
                                                     "speedup": round(row["cublas"]["us"] / row[best]["us"], 3)}}),
                  flush=True)
        names = [str(c) for c in configs]
        chosen = min(names, key=lambda c: sum(shape["rows"][m][c]["us"] for m in ROWS))
        shape["chosen"] = chosen
        shape["speedup"] = {m: round(shape["rows"][m]["cublas"]["us"] / shape["rows"][m][chosen]["us"], 3) for m in ROWS}
        shape["module_config"] = str(skinny_gemv.CONFIGS.get((n, k)))
        print(json.dumps({label: {"chosen": chosen, "speedup": shape["speedup"],
                                  "module_config": shape["module_config"]}}), flush=True)
        report["shapes"][label] = shape
        del weights
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
