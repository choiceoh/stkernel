"""Compare GB10 query quantization with a Git-exported baseline and real L3 weights.

Raw kernel timings exclude allocations and Python launch overhead. The optional
real indexer comparison changes only the query quantizer in the served lane table.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from engine.kernels import kpool


def exact(a, b):
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)), "FP8 bytes"
    assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32)), "FP32 scale bits"


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_indexer_quant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arithmetic(path):
    names = {"_fwht_stage", "_fwht_quant_kernel"}
    return [ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--rank-file", type=Path)
    args = ap.parse_args()
    assert bool(args.checkpoint) == bool(args.rank_file)
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.cuda.set_per_process_memory_fraction((1536 * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    old = load_baseline(args.baseline)
    candidate = Path(kpool.__file__)
    assert arithmetic(args.baseline) == arithmetic(candidate), "quantization arithmetic changed"
    torch.manual_seed(20260911)
    report = {"torch": torch.__version__, "triton": kpool.triton.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "arithmetic_ast_identical": True,
              "protocol": {"rounds": 9, "warm_graph_launches": 32, "evicted_graph_launches": 1,
                           "eviction_bytes": 64 * 2**20, "order": "alternating AB/BA",
                           "existing_services_running": True, "gpu_clocks_locked": False}, "cases": []}
    eviction = torch.zeros(16 * 2**20, device="cuda")
    for rows in (1, 32, 192, 768, 1024, 1025, 1536, 2048, 4096, 8192,
                 16384, 32768, 65536, 65537, 131072):
        q = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        exact(old.fwht128_quant_fp8(q), kpool.fwht128_quant_fp8(q))
        functions, values, metadata, graphs = {}, {}, {}, {}
        for name, module, (tile, warps) in (("baseline", old, (32, 2)),
                                           ("optimized", kpool, kpool._fwht_quant_config(rows))):
            out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
            scales = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
            def run(module=module, tile=tile, warps=warps, out=out, scales=scales):
                return module._fwht_quant_kernel[((rows + tile - 1) // tile,)](
                    q, out, scales, rows, BLOCK_R=tile, num_warps=warps)
            kernel = run()
            values[name] = out, scales
            functions[name] = run
            metadata[name] = {"rows_per_cta": tile, "warps_per_cta": warps,
                              "grid_ctas": (rows + tile - 1) // tile,
                              "registers_per_thread": kernel.n_regs,
                              "shared_bytes_per_cta": kernel.metadata.shared}
            graphs[name] = {}
            for mode, count in (("warm", 32), ("evicted", 1)):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count): run()
                graphs[name][mode] = graph
        samples = {name: {mode: [] for mode in ("warm", "evicted")} for name in functions}
        for mode, count in (("warm", 32), ("evicted", 1)):
            for iteration in range(9):
                for name in (("baseline", "optimized") if iteration % 2 == 0 else ("optimized", "baseline")):
                    if mode == "warm": graphs[name][mode].replay()
                    else: eviction.add_(1)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[name][mode].replay(); end.record(); end.synchronize()
                    samples[name][mode].append(start.elapsed_time(end) * 1000 / count)
        for tick in range(4):
            if tick == 0: q.zero_()
            else: q.normal_(std=10. ** (tick - 2))
            for pair in graphs.values(): pair["evicted"].replay()
            exact(values["baseline"], values["optimized"])
        row = {"rows": rows, "bits_exact": True, "graph_replays_exact": 4, "metadata": metadata,
               "samples_us": samples,
               "median_us": {name: {mode: statistics.median(vals) for mode, vals in modes.items()}
                             for name, modes in samples.items()}}
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "samples_us"}), flush=True)
    if args.checkpoint:
        from dataclasses import replace
        from engine.profiles.glm53 import lanes
        from engine_indexer_lanes import real_indexer, threaded_dispatch
        ref, served = lanes.reference(), lanes.served()
        report["local_tp"] = threaded_dispatch(served)
        report["real_indexer"] = real_indexer(args.checkpoint, args.rank_file, ref, served,
            baseline_lanes=replace(served, indexer_quant=old.fwht128_quant_fp8), graph_measurements=True)
        report["rank_file_sha256"] = hashlib.sha256(args.rank_file.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("real_indexer", json.dumps(report["real_indexer"]), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__": main()
