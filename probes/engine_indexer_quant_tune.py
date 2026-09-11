"""Bounded GB10 tuning of the indexer's exact Hadamard/FP8 transformation."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from engine.kernels.kpool import _fwht_quant_kernel


@triton.jit
def _warp_quant(q, out, scales, rows, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, 128)
    x = tl.load(q + r[:, None] * 128 + c[None, :], r[:, None] < rows, 0).to(tl.float32)
    for stage in tl.static_range(7):
        stride = 1 << stage
        other = tl.gather(x, tl.broadcast_to((c ^ stride)[None, :], (ROWS, 128)), 1)
        x = tl.where((c[None, :] & stride) == 0, x + other, other - x)
    x = (x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(maximum * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
    tl.store(out + r[:, None] * 128 + c[None, :], y, r[:, None] < rows)
    tl.store(scales + r, scale, r < rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.cuda.set_per_process_memory_fraction((512 * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(128)
    reports = []
    eviction = torch.zeros(16 * 2**20, device="cuda")
    for rows in (1, 16, 32, 64, 96, 192, 384, 768, 1024, 1536, 2048, 4096, 8192, 16384, 65536):
        q = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        configurations = [("baseline", 32, 2), ("warp1", 1, 1), ("warp4", 4, 4),
                          ("warp8", 8, 4), ("warp16", 16, 4), ("warp32", 32, 4), ("warp32w2", 32, 2),
                          ("batch1", 1, 1), ("batch4", 4, 4), ("batch4w1", 4, 1), ("batch8w1", 8, 1)]
        functions, outputs, compiled, graphs = {}, {}, {}, {}
        for name, tile, warps in configurations:
            output = torch.empty_like(q, dtype=torch.float8_e4m3fn)
            scales = torch.empty(rows, 1, device="cuda")
            def run(name=name, tile=tile, warps=warps, output=output, scales=scales):
                if name == "baseline" or name.startswith("batch"):
                    return _fwht_quant_kernel[(triton.cdiv(rows, tile),)](
                        q, output, scales, rows, BLOCK_R=tile, num_warps=warps)
                return _warp_quant[(triton.cdiv(rows, tile),)](
                    q, output, scales, rows, ROWS=tile, num_warps=warps)
            functions[name] = run
            outputs[name] = output, scales
            kernel = run()
            compiled[name] = {"registers": kernel.n_regs, "shared": kernel.metadata.shared}
        torch.cuda.synchronize()
        reference = outputs["baseline"]
        for name, (out, scale) in outputs.items():
            assert torch.equal(out.view(torch.uint8), reference[0].view(torch.uint8)), (rows, name, "fp8")
            assert torch.equal(scale.view(torch.int32), reference[1].view(torch.int32)), (rows, name, "scale")
        for name, run in functions.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32): run()
            graphs[name] = graph
        # Eviction precedes the entire 32-launch batch, not every kernel.
        samples = {name: {mode: [] for mode in ("warm", "batch_after_eviction")} for name in functions}
        for mode in ("warm", "batch_after_eviction"):
            for iteration in range(7):
                names = list(functions)
                if iteration % 2: names.reverse()
                for name in names:
                    if mode == "warm": graphs[name].replay()
                    else: eviction.add_(1)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[name].replay(); end.record(); end.synchronize()
                    samples[name][mode].append(start.elapsed_time(end) * 1000 / 32)
        row = {"rows": rows, "bits_exact": True, "compiled": compiled, "samples_us": samples,
               "median_us": {name: {mode: statistics.median(values) for mode, values in modes.items()}
                             for name, modes in samples.items()}}
        reports.append(row)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps({"rows": rows, "median_us": row["median_us"], "compiled": compiled}), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__": main()
