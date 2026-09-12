"""Judge terminal mHC contraction and direct five-layer feature packing.

The baseline is the served TileLang post followed by the actual Torch casts,
mean and concat. No model layers or communication are included in this timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import unittest

import torch


def compile_only():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.mhc_contract import _contract
    kernel = triton.compile(ASTSource(_contract,
        {"X": "*bf16", "Residual": "*bf16", "Post": "*fp32", "Comb": "*fp32",
         "Out": "*bf16", "OUT_STRIDE": "i64"}, constexprs={"H": 4096, "B": 512}),
        target=GPUTarget("cuda", 121, 32), options={"num_warps": 4})
    assert not torch.cuda.is_initialized()
    reduce_header = Path(torch.__file__).parent / "include/ATen/native/cuda/Reduce.cuh"
    return dict(scope="SM121 compilation only; no numerical or speed verdict", gpu_used=False,
                torch=torch.__version__, triton=triton.__version__, status="PASS",
                cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
                shared_bytes=kernel.metadata.shared,
                torch_reduce_header_sha256=hashlib.sha256(reduce_header.read_bytes()).hexdigest())


def measure(samples):
    from engine.kernels.mhc_contract import contract
    from tests.test_engine_mhc_contract import MhcContractTests
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("this experiment targets GB10/SM121")
    torch.cuda.set_per_process_memory_fraction((2 << 30) / torch.cuda.get_device_properties(0).total_memory)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MhcContractTests))
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("terminal contraction correctness must pass without skips")
    torch.cuda.empty_cache()
    cases = []
    trash = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    # Five observed layers, with full BF16 carries as the real producer hands
    # them out. One feature also prices the final hidden-state contraction.
    for rows, count in ((7, 1), (7, 5), (1728, 1), (1728, 5)):
        values = [MhcContractTests.inputs(rows, seed=i + 2) for i in range(count)]

        def baseline():
            parts = [MhcContractTests.baseline(*v) for v in values]
            return torch.cat(parts, dim=-1) if count > 1 else parts[0]

        def candidate():
            out = torch.empty((rows, count * 4096), device="cuda", dtype=torch.bfloat16)
            for i, inputs in enumerate(values):
                contract(*inputs, out=out[:, i * 4096:(i + 1) * 4096])
            return out

        graphs, outputs = [], []
        for fn in (baseline, candidate):
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            graphs.append(graph)
            outputs.append(out)
        try:
            for graph in graphs:
                graph.replay()
            if not torch.equal(outputs[0], outputs[1]):
                raise RuntimeError("timed graphs differ before measurement")
            output_hash = hashlib.sha256(outputs[0].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for regime in ("warm", "evicted"):
                times = [[], []]
                for sample in range(samples):
                    for arm in ((0, 1) if sample % 2 == 0 else (1, 0)):
                        graphs[arm].replay()
                        if regime == "evicted":
                            trash.zero_()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        times[arm].append(start.elapsed_time(end) * 1000)
                medians = [statistics.median(t) for t in times]
                row = dict(local_rows=rows, features=count, regime=regime,
                           baseline_us=medians[0], fused_us=medians[1],
                           change_pct=100 * (medians[1] / medians[0] - 1),
                           output_sha256=output_hash, samples_us=times)
                cases.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples_us"}), flush=True)
        finally:
            for graph in graphs:
                graph.reset()
        del graphs, outputs, values, out
        torch.cuda.empty_cache()
    return dict(scope="terminal contraction and feature assembly only; not model, RMSNorm, communication or serving",
                correctness=True, gpu_used=True, torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("/cache/mhc-contract.json"))
    args = parser.parse_args()
    if args.samples < 4:
        parser.error("at least four paired samples are required")
    report = compile_only() if args.compile_only else measure(args.samples)
    files = ("engine/kernels/mhc_contract.py", "engine/kernels/mhc/__init__.py",
             "engine/kernels/mhc/tilelang_kernels.py", "engine/profiles/glm53/net.py",
             "engine/profiles/glm53/drafter.py", "tests/test_engine_mhc_contract.py",
             "probes/engine_mhc_contract_check.py")
    report["source_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
