"""Price the FP8 packet-to-GEMM boundary; this is not a serving benchmark."""
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
    from engine.kernels.prefill_collectives.consumer import _quantize_gather
    kernel = triton.compile(ASTSource(_quantize_gather,
        {"Packed":"*fp8e4nv", "Scales":"*fp32", "Q":"*fp8e4nv", "S":"*fp32",
         "LOCAL_N":"i32", "PAYLOAD_BYTES":"i32"},
        constexprs={"K":4096, "G":32, "PACK_BLOCK":2048}),
        target=GPUTarget("cuda", 121, 32), options={"num_warps":4})
    assert not torch.cuda.is_initialized()
    return {"scope":"SM121 compilation only", "gpu_used":False,
            "torch":torch.__version__, "triton":triton.__version__, "status":"PASS",
            "cubin_sha256":hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
            "shared_bytes":kernel.metadata.shared}


def measure(samples):
    from engine.kernels.prefill_collectives.consumer import quantize_gather
    from tests.test_engine_prefill_fp8_consumer import PrefillConsumerTests
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("this experiment targets GB10/SM121")
    torch.cuda.set_per_process_memory_fraction((2 << 30)/torch.cuda.get_device_properties(0).total_memory)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PrefillConsumerTests))
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("packet consumer correctness must pass without skips")
    cases = []
    trash = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    for rows in (1024, 1728):
        received = PrefillConsumerTests.received(rows)
        graphs = []
        for fn in (PrefillConsumerTests.baseline, quantize_gather):
            fn(received, rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(received, rows)
            graphs.append(graph)
        try:
            for regime in ("warm", "evicted"):
                times = [[], []]
                for i in range(samples):
                    for arm in ((0, 1) if i % 2 == 0 else (1, 0)):
                        graphs[arm].replay()
                        if regime == "evicted":
                            trash.zero_()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                        times[arm].append(start.elapsed_time(end)*1000)
                medians = [statistics.median(t) for t in times]
                row = dict(rows=4*rows, hidden=4096, regime=regime,
                           baseline_us=medians[0], fused_us=medians[1],
                           change_pct=100*(medians[1]/medians[0]-1), samples_us=times)
                cases.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != "samples_us"}), flush=True)
        finally:
            for graph in graphs:
                graph.reset()
    return dict(scope="received packet to GEMM activation only, not communication or serving throughput",
                correctness=True, gpu_used=True, torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/cache/prefill-fp8-consumer.json"))
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 4:
        parser.error("at least four paired samples are required")
    report = compile_only() if args.compile_only else measure(args.samples)
    files = ("engine/kernels/prefill_collectives/consumer.py", "engine/kernels/prefill_collectives/kernels.py",
             "engine/kernels/dense/fp8.py", "tests/test_engine_prefill_fp8_consumer.py",
             "probes/engine_prefill_fp8_consumer_check.py")
    report["source_sha256"] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
