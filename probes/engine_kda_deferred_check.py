"""Judge deferred KDA state writes before changing the engine's state contract.

Runs GPU correctness first, then paired CUDA Graph timings for verify+commit.
This measures a kernel, not serving throughput. An engine integration needs
the TP4 onepass gate separately. Outputs are written under /cache by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import unittest

import torch

from engine.kernels.kda.deferred import verify, commit
from engine.kernels.kda.ring import recurrent_kda_ring


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("/cache/kda-deferred.json"))
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    if args.samples < 4:
        ap.error("at least four timing samples are required")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("the deferred-state experiment targets GB10/SM121")
    torch.cuda.set_per_process_memory_fraction((2 << 30) / torch.cuda.get_device_properties(0).total_memory)
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_kda_deferred")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("deferred-state correctness must pass without skips")
    torch.manual_seed(20260912)
    from tests.test_engine_kda_deferred import DeferredKdaTests
    fixture = DeferredKdaTests()
    trash = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    cases = []
    with torch.inference_mode():
        for t in (1, 6, 7):
            inputs, backing, _ = fixture.inputs(t)
            width = 16*128*128
            # Match production SPEC_K=6: seven dense state positions.
            ring = backing.as_strided((3, 7, 16, 128, 128),
                                      (7*width+64, width, 128*128, 128, 1), 64)
            baseline_backing = backing.clone()
            baseline_ring = baseline_backing.as_strided(ring.shape, ring.stride(), ring.storage_offset())
            initial = torch.randn((16, 128, 128), device="cuda")*.1
            slot, context, count = (torch.tensor(x, device="cuda") for x in (1, 4096, t))
            def old():
                return recurrent_kda_ring(*inputs, baseline_ring, slot, context, -5.)
            def new():
                out, factors = verify(*inputs, ring, slot, context, -5.)
                commit(factors, ring, slot, context, count, block=768)
                return out
            for fn in (old, new):
                fn()
            graphs = []
            for fn in (old, new):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs.append(graph)
            try:
                for accepted in sorted({1, min(3, t), t}):
                    count.fill_(accepted)
                    for ctx in (4096, 4607):  # no prefix mark, then a crossed 768-token boundary
                        context.fill_(ctx)
                        for regime in ("warm", "evicted"):
                            samples = [[], []]
                            for iteration in range(args.samples):
                                for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                                    graphs[arm].replay()
                                    # T==R overwrites the initial row, and the
                                    # two policies write different future rows.
                                    # Restore equal inputs outside the timer.
                                    (baseline_ring, ring)[arm][1, (ctx-1) % 7].copy_(initial)
                                    if regime == "evicted":
                                        trash.zero_()
                                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                                    samples[arm].append(start.elapsed_time(end)*1000)
                            medians = [statistics.median(s) for s in samples]
                            row = dict(tokens=t, ring_width=7, accepted=accepted, context=ctx, regime=regime,
                                       baseline_us=medians[0], deferred_us=medians[1],
                                       change_pct=100*(medians[1]/medians[0]-1), samples_us=samples)
                            cases.append(row)
                            print(json.dumps({k:v for k,v in row.items() if k != "samples_us"}), flush=True)
            finally:
                for graph in graphs:
                    graph.reset()
    files = ("engine/kernels/kda/deferred.py", "engine/kernels/kda/ring.py",
             "engine/kernels/kda/fused_recurrent.py", "probes/engine_kda_deferred_check.py",
             "tests/test_engine_kda_deferred.py")
    report = dict(scope="kernel verify plus accepted-state commit, not engine throughput", correctness=True,
                  torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  cases=cases, source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
