"""Paired 34-layer FP32 ring work, with identical state restored outside timing."""
import statistics

import torch


def measure(samples=12):
    from engine.kernels.kda.deferred import Batch
    from engine.kernels.kda.ring import recurrent_kda_ring_rows
    from tests.test_engine_kda_deferred_batch import fixture, views
    report = []
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    for rows in (1, 4):
        args, initial, initial_rings = fixture(rows, 7, layers=34)
        backing = [initial.clone(), initial.clone()]
        rings = [views(value, initial_rings) for value in backing]
        owner = Batch(rings[1], rows, 7, block=768)
        slots = torch.arange(1, rows+1, device="cuda")
        contexts = torch.full((rows,), 4096, device="cuda", dtype=torch.int64)
        counts = torch.full_like(contexts, 7)
        def ordinary():
            return [recurrent_kda_ring_rows(*a, r, slots, contexts, -5.) for a, r in zip(args, rings[0])]
        def deferred():
            out = [owner.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
            owner.commit(slots, contexts, counts)
            return out
        graphs, outputs = [], []
        try:
            for fn in (ordinary, deferred):
                fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = fn()
                graphs.append(graph)
                outputs.append(out)
            for accepted in (1, 3, 7):
                counts.fill_(accepted)
                for context in (4096, 4607):
                    contexts.fill_(context)
                    for arm in (0, 1):
                        backing[arm].copy_(initial)
                        graphs[arm].replay()
                    for actual, expected in zip(outputs[1], outputs[0]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for left, right in zip(rings[0], rings[1]):
                        for pos in range(context, context+accepted):
                            if pos == context+accepted-1 or (pos+1) % 768 == 0:
                                torch.testing.assert_close(left[1:rows+1, pos % 7], right[1:rows+1, pos % 7], rtol=0, atol=0)
                    for cold in (False, True):
                        elapsed = [[], []]
                        for trial in range(samples):
                            for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                                backing[arm].copy_(initial)
                                if cold:
                                    flush.zero_()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                                elapsed[arm].append(start.elapsed_time(end)*1000)
                        med = [statistics.median(a) for a in elapsed]
                        row = dict(rows=rows, layers=34, tokens=7, accepted=accepted, context=context,
                                   cache="evicted-64MiB" if cold else "warm", ordinary_us=med[0], deferred_us=med[1],
                                   change_pct=100*(med[1]/med[0]-1), factor_bytes=owner.nbytes, samples_us=elapsed)
                        report.append(row)
                        print({k: v for k, v in row.items() if k != "samples_us"}, flush=True)
        finally:
            for graph in graphs:
                graph.reset()
        # Release C=1's arena before allocating the C=4 pair. Both graphs own
        # full model state arenas, so letting loop locals linger costs GiBs.
        del owner, rings, backing, args, initial_rings, initial, outputs, graphs, ordinary, deferred, fn, out, left, right
    return dict(scope="34-layer KDA verify plus accepted-state commit, not model throughput", cases=report)
