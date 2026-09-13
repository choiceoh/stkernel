"""Matched KDA recurrence and commit; allocation is outside graph timing."""
import statistics

import torch


def measure(samples=12):
    from engine.kernels.kda.deferred import Batch
    from engine.kernels.kda.ring import recurrent_kda_ring_rows
    from engine.base.kernel_shape import bound
    from engine.profiles.glm53.facts import BLOCK, SPEC_K
    from tests.test_engine_kda_compact import compact_storage
    from tests.test_engine_kda_deferred_batch import fixture

    # Standalone measured GLM cell, without loading weights or checkpoint
    # metadata. The active serving K and prefix block come from the profile.
    shape = bound()
    if shape.linear is None or (shape.linear.v_heads, shape.linear.k_dim, shape.linear.v_dim) != (16, 128, 128):
        raise ValueError("this aggregate probe models GLM's 34 KDA layers on TP4")
    tokens, h, d, block, layers = SPEC_K+1, shape.linear.v_heads, shape.linear.k_dim, BLOCK, 34
    names = ("ordinary-ring", "deferred-ring", "compact-current-boundary")

    def one_width(rows):
        torch.manual_seed(91351+rows)
        args, backing, rings = fixture(rows, tokens, layers=layers, h=h, k=d, v=d, width=tokens)
        compact, current, boundary = compact_storage(rows, layers, h, d, d)
        deferred = Batch(rings, rows, tokens, block=block)
        owner = Batch(current, rows, tokens, block=block, boundaries=boundary)
        slots = torch.arange(1, rows+1, device="cuda")
        contexts = torch.full((rows,), 4096, device="cuda", dtype=torch.int64)
        counts = torch.ones_like(contexts)
        flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)

        def ordinary():
            return [recurrent_kda_ring_rows(*a, r, slots, contexts, -5.) for a, r in zip(args, rings)]

        def accepted(batch):
            out = [batch.verify(i, *a, slots, contexts, -5.) for i, a in enumerate(args)]
            batch.commit(slots, contexts, counts)
            return out

        graphs, outputs, result = [], [], []
        try:
            for fn in (ordinary, lambda: accepted(deferred), lambda: accepted(owner)):
                backing.zero_(); compact.zero_()
                fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs.append(fn())
                graphs.append(graph)
            for count in (1, min(3, tokens), tokens):
                counts.fill_(count)
                for context in (block*5+block//2, block*6-1):
                    contexts.fill_(context)
                    for arm, graph in enumerate(graphs):
                        backing.zero_(); compact.zero_()
                        graph.replay()
                        if arm == 0:
                            expected_outputs = [x.clone() for x in outputs[arm]]
                            expected_current = [x[1:rows+1, (context+count-1) % tokens].clone() for x in rings]
                            crossed = (context+count)//block*block
                            expected_boundary = ([x[1:rows+1, (crossed-1) % tokens].clone() for x in rings]
                                                 if crossed > context else None)
                        else:
                            for actual, expected in zip(outputs[arm], expected_outputs):
                                if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
                                    raise RuntimeError(f"{names[arm]} changed recurrence output")
                            actual_current = ([x[1:rows+1, (context+count-1) % tokens] for x in rings] if arm == 1
                                              else [x[1:rows+1, 0] for x in current])
                            for actual, expected in zip(actual_current, expected_current):
                                if not torch.equal(actual, expected):
                                    raise RuntimeError(f"{names[arm]} changed accepted state")
                            if expected_boundary is not None:
                                actual_boundary = ([x[1:rows+1, (crossed-1) % tokens] for x in rings] if arm == 1
                                                   else [x[1:rows+1] for x in boundary])
                                for actual, expected in zip(actual_boundary, expected_boundary):
                                    if not torch.equal(actual, expected):
                                        raise RuntimeError(f"{names[arm]} changed prefix boundary")
                    for cold in (False, True):
                        elapsed = [[] for _ in names]
                        for trial in range(samples):
                            for arm in ((0, 1, 2) if trial % 2 == 0 else (2, 1, 0)):
                                # Both layouts receive identical zero state;
                                # allocation, reset, compilation and eviction
                                # all precede the timed graph replay.
                                (compact if arm == 2 else backing).zero_()
                                if cold:
                                    flush.zero_()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record(); graphs[arm].replay(); end.record(); end.synchronize()
                                elapsed[arm].append(start.elapsed_time(end)*1000)
                        med = [statistics.median(x) for x in elapsed]
                        row = dict(rows=rows, layers=layers, tokens=tokens, accepted=count, context=context,
                                   cache="evicted-64MiB" if cold else "after-state-reset",
                                   medians_us=dict(zip(names, med)), samples_us=dict(zip(names, elapsed)),
                                   compact_change_vs_ordinary_pct=100*(med[2]/med[0]-1),
                                   compact_change_vs_deferred_pct=100*(med[2]/med[1]-1),
                                   recurrent_bytes=rows*layers*tokens*h*d*d*4,
                                   compact_current_boundary_bytes=rows*layers*2*h*d*d*4,
                                   allocated_ring_arena_bytes=backing.numel()*4,
                                   allocated_compact_arena_bytes=compact.numel()*4,
                                   deferred_workspace_bytes=deferred.nbytes, compact_workspace_bytes=owner.nbytes,
                                   exact=True)
                        result.append(row)
                        print({k: v for k, v in row.items() if k != "samples_us"}, flush=True)
        finally:
            for graph in graphs:
                graph.reset()
        return result

    cases = []
    for rows in (1, 4):
        cases.extend(one_width(rows))
    return dict(scope="34-layer FP32 KDA recurrence plus commit; no projection, conv, prefix-copy, NIC or model tok/s",
                spec_k=SPEC_K, block=block, variants=names, cases=cases)
