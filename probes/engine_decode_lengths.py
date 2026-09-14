"""Captured native length reuse, changed request order and rollback; no timing."""
from types import SimpleNamespace as NS

import torch

from engine.kernels.indexer import row_lengths
from engine.profiles.glm53.decode_graphs import GraphCaches
from probes.engine_decode_fusions import _capture


def check(report):
    for n, t, split in ((1, 1, False), (4, 1, False), (1, 8, False), (2, 8, False),
                        (3, 8, False), (4, 8, False), (4, 1, True), (4, 8, True)):
        contexts = torch.zeros(n, dtype=torch.int64, device='cuda')
        ids = torch.arange(n, dtype=torch.int64, device='cuda')
        real = NS(F=NS(kpool=4), layout=None, block_table=torch.zeros(n, 176, dtype=torch.int32, device='cuda'))
        caches = GraphCaches(real, ids, ids, 131072)
        groups = ((0, 2), (2, 4)) if split else ((0, n),)
        graphs, outputs, calls = [], [], []
        try:
            for shared in (False, True):
                recorded = []
                def run():
                    caches.gather()
                    count, result = [0], []
                    def produce(ctx, tokens, kp):
                        count[0] += 1
                        return row_lengths(ctx, tokens, kp)
                    try:
                        for a, b in groups:
                            child, ctx = (caches.subset(a, b), contexts[a:b]) if split else (caches, contexts)
                            result.append([child.row_lengths(ctx, t, 4, produce) if shared else produce(ctx, t, 4)
                                           for _ in range(11)])
                        recorded.append(count[0])
                        return result
                    finally:
                        caches._decode_lengths = None
                        del caches.block_table
                graph, out = _capture(run)
                graphs.append(graph); outputs.append(out); calls.append(recorded)
                assert recorded == [len(groups)*(1 if shared else 11)]*2
            for phase, start in enumerate((0, 31997, 131061, 31990, 32252, 3)):
                contexts.copy_((torch.arange(n, device='cuda')+start).flip(0))
                ids.copy_((torch.arange(n, device='cuda')+phase).flip(0) % n)
                expected = (contexts[:, None]+torch.arange(t, device='cuda')+1).int()
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        for group in outputs[arm]:
                            for pair in group:
                                for value in pair:
                                    value.fill_(-77)
                        graphs[arm].replay()
                    for arm in outputs:
                        for (a, b), group in zip(groups, arm):
                            want = expected[a:b].flatten()
                            for seq, ke in group:
                                assert torch.equal(seq, want) and torch.equal(ke, want//4)
            report('decode_length_reuse', sequences=n, tokens=t, split_groups=len(groups), exact=True,
                   producer_calls_per_recording=[c[0] for c in calls], replay_orders='BA/AB',
                   changed_contexts=True, rollback=True, consumer_metrics_measured=False)
        finally:
            for graph in graphs:
                graph.reset()
