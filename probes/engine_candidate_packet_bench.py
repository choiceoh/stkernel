"""Captured candidate work only; no network, weights or consumer speed claim."""
import statistics
import torch


def measure():
    from engine.kernels.common.vocab_candidates import pack, select, select_logits, restore
    from engine.modules.vocab import CandidateBuffer
    report = []
    generator = torch.Generator(device='cuda').manual_seed(91328)
    width, k = 38720, 16
    for rows in (6, 24):
        logits = torch.randn(4, rows, width, generator=generator, device='cuda').bfloat16()
        x = logits[0]
        foreign = torch.cat([select(pack(logits[r], r*width, width), k) for r in (1, 2, 3)], -1)
        workspace = CandidateBuffer(rows, width*4, k*4, 'cuda')

        def call(fused, reuse):
            packet = select_logits(x, 0, width, k) if fused else select(pack(x, 0, width), k)
            gathered = torch.cat((packet, foreign), -1)  # identity stand-in for the same exchange
            dense = workspace.restore(gathered, width*4) if reuse else restore(gathered, width*4)
            return dense.topk(k, dim=-1)

        graphs = {}
        outputs = {}
        flush = torch.empty(64*1024*1024, dtype=torch.uint8, device='cuda')
        try:
            for name, fused, reuse in (('original', False, False), ('fused_select', True, False),
                                       ('fused_select_reuse', True, True)):
                for _ in range(3):
                    call(fused, reuse)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call(fused, reuse)
                graphs[name], outputs[name] = graph, output
                graph.replay()
                for got, want in zip(output, outputs['original']):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            for cold in (False, True):
                samples = {name: [] for name in graphs}
                for trial in range(4):
                    names = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
                    for name in names:
                        spans = []
                        for _ in range(40):
                            if cold:
                                flush.zero_()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record(); graphs[name].replay(); end.record()
                            spans.append((start, end))
                        spans[-1][1].synchronize()
                        samples[name] += [a.elapsed_time(b)*1000 for a, b in spans]
                report.append(dict(rows=rows, cache='cold-64MiB' if cold else 'warm',
                                   median_us={name: statistics.median(values) for name, values in samples.items()},
                                   samples_us=samples))
        finally:
            for graph in graphs.values():
                graph.reset()
    return dict(scope='captured local selection, synthetic gather copy and exact dense topk; no NIC/model proof',
                results=report)
