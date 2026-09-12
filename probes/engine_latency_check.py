"""GPU contract for onepass graph attribution and profiler isolation (no model speed claim)."""
import gzip
import json
from pathlib import Path
import tempfile


def check():
    import torch
    from engine.base.graphs import DecodeGraphs
    from engine.base.graph_labels import operation
    from engine.base.latency import Recorder
    from engine.base.latency_trace import attribute

    @operation('add')
    def add(x): return x + 2

    @operation('multiply')
    def multiply(x): return x * 3

    def forward(x): return multiply(add(x))
    graphs = DecodeGraphs(forward, lambda c, t: torch.ones((c, t), device='cuda'), [(1, 1024), (4, 1024)], label='latency-proof')
    reports = []
    with tempfile.TemporaryDirectory() as root:
        rec = Recorder(0, root)
        for width in (1, 4):
            shape = (width, 1024)
            rec.begin(f'measure-{width}', concurrency=width)
            with rec.step('decode', list(range(width)), [0] * width, width):
                graphs.graphs[shape].replay()
            torch.cuda.synchronize()
            normal = rec.finish(f'measure-{width}')
            assert normal['complete'] and not normal['traces'] and not normal['preparation_changed'], normal
            rec.begin(f'diagnostic-{width}', diagnostic=True, concurrency=width)
            with rec.step('decode', list(range(width)), [0] * width, width):
                graphs.graphs[shape].replay()
            diagnostic = rec.finish(f'diagnostic-{width}')
            assert diagnostic['complete'] and diagnostic['traces'], diagnostic
            trace = diagnostic['traces'][0]
            path = Path(diagnostic['server_directory']) / trace['file']
            rows = attribute(json.loads(gzip.decompress(path.read_bytes())))
            operations = [r['operation'] for r in rows if r['attribution'] == 'graph_node']
            assert any(op.endswith('/add') for op in operations), (rows, diagnostic['graph_attribution_errors'])
            assert any(op.endswith('/multiply') for op in operations), (rows, diagnostic['graph_attribution_errors'])
            torch.testing.assert_close(graphs.outputs[shape], torch.full_like(graphs.outputs[shape], 9))
            block = rec.artifact(f'diagnostic-{width}', trace['file'], 0)
            assert block['bytes'] == trace['bytes']
            reports.append(dict(concurrency=width, graph_operations=operations, activities=len(rows),
                                normal_profiled=False, numerics=True))
    return dict(passed=True, checks=reports, scope='toy CUDA graph; instrumentation correctness, not GLM throughput')
