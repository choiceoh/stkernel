"""Bounded same-pack DSA query and latent-write qualification; never consumer-speed proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_fusions import _capture, _time

ROWS = (8, 16, 24, 32)


def timings(report, name, rows, graphs, **metadata):
    # The same compiled graphs/weights in both orders. Eviction is outside
    # each event interval; include no cache-flush kernel in reported time.
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    for evicted in (False, True):
        samples = []
        for arm, i in (('B', 0), ('A', 1), ('A', 1), ('B', 0)):
            if not evicted:
                ms = _time(graphs[i], iterations=64)
            else:
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(32)]
                for start, end in events:
                    flush.zero_()
                    start.record()
                    graphs[i].replay()
                    end.record()
                events[-1][1].synchronize()
                ms = sum(start.elapsed_time(end) for start, end in events) / len(events)
            samples.append(dict(arm=arm, ms=ms))
        report('timing', candidate=name, rows=rows, cache='evicted' if evicted else 'warm',
               samples=samples, **metadata)


def query_check(report, *, layers=11, timing=True, ranks=None):
    from engine.kernels.dense import DenseLinear, extension
    from engine.kernels.dense.query_pair import QueryPair
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loader = rank_loader(path)
        keys = sorted(k for k in loader.keys() if k.endswith('.mla.q_b'))
        if len(keys) != 11:
            raise RuntimeError('query gate requires all 11 real DSA layers')
        names = [(k, k.removesuffix('.mla.q_b') + '.idx.wq_b') for k in keys]
        loaded = loader.load([k for pair in names for k in pair], device='cuda')
        weights = [(loaded[a], loaded[b]) for a, b in names]
        origin = str(path)
    else:
        weights = [tuple((torch.randn(4096, 1536, device='cuda') * .02).bfloat16() for _ in range(2)) for _ in range(layers)]
        origin = 'synthetic BF16 weights'
    owners = [tuple(DenseLinear(w, prefill=False) for w in pair) for pair in weights]
    for pair in owners:
        for layer in pair:
            layer.decode_input_rows = ROWS
    paired = [QueryPair(*pair, rows=ROWS) for pair in owners]
    report('query_weights', source=origin, layers=len(owners), packing='same RTN W4 packs in both arms',
           shapes=[[list(w.shape) for w in pair] for pair in weights],
           weight_sha256=[[hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                           for w in pair] for pair in weights])
    del weights
    for rows, row_stride in ((m, s) for m in ROWS for s in (1536, 1544)):
        parent = torch.randn(rows, row_stride, device='cuda', dtype=torch.bfloat16)
        x = parent if row_stride == 1536 else parent[:, 4:1540]
        graphs, outputs = [], []
        try:
            for fn in (lambda: [tuple(layer(x) for layer in pair) for pair in owners],
                       lambda: [pair(x) for pair in paired]):
                graph, out = _capture(fn)
                graphs.append(graph); outputs.append(out)
            for scale in (0., .01, 1., 16.):
                parent.normal_().mul_(scale)
                for order in ((0, 1), (1, 0)):
                    for i in order:
                        for pair in outputs[i]:
                            for value in pair:
                                value.fill_(float('nan'))
                        graphs[i].replay()
                    for a, b in zip(outputs[1], outputs[0]):
                        for got, want in zip(a, b):
                            if not got.isfinite().all().item():
                                raise RuntimeError('query candidate left non-finite output')
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
            report('query_numerics', rows=rows, layers=len(owners), exact=True, replay_orders='BA/AB',
                   input_stride=x.stride(0), shape_plans=[extension().gemm2_plan(rows, layer.rows, 1536)
                                                         for layer in owners[0]])
            if timing and row_stride == 1536:
                timings(report, 'query_pair', rows, graphs, layers=len(owners))
        finally:
            for graph in graphs:
                graph.reset()


def latent_check(report, *, timing=True):
    from engine.kernels.common.norm_rope import norm
    from engine.kernels.indexer import latent_write_rows
    from engine.kernels.mla.decode_inputs import latent_norm_write
    block, stride, pages, table_width = 768, 11 * 768, 16, 176
    for rows in ROWS:
        c = rows // 8
        contexts = torch.empty(c, device='cuda', dtype=torch.int64)
        table_parent = torch.empty(c, 2 * table_width, device='cuda', dtype=torch.int32)
        table = table_parent[:, ::2]
        source = torch.randn(rows, 2056, device='cuda', dtype=torch.bfloat16)
        x = source[:, 1536:2048]
        weights = [torch.randn(512, device='cuda', dtype=torch.bfloat16) for _ in range(11)]
        raw = [torch.empty(pages * stride, 520, device='cuda', dtype=torch.uint8) for _ in range(2)]
        latent = [t[:, :512].view(torch.float8_e4m3fn) for t in raw]
        graphs = []
        try:
            for i in range(2):
                def run():
                    for L, w in enumerate(weights):
                        if i == 0:
                            latent_write_rows(norm(x, w, 1e-6).to(torch.float8_e4m3fn), latent[i], table,
                                              block, stride, L * block, contexts, 8)
                        else:
                            latent_norm_write(x, w, latent[i], table, block, stride, L * block, contexts, 8, 1e-6)
                # A valid table/context is needed during warmup, before capture.
                contexts.fill_(31997)
                table.copy_((torch.arange(table_width, device='cuda')[None, :] +
                             3 * torch.arange(c, device='cuda')[:, None]) % pages)
                graph, _ = _capture(run)
                graphs.append(graph)
            for ctx, shift in ((31997, 0), (131069, 1), (32252, 5)):
                contexts.fill_(ctx)
                table.copy_((torch.arange(table_width, device='cuda')[None, :] + shift +
                             3 * torch.arange(c, device='cuda')[:, None]) % pages)
                for magnitude in (0., .01, 1., 16.):
                    source.normal_().mul_(magnitude)
                    for order in ((0, 1), (1, 0)):
                        for i in order:
                            raw[i].fill_(0x55)
                            graphs[i].replay()
                        torch.testing.assert_close(raw[1], raw[0], rtol=0, atol=0)
            report('latent_numerics', rows=rows, layers=11, exact=True, input_stride=x.stride(0),
                   latent_stride=latent[0].stride(0), contexts=[31997, 131069, 32252], block=block,
                   rebound_tables=True, layer_offsets=True, poison_padding=True, replay_orders='BA/AB')
            if timing:
                timings(report, 'latent_norm_write', rows, graphs, layers=11)
        finally:
            for graph in graphs:
                graph.reset()


def check(ranks=None):
    from engine.base.kernel_shape import bound, to_dict
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/query_pair.py',
             'engine/kernels/mla/decode_inputs.py', 'engine/kernels/indexer_gate.py',
             'engine/kernels/decode_projection.py', 'engine/profiles/glm53/net.py')
    def report(event, **values):
        print(json.dumps(dict(event=event, **values)), flush=True)
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           engine_shape=to_dict(bound()), source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
           scope='captured same-build component checks; no answer, acceptance or consumer-speed verdict')
    torch.manual_seed(914875)
    from probes.engine_decode_indexer_gate import check as head_gate_check
    failures = []
    for name, fn in (('query_pair', lambda: query_check(report, ranks=ranks)),
                     ('latent_norm_write', lambda: latent_check(report)),
                     ('indexer_head_gate', lambda: head_gate_check(report, ranks))):
        try:
            fn()
        except Exception as exc:
            failures.append(name)
            report('component_failed', candidate=name, error=f'{type(exc).__name__}: {exc}')
    report('complete', status='FAIL' if failures else 'PASS', failed_components=failures, consumer_metrics_measured=False)
    if failures:
        raise RuntimeError(f'DSA qualification failed: {failures}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks', help='consumer rank directory; otherwise synthetic weights are explicitly reported')
    check(ap.parse_args().ranks)
