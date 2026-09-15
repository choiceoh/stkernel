"""C1 input packing: same bytes and W4 consumers with 1/2/4/8 warps per CTA.

The eight-warp arm is the previous kernel's layout and arithmetic. Only work
ownership changes. GPU event intervals live inside captured graphs; eviction
and Python enqueue gaps are outside. No consumer tok/s or TP4 verdict.
"""
import hashlib
import json
from pathlib import Path

import torch

from probes.engine_decode_fusions import _capture

WARPS = (8, 4, 2, 1)
WIDTHS = (1536, 2048, 3072, 4096)


def exact_packs(report, ext):
    from engine.kernels.dense import producer_pack_nbytes
    for width in (128, *WIDTHS, 20480):
        parent = torch.empty(8, width + 8, device='cuda', dtype=torch.bfloat16)
        x = parent[:, 4:width + 4]
        size = producer_pack_nbytes(8, width)
        guards = {w: torch.full((size + 32,), 165, device='cuda', dtype=torch.uint8) for w in WARPS}
        packs = {w: guard[16:-16] for w, guard in guards.items()}
        graphs = {}
        try:
            for w in WARPS:
                graphs[w], _ = _capture(lambda w=w: ext.run_input_pack(x, packs[w], w))
            for magnitude in (0., 1e-35, .001, 1., 50., 1e20):
                parent.normal_().mul_(magnitude)
                before = parent.clone()
                for order in (WARPS, WARPS[::-1]):
                    for guard in guards.values():
                        guard.fill_(165)
                    for w in order:
                        graphs[w].replay()
                    torch.cuda.synchronize()
                    for w in WARPS:
                        if not torch.equal(packs[w], packs[8]):
                            raise AssertionError(f'K={width}, warps={w}: pack bytes differ')
                        if not (guards[w][:16].eq(165).all() and guards[w][-16:].eq(165).all()):
                            raise AssertionError('pack wrote outside its allocation')
                    if not torch.equal(parent, before):
                        raise AssertionError('packing modified its input')
            # Unsupported controls must fail before touching the destination.
            for w in (-1, 3, 16):
                try:
                    ext.run_input_pack(x, packs[8], w)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f'invalid geometry {w} was accepted')
            report('exact_pack', width=width, warps=WARPS, stride=x.stride(0),
                   magnitudes=6, replay_orders=2, guarded=True, bytes=size)
        finally:
            for graph in graphs.values():
                graph.reset()


def timings(report, calls, *, brackets, **meta):
    """B/A/A/B of device event intervals, excluding host enqueue and eviction."""
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    for cache, repeats in (('warm', 32), ('evicted', 1)):
        graphs, events = {}, {}
        try:
            for w, fn in calls.items():
                start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                def timed(fn=fn, start=start, end=end):
                    start.record()
                    for _ in range(repeats):
                        fn()
                    end.record()
                graphs[w], _ = _capture(timed)
                events[w] = start, end
            for candidate in (w for w in calls if w != 8):
                samples = []
                for _ in range(brackets):
                    for w in (8, candidate, candidate, 8):
                        values = []
                        for _ in range(16):
                            if cache == 'evicted':
                                flush.zero_()
                            graphs[w].replay()
                            start, end = events[w]
                            end.synchronize()
                            values.append(start.elapsed_time(end) * 1000 / repeats)
                        samples.append(dict(warps=w, us=sum(values) / len(values)))
                control = [s['us'] for s in samples if s['warps'] == 8]
                changed = [s['us'] for s in samples if s['warps'] == candidate]
                report('timing', cache=cache, candidate=candidate, samples=samples,
                       control_us=sum(control)/len(control), candidate_us=sum(changed)/len(changed),
                       change_pct=100*(sum(changed)/sum(control)-1), **meta)
        finally:
            for graph in graphs.values():
                graph.reset()


def pack_timings(report, ext, brackets):
    from engine.kernels.dense import producer_pack_nbytes
    for width in WIDTHS:
        x = torch.randn(8, width, device='cuda', dtype=torch.bfloat16)
        packs = {w: torch.empty(producer_pack_nbytes(8, width), device='cuda', dtype=torch.uint8) for w in WARPS}
        timings(report, {w: lambda w=w: ext.run_input_pack(x, packs[w], w) for w in WARPS},
                brackets=brackets, component='pack', width=width)


def projection_checks(report, ranks, brackets, timing=True):
    """Use the existing real-weight, poisoned/rebound W4 consumer gate."""
    from probes import engine_dense_cells as cells
    def project(ext, owners, x, destination, warps):
        if len(owners) == 2:
            outputs = [torch.empty(8, o.rows, device=x.device, dtype=x.dtype) for o in owners]
            ext.run_query_pair(x, [o.packs[0].data for o in owners], [o.packs[0].scale for o in owners],
                               [o.packs[0].rowscale for o in owners], outputs, input_pack_rows=warps)
            return tuple(outputs)
        owner, = owners
        pack, = owner.packs
        out = (torch.empty(8, owner.rows, device=x.device, dtype=x.dtype)
               if destination is None else destination)
        ext.run_gemm_bound_input(x, pack.data, pack.scale, out, owner.rows, pack.rowscale.data_ptr(),
                                 owner.workspace, destination, input_pack_rows=warps)
        return out if destination is None else None
    original_cells, original_routes, original_bracket = cells.CELLS, cells.ROUTES, cells.bracket
    try:
        # Probe-only route controls; production has no runtime geometry knob.
        cells.ROUTES = {f'pack{w}': lambda ext, owners, x, d, w=w: project(ext, owners, x, d, w) for w in WARPS}
        cells.CELLS = tuple((*c[:-1], {8: tuple(('pack8', f'pack{w}') for w in WARPS if w != 8)})
                            for c in original_cells if 8 in c[-1] and not c[0].startswith('drafter.'))
        def bracket_rows(report, graphs, control, candidate, *, brackets, calls, **meta):
            # Capture the native calls and timing events together, retaining
            # the exact inputs and rebound destinations from the output gate.
            w = int(candidate[4:])
            timings(report, {8: calls[control], w: calls[candidate]},
                    brackets=brackets, component='projection', **meta)
        cells.bracket = bracket_rows
        failures = cells.check(report, ranks, rows=(8,), brackets=brackets, timing=timing)
        if failures:
            raise AssertionError(f'projection gate failed: {failures}')
    finally:
        cells.CELLS, cells.ROUTES, cells.bracket = original_cells, original_routes, original_bracket


def main(ranks=None, *, samples=None, output=None):
    from engine.kernels.dense import extension
    brackets = int(samples) if samples else 2
    if brackets < 1:
        raise ValueError('at least one bracket is required')
    root = Path(__file__).resolve().parents[1]
    sink = open(output, 'w') if output else None
    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink:
            sink.write(line+'\n'); sink.flush()
    try:
        torch.manual_seed(915)
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
               source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                              ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py',
                               'probes/engine_input_pack_grid.py', 'probes/engine_dense_cells.py',
                               'probes/engine_decode_fusions.py')},
               scope='same-build C1 GPU components; not TP4 serving performance')
        ext = extension()
        exact_packs(report, ext)
        pack_timings(report, ext, brackets)
        if ranks:
            projection_checks(report, ranks, brackets)
        report('complete', status='PASS', consumer_metrics_measured=False,
               peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if sink:
            sink.close()
