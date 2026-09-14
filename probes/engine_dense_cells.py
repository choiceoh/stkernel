"""Same-build K=7 dense W4 cells against their generic routes; no model boot or consumer verdict.

Real rank weights at 8 rows (C=1) and 16 rows (C=2). Every comparison replays the
captured graph the projection serves in -- its input pack and output included -- as
B/A/A/B brackets of warm samples (64 replays each) and evicted samples (a 128 MiB
flush before every replay, outside the event interval). All arms are named routes
over the SAME DenseLinear packs of one build, so each cell is also a zero-tolerance
gate: every arm must reproduce the first arm's BF16 bytes, including changed and
strided inputs, poisoned outputs, both replay orders and rebound TX descriptors.

Routes (ROUTES), each `(ext, owners, x, destination) -> outputs`:
  bound         serving: decode_input_rows=(8,16,24,32) -> run_gemm_bound_input(forward_pipeline=True);
                C1 ordered/joined cells at 8 rows, the wide pack cells at 16 rows
  generic       decode_input_rows=() -> run_gemm (mk_gemm2_kernel<RQ>) or run_gemm_to_slot
  wide          run_gemm_wide_input: invocation-owned FP8 pack + mk_gemm2_kernel<RQ,PACKED> (9..32 rows)
  pair          QueryPair, the serving DSA query owner
  pair_generic  two DenseLinear readers with decode_input_rows=()
  pair_wide     run_query_pair(local_c1=False): wide pack + two packed mk_gemm2_kernel launches
  pack          run_input_pack: the C1 cell's own input pack alone (8 rows), timed against the whole cell
                to size a producer-side pack; it has no projection output, so the exactness gate skips it

To add an arm: put a route in ROUTES and a (control, candidate) pair in the cell's
row plan below. Scope `single` is one layer; `chain` calls every listed layer in
model order with its own weights, which no L2 holds at once.

Selection through the queue's literal flags: `--lanes dense_cells` runs every cell,
`--lanes dense_cells:kda.o_proj:kda.in_proj=16` names cells (optionally one row
count); `--seqs 1,2` picks 8/16 rows; `--samples N` sets B/A/A/B brackets (default 2);
`--output /cache/<name>.jsonl` keeps the events where the queue collects them.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_fusions import _capture, _time

BOUND = (8, 16, 24, 32)
KDA_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9)
DSA_LAYERS = (3, 7, 11, 15)
DENSE_LAYERS = (0, 1, 2)
# name, weight keys per layer, layers, direct TX output, input width, {rows: ((control, candidate), ...)}
CELLS = (
    ('kda.in_proj', ('kda.in_proj',), KDA_LAYERS, False, 4096,
     {8: (('bound', 'generic'), ('bound', 'pack')), 16: (('generic', 'wide'),)}),
    ('kda.o_proj', ('kda.o_proj',), KDA_LAYERS, True, 2048,
     {8: (('bound', 'generic'), ('bound', 'pack')), 16: (('bound', 'generic'),)}),
    ('mla.o_proj', ('mla.o_proj',), DSA_LAYERS, True, 4096,
     {8: (('bound', 'generic'), ('bound', 'pack')), 16: (('bound', 'generic'),)}),
    ('mla.query', ('mla.q_b', 'idx.wq_b'), DSA_LAYERS, False, 1536,
     {8: (('pair', 'pair_generic'), ('pair', 'pair_wide'), ('pair', 'pack')), 16: (('pair', 'pair_generic'),)}),
    ('mla.qkv_a', ('mla.qkv_a',), DSA_LAYERS, False, 4096,
     {16: (('bound', 'generic'),)}),
    ('mlp.gate_up', ('mlp.gate_up',), DENSE_LAYERS, False, 4096,
     {8: (('bound', 'generic'), ('bound', 'pack')), 16: (('bound', 'generic'),)}),
    ('mlp.down', ('mlp.down',), DENSE_LAYERS, True, 3072,
     {8: (('bound', 'generic'), ('bound', 'pack')), 16: (('bound', 'generic'),)}),
)
SHAPES = {'kda.in_proj': (6416, 4096), 'kda.o_proj': (4096, 2048), 'mla.o_proj': (4096, 4096),
          'mla.q_b': (4096, 1536), 'idx.wq_b': (4096, 1536), 'mla.qkv_a': (2048, 4096),
          'mlp.gate_up': (6144, 4096), 'mlp.down': (4096, 3072)}


def _dense(owner, x, destination, rows):
    owner.decode_input_rows = rows
    if destination is None:
        return owner(x)
    owner._write_slot(x, destination)
    return None


def _wide(ext, owner, x):
    p = owner.packs[0]
    y = torch.empty(x.shape[0], owner.rows, device=x.device, dtype=x.dtype)
    ext.run_gemm_wide_input(x, p.data, p.scale, y, owner.rows, 1., 0, p.rowscale.data_ptr(), 0, 0, 0)
    return y


def _pair(owners, x):
    from engine.kernels.dense.query_pair import QueryPair
    for owner in owners:
        owner.decode_input_rows = BOUND
    return QueryPair(*owners, rows=BOUND)(x)


def _pair_wide(ext, owners, x):
    packs = [o.packs[0] for o in owners]
    outputs = [torch.empty(x.shape[0], o.rows, device=x.device, dtype=x.dtype) for o in owners]
    ext.run_query_pair(x, [p.data for p in packs], [p.scale for p in packs],
                       [p.rowscale for p in packs], outputs, False)
    return tuple(outputs)


def _pack(ext, x):
    # The cell's own pack layout: k/128 blocks of 1024 FP8 bytes, then k/128 x 8 row scales.
    blocks = x.shape[1] // 128
    packed = torch.empty(blocks * 1024 + blocks * 8 * 4, device=x.device, dtype=torch.uint8)
    ext.run_input_pack(x, packed)
    return packed


PACK_ARMS = ('pack',)

ROUTES = {
    'bound': lambda ext, owners, x, d: _dense(owners[0], x, d, BOUND),
    'generic': lambda ext, owners, x, d: _dense(owners[0], x, d, ()),
    'wide': lambda ext, owners, x, d: _wide(ext, owners[0], x),
    'pair': lambda ext, owners, x, d: _pair(owners, x),
    'pair_generic': lambda ext, owners, x, d: tuple(_dense(o, x, None, ()) for o in owners),
    'pair_wide': lambda ext, owners, x, d: _pair_wide(ext, owners, x),
    'pack': lambda ext, owners, x, d: _pack(ext, x),
}


def _stats(values):
    return dict(mean=sum(values) / len(values), min=min(values))


def bracket(report, graphs, control, candidate, *, brackets, **meta):
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    for cache in ('warm', 'evicted'):
        samples = []
        for _ in range(brackets):
            for arm in (control, candidate, candidate, control):
                if cache == 'warm':
                    us = _time(graphs[arm], iterations=64) * 1000
                else:
                    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                              for _ in range(32)]
                    for start, end in events:
                        flush.zero_()
                        start.record()
                        graphs[arm].replay()
                        end.record()
                    events[-1][1].synchronize()
                    us = sum(s.elapsed_time(e) for s, e in events) / len(events) * 1000
                samples.append(dict(arm=arm, us=us))
        b = _stats([s['us'] for s in samples if s['arm'] == control])
        a = _stats([s['us'] for s in samples if s['arm'] == candidate])
        report('timing', cache=cache, control=control, candidate=candidate, samples=samples,
               control_us=b, candidate_us=a, mean_change_pct=100 * (a['mean'] / b['mean'] - 1),
               min_change_pct=100 * (a['min'] / b['min'] - 1), **meta)


def _values(out):
    return out if isinstance(out, tuple) else (out,)


def cell_check(report, ext, cell, owners, rows, *, brackets, timing=True):
    name, _, layers, direct, width, plan = cell
    arms = list(dict.fromkeys(a for pair in plan[rows] for a in pair))
    n = owners[0][0].rows
    for scope, group in (('single', owners[:1]), ('chain', owners)):
        parent = torch.randn(rows, width + 8, device='cuda', dtype=torch.bfloat16)
        x = parent[:, 4:width + 4]
        guards = {a: [torch.full((2, rows + 2, n), -123., device='cuda', dtype=torch.bfloat16) for _ in group]
                  for a in arms} if direct else {}
        addresses = {a: [torch.tensor([g[0, 1].data_ptr()], device='cuda', dtype=torch.int64) for g in guards[a]]
                     for a in arms} if direct else {}
        graphs, outputs = {}, {}
        try:
            for arm in arms:
                def run(arm=arm):
                    return [ROUTES[arm](ext, layer, x, addresses[arm][i] if direct else None)
                            for i, layer in enumerate(group)]
                graphs[arm], outputs[arm] = _capture(run)
            magnitudes = (0., .001, .1, 1., 50., 0.) if scope == 'single' else (1., 0.)
            for step, magnitude in enumerate(magnitudes):
                parent.normal_().mul_(magnitude)
                for order in (arms, arms[::-1]):
                    for arm in arms:
                        if direct:
                            for g, address in zip(guards[arm], addresses[arm]):
                                g.fill_(-123.)
                                address.fill_(g[step % 2, 1].data_ptr())
                        elif arm not in PACK_ARMS:
                            for out in outputs[arm]:
                                for y in _values(out):
                                    y.fill_(float('nan'))
                    for arm in order:
                        graphs[arm].replay()
                    torch.cuda.synchronize()
                    for arm in (a for a in arms[1:] if a not in PACK_ARMS):
                        if direct:
                            for g, want in zip(guards[arm], guards[arms[0]]):
                                inner = g[step % 2, 1:-1]
                                if not inner.isfinite().all().item():
                                    raise RuntimeError(f'{name} {arm} left a non-finite direct output')
                                torch.testing.assert_close(inner, want[step % 2, 1:-1], rtol=0, atol=0)
                                if not (g[step % 2, (0, -1)].eq(-123.).all().item()
                                        and g[1 - step % 2].eq(-123.).all().item()):
                                    raise RuntimeError(f'{name} {arm} wrote outside its rebound destination')
                        else:
                            for got, want in zip(outputs[arm], outputs[arms[0]]):
                                for a, b in zip(_values(got), _values(want)):
                                    if not a.isfinite().all().item():
                                        raise RuntimeError(f'{name} {arm} left a non-finite output')
                                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            report('exact', cell=name, rows=rows, scope=scope, layers=list(layers[:len(group)]), arms=arms,
                   reference=arms[0], not_projections=[a for a in arms if a in PACK_ARMS], magnitudes=magnitudes, replay_orders='forward/reverse', direct_output=direct,
                   rebound_descriptor=direct, input_stride=x.stride(0),
                   plan=[ext.gemm2_plan(rows, o.rows, o.cols) for o in owners[0]])
            if timing:
                for control, candidate in plan[rows]:
                    bracket(report, graphs, control, candidate, brackets=brackets, cell=name, rows=rows,
                            scope=scope, layers=len(group), direct_output=direct)
        finally:
            for graph in graphs.values():
                graph.reset()


def selected_cells(names=(), rows=(8, 16)):
    """[(cell, rows)] for `name` or `name=rows` tokens; every cell at every requested row count when empty."""
    wanted = {}
    for token in names:
        name, _, only = token.partition('=')
        wanted.setdefault(name, set()).update({int(only)} if only else set(rows))
    known = {cell[0] for cell in CELLS}
    if set(wanted) - known:
        raise ValueError(f'unknown dense cells: {sorted(set(wanted) - known)}')
    return [(cell, m) for m in rows for cell in CELLS
            if m in cell[5] and (not wanted or m in wanted.get(cell[0], ()))]


def check(report, ranks=None, *, cells=(), rows=(8, 16), brackets=2, timing=True):
    from engine.kernels.dense import DenseLinear, extension
    plan = selected_cells(cells, rows)
    if not plan:
        raise ValueError('no dense cell selected')
    keys = sorted({f'L{L}.{key}' for cell, _ in plan for key in cell[1] for L in cell[2]})
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loaded = rank_loader(path).load(keys, device='cuda')
        origin = str(path)
    else:
        loaded = {k: (torch.randn(*SHAPES[k.split('.', 1)[1]], device='cuda') * .02).bfloat16() for k in keys}
        origin = 'synthetic BF16 weights'
    report('weights', source=origin, packing='identical RTN W4 packs in every arm; not the consumer GPTQ packs',
           tensors={k: dict(shape=list(loaded[k].shape), sha256=hashlib.sha256(
               loaded[k].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for k in keys})
    dense = {k: DenseLinear(loaded[k], prefill=False) for k in keys}
    del loaded
    ext = extension()
    failures = []
    for cell, m in plan:
        owners = [tuple(dense[f'L{L}.{key}'] for key in cell[1]) for L in cell[2]]
        try:
            cell_check(report, ext, cell, owners, m, brackets=brackets, timing=timing)
        except Exception as exc:  # the other cells' evidence is kept; the run still fails
            failures.append(f'{cell[0]}={m}')
            report('component_failed', cell=cell[0], rows=m, error=f'{type(exc).__name__}: {exc}')
    return failures


def main(ranks=None, *, cells=(), seqs=None, samples=None, output=None):
    rows = tuple(8 * int(c) for c in seqs.split(',')) if seqs else (8, 16)
    if not rows or any(m not in (8, 16) for m in rows):
        raise ValueError('dense cells are compared at C=1 and C=2 (8 and 16 rows)')
    brackets = int(samples) if samples else 2
    sink = open(output, 'w') if output else None

    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink is not None:
            sink.write(line + '\n')
            sink.flush()
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py',
             'engine/kernels/dense/query_pair.py', 'probes/engine_dense_cells.py')
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           rows=rows, cells=list(cells), brackets=brackets,
           source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
           scope='captured same-build dense components beside production; no answer, acceptance or consumer-speed verdict')
    torch.manual_seed(915)
    failures = check(report, ranks, cells=cells, rows=rows, brackets=brackets)
    report('complete', status='FAIL' if failures else 'PASS', failed=failures, consumer_metrics_measured=False)
    if sink is not None:
        sink.close()
    if failures:
        raise RuntimeError(f'dense cell comparison failed: {failures}')
