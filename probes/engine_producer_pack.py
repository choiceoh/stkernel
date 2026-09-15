"""KDA o_proj's input pack written by its producer, the output norm, against the cell's own pack launch.

Same build, real rank weights, 8 rows (C=1) and 16 rows (C=2). Control: kda_output_norm, then the bound
o_proj cell with its own pack launch (run_gemm_bound_input, direct TX output) -- at 8 rows
mk_input_pack_kernel + the ordered in-CTA kernel, at 16 rows mk_wide_input_pack_kernel + the sixteen-row
CTA. Candidate: the norm writes the cell's pack beside its output and the cell reads it (producer_pack).
Zero tolerance first: the norm output, the TX slot bytes and the pack itself -- against the cell's pack
kernel run on the control's own output (run_input_pack) -- over changed magnitudes, subnormal gates,
poisoned slots and both replay orders; at 16 rows the pack comparison covers the 16 used rows of each K
block. Then warm and evicted B/A/A/B for one layer and for a chain of distinct KDA layers; each interval
is the whole norm + projection. No model boot, consumer speed or acceptance verdict.

    bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes producer_pack --seqs 1,2 \
        --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors --output /cache/<name>.jsonl
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_fusions import _capture
from probes.engine_dense_cells import BOUND, KDA_LAYERS, bracket

HEADS, DIM = 16, 128
EPS = 1e-5


def _used(pack, rows):
    """The bytes a cell reads: all of the C1 pack; the 16 used rows of each K block of the wide pack."""
    if rows == 8:
        return (pack,)
    words = pack[:HEADS * 32 * 128].view(HEADS, 32, 128)[:, :16]
    scales = pack[HEADS * 32 * 128:].view(HEADS, 32, 4)[:, :16]
    return words, scales


def _arms(kda_output_norm, layers, core, gate, addresses, rows, pack_bytes):
    def control():
        outs = []
        for (owner, weight), address in zip(layers, addresses['control']):
            y = kda_output_norm(core, gate, weight, EPS)
            owner._write_slot(y.reshape(rows, HEADS * DIM), address)
            outs.append(y)
        return outs

    def producer():
        outs = []
        for (owner, weight), address in zip(layers, addresses['producer']):
            pack = torch.empty(pack_bytes, dtype=torch.uint8, device=core.device)
            y = kda_output_norm(core, gate, weight, EPS, pack=pack)
            owner._write_slot(y.reshape(rows, HEADS * DIM), address, pack=pack)
            outs.append((y, pack))
        return outs
    return {'control': control, 'producer': producer}


def check(report, ranks=None, *, rows_list=(8,), brackets=2, timing=True):
    from engine.kernels.dense import DenseLinear, extension, producer_pack_nbytes
    from engine.kernels.kda.output import kda_output_norm
    keys = [f'L{L}.kda.{w}' for L in KDA_LAYERS for w in ('o_proj', 'o_norm')]
    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loaded = rank_loader(path).load(keys, device='cuda')
        origin = str(path)
    else:
        loaded = {}
        for L in KDA_LAYERS:
            loaded[f'L{L}.kda.o_proj'] = (torch.randn(4096, HEADS * DIM, device='cuda') * .02).bfloat16()
            loaded[f'L{L}.kda.o_norm'] = (1 + .1 * torch.randn(DIM, device='cuda')).bfloat16()
        origin = 'synthetic BF16 weights'
    report('weights', source=origin, tensors={k: dict(shape=list(loaded[k].shape), dtype=str(loaded[k].dtype),
           sha256=hashlib.sha256(loaded[k].cpu().float().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
           for k in keys}, packing='identical RTN W4 packs in both arms')
    owners = []
    for L in KDA_LAYERS:
        owner = DenseLinear(loaded[f'L{L}.kda.o_proj'], prefill=False)
        owner.decode_input_rows = BOUND
        owners.append((owner, loaded[f'L{L}.kda.o_norm'].contiguous()))
    ext = extension()
    n = 4096
    for rows in rows_list:
        pack_bytes = producer_pack_nbytes(rows, HEADS * DIM)
        for owner, _ in owners:
            if not owner.producer_pack_rows(rows):
                raise RuntimeError(f'kda.o_proj is not a bound direct writer that reads producer packs at {rows} rows')
        for scope, group in (('single', owners[:1]), ('chain', owners)):
            core = torch.empty(rows, HEADS, DIM, device='cuda', dtype=torch.bfloat16)
            gate = torch.empty_like(core)
            guards = {a: [torch.full((2, rows + 2, n), -123., device='cuda', dtype=torch.bfloat16) for _ in group]
                      for a in ('control', 'producer')}
            addresses = {a: [torch.tensor([g[0, 1].data_ptr()], device='cuda', dtype=torch.int64) for g in guards[a]]
                         for a in guards}
            core.normal_(); gate.normal_()
            graphs, outputs = {}, {}
            try:
                for arm, fn in _arms(kda_output_norm, group, core, gate, addresses, rows, pack_bytes).items():
                    graphs[arm], outputs[arm] = _capture(fn)
                reference = torch.empty(pack_bytes, dtype=torch.uint8, device='cuda')
                cases = ([(0., 0.), (.001, 1.), (1., 1.), (50., 4.), (1., 30.), (1e-3, 88.)] if scope == 'single'
                         else [(1., 1.), (0., 0.)])
                for step, (magnitude, spread) in enumerate(cases):
                    core.normal_().mul_(magnitude)
                    gate.normal_().mul_(spread)
                    if spread >= 88.:                   # subnormal sigmoids: -88 gates on a band of columns
                        gate[:, :, ::7] = -88.
                    for order in (('control', 'producer'), ('producer', 'control')):
                        for arm in order:
                            for g, address in zip(guards[arm], addresses[arm]):
                                g.fill_(-123.)
                                address.fill_(g[step % 2, 1].data_ptr())
                            for item in outputs[arm]:
                                (item[0] if isinstance(item, tuple) else item).fill_(float('nan'))
                        for arm in order:
                            graphs[arm].replay()
                        torch.cuda.synchronize()
                        for i, (want_y, (got_y, pack)) in enumerate(zip(outputs['control'], outputs['producer'])):
                            torch.testing.assert_close(got_y, want_y, rtol=0, atol=0)
                            want_slot = guards['control'][i][step % 2, 1:-1]
                            got_slot = guards['producer'][i][step % 2, 1:-1]
                            if not got_slot.isfinite().all().item():
                                raise RuntimeError('producer arm left a non-finite TX slot')
                            torch.testing.assert_close(got_slot, want_slot, rtol=0, atol=0)
                            for arm in ('control', 'producer'):
                                g = guards[arm][i]
                                if not (g[step % 2, (0, -1)].eq(-123.).all().item()
                                        and g[1 - step % 2].eq(-123.).all().item()):
                                    raise RuntimeError(f'{arm} wrote outside its rebound TX slot')
                            ext.run_input_pack(want_y.reshape(rows, HEADS * DIM), reference)
                            torch.cuda.synchronize()
                            for got, want in zip(_used(pack, rows), _used(reference, rows)):
                                if not torch.equal(got, want):
                                    bad = (got != want).nonzero()
                                    raise RuntimeError(f'{rows}-row producer pack differs from the cell\'s pack kernel at '
                                                       f'{len(bad)} places, first {bad[:4].tolist()}')
                report('exact', rows=rows, scope=scope, layers=list(KDA_LAYERS[:len(group)]), cases=cases,
                       checked=['norm output', 'TX slot', 'pack bytes vs the cell\'s pack kernel'],
                       replay_orders='forward/reverse', rebound_descriptor=True, pack_bytes=pack_bytes)
                if timing:
                    bracket(report, graphs, 'control', 'producer', brackets=brackets, cell='kda.o_proj+norm',
                            rows=rows, scope=scope, layers=len(group), direct_output=True)
            finally:
                for graph in graphs.values():
                    graph.reset()


def main(ranks=None, *, seqs=None, samples=None, output=None):
    rows_list = tuple(8 * int(c) for c in seqs.split(',')) if seqs else (8,)
    if not rows_list or any(m not in (8, 16) for m in rows_list):
        raise ValueError('producer packs are compared at C=1 and C=2 (8 and 16 rows)')
    brackets = int(samples) if samples else 2
    sink = open(output, 'w') if output else None

    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink is not None:
            sink.write(line + '\n')
            sink.flush()
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py', 'engine/kernels/kda/output.py',
             'probes/engine_producer_pack.py', 'probes/engine_dense_cells.py')
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           rows=rows_list, brackets=brackets,
           source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
           scope='captured same-build KDA norm + o_proj beside production; no consumer verdict')
    torch.manual_seed(916)
    status, failure = 'PASS', None
    try:
        check(report, ranks, rows_list=rows_list, brackets=brackets)
    except Exception as exc:
        status, failure = 'FAIL', f'{type(exc).__name__}: {exc}'
    report('complete', status=status, failure=failure)
    if sink is not None:
        sink.close()
    if failure:
        raise RuntimeError(failure)
