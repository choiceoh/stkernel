"""Compare direct-state memory tiles with an exported baseline state.py.

Real GLM cache layouts, no model/collective execution. Independent processes
are required; each process alternates AB/BA for five rounds of 30 samples.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from engine.base.arena import Arena
from engine.kernels import state
from engine.profiles.glm53 import facts
from engine.profiles.glm53.caches import Glm53Caches, layout


def load(path):
    spec = importlib.util.spec_from_file_location('direct_state_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def paired(graphs):
    values = [[], []]
    for round_id in range(5):
        for which in ((0,1) if round_id % 2 == 0 else (1,0)):
            graph = graphs[which]
            for _ in range(5):
                graph.replay()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(30)]
            for start,end in events:
                start.record(); graph.replay(); end.record()
            events[-1][1].synchronize()
            values[which].append([a.elapsed_time(b)*1000 for a,b in events])
    return {name: dict(median_us=statistics.median(sum(rows, [])),
                       rounds_median_us=[statistics.median(r) for r in rows], samples_us=rows)
            for name,rows in zip(('baseline','optimized'),values)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--ckpt-meta', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(13)
    old = load(args.baseline)
    F = facts.load(args.ckpt_meta)
    report = dict(scope=__doc__, torch=torch.__version__, triton=triton.__version__,
                  device=torch.cuda.get_device_name(),
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  state_sha256=hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest(), results=[])
    for layers in ([0], list(range(F.layers))):
        for n in (1,4):
            p = layout(F, layers)
            caches = Glm53Caches(Arena(p.nbytes(2, n)), F, layers, 2, n)
            for field in caches._fields.values():
                field.copy_(torch.randn_like(field))
            slots = [torch.tensor([i], device='cuda') for i in range(n,0,-1)]
            contexts = [torch.tensor(32768+i, device='cuda') for i in range(n)]
            calls = [(caches._fields['conv', L], caches._fields['rec', L], slot, ctx)
                     for L in F.kda_layers if L in layers for slot,ctx in zip(slots,contexts)]
            functions = [lambda module=m: tuple(module.kda_history(*c, F.conv-1) for c in calls)
                         for m in (old,state)]
            captured = [capture(fn) for fn in functions]
            for a,b in zip(captured[0][1], captured[1][1]):
                assert all(torch.equal(x,y) for x,y in zip(a,b))
            history = paired([x[0] for x in captured])
            for graph,_ in captured:
                graph.reset()
            del captured, functions, graph, a, b
            writes = {}
            for tokens in (1,6):
                inputs = torch.randn(len(calls), tokens, F.kda_heads_local, F.kda_dim, F.kda_dim, device='cuda')
                before = caches.state.clone()
                def write(module):
                    for values, (_,rec,slot,ctx) in zip(inputs,calls):
                        module.write_ring(values, rec, slot, ctx)
                graphs = [capture(lambda module=m: write(module))[0] for m in (old,state)]
                caches.state.copy_(before); graphs[0].replay()
                expected = caches.state.clone()
                caches.state.copy_(before); graphs[1].replay()
                assert torch.equal(caches.state, expected)
                del before, expected
                writes[tokens] = paired(graphs)
                for graph in graphs:
                    graph.reset()
                del graph, graphs, inputs
            result = dict(kda_layers=len(calls)//n, active_sequences=n,
                          history=history, writes=writes, exact=True)
            report['results'].append(result)
            print('case', result['kda_layers'], n,
                  {k:round(v['median_us'],2) for k,v in history.items()}, flush=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            del calls, caches, slots, contexts, field
            gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
