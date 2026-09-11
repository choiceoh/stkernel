"""Real DFlash2 weights with one target vocabulary shard, on a single GPU.

Checks legacy/new eager and graph proposals and unmodified context rings.
Synthetic accepted-context activations exercise real weights, not generation
quality or draft acceptance. Identity collectives cannot qualify TP4 speed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as fn
from engine.base.comm import Comm
from engine.base.loader import RankLoader
from engine.profiles.glm53.drafter import Drafter, load


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out = call()
    torch.cuda.synchronize()
    return graph, out


def paired(graphs):
    values=[[],[]]
    for round_id in range(3):
        for which in ((0,1) if round_id%2==0 else (1,0)):
            g=graphs[which]
            for _ in range(3):g.replay()
            pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(30)]
            for a,b in pairs:
                a.record();g.replay();b.record()
            pairs[-1][1].synchronize()
            values[which].append([a.elapsed_time(b)*1000 for a,b in pairs])
    return {name:dict(median_us=statistics.median(sum(rows,[])),samples_us=rows)
            for name,rows in zip(('legacy','candidate'),values)}


class Target:
    rank=0
    comm=Comm()
    def __init__(self, path):
        self.p=RankLoader(path).load(['embed','head'],max_run=64<<20)
        self.vp=self.p['head'].shape[0]
    def embed(self, ids):
        invalid=ids>=self.vp
        return fn.embedding(ids.masked_fill(invalid,0),self.p['embed']).masked_fill(invalid[:,None],0)
    def head_local(self,h):return fn.linear(h,self.p['head'])
    def head(self,h):return self.head_local(h)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head',required=True)
    ap.add_argument('--drafter-dir',required=True)
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    free,total=torch.cuda.mem_get_info()
    if free<5*2**30:raise RuntimeError('single-GPU real-weight check requires 5 GiB free')
    torch.cuda.set_per_process_memory_fraction((5*2**30)/total)
    spec=importlib.util.spec_from_file_location('draft_full_gather_baseline',a.baseline)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    F=load(a.drafter_dir)
    d=Drafter(F,Target(a.head),153880)
    d.bind(RankLoader(Path(a.drafter_dir)/'model.safetensors').load([s.name for s in d.specs()],max_run=64<<20))
    ring=torch.zeros(F.layers,2,F.window,F.kv_heads,F.head_dim,device='cuda',dtype=torch.bfloat16)
    anchor=torch.tensor([1234],device='cuda')
    position=torch.tensor(0,device='cuda')
    old_graph,old_out=capture(lambda:old.Drafter.propose_tensor(d,anchor,position,ring))
    new_graph,new_out=capture(lambda:d.propose_tensor(anchor,position,ring))
    report=dict(scope=__doc__,torch=torch.__version__,torch_git=torch.version.git_version,
                cuda=torch.version.cuda,device=torch.cuda.get_device_name(),cases=[],
                baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest(),
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                    'engine/modules/vocab.py','engine/profiles/glm53/drafter.py','probes/engine_draft_candidate_check.py')})
    for context in (0,1,17,2047,2048,2057):
        ring.zero_()
        ids=torch.arange(max(0,context-F.window),context,device='cuda')
        aux=torch.randn(ids.numel(),F.hidden*len(F.target_layers),device='cuda',dtype=torch.bfloat16)*.02
        d.observe(ring,ids,aux)
        position.fill_(context)
        before=ring.clone()
        expected=old.Drafter.propose_tensor(d,anchor,context,ring)
        actual=d.propose_tensor(anchor,context,ring)
        old_graph.replay();new_graph.replay();torch.cuda.synchronize()
        assert all(torch.equal(expected,x) for x in (actual,old_out,new_out)),(context,expected,actual,old_out,new_out)
        assert torch.equal(before,ring)
        row=dict(context=context,exact=True,ids=actual.tolist(),timing=paired([old_graph,new_graph]))
        assert torch.equal(before,ring)
        report['cases'].append(row)
        print(context,actual.tolist(),{k:round(v['median_us'],2) for k,v in row['timing'].items()},flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    old_graph.reset();new_graph.reset()
    report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
