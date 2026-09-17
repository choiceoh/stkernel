"""Owned-GPU exact sampled-walk comparison, including keyed graph replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline-drafter', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('explicit --gpu required')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from engine.kernels.draft_sample import sampled_walk, _by_torch
    from engine.kernels.dense.cublaslt import _measure, require_timing_target
    prop = torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major, prop.minor), sms=prop.multi_processor_count), 'sm120-probe')
    report = dict(status='RUNNING', device=prop.name, torch=torch.__version__, cuda=torch.version.cuda,
                  scope='conditional probability construction and walk, excludes score construction and drafter model', cells=[])
    paths = ('engine/kernels/draft_sample.py', 'engine/base/sampler.py', 'probes/engine_draft_sample_check.py',
             'engine/profiles/glm53/drafter.py')
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    report['baseline_drafter_sha256'] = hashlib.sha256(args.baseline_drafter.read_bytes()).hexdigest()
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        for n,k,c in ((1,7,16),(2,7,16),(4,7,16),(8,7,16),(1,8,16),(2,5,4),(2,7,8)):
            torch.manual_seed(193+n+k+c)
            scores = torch.randn(n,k,c,c,device='cuda')
            cand = torch.randint(0,154880,(n,k,c),device='cuda')
            temps = torch.ones(n,device='cuda')
            uniforms = torch.rand(n,2*k+1,device='cuda')[:,:k]  # actual keyed block stride
            for greedy_rows,last_mass in ((True,False),(False,True)):
                kw=dict(greedy_rows=greedy_rows,last_mass=last_mass)
                def old(): return _by_torch(scores,cand,temps,uniforms,**kw)
                def new(): return sampled_walk(scores,cand,temps,uniforms,**kw)
                for trial in range(12):
                    scores.normal_().mul_((.01,1.,30.)[trial%3]);temps.fill_((.01,.8,2.)[trial%3]);uniforms.uniform_()
                    if trial%4==0: scores.round_();temps[::2]=0
                    if trial%4==1: scores.zero_();uniforms[:,::2]=0;uniforms[:,1::2]=1
                    if trial%4==2: scores[...,c//2:]=-float('inf')
                    for a,b in zip(new(),old()): torch.testing.assert_close(a,b,rtol=0,atol=0)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream): new()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): actual=new()
                for trial in range(2):
                    scores.normal_();temps.fill_(.8);uniforms.uniform_();expected=old();torch.cuda.synchronize()
                    allocation=torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                    graph.replay();torch.cuda.synchronize()
                    assert allocation==torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
                graph.reset()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    _measure(old,None);_measure(new,None)
                    times=[[_measure(f,None,repeats=12) for f in (old,new,new,old)] for _ in range(3)]
                cell=dict(rows=n,k=k,candidates=c,**kw,bit_exact=True,graph_replays=2,baab_ms=times)
                report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        # Extract the unchanged baseline method rather than approximate its
        # launch count with a hand-written benchmark. candidate_rows is held
        # fixed to isolate the actual selector pipeline after the model/head.
        import ast
        from types import SimpleNamespace
        from engine.profiles.glm53.drafter import Drafter
        tree = ast.parse(args.baseline_drafter.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Drafter')
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'propose_rows')
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<baseline-propose-rows>', 'exec'), namespace)
        baseline = namespace['propose_rows']
        report['selector_pipeline_cells'] = []
        for n in (1,2,4,8):
            k,c,r,v = 7,16,256,4096
            torch.manual_seed(250+n)
            unary = torch.randn(n,k,c,device='cuda')
            cand = torch.randint(0,v,(n,k,c),device='cuda')
            proj = torch.randn(n,k,r,device='cuda') * .1
            anchors = torch.randint(0,v,(n,),device='cuda')
            temps = torch.full((n,),.8,device='cuda')
            uniforms = torch.rand(n,2*k+1,device='cuda')[:,:k]
            codes = [torch.randn(v,r,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            d = SimpleNamespace(F=SimpleNamespace(sel_top_k=c), k=k, selector_alpha=(1.,)*k,
                                p=dict(zip(('candidate_selector.predecessor_codebook',
                                            'candidate_selector.successor_codebook'),codes)),
                                target=SimpleNamespace(comm=SimpleNamespace(world_size=1)),
                                candidate_rows=lambda *a:(unary,cand,proj))
            def call(fn): return fn(d,None,None,anchors,None,temps=temps,uniforms=uniforms)
            for trial in range(5):
                unary.normal_();proj.normal_().mul_(.1);uniforms.uniform_()
                for a,b in zip(call(Drafter.propose_rows),call(baseline)):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                old = lambda:call(baseline)
                new = lambda:call(Drafter.propose_rows)
                _measure(old,None);_measure(new,None)
                times=[[_measure(f,None,repeats=12) for f in (old,new,new,old)] for _ in range(3)]
            cell=dict(rows=n,k=k,candidates=c,rank=r,codebook_vocab=v,bit_exact=True,baab_ms=times)
            report['selector_pipeline_cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));raise
    finally: save()


if __name__=='__main__': main()
