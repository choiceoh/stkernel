"""Matched old/new served KDA adapters, including copies and all snapshots."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from engine.profiles.glm53 import lanes


def load(path, name):
    spec = importlib.util.spec_from_file_location("engine.kernels.kda."+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_lane(directory):
    old = load(directory/"kda.py", "baseline_kda")
    old.fused_recurrent_gated_delta_rule_fwd_kernel = load(directory/"fused_recurrent.py", "baseline_recurrent").fused_recurrent_gated_delta_rule_fwd_kernel
    def baseline(q,k,v,g,beta,alog,bias,state,lb):
        initial = (state.transpose(-1,-2).contiguous() if state is not None else
                   torch.zeros(1,v.shape[2],v.shape[-1],k.shape[-1],device=q.device,dtype=torch.float32))
        output, states = old.fused_recurrent_kda(q,k,v,g,beta,scale=k.shape[-1]**-.5,
            initial_state=initial,inplace_final_state=False,use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,a_log=alog,g_bias=bias,compute_gate=True,lower_bound=lb)
        return output, states.transpose(-1,-2).contiguous()
    return baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((768*2**20)/torch.cuda.get_device_properties(0).total_memory)
    baseline = baseline_lane(a.baseline_dir)
    candidate = lanes.served(reference_for=("expert",)).kda_recurrent
    oracle = lanes.reference().kda_recurrent
    def errors(actual, expected):
        return {name: {"bits_exact":torch.equal(x.view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32),
                                               y.view(torch.int16 if y.dtype==torch.bfloat16 else torch.int32)),
                       "relative_max":((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-6)).item(),
                       "absolute_max":(x.float()-y.float()).abs().max().item()}
                for name,x,y in zip(("output","every_state"),actual,expected)}
    torch.manual_seed(92812)
    trash = torch.empty(64*2**20//4, device="cuda")
    reports=[]
    for t in (1,2,3,4,5,6,12):
        for seeded in (False,True):
            # Dense inputs isolate the adapter. The regression suite also
            # tests the strided projection slices passed by the real net.
            q,k,v,g=[torch.randn(1,t,16,128,device="cuda",dtype=torch.bfloat16) for _ in range(4)]
            beta=torch.randn(1,t,16,device="cuda",dtype=torch.bfloat16)
            alog=torch.randn(16,device="cuda")*.2
            bias=torch.randn(16*128,device="cuda")*.1
            initial=torch.randn(1,16,128,128,device="cuda")*.1 if seeded else None
            args=(q,k,v,g,beta,alog,bias,initial,-5.)
            saved=initial.clone() if seeded else None
            ref=baseline(*args); new=candidate(*args); expected=oracle(*args)
            comparisons={"old":errors(new,ref), "oracle":errors(new,expected)}
            for comparison in comparisons.values():
                assert comparison["output"]["relative_max"]<.008,comparison
                assert comparison["every_state"]["relative_max"]<2e-6,comparison
            if seeded: assert torch.equal(initial,saved)
            graphs={}
            for name,fn in (("old",baseline),("new",candidate)):
                for count in (1,8):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count): result=fn(*args)
                    graphs[name,count]=(graph,result)
            samples={regime:{name:[] for name in ("old","new")} for regime in ("warm","evicted")}
            for iteration in range(11):
                for regime,count in (("warm",8),("evicted",1)):
                    for name in (("old","new") if iteration%2==0 else ("new","old")):
                        graph,_=graphs[name,count]
                        if regime=="evicted": trash.zero_()
                        else: graph.replay()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples[regime][name].append(start.elapsed_time(end)*1000/count)
            row={"tokens":t,"seeded":seeded,"comparisons":comparisons,"samples_us":samples,
                 "median_us":{r:{n:statistics.median(s) for n,s in ns.items()} for r,ns in samples.items()}}
            reports.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
            for graph,_ in graphs.values(): graph.reset()
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [a.baseline_dir/"kda.py",a.baseline_dir/"fused_recurrent.py",
               Path("engine/kernels/kda/kda.py"),Path("engine/kernels/kda/fused_recurrent.py"),
               Path("engine/profiles/glm53/lanes.py")]}
    a.output.write_text(json.dumps({"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "source_sha256":manifest,"cases":reports},indent=2)+"\n")
    print("PASS",flush=True)


if __name__ == "__main__": main()
