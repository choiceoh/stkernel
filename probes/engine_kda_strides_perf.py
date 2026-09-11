"""Frozen canonical KDA driver versus direct strided loads, including input copies."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch
from engine.profiles.glm53 import lanes
from engine_kda_state_perf import load
from engine_causal_conv_perf import profile


def baseline_strides(directory):
    old = load(directory/"kda.py", "baseline_kda_strides")
    old.fused_recurrent_gated_delta_rule_fwd_kernel = load(
        directory/"fused_recurrent.py", "baseline_recurrent_strides").fused_recurrent_gated_delta_rule_fwd_kernel
    def run(q,k,v,g,beta,a,bias,state,lb):
        initial = state if state is not None else torch.zeros(
            1,v.shape[2],k.shape[-1],v.shape[-1],device=q.device,dtype=torch.float32)
        return old.fused_recurrent_kda(q,k,v,g,beta,initial_state=initial,
            inplace_final_state=False,use_qk_l2norm_in_kernel=True,sigmoid_beta=True,
            a_log=a,g_bias=bias,compute_gate=True,lower_bound=lb,state_layout="kv")
    run.driver = old
    return run


def same_bits(actual, expected):
    for x,y in zip(actual,expected):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))


def kernel_stats(fn, args):
    driver=getattr(fn,"driver",None) or importlib.import_module("engine.kernels.kda.kda")
    kernel=driver.fused_recurrent_gated_delta_rule_fwd_kernel
    result={}
    class Recorder:
        def __getitem__(self,grid):
            def launch(*values,**kwargs):
                compiled=kernel[grid](*values,**kwargs)
                result.update(registers=compiled.n_regs,shared=compiled.metadata.shared,
                              spills=compiled.n_spills,BK=kwargs["BK"],BV=kwargs["BV"],
                              warps=kwargs["num_warps"],grid=list(grid))
                return compiled
            return launch
    with patch.object(driver,"fused_recurrent_gated_delta_rule_fwd_kernel",Recorder()):
        fn(*args)
    return result


def inputs(tokens, initialized):
    x=torch.randn(tokens,6144,device="cuda",dtype=torch.bfloat16)
    q,k,v=(p.reshape(1,tokens,16,128) for p in x.split(2048,dim=-1))
    proj=torch.randn(tokens,6416,device="cuda",dtype=torch.bfloat16)
    beta=proj[:,6144:6160][None]
    g=torch.randn(1,tokens,16,128,device="cuda",dtype=torch.bfloat16)
    a=torch.randn(16,device="cuda")*.2
    bias=torch.randn(2048,device="cuda")*.1
    state=torch.randn(1,16,128,128,device="cuda")*.1 if initialized else None
    return q,k,v,g,beta,a,bias,state,-5.


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(129233)
    old=baseline_strides(a.baseline_dir)
    new=lanes.served(reference_for=("expert",)).kda_recurrent
    trash=torch.empty(64*2**20//4,device="cuda")
    cases=[]
    for tokens in (1,2,3,4,5,6,7,12):
        for initialized in (False,True):
            args=inputs(tokens,initialized)
            expected=old(*args);same_bits(new(*args),expected)
            graphs={}
            for name,fn in (("old",old),("new",new)):
                for count in (1,8):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):result=fn(*args)
                    graphs[name,count]=(graph,result)
            samples={r:{n:[] for n in ("old","new")} for r in ("warm","evicted")}
            for iteration in range(11):
                for regime,count in (("warm",8),("evicted",1)):
                    for name in (("old","new") if iteration%2==0 else ("new","old")):
                        graph,_=graphs[name,count]
                        if regime=="warm":graph.replay()
                        else:trash.zero_()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples[regime][name].append(start.elapsed_time(end)*1000/count)
            for graph,result in graphs.values():same_bits(result,expected);graph.reset()
            cases.append({"tokens":tokens,"initialized":initialized,"bits_exact":True,"samples_us":samples,
                          "median_us":{r:{n:statistics.median(s) for n,s in ns.items()} for r,ns in samples.items()}})
            print(json.dumps({k:v for k,v in cases[-1].items() if k!='samples_us'}),flush=True)
    operators={}
    kernels={}
    for tokens in (1,6):
        args=inputs(tokens,True)
        operators[tokens]={"old":profile(lambda:old(*args)),"new":profile(lambda:new(*args))}
        kernels[tokens]={"old":kernel_stats(old,args),"new":kernel_stats(new,args)}
    files=[Path(p) for p in ("engine/kernels/kda/kda.py","engine/kernels/kda/fused_recurrent.py",
                             "engine/profiles/glm53/net.py","engine/profiles/glm53/lanes.py")]
    files += [a.baseline_dir/p for p in ("kda.py","fused_recurrent.py")]
    report={"passed":True,"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(),"cases":cases,"operators":operators,"kernels":kernels,
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS",flush=True)


if __name__=="__main__":main()
