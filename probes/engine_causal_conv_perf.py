"""Compare complete old/new single-sequence conv calls on a TP4 projection slice."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
from engine.kernels.causal_conv_single import causal_conv1d_single, _single_conv


def baseline_conv(path):
    spec = importlib.util.spec_from_file_location("engine.profiles.glm53.baseline_lanes",path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.served(reference_for=("expert",)).conv_prefill


def exact(actual,expected):
    for x,y in zip(actual,expected):
        assert x.shape==y.shape and x.dtype==y.dtype
        assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))


def profile(fn):
    fn();torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial=torch.cuda.memory_allocated()
    output=fn();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-initial
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as p:
        output=fn()
    torch.cuda.synchronize()
    gpu=Counter(e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA)
    assert gpu,"CUDA profiler did not capture kernel events"
    return {"peak_increment_bytes":peak,"cuda_events":dict(gpu),"cuda_event_count":sum(gpu.values()),
            "aten_ops":{e.key:e.count for e in p.key_averages() if e.key.startswith("aten::")}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-lanes",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(129128)
    old=baseline_conv(a.baseline_lanes)
    trash=torch.empty(64*2**20//4,device="cuda")
    cases=[]
    for tokens in (1,6,24,64,256,1024,4096,6912):
        for initialized in (False,True):
            x=torch.randn(tokens,6416,device="cuda",dtype=torch.bfloat16)[:, :6144]
            w=torch.randn(6144,4,device="cuda")
            s=torch.randn(6144,3,device="cuda",dtype=x.dtype) if initialized else None
            expected=old(x,w,s);exact(causal_conv1d_single(x,w,s),expected)
            fns={"old":lambda:old(x,w,s),"new":lambda:causal_conv1d_single(x,w,s)}
            graphs={}
            for name,fn in fns.items():
                for count in (1,16):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):out=fn()
                    graphs[name,count]=(graph,out)
            samples={r:{n:[] for n in fns} for r in ("warm","evicted")}
            for iteration in range(11):
                for regime,count in (("warm",16),("evicted",1)):
                    for name in (("old","new") if iteration%2==0 else ("new","old")):
                        graph,_=graphs[name,count]
                        if regime=="warm":graph.replay()
                        else:trash.zero_()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples[regime][name].append(start.elapsed_time(end)*1000/count)
            for graph,out in graphs.values():
                exact(out,expected);graph.reset()
            cases.append({"tokens":tokens,"initialized":initialized,"bits_exact":True,"samples_us":samples,
                          "median_us":{r:{n:statistics.median(v) for n,v in ns.items()} for r,ns in samples.items()}})
            print(json.dumps({k:v for k,v in cases[-1].items() if k!='samples_us'}),flush=True)
    operators={}
    kernels={}
    for tokens in (1,6,6912):
        x=torch.randn(tokens,6416,device="cuda",dtype=torch.bfloat16)[:, :6144]
        w=torch.randn(6144,4,device="cuda");s=torch.randn(6144,3,device="cuda",dtype=x.dtype)
        operators[tokens]={name:profile(fn) for name,fn in
                          (("old",lambda:old(x,w,s)),("new",lambda:causal_conv1d_single(x,w,s)))}
        y,f=causal_conv1d_single(x,w,s)
        compiled=_single_conv[((6144+127)//128,(tokens+7)//8)](
            x,w,s,y,f,tokens,6144,*x.stride(),*w.stride(),*s.stride(),4,True,128,8,
            num_warps=4,num_stages=2)
        kernels[tokens]={"registers":compiled.n_regs,"shared":compiled.metadata.shared,
                         "spills":compiled.n_spills,"channels_per_cta":128,"tokens_per_cta":8,"warps":4}
    files=[Path(p) for p in ("engine/kernels/causal_conv_single.py","engine/kernels/causal_conv.py",
                             "engine/profiles/glm53/net.py","engine/profiles/glm53/lanes.py")]
    report={"passed":True,"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(),"cases":cases,"operators":operators,"kernels":kernels,
            "baseline_lanes_sha256":hashlib.sha256(a.baseline_lanes.read_bytes()).hexdigest(),
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS",flush=True)


if __name__=="__main__":main()
