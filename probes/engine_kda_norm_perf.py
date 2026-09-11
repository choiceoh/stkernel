"""Compare the public fused output norm with the prior torch expression."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import torch
from engine.kernels.kda.output import kda_output_norm, _output_norm
from engine.modules.linear_attention import kda_output_norm as reference


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
    return {"peak_increment_bytes":peak,"cuda_events":dict(gpu),
            "cuda_event_count":sum(gpu.values()),
            "aten_ops":{e.key:e.count for e in p.key_averages() if e.key.startswith("aten::")}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(92328)
    trash=torch.empty(64*2**20//4,device="cuda")
    cases=[]
    for tokens in (1,6,24,64,256,1024,4096,6912):
        x=torch.randn(tokens,16,128,device="cuda",dtype=torch.bfloat16)*.1
        g=torch.randn_like(x)*3
        w=torch.randn(128,device="cuda",dtype=torch.bfloat16)
        expected=reference(x,g,w);actual=kda_output_norm(x,g,w)
        assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),tokens
        fns={"old":lambda:reference(x,g,w),"new":lambda:kda_output_norm(x,g,w)}
        graphs={}
        for name,fn in fns.items():
            for count in (1,4):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count):out=fn()
                graphs[name,count]=(graph,out)
        samples={r:{n:[] for n in fns} for r in ("warm","evicted")}
        for iteration in range(11):
            for regime,count in (("warm",4),("evicted",1)):
                for name in (("old","new") if iteration%2==0 else ("new","old")):
                    graph,_=graphs[name,count]
                    if regime=="warm":graph.replay()
                    else:trash.zero_()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    samples[regime][name].append(start.elapsed_time(end)*1000/count)
        for graph,out in graphs.values():
            assert torch.equal(out.view(torch.int16),expected.view(torch.int16)),tokens
            graph.reset()
        cases.append({"tokens":tokens,"rows":tokens*16,"bits_exact":True,"samples_us":samples,
                      "median_us":{r:{n:statistics.median(s) for n,s in ns.items()} for r,ns in samples.items()}})
        print(json.dumps({k:v for k,v in cases[-1].items() if k!='samples_us'}),flush=True)
    # Profile after all timing, outside CUDA graph pools.
    operators={}
    for tokens in (6,6912):
        x=torch.randn(tokens,16,128,device="cuda",dtype=torch.bfloat16)
        g=torch.randn_like(x);w=torch.randn(128,device="cuda",dtype=torch.bfloat16)
        operators[tokens]={name:profile(fn) for name,fn in
                          (("old",lambda:reference(x,g,w)),("new",lambda:kda_output_norm(x,g,w)))}
    y=torch.empty_like(x)
    compiled=_output_norm[(tokens*16,)](x,g,w,y,tokens*16,128,1e-6,128,1,
                                        PRECISE=True,num_warps=1,enable_fp_fusion=False)
    files=[Path(p) for p in ("engine/kernels/kda/output.py","engine/modules/linear_attention.py",
                             "engine/profiles/glm53/net.py","engine/profiles/glm53/lanes.py")]
    report={"passed":True,"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(),"cases":cases,"operators":operators,
            "kernel":{"registers":compiled.n_regs,"shared":compiled.metadata.shared,"spills":compiled.n_spills},
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS",flush=True)


if __name__=="__main__":main()
