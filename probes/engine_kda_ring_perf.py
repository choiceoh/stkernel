"""Direct KDA ring versus frozen recurrence with initial/final state staging."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics

import torch
import triton
from engine.kernels.kda.ring import recurrent_kda_ring
from engine.kernels.state import _read_rec, write_ring
from engine_kda_strides_perf import baseline_strides, inputs, paired_timing, kernel_stats, same_bits
from engine_causal_conv_perf import profile


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(129613)
    old_lane=baseline_strides(a.baseline_dir)
    trash=torch.empty(64*2**20//4,device='cuda')
    cases=[];operators={};kernels={}
    for t in (1,2,6):
        for context in (0,4096):
            args=inputs(t,True)
            core_args=args[:7]
            rings=[torch.randn(3,6,16,128,128,device='cuda')*.1 for _ in range(2)]
            slot=torch.tensor(2,device='cuda',dtype=torch.int64)
            ctx=torch.tensor(context,device='cuda',dtype=torch.int64)
            seed=rings[0].clone()
            def old():
                ring=rings[0]
                initial=torch.empty((1,16,128,128),device='cuda')
                _read_rec[(triton.cdiv(initial.numel(),256),)](ring,slot,ctx,initial,
                    ring.stride(0),ring.stride(1),6,initial.numel(),256)
                out,states=old_lane(*core_args,initial,-5.)
                write_ring(states,ring,slot,ctx)
                return out
            def new():return recurrent_kda_ring(*core_args,rings[1],slot,ctx,-5.)
            for r in rings:r.copy_(seed)
            expected=old();actual=new()
            same_bits((actual,rings[1]),(expected,rings[0]))
            graphs={}
            for index,fn in enumerate((old,new)):
                for count in (1,8):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):result=fn()
                    graphs[index,count]=(graph,result)
            samples={regime:[[],[]] for regime in ('warm','evicted')}
            for iteration in range(12):
                for regime,count in (('warm',8),('evicted',1)):
                    for index in ((0,1) if iteration%2==0 else (1,0)):
                        graph,_=graphs[index,count]
                        if regime=='warm':graph.replay()
                        else:trash.zero_()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples[regime][index].append(start.elapsed_time(end)*1000/count)
            # Graphs mutate rings: restore identical input before checking each
            # complete graph (including eight dependent recurrent invocations).
            for count in (1,8):
                for r in rings:r.copy_(seed)
                for index in range(2):graphs[index,count][0].replay()
                same_bits((graphs[0,count][1],rings[0]),(graphs[1,count][1],rings[1]))
            for graph,_ in graphs.values():graph.reset()
            case={'tokens':t,'context':context,'bits_exact':True,'samples_us':samples,
                  'paired':{r:paired_timing(s) for r,s in samples.items()}}
            cases.append(case)
            print(json.dumps(case),flush=True)
    # All timing precedes profiler/CUPTI overhead and kernel introspection.
    for t in (1,6):
        args=inputs(t,True);core_args=args[:7]
        rings=[torch.randn(3,6,16,128,128,device='cuda')*.1 for _ in range(2)]
        slot=torch.tensor(2,device='cuda');ctx=torch.tensor(4096,device='cuda')
        operators[t]={'old':profile(old),'new':profile(new)}
        def direct(*values):return recurrent_kda_ring(*values)
        direct.driver=importlib.import_module('engine.kernels.kda.ring')
        kernels[t]={'old':kernel_stats(old_lane,args),
                    'new':kernel_stats(direct,(*core_args,rings[1],slot,ctx,-5.))}
    files=[Path(p) for p in ('engine/kernels/kda/ring.py','engine/kernels/kda/fused_recurrent.py',
                            'engine/kernels/state.py','engine/profiles/glm53/net.py',
                            'engine/profiles/glm53/lanes.py','engine/profiles/glm53/decode_graphs.py')]
    files += [a.baseline_dir/p for p in ('kda.py','fused_recurrent.py')]
    report={'passed':True,'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(),'cases':cases,'operators':operators,'kernels':kernels,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',flush=True)


if __name__=='__main__':main()
