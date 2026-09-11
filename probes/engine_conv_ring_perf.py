"""Short conv with ring gather/write versus one direct ring kernel on GB10."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from engine.kernels.causal_conv_ring import causal_conv1d_ring
from engine.kernels.state import conv_history, write_conv
from engine_kda_strides_perf import paired_timing, same_bits
from engine_causal_conv_perf import profile


def load_baseline(path):
    spec=importlib.util.spec_from_file_location('st_conv_ring_baseline',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def compiled_stats(driver, fn):
    kernel=driver._single_conv
    result={}
    class Recorder:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                compiled=kernel[grid](*args,**kwargs)
                result.update(registers=compiled.n_regs,shared_bytes=compiled.metadata.shared,
                    spills=compiled.n_spills,grid=list(grid),warps=kwargs['num_warps'],
                    ptx_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest())
                return compiled
            return launch
    with patch.object(driver,'_single_conv',Recorder()):fn()
    return result


def inputs(t):
    x=torch.randn(t,6416,device='cuda',dtype=torch.bfloat16)[:,:6144]
    w=torch.randn(6144,4,device='cuda')
    storage=torch.randn(3*(6144*8+64)+64,device='cuda',dtype=x.dtype)
    ring=storage.as_strided((3,6144,8),(6144*8+64,8,1),64)
    return x,w,storage,ring


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-conv',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(129621)
    baseline=load_baseline(a.baseline_conv)
    new_driver=importlib.import_module('engine.kernels.causal_conv_ring')
    trash=torch.empty(64*2**20//4,device='cuda')
    cases=[];operators={};kernels={}
    def pipelines(t,context):
        x,w,storage,ring=inputs(t)
        other=storage.clone();new_ring=other.as_strided(ring.shape,ring.stride(),ring.storage_offset())
        slot=torch.tensor(2,device='cuda',dtype=torch.int64)
        ctx=torch.tensor(context,device='cuda',dtype=torch.int64)
        def old():
            hist=conv_history(ring,slot,ctx,3)
            y,_=baseline.causal_conv1d_single(x,w,hist)
            write_conv(x,ring,slot,ctx)
            return y
        def new():return causal_conv1d_ring(x,w,new_ring,slot,ctx)
        return (old,new),(storage,other)
    for t in (1,2,6,8):
        for context in (0,1,4096):
            fns,storages=pipelines(t,context)
            expected=fns[0]();actual=fns[1]()
            same_bits((actual,storages[1]),(expected,storages[0]))
            seed=storages[0].clone()
            graphs={}
            for i,fn in enumerate(fns):
                for count in (1,16):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):result=fn()
                    graphs[i,count]=(graph,result)
            samples={regime:[[],[]] for regime in ('warm','evicted')}
            for iteration in range(12):
                for regime,count in (('warm',16),('evicted',1)):
                    for i in ((0,1) if iteration%2==0 else (1,0)):
                        graph,_=graphs[i,count]
                        if regime=='warm':graph.replay()
                        else:trash.zero_()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples[regime][i].append(start.elapsed_time(end)*1000/count)
            for count in (1,16):
                for s in storages:s.copy_(seed)
                for i in range(2):graphs[i,count][0].replay()
                same_bits((graphs[0,count][1],storages[0]),(graphs[1,count][1],storages[1]))
            for graph,_ in graphs.values():graph.reset()
            case={'tokens':t,'context':context,'bits_exact':True,'samples_us':samples,
                  'paired':{r:paired_timing(s) for r,s in samples.items()}}
            cases.append(case)
            print(json.dumps(case),flush=True)
    # CUPTI is enabled only after all timings have completed.
    for t in (1,6):
        fns,_=pipelines(t,4096)
        operators[t]={name:profile(fn) for name,fn in zip(('old','new'),fns)}
        kernels[t]={name:compiled_stats(driver,fn) for name,driver,fn in zip(
            ('old','new'),(baseline,new_driver),fns)}
    sources=[Path(p) for p in ('engine/kernels/causal_conv_single.py','engine/kernels/causal_conv_ring.py',
        'engine/kernels/state.py','engine/profiles/glm53/net.py','engine/profiles/glm53/lanes.py',
        'engine/profiles/glm53/decode_graphs.py')]+[a.baseline_conv]
    result={'passed':True,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cuda':torch.version.cuda,'cases':cases,'operators':operators,'kernels':kernels,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',flush=True)


if __name__=='__main__':main()
