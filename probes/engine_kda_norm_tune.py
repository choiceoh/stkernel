"""Bounded GB10 KDA output fusion sweep versus the previous torch expression."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from engine.kernels.kda.output import _output_norm


def reference(x,g,w,eps=1e-6):
    f=x.float()
    return (f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+eps)*w.float()*torch.sigmoid(g.float())).to(x.dtype)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rows',type=int,nargs='+',default=[16,96,384,1024,4096,16384,65536,110592])
    a=ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((1536*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(61128)
    report=[]
    for rows in a.rows:
        x=torch.randn(rows,128,device='cuda',dtype=torch.bfloat16)*.1
        g=torch.randn_like(x)*3
        w=torch.randn(128,device='cuda',dtype=torch.bfloat16)
        expected=reference(x,g,w)
        configs=[('torch',None,None,False)]+[(f'r{br}w{nw}p{int(p)}',br,nw,p)
                 for br,nw in ((1,1),(4,1),(8,1),(16,2),(32,4)) for p in (False,True)]
        functions,graphs,metadata,errors={},{},{},{}
        for name,br,nw,p in configs:
            out=torch.empty_like(x)
            def run(br=br,nw=nw,p=p,out=out):
                if br is None:return reference(x,g,w),None
                kernel=_output_norm[((rows+br-1)//br,)](x,g,w,out,rows,128,1e-6,128,br,
                    PRECISE=p,num_warps=nw,enable_fp_fusion=False)
                return out,kernel
            out,kernel=run()
            functions[name]=run
            diff=(out.float()-expected.float()).abs()
            errors[name]={'bits_exact':torch.equal(out.view(torch.int16),expected.view(torch.int16)),
                'relative_max':(diff.max()/expected.float().abs().max()).item(),
                'different_elements':int((out!=expected).sum()),'elements':x.numel()}
            assert errors[name]['relative_max']<.008,(rows,name,errors[name])
            if kernel:metadata[name]={'registers':kernel.n_regs,'shared':kernel.metadata.shared,'spills':kernel.n_spills}
        for name,run in functions.items():
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4): result,_=run()
            graphs[name]=(graph,result)
        samples={name:[] for name in graphs}
        for tick in range(9):
            names=list(graphs)
            if tick%2:names.reverse()
            for name in names:
                graph,_=graphs[name];graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/4)
        row={'rows':rows,'errors':errors,'metadata':metadata,'samples_us':samples,
             'median_us':{n:statistics.median(s) for n,s in samples.items()}}
        report.append(row)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
        for graph,_ in graphs.values():graph.reset()
    print('PASS',flush=True)


if __name__=='__main__':main()
