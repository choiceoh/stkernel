"""Owned-GPU matched post-convolution residual/RMS fusion, no fleet operation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.gpu:ap.error('explicit --gpu required')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import torch
    from engine.kernels.draft_conv import tap_mix,tap_add_norm
    from engine.kernels.common.norm_rope import add_norm
    from engine.kernels.dense.cublaslt import _measure,require_timing_target
    prop=torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major,prop.minor),sms=prop.multi_processor_count),'sm120-probe')
    report=dict(status='RUNNING',device=prop.name,torch=torch.__version__,cuda=torch.version.cuda,cells=[],
                scope='synthetic post-convolution/RMS component; not engine step/s or acceptance')
    paths=['engine/kernels/draft_conv.py','engine/kernels/common/norm_rope.py','engine/profiles/glm53/drafter.py','probes/engine_draft_post_norm_check.py']
    report['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        for n,t,g,taps in ((1,6,256,2),(1,7,256,2),(1,8,256,2),(2,8,256,2),(4,8,256,2),
                           (1,8,16,2),(2,8,64,4)):
            m=n*t;h=4096
            torch.manual_seed(1718+m)
            x,res=[torch.randn(m,h,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            delta=torch.randn(m,2,taps,h//g,device='cuda',dtype=torch.bfloat16)[:,1]
            base=torch.randn(taps,h,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(h,device='cuda',dtype=torch.bfloat16)
            def old():return add_norm(res,tap_mix(x,delta,base,g,t),w,1e-6)
            def new():return tap_add_norm(x,delta,base,res,w,1e-6,g,t)
            for scale in (0.,.01,1.,10.):
                x.normal_().mul_(scale);res.normal_().mul_(scale)
                expected=old();actual=new()
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            # Nine post-convolution boundaries in a five-layer proposal.
            # Other drafter work is deliberately excluded from this component chain.
            def chain(fused):
                carry=res;value=x
                for _ in range(9):
                    if fused:carry,value=tap_add_norm(value,delta,base,carry,w,1e-6,g,t)
                    else:carry,value=add_norm(carry,tap_mix(value,delta,base,g,t),w,1e-6)
                return carry,value
            for a,b in zip(chain(True),chain(False)):torch.testing.assert_close(a,b,rtol=0,atol=0)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):new()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):actual=new()
            for scale in (.5,2.):
                x.normal_().mul_(scale);expected=old();torch.cuda.synchronize()
                allocation=torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                graph.replay();torch.cuda.synchronize()
                assert allocation==torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            graph.reset()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _measure(old,None);_measure(new,None)
                times=[[_measure(f,None,repeats=12) for f in (old,new,new,old)] for _ in range(3)]
                _measure(lambda:chain(False),None);_measure(lambda:chain(True),None)
                chain_times=[[_measure(lambda f=f:chain(f),None,repeats=8) for f in (False,True,True,False)]
                             for _ in range(2)]
            cell=dict(rows=m,block=t,group=g,taps=taps,bit_exact=True,graph_replays=2,allocation_bytes=0,baab_ms=times,nine_boundary_baab_ms=chain_times)
            report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
