"""Direct owned-device residual/RMS/head producer comparison; no fleet operations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--packed-weight', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('explicit --gpu on an owned device required')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import torch
    from engine.kernels.common.norm_rope import add_norm
    from engine.kernels.dense.cublaslt_producer import add_norm_head
    from engine.kernels.dense import FP8Linear
    from engine.profiles.glm53.net import Glm53Net
    from engine.kernels.dense.cublaslt import _measure, require_timing_target
    from engine.kernels.dense import mxfp8
    from probes.engine_cublaslt_check import packed_weight
    prop=torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major,prop.minor),sms=prop.multi_processor_count),'sm120-probe')
    weight,receipt=packed_weight(args.packed_weight,38784,4096)
    weight=tuple(x.cuda() for x in weight)
    layer=FP8Linear(weight[0],quantized=weight)
    layer.rows=38720  # logical vocabulary partition excludes physical GEMM padding
    layer.prepare_cublas()
    net=Glm53Net.__new__(Glm53Net);net.dense={'head':layer}
    reader=layer.cublas
    report=dict(status='RUNNING',device=prop.name,torch=torch.__version__,cuda=torch.version.cuda,
                weight=receipt,cells=[],scope='5050 component, synthetic activations; no GB10/engine speed claim')
    paths=['engine/kernels/dense/cublaslt_producer.py','engine/kernels/dense/cublaslt_serving.py',
           'engine/kernels/common/norm_rope.py','engine/kernels/dense/__init__.py',
           'engine/profiles/glm53/net.py','engine/profiles/glm53/drafter.py','engine/profiles/glm53/cublas.py','probes/engine_cublaslt_producer_check.py']
    report['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        for n,t in ((1,8),(2,8),(4,8),(8,8),(1,2),(2,5),(128,2),(19,8)):
            torch.manual_seed(2026+n+t)
            a,b=[torch.randn(n*t,4096,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            w=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
            def inputs_base():
                h=add_norm(a,b,w,1e-6)[1]
                selected=h.view(n,t,4096)[:,1:].reshape(-1,4096).contiguous()
                return selected,mxfp8.quantize(selected,num_warps=1)
            def inputs_trial():return add_norm_head(a,b,w,1e-6,t)
            def check():
                h,(q,s)=inputs_base();hh,(qq,ss)=inputs_trial()
                torch.testing.assert_close(hh,h,rtol=0,atol=0)
                assert torch.equal(q.view(torch.uint8),qq.view(torch.uint8))
                assert torch.equal(s,ss)
            check()
            def base():
                h=add_norm(a,b,w,1e-6)[1].view(n,t,4096)[:,1:].reshape(-1,4096).contiguous()
                return net.head_local(h)
            def trial():
                h,pack=inputs_trial()
                return net.head_local(h,producer_pack=pack)
            expected=base();actual=trial();torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):trial()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):actual=trial()
            second=torch.cuda.CUDAGraph()
            with torch.cuda.graph(second,stream=stream):independent=trial()
            second.replay();torch.cuda.synchronize()
            snapshot=independent.clone()
            assert actual.data_ptr()!=independent.data_ptr()
            for scale in (.5,2.):
                a.normal_().mul_(scale);b.normal_();check();expected=base();torch.cuda.synchronize()
                allocated=torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                graph.replay();torch.cuda.synchronize()
                assert allocated==torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(independent,snapshot,rtol=0,atol=0)
            graph.reset();second.reset()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _measure(base,None);_measure(trial,None)
                timings=[[_measure(fn,None) for fn in (base,trial,trial,base)] for _ in range(2)]
                producer=[[_measure(fn,None) for fn in (inputs_base,inputs_trial,inputs_trial,inputs_base)] for _ in range(2)]
            cell=dict(blocks=n,block_rows=t,head_rows=n*(t-1),bit_exact=True,graph_replays=2,
                      replay_allocation_bytes=0,independent_graph_outputs=True,whole_baab_ms=timings,producer_baab_ms=producer)
            report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
