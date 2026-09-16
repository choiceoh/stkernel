"""Explicit FC reduction/RMS fusion comparison on the owned SM120 probe."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch, triton
from engine.kernels.dense.cublaslt_split import _reduce, _reduce_norm
from engine.kernels.dense.cublaslt import _measure
from engine.kernels.common.norm_rope import norm

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',action='store_true');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.gpu or torch.cuda.get_device_capability()!=(12,0):ap.error('owned SM120 GPU required')
    report=dict(status='RUNNING',cells=[])
    try:
        for m in (1,7,8,16,24,32,64):
            for has_bias in (False,True):
                torch.manual_seed(1700+m)
                x=torch.randn(5,m,4096,device='cuda',dtype=torch.float32)
                w=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
                b=torch.randn(4096,device='cuda',dtype=torch.float32) if has_bias else None
                temp=torch.empty(m,4096,device='cuda',dtype=torch.bfloat16);out=torch.empty_like(temp)
                result=None
                def base():
                    nonlocal result
                    _reduce[(triton.cdiv(m*4096,256),)](x,temp,m*4096,5,num_warps=4)
                    result=norm(temp,w,1e-6,bias=b)
                def trial():_reduce_norm[(m,)](x,w,w if b is None else b,out,m,4096,5,1e-6,has_bias,4096,num_warps=8)
                base();trial();torch.testing.assert_close(out,result,rtol=0,atol=0)
                _measure(base,None);_measure(trial,None)
                times=[_measure(fn,None) for fn in (base,trial,trial,base)]
                x.normal_();base();trial();torch.testing.assert_close(out,result,rtol=0,atol=0)
                report['cells'].append(dict(rows=m,bias=has_bias,bit_exact=True,base_trial_trial_base_ms=times))
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=repr(e));raise
    finally:args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()):main()
