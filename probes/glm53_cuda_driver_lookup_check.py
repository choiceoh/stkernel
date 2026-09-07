"""Minimal CUDA binding initialization diagnostic; no MoE or INT8 kernels."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('torch-only','driver-only','torch-driver'),required=True)
    args=ap.parse_args()
    versions={k:importlib.metadata.version(k) for k in ('cuda-bindings','cuda-python','nvidia-cutlass-dsl','torch')}
    print(json.dumps(dict(kind='DRIVER_LOOKUP_BEGIN',mode=args.mode,versions=versions)),flush=True)
    if args.mode!='driver-only':
        import torch
        torch.cuda.init()
        x=torch.ones(1,device='cuda');x.add_(1);torch.cuda.synchronize()
        assert x.item()==2
        print(json.dumps(dict(kind='TORCH_CONTROL_PASS',mode=args.mode,capability=torch.cuda.get_device_capability())),flush=True)
    calls=[]
    if args.mode!='torch-only':
        from cuda.bindings import driver
        for round in range(2):
            for name,arguments in (('cuDriverGetVersion',()),('cuInit',(0,)),('cuDeviceGetCount',())):
                print(json.dumps(dict(kind='DRIVER_CALL_BEGIN',round=round,call=name)),flush=True)
                values=getattr(driver,name)(*arguments)
                record=dict(round=round,call=name,code=int(values[0]),values=[int(v) for v in values[1:]])
                calls.append(record);print(json.dumps(dict(kind='DRIVER_CALL_END',**record)),flush=True)
                assert record['code']==0,record
    print(json.dumps(dict(kind='DRIVER_LOOKUP_PROGRAM_COMPLETE',mode=args.mode,calls=calls,versions=versions,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        serving_gate=False,numerical_acceptance=False)),flush=True)


if __name__=='__main__':main()
