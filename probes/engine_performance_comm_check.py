"""Bounded TP4 correctness/capture check for native AR and prefill FP8 packets."""
import json
import os
import torch
import torch.distributed as dist
from engine.base.comm import Comm
from engine.kernels.prefill_collectives import PrefillCollectives


def quantized(x):
    blocks=x.float().reshape(-1,2048)
    scale=torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(1).clamp_min(1e-30)/448.)))
    return ((blocks/scale[:,None]).to(torch.float8_e4m3fn).float()*scale[:,None]).reshape(x.shape)


def main():
    torch.cuda.set_device(0)
    comm=Comm.init(timeout_s=180)
    graphs=[]
    try:
        comm.prepare_oneshot()
        torch.manual_seed(138+comm.rank)
        for n in (1,6,12,24,32):
            x=torch.randn(n,4096,device='cuda',dtype=torch.bfloat16)
            comm.all_reduce(x)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y=comm.all_reduce(x)
                y=comm.all_reduce(y)
            graphs.append(g)
            for _ in range(5):
                x.normal_()
                ref=x.clone();dist.all_reduce(ref,group=comm.group)
                ref=ref*4
                g.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(y,ref,rtol=.02,atol=.125)
            print(json.dumps(dict(rank=comm.rank,ar_rows=n,passed=True)),flush=True)
        sp=PrefillCollectives(comm)
        for rows in (128,2128,4096,4100,6912):
            x=torch.randn(rows//4,4096,device='cuda',dtype=torch.bfloat16)
            expected=comm.all_gather(quantized(x).bfloat16() if rows>=4096 else x,dim=0)
            actual=sp.all_gather(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            partial=torch.randn(rows,4096,device='cuda',dtype=torch.bfloat16)
            if rows>=4096:
                ref=quantized(partial)
                dist.all_reduce(ref,group=comm.group)
                ref=ref.chunk(4)[comm.rank].bfloat16()
            else:
                ref=comm.reduce_scatter_rows(partial)
            actual=sp.reduce_scatter(partial)
            torch.testing.assert_close(actual,ref,rtol=.008,atol=.03125)
            print(json.dumps(dict(rank=comm.rank,prefill_rows=rows,passed=True)),flush=True)
        comm.barrier()
    finally:
        for g in graphs:g.reset()
        comm.close()


if __name__=='__main__':main()
