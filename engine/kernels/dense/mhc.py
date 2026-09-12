"""Model-owned MK post/pre fusion and one-shot AR consumer PDL.

Immutable coefficient packs and scratch stay alive with the model, across
every captured graph. Calls must be ordered on its execution stream.
"""
import torch
from engine.kernels.dense import extension


class MHC:
    def __init__(self, weights):
        self.ext = extension()
        device = next(iter(weights.values())).device
        self.weights = {}
        self.executed = set()
        for key, fn in weights.items():
            if fn.shape != (24,16384) or fn.dtype != torch.float32 or not fn.is_contiguous():
                raise ValueError("MK MHC requires FP32 [24,4*4096] weights")
            bf16 = fn.bfloat16()
            # BF16 storage is lossless only for BF16-origin checkpoint values.
            packed = bf16.reshape(24,4,4096).transpose(1,2).contiguous() if torch.equal(fn,bf16.float()) else None
            self.weights[key] = fn, packed
        self.workspace = [torch.zeros(size,device=device,dtype=dtype) for size,dtype in (
            (16*128*24,torch.float32),(16*128,torch.float32),(16*128,torch.float32),
            (128*4,torch.float32),(128*4096,torch.bfloat16),(8,torch.int32))]

    def __call__(self,key,x,res,post,comb,scale,base,norm,eps,hc_eps,post_mult,sinkhorn):
        n = x.shape[0]
        if not 1 <= n <= 32 or res.shape != (n,4,4096):
            raise ValueError("MK MHC decode geometry mismatch")
        small = n <= 8
        fp32, packed = self.weights[key]
        weight = packed if small and packed is not None else fp32
        rc = torch.empty_like(res)
        pm = torch.empty((n,4,1),device=x.device,dtype=torch.float32)
        cm = torch.empty((n,4,4),device=x.device,dtype=torch.float32)
        li = torch.empty_like(x)
        tensors = [x,res,post,comb,weight,scale,base,norm,rc,pm,cm,li,*self.workspace]
        self.ext.run_mhc([t.data_ptr() for t in tensors],
                         [eps,hc_eps,hc_eps,post_mult,eps],[n,sinkhorn,4096],
                         weight is packed,small)
        self.executed.add(key)
        return rc,pm,cm,li
