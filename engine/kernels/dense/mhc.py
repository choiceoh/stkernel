"""Model-owned MK post/pre fusion and one-shot AR consumer PDL.

Immutable coefficient packs and scratch stay alive with the model, across
every captured graph. Calls must be ordered on its execution stream.

The geometry comes from the bound kernel shape (engine/base/kernel_shape):
`hidden` residual streams `hc` wide. The kernel itself is compiled for two
hidden widths (kernels.cu: HIDDEN 4096 and HIDDEN_V41 5120) at hc 4, so a
shape outside that is refused here, by name, before any weight is packed.
"""
import torch
from engine.kernels.dense import extension

MHC_MAX_TOK = 128               # kernels.cu MHC_MAX_TOK_DEF: the token rows the workspace holds
HCHUNK = 256                    # kernels.cu HCHUNK: hidden is walked in chunks of this many
COMPILED_HIDDEN = (4096, 5120)  # kernels.cu HIDDEN and HIDDEN_V41: what mk_run_mhc accepts
COMPILED_HC = 4                 # kernels.cu HC


def geometry(shape=None) -> "tuple[int, int, int, int]":
    """(hidden, hc, nout, nchunk) for the bound kernel shape; a shape the kernel is not compiled for is refused."""
    if shape is None:
        from engine.base.kernel_shape import bound
        shape = bound()
    if shape.hidden not in COMPILED_HIDDEN or shape.hc != COMPILED_HC:
        raise ValueError(f"MK MHC is compiled for hidden {COMPILED_HIDDEN} at hc {COMPILED_HC}; "
                         f"the bound kernel shape asks for hidden {shape.hidden} hc {shape.hc}")
    return shape.hidden, shape.hc, shape.hc * (2 + shape.hc), shape.hidden // HCHUNK


def workspace_sizes(hidden: int, hc: int, nout: int, nchunk: int) -> "list[tuple[int, torch.dtype]]":
    """The kernel's scratch, in its argument order: yp [nchunk, tok, nout], rp [nchunk, tok], sq, pmix [tok, hc],
    ol_stash [tok, hidden] and the barrier word."""
    return [(nchunk * MHC_MAX_TOK * nout, torch.float32), (nchunk * MHC_MAX_TOK, torch.float32),
            (nchunk * MHC_MAX_TOK, torch.float32), (MHC_MAX_TOK * hc, torch.float32),
            (MHC_MAX_TOK * hidden, torch.bfloat16), (8, torch.int32)]


class MHC:
    def __init__(self, weights):
        self.ext = extension()
        self.hidden, self.hc, self.nout, nchunk = geometry()
        device = next(iter(weights.values())).device
        self.weights = {}
        self.executed = set()
        for key, fn in weights.items():
            if fn.shape != (self.nout, self.hc * self.hidden) or fn.dtype != torch.float32 or not fn.is_contiguous():
                raise ValueError(f"MK MHC requires FP32 [{self.nout},{self.hc}*{self.hidden}] weights")
            bf16 = fn.bfloat16()
            # BF16 storage is lossless only for BF16-origin checkpoint values.
            packed = (bf16.reshape(self.nout, self.hc, self.hidden).transpose(1, 2).contiguous()
                      if torch.equal(fn, bf16.float()) else None)
            self.weights[key] = fn, packed
        self.workspace = [torch.zeros(size, device=device, dtype=dtype)
                          for size, dtype in workspace_sizes(self.hidden, self.hc, self.nout, nchunk)]

    def __call__(self,key,x,res,post,comb,scale,base,norm,eps,hc_eps,post_mult,sinkhorn,*,packets=None):
        n = x.shape[0]
        if not 1 <= n <= 64 or res.shape != (n, self.hc, self.hidden):
            raise ValueError("MK MHC decode geometry mismatch")
        small = n <= 8
        fp32, packed = self.weights[key]
        weight = packed if small and packed is not None else fp32
        rc = torch.empty_like(res)
        pm = torch.empty((n, self.hc, 1), device=x.device, dtype=torch.float32)
        cm = torch.empty((n, self.hc, self.hc), device=x.device, dtype=torch.float32)
        li = torch.empty_like(x)
        tensors = [x,res,post,comb,weight,scale,base,norm,rc,pm,cm,li,*self.workspace]
        args = ([t.data_ptr() for t in tensors], [eps,hc_eps,hc_eps,post_mult,eps], [n, sinkhorn, self.hidden])
        if packets is None:
            self.ext.run_mhc(*args,weight is packed,small)
        else:
            if (packets.device != x.device or packets.dtype != torch.int64 or
                    packets.shape != (4,) or not packets.is_contiguous()):
                raise ValueError("MHC needs a same-device contiguous int64[4] rank descriptor")
            self.ext.run_mhc_packets(*args,packets,weight is packed)
        self.executed.add(key)
        return rc,pm,cm,li
