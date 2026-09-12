"""Real-weight b12x migration and activation-rounding diagnosis on SM121.

Compares the original FlashInfer implementation on identical rank bytes.
Repeats measure scatter stability separately from activation quantization.
The native result is gated against the independent reciprocal oracle and
its own repeat spread. Legacy BF16 scatter results remain a diagnostic
control because the native GLM decode lane now accumulates in FP32.
An independent dequant/GEMM oracle evaluates both exact division and the
PTX reciprocal/FP8 rounding sequence used by the native activation packer.
This diagnostic never changes the serving kernel or the reference lane.
"""
import torch
import triton as tr
import triton.language as tl
from engine.modules import moe
from engine.modules.quant import _fp4_encode

@tr.jit
def scaled(X, Y, S, GS, N: tl.constexpr, MULTIPLIER: tl.constexpr = False):
    block = tl.program_id(0)
    i = block * 16 + tl.arange(0, 16)
    x = tl.load(X+i).to(tl.float32)
    gs = tl.load(GS)
    rgs = gs if MULTIPLIER else tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;', '=f,f', [gs], dtype=tl.float32, is_pure=True, pack=1)
    rsix = tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;', '=f,f', [tl.full((), 6., tl.float32)], dtype=tl.float32, is_pure=True, pack=1)
    s = tl.minimum(rgs * (tl.max(tl.abs(x),0) * rsix), 448.).to(tl.float8e4nv).to(tl.float32)
    inv = tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;', '=f,f', [s], dtype=tl.float32, is_pure=True, pack=1)
    inv = tl.where(s == 0., 0., inv)
    tl.store(Y+i, x * (inv * rgs))
    tl.store(S+block, s)

def hardware_quant(x, gs, *, multiplier=False):
    x = x.contiguous()
    y = torch.empty_like(x, dtype=torch.float32)
    s = torch.empty((*x.shape[:-1],x.shape[-1]//16), device=x.device, dtype=torch.float32)
    scaled[(x.numel()//16,)](x,y,s,gs, x.numel(), MULTIPLIER=multiplier, num_warps=1, enable_fp_fusion=False)
    nib = _fp4_encode(y).unflatten(-1,(x.shape[-1]//2,2))
    return (nib[...,0] | (nib[...,1]<<4)).view(torch.float4_e2m1fn_x2), s.to(torch.float8_e4m3fn)


def relative(a,b):
    return ((a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1e-6)).item()


def main():
    import argparse
    import json
    from pathlib import Path
    from engine.profiles.glm53.weights import rank_loader
    from engine.profiles.glm53.lanes import served, swiglu_clamped
    from engine.modules.nvfp4_sf import unswizzle_sf, mma_sf_view
    from flashinfer.fused_moe import b12x_fused_moe as original
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--layer',type=int,default=3)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=64)
    ap.add_argument('--tokens',type=int,nargs='+',default=[1,4,6,12,18,24])
    ap.add_argument('--moe-static',default='stock')
    a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    names=[f'L{a.layer}.moe.{key}' for key in ('w13','w13_sf','w2','w2_sf')]
    got=rank_loader(Path(a.ranks)/'rank0of4.safetensors').load(names,device='cuda')
    w13,s13,w2,s2=[got[name] for name in names]
    e,two_i,half_h=w13.shape
    h,i=half_h*2,two_i//2
    sf13,sf2=mma_sf_view(s13,two_i,h),mma_sf_view(s2,h,i)
    ones=torch.ones(e,device='cuda'); one=ones[0]
    weights=[]
    for expert in range(16):
        first=unswizzle_sf(s13[expert].view(torch.uint8),two_i,h//16).view(torch.float8_e4m3fn)
        second=unswizzle_sf(s2[expert].view(torch.uint8),h,i//16).view(torch.float8_e4m3fn)
        weights.append((moe.dequant_nvfp4(w13[expert],first,one),moe.dequant_nvfp4(w2[expert],second,one)))
    def oracle(x,sel,route,quant):
        packed,scale=quant(x,one)
        xq=moe.dequant_nvfp4_act(packed,scale,one)
        out=torch.zeros_like(x,dtype=torch.float32)
        for expert,(first,second) in enumerate(weights):
            up,gate=(xq@first.T).chunk(2,-1)
            hidden=swiglu_clamped(gate,up,10.)
            hp,hs=quant(hidden,one)
            hq=moe.dequant_nvfp4_act(hp,hs,one)
            coefficient=((sel==expert)*route).sum(-1,keepdim=True)
            out+=((hq@second.T)*coefficient).bfloat16().float()
        return out.bfloat16()
    lane=served(moe_static=a.moe_static)
    rows=[]
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        for tokens,mode in ((t,m) for t in a.tokens for m in ('shared','mixed')):
            x=torch.randn(tokens,h,device='cuda',dtype=torch.bfloat16)*.5
            if mode=='shared':
                sel=torch.arange(8,device='cuda',dtype=torch.int32).repeat(tokens,1)
                route=torch.full((tokens,8),1/8,device='cuda')
            else:
                sel=torch.rand(tokens,16,device='cuda').topk(8,-1).indices.int()
                route=torch.rand(tokens,8,device='cuda')
                route[0,0]=0.  # zero-weight routes must not contribute
                route/=route.sum(-1,keepdim=True)
            def native_call():
                return lane.moe(x,sel,route,w13,s13,w2,s2,10.)
            def original_call():
                return original(x=x,w1_weight=w13,w1_weight_sf=sf13,w2_weight=w2,w2_weight_sf=sf2,
                token_selected_experts=sel,token_final_scales=route,num_experts=e,top_k=8,
                w1_alpha=ones,w2_alpha=ones,fc2_input_scale=ones,
                activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,
                swiglu_limit=10.,activation_precision='fp4',quant_mode='nvfp4')
            for _ in range(3):
                native_call(); original_call()
            native_runs=torch.stack([native_call().clone().float() for _ in range(a.repeats)])
            original_runs=torch.stack([original_call().clone().float() for _ in range(a.repeats)])
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured=native_call()
            graph_runs=[]
            for _ in range(a.repeats):
                graph.replay()
                graph_runs.append(captured.clone().float())
            graph_runs=torch.stack(graph_runs)
            out=native_runs.mean(0); expect=original_runs.mean(0)
            exact=oracle(x,sel,route,moe.quant_nvfp4_act); fast=oracle(x,sel,route,hardware_quant)
            p,s=moe.quant_nvfp4_act(x,one); q,t=hardware_quant(x,one)
            def milliseconds(call):
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(a.repeats): call()
                end.record(); end.synchronize()
                return start.elapsed_time(end)/a.repeats
            row=dict(seed=seed,tokens=tokens,routing=mode,repeats=a.repeats,moe_static=a.moe_static,original_relative=relative(out,expect),
                native_repeat_relative=relative(native_runs,native_runs[0]),
                graph_relative=relative(graph_runs,native_runs[0]),
                original_repeat_relative=relative(original_runs,original_runs[0]),
                unaveraged_pair_relative=relative(native_runs,original_runs),
                division_oracle_relative=relative(out,exact),reciprocal_oracle_relative=relative(out,fast),
                activation_changed_bytes=int((p.view(torch.uint8)!=q.view(torch.uint8)).sum()),
                activation_scale_changes=int((s.float()!=t.float()).sum()),
                native_ms=milliseconds(native_call),original_ms=milliseconds(original_call),
                graph_ms=milliseconds(graph.replay))
            graph.reset()
            rows.append(row); print(json.dumps(row),flush=True)
            assert torch.isfinite(native_runs).all() and relative(native_runs,fast)<=.02,row
            assert row['native_repeat_relative']<=.001,row
            assert row['graph_relative']<=.001,row
    print(json.dumps(dict(passed=True,checks=rows)),flush=True)

if __name__=='__main__':
    main()
