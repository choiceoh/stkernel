"""Read-only L3 pilot: magnitude pair pruning, K32 rescaling, native GEMM.

This is an uncalibrated diagnostic, NOT an accuracy-preserving conversion.
Gaussian inputs measure local projection distortion, not language-model
quality, real activation distributions, or downstream routing changes.
No checkpoint files are written; at most eight selected experts are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import torch
from engine.modules.nvfp4_sf import unswizzle_sf
from probes.engine_sparse_nvfp4 import Library, Projection, dequant, qualify, benchmark


def read_experts(path, layer, name, experts):
    """Read bounded byte ranges, avoiding whole-rank loading or GPU allocation."""
    with open(path,'rb') as f:
        count = struct.unpack('<Q',f.read(8))[0]
        if count > 64*1024**2:
            raise ValueError('unreasonably large safetensors header')
        header = json.loads(f.read(count))
        if header.get('__metadata__',{}).get('weight_layout') != 'st-glm53-b12x-up-gate-v1':
            raise ValueError('requires known up|gate rank layout')
        base = 8+count
        spec = header[f'L{layer}.moe.{name}']
        e,m,hk = spec['shape']
        expected = {'w13':(288,1024,2048),'w2':(288,4096,256)}[name]
        if (e,m,hk) != expected or spec['dtype'] != 'U8':
            raise ValueError('requires GLM TP4 expert shapes')
        if any(i < 0 or i >= e for i in experts):
            raise ValueError('expert index out of bounds')
        result, hashes = [], {}
        for suffix in ('','_sf'):
            key=f'L{layer}.moe.{name}{suffix}'
            entry=header[key]
            lo,hi=entry['data_offsets']
            size=(hi-lo)//e
            expected_bytes=m*(hk//8 if suffix else hk)
            if hi-lo != size*e or size != expected_bytes:
                raise ValueError('unexpected tensor byte size')
            selected=[]
            for expert in experts:
                raw=os.pread(f.fileno(),size,base+lo+expert*size)
                if len(raw)!=size:
                    raise ValueError('truncated tensor')
                hashes[f'{key}.{expert}']=hashlib.sha256(raw).hexdigest()
                tensor=torch.frombuffer(bytearray(raw),dtype=torch.uint8)
                if suffix:
                    tensor=unswizzle_sf(tensor,m,hk//8)
                else:
                    tensor=tensor.reshape(m,hk)
                selected.append(tensor)
            result.append(torch.stack(selected).cuda())
    return *result, hashes


def prune_pairs(weight):
    """Keep two adjacent pairs with greatest squared magnitude per K8."""
    pairs=weight.reshape(*weight.shape[:-1],-1,4,2)
    score=pairs.square().sum(-1)
    # Stable ordering gives deterministic tie breaking; magnitude only.
    selected=score.argsort(dim=-1,descending=True,stable=True)[...,:2]
    keep=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,selected,True)
    return (pairs*keep.unsqueeze(-1)).reshape_as(weight)


def quantize32(values):
    """Nearest E2M1, ties to even, positive E4M3 scale per logical K32."""
    blocks=values.reshape(*values.shape[:-1],-1,32)
    scale=(blocks.abs().amax(-1)/6).clamp(max=448).to(torch.float8_e4m3fn)
    sf=scale.float()
    inverse=torch.where(sf>0,sf.reciprocal(),0.)
    normalized=(blocks*inverse.unsqueeze(-1)).flatten(-2)
    # Sorted positive representable values have midpoint ties at these points.
    boundaries=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=values.device)
    magnitude=normalized.abs().contiguous()
    code=torch.bucketize(magnitude,boundaries,right=False)
    # Even FP4 encoding wins a tie. Odd lower codes move to the upper code.
    ties=(code < 7) & (magnitude == boundaries[code.clamp(max=6)])
    code=code+((code%2==1)&ties)
    code=code.to(torch.uint8) | ((normalized<0).to(torch.uint8)<<3)
    packed=(code[...,0::2] | (code[...,1::2]<<4)).contiguous()
    return packed,scale.view(torch.uint8).contiguous()


def distortion(a,b):
    delta=a.float()-b.float()
    return dict(relative_l2=(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(b.float()).clamp_min(1e-20)).item(),
                relative_max=(delta.abs().max()/b.float().abs().max().clamp_min(1e-20)).item())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layer',type=int,default=3)
    ap.add_argument('--experts',type=int,nargs='+',default=[0,1,2,3,4,5,6,7])
    ap.add_argument('--tokens',type=int,default=6)
    ap.add_argument('--rounds',type=int,default=12)
    args=ap.parse_args()
    if not 1<=len(args.experts)<=8 or not 1<=args.tokens<=128 or not 3<=args.layer<=44:
        ap.error('bounded pilot: 1..8 experts, 1..128 tokens, layer 3..44')
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    library=Library(args.library)
    results=dict(scope='uncalibrated magnitude-only pruning and rescaling; synthetic Gaussian activations; no language-model quality measurement',
                 layer=args.layer, experts=args.experts, tokens_per_expert=args.tokens,
                 rank_name=args.rank.name, rank_size_bytes=args.rank.stat().st_size,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    for name in ('w13','w2'):
        raw,sf,hashes=read_experts(args.rank,args.layer,name,args.experts)
        w=dequant(raw,sf)
        pruned=prune_pairs(w)
        packed,scale=quantize32(pruned)
        reconstructed=dequant(packed,scale)
        rescale_packed,rescale_sf=quantize32(w)
        rescale_only=dequant(rescale_packed,rescale_sf)
        torch.manual_seed(271+int(name=='w2'))
        x=torch.randn(len(args.experts),args.tokens,w.shape[-1],device='cuda')*.5
        xp,xs=quantize32(x)
        xq=dequant(xp,xs)
        baseline=torch.bmm(xq,w.transpose(1,2))
        pruned_out=torch.bmm(xq,pruned.transpose(1,2))
        reencoded_out=torch.bmm(xq,reconstructed.transpose(1,2))
        rescale_out=torch.bmm(xq,rescale_only.transpose(1,2))
        projection=Projection(library,packed,xp,scale,xs)
        try:
            row=dict(projection=name, tensor_sha256=hashes,
                     original_zero_fraction=(w==0).float().mean().item(),
                     sparse_zero_fraction=(reconstructed==0).float().mean().item(),
                     weight_error=dict(prune_only=distortion(pruned,w),
                                       rescale_only=distortion(rescale_only,w),
                                       prune_and_rescale=distortion(reconstructed,w)),
                     projection_error=dict(prune_only=distortion(pruned_out,baseline),
                                           rescale_only=distortion(rescale_out,baseline),
                                           prune_and_rescale=distortion(reencoded_out,baseline)),
                     kernel_correctness=qualify(projection), bytes=projection.bytes,
                     timing=benchmark(projection,args.rounds))
            results['cases'].append(row)
            results['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('tensor_sha256','timing')}),flush=True)
        finally:
            projection.close()


if __name__ == '__main__':
    main()
