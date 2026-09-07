"""INT8 CPU recipe, explicit no-GPU compilation and owned-probe codec checks."""
import argparse
import hashlib
import json
from pathlib import Path


def geometry(rows):
    if rows < 128:
        raise ValueError('at least 128 rows required')
    local_rows=(rows+3)//4
    local_n=local_rows*4096
    payload_bytes=((local_n+4*(local_n//2048)+127)//128)*128
    return dict(rows=rows,padded_rows=local_rows*4,local_n=local_n,payload_bytes=payload_bytes)


def reference_packet(torch, tensor):
    """Independent tensor recipe; accepts CPU or CUDA, preserves all padding."""
    g=geometry(tensor.shape[0])
    padded=torch.zeros((g['padded_rows'],4096),dtype=torch.float32,device=tensor.device)
    padded[:tensor.shape[0]]=tensor.float()
    blocks=padded.reshape(-1,2048)
    scales=torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=1).clamp_min(1e-30)/127)))
    q=torch.round(blocks/scales[:,None]).clamp(-127,127).to(torch.int8)
    packet=torch.zeros((4,g['payload_bytes']),dtype=torch.uint8,device=tensor.device)
    packet[:,:g['local_n']]=q.view(torch.uint8).reshape(4,g['local_n'])
    count=g['local_n']//2048
    packet.view(torch.float32)[:,g['local_n']//4:g['local_n']//4+count]=scales.reshape(4,count)
    decoded=(q.float()*scales[:,None]).reshape(g['padded_rows'],4096)
    return packet.reshape(-1),decoded,g


def check_packet(torch,h,tensor,*,cpu_reference):
    before=tensor.clone()
    g=geometry(tensor.shape[0])
    actual=torch.full((4*g['payload_bytes'],),0xA5,dtype=torch.uint8,device=tensor.device)
    h._pack_rs_payload_int8[(g['padded_rows']*2,)](tensor,actual.view(torch.int8),actual.view(torch.float32),
        N=tensor.numel(),LOCAL_N=g['local_n'],PAYLOAD_BYTES=g['payload_bytes'],BLOCK=2048)
    expected,decoded,_=reference_packet(torch,tensor.cpu() if cpu_reference else tensor)
    compared=actual.cpu() if cpu_reference else actual
    bad=int((compared!=expected).sum())
    indices=(compared!=expected).nonzero().flatten()[:16]
    return dict(rows=tensor.shape[0],cpu_reference=cpu_reference,bad_bytes=bad,
        first_bad_indices=indices.cpu().tolist(),actual_bytes=compared[indices].cpu().tolist(),
        expected_bytes=expected[indices].cpu().tolist(),
        source_unchanged=torch.equal(tensor.view(torch.int16),before.view(torch.int16)),
        finite=bool(torch.isfinite(decoded).all())),decoded.to(tensor.device)


def codec_cases(torch,h,rank):
    results=[]
    for rows in (128,129,130,131,4095,4096,4097,8192):
        for case in ('zero','random','ties','extreme'):
            generator=torch.Generator(device='cuda').manual_seed(42019+rows+rank)
            x=torch.randn((rows,4096),generator=generator,device='cuda',dtype=torch.bfloat16)
            if case=='zero':x.zero_()
            if case=='ties':
                x.zero_()
                x[:,:8]=torch.tensor([-127.,-126.5,-1.5,-.5,.5,1.5,126.5,127.],device='cuda')
                x[:,2048:2052]=torch.tensor([-128.,-3.,3.,128.],device='cuda')
            if case=='extreme':
                x*=2**-20;x[:,0]=(-1 if rank%2 else 1)*2**60;x[:,2048]=2**-60
            report,_=check_packet(torch,h,x,cpu_reference=True)
            results.append(dict(report,case=case))
    return dict(rank=rank,cases=results)


def compile_kernels(h):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    compiled=[]
    for function,signature,constants in (
        (h._pack_rs_payload_int8,{'X':'*bf16','Packed':'*i8','Scales':'*fp32','N':'i32',
            'LOCAL_N':'i32','PAYLOAD_BYTES':'i32'},{'BLOCK':2048}),
        (h._unpack_sum_payload,{'Packed':'*i8','Scales':'*fp32','Out':'*bf16',
            'LOCAL_N':'i32','PAYLOAD_BYTES':'i32'},{'TP':4,'BLOCK':2048}),
    ):
        kernel=triton.compile(ASTSource(function,signature,constexprs=constants),
            target=GPUTarget('cuda',121,32),options={'num_warps':4})
        compiled.append(dict(kernel=function.__name__,hash=kernel.hash,shared_bytes=kernel.metadata.shared))
    return compiled


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    if not args.compile_only:ap.error('GPU codec checks require the owned TP4 runner')
    from vllm.distributed.device_communicators import glm53_prefill_collectives as h
    expected=Path(__file__).resolve().parents[1]/'build/glm53/glm53_prefill_collectives.py'
    digest=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
    assert digest==hashlib.sha256(expected.read_bytes()).hexdigest()
    print(json.dumps(dict(verdict='INT8_CPU_COMPILE_PASS',gpu_execution=False,source_sha256=digest,
        compiled=compile_kernels(h))),flush=True)
