#!/usr/bin/env python3
"""Four-rank stock b12x MoE + overlapped TP transport correctness/timing.

This is an isolated kernel/transport gate, not serving TTFT or text quality.
Use the owned offline launcher; a matched full-model bracket follows only
if every row, changed input and repeated-use case passes on every rank.
"""
import argparse
import ast
from contextlib import contextmanager
import inspect
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace


@contextmanager
def distributed_probe(ps, *, rank, local_rank):
    """Cover partial initialization as well as the measurement body."""
    ps.set_custom_all_reduce(False)
    try:
        ps.init_distributed_environment(world_size=4,rank=rank,local_rank=local_rank,
            distributed_init_method='env://',backend='nccl',timeout=timedelta(minutes=5))
        ps.initialize_model_parallel(tensor_model_parallel_size=4,pipeline_model_parallel_size=1)
        yield
    finally:
        try:
            ps.destroy_model_parallel()
        finally:
            ps.destroy_distributed_environment()


def validate_distributed_api():
    """Check probe calls against the mounted source without importing CUDA."""
    source=Path(__file__).resolve().parents[1]/'overlay/modules/glm53_runtime/parallel_state.py'
    definitions={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)}
    helper=next(n for n in ast.parse(Path(__file__).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='distributed_probe')
    checked=[]
    for call in ast.walk(helper):
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute):continue
        if not isinstance(call.func.value,ast.Name) or call.func.value.id!='ps':continue
        definition=definitions[call.func.attr]
        args=definition.args;parameters=[]
        positional=args.posonlyargs+args.args;required=len(positional)-len(args.defaults)
        for i,arg in enumerate(positional):
            kind=inspect.Parameter.POSITIONAL_ONLY if i<len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
            parameters.append(inspect.Parameter(arg.arg,kind,default=inspect.Parameter.empty if i<required else None))
        if args.vararg:parameters.append(inspect.Parameter(args.vararg.arg,inspect.Parameter.VAR_POSITIONAL))
        for arg,default in zip(args.kwonlyargs,args.kw_defaults):
            parameters.append(inspect.Parameter(arg.arg,inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None))
        if args.kwarg:parameters.append(inspect.Parameter(args.kwarg.arg,inspect.Parameter.VAR_KEYWORD))
        inspect.Signature(parameters).bind(*[None for _ in call.args],**{k.arg:None for k in call.keywords})
        checked.append(call.func.attr)
    if set(checked)!={'set_custom_all_reduce','init_distributed_environment','initialize_model_parallel',
                     'destroy_model_parallel','destroy_distributed_environment'}:
        raise RuntimeError('distributed lifecycle call coverage changed')
    return checked


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transport',choices=('bf16','fp8-v3'),required=True)
    ap.add_argument('--rows',nargs='+',type=int,default=[4096,4143,6912,8192])
    ap.add_argument('--check-api',action='store_true',help='CPU-only binding against the frozen distributed API')
    args=ap.parse_args()
    checked=validate_distributed_api()
    if args.check_api:
        print(json.dumps(dict(distributed_api='PASS',calls=checked)));return
    if int(os.environ.get('WORLD_SIZE','0'))!=4 or any(not 4096<=n<=8192 for n in args.rows):
        ap.error('four real ranks and 4096..8192 rows required')
    os.environ.update(VLLM_GLM53_PREFILL_SP='1',VLLM_GLM53_PREFILL_MOE_OVERLAP='1',
        VLLM_GLM53_PREFILL_SP_FP8='0' if args.transport=='bf16' else '3',
        VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS='4096',VLLM_DSV4_ONESHOT_AR='0',
        VLLM_GLM53_B12X_STATIC_V2='t',VLLM_GLM53_B12X_PREFILL_REUSE='0',
        VLLM_GLM53_B12X_PREFILL_FC1_N128='0')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from vllm.config import VllmConfig,ParallelConfig,set_current_vllm_config
    from vllm.distributed import get_tp_group,tensor_model_parallel_all_reduce
    from vllm.distributed import parallel_state as ps
    from vllm.forward_context import override_forward_context,get_forward_context
    from vllm.distributed.device_communicators import glm53_prefill_collectives as h
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from b12x_static_probe import expert_set

    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    assert torch.cuda.get_device_capability()==(12,1)
    config=VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4,
        pipeline_parallel_size=1,distributed_executor_backend='external_launcher'))
    with set_current_vllm_config(config), distributed_probe(ps,rank=rank,local_rank=local_rank):
        group=get_tp_group()
        def require(ok,message):
            reports=[None]*4
            dist.all_gather_object(reports,None if bool(ok) else message,group=group.cpu_group)
            if any(r is not None for r in reports):raise AssertionError(reports)
        # Every imported overlay and the probe must match the frozen checkout.
        manifest=Path('/repo/build/glm53/manifest.tsv')
        provenance={}
        for line in manifest.read_text().splitlines():
            if not line or line.startswith('#'):continue
            name,target,*_=line.split('\t')
            actual=hashlib.sha256(Path(target).read_bytes()).hexdigest()
            expected=hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()
            provenance[name]=actual
            require(actual==expected,'source mismatch: '+name)
        identities=[None]*4
        dist.all_gather_object(identities,(provenance,args.transport,args.rows),group=group.cpu_group)
        require(all(x==identities[0] for x in identities),'rank source/plan mismatch')
        md._STATIC_V2_OVERRIDE=md._parse_glm53_static_v2('t',probe=True)
        torch.manual_seed(73209+rank)
        weights=expert_set(torch.Generator().manual_seed(73209+rank))
        w13,sf13,w2,sf2=weights;md.tile_expert_weights_inplace(w13,w2)
        wrapper=B12xMoEWrapper(num_experts=288,top_k=8,hidden_size=4096,
            intermediate_size=512,use_cuda_graph=True,max_num_tokens=8192,num_local_experts=288,
            activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
        ones=torch.ones(288,device='cuda')
        shared_stream=torch.cuda.Stream()
        sw1=torch.randn(512,4096,device='cuda',dtype=torch.bfloat16)*.01
        sw2=torch.randn(4096,256,device='cuda',dtype=torch.bfloat16)*.01
        class MLP:
            experts=SimpleNamespace(layer_name='probe.moe')
            skew=False
            def __call__(self,x):
                # Routing depends on token content, never stripe row index.
                ids=((x[:,0].float().abs()*1024).int()[:,None]+torch.arange(8,device=x.device))% (8 if self.skew else 288)
                ids=ids.to(torch.int32)
                scales=torch.softmax(x[:,:8].float(),dim=1)
                stream=torch.cuda.current_stream();shared_stream.wait_stream(stream)
                x.record_stream(shared_stream)
                with torch.cuda.stream(shared_stream):
                    gu=F.linear(x,sw1);shared=F.linear(F.silu(gu[:,:256])*gu[:,256:],sw2)
                output=torch.empty_like(x)
                wrapper.run(x,w13,sf13,w2,sf2,ids,scales,
                    w1_alpha=ones,w2_alpha=ones,fc2_input_scale=ones,out=output)
                stream.wait_stream(shared_stream);shared.record_stream(stream)
                return tensor_model_parallel_all_reduce(output+shared)
        mlp=MLP()
        @dataclass
        class Context:
            no_compile_layers:dict
            all_moe_layers:object=None
            moe_layer_index:int=0
            dp_metadata:object=None
            ubatch_slices:object=None
        context=Context({'probe.moe':mlp.experts})
        results=[]
        def compare(a,b,repeat):
            a,b,r=(v.float() for v in (a,b,repeat))
            finite=all(bool(torch.isfinite(v).all()) for v in (a,b,r))
            norm=b.norm(dim=1).clamp_min(1e-6);peak=b.abs().amax(dim=1).clamp_min(1e-6)
            error=(a-b).norm(dim=1)/norm;noise=(r-b).norm(dim=1)/norm
            worst=(a-b).abs().amax(dim=1)/peak;npeak=(r-b).abs().amax(dim=1)/peak
            bad=(error>torch.maximum(3*noise,torch.full_like(noise,.02))) | (worst>torch.maximum(3*npeak,torch.full_like(npeak,.04)))
            report=dict(bad_rows=int(bad.sum()),max_row_relative_l2=float(error.max()),max_row_relative_abs=float(worst.max()),repeat_l2=float(noise.max()))
            require(finite and not report['bad_rows'],str(report))
            return report
        for rows in args.rows:
            for skew in (False,True):
                mlp.skew=skew
                generator=torch.Generator(device='cuda').manual_seed(9211+rows)
                full=torch.randn(rows,4096,generator=generator,device='cuda',dtype=torch.bfloat16)*.5
                shard=h.prefill_shard(full);original=shard.clone()
                def call(candidate):
                    with override_forward_context(context):
                        if candidate:
                            overlapped=h.prefill_moe_overlap(mlp,shard,num_tokens=rows)
                            if overlapped is not None:return overlapped
                        x=h.prefill_all_gather(shard,num_tokens=rows)
                        with h.partial_tp_output(num_tokens=rows):x=mlp(x)
                        return h.prefill_reduce_scatter(x)
                b=call(False);b2=call(False);a=call(True);torch.cuda.synchronize()
                valid_rows=min(shard.shape[0],max(0,rows-rank*shard.shape[0]))
                result=dict(rows=rows,skew=skew,overlap_admitted=rows>=6144,eager=compare(a[:valid_rows],b[:valid_rows],b2[:valid_rows]))
                require(torch.equal(shard,original),'input was modified')
                # Keep prior output alive, change activations/routes, and churn
                # allocator storage to expose missing cross-stream ownership.
                retained=a.clone();shard.mul_(-.75)
                trash=[torch.empty_like(full) for _ in range(3)];del trash
                b=call(False);b2=call(False)
                for _ in range(3):
                    a2=call(True);torch.cuda.synchronize()
                    result['changed']=compare(a2[:valid_rows],b[:valid_rows],b2[:valid_rows])
                require(torch.equal(a,retained),'earlier output overwritten by later MoE call')
                timing=[[],[]]
                for iteration in range(6):
                    for arm in ((0,1) if iteration%2==0 else (1,0)):
                        dist.barrier(group=group.device_group)
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();value=call(bool(arm));end.record();end.synchronize()
                        elapsed=torch.tensor(start.elapsed_time(end),device='cuda')
                        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX,group=group.device_group)
                        timing[arm].append(float(elapsed))
                result['slowest_rank_ms']=dict(baseline=timing[0],overlap=timing[1])
                result['median_speedup_pct']=100*(statistics.median(timing[0])/statistics.median(timing[1])-1)
                results.append(result)
                if rank==0:print(json.dumps(result),flush=True)
        if rank==0:print(json.dumps(dict(verdict='MOE_OVERLAP_GPU_PASS',transport=args.transport,provenance=provenance,results=results)),flush=True)

if __name__=='__main__':main()
