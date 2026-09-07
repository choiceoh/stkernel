#!/usr/bin/env python3
"""Four-rank stock b12x MoE + M64 full-chunk tiles and TP transport correctness/timing.

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
    ap.add_argument('--rows',nargs='+',type=int,default=[4096,6143,6144,6912,8192])
    ap.add_argument('--check-api',action='store_true',help='CPU-only binding against the frozen distributed API')
    diagnostic=ap.add_mutually_exclusive_group()
    diagnostic.add_argument('--fp8-diagnostic',action='store_true',help='Fixed repeated FP8 controls; never a serving gate')
    diagnostic.add_argument('--fp8-trace',action='store_true',help='Actual partial/packet replay diagnostic; never a serving gate')
    diagnostic.add_argument('--int8-diagnostic',action='store_true',help='All-row INT8/FP8 comparison; never a serving gate')
    args=ap.parse_args()
    checked=validate_distributed_api()
    if args.check_api:
        print(json.dumps(dict(distributed_api='PASS',calls=checked)));return
    if (args.fp8_diagnostic or args.fp8_trace or args.int8_diagnostic) and args.transport != 'fp8-v3':
        ap.error('diagnostic requires FP8-v3')
    if int(os.environ.get('WORLD_SIZE','0'))!=4 or any(not 4096<=n<=8192 for n in args.rows):
        ap.error('four real ranks and 4096..8192 rows required')
    os.environ.update(VLLM_GLM53_PREFILL_SP='1',VLLM_GLM53_B12X_PREFILL_M64='1',
        VLLM_GLM53_PREFILL_SP_FP8='0' if args.transport=='bf16' else '3',
        VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS='4096',VLLM_GLM53_PREFILL_SP_RS_INT8='0',VLLM_DSV4_ONESHOT_AR='0',
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
        dist.all_gather_object(identities,(provenance,args.transport,args.rows,hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),group=group.cpu_group)
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
            def routes(self,x):
                # Routing depends on token content, never stripe row index.
                ids=((x[:,0].float().abs()*1024).int()[:,None]+torch.arange(8,device=x.device))% (8 if self.skew else 288)
                ids=ids.to(torch.int32)
                scales=torch.softmax(x[:,:8].float(),dim=1)
                return ids,scales
            def __call__(self,x):
                ids,scales=self.routes(x)
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
        require(wrapper._dynamic_workspace.tile_m==128 and wrapper._prefill_m64_workspace.tile_m==64, 'both workspace geometries must be active')
        candidate_workspace=wrapper._prefill_m64_workspace
        results=[]
        all_pass=True
        def reports(value):
            values=[None]*4
            dist.all_gather_object(values,value,group=group.cpu_group)
            return values
        def compare(a,b,repeat):
            a,b,r=(v.float() for v in (a,b,repeat))
            finite=all(bool(torch.isfinite(v).all()) for v in (a,b,r))
            norm=b.norm(dim=1).clamp_min(1e-6);peak=b.abs().amax(dim=1).clamp_min(1e-6)
            error=(a-b).norm(dim=1)/norm;noise=(r-b).norm(dim=1)/norm
            worst=(a-b).abs().amax(dim=1)/peak;npeak=(r-b).abs().amax(dim=1)/peak
            bad=(error>torch.maximum(3*noise,torch.full_like(noise,.02))) | (worst>torch.maximum(3*npeak,torch.full_like(npeak,.04)))
            report=dict(bad_rows=int(bad.sum()),max_row_relative_l2=float(error.max()),max_row_relative_abs=float(worst.max()),repeat_l2=float(noise.max()))
            report['finite']=finite
            report['pass']=finite and not report['bad_rows']
            return reports(report)
        if args.fp8_diagnostic or args.fp8_trace or args.int8_diagnostic:
            from glm53_moe_m64_fp8_diagnostic import run as run_diagnostic
            def case_factory(rows, skew, seed):
                mlp.skew=skew
                generator=torch.Generator(device='cuda').manual_seed(seed)
                full=torch.randn(rows,4096,generator=generator,device='cuda',dtype=torch.bfloat16)*.5
                shard=h.prefill_shard(full);original=shard.clone()
                gathered=h.prefill_all_gather(shard,num_tokens=rows)
                ids,scales=mlp.routes(gathered)
                def select(candidate):
                    wrapper._prefill_m64_workspace=candidate_workspace if candidate else None
                    expected=candidate_workspace if candidate and rows>=6144 else wrapper._dynamic_workspace
                    if wrapper._workspace_for_prefill(wrapper._dynamic_workspace,rows) is not expected:
                        raise AssertionError('diagnostic selected wrong workspace')
                def call(candidate):
                    select(candidate)
                    with override_forward_context(context):
                        x=h.prefill_all_gather(shard,num_tokens=rows)
                        if args.fp8_trace or args.int8_diagnostic:
                            gather_unchanged=torch.equal(x.view(torch.int16),gathered.view(torch.int16))
                        with h.partial_tp_output(num_tokens=rows):x=mlp(x)
                        if args.fp8_trace or args.int8_diagnostic:
                            partial=x.clone()
                            output=h.prefill_reduce_scatter(x)
                            return dict(partial=partial,output=output,gather_unchanged=gather_unchanged,
                                        source_unchanged=torch.equal(x.view(torch.int16),partial.view(torch.int16)))
                        return h.prefill_reduce_scatter(x)
                def local_call(candidate):
                    select(candidate)
                    return wrapper.run(gathered,w13,sf13,w2,sf2,ids,scales,
                        w1_alpha=ones,w2_alpha=ones,fc2_input_scale=ones,out=torch.empty_like(gathered))
                return call,local_call,lambda:torch.equal(shard,original)
            if args.int8_diagnostic:
                from glm53_moe_m64_int8_diagnostic import run as run_int8
                run_int8(torch=torch,h=h,rank=rank,provenance=provenance,reports=reports,
                         require=require,case_factory=case_factory)
            elif args.fp8_trace:
                from glm53_moe_m64_fp8_trace import run as run_trace
                run_trace(torch=torch,h=h,rank=rank,provenance=provenance,reports=reports,
                          require=require,case_factory=case_factory)
            else:
                run_diagnostic(torch=torch,rank=rank,provenance=provenance,reports=reports,
                               require=require,case_factory=case_factory)
            return
        for rows in args.rows:
            for skew in (False,True):
                mlp.skew=skew
                generator=torch.Generator(device='cuda').manual_seed(9211+rows)
                full=torch.randn(rows,4096,generator=generator,device='cuda',dtype=torch.bfloat16)*.5
                shard=h.prefill_shard(full);original=shard.clone()
                def call(candidate):
                    with override_forward_context(context):
                        wrapper._prefill_m64_workspace=candidate_workspace if candidate else None
                        selected=wrapper._workspace_for_prefill(wrapper._dynamic_workspace,rows)
                        expected=candidate_workspace if candidate and rows>=6144 else wrapper._dynamic_workspace
                        if selected is not expected:raise AssertionError('actual call selected wrong workspace')
                        x=h.prefill_all_gather(shard,num_tokens=rows)
                        with h.partial_tp_output(num_tokens=rows):x=mlp(x)
                        return h.prefill_reduce_scatter(x)
                b=call(False);b2=call(False);control=call(False);a=call(True);torch.cuda.synchronize()
                valid_rows=min(shard.shape[0],max(0,rows-rank*shard.shape[0]))
                result=dict(rows=rows,skew=skew,m64_admitted=rows>=6144,eager=compare(a[:valid_rows],b[:valid_rows],b2[:valid_rows]),stock_control=compare(control[:valid_rows],b[:valid_rows],b2[:valid_rows]))
                require(torch.equal(shard,original),'input was modified')
                # Keep prior output alive, change activations/routes, and churn
                # allocator storage to expose missing cross-stream ownership.
                retained=a.clone();shard.mul_(-.75)
                trash=[torch.empty_like(full) for _ in range(3)];del trash
                b=call(False);b2=call(False);control=call(False)
                result['changed_control']=compare(control[:valid_rows],b[:valid_rows],b2[:valid_rows])
                result['changed']=[]
                for _ in range(3):
                    a2=call(True);torch.cuda.synchronize()
                    result['changed'].append(compare(a2[:valid_rows],b[:valid_rows],b2[:valid_rows]))
                require(torch.equal(a,retained),'earlier output overwritten by later MoE call')
                # Compare local MoE before TP quantization as well. This
                # keeps transport amplification distinguishable from the tile.
                gathered=h.prefill_all_gather(shard,num_tokens=rows)
                ids,scales=mlp.routes(gathered)
                def local_call(candidate, output=None):
                    wrapper._prefill_m64_workspace=candidate_workspace if candidate else None
                    if output is None:output=torch.empty_like(gathered)
                    return wrapper.run(gathered,w13,sf13,w2,sf2,ids,scales,
                        w1_alpha=ones,w2_alpha=ones,fc2_input_scale=ones,out=output)
                lb=local_call(False);lb2=local_call(False);lc=local_call(False);la=local_call(True)
                result['local_moe']=compare(la,lb,lb2)
                result['local_control']=compare(lc,lb,lb2)
                if rows==8192:
                    # Capture must choose M128 even with the candidate enabled.
                    # Replay changes both inputs and expert assignments in place.
                    gout=torch.empty_like(gathered);stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        local_call(False,gout);local_call(False,gout)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    wrapper._prefill_m64_workspace=candidate_workspace
                    with torch.cuda.graph(graph,stream=stream):
                        if wrapper._workspace_for_prefill(wrapper._dynamic_workspace,rows) is not wrapper._dynamic_workspace:
                            raise AssertionError('M64 was admitted during capture')
                        local_call(True,gout)
                    graph.replay();torch.cuda.synchronize()
                    result['graph']=compare(gout,lb,lb2)
                    saved=gout.clone();gathered.mul_(-.5)
                    new_ids,new_scales=mlp.routes(gathered);ids.copy_(new_ids);scales.copy_(new_scales)
                    gb=local_call(False);gb2=local_call(False)
                    graph.replay();torch.cuda.synchronize()
                    result['graph_changed']=compare(gout,gb,gb2)
                    require(not torch.equal(gout,saved),'changed-input graph did not change output')
                    del graph
                timing=[[],[]]
                for iteration in range(6):
                    for arm in ((0,1) if iteration%2==0 else (1,0)):
                        dist.barrier(group=group.device_group)
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();value=call(bool(arm));end.record();end.synchronize()
                        elapsed=torch.tensor(start.elapsed_time(end),device='cuda')
                        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX,group=group.device_group)
                        timing[arm].append(float(elapsed))
                result['slowest_rank_ms']=dict(baseline=timing[0],m64=timing[1])
                result['median_speedup_pct']=100*(statistics.median(timing[0])/statistics.median(timing[1])-1)
                checks=result['eager']+result['stock_control']+result['changed_control']+result['local_moe']+result['local_control']+result.get('graph',[])+result.get('graph_changed',[])+[r for rep in result['changed'] for r in rep]
                result['pass']=all(r['pass'] for r in checks)
                all_pass=all_pass and result['pass']
                results.append(result)
                if rank==0:print(json.dumps(result),flush=True)
        if rank==0:print(json.dumps(dict(verdict='MOE_M64_GPU_PASS' if all_pass else 'MOE_M64_GPU_FAIL',transport=args.transport,provenance=provenance,results=results)),flush=True)

        require(all_pass,'one or more numerical or stock control checks failed')

if __name__=='__main__':main()
