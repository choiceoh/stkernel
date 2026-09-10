#!/usr/bin/env python3
"""Independent CPU oracle for original-output packed E2M1/E4M3 KV storage.

The pinned official Python quantizer and Attention AST execute with explicit
CPU kernel/projection fixtures. TileLang's hardware FP4 packing/conversion,
Triton device numerics, distributed execution and speed remain unvalidated.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import struct
import sys
import weakref

ROOT=Path(__file__).resolve().parents[1]
MODEL_SHA="4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_SHA="1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
LEVELS=(0.,.5,1.,1.5,2.,3.,4.,6.)
OWNERS=(2,8,14,20)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def f32(value):
    try:return struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError:return math.copysign(math.inf,value)


def bf16_bits(value):
    bits=struct.unpack('<I',struct.pack('<f',f32(value)))[0]
    if bits&0x7f800000==0x7f800000:
        return 0x7fc0 if bits&0x7fffff else bits>>16
    return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff


def e4m3_value(byte):
    """Arithmetic E4M3FN decoder independent of the candidate bit decoder."""
    sign=-1. if byte&128 else 1.
    exponent,mantissa=(byte>>3)&15,byte&7
    if exponent==15 and mantissa==7:return math.nan
    magnitude=math.ldexp(mantissa,-9) if exponent==0 else math.ldexp(8+mantissa,exponent-10)
    return math.copysign(magnitude,sign)


E4M3_POSITIVE=tuple(e4m3_value(i) for i in range(127))


def e4m3_rne(value):
    """CPU IEEE nearest-even conversion, including the finite->NaN boundary.

    This models nonsaturating E4M3FN casting. Whether a particular TileLang
    hardware conversion saturates or how it packs a NaN FP4 intermediate is
    deliberately not claimed by this CPU substitute.
    """
    value=f32(value)
    sign=128 if math.copysign(1.,value)<0 else 0
    magnitude=abs(value)
    if not math.isfinite(magnitude) or magnitude>464:return sign|127
    if magnitude>=448:return sign|126  # 464 is the tie to the even finite code.
    upper=bisect.bisect_left(E4M3_POSITIVE,magnitude)
    if upper==0:return sign
    lower=upper-1
    low_distance=magnitude-E4M3_POSITIVE[lower]
    high_distance=E4M3_POSITIVE[upper]-magnitude
    if low_distance<high_distance or (low_distance==high_distance and lower%2==0):return sign|lower
    return sign|upper


def scalar_decode_bits(code,scale):
    value=LEVELS[code&7]
    if code&8:value=-value
    return bf16_bits(value*e4m3_value(scale))


def arithmetic_unpack(packed,scales):
    import torch
    codes=torch.stack((packed&15,packed>>4),dim=-1).flatten(-2)
    values=torch.tensor(LEVELS,dtype=torch.float32)[(codes&7).long()]
    values=torch.where((codes&8)!=0,-values,values)
    scale_values=torch.tensor([e4m3_value(i) for i in range(256)],dtype=torch.float32)
    scale=scale_values[scales.long()].repeat_interleave(16,dim=-1)
    return (values*scale).to(torch.bfloat16)


def oracle_quantize(x, *, overflow="nan"):
    """Official FP32 /6 scale expression; independent E4M3 and E2M1 RNE.

    NaN scale makes every reconstructed element NaN. Its FP4 packed nibble is
    set to zero by this CPU fixture, because GPU NaN-to-FP4 bits are unproven.
    Finite BF16 producer inputs can reach this case by E4M3 scale overflow.
    """
    import torch
    if x.dtype!=torch.bfloat16 or x.shape[-1]%16:
        raise ValueError('CPU quantizer requires BF16 blocks16')
    if not bool(x.isfinite().all()):
        raise ValueError('nonfinite producer input is outside this CPU fixture')
    groups=x.float().unflatten(-1,(-1,16))
    amax=groups.abs().amax(-1).clamp_min(6*2.**-9)
    scale_fp32=amax/6.
    raw=torch.tensor([e4m3_rne(v) for v in scale_fp32.flatten().tolist()],dtype=torch.uint8).reshape(scale_fp32.shape)
    if overflow not in ("nan","satfinite"):raise ValueError("explicit CPU overflow policy required")
    if overflow=="satfinite":raw=raw.masked_fill(raw==127,126)
    table=torch.tensor([e4m3_value(i) for i in range(256)],dtype=torch.float32)
    scales=table[raw.long()]
    normalized=(groups/scales[...,None]).clamp(-6,6)
    magnitude=normalized.abs()
    boundaries=(.25,.75,1.25,1.75,2.5,3.5,5.)
    codes=torch.bucketize(magnitude,torch.tensor(boundaries,dtype=torch.float32),right=False)
    for i,midpoint in enumerate(boundaries):
        if i&1:codes=codes+(magnitude==midpoint).to(codes.dtype)
    codes=codes.to(torch.uint8)|(normalized.signbit().to(torch.uint8)<<3)
    codes=codes.masked_fill(normalized.isnan(),0).flatten(-2)
    packed=codes[...,::2]|(codes[...,1::2]<<4)
    return packed,raw,arithmetic_unpack(packed,raw)


def install_quantizer_fixture(kernel_namespace, *, overflow="nan"):
    """Replace TileLang dispatch only, retaining the actual pinned wrapper."""
    original=kernel_namespace['fp4_quant_kernel']
    def factory(n,block_size,*,scale_dtype,inplace):
        if block_size!=16:
            return original(n,block_size,scale_dtype=scale_dtype,inplace=inplace)
        assert n%16==0 and scale_dtype==kernel_namespace['FP8']
        def run(x,y,sf):
            import torch
            packed,scales,dequantized=oracle_quantize(x,overflow=overflow)
            if inplace:y.copy_(dequantized)
            else:y.view(torch.uint8).copy_(packed)
            sf.view(torch.uint8).copy_(scales)
        return run
    kernel_namespace['fp4_quant_kernel']=factory


def reference_quantizer(path,prior):
    module=prior.reference_kernel(path)
    assert sha(path)==KERNEL_SHA
    install_quantizer_fixture(vars(module))
    return module


def format_checks(core,kernel,dual):
    import torch
    codes=torch.arange(16,dtype=torch.uint8).repeat(256)
    scales=torch.arange(256,dtype=torch.int32).repeat_interleave(16).to(torch.uint8)
    packed=(codes|(codes<<4))[:,None].expand(-1,256).contiguous()
    sf=scales[:,None].expand(-1,32).contiguous()
    bits=torch.tensor([scalar_decode_bits(int(c),int(s)) for c,s in zip(codes,scales)],dtype=torch.int32).to(torch.int16)
    expected=bits[:,None].expand(-1,512).contiguous().view(torch.bfloat16)
    actual=torch.cat([core.unpack_kv_tile(packed[lo:lo+128],sf[lo:lo+128]) for lo in range(0,4096,128)])
    dual.equal_output(actual,expected,'all16 codes x all256 scale bytes')
    raw=torch.arange(256,dtype=torch.int32).to(torch.uint8)[:,None].expand(-1,256).contiguous()
    rawsf=torch.full((256,32),56,dtype=torch.uint8)
    dual.equal_output(core.unpack_kv_tile(raw,rawsf),arithmetic_unpack(raw,rawsf),'distinct packed bytes')
    # Verify the independent conversion table against CPU's E4M3FN cast for
    # every finite BF16 amax after the ORIGINAL max floor and FP32 division.
    bf16=torch.arange(0x7f80,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    scale=(bf16.float().clamp_min(6*2.**-9)/6.)
    expected_sf=torch.tensor([e4m3_rne(v) for v in scale.tolist()],dtype=torch.uint8)
    actual_sf=scale.to(torch.float8_e4m3fn).view(torch.uint8)
    finite_scale=scale<=464
    assert torch.equal(expected_sf[finite_scale],actual_sf[finite_scale]),'nonoverflow E4M3 RNE differs'
    overflow_counts={str(int(code)):int(count) for code,count in zip(*actual_sf[~finite_scale].unique(return_counts=True))}
    fixtures=[0.,-0.,2.**-133,-2.**-133,2.**-10,-2.**-10,6*2.**-9,
              .25,.75,1.25,1.75,2.5,3.5,5.,6.,2688.,2776.,2784.,2800.,
              3.3895313892515355e38,-3.3895313892515355e38]
    x=torch.tensor(fixtures,dtype=torch.bfloat16)[:,None].expand(-1,512).contiguous()
    # Mixed groups exercise /scale RNE ties; slot15 fixes scale exactly1.
    ties=torch.tensor([0.,-0.,.25,-.25,.75,-.75,1.25,-1.25,1.75,-1.75,2.5,-2.5,3.5,5.,-5.,6.],dtype=torch.bfloat16)
    x=torch.cat((x,ties.repeat(32)[None]),dim=0)
    y,s=kernel.fp4_act_quant(x,16,False,scale_dtype=torch.float8_e4m3fn)
    inplace=kernel.fp4_act_quant(x.clone(),16,True,scale_dtype=torch.float8_e4m3fn)
    dual.equal_output(core.unpack_kv_tile(y.view(torch.uint8),s.view(torch.uint8)),inplace,'original wrapper packed/inplace pair')
    oracle_y,oracle_s,oracle_x=oracle_quantize(x)
    assert torch.equal(y.view(torch.uint8),oracle_y) and torch.equal(s.view(torch.uint8),oracle_s)
    dual.equal_output(inplace,oracle_x,'quantizer fixture arithmetic')
    assert bool(inplace[-3:-1].isnan().all()),'finite producer overflow fixture must expose NaN scale'
    assert inplace[1,0].view(torch.int16).item()==-32768
    # Non-saturating FP8 RNE halfway boundary is declared and directly checked.
    assert e4m3_rne(464.)==126 and e4m3_rne(464.01)==127
    # Check BOTH plausible producer-overflow contracts without pretending that
    # the CPU cast establishes the pinned TileLang conversion policy.
    old_factory=kernel.fp4_quant_kernel
    try:
        install_quantizer_fixture(vars(kernel),overflow="satfinite")
        sat_y,sat_s=kernel.fp4_act_quant(x,16,False,scale_dtype=torch.float8_e4m3fn)
        sat_inplace=kernel.fp4_act_quant(x.clone(),16,True,scale_dtype=torch.float8_e4m3fn)
        dual.equal_output(core.unpack_kv_tile(sat_y.view(torch.uint8),sat_s.view(torch.uint8)),sat_inplace,'satfinite CPU fixture paired outputs')
        dual.equal_output(sat_inplace,oracle_quantize(x,overflow="satfinite")[2],'satfinite independent arithmetic')
        assert bool(sat_inplace.isfinite().all())
    finally:kernel.fp4_quant_kernel=old_factory
    return dict(nibble_scale_pairs=4096,packed_byte_patterns=256,finite_bf16_amax_values=len(scale),
        producer_rows=len(x),positive_zero_and_negative_zero_exact=True,nan_positions_exact=True,
        cpu_producer_policies=['nonsaturating overflow to NaN','satfinite overflow to 448'],
        cpu_cast_nonoverflow_values=int(finite_scale.sum()),cpu_cast_overflow_raw_counts=overflow_counts,
        nan_payload_exact=False,gpu_producer_bytes_validated=False,
        nan_fp4_nibble_fixture=0,scale_cast='Two explicit CPU E4M3FN policies; hardware conversion unvalidated')


def score_checks(core,kernel,dual):
    import torch
    rows=[]
    cases=[('h8_partial',1,1,8,127,7,63),('h16_boundary',1,2,16,128,9,64),
           ('h64_tail',1,1,64,129,11,65),('strided_batch2',2,3,16,128,9,129),
           ('empty_prefix',1,1,16,128,0,65),('ties',1,1,16,128,9,65)]
    for case,(name,b,q,h,w,c,k) in enumerate(cases):
        rng=torch.Generator().manual_seed(541000+case)
        query=(torch.randn(b,q,h,512,generator=rng,dtype=torch.float32)*.2).to(torch.bfloat16)
        window=torch.randn(b,w*2,1024,generator=rng,dtype=torch.float32).to(torch.bfloat16)[:,::2,::2]
        raw=(torch.randn(b,max(c,1),512,generator=rng,dtype=torch.float32)*3).to(torch.bfloat16)
        y,s=kernel.fp4_act_quant(raw,16,False,scale_dtype=torch.float8_e4m3fn)
        dequantized=kernel.fp4_act_quant(raw.clone(),16,True,scale_dtype=torch.float8_e4m3fn)[:,:c]
        cache=core.PackedKVCache(b,max(c+5,8),'cpu',20,1)
        core.write_packed_kv(cache,y,s,start_slot=0,rows=c)
        ids=torch.randint(w+c,(b,q,k),generator=rng,dtype=torch.int32); ids[...,::7]=-1
        if c:ids[...,1:4]=w+c-1
        if name=='ties':query.zero_()
        sink=torch.linspace(-2,2,h,dtype=torch.float32)
        combined=torch.cat((window,dequantized),1)
        # Inspect the actual selected tile path: width is an active prefix,
        # not allocation capacity, and invalid slots must not expose tail bytes.
        for batch_id in range(b):
            for row in range(q):
                for lo in range(0,k,64):
                    selected=torch.full((1,64),-1,dtype=torch.int64)
                    count=min(64,k-lo)
                    selected[:,:count]=ids[batch_id,row,lo:lo+count]
                    gathered,valid=core._gather_tile(window,cache,c,torch.tensor([batch_id]),selected)
                    expected_tile=dual.gather_concat(combined[batch_id:batch_id+1],selected[:,None].to(torch.int32))[:,0]
                    dual.equal_output(gathered,expected_tile,'actual selected packed gather')
                    assert torch.equal(valid,selected!=-1)
        expected=dual.online64(query,combined,sink,ids,512**-.5)
        output=core.packed_sparse_attn(query,window,cache,sink,ids,512**-.5,width=c)
        dual.equal_output(output,expected,name)
        assert output.is_contiguous()
        rows.append(dict(name=name,batch=b,queries=q,heads=h,window=w,compressed=c,slots=k,
                         output_sha256=dual.tensor_sha(output),cpu_bf16_bits_exact=True,actual_gather_bytes_exact=True))
        # Update existing storage through the official-byte writer; metadata
        # stays stable and the next call must decode fresh content.
        if c:
            ptrs=(cache.packed.data_ptr(),cache.scales.data_ptr())
            changed=raw.neg().add(1)
            y2,s2=kernel.fp4_act_quant(changed,16,False,scale_dtype=torch.float8_e4m3fn)
            core.write_packed_kv(cache,y2,s2,start_slot=0,rows=c)
            key2=kernel.fp4_act_quant(changed.clone(),16,True,scale_dtype=torch.float8_e4m3fn)[:,:c]
            expected2=dual.online64(query,torch.cat((window,key2),1),sink,ids,512**-.5)
            actual2=core.packed_sparse_attn(query,window,cache,sink,ids,512**-.5,width=c)
            dual.equal_output(actual2,expected2,'same address new packed bytes')
            assert ptrs==(cache.packed.data_ptr(),cache.scales.data_ptr())
            assert not torch.equal(actual2,output)
    return rows


def adapter_checks(core,adapter,packed_index_adapter,reference_dir,prior,dual):
    import torch
    rows=[]; released_bf16=0; released_packed=0
    # B2 is a separate fresh lifecycle. No checkpoint weights are loaded.
    for batch in (1,2):
        kernel=dual.reference_kernel(reference_dir/'kernel.py')
        ref=dual.full_reference(reference_dir/'model.py',kernel,prior,4)
        opt=dual.full_reference(reference_dir/'model.py',kernel,prior,4)
        install_quantizer_fixture(ref.fp4_act_quant.__globals__)
        install_quantizer_fixture(opt.fp4_act_quant.__globals__)
        capacity=144
        ref_model=dual.attention_fixture(ref,prior,4,capacity)
        opt_model=dual.attention_fixture(opt,prior,4,capacity)
        for module,model in ((ref,ref_model),(opt,opt_model)):
            for layer in range(2,40):
                obj=model.layers[layer].attn
                obj.window_kv_cache=obj.window_kv_cache.repeat(batch,1,1)
                if layer in OWNERS:
                    obj.compress_kv_cache=obj.compress_kv_cache.repeat(batch,1,1)
                    obj.indexer.k_cache=obj.indexer.k_cache.repeat(batch,1,1)
            def reduce(value):
                base=value.clone()
                for _ in range(3):value.add_(base)
            module.dist.all_reduce=reduce
        old=[weakref.ref(opt_model.layers[layer].attn.compress_kv_cache) for layer in OWNERS]
        # A live shared pointer must not keep the displaced BF16 allocation.
        opt.shared_attn.compress_kv=opt_model.layers[20].attn.compress_kv_cache
        index_handle=packed_index_adapter.install_reference_packed_indexer(opt_model,opt,enabled=True)
        handle=adapter.install_reference_packed_kv_attention(opt_model,opt,enabled=True)
        gc.collect()
        assert all(value() is None for value in old),'old BF16 compressed owner was retained'
        released_bf16+=len(old)
        try:
            steps=[('prefill_ring_seed',0,129),('even_end_decode',129,1),('odd_end_no_latent',130,1)] if batch==1 else [('batch2_prefill',0,4),('batch2_odd_end',4,1),('batch2_even_end',5,1)]
            for step,(name,start,queries) in enumerate(steps):
                rng=torch.Generator().manual_seed(542000+batch*100+step)
                x=(torch.randn(batch,queries,5120,generator=rng,dtype=torch.float32)*.3).to(torch.bfloat16)
                outputs=[]
                for layer in range(2,40):
                    expected=ref_model.layers[layer].attn(x.clone(),start)
                    actual=opt_model.layers[layer].attn(x.clone(),start)
                    dual.equal_output(actual,expected,f'{name}/layer{layer}')
                    dual.equal_output(ref_model.layers[layer].attn.window_kv_cache,
                                      opt_model.layers[layer].attn.window_kv_cache,'ring cache exact')
                    assert torch.equal(ref.shared_attn.topk_idxs,opt.shared_attn.topk_idxs)
                    outputs.append(dict(layer=layer,output_sha256=dual.tensor_sha(actual)))
                rows.append(dict(name=name,batch=batch,start_pos=start,queries=queries,
                                 actual_attention_layers=38,packed_indexer_enabled=True,outputs=outputs))
            counters=dict(handle.counters)
            index_counters=dict(index_handle.counters)
            assert index_counters['dense_calls']==24 and index_counters['consumer_compact_calls']==0
            # Cache-owning handle schema is checked after adapter integration.
            owned=list(handle._caches.values())
            packed_refs=[weakref.ref(v) for cache in owned for v in (cache.packed,cache.scales)]
            del owned
            handle.restore(reset=True)
            assert not handle.active
            gc.collect()
            assert all(value() is None for value in packed_refs),'restore retained packed compressed owners'
            released_packed+=len(packed_refs)
            for layer in OWNERS:
                buf=opt_model.layers[layer].attn.compress_kv_cache
                assert buf.dtype==torch.bfloat16 and bool((buf==0).all())
            # Both histories were deliberately discarded; reset the independent
            # index adapter too, then execute an actual fresh original prefill.
            index_handle.restore(reset=True)
            fresh=dual.attention_fixture(ref,prior,4,capacity)
            if batch>1:
                for layer in range(2,40):
                    obj=fresh.layers[layer].attn
                    obj.window_kv_cache=obj.window_kv_cache.repeat(batch,1,1)
                    if layer in OWNERS:
                        obj.compress_kv_cache=obj.compress_kv_cache.repeat(batch,1,1)
                        obj.indexer.k_cache=obj.indexer.k_cache.repeat(batch,1,1)
            x=torch.zeros(batch,2,5120,dtype=torch.bfloat16)
            for layer in range(2,40):
                dual.equal_output(opt_model.layers[layer].attn(x.clone(),0),fresh.layers[layer].attn(x.clone(),0),'fresh prefill after restore')
            rows.append(dict(name='restore_fresh_prefill',batch=batch,queries=2,actual_attention_layers=38,
                             packed_indexer_enabled=False,counters_before_restore=counters,index_counters_before_restore=index_counters))
        finally:
            if handle.active:handle.restore(reset=True)
            if index_handle.active:index_handle.restore(reset=True)
    return dict(rows=rows,displaced_bf16_owners_released=released_bf16,
                packed_planes_released_on_restore=released_packed,
                fixture_stubs=['synthetic projections/identity RMS and identity RoPE frequencies',
                               'deterministic compressor latents','identity FP8 window quantizer',
                               'E4M3 and E8M0 quantizers use independent CPU arithmetic, not TileLang',
                               'TP4 repeated-local BF16 collective simulator, not NCCL'],
                executed_long_prefill_tokens=129,model_weight_equivalence=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('output exists; preserve previous evidence')
    import torch
    initial_default_dtype=str(torch.get_default_dtype())
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    source=ROOT/'overlay/modules/dsv41_model'
    sys.path.insert(0,str(source))
    prior=load(ROOT/'probes/dsv41_packed_index_diff.py','dsv41_kv_prior_packed_oracle')
    dual=load(ROOT/'probes/dsv41_dual_sparse_diff.py','dsv41_kv_dual_oracle')
    core=load(source/'dsv41_packed_kv.py','dsv41_packed_kv')
    adapter=load(source/'dsv41_packed_kv_reference_adapter.py','dsv41_packed_kv_reference_adapter')
    index_adapter=load(source/'dsv41_packed_reference_adapter.py','dsv41_packed_reference_adapter')
    deps=[ROOT/'probes/dsv41_packed_index_diff.py',ROOT/'probes/dsv41_dual_sparse_diff.py']+[source/name for name in (
        'dsv41_packed_kv.py','dsv41_packed_kv_triton.py','dsv41_packed_kv_reference_adapter.py',
        'dsv41_dual_sparse.py','dsv41_dual_sparse_reference_adapter.py','dsv41_packed_reference_adapter.py',
        'dsv41_packed_index.py','dsv41_indexer.py','dsv41_reference_adapter.py')]
    dependencies={str(path.relative_to(ROOT)):sha(path) for path in deps}
    kernel=reference_quantizer(args.reference_dir/'kernel.py',prior)
    result=dict(schema=1,passed=False,cpu_only=True,gpu_numerics=False,gpu_producer_bytes_validated=False,
                gpu_performance=False,v41_model_equivalence=False,model_sha256=MODEL_SHA,kernel_sha256=KERNEL_SHA,
                probe_sha256=sha(__file__),dependencies_sha256=dependencies,
                runtime=dict(torch_version=torch.__version__,python_version=sys.version,
                             platform=platform.platform(),initial_default_dtype=initial_default_dtype),
                cpu_overflow_observation_scope='This recorded Torch CPU runtime only; not TileLang/GPU policy')
    result['format']=format_checks(core,kernel,dual)
    result['scores']=score_checks(core,kernel,dual)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        result['bf16_default_scores']=score_checks(core,kernel,dual)
    finally:torch.set_default_dtype(previous)
    result['adapter']=adapter_checks(core,adapter,index_adapter,args.reference_dir,prior,dual)
    assert dependencies=={str(path.relative_to(ROOT)):sha(path) for path in deps},'dependencies changed during run'
    assert not torch.cuda.is_initialized()
    result.update(passed=True,cuda_initialized=False,default_dtype_restored=torch.get_default_dtype()==previous)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=True,format_pairs=4096,score_cases=len(result['scores']),
                         adapter_steps=len(result['adapter']['rows']),output=str(args.output))))


if __name__=='__main__':main()
