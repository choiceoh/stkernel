#!/usr/bin/env python3
"""Independent CPU format/score/cache oracle for the V4.1 packed indexer.

The model and quantizer wrapper are extracted from SHA-pinned official source.
The TileLang quantization kernel is replaced by an explicitly identified CPU
arithmetic oracle; this does not validate TileLang code generation or GPU
numerics. No model weights, CUDA contexts, downloads or distributed jobs run.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)
MIDPOINTS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
INDEX_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
OWNERS = (2, 8, 14, 20)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bf16_bits(value):
    """Arithmetic IEEE FP32 then independent ties-even narrowing to BF16."""
    bits = f32_bits(value)
    if bits & 0x7f800000 == 0x7f800000:
        return 0x7fc0 if bits & 0x7fffff else bits >> 16
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def scalar_decode_bits(code, scale_byte):
    if scale_byte == 255:
        return 0x7fc0
    value = math.ldexp(LEVELS[code & 7], scale_byte - 127)
    if code & 8:
        value = -value
    return bf16_bits(value)


def scalar_scale(amax):
    """Independent arithmetic form, intentionally not bit-ceil implementation."""
    product = f32(max(float(amax), 6 * 2.**-126) * f32(1 / 6))
    mantissa, exponent = math.frexp(product)
    exponent -= int(mantissa == .5)
    return math.ldexp(1., exponent)


def arithmetic_unpack(packed, scales):
    import torch
    low, high = packed & 15, packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    values = torch.tensor(LEVELS, dtype=torch.float32)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    scale = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.to(torch.int32) - 127)
    scale = scale.masked_fill(scales == 255, torch.nan).repeat_interleave(32, dim=-1)
    return (values * scale).to(torch.bfloat16)


def oracle_quantize(x):
    """Official FP32 scale formula plus explicit E2M1 RN-even code selection.

    This is a CPU semantics oracle, not the vendor TileLang kernel. The finite
    producer domain is separate from decoding all 256 possible scale bytes.
    """
    import torch
    if x.dtype != torch.bfloat16 or x.shape[-1] % 32:
        raise ValueError("quantizer fixture expects BF16 groups of 32")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite producer fixture is outside the admitted domain")
    groups = x.float().unflatten(-1, (-1, 32))
    amax = groups.abs().amax(dim=-1).clamp_min(6 * 2.**-126)
    product = amax * torch.tensor(1 / 6, dtype=torch.float32)
    mantissa, exponent = torch.frexp(product)
    exponent = exponent - (mantissa == .5).to(exponent.dtype)
    scales = torch.ldexp(torch.ones_like(product), exponent)
    scaled = (groups / scales.unsqueeze(-1)).clamp(-6, 6)
    magnitude = scaled.abs()
    boundaries = torch.tensor(MIDPOINTS, dtype=torch.float32)
    code = torch.bucketize(magnitude, boundaries, right=False)
    # At an exact midpoint the lower code wins iff its mantissa LSB is even.
    for index, midpoint in enumerate(MIDPOINTS):
        if index & 1:
            code = code + (magnitude == midpoint).to(code.dtype)
    code = (code.to(torch.uint8) | (scaled.signbit().to(torch.uint8) << 3)).flatten(-2)
    packed = code[..., ::2] | (code[..., 1::2] << 4)
    sf = (exponent + 127).to(torch.uint8)
    assert bool(((sf >= 1) & (sf <= 253)).all())
    return packed, sf, arithmetic_unpack(packed, sf)


def reference_kernel(path):
    """Execute the original Python wrapper, replacing only TileLang dispatch."""
    import torch
    if sha(path) != KERNEL_SHA:
        raise ValueError("official kernel.py SHA mismatch")
    tree = ast.parse(path.read_bytes(), filename=str(path))
    names = {"fp4_act_quant", "fast_log2_ceil", "fast_pow2", "fast_round_scale"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    module = ModuleType("dsv41_pinned_quantizer_cpu_oracle")
    ns = vars(module)
    ns.update(__file__=str(path), torch=torch, FP8="float8_e4m3", FE8M0="float8_e8m0fnu")
    class Intrinsics:
        @staticmethod
        def reinterpret(dtype, value):
            if dtype == "uint32":
                return f32_bits(value)
            if dtype == "float32":
                return struct.unpack("<f", struct.pack("<I", int(value)))[0]
            raise AssertionError(dtype)
        @staticmethod
        def Cast(dtype, value):
            assert dtype == "int32"
            return int(value)
        @staticmethod
        def if_then_else(condition, yes, no):
            return yes if condition else no
    ns["T"] = Intrinsics
    def kernel_factory(n, block_size, *, scale_dtype, inplace):
        assert n % 32 == 0 and block_size == 32 and scale_dtype == ns["FE8M0"]
        def kernel(x, y, sf):
            packed, scales, dequantized = oracle_quantize(x)
            if inplace:
                y.copy_(dequantized)
            else:
                y.view(torch.uint8).copy_(packed)
            sf.view(torch.uint8).copy_(scales)
        return kernel
    ns["fp4_quant_kernel"] = kernel_factory
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec", dont_inherit=True), ns)
    return module


def full_reference(model_path, kernel, world_size):
    import torch
    import torch.nn.functional as F
    if sha(model_path) != MODEL_SHA:
        raise ValueError("official model.py SHA mismatch")
    tree = ast.parse(model_path.read_bytes(), filename=str(model_path))
    names = {"Indexer", "select_candidate_blocks", "apply_rotary_emb"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    module = ModuleType("dsv41_pinned_model_cpu_oracle")
    ns = vars(module)
    ns.update(__file__=str(model_path), torch=torch, nn=torch.nn, F=F, ModelArgs=object,
              world_size=world_size, fp4_block_size=32, fp4_act_quant=kernel.fp4_act_quant,
              shared_attn=SimpleNamespace(index_k=None, candidates=None),
              dist=SimpleNamespace(all_reduce=lambda value: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(model_path), "exec", dont_inherit=True), ns)
    forward = next(n for n in nodes if isinstance(n, ast.ClassDef)).body
    forward = next(n for n in forward if isinstance(n, ast.FunctionDef) and n.name == "forward")
    scores = [n for n in forward.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "index_score" for t in n.targets)]
    assert len(scores) == 2
    function = ast.parse("def source_score(q, index_k, weights): pass").body[0]
    function.body = scores + [ast.Return(ast.Name("index_score", ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(model_path), "exec", dont_inherit=True), ns)
    return module


def tensor_sha(x):
    import torch
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def exact_bits(actual, expected, label):
    import torch
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), label


def format_checks(core, kernel):
    import torch
    scales = torch.arange(256, dtype=torch.int32).repeat_interleave(16).to(torch.uint8)
    codes = torch.arange(16, dtype=torch.uint8).repeat(256)
    packed = (codes | (codes << 4)).reshape(-1, 1).expand(-1, 64).contiguous()
    sf = scales.reshape(-1, 1).expand(-1, 4).contiguous()
    expected_bits = torch.tensor([scalar_decode_bits(int(c), int(s)) for c, s in zip(codes, scales)],
                                 dtype=torch.int32).to(torch.int16)
    expected = expected_bits[:, None].expand(-1, 128).contiguous().view(torch.bfloat16)
    actual = torch.cat([core.unpack_index_tile(packed[lo:lo+1024], sf[lo:lo+1024])
                        for lo in range(0, len(packed), 1024)])
    exact_bits(actual, expected, "exhaustive nibble/scale decoding differs")
    # Distinct low/high nibbles catch a lane swap, even when uniform-code
    # dequantization was otherwise numerically correct.
    byte_values = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    mixed = byte_values[:, None].expand(-1, 64).contiguous()
    mixed_sf = torch.full((256, 4), 127, dtype=torch.uint8)
    exact_bits(core.unpack_index_tile(mixed, mixed_sf), arithmetic_unpack(mixed, mixed_sf), "nibble order")
    # Compare every nonnegative finite BF16 amax against ORIGINAL source helper
    # expressions, with the product explicitly rounded to FP32 as in the DSL.
    class FP32(float):
        def __mul__(self, other):
            return FP32(f32(float(self) * float(other)))
    values = torch.arange(0x7f80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().tolist()
    for value in values:
        got = kernel.fast_round_scale(FP32(max(value, 6*2.**-126)), FP32(f32(1/6)))
        assert f32_bits(got) == f32_bits(scalar_scale(value)), "official scale expression drift"
    # The official Python wrapper actually allocates the packed shell and scale
    # dtype, invokes the substitute kernel, and copies BF16 for inplace=True.
    fixtures = torch.tensor([0., -0., 2.**-133, -2.**-133, 2.**-127, -2.**-127,
                             6*2.**-126, -6*2.**-126, .25, -.25, .75, -.75,
                             1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5,
                             5., -5., 6., -6., 3.3895313892515355e38, -3.3895313892515355e38],
                            dtype=torch.bfloat16)[:, None].expand(-1, 128).contiguous()
    y, s = kernel.fp4_act_quant(fixtures, 32, False)
    original_inplace = kernel.fp4_act_quant(fixtures.clone(), 32, True)
    exact_bits(core.unpack_index_tile(y.view(torch.uint8), s.view(torch.uint8)), original_inplace,
               "packed and inplace official-wrapper CPU oracle disagree")
    assert bool(original_inplace[-2:].isinf().all()), "overflow fixture did not reach infinity"
    assert original_inplace[1, 0].view(torch.int16).item() == -32768
    assert original_inplace[4, 0].view(torch.int16).item() == 0x40
    return dict(nibble_scale_pairs=4096, distinct_packed_bytes=256, scale_amax_bf16_values=len(values),
                packed_inplace_wrapper_rows=len(fixtures), exact_bf16_bits=True,
                signed_zero=True, subnormal=True, finite_input_overflow=True,
                nan_scale_byte_255=True, gpu_quantizer_validated=False)


def bf16_sum(parts):
    value = parts[0].clone()
    for part in parts[1:]:
        value.add_(part)
    return value


def model_fixture(module, world, capacity):
    import torch
    model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(indexer=None)) for _ in range(40)])
    frequencies = torch.ones(capacity + 8, 32, dtype=torch.complex64)
    for layer in INDEX_LAYERS:
        obj = module.Indexer.__new__(module.Indexer)
        torch.nn.Module.__init__(obj)
        obj.owns_k, obj.compress_ratio = layer in OWNERS, 2 if layer < 20 else 1
        obj.is_candidate_source, obj.uses_candidates = layer == 20, layer > 20
        obj.candidate_topk_blocks, obj.candidate_block_size = 2048, 8
        obj.dim, obj.n_heads, obj.n_local_heads = 5120, 32, 32 // world
        obj.index_head_dim, obj.rope_head_dim, obj.index_topk = 128, 64, 512
        obj.q_lora_rank, obj.softmax_scale = 1280, 128**-.5
        obj.freqs_cis = frequencies
        obj.wq_b = lambda x: x
        obj.weights_proj = lambda x: x
        if obj.owns_k:
            obj.wk, obj.k_norm = lambda x:x.clone(), lambda x:x
            obj.register_buffer("k_cache", torch.zeros(1, capacity // obj.compress_ratio, 128,
                                                       dtype=torch.bfloat16), persistent=False)
        model.layers[layer].attn.indexer = obj
    return model


def adapter_checks(core, adapter, model_path, kernel, world):
    """Exact short history, then explicitly seeded long-history decode.

    The long prefix is NOT executed as prefill. It is seeded identically into
    reference BF16 and candidate packed owners so the real long decode source,
    consumers and shared-slot transitions can be compared at bounded CPU cost.
    """
    import torch
    capacity, long_prefix = 32780, 32776
    states, old_weakrefs = [], []
    for rank in range(world):
        ref = full_reference(model_path, kernel, world)
        opt = full_reference(model_path, kernel, world)
        ref_model, opt_model = model_fixture(ref, world, capacity), model_fixture(opt, world, capacity)
        old_weakrefs += [weakref.ref(opt_model.layers[layer].attn.indexer.k_cache) for layer in OWNERS]
        # Even a preexisting shared pointer must not retain an old BF16 owner.
        opt.shared_attn.index_k = opt_model.layers[20].attn.indexer.k_cache
        handle = adapter.install_reference_packed_indexer(opt_model, opt, enabled=True)
        assert handle.active
        states.append((ref, opt, ref_model, opt_model, handle))
    gc.collect()
    assert all(ref() is None for ref in old_weakrefs), "installation retained a displaced BF16 cache"
    oracle_caches = {layer:torch.zeros(1, capacity // (2 if layer < 20 else 1),128,dtype=torch.bfloat16)
                     for layer in OWNERS}
    active_owner = None
    rows, previous_long_mask = [], None
    steps = [("warm_prefill",0,8), ("ratio2_odd_end_no_publish",8,1),
             ("ratio2_even_end_publish",9,1), ("ratio2_odd_end_again",10,1),
             ("ratio2_even_end_again",11,1),
             ("seeded_long_odd_end",long_prefix,1), ("seeded_long_even_end",long_prefix+1,1)]
    try:
        for step_index,(name,start,queries) in enumerate(steps):
            rng=torch.Generator().manual_seed(521000+step_index)
            if start == long_prefix:
                # This seed covers only history storage/cursors. Every next
                # Indexer.forward, producer update, score and collective is real.
                for layer in OWNERS:
                    ratio=2 if layer<20 else 1
                    count=long_prefix//ratio
                    pre=(torch.randn(1,count,128,generator=rng)*.5).to(torch.bfloat16)
                    y,sf=kernel.fp4_act_quant(pre,32,False)
                    dequant=kernel.fp4_act_quant(pre.clone(),32,True)
                    oracle_caches[layer][:,:count].copy_(dequant)
                    for ref,opt,ref_model,opt_model,handle in states:
                        ref_model.layers[layer].attn.indexer.k_cache[:,:count].copy_(dequant)
                        core.write_packed_index(handle._caches[layer],y,sf,start_slot=0)
                        handle._valid[layer]=count
                for ref,opt,ref_model,opt_model,handle in states:
                    ref.shared_attn.index_k=ref_model.layers[20].attn.indexer.k_cache
                    ref.shared_attn.candidates=None
                    opt.shared_attn.index_k=handle._caches[20]
                    opt.shared_attn.candidates=None
                    handle._step=(0,1,long_prefix,128)
                    handle._last_layer=36
                    handle._state=None
                active_owner=20
            x=torch.zeros(1,queries,5120,dtype=torch.bfloat16)
            qr=torch.zeros(1,queries,1280,dtype=torch.bfloat16)
            payloads, layer_rows, outputs = [],[],[]
            for layer in INDEX_LAYERS:
                ratio=2 if layer<20 else 1
                width=(start+queries)//ratio
                qraw=(torch.randn(1,queries,32,128,generator=rng)*.5).to(torch.bfloat16)
                weights_raw=(torch.randn(1,queries,32,generator=rng)*3).to(torch.bfloat16)
                weights_raw[...,0],weights_raw[...,1]=-8,8
                q=kernel.fp4_act_quant(qraw.clone(),32,True)
                weights=weights_raw*(128**-.5*32**-.5)
                latent=None
                publish=layer in OWNERS and (start==0 or (start+1)%ratio==0)
                if publish:
                    count=queries//ratio if start==0 else 1
                    latent=(torch.randn(1,count,128,generator=rng)*.5).to(torch.bfloat16)
                    k=kernel.fp4_act_quant(latent.clone(),32,True)
                    oracle_caches[layer][:,start//ratio:start//ratio+count].copy_(k)
                    active_owner=layer
                assert active_owner is not None
                golden_key=oracle_caches[active_owner][:,:width]
                qparts=q.split(32//world,dim=2)
                rawparts=qraw.split(32//world,dim=2)
                wparts=weights.split(32//world,dim=2)
                wrawparts=weights_raw.split(32//world,dim=2)
                dense_parts=[states[0][0].source_score(qv.clone(),golden_key,wv) for qv,wv in zip(qparts,wparts)]
                total=bf16_sum(dense_parts)
                for rank,(ref,opt,ref_model,opt_model,handle) in enumerate(states):
                    for fixture in (ref_model,opt_model):
                        instance=fixture.layers[layer].attn.indexer
                        instance.wq_b=lambda unused,qv=rawparts[rank]:qv.clone().flatten(2)
                        instance.weights_proj=lambda unused,wv=wrawparts[rank]:wv.clone()
                    ref_calls=[]
                    def reference_reduce(value):
                        exact_bits(value,dense_parts[rank],"original local score vs independent cache")
                        ref_calls.append(list(value.shape)); value.copy_(total)
                    ref.dist.all_reduce=reference_reduce
                    expected=ref_model.layers[layer].attn.indexer(x,qr,None if latent is None else latent.clone(),start,128)
                    assert len(ref_calls)==int(world>1)
                    ref_owner=next(owner for owner in OWNERS if ref.shared_attn.index_k is ref_model.layers[owner].attn.indexer.k_cache)
                    assert ref_owner==active_owner
                    compact=layer>20 and start>0 and queries==1 and start+1>16384
                    mask=ref.shared_attn.candidates
                    if compact:
                        ids=torch.arange(width).expand(1,queries,width).masked_fill(~mask,width).sort(dim=-1).values[...,:16384]
                        valid=ids<width
                        safe=ids.clamp_max(width-1)
                        local=dense_parts[rank].gather(-1,safe).masked_fill(~valid,0)
                        global_scores=bf16_sum([p.gather(-1,safe).masked_fill(~valid,0) for p in dense_parts])
                    else:
                        local,global_scores=dense_parts[rank],total
                    def candidate_reduce(value):
                        exact_bits(value,local,"packed adapter local collective payload")
                        payloads.append(dict(rank=rank,layer=layer,shape=list(value.shape),compact=compact))
                        value.copy_(global_scores)
                    opt.dist.all_reduce=candidate_reduce
                    actual=opt_model.layers[layer].attn.indexer(x,qr,None if latent is None else latent.clone(),start,128)
                    assert torch.equal(actual,expected), f"{name}/rank{rank}/layer{layer} selected-index mismatch"
                    assert opt.shared_attn.index_k.owner_layer==active_owner
                    # Verify the exact consumed slot, including ratio2 reading
                    # the previous ratio1 owner when latent was None.
                    for lo in range(0,width,1024):
                        packed_key=core.unpack_index_tile(opt.shared_attn.index_k.packed[:,lo:lo+1024],
                                                         opt.shared_attn.index_k.scales[:,lo:lo+1024])
                        stop=min(lo+1024,width)
                        exact_bits(packed_key[:,:stop-lo],golden_key[:,lo:stop],"published slot BF16 bytes")
                    outputs.append(dict(rank=rank,layer=layer,sha256=tensor_sha(actual)))
                layer_rows.append(dict(layer=layer,ratio=ratio,published=publish,
                                       active_owner=active_owner,consumed_width=width))
            expected_calls=world*8 if world>1 else 0
            assert len(payloads)==expected_calls
            current_mask=tensor_sha(states[0][0].shared_attn.candidates)
            if name=="seeded_long_even_end":
                assert current_mask!=previous_long_mask,"long candidate fixture did not change source mask"
            previous_long_mask=current_mask
            rows.append(dict(name=name,world_size=world,start_pos=start,queries=queries,
                             seeded_long_history=start>=long_prefix,layers=layer_rows,
                             collective_payloads=payloads,output_sha256=outputs,
                             exact_original_forward=True,source_mask_sha256=current_mask))
        packed_refs=[weakref.ref(handle._caches[layer].packed) for *_,handle in states for layer in OWNERS]
        for ref,opt,ref_model,opt_model,handle in states:
            handle.restore(reset=True)
            assert handle.active is False and handle.reset_pending is True
            assert opt.shared_attn.index_k is None
            for layer in OWNERS:
                cache=opt_model.layers[layer].attn.indexer.k_cache
                assert cache.dtype==torch.bfloat16 and bool((cache==0).all())
            with __import__("unittest").TestCase().assertRaisesRegex(RuntimeError,"fresh prefill"):
                opt_model.layers[20].attn.indexer(x,qr,None,start+1,128)
        gc.collect()
        assert all(ref() is None for ref in packed_refs),"restore retained packed owner tensors"
    finally:
        for *_,handle in states:
            if handle.active:
                handle.restore(reset=True)
    return dict(rows=rows,displaced_bf16_owners_released=len(old_weakrefs),
                packed_owners_released_on_restore=len(packed_refs),
                long_seed=dict(tokens=long_prefix,executed_long_prefill=False,
                  synthetic_fields=["reference cache prefix bytes","packed prefix bytes",
                                    "per-owner valid prefix","completed previous step and shared slot"]),
                projection_fixture="synthetic BF16 projections and identity RoPE; original forward operations retained")


def score_checks(core, compact_core, model, kernel):
    import torch
    specs = [("dense_partial", 2, 3, 37, False, False),
             ("compact_partial", 2, 3, 37, True, False),
             ("ties_full_domain", 1, 1, 65, True, True),
             ("long_dense", 1, 1, 32777, False, False),
             ("long_compact", 1, 1, 32777, True, False)]
    rows = []
    for case_i, (name, batch, queries, width, compact, ties) in enumerate(specs):
        rng = torch.Generator().manual_seed(931000 + case_i)
        before = (torch.randn(batch, width, 128, generator=rng)*.5).to(torch.bfloat16)
        q = (torch.randn(batch, queries, 32, 128, generator=rng)*.5).to(torch.bfloat16)
        if ties:
            before.zero_(); q.zero_()
        y, sf = kernel.fp4_act_quant(before, 32, False)
        key = kernel.fp4_act_quant(before.clone(), 32, True)
        q = kernel.fp4_act_quant(q, 32, True)
        raw_weights = (torch.randn(batch, queries, 32, generator=rng)*3).to(torch.bfloat16)
        raw_weights[..., 0], raw_weights[..., 1] = -8, 8
        weights = raw_weights * (128**-.5 * 32**-.5)
        cache = core.allocate_packed_index_cache(batch, width + 7, device="cpu", owner_layer=20, ratio=1)
        core.write_packed_index(cache, y, sf, start_slot=0)
        selected = None
        mask = torch.ones(batch, queries, width, dtype=torch.bool)
        blocks = 1 if ties else (2048 if width > 16384 else 2)
        if compact:
            source = torch.zeros(batch, queries, width, dtype=torch.bfloat16) if ties else torch.randn(batch, queries, width, generator=rng).to(torch.bfloat16)
            mask = model.select_candidate_blocks(source, width, blocks, 8)
            selected = torch.arange(width).expand(batch, queries, width).masked_fill(~mask, width).sort(dim=-1).values
            selected = selected[..., :min(width, blocks*8)]
            selected = selected.masked_fill(selected == width, -1).to(torch.int32).contiguous()
        for world in (1, 4):
            qs, ws = q.split(32//world, dim=2), weights.split(32//world, dim=2)
            dense_parts = [model.source_score(qv.clone(), key, wv) for qv, wv in zip(qs, ws)]
            parts = dense_parts if selected is None else [p.gather(-1, selected.clamp_min(0).long()).masked_fill(selected < 0, 0) for p in dense_parts]
            total = bf16_sum(parts)
            expected_dense = bf16_sum(dense_parts).masked_fill(~mask, -torch.inf)
            indices = expected_dense.topk(min(512, width), dim=-1, sorted=False).indices.sort(dim=-1).values
            expected_indices = (indices + 256).int()
            payloads = []
            for rank in range(world):
                def reduce(value):
                    exact_bits(value, parts[rank], "pre-collective BF16 scores differ")
                    payloads.append(dict(rank=rank, shape=list(value.shape), elements=value.numel(), bytes=value.numel()*2))
                    value.copy_(total)
                scores = core.packed_index_scores(qs[rank], cache, ws[rank], width=width, ids=selected,
                            reduce_fn=reduce if world > 1 else None, backend="torch")
                exact_bits(scores, total, "packed index scores differ")
                if selected is None:
                    got = scores.topk(min(512, width), dim=-1, sorted=False).indices.sort(dim=-1).values.int() + 256
                else:
                    got = compact_core.compact_topk(scores, selected, width, 256, 512, full_width=width)
                assert torch.equal(got, expected_indices), "final full-width topk differs"
            assert len(payloads) == (world if world > 1 else 0)
            rows.append(dict(name=name, world_size=world, batch=batch, queries=queries, width=width,
                             compact=compact, exact_scores=True, exact_indices=True,
                             collective_payloads=payloads, key_sha256=tensor_sha(key),
                             packed_sha256=tensor_sha(y.view(torch.uint8)), scale_sha256=tensor_sha(sf.view(torch.uint8))))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve the previous receipt")
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    source_dir = ROOT / "overlay/modules/dsv41_model"
    core_path = source_dir / "dsv41_packed_index.py"
    compact_path = source_dir / "dsv41_indexer.py"
    sys.path.insert(0,str(source_dir))
    compact_core = load(compact_path, "dsv41_indexer")
    core = load(core_path, "dsv41_packed_index")
    adapter_path=source_dir/"dsv41_packed_reference_adapter.py"
    adapter=load(adapter_path,"dsv41_packed_reference_adapter")
    kernel = reference_kernel(args.reference_dir / "kernel.py")
    model = full_reference(args.reference_dir / "model.py", kernel, 1)
    receipt = dict(schema=1, passed=False, scope="independent CPU bit and source-extracted arithmetic oracle",
                   model_sha256=MODEL_SHA, kernel_sha256=KERNEL_SHA, probe_sha256=sha(__file__),
                   candidate_sha256={p.name:sha(p) for p in (core_path, compact_path,adapter_path)},
                   torch_version=torch.__version__, gpu_quantizer_validated=False,
                   gpu_numerics=False, gpu_performance=False, model_equivalence=False)
    try:
        receipt["format"] = format_checks(core, kernel)
        receipt["scores"] = score_checks(core, compact_core, model, kernel)
        receipt["adapter"] = [adapter_checks(core,adapter,args.reference_dir/"model.py",kernel,world)
                              for world in (1,4)]
        receipt["static_cache_bytes"] = [dict(positions=s, bf16=s*256, packed=s*68,
            scope="one D128 cache; excludes allocator/state metadata and other model caches") for s in (131072,1048576)]
        receipt["cuda_initialized"] = torch.cuda.is_initialized()
        assert receipt["cuda_initialized"] is False
        receipt["passed"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed":True,"output":str(args.output),"sha256":sha(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
