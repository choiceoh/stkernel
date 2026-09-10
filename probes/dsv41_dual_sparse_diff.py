#!/usr/bin/env python3
"""Source-pinned CPU oracle for dual-pool V4.1 sparse attention.

The official sparse_attn wrapper and Attention methods are executed from their
original AST. TileLang dispatch is replaced by the explicit 64-slot arithmetic
below, not by full softmax. CPU equality does not establish GPU GEMM/exp or
TileLang/Triton numerical equivalence, model equivalence, or performance.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
OWNERS = (2, 8, 14, 20)
INDEXERS = (2, 8, 14, 20, 24, 28, 32, 36)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def equal_output(actual, expected, label):
    """Exact finite BF16 bits and equal NaN locations; NaN payload unspecified."""
    import torch
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape, label
    nan = expected.isnan()
    assert torch.equal(actual.isnan(), nan), f"{label}: NaN domain differs"
    assert torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16)), f"{label}: BF16 bits differ"


def gather_concat(kv, ids):
    """Independent original-domain (-1 or in-range) concatenated gather."""
    import torch
    assert ids.dtype == torch.int32
    if bool(((ids < -1) | (ids >= kv.shape[1])).any()):
        raise ValueError("outside the defined official index domain")
    out = torch.zeros((*ids.shape, kv.shape[-1]), dtype=kv.dtype)
    for batch in range(ids.shape[0]):
        valid = ids[batch] >= 0
        out[batch][valid] = kv[batch, ids[batch][valid].long()]
    return out


def gather_dual(window, compressed, ids):
    """Two actual pool reads, independently checked against concat bytes."""
    import torch
    width = window.shape[1]
    extra = 0 if compressed is None else compressed.shape[1]
    if bool(((ids < -1) | (ids >= width + extra)).any()):
        raise ValueError("outside the defined official index domain")
    out = torch.zeros((*ids.shape, window.shape[-1]), dtype=window.dtype)
    for batch in range(ids.shape[0]):
        win = (ids[batch] >= 0) & (ids[batch] < width)
        comp = ids[batch] >= width
        out[batch][win] = window[batch, ids[batch][win].long()]
        if compressed is not None:
            out[batch][comp] = compressed[batch, (ids[batch][comp] - width).long()]
    return out


def check_actual_gather(core, window, compressed, ids):
    """Check the candidate's actual bounded gather, not a second mock gather."""
    import torch
    empty = window[:, :0] if compressed is None else compressed
    joined = torch.cat((window, empty), dim=1)
    for batch in range(ids.shape[0]):
        for query in range(ids.shape[1]):
            for lo in range(0, ids.shape[-1], 64):
                selected = torch.full((1, 64), -1, dtype=torch.int64)
                count = min(64, ids.shape[-1] - lo)
                selected[:, :count] = ids[batch, query, lo:lo+count]
                keys, valid = core._gather_tile(window, compressed, torch.tensor([batch]), selected)
                expected = gather_concat(joined[batch:batch+1], selected[:, None].to(torch.int32))[:, 0]
                equal_output(keys, expected, "actual candidate gathered bytes")
                assert torch.equal(valid, selected != -1)


def online64(q, kv, sink, ids, scale):
    """Literal staged sparse_attn_kernel arithmetic, with CPU FP32 GEMMs.

    Keep the finite initial maximum, tail padding, index order, duplicate slots,
    pre-PV BF16 probability conversion and one final sink denominator term.
    The reference DSL uses GEMM accumulation and GPU exp/reductions; this CPU
    stand-in is not a GPU-bit oracle.
    """
    import torch
    gathered = gather_concat(kv, ids)
    bsz, queries, heads, dim = q.shape
    output = torch.empty_like(q)
    for b in range(bsz):
        for row in range(queries):
            maximum = torch.full((heads,), -1e30, dtype=torch.float32)
            denominator = torch.zeros(heads, dtype=torch.float32)
            accumulator = torch.zeros((heads, dim), dtype=torch.float32)
            for lo in range(0, ids.shape[-1], 64):
                take = min(64, ids.shape[-1] - lo)
                tile = torch.zeros((64, dim), dtype=torch.bfloat16)
                tile[:take] = gathered[b, row, lo:lo+take]
                valid = torch.zeros(64, dtype=torch.bool)
                valid[:take] = ids[b, row, lo:lo+take] != -1
                # Start invalid accumulators at -inf before the QK GEMM.
                scores = torch.zeros((heads, 64), dtype=torch.float32).masked_fill(~valid, -torch.inf)
                scores = torch.addmm(scores, q[b, row].float(), tile.float().T)
                scores = scores * scale
                previous = maximum.clone()
                maximum = torch.maximum(maximum, scores.amax(dim=1))
                rescale = torch.exp(previous - maximum)
                probabilities = torch.exp(scores - maximum[:, None])
                denominator = denominator * rescale + probabilities.sum(dim=1)
                accumulator = accumulator * rescale[:, None]
                accumulator = torch.addmm(accumulator, probabilities.to(torch.bfloat16).float(), tile.float())
            denominator = denominator + torch.exp(sink - maximum)
            output[b, row] = (accumulator / denominator[:, None]).to(torch.bfloat16)
    return output


def reference_kernel(path):
    import torch
    path = Path(path)
    if sha(path) != KERNEL_SHA:
        raise ValueError("official kernel.py SHA mismatch")
    tree = ast.parse(path.read_bytes(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn")
    module = ModuleType("dsv41_official_sparse_cpu_oracle")
    module.__file__ = str(path)
    module.torch = torch
    module.calls = []
    def factory(heads, dim, scale):
        assert dim == 512 and heads >= 16
        def run(q, kv, out, sink, ids):
            module.calls.append(dict(heads=heads, queries=q.shape[1], width=kv.shape[1], slots=ids.shape[-1]))
            out.copy_(online64(q, kv, sink, ids, scale))
        return run
    module.sparse_attn_kernel = factory
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), vars(module))
    return module


def numerical_checks(core, kernel):
    import torch
    specs = [
        ("slots63_win127_heads8", 1, 1, 8, 127, 5, 63, False),
        ("slots64_win128_heads16", 1, 2, 16, 128, 7, 64, False),
        ("slots65_win129_heads64", 1, 1, 64, 129, 9, 65, False),
        ("strided_batch2_duplicates", 2, 3, 16, 128, 17, 129, True),
        ("empty_compressed", 1, 2, 16, 128, 0, 65, False),
        ("empty_window", 1, 1, 16, 0, 9, 65, False),
        ("all_invalid_both_empty", 1, 1, 8, 0, 0, 65, False),
        ("empty_indices", 1, 1, 16, 128, 7, 0, False),
        ("ties_duplicate_order", 1, 1, 16, 128, 7, 65, False),
    ]
    rows = []
    for case, (name, b, qn, h, w, c, slots, strided) in enumerate(specs):
        rng = torch.Generator().manual_seed(531000 + case)
        def tensor(shape):
            return (torch.randn(shape, generator=rng) * .3).to(torch.bfloat16)
        q = tensor((b, qn, h, 512))
        if strided:
            window = tensor((b, w * 2 + 4, 1024))[:, 2:2+2*w:2, ::2]
            compressed = tensor((b, c * 2 + 6, 1024))[:, 3:3+2*c:2, ::2]
        else:
            window, compressed = tensor((b, w, 512)), tensor((b, c, 512))
        ids = torch.randint(max(1, w+c), (b, qn, slots), generator=rng, dtype=torch.int32)
        ids[..., ::7] = -1
        if w+c == 0:
            ids.fill_(-1)
        if slots > 4 and w+c:
            ids[..., 1:4] = min(w, w+c-1)  # repeat one exact slot across pools/boundaries
        if name == "ties_duplicate_order":
            q.zero_()
        sink = torch.linspace(-3, 3, h, dtype=torch.float32)
        joined = torch.cat((window, compressed), dim=1)
        equal_output(gather_dual(window, compressed, ids), gather_concat(joined, ids), name+" gather")
        check_actual_gather(core, window, compressed, ids)
        expected = kernel.sparse_attn(q, joined, sink, ids, 512**-.5)
        actual = core.dual_sparse_attn(q, window, compressed, sink, ids, 512**-.5)
        equal_output(actual, expected, name)
        assert actual.is_contiguous()
        if c == 0:
            equal_output(core.dual_sparse_attn(q, window, None, sink, ids, 512**-.5), expected, name+" None")
        rows.append(dict(name=name, batch=b, queries=qn, heads=h, window=w, compressed=c, slots=slots,
                         strided=strided, gather_bytes_exact=True, cpu_output_bf16_bits_exact=True,
                         output_sha256=tensor_sha(actual)))
    # Nonstandard sink values expose why all-invalid rows cannot just return 0.
    q = torch.ones((1, 1, 16, 512), dtype=torch.bfloat16)
    window = torch.ones((1, 2, 512), dtype=torch.bfloat16)
    ids = torch.full((1, 1, 65), -1, dtype=torch.int32)
    for value in (0., -float("inf"), -3e30, 3e30, float("inf"), float("nan")):
        sink = torch.full((16,), value, dtype=torch.float32)
        expected = kernel.sparse_attn(q, window, sink, ids, 512**-.5)
        actual = core.dual_sparse_attn(q, window, None, sink, ids, 512**-.5)
        equal_output(actual, expected, "all-invalid sink")
        rows.append(dict(name="all_invalid_sink", sink=str(value), nan_elements=int(actual.isnan().sum()),
                         cpu_output_bf16_bits_exact=True, nan_payload_equivalence=False))
    # Same storage, new content must be consumed; no cached materialized pool.
    rng = torch.Generator().manual_seed(531111)
    window = torch.randn(1, 128, 512, generator=rng).to(torch.bfloat16)
    compressed = torch.randn(1, 7, 512, generator=rng).to(torch.bfloat16)
    ids = torch.tensor([[[127,128,134,0,128,-1]]], dtype=torch.int32)
    sink = torch.zeros(16, dtype=torch.float32)
    pointers = (window.data_ptr(), compressed.data_ptr(), q.data_ptr(), sink.data_ptr(), ids.data_ptr())
    outputs = []
    for change in range(3):
        expected = kernel.sparse_attn(q, torch.cat((window,compressed),1), sink, ids, 512**-.5)
        actual = core.dual_sparse_attn(q, window, compressed, sink, ids, 512**-.5)
        equal_output(actual, expected, "same-address mutation")
        outputs.append(tensor_sha(actual))
        window.add_(.25); compressed.mul_(-.75); q.mul_(-.5); sink.add_(.3)
        ids[...,0] = change
    assert len(set(outputs)) == 3 and pointers == (window.data_ptr(),compressed.data_ptr(),q.data_ptr(),sink.data_ptr(),ids.data_ptr())
    rows.append(dict(name="same_address_changed_inputs", outputs=outputs, same_pointers=True))
    return rows


def full_reference(model_path, kernel, quant_oracle, world):
    import torch
    # Reuse the independent FP4/indexer source loader from the previous probe;
    # attention's E4M3 and FP8 quantizers below are explicit identity fixtures.
    quant = quant_oracle.reference_kernel(model_path.with_name("kernel.py"))
    original_factory = quant.fp4_quant_kernel
    def factory(n, block_size, *, scale_dtype, inplace):
        if block_size == 16:
            assert inplace and scale_dtype == quant.FP8
            def identity_compressed(x, y, sf):
                y.copy_(x); sf.fill_(1)
            return identity_compressed
        return original_factory(n, block_size, scale_dtype=scale_dtype, inplace=inplace)
    quant.fp4_quant_kernel = factory
    module = quant_oracle.full_reference(model_path, quant, world)
    tree = ast.parse(model_path.read_bytes(), filename=str(model_path))
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {"Attention","get_window_topk_idxs"}]
    class ProjectionTorch:
        # The real Attention AST reaches its output projection; the fixture
        # avoids allocating/contracting 38 real 5120-dimension weight matrices.
        def __getattr__(self, name):
            return getattr(torch, name)
        def einsum(self, equation, value, weight):
            if equation == "bsgd,grd->bsgr":
                assert weight.shape[-2] == 1024
                return value[..., :1024].clone()
            return torch.einsum(equation, value, weight)
    module.torch = ProjectionTorch()
    module.lru_cache = functools.lru_cache
    module.sparse_attn = kernel.sparse_attn
    module.act_quant = lambda x, *args, **kwargs:x
    module.fp8_block_size, module.scale_fmt, module.scale_dtype = 128, "ue8m0", torch.float8_e8m0fnu
    module.shared_attn.compress_kv = None
    module.shared_attn.topk_idxs = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(model_path),"exec",dont_inherit=True),vars(module))
    return module


def attention_fixture(module, quant_oracle, world, capacity):
    import torch
    model = quant_oracle.model_fixture(module, world, capacity)
    frequencies = torch.ones(capacity,32,dtype=torch.complex64)
    for layer in range(2,40):
        old_indexer = model.layers[layer].attn.indexer
        obj = module.Attention.__new__(module.Attention)
        torch.nn.Module.__init__(obj)
        geometry = dict(layer_id=layer,dim=5120,n_heads=64,n_local_heads=64//world,
            q_lora_rank=1280,o_lora_rank=1024,head_dim=512,rope_head_dim=64,nope_head_dim=448,
            n_groups=8,n_local_groups=8//world,window_size=128,compress_ratio=2 if layer<20 else 1,
            eps=1e-20,softmax_scale=512**-.5,is_kv_source=layer in OWNERS,is_index_source=layer in INDEXERS)
        for name,value in geometry.items():setattr(obj,name,value)
        obj.attn_sink = torch.linspace(-1,1,obj.n_local_heads)
        obj.wq_a = lambda x:x[...,:1280].clone()
        obj.q_norm = lambda x:x
        obj.wq_b = lambda x,h=obj.n_local_heads:x[...,:512].repeat(1,1,h)
        obj.wkv = lambda x:x[...,:512].clone()
        obj.kv_norm = lambda x:x
        obj.wo_a = SimpleNamespace(weight=torch.zeros(1,dtype=torch.bfloat16).expand(obj.n_local_groups*1024,4096))
        obj.wo_b = lambda x:x.repeat(1,1,(5120+x.shape[-1]-1)//x.shape[-1])[...,:5120].clone()
        obj.register_buffer("window_kv_cache",torch.zeros(1,128,512,dtype=torch.bfloat16),persistent=False)
        obj.register_buffer("freqs_cis",frequencies,persistent=False)
        obj.indexer = old_indexer
        if old_indexer is not None:
            old_indexer.freqs_cis = frequencies
            old_indexer.wq_b = lambda x,h=32//world:x[...,:128].repeat(1,1,h)
            old_indexer.weights_proj = lambda x,h=32//world:x[...,:h].clone()
            if old_indexer.owns_k:
                old_indexer.wk = lambda x:x[...,:128].clone()
        if obj.is_kv_source:
            ratio=obj.compress_ratio
            def compressor(x,start,ratio=ratio):
                if start==0:return x[:,:x.shape[1]//ratio*ratio:ratio,:512].clone()
                return x[...,:512].clone() if (start+1)%ratio==0 else None
            obj.compressor=compressor
            obj.register_buffer("compress_kv_cache",torch.zeros(1,capacity//ratio,512,dtype=torch.bfloat16),persistent=False)
        else:obj.compressor=None
        model.layers[layer].attn=obj
    return model


def adapter_checks(adapter, packed_adapter, kernel, model_path, quant_oracle):
    import torch
    rows=[]
    # world4 exercises real local H16; world1 H64 is independently covered above.
    world,capacity=4,144
    reference=full_reference(model_path,kernel,quant_oracle,world)
    candidate=full_reference(model_path,kernel,quant_oracle,world)
    ref_model=attention_fixture(reference,quant_oracle,world,capacity)
    opt_model=attention_fixture(candidate,quant_oracle,world,capacity)
    # Single-rank CPU simulator: the other three ranks repeat the same local
    # score; BF16 in-place sum models collective rounding, not NCCL execution.
    def reduce(value):
        original=value.clone()
        for _ in range(world-1):value.add_(original)
    reference.dist.all_reduce=candidate.dist.all_reduce=reduce
    packed=packed_adapter.install_reference_packed_indexer(opt_model,candidate,enabled=True)
    handle=adapter.install_reference_dual_sparse_attention(opt_model,candidate,enabled=True)
    original_windows=[obj.attn.window_kv_cache.data_ptr() for obj in opt_model.layers[2:]]
    rng=torch.Generator().manual_seed(532000)
    try:
        for name,start,queries in [("warm_prefill",0,4),("odd_end_latent_none",4,1),("even_end_write",5,1)]:
            x=(torch.randn(1,queries,5120,generator=rng)*.3).to(torch.bfloat16)
            signatures=[]
            for layer in range(2,40):
                expected=ref_model.layers[layer].attn(x.clone(),start)
                actual=opt_model.layers[layer].attn(x.clone(),start)
                equal_output(actual,expected,f"{name}/layer{layer}")
                a,b=ref_model.layers[layer].attn,opt_model.layers[layer].attn
                equal_output(a.window_kv_cache,b.window_kv_cache,"window cache mutation")
                if layer in OWNERS:
                    equal_output(a.compress_kv_cache,b.compress_kv_cache,"compressed cache mutation")
                assert torch.equal(reference.shared_attn.topk_idxs,candidate.shared_attn.topk_idxs)
                # Attention publishes its own compressed owner EVEN when the
                # compressor returned None; Indexer has a different rule.
                if layer in OWNERS:
                    assert reference.shared_attn.compress_kv is a.compress_kv_cache
                    assert candidate.shared_attn.compress_kv is b.compress_kv_cache
                signatures.append(dict(layer=layer,output_sha256=tensor_sha(actual)))
            rows.append(dict(name=name,start_pos=start,queries=queries,layers=signatures,
                             packed_indexer_also_enabled=True,actual_attention_forward=True))
        assert handle.counters["dual_calls"]==38*3
        assert original_windows==[obj.attn.window_kv_cache.data_ptr() for obj in opt_model.layers[2:]]
        counters=dict(handle.counters)
        handle.restore()
        assert not handle.active and packed.active
        # After restoration the original concat path must read the same live
        # window/compressed history, with the packed indexer still installed.
        x=(torch.randn(1,1,5120,generator=rng)*.3).to(torch.bfloat16)
        for layer in range(2,40):
            equal_output(opt_model.layers[layer].attn(x.clone(),6),ref_model.layers[layer].attn(x.clone(),6),"restored live history")
        rows.append(dict(name="restored_live_history",start_pos=6,queries=1,layers=38,
                         packed_indexer_also_enabled=True,dual_adapter_enabled=False))
        packed_counters=dict(packed.counters)
        assert packed_counters["dense_calls"]==4*8
        assert packed_counters["consumer_compact_calls"]==0
    finally:
        if handle.active:handle.restore()
        if packed.active:packed.restore(reset=True)
    return dict(rows=rows,dual_calls=counters["dual_calls"],packed_counters=packed_counters,world_size=world,
                collective="CPU repeated-local BF16 sum; no distributed backend",
                fixture_stubs=["synthetic low-rank Q/K/output projections", "identity RMS norms and RoPE frequencies",
                               "deterministic compressor latents", "identity FP8 window quantizer",
                               "identity E4M3 compressed quantizer; original Python wrapper executed",
                               "E8M0 index quantizer uses independent CPU oracle, not TileLang"],
                model_weight_equivalence=False)


def window_checks(core,kernel,model_path):
    import torch
    tree=ast.parse(model_path.read_bytes(),filename=str(model_path))
    assert sha(model_path)==MODEL_SHA
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_window_topk_idxs")
    ns=dict(torch=torch,lru_cache=functools.lru_cache)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(model_path),"exec",dont_inherit=True),ns)
    rows=[]
    rng=torch.Generator().manual_seed(533000)
    for start,queries in [(0,129),(127,1),(128,1),(129,1)]:
        width=queries if start==0 else 128
        window=torch.randn(1,width,512,generator=rng).to(torch.bfloat16)
        compressed=torch.randn(1,3,512,generator=rng).to(torch.bfloat16)
        q=(torch.randn(1,queries,8,512,generator=rng)*.1).to(torch.bfloat16)
        win_ids=ns["get_window_topk_idxs"](128,1,queries,start)
        comp_ids=torch.full((1,queries,1),width+2,dtype=torch.int32)
        ids=torch.cat((win_ids,comp_ids),dim=-1)
        sink=torch.zeros(8,dtype=torch.float32)
        expected=kernel.sparse_attn(q,torch.cat((window,compressed),1),sink,ids,512**-.5)
        actual=core.dual_sparse_attn(q,window,compressed,sink,ids,512**-.5)
        equal_output(actual,expected,"prefill/ringwrap")
        rows.append(dict(start_pos=start,queries=queries,actual_window_width=width,
                         compressed_offset=width,original_window_indices_sha256=tensor_sha(win_ids),
                         output_sha256=tensor_sha(actual)))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error("output exists; preserve prior receipt")
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    module_dir=ROOT/"overlay/modules/dsv41_model"
    sys.path.insert(0,str(module_dir))
    core=load(module_dir/"dsv41_dual_sparse.py","dsv41_dual_sparse")
    adapter=load(module_dir/"dsv41_dual_sparse_reference_adapter.py","dsv41_dual_sparse_reference_adapter")
    packed=load(module_dir/"dsv41_packed_reference_adapter.py","dsv41_packed_reference_adapter")
    quant_oracle=load(ROOT/"probes/dsv41_packed_index_diff.py","dsv41_prior_packed_oracle")
    kernel=reference_kernel(args.reference_dir/"kernel.py")
    result=dict(schema=1,passed=False,cpu_only=True,gpu_numerics=False,gpu_performance=False,
                v41_model_equivalence=False,model_sha256=MODEL_SHA,kernel_sha256=KERNEL_SHA,
                core_sha256=sha(module_dir/"dsv41_dual_sparse.py"),
                adapter_sha256=sha(module_dir/"dsv41_dual_sparse_reference_adapter.py"),
                probe_sha256=sha(__file__),
                arithmetic="official 64-slot staged FP32 recurrence with BF16 probabilities and output; CPU GEMM/exp stand-in",
                comparison="exact finite BF16 bits; exact NaN positions; no NaN payload claim")
    dependency_paths=[ROOT/"probes/dsv41_packed_index_diff.py"] + [module_dir/name for name in (
        "dsv41_packed_reference_adapter.py", "dsv41_packed_index.py", "dsv41_indexer.py",
        "dsv41_reference_adapter.py", "dsv41_dual_sparse_triton.py")]
    dependencies={str(path.relative_to(ROOT)):sha(path) for path in dependency_paths}
    result["dependencies_sha256"]=dependencies
    result["numerical"]=numerical_checks(core,kernel)
    result["window"]=window_checks(core,kernel,args.reference_dir/"model.py")
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        result["bf16_default_numerical"]=numerical_checks(core,kernel)
    finally:
        torch.set_default_dtype(previous_dtype)
    result["default_dtype_restored"]=torch.get_default_dtype()==previous_dtype
    result["adapter"]=adapter_checks(adapter,packed,kernel,args.reference_dir/"model.py",quant_oracle)
    assert not torch.cuda.is_initialized()
    assert dependencies=={str(path.relative_to(ROOT)):sha(path) for path in dependency_paths}
    result.update(passed=True,cuda_initialized=False,wrapper_calls=len(kernel.calls))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(passed=True,numerical_cases=len(result["numerical"]),window_cases=len(result["window"]),
                         bf16_default_cases=len(result["bf16_default_numerical"]),
                         adapter_steps=len(result["adapter"]["rows"]),output=str(args.output))))


if __name__=="__main__":main()
