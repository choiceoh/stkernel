"""Dense/sparse NVFP4 qualification at GLM TP=4 expert projection shapes.

Uses prepacked operands, FP32 accumulation and BF16 outputs. Measures one
batched projection, not a fused MoE or end-to-end serving throughput. Each
batch is a different expert; tokens is the number of rows per active expert.
Build engine_sparse_nvfp4.cu with the installed CUTLASS headers (see report).
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics

import torch


def validate_sparse(packed):
    """Native NVFP4 sparsity is two retained adjacent pairs out of four."""
    if packed.dtype != torch.uint8 or packed.shape[-1] % 4:
        raise ValueError('expected packed uint8 FP4 with K divisible by 8')
    active = (packed & 0x77).ne(0).reshape(*packed.shape[:-1], -1, 4)
    if (active.sum(-1) > 2).any().item():
        raise ValueError('weight violates pairwise 4:8 sparse NVFP4 pattern')


def dequant(packed, scales):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    return table[codes.long()] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(packed.shape[-1]*2//scales.shape[-1], -1)


def synthetic(experts, rows, k, *, sparse, seed):
    """All six legal pair masks, signs, FP4 codes and varying positive scales."""
    generator = torch.Generator(device='cuda').manual_seed(seed)
    packed = torch.randint(0, 256, (experts, rows, k//2), dtype=torch.uint8,
                           device='cuda', generator=generator)
    if sparse:
        masks = torch.tensor([[1,1,0,0], [1,0,1,0], [1,0,0,1],
                              [0,1,1,0], [0,1,0,1], [0,0,1,1]],
                             dtype=torch.uint8, device='cuda')
        indices = torch.arange(experts*rows*k//8, device='cuda') % 6
        packed.mul_(masks[indices].reshape_as(packed))
    # E4M3 raw bytes 32..64 represent finite, nonzero positive values.
    scales = torch.randint(32, 65, (experts, rows, k//32), device='cuda',
                           dtype=torch.uint8, generator=generator)
    return packed, scales


class Library:
    def __init__(self, path):
        self.lib = ct.CDLL(str(Path(path).resolve()))
        v, i = ct.c_void_p, ct.c_int
        self.lib.sparse_probe_error.restype = ct.c_char_p
        self.lib.sparse_probe_sizes.argtypes = [i]*4 + [ct.POINTER(ct.c_int64)]
        self.lib.sparse_probe_prepare.argtypes = [i]*4 + [v]*10
        self.lib.sparse_probe_create.argtypes = [i]*5 + [v]*7
        self.lib.sparse_probe_create.restype = v
        self.lib.sparse_probe_run.argtypes = [v, v]
        self.lib.sparse_probe_destroy.argtypes = [v]
        self.lib.sparse_probe_destroy.restype = None

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.sparse_probe_error().decode())


class Projection:
    def __init__(self, library, weight, x, weight_sf, x_sf):
        validate_sparse(weight)
        self.library = library
        self.shape = (weight.shape[1], x.shape[1], weight.shape[2]*2, weight.shape[0])
        m, n, k, batches = self.shape
        for tensor, expected in [(weight, (batches,m,k//2)), (x, (batches,n,k//2)),
                                 (weight_sf,(batches,m,k//32)), (x_sf,(batches,n,k//32))]:
            if tensor.dtype != torch.uint8 or not tensor.is_cuda or not tensor.is_contiguous() or tuple(tensor.shape) != expected:
                raise ValueError(f'expected contiguous CUDA uint8 {expected}')
        for sf in (weight_sf, x_sf):
            if (sf > 126).any().item():
                raise ValueError('scales must be nonnegative finite E4M3')
        self.inputs = (weight, x, weight_sf, x_sf)
        sizes = (ct.c_int64*6)()
        library.check(library.lib.sparse_probe_sizes(*self.shape, sizes))
        self.buffers = [torch.zeros(size, dtype=torch.uint8, device='cuda') for size in sizes]
        comp, metadata, sf_a, sf_b, dense_sf_a, dense_sf_b = self.buffers
        self.bytes = dict(dense_weight=weight.numel(), sparse_weight=comp.numel(),
                          metadata=metadata.numel(), sparse_weight_scales=sf_a.numel(), dense_weight_scales=dense_sf_a.numel(),
                          activation=x.numel(), sparse_activation_scales=sf_b.numel(), dense_activation_scales=dense_sf_b.numel())
        self.outputs = [torch.empty((batches,n,m), dtype=torch.bfloat16, device='cuda') for _ in range(2)]
        stream = torch.cuda.current_stream().cuda_stream
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        library.check(library.lib.sparse_probe_prepare(*self.shape, *[t.data_ptr() for t in
            (weight,comp,metadata,weight_sf,x_sf,sf_a,sf_b,dense_sf_a,dense_sf_b)], stream))
        end.record(); end.synchronize()
        self.prepare_us = start.elapsed_time(end)*1000
        self.contexts = []
        try:
            for variant, a in enumerate((weight, comp)):
                sa,sb = (sf_a,sf_b) if variant else (dense_sf_a,dense_sf_b)
                context = library.lib.sparse_probe_create(variant, *self.shape,
                    *[t.data_ptr() for t in (a,x,sa,sb,metadata,self.outputs[variant])], stream)
                if not context:
                    raise RuntimeError(library.lib.sparse_probe_error().decode())
                self.contexts.append(context)
        except Exception:
            self.close()
            raise

    def run(self, variant):
        self.library.check(self.library.lib.sparse_probe_run(
            self.contexts[variant], torch.cuda.current_stream().cuda_stream))
        return self.outputs[variant]

    def close(self):
        for context in self.contexts:
            self.library.lib.sparse_probe_destroy(context)
        self.contexts.clear()


def error(actual, reference):
    a, b = actual.float(), reference.float()
    assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    d = a-b
    return dict(relative_max=(d.abs().max()/b.abs().max().clamp_min(1e-20)).item(),
                relative_l2=(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(b).clamp_min(1e-20)).item(),
                bits_equal=torch.equal(actual.view(torch.uint8),reference.view(torch.uint8)))


def qualify(projection):
    w,x,ws,xs = projection.inputs
    reference = torch.bmm(dequant(x,xs), dequant(w,ws).transpose(1,2)).bfloat16()
    rows = []
    for variant in (0,1):
        output = projection.run(variant)
        result = error(output,reference)
        assert result['relative_l2'] <= .003 and result['relative_max'] <= .008, result
        rows.append(result)
    comparison = error(projection.outputs[1],projection.outputs[0])
    assert comparison['relative_l2'] <= .003, comparison
    return dict(reference=rows, sparse_vs_dense=comparison)


def paired(samples):
    cycles = [[statistics.mean(s[i:i+2]) for i in range(0,len(s),2)] for s in samples]
    return dict(cycle_means_us=cycles, median_us=[statistics.median(s) for s in cycles],
                speedup=statistics.median(a/b for a,b in zip(*cycles)),
                reduction_percent=statistics.median(100*(a-b)/a for a,b in zip(*cycles)))


def benchmark(projection, rounds=12):
    if rounds < 2 or rounds % 2:
        raise ValueError('timing requires an even number of AB/BA rounds >= 2')
    trash = torch.empty(64*1024**2, dtype=torch.uint8, device='cuda')
    graphs = {}
    samples = {regime:[[],[]] for regime in ('warm','evicted')}
    try:
        for variant in (0,1):
            expected = projection.run(variant).clone()
            for count in (1,16):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count):
                        projection.run(variant)
                graphs[variant,count] = graph
                projection.outputs[variant].fill_(float('nan'))
                graph.replay()
                assert torch.equal(projection.outputs[variant], expected), 'graph failed to overwrite outputs'
        for r in range(rounds):
            for regime,count in (('warm',16),('evicted',1)):
                for variant in ((0,1) if r%2 == 0 else (1,0)):
                    graph = graphs[variant,count]
                    if regime == 'warm':
                        graph.replay()
                    else:
                        trash.zero_()
                    start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record(); graph.replay(); end.record(); end.synchronize()
                    samples[regime][variant].append(start.elapsed_time(end)*1000/count)
        return dict(samples_us=samples, paired={k:paired(v) for k,v in samples.items()})
    finally:
        for graph in graphs.values():
            graph.reset()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, nargs='+', default=[1,6,32,128])
    ap.add_argument('--experts', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=12)
    args = ap.parse_args()
    if any(n <= 0 or n > 512 for n in args.tokens) or not 1 <= args.experts <= 16:
        ap.error('bounded probe requires tokens 1..512, experts 1..16')
    if torch.cuda.get_device_capability() != (12,1):
        raise RuntimeError('this probe targets GB10 SM121a')
    torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    library = Library(args.library)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, tile=[128,128,256], logical_scale_group=32,
                  scope='batched standalone projections; synthetic prequantized inputs; no routing, activation quantization or TP communication',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__),Path(__file__).with_suffix('.cu'))},
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), cases=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name,m,k in (('w13',1024,4096),('w2',4096,512)):
        for n in args.tokens:
            w,ws = synthetic(args.experts,m,k,sparse=True,seed=1523)
            x,xs = synthetic(args.experts,n,k,sparse=False,seed=9153+n)
            projection = Projection(library,w,x,ws,xs)
            try:
                row = dict(projection=name, tokens_per_expert=n, active_experts=args.experts,
                           shape=list(projection.shape), bytes=projection.bytes,
                           prepare_us=projection.prepare_us, correctness=qualify(projection),
                           timing=benchmark(projection,args.rounds))
                result['cases'].append(row)
                result['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
                args.out.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='timing'} |
                                 {'paired':row['timing']['paired']}),flush=True)
            finally:
                projection.close()


if __name__ == '__main__':
    main()
