"""Compare GB10 pooling against an exact pre-change kpool.py exported from git.

Measures the complete return-only lane and the arithmetic kernel separately.
Graph times average 100 launches per replay; all pairs alternate AB/BA.
Optional real-weight L3 comparisons change only the pooling lane.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from engine.kernels import kpool
from engine_decode_overhead import paired
from engine_pool_slots import capture
from engine_indexer_lanes import operators, real_indexer


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('pooling_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def previous(module, keys, scores, ape):
    # Exact adapter body in PR #546's lanes.served().kpool.
    pools = keys.shape[0]
    dummy = torch.zeros(1, 64, 132, device=keys.device, dtype=torch.uint8)
    result = module.kpool_compress_and_write_cache(dummy, keys, scores, ape,
        torch.arange(pools, device=keys.device, dtype=torch.int64), keys.shape[1],
        return_compressed=True, write_cache=False)
    quant = result[0].view(torch.uint8).reshape(pools, -1)[:, :128].contiguous().view(torch.float8_e4m3fn)
    return quant, result[1].reshape(pools, 1).float()


def raw(module, keys, scores, ape, outputs, warps):
    q, scale = outputs
    return module._kpool_softmax_rotate_write_cache_kernel[(keys.shape[0],)](
        q, scale, keys, scores, ape, q, q, q, scale,
        keys.stride(0), keys.stride(1), scores.stride(0), scores.stride(1), ape.stride(0),
        PAGE_SIZE=1, BUF_NUMEL_PER_PAGE=1, POOL_SIZE=keys.shape[1], HEAD_DIM=128,
        S_OFFSET_NBYTES_IN_PAGE=0, ROUND_SCALE=True, HAS_WRITE_MASK=False,
        RETURN_COMPRESSED=True, WRITE_CACHE=False, BLOCK_D=128, num_warps=warps,
        **({'WARP_LOCAL_ROTATION': True} if module is kpool else {}))


def graph_pair(functions):
    graphs = [capture(fn, 100) for fn in functions]
    times = paired([graph.replay for graph in graphs], rounds=5, samples=50, warmup=10)
    return dict(graph_100_calls=times, device_us_per_call={name: times[name]['cuda_stream']['median_us'] / 100
                                                        for name in ('baseline', 'optimized')})


def resource(kernel):
    return dict(warps=kernel.metadata.num_warps, registers_per_thread=kernel.n_regs,
                shared_bytes=kernel.metadata.shared, spills=kernel.n_spills)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-kpool', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--rank-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert importlib.util.find_spec('vllm') is None
    assert torch.cuda.get_device_capability() == (12, 1)
    baseline = load_baseline(args.baseline_kpool)
    old = lambda k, s, a: previous(baseline, k, s, a)
    report = dict(scope='kpool lane and optional real-weight L3 indexer, not full-model ITL',
        baseline_kpool_sha256=hashlib.sha256(args.baseline_kpool.read_bytes()).hexdigest(),
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), measurements=[], protocol=dict(rounds=5, eager_samples=100,
        graph_samples=50, launches_per_graph=100, order='alternating AB/BA'))
    if args.checkpoint:
        assert args.rank_file
        from engine.profiles.glm53 import lanes
        ref, fused = lanes.reference(), lanes.served()
        report['real_indexer'] = real_indexer(args.checkpoint, args.rank_file, ref, fused,
                                              baseline_lanes=replace(fused, kpool_compress=old))
        print('real indexer:', json.dumps(report['real_indexer']), flush=True)
    profiles = []
    generator = torch.Generator(device='cuda').manual_seed(270)
    for pools in (1, 2, 16, 64, 512, 4096):
        keys = torch.randn(pools, 4, 128, device='cuda', dtype=torch.bfloat16, generator=generator)
        scores = torch.randn_like(keys)
        ape = torch.randn(4, 128, device='cuda', generator=generator)
        functions = [lambda fn=fn, k=keys, s=scores, a=ape: fn(k, s, a) for fn in (old, kpool.compress_pool_keys)]
        outputs = [fn() for fn in functions]
        assert torch.equal(outputs[0][0].view(torch.uint8), outputs[1][0].view(torch.uint8))
        assert torch.equal(outputs[0][1], outputs[1][1])
        kernels = [lambda module=module, warps=warps, out=out, k=keys, s=scores, a=ape: raw(module,k,s,a,out,warps)
                   for module, warps, out in zip((baseline, kpool), (4, 1), outputs)]
        resources = [resource(fn()) for fn in kernels]
        assert torch.equal(outputs[0][0].view(torch.uint8), outputs[1][0].view(torch.uint8))
        assert torch.equal(outputs[0][1], outputs[1][1])
        item = dict(pools=pools, exact=True, resources=dict(zip(('baseline','optimized'), resources)),
                    eager=paired(functions, rounds=5, samples=100, warmup=40),
                    lane=graph_pair(functions), arithmetic=graph_pair(kernels))
        profiles.append((item,functions))
        report['measurements'].append(item)
        print('measurement:',json.dumps(item),flush=True)
    for item, functions in profiles:
        item['operators'] = {name:operators(fn) for name,fn in zip(('baseline','optimized'),functions)}
    assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
