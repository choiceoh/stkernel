#!/usr/bin/env python3
"""Same-binary AR -> exact MHC -> W4 GEMM ordering and segment timing.

--compile-only never initializes CUDA. --distributed uses the real four-node
OSAR protocol; the local mode uses a delayed input producer, not a network
performance model. Only a serving bracket decides engine-step speed.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'overlay/modules'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--distributed', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('/evidence/result.json'))
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()
    os.environ.update(MAX_JOBS='1', VLLM_GLM53_MK_PDL='1',
        VLLM_GLM53_MK_MHC_BF16='1', VLLM_GLM53_MEGAKERNEL='1',
        VLLM_GLM53_MK_MHC='1', VLLM_GLM53_MK_GEMM='1',
        VLLM_GLM53_MK_FP8_PACK2='1', VLLM_GLM53_MK_GEMM_TRANSPOSE_M8='2',
        VLLM_GLM53_MK_M8_FASTPATH='1', VLLM_GLM53_MK_INPUT_REUSE='1',
        VLLM_GLM53_MK_INPUT_CTA='4')
    import torch
    from torch.utils.cpp_extension import load
    mk = module('ar_probe_mk', MOD / 'glm53_megakernel/glm53_megakernel.py')
    shim = module('ar_probe_osar', MOD / 'tp_oneshot_ar/dsv4_oneshot_shim.py')
    ext = mk._build()
    ar = shim._build()
    delay_path = ROOT / 'probes/ar_consumer_delay.cu'
    build = Path(os.environ.get('AR_CONSUMER_BUILD', '/build')) / 'delay'
    build.mkdir(parents=True, exist_ok=True)
    delay = load(name='ar_consumer_delay', sources=[str(delay_path)],
        extra_cuda_cflags=['-O2', '-arch=sm_121a'], build_directory=str(build), verbose=False)
    receipt = {'status': 'RUNNING', 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'mode': 'distributed' if args.distributed else 'delayed-producer',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(mk._SRC), Path(shim._SRC), delay_path)}, 'cases': [], 'samples': []}
    if args.compile_only:
        assert hasattr(ar, 'oneshot_ar_consumer')
        receipt.update(status='PASS', mode='compile-only')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + '\n')
        print('PASS compile-only: production MK and OSAR bindings plus delayed producer', flush=True)
        return
    torch.cuda.set_device(0)
    assert ext.probe_device()[:3] == [12, 1, 48] or tuple(ext.probe_device()[:3]) == (12, 1, 48)
    receipt['device'] = torch.cuda.get_device_name()
    mk._AR_NOTE = False
    mk._MHC_BF16_OK = True
    mk._ensure_workspace('cuda')
    rank = int(os.environ.get('AR_CONSUMER_RANK', '0'))
    if args.distributed:
        import torch.distributed as dist
        ips = os.environ['AR_CONSUMER_IPS'].split(',')
        assert len(ips) == 4
        dist.init_process_group('gloo', init_method=os.environ['AR_CONSUMER_INIT'],
                                rank=rank, world_size=4, timeout=timedelta(seconds=180))
        ar.init(rank, 4, ips[rank])
        infos = [None] * 4
        dist.all_gather_object(infos, ar.local_infos())
        ar.connect(infos)
        dist.barrier()
    receipt['rank'] = rank
    torch.manual_seed(53)
    packs = {n: mk.build_mk_weight_w4(torch.randn(n, 4096, device='cuda',
                   dtype=torch.bfloat16) * .02) for n in (4096, 6144, 6416)}
    from mhc_reuse_bench import mathematical_reference
    from megakernel_glm53_bench import _l2_flush
    _l2_flush()
    torch.cuda.synchronize()
    fn = (torch.randn(24, 16384, device='cuda') * .02).bfloat16().float()
    assert mk._mhc_bf16_weight(fn) is not None
    fixed = (fn, torch.tensor([.6, .9, .3], device='cuda'),
             torch.randn(24, device='cuda') * .1,
             torch.randn(4096, device='cuda', dtype=torch.bfloat16))

    def inputs(t):
        return (torch.randn(t, 4096, device='cuda', dtype=torch.bfloat16) * .1,
                torch.randn(t, 4, 4096, device='cuda', dtype=torch.bfloat16) * .1,
                torch.rand(t, 4, device='cuda'), torch.rand(t, 16, device='cuda'))

    def segment(values, early, fp32, n, cycles=64000):
        x, res, pm, cm = values
        if args.distributed:
            reduced = (ar.oneshot_ar_consumer(x) if early and x.numel() <= 32768
                       else ar.oneshot_ar(x))
        else:
            reduced = torch.empty_like(x)
            delay.produce(x, reduced, cycles, early)
        outs = mk._mhc_call(reduced, res, pm, cm, *fixed, x.shape[0],
                           1e-6, 1e-6, 1e-6, 1., 1e-6, 20,
                           _fp32_fn=fp32, _ar_consumer=early)
        gemm = mk._gemm_call(outs[-1], packs[n], n)
        return (reduced, *outs, gemm)

    graphs = {}
    for t in (1, 2, 6, 8, 16, 32):
        vals = inputs(t)
        for fp32 in (True, False):
            for early in (False, True):
                segment(vals, early, fp32, 4096)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    got = segment(vals, early, fp32, 4096)
                graphs[early] = (g, got)
            for seed in (17, 0, 29):
                # Values change behind fixed graph pointers. Integer/32
                # sums are exact in BF16 and make an independent AR oracle.
                torch.manual_seed(seed)
                if seed:
                    vals[0].copy_(torch.randint(-8, 9, vals[0].shape, device='cuda').bfloat16() / 32)
                    vals[0].add_(rank / 32 if args.distributed else 0)
                    vals[1].normal_(0, .1)
                    vals[2].uniform_(); vals[3].uniform_()
                else:
                    vals[0].zero_(); vals[1].zero_(); vals[2].zero_(); vals[3].zero_()
                torch.cuda.synchronize()
                expected = vals[0].clone()
                if args.distributed:
                    host = vals[0].cpu()
                    dist.all_reduce(host)
                    expected.copy_(host)
                    dist.barrier()
                graphs[False][0].replay()
                torch.cuda.synchronize()
                base = tuple(x.clone() for x in graphs[False][1])
                graphs[True][0].replay()
                torch.cuda.synchronize()
                cand = graphs[True][1]
                assert torch.equal(cand[0], expected), ('AR oracle', t, seed, rank)
                assert all(torch.equal(a, b) for a, b in zip(base, cand)), ('exact segment', t, fp32, seed, rank)
                if seed == 17 and not fp32:
                    oracle = mathematical_reference((expected, *vals[1:], *fixed), torch)
                    errors = [mk._rel_err(a, b) for a, b in zip(cand[1:5], oracle)]
                    assert max(errors) <= mk._TOL_MHC, ('independent MHC oracle', errors)
                receipt['cases'].append({'tokens': t, 'fp32_fn': fp32, 'seed': seed,
                                         'exact_outputs': 6, 'pass': True})
        print('PASS graph input updates and independent oracle T=' + str(t), flush=True)

    if not args.check_only:
        vals = inputs(6)
        for n in (4096, 6144, 6416):
            for cold in (False, True):
                timed = {}
                for early in (False, True):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        if cold: _l2_flush(hot=vals)
                        start.record()
                        segment(vals, early, False, n)
                        end.record()
                    timed[early] = (g, start, end)
                for rep in range(12):
                    for early in ((False, True) if rep % 2 == 0 else (True, False)):
                        if args.distributed: dist.barrier()
                        g, start, end = timed[early]
                        g.replay(); torch.cuda.synchronize()
                        receipt['samples'].append({'n': n, 'cold': cold, 'early': early,
                                                   'rep': rep, 'us': start.elapsed_time(end) * 1000})
                stats = {str(e): statistics.median(s['us'] for s in receipt['samples']
                    if s['n'] == n and s['cold'] == cold and s['early'] == e) for e in (False, True)}
                print(json.dumps({'n': n, 'cold': cold, 'median_us': stats}), flush=True)
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for early in (False, True):
                    timed[early][0].replay(); torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.out.with_suffix('.trace.json')))
    if args.distributed:
        dist.barrier()
        ar.shutdown()
        dist.destroy_process_group()
    receipt['status'] = 'PASS'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print('PASS all exact segment, graph and independent oracle checks', flush=True)


if __name__ == '__main__':
    main()
