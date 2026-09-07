#!/usr/bin/env python3
"""C=1 register scatter vs the same-build shared epilogue, in one CUDA context.

Run one variant per fresh container. Every arm is compared with the independent
stock backend before timing; input/routing mutations test captured state reuse.
The fixture is synthetic. This is not a serving quality or step-speed verdict.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    import moe_direct_scatter
    ap.add_argument('--variant', choices=moe_direct_scatter.VARIANTS, required=True)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--rounds', type=int, default=16)
    ap.add_argument('--out', default='/tmp/moe-c1-tiles.json')
    args = ap.parse_args()
    if args.rounds < 8:
        ap.error('at least eight balanced rounds required')
    os.environ['VLLM_GLM53_B12X_FORCE_BACKEND'] = 'static'
    os.environ['VLLM_GLM53_B12X_STATIC_V2'] = '0'
    sys.path.insert(0, os.environ.get('MK_PKG_PATH', '/usr/local/lib/python3.12/dist-packages'))
    import torch
    import moe_decode_stream_probe as fixture
    from megakernel_glm53_bench import _l2_flush
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    import moe_direct_scatter
    torch.manual_seed(53)
    assert md._GLM53_B12X_FORCE_BACKEND == 'static'
    assert md._GLM53_B12X_STATIC_V2 is None
    stock_classes = md.MoEStaticKernelV4, md.MoEStaticKernelV5
    stock_sources = md._kernel_source_files
    w13, sf13, w2, sf2 = fixture.expert_set(torch.Generator().manual_seed(53))
    scales = torch.ones(fixture.E, device='cuda')
    wrapper = fixture.served_wrapper()
    info = moe_direct_scatter.install(md, args.variant)
    candidate_classes = md.MoEStaticKernelV4, md.MoEStaticKernelV5
    candidate_sources = md._kernel_source_files
    graphs, gates, samples, kernels = {}, [], {}, []

    def select(arm):
        md.MoEStaticKernelV4, md.MoEStaticKernelV5 = (
            candidate_classes if arm == 'candidate' else stock_classes)
        md._kernel_source_files = candidate_sources if arm == 'candidate' else stock_sources
        md._STATIC_V2_KERNEL_CACHE.clear()
        if arm == 'stock':
            md._STATIC_V2_OVERRIDE = None
        else:
            cfg = md._parse_glm53_static_v2('t', probe=True)
            md._STATIC_V2_OVERRIDE = cfg

    # M=1/2 force the static path for correctness; timing targets M=6/8.
    for tokens, unique in ((1,8), (2,8), (2,16), (6,8), (6,40), (6,48), (8,8), (8,40), (8,64), (16,8), (16,40), (32,8), (32,40)):
        fixture.T = tokens
        ids, weights = fixture._routing(unique)
        x = torch.randn(tokens,4096,device='cuda',dtype=torch.bfloat16)*.5
        outputs = {arm:torch.empty_like(x) for arm in ('stock','baseline','candidate')}
        def run(arm):
            wrapper.run(x,w13,sf13,w2,sf2,ids,weights,w1_alpha=scales,
                        w2_alpha=scales,fc2_input_scale=scales,out=outputs[arm])
        current = {}
        for arm in outputs:
            select(arm)
            current[arm] = fixture._graph(lambda arm=arm:run(arm),torch.cuda.Stream())
            if arm != 'stock':
                assert md._STATIC_V2_KERNEL_CACHE, (arm,'static tile path not taken')
                # CUDA graph nodes retain raw function handles; keep the CuTe
                # executables alive when the next arm clears the dispatch cache.
                kernels.extend(md._STATIC_V2_KERNEL_CACHE.values())
        for replay in range(5):
            # Update addresses already captured, including routing. This catches
            # stale row/scale mappings as well as accidental constant outputs.
            x.copy_(torch.randn_like(x)*(.15+.15*replay))
            ids.add_(17).remainder_(fixture.E)
            if replay == 4:
                weights.zero_()
            current['stock'].replay(); torch.cuda.synchronize()
            reference = outputs['stock'].clone()
            current['stock'].replay(); torch.cuda.synchronize()
            noise = float((reference.float()-outputs['stock'].float()).abs().max())
            limit = max(4*noise,.01*float(reference.float().abs().max()))
            for arm in ('baseline','candidate'):
                current[arm].replay(); torch.cuda.synchronize()
                got = outputs[arm]
                assert torch.isfinite(got).all(), (tokens,unique,arm,replay,'nonfinite')
                diff = float((got.float()-reference.float()).abs().max())
                assert diff <= limit, (tokens,unique,arm,replay,diff,limit)
                if replay == 4:
                    assert torch.count_nonzero(got) == 0, (arm,'zero route weights')
                gates.append(dict(tokens=tokens,unique=unique,arm=arm,replay=replay,
                                  max_error=diff,stock_noise=noise,limit=limit))
        # Restore real nonzero routing before any timing.
        _, new_weights = fixture._routing(unique)
        weights.copy_(new_weights)
        for graph in current.values():
            for _ in range(3):
                graph.replay()
        torch.cuda.synchronize()
        if tokens in (6,8):
            name = f'M{tokens}-U{unique}'
            graphs[name] = (current,x,ids,weights,outputs)
        print(f'GATE M{tokens} U{unique}: stock differential, mutated graph and exact-zero PASS',flush=True)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for name,(current,x,ids,weights,outputs) in ({} if args.check_only else graphs).items():
        samples[name] = {state:{arm:[] for arm in ('baseline','candidate')} for state in ('cold','warm')}
        for state in ('cold','warm'):
            for repeat in range(args.rounds):
                order = ('baseline','candidate') if repeat % 2 == 0 else ('candidate','baseline')
                for arm in order:
                    if state == 'cold':
                        _l2_flush(hot=(x,ids,weights))
                    else:
                        for _ in range(3):
                            current[arm].replay()
                    start.record(); current[arm].replay(); end.record(); end.synchronize()
                    samples[name][state][arm].append(start.elapsed_time(end)*1000)
    medians = {name:{state:{arm:statistics.median(v) for arm,v in arms.items()}
                     for state,arms in states.items()} for name,states in samples.items()}
    report = dict(status='PASS',candidate=info,device=torch.cuda.get_device_name(),torch=torch.__version__,
                  gates=gates,samples_us=samples,median_us=medians,
                  scope='synthetic C=1 static MoE only; M1/2 correctness uses forced static')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print('JSON_RESULT '+json.dumps(report),flush=True)
    print('VERDICT: PASS (MoE GPU correctness/replay; serving unmeasured)',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
