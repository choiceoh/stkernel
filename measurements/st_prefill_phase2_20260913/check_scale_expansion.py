"""Actual-weight eager prefill comparison, including scale reconstruction cost.

Run only inside a granted fleet GPU campaign. Kernel controls do not boot a
baseline engine and their event timings are not consumer TTFT measurements.
The independent byte map, established numerical bounds and route oracle are
retained; temporary expansion is intentionally forbidden inside CUDA capture.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.kernels.b12x import moe_dispatch as md
from engine.kernels.b12x.moe_sf6_prefill_scales import expand_scales
from check_long_q0 import oracle

CASES = ((2672, 'balanced'), (2675, 'zeros'),
         (8192, 'duplicate'), (32256, 'concentrated'))


def weight_identity(views, judge):
    return {name: judge._tensor_identity(value) for name, value in (
        ('w13', views.w13_tiled_storage), ('w2', views.w2_tiled_storage),
        ('fc1', views.reform_scales.fc1), ('fc2', views.reform_scales.fc2),
        ('alpha1', views.w1_alpha), ('alpha2', views.w2_alpha)) if value is not None}


def prepare_layer(loader, lanes, layer):
    prefix = f'L{layer}.moe.'
    suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    loaded = loader.load([prefix + k for k in suffixes])
    p = {k: loaded[prefix + k] for k in suffixes}
    scales = ModelOptScales.bind(*(p[k] for k in suffixes[4:]), experts=288, device=p['w13'].device)
    captured, raw = [], []
    get_views, pack_scales = md._get_weight_views, md._prepared_reform_scales

    def capture_views(*args, **kwargs):
        views = get_views(*args, **kwargs)
        captured.append(views)
        return views

    def capture_raw(source1, source2, raw1, raw2, **kwargs):
        # These are the original TMA bytes before SF6 compression. Retain
        # private copies only in this gate, never in the serving weight owner.
        raw.extend((raw1.view(torch.uint8).flatten().clone(),
                    raw2.view(torch.uint8).flatten().clone()))
        return pack_scales(source1, source2, raw1, raw2, **kwargs)

    with patch.object(md, '_get_weight_views', capture_views), \
            patch.object(md, '_prepared_reform_scales', capture_raw):
        lanes.moe_prepare(*(p[k] for k in suffixes[:4]), 8, 10., scales=scales)
    assert len(captured) == 1 and len(raw) == 2, 'missing original scale planes'
    return p, scales, captured[0], raw


def check_case(judge, views, scales, workspace, rows, kind, sink):
    device = views.w1_alpha.device
    generator = torch.Generator(device=device).manual_seed(judge.SEED)
    x = torch.randn(rows, 4096, dtype=torch.bfloat16, device=device, generator=generator) * .5
    ids, weights = judge._routing(torch, rows, kind, generator, device, False)
    first, second = scales.input13.float().clone(), scales.input2.float().clone()
    output = torch.empty_like(x)
    pointers = tuple(t.data_ptr() for t in (x, ids, weights, first, second, output))
    mapping = torch.arange(288, dtype=torch.int32, device=device)
    backing = dict(fc1_input=first, fc1_alpha=views.w1_alpha,
                   fc2_input=second, fc2_alpha=views.w2_alpha)

    def call(expansion):
        got = md.launch_sm120_dynamic_moe(
            workspace=workspace, weights=views, a=x, topk_ids=ids, topk_weights=weights,
            input_gs=first, down_input_scale=second, scatter_output=output,
            num_experts=288, num_tokens=rows, k=4096, n=512, top_k=8,
            activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
            _prefill_scale_expansion=expansion, _prefill_tile64=False)
        assert got is output and got.dtype == torch.bfloat16

    def eager(expansion, side=None):
        output.fill_(float('nan'))
        if side is None:
            call(expansion)
        else:
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                call(expansion)
                # Queue allocator pressure after the CuTe reader. Reused
                # temporary storage must stay ordered on the execution stream.
                pressure = torch.empty(113246208, dtype=torch.uint8, device=device)
                pressure.fill_(0xA5)
            torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        return output.clone()

    side = torch.cuda.Stream(device=device)
    sink.update(rows=rows, routes=kind, phases=[], status='RUNNING')
    for changed in (False, True):
        if changed:
            x.mul_(-.75)
            new_ids, new_weights = judge._routing(torch, rows, kind, generator, device, True)
            ids.copy_(new_ids)
            weights.copy_(new_weights)
            first[::2].mul_(.875)
            first[1::2].mul_(1.125)
        assert pointers == tuple(t.data_ptr() for t in (x, ids, weights, first, second, output))
        identity = judge._inputs(x, ids, weights, backing)
        b1 = eager(False)
        reference_q0 = judge._q0(torch, workspace, ids, weights)
        b2, b3 = eager(False), eager(False)
        phase = dict(changed=changed, controls=[judge.check_control(b1, b2),
                     judge.check_control(b1, b3), judge.check_control(b2, b3)], candidate=[])
        sink['phases'].append(phase)
        context = dict(result=sink, third=b3, inputs=x, route_ids=ids,
                       route_weights=weights, expert_map=mapping, scales=backing)
        for stream_name, expansion, stream in (('explicit-current', True, None),
                                               ('automatic-side', None, side)):
            candidate = eager(expansion, stream)
            phase['candidate'].append(dict(stream=stream_name,
                **judge.compare(candidate, b1, b2, failure_context=context)))
            got_q0 = judge._q0(torch, workspace, ids, weights)
            assert got_q0 == reference_q0, 'route IDs/weights/counts or sampled activation bytes changed'
            zero = (weights == 0).all(dim=1)
            if bool(zero.any()):
                assert bool((b1[zero] == 0).all()) and bool((candidate[zero] == 0).all())
        assert judge._inputs(x, ids, weights, backing) == identity, 'fixture inputs mutated'
        phase['route_sha256'] = hashlib.sha256(json.dumps(reference_q0, sort_keys=True).encode()).hexdigest()
        del b1, b2, b3, candidate
    # Full launcher cost, including two expansions/allocations and final cast.
    # Alternating controls keep first/last scheduling effects visible. This is
    # a bounded component decision, never a claim about served token rate.
    timings = {False: [], True: []}
    for expansion in (False, True, True, False) * 3:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call(expansion)
        end.record()
        end.synchronize()
        timings[expansion].append(start.elapsed_time(end))
    sink.update(status='PASS', packed_ms=timings[False], expanded_ms=timings[True],
                median_speedup=statistics.median(timings[False]) / statistics.median(timings[True]),
                graph_capture=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ranks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a fresh output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.get_device_capability() == (12, 1)
    report = dict(status='RUNNING', scope='actual rank0 layer 3; eager component numerics and event timing',
                  consumer_performance=False, layers=[], source_sha256={})
    for path in sorted((ROOT / 'engine/kernels/b12x').rglob('*.py')):
        report['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    started = time.monotonic()
    try:
        # The small fixtures exercise changed packed bytes on the same storage;
        # the real-weight checks below cover both complete 108 MiB scale planes.
        import unittest
        from tests.test_moe_prefill_scale_expansion import ScaleExpansionCudaTests
        result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ScaleExpansionCudaTests))
        assert result.wasSuccessful() and result.testsRun == 2 and not result.skipped
        report['cuda_byte_tests'] = 2
        judge, lanes = oracle(), served(moe_static='t,r,sf6,q0')
        loader = RankLoader(str(args.ranks))
        p, scales, views, raw = prepare_layer(loader, lanes, 3)
        assert views.packed_only and views.reform_scales.enabled, 'actual layer 3 must use SF6'
        before = weight_identity(views, judge)
        expanded = expand_scales(views.reform_scales, experts=288, hidden=4096, intermediate=512)
        assert sum(x.numel() for x in expanded) == 113246208
        assert all(torch.equal(a, b) for a, b in zip(expanded, raw)), 'actual original scale bytes differ'
        report['actual_scale_bytes'] = sum(x.numel() for x in raw)
        del raw, expanded
        workspace = md.allocate_sm120_dynamic_workspace(
            state_E=288, weight_E=288, routed_rows=32256*8, k=4096, n=512,
            num_topk=8, device=p['w13'].device, activation='swigluoai_uninterleave',
            quant_mode='nvfp4', tile_m=128)
        layer = dict(layer=3, cases=[])
        report['layers'].append(layer)
        cases = ((9216, 'balanced'), (32256, 'concentrated')) if args.long_only else CASES
        for rows, kind in cases:
            cell = {}
            layer['cases'].append(cell)
            check_case(judge, views, scales, workspace, rows, kind, cell)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(cell), flush=True)
        assert weight_identity(views, judge) == before, 'actual weights/scales mutated'
        report.update(status='PASS', actual_weights_preserved=True,
                      candidate_keys=[repr(k) for k in md._DYNAMIC_KERNEL_CACHE
                                      if k[-1] == 'temporary_prefill_raw_scales_v1'])
        assert len(report['candidate_keys']) == (1 if args.long_only else 2), 'requested readers must execute'
        del workspace
        gc.collect()
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('status', 'elapsed_s', 'scope')}), flush=True)


if __name__ == '__main__':
    main()
