"""Actual-weight M128/M64 eager component gate; run only with a GPU grant."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served
from engine.kernels.b12x import moe_dispatch as md
from check_long_q0 import oracle
from check_scale_expansion import prepare_layer, weight_identity

CASES = ((2672, 'balanced'), (2675, 'zeros'), (2121, 'concentrated'),
         (8192, 'duplicate'), (65, 'balanced'))


def route_samples(counts, bases, tokens, weights, input_ids, input_weights):
    """Check all original routes against the independently read M64 table."""
    bits = lambda value: struct.pack('<f', value).hex()
    rows = len(input_ids)
    expected = Counter((expert, token, bits(weight))
        for token, (ids, values) in enumerate(zip(input_ids, input_weights))
        for expert, weight in zip(ids, values))
    assert len(counts) == 288 and len(bases) == 289 and len(expected)
    observed, sample, tile = Counter(), [], 0
    selected = {0, rows // 2, rows - 1}
    for expert, count in enumerate(counts):
        assert 0 <= count <= rows * 8 and bases[expert] == tile
        tile += (count + 63) // 64
        for physical in range(bases[expert] * 64, bases[expert] * 64 + count):
            assert physical < len(tokens) and physical < len(weights)
            token, weight = tokens[physical], weights[physical]
            assert 0 <= token < rows
            record = expert, token, bits(weight)
            observed[record] += 1
            if token in selected:
                sample.append((record, physical))
    assert bases[-1] == tile and observed == expected and sum(counts) == rows * 8
    return sorted(sample)


def q0_m64(judge, workspace, ids, weights):
    counts = workspace.row_counts.cpu().tolist()
    samples = route_samples(counts, workspace.expert_tile_base.cpu().tolist(),
        workspace.token_map.cpu().tolist(), workspace.token_weights.cpu().tolist(),
        ids.cpu().tolist(), weights.cpu().tolist())
    packed, sf = workspace.packed_input.reshape(-1, 2048), workspace.scale_flat
    records = []
    for key, physical in samples:
        atom, row = divmod(physical, 64)
        base = atom * 32768 + (row % 32) * 16 + (row // 32) * 4
        offsets = [base + (block // 4) * 512 + block % 4 for block in range(256)]
        assert max(offsets) < sf.numel()
        index = torch.tensor(offsets, device=sf.device, dtype=torch.int64)
        records.append((*key, judge._tensor_identity(packed[physical])['sha256'],
                        judge._tensor_identity(sf.index_select(0, index))['sha256']))
    return dict(row_counts=counts, routes=sum(counts), payload_sample=sorted(records))


def check_case(judge, views, scales, workspaces, rows, kind, result):
    device = views.w1_alpha.device
    generator = torch.Generator(device=device).manual_seed(judge.SEED)
    x = torch.randn(rows, 4096, dtype=torch.bfloat16, device=device, generator=generator) * .5
    ids, weights = judge._routing(torch, rows, kind, generator, device, False)
    first, second, out = scales.input13.float().clone(), scales.input2.float().clone(), torch.empty_like(x)
    pointers = tuple(t.data_ptr() for t in (x, ids, weights, first, second, out))
    backing = dict(fc1_input=first, fc1_alpha=views.w1_alpha,
                   fc2_input=second, fc2_alpha=views.w2_alpha)
    mapping, side = torch.arange(288, dtype=torch.int32, device=device), torch.cuda.Stream(device=device)

    def call(m64, automatic=False):
        got = md.launch_sm120_dynamic_moe(workspace=workspaces[False if automatic else m64], weights=views,
            a=x, topk_ids=ids, topk_weights=weights, input_gs=first, down_input_scale=second,
            scatter_output=out, num_experts=288, num_tokens=rows, k=4096, n=512, top_k=8,
            activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
            _prefill_scale_expansion=None if automatic else False,
            _prefill_tile64=None if automatic else m64)
        assert got is out and out.dtype == torch.bfloat16

    def eager(m64, alternate=False):
        out.fill_(float('nan'))
        if alternate:
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                call(m64)
            torch.cuda.current_stream(device).wait_stream(side)
        else:
            call(m64)
        torch.cuda.synchronize(device)
        return out.clone()

    result.update(rows=rows, routes=kind, phases=[], status='RUNNING')
    for changed in (False, True):
        if changed:
            x.mul_(-.75)
            new_ids, new_weights = judge._routing(torch, rows, kind, generator, device, True)
            ids.copy_(new_ids); weights.copy_(new_weights)
            first[::2].mul_(.875); first[1::2].mul_(1.125)
        identity = judge._inputs(x, ids, weights, backing)
        a, b, c = eager(False), eager(False), eager(False)
        expected = judge._q0(torch, workspaces[False], ids, weights)
        expected = {key: expected[key] for key in ('row_counts', 'routes', 'payload_sample')}
        phase = dict(changed=changed, controls=[judge.check_control(a, b), judge.check_control(a, c),
                     judge.check_control(b, c)], q0_reference=expected, candidate=[])
        result['phases'].append(phase)
        context = dict(result=result, third=c, inputs=x, route_ids=ids, route_weights=weights,
                       expert_map=mapping, scales=backing)
        for alternate in (False, True):
            candidate = eager(True, alternate)
            observation = dict(alternate_stream=alternate)
            phase['candidate'].append(observation)
            actual_workspace = workspaces[True]
            try:
                observation['q0'] = q0_m64(judge, actual_workspace, ids, weights)
                observation['q0_matches'] = observation['q0'] == expected
                assert observation['q0_matches'], 'M64 route/Q0 payload differs from the M128 control'
            except BaseException as error:
                observation['q0_error'] = repr(error)
                raise
            observation.update(judge.compare(candidate, a, b, failure_context=context))
            zero = (weights == 0).all(dim=1)
            if bool(zero.any()):
                assert bool((a[zero] == 0).all()) and bool((candidate[zero] == 0).all())
            assert bool(torch.isfinite(candidate).all())
            assert bool((a[~zero] != 0).any()), 'degenerate all-zero numerical fixture'
        assert judge._inputs(x, ids, weights, backing) == identity
        assert pointers == tuple(t.data_ptr() for t in (x, ids, weights, first, second, out))
    timings = {False: [], True: []}
    for m64 in (False, True, True, False) * 3:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); call(m64); end.record(); end.synchronize()
        timings[m64].append(start.elapsed_time(end))
    result.update(status='PASS', m128_ms=timings[False], m64_ms=timings[True],
        median_speedup=statistics.median(timings[False]) / statistics.median(timings[True]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ranks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.get_device_capability() == (12, 1)
    started = time.monotonic()
    report = dict(status='RUNNING', consumer_performance=False, cases=[],
                  scope='actual rank0 layer3; M64 component numerics, routing and timing')
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / 'engine/kernels/b12x').rglob('*.py'))}
    try:
        judge, lanes = oracle(), served(moe_static='t,r,sf6,q0')
        params, scales, views, raw = prepare_layer(RankLoader(str(args.ranks)), lanes, 3)
        del raw
        assert views.packed_only and views.reform_scales.enabled
        identity = weight_identity(views, judge)
        workspaces = {m64: md.allocate_sm120_dynamic_workspace(state_E=288, weight_E=288,
            routed_rows=8192 * 8, k=4096, n=512, num_topk=8, device=params['w13'].device,
            activation='swigluoai_uninterleave', quant_mode='nvfp4', tile_m=64 if m64 else 128,
            _prefill_tile64=m64) for m64 in (False, True)}
        for rows, kind in CASES:
            case = {}
            report['cases'].append(case)
            check_case(judge, views, scales, workspaces, rows, kind, case)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(case), flush=True)
        assert weight_identity(views, judge) == identity
        report.update(status='PASS', actual_weights_preserved=True,
            candidate_keys=[repr(k) for k in md._DYNAMIC_KERNEL_CACHE if k[-1] == 'private_prefill_m64_fp32_v1'])
        assert report['candidate_keys'], 'no private M64 compiled path executed'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        if 'identity' in locals():
            try:
                report['actual_weights_preserved'] = weight_identity(views, judge) == identity
            except BaseException as error:
                report['weight_identity_error'] = repr(error)
        report['elapsed_s'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
