"""Previous packed preparation choices and a fused-value-check device gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time


def torch_check_values(inputs, scales):
    import torch
    for value in inputs:
        if not bool(torch.isfinite(value).all()):
            raise ValueError('mixed source values must be finite')
    for value in scales:
        if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise ValueError('mixed experts require positive finite per-expert scales')


def previous_routes(decode, prefill, *, identity, hot_route_quota=128, cold_task_quota=None):
    from engine.modules.mixed_experts import plan_experts_packed
    from engine.modules.mixed_completion import plan_cold
    plan = plan_experts_packed(decode, prefill, identity=identity, hot_route_quota=hot_route_quota)
    return plan, None if cold_task_quota is None else plan_cold(plan, task_quota=cold_task_quota)


class BlockingMetadata:
    """Same tables/owner lifecycle, previous per-table blocking CUDA copies."""
    def __init__(self, plan, cold, device):
        from engine.modules.mixed_metadata import metadata_tables
        self.tables, self.device = metadata_tables(plan, cold), device

    def __getitem__(self, name):
        import torch
        return torch.tensor(self.tables[name], dtype=torch.int32, device=self.device)


def value_check_gate(device='cuda'):
    import torch
    from engine.kernels.mixed_checks import check_values
    checked = 0
    # Cover masked blocks, D > P, P > D, every source/scale plane, first/last
    # values, IEEE exceptional inputs and positive-scale boundaries.
    for d, p in ((9, 19), (32, 1)):
        inputs = (torch.zeros(d, 4096, dtype=torch.bfloat16, device=device),
            torch.zeros(p, 4096, dtype=torch.bfloat16, device=device),
            torch.ones(d, 8, dtype=torch.float32, device=device),
            torch.ones(p, 8, dtype=torch.float32, device=device))
        scales = tuple(torch.ones(288, dtype=torch.float32, device=device) for _ in range(4))
        def compare():
            nonlocal checked
            outcomes = []
            for fn in (torch_check_values, check_values):
                try:
                    fn(inputs, scales)
                    outcomes.append('valid')
                except ValueError as e:
                    outcomes.append(str(e))
            if outcomes[0] != outcomes[1]:
                raise RuntimeError(f'fused value contract differs: {outcomes}')
            checked += 1
        compare()
        for i, tensor in enumerate(inputs + scales):
            values = (float('nan'), float('inf'), -float('inf'), torch.finfo(tensor.dtype).max,
                      torch.finfo(tensor.dtype).tiny, 0., -0., -1.)
            for pos in (0, tensor.numel()-1):
                for value in values:
                    before = tensor.flatten()[pos].item()
                    tensor.flatten()[pos] = value
                    compare()
                    tensor.flatten()[pos] = before
        compare()
    return dict(status='PASS', cases=checked, source_and_scale_planes=8,
                decode_longer_than_prefill=True)


def run(samples):
    import numpy as np
    import torch
    from engine.modules.mixed_experts import ExpertInvocation
    from engine.modules.mixed_route_plan import prepare_routes
    torch.set_num_threads(1)
    files = ('engine/modules/mixed_experts.py', 'engine/modules/mixed_completion.py',
        'engine/modules/mixed_route_plan.py', 'engine/modules/mixed_metadata.py',
        'engine/modules/route_table.py', 'probes/engine_mixed_prepare_bench.py')
    root = Path(__file__).resolve().parents[1]
    report = dict(status='PASS', scope='CPU route planning only, fresh input snapshots per invocation; '
        'alternating previous packed two-stage and joint planning; not GPU or serving timing',
        source_revision=os.environ.get('TEST_REVISION'), image=os.environ.get('ST_IMAGE'),
        numpy=np.__version__, torch=torch.__version__, gpu_used=False,
        source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}, cases=[])
    rng = np.random.default_rng(89521)
    for d, p in ((8, 9240), (32, 9240), (8, 32768), (32, 32768)):
        rows = np.argsort(rng.random((p, 288)), axis=1)[:, :8].astype(np.int32)
        cell = dict(decode_rows=d, prefill_rows=p, hot_quota=128, samples=[])
        for sample in range(-1, samples):
            for arm in ('packed_v1', 'packed_v2') if sample % 2 == 0 else ('packed_v2', 'packed_v1'):
                fn = previous_routes if arm == 'packed_v1' else prepare_routes
                start = time.perf_counter()
                plan, cold = fn(rows[:d], rows, identity=ExpertInvocation(3, sample+1, sample+1, sample+1),
                    cold_task_quota=48)
                elapsed = (time.perf_counter()-start)*1000
                if sample >= 0:
                    cell['samples'].append(dict(arm=arm, sample=sample, plan_ms=elapsed))
                if sample == -1:
                    digest = hashlib.sha256(cold.sources.data).hexdigest()
                    if cell.setdefault('cold_sha256', digest) != digest:
                        raise RuntimeError('joint source layout differs from previous packed planning')
                del plan, cold
        cell['median_ms'] = {arm: median(s['plan_ms'] for s in cell['samples'] if s['arm']==arm)
                            for arm in ('packed_v1', 'packed_v2')}
        report['cases'].append(cell)
        print(json.dumps({k:v for k,v in cell.items() if k!='samples'}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU benchmark initialized CUDA')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mixed-prepare-cpu.json'))
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error('--samples must be even and at least two')
    result = run(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
