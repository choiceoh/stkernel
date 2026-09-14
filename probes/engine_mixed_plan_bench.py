"""Matched CPU planning/metadata benchmark; no GPU or serving performance claim."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time
from types import SimpleNamespace


def legacy_plan(decode, prefill, **options):
    from engine.modules.mixed_experts import plan_experts
    return plan_experts(decode.tolist(), prefill.tolist(), **options)


def legacy_signature(owner):
    """The pre-optimization M2 JSON agreement, retained only as a timing arm."""
    plan, cold = owner.plan, owner.cold
    data = (asdict(plan.identity), plan.decode, plan.prefill, plan.sources,
            plan.experts, plan.tile_m, plan.quota, cold.task_quota,
            cold.task_expert, cold.task_valid_rows, cold.windows)
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def fingerprint():
    paths = ('engine/modules/route_table.py', 'engine/modules/mixed_experts.py',
        'engine/modules/mixed_completion.py', 'engine/modules/mixed_tickets.py',
        'tests/test_engine_mixed_plan.py', 'probes/engine_mixed_plan_bench.py')
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def run(samples):
    import numpy as np
    import torch
    from engine.modules.mixed_experts import ExpertInvocation, plan_experts_packed
    from engine.modules.mixed_completion import plan_cold
    from engine.modules.mixed_tickets import signature
    from engine.modules.route_table import RouteTable
    torch.set_num_threads(1)
    report = dict(status='PASS', scope=__doc__, platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, image=os.environ.get('ST_IMAGE'),
        source_revision=os.environ.get('TEST_REVISION'), source_sha256=fingerprint(),
        gpu_used=False, cuda_initialized=torch.cuda.is_initialized(), samples_per_arm=samples,
        timing='CPU arrays through fresh planning, descriptor hash and cold source CPU tensor copy; '
               'one unmeasured warmup per arm; alternate A/B and B/A; input generation/GC outside timing', cases=[])
    rng = np.random.default_rng(89520)
    for d, p in ((8, 9240), (32, 9240), (8, 32768), (32, 32768)):
        rows = np.argsort(rng.random((p, 288)), axis=1)[:, :8].astype(np.int32)
        decode = rows[:d].copy()
        for quota in (0, 128):
            cell = dict(decode_rows=d, prefill_rows=p, hot_quota=quota,
                input_sha256=hashlib.sha256(decode.tobytes()+rows.tobytes()).hexdigest(), samples=[])
            for sample in range(-1, samples):
                for arm in (('legacy', 'packed') if sample % 2 == 0 else ('packed', 'legacy')):
                    gc.collect()
                    identity = ExpertInvocation(3, sample+1, sample+1, sample+1)
                    planner, agree = (legacy_plan, legacy_signature) if arm == 'legacy' else (plan_experts_packed, signature)
                    start = time.perf_counter()
                    plan = planner(decode, rows, identity=identity, hot_route_quota=quota)
                    hot = time.perf_counter()
                    cold = plan_cold(plan)
                    planned = time.perf_counter()
                    agree(SimpleNamespace(plan=plan, cold=cold))
                    agreed = time.perf_counter()
                    sources = cold.sources.array() if isinstance(cold.sources, RouteTable) else cold.sources
                    tensor = torch.tensor(sources, dtype=torch.int32).reshape(-1, 4)
                    copied = time.perf_counter()
                    if sample >= 0:
                        cell['samples'].append(dict(arm=arm, sample=sample,
                            hot_plan_ms=(hot-start)*1000, cold_plan_ms=(planned-hot)*1000,
                            agreement_ms=(agreed-planned)*1000, tensor_copy_ms=(copied-agreed)*1000,
                            total_ms=(copied-start)*1000))
                    # Compare the complete upload outside timing, including the
                    # zero-padding positions, token/slot order and expert IDs.
                    upload_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                    if sample == -1:
                        expected = cell.setdefault('cold_source_sha256', upload_hash)
                        if upload_hash != expected:
                            raise RuntimeError('packed cold upload differs from the scalar reference')
                    del plan, cold, sources, tensor
            cell['median_ms'] = {arm: median(s['total_ms'] for s in cell['samples'] if s['arm'] == arm)
                                 for arm in ('legacy', 'packed')}
            cell['reduction_percent'] = 100 * (1 - cell['median_ms']['packed']/cell['median_ms']['legacy'])
            report['cases'].append(cell)
            print(json.dumps({k:v for k,v in cell.items() if k != 'samples'}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU planning benchmark initialized CUDA')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mixed-plan-cpu.json'))
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error('--samples must be even and at least two')
    report = run(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
