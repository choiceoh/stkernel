"""Matched NumPy/native host planning; no device and no FFN latency claim."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import time
from unittest.mock import patch


def run(samples):
    import numpy as np
    from engine.modules.mixed_experts import ExpertInvocation
    from engine.modules.mixed_route_native import planner, prepare_native
    from engine.modules.mixed_route_plan import prepare_routes_numpy
    from engine.modules.route_table import RouteTable
    from probes.engine_mixed_experts_compile import fingerprint
    report = dict(status='PASS', scope=__doc__, source_revision=os.environ.get('TEST_REVISION'),
                  source_sha256=fingerprint(), samples=samples, cases=[], gpu_used=False)
    report['source_sha256']['probes/engine_mixed_native_plan_bench.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='mixed-host-build-') as cache:
        with patch.dict(os.environ, XDG_CACHE_HOME=cache):
            planner.cache_clear()
            start = time.perf_counter(); handle = planner()
            report['fresh_host_build_ms'] = (time.perf_counter()-start)*1000
            report['library_sha256'] = hashlib.sha256(Path(handle._library._name).read_bytes()).hexdigest()
    rng = np.random.default_rng(89552)
    for d, p in ((8, 9240), (32, 9240), (8, 32768), (32, 32768)):
        routes = np.argsort(rng.random((p, 288)), axis=1)[:, :8].astype(np.int32)
        cell = dict(decode_rows=d, prefill_rows=p, timings=[])
        for sample in range(-1, samples):
            values = []
            for arm in ('numpy', 'native') if sample % 2 == 0 else ('native', 'numpy'):
                fn = prepare_native if arm == 'native' else prepare_routes_numpy
                start = time.perf_counter()
                result = fn(routes[:d], routes, identity=ExpertInvocation(3, sample+1, 0, sample+1),
                            hot_route_quota=128, cold_task_quota=48)
                elapsed = (time.perf_counter()-start)*1000
                if sample >= 0:
                    cell['timings'].append(dict(arm=arm, sample=sample, plan_ms=elapsed))
                values.append(result)
            for left, right in zip(*values):
                for field in fields(left):
                    a, b = getattr(left, field.name), getattr(right, field.name)
                    if isinstance(a, RouteTable):
                        if a != b:
                            raise RuntimeError('native planner changed immutable descriptor bytes')
                    elif a != b:
                        raise RuntimeError('native planner changed descriptor '+field.name)
        cell['median_ms'] = {arm: median(x['plan_ms'] for x in cell['timings'] if x['arm']==arm)
                             for arm in ('numpy', 'native')}
        report['cases'].append(cell)
        print(json.dumps({k:v for k,v in cell.items() if k!='timings'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error('samples must be even and at least two')
    report = run(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
