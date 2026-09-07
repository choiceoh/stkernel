#!/usr/bin/env python3
"""Controlled heterogeneous CPU tests: cold round robin vs recorded durations.

Sleeps are deliberate fixture service times, not simulated production timings.
Both arms execute the same test IDs once with the same two-worker budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT/'bench/cpu_unittest.py'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=5)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if not 1 <= args.rounds <= 10:
        ap.error('rounds must be 1..10')
    rows = []
    with tempfile.TemporaryDirectory(prefix='fleet-sharding-comparison-') as directory:
        root = Path(directory); (root/'tests').mkdir()
        source = 'import time,unittest\nclass C(unittest.TestCase):\n'+''.join(
            f' def test_{i}(self): time.sleep({seconds}); self.assertEqual(2+2,4)\n'
            for i,seconds in enumerate((.6,.05,.55,.05)))
        (root/'tests/test_fleet_timing_comparison.py').write_text(source)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',FLEET_CPU_SLOTS='2',FLEET_EXPERIMENT_ROOT=str(root/'history'))
        output = root/'report.json'; history = root/'history/cpu-test-timings.json'
        def run():
            start = time.monotonic()
            p = subprocess.run([sys.executable,str(RUNNER),'tests/test_fleet*.py',str(output),
                                '--root',str(root),'--jobs','2'],env=env,text=True,capture_output=True,check=True)
            elapsed = time.monotonic()-start; report=json.loads(output.read_text())
            assert report['passed'] and report['coverage_complete'] and report['tests_run']==4, p.stdout+p.stderr
            return dict(wall_s=elapsed,**report)
        seed = run(); hints = history.read_bytes()
        for index in range(args.rounds):
            for variant in (('round-robin','duration-balanced') if index%2==0 else ('duration-balanced','round-robin')):
                if variant=='round-robin':history.unlink(missing_ok=True)
                else:history.write_bytes(hints)
                row = run()
                assert row['scheduling']==variant and set(row['test_ids'])==set(seed['test_ids'])
                row.update(round=index+1,variant=variant)
                rows.append(row)
                print(json.dumps({key:row[key] for key in ('round','variant','wall_s','tests_run')}),flush=True)
    result=dict(python=sys.version,platform=platform.platform(),rounds=args.rounds,
                runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                fixture_sha256=hashlib.sha256(source.encode()).hexdigest(),fixture_service_times_s=[.6,.05,.55,.05],
                workers=2,samples=rows,summary={variant:statistics.median(row['wall_s'] for row in rows if row['variant']==variant)
                    for variant in ('round-robin','duration-balanced')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
