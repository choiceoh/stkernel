#!/usr/bin/env python3
"""Paired CPU result timing with controlled unavailable deployment attestation.

Uses committed temporary repositories and the existing fake fleet boundary.
No real deployment, SSH, Docker daemon or GPU hold is used.
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
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
import test_fleet_experiments as fixtures
import experiments as ex


def sample(source, delay):
    implementation = ModuleType('measured_plan')
    exec(compile(source, '<measured-plan>', 'exec'), implementation.__dict__)
    fixture = fixtures.SubmissionTests()
    fixture.setUp()
    store = ex.Store(fixture.jobs)
    try:
        (fixture.repo / 'tests').mkdir()
        (fixture.repo / 'tests/test_cpu.py').write_text(
            'import unittest\nclass C(unittest.TestCase):\n'
            ' def test_ok(self): self.assertEqual(sum(range(10000)),49995000)\n')
        fixture.refresh_deployed_fixture()
        raw = dict(hypothesis='controlled attestation delay', knobs={'VLLM_TEST':'1'},
                   context=fixture.pair_context(), objective={'metric':'quality'},
                   cpu_suites=[], cpu_tests=['tests/test_cpu.py'],
                   prepare=[dict(command=[sys.executable, '-c', 'print("prepared")'], requires=['checks'])])
        manifest = fixture.root / 'input.json'
        manifest.write_text(json.dumps(raw))
        args = SimpleNamespace(manifest=manifest, base=None, submit=True, prepare_only=False,
                               session='comparison', supersedes=[])
        original = ex.snapshot
        observation = {}

        def delayed(repo, spec, stamp):
            if spec['kind'] != 'pair':
                return original(repo, spec, stamp)
            saved = json.loads(next((store.root / 'plans').glob('*/plan.json')).read_text())
            cpu = [stage for stage in saved['stages'] if stage['name'] != 'gpu']
            observation['cpu_ids_visible_during_attestation'] = all('submission' in stage for stage in cpu)
            began = time.perf_counter()
            time.sleep(delay)  # Deliberate, identical external delay in both arms.
            observation['attestation_delay_s'] = time.perf_counter() - began
            raise ValueError('controlled unavailable deployment')

        with patch.dict(os.environ, fixture.env, clear=True), patch.object(ex, 'snapshot', side_effect=delayed):
            started = time.time()
            value = implementation.run(args, store, fixture.repo)
            observation['plan_return_s'] = time.time() - started
        assert value['error'] == 'controlled unavailable deployment', value
        ids = [stage['submission']['id'] for stage in value['stages'] if stage['name'] != 'gpu']
        for job in ids:
            result = fixture.wait(job)
            assert result['state'] == 'succeeded', result
            assert not result['result'].get('cache_source'), result
        rows = [store.get(job) for job in ids]
        observation.update(first_cpu_result_s=min(row['finished'] for row in rows)-started,
                           all_cpu_results_s=max(row['finished'] for row in rows)-started,
                           cpu_jobs=len(rows), cache_hits=0)
        assert not (fixture.logs / 'arms').exists()
        assert not (fixture.fleet / 'holder').exists()
        return observation
    finally:
        store.db.close()
        fixture.doCleanups()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='69ea76f6d6edf704d5d55a7f9bf366a0d3f4fe84')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10 or not 0 < args.delay <= 10:
        parser.error('rounds must be 1..10 and delay must be 0..10 seconds (exclusive zero)')
    base = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.baseline], text=True).strip()
    sources = dict(baseline=subprocess.check_output(['git', '-C', str(ROOT), 'show', base+':bench/experiment_plan.py'], text=True),
                   candidate=(ROOT / 'bench/experiment_plan.py').read_text())
    result = dict(baseline_revision=base, python=sys.version, platform=platform.platform(),
                  controlled_delay_s=args.delay, rounds=args.rounds,
                  plan_sha256={key:hashlib.sha256(value.encode()).hexdigest() for key,value in sources.items()},
                  samples=[], summary={})
    for index in range(args.rounds):
        for variant in (('baseline','candidate') if index % 2 == 0 else ('candidate','baseline')):
            row = dict(round=index+1, variant=variant, **sample(sources[variant],args.delay))
            result['samples'].append(row)
            print(json.dumps(row),flush=True)
    for variant in sources:
        rows = [row for row in result['samples'] if row['variant'] == variant]
        result['summary'][variant] = {key:dict(p50=statistics.median(row[key] for row in rows),
                                             minimum=min(row[key] for row in rows), maximum=max(row[key] for row in rows))
                                      for key in ('first_cpu_result_s','all_cpu_results_s','plan_return_s')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
