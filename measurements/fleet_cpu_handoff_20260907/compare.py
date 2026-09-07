#!/usr/bin/env python3
"""Measure real CPU DAG completion using isolated fake-fleet repositories."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
import test_fleet_experiments as fixtures
import experiments as ex


def sample(source, depth):
    fixture = fixtures.SubmissionTests()
    fixture.setUp()
    store = ex.Store(fixture.jobs)
    try:
        # Only the worker implementation differs; all other fixture code,
        # checks, artifact commands, and CPU budgets are identical in both arms.
        (fixture.repo / 'bench/experiments.py').write_text(source)
        def cli(*args):
            # Submission and execution must attest the same runner bytes.
            process = subprocess.run([sys.executable,str(fixture.repo / 'bench/experiments.py'),*args],
                                     env=fixture.env,text=True,capture_output=True,timeout=30)
            assert process.returncode == 0, process.stdout+process.stderr
            return json.loads(process.stdout)
        fixture.cli = cli
        (fixture.repo / 'tests').mkdir()
        (fixture.repo / 'tests/test_cpu.py').write_text(
            'import unittest\nclass C(unittest.TestCase):\n'
            ' def test_ok(self): self.assertEqual(sum(range(10000)),49995000)\n')
        fixture.refresh_deployed_fixture()
        stages = []
        for index in range(1, depth+1):
            code = 'from pathlib import Path;Path("build").mkdir(exist_ok=True);'
            if index > 1:
                code += f'assert Path("build/value{index-1}").read_text()=="{index-1}";'
            code += f'Path("build/value{index}").write_text("{index}")'
            stages.append(dict(command=[sys.executable,'-c',code],outputs=[f'build/value{index}'],
                               requires=[f'prepare-{index-1}' if index > 1 else 'checks']))
        raw = dict(hypothesis='CPU handoff comparison',knobs={'VLLM_TEST':'1'},
                   context=fixture.pair_context(),cpu_suites=[],cpu_tests=['tests/test_cpu.py'],prepare=stages)
        manifest = fixture.root / 'input.json'
        manifest.write_text(json.dumps(raw))
        started = time.time()
        plan = fixture.cli('plan','comparison',str(manifest),'--prepare-only')
        plan_return = time.time()-started
        ids = [stage['submission']['id'] for stage in plan['stages'] if stage['name'] != 'gpu']
        for job in ids:
            result = fixture.wait(job)
            assert result['state'] == 'succeeded', result
            assert not result['result'].get('cache_source'), result
        rows = [store.get(job) for job in ids]
        gaps = [max(0,after['started']-before['finished']) for before,after in zip(rows,rows[1:])]
        checks = rows[0]['result']['checks']
        assert checks['passed'] and checks['coverage_complete'] and checks['tests_run'] == 1, checks
        value = Path(rows[-1]['result']['artifacts'][0]['path']).read_text()
        assert value == str(depth), value
        assert not (fixture.logs / 'arms').exists()
        assert not (fixture.fleet / 'holder').exists()
        return dict(plan_return_s=plan_return,first_cpu_result_s=rows[0]['finished']-started,
                    all_cpu_results_s=rows[-1]['finished']-started,handoff_gaps_s=gaps,
                    handoff_gap_sum_s=sum(gaps),cpu_jobs=len(rows),cache_hits=0,
                    verified_artifact=value,cpu_checks=checks['tests_run'])
    finally:
        store.db.close()
        fixture.doCleanups()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='767469f613c2f8b1a5a366626f66c97ac4fbd14a')
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--depth',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10 or not 1 <= args.depth <= 8:
        parser.error('rounds must be 1..10 and depth must be 1..8')
    baseline = subprocess.check_output(['git','-C',str(ROOT),'rev-parse',args.baseline],text=True).strip()
    sources = dict(baseline=subprocess.check_output(['git','-C',str(ROOT),'show',baseline+':bench/experiments.py'],text=True),
                   candidate=(ROOT / 'bench/experiments.py').read_text())
    result = dict(baseline_revision=baseline,python=sys.version,platform=platform.platform(),
                  rounds=args.rounds,preparation_stages=args.depth,
                  worker_sha256={key:hashlib.sha256(value.encode()).hexdigest() for key,value in sources.items()},
                  samples=[],summary={})
    for index in range(args.rounds):
        for variant in (('baseline','candidate') if index % 2 == 0 else ('candidate','baseline')):
            row = dict(round=index+1,variant=variant,**sample(sources[variant],args.depth))
            result['samples'].append(row)
            print(json.dumps(row),flush=True)
    for variant in sources:
        rows = [row for row in result['samples'] if row['variant'] == variant]
        result['summary'][variant] = {key:dict(p50=statistics.median(row[key] for row in rows),
                                             minimum=min(row[key] for row in rows),maximum=max(row[key] for row in rows))
                                      for key in ('first_cpu_result_s','all_cpu_results_s','handoff_gap_sum_s','plan_return_s')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
