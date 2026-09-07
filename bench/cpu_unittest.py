# SPDX-License-Identifier: Apache-2.0
"""Count every unittest; shard only isolated fleet tests within reserved CPU slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import unittest


def cases(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def history(root, tests, workers):
    path = Path(os.environ.get('FLEET_EXPERIMENT_ROOT', str(root/'build'))) / 'cpu-test-timings.json'
    family = hashlib.sha256(json.dumps([platform.node(),platform.machine(),list(sys.version_info[:2]),workers]).encode()).hexdigest()
    files = {}
    keys = {}
    for test in tests:
        name = test.id().split('.',1)[0]+'.py'
        if name not in files:
            source = root/'tests'/name
            files[name] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else 'unknown'
        keys[test.id()] = files[name]+':'+test.id()
    return path, family, keys


def read_history(path):
    try:
        if path.stat().st_size > 2_000_000:
            return {}
        value = json.loads(path.read_text())
        if value.get('schema') == 1 and isinstance(value.get('profiles'),dict):
            return value['profiles']
    except (OSError,ValueError,AttributeError):
        pass
    return {}


def duration(value):
    return type(value) in (int,float) and 0 < value <= 86400 and math.isfinite(value)


def shard_assignment(ids, workers, hints):
    known = [hints[name] for name in ids if duration(hints.get(name))]
    if not known:
        return [ids[i::workers] for i in range(workers)], 'round-robin'
    default = statistics.median(known)
    weights = {name:hints[name] if duration(hints.get(name)) else default for name in ids}
    groups, loads = [[] for _ in range(workers)], [0.0]*workers
    for name in sorted(ids,key=lambda name:(-weights[name],name)):
        index = min(range(workers),key=lambda i:(loads[i],len(groups[i]),i))
        groups[index].append(name)
        loads[index] += weights[name]
    return [sorted(group) for group in groups], 'duration-balanced'


def save_history(info, report):
    if not report['coverage_complete']:
        return
    path, family, keys = info
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            profiles = read_history(path)
            prior = profiles.get(family,{})
            values = dict(prior) if isinstance(prior,dict) else {}
            for name, seconds in report['test_durations_s'].items():
                if name in keys and duration(seconds):
                    old = values.get(keys[name])
                    values[keys[name]] = (old*2+seconds)/3 if duration(old) else seconds
            profiles[family] = dict(list(values.items())[-10000:])
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
                json.dump(dict(schema=1,profiles=dict(list(profiles.items())[-32:])),stream)
                temporary = Path(stream.name)
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    except (OSError,ValueError,TypeError):
        pass  # Timing hints never decide whether the test gate passes.


def execute(target, root, jobs=None, shard=None, shard_plan=None):
    path = (root/target).resolve()
    if path.parent != (root/'tests').resolve() or not path.name.startswith('test_') or path.suffix != '.py':
        raise ValueError('test must be a tests/test_*.py file')
    if not path.is_file() and path.name != 'test_fleet*.py':
        raise ValueError('test does not exist: '+target)
    # A loader retains its top-level import directory on Python 3.12. Nested
    # fixture discovery must not inherit the outer repository's directory.
    all_cases = sorted(cases(unittest.TestLoader().discover(str(path.parent),pattern=path.name)),key=lambda t:t.id())
    explicit_jobs = jobs is not None
    budget = int(os.environ.get('FLEET_CPU_SLOTS',min(2,os.cpu_count() or 1)))
    jobs = budget if jobs is None else jobs
    if not 1 <= jobs <= min(8,budget):
        raise ValueError('test workers must fit the reserved CPU slots (maximum eight)')
    # Arbitrary test files may share class/module fixtures; only reviewed fleet
    # cases use a separate temporary checkout/DB/queue for every test method.
    from cpu_evidence import FLEET_AUDIT, sha
    reviewed = {str(p.relative_to(root)):sha(p) for p in (root/'tests').glob('test_fleet*.py')} == FLEET_AUDIT and bool(FLEET_AUDIT)
    workers = min(jobs,len(all_cases)) if path.name == 'test_fleet*.py' and shard is None and (reviewed or explicit_jobs) else 1
    if shard_plan is not None and shard is None:
        raise ValueError('a shard plan requires a shard index')
    info = history(root,all_cases,workers) if path.name == 'test_fleet*.py' and shard is None else None
    if workers > 1:
        saved = read_history(info[0]).get(info[1],{})
        saved = saved if isinstance(saved,dict) else {}
        hints = {name:saved.get(key) for name,key in info[2].items()}
        groups, scheduling = shard_assignment([test.id() for test in all_cases],workers,hints)
        with tempfile.TemporaryDirectory(prefix='fleet-test-shards-') as directory:
            plan = Path(directory)/'shards.json'
            plan.write_text(json.dumps(groups))
            def child(index):
                out = Path(directory)/f'{index}.json'
                process = subprocess.run([sys.executable,str(Path(__file__).resolve()),target,str(out),
                    '--jobs','1','--shard',f'{index}/{workers}','--shard-plan',str(plan),'--root',str(root)],cwd=root,text=True,capture_output=True)
                report = json.loads(out.read_text()) if out.exists() else dict(passed=False,coverage_complete=False,
                    tests_run=0,failures=0,errors=1,skipped=[],expected_failures=0,unexpected_successes=0,
                    test_ids=[],test_durations_s={},failure_details=[dict(test='shard-'+str(index),traceback=process.stderr[-6000:])])
                report['passed'] &= process.returncode == 0
                if not report['passed']:
                    print(process.stdout[-12000:]+process.stderr[-12000:],file=sys.stderr)
                return report
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reports = list(pool.map(child,range(workers)))
        report = {key:sum(r[key] for r in reports) for key in ('tests_run','failures','errors','expected_failures','unexpected_successes')}
        report.update({key:[item for r in reports for item in r[key]] for key in ('skipped','test_ids','failure_details')})
        report['test_durations_s'] = {name:seconds for r in reports for name,seconds in r['test_durations_s'].items()}
        exact = sorted(report['test_ids']) == sorted(t.id() for t in all_cases) and len(set(report['test_ids'])) == len(all_cases)
        report.update(coverage_complete=exact and all(r['coverage_complete'] for r in reports),
                      passed=exact and all(r['passed'] for r in reports),workers=workers,scheduling=scheduling,
                      timing_hints=sum(duration(value) for value in hints.values()))
        save_history(info,report)
        print(f"Ran {report['tests_run']} tests in {workers} isolated shards: {'OK' if report['passed'] else 'FAILED'}")
        return report
    selected = all_cases
    if shard is not None:
        index,count = (int(v) for v in shard.split('/'))
        if path.name != 'test_fleet*.py' or not 0 <= index < count <= 8:
            raise ValueError('invalid fleet test shard')
        selected = all_cases[index::count]
        if shard_plan is not None:
            groups = json.loads(Path(shard_plan).read_text())
            if (not isinstance(groups,list) or len(groups) != count or
                    any(not isinstance(group,list) or any(not isinstance(name,str) for name in group) for group in groups)):
                raise ValueError('invalid shard assignment')
            flat = [name for group in groups for name in group]
            if sorted(flat) != sorted(test.id() for test in all_cases) or len(set(flat)) != len(flat):
                raise ValueError('shard assignment must cover every test exactly once')
            wanted = set(groups[index])
            selected = [test for test in all_cases if test.id() in wanted]
    stream = io.StringIO()
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            self.ids.append(test.id())
            self.began = time.monotonic()
            super().startTest(test)
        def stopTest(self,test):
            self.seconds[test.id()] = max(.000001,time.monotonic()-self.began)
            super().stopTest(test)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.ids=[]
            self.seconds={}
    result = unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(unittest.TestSuite(selected))
    print(stream.getvalue(),end='')
    complete = (result.testsRun > 0 and not (result.skipped or result.expectedFailures)
                and result.ids == [t.id() for t in selected])
    report = dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=[reason for _,reason in result.skipped],expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses),coverage_complete=complete,
        passed=result.wasSuccessful() and complete,test_ids=result.ids,workers=1,test_durations_s=result.seconds,
        failure_details=[dict(test=t.id(),traceback=trace[-6000:]) for t,trace in result.failures+result.errors])
    if info:
        save_history(info,report)
    return report


def run(target,out,root,jobs=None,shard=None,shard_plan=None):
    report = execute(target,root,jobs,shard,shard_plan)
    out.write_text(json.dumps(report)+'\n')
    return 0 if report['passed'] else 3 if not (report['failures'] or report['errors']) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('test')
    ap.add_argument('report',type=Path)
    ap.add_argument('--jobs',type=int)
    ap.add_argument('--shard')
    ap.add_argument('--shard-plan',type=Path)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    raise SystemExit(run(args.test,args.report,args.root,args.jobs,args.shard,args.shard_plan))
