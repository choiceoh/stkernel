# SPDX-License-Identifier: Apache-2.0
"""Count every unittest; shard only isolated fleet tests within reserved CPU slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def cases(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def execute(target, root, jobs=None, shard=None):
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
    if workers > 1:
        with tempfile.TemporaryDirectory(prefix='fleet-test-shards-') as directory:
            def child(index):
                out = Path(directory)/f'{index}.json'
                process = subprocess.run([sys.executable,str(Path(__file__).resolve()),target,str(out),
                    '--jobs','1','--shard',f'{index}/{workers}','--root',str(root)],cwd=root,text=True,capture_output=True)
                report = json.loads(out.read_text()) if out.exists() else dict(passed=False,coverage_complete=False,
                    tests_run=0,failures=0,errors=1,skipped=[],expected_failures=0,unexpected_successes=0,
                    test_ids=[],failure_details=[dict(test='shard-'+str(index),traceback=process.stderr[-6000:])])
                report['passed'] &= process.returncode == 0
                if not report['passed']:
                    print(process.stdout[-12000:]+process.stderr[-12000:],file=sys.stderr)
                return report
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reports = list(pool.map(child,range(workers)))
        report = {key:sum(r[key] for r in reports) for key in ('tests_run','failures','errors','expected_failures','unexpected_successes')}
        report.update({key:[item for r in reports for item in r[key]] for key in ('skipped','test_ids','failure_details')})
        exact = sorted(report['test_ids']) == sorted(t.id() for t in all_cases) and len(set(report['test_ids'])) == len(all_cases)
        report.update(coverage_complete=exact and all(r['coverage_complete'] for r in reports),
                      passed=exact and all(r['passed'] for r in reports),workers=workers)
        print(f"Ran {report['tests_run']} tests in {workers} isolated shards: {'OK' if report['passed'] else 'FAILED'}")
        return report
    selected = all_cases
    if shard is not None:
        index,count = (int(v) for v in shard.split('/'))
        if path.name != 'test_fleet*.py' or not 0 <= index < count <= 8:
            raise ValueError('invalid fleet test shard')
        selected = all_cases[index::count]
    stream = io.StringIO()
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            self.ids.append(test.id())
            super().startTest(test)
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.ids=[]
    result = unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(unittest.TestSuite(selected))
    print(stream.getvalue(),end='')
    complete = (result.testsRun > 0 and not (result.skipped or result.expectedFailures)
                and result.ids == [t.id() for t in selected])
    return dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=[reason for _,reason in result.skipped],expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses),coverage_complete=complete,
        passed=result.wasSuccessful() and complete,test_ids=result.ids,workers=1,
        failure_details=[dict(test=t.id(),traceback=trace[-6000:]) for t,trace in result.failures+result.errors])


def run(target,out,root,jobs=None,shard=None):
    report = execute(target,root,jobs,shard)
    out.write_text(json.dumps(report)+'\n')
    return 0 if report['passed'] else 3 if not (report['failures'] or report['errors']) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('test')
    ap.add_argument('report',type=Path)
    ap.add_argument('--jobs',type=int)
    ap.add_argument('--shard')
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    raise SystemExit(run(args.test,args.report,args.root,args.jobs,args.shard))
