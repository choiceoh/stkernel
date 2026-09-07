# SPDX-License-Identifier: Apache-2.0
"""Run a repository unittest file and publish executed-test counts, not log guesses."""
import argparse
import json
from pathlib import Path
import unittest


def run(target, out, root):
    path = (root / target).resolve()
    if path.parent != (root / 'tests').resolve() or not path.name.startswith('test_') or path.suffix != '.py':
        raise ValueError('test must be a tests/test_*.py file')
    if not path.is_file() and path.name != 'test_fleet*.py':
        raise ValueError('test does not exist: ' + target)
    suite = unittest.defaultTestLoader.discover(str(path.parent), pattern=path.name)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    complete = result.testsRun > 0 and not (result.skipped or result.expectedFailures)
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=[reason for _, reason in result.skipped], expected_failures=len(result.expectedFailures),
                  unexpected_successes=len(result.unexpectedSuccesses), coverage_complete=complete,
                  passed=result.wasSuccessful() and complete)
    out.write_text(json.dumps(report) + '\n')
    return 0 if report['passed'] else 3 if result.wasSuccessful() else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('test')
    ap.add_argument('report', type=Path)
    args = ap.parse_args()
    raise SystemExit(run(args.test, args.report, Path(__file__).resolve().parents[1]))
