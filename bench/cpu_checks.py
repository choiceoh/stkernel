#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Named CPU-only behavioral gates, with per-suite logs and a JSON report.

Choose the suite relevant to the change; reuse the result for the same source
and inputs. These gates establish CPU contracts, never device correctness or
throughput. No package installation, deployment or remote commands are run.
"""
import argparse
import ast
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "logic": [[sys.executable, "tests/test_logic.py"]],
    "core": [[sys.executable, "tests/test_logic.py", "--component", "core"]],
    "fleet": [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_fleet*.py", "-v"]],
    "startup": [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"]
                for name in ("test_glm53_startup.py", "test_glm53_attestation.py",
                             "test_glm53_reclaim.py", "test_memfree_preflight.py",
                             "test_glm53_memory_preflight.py")],
    'sensitivity': [[sys.executable,'bench/cpu_contracts.py','--mutation-audit']],
}


def canonical(command, repo):
    """Adapt common unittest invocations without changing arbitrary commands."""
    if not command or not re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(command[0]).name):
        return command
    target = None
    if len(command) == 2 and command[1] == 'tests/test_logic.py':
        return [command[0], 'bench/cpu_checks.py', '--suite', 'logic']
    if len(command) == 2 and re.fullmatch(r'tests/test_[A-Za-z0-9_]+\.py', command[1]):
        target = command[1]
    elif command[1:6] == ['-m', 'unittest', 'discover', '-s', 'tests'] and len(command) in (8, 9):
        if command[6] == '-p' and (len(command) == 8 or command[8] == '-v'):
            if re.fullmatch(r'test_[A-Za-z0-9_]+\.py', command[7]):
                target = 'tests/' + command[7]
    if target and (repo / target).is_file():
        tree = ast.parse((repo / target).read_text())
        if any(isinstance(n, ast.Import) and any(v.name == 'unittest' for v in n.names)
               or isinstance(n, ast.ImportFrom) and n.module == 'unittest' for n in ast.walk(tree)):
            return [command[0], 'bench/cpu_checks.py', '--test', target]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=sorted(SUITES), action="append", default=[])
    ap.add_argument("--test", action="append", default=[], help="individual tests/test_*.py unittest file")
    from cpu_contracts import CONTRACTS
    ap.add_argument('--contract',choices=sorted(CONTRACTS),action='append',default=[])
    ap.add_argument("--out", type=Path, default=os.environ.get("FLEET_CPU_REPORT", "/tmp/stkernel-cpu-report.json"))
    args = ap.parse_args()
    if not args.suite and not args.test and not args.contract:
        ap.error('choose --suite, --test or --contract')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"evidence": "cpu-only", "checks": [], "passed": True, "coverage_complete": True, "tests_run": 0}
    selected = [(suite, command) for suite in dict.fromkeys(args.suite) for command in SUITES[suite]]
    selected += [('individual', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', Path(t).name, '-v'])
                 for t in dict.fromkeys(args.test)]
    selected += [('contract-'+c,[sys.executable,'bench/cpu_contracts.py','--contract',c]) for c in dict.fromkeys(args.contract)]
    for target in args.test:
        if not re.fullmatch(r'tests/test_[A-Za-z0-9_]+\.py', target):
            ap.error('individual test must be tests/test_*.py')
    for index, (suite, command) in enumerate(selected):
        log = args.out.with_name(args.out.stem + f"-{suite}-{index}.log")
        counts_path = log.with_suffix('.counts.json')
        counts_path.unlink(missing_ok=True)
        if '-p' in command:
            target = command[command.index('-p') + 1]
            command = [sys.executable, 'bench/cpu_unittest.py', 'tests/' + target, str(counts_path)]
        elif command[1] == 'bench/cpu_contracts.py':
            command = [*command,'--report',str(counts_path)]
        started = time.monotonic()
        print(f"CPU {suite}: {' '.join(command)}", flush=True)
        with log.open("w") as output:
            rc = subprocess.call(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        text = log.read_text()
        counts = json.loads(counts_path.read_text()) if counts_path.exists() else None
        skipped = counts['skipped'] if counts else [line.strip() for line in text.splitlines() if re.search(r"\bskip(?:ped)?\b", line, re.I)]
        matches = re.findall(r'^(?:all|core) OK \((\d+) checks;', text, re.M)
        executed = counts['tests_run'] if counts else int(matches[-1]) if matches else 0
        complete = counts['coverage_complete'] if counts else executed > 0 and not skipped
        report['tests_run'] += executed
        report["checks"].append(dict(suite=suite, command=command, returncode=rc,
                                      seconds=round(time.monotonic() - started, 3), log=str(log), skipped=skipped,
                                      tests_run=executed, counts=counts))
        report["passed"] &= rc == 0 and complete
        report["coverage_complete"] &= complete
        print(text[-4000:], flush=True)
        # Persist even on failure; downstream GPU work gets the exact gate.
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out)
        if rc:
            return rc
        if not complete:
            print("CPU coverage incomplete: supply the missing dependencies before using this as a prerequisite", flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
