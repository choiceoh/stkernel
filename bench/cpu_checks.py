#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Named CPU-only behavioral gates, with per-suite logs and a JSON report.

Choose the suite relevant to the change; reuse the result for the same source
and inputs. These gates establish CPU contracts, never device correctness or
throughput. No package installation, deployment or remote commands are run.
"""
import argparse
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
    "fleet": [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_fleet_experiments.py", "-v"]],
    "startup": [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"]
                for name in ("test_glm53_startup.py", "test_glm53_attestation.py",
                             "test_glm53_reclaim.py", "test_memfree_preflight.py")],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=sorted(SUITES), action="append", required=True)
    ap.add_argument("--out", type=Path, default=os.environ.get("FLEET_CPU_REPORT", "/tmp/stkernel-cpu-report.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"evidence": "cpu-only", "checks": [], "passed": True, "coverage_complete": True}
    for suite in dict.fromkeys(args.suite):
        for index, command in enumerate(SUITES[suite]):
            log = args.out.with_name(args.out.stem + f"-{suite}-{index}.log")
            started = time.monotonic()
            print(f"CPU {suite}: {' '.join(command)}", flush=True)
            with log.open("w") as output:
                rc = subprocess.call(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            text = log.read_text()
            skipped = [line.strip() for line in text.splitlines() if re.search(r"\bskip(?:ped)?\b", line, re.I)]
            report["checks"].append(dict(suite=suite, command=command, returncode=rc,
                                          seconds=round(time.monotonic() - started, 3), log=str(log), skipped=skipped))
            report["passed"] &= rc == 0 and not skipped
            report["coverage_complete"] &= not skipped
            print(text[-4000:], flush=True)
            # Persist even on failure; downstream GPU work gets the exact gate.
            temporary = args.out.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.out)
            if rc:
                return rc
            if skipped:
                print("CPU coverage incomplete: supply the missing dependencies before using this as a prerequisite", flush=True)
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
