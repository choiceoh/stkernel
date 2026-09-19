#!/usr/bin/env python3
"""One command, one verdict: does this tree's test suite pass, and what could not be judged?

Written because the verdict used to depend on things nobody could read off the repository. Two files
(test_engine_prefix, test_engine_fleet_lease) passed or failed according to whether `PYTHONPATH=tests` happened to be
set, which appeared in no file; and "FAILED" covered both a real broken guard and a module that could not be imported
at all. An agent that cannot tell those apart will either chase a bug that is not there or ship one that is.

So this reports four states, not two:

    ok            every test ran and passed
    ok (skipped)  it passed, but some tests opted out -- a dependency is missing, not a guarantee
    FAILED        a test ran and did not pass. This is the only state that sets the exit code
    CANNOT RUN    the module could not even be imported here

Usage
    python3 tools/check.py                       # every tests/test_engine_*.py, on the CPU
    python3 tools/check.py --pattern 'test_*'    # a different slice; repeat --pattern for a union
    python3 tools/check.py --list                # every file, not just the interesting ones, with its seconds
    python3 tools/check.py --gpu                 # leave CUDA_VISIBLE_DEVICES alone (see below)
    python3 tools/check.py --shard 2/4           # the second of four parts -- CI runs the parts on four runners

`--gpu` is off by default on purpose: production GLM-5.3 *is* this engine, and a test that takes the GPU beside it
can get the server killed by earlyoom. Run with `--gpu` only when you know the box is yours.

Every file gets ONE OpenMP thread unless OMP_NUM_THREADS already says otherwise. torch's default is a thread per core,
and these tests are thousands of tiny ops, which a thread pool only slows down: on 2026-09-19, in the CPU container
(10 cores), test_engine_composed_options took 20.8 s alone at the default and 1.5 s at one thread -- and 280 s with
three other files running beside it, each with its own pool of ten. The engine suite at `--jobs 4` went from 793 s to
288 s, every file with the same verdict. It was most of what GitHub's runner spent on this verdict, too.

`--shard k/n` cuts the files into n parts longest first, by the seconds recorded in tools/check_durations.json (a file
the table has not seen counts as its median), so every runner computes the same split and each file lands in exactly
one part. The same table orders a run longest first, so the slowest file does not start last. Refresh it from a full
run with `--durations tools/check_durations.json`; a stale table costs balance, never a verdict.

A file recorded longer than one slot's fair share of the run (the table's total over n runners x --jobs) would hold
its part up by itself, so it runs as that many slices instead: each slice a process that loads the whole file and runs
every k-th test of it, the tests in the loader's order. Its line in the output is the slices together -- the worst
state, the counts and seconds summed -- or, when the slices landed on different runners, the ones here, named
`tests.x[1/2]`. Nothing is sliced on one runner with the default table: the share there is minutes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DURATIONS = ROOT / "tools" / "check_durations.json"
# Run as `python -c SLICE <module> <j> <k>`: `python -m unittest <module>`'s loader and runner over every k-th test,
# starting at the j-th. An empty slice -- a file with fewer tests than slices -- reads as a pass of none; unittest's
# own "NO TESTS RAN" would read as a file that could not run.
SLICE = """\
import sys, unittest
name, j, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["python -m unittest", name]
def flat(suite):
    for test in suite:
        yield from (flat(test) if isinstance(test, unittest.TestSuite) else [test])
mine = list(flat(unittest.defaultTestLoader.loadTestsFromName(name)))[j::k]
if not mine:
    print("Ran 0 tests in 0.000s\\n\\nOK", file=sys.stderr)
    sys.exit(0)
runner =unittest.TextTestRunner(warnings=None if sys.warnoptions else "default")
sys.exit(not runner.run(unittest.TestSuite(mine)).wasSuccessful())
"""
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
DONE = re.compile(r"^(OK|FAILED)(?: \((.*)\))?$", re.M)
CASE = re.compile(r"^(?:ERROR|FAIL): (\S+)")
RULE = re.compile(r"^[=-]{3,}\s*$").match                   # unittest's banners, whatever width it chose
EXC = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception|Failure|Exit)\b.*$")


class Verdict:
    __slots__ = ("module", "state", "tests", "skipped", "detail", "seconds")

    def __init__(self, module, state, tests=0, skipped=0, detail="", seconds=0.0):
        self.module, self.state, self.tests, self.skipped, self.detail = module, state, tests, skipped, detail
        self.seconds = seconds

    @property
    def line(self) -> str:
        note = f"  {self.detail}" if self.detail else ""
        skip = f", {self.skipped} skipped" if self.skipped else ""
        return f"  {self.state:<11} {self.module:<44} {self.seconds:6.1f}s  {self.tests} tests{skip}{note}"


def first_failure(out: str) -> str:
    """The first failing test and the line that explains it.

    A verdict that reads `FAILED  6 errors` and nothing else sends the reader back to a terminal
    -- and on CI there is no terminal to go back to. The first run of this tool on GitHub said
    exactly that about two files, and finding out that the answer was a missing `triton` wheel
    took a local harness that faked the runner. One line would have done it.
    """
    lines = out.splitlines()
    for i, line in enumerate(lines):
        case = CASE.match(line)
        if not case:
            continue
        block = lines[i + 1:i + 80]
        while block and RULE(block[0]):
            block = block[1:]                               # the rule under the case's own heading
        for stop, text in enumerate(block):                 # ends at the next banner or the next case
            if RULE(text) or CASE.match(text):
                block = block[:stop]
                break
        why = next((t.strip() for t in reversed(block) if EXC.match(t.strip())), "")
        return f" -- {case.group(1)}: {why[:90]}" if why else f" -- {case.group(1)}"
    return ""


def judge(module: str, out: str, code: int) -> Verdict:
    ran = RAN.search(out)
    tests = int(ran.group(1)) if ran else 0
    done = DONE.search(out)
    counts = dict(p.split("=", 1) for p in (done.group(2) or "").split(", ") if "=" in p) if done else {}
    skipped = int(counts.get("skipped", 0))
    if "unittest.loader._FailedTest" in out:                 # the module never imported: this is not a test result
        why = next((l.strip() for l in out.splitlines() if "Error:" in l), "import failed")
        return Verdict(module, "CANNOT RUN", tests, skipped, why)
    if done is None:
        tail = out.strip().splitlines()[-1:] or [f"no result line, exit {code}"]
        return Verdict(module, "CANNOT RUN", tests, skipped, tail[0][:90])
    if done.group(1) == "FAILED":
        what = ", ".join(f"{v} {k}" for k, v in counts.items() if k != "skipped")
        return Verdict(module, "FAILED", tests, skipped, what + first_failure(out))
    return Verdict(module, "ok", tests, skipped)


def run(module: str, gpu: bool, timeout: int, j: int = 0, k: int = 1) -> Verdict:
    """The module in a process of its own -- or, with k > 1, its j-th slice (SLICE)."""
    env = dict(os.environ)
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")                  # a thread per core made this suite 2.8x slower (docstring)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH", "")]))
    how = ["-m", "unittest", module] if k == 1 else ["-c", SLICE, module, str(j), str(k)]
    name = module if k == 1 else f"{module}[{j + 1}/{k}]"
    start = time.monotonic()
    try:
        p = subprocess.run([sys.executable, *how], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Verdict(name, "CANNOT RUN", detail=f"no verdict in {timeout}s", seconds=time.monotonic() - start)
    verdict = judge(name, p.stdout + p.stderr, p.returncode)
    verdict.seconds = time.monotonic() - start
    return verdict


def shard(text: str) -> "tuple[int, int]":
    k, _, n = text.partition("/")
    if not (k.isdigit() and n.isdigit() and 1 <= int(k) <= int(n)):
        raise argparse.ArgumentTypeError(f"{text!r} is not k/n with 1 <= k <= n")
    return int(k), int(n)


def read_table(path: pathlib.Path) -> "dict[str, float]":
    try:
        return {m: float(s) for m, s in json.loads(path.read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def weights(modules: "list[str]", table: "dict[str, float]") -> "dict[str, float]":
    """Seconds per module as recorded; a module the table has not seen yet counts as the median of those it has."""
    known = [table[m] for m in modules if m in table]
    default = statistics.median(known) if known else 1.0
    return {m: table.get(m, default) for m in modules}


def units(modules: "list[str]", weight: "dict[str, float]", n: int, jobs: int) -> "dict[tuple, float]":
    """What the parts are cut from, with the seconds each is expected to take: (module, j, k), the j-th of k slices.
    A module longer than one slot's fair share -- the whole table over n runners x `jobs` -- gets as many slices as it
    has shares; every other module is one slice of one."""
    share = max(sum(weight.values()) / (n * jobs), 1e-9)
    return {(m, j, k): weight[m] / k for m in modules for k in [max(1, math.ceil(weight[m] / share))] for j in range(k)}


def part(items: list, weight: dict, k: int, n: int) -> list:
    """The k-th of n parts, longest first. Each item goes to the part with the fewest seconds so far, so the parts
    come out even, and the cut depends only on the list and the table: n runners computing it apart from each other
    cover every item exactly once."""
    loads, parts = [0.0] * n, [[] for _ in range(n)]
    for item in sorted(items, key=lambda item: (-weight[item], item)):
        i = min(range(n), key=lambda i: (loads[i], i))
        parts[i].append(item)
        loads[i] += weight[item]
    return parts[k - 1]


RANK = {"FAILED": 0, "CANNOT RUN": 1, "ok": 2}


def gather(done: "list[tuple[tuple, Verdict]]") -> "list[Verdict]":
    """One line per module: a module run as slices reads as its worst slice, with the tests, skips and seconds of all
    of them. A module whose other slices ran on another runner keeps the numbers of the slices it has in its name."""
    groups: "dict[str, list]" = {}
    for (module, j, k), verdict in done:
        groups.setdefault(module, []).append((j, k, verdict))
    out = []
    for module, got in sorted(groups.items()):
        if got[0][1] == 1:
            out.append(got[0][2])
            continue
        got.sort(key=lambda g: g[0])
        k, slices = got[0][1], [v for _, _, v in got]
        worst = min(slices, key=lambda v: RANK[v.state])
        name = module if len(got) == k else f"{module}[{','.join(str(j + 1) for j, _, _ in got)}/{k}]"
        out.append(Verdict(name, worst.state, sum(v.tests for v in slices), sum(v.skipped for v in slices),
                           worst.detail, sum(v.seconds for v in slices)))
    return out


def record(path: pathlib.Path, verdicts: "list[Verdict]") -> None:
    """Fold this run's seconds into the table. A module that never imported took no time worth keeping, one with only
    some of its slices here has no whole to report, and one whose file is gone should stop pulling its old weight into
    the median."""
    table = read_table(path)
    table.update({v.module: round(v.seconds, 1) for v in verdicts if v.state != "CANNOT RUN" and "[" not in v.module})
    alive = {m: s for m, s in table.items() if (ROOT / "tests" / (m.split(".", 1)[-1] + ".py")).exists()}
    path.write_text(json.dumps(dict(sorted(alive.items())), indent=0) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", action="append", help="test file stem glob, repeatable (default: test_engine_*)")
    ap.add_argument("--jobs", type=int, default=4, help="files at a time (default 4: this box also serves)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds one file may take")
    ap.add_argument("--gpu", action="store_true", help="do not hide the GPU from the tests")
    ap.add_argument("--list", action="store_true", help="print every file, not only the ones worth reading")
    ap.add_argument("--shard", type=shard, default=(1, 1), metavar="K/N", help="run only the k-th of n parts")
    ap.add_argument("--durations", metavar="JSON", help="fold this run's seconds per file into this table")
    a = ap.parse_args()

    patterns = a.pattern or ["test_engine_*"]
    files = sorted({f for p in patterns for f in (ROOT / "tests").glob(p + ".py")})
    if not files:
        print(f"no tests match {' '.join(patterns)!r} under {ROOT / 'tests'}")
        return 2
    modules = [f"tests.{f.stem}" for f in files]
    (k, n), start = a.shard, time.monotonic()
    pieces = units(modules, weights(modules, read_table(DURATIONS)), n, a.jobs)
    mine = part(list(pieces), pieces, k, n)
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        verdicts = gather(list(zip(mine, pool.map(lambda u: run(u[0], a.gpu, a.timeout, u[1], u[2]), mine))))
    if a.durations:
        record(pathlib.Path(a.durations), verdicts)

    bad = [v for v in verdicts if v.state != "ok"]
    for v in (verdicts if a.list else bad):
        print(v.line)
    tests = sum(v.tests for v in verdicts)
    skipped = sum(v.skipped for v in verdicts)
    failed = [v for v in verdicts if v.state == "FAILED"]
    stuck = [v for v in verdicts if v.state == "CANNOT RUN"]
    if bad and not a.list:
        print()
    scope = f"part {k}/{n}: {len(verdicts)} of {len(modules)} files" if n > 1 else f"{len(modules)} files"
    print(f"  {scope}, {tests} tests: {len(verdicts) - len(bad)} ok, {len(failed)} failed, "
          f"{len(stuck)} cannot run" + (f", {skipped} skipped" if skipped else "")
          + f" in {time.monotonic() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
