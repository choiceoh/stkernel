"""The served kernels' GPU cases under compute-sanitizer on one GB10 (probe, single-GPU lane; engine/SM121_INTAKE.md
U15).

On GB10's unified memory a read past a tensor lands inside another live allocation more often than past mapped memory,
so it returns another tensor's bytes instead of faulting (vllm#49049: an unclamped query row in a gather, silent at
four sequences and a fault only at eight). The code audit of 2026-09-19 found no such index in the served paths; this
lane asks the tool that tracks every allocation's bounds instead of the page table: the GPU cases the Qwen3.8 lane runs
(probes/engine_qwen38_cells.GLUE_CASES -- QSA, GDN/KDA, conv, MoE routing and sums, gates) under `memcheck`, in a child
process, with the child's report and the tool's error summary kept.

Left out: the MLA glue class (tests.test_engine_kernel_glue.GlueOnTheGpuTests) -- the megakernel spins on a grid
barrier that needs every block resident, which the tool's instrumentation does not promise -- and the cases the Qwen3.8
lane leaves out itself (GLUE_LEFT_OUT).

    python3 probes/engine_kernel_check.py --lanes sm121_sanitizer --output /cache/sm121-sanitizer.json

A clean run is `errors: 0` with every case passed; a non-zero count names the kernel and the address in `first_errors`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOL = "memcheck"
TIMEOUT_S = 1500
LEFT_OUT = ("tests.test_engine_kernel_glue.GlueOnTheGpuTests",)
FIRST_ERRORS = 12


def cases() -> list:
    from probes.engine_qwen38_cells import GLUE_CASES
    return [c for c in GLUE_CASES if c not in LEFT_OUT]


def sanitizer() -> "str | None":
    home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    for candidate in (Path(home) / "bin" / "compute-sanitizer", shutil.which("compute-sanitizer")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def child(output) -> int:
    """Inside the tool: the cases, each case's outcome written as JSON (the tool's own summary goes to its log)."""
    import unittest
    from probes.engine_qwen38_cells import GLUE_LEFT_OUT, _cases
    suite = unittest.TestSuite(case for case in _cases(unittest.defaultTestLoader.loadTestsFromNames(cases()))
                               if case.id() not in GLUE_LEFT_OUT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(output).write_text(json.dumps({
        "tests": result.testsRun, "failures": [c.id() for c, _ in result.failures],
        "errors": [c.id() for c, _ in result.errors], "skipped": [c.id() for c, _ in result.skipped]}) + "\n")
    return 0 if result.wasSuccessful() else 1


def summary(log: str) -> dict:
    counts = [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) error", log)]
    blocks = [b.strip() for b in re.split(r"\n(?==========\s+(?:Invalid|Program hit|Uninitialized|Race))", log)
              if re.match(r"=========\s+(?:Invalid|Program hit|Uninitialized|Race)", b.strip())]
    return {"errors": sum(counts) if counts else None, "first_errors": [b[:600] for b in blocks[:FIRST_ERRORS]]}


def preflight(tool, work) -> dict:
    """What the tool says about itself, and one CUDA allocation under it -- sm121-batchA-0919b's first run ended in 1.2 s
    with "Target application terminated before first instrumented API call" and nothing on the child's stdout/stderr."""
    rows = {}
    for name, cmd in (("version", [tool, "--version"]),
                      ("minimal", [tool, "--tool", TOOL, "--print-level", "info", "--log-file", str(work / "sm121-sanitizer-minimal.log"),
                                   sys.executable, "-c", "import torch; torch.zeros(1, device='cuda'); print('cuda ok')"]),
                      ("plain", [sys.executable, "-c", "import torch; torch.zeros(1, device='cuda'); print('cuda ok')"])):
        try:
            done = subprocess.run(cmd, cwd=str(ROOT), timeout=300, capture_output=True, text=True)
            rows[name] = {"returncode": done.returncode, "stdout": done.stdout[-1500:], "stderr": done.stderr[-1500:]}
        except Exception as exc:                                        # noqa: BLE001
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    log = work / "sm121-sanitizer-minimal.log"
    if log.is_file():
        rows["minimal"]["log"] = log.read_text(errors="replace")[-3000:]
    return rows


def run(output=None) -> dict:
    tool = sanitizer()
    report = {"lane": "sm121_sanitizer", "tool": TOOL, "sanitizer": tool, "cases": cases(), "left_out": list(LEFT_OUT)}
    if tool is None:
        report["unavailable"] = "compute-sanitizer is not in CUDA_HOME/bin or on PATH"
    else:
        work = Path(output).parent if output else Path("/tmp")
        work.mkdir(parents=True, exist_ok=True)
        report["preflight"] = preflight(tool, work)
        log, result = work / "sm121-sanitizer.log", work / "sm121-sanitizer-child.json"
        cmd = [tool, "--tool", TOOL, "--error-exitcode", "99", "--print-limit", "100", "--log-file", str(log),
               sys.executable, str(Path(__file__).resolve()), "child", str(result)]
        began = time.perf_counter()
        try:
            done = subprocess.run(cmd, cwd=str(ROOT), timeout=TIMEOUT_S, capture_output=True, text=True)
            report.update(returncode=done.returncode, stdout_tail=done.stdout[-3000:], stderr_tail=done.stderr[-3000:])
        except subprocess.TimeoutExpired:
            report["returncode"] = "timeout"
        report["seconds"] = round(time.perf_counter() - began, 1)
        report.update(summary(log.read_text(errors="replace") if log.is_file() else ""))
        if result.is_file():
            report["child"] = json.loads(result.read_text())
    text = json.dumps(report, indent=1)
    if output:
        Path(output).write_text(text + "\n")
    print(text, flush=True)
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        raise SystemExit(child(sys.argv[2]))
    run(sys.argv[1] if len(sys.argv) > 1 else None)
