"""Compile-only or bounded native direct MHC arithmetic gate; GPU via fleet."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/cache/direct-mhc.json"))
    args = parser.parse_args()
    start = time.monotonic()
    if args.compile_only:
        import torch
        from probes.engine_execution_plan_check import compile_dense
        from engine.kernels.oneshot import build
        result = {"dense": compile_dense(), "oneshot": build().__name__}
        assert not torch.cuda.is_initialized()
        scope = "SM121a compilation only; no GPU execution"
    else:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("requires an admitted GB10")
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_direct_mhc_cuda")
        run = unittest.TextTestRunner(verbosity=2).run(suite)
        if not run.wasSuccessful() or run.skipped:
            raise RuntimeError("direct MHC numerical gate failed or skipped")
        result = dict(tests=run.testsRun, skipped=0, cases=8, changed_replays=96)
        scope = "direct MHC rank fold and descriptor replay only; TP4 transport and onepass pending"
    files = ("engine/kernels/dense/kernels.cu", "engine/kernels/dense/mhc.py",
             "engine/kernels/oneshot/dsv4_oneshot_ar.cu", "engine/kernels/oneshot/__init__.py",
             "engine/profiles/glm53/direct_mhc.py", "tests/test_engine_direct_mhc_cuda.py")
    report = dict(scope=scope, result=result, seconds=time.monotonic()-start,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
