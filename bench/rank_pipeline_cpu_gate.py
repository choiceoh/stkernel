#!/usr/bin/env python3
"""Pinned-image CPU gate; successful CUDA initialization is forbidden."""
import json
from pathlib import Path
import sys
import unittest
import torch

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    suite = unittest.TestSuite()
    for pattern in ("test_glm53_startup_artifacts.py", "test_glm53_rank_pipeline.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skips=len(result.skipped), torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized())
    print("CPU_RESULT="+json.dumps(report), flush=True)
    raise SystemExit(not result.wasSuccessful() or result.testsRun < 37 or bool(result.skipped) or torch.cuda.is_initialized())


if __name__ == "__main__":
    main()
