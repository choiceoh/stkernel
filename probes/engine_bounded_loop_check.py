"""Compile-only or admitted GB10 conditional graph and decode-commit gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if args.compile_only:
        from engine.kernels.bounded_graph import build
        result = dict(module=build().__name__)
        assert not torch.cuda.is_initialized()
        scope = "SM121a conditional graph compilation only; no CUDA context"
    else:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("requires an admitted GB10")
        run = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_bounded_loop_cuda"))
        if not run.wasSuccessful() or run.skipped:
            raise RuntimeError("bounded loop gate failed or skipped")
        result = dict(tests=run.testsRun, skipped=0)
        scope = "single GB10 conditional graph, real token commit, stop and lifetime; not TP4 or serving proof"
    root = Path(__file__).resolve().parents[1]
    files = ("engine/kernels/bounded_graph/loop.cu", "engine/kernels/bounded_graph/__init__.py",
             "engine/profiles/glm53/bounded_loop.py", "engine/kernels/decode_commit.py",
             "tests/test_engine_bounded_loop_cuda.py")
    report = dict(scope=scope, result=result, seconds=time.monotonic()-start,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    Path("/cache/bounded-loop.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
